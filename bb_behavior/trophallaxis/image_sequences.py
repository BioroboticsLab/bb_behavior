from bb_backend.api import FramePlotter

from ..db import DatabaseCursorContext, get_neighbour_frames, get_interpolated_trajectory
from ..io import BeesbookVideoManager

from collections import defaultdict
import json
import skimage.exposure
import scipy.ndimage
import scipy.ndimage.interpolation
import numba
import numpy as np
import pandas as pd
import prefetch_generator
import zipfile
import itertools
import io
import torch.utils.data
from tqdm import tqdm_notebook


def get_whole_frame_image_sequences(video_manager, frame_ids_fcs, n_frames_before_after=5, verbose=False):
    """Takes a list of frames and, for each frame, fetches +/-n_frames_before_after neighbour frames and retrieves everything
    as images. To speed up the process, the video_manager is used to cache image files.
    Ideally, the frames should be sorted by cam ID and timestamp.
    
    Arguments:
        video_manager: bb_behavior.io.videos.BeesbookVideoManager
            Video manager that is aware where the videos are located and where to cache the frame images.
        frame_ids_fcs: list(tuple(int, int))
            List of frame IDs and frame container IDs that each constitute the middle frame of a sequence.
        n_frames_before_after: int
            How many frames to fetch around the center frames. Yields 1 + 2 * n_frames_before_after frames.
        verbose: bool
            Whether to print additional information.
    Yields:
        images, neighbour_frames: list(np.array), list(tuple)
            For each input frame ID, a pair of the images and neighbour information
            (see bb_behavior.db.get_neighbour_frames) is returned.
    """
    
    with DatabaseCursorContext("Troph. whole frame retrieval") as prepared_cursor:
        # Cache all frames that are bound to be queried.
        all_neighbour_frames = [None] * len(frame_ids_fcs)
        def get_neighbour_frames_for_index(index):
            if all_neighbour_frames[index] is None:
                frame_id = frame_ids_fcs[index][0]
                all_neighbour_frames[index] = get_neighbour_frames(frame_id, n_frames=n_frames_before_after) 
            return all_neighbour_frames[index]
            
        for idx, (_, fc_id) in enumerate(frame_ids_fcs):
            # A yet unseen frame container? Note that they are sorted.
            if idx == 0 or (frame_ids_fcs[idx - 1][1] != fc_id):
                video_manager.clear_video_cache(retain_last_n_requests=1)
                new_frame_ids_neighbours = [(frame_ids_fcs[i][0], get_neighbour_frames_for_index(i)) \
                                            for i in range(idx, len(frame_ids_fcs)) \
                                            if frame_ids_fcs[i][1] == fc_id]
                new_frame_ids = set()
                for (_, neighbours) in new_frame_ids_neighbours:
                    new_frame_ids |= {n[1] for n in neighbours if n[1] is not None}
                if verbose:
                    print("New frame container (id={})! Caching {} frames.".format(fc_id, len(new_frame_ids)))
                video_manager.cache_frames(new_frame_ids, cursor=prepared_cursor, verbose=verbose)
            neighbour_frames = get_neighbour_frames_for_index(idx)
            
            frame_ids = [f[1] for f in neighbour_frames]
            try:
                yield video_manager.get_frames(frame_ids, cursor=prepared_cursor, verbose=verbose), neighbour_frames
            except Exception as e:
                print("VideoManager: error getting frames for container {}.".format(fc_id))
                for frame_id in frame_ids:
                    if "{}.bmp".format(frame_id) in video_manager.deleted_images:
                        print("Image was already removed from cache ({})".format(frame_id))
                print(str(e))
                yield None, None

@numba.njit
def get_affine_transform(xy0, xy1):
    """Returns the affine transformation that puts xy0 at the upper left corner, facing xy1 to the lower right.
    
    Arguments:
        xy0, xy1: tuple(x, y, orientation)
    Returns:
        R, offset: np.array, np.array
            R is a 2x2 rotation matrix and offset is the translation.
            Both together constitute the input to scipy.ndimage.affine_transform.
    """
    r = np.arctan2(xy1[1] - xy0[1], xy1[0] - xy0[0])
    r -= np.pi / 4.0
    c, s = np.cos(r), np.sin(r)
    R = np.array(((c,-s), (s, c))).T
    offset = np.array((xy0[1], xy0[0])) - np.dot(R, np.array([32.0, 32.0]))
    
    return R, offset

def rotate_crops(traj0, traj1, images):
    """Takes trajectories in sub-image coordinates and places/rotates one bee to the upper left corner.
    
    Arguments:
        traj0, traj1: list(tuple(x, y, orientation))
            Two lists containing the position of two individuals in every frame.
        images: list(np.array)
            List of images.
    Returns:
        images: list(np.array)
            Cropped, rotated and translated image regions.
    """
    results = []
    for xy0, xy1, im in zip(traj0, traj1, images):
        R, offset = get_affine_transform(xy0, xy1)
        #output = np.zeros(shape=(64, 64), dtype=np.float32)
        im2 = scipy.ndimage.affine_transform(im, R, offset=offset, order=1, output_shape=(128, 128), mode='mirror')
        results.append(im2)
    return results

@numba.njit
def to_head_position(traj):
    """Extrapolates a position in pixel coordinates to the location of the head.
    Works in place.
    
    Arguments:
        traj: np.array
            shape=(N, 3) where each row is np.array(x, y, orientation).
    """
    head_distance = 50
    traj[:, 0] += np.cos(traj[:, 2]) * head_distance
    traj[:, 1] += np.sin(traj[:, 2]) * head_distance

def get_crops_for_bees(traj0, traj1, images):
    """Takes trajectories and a list of full frame images and returns small crops around each meeting point.
    
    Arguments:
        traj0, traj1: np.array
            shape=(N, 3) where each row is np.array(x, y, orientation).
        images: list(np.array)
    
    Returns:
        xy0, xy1, cropped_images: list(tuple), list(tuple), list(np.array)
            Rotated and cropped positions and images.
    """
    to_head_position(traj0)
    to_head_position(traj1)
    
    s = 128
    xs = (((traj0[:, 0] + traj1[:, 0]) / 2.0) - s // 2).astype(np.int32)
    ys = (((traj0[:, 1] + traj1[:, 1]) / 2.0) - s // 2).astype(np.int32)
    
    all_xy0, all_xy1, cropped_images = [], [], []
    
    for idx, (x, y, im) in enumerate(zip(xs, ys, images)):
        sub_im = np.zeros(shape=(s, s), dtype=np.uint8)
        x_begin, y_begin = x, y
        need_fill = False
        if x_begin < 0:
            x_begin = 0
            need_fill = True
        if y_begin < 0:
            y_begin = 0
            need_fill = True
        x_end, y_end = x + s, y + s
        if x_end > im.shape[1]:
            x_end = im.shape[1]
            need_fill = True
        if y_end > im.shape[0]:
            y_end = im.shape[0]
            need_fill = True
        if need_fill:
            sub_im += int(np.mean(im) * 255.0)
        to_end_x = sub_im.shape[1] - ((x + s) - x_end)
        to_end_y = sub_im.shape[0] - ((y + s) - y_end)
        sub_im[(y_begin - y):to_end_y, (x_begin - x):to_end_x] = \
                255.0 * im[y_begin:y_end, x_begin:x_end]
        sub_im = skimage.exposure.equalize_adapthist(sub_im).astype(np.float32)
        xy0 = float(traj0[idx, 0] - x), float(traj0[idx, 1] - y), float(traj0[idx, 2])
        xy1 = float(traj1[idx, 0] - x), float(traj1[idx, 1] - y), float(traj1[idx, 2])
        """
        try:
            sub_im[int(xy0[1]), int(xy0[0])] = 255
        except:
            pass
        try:
            sub_im[int(xy1[1]), int(xy1[0])] = 255
        except:
            pass
        """
        all_xy0.append(xy0)
        all_xy1.append(xy1)
        cropped_images.append(sub_im)
        
    return all_xy0, all_xy1, cropped_images

def get_all_crops_for_frame(frame_id, df, images, neighbour_frames, cursor=None, verbose=False):
    """Takes a center frame ID, a list of neighbour frame information (see bb_behavior.db.get_neighbour_frames) and a list of images.
    In addition a dataframe of (labeled) bee ID pairs is taken to cut out all interactions in the supplied frames.
    
    Arguments:
        frame_id: int
        df: pandas.DataFrame
            DataFrame containing at least the columns bee_id0, bee_id1, label, event_id.
        images: list(np.array)
            List of full frame images that correspond to neighbour_frames.
        neighbour_frames: list(tuple(timestamp, frame_id, cam_id))
            As returned by bb_behavior.db.get_neighbour_frames.
        cursor: psycopg2.cursor
            Prepared database cursor.
        verbose: bool
            Whether to display the retrieved images.
    """
    all_results = []
    for bee_id0, bee_id1, label, event_id in df[["bee_id0", "bee_id1", "label", "event_id"]].itertuples(index=False):
        cropped_images = None
        try:
            traj0, mask0 = get_interpolated_trajectory(int(bee_id0), frames=neighbour_frames, cursor=cursor, cursor_is_prepared=cursor is not None)
            traj1, mask1 = get_interpolated_trajectory(int(bee_id1), frames=neighbour_frames, cursor=cursor, cursor_is_prepared=cursor is not None)

            if mask0.sum() < mask0.shape[0] // 3:
                raise ValueError("Bee {} has no data available around frame {}.".format(bee_id0, neighbour_frames[len(neighbour_frames) // 2][1]))
            if mask1.sum() < mask1.shape[0] // 3:
                raise ValueError("Bee {} has no data available around frame {}.".format(bee_id1, neighbour_frames[len(neighbour_frames) // 2][1]))

            local_traj0, local_traj1, cropped_images = get_crops_for_bees(traj0, traj1, images)
            
            all_results.append((dict(
                    frame_id=int(frame_id), event_id=event_id,
                    bee_id0=bee_id0, bee_id1=bee_id1, label=label,
                    local_traj0=local_traj0, local_traj1=local_traj1,
                    traj0=list(map(tuple, traj0.astype(float))), traj1=list(map(tuple, traj1.astype(float))),
                    mask0=list(mask0.astype(float)), mask1=list(mask1.astype(float))
                ), np.stack(cropped_images)))
        except Exception as e:
            print("Error at {}_{}_{}_{} - neighbour frames: {}".format(
                event_id, frame_id, bee_id0, bee_id1, str([int(f[1]) for f in neighbour_frames])))
            print(str(e))
            if verbose and cropped_images is not None:
                import matplotlib.pyplot as plt
                from IPython.display import display
                display((mask0, mask1))
                fig, axes = plt.subplots(1, len(cropped_images), figsize=(20, 5))
                for ax, im in zip(axes, cropped_images):
                    ax.imshow(im, cmap="gray")
                    ax.set_title(str(im.shape))
                    ax.set_axis_off()
                plt.tight_layout()
                plt.show()
        
    return all_results

def generate_image_sequence_data(dataframe, output_file, video_root, video_cache_path,
                                    n_sequences=None, append=True,
                                    verbose=False, dry=False, skip_first_n_frames=0):
    """Generates and stores image sequences for all rows in a pandas.DataFrame.

    Arguments:
        dataframe: pandas.DataFrame
            DataFrame containing at least the columns bee_id0, bee_Id1, label, event_id.
        output_file: string
            Path to .zip file where the output is stored.
        video_root: string
            Path to the beesbook video location.
        video_cache_path: string
            Path where frame images can be cached (recommendation: use a ramdisk).
        n_sequences: int
            Optional. Whether to stop after having generated at least n_sequence image sequences.
        append: bool
            Whether to append to the output file instead of clearing it first.
        verbose: bool
            Whether to print additional output.
        dry: bool
            If set, no data is actually stored to the file.
        skip_first_n_frames: int
            Skips the first skip_first_n_frames frame IDs (after sorting).
    """
    video_manager = BeesbookVideoManager(video_root=video_root,
                                        cache_path=video_cache_path)
    
    dataframe.sort_values(["cam_id", "timestamp"], inplace=True)
    # Get a list containing both the unique frame IDs and the respective frame containers to make caching faster.
    unique_frames_fcs = list(dataframe[["frame_id", "fc_id"]].drop_duplicates().itertuples(index=False, name=None))
    image_source = get_whole_frame_image_sequences(video_manager, unique_frames_fcs[skip_first_n_frames:], verbose=verbose)
    image_source = prefetch_generator.BackgroundGenerator(image_source, max_prefetch=1)

    iterable = zip(itertools.islice(dataframe.groupby("frame_id", sort=False), skip_first_n_frames, None), image_source)
    iterable = tqdm_notebook(iterable, total=len(unique_frames_fcs))
    
    generated_sequence_count = 0
    with DatabaseCursorContext("Troph. image retrieval") as cursor:

        mode = "a" if append else "w"
        with zipfile.ZipFile(output_file, mode=mode, compression=zipfile.ZIP_DEFLATED) as zf:

            for (frame_id, df), (images, neighbour_frames) in iterable:
                if images is None:
                    continue
                if frame_id != neighbour_frames[len(neighbour_frames) // 2][1]:
                    print("Error extracting around frame_id {}".format(frame_id))
                    print("Neighbours: {}".format(str([int(n[1]) for n in neighbour_frames])))
                    raise ValueError("Wrong neighbour frames returned.")
                results = get_all_crops_for_frame(frame_id, df, images, neighbour_frames, cursor=cursor, verbose=verbose)

                for (metadata, images) in results:
                    
                    event_id = metadata["event_id"]
                    frame_id = metadata["frame_id"]
                    bee_id0 = metadata["bee_id0"]
                    bee_id1 = metadata["bee_id1"]
                    filename = "{}_{}_{}_{}".format(event_id, frame_id, bee_id0, bee_id1)
                    if not dry:
                        zf.writestr(filename + ".json", json.dumps(metadata))

                        with zf.open(filename + ".npy", mode="w") as image_file:
                            np.save(image_file, images)

                    if verbose:
                        import matplotlib.pyplot as plt
                        fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
                        print(filename)
                        for idx, (xy0, xy1, im) in enumerate(zip(metadata["local_traj0"], metadata["local_traj1"], images)):
                            ax = axes[idx]
                            ax.imshow(im, cmap="gray")
                            ax.set_axis_off()
                            label = metadata["label"]
                            ax.set_title("{} {:2.2f}".format(int(label), xy0[2] / np.pi * 180.0))
                        plt.show()

                    generated_sequence_count += 1

                if n_sequences and generated_sequence_count >= n_sequences:
                    break

def get_available_events(path, with_event_metadata=False, with_frame_metadata=False, progress=None):
    """Loads information about all available events from a zip file generated by generate_image_sequence_data.

    Arguments:
        path: string
            Path to zip file.
        with_event_metadata: bool
            Whether to load additional metadata about the available data from the zip archives.
        with_frame_metadata: bool
            Whether to load additional metadata about the frame from the database.
        progress: None, "tqdm" or "tqdm_notebook"
            Whether to display a progress bar.
    Returns:
        pandas.DataFrame
    """
    from concurrent.futures import ThreadPoolExecutor as PoolExecutor
    from ..utils.processing import get_progress_bar_fn
    progress_bar_fn = get_progress_bar_fn(progress)

    def load_metadata(zf, filename):
        with zf.open(filename) as file:
            meta = json.load(file)
            if meta["bee_id0"] >= meta["bee_id1"]:
                from IPython.display import display
                display(meta)
            assert(meta["bee_id0"] < meta["bee_id1"])
            return meta
        
    executor = None
    if with_event_metadata:
        executor = PoolExecutor(max_workers=8)
        
    df = None
    available = defaultdict(list)
    with zipfile.ZipFile(path, mode="r", compression=zipfile.ZIP_DEFLATED) as zf:
        for obj in zf.infolist():
            if obj.filename.endswith(".npy") and obj.file_size > 0:
                event_id, frame_id, bee_id0, bee_id1 = map(int, obj.filename[:-4].split("_"))
                bee_id0, bee_id1 = min(bee_id0, bee_id1), max(bee_id0, bee_id1)
                available["event_id"].append(event_id)
                available["frame_id"].append(frame_id)
                available["bee_id0"].append(bee_id0)
                available["bee_id1"].append(bee_id1)

                if executor:
                    available["metadata"].append(executor.submit(load_metadata, zf, obj.filename[:-3]+"json"))
                    available["filename"].append(obj.filename[:-4])
                    
        if len(available) == 0:
            return None

        metadata = available["metadata"]
        del available["metadata"]
        df = pd.DataFrame(available)

        if executor:
            metadata = [meta.result() for meta in progress_bar_fn(metadata)]
            executor.shutdown()
            
            meta_df = pd.DataFrame(metadata)
            meta_df.drop(["event_id", "frame_id", "bee_id0", "bee_id1"], axis=1, inplace=True)
            df = pd.concat((df, meta_df), axis=1)

        if with_frame_metadata:
            from ..db.metadata import get_frame_metadata
            metadata = get_frame_metadata(df.frame_id.unique())
            df = pd.merge(df, metadata, how="left", left_on="frame_id", right_on="frame_id")
    return df

class TrophallaxisImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, extracted_path=None, data=None, n_channels=5, use_augmentations=True):
        self.path = path
        self.n_channels = n_channels
        
        if data is None:
            self.data = get_available_events(path, with_event_metadata=True)
        else:
            self.data = data
        
        self.extracted_path = extracted_path
        
        if use_augmentations:
            from imgaug import augmenters as iaa
            self.augmentations = iaa.Sequential([
                # Strengthen or weaken the contrast in each image.
                iaa.ContrastNormalization((0.8, 1.2)),
                # Make some images brighter and some darker.
                iaa.Multiply((0.8, 1.2)),
                # Apply affine transformations to each image.
                # Scale/zoom them, translate/move them, rotate them and shear them.
                iaa.Affine(
                    scale={"x": (0.96, 1.04), "y": (0.96, 1.04)},
                    rotate=(-5, 5),
                    shear=(-2, 2),
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
                )
            ], random_order=True)
        else:
            self.augmentations = None
        
    def __len__(self):
        return self.data.shape[0]
    
    
    def __getitem__(self, idx):
        filename = self.data.filename.iloc[idx]
        trajs = self.data.local_traj0.iloc[idx], self.data.local_traj1.iloc[idx]
        label = self.data.label.iloc[idx]
                
        if not self.extracted_path:
            with open(self.path, "rb") as zf_handle:
                zf = zipfile.ZipFile(zf_handle, mode="r", compression=zipfile.ZIP_DEFLATED)
                image_data = zf.read(filename + ".npy")
                bytes_io = io.BytesIO(image_data)
                images = np.load(bytes_io)
        else:
            path = self.extracted_path + "/" + filename + ".npy"
            images = np.load(path)
        
        if False:
            images = images[len(images) // 2]
            images = images.reshape((1, images.shape[0], images.shape[1]))
        else:
            mid = len(images) // 2
            channel_offset = self.n_channels // 2
            images = images[(mid - channel_offset):(mid + channel_offset + 1)]
        images = rotate_crops(*trajs, images=images)
        
        if self.augmentations is not None:
            deterministic = self.augmentations.to_deterministic()
            #print(images[0].dtype, images[0].shape, images[0].min(), images[0].max())
            images = [deterministic.augment_image(img) for img in images]
            #print(images[0].dtype, images[0].shape, images[0].min(), images[0].max())

        if False:#label:
            from IPython.display import display
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, len(images), figsize=(20, 5))
            for ax, im in zip(ax, images):
                ax.imshow(im, cmap="gray")
            display(fig)
            plt.show()
            display(self.data.iloc[idx, :])
        
        images = np.stack(images).astype(np.float32)
        return images, float(label)
    
    def get_splits(self, p=0.8):
        # Stratified group split.
        events_per_label = defaultdict(list)
        for label, df in self.data.groupby("label"):
            events = list(df.event_id.unique())
            np.random.seed(42)
            np.random.shuffle(events_per_label)
            events_per_label[label] = events
        
        splits = [[], []]
        for label, events in events_per_label.items():
            n_first_split = int(p * len(events))
            splits[0].extend(events[:n_first_split])
            splits[1].extend(events[n_first_split:])
        
        for split in splits:
            idx = self.data.event_id.isin(split)
            yield TrophallaxisImageDataset(path=self.path, data=self.data[idx], extracted_path=self.extracted_path)