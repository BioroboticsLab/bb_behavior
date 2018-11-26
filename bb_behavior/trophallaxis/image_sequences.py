from bb_backend.api import FramePlotter

from ..db import DatabaseCursorContext, get_neighbour_frames, get_interpolated_trajectory
from ..io import BeesbookVideoManager

import json
import skimage.exposure
import scipy.ndimage
import scipy.ndimage.interpolation
import numba
import numpy as np
import pandas as pd
import prefetch_generator
import zipfile
from tqdm import tqdm_notebook


def get_whole_frame_image_sequences(video_manager, frame_ids_fcs, n_frames_before_after=5):
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
    Yields:
        images, neighbour_frames: list(np.array), list(tuple)
            For each input frame ID, a pair of the images and neighbour information
            (see bb_behavior.db.get_neighbour_frames) is returned.
    """
    
    with DatabaseCursorContext("Troph. whole frame retrieval") as prepared_cursor:
        for idx, (frame_id, fc_id) in enumerate(frame_ids_fcs):
            # A yet unseen frame container? Note that they are sorted.
            if idx == 0 or (frame_ids_fcs[idx - 1][1] != fc_id):
                video_manager.clear_video_cache(retain_last_n_requests=4)
                new_frame_ids = [f[0] for f in frame_ids_fcs[idx:] if f[1] == fc_id]
                video_manager.cache_frames(new_frame_ids, cursor=prepared_cursor)
            neighbour_frames = get_neighbour_frames(frame_id, n_frames=n_frames_before_after)        
            frame_ids = [f[1] for f in neighbour_frames]
            yield video_manager.get_frames(frame_ids, cursor=prepared_cursor), neighbour_frames

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
        im2 = scipy.ndimage.affine_transform(im, R, offset=offset, output_shape=(128, 128), mode='mirror')
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
        sub_im = (255.0 * im[y:(y+s), x:(x+s)]).astype(np.uint8)
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
        try:
            traj0, mask0 = get_interpolated_trajectory(int(bee_id0), frames=neighbour_frames, cursor=cursor, cursor_is_prepared=cursor is not None)
            traj1, mask1 = get_interpolated_trajectory(int(bee_id1), frames=neighbour_frames, cursor=cursor, cursor_is_prepared=cursor is not None)
        
            local_traj0, local_traj1, cropped_images = get_crops_for_bees(traj0, traj1, images)
            
            all_results.append((dict(
                    frame_id=frame_id, event_id=event_id,
                    bee_id0=bee_id0, bee_id1=bee_id1, label=label,
                    local_traj0=local_traj0, local_traj1=local_traj1,
                    traj0=list(map(tuple, traj0.astype(float))), traj1=list(map(tuple, traj1.astype(float))),
                    mask0=list(mask0.astype(float)), mask1=list(mask1.astype(float))
                ), np.stack(cropped_images)))

            if verbose:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
                for idx, (xy0, xy1, im) in enumerate(zip(local_traj0, local_traj1, cropped_images)):
                    ax = axes[idx]
                    ax.imshow(im, cmap="gray")
                    ax.set_axis_off()
                    ax.set_title("{} {:2.2f}".format(int(label), xy0[2] / np.pi * 180.0))
                plt.show()
        except Exception as e:
            print("Error at {}_{}_{}_{}".format(event_id, frame_id, bee_id0, bee_id1))
            print(str(e))
        
    return all_results

def generate_image_sequence_data(dataframe, output_file, video_root, video_cache_path, n_sequences=None):
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
    """
    video_manager = BeesbookVideoManager(video_root=video_root,
                                        cache_path=video_cache_path)
    
    dataframe.sort_values(["cam_id", "timestamp"], inplace=True)
    # Get a list containing both the unique frame IDs and the respective frame containers to make caching faster.
    unique_frames_fcs = list(dataframe[["frame_id", "fc_id"]].drop_duplicates().itertuples(index=False, name=None))
    image_source = get_whole_frame_image_sequences(video_manager, unique_frames_fcs)
    image_source = prefetch_generator.BackgroundGenerator(image_source, max_prefetch=1)

    iterable = zip(dataframe.groupby("frame_id", sort=False), image_source)
    iterable = tqdm_notebook(iterable, total=len(unique_frames_fcs))
    
    generated_sequence_count = 0
    with DatabaseCursorContext("Troph. image retrieval") as cursor:

        with zipfile.ZipFile(output_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:

            for request_idx, ((frame_id, df), (images, neighbour_frames)) in enumerate(iterable):
                results = get_all_crops_for_frame(frame_id, df, images, neighbour_frames, cursor=cursor, verbose=False)

                for (metadata, images) in results:
                    
                    event_id = metadata["event_id"]
                    frame_id = metadata["frame_id"]
                    bee_id0 = metadata["bee_id0"]
                    bee_id1 = metadata["bee_id1"]
                    filename = "{}_{}_{}_{}".format(event_id, frame_id, bee_id0, bee_id1)
                    zf.writestr(filename + ".json", json.dumps(metadata))

                    with zf.open(filename + ".npy", mode="w") as image_file:
                        np.save(image_file, images)

                    generated_sequence_count += 1

                if n_sequences and generated_sequence_count >= n_sequences:
                    break