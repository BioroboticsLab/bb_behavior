"""This module wraps the BeesBook detection pipeline and the tracking.
It is not optimized for high-performance throughput but for accessibility.

Usage:
    Have video (or list of images) ready.
    Have stored tracking parameters ready.

    ```
    # For the stored tracking parameters to resolve to the correct names.
    import math
    import numpy as np
    from bb_tracking.data.constants import DETKEY
    from bb_tracking.tracking import distance_orientations_v, distance_positions_v

    frame_info, detections = detect_markers_in_video(video_path, ...)
    tracks = track_detections_dataframe(detections, tracker=stored_tracking_parameters, ...)
    display_tracking_results(video_path, frame_info, detections, tracks)
    ```


If you want the detection pipeline to use GPU acceleration, you have to configure your theano accordingly.
E.g. in a jupyter notebook if everything is installed, use
```
%env KERAS_BACKEND=theano
%env THEANO_FLAGS=floatX=float32,device=cuda0
```
"""
from ..plot.misc import draw_ferwar_id_on_axis

import bb_tracking
import bb_tracking.data
import bb_tracking.tracking

import numpy as np
import pandas as pd
import bb_trajectory.utils
from collections import defaultdict

import bb_tracking.data
import bb_tracking.tracking
import dill

import pipeline.io
from pipeline.objects import PipelineResult

# These imports are necessary so that the bb_tracking module works correctly.
# And this, my kids, is why we don't import into the global namespace.
import math
from bb_tracking.data.constants import DETKEY
from bb_tracking.tracking import score_id_sim_v
from bb_tracking.tracking import distance_orientations_v, distance_positions_v

import bb_utils.ids

def get_timestamps_for_beesbook_video(path):
    """Takes the file path to a beesbook video (e.g. "/home/david/video.mp4").
    The function loads the corresponding timestamp-file (e.g. "/home/david/video.txt") and parses the
    timestamps for all frames.
    
    Arguments:
        path: file path to video (incl. .mp4 or .avi extension).
            A file with the same name but a .txt extension has to exist in the same directory.
    Returns:
        list containing UNIX timestamps with milliseconds. Length corresponds to the number of frames in the video.
    """
    import time, datetime
    def to_timestamp(datetime_string):
        # e.g. Cam_0_2018-09-13T17:13:49.501824Z
        dt = datetime.datetime.strptime(datetime_string[6:-1], "%Y-%m-%dT%H:%M:%S.%f")
        return time.mktime(dt.utctimetuple()) + dt.microsecond * 0.000001
    ts_path = path[:-3] + "txt"
    with open(ts_path, "r") as f:
        return [to_timestamp(l) for l in f.read().splitlines()]
    
def detect_markers_in_beesbook_video(video_path, *args, **kwargs):
    """Wraps detect_markers_in_video but loads timestamps from a .txt file next to the video file.
    See get_timestamps_for_beesbook_video.
    
    Arguments:
        video_path: Path to beesbook video. A .txt-extension file with the same name has to exist in the same directory.
    Returns:
        frame_info, video_dataframe: see detect_markers_in_video
    """
    timestamps = get_timestamps_for_beesbook_video(video_path)
    return detect_markers_in_video(video_path, timestamps=timestamps, *args, **kwargs)

def get_default_pipeline(localizer_threshold=None, verbose=False):
    """Creates and returns a bb_pipeline Pipeline object that is configured to
    take an image and return all info required for bb_binary (PipelineResult).
    
    Arguments:
        localizer_threshold: float
            Threshold for the localizer in the pipeline.
        verbose: bool
            Whether to also provide the CrownOverlay output for display purposes (slower).
        
    Returns:
        bb_pipeline.pipeline.Pipeline object, ready to be used.
    """
    import pipeline
    import pipeline.pipeline
    import pipeline.objects
    
    outputs = [pipeline.objects.PipelineResult]
    if verbose:
        import pipeline.objects.CrownOverlay
        outputs += [pipeline.objects.CrownOverlay]
    conf = pipeline.pipeline.get_auto_config()
    if localizer_threshold is not None:
        conf['Localizer']['threshold'] = localizer_threshold
    pipeline = pipeline.Pipeline([pipeline.objects.Image],  # inputs
                        outputs,  # outputs
                        **conf)
    return pipeline

def detect_markers_in_video(source_path, source_type="auto", pipeline=None, pipeline_factory=None,
                            tag_pixel_diameter=50.0, timestamps=None,
                            start_timestamp=None, fps=3.0, cam_id=0,
                            verbose=False, n_frames=None, progress="tqdm",
                            calculate_confidences=True, confidence_filter=None,
                           use_parallel_jobs=False):
    """Takes a video or a sequence of images, applies the BeesBook tag detection pipeline on the video and puts the results in a pandas DataFrame.
    Note that this is not optimized for high performance cluster computing but instead for end-user usability.

    Arguments:
        source_path: string
            Path to video file or list of paths to images.
        source_type: ("auto", "video", "image")
            Type of media file behind 'source_path'.
        pipeline: bb_pipeline.pipeline.Pipeline
            Instantiated pipeline object used for localizing and decoding tags.
            Can be None.
        pipeline_factory: callable
            Used with pipeline=None. Function that returns a unique object suitable for the 'pipeline' argument.
            If given, the tag detection will be multithreaded with one pipeline object per threads.
        tag_pixel_diameter: float
            Diameter of the outer border of a flat BeesBook tag in the image (in pixels).
        timestamps: list(float)
            List of timestamps of a length that corresponds to the number of frames in the video.
            Can be None.
        start_timestamps: float
            Used with timestamps=None. Timestamp of first frame of the video. Defaults to 0.
        fps: float
            Used with timestamps=None. Frames per second in the video to auto-generate timestamps.
        cam_id: int
            Additional identifier that can be used to differentiate between recording instances.
            Will be used in the identifiers of the detections. Defaults to 0.
        verbose: bool
            Whether to display the detections in an image.
            Note that the pipeline has to be set to provide the CrownOverlay output.
        n_frames: int
            Whether to only return results for the first n_frames frames of the video.
        progress: (None, "tqdm", "tqdm_notebook")
            Whether to draw a progress bar.
        calculate_confidences: bool
            Whether to include a 'confidence' column in the results. Defaults to True.
        confidence_filter: float
            If set, specifies a threshold that disregards all detections from the results
            with a lower confidence value.
        use_parallel_jobs: bool
            Whether to use threading/multiprocessing.
    """
    if source_type == "auto":
        if isinstance(source_path, str):
            if source_path.endswith("jpg"):
                source_type = "image"
            else:
                source_type = "video"
        elif isinstance(source_path, list):
            source_type = "image"
    calculate_confidences = calculate_confidences or confidence_filter is not None
    scale = 50.0 / tag_pixel_diameter
    if pipeline is None and pipeline_factory is None:
        pipeline = get_default_pipeline()

    if timestamps is None:
        def generate_timestamps(start_from):
            def gen():
                i = 0.0
                while True:
                    yield start_from + i * 1.0 / fps
                    i += 1.0
            yield from gen()
        timestamps = generate_timestamps(start_timestamp or 0.0)
        
    import skimage.transform
    import pipeline as bb_pipeline
        
    interrupted = False
    def interruptable_frame_generator(gen):
        nonlocal interrupted
        for g in gen:
            yield g
            if interrupted:
                print("Stopping early...")
                break
    
    def get_frames_from_images():
        import skimage.io
        for idx, (path, ts) in enumerate(zip(interruptable_frame_generator(source_path), timestamps)):
            im = skimage.io.imread(path, as_gray=True)            
            im = (skimage.transform.rescale(im, scale, order=1, multichannel=False, anti_aliasing=False, mode='constant') * 255).astype(np.uint8)
            yield idx, ts, im

            if n_frames is not None and idx >= n_frames - 1:
                break

    def get_frames_from_video():
        frames_generator = bb_pipeline.io.raw_frames_generator(source_path, format=None)
        
        for idx, (im, ts) in enumerate(zip(interruptable_frame_generator(frames_generator), timestamps)):
            im = (skimage.transform.rescale(im, scale, order=1, multichannel=False, anti_aliasing=False, mode='constant') * 255).astype(np.uint8)
            yield idx, ts, im
            
            if n_frames is not None and idx >= n_frames - 1:
                break
    
    def get_detections_from_frame(idx, ts, im, thread_context=None, **kwargs):
        nonlocal pipeline
        if pipeline is not None:
            pipeline_results = pipeline([im])
        else:
            if "pipeline" not in thread_context:
                thread_context["pipeline"] = pipeline_factory()
            pipeline_results = thread_context["pipeline"]([im])
            
        #confident_ids = [r for c, r in zip(confidences, decoded_ids) if c >= threshold]
        #decimal_ids = set([ids.BeesbookID.from_bb_binary(i).as_ferwar() for i in confident_ids])

        if verbose:
            crowns = pipeline_results[CrownOverlay]
            frame = ResultCrownVisualizer.add_overlay(im.astype(np.float64) / 255, crowns)
            fig, ax = plt.subplots(figsize=(20, 10))
            plt.imshow(frame)
            plt.axis("off")
            plt.show()
        
        frame_id = bb_pipeline.io.unique_id()
        required_results = pipeline_results[PipelineResult]    
        n_detections = required_results.orientations.shape[0]
        detection_ids = ['f{}d{}c{}'.format(frame_id, detection_idx, cam_id) for detection_idx in range(n_detections)]
        decoded_ids = [list(r) for r in list(required_results.ids)]
        
        if n_detections > 0:
            frame_data = {
                "id": detection_ids,
                "localizerSaliency": required_results.saliencies.flatten(),
                "beeID": decoded_ids,
                "xpos": required_results.positions[:, 1] / scale,
                "ypos": required_results.positions[:, 0] / scale,
                "camID": [cam_id] * n_detections, 
                "zrotation": required_results.orientations[:, 0],
                "timestamp": [ts] * n_detections,
                "frameIdx": [idx] * n_detections
            }
            
            if calculate_confidences:
                confidences = np.array([np.product(np.abs(0.5 - np.array(r)) * 2) for r in decoded_ids])
                frame_data["confidence"] = confidences
                if confidence_filter is not None:
                    frame_data = frame_data[frame_data.confidence >= confidence_filter]
        else:
            frame_data = None
        
        
        return idx, frame_id, ts, frame_data
    
    
    progress_bar = None
    if progress == "tqdm":
        import tqdm
        progress_bar = tqdm.tqdm(total=n_frames)
    elif progress == "tqdm_notebook":
        import tqdm
        progress_bar = tqdm.tqdm_notebook(total=n_frames)
        
    def save_frame_data(idx, frame_id, ts, frame_data, **kwargs):
        nonlocal frame_info
        nonlocal video_dataframe
        frame_info.append((idx, frame_id, ts))
        if frame_data is not None:
            video_dataframe.append(pd.DataFrame(frame_data))
        progress_bar.update()
        
    source = get_frames_from_video
    if source_type == "image":
        source = get_frames_from_images
        if isinstance(source_path, str):
            source_path = [source_path]
            
    if use_parallel_jobs:
        thread_context_factory = None
        n_pipeline_jobs = 1
        if not pipeline:
            # The thread context is used to provide each thread with a unique pipeline object.
            class Ctx():
                def __enter__(self):
                    return dict()
                def __exit__(self, *args):
                    pass
                
            n_pipeline_jobs = 4
            thread_context_factory = Ctx

        jobs = bb_trajectory.utils.ParallelPipeline([source, get_detections_from_frame, save_frame_data],
                                                    n_thread_map={0: 1, 1: n_pipeline_jobs},
                                                   thread_context_factory=thread_context_factory)
    else:
        def sequential_jobs():
            for im in source():
                if im is not None:
                    detections = get_detections_from_frame(*im)
                    save_frame_data(*detections)
        jobs = sequential_jobs

    frame_info = []
    video_dataframe = []
    try:
        jobs()
    except KeyboardInterrupt:
        interrupted = True
    frame_info = list(sorted(frame_info))
    video_dataframe = pd.concat(video_dataframe)
    video_dataframe.sort_values("frameIdx", inplace=True)
    
    return frame_info, video_dataframe

class PandasTracker():
    def __init__(self, det_score_fun, frag_score_fun):
        self.det_score_fun = det_score_fun
        self.frag_score_fun = frag_score_fun
        
    def merge_fragments(self, dw_tracks, frame_diff):
        gap = frame_diff - 1
        walker = bb_tracking.tracking.SimpleWalker(dw_tracks, self.frag_score_fun, frame_diff, np.inf)
        merged_fragments = walker.calc_tracks()
        return merged_fragments
        
    def __call__(self, dataframe, max_distance=200, max_track_length=15, **kwargs):
        meta_keys = kwargs["meta_keys"] if "meta_keys" in kwargs else dict()
        if "camID" in dataframe.columns:
            meta_keys["camID"] = "camId"
        
        dw_final = bb_tracking.data.DataWrapperPandas(dataframe, meta_keys=meta_keys, **kwargs)
        
        walker_final = bb_tracking.tracking.SimpleWalker(dw_final, self.det_score_fun, 1, max_distance)
        tracks_final  = walker_final.calc_tracks()
        del(walker_final)
        self.tracks_final = tracks_final
        self.dw_final = dw_final
        
        dw_tracks_final = bb_tracking.data.DataWrapperTracks(tracks_final, dw_final.cam_timestamps)
        del(dw_final)

        tracks_final = self.merge_fragments(dw_tracks_final, max_track_length)
        del(dw_tracks_final)

        return tracks_final
    
    @classmethod
    def from_dict(cls, fun_dict):
        return PandasTracker(fun_dict["det_score_fun"], fun_dict["frag_score_fun"])
    @classmethod
    def from_dill_dict(cls, filename):
        import dill
        with open(filename, 'rb') as f:
            fun_dict = dill.load(f)
            return cls.from_dict(fun_dict)
    
    def to_dill_dict(self, filename):
        if self.det_score_fun is None or self.frag_score_fun is None:
            raise ValueError("Tracker not instantiated/trained.")
        import dill
        with open(filename, 'wb') as f:
            d = dict(det_score_fun = self.det_score_fun,
                     frag_score_fun = self.frag_score_fun)
            dill.dump(d, f)
                

def track_detections_dataframe(dataframe, tracker=None,
                               confidence_filter_detections=None,
                              confidence_filter_tracks=None,
                              coordinate_scale=None,
                              use_weights_for_tracked_id=True):
    """Takes a dataframe as generated by detect_markers_in_video and returns connected tracks.
    
    Arguments:
        dataframe: pandas.DataFrame
            Detections per frame as returned by detect_markers_in_video.
        tracker: string or callable or dict
            Either an instantiated PandasTracker
            or the input to either PandasTracker.from_dill_dict or PandasTracker.from_dict.
        confidence_filter_detections: float
            If given, disregards detections with a lower confidence value prior to tracking.
        confidence_filter_tracks: float
            If given, disregards tracks with a lower mean confidence after tracking.
        coordinate_scale: float
            If given, scales the detection coordinates prior to tracking.
            Can be used to ensure that the input coordinates are on a similar scale as the
            data that has been originally used to train the tracker.
            Higher value means that distances are more important.
        use_weights_for_track_id: bool
            Whether to use a weighted average for the tracked ID based on the detection confidences.
            If false, the median will be used.
    Returns:
        pandas.DataFrame similar to the input 'dataframe', containing the additional columns:
        "track_id", "track_confidence", "bee_id".
    """
    if tracker is not None:
        if isinstance(tracker, dict):
            tracker = PandasTracker.from_dict(tracker)
        elif isinstance(tracker, str):
            tracker = PandasTracker.from_dill_dict(tracker)
    
    has_confidences = "confidence" in dataframe.columns
    if confidence_filter_detections and confidence_filter_detections > 0.0:
        if not has_confidences:
            raise ValueError("Can not filter for confidence ('confidence' column not available).")    
        filtered = dataframe[dataframe.confidence >= confidence_filter_detections]
    else:
        filtered = dataframe
    
    if coordinate_scale is not None and coordinate_scale != 1.0:
        filtered = filtered.copy()
        filtered.xpos = filtered.xpos * coordinate_scale
        filtered.ypos = filtered.ypos * coordinate_scale
        
    tracks = tracker(filtered)
    detection_ids = {dataframe.id.values[idx]: idx for idx in range(dataframe.shape[0])}
    tracked_ids = [None] * dataframe.shape[0]
    track_ids = [None] * dataframe.shape[0]
    track_confidences = [None] * dataframe.shape[0]
    
    for track in tracks:
        median_id = np.nan * np.zeros(shape=(len(track.ids), len(track.meta["detections"][0].beeId)))
        for i, detection in enumerate(track.meta["detections"]):
            median_id[i, :] = detection.beeId
        if use_weights_for_tracked_id:
            confidences = np.product(np.abs(0.5 - median_id) * 2, axis=1)
            median_id = np.average(median_id, axis=0, weights=confidences)
        else:
            median_id = np.median(median_id[:, 1:], axis=0)
        track_confidence = np.product(np.abs(0.5 - median_id) * 2)
        tracked_id = bb_utils.ids.BeesbookID.from_bb_binary(median_id).as_ferwar()
        for ID in track.ids:
            idx = detection_ids[ID]
            tracked_ids[idx] = tracked_id
            track_ids[idx] = track.id
            track_confidences[idx] = track_confidence
            
    dataframe = dataframe.copy()
    dataframe["track_id"] = track_ids
    dataframe["track_confidence"] = track_confidences
    dataframe["bee_id"] = tracked_ids
    
    if confidence_filter_tracks is not None and confidence_filter_tracks > 0.0:
        dataframe = dataframe[~pd.isnull(dataframe.track_confidence)]
        dataframe = dataframe[dataframe.track_confidence >= confidence_filter_tracks]
    return dataframe

def plot_tracks(detections, tracks, fig=None, use_individual_id_for_color=True):
    import matplotlib.cm
    import matplotlib.pyplot as plt
    
    new_figure = fig is None
    if new_figure:
        fig = plt.figure(figsize=(20, 20))
    if not use_individual_id_for_color:
        color_count = len(tracks.track_id.unique())
    else:
        id_to_color = list(set(tracks.bee_id.unique()))
        color_count = len(id_to_color)
    colors = np.linspace(0.0, 1.0, num=color_count)
    
    cm = matplotlib.cm.hsv
    for track_idx, (track_id, df) in enumerate(tracks.groupby("track_id")):
        df = df.sort_values("timestamp")
        color_idx = track_idx
        if use_individual_id_for_color:
            color_idx = id_to_color.index(df.bee_id.iloc[0])
        color = cm(colors[color_idx])
        bee_id = int(df.bee_id.values[0])
        plt.scatter(df.xpos, df.ypos, c=color, alpha=0.2)
        plt.plot(df.xpos.values, df.ypos.values, "k--")
        plt.text(float(df.xpos.values[0]), float(df.ypos.values[0]), "#{:02d}({})".format(track_idx, bee_id),
                 color=color)
    if new_figure:
        plt.show()

def display_tracking_results(path, frame_info, detections, tracks, image=None, fig_width=12):
    import matplotlib.pyplot as plt

    track_count = len(tracks.track_id.unique())
    print("Found {} detections belonging to {} unique tracks and {} individuals.".format(
            detections.shape[0],
            track_count,
            len(tracks.bee_id.unique())))
    tracks_available = tracks.shape[0] > 0
    if not tracks_available:
        print("No tracks found. You might need to lower the confidence thresholds.")
    
    # Print result image first.
    if image is None and path is not None:
        from ..io.videos import get_first_frame_from_video
        image = get_first_frame_from_video(path)
    
    fig = plt.figure(figsize=(fig_width, fig_width))
    if image is not None:
        plt.imshow(image)
    plot_tracks(detections, tracks, fig=fig)
    plt.show()
    
    print("Detection/track statistics:")
    # Show histogram of confidences, allowing to tune the tracking paramters.
    if "confidence" in detections.columns:
        plt.figure(figsize=(fig_width, 2))
        plt.hist(detections.confidence)
        plt.xlabel("Detection confidence\n(decoded ID correctness)")
        plt.xlim(-0.1, 1.1)
        plt.show()
    
    plt.figure(figsize=(fig_width, 2))
    plt.hist(tracks.track_confidence[~pd.isnull(tracks.track_confidence)])
    plt.xlabel("Track confidence\n(track ID correctness)")
    plt.xlim(-0.1, 1.1)
    plt.show()
    
    # Show statistics about connected tracks
    if tracks_available:
        track_data = defaultdict(list)
        for track_id, df in tracks.groupby("track_id"):
            df = df.sort_values("timestamp")
            frame_differences = np.diff(df.frameIdx.values)
            xy = df[["xpos", "ypos"]].values
            distances = np.linalg.norm(xy[1:, :] - xy[:-1, :], axis=1)
            distances = distances[frame_differences == 1]
            time_differences = np.diff(df.timestamp.values)
            
            
            track_data["distances"] += list(distances)
            track_data["gap_length_seconds"] += list(time_differences)
            track_data["gap_length_frames"] += list(frame_differences)
            
        plt.figure(figsize=(fig_width, 2))
        plt.hist(track_data["distances"])
        plt.xlabel("Pixel distances (Speed per frame)\nof tracked individuals (gaps ignored)")
        plt.show()
        
        fig, ax = plt.subplots(figsize=(fig_width, 2))
        ax.hist(track_data["gap_length_seconds"])
        ax.set_xlabel("Seconds")
        #plt.xlabel("Gap length of individuals in successive frames")
        #plt.show()
        #fig, ax = plt.subplots(figsize=(fig_width, 2))
        ax = ax.twiny()
        ax.hist(track_data["gap_length_frames"])
        ax.set_xlabel("Frames")
        plt.title("Gap length of individuals in successive frames")
        plt.tight_layout()
        plt.show()
        
    # Print information about the identified individuals based on their ID.
    print("Individual statistics:")
    for idx, (bee_id, df) in enumerate(tracks.groupby("bee_id")):
        bee_id = int(bee_id)
        if df.shape[0] <= 3:
            print("Individual {} in track {} has only {} detection(s). Skipping.".format(
                bee_id, df.track_id.values[0], df.shape[0]))
            continue
        track_count = len(df.track_id.unique())
        df = df.sort_values("timestamp")
        xy = df[["xpos", "ypos"]].values
        distances = np.linalg.norm(xy[1:, :] - xy[:-1, :], axis=1)
        time_differences = np.diff(df.timestamp.values)
        distances /= time_differences
        frame_idx = df.frameIdx.values[1:]
        
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(fig_width, 2) , gridspec_kw = {'width_ratios':[(fig_width - 2), 1]})
        ax.plot(frame_idx, distances, "k-", label="Speed")
        ax.set_ylabel("Pixels per second")
        ax.set_xlabel("Frames")
        ax.legend(loc="upper left")
        ax = ax.twinx()
        ax.plot(frame_idx, df.confidence.values[1:], "g--", label="Decoder Confidence")
        ax.plot(frame_idx, df.track_confidence.values[1:], "b:", label="Track Confidence", alpha=0.5)
        ax.set_ylabel("Confidence\nHigher is better")
        ax.legend(loc="upper right")
        ax.set_xlim(0, tracks.frameIdx.max())
        plt.title("Individual {} ({} different tracks)".format(bee_id, track_count))
        draw_ferwar_id_on_axis(bee_id, ax2)
        plt.tight_layout()
        plt.show()