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

from .pipeline import get_timestamps_for_beesbook_video,\
        detect_markers_in_beesbook_video,\
        detect_markers_in_video


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