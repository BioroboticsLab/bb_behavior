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

from collections import defaultdict, namedtuple
import datetime, pytz
import dill
# import joblib
import xgboost as xgb
import math
import numpy as np
import pandas as pd
import scipy.spatial

import bb_tracking.data_walker
import bb_tracking.features
import bb_tracking.models
import bb_tracking.repository_tracker
import bb_tracking.types

import bb_utils.ids

def make_scaling_homography_fn(pixels_to_millimeter_ratio):
    """Returns a homography generator function that always returns a homography with a fixed pixel-to-mm scaling.

    Arguments:
        pixels_to_millimeter_ratio: float
            Millimeters / pixels for the video data.
    """
    def scaling_homography_fn(cam_id, dt):
        return np.array([[pixels_to_millimeter_ratio, 0.0, 0.0],
                         [0.0, pixels_to_millimeter_ratio, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
    return scaling_homography_fn

def get_default_tracker_settings(detection_model_path, tracklet_model_path,
        detection_classification_threshold=0.6, tracklet_classification_threshold=0.5):

    #detection_model = bb_tracking.models.XGBoostRankingClassifier.load(detection_model_path)
    #tracklet_model = bb_tracking.models.XGBoostRankingClassifier.load(tracklet_model_path)

    # with open(detection_model_path, "rb") as f:
    #     detection_model = joblib.load(f)
    # with open(tracklet_model_path, "rb") as f:
    #     tracklet_model = joblib.load(f)

    # Load the detection model
    detection_model_booster = xgb.Booster()
    detection_model_booster.load_model(detection_model_path)
    # Wrap the Booster in an XGBClassifier
    detection_model = xgb.XGBClassifier()
    detection_model._Booster = tracklet_model_booster

    # Load the tracklet model as a Booster object
    tracklet_model_booster = xgb.Booster()
    tracklet_model_booster.load_model(tracklet_model_path)
    # Wrap the Booster in an XGBClassifier
    tracklet_model = xgb.XGBClassifier()
    tracklet_model._Booster = tracklet_model_booster


    tracklet_kwargs = dict(
        max_distance_per_second = 30.0,
        n_features=18,
        detection_feature_fn=bb_tracking.features.get_detection_features,
        detection_cost_fn=lambda f: 1 - detection_model.predict_proba(f)[:, 1],
        max_cost=1.0 - detection_classification_threshold
        )

    track_kwargs = dict(
        max_distance_per_second = 20.0,
        max_seconds_gap=4.0,
        n_features=14,
        tracklet_feature_fn=bb_tracking.features.get_track_features,
        tracklet_cost_fn=lambda f: 1 - tracklet_model.predict_proba(f)[:, 1],
        max_cost=1.0 - tracklet_classification_threshold
        )
    return dict(tracklet_kwargs=tracklet_kwargs, track_kwargs=track_kwargs)

def iterate_dataframe_as_detections(dataframe, H):
    """Takes a dataframe (e.g. as returned by pipeline.detect_markers_in_video) and yields the rows in a tracking-friendly format.
    """
    if dataframe is None:
        return
    if dataframe.shape[0] == 0:
        return

    Bee = namedtuple("bee", ("xpos", "ypos", "idx", "localizerSaliency"))

    dataframe = dataframe.sort_values("timestamp")

    detection_type_map = dict(
        TaggedBee=bb_tracking.types.DetectionType.TaggedBee,
        UnmarkedBee=bb_tracking.types.DetectionType.UntaggedBee,
        BeeInCell=bb_tracking.types.DetectionType.BeeInCell,
        UpsideDownBee=bb_tracking.types.DetectionType.BeeOnGlass,
    )

    assert len(dataframe.camID.unique()) == 1
    cam_id = dataframe.camID.values[0]

    for (ts, frame_id), frame_df in dataframe.groupby(["timestamp", "frameId"], sort=True):
        frame_detections = []
        frame_datetime = None
        for (xpos, ypos, idx, localizer_saliency, timestamp, orientation, detection_type, bit_probabilities) in \
            frame_df[["xpos", "ypos", "detection_index", "localizerSaliency",
                       "timestamp", "zrotation", "detection_type", "beeID"]].itertuples(index=False):

            xpos = np.float64(xpos)
            ypos = np.float64(ypos)
            orientation = np.float64(orientation)
            timestamp = np.float64(timestamp)

            bee = Bee(xpos, ypos, idx, localizer_saliency)
            if type(bit_probabilities) is list:
                bit_probabilities = np.array(bit_probabilities)
            elif isinstance(bit_probabilities, np.ndarray):
                pass
            else:
                assert (bit_probabilities is None) or math.isnan(bit_probabilities)
                bit_probabilities = None

            detection_type = detection_type_map[detection_type]
            detection = bb_tracking.data_walker.make_detection(bee, H=H,
                                       frame_id=frame_id, timestamp=timestamp, orientation=float(orientation),
                                       detection_type=detection_type, bit_probabilities=bit_probabilities,
                                       no_datetime_timestamps=True)
            frame_detections.append(detection)

            if frame_datetime is None:
                frame_datetime = datetime.datetime.fromtimestamp(timestamp, tz=pytz.UTC)

        xy = [(detection.x_hive, detection.y_hive) for detection in frame_detections]
        frame_kdtree = scipy.spatial.cKDTree(xy)
        yield (cam_id, frame_id, frame_datetime, frame_detections, frame_kdtree)



def track_detections_dataframe(dataframe_or_generator,
                                tracker_settings=None, use_threading=False,
                                homography_fn=None, homography_scale=None,
                                cam_id=None,
                                tracker_settings_kwargs=dict()):
    """Takes a dataframe as generated by detect_markers_in_video and returns connected tracks.

    Arguments:
        dataframe_or_generator: pandas.DataFrame or generator
            Detections per frame as returned by detect_markers_in_video.
            Can be a generator yielding multiple dataframes for a single camera (must be ordered by time).
        tracker_settings: dict
            Dictionary containing the keys 'detection_kwargs' and 'tracklet_kwargs'.
        homography_fn: callable
            Optional. A function taking a cam_id and datetime object and returning a pixels-to-mm homography.
            If None, homography_scale will be used to get a simple scaling homography.
        homography_scale: float
            Optional. If homography_fn is not given, this pixels-to-millimeters ratio is used as a homography.
    Returns:
        pandas.DataFrame similar to the input, containing the additional columns:
        "track_id", "bee_id".
    """
    if tracker_settings is None:
        tracker_settings = get_default_tracker_settings(**tracker_settings_kwargs)

    if homography_fn is None:
        if homography_scale is None:
            raise ValueError("Either homography_fn or homography_scale must be given.")
        homography_fn = make_scaling_homography_fn(homography_scale)

    if type(dataframe_or_generator) is pd.DataFrame:
        dataframe_or_generator = (dataframe_or_generator,)

    def iterate_dataframes():
        for df in dataframe_or_generator:
            if df is None:
                continue
            if df.shape[0] == 0:
                continue
            dt = df.timestamp.values[0]
            assert isinstance(dt, (np.floating, float))
            homography = homography_fn(cam_id, dt)
            yield from iterate_dataframe_as_detections(df, homography)

    tracker = bb_tracking.repository_tracker.CamDataGeneratorTracker(
        iterate_dataframes(),
        cam_ids=(cam_id,),
        progress_bar=None, use_threading=use_threading,
        **tracker_settings)

    tracks_dataframe = []
    for track in tracker:
        for detection in track.detections:
            tracks_dataframe.append(dict(
                bee_id=track.bee_id, bee_id_confidence=track.bee_id_confidence,
                track_id=track.id,
                x_pixels=detection.x_pixels, y_pixels=detection.y_pixels, orientation_pixels=detection.orientation_pixels,
                x_hive=detection.x_hive, y_hive=detection.y_hive, orientation_hive=detection.orientation_hive,
                timestamp_posix=detection.timestamp_posix, timestamp=detection.timestamp,
                frame_id=detection.frame_id, detection_type=detection.detection_type, detection_index=detection.detection_index,
                detection_confidence=bb_tracking.features.get_detection_confidence(detection)))
    if len(tracks_dataframe) == 0:
        return None
    return pd.DataFrame(tracks_dataframe)

def plot_tracks(tracks, fig=None, use_individual_id_for_color=True, use_pixel_coordinates=False, draw_labels=True, draw_lines=True):
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
    colors = np.linspace(0.0, 1.0, num=color_count+1)

    x_coord, y_coord = "x_hive", "y_hive"
    if use_pixel_coordinates:
        x_coord, y_coord = "x_pixels", "y_pixels"

    cm = matplotlib.cm.hsv
    for track_idx, (track_id, df) in enumerate(tracks.groupby("track_id")):
        df = df.sort_values("timestamp")
        color_idx = track_idx
        if use_individual_id_for_color:
            color_idx = id_to_color.index(df.bee_id.iloc[0])
        color = cm(colors[color_idx])
        bee_id = int(df.bee_id.values[0])
        plt.scatter(df[x_coord], df[y_coord], c=np.array([color]), alpha=0.2)
        if draw_lines:
            plt.plot(df[x_coord].values, df[y_coord].values, "k--")
        if draw_labels:
            plt.text(float(df[x_coord].values[0]), float(df[y_coord].values[0]), "#{:02d}({})".format(track_idx, bee_id),
                 color=color)
    if new_figure:
        plt.show()

def display_tracking_results(tracks, path=None, image=None, fig_width=12, track_plot_kws=dict()):
    import matplotlib.pyplot as plt

    track_count = len(tracks.track_id.unique())
    print("Found {} detections belonging to {} unique tracks and {} individuals.".format(
            tracks.shape[0],
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
        plt.imshow(image, cmap="gray")
    plot_tracks(tracks, fig=fig, use_pixel_coordinates=True, **track_plot_kws)
    plt.show()

    print("Detection/track statistics:")
    # Show histogram of confidences, allowing to tune the tracking paramters.
    if "detection_confidence" in tracks.columns:
        plt.figure(figsize=(fig_width, 2))
        plt.hist(tracks.detection_confidence)
        plt.xlabel("Detection confidence\n(decoded ID correctness)")
        plt.xlim(-0.1, 1.1)
        plt.show()

    if "track_confidence" in tracks.columns:
        plt.figure(figsize=(fig_width, 2))
        plt.hist(tracks.track_confidence[~pd.isnull(tracks.track_confidence)])
        plt.xlabel("Track confidence\n(track ID correctness)")
        plt.xlim(-0.1, 1.1)
        plt.show()

    # Show statistics about connected tracks
    if tracks_available:
        track_data = defaultdict(list)
        for track_id, df in tracks.groupby("track_id"):
            if df.shape[0] < 2:
                continue
            df = df.sort_values("timestamp_posix")
            xy = df[["x_hive", "y_hive"]].values
            distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            time_differences = np.diff(df.timestamp_posix.values)
            assert (time_differences.shape[0] == distances.shape[0])
            track_data["speed"] += list(distances / time_differences)
            track_data["gap_length_seconds"] += list(time_differences)

        plt.figure(figsize=(fig_width, 2))
        plt.hist(track_data["speed"], bins=50)
        plt.xlabel("Speed per frame in mm/s of tracked individuals")
        plt.show()

        fig, ax = plt.subplots(figsize=(fig_width, 2))
        ax.hist(track_data["gap_length_seconds"], bins=50)
        ax.set_xlabel("Seconds")
        #plt.xlabel("Gap length of individuals in successive frames")
        #plt.show()
        #fig, ax = plt.subplots(figsize=(fig_width, 2))
        plt.title("Gap length of individuals in successive frames")
        plt.tight_layout()
        plt.show()

    # Print information about the identified individuals based on their ID.
    min_timestamp, max_timestamp = tracks.timestamp_posix.min(), tracks.timestamp_posix.max()
    print("Individual statistics:")
    for idx, (bee_id, df) in enumerate(tracks.groupby("bee_id")):
        bee_id = int(bee_id)
        if df.shape[0] <= 3:
            print("Individual {} in track {} has only {} detection(s). Skipping.".format(
                bee_id, df.track_id.values[0], df.shape[0]))
            continue
        track_count = len(df.track_id.unique())
        df = df.sort_values("timestamp_posix")
        xy = df[["x_hive", "y_hive"]].values
        distances = np.linalg.norm(xy[1:, :] - xy[:-1, :], axis=1)
        time_differences = np.diff(df.timestamp_posix.values)
        distances /= time_differences
        timestamps = df.timestamp_posix.values[1:]

        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(fig_width, 2) , gridspec_kw = {'width_ratios':[(fig_width - 2), 1]})
        ax.plot(timestamps, distances, "k-", label="Speed")
        ax.set_ylabel("mm/s")
        ax.set_xlabel("Timestamp")
        ax.legend(loc="upper left")
        ax.set_xlim(min_timestamp, max_timestamp)
        if "detection_confidence" in df:
            ax = ax.twinx()
            ax.plot(timestamps, df.detection_confidence.values[1:], "g--", label="Decoder Confidence")
            if "track_confidence" in df:
                ax.plot(timestamps, df.track_confidence.values[1:], "b:", label="Track Confidence", alpha=0.5)
            ax.set_xlim(min_timestamp, max_timestamp)
            ax.set_ylabel("Confidence\nHigher is better")
            ax.legend(loc="upper right")

        plt.title("#{} (from {} tracks)".format(bee_id, track_count))
        draw_ferwar_id_on_axis(bee_id, ax2)
        plt.tight_layout()
        plt.show()
