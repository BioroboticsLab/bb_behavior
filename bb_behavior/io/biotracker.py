import json
import math
import numpy as np
import pandas as pd
from ..db.base import get_database_connection
import datetime

def save_tracks(track_map, filename, timestamps=None, frame_ids=None, track_names=None, meta=None):
    meta = meta or dict()
    if timestamps is not None:
        meta["timestamps"] = timestamps
    if frame_ids is not None:
        meta["frame_ids"] = [int(fid) for fid in frame_ids]
    
    trajectories = dict(objectName="", valid="true", id="0", childNodes="", meta=meta)
    
    for track_idx, (track_id, xy) in enumerate(track_map.items()):
        track_name = str(track_id)
        if track_names and track_id in track_names:
            track_name = track_names[track_id]
        traj = dict(valid="true", id=str(track_id), childNodes="", objectName=str(track_name))
        trajectories["Trajectory_{}".format(track_idx)] = traj
        
        object_names = None
        if type(xy) is tuple:
            object_names, xy = xy

        for node_idx, frame_xy in enumerate(xy):
            x, y = frame_xy[0], frame_xy[1]
            if np.isnan(x):
                continue
            node = dict(valid="true", id=str(track_id), coordinateUnit="px",
                       x="{:1.3f}".format(x), y="{:1.3f}".format(y), time="0", timeString="")
            if len(frame_xy) > 2:
                rad = frame_xy[2]
                deg = math.degrees(rad)
                node["rad"] = "{:1.3f}".format(rad)
                node["deg"] = "{:1.3f}".format(deg)
            if timestamps is not None:
                node["time"] = str(timestamps[node_idx])
                node["timeString"] = str(datetime.datetime.utcfromtimestamp(timestamps[node_idx]))
            if object_names is not None:
                object_name = object_names[node_idx]
            else:
                object_name = track_name
                if frame_ids is not None:
                    object_name = "{} {}".format(frame_ids[node_idx], track_name)
            node["objectName"] = object_name
            
            traj["Element_{}".format(node_idx)] = node
    with open(filename, "w") as f:
        json.dump(trajectories, f)

def load_tracks(tracks_filename, using_bee_ids=True):
    """
    Arguments:
        tracks_filename: string
            File path of the .json track data loaded/saved by the biotracker.

    Returns:
        track_map: dict(track_id=list(dict(object_name=string, x=string, y=string))))
    """
    with open(tracks_filename) as f:
        tracking_data = json.load(f)

    track_map = dict()
    for key in tracking_data:
        if key == "meta":
            track_map["frame_ids"] = tracking_data["meta"]["frame_ids"]
            track_map["timestamps"] = tracking_data["meta"]["timestamps"]
        if not key.startswith("Trajectory_"):
            continue
        trajectory = tracking_data[key]
        available_frame_indices = list(sorted([int(e.split("_")[1]) for e in trajectory.keys() if e.startswith("Element_")]))
        highest_valid_frame_number = available_frame_indices[-1]
        
        trajectory_list = []
        object_id = None
        for frame_index in range(highest_valid_frame_number + 1):
            node_name = "Element_{}".format(frame_index)
            x, y, object_name, valid = None, None, None, False

            if node_name in trajectory:
                valid = True
                track_frame = trajectory[node_name]

                ID = int(track_frame["id"])
                assert object_id is None or object_id == ID
                object_id = ID

                object_name = track_frame["objectName"]
                x, y = track_frame["x"], track_frame["y"]

            trajectory_list.append(dict(object_name=object_name, x=x, y=y, frame_index=frame_index, valid=valid))
        if object_id is None:
            continue
        assert object_id not in track_map

        track_map[object_id] = trajectory_list
    
    return track_map

def load_annotations(annotations_filename):
    with open(annotations_filename, "r") as f:
        annotation_data = json.load(f)

    annotations_list = []
    for annotation in annotation_data:
        if not "origin_track_id" in annotation:
            continue
        track_id0 = annotation["origin_track_id"]
        start_frame = annotation["start_frame"]
        end_frame = annotation["end_frame"]
        
        track_id1 = None
        if "end_track_id" in annotation:
            track_id1 = annotation["end_track_id"]

        annotations_list.append(dict(
            track_id0 = track_id0,
            track_id1 = track_id1,
            comment = annotation["comment"],
            start_frame = start_frame,
            end_frame = end_frame
        ))
    return pd.DataFrame(annotations_list)

def load_annotated_bee_video(video_filename=None, tracks_filename=None, annotations_filename=None, cursor=None, object_name="bee_id"):
    """
    Arguments:
        video_filename: string
            Path to the video that was annotated using the biotracker.
        tracks_filename: string
            Optional. Path to JSON output of the biotracker. Defaults to the video name with json extension instead of mp4.
        annotations_filename: string
            Optional. Path to JSON annotation output of the biotracker. Defaults to the video name with annotations.json extension instead of mp4.
        cursor: psycopg2.Cursor
            Optional database cursor.
        object_name: "bee_id" or "detection_idx"
            Determines what the data constitutes of. frame_id and bee ID or frame_id and detection index.
    """
    if cursor is None:
        with get_database_connection("biotracker_data") as db:
            cursor = db.cursor()

            cursor.execute("PREPARE get_timestamp AS SELECT timestamp FROM plotter_frame WHERE frame_id=$1")
            cursor.execute("PREPARE get_bee_id_track_id AS SELECT bee_id, track_id FROM bb_detections_2016_stitched WHERE frame_id=$1 AND detection_idx=$2")

            return load_annotated_bee_video(video_filename, tracks_filename, annotations_filename, cursor=cursor, object_name=object_name)

    def get_datetime_for_frame_id(frame_id):
        cursor.execute("EXECUTE get_timestamp (%s)", (frame_id, ))
        ts = cursor.fetchone()[0]
        dt = datetime.datetime.utcfromtimestamp(ts)
        return dt.replace(tzinfo=datetime.timezone.utc)

    def get_bee_id_track_id_for_detection_idx(frame_id, detection_idx):
        if detection_idx is None:
            return None
        cursor.execute("EXECUTE get_bee_id_track_id (%s, %s)", (frame_id, detection_idx))
        bee_id, track_id = cursor.fetchone()[0:2]
        return bee_id, track_id

    has_detection_index_instead_of_id = object_name == "detection_idx"

    if tracks_filename is None:
        tracks_filename = video_filename[:-4] + ".json"
    if annotations_filename is None:
        annotations_filename = video_filename + ".annotations.json"
    
    track_map = load_tracks(tracks_filename)
    all_frame_ids = track_map["frame_ids"]
    annotations_df = load_annotations(annotations_filename)
    
    additional_columns = []
    for i in range(annotations_df.shape[0]):
        track_id0 = annotations_df.track_id0.iloc[i]
        track_id1 = annotations_df.track_id1.iloc[i]
        start_frame = annotations_df.start_frame.iloc[i]
        end_frame = annotations_df.end_frame.iloc[i]
        if end_frame == 0:
            end_frame = start_frame
        # Find out valid frame id / detection idx close to the start of the interaction.
        def get_first_valid_object_name(track, start_idx):
            while True:
                if start_idx < 0:
                    break
                object_name = track[start_idx]["object_name"]
                if object_name == "None":
                    start_idx -= 1
                    continue
                return object_name
            return None
            
        start_frame_id, end_frame_id = all_frame_ids[start_frame], all_frame_ids[end_frame]
        start_datetime = get_datetime_for_frame_id(start_frame_id)
        end_datetime = get_datetime_for_frame_id(end_frame_id)

        object_names = [get_first_valid_object_name(track_map[track_id0], start_frame), None]
        if track_id1 is not None:
            object_names[1] = get_first_valid_object_name(track_map[track_id1], start_frame)

        bee_ids = None, None
        track_ids = None, None
        if has_detection_index_instead_of_id:
            def object_name_to_bee_id_track_id(object_name):
                if object_name is None:
                    return (None, None)
                frame_id, detection_idx = object_name.split(" ")
                return get_bee_id_track_id_for_detection_idx(int(frame_id), int(detection_idx))
            bee_ids_track_ids = list(map(object_name_to_bee_id_track_id, object_names))
            bee_ids, track_ids = zip(*bee_ids_track_ids)
        else:
            bee_ids = [int(object_names[0]), None]
            if object_names[0] is not None:
                bee_ids[1] = int(object_names[1])

        annotation_data = dict(
            start_frame=start_frame,
            end_frame=end_frame,
            comment=annotations_df.comment.iloc[i],
            start_frame_id=start_frame_id,
            end_frame_id=end_frame_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            start_timestamp=start_datetime.timestamp(),
            end_timestamp=end_datetime.timestamp(),
            bee_id0=bee_ids[0],
            bee_id1=bee_ids[1],
            track_id0=track_ids[0],
            track_id1=track_ids[1],
            )

        additional_columns.append(annotation_data)
    annotations_df = pd.DataFrame(additional_columns)
    return annotations_df
