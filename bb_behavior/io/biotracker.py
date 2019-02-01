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

def load_tracks(tracks_filename):
    with open(tracks_filename) as f:
        tracking_data = json.load(f)

    track_map = dict()
    for key in tracking_data:
        if not key.startswith("Trajectory_"):
            continue
        trajectory = tracking_data[key]


        trajectory_list = []
        track_index = 0
        object_id = None
        while True:
            track_frame_name = "Element_{}".format(track_index)
            track_index += 1
            if not track_frame_name in trajectory:
                break
            track_frame = trajectory[track_frame_name]

            ID = int(track_frame["id"])
            assert object_id is None or object_id == ID
            object_id = ID

            object_name = track_frame["objectName"]
            x, y = track_frame["x"], track_frame["y"]
            trajectory_list.append(dict(object_name=object_name, x=x, y=y))

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
        object_id0 = annotation["origin_track_id"]
        start_frame = annotation["start_frame"]
        end_frame = annotation["end_frame"]
        
        object_id1 = None
        if "end_track_id" in annotation:
            object_id1 = annotation["end_track_id"]

        annotations_list.append(dict(
            object_id0 = object_id0,
            object_id1 = object_id1,
            comment = annotation["comment"],
            start_frame = start_frame,
            end_frame = end_frame
        ))
    return pd.DataFrame(annotations_list)

def load_annotated_bee_video(video_filename=None, tracks_filename=None, annotations_filename=None, cursor=None):
    if cursor is None:
        with get_database_connection("biotracker_data") as db:
            cursor = db.cursor()

            cursor.execute("PREPARE get_timestamp AS SELECT timestamp FROM plotter_frame WHERE frame_id=$1")

            return load_annotated_bee_video(video_filename, tracks_filename, annotations_filename, cursor=cursor)

    def get_datetime_for_frame_id(frame_id):
        cursor.execute("EXECUTE get_timestamp (%s)", (frame_id, ))
        ts = cursor.fetchone()[0]
        dt = datetime.datetime.utcfromtimestamp(ts)
        return dt.replace(tzinfo=datetime.timezone.utc)

    if tracks_filename is None:
        tracks_filename = video_filename[:-4] + ".json"
    if annotations_filename is None:
        annotations_filename = video_filename + ".annotations.json"
    
    track_map = load_tracks(tracks_filename)
    annotations_df = load_annotations(annotations_filename)
    
    additional_columns = []
    for i in range(annotations_df.shape[0]):
        bee_id0 = annotations_df.object_id0.iloc[i]
        start_frame = annotations_df.start_frame.iloc[i]
        end_frame = annotations_df.end_frame.iloc[i]

        start_frame_id = track_map[bee_id0][start_frame]["object_name"].split(" ")[0]
        end_frame_id = track_map[bee_id0][end_frame]["object_name"].split(" ")[0]
        start_datetime = get_datetime_for_frame_id(start_frame_id)
        end_datetime = get_datetime_for_frame_id(end_frame_id)

        additional_columns.append(dict(
            start_frame_id=start_frame_id,
            end_frame_id=end_frame_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            start_timestamp=start_datetime.timestamp(),
            end_timestamp=end_datetime.timestamp()))
    additional_columns = pd.DataFrame(additional_columns)
    annotations_df = pd.concat((annotations_df, additional_columns), axis=1)
    annotations_df.rename(columns=dict(object_id0="bee_id0", object_id1="bee_id1"), inplace=True)
    
    return annotations_df
