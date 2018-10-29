import json
import pandas as pd
from ..db.base import get_database_connection
import datetime

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
