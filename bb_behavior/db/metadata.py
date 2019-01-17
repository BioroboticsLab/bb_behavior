from . import base
import numpy as np
import pandas as pd
import psycopg2.extras

def get_frame_metadata(frames, cursor=None, cursor_is_prepared=False, return_dataframe=True, include_video_name=False):
    """Takes a list of frame IDs and fetches additional data such as timestamps, cam_id, frame container ID, ...

    Arguments:
        frames: list(int)
            Database frame IDs for which to fetch the metadata.
        cursor: psycopg2.cursor
            Optional. Cursor connected to the database.
        cursor_is_prepared: bool
            Whether the cursor already has the SQL statements prepared.
        return_dataframe: bool
            Whether to return a pandas.DataFrame (instead of a list of tuples).
        include_video_name: bool
            Whether to include the original filename of the video in the results.
    Returns:
        annotated_frames: pandas.DataFrame
            DataFrame with the columns frame_id, timestamp, frame_idx, fc_id, cam_id, [video_name].
            Note that 'frame_idx' is the index of the respective video (fc_id).

            If 'return_dataframe' is true, returns a list with one tuple per row of the data frame.
    """
    if cursor is None:
        with base.get_database_connection("Frame metadata") as con:
            return get_frame_metadata(frames, cursor=con.cursor(), cursor_is_prepared=False, return_dataframe=return_dataframe, include_video_name=include_video_name)
    if not cursor_is_prepared:
        cursor.execute("PREPARE get_frame_metadata AS "
           "SELECT frame_id, timestamp, index, fc_id FROM plotter_frame WHERE frame_id = ANY($1)")
        cursor.execute("PREPARE get_frame_container_info AS "
          "SELECT video_name FROM plotter_framecontainer "
          "WHERE id = $1 LIMIT 1")

    # Fetch the widely available metadata.
    frames = list(map(int, frames))
    cursor.execute("EXECUTE get_frame_metadata (%s)", (frames,))
    metadata = cursor.fetchall()
    frame_id_dict = dict()
    required_frame_containers = dict()
    for m in metadata:
        frame_id_dict[m[0]] = m
        required_frame_containers[m[-1]] = None
    
    # And add the frame_ids.
    for fc_id in required_frame_containers.keys():
        cursor.execute("EXECUTE get_frame_container_info (%s)", (fc_id,))
        video_name = cursor.fetchone()[0]
        cam_id = int(video_name[4])
        required_frame_containers[fc_id] = (cam_id, video_name)
    
    annotated_frames = []
    for idx, frame_id in enumerate(frames):
        if frame_id not in frame_id_dict:
            import warnings
            warnings.warn("Frame ID {} not found in database.".format(frame_id))
            continue
        meta = list(frame_id_dict[frame_id])
        #meta[0] = int(meta[0])
        if include_video_name:
            meta.extend(required_frame_containers[meta[-1]])
        else:
            meta.append(required_frame_containers[meta[-1]][0])
        annotated_frames.append(meta)
    
    if return_dataframe:
        columns = ("frame_id", "timestamp", "frame_idx", "fc_id", "cam_id")
        if include_video_name:
            columns = columns + ("video_name",)
        annotated_frames = pd.DataFrame(annotated_frames, columns=columns)

    return annotated_frames

def get_alive_bees(dt_from, dt_to, cursor=None):
    """Returns all bees for a time window that have been tagged and did not die yet.

    Arguments:
        dt_from: datetime.datetime
            Begin of the period (inclusive).
        dt_to: datetime.datetime
            End of the period (exclusive).
        cursor: psycopg2.cursor
            Optional. Connected database cursor.
    Returns:
        bee_ids: set(int)
            All bee IDs (ferwar format) that are alive between the given timestamps.
    """
    if cursor is None:
        with base.get_database_connection("Frame metadata") as con:
            return get_alive_bees(dt_from, dt_to, cursor=con.cursor())
    cursor.execute("SELECT bee_id from alive_bees_2016 WHERE timestamp >= %s and timestamp < %s ", (dt_from, dt_to))
    bee_ids = {result[0] for result in cursor.fetchall()}
    return bee_ids
