from . import base
import numpy as np
import pandas as pd
import psycopg2.extras

def get_frame_metadata(frames, cursor=None, cursor_is_prepared=False, return_dataframe=True):
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
    Returns:
        annotated_frames: pandas.DataFrame
            DataFrame with the columns frame_id, timestamp, frame_idx, fc_id, cam_id.
            Note that 'frame_idx' is the index of the respective video (fc_id).

            If 'return_dataframe' is true, returns a list with one tuple per row of the data frame.
    """
    if cursor is None:
        with base.get_database_connection("Frame metadata") as con:
            return get_frame_metadata(frames, cursor=con.cursor(), cursor_is_prepared=False, return_dataframe=return_dataframe)
    if not cursor_is_prepared:
        cursor.execute("PREPARE get_frame_metadata AS "
           "SELECT frame_id, timestamp, index, fc_id FROM plotter_frame WHERE frame_id = ANY($1)")
        cursor.execute("PREPARE get_frame_id_for_container AS "
          "SELECT CAST(SUBSTR(video_name, 5, 1) AS INT) FROM plotter_framecontainer "
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
        cursor.execute("EXECUTE get_frame_id_for_container (%s)", (fc_id,))
        cam_id = cursor.fetchone()[0]
        required_frame_containers[fc_id] = cam_id
    
    annotated_frames = []
    for idx, frame_id in enumerate(frames):
        meta = list(frame_id_dict[frame_id])
        #meta[0] = int(meta[0])
        meta.append(required_frame_containers[meta[-1]])
        annotated_frames.append(meta)
    
    if return_dataframe:
        annotated_frames = pd.DataFrame(annotated_frames, columns=("frame_id", "timestamp", "frame_idx", "fc_id", "cam_id"))

    return annotated_frames
