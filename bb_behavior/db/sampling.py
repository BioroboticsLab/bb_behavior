import math
import numba
import numpy as np

from . import base

def sample_frame_ids(n_samples=100, ts_from=None, ts_to=None, cursor=None):
    """Uniformely samples random frame IDs.
    
    Arguments:
        n_samples: Number of frame_ids to return.
        ts_from: Optional. Unix timestamp. Only return frames starting from this timestamp.
        ts_to: Optional. Unix timestamp. Only return frames before this timestamp.
            Required when ts_from is set.
        cursor: Optional. Database cursor connected to the DB.
        
    Returns:
        List containing tuples with (frame_id, timestamp).
    """
    if cursor is None:
        with base.get_database_connection(application_name="sample_frames") as db:
            cursor = db.cursor()
            return sample_frame_ids(n_samples=n_samples, ts_from=ts_from, ts_to=ts_to, cursor=cursor)
    timestamp_condition = " WHERE True "
    query_parameters = None
    if ts_from is not None:
        query_parameters = (ts_from, ts_to)
        timestamp_condition = " WHERE timestamp >= %s AND timestamp < %s "
        
    query = "SELECT COUNT(*) from plotter_frame" + timestamp_condition
    cursor.execute(query, query_parameters)
    count = cursor.fetchone()[0]
    if count == 0:
        return []

    fraction_to_sample = 100 * 1.10 * n_samples / count
    if fraction_to_sample < 0.01:
        fraction_to_sample = 0.01
    query = ("""
    SELECT * FROM (SELECT frame_id, timestamp FROM plotter_frame
    TABLESAMPLE BERNOULLI({:5.3f})
    """ + timestamp_condition + """) as sub
    ORDER BY random()
    LIMIT {}
    """).format(fraction_to_sample, n_samples)

    cursor.execute(query, query_parameters)
    
    return cursor.fetchall()

def get_frames(cam_id, ts_from, ts_to, cursor=None, frame_container_id=None, cursor_is_prepared=False):
    """Retrieves a list of frames for a camera between two time points.
        
        Arguments:
            cam_id: database camera id (0-4)
            ts_from: Begin (included); unix timestamp with milliseconds accuracy
            ts_to: End (excluded); unix timestamp with milliseconds accuracy
            cursor: optional database cursor to work on
            frame_container_id: required when cam_id==None; database frame_container_id to retrieve the camera ID from
        
        Returns:
            List containing tuples of (timestamp, frame_id, cam_id), which are sorted by timestamp.
    """
    if cursor is None:
        with base.get_database_connection(application_name="get_frames") as db:
            return get_frames(cam_id, ts_from, ts_to, cursor=db.cursor(), frame_container_id=frame_container_id)
    if not cursor_is_prepared:
        cursor.execute("SELECT timestamp, frame_id, fc_id FROM plotter_frame WHERE timestamp >= %s AND timestamp < %s", (ts_from, ts_to))
    else:
        cursor.execute("EXECUTE get_frames_all_frame_ids (%s, %s)",  (ts_from, ts_to))
    results = list(cursor)
    containers = {fc_id for (_, _, fc_id) in results}
    
    if not cursor_is_prepared:
        cursor.execute("PREPARE get_frames_fetch_container AS "
            "SELECT CAST(SUBSTR(video_name, 5, 1) AS INT) FROM plotter_framecontainer "
            "WHERE id = $1")

    if cam_id is None:
        if frame_container_id is None:
            raise ValueError("frame_container_id required when no cam_id provided.")
        cursor.execute("EXECUTE get_frames_fetch_container (%s)", (frame_container_id,))
        cam_id = cursor.fetchone()[0]

    matching_cam = set()
    for container in containers:
        cursor.execute("EXECUTE get_frames_fetch_container (%s)", (container,))
        cam = cursor.fetchone()[0]
        if cam == cam_id:
            matching_cam.add(container)
    results = [(timestamp, frame_id, cam_id) for (timestamp, frame_id, fc_id) in results if fc_id in matching_cam]
    return sorted(results)

def get_detections_for_frames(frames, use_hive_coordinates=True, confidence_threshold=0.1, sample_fraction=1.0, cursor=None):
    if cursor is None:
        with base.get_database_connection(application_name="get_detections_for_frames") as db:
            yield from get_detections_for_frames(frames, cursor=db.cursor("detection cursor"),
                    use_hive_coordinates=use_hive_coordinates,
                    confidence_threshold=confidence_threshold,
                    sample_fraction=sample_fraction)
            return
    
    coordinate_string = "x_pos as x, y_pos as y, orientation "
    if use_hive_coordinates:
        coordinate_string = "x_pos_hive as x, y_pos_hive as y, orientation_hive as orientation "
    sample_string = ""
    if sample_fraction is not None and sample_fraction < 1.0:
        sample_string = "TABLESAMPLE BERNOULLI ({}) ".format(sample_fraction * 100)
    cursor.execute("SELECT timestamp, frame_id, " + coordinate_string + ", track_id, bee_id "
            "FROM  " + base.get_detections_tablename() + " " + sample_string + \
            "WHERE frame_id = ANY(%s) "
            "AND bee_id_confidence > %s "
            "ORDER BY timestamp ASC", (list(map(int, frames)), confidence_threshold))

    yield from cursor

def get_track_ids(frames, cursor=None):
    """Retrieves all unique track IDs from the database that occur in a given set of frame_ids.
    Arguments:
        frames: list(int) or list(tuple(timestamp, frame_id, cam_id))
            Frames for which the track IDs are fetched.
        cursor: psycopg2.Cursor
            Optional database connection cursor.
    Returns:
        list(int)
            List containing the uint64 track IDs.
    """
    if cursor is None:
        with base.get_database_connection(application_name="get_track_ids") as db:
            return get_track_ids(frames, cursor=db.cursor())
    if len(frames) == 0:
        return []
    if type(frames[0]) is tuple: # (timestamp, frame_id, cam_id) style
        frames = [f[1] for f in frames]
    frames = [int(f) for f in frames]
    cursor.execute("SELECT DISTINCT track_id FROM {} WHERE frame_id = ANY(%s)".format(base.get_detections_tablename()), (frames,))
    track_ids = cursor.fetchall()
    track_ids = [int(t[0]) for t in track_ids]
    return track_ids

def get_track_ids_for_bee(bee_id, dt_from, dt_to, n_track_ids=None, cursor=None):
    """Retrieves all unique track IDs for a timespan from the database that belong to a certain bee.
    Arguments:
        bee_id: int
            Beesbook ID (ferwar style).
        dt_from, dt_to: datetime.datetime
            Points in time between which all track IDs will be returned.
        n_track_ids: int
            Optional. If given, n_track_ids random IDs will be sampled and returned.
        cursor: psycopg2.Cursor
            Optional database connection cursor.
    Returns:
        list(int)
            List containing the uint64 track IDs.
    """
    if cursor is None:
        with base.get_database_connection(application_name="get_track_ids_for_bee") as db:
            return get_track_ids_for_bee(bee_id, dt_from, dt_to, n_track_ids=n_track_ids, cursor=db.cursor())

    cursor.execute("SELECT DISTINCT track_id FROM {} WHERE bee_id = %s AND "
                    "timestamp >= %s AND timestamp < %s".format(base.get_detections_tablename()), (int(bee_id), dt_from, dt_to))
    track_ids = cursor.fetchall()
    track_ids = [int(t[0]) for t in track_ids]
    if (n_track_ids is not None) and len(track_ids) > n_track_ids:
        track_ids = np.random.choice(track_ids, size=n_track_ids, replace=False)
    return track_ids

def get_bee_ids(frames, cursor=None):
    """Retrieves all unique bee IDS from the database that occur in a given set of frame_ids.
    Arguments:
        frames: list(int) or list(tuple(timestamp, frame_id, cam_id))
            Frames for which the track IDs are fetched.
        cursor: psycopg2.Cursor
            Optional database connection cursor.
    Returns:
        list(int)
            List containing the bee IDs.
    """
    if cursor is None:
        with base.get_database_connection(application_name="get_bee_ids") as db:
            return get_bee_ids(frames, cursor=db.cursor())
    if len(frames) == 0:
        return []
    if type(frames[0]) is tuple: # (timestamp, frame_id, cam_id) style
        frames = [f[1] for f in frames]
    frames = [int(f) for f in frames]
    cursor.execute("SELECT DISTINCT bee_id FROM {} WHERE frame_id = ANY(%s)".format(base.get_detections_tablename()), (frames,))
    track_ids = cursor.fetchall()
    track_ids = [int(t[0]) for t in track_ids]
    return track_ids

def get_neighbour_frames(frame_id, n_frames=None, seconds=None, cursor=None, cursor_is_prepared=False,
                        n_frames_left=None, n_frames_right=None, seconds_left=None, seconds_right=None,
                        s_frame_margin_leeway=2.0):
    """Retrieves a specified number of frames around a center frame from the database.
        
        Arguments:
            frame_id: database ID of the middle frame
            n_frames: number of frames for the margin; attempts to return 2 * n_frames + 1 frames
            seconds: optional instead of n_frames; generally seconds = n_frames * 3.0
            cursor: optional database cursor to work on

            n_frames_left: left margin; defaults to n_frames
            n_frames_right: right margin; defaults to n_frames
            seconds_left: optional left margin in seconds; defaults to seconds
            seconds_right: optional right margin in seconds; defaults to seconds
            s_frame_margin_leeway: float or None
                If given, the initial query is done with a bit more leeway around the timestamps to allow
                for larger temporal gaps between images, trying to get exactly n_frames neighbours.
        Returns:
            List containing tuples of (timestamp, frame_id, cam_id), which are sorted by timestamp.
    """
    n_frames_left = n_frames_left or n_frames
    n_frames_right = n_frames_right or n_frames
    seconds = seconds or (n_frames / 3 if n_frames else 5.0)
    seconds_left = seconds_left or (seconds if n_frames_left is None else n_frames_left / 3)
    seconds_right = seconds_right or (seconds if n_frames_right is None else n_frames_right / 3)
    if s_frame_margin_leeway is None:
        s_frame_margin_leeway = 0.0

    if frame_id is None:
        raise ValueError("frame_id must not be None.")
    else:
        frame_id = int(frame_id)
    if cursor is None:
        with base.get_database_connection(application_name="get_neighbour_frames") as db:
            return get_neighbour_frames(frame_id=frame_id, n_frames=n_frames, seconds=seconds, cursor=db.cursor(),
                                        n_frames_left=n_frames_left, n_frames_right=n_frames_right, seconds_left=seconds_left,
                                        seconds_right=seconds_right, s_frame_margin_leeway=s_frame_margin_leeway)
    
    f_index, frame_container_id, timestamp = None, None, None
    if not cursor_is_prepared:
        cursor.execute("SELECT index, fc_id, timestamp FROM plotter_frame WHERE frame_id = %s LIMIT 1", (frame_id,))
    else:
        cursor.execute("EXECUTE get_neighbour_frames (%s)", (frame_id,))

    results = cursor.fetchone()
    f_index, frame_container_id, timestamp = results
    ts_from = timestamp - seconds_left - s_frame_margin_leeway
    ts_to = timestamp + seconds_right + s_frame_margin_leeway
    
    neighbour_frames = get_frames(cam_id=None, ts_from=ts_from, ts_to=ts_to, cursor=cursor, cursor_is_prepared=cursor_is_prepared, frame_container_id=frame_container_id)

    if s_frame_margin_leeway > 0.0:
        # We have potentially queried more frames and now need to filter them.
        middle_frame_idx = None
        for idx, (_, f_id, _) in enumerate(neighbour_frames):
            if f_id != frame_id:
                continue
            middle_frame_idx = idx
            break
        if middle_frame_idx is None:
            raise ValueError("frame_id {} not a neighbour of itself.".format(frame_id))
        left = []
        if middle_frame_idx < n_frames_left:
            left = [(None, None, None)] * (n_frames_left - middle_frame_idx) + neighbour_frames[:(middle_frame_idx + 1)]
        else:
            left = neighbour_frames[(middle_frame_idx - n_frames_left):(middle_frame_idx + 1)]
        right = []
        n_neighbours = len(neighbour_frames)
        n_right_neighbours = n_neighbours - middle_frame_idx
        if n_right_neighbours < n_frames_right:
            right = neighbour_frames[(middle_frame_idx + 1):] + [(None, None, None)] * (n_frames_right - n_right_neighbours + 1)
        else:
            right =  neighbour_frames[(middle_frame_idx + 1):(middle_frame_idx + n_frames_right + 1)]
        neighbour_frames = left + right

    return neighbour_frames