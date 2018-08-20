import psycopg2
import math
import numba
import numpy as np
from . import utils

server_address = "localhost:5432"

def get_database_connection(application_name="bb_trajectory"):
    database_host = server_address
    database_port = 5432

    if ":" in database_host:
        database_host, database_port = database_host.split(":")
    return psycopg2.connect("dbname='beesbook' user='reader' host='{}' port='{}' password='reader'".format(database_host, database_port),
                          application_name=application_name)

class DatabaseCursorContext(object):
    """Helper objects to create a database cursor with pre-defined queries.
    This is used as a thread_context in loops that call e.g. get_frames
    multiple times and don't want to re-create the cursor and query definitions
    every time.
    """
    _db = None
    _cursor = None
    _application_name = None
    def __init__(self, application_name="DatabaseCursorContext"):
        self._application_name = application_name

    def __enter__(self):
        self._db = get_database_connection(application_name=self._application_name)
        self._cursor = self._db.cursor()

        self._cursor.execute("""
            SET geqo_effort to 10;
            SET max_parallel_workers_per_gather TO 8;
            SET temp_buffers to "32GB";
            SET work_mem to "1GB";
            SET temp_tablespaces to "ssdspace";""")

        self._cursor.execute("PREPARE get_neighbour_frames AS "
           "SELECT index, fc_id, timestamp FROM plotter_frame WHERE frame_id = $1 LIMIT 1")

        self._cursor.execute("PREPARE get_frames_all_frame_ids AS "
           "SELECT timestamp, frame_id, fc_id FROM plotter_frame WHERE timestamp >= $1 AND timestamp <= $2")

        self._cursor.execute("PREPARE get_frames_fetch_container AS "
           "SELECT CAST(SUBSTR(video_name, 5, 1) AS INT) FROM plotter_framecontainer "
           "WHERE id = $1")

        self._cursor.execute("PREPARE get_bee_detections AS "
           "SELECT timestamp, frame_id, x_pos AS x, y_pos AS y, orientation, track_id FROM bb_detections_2016_stitched "
           "WHERE frame_id = ANY($1) AND bee_id = $2 ORDER BY timestamp ASC")

        self._cursor.execute("PREPARE get_bee_detections_hive_coords AS "
           "SELECT timestamp, frame_id, x_pos_hive AS x, y_pos_hive AS y, orientation, track_id FROM bb_detections_2016_stitched "
           "WHERE frame_id = ANY($1) AND bee_id = $2 ORDER BY timestamp ASC")
        return self._cursor

    def __exit__(self, type, value, traceback):
        self._cursor.close()
        self._db.close()

    @property
    def cursor(self):
        return self._cursor

def get_frames(cam_id, ts_from, ts_to, cursor=None, frame_container_id=None, cursor_is_prepared=False):
    """Retrieves a list of frames for a camera between two time points.
        
        Arguments:
            cam_id: database camera id (0-4)
            ts_from: Begin (included); unix timestamp with milliseconds accuracy
            ts_to: End (included); unix timestamp with milliseconds accuracy
            cursor: optional database cursor to work on
            frame_container_id: required when cam_id==None; database frame_container_id to retrieve the camera ID from
        
        Returns:
            List containing tuples of (timestamp, frame_id, cam_id), which are sorted by timestamp.
    """
    if cursor is None:
        with get_database_connection(application_name="get_frames") as db:
            return get_frames(cam_id, ts_from, ts_to, cursor=db.cursor(), frame_container_id=frame_container_id)
    if not cursor_is_prepared:
        cursor.execute("SELECT timestamp, frame_id, fc_id FROM plotter_frame WHERE timestamp >= %s AND timestamp <= %s", (ts_from, ts_to))
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

def get_neighbour_frames(frame_id, n_frames=None, seconds=None, cursor=None, cursor_is_prepared=False,
                        n_frames_left=None, n_frames_right=None, seconds_left=None, seconds_right=None):
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
        Returns:
            List containing tuples of (timestamp, frame_id, cam_id), which are sorted by timestamp.
    """
    n_frames_left = n_frames_left or n_frames
    n_frames_right = n_frames_right or n_frames
    seconds = seconds or (n_frames / 3 if n_frames else 5.0)
    seconds_left = seconds_left or (seconds if n_frames_left is None else n_frames_left / 3)
    seconds_right = seconds_right or (seconds if n_frames_right is None else n_frames_right / 3)
    
    if frame_id is None:
        raise ValueError("frame_id must not be None.")

    if cursor is None:
        with get_database_connection(application_name="get_neighbour_frames") as db:
            return get_neighbour_frames(frame_id=frame_id, n_frames=n_frames, seconds=seconds, cursor=db.cursor(),
                                        n_frames_left=n_frames_left, n_frames_right=n_frames_right, seconds_left=seconds_left,
                                        seconds_right=seconds_right)
    
    f_index, frame_container_id, timestamp = None, None, None
    if not cursor_is_prepared:
        cursor.execute("SELECT index, fc_id, timestamp FROM plotter_frame WHERE frame_id = %s LIMIT 1", (frame_id,))
    else:
        cursor.execute("EXECUTE get_neighbour_frames (%s)", (frame_id,))

    results = cursor.fetchone()
    f_index, frame_container_id, timestamp = results
    ts_from = timestamp - seconds_left
    ts_to = timestamp + seconds_right
    
    return get_frames(cam_id=None, ts_from=ts_from, ts_to=ts_to, cursor=cursor, cursor_is_prepared=cursor_is_prepared, frame_container_id=frame_container_id)
        
    
def get_bee_detections(bee_id, verbose=False, frame_id=None, frames=None,
                        use_hive_coords=False,
                        cursor=None, cursor_is_prepared=False, **kwargs):
    """Fetches all detections for a bee between some time points or around a center frame.
        The results include "None" when no detection was found for a time step.
        
        Arguments:
            bee_id: database ID (ferwar format) of the focal individual
            verbose: whether to print extra information
            frame_id: optional center frame ID, use with n_frames or seconds (see get_neighbour_frames)
            frames: optional list of frames containing tuples of (timestamp, frame_id, cam_id), see get_frames
            use_hive_coords: (default False) whether to retrieve hive coordinates
            cursor: optional database cursor to work on
            
        Returns:
            List containing tuples of (timestamp, frame_id, x_pos, y_pos, orientation, track_id) sorted by timestamp; can contain None
    """
    if type(bee_id) != int:
        bee_id = bee_id.as_ferwar()
    if frames is None and frame_id is None:
        raise ValueError("Either frame_id or frames must be provided.")
    if not frames and not frame_id:
        raise ValueError("frames must not be empty.")

    if cursor is None:
        with get_database_connection(application_name="get_bee_detections") as db:
            return get_bee_detections(bee_id, verbose=verbose, frame_id=frame_id, frames=frames, cursor=db.cursor(), **kwargs)
    
    frames = frames or get_neighbour_frames(frame_id=frame_id, cursor=cursor, cursor_is_prepared=cursor_is_prepared, **kwargs)
    frame_ids = [f[1] for f in frames]
    
    if not cursor_is_prepared:
        coords_string = "x_pos AS x, y_pos AS y"
        if use_hive_coords:
            coords_string = "x_pos_hive AS x, y_pos_hive AS y"
        cursor.execute("SELECT timestamp, frame_id, " + coords_string + ", orientation, track_id FROM bb_detections_2016_stitched WHERE frame_id=ANY(%s) AND bee_id = %s ORDER BY timestamp ASC",
                        (frame_ids, bee_id))
    else:
        prepared_statement_name = "get_bee_detections" if not use_hive_coords else "get_bee_detections_hive_coords"
        cursor.execute("EXECUTE " + prepared_statement_name + " (%s, %s)", (frame_ids, bee_id))
    detections = cursor.fetchall()
    
    results = []
    for n_idx, (timestamp, frame_id, _) in enumerate(frames):
        if len(detections) == 0:
            results.append(None)
            continue
        if frame_id == detections[0][1]:
            if len(detections) == 1 or frame_id != detections[1][1]:
                results.append(detections[0])
                detections.pop(0)
            else:
                candidates = [d for d in detections if d[1] == frame_id]
                if verbose:
                    print("Warning: more than one candidate! ({})".format(len(candidates)))
                closest_candidate = None
                for i, r in reversed(list(enumerate(results))):
                    if r is None:
                        continue
                    closest_candidate = r
                    break
                candidate = None
                if closest_candidate is not None:
                    for c in candidates:
                        if c[-1] == closest_candidate[-1]: # track_id
                            candidate = c
                            break
                if verbose and candidate is not None:
                    print("\t..resolved via track ID.")
                else:
                    distances = np.array([[x, y] for (_, _, x, y, _, _) in candidates])
                    if closest_candidate:
                        distances -= np.array([closest_candidate[2], closest_candidate[3]])
                        distances = np.linalg.norm(distances, axis=1)
                        min_d = np.argmin(distances)
                        candidate = candidates[min_d]
                        if verbose:
                            print("\t..resolved via distance.")
                    
                results.append(candidate)#candidates[0])
                for i in range(len(candidates)):
                    detections.pop(0)
        else:
            results.append(None)
    return results

@numba.njit
def short_angle_dist(a0,a1):
    """Returns the signed distance between two angles in radians.
    """
    max = math.pi*2
    da = (a1 - a0) % max
    return 2*da % max - da
@numba.njit
def angle_lerp(a0,a1,t):
    """Simple linear interpolation between two angles.
        
        Arguments:
            a0: angle 1
            a1: angle 2
            t: interpolation value between 0 and 1
    """
    return a0 + short_angle_dist(a0,a1)*t

def get_bee_trajectory(bee_id, frame_id=None, frames=None, **kwargs):
    """Returns the trajectory (x, y, orientation) of a bee as a numpy array.
        Missing detections will be filled with np.nan.
        
        Arguments:
            bee_id: database ID (ferwar format) for the focal individual
            frame_id: optional center frame ID, use with n_frames or seconds (see get_neighbour_frames)
            frames: optional list of frames containing tuples of (timestamp, frame_id, cam_id), see get_frames

        Returns:
            numpy array (float 32) of shape (N, 3)
    """
    detections = get_bee_detections(bee_id, frame_id=frame_id, frames=frames, **kwargs)
    # (dt, frame_id, x, y, alpha)
    def unpack(d):
        if d is None:
            return [np.nan, np.nan, np.nan]
        (dt, frame_id, x, y, alpha, track_id) = d
        return [x, y, alpha]
    trajectory = np.array([unpack(d) for d in detections], dtype=np.float32)
    return trajectory
        
@numba.njit(numba.float32[:](numba.float32[:, :]))
def interpolate_trajectory(trajectory):
    """
        Linearly interpolates a bee trajectory consisting of position and angle in place.
        np.nan in the data are filled by linear interpolation or extrapolation the the closest neighbours.
        Args:
            trajectory: numpy array (float 32) of shape (N, 3) containing (x_pos, y_pos, orientation) (return value of get_bee_trajectory)
        Returns:
            numpy array (float 32) of shape (N,) containing 1.0 in places where the original trajectory contained values and 0.0 for interpolated values.
    """
    
    nans = np.isnan(trajectory[:,0])
    not_nans = ~nans
    
    nans_idx = np.where(nans)[0]
    valid_idx = np.where(not_nans)[0]
    if len(valid_idx) < 2:
        return np.zeros(shape=(trajectory.shape[0]), dtype=np.float32)
    
    # Interpolate gaps.
    for i in nans_idx:
        # Find closest two points to use for interpolation.
        begin_t = np.searchsorted(valid_idx, i) - 1
        if begin_t == len(valid_idx) - 1:
            begin_t -= 1 # extrapolate right
        elif begin_t == -1:
            begin_t = 0 # extrapolate left
        
        begin_t_idx = valid_idx[begin_t]
        end_t_idx = valid_idx[begin_t + 1]
        
        last_t = trajectory[begin_t_idx]
        next_t = trajectory[end_t_idx]
        dx = (end_t_idx - begin_t_idx) / 3.0
        m = [(next_t[0] - last_t[0]) / dx,
             (next_t[1] - last_t[1]) / dx,
             short_angle_dist(last_t[2], next_t[2]) / dx]

        dt = (i - begin_t_idx) / 3.0
        e = [m[i] * dt + last_t[i] for i in range(3)]
        trajectory[i] = e
        
    return not_nans.astype(np.float32)

def get_interpolated_trajectory(bee_id, frame_id=None, frames=None, interpolate=True, **kwargs):
    """Fetches detections from the database and interpolates missing detections linearly.
        
        Arguments:
            bee_id: database ID (ferwar format) for the focal individual
            frame_id: optional center frame ID, use with n_frames or seconds (see get_neighbour_frames)
            frames: optional list of frames containing tuples of (timestamp, frame_id, cam_id), see get_frames
            interpolate: whether to fill missing detections with a linear interpolation (instead of np.nan)
        
        Returns:
            (trajectory, mask): numpy arrays (float 32).
                                trajectory is of shape (N, 3) containing (x_pos, y_pos, orientation).
                                mask is of shape (N,) containing 1.0 for original and 0.0 for interpolated values.
    """
    trajectory = get_bee_trajectory(bee_id, frame_id=frame_id, frames=frames, **kwargs)
    mask = None
    if interpolate:
        mask = interpolate_trajectory(trajectory)
    return trajectory, mask

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
        with get_database_connection(application_name="sample_frames") as db:
            cursor = db.cursor()
            return sample_frame_ids(n_samples=n_samples, ts_from=ts_from, ts_to=ts_to, cursor=cursor)
    timestamp_condition = " WHERE True "
    query_parameters = None
    if ts_from is not None:
        query_parameters = (ts_from, ts_to)
        timestamp_condition = " WHERE timestamp >= %s AND timestamp < %s ";
        
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

def find_interactions_in_frame(frame_id, max_distance=20.0, min_distance=0.0, confidence_threshold=0.25, cursor=None):
    """Takes a frame id and finds all the possible interactions consisting of close bees.
    
    Arguments:
        frame_id: Database frame id of the frame to search.
        max_distance: Maximum hive distance (mm) of interaction partners.
        min_distance: Minimum hive distance (mm) of interaction partners.
        confidence_threshold: Minimum confidence of detections. Others are ignored.
        cursor: Optional. Database cursor connected to the DB.
        
    Returns:
        List containing interaction partners as tuples of
        (frame_id, bee_id0, bee_id1, detection_idx0, detection_idx1,
        x_pos_hive0, y_pos_hive0, x_pos_hive1, x_pos_hive1, cam_id).
    """
    import pandas

    if cursor is None:
        with get_database_connection(application_name="sample_frames") as db:
            cursor = db.cursor()
            return find_interactions_in_frame(frame_id=frame_id,
                                              max_distance=max_distance,
                                              min_distance=min_distance,
                                              confidence_threshold=confidence_threshold,
                                              cursor=cursor)
        
    query = """
    SELECT x_pos_hive, y_pos_hive, orientation_hive, bee_id, detection_idx, cam_id FROM bb_detections_2016_stitched
        WHERE frame_id = %s
        AND bee_id_confidence >= %s
    """
    
    df = pandas.read_sql_query(query, cursor.connection, coerce_float=False,
                               params=(int(frame_id), confidence_threshold))
    if df.shape[0] == 0:
        return []
    
    close_pairs = utils.find_close_points(df[["x_pos_hive", "y_pos_hive"]].values,
                                   max_distance, min_distance)
    
    results = []
    for (i, j) in close_pairs:
        results.append((int(frame_id),
                        int(df.bee_id.iloc[i]), int(df.bee_id.iloc[j]),
                        int(df.detection_idx.iloc[i]), int(df.detection_idx.iloc[j]),
                        df.x_pos_hive.iloc[i], df.y_pos_hive.iloc[i], df.orientation_hive.iloc[i],
                        df.x_pos_hive.iloc[j], df.y_pos_hive.iloc[j], df.orientation_hive.iloc[j],
                        int(df.cam_id.iloc[j])
                       ))
    return results