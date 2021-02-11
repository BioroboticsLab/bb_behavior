import datetime
import math
import numba
import numpy as np
import pandas as pd
import itertools

from .. import utils
from . import base
from . import sampling

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
        self._db = base.get_database_connection(application_name=self._application_name)
        self._cursor = self._db.cursor()

        self._cursor.execute("""
            SET geqo_effort to 10;
            SET max_parallel_workers_per_gather TO 8;
            SET temp_buffers to "32GB";
            SET work_mem to "1GB";""")

        temp_tablespace = base.get_temp_tablespace()
        if temp_tablespace:
            self._cursor.execute("""SET temp_tablespaces to "{}";""".format(temp_tablespace))

        self._cursor.execute("PREPARE get_frame_info AS "
            "SELECT cam_id, index, fc_id, timestamp FROM {} WHERE frame_id = $1 LIMIT 1".format(base.get_frame_metadata_tablename()))

        self._cursor.execute("PREPARE get_all_frame_ids AS "
        "SELECT datetime, frame_id, cam_id FROM {} WHERE datetime >= $1 AND datetime < $2".format(base.get_frame_metadata_tablename()))

        self._cursor.execute("PREPARE get_all_frame_ids_for_cam AS "
        "SELECT datetime, frame_id, cam_id FROM {} WHERE datetime >= $1 AND datetime < $2 AND cam_id = $3".format(base.get_frame_metadata_tablename()))

        self._cursor.execute("PREPARE get_bee_detections AS "
           "SELECT timestamp, frame_id, x_pos AS x, y_pos AS y, orientation, track_id FROM {} "
           "WHERE frame_id = ANY($1) AND bee_id = $2 AND bee_id_confidence >= $3 ORDER BY timestamp ASC".format(base.get_detections_tablename()))
        
        self._cursor.execute("PREPARE get_bee_detections_hive_coords AS "
           "SELECT timestamp, frame_id, x_pos_hive AS x, y_pos_hive AS y, orientation_hive as orientation, track_id FROM {} "
           "WHERE frame_id = ANY($1) AND bee_id = $2 AND bee_id_confidence >= $3 ORDER BY timestamp ASC".format(base.get_detections_tablename()))

        self._cursor.execute("PREPARE find_interaction_candidates AS "
            "SELECT x_pos_hive, y_pos_hive, orientation_hive, bee_id, detection_idx, cam_id FROM {} "
            "WHERE frame_id = $1 AND bee_id_confidence >= $2".format(base.get_detections_tablename()))

        self._cursor.execute("""PREPARE get_all_bee_hive_detections_for_frames AS 
            SELECT bee_id, timestamp, frame_id,
            x_pos_hive AS x, y_pos_hive AS y, orientation_hive as orientation,
            track_id FROM {} WHERE frame_id=ANY($1) AND bee_id=ANY($2)
            ORDER BY timestamp ASC""".format(base.get_detections_tablename()))

        self._cursor.execute("""PREPARE get_all_bee_pixel_detections_for_frames AS 
            SELECT bee_id, timestamp, frame_id,
            x_pos AS x, y_pos AS y, orientation,
            track_id FROM {} WHERE frame_id=ANY($1) AND bee_id=ANY($2)
            ORDER BY timestamp ASC""".format(base.get_detections_tablename()))

        self._cursor.execute("PREPARE get_frame_metadata AS "
            "SELECT frame_id, timestamp, index, fc_id FROM {} WHERE frame_id = ANY($1)".format(base.get_frame_metadata_tablename()))
        # For metadata.get_frame_metadata
        self._cursor.execute("PREPARE get_frame_container_info AS "
            "SELECT video_name FROM {} "
            "WHERE fc_id = $1 LIMIT 1".format(base.get_framecontainer_metadata_tablename()))

        return self._cursor

    def __exit__(self, type, value, traceback):
        self._cursor.close()
        self._db.close()

    @property
    def cursor(self):
        return self._cursor

def get_consistent_track_from_detections(frames, detections, verbose=False):
    """Takes an ordered list of frames and detections for that frames and filters out duplicate detections.

    Arguments:
        frames: list(tuple(timestmap, frame_id, cam_id))
            See get_neighbour_frames.
        detections: list(tuple(timestamp, frame_id, x, y, orientation, track_id))
            Ordered by timestamp.
            Possible detections (can contain duplicates and missing detections).
        verbose: bool
            Whether to print additional output.

    Returns:
        list(tuple(timestamp, frame_id, x_pos, y_pos, orientation, track_id)) sorted by timestamp; can contain None.
        Length is the same as the length of 'frames'.
    """
    # frames can be a list of tuples or a list of ints.
    if len(frames) > 0 and type(frames[0]) is tuple:
        frame_ids = [f[1] for f in frames]
    else:
        frame_ids = frames
    results = []
    for n_idx, frame_id in enumerate(frame_ids):
        if (len(detections) == 0) or (frame_id is None):
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

def get_bee_detections(bee_id, verbose=False, frame_id=None, frames=None,
                        use_hive_coords=False,
                        cursor=None, cursor_is_prepared=False, make_consistent=True,
                        confidence_threshold=0.0, **kwargs):
    """Fetches all detections for a bee between some time points or around a center frame.
        The results include "None" when no detection was found for a time step.
        
        Arguments:
            bee_id: database ID (ferwar format) of the focal individual
            verbose: whether to print extra information
            frame_id: optional center frame ID, use with n_frames or seconds (see get_neighbour_frames)
            frames: optional list of frames containing tuples of (timestamp, frame_id, cam_id), see get_frames
            use_hive_coords: (default False) whether to retrieve hive coordinates
            cursor: optional database cursor to work on
            make_consistent: bool
                Whether the frames are a consecutive track. Filters out duplicate detections per frame and orders the return values.
            confidence_threshold: minimum bee_id_confidence to allow
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
        with base.get_database_connection(application_name="get_bee_detections") as db:
            return get_bee_detections(bee_id, verbose=verbose, frame_id=frame_id, frames=frames, cursor=db.cursor(), use_hive_coords=use_hive_coords, **kwargs)
    
    frames = frames or sampling.get_neighbour_frames(frame_id=frame_id, cursor=cursor, cursor_is_prepared=cursor_is_prepared, **kwargs)
    # Is frames a list of tuples? (Can also be a simple list of frame_ids.)
    if len(frames) > 0 and type(frames[0]) is tuple:
        frame_ids = [int(f[1]) for f in frames if f[1] is not None]
    else:
        frame_ids = frames
    if not cursor_is_prepared:
        coords_string = "x_pos AS x, y_pos AS y, orientation"
        if use_hive_coords:
            coords_string = "x_pos_hive AS x, y_pos_hive AS y, orientation_hive as orientation"
        cursor.execute("SELECT timestamp, frame_id, " + coords_string + ", track_id FROM {} WHERE frame_id=ANY(%s) AND bee_id = %s AND bee_id_confidence >= %s ORDER BY timestamp ASC".format(base.get_detections_tablename()),
                        (frame_ids, bee_id, confidence_threshold))
    else:
        prepared_statement_name = "get_bee_detections" if not use_hive_coords else "get_bee_detections_hive_coords"
        cursor.execute("EXECUTE " + prepared_statement_name + " (%s, %s, %s)", (frame_ids, bee_id, confidence_threshold))
    detections = cursor.fetchall()
    if make_consistent:
        return get_consistent_track_from_detections(frames, detections, verbose=verbose)
    else:
        return detections

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

def get_bee_trajectory(bee_id, frame_id=None, frames=None, detections=None, **kwargs):
    """Returns the trajectory (x, y, orientation) of a bee as a numpy array.
        Missing detections will be filled with np.nan.
        
        Arguments:
            bee_id: database ID (ferwar format) for the focal individual
            frame_id: optional center frame ID, use with n_frames or seconds (see get_neighbour_frames)
            frames: optional list of frames containing tuples of (timestamp, frame_id, cam_id), see get_frames
            detections: list(tuple())
                Optional. Same format as returned by get_bee_detections. If given, all other parameters can be None.
        Returns:
            numpy array (float 32) of shape (N, 3)
    """
    if detections is None:
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
    # There are three states: all values valid, orientation nan, all values nan.
    # Thus, checking the orientation suffices.
    rows_with_nans = np.isnan(trajectory[:, 2])
    assert rows_with_nans.shape[0] == trajectory.shape[0]
    not_nans = ~rows_with_nans
    
    nans_idx = np.where(rows_with_nans)[0]
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
        # Only replace previously missing values.
        for j in range(3):
            if np.isnan(trajectory[i, j]):
                trajectory[i, j] = e[j]
        
    return not_nans.astype(np.float32)

def get_interpolated_trajectory(bee_id, frame_id=None, frames=None, interpolate=True, detections=None, **kwargs):
    """Fetches detections from the database and interpolates missing detections linearly.
        
        Arguments:
            bee_id: database ID (ferwar format) for the focal individual
            frame_id: optional center frame ID, use with n_frames or seconds (see get_neighbour_frames)
            frames: optional list of frames containing tuples of (timestamp, frame_id, cam_id), see get_frames
            interpolate: whether to fill missing detections with a linear interpolation (instead of np.nan)
            detections: list(tuple())
                Optional. Same format as returned by get_bee_detections. If given, all other parameters can be None.
        
        Returns:
            (trajectory, mask): numpy arrays (float 32).
                                trajectory is of shape (N, 3) containing (x_pos, y_pos, orientation).
                                mask is of shape (N,) containing 1.0 for original and 0.0 for interpolated values.
    """
    trajectory = get_bee_trajectory(bee_id, frame_id=frame_id, frames=frames, detections=detections, **kwargs)
    if trajectory.shape[0] == 0:
        return None, None

    mask = None
    if interpolate:
        mask = interpolate_trajectory(trajectory)
    return trajectory, mask

def get_track(track_id, frames, use_hive_coords=False, cursor=None, make_consistent=True, interpolate=True):
    """Retrieves the coherent, short track for a track ID that was produced by the tracking mapping.
    The track can contain gaps and might not start at the beginning of the given frames.

    Arguments:
        track_id: uint64
            Track ID that is fetched from the database.
        frames: list(int) or list((timestamp, frame_id, cam_id))
            Frames for which the track is fetched.
        use_hive_coords: bool
            Whether to return the trajectory in hive coordinates.
        cursor: psycopg2.Cursor
            Optional database connection.
        make_consistent: bool
            Whether the resulting trajectory is matched to the 'frames' list, possibly producing gaps.
        interpolate: bool
            Whether to interpolate the resulting trajectory.

    Returns:
        list(np.array(shape=(N, 3))), list(tuple(frame_id, detection_idx))
            Yields the trajectory (x, y, orientation) and a list of frame_ids and detection indices
            that can uniquely indentify a detection. The latter can contain None where no data is available.
            The former will be interpolated, if set, but never extrapolated beyond the track boundaries.
    """
    if (frames is not None) and (len(frames) == 0):
        return []
    if cursor is None:
        with base.get_database_connection(application_name="get_track") as db:
            return get_track(track_id, frames,
                make_consistent=make_consistent, interpolate=interpolate,
                use_hive_coords=use_hive_coords, cursor=db.cursor())
    coords = ("x_pos", "y_pos", "orientation")
    if use_hive_coords:
        coords = (c + "_hive" for c in coords)
    
    frame_condition = ""
    query_arguments = (track_id, )
    if frames is not None:
        if type(frames[0]) is tuple: # (timestamp, frame_id, cam_id) style.
            frame_ids = [f[1] for f in frames]
        else:
            frame_ids = frames
            if make_consistent:
                raise ValueError("get_track: make_consistent==True requires frames given as (timestamp, frame_id, cam_id) tuples.")
        frame_ids = list(map(int, frame_ids))

        frame_condition = " frame_id = ANY(%s) AND "
        query_arguments = (frame_ids, track_id)
    cursor.execute(
           "SELECT detection_idx, timestamp, frame_id, {} AS x, {} AS y, {} as orientation, track_id FROM {} "
           "WHERE {} track_id = %s ORDER BY timestamp ASC".format(*coords, base.get_detections_tablename(), frame_condition), query_arguments)
    track = cursor.fetchall()
    detection_keys = {t[2]: t[0] for t in track}
    track = [t[1:] for t in track]
    if make_consistent:
        track = get_consistent_track_from_detections(frames, track)

    keys = []
    for t in track:
        if t is None:
            keys.append(None)
        else:
            frame_id = t[1]
            keys.append((int(frame_id), detection_keys[frame_id]))

    track = get_bee_trajectory(bee_id=None, detections=track)
    if interpolate:
        interpolate_trajectory(track)
    return track, keys

def get_bee_velocities(bee_id, dt_from, dt_to, cursor=None,
                       cursor_is_prepared=False, progress=None,
                       confidence_threshold=0.1, fixup_velocities="auto",
                       additional_columns=set(), max_mm_per_second=None):
    """Retrieves the velocities of a bee over time.

    Arguments:
        bee_id: int
            To query data from the database. In ferwar format.
        dt_from, dt_to: datetime.datetime
            Time interval to search for results.
        cursor: pyscopg2.cursor
            Optional. Connected database cursor.
        cursor_is_prepared: bool
            Whether to execute prepare statements on the cursor.
        progress: "tqdm", "tqdm_notebook" or None
            Progress bar to display.
        confidence_threshold: float
            Retrieves only detections above this threshold.
        fixup_velocities: bool
            Whether to assume that the timestamps are at the FPS given in the base config and smoothing them is okay.
            If "auto", it's true for the 2016_berlin season and False otherwise.
        additional_columns: iterable(string)
            Iterable of additional column names to query from the database.
        max_mm_per_second: float
            Optional. All velocities above this value are considered unrealistic outliers and will be ignored.
    """
    if not cursor:
        from contextlib import closing
        with closing(base.get_database_connection("get_bee_velocities")) as con:
            return get_bee_velocities(bee_id, dt_from, dt_to, cursor=con.cursor(), cursor_is_prepared=False,
                                      progress=progress, confidence_threshold=confidence_threshold, fixup_velocities=fixup_velocities,
                                      additional_columns=additional_columns, max_mm_per_second=max_mm_per_second)
    
    import pytz
    import scipy.signal

    required_columns = list(set(("cam_id", "timestamp", "x_pos_hive", "y_pos_hive", "orientation_hive", "track_id")) | set(additional_columns))

    if not cursor_is_prepared:       
        cursor.execute("""PREPARE fetch_detections AS
                SELECT {}
                   FROM {} 
                    WHERE timestamp >= $1
                    AND timestamp < $2
                    AND bee_id = $3
                    AND bee_id_confidence > {}
                """.format(", ".join(required_columns), base.get_detections_tablename(), confidence_threshold))
    
    progress_bar = lambda x: x
    if progress == "tqdm":
        from tqdm import tqdm
        progress_bar = tqdm
    elif progress == "tqdm_notebook":
        from tqdm import tqdm_notebook
        progress_bar = tqdm_notebook
    
    if fixup_velocities == "auto":
        if base.get_season_identifier() == "berlin_2016":
            fixup_velocities = True
        else:
            fixup_velocities = False

    x_col_index = required_columns.index("x_pos_hive")
    y_col_index = required_columns.index("y_pos_hive")
    cam_id_col_index = required_columns.index("cam_id")
    timestamp_col_index = required_columns.index("timestamp")
    track_id_col_index = required_columns.index("track_id")

    query_args = (dt_from, dt_to, bee_id)
    cursor.execute("EXECUTE fetch_detections (%s, %s, %s)", query_args)
    all_track_data = cursor.fetchall()

    # Order by track ID.
    track_id_data = dict()
    for row in all_track_data:
        track_id = row[track_id_col_index]
        if track_id not in track_id_data:
            track_id_data[track_id] = []
        track_id_data[track_id].append(row)

    # Order by time.
    sorted_track_ids = list()
    for key, val in track_id_data.items():
        val = sorted(val, key=lambda x: x[timestamp_col_index])
        track_id_data[key] = val
        sorted_track_ids.append((val[0][timestamp_col_index], key))
    sorted_track_ids = sorted(sorted_track_ids)

    all_velocities = []
    last_track_end_timestamp = None

    for _, track_id in progress_bar(sorted_track_ids):
        track = track_id_data[track_id]

        if not track:
            continue
        
        if last_track_end_timestamp is not None:
            track_begin_timestamp = track[0][timestamp_col_index]
            if track_begin_timestamp < last_track_end_timestamp:
                continue
        last_track_end_timestamp = track[-1][timestamp_col_index]

        value_series = tuple(zip(*track))
        datetimes = value_series[timestamp_col_index]
        if len(datetimes) < 2:
            continue
        x = value_series[x_col_index]
        y = value_series[y_col_index]
        
        timestamp_deltas = [dt.timestamp() for dt in datetimes]
        x, y, timestamp_deltas = np.diff(x), np.diff(y), np.diff(timestamp_deltas)
        assert np.all(timestamp_deltas > 0.0)
        
        v = np.sqrt(np.square(x) + np.square(y))
        
        if fixup_velocities:
            timestamp_deltas = np.round(timestamp_deltas * float(base.get_fps())) / 3.0
            timestamp_deltas[timestamp_deltas == 0.0] = 1.0 / base.get_fps()
        v = v / timestamp_deltas
        if max_mm_per_second is not None:
            v[v > max_mm_per_second] = np.nan
        if v.shape[0] > 3:
            v = scipy.signal.medfilt(v, kernel_size=3)
        
        valid_idx = ~pd.isnull(v)
        timestamp_deltas = timestamp_deltas[valid_idx]
        v = v[valid_idx]
        datetimes = list(map(lambda x: x.replace(tzinfo=pytz.UTC), datetimes[:-1]))
        datetimes = [datetimes[i] for i in range(len(datetimes)) if valid_idx[i]]

        columns_dict = dict(
                velocity=v,
                time_passed=timestamp_deltas,
                datetime=datetimes)
        for additional_column in additional_columns:
            columns_dict[additional_column] = value_series[required_columns.index(additional_column)][1:]

        df = pd.DataFrame(columns_dict)
        all_velocities.append(df)
    if not all_velocities:
        return None
    all_velocities = pd.concat(all_velocities, axis=0)
    all_velocities = all_velocities[(all_velocities.datetime >= dt_from) & (all_velocities.datetime < dt_to)]
    all_velocities["datetime"] = pd.to_datetime(all_velocities.datetime)
    return all_velocities

def get_bee_velocities_from_detections(bee_id, dt_from, dt_to, cursor=None,
                                    cursor_is_prepared=False,
                                    confidence_threshold=0.1,
                                    additional_columns=set(),
                                    window_size=datetime.timedelta(minutes=10),
                                    max_gap_length=5, max_distance=25, max_time_distance=2.0,
                                    max_mm_per_second=15.0):
    """Retrieves the velocities of a bee over time.

    Arguments:
        bee_id: int
            To query data from the database. In ferwar format.
        dt_from, dt_to: datetime.datetime
            Time interval to search for results.
        cursor: pyscopg2.cursor
            Optional. Connected database cursor.
        cursor_is_prepared: bool
            Whether to not execute prepare statements on the cursor.
        confidence_threshold: float
            Retrieves only detections above this threshold.
        additional_columns: iterable(string)
            Iterable of additional column names to query from the database.
        window_size: datetime.timedelta
            Specifies the chunk size in which the data is iterated over.
        max_gap_length: int
            Maximum number of detections to look ahead and look for a different detection
            in case of encountering detections  a different camera or too far from the last detection.
        max_distance: float
            Maximum distance in mm over which calculate a velocity for two temporally adjacent detections.
        max_time_distance: float
            Maximum time duration in seconds over which calculate a velocity for two temporally adjacent detections.
        max_mm_per_second: float
            All velocities above this value are considered unrealistic outliers and will be ignored.
    """
    if not cursor:
        from contextlib import closing
        with closing(base.get_database_connection("get_bee_velocities_from_detections")) as con:
            return get_bee_velocities_from_detections(bee_id, dt_from, dt_to, cursor=con.cursor(), cursor_is_prepared=False,
                                      confidence_threshold=confidence_threshold,
                                      additional_columns=additional_columns,
                                      window_size=window_size, max_gap_length=max_gap_length,
                                      max_distance=max_distance, max_time_distance=max_time_distance, max_mm_per_second=max_mm_per_second)
    
    import pytz
    import scipy.signal

    required_columns = list(set(("cam_id", "timestamp", "x_pos_hive", "y_pos_hive", "orientation_hive")) | set(additional_columns))

    if not cursor_is_prepared:       
        cursor.execute("""PREPARE fetch_detections AS
                SELECT {}
                   FROM {} 
                    WHERE timestamp >= $1
                    AND timestamp < $2
                    AND bee_id = $3
                    AND bee_id_confidence > {}
                    ORDER BY timestamp ASC
                """.format(", ".join(required_columns), base.get_detections_tablename(), confidence_threshold))
    
    x_col_index = required_columns.index("x_pos_hive")
    y_col_index = required_columns.index("y_pos_hive")
    cam_id_col_index = required_columns.index("cam_id")
    timestamp_col_index = required_columns.index("timestamp")

    all_velocities = []

    dt_current = dt_from
    is_first = True
    while dt_current < dt_to:
        is_last = False
        dt_current_end = dt_current + window_size
        if dt_current_end > dt_to:
            dt_current_end = dt_to
            is_last = True

        if is_first:
            cursor.execute("EXECUTE fetch_detections(%s, %s, %s)", (dt_current, dt_current_end, bee_id))
            is_first = False
        detections = cursor.fetchall()
        n_detections = len(detections)
        # Let DB server prepare next chunk.
        if not is_last:
            dt_next_end = dt_current_end + window_size
            if dt_next_end > dt_to:
                dt_next_end = dt_to
            cursor.execute("EXECUTE fetch_detections(%s, %s, %s)", (dt_current_end, dt_next_end, bee_id))

        def get_next_detection(det_idx, last_x, last_y, last_ts, last_cam_id):
            for next_idx in range(det_idx + 1, min(n_detections, det_idx + max_gap_length)):
                next_detection = detections[next_idx]
                _x, _y = next_detection[x_col_index], next_detection[y_col_index]
                _cam_id, _ts = next_detection[cam_id_col_index], next_detection[timestamp_col_index]
                distance = np.sqrt((last_x - _x) ** 2.0 + (last_y - _y) ** 2.0)

                if (_ts > last_ts) and (_cam_id == last_cam_id) and (distance <= max_distance):
                    return next_idx
            return -1

        last_detection = None
        for det_idx in range(n_detections):
            detection = detections[det_idx]
            x, y = detection[x_col_index], detection[y_col_index]
            cam_id, ts = detection[cam_id_col_index], detection[timestamp_col_index]

            if last_detection is not None:
                last_x, last_y, last_cam_id, last_ts = last_detection
                if last_ts >= ts:
                    continue

                distance = np.sqrt((last_x - x) ** 2.0 + (last_y - y) ** 2.0)
                if (last_cam_id != cam_id) or (distance > max_distance):
                    next_idx = get_next_detection(det_idx, last_x, last_y, last_ts, last_cam_id)
                    if next_idx != -1:
                        det_idx = next_idx
                        continue
            
            all_velocities.append(detection)
                
            last_detection = (x, y, cam_id, ts)

        dt_current = dt_current_end
    
    if len(all_velocities) <= 2:
        return None
        
    all_velocities = pd.DataFrame(all_velocities, columns=required_columns)
    timestamp_deltas = np.diff([dt.timestamp() for dt in all_velocities.timestamp])
    x, y = np.diff(all_velocities.x_pos_hive.values), np.diff(all_velocities.y_pos_hive.values)
    distances = np.sqrt(np.square(x) + np.square(y))
    v = distances / timestamp_deltas
    cam_differences = np.diff(all_velocities.cam_id.values)
    v[cam_differences != 0] = np.nan
    v[timestamp_deltas > max_time_distance] = np.nan
    v[distances > max_distance] = np.nan
    v[v > max_mm_per_second] = np.nan
    v = scipy.signal.medfilt(v, kernel_size=3)

    all_velocities = all_velocities.iloc[1:, :]
    all_velocities["time_passed"] = timestamp_deltas
    all_velocities["velocity"] = v
    all_velocities.rename(dict(timestamp="datetime"), axis=1, inplace=True)
    all_velocities["datetime"] = [dt.astimezone(pytz.UTC) for dt in all_velocities.datetime]
    required_columns = list(set(["datetime", "velocity", "time_passed"]) | set(additional_columns))
    all_velocities = all_velocities[required_columns]
    return all_velocities
