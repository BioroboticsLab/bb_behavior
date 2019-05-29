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
           "SELECT timestamp, frame_id, x_pos_hive AS x, y_pos_hive AS y, orientation_hive as orientation, track_id FROM bb_detections_2016_stitched "
           "WHERE frame_id = ANY($1) AND bee_id = $2 ORDER BY timestamp ASC")

        self._cursor.execute("PREPARE find_interaction_candidates AS "
            "SELECT x_pos_hive, y_pos_hive, orientation_hive, bee_id, detection_idx, cam_id FROM bb_detections_2016_stitched "
            "WHERE frame_id = $1 AND bee_id_confidence >= $2")

        # For metadata.get_frame_metadata
        self._cursor.execute("PREPARE get_frame_metadata AS "
           "SELECT frame_id, timestamp, index, fc_id FROM plotter_frame WHERE frame_id = ANY($1)")
        # For metadata.get_frame_metadata
        self._cursor.execute("PREPARE get_frame_container_info AS "
          "SELECT video_name FROM plotter_framecontainer "
          "WHERE id = $1 LIMIT 1")
          
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
                        cursor=None, cursor_is_prepared=False, make_consistent=True, **kwargs):
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
            return get_bee_detections(bee_id, verbose=verbose, frame_id=frame_id, frames=frames, cursor=db.cursor(), **kwargs)
    
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
        cursor.execute("SELECT timestamp, frame_id, " + coords_string + ", track_id FROM bb_detections_2016_stitched WHERE frame_id=ANY(%s) AND bee_id = %s ORDER BY timestamp ASC",
                        (frame_ids, bee_id))
    else:
        prepared_statement_name = "get_bee_detections" if not use_hive_coords else "get_bee_detections_hive_coords"
        cursor.execute("EXECUTE " + prepared_statement_name + " (%s, %s)", (frame_ids, bee_id))
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
    mask = None
    if interpolate:
        mask = interpolate_trajectory(trajectory)
    return trajectory, mask

def get_track(track_id, frames, use_hive_coords=False, bee_id=None, cursor=None, make_consistent=True, interpolate=True):
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
           "SELECT detection_idx, timestamp, frame_id, {} AS x, {} AS y, {} as orientation, track_id FROM bb_detections_2016_stitched "
           "WHERE {} track_id = %s ORDER BY timestamp ASC".format(*coords, frame_condition), query_arguments)
    track = cursor.fetchall()
    detection_keys = {t[2]: t[0] for t in track}
    if make_consistent:
        track = get_consistent_track_from_detections(frames, [t[1:] for t in track])

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
                       confidence_threshold=0.1, fixup_velocities=True,
                       additional_columns=set()):
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
            Whether to assume that the timestamps are at 3Hz and smoothing them is okay.
        additional_columns: iterable(string)
            Iterable of additional column names to query from the database.
    """
    if not cursor:
        from contextlib import closing
        with closing(base.get_database_connection("get_bee_velocities")) as con:
            return get_bee_velocities(bee_id, dt_from, dt_to, cursor=con.cursor(), cursor_is_prepared=False,
                                      progress=progress, confidence_threshold=confidence_threshold, fixup_velocities=fixup_velocities,
                                      additional_columns=additional_columns)
    
    import pytz
    import scipy.signal

    required_columns = list(set(("timestamp", "x_pos_hive", "y_pos_hive", "orientation_hive", "track_id")) | set(additional_columns))

    if not cursor_is_prepared:
        cursor.execute("""PREPARE fetch_track_ids_for_bee AS
                SELECT 
                    track_id, MIN(timestamp), MAX(timestamp)
                    FROM bb_detections_2016_stitched
                    WHERE timestamp >= $1
                       AND timestamp <= $2
                       AND bee_id = $3
                    GROUP BY track_id
                    """)
        
        cursor.execute("""PREPARE fetch_tracks AS
                SELECT {}
                   FROM bb_detections_2016_stitched 
                   WHERE
                       track_id = ANY($1)
                       AND bee_id_confidence > {}
                       ORDER BY track_id, timestamp ASC
                """.format(", ".join(required_columns), confidence_threshold))
    
    progress_bar = lambda x: x
    if progress == "tqdm":
        from tqdm import tqdm
        progress_bar = tqdm
    elif progress == "tqdm_notebook":
        from tqdm import tqdm_notebook
        progress_bar = tqdm_notebook
    
    query_args = (dt_from, dt_to, bee_id)
    cursor.execute("EXECUTE fetch_track_ids_for_bee (%s, %s, %s)", query_args)
    track_ids = cursor.fetchall()
    track_ids = list(sorted(track_ids, key=lambda x: x[1]))
    track_ids = [track_ids[i][0] for i in range(len(track_ids)-1) if track_ids[i][2] > track_ids[i+1][1]]
    
    cursor.execute("EXECUTE fetch_tracks (%s)", (track_ids, ))
    all_track_data = cursor.fetchall()

    def iterate_tracks():
        track_id_index = required_columns.index("track_id")
        track_data = []
        dummy_row = (None,) * len(required_columns)
        for row in itertools.chain(all_track_data, [dummy_row]):
            if track_data and row[track_id_index] != track_data[-1][track_id_index]:
                yield track_data[-1][track_id_index], track_data
                track_data = []
            track_data.append(row)

    all_velocities = []
    
    for _, track in progress_bar(iterate_tracks()):
        if not track:
            continue
        value_series = tuple(zip(*track))
        datetimes = value_series[required_columns.index("timestamp")]
        if len(datetimes) < 2:
            continue
        x = value_series[required_columns.index("x_pos_hive")]
        y = value_series[required_columns.index("y_pos_hive")]
        
        timestamps = [dt.timestamp() for dt in datetimes]
        x, y, timestamps = np.diff(x), np.diff(y), np.diff(timestamps)
        assert np.all(timestamps > 0.0)
        
        v = np.sqrt(np.square(x) + np.square(y))
        
        if fixup_velocities:
            timestamps = np.round(timestamps * 3.0) / 3.0
        v = v / timestamps
        v = scipy.signal.medfilt(v, kernel_size=3)
        
        columns_dict = dict(
                velocity=v,
                time_passed=timestamps,
                datetime=list(map(lambda x: x.replace(tzinfo=pytz.UTC), datetimes[:-1])))
        for additional_column in additional_columns:
            columns_dict[additional_column] = value_series[required_columns.index(additional_column)][1:]

        df = pd.DataFrame(columns_dict)
        all_velocities.append(df)
    if not all_velocities:
        return None
    all_velocities = pd.concat(all_velocities, axis=0)
    all_velocities = all_velocities[(all_velocities.datetime >= dt_from) & (all_velocities.datetime <= dt_to)]
    return all_velocities
