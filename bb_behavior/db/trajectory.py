import math
import numba
import numpy as np

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
    results = []
    for n_idx, (timestamp, frame_id, _) in enumerate(frames):
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
        with base.get_database_connection(application_name="get_bee_detections") as db:
            return get_bee_detections(bee_id, verbose=verbose, frame_id=frame_id, frames=frames, cursor=db.cursor(), **kwargs)
    
    frames = frames or sampling.get_neighbour_frames(frame_id=frame_id, cursor=cursor, cursor_is_prepared=cursor_is_prepared, **kwargs)
    frame_ids = [int(f[1]) for f in frames if f[1] is not None]
    
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
    
    return get_consistent_track_from_detections(frames, detections, verbose=verbose)

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
    detections = detections or get_bee_detections(bee_id, frame_id=frame_id, frames=frames, **kwargs)
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
