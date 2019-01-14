from tqdm import tqdm_notebook
import numpy as np
import datetime, pytz

from ..db import sample_frame_ids, get_interpolated_trajectory, DatabaseCursorContext, find_interactions_in_frame
from .features import is_valid_relative_rotation
from ..utils.processing import ParallelPipeline

def iter_frames(number_of_frames, dt_from=None, dt_to=None):
    dt_from = dt_from or datetime.datetime(year=2016, month=8, day=1, tzinfo=pytz.UTC)
    dt_to = dt_to or datetime.datetime(year=2016, month=9, day=1, tzinfo=pytz.UTC)
    samples = sample_frame_ids(number_of_frames, ts_from=dt_from.timestamp(), ts_to=dt_to.timestamp())
    
    trange = tqdm_notebook(samples)
    for sample in trange:
        yield sample[0]

def get_interactions_for_frame_id(frame_id, 
                                  max_distance, min_distance,
                                  thread_context=None, **kwargs):
    r = find_interactions_in_frame(
            frame_id, max_distance=max_distance, min_distance=min_distance,
            cursor=thread_context)
    for interaction in r:
        yield (interaction,)

def filter_orientations(interaction, minimum_relative_orientation, **kwargs):
    x1, y1, r1 = interaction[5:8]
    x2, y2, r2 = interaction[8:11]
    
    valid = is_valid_relative_rotation(np.array([x1, y1]), r1, np.array([x2, y2]), r2,
                                      minimum_relative_orientation)
    if not valid:
        return None
    return (interaction,)

def validate_interaction_duration(interaction,
                                  max_distance, min_distance, minimum_relative_orientation,
                                  thread_context=None, **kwargs):
    frame_id = interaction[0]
    bee_ids = interaction[1:3]
    
    n_margin = 3
    
    trajs = [get_interpolated_trajectory(
                int(bee_id),
                frame_id=frame_id, n_frames=n_margin, use_hive_coords=True,
                interpolate=True, verbose=False,
                cursor=thread_context, cursor_is_prepared=(thread_context is not None)) for bee_id in bee_ids]
    masks = [trajs[0][1], trajs[1][1]]
    trajs = [trajs[0][0], trajs[1][0]]
    if np.any(np.isnan(trajs)): # Is at least one trajectory invalid?
        return None
    
    distances = np.linalg.norm(trajs[0][:, :2] - trajs[1][:, :2], axis=1)
    valid_distances = (distances > min_distance) & (distances <= max_distance)
    
    valid_angles = np.array([is_valid_relative_rotation(trajs[0][i, :2], trajs[0][i, 2],
                                               trajs[1][i, :2], trajs[1][i, 2],
                                               minimum_relative_orientation)
                    for i in range(trajs[0].shape[0])])
    valid = valid_distances & valid_angles
    
    if np.sum(valid) >= 3:
        return (interaction, )
    return None

def get_trophallaxis_samples(number_of_frames, max_distance, min_distance, minimum_relative_orientation, dt_from=None, dt_to=None):
    def make_thread_context():
        return DatabaseCursorContext(application_name="Troph. Sample")
    
    all_interaction_results = []
    def save_interaction(interaction, **kwargs):
        nonlocal all_interaction_results
        all_interaction_results.append(interaction)
    
    def _get_interactions_for_frame_id(*args, thread_context=None, **kwargs):
        yield from get_interactions_for_frame_id(*args, max_distance=max_distance, min_distance=min_distance,
                    thread_context=thread_context, **kwargs)

    data_source = iter_frames
    if dt_from is not None and dt_to is not None:
        def iter_frames_from_range(n_samples):
            yield from iter_frames(n_samples, dt_from=dt_from, dt_to=dt_to)
        data_source = iter_frames_from_range

    pipeline = ParallelPipeline([data_source, _get_interactions_for_frame_id,
                                lambda *args, thread_context=None, **kwargs: filter_orientations(*args,
                                        minimum_relative_orientation=minimum_relative_orientation,
                                        thread_context=thread_context, **kwargs),
                                lambda *args, thread_context=None, **kwargs: validate_interaction_duration(*args,
                                        max_distance=max_distance, min_distance=min_distance,
                                        minimum_relative_orientation=minimum_relative_orientation, thread_context=thread_context, **kwargs),
                                save_interaction],
                                n_thread_map={3:4},
                                thread_context_factory=make_thread_context)
    pipeline(number_of_frames)
    np.random.shuffle(all_interaction_results)
    return all_interaction_results