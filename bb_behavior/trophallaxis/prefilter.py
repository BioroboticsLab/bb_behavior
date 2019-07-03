from concurrent.futures import ProcessPoolExecutor
import datetime
import msgpack
import numba
import numpy as np
import os.path
import pandas as pd
import warnings
import zipfile

from ..db import find_interactions_in_frame, get_frame_metadata, get_frames

@numba.njit
def calculate_head_distance(xy0, xy1):
    head0 = np.array([xy0[0], xy0[1]])
    head0[0] += 3.19 * np.cos(xy0[2])
    head0[1] += 3.19 * np.sin(xy0[2])
    head1 = np.array([xy1[0], xy1[1]])
    head1[0] += 3.19 * np.cos(xy1[2])
    head1[1] += 3.19 * np.sin(xy1[2])
    
    d = 0.0
    d = np.linalg.norm(head0 - head1)
    return d

@numba.njit
def calculate_angle_dot_distance(r0, r1):
    vec0 = np.array([np.cos(r0), np.sin(r0)])
    vec1 = np.array([np.cos(r1), np.sin(r1)])
    d = 0.0
    d = np.dot(vec0, vec1)
    return d
    
@numba.njit
def probability_distance_fun_(xy0, xy1, beta0, beta1, beta2, bias, hard_min, hard_max):
    distance = np.linalg.norm(xy0[:2] - xy1[:2])
    if distance <= hard_min or distance >= hard_max:
        return 0.0

    head_distance = calculate_head_distance(xy0, xy1)
    angle_dot_distance = calculate_angle_dot_distance(xy0[2], xy1[2])
    
    distance = distance * beta0 + head_distance * beta1 + angle_dot_distance * beta2 + bias
    distance = (1.0 / (1.0 + np.exp(-distance)))
    return distance

@numba.njit
def probability_distance_fun_vectorized_(xy0, xy1, out):
    out = np.zeros(shape=(xy0.shape[0],), dtype=np.float32)
    for i in range(xy0.shape[0]):
        out[i] = probability_distance_fun(xy0[i,:], xy1[i,:])
        
@numba.jit
def probability_distance_fun_vectorized(xy0, xy1):
    out = np.zeros(shape=(xy0.shape[0],), dtype=np.float32)
    return probability_distance_fun_vectorized_(xy0, xy1, out)

def get_data_for_frame_id(timestamp, frame_id, cam_id,
                                  max_distance, min_distance, distance_func,
                                  thread_context=None, **kwargs):
    r = find_interactions_in_frame(
            frame_id, max_distance=max_distance, min_distance=min_distance,
            distance_func=distance_func,
                features=["x_pos_hive", "y_pos_hive", "orientation_hive"],
            cursor=thread_context, cursor_is_prepared=thread_context is not None)
    
    core_data = [i[:3] for i in r]
    core_data = pd.DataFrame(core_data, columns=("frame_id", "bee_id0", "bee_id1"), dtype=np.uint64)
    return timestamp, frame_id, cam_id, core_data

@numba.njit
def probability_distance_fun(xy0, xy1):
    return probability_distance_fun_(xy0, xy1, 2.19928889, -1.57782416, 1.75782411, -13.51532839,
                                   7.31, 12.04) # The min/max distance are 99 percentiles.
high_recall_threshold = 0.45185223  # 85% recall, 21% precision

def get_data_for_frame_id_high_recall(*args, min_distance=high_recall_threshold, 
                                        max_distance=2.0, distance_func='auto', **kwargs):
    if distance_func == 'auto':
        distance_func = probability_distance_fun

    return get_data_for_frame_id(*args, 
                     min_distance=min_distance, max_distance=max_distance,
                     distance_func=distance_func,
                     **kwargs)

def process_frame_with_prefilter(frame_info, **kwargs):
    results = get_data_for_frame_id_high_recall(*frame_info, **kwargs)
    return results
            
def iter_frames_to_filter(cam_id, from_, to_):
    all_frames = list(get_frames(cam_id, from_.timestamp(), to_.timestamp()))

    for idx, (timestamp, frame_id, cam_id) in enumerate(all_frames):
        if idx % 3 == 0:
            yield (timestamp, frame_id, cam_id)
