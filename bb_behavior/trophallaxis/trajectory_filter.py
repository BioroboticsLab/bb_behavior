import datetime
import msgpack
import numba
import numpy as np
import pandas as pd
import prefetch_generator
import scipy.signal
import concurrent.futures
import torch
import tqdm
import os.path
import warnings
import zipfile

from . import prefilter
from .. import trajectory
from .. import utils
from .. import db

@numba.jit
def traj2cossin(traj):
    return traj[0, 2:4, :][::-1, :]
@numba.jit
def traj2headpos(traj):
    cossin = traj2cossin(traj)
    headpos = np.array([traj[0, 0, :] + 0.319 * cossin[0],
                        traj[0, 1, :] + 0.319 * cossin[1]])
    return headpos

def pairwise_measures(trajectories):
    diffs = list()
    min_angle = 0.0
    
    trajectories[0] = np.concatenate((trajectories[0], np.zeros(shape=(1, 3, trajectories[0].shape[2]), dtype=np.float32)), axis=1).astype(np.float32)
    traj0 = trajectories[0]
    
    head0 = traj2headpos(traj0)
    
    for idx, traj in enumerate(trajectories):
        if idx == 0:
            continue
        
        dist = np.linalg.norm(traj[0, 0:2, :] - traj0[0, 0:2, :], axis=0)
        motion_difference = np.zeros(shape=(1, 1, traj.shape[2]), dtype=np.float32)
        motion_difference[:, :, 1:] = np.linalg.norm(np.diff(traj[0, 0:2, :], axis=1) - np.diff(traj0[0, 0:2, :], axis=1), axis=0)
        head = traj2headpos(traj)
        diffs = [head0 - traj[0, 0:2, :], head - traj0[0, 0:2, :]]
        diffs = [d / np.linalg.norm(d, axis=0) for d in diffs]
        min_angle = np.array([min(
                abs(np.dot(traj[0, 2:4, i][::-1], diffs[0][:, i])),
                abs(np.dot(traj0[0, 2:4, i][::-1], diffs[1][:, i]))
            ) for i in range(traj.shape[2])])
        
        trajectories[idx] = np.concatenate((traj,
                                            dist.reshape(1, 1, dist.shape[0]),
                                            motion_difference,
                                            min_angle.reshape(1, 1, min_angle.shape[0])
                                           ),
                                           axis=1).astype(np.float32)
@numba.jit(parallel=True)
def smooth_temporal_features(trajectories):
    for traj in trajectories:
        for f in numba.prange(traj.shape[1]):
            traj[0, f, :] = scipy.signal.medfilt(traj[0, f, :], kernel_size=3)

feature_procs = [trajectory.FeatureTransform.Normalizer(downscale_by=10.0),
                 trajectory.FeatureTransform.Angle2Geometric(),
                 trajectory.FeatureTransform(fun=pairwise_measures,
                                                    input=("x", "y", "r_sin", "r_cos", "mask"),
                                                    output=("x", "y", "r_sin", "r_cos", "mask", "dist", "motion_difference", "head_angle")),
                 trajectory.FeatureTransform(fun=smooth_temporal_features,
                                                    input=("x", "y", "r_sin", "r_cos", "mask", "dist", "motion_difference", "head_angle"))]


def load_features(data):
    import bb_behavior.trajectory
    datareader = bb_behavior.trajectory.DataReader(dataframe=data, sample_count=None, target_column=None,
                                              feature_procs=feature_procs, n_threads=2, progress=None, frame_margin=7,
                                              use_hive_coords=True, chunk_frame_id_queries=True, verbose=False)
    try:
        datareader.create_features()
    except Exception as e:
        from IPython.display import display
        display(data)
        print("Chunk failed with {}".format(str(e)))
        return None
    return datareader.X, datareader.samples, datareader._valid_sample_indices

def load_model(trajectory_model_path="/mnt/storage/david/cache/beesbook/trophallaxis/1dcnn.cache", use_cuda=True):
    torch_kwargs = dict()
    if not use_cuda:
        torch_kwargs["map_location"] = torch.device("cpu")
    return torch.load(trajectory_model_path, **torch_kwargs)

def predict(X, samples, valid_sample_indices, min_threshold=0.0, thread_context=None, **kwargs):
    if not "model" in thread_context:
        thread_context["model"] = thread_context["model_factory"]()
    model = thread_context["model"]
    Y = model.predict_proba(X)[:, 1]
    results = []
    for idx in range(Y.shape[0]):
        y = Y[idx]
        if y >= min_threshold:    
            sample_idx = valid_sample_indices[idx]
            frame_id, bee_id0, bee_id1 = samples.frame_id.iloc[sample_idx], \
                                            samples.bee_id0.iloc[sample_idx], \
                                            samples.bee_id1.iloc[sample_idx]
            results.append(dict(frame_id=frame_id, bee_id0=bee_id0, bee_id1=bee_id1, score=y))
    results = pd.DataFrame(results)
    return results

def get_available_processed_days(base_path=None):
    """Loads the available processed trajectory data chunks' metadata and returns them as a data frame.
    Arguments:
        base_path: string
            Path of the directory that contains the files matching 'trajfilter.*.zip'.
    Returns:
        pandas.DataFrame
        Containing at least the columns cam_id, begin, end, filename.
    """
    import glob
    if base_path is None:
        base_path = "/mnt/storage/david/cache/beesbook/trophallaxis"
    available_files = set()
    for ext in ("zip",):
        available_files |= set(glob.glob(base_path + "/trophallaxis.*." + ext))
    
    available_files = list(sorted(list(available_files)))
    available_files_df = []
    for filename in available_files:
        leaf_name = filename.split("/")[-1]
        infos = leaf_name.split(".")[1]
        infos = infos.split("_")
        cam_id = int(infos[0])
        datetime_from, datetime_to = [datetime.datetime.strptime(dt.split("+")[0], "%Y-%m-%d %H:%M:%S") for dt in infos[1:]]
        available_files_df.append(dict(
            cam_id=cam_id,
            begin=datetime_from,
            end=datetime_to,
            filename=filename
        ))
        
    available_files_df = pd.DataFrame(available_files_df)
    return available_files_df

def load_processed_data(f, threshold):
    """Takes a file path containing trajectory data and loads it, returning a data frame with additional metadata.

    Arguments:
        f: string
            Path to trajectory data archive file.
        threshold:
            Classifier threshold to filter the results before querying the metadata from the database.

    Returns:
        pandas.DataFrame
    """
    data = None
    try:
        with zipfile.ZipFile(f, "r", zipfile.ZIP_DEFLATED) as zf:
            with zf.open(f.split("/")[-1].replace("zip", "msgpack"), "r") as file:
                data = msgpack.load(file)
    except Exception as e:
        print("Error loading {}".format(f))
        print(str(e))
        return None

    if len(data) == 0:
        return None
    bee_id0, bee_id1, frame_id, score = zip(*data)
    frame_id = pd.Series(frame_id, dtype=np.uint64)
    bee_id0 = pd.Series(bee_id0, dtype=np.uint16)
    bee_id1 = pd.Series(bee_id1, dtype=np.uint16)
    score = pd.Series(score, dtype=np.float32)
    data = [frame_id, bee_id0, bee_id1, score]
    data = pd.concat(data, axis=1)
    data.columns = ["frame_id", "bee_id0", "bee_id1", "score"]
    data = data[(~pd.isnull(data.score)) & (data.score >= threshold)]
    all_frame_ids = data.frame_id.unique()
    
    metadata = db.metadata.get_frame_metadata(all_frame_ids)
    metadata.frame_id = metadata.frame_id.astype(np.uint64)
    metadata["datetime"] = pd.to_datetime(metadata.timestamp, unit="s")
    
    data = data.merge(metadata, on="frame_id", how="inner")
    return data

def load_all_processed_data(paths, threshold=0.0):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        data = executor.map(lambda f: load_processed_data(f, threshold=threshold), paths)
    return pd.concat([d for d in data if d is not None], axis=0, ignore_index=True)