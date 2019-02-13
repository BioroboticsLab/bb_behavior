import datetime
import msgpack
import numba
import numpy as np
import pandas as pd
import prefetch_generator
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
    headpos = traj[0, 0:2, :] + 0.319 * traj2cossin(traj)
    return headpos

#@numba.jit
def pairwise_measures(trajectories):
    diffs = list()
    min_angle = 0.0
    
    trajectories[0] = np.concatenate((trajectories[0], np.zeros(shape=(1, 1, trajectories[0].shape[2]), dtype=np.float32)), axis=1).astype(np.float32)
    traj0 = trajectories[0]

    head0 = traj2headpos(traj0)
    
    for idx, traj in enumerate(trajectories):
        if idx == 0:
            continue
        
        dist = np.linalg.norm(traj[0, 0:2, :] - traj0[0, 0:2, :], axis=0)
        head = traj2headpos(traj)
        diffs = [head0 - traj[0, 0:2, :], head - traj0[0, 0:2, :]]
        diffs = [d / np.linalg.norm(d, axis=0) for d in diffs]
        min_angle = np.array([min(
                abs(np.dot(traj[0, 2:4, i][::-1], diffs[0][:, i])),
                abs(np.dot(traj0[0, 2:4, i][::-1], diffs[1][:, i]))
            ) for i in range(traj.shape[2])])
        
        trajectories[idx] = np.concatenate((traj,
                                            dist.reshape(1, 1, dist.shape[0]),
                                            min_angle.reshape(1, 1, min_angle.shape[0])
                                           ),
                                           axis=1).astype(np.float32)
             

feature_procs = [trajectory.FeatureTransform.Normalizer(downscale_by=10.0),
                 trajectory.FeatureTransform.Angle2Geometric(),
                 trajectory.FeatureTransform(fun=pairwise_measures,
                                                    input=("x", "y", "r_sin", "r_cos", "mask"),
                                                    output=("x", "y", "r_sin", "r_cos", "mask", "dist", "head_angle"))]


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

def to_output_filename(path):
    return path.replace("prefilter", "trajfilter").replace("msgpack", "zip")

def process_preprocessed_data(progress="tqdm", use_cuda=True, n_loader_processes=16, n_prediction_threads=3):
    class ThreadCtx():
        def __enter__(self):
            return dict()
        def __exit__(self, *args):
            pass
    progress_bar_fun = lambda x, **kwargs: x
    if progress == "tqdm":
        import tqdm
        progress_bar_fun = tqdm.tqdm
    elif progress == "tqdm_notebook":
        import tqdm
        progress_bar_fun = tqdm.tqdm_notebook

    available_files_df = prefilter.get_available_processed_days()
    processed = list(map(lambda x: os.path.isfile(to_output_filename(x)), available_files_df.filename.values))
    available_files_df = available_files_df.iloc[~np.array(processed), :]
    
    def iter_and_load_data():
        for cam_id, dt_begin, dt_end, filepath in available_files_df[["cam_id", "begin", "end", "filename"]].itertuples(index=False):
            if os.path.isfile(to_output_filename(filepath)):
                print("Skipping {}".format(filepath))
                continue
            data = None
            try:
                data = prefilter.load_processed_data(filepath, warnings_as_errors=True)
            except Exception as e:
                e = "Error! Skipping file {}. [{}]".format(filepath, str(e))
                print(e)
                continue
            if data is None:
                continue
            data.sort_values("frame_id", inplace=True)
            yield filepath, data
    
    n_features_loaded = 0
    
    def predict(X, samples, valid_sample_indices, thread_context, **kwargs):
        if not "model" in thread_context:
            torch_kwargs = dict()
            if not use_cuda:
                torch_kwargs["map_location"] = torch.device("cpu")
            thread_context["model"] = torch.load("/mnt/storage/david/cache/beesbook/trophallaxis/1dcnn.cache", **torch_kwargs)
            if not use_cuda:
                thread_context["model"].use_cuda = False
        model = thread_context["model"]
        Y = model.predict_proba(X)[:, 1]
        results = []
        #samples = datareader.samples
        for idx in range(Y.shape[0]):
            y = Y[idx]
            sample_idx = valid_sample_indices[idx]
            frame_id, bee_id0, bee_id1 = samples.frame_id.iloc[sample_idx], \
                                            samples.bee_id0.iloc[sample_idx], \
                                            samples.bee_id1.iloc[sample_idx]
            results.append(dict(frame_id=frame_id, bee_id0=bee_id0, bee_id1=bee_id1, score=y))
        results = pd.DataFrame(results)
        return results
    
    generator = prefetch_generator.BackgroundGenerator(iter_and_load_data(), max_prefetch=1)
    trange = progress_bar_fun(generator, desc="Input", total=available_files_df.shape[0])
    total_output = 0
    for filepath, filepath_data in trange:
        if progress is not None:
            trange.set_postfix(classified_samples=total_output)
        
        batchsize = 2000
        n_chunks = (filepath_data.shape[0] // batchsize) + 1
        
        
        
        chunk_results = []
        chunk_range = None
        def save_chunk_results(results, **kwargs):
            nonlocal chunk_results
            nonlocal chunk_range
            nonlocal n_features_loaded
            if progress is not None:
                if chunk_range is None:
                    chunk_range = progress_bar_fun(total=n_chunks - 1, desc="Chunks")
                else:
                    chunk_range.update()
            chunk_results.append(results)
        
         
        def generate_chunks():
            yield from utils.iterate_minibatches(filepath_data, targets=None, batchsize=batchsize) 
        def generate_chunked_features():
            yield from utils.prefetch_map(load_features, generate_chunks(), max_workers=n_loader_processes)
        
        pipeline = utils.ParallelPipeline([generate_chunked_features, predict, save_chunk_results],
                                                      n_thread_map={1:n_prediction_threads}, thread_context_factory=lambda: ThreadCtx())
        n_features_loaded = 0
        pipeline()
        if chunk_range is not None:
            chunk_range.close()
        if len(chunk_results) == 0:
            print("No results for {}".format(filepath))
            continue
        results_df = pd.concat(chunk_results, axis=0)
        
        df = filepath_data.merge(results_df, how="left", on=("frame_id", "bee_id0", "bee_id1"))
        df = df[["frame_id", "bee_id0", "bee_id1", "score"]]
        total_output += results_df.shape[0]
        
        df.frame_id = df.frame_id.astype(np.uint64)
        df.bee_id0 = df.bee_id0.astype(np.uint16)
        df.bee_id1 = df.bee_id1.astype(np.uint16)
        df.score = df.score.astype(np.float32)
        raw_df = list(df.itertuples(index=False))

        output_filename = to_output_filename(filepath)
        with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zf:
                with zf.open(output_filename.split("/")[-1].replace("zip", "msgpack"), "w") as file:
                    msgpack.dump(raw_df, file, use_bin_type=True)

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
        available_files |= set(glob.glob(base_path + "/trajfilter.*." + ext))
    
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
    from IPython.display import display
    frame_id, bee_id0, bee_id1, score = zip(*data)
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