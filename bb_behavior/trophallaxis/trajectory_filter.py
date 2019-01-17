import msgpack
import numba
import numpy as np
import pandas as pd
import prefetch_generator
import concurrent.futures
import torch
import tqdm
import os.path
import zipfile

from . import prefilter
from .. import trajectory
from .. import utils

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
    #nonlocal n_features_loaded
    #nonlocal n_failed_chunks
    datareader = bb_behavior.trajectory.DataReader(dataframe=data, sample_count=None, target_column=None,
                                              feature_procs=feature_procs, n_threads=2, progress=None, frame_margin=7,
                                              use_hive_coords=True, chunk_frame_id_queries=True, verbose=False)
    try:
        datareader.create_features()
    except Exception as e:
        #n_failed_chunks += 1
        print("Chunk failed with {}".format(str(e)))
        return None
    #n_features_loaded += 1
    return datareader.X, datareader.samples, datareader._valid_sample_indices

def to_output_filename(path):
    return path.replace("prefilter", "trajfilter").replace("msgpack", "zip")

def process_preprocessed_data(progress="tqdm"):
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
            data = prefilter.load_processed_data(filepath)
            if data is None:
                continue
            data.sort_values("frame_id", inplace=True)
            yield filepath, data
    
    n_features_loaded = 0
    n_failed_chunks = 0
    
    
    #def predict(datareader, thread_context, **kwargs):
    def predict(X, samples, valid_sample_indices, thread_context, **kwargs):
        if not "model" in thread_context:
            thread_context["model"] = torch.load("/mnt/storage/david/cache/beesbook/trophallaxis/1dcnn.cache")
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
                
                chunk_range.set_postfix(chunk_results_size="{:01d}".format(results.shape[0]),
                                        chunk_size="{:01d}".format(batchsize),
                                    chunks_loaded="{:01d}".format(n_features_loaded),
                                    chunks_failed="{:01d}".format(n_failed_chunks))
            chunk_results.append(results)
        
         
        def generate_chunks():
            yield from utils.iterate_minibatches(filepath_data, targets=None, batchsize=batchsize) 
        def generate_chunked_features():
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=16)
            chunks = executor.map(load_features, generate_chunks())
            yield from chunks
        
        pipeline = utils.ParallelPipeline([generate_chunked_features, predict, save_chunk_results],
                                                      n_thread_map={1:3}, thread_context_factory=lambda: ThreadCtx())
        n_features_loaded = 0
        pipeline()
        if chunk_range is not None:
            chunk_range.close()
        
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
