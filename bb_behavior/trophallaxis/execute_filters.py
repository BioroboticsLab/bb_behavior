from concurrent.futures import ProcessPoolExecutor
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
import pickle

from . import prefilter
from . import trajectory_filter
from .. import trajectory
from .. import utils
from .. import db

def get_job_filename(cam_id, dt_from, dt_to, target_dir):
    return target_dir + "/trophallaxis.{}_{}_{}.zip".format(
                    cam_id, str(dt_from), str(dt_to))

def generate_jobs(dt_from, dt_to, target_dir=None):
    written = 0
    already_done = 0
    already_open = 0
    already_in_progress = 0
    current_day_start = dt_from
    while current_day_start < dt_to:
        current_day_end = current_day_start + datetime.timedelta(days=1)

        for cam_id in range(4):
            job_name = get_job_filename(cam_id, current_day_start, current_day_end, target_dir)
            # Already finished?
            if os.path.isfile(job_name + ".zip"):
                already_done += 1
                continue
            # Already open?
            parameter_filename = job_name + ".job.pickle"
            if os.path.isfile(parameter_filename):
                already_open += 1
                continue
            # Already being worked on?
            temp_filename = parameter_filename + ".temp"
            if os.path.isfile(temp_filename):
                already_in_progress += 1
                continue
            with open(parameter_filename, "wb") as f:
                pickle.dump(dict(start=current_day_start, end=current_day_end, cam_id=cam_id), f)
            written += 1

        current_day_start = current_day_end

    print("Jobs created: {} (already open: {}, already done: {}, in progress: {})".format(written, already_open, already_done, already_in_progress))


def execute_job(job_filename, use_cuda, trajectory_model_path, max_workers, min_threshold, progress):
    if not os.path.isfile(job_filename):
        print("Job race condition.")
        return
    temp_filename = job_filename + ".temp"
    os.rename(job_filename, temp_filename)
    with open(temp_filename , "rb") as f:
        job = pickle.load(f)

    cam_id, start, end = job["cam_id"], job["start"], job["end"]
    try:
        # Load trajectory model.
        model = trajectory_filter.load_model(trajectory_model_path, use_cuda=use_cuda)
        
        job_results = []
        # Filter data.
        with db.DatabaseCursorContext() as cursor:
            for (timestamp, frame_id, cam_id) in progress(prefilter.iter_frames_to_filter(cam_id, start, end), leave=False):
                _, _, _, prefiltered_data = prefilter.process_frame_with_prefilter((timestamp, frame_id, cam_id), thread_context=cursor)

                features = trajectory_filter.load_features(prefiltered_data)
                if features is None:
                    print("Failed loading features for frame {}".format(frame_id))
                    continue
                filtered = trajectory_filter.predict(*features, min_threshold=min_threshold, thread_context=dict(model=model))
                job_results.append(filtered)
        
        # Save all results.
        job_results = pd.concat(job_results, axis=0)
        
        job_results.frame_id = job_results.frame_id.astype(np.uint64)
        job_results.bee_id0 = job_results.bee_id0.astype(np.uint16)
        job_results.bee_id1 = job_results.bee_id1.astype(np.uint16)
        job_results.score = job_results.score.astype(np.float32)
        job_results = list(job_results[["frame_id", "bee_id0", "bee_id1", "score"]].itertuples(index=False))

        output_filename = job_filename[:-len(".job.pickle")]
        with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zf:
                with zf.open(output_filename.split("/")[-1].replace("zip", "msgpack"), "w") as file:
                    msgpack.dump(job_results, file, use_bin_type=True)

        
        os.remove(temp_filename)

    except:
        os.rename(temp_filename, job_filename)
        raise

def execute_all_jobs(target_dir, progress="tqdm", use_cuda=True, trajectory_model_path=None, max_workers=32, min_threshold=0.1):
    if progress is None:
        progress = lambda x=None, **kwargs: x
    else:
        import tqdm
        if progress == "tqdm":
            progress = tqdm.tqdm
        elif progress == "tqdm_notebook":
            progress = tqdm.tqdm_notebook
    
    trajectory_model_path = trajectory_model_path or "/mnt/storage/david/cache/beesbook/trophallaxis/1dcnn.cache"

    open_jobs = [f for f in os.listdir(target_dir) if f.endswith(".job.pickle")]
    np.random.shuffle(open_jobs)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for job in open_jobs:
            future = executor.submit(execute_job, target_dir + "/" + job, use_cuda, trajectory_model_path, 
                                        max_workers, min_threshold, progress)
            futures.append(future)

        for future in progress(futures):
            future.result()

