from ..utils.interaction_model_evaluation import order_bee_ids, evaluate_interaction_model
from . import prefilter, trajectory_filter
from ..io import biotracker
from .. import db
import numpy as np
import pandas as pd

def load_test_set(root_path=None):
    root_path = root_path or "/mnt/storage/david/data/beesbook/trophallaxis/testset/"
    import os
    file_ending = ".annotations.json"
    annotation_files = [f for f in os.listdir(root_path) if f.endswith(file_ending)]
    print("Found {} annotated files.".format(len(annotation_files)))

    all_annotations = []
    annotated_frame_ids = []
    for f_idx, f in enumerate(annotation_files):
        filename = f.split(".")[0]
        frame_id = int(filename.split("_")[1])
        annotated_frame_ids.append(frame_id)
        filename = root_path + filename
        annotations = biotracker.load_annotated_bee_video(video_filename=filename+".mp4", object_name="detection_idx")
        #tracks = bb_behavior.io.biotracker.load_tracks(filename+".json", using_bee_ids=False)
        #display(annotations.head())
        annotations["frame_id"] = frame_id
        annotations.frame_id = annotations.frame_id.astype(np.uint64)
        all_annotations.append(annotations)
    annotated_frame_ids = set(annotated_frame_ids)
    print("Total annotated frame IDs: {}".format(len(annotated_frame_ids)))
    all_annotations = pd.concat(all_annotations, axis=0)
    trophallaxis_annotations = all_annotations[all_annotations.comment.str.contains("trophallaxis")]
    trophallaxis_annotations = trophallaxis_annotations[~pd.isnull(trophallaxis_annotations.bee_id1)]
    gt_data = trophallaxis_annotations[["frame_id", "bee_id0", "bee_id1"]].copy()
    gt_data.frame_id = gt_data.frame_id.astype(np.uint64)
    gt_data.bee_id0 = gt_data.bee_id0.astype(np.uint16)
    gt_data.bee_id1 = gt_data.bee_id1.astype(np.uint16)
    return annotated_frame_ids, gt_data

def apply_prefilter_to_frame(frame_id, **kwargs):
    _, _, _, data = prefilter.get_data_for_frame_id_high_recall(None, frame_id, None)
    
    return data

def apply_prefilter_to_frame_sequence(frame_id, cursor=None):
    neighbour_frames = db.get_neighbour_frames(frame_id=frame_id, n_frames=5,
                                                           cursor=cursor, cursor_is_prepared=cursor is not None)
    frame_ids = [f_id for (_, f_id, _) in neighbour_frames[::2]]
    results = []
    for frame_id in frame_ids:
        res = apply_prefilter_to_frame(frame_id, cursor=cursor)
        if res.shape[0] > 0:
            res = set(order_bee_ids(b0, b1) for b0, b1 in res[["bee_id0", "bee_id1"]].itertuples(index=False))
        else:
            res = set()
        results.append(res)
    before_results = set().union(*results[:(len(results)//2)])
    after_results  = set().union(*results[ (len(results)//2):])
    results = list(before_results & after_results)
    return pd.DataFrame(results, columns=["bee_id0", "bee_id1"])

def apply_prefilter_and_trajfilter_to_frame(frame_id, threshold=0.5, cursor=None):
    import torch
    _, _, _, data = prefilter.get_data_for_frame_id_high_recall(
                                                    None, frame_id, None, 
                                                    cursor=cursor, cursor_is_prepared=cursor is not None)
    X, samples, valid = trajectory_filter.load_features(data)
    
    model = torch.load("/mnt/storage/david/cache/beesbook/trophallaxis/1dcnn.cache")
    Y = model.predict_proba(X)[:, 1]
    df = []
    for idx, (sample_idx, y) in enumerate(zip(valid, Y)):
        if y < threshold:
            continue
        df.append(dict(
            bee_id0=samples.bee_id0.iloc[sample_idx],
            bee_id1=samples.bee_id1.iloc[sample_idx]))
    df = pd.DataFrame(df)
    return df

def apply_prefilter_and_trajfilter_to_frame_sequence(frame_id, threshold=0.5, cursor=None):
    neighbour_frames = db.get_neighbour_frames(frame_id=frame_id, n_frames=5,
                                                           cursor=cursor, cursor_is_prepared=cursor is not None)
    frame_ids = [f_id for (_, f_id, _) in neighbour_frames[::2]]
    results = []
    for frame_id in frame_ids:
        res = apply_prefilter_and_trajfilter_to_frame(frame_id, threshold=threshold, cursor=cursor)
        if res.shape[0] > 0:
            res = set(order_bee_ids(b0, b1) for b0, b1 in res[["bee_id0", "bee_id1"]].itertuples(index=False))
        else:
            res = set()
        results.append(res)
    before_results = set().union(*results[:(len(results)//2)])
    after_results  = set().union(*results[ (len(results)//2):])
    results = list(before_results & after_results)
    return pd.DataFrame(results, columns=["bee_id0", "bee_id1"])

def evaluate(root_path="/mnt/storage/david/data/beesbook/trophallaxis/testset/"):
    frame_ids, ground_truth = load_test_set()


    evaluate_interaction_model(apply_prefilter_to_frame,
               frame_ids, ground_truth, label="Trophallaxis Prefilter")
    evaluate_interaction_model(apply_prefilter_to_frame_sequence,
               frame_ids, ground_truth, label="Trophallaxis Prefilter (Multi-Frame)")
    evaluate_interaction_model(apply_prefilter_and_trajfilter_to_frame,
               frame_ids, ground_truth, label="Trajectory Filter")
    evaluate_interaction_model(apply_prefilter_and_trajfilter_to_frame_sequence,
               frame_ids, ground_truth, label="Trajectory Filter (Multi-Frame)")
    