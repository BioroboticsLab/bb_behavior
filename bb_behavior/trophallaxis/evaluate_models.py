from ..utils.interaction_model_evaluation import order_bee_ids, evaluate_interaction_model
from . import prefilter, trajectory_filter, image_sequences
from ..io import biotracker
from .. import db
import numpy as np
import pandas as pd
import os.path

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

def apply_prefilter_and_imagefilter_to_frame(frame_id,
                                            video_root, video_cache_path, output_path,
                                            threshold=0.5, cursor=None, gt=None):
    import torch
    _, _, _, data = prefilter.get_data_for_frame_id_high_recall(
                                                    None, frame_id, None, 
                                                    cursor=cursor, cursor_is_prepared=cursor is not None)
    
    output_file = output_path + "/{}.zip".format(frame_id)

    if not os.path.isfile(output_file):
        print("Generating image sequence data for {} interaction events...".format(data.shape[0]))
        bee_ids = data[["bee_id0", "bee_id1"]].values
        data["bee_id0"], data["bee_id1"] = bee_ids.min(axis=1), bee_ids.max(axis=1)
        data = data[data.bee_id0 != data.bee_id1].reset_index(drop=True)
        from ..db.metadata import get_frame_metadata
        metadata = get_frame_metadata([frame_id], cursor=cursor, cursor_is_prepared=cursor is not None)
        data["cam_id"] = metadata.cam_id.values[0]
        data["timestamp"] = metadata.timestamp.values[0]
        data["fc_id"] = metadata.fc_id.values[0]
        gt_bees = gt[gt.frame_id == frame_id][["bee_id0", "bee_id1"]].values
        gt_bees = set((min(bees), max(bees)) for bees in gt_bees)
        data["label"] = [(1 if bee_pair in gt_bees else 0) for bee_pair in data[["bee_id0", "bee_id1"]].itertuples(index=False)]
        print("True labels: {}".format(data.label.sum()))
        data["event_id"] = np.arange(data.shape[0], dtype=np.uint32)
        image_sequences.generate_image_sequence_data(data, output_file,
                                     video_root=video_root, video_cache_path=video_cache_path,
                                    n_sequences=None, append=False, verbose=False, dry=False)
    
    available_events = image_sequences.get_available_events(output_file, with_event_metadata=True, with_frame_metadata=True, progress=None)
    from IPython.display import display
    display(available_events.shape, data.shape)
    assert available_events.shape[0] <= data.shape[0]

    if available_events is None:
        print("No image data available for frame {}.".format(frame_id))
        return None

    
    dataset = image_sequences.TrophallaxisImageDataset(data=available_events, use_augmentations=False,
                             path=output_file)#, extracted_path="/home/david/Documents/troph_gt/gt_data")
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=2, drop_last=False)
    """for i in range(len(dataset)):
        im = dataset[i]
    return available_events"""
    model = torch.load("/mnt/storage/david/cache/beesbook/trophallaxis/image_cnn_model_.torch")
    predictions = model.predict_proba_from_dataloader(dataset_loader)[:, 1]
    print("For dataset of shape {}, predictions are of shape {} (Original events: {}).".format(len(dataset), predictions.shape, data.shape))
    assert predictions.shape[0] == len(dataset)
    assert predictions.shape[0] == available_events.shape[0]

    df = dict(bee_id0=[], bee_id1=[])
    for idx, y_hat in enumerate(predictions):
        if y_hat >= threshold:
            df["bee_id0"].append(available_events.bee_id0.iloc[idx])
            df["bee_id1"].append(available_events.bee_id1.iloc[idx])
    return pd.DataFrame(df)
    
def evaluate(root_path="/mnt/storage/david/data/beesbook/trophallaxis/testset/",
            video_root="/mnt/cray/recordings_2016/", video_cache_path="/mnt/ramdisk/",
            image_output_path="/home/david/Documents/troph_gt/eval_data/"):
    frame_ids, ground_truth = load_test_set()

    evaluate_interaction_model(apply_prefilter_to_frame,
               frame_ids, ground_truth, label="Trophallaxis Prefilter")

    if video_root is not None:
        evaluate_interaction_model(apply_prefilter_and_imagefilter_to_frame,
                frame_ids, ground_truth, label="Trophallaxis Image Filter",
                video_root=video_root, video_cache_path=video_cache_path, output_path=image_output_path,
                gt=ground_truth)
        return

    
    evaluate_interaction_model(apply_prefilter_to_frame_sequence,
               frame_ids, ground_truth, label="Trophallaxis Prefilter (Multi-Frame)")
    evaluate_interaction_model(apply_prefilter_and_trajfilter_to_frame,
               frame_ids, ground_truth, label="Trajectory Filter")
    evaluate_interaction_model(apply_prefilter_and_trajfilter_to_frame_sequence,
               frame_ids, ground_truth, label="Trajectory Filter (Multi-Frame)")
    