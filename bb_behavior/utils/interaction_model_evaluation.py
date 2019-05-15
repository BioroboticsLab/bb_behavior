from collections import defaultdict
import numpy as np
import pandas as pd

from ..db import DatabaseCursorContext, find_interactions_in_frame

def order_bee_ids(b0, b1):
    return min(b0, b1), max(b0, b1)


def evaluate_interaction_model(model_fun, frame_ids, gt_df, label="Model Evaluation", **kwargs):
    """Takes an evaluation function and ground truth and assesses the performance of the function.

    Arguments:
        model_fun: callable
            Function of the form (frame_id, cursor) -> pandas.DataFrame[["bee_id0", "bee_id1"]].
        frame_ids: list(int)
            List of frame IDs for which to call model_fun.
        gt_df: pandas.DataFrame
            DataFrame containing at least the columns 'frame_id, 'bee_id0', 'bee_id1'.
            Evaluated against model_fun.
        **kwargs: dict
            Additional keyword arguments are passed to the model evaluation function.
    """
    results = defaultdict(int)
    
    with DatabaseCursorContext() as cursor:
        
        for frame_id in frame_ids:
            
            
            # Ground truth.
            df = gt_df[gt_df.frame_id == frame_id]
            
            pairs = list()
            for bee_id0, bee_id1 in df[["bee_id0", "bee_id1"]].itertuples(index=False):
                pairs.append(order_bee_ids(bee_id0, bee_id1))
            pairs = set(pairs)
            
            # Prediction.
            frame_results = model_fun(frame_id, cursor=cursor, **kwargs)
            try:
                frame_results = set(order_bee_ids(*p) for p in frame_results[["bee_id0", "bee_id1"]].itertuples(index=False))
            except:
                frame_results = set()
                print("Warning: Frame results empty. (Frame {})".format(frame_id))
            
            # Total interactions.
            interactions = find_interactions_in_frame(frame_id, max_distance=40.0,
                                                                     confidence_threshold=0.0,
                                                                     cursor=cursor, cursor_is_prepared=True)
            all_candidates = set(order_bee_ids(inter[1], inter[2]) for inter in interactions)                                                        
            assert (len(pairs & all_candidates) == len(pairs))
            negative_pairs = all_candidates - pairs
            
            P = len(pairs)
            N = len(negative_pairs)
            TP = len(frame_results & pairs)
            FP = len(frame_results) - TP
            TN = len(negative_pairs - pairs - frame_results)
            FN = len(pairs - frame_results)

            results["P"] += P
            results["N"] += N
            results["TP"] += TP
            results["FP"] += FP
            results["TN"] += TN
            results["FN"] += FN

    P, TP, FP = results["P"], results["TP"], results["FP"]
    N, TN, FN = results["N"], results["TN"], results["FN"]
    TPR = TP / P # Recall, hit rate
    TNR = TN / N # Specificity
    PPV = TP / (TP + FP) # Precision
    NPV = TN / (TN + FN) # Neg. pred. val
    F1  = np.nan
    try:
        F1 = 2 * (PPV * TPR) / (PPV + TPR)
    except:
        pass
    
    results = (f"{label}\n"
                "==============================\n"
               f"Recall     (TPR): {TPR:6.2%}\n"
               f"Precision  (PPV): {PPV:6.2%}\n"
               f"F1-Score   ( F1): {F1:6.2%}\n"
               f"Specificity(TNR): {TNR:6.2%}\n"
               f"Neg.Pred.V.(NPV): {NPV:6.2%}\n"
               "\n"
               f"Positive   (  P): {P:6d}\n"
               f"True Pos.  ( TP): {TP:6d}\n"
               f"False Pos. ( FP): {FP:6d}\n"
               "\n"
               f"Negative   (  N): {N:6d}\n"
               f"True Neg.  ( TN): {TN:6d}\n"
               f"False Neg. ( FN): {FN:6d}\n"
               f"Frames          : {len(frame_ids):6d}\n"
              )
    print(results)
            
            
            
        