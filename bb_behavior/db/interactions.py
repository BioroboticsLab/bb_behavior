from . import base
from .. import utils

def find_interactions_in_frame(frame_id, max_distance=20.0, min_distance=0.0, confidence_threshold=0.25, cursor=None,
                                distance_func=None, features=["x_pos_hive", "y_pos_hive"]):
    """Takes a frame id and finds all the possible interactions consisting of close bees.
    
    Arguments:
        frame_id: Database frame id of the frame to search.
        max_distance: Maximum hive distance (mm) of interaction partners.
        min_distance: Minimum hive distance (mm) of interaction partners.
        confidence_threshold: Minimum confidence of detections. Others are ignored.
        cursor: Optional. Database cursor connected to the DB.
        distance_func: callable
            Custom callable to use as a distance metric.
            Will be passed to scipy.spatial.distance.pdist.
        features: list
            Defines the fields that are queried from the DB and passed to the distance function.
    Returns:
        List containing interaction partners as tuples of
        (frame_id, bee_id0, bee_id1, detection_idx0, detection_idx1,
        x_pos_hive0, y_pos_hive0, x_pos_hive1, x_pos_hive1, cam_id).
    """
    import pandas

    if cursor is None:
        with base.get_database_connection(application_name="sample_frames") as db:
            cursor = db.cursor()
            return find_interactions_in_frame(frame_id=frame_id,
                                              max_distance=max_distance,
                                              min_distance=min_distance,
                                              confidence_threshold=confidence_threshold,
                                              distance_func=distance_func,
                                              cursor=cursor)
    
    fields = set(("x_pos_hive", "y_pos_hive", "orientation_hive")) | set(features)
    query = """
    SELECT {}, bee_id, detection_idx, cam_id FROM bb_detections_2016_stitched
        WHERE frame_id = %s
        AND bee_id_confidence >= %s
    """.format(",".join(fields))
    
    df = pandas.read_sql_query(query, cursor.connection, coerce_float=False,
                               params=(int(frame_id), confidence_threshold))
    if df.shape[0] == 0:
        return []
    
    close_pairs = utils.find_close_points(df[features].values,
                                   max_distance, min_distance, distance_func=distance_func)
    
    results = []
    for (i, j) in close_pairs:
        results.append((int(frame_id),
                        int(df.bee_id.iloc[i]), int(df.bee_id.iloc[j]),
                        int(df.detection_idx.iloc[i]), int(df.detection_idx.iloc[j]),
                        df.x_pos_hive.iloc[i], df.y_pos_hive.iloc[i], df.orientation_hive.iloc[i],
                        df.x_pos_hive.iloc[j], df.y_pos_hive.iloc[j], df.orientation_hive.iloc[j],
                        int(df.cam_id.iloc[j])
                       ))
    return results

