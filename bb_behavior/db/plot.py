import datetime
import numpy as np
import pandas as pd
import psycopg2

def plot_timeline_from_database(
              host="localhost", port=5432, user="reader", password="reader", database="beesbook",
               table_name="bb_frame_metadata_", title="BeesBook", progress="tqdm_notebook", plot_kws=dict(),
               use_detections="auto"
              ):
    """Takes data from either the metadata table or the detections table and plots a timeline with the state of the cameras.
    This can be used as sanity checking for whether the cameras have been recording all the time.

    Arguments:
        host, port, user, password, database:
            credentials for the database
        table_name: string
            Name of the table to fetch the metadata from.
            If the name does not contain bb_detections_, it is assumed that it comes from the frame metadata table.
        title: string
            Title of the plot.
        progress: "tqdm", "tqdm_notebook", None or callable
            Callable to display the query progress.
        plot_kws:
            Additional arguments to be passed to bb_behavior.plot.time.plot_timeline.
        use_detections: True, False or "auto"
            Whether to use a table in the bb_detections format or in the metadata format.
            When True, the table just has to contain timestamp and cam_id.
    """
    from ..plot.time import plot_timeline

    from collections import defaultdict
    if use_detections == "auto":
        use_detections = "bb_detections_" in table_name
    
    if progress == "tqdm":
        import tqdm
        progress = tqdm.tqdm
    elif progress == "tqdm_notebook":
        import tqdm
        progress = tqdm.tqdm_notebook
    elif progress is None:
        progress = lambda x, **kwargs: x

    with psycopg2.connect(host=host, port=port, user=user, password=password, database=database,
                          application_name="timeline") as con:
        
        cursor = con.cursor(name="iterator")
        if not use_detections:
            
            cursor.execute("""SELECT datetime, frame_number, cam_id, fc_id, index, frame_id
                    FROM {}
                    ORDER BY timestamp
                    ;""".format(table_name))
        else:
            cursor.execute("""SELECT DISTINCT timestamp, cam_id FROM {};""".format(table_name))
            results = {f for f in progress(cursor)}
            cursor = ((dt, 0, cam_id, 0, 0, 0) for (dt, cam_id) in progress(sorted(results)))
            
        def iterate():
            cam_id_idx = defaultdict(int)
            for result in progress(cursor):
                dt, cam_idx, cam_id, fc_id, fc_idx, frame_id = result
                if use_detections:
                    cam_idx = cam_id_idx[cam_id]
                    cam_id_idx[cam_id] += 1
                yield dict(time=dt, y="Cam {}".format(cam_id), color="Recording",
                          frame_id=frame_id, index=cam_idx)
        
        colormap = dict(Recording = 'rgb(50, 150, 30)',
                          Gap = 'rgb(250, 100, 5)')
        
        def make_description(meta_begin, meta_end):
            if meta_end is None:
                return ""
            time_start, time_end = meta_begin["time"], meta_end["time"]
            duration = time_end - time_start
            index_start, index_end = meta_begin["index"], meta_end["index"]
            n_frames = index_end - index_start
            fps = np.nan
            if duration.total_seconds() > 0:
                fps = n_frames / duration.total_seconds()
            return "#Frames: {}\n<br>FPS: {:5.3f}".format(n_frames, fps)
        
        plot_timeline(iterate(),
                      time="time", y="y", color="color",
                      colormap=colormap, meta_keys=("frame_id", "index", "time"),
                      description_fun=make_description,
                      **plot_kws)

def fetch_sampled_age_values_from_database(dt_from, dt_to, n_frames=10, verbose=False, check_is_alive=True):
    from ..db.trajectory import DatabaseCursorContext, get_bee_detections
    from ..db.sampling import sample_frame_ids, get_bee_ids
    from ..db.metadata import get_frame_metadata, get_alive_bees
    from bb_utils.meta import BeeMetaInfo
    from bb_utils.ids import BeesbookID
    bb_meta_info = BeeMetaInfo()
    
    with DatabaseCursorContext("heatmap") as cursor:
        frames = sample_frame_ids(n_samples=n_frames, ts_from=dt_from.timestamp(), ts_to=dt_to.timestamp(),
                                cursor=cursor)
        frames, _ = zip(*frames)
        meta = get_frame_metadata(frames, cursor=cursor, cursor_is_prepared=True, return_dataframe=False)
        if verbose:
            print("Sampled {} frames, fetched metadata for {}.".format(len(frames), len(meta)))
        frames, _, _, _, cam_ids = zip(*meta)
        frame_id_to_cam_id = {f:c for f, c in zip(frames, cam_ids)}
        bee_ids = get_bee_ids(frames)
        if verbose:
            print("Found {} bee IDs in the data for those frames.".format(len(bee_ids)))

        bee_alive_cache = dict()
        def bee_is_alive(bee_id, dt):
            if not check_is_alive:
                return True
            if not dt in bee_alive_cache:
                bee_alive_cache[dt] = get_alive_bees(dt, dt + datetime.timedelta(days=1), cursor=cursor)
            return bee_id in bee_alive_cache[dt]

        for bee_id in bee_ids:
            detections = get_bee_detections(bee_id, frames=list(map(int, frames)), use_hive_coords=True,
                                            cursor=cursor, cursor_is_prepared=True, make_consistent=False)
            bb_id = BeesbookID.from_ferwar(bee_id)
            
            for detection in detections:
                if detection is None:
                    continue
                dt = detection[0]
                age = bb_meta_info.get_age(bb_id, dt.replace(tzinfo=None))
                if pd.isnull(age) or age.days < 0:
                    continue
                if not bee_is_alive(bee_id, dt):
                    continue
                age = age.days
                cam_id = frame_id_to_cam_id[detection[1]]
                hive_side = ["Cam 0/1", "Cam 2/3"][cam_id // 2]
                x, y = detection[2], detection[3]
                if cam_id == 1 and x <= 185:
                    continue
                elif cam_id == 3 and x <= 182:
                    continue
                yield dict(x=x, y=y, value=age, category=hive_side, bee_id=bee_id, datetime=dt)

def plot_age_distribution_from_database(dt_from, dt_to, n_frames=10, verbose=False, sample_positions=None, plot_kwargs=dict()):
    from ..plot.spatial import plot_spatial_values
    sample_positions = sample_positions or list(fetch_sampled_age_values_from_database(dt_from, dt_to, n_frames=n_frames, verbose=verbose))
    if verbose:
        print("Total samples: {}".format(len(sample_positions)))
    plot_kwargs = {**dict(cmap="viridis", figsize=(20, 20), metric="mean", alpha="count"), **plot_kwargs}
    plot_spatial_values(sample_positions,
                          bin_width=5, interpolation="none", verbose=False,
                         x_lim=(0, 350), y_lim=(0, 240), **plot_kwargs)