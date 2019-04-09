import numpy as np
import psycopg2

def plot_timeline_from_database(
              host="localhost", port=5432, user="reader", password="reader", database="beesbook",
               table_name="bb_frame_metadata_", title="BeesBook", progress="tqdm_notebook", **kwargs
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
        **kwargs:
            Additional arguments are passed to bb_behavior.plot.time.plot_timeline.
    """
    from ..plot.time import plot_timeline

    from collections import defaultdict
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
                          application_name="timtqdm_notebookeline") as con:
        
        cursor = con.cursor(name="iterator")
        if not use_detections:
            
            cursor.execute("""SELECT datetime, frame_number, cam_id, fc_id, index, frame_id
                    FROM {}
                    ORDER BY timestamp
                    ;""".format(table_name))
        else:
            cursor.execute("""SELECT timestamp, frame_id, cam_id FROM {};""".format(table_name))
            results = {f for f in progress(cursor)}
            cursor = ((dt, 0, cam_id, 0, 0, frame_id) for (dt, frame_id, cam_id) in progress(sorted(results)))
            
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
                      **kwargs)