from . import base
import datetime, pytz
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

def get_frame_metadata(frames, cursor=None, cursor_is_prepared=False, return_dataframe=True, include_video_name=False,
    warnings_as_errors=False):
    """Takes a list of frame IDs and fetches additional data such as timestamps, cam_id, frame container ID, ...

    Arguments:
        frames: list(int)
            Database frame IDs for which to fetch the metadata.
        cursor: psycopg2.cursor
            Optional. Cursor connected to the database.
        cursor_is_prepared: bool
            Whether the cursor already has the SQL statements prepared.
        return_dataframe: bool
            Whether to return a pandas.DataFrame (instead of a list of tuples).
        include_video_name: bool
            Whether to include the original filename of the video in the results.
        warnings_as_errors: bool
            Whether to raise an exception instead of a warning.
    Returns:
        annotated_frames: pandas.DataFrame
            DataFrame with the columns frame_id, timestamp, frame_idx, fc_id, cam_id, [video_name].
            Note that 'frame_idx' is the index of the respective video (fc_id).

            If 'return_dataframe' is true, returns a list with one tuple per row of the data frame.
    """
    if cursor is None:
        from contextlib import closing
        with closing(base.get_database_connection("Frame metadata")) as con:
            return get_frame_metadata(frames, cursor=con.cursor(), cursor_is_prepared=False, return_dataframe=return_dataframe,
                include_video_name=include_video_name, warnings_as_errors=warnings_as_errors)
    if not cursor_is_prepared:
        cursor.execute("PREPARE get_frame_metadata AS "
           "SELECT frame_id, timestamp, index, fc_id FROM {} WHERE frame_id = ANY($1)".format(base.get_frame_metadata_tablename()))
        cursor.execute("PREPARE get_frame_container_info AS "
          "SELECT video_name FROM {} "
          "WHERE id = $1 LIMIT 1".format(base.get_framecontainer_metadata_tablename()))

    # Fetch the widely available metadata.
    frames = list(map(int, frames))
    cursor.execute("EXECUTE get_frame_metadata (%s)", (frames,))
    metadata = cursor.fetchall()
    frame_id_dict = dict()
    required_frame_containers = dict()
    for m in metadata:
        frame_id_dict[m[0]] = m
        required_frame_containers[m[-1]] = None
    
    # And add the frame_ids.
    for fc_id in required_frame_containers.keys():
        cursor.execute("EXECUTE get_frame_container_info (%s)", (fc_id,))
        video_name = cursor.fetchone()[0]
        cam_id = int(video_name[4])
        required_frame_containers[fc_id] = (cam_id, video_name)
    
    annotated_frames = []
    for idx, frame_id in enumerate(frames):
        if frame_id not in frame_id_dict:
            warning = "Frame ID {} not found in database.".format(frame_id)
            if warnings_as_errors:
                raise ValueError(warning)
            else:
                import warnings
                warnings.warn(warning)
                continue
        meta = list(frame_id_dict[frame_id])
        #meta[0] = int(meta[0])
        if include_video_name:
            meta.extend(required_frame_containers[meta[-1]])
        else:
            meta.append(required_frame_containers[meta[-1]][0])
        annotated_frames.append(meta)
    
    if return_dataframe:
        columns = ("frame_id", "timestamp", "frame_idx", "fc_id", "cam_id")
        if include_video_name:
            columns = columns + ("video_name",)
        annotated_frames = pd.DataFrame(annotated_frames, columns=columns)

    return annotated_frames

def get_bee_hatch_dates(bee_ids, cursor=None, return_maximum_date=False, respect_bounds=True):
    """Returns the introduction/hatch dates for a list of bees from the database.
    Note that the first date is not guaranteed to be the hatch date if some bees were marked before the recording began.
    Similarly, the last date is not guaranteed to be the death date if the recording stopped when some bees were still alive.
    To set all bees with hatch dates before the season begin (resp. death date after the season ended) to None, use respect_bounds.
    Arguments:
        bee_ids: list(int)
            List of ferwar-style IDs.
        cursor: psycopg2.cursor
            Optional. Connected database cursor.
        return_maximum_date: bool
            Optional. Whether to return the maximum date (the last day of life) instead.
        respect_bounds: bool
            Whether to return None for dates that would be the first/last day of the recording.
            This is true by default to prevent wrong assumptions about the hatching or death of a bee.
    Returns:
        hatch_dates: list(int, datetime.date)
            List of (bee_id, hatch_date). Same order as bee_ids
            hatch_date can be None if no data is available or respect_bounds is True and the date would be the start/end of the season.
    """
    if cursor is None:
        from contextlib import closing
        with closing(base.get_database_connection("get_bee_hatch_dates")) as con:
            return get_bee_hatch_dates(bee_ids, cursor=con.cursor(), return_maximum_date=return_maximum_date, respect_bounds=respect_bounds)
    
    # For the 2016 berlin data, the hatch dates are stored somewhere else.
    bee_id_to_date_override_map = dict()
    if not return_maximum_date and base.beesbook_season_config["identifier"] == "berlin_2016":
        import bb_utils.meta, bb_utils.ids
        hatch_metadata = bb_utils.meta.BeeMetaInfo()
        for bee_id in bee_ids:
            date = hatch_metadata.get_hatchdate(bb_utils.ids.BeesbookID.from_ferwar(bee_id))
            date = date.to_pydatetime().date()
            if pd.isnull(date):
                date = None
            bee_id_to_date_override_map[bee_id] = date

    bee_ids = list(map(int, bee_ids))
    fun = "MIN" if not return_maximum_date else "MAX"
    cursor.execute("SELECT {}(timestamp) FROM {};".format(fun, base.get_alive_bees_tablename()))
    bounding_date = cursor.fetchone()[0]

    cursor.execute("SELECT bee_id, {}(timestamp) FROM {} WHERE bee_id=ANY(%s) GROUP BY bee_id".format(
                    fun, base.get_alive_bees_tablename()),
                    (bee_ids,))
    lookup_dict = {b: d for (b, d) in cursor.fetchall()}
    hatch_dates = []
    for bee_id in bee_ids:
        date = lookup_dict[bee_id] if bee_id in lookup_dict else None
        if date is not None and bee_id in bee_id_to_date_override_map:
            date = bee_id_to_date_override_map[bee_id]
            assert date is not None
        else:
            if respect_bounds:
                if date == bounding_date:
                    date = None
        hatch_dates.append((bee_id, date))

    return hatch_dates

def get_bee_death_dates(bee_ids, cursor=None):
    """Compare get_bee_hatch_dates for further info.
    Returns the last day of life of the bees."""
    return get_bee_hatch_dates(bee_ids, cursor=cursor, return_maximum_date=True)

def get_bee_ages(bee_id_dates, clip=True, cursor=None):
    """Takes a list of bee_ids and dates and returns a list of bee_ids, dates and ages on that day.
    Arguments:
        bee_id_dates: list(int, datetime.date)
            List of ferwar-style bee_ids and dates.
        cursor: psycopg2.cursor
            Optional. Connected database cursor.
        clip: bool
            Whether to clip negative values to -1 and also change bees that have died already to -1.
            If clip is False, then bees without data will have the age set to None.
    Returns:
        bee_ages: list((int, datetime.date, int))
            List of (bee_id, date, age).
    """
    
    bee_ids = [bee_id for bee_id, _ in bee_id_dates]
    hatch_dates = {b: d for (b, d) in get_bee_hatch_dates(bee_ids, cursor=cursor)}
    last_days = None
    if clip:
        last_days = {b: d for (b, d) in get_bee_death_dates(bee_ids, cursor=cursor)}
    
    results = []
    for bee_id, date in bee_id_dates:
        hatch_date = hatch_dates[bee_id]
        if hatch_date is None:
            age = None
        else:
            age = (date - hatch_date).days

        if clip:
            if age is None or age < 0:
                age = -1
            elif last_days[bee_id] is not None and date > last_days[bee_id]:
                age = -1
        results.append((bee_id, date, age))
    return results

def get_alive_bees(dt_from, dt_to, cursor=None):
    """Returns all bees for a time window that have been tagged and did not die yet.

    Arguments:
        dt_from: datetime.datetime
            Begin of the period (inclusive).
        dt_to: datetime.datetime
            End of the period (exclusive).
        cursor: psycopg2.cursor
            Optional. Connected database cursor.
    Returns:
        bee_ids: set(int)
            All bee IDs (ferwar format) that are alive between the given timestamps.
    """
    if cursor is None:
        from contextlib import closing
        with closing(base.get_database_connection("get_alive_bees")) as con:
            return get_alive_bees(dt_from, dt_to, cursor=con.cursor())
    cursor.execute("SELECT bee_id from {} WHERE timestamp >= %s and timestamp < %s ".format(base.get_alive_bees_tablename()), (dt_from, dt_to))
    bee_ids = {result[0] for result in cursor.fetchall()}
    return bee_ids

def create_frame_metadata_table(repository_path, host, user, password, database="beesbook", tablename_suffix="", progress="tqdm"):
    """Reads a bb_binary.Repository and puts all the frame IDs, frame containers and their metadata (e.g. global index for frames)
    into two new database tables.

    Arguments:
        repository_path: string
            Path to a bb_binary.Repository.
        host, user, password, database: string
            Credentials for the database server.
        tablename_suffix: string
            Suffix for the table names (e.g. "2019_berlin").
        progress: string ("tqdm"/"tqdm_notebook") or callable
            Optional. Used to display the import progress.
    """
    from collections import defaultdict
    import bb_binary
    repo = bb_binary.Repository(repository_path)
    
    cam_id_indices = defaultdict(int)
    next_frame_container_id = 0
    
    if progress is not None:
        if progress == "tqdm":
            import tqdm
            progress = tqdm.tqdm
        elif progress == "tqdm_notebook":
            import tqdm
            progress = tqdm.tqdm_notebook
    else:
        progress = lambda x: x

    with psycopg2.connect(host=host, user=user, password=password, database=database) as con:
        cursor = con.cursor()
        
        framecontainer_tablename = "bb_framecontainer_metadata_" + tablename_suffix
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS {} (
            id integer NOT NULL,
            fc_id numeric(32,0) NOT NULL,
            fc_path text NOT NULL,
            video_name text NOT NULL
        );

        """.format(framecontainer_tablename))
        
        frame_tablename = "bb_frame_metadata_" + tablename_suffix
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS {} (
            frame_id numeric(32,0) NOT NULL,
            frame_number bigint NOT NULL,
            cam_id smallint NOT NULL,
            index integer NOT NULL,
            fc_id integer NOT NULL,
            "timestamp" double precision NOT NULL,
            "datetime" timestamp with time zone NOT NULL
        );

        """.format(frame_tablename))
        
        framecontainer_statement = """
                       INSERT INTO {} (id, fc_id, fc_path, video_name) VALUES %s
                        """.format(framecontainer_tablename)
        frame_statement = """
                       INSERT INTO {} (frame_id, frame_number, cam_id,
                           index, fc_id, "timestamp", "datetime") VALUES %s
                        """.format(frame_tablename)
        
        
        def commit_batch(batch, statement):
            if len(batch) == 0:
                return
            psycopg2.extras.execute_values(cursor, statement,
                                           batch, page_size=200)
            del batch[:]
            
        frame_batch = []    
        def commit_frame_batch():
            commit_batch(frame_batch, frame_statement)
            
        framecontainer_batch = []    
        def commit_framecontainer_batch():
            commit_batch(framecontainer_batch, framecontainer_statement)

        for fc_path in progress(repo.iter_fnames()):
            fc = bb_binary.load_frame_container(fc_path)
            
            fc_id = fc.id
            cam_id = fc.camId
            video_path = fc.dataSources[0].filename
            
            next_fc_frame_number = cam_id_indices[cam_id]
            
            for frame in fc.frames:
                frame_id = frame.id
                frame_timestamp = frame.timestamp
                frame_datetime = datetime.datetime.utcfromtimestamp(frame_timestamp)
                frame_datetime = pytz.utc.localize(frame_datetime)
                frame_index = frame.frameIdx

                frame_batch.append((
                    frame_id, next_fc_frame_number, cam_id,
                    frame_index, next_frame_container_id,
                    frame_timestamp, frame_datetime))
                
                next_fc_frame_number += 1
                
                if len(frame_batch) > 2000:
                    commit_frame_batch()
                    
            cam_id_indices[cam_id] = next_fc_frame_number
            commit_frame_batch()
            
            framecontainer_batch.append((next_frame_container_id, fc_id, fc_path, video_path))
            if len(framecontainer_batch) > 100:
                    commit_framecontainer_batch()
            
            next_frame_container_id += 1
        
        commit_framecontainer_batch()
    con.close()