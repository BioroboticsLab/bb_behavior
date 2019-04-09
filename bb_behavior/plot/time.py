import datetime
import itertools

def plot_timeline(iterable, min_gap_size=datetime.timedelta(seconds=1),
                 y="y", time="time", color="color", title="Timeline", filename="out.html",
                colormap=None, meta_keys=None, description_fun=None, fill_gaps=True):
    """Takes an iterable data source and merges the returned events into chunks that are max. min_gap_size apart.
    
    Arguments:
        iterable: iterable
            An expression yielding a mapping with keys matching the arguments y, time, color.
            The events have to be sorted by time.
            The y_value will be displayed on the Y-axis.
            The category determines the color/legend of the time slot.
        min_gap_size: datetime.timedelta
            If not None, determines the maximum time that two timestamps can be apart to be considerend one event.
        y: any
            Key to retrieve the Y value from an object yielded by iterable.
            The Y value can be considered a unique measurement unit (e.g. camera).
        time: any
            Key to retrieve the datetime.datetime from an object yielded by iterable.
        color: any
            Key to retrieve the category from an object yielded by iterable.
            The color can be considered the task/resource that the measurement unit uses.
            E.g. "Recording"/"Off" or "Sleeping"/"Walking".
        title: string
            The title of the plot.
        filename: string
            The plot will be exported to a file, if given.
            If set to None, the plot will be in Jupyter mode.
        colormap: mapping of /color/ to RGB-string.
            E.g. dict(Recording = 'rgb(50, 150, 30)', Gap = 'rgb(250, 100, 5)')
        meta_keys: iterable(any)
            Additional fields taken from the iterable and passed to the description_fun.
        description_fun: callable
            Callable of the form (dict:meta_keys, dict:meta_keys_end) -> string.
            Taking the meta information of the beginning and ending of a sequence and returning a description.
        fill_gaps: boolean
            Whether empty durations longer than min_gap_size are filled with a special "Gap" color.
    """
    import plotly.figure_factory as ff
    import plotly.offline
    if filename is None:
        plotly.offline.init_notebook_mode()
    
    from collections import defaultdict
    last_x_for_y = defaultdict(list)
    
    if meta_keys is not None and description_fun is None:
        description_fun = lambda a, b: ",<br>".join("{}: {}".format(m, a[m]) for m in meta_keys)
    
    for timepoint in iterable:
        dt, y_value, category = timepoint[time], timepoint[y], timepoint[color]
        meta_values = {m: timepoint[m] for m in meta_keys}
        last_x = last_x_for_y[y_value]

        def push():
            last_x_for_y[y_value].append(dict(
                Task=y_value,
                Resource=category,
                Start = dt.isoformat(),
                Finish = dt.isoformat(),
                meta_values = meta_values,
                meta_values_end = meta_values,
                dt_start = dt,
                dt_end = dt
            ))

        if not last_x or (category != last_x["Resource"]):
            push()
            continue
        last_x = last_x[-1]

        delay = dt - last_x["dt_end"]
        if min_gap_size is not None and delay > min_gap_size:
            if fill_gaps:
                last_x_for_y[y_value].append(dict(
                    Task=y_value,
                    Resource="Gap",
                    Start = last_x["Finish"],
                    Finish = dt.isoformat(),
                    dt_start = last_x["dt_end"],
                    dt_end = dt,
                    meta_values = None,
                    meta_values_end = None
                ))
            push()
            continue
            
        last_x["Finish"] = dt.isoformat()
        last_x["dt_end"] = dt
        last_x["meta_values_end"] = meta_values

    for _, sessions in last_x_for_y.items():
        for s in sessions:
            dt_start, dt_end = s["dt_start"], s["dt_end"]
            duration = (dt_end - dt_start)
            meta_values = s["meta_values"]
            meta_values_end = s["meta_values_end"]
            description = "Duration: {}".format(duration)
            if description_fun is not None and (meta_values or meta_values_end):
                description += "\n<br>" + description_fun(meta_values, meta_values_end)
                
    df = list(itertools.chain(*list(last_x_for_y.values())))
    fig = ff.create_gantt(df, colors=colormap, group_tasks=True, index_col="Resource", title=title)
    fig = plotly.offline.plot(fig, filename=filename, image_width=1024, image_height=600)
    if filename is None:
        from IPython.display import display
        display(fig)
    return last_x_for_y