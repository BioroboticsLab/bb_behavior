import pandas as pd
import numpy as np

def plot_bees(bee_ids, colormap=None, frame_id=None, frame_margin = 26, frame_ids=None,
              path_alpha=0.3, cam_id=None, bt_export=None, plot_markers=True, plot_labels=True,
             n_frames_left=None, n_frames_right=None, backend_server_address=None):
    import bb_backend.api
    from bb_backend.api import FramePlotter, VideoPlotter
    if backend_server_address is not None:
        bb_backend.api.server_adress = backend_server_address
    
    from ..db import get_neighbour_frames, get_interpolated_trajectory
    timestamps = None
    
    if frame_ids is None:
        n_frames_left = n_frames_left or frame_margin
        n_frames_right = n_frames_right or frame_margin
        frames = get_neighbour_frames(int(frame_id),
                                                       n_frames_left=n_frames_left,
                                                       n_frames_right=n_frames_right,)
        timestamps, frame_ids, cam_ids = zip(*frames)
        cam_id = [c for c in cam_ids if c is not None][0]
    else:
        frames = [(None, fid, None) for fid in frame_ids]
        
    bee_map = {}
    for bee_id in bee_ids:
        traj, _ = get_interpolated_trajectory(int(bee_id), frames=frames)
        bee_map[bee_id] = traj[:, :2]
    
    # JSON export
    if bt_export is not None:
        from ..io.biotracker import save_tracks
        bee_map_video_cords = dict()
        for bee_id, xy in bee_map.items():
            xy = xy.astype(int)
            
            xs, ys = bb_backend.api.get_plot_coordinates(xy[:,0], xy[:,1])
            origin = bb_backend.api.get_image_origin(cam_id, year=2016)
            
            if origin[0] == 1:
                xs = 3000 - xs
            if origin[1] == 1:
                ys = 4000 - ys
            bee_map_video_cords[bee_id] = np.array([xs, ys]).T
        save_tracks(bee_map_video_cords, bt_export, timestamps=timestamps, frame_ids=frame_ids, meta=dict(cam_id=cam_id))
        
    
    # Video export
    frame_plotters = []
    for fidx, frame_id in enumerate(frame_ids):
        labels, xs, ys, sizes = [], [], [], []
        colors = []
        for bee_id in bee_ids:
            x, y = bee_map[bee_id][fidx, 0], bee_map[bee_id][fidx, 1]
            if pd.isnull(x) or pd.isnull(y):
                continue
            plot_label = (isinstance(plot_labels, set) and bee_id in plot_labels) or (plot_labels == True)
            plot_marker = (isinstance(plot_markers, set) and bee_id in plot_markers) or (plot_markers == True)
            
            
            if plot_marker or plot_label:
                xs.append(int(x))
                ys.append(int(y))
                
                color = "y"
                if (colormap is not None):
                    if bee_id in colormap:
                        color = colormap[bee_id]
                colors.append(color)

                if plot_label:
                    labels.append(str(bee_id))
                else:
                    labels.append(None)
                    
                if plot_marker:
                    sizes.append(20)
                else:
                    sizes.append(None)
        assert (len(xs) == len(ys))
        assert (len(xs) == len(labels))
        fp = FramePlotter(frame_id=int(frame_id),
                          labels=labels, xs=xs, ys=ys, sizes=sizes, colors=colors)
        frame_plotters.append(fp)
    track_labels = path_alpha is not None and path_alpha > 0.0
    crop_margin = 100 if not bt_export else None
    video = VideoPlotter(frames=frame_plotters, crop_margin=crop_margin, title="auto", track_labels=track_labels, scale=1.0, path_alpha=path_alpha)
    
    return video