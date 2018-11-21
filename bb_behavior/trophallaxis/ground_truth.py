"""Helper functions to generate ground truth data for trophallaxis.

Usage:
    Execute the following in a jupyter notebook.
    ```
    interactions = bb_behavior.trophallaxis.sampling.get_trophallaxis_samples(100, 260.0/15.65, 100/15.65, 0.5)
    GUI = bb_behavior.trophallaxis.ground_truth.GUI("Your name")
    GUI(interactions)
    ```
"""
import numpy as np
import pandas as pd
import skimage.exposure
import skimage.draw
import datetime
import pytz

import bb_utils.meta
import bb_utils.ids
from bb_backend.api import FramePlotter

from ..db import get_neighbour_frames, get_bee_detections
from ..utils import misc

ground_truth_save_path = "/mnt/storage/david/data/beesbook/trophallaxis/ground_truth.csv"
# When the process of annotating ground-truth data changes, the data version should be increased.
ground_truth_data_version = 4

def load_ground_truth_data():
    """Loads the available ground truth data as a pandas.DataFrame.

    The data has a column 'version'.
    This column determines different steps in the process of annotating ground truth data.

        1: Beginning. Annotated with half resolution images.
        2: Displayed image changed to full resolution and raw quality.
        3: Antennation added as a label. Negative class changed to 'nothing' (from 'not trophallaxis').
        4: Slider added to annotate more than one frame at a time.
    Returns:
        pandas.DataFrame
    """
    global ground_truth_save_path
    try:
        data = pd.read_csv(ground_truth_save_path, header=None,
            names=["frame_id", "bee_id0", "bee_id1", "author", "annotation_timestamp", "label", "version", "event_id", "event_frame_idx"],
            dtype={"frame_id": np.uint64, "event_id": np.uint64})
    except:
        return None
    return data

def get_ground_truth_events(data=None):
    """Loads and returns frame_id, bee_id0, bee_id1 for all available ground truth events.
    Arguments:
        data: pandas.DataFrame
            Optional. DataFrame with the first three columns being frame_id, bee_id0, bee_id1.
    Returns:
        set(tuple(frame_id, bee_id0, bee_id1))
        All available events.
    """
    if data is None:
        data = load_ground_truth_data()
        if data is None:
            return set()
    events = set()
    for row in data.itertuples(index=False):
        key = (row[0], row[1], row[2])
        events.add(key)

    return events


def get_frames_for_interaction(frame_id, bee_id0, bee_id1, n_frames=31, width=200):
    """Fetches n_frames images (width x width) that are centered around two bee detections.

    Arguments:
        frame_id: int
            frame_id of the center frame.
        bee_id0: int
            Bee ID (ferwar style) of the first individual.
        bee_id1: int
            Bee ID (ferwar style) of the second individual.
        n_frames: int
            Number of total images returned.
        width: int
            Width (and height) of the square cutout around the bees.

    Returns:
        tuple(list(np.array), list(int))
            First list containing grey-scale images, second list containing the respective frame IDs.
    """
    target_len = n_frames
    
    neighbour_frames = get_neighbour_frames(frame_id, n_frames=(target_len + 4) // 2)
    if len(neighbour_frames) <= 3:
        return None

    margin_left = (len(neighbour_frames) - target_len) // 2
    margin_right = (len(neighbour_frames) - target_len) - margin_left
    if margin_left >= 0 and margin_right >= 0:
        neighbour_frames = neighbour_frames[margin_left:-margin_right]
    assert len(neighbour_frames) <= target_len
    assert len(neighbour_frames) > 3

    # Find index of originally requested frame.
    mid = None
    original_frame_id = frame_id
    for idx, (ts, frame_id, _) in enumerate(neighbour_frames):
        if frame_id == original_frame_id:
            mid = idx
            break
    if mid is None:
        return None
    det = get_bee_detections(bee_id0, frames=neighbour_frames)
    det2 = get_bee_detections(bee_id1, frames=neighbour_frames)
    if det is None:
        return None
    if det[mid] is None:
        return None
    x, y = det[mid][2], det[mid][3]
    s = width
    w, h = x + s, y + s
    x, y = x - s, y - s
    
    frames = []
    
    for idx, (ts, frame_id, _) in enumerate(neighbour_frames):
        is_focal_frame = frame_id == original_frame_id
        decode_n_frames = None
        if idx == 0:
            decode_n_frames = len(neighbour_frames)
        #im = np.zeros(shape=(800, 800), dtype=np.float32)
        xs, ys = [], []
        colors, sizes = [], None
        for (det_idx, bee) in enumerate((det, det2)):
            if bee[idx] is not None:
                xs.append(bee[idx][2])
                ys.append(bee[idx][3])
                if is_focal_frame:
                    colors.append("white")
                else:
                    colors.append(["gray", "silver"][det_idx])
        if not xs:
            xs, ys, colors = None, None, None
        else:
            sizes = [25] * len(xs)
        im = FramePlotter(frame_id=int(frame_id),
                          xs=xs, ys=ys, colors=colors,
                          sizes=sizes,
                          crop_coordinates=(x, y, w, h),
                          decode_n_frames=decode_n_frames,
                          raw=True, scale=1.0).get_image()
        im = skimage.exposure.equalize_adapthist(im)
        frames.append(im)
    
    return frames, [f[1] for f in neighbour_frames]

def generate_frames_for_interactions(interactions):
    """Takes a list of frame_ids and bee_ids and wraps get_frames_for_interaction as a generator.

    Arguments:
        interactions: list(tuple(frame_id, bee_id0, bee_id1))
            List of possible interactions.
    Returns:
        i: int
        frame_info: tuple(list(np.array), list(int))
        frame_id: int
        bee_id0: int
        bee_id1: int
        bee_name0: string
        bee_name1: string
    """
    meta = bb_utils.meta.BeeMetaInfo()
    for i in range(len(interactions)):
        frame_id, bee_id0, bee_id1 = interactions[i][:3]
        frame_info = get_frames_for_interaction(int(frame_id), bee_id0, bee_id1)
        
        if frame_info is None:
            continue
        bee_name0 = meta.get_beename(bb_utils.ids.BeesbookID.from_ferwar(bee_id0))
        bee_name1 = meta.get_beename(bb_utils.ids.BeesbookID.from_ferwar(bee_id1))
        yield i, frame_info, frame_id, bee_id0, bee_id1, bee_name0, bee_name1


class GUI():
    def __init__(self, name="Unknown"):
        import ipywidgets

        self.image_widget = ipywidgets.Output(layout={'border': '1px solid black'})
        self.image_widget2 = ipywidgets.Output()

        self.frame_bounds_widgets = [ipywidgets.Output(), ipywidgets.Output()]

        self.name_widget = ipywidgets.Text(name)
        self.ok = ipywidgets.Button(description="\tTrophallaxis", icon="check-circle")
        self.ok.on_click(lambda x: self.on_click(action="trophallaxis"))
        self.antennation = ipywidgets.Button(description="\tAntennation", icon="check-circle")
        self.antennation.on_click(lambda x: self.on_click(action="antennation"))
        self.nope = ipywidgets.Button(description="\tNothing", icon="ban")
        self.nope.on_click(lambda x: self.on_click(action="nothing"))
        self.idk = ipywidgets.Button(description="\tMaybe troph.", icon="puzzle-piece")
        self.idk.on_click(lambda x: self.on_click(action="unsure"))
        self.skip = ipywidgets.Button(description="\tSkip", icon="recycle")
        self.skip.on_click(lambda x: self.on_click(action="skip"))

        self.frame_idx_slider = ipywidgets.IntRangeSlider(value=[0, 0], step=1, description="Frames:", min=0,
            readout=True, readout_format="d", continuous_update=False)
        self.frame_idx_slider.observe(self.on_slider_changed, names='value')

        from IPython.display import display
        display(self.name_widget)
        display(self.image_widget)
        display(ipywidgets.VBox([
                    ipywidgets.HBox([self.frame_idx_slider]),
                    ipywidgets.HBox([self.ok, self.antennation, self.idk]),
                    ipywidgets.HBox([self.nope, self.skip])
                    ])
            )
        display(ipywidgets.HBox([self.image_widget2, ipywidgets.VBox(self.frame_bounds_widgets)]))

        self.current_interaction_idx = 0
        self.generator = None
    
    def on_slider_changed(self, change):
        if self.frames is None:
            return
        import matplotlib.pyplot as plt
        for w in self.frame_bounds_widgets:
            w.clear_output()
        
        def display(im):
            fig, ax = plt.subplots(figsize=(5, 5))
            plt.imshow(im, cmap="gray")
            plt.axis("off")
            plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
            fig.subplots_adjust(0,0,1,1)
            plt.show()
        with self.frame_bounds_widgets[0]:
            display(self.frames[change["new"][0]])
        with self.frame_bounds_widgets[1]:
            display(self.frames[change["new"][1]])

    def view_next_interaction(self, info_text=""):
        import matplotlib.pyplot as plt
        from ..plot.misc import plot_images_as_video

        self.frames = None
        for w in self.frame_bounds_widgets:
            w.clear_output()
        self.image_widget.clear_output()
        self.image_widget2.clear_output()
        with self.image_widget:
            if info_text:
                print(info_text)
            print("Please stand by...\nLoading next frame...")
        
        self.current_interaction_idx, (self.frames, self.frame_ids), frame_id, bee_id0, bee_id1, bee_name0, bee_name1 = \
                                        next(self.generator)

        self.frame_idx_slider.value = [0, 0]
        self.frame_idx_slider.max = len(self.frames) - 1

        self.image_widget.clear_output()
        with self.image_widget:
            fig, axes = plt.subplots(1, 5, figsize=(40, 10))
            print(f"frame: {frame_id}, {bee_id0} and {bee_id1}")
            plt.title(f"frame: {frame_id}, {bee_name0} and {bee_name1} ({bee_id0} and {bee_id1})")
            middle_frames = self.frames[int(len(self.frames) / 2 - len(axes) / 2):]
            for idx, ax in enumerate(axes):
                ax.imshow(middle_frames[idx], cmap="gray")
                ax.set_aspect("equal")
                ax.set_axis_off()
            plt.show()
        
        with self.image_widget2:
            plot_images_as_video(self.frames, display_index=True)


    def on_click(self, action, name=None):
    
        if name is None:
            name = self.name_widget.value
        
        info_text = "Skipped..."
        if action != "skip":
            now = datetime.datetime.now(tz=pytz.UTC).timestamp()

            begin_idx, end_idx = self.frame_idx_slider.value
            frame_id, bee_id0, bee_id1 = self.interactions[self.current_interaction_idx][:3]
            event_id = misc.generate_64bit_id()

            with open(ground_truth_save_path, 'a') as file:
                # Write only one frame if begin and end are not specified (assume middle frame).
                if end_idx <= begin_idx and action != "nothing":
                    file.write(f"{frame_id},{bee_id0},{bee_id1},{name},{now},{action},{ground_truth_data_version}, {event_id}, 0\n")
                else:
                    # Annotate whole event.
                    for i in range(len(self.frames)):
                        frame_id = self.frame_ids[i]
                        frame_action = action
                        if (end_idx > begin_idx) and (i < begin_idx or i > end_idx):
                            if action == "nothing":
                                # If the user specifically set the nothing-slider to a range, then the other
                                # frames are thought to not contain nothing and so we skip them.
                                continue
                            # Otherwise, if the user gave a label for a range, then the rest is considered 'nothing'.
                            frame_action = "nothing"
                        file.write(f"{frame_id},{bee_id0},{bee_id1},{name},{now},{frame_action},{ground_truth_data_version},{event_id},{i}\n")
            info_text = f"{action}! \t (sample was {frame_id}, {bee_id0} and {bee_id1})"
        self.view_next_interaction(info_text)

    def __call__(self, interactions, max_prefetch=20):
        import prefetch_generator
        self.interactions = interactions
        self.generator = prefetch_generator.BackgroundGenerator(generate_frames_for_interactions(interactions),
                                                                   max_prefetch=max_prefetch)
        self.view_next_interaction()

