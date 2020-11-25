"""This module wraps the BeesBook detection pipeline.
It is not optimized for high-performance throughput but for accessibility.

If you want the detection pipeline to use GPU acceleration, you have to configure your theano accordingly.
E.g. in a jupyter notebook if everything is installed, use
```
%env KERAS_BACKEND=theano
%env THEANO_FLAGS=floatX=float32,device=cuda0
```
"""
from ..plot.misc import draw_ferwar_id_on_axis

import numpy as np
import pandas as pd
from collections import defaultdict

import dill

import pipeline.io
from pipeline.objects import PipelineResult

import tqdm.auto

def get_timestamps_for_beesbook_video(path):
    """Takes the file path to a beesbook video (e.g. "/home/david/video.mp4").
    The function loads the corresponding timestamp-file (e.g. "/home/david/video.txt") and parses the
    timestamps for all frames.
    
    Arguments:
        path: file path to video (incl. .mp4 or .avi extension).
            A file with the same name but a .txt extension has to exist in the same directory.
    Returns:
        list containing UNIX timestamps with milliseconds. Length corresponds to the number of frames in the video.
    """
    import time, datetime
    def to_timestamp(datetime_string):
        # e.g. Cam_0_2018-09-13T17:13:49.501824Z
        dt = datetime.datetime.strptime(datetime_string[6:-1], "%Y-%m-%dT%H:%M:%S.%f")
        return time.mktime(dt.utctimetuple()) + dt.microsecond * 0.000001
    ts_path = path[:-3] + "txt"
    with open(ts_path, "r") as f:
        return [to_timestamp(l) for l in f.read().splitlines()]
    
def detect_markers_in_beesbook_video(video_path, *args, **kwargs):
    """Wraps detect_markers_in_video but loads timestamps from a .txt file next to the video file.
    See get_timestamps_for_beesbook_video.
    
    Arguments:
        video_path: Path to beesbook video. A .txt-extension file with the same name has to exist in the same directory.
    Returns:
        frame_info, video_dataframe: see detect_markers_in_video
    """
    timestamps = get_timestamps_for_beesbook_video(video_path)
    return detect_markers_in_video(video_path, timestamps=timestamps, *args, **kwargs)

def get_default_pipeline(localizer_threshold=None, verbose=False):
    """Creates and returns a bb_pipeline Pipeline object that is configured to
    take an image and return all info required for bb_binary (PipelineResult).
    
    Arguments:
        localizer_threshold: float
            Threshold for the localizer in the pipeline.
        verbose: bool
            Whether to also provide the CrownOverlay output for display purposes (slower).
        
    Returns:
        bb_pipeline.pipeline.Pipeline object, ready to be used.
    """
    import pipeline
    import pipeline.pipeline
    import pipeline.objects
    
    outputs = [pipeline.objects.PipelineResult]
    if verbose:
        outputs += [pipeline.objects.CrownOverlay]
    conf = pipeline.pipeline.get_auto_config()
    if localizer_threshold is not None:
        conf['Localizer']['threshold_tag'] = localizer_threshold
    decoder_pipeline = pipeline.Pipeline([pipeline.objects.Image],  # inputs
                        outputs,  # outputs
                        **conf)
    return decoder_pipeline

def detect_markers_in_video(source_path, source_type="auto", decoder_pipeline=None, pipeline_factory=None,
                            tag_pixel_diameter=30.0, timestamps=None,
                            start_timestamp=None, fps=3.0, cam_id=0,
                            verbose=False, n_frames=None, progress=tqdm.auto.tqdm,
                            calculate_confidences=True, confidence_filter=None,
                           use_parallel_jobs=False, clahe=False):
    """Takes a video or a sequence of images, applies the BeesBook tag detection pipeline on the video and puts the results in a pandas DataFrame.
    Note that this is not optimized for high performance cluster computing but instead for end-user usability.

    Arguments:
        source_path: string
            Path to video file or list of paths to images.
            Can alternatively also be a list of greyscale (one channel) numpy images.
            E.g. as loaded by skimage.io.imread.
        source_type: ("auto", "video", "image")
            Type of media file behind 'source_path'.
        decoder_pipeline: pipeline.Pipeline
            Instantiated pipeline object used for localizing and decoding tags.
            Can be None.
        pipeline_factory: callable
            Used with decoder_pipeline=None. Function that returns a unique object suitable for the 'decoder_pipeline' argument.
            If given, the tag detection will be multithreaded with one pipeline object per threads.
        tag_pixel_diameter: float
            Diameter of the outer border of a flat BeesBook tag in the image (in pixels).
        timestamps: list(float)
            List of timestamps of a length that corresponds to the number of frames in the video.
            Can be None.
        start_timestamps: float
            Used with timestamps=None. Timestamp of first frame of the video. Defaults to 0.
        fps: float
            Used with timestamps=None. Frames per second in the video to auto-generate timestamps.
        cam_id: int
            Additional identifier that can be used to differentiate between recording instances.
            Will be used in the identifiers of the detections. Defaults to 0.
        verbose: bool
            Whether to display the detections in an image.
            Note that the pipeline has to be set to provide the CrownOverlay output.
        n_frames: int
            Whether to only return results for the first n_frames frames of the video.
        progress: callable or None.
            Whether to draw a progress bar.
        calculate_confidences: bool
            Whether to include a 'confidence' column in the results. Defaults to True.
        confidence_filter: float
            If set, specifies a threshold that disregards all detections from the results
            with a lower confidence value.
        use_parallel_jobs: bool
            Whether to use threading/multiprocessing.
        clahe: bool
            Whether to apply histogram equalization to the images.
    """
    if source_type == "auto":
        if isinstance(source_path, str):
            if source_path.endswith("jpg") or source_path.endswith("png"):
                source_type = "image"
            else:
                source_type = "video"
        elif isinstance(source_path, list):
            source_type = "image"
    calculate_confidences = calculate_confidences or confidence_filter is not None
    scale = 30.0 / tag_pixel_diameter
    if decoder_pipeline is None and pipeline_factory is None:
        decoder_pipeline = get_default_pipeline()

    if timestamps is None:
        def generate_timestamps(start_from):
            def gen():
                i = 0.0
                while True:
                    yield start_from + i * 1.0 / fps
                    i += 1.0
            yield from gen()
        timestamps = generate_timestamps(start_timestamp or 0.0)
        
    import skimage.transform
    import pipeline as bb_pipeline
        
    interrupted = False
    def interruptable_frame_generator(gen):
        nonlocal interrupted
        for g in gen:
            yield g
            if interrupted:
                print("Stopping early...")
                break
    
    def preprocess_image(im):
        if clahe:
            im = skimage.exposure.equalize_adapthist(im, kernel_size=3 * tag_pixel_diameter)
        im = (skimage.transform.rescale(im, scale, order=1, multichannel=False, anti_aliasing=False, mode='constant') * 255).astype(np.uint8)
        return im

    def get_frames_from_images():
        import skimage.io
        for idx, (path, ts) in enumerate(zip(interruptable_frame_generator(source_path), timestamps)):
            if type(path) is str:
                im = skimage.io.imread(path, as_gray=True)            
            else:
                im = path
            im = preprocess_image(im)
            yield idx, ts, im

            if n_frames is not None and idx >= n_frames - 1:
                break

    def get_frames_from_video():
        frames_generator = bb_pipeline.io.raw_frames_generator(source_path, format=None)
        
        for idx, (im, ts) in enumerate(zip(interruptable_frame_generator(frames_generator), timestamps)):
            assert im is not None
            # Skip broken videos.
            if im.shape[0] <= 0:
                print("Warning: Could not read frame {} of file {}. The video is corrupt.".format(idx, source_path))
                break
            # Skip broken frames because clahe would fail on constant input.
            if im.min() == im.max():
                print("Warning: Frame {} of file {} is empty.".format(idx, source_path))
                continue
            
            assert im.shape[0] > 0
            assert im.shape[1] > 0

            im = preprocess_image(im)

            yield idx, ts, im
            
            if n_frames is not None and idx >= n_frames - 1:
                break
    
    def get_detections_from_frame(idx, ts, im, thread_context=None, **kwargs):
        nonlocal decoder_pipeline
        if decoder_pipeline is not None:
            pipeline_results = decoder_pipeline([im])
        else:
            if "pipeline" not in thread_context:
                thread_context["pipeline"] = pipeline_factory()
            pipeline_results = thread_context["pipeline"]([im])
            
        #confident_ids = [r for c, r in zip(confidences, decoded_ids) if c >= threshold]
        #decimal_ids = set([ids.BeesbookID.from_bb_binary(i).as_ferwar() for i in confident_ids])

        if verbose:
            import pipeline.objects
            from pipeline.stages.visualization import ResultCrownVisualizer
            import matplotlib.pyplot as plt
            crowns = pipeline_results[pipeline.objects.CrownOverlay]
            frame = ResultCrownVisualizer.add_overlay(im.astype(np.float64) / 255, crowns)
            fig, ax = plt.subplots(figsize=(20, 10))
            plt.imshow(frame)
            plt.axis("off")
            plt.show()
        
        frame_id = bb_pipeline.io.unique_id()
        required_results = pipeline_results[PipelineResult]    
        n_detections = required_results.orientations.shape[0]
        decoded_ids = [list(r) for r in list(required_results.ids)]

        if n_detections > 0:
            frame_data = {
                "localizerSaliency": required_results.tag_saliencies.flatten(),
                "beeID": decoded_ids,
                "xpos": required_results.tag_positions[:, 1] / scale,
                "ypos": required_results.tag_positions[:, 0] / scale,
                "camID": [cam_id] * n_detections, 
                "zrotation": required_results.orientations[:, 0],
                "timestamp": [ts] * n_detections,
                "frameIdx": [idx] * n_detections,
                "frameId": frame_id,
                "detection_index": range(n_detections),
                "detection_type": "TaggedBee"
            }
            
            frame_data = pd.DataFrame(frame_data)

            if calculate_confidences:
                confidences = np.array([np.product(np.abs(0.5 - np.array(r)) * 2) for r in decoded_ids])
                frame_data["confidence"] = confidences
                if confidence_filter is not None:
                    frame_data = frame_data[frame_data.confidence >= confidence_filter]

        else:
            frame_data = None

        n_bees = required_results.bee_positions.shape[0]

        if n_bees > 0:
            bee_data = {
                "localizerSaliency": required_results.bee_saliencies.flatten(),
                "beeID": [np.nan] * n_bees,
                "xpos": required_results.bee_positions[:, 1] / scale,
                "ypos": required_results.bee_positions[:, 0] / scale,
                "camID": [cam_id] * n_bees, 
                "zrotation": [np.nan] * n_bees,
                "timestamp": [ts] * n_bees,
                "frameIdx": [idx] * n_bees,
                "frameId": frame_id,
                "detection_index": range(n_bees),
                "detection_type": required_results.bee_types
            }

            if calculate_confidences:
                bee_data["confidence"] = [np.nan] * n_bees

            bee_data = pd.DataFrame(bee_data)

            if frame_data is not None:
                frame_data = pd.concat((frame_data, bee_data))
            else:
                frame_data = bee_data

        return idx, frame_id, ts, frame_data
    
    
    progress_bar = None
    if progress is not None:
        progress_bar = progress(total=n_frames)
        
    def save_frame_data(idx, frame_id, ts, frame_data, **kwargs):
        nonlocal frame_info
        nonlocal video_dataframe
        frame_info.append((idx, frame_id, ts))
        if frame_data is not None:
            video_dataframe.append(frame_data)
        if progress is not None:
            progress_bar.update()
        
    source = get_frames_from_video
    if source_type == "image":
        source = get_frames_from_images
        if isinstance(source_path, str):
            source_path = [source_path]
            
    if use_parallel_jobs:
        from ..utils.processing import ParallelPipeline

        thread_context_factory = None
        n_pipeline_jobs = 1
        if not decoder_pipeline:
            # The thread context is used to provide each thread with a unique pipeline object.
            class Ctx():
                def __enter__(self):
                    return dict()
                def __exit__(self, *args):
                    pass
                
            n_pipeline_jobs = 4
            thread_context_factory = Ctx

        jobs = ParallelPipeline([source, get_detections_from_frame, save_frame_data],
                                                    n_thread_map={0: 1, 1: n_pipeline_jobs},
                                                   thread_context_factory=thread_context_factory)
    else:
        def sequential_jobs():
            for im in source():
                if im is not None:
                    detections = get_detections_from_frame(*im)
                    save_frame_data(*detections)
        jobs = sequential_jobs

    frame_info = []
    video_dataframe = []
    try:
        jobs()
    except KeyboardInterrupt:
        interrupted = True
    frame_info = list(sorted(frame_info))
    if len(video_dataframe) > 0:
        video_dataframe = pd.concat(video_dataframe)
        video_dataframe.sort_values("frameIdx", inplace=True)
        # Enfore frame ID datatype to be unsigned which may have gotten lost when concatenating the data frames.
        video_dataframe.frameId = video_dataframe.frameId.astype(np.uint64)
    else:
        video_dataframe = None
    
    return frame_info, video_dataframe
