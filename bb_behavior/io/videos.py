import copy
import os
import shutil
import skimage.io
import tempfile
from concurrent.futures import ThreadPoolExecutor as PoolExecutor


from ..db.metadata import get_frame_metadata

def parse_beesbook_video_filename(filename, time_delimiter=None, single_timestamp=False):
    """
    Parses a filename of a video generated by the BeesBook system.
    Returns the camera ID and the timestamps. Ignores the file extension.
    
    Arguments:
        filename: string
            Video filename. E.g. Cam_0_2019-08-30T13_13_16.761604Z--2019-08-30T13_14_40.016808Z.avi
        time_delimiter: string, optional
            Specifies the separator between hour, minutes and seconds.
            If None, ":" and "_" are tried.
            Some filesystems might not support ":" in a filename.
        single_timestamp: bool, optional
            Whether to also accept filenames that contain only one timestamp without the '--' separation.
            In that case, datetime_end will be None.
            
    Returns:
        (cam_id, datetime_begin, datetime_end): tuple(int, datetime.datetime, datetime.datetime)
            The timezone information is set to pytz.UTC.
    """
    import datetime, pytz
    
    cam_prefix_len = len("Cam_X_")
    cam_string = filename[:cam_prefix_len]
    datetime_suffix = filename[cam_prefix_len:]
    datetime_suffix = datetime_suffix[:56] # The datetime format has a fixed length.
    if "--" in datetime_suffix:
        begin, end = datetime_suffix.split("--")
    else:
        begin = datetime_suffix
        end = None
    
    # Assume we are using UTC strings.
    if not begin or (not end and not single_timestamp) or begin[-1] != "Z" or (end and end[-1] != "Z"):
        raise ValueError("Invalid video name encountered: {}".format(filename))
    
    if time_delimiter is None:
        time_delimiter = ":"
        if not time_delimiter in begin:
            time_delimiter = "_"
            
    dt_format_string = "%Y-%m-%dT%H{}%M{}%S.%fZ".format(time_delimiter, time_delimiter)
    begin = pytz.UTC.localize(datetime.datetime.strptime(begin, dt_format_string))
    if end is not None:
        end = pytz.UTC.localize(datetime.datetime.strptime(end, dt_format_string))
    return int(cam_string.split("_")[1]), begin, end

def get_first_frame_from_video(vid_file):
    import cv2
    capture = cv2.VideoCapture(vid_file)
    success, image = capture.read()
    if not success:
        raise ValueError("Could not read first frame. Is the video file valid? ({})".format(vid_file))
    return image

def extract_frames_from_video(video_path, target_directory, start_frame=0, n_frames=1,codec="hevc_cuvid", command="ffmpeg", scale=1.0, framerate=3, output_format="bmp"):
    """Takes a video filename, frame start index and number of frames and uses FFMPEG to extract these frames into a target directory.
    """
    import subprocess

    if codec is not None:
        codec = ['-vcodec', codec]
    else:
        codec = []

    if scale != 1.0:
        scale = ",scale=iw*{scale}:ih*{scale}"
    else:
        scale = ""

    call_args = codec + [
        "-y", "-v", "24", "-r", str(framerate), "-i", video_path, "-start_number", "0",
        "-vf", "select=gte(n\\,{start_frame}){scale}".format(start_frame=start_frame, scale=scale), "-qscale:v", "2",
        "-vframes", str(n_frames), "{target_directory}/%04d.{output_format}".format(target_directory=target_directory, output_format=output_format)]

    p = subprocess.Popen([command] + call_args, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.wait() != 0:
        raise ValueError("FFMPEG Error: {}".format(stderr))
    


def get_frames_from_video(video_path, **kwargs):
    """Like extract_frames_from_video but loads the frames and yields them as numpy arrays.
    """
    import tempfile
    import matplotlib.pyplot as plt
    import numpy as np

    if "output_format" not in kwargs:
        kwargs["output_format"] = "png"
    n_frames = 1
    if "n_frames" in kwargs:
        n_frames = kwargs["n_frames"]

    
    with tempfile.TemporaryDirectory() as temp_dir:
        extract_frames_from_video(video_path, temp_dir, **kwargs)
        
        all_frames = sorted(os.listdir(temp_dir))
        if len(all_frames) != n_frames:
            start_frame = 0
            if "start_frame" in kwargs:
                start_frame = kwargs["start_frame"]
            raise ValueError("Invalid frame number (start: {}, N: {}) for video file {}.".format(
                                start_frame, n_frames, video_path))
        
        for f in all_frames:
            image = plt.imread(temp_dir + "/" + f).astype(np.float32)
            yield image

class BeesbookVideoManager():
    def __init__(self, video_root, cache_path, videos_in_subdirectories=True):
        if video_root[-1] != "/":
            video_root += "/"
        self.video_root = video_root
        if cache_path[-1] != "/":
            cache_path += "/"
        self.cache_path = cache_path
        self.videos_in_subdirectories = videos_in_subdirectories
        self.last_requests = []

        self.command = "ffmpeg"
        self.codec = "hevc"
        self.output_format = "png"
        self.scale = 1.0

        self.loader_thread_pool = PoolExecutor(max_workers=8)

    def clear_video_cache(self, retain_last_n_requests=None):
        """Removes all cached image files.
        To be called by user in regular intervals.

        Optionally, retains the last retain_last_n_requests requests' images in the cache.

        Arguments:
            retain_last_n_requests: bool
                Whether not to delete the images that were generated in the last
                retain_last_n_requests calls to get_frames.
        """
        # Allow some images to be retained on order to only clear the oldest images in the cache.
        retained_images = set()
        if retain_last_n_requests:
            retain_last_n_requests = min(retain_last_n_requests, len(self.last_requests))

            for i in range(retain_last_n_requests):
                retained_images |= {str(f_id) + "." + self.output_format for f_id in self.last_requests[-(i+1)]}
            del self.last_requests[0:(len(self.last_requests) - retain_last_n_requests)]

        # Check the directory for files and remove all old images.
        for filename in os.listdir(self.cache_path):
            # Safety - only remove our image files.
            if filename.endswith("." + self.output_format) and filename not in retained_images:
                os.remove(self.cache_path + filename)

    def get_raw_video_path(self, video_name):
        """
        Arguments:
            video_name: string
                Name of the video file located in self.video_root.
                If videos_in_subdirectories is set, the file needs to follow the BeesBook video naming
                convention and be located in video_root/{date}/{cam}.
        Returns:
            video_path: string
                Full path to video.
        """
        path = self.video_root

        if self.videos_in_subdirectories:
            cam = video_name[:5]
            date = video_name[6:(6 + 10)]
            path += "{date}/{cam}/".format(date=date, cam=cam)

        return path + video_name

    def extract_frames(self, video_name, start_frame, n_frames, frame_ids):
        """Extracts and caches frames from a video using ffmpeg.

        Arguments:
            video_name: string
                Name of the video (also see get_raw_video_path).
            start_frame: int
                Index of the first frame to extract.
            n_frames: int
                Number of frames to extract.
            frame_ids: list(int)
                List of length n_frames with the frame ids belonging to the video subsequence.
                These will determine the final filenames.

        """
        assert (n_frames == len(frame_ids))
        with tempfile.TemporaryDirectory(dir=self.cache_path, prefix="vidtmp_") as dirpath:
            try:
                extract_frames_from_video(self.get_raw_video_path(video_name), target_directory=dirpath,
                    start_frame=start_frame, n_frames=n_frames,
                    command=self.command, codec=self.codec, output_format=self.output_format,
                    scale=self.scale)
            except Exception as e:
                print("Error at video {}".format(self.get_raw_video_path(video_name),))
                raise e
            
            for (frame_id, filepath) in zip(frame_ids, sorted(os.listdir(dirpath))):
                full_filepath = dirpath+"/"+filepath
                if os.path.getsize(full_filepath) > 0:
                    shutil.move(full_filepath, self.cache_path + str(frame_id) + "." + self.output_format)
                else:
                    print("Zero-size file created by ffmpeg.")

    def get_frame_id_path(self, frame_id):
        """
        Arguments:
            frame_id: int
        Returns:
            frame_path: string
                Path to the image belonging to frame_id.
        """
        return self.cache_path + str(frame_id) + "." + self.output_format
    
    def is_frame_cached(self, frame_id):
        """Checks whether a specific frame is available on disk.
        
        Arguments:
            frame_id: int
        Returns:
            is_cached: bool
                Whether the frame is ready to be read.
        """
        in_cache = os.path.isfile(self.get_frame_id_path(frame_id))
        #print("Frame {} is in cache {} [at path {}]".format(frame_id, in_cache, self.get_frame_id_path(frame_id)))
        return in_cache

    def get_frame(self, frame_id):
        """Returns a loaded image belonging to a cached frame_id.

        Arguments:
            frame_id: int
        Returns:
            frame: np.array
        """
        if frame_id is None:
            return None
        return skimage.io.imread(self.get_frame_id_path(frame_id), as_gray=True, plugin="matplotlib")

    def extract_frames_from_metadata(self, frame_ids, frame_indices, video_names, verbose=False):
        """Takes lists of frame IDs and metadata (coming from db.get_frame_metadata)
        and extracts the video files into the cache.

        Arguments:
            frame_ids: list(int)
            frame_indices: list(int)
                Index of the frame in its video file.
            video_name: list(string)
                Name of video file of each frame.
            verbose: bool
                Whether to print additional information.
        """
        unique_videos = set(video_names)

        for video_name in unique_videos:
            video_data = [(frame_indices[i], frame_ids[i]) for i in range(len(video_names)) if video_names[i] == video_name]
            video_data = sorted(video_data)
            # The frame_ids must not necessarily be in sequence.
            # If we need the first two frames and the last two, don't extract the full video.
            cuts = [i for i in range(1, len(video_data) + 1) if (i == len(video_data)) or (video_data[i-1][0] + 1 != video_data[i][0])]
            last_cut = 0
            for cut in cuts:
                video_indices, video_frame_ids = zip(*video_data[last_cut:cut])
                start_idx, end_idx = video_indices[0], video_indices[-1]
                assert end_idx >= start_idx
                if verbose:
                    print("Extracting from {}: frame {}-{}".format(self.get_raw_video_path(video_name), start_idx, end_idx))
                self.extract_frames(video_name, start_idx, end_idx - start_idx + 1, video_frame_ids)
                last_cut = cut

    def cache_frames(self, frame_ids, cursor=None, verbose=False):
        """Makes sure that all frames in frame_ids are cached.
        Unlike get_frames it does not load and return the images.

        Arguments:
            frame_ids: list(int)
            verbose: bool
                Whether to print additional output.
        """
        all_n = len(frame_ids)
        frame_ids = [f_id for f_id in frame_ids if not self.is_frame_cached(f_id)]
        if frame_ids:
            if verbose:
                print("Extracting {} frames ({} were requested).".format(len(frame_ids), all_n))
            metadata = get_frame_metadata(frame_ids, return_dataframe=False, include_video_name=True,
                                        cursor=cursor, cursor_is_prepared=cursor is not None)
            frame_indices = [meta[2] for meta in metadata]
            video_names = [meta[5] for meta in metadata]
            assert frame_ids == [meta[0] for meta in metadata]
            self.extract_frames_from_metadata(frame_ids, frame_indices, video_names, verbose=verbose)

    def get_frames(self, frame_ids, cursor=None, verbose=False):
        """Returns a list of numpy arrays that correspond to whole frame images for the given frame IDs.

        Arguments:
            frame_ids: list(int)
            verbose: bool
                Whether to print additional output.
        Returns:
            images: list(np.array)
        """
        all_frame_ids = copy.copy(frame_ids)
        self.cache_frames(all_frame_ids, verbose=verbose)
        self.last_requests.append(all_frame_ids)

        for frame_id in frame_ids:
            assert self.is_frame_cached(frame_id)

        images = list(self.loader_thread_pool.map(self.get_frame, all_frame_ids))
        return images
