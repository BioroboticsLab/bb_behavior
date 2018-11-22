import os
import shutil
import tempfile

from ..db.metadata import get_frame_metadata

def get_first_frame_from_video(vid_file):
    import cv2
    capture = cv2.VideoCapture(vid_file)
    success, image = capture.read()
    if not success:
        return None
    return image

def extract_frames_from_video(video_path, target_directory, start_frame=0, n_frames=1, codec="hevc_cuvid", command="ffmpeg", scale=1.0, framerate=3):

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
        "-vf", f"select=gte(n\\,{start_frame}){scale}", "-qscale:v", "2",
        "-vframes", str(n_frames), f"{target_directory}/%04d.bmp"]

    subprocess.run([command] + call_args, stderr=subprocess.PIPE)
    

class BeesbookVideoManager():
    def __init__(self, video_root, cache_path, videos_in_subdirectories=True):
        if video_root[-1] != "/":
            video_root += "/"
        self.video_root = video_root
        if cache_path[-1] != "/":
            cache_path += "/"
        self.cache_path = cache_path
        self.videos_in_subdirectories = videos_in_subdirectories


        self.command = "ffmpeg"
        self.codec = "hevc_cuvid"

    def get_raw_video_path(self, video_name):
        path = self.video_root

        if self.videos_in_subdirectories:
            cam = video_name[:5]
            date = video_name[6:(6 + 10)]
            path += f"{date}/{cam}/"

        return path + video_name

    def extract_frames(self, video_name, start_frame, n_frames, frame_ids):
        
        with tempfile.TemporaryDirectory(prefix="vidtmp_") as dirpath:
            extract_frames_from_video(self.get_raw_video_path(video_name), target_directory=dirpath,
                start_frame=start_frame, n_frames=n_frames,
                command=self.command, codec=self.codec)
            
            for (frame_id, filepath) in zip(frame_ids, os.listdir(dirpath)):
                shutil.move(dirpath+"/"+filepath, self.cache_path + str(frame_id) + ".bmp")

    def get_frame_id_path(self, frame_id):
        return self.cache_path + str(frame_id) + ".bmp"
    
    def is_frame_cached(self, frame_id):
        return os.path.isfile(self.get_frame_id_path(frame_id))

    def get_frame(self, frame_id):
        import skimage.io
        return skimage.io.imread(self.get_frame_id_path(frame_id), as_grey=True)

    def extract_frames_from_metadata(self, frame_ids, frame_indices, video_names):
        unique_videos = set(video_names)

        for video_name in unique_videos:
            video_data = [(frame_indices[i], frame_ids[i]) for i in range(len(video_names)) if video_names[i] == video_name]
            video_data = sorted(video_data)
            video_indices, video_frame_ids = zip(*video_data)
            start_idx, end_idx = video_indices[0], video_indices[-1]
            
            self.extract_frames(video_name, start_idx, end_idx - start_idx + 1, video_frame_ids)

    def get_frames_for_metadata(self, frame_ids, frame_indices, video_names):
        required_frames = [meta for meta in zip(frame_ids, frame_indices, video_names) if not self.is_frame_cached(meta[0])]
        if required_frames:
            self.extract_frames_from_metadata(*zip(*required_frames))
        
        return [self.get_frame(f_id) for f_id in frame_ids]

    def get_frames(self, frame_ids):

        metadata = get_frame_metadata(frame_ids, return_dataframe=False, include_video_name=True)
        frame_indices = [meta[2] for meta in metadata]
        video_names = [meta[5] for meta in metadata]

        return self.get_frames_for_metadata(frame_ids, frame_indices, video_names)
