def get_first_frame_from_video(vid_file):
    import cv2
    capture = cv2.VideoCapture(vid_file)
    success, image = capture.read()
    if not success:
        return None
    return image