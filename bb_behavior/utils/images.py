import numpy as np
import skimage.exposure

def get_crop_from_image(xy, image, width=128, clahe=True):
    """Takes one pair of pixel coordinates and the corresponding image and returns a small crops around the point.
    
    Arguments:
        xy: np.array
            shape=(2,) with np.array(x, y).
        images: np.array
    
    Returns:
        crop: np.array
            Cropped image.
    """
    
    xy = (xy - width // 2).reshape(1, 2).astype(np.int32)
    if type(image) is not list:
        image = [image]

    for idx, (xy, im) in enumerate(zip(xy, image)):
        sub_im = np.zeros(shape=(width, width), dtype=np.uint8)
        x, y = xy
        x_begin, y_begin = x, y
        need_fill = False
        if x_begin < 0:
            x_begin = 0
            need_fill = True
        if y_begin < 0:
            y_begin = 0
            need_fill = True
        x_end, y_end = x + width, y + width
        if x_end > im.shape[1]:
            x_end = im.shape[1]
            need_fill = True
        if y_end > im.shape[0]:
            y_end = im.shape[0]
            need_fill = True
        
        if need_fill:
            sub_im += int(np.mean(im) * 255.0)
        to_end_x = sub_im.shape[1] - ((x + width) - x_end)
        to_end_y = sub_im.shape[0] - ((y + width) - y_end)
        sub_im[(y_begin - y):to_end_y, (x_begin - x):to_end_x] = \
                255.0 * im[y_begin:y_end, x_begin:x_end]

        if clahe:
            sub_im = skimage.exposure.equalize_adapthist(sub_im).astype(np.float32)
    
    return sub_im
