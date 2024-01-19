import cv2
import numba
import numpy as np

def generate_median_images(gen, N=10):
    """ Takes an image generator and yields a median image of the last N images every N images.

    Arguments:
        gen: generator
            A generator that yields greyscale images.
        N: int
            A median image is generated every N images.
    Yields:
        smooth_image: np.array(dtype=np.float32)
            Image of same shape as original images.
    """
    idx = 1
    try:
        im = next(gen)
    except StopIteration:
        return

    buffer = np.zeros(shape=(N, im.shape[0], im.shape[1]), dtype=np.float32)
    buffer[0] = im

    for i, im in enumerate(gen):
        buffer[idx] = im
        idx = (idx + 1) % N

        if idx == 0: # Output every N images
            smooth_image = np.nanmedian(buffer, axis=0)
            yield smooth_image

@numba.njit
def increase_histogram(hist, image, is_float=False):
    """Takes a histogram and an image. Increases histogram counters for this image in-place.

    Arguments:
        hist: np.array(shape=(255, H, W), dtype=np.float32)
            Histogram that will be changed in-place.
        image: np.array(shape=(H, W))
            Counts for color values from this image will be increased in the histogram.
            Datatype can either be int or float, see is_float.
        is_float: bool
            If True, values from image should be in range (0, 1).
            Otherwise, values should be in range(0, 255).
    """
    for y in numba.prange(image.shape[0]):
        for x in numba.prange(image.shape[1]):
            value = image[y, x]
            if is_float:
                int_value = int(255.0 * value)
            else:
                int_value = int(value)
            hist[int_value, y, x] += 1.0

def generate_mode_images(gen, only_return_one=False, smoothing=0.95):
    """Takes an image generator and yields new images
    with pixel values being the mode of the original images' values.

    Arguments:
        gen: generator
            Generator yielding greyscale images.
            Either integer-based data type and range (0, 255).
            Or float data type and range (0, 1).
        only_return_one: bool
            If true, one modal image will be generated for the whole generator.
            Otherwise, each image will yield the current state of the mode.
        smoothing: float
            Value between [0, 1] to control how fast the histogram adjusts to changes.
            Will be set to 1 if only_return_one is set. 1 = slowest adjustment. 0 = instant.

    Yields:
        modal_image: np.array(dtype=np.uint8)
        Images of the same shape as the original image.
    """
    import itertools

    bins=256
    try:
        first_image = next(gen)
    except StopIteration:
        first_image = None

    if first_image is None:
        return None

    is_float = not np.issubdtype(first_image.dtype, np.integer)
    if is_float and first_image.max() > 1.0:
        raise ValueError("Image appears to be floating data type but max value is above 1.0.")
    histogram = np.zeros(shape=(bins, first_image.shape[0], first_image.shape[1]), dtype=np.float32)
    last_background_image = None

    for i, im in enumerate(itertools.chain((first_image,), gen)):
        diff = 0.0
        increase_histogram(histogram, im, is_float=is_float)

        if (not only_return_one) and i > 3:
            last_background_image = np.argmax(histogram, axis=0)
            yield last_background_image.astype(np.uint8)

        if not only_return_one:
            histogram *= smoothing

    if only_return_one:
        yield np.argmax(histogram, axis=0).astype(np.uint8)

def make_background_video(image_generator, output_filename,
               mode_smoothing=0.95, median_steps=10, codec="XVID", fps=10.0):
    """Takes an image generator and creates and saves a background-subtracted video.

    Arguments:
        image_generator: generator
            Generator yielding grescale images as np.array with shape (H, W).
        output_filename: string
            Filename of the resulting video. Recommended file type: mp4.
        mode_smoothing: float
            Value in range (0, 1). Controls how fast the modal image adjusts to background changes.
            1 = slowest adjustment. 0 = instant.
        median_steps: int
            One median image will be generated for every median_steps images.
        codec: string
            Fourcc codec to be passed to OpenCV.
        fps: float
            Replay speed of the output video.

    """
    from prefetch_generator import BackgroundGenerator

    background_writer = None
    for image in BackgroundGenerator(
                    generate_mode_images(generate_median_images(
                        image_generator, N=median_steps),
                                        smoothing=mode_smoothing),
                    4):
        if image is None:
            return

        if background_writer is None:
            fourcc =  cv2.VideoWriter_fourcc(*codec)
            background_writer = cv2.VideoWriter(output_filename, fourcc, fps, (image.shape[1],image.shape[0]), False)
        background_writer.write(image)

def make_background_image(image_generator, output_filename=None,
                          mode_smoothing=0.95, median_steps=10, use_threading=True):
    """Takes an image generator and creates and returns and optionally saves a background-subtracted image.

    Arguments:
        image_generator: generator
            Generator yielding grescale images as np.array with shape (H, W).
            The images should be dtype=float32 and the values should be between 0.0 and 1.0.
        output_filename: string
            Optional. Filename of the resulting image.
        mode_smoothing: float
            Value in range (0, 1). Controls how fast the modal image adjusts to background changes.
            1 = slowest adjustment. 0 = instant.
        median_steps: int
            One median image will be generated for every median_steps images.
        use_threading: bool
            Default true. Whether to load the images and generate the median in a different thread.
    Returns:
        image: np.array(shape=(H, W), dtype=np.uint8)
            Greyscale background image.
    """
    import imageio

    median_image_generator = generate_median_images(image_generator, N=median_steps)
    if use_threading:
        from prefetch_generator import BackgroundGenerator
        median_image_generator = BackgroundGenerator(median_image_generator, 6)

    image = list(generate_mode_images(median_image_generator,
                only_return_one=True, smoothing=mode_smoothing))
    if image is None or len(image) == 0:
        return None
    image = image[0] # List should contain only one element.

    if output_filename is not None:
        imageio.imwrite(output_filename, image)

    return image
