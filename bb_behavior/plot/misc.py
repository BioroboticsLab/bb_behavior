import numpy as np

def draw_ferwar_id_on_axis(ID, ax):
    import bb_utils.visualization
    import skimage.io
    import io
    
    ID = bb_utils.ids.BeesbookID.from_ferwar(ID)
    png = bb_utils.visualization.TagArtist().draw(ID.as_bb_binary())
    png = io.BytesIO(png)
    im = skimage.io.imread(png)
    ax.imshow(im)
    ax.set_axis_off()

def plot_images_as_video(image_array, dpi=72.0, format="html5", jupyter=True):
    """Takes a list of images and plots them as a playable video sequence.

    Arguments:
        image_array: list(np.array)
            List of successive frames.
        dpi: float
            Determines the resolution of the figure.
        format: ("html5", "jshtml")
            Format of the video.
        jupyter: bool
            Whether to display the video in a jupyter notebook.

    Returns:
        fig, anim: matplotlib figure, animation object. If jupyter=True, None is returned.
    """
    import matplotlib.animation
    import matplotlib.pyplot as plt
    
    xpixels, ypixels = image_array[0].shape[0], image_array[0].shape[1]
    
    fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)
    im = plt.figimage(image_array[0].astype(np.float32))
    def animate(i):
        im.set_array(image_array[i].astype(np.float32))
        return (im,)

    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=len(image_array))
    if format == "html5":
        anim = anim.to_html5_video()
    elif format == "jshtml":
        anim = anim.to_jshtml()
    else:
        raise ValueError("'format' must be one of 'html5', 'jshtml'!")
    
    if jupyter:
        from IPython.display import display, HTML
        display(HTML(anim))
        plt.close()
        return None
    return fig, anim