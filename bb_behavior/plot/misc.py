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

