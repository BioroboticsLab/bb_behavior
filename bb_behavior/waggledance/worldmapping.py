import numpy as np
import matplotlib.pyplot as plt

import astropy.coordinates 
import astropy.units as u
import astropy.time
import pytz

import math

def extract_azimuth(time, latitude, longitude):
    """Calculates the azimuth from a datetime and location.
    Arguments:
        time: datetime.datetime with timezone
        latitude, longitude: float

    Returns:
        azimuth: float
    """
    earth_loc = astropy.coordinates.EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg, height=0*u.m)
    
    time = astropy.time.Time(time.astimezone(pytz.UTC), scale="utc")
    sun_loc = astropy.coordinates.get_sun(time)
    azimuth = -sun_loc.transform_to(astropy.coordinates.AltAz(obstime=time, location=earth_loc)).az

    return azimuth.rad

def decode_waggle_dance_angle(hive_angle, time, distance=10.0, latitude=52.457130, longitude=13.296285):
    """Translates an in-hive waggle dance angle to a world angle.
    Arguments:
        hive_angle: float
            In-hive angle, relative to gravity (in radians).
        time: datetime.datetime with timezone
        distance: float
            Distance in arbitrary units (e.g. meters or pixels) to scale the resulting coordinates.
        latitude, longitude: float

    Returns:
        world_angle: float
            Angle in the world (radians).
        x, y: float, float
            Offset of the indicated foraging position with the hive being at 0|0.
            ||[x, y]||_2 == distance.
        
    """
    # Angle is in radians.
    assert hive_angle >= -2.0 * math.pi
    assert hive_angle <= +2.0 * math.pi
    
    azimuth_rad = extract_azimuth(time, latitude, longitude)
    
    world_angle = azimuth_rad + hive_angle
    return world_angle, distance * np.cos(world_angle), distance * np.sin(world_angle)

def get_default_map_image():
    """Returns a screenshot of a map as well as the coordinates in pixels of the hive and a resolution.

    Returns:
        map_image: numpy.array
            Containing the image of a map.
        map_hive_x, map_hive_y: float
            Pixel coordinates of the hive.
        meters_to_pixels: float
            Scale factor that can be taken times meters (meters_to_pixels * 10m) = ?? pixels.
    """
    map_image = plt.imread("/mnt/storage/david/data/beesbook/foragergroups/map.png")
    map_hive_x, map_hive_y = 2185, 1563
    meters_to_pixels = 320.0 / 100.0
    return map_image, map_hive_x, map_hive_y, meters_to_pixels

def plot_hive_map(map_image=None, map_hive_x=None, map_hive_y=None, figsize=(16, 16), title=None, ax=None):
    """Plots an image and marks a specific location.
    """

    if map_image is None:
        map_image, map_hive_x, map_hive_y, _= get_default_map_image()
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(map_image)
    ax.plot([map_hive_x], [map_hive_y], marker="x", markersize=20, markerfacecolor="k", markeredgecolor="k")
    if title is not None:
        plt.title(str(title))
    
    return ax

def plot_waggle_locations(world_locations, ax=None, map_image=None, map_hive_x=None, map_hive_y=None, meters_to_pixels=None, color="g", kde_cmap="Greens"):
    """Takes a list of world locations (e.g. output from decode_waggle_dance) and plots them on an axis.
    """
    
    if map_image is None:
        map_image, map_hive_x, map_hive_y, meters_to_pixels= get_default_map_image()
    
    if ax is None:
        ax = plt.gca()
    pixel_locations = world_locations * meters_to_pixels
    pixel_locations[:, 0] += map_hive_x
    pixel_locations[:, 1] += map_hive_y


    ax.scatter([pixel_locations[:,0]], [pixel_locations[:, 1]], alpha=0.25, marker="o", c=color)
    if kde_cmap:
        import seaborn as sns
        sns.kdeplot(pixel_locations[:,0], pixel_locations[:,1],
                    ax=ax, alpha=0.5, cmap=kde_cmap, legend=True, shade=False, shade_lowest=False)
    
    plt.axis("off")
    plt.xlim(0, map_image.shape[1])
    plt.ylim(map_image.shape[0], 0)

