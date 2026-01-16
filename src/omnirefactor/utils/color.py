from .imports import *

def rgb_to_hsv(arr):
    rgb_to_hsv_channels = np.vectorize(colorsys.rgb_to_hsv)
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv_channels(r, g, b)
    hsv = np.stack((h, s, v), axis=-1)
    return hsv


def hsv_to_rgb(arr):
    hsv_to_rgb_channels = np.vectorize(colorsys.hsv_to_rgb)
    h, s, v = np.rollaxis(arr, axis=-1)
    r, g, b = hsv_to_rgb_channels(h, s, v)
    rgb = np.stack((r, g, b), axis=-1)
    return rgb



def sinebow(N, bg_color=[0, 0, 0, 0], offset=0):
    """Generate a color dictionary for N-colored labels.

    Parameters
    ----------
    N: int
        number of distinct colors to generate (excluding background)

    bg_color: ndarray, list, or tuple of length 4
        RGBA values specifying the background color at the front of the dictionary.

    Returns
    -------
    dict
        {int: RGBA array} mapping integer labels to RGBA colors.
    """
    colordict = {0: bg_color}
    for j in range(N):
        k = j + offset
        angle = k * 2 * np.pi / N
        r = ((np.cos(angle) + 1) / 2)
        g = ((np.cos(angle + 2 * np.pi / 3) + 1) / 2)
        b = ((np.cos(angle + 4 * np.pi / 3) + 1) / 2)
        colordict.update({j + 1: [r, g, b, 1]})
    return colordict
