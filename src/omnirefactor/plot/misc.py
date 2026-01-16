import numpy as np
import matplotlib as mpl
from operator import sub


def split_list(lst, N):
    return [lst[i:i + N] for i in range(0, len(lst), N)]


# from https://stackoverflow.com/a/63530703/13326811
def colored_line_segments(xs, ys, zs=None, color='k', mid_colors=False):
    from scipy.interpolate import interp1d
    from matplotlib.colors import colorConverter

    if isinstance(color, str):
        color = colorConverter.to_rgba(color)[:-1]
        color = np.array([color for i in range(len(xs))])
    segs = []
    seg_colors = []
    lastColor = [color[0][0], color[0][1], color[0][2]]
    start = [xs[0], ys[0]]
    end = [xs[0], ys[0]]
    if not zs is None:
        start.append(zs[0])
        end.append(zs[0])
    else:
        zs = [zs] * len(xs)
    for x, y, z, c in zip(xs, ys, zs, color):
        if mid_colors:
            seg_colors.append([(chan + lastChan) * .5 for chan, lastChan in zip(c, lastColor)])
        else:
            seg_colors.append(c)
        lastColor = c[:-1]
        if not z is None:
            start = [end[0], end[1], end[2]]
            end = [x, y, z]
        else:
            start = [end[0], end[1]]
            end = [x, y]
        segs.append([start, end])
    colors = [(*color, 1) for color in seg_colors]
    return segs, colors


def segmented_resample(xs, ys, zs=None, color='k', n_resample=100, mid_colors=False):
    from scipy.interpolate import interp1d
    from matplotlib.colors import colorConverter

    n_points = len(xs)
    if isinstance(color, str):
        color = colorConverter.to_rgba(color)[:-1]
        color = np.array([color for i in range(n_points)])
    n_segs = (n_points - 1) * (n_resample - 1)
    xsInterp = np.linspace(0, 1, n_resample)
    segs = []
    seg_colors = []
    hiResXs = [xs[0]]
    hiResYs = [ys[0]]
    if not zs is None:
        hiResZs = [zs[0]]
    RGB = color.swapaxes(0, 1)
    for i in range(n_points - 1):
        fit_xHiRes = interp1d([0, 1], xs[i:i + 2])
        fit_yHiRes = interp1d([0, 1], ys[i:i + 2])
        xHiRes = fit_xHiRes(xsInterp)
        yHiRes = fit_yHiRes(xsInterp)
        hiResXs = hiResXs + list(xHiRes[1:])
        hiResYs = hiResYs + list(yHiRes[1:])
        R_HiRes = interp1d([0, 1], RGB[0][i:i + 2])(xsInterp)
        G_HiRes = interp1d([0, 1], RGB[1][i:i + 2])(xsInterp)
        B_HiRes = interp1d([0, 1], RGB[2][i:i + 2])(xsInterp)
        lastColor = [R_HiRes[0], G_HiRes[0], B_HiRes[0]]
        start = [xHiRes[0], yHiRes[0]]
        end = [xHiRes[0], yHiRes[0]]
        if not zs is None:
            fit_zHiRes = interp1d([0, 1], zs[i:i + 2])
            zHiRes = fit_zHiRes(xsInterp)
            hiResZs = hiResZs + list(zHiRes[1:])
            start.append(zHiRes[0])
            end.append(zHiRes[0])
        else:
            zHiRes = [zs] * len(xHiRes)

        if mid_colors:
            seg_colors.append([R_HiRes[0], G_HiRes[0], B_HiRes[0]])
        for x, y, z, r, g, b in zip(xHiRes[1:], yHiRes[1:], zHiRes[1:], R_HiRes[1:], G_HiRes[1:], B_HiRes[1:]):
            if mid_colors:
                seg_colors.append([(chan + lastChan) * .5 for chan, lastChan in zip((r, g, b), lastColor)])
            else:
                seg_colors.append([r, g, b])
            lastColor = [r, g, b]
            if not z is None:
                start = [end[0], end[1], end[2]]
                end = [x, y, z]
            else:
                start = [end[0], end[1]]
                end = [x, y]
            segs.append([start, end])

    colors = [(*color, 1) for color in seg_colors]
    data = [hiResXs, hiResYs]
    if not zs is None:
        data = [hiResXs, hiResYs, hiResZs]
    return segs, colors, data


def faded_segment_resample(xs, ys, zs=None, color='k', fade_len=20, n_resample=100, direction='Head'):
    segs, colors, hiResData = segmented_resample(xs, ys, zs, color, n_resample)
    n_segs = len(segs)
    if fade_len > len(segs):
        fade_len = n_segs
    if direction == 'Head':
        # Head fade
        alphas = np.concatenate((np.zeros(n_segs - fade_len), np.linspace(0, 1, fade_len)))
    else:
        # Tail fade
        alphas = np.concatenate((np.linspace(1, 0, fade_len), np.zeros(n_segs - fade_len)))
    colors = [(*color[:-1], alpha) for color, alpha in zip(colors, alphas)]
    return segs, colors, hiResData


# https://stackoverflow.com/a/27537018/13326811
def _get_perp_line(current_seg, out_of_page, linewidth):
    perp = np.cross(current_seg, out_of_page)[0:2]
    perp_unit = _get_unit_vector(perp)
    current_seg_perp_line = perp_unit * linewidth
    return current_seg_perp_line


def _get_unit_vector(vector):
    vector_size = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
    vector_unit = vector / vector_size
    return vector_unit[0:2]


def colored_line(x, y, ax, z=None, line_width=1, MAP='jet'):
    # use pcolormesh to make interpolated rectangles
    num_pts = len(x)
    [xs, ys, zs] = [
        np.zeros((num_pts, 2)),
        np.zeros((num_pts, 2)),
        np.zeros((num_pts, 2))
    ]

    dist = 0
    out_of_page = [0, 0, 1]
    for i in range(num_pts):
        # set the colors and the x,y locations of the source line
        xs[i][0] = x[i]
        ys[i][0] = y[i]
        if i > 0:
            x_delta = x[i] - x[i - 1]
            y_delta = y[i] - y[i - 1]
            seg_length = (x_delta ** 2 + y_delta ** 2) ** 0.5
            dist += seg_length
            zs[i] = [dist, dist]

        # define the offset perpendicular points
        if i == num_pts - 1:
            current_seg = [x[i] - x[i - 1], y[i] - y[i - 1], 0]
        else:
            current_seg = [x[i + 1] - x[i], y[i + 1] - y[i], 0]
        current_seg_perp = _get_perp_line(
            current_seg, out_of_page, line_width)
        if i == 0 or i == num_pts - 1:
            xs[i][1] = xs[i][0] + current_seg_perp[0]
            ys[i][1] = ys[i][0] + current_seg_perp[1]
            continue
        current_pt = [x[i], y[i]]
        current_seg_unit = _get_unit_vector(current_seg)
        previous_seg = [x[i] - x[i - 1], y[i] - y[i - 1], 0]
        previous_seg_perp = _get_perp_line(
            previous_seg, out_of_page, line_width)
        previous_seg_unit = _get_unit_vector(previous_seg)
        # current_pt + previous_seg_perp + scalar * previous_seg_unit =
        # current_pt + current_seg_perp - scalar * current_seg_unit =
        scalar = (
            (current_seg_perp - previous_seg_perp) /
            (previous_seg_unit + current_seg_unit)
        )
        new_pt = current_pt + previous_seg_perp + scalar[0] * previous_seg_unit
        xs[i][1] = new_pt[0]
        ys[i][1] = new_pt[1]

    cm = mpl.colormaps[MAP]

    ax.pcolormesh(xs, ys, zs, shading='gouraud', cmap=cm)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap

    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def get_aspect(ax):
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio
