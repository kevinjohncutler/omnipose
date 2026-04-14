"""Segmentation result visualization (mask overlays, prediction strips).

Generic image display helpers (``imshow``, ``set_outline``) live in
``ocdkit.plot.display``; this module focuses on the omnipose-specific
overlays for showing segmentation results alongside source images.
"""

import os

import numpy as np

from . import figure, imshow, rgb_to_hsv, hsv_to_rgb, normalize99, masks_to_outlines


def image_to_rgb(img0, channels=None, channel_axis=-1, omni=False):
    """Convert image to RGB for visualization."""
    img = img0.copy().astype(np.float32)
    if img.ndim < 3:
        img = img[..., np.newaxis]
        channels = [0, 0]
    if img.shape[0] < 5:
        img = np.transpose(img, (1, 2, 0))

    if channels is None:
        if np.all(img0[..., 0] == img0[..., 1]):
            channels = [0, 0]
        else:
            channels = [i + 1 for i in range(img0.shape[channel_axis])]

    if channels[0] == 0:
        img = img.mean(axis=-1)[:, :, np.newaxis]
    for i in range(img.shape[-1]):
        if np.ptp(img[:, :, i]) > 0:
            img[:, :, i] = normalize99(img[:, :, i], omni=omni)
            img[:, :, i] = np.clip(img[:, :, i], 0, 1)

    img *= 255
    img = np.uint8(img)
    rgb = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    if img.shape[-1] == 1:
        rgb = np.tile(img, (1, 1, 3))
    else:
        rgb[:, :, channels[0] - 1] = img[:, :, 0]
        if channels[1] > 0:
            rgb[:, :, channels[1] - 1] = img[:, :, 1]
    return rgb


def outline_view(img0, maski, boundaries=None, color=[1, 0, 0],
                 channels=None, channel_axis=-1,
                 mode="inner", connectivity=2, skip_formatting=False):
    """Overlay outlines on an image."""
    try:
        from skimage.segmentation import find_boundaries
        skimage_enabled = True
    except Exception:
        skimage_enabled = False

    if np.max(color) <= 1 and not skip_formatting:
        color = np.array(color) * (2**8 - 1)

    if not skip_formatting:
        img0 = image_to_rgb(img0, channels=channels, channel_axis=channel_axis, omni=True)

    if boundaries is None:
        if skimage_enabled:
            outlines = find_boundaries(maski, mode=mode, connectivity=connectivity)
        else:
            outlines = masks_to_outlines(maski, mode=mode)
    else:
        outlines = boundaries

    out_y, out_x = np.nonzero(outlines)
    imgout = img0.copy()
    imgout[out_y, out_x] = color
    return imgout


def mask_overlay(img, masks, colors=None, omni=False):
    """Overlay masks on grayscale image."""
    if colors is not None:
        if colors.max() > 1:
            colors = np.float32(colors)
            colors /= 255
        colors = rgb_to_hsv(colors)

    if img.ndim > 2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)

    hsv = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    hsv[:, :, 2] = np.clip((img / 255.0 if img.max() > 1 else img) * 1.5, 0, 1)
    hues = np.linspace(0, 1, masks.max() + 1)[np.random.permutation(masks.max())]
    for n in range(int(masks.max())):
        ipix = (masks == n + 1).nonzero()
        if colors is None:
            hsv[ipix[0], ipix[1], 0] = hues[n]
        else:
            hsv[ipix[0], ipix[1], 0] = colors[n, 0]
        hsv[ipix[0], ipix[1], 1] = 1.0
    rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
    return rgb


def show_segmentation(fig, img, maski, flowi, bdi=None, channels=None, file_name=None, omni=False,
                      seg_norm=False, bg_color=None, outline_color=[1, 0, 0], img_colors=None,
                      channel_axis=-1, figsize=(12, 3), dpi=300, hold=False,
                      interpolation="bilinear"):
    """Plot segmentation results."""
    from .. import io, transforms
    try:
        from skimage import color as _skcolor  # noqa: F401
        skimage_enabled = True
    except Exception:
        skimage_enabled = False
    from . import colorize
    from .overlay import mask_outline_overlay

    if fig is None:
        fig, ax = figure(figsize=figsize, dpi=dpi)

    if channels is None:
        channels = [0, 0]
    img0 = img.copy()

    if img0.ndim == 2:
        img0 = image_to_rgb(img0, channels=channels, omni=omni)
    else:
        if channel_axis is None:
            channel_axis = 0
        if img0.shape[0] == 3 and channel_axis != -1:
            img0 = np.transpose(img0, (1, 2, 0))
        if img0.shape[channel_axis] != 3:
            img0 = transforms.move_axis(img0, channel_axis, "first")
            img0 = colorize(img0, colors=img_colors)

    img0 = (normalize99(img0, omni=omni) * (2**8 - 1)).astype(np.uint8)

    if bdi is None or not bdi.shape:
        outlines = masks_to_outlines(maski, omni)
    else:
        outlines = bdi

    if seg_norm:
        fg = 1 / 9
        p = np.clip(normalize99(img0, omni=omni), 0, 1)
        img1 = p ** (np.log(fg) / np.log(np.mean(p[maski > 0])))
    else:
        img1 = img0

    if omni and skimage_enabled:
        overlay = mask_outline_overlay(img1, maski, outlines)
    else:
        overlay = mask_overlay(img0, maski)

    outli = outline_view(
        img0,
        maski,
        boundaries=outlines,
        color=np.array(outline_color) * 255,
        channels=channels,
        channel_axis=channel_axis,
        skip_formatting=True,
    )

    ax = fig.get_axes()[0]
    fig = imshow(
        [img0, outli, overlay, flowi],
        ax=ax,
        titles=["original image", "predicted outlines", "predicted masks", "predicted flow field"],
        interpolation=interpolation,
        hold=hold,
        figsize=figsize,
        dpi=dpi,
    )

    if file_name is not None:
        save_path = os.path.splitext(file_name)[0]
        io.imwrite(save_path + "_overlay.jpg", overlay)
        io.imwrite(save_path + "_outlines.jpg", outli)
        io.imwrite(save_path + "_flows.jpg", flowi)

    if hold:
        return fig, img1, outlines, overlay
