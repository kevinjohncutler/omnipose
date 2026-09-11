"""Segmentation result visualization (mask overlays, prediction strips).

Generic image display helpers (``imshow``, ``set_outline``, ``outline_view``)
live in ``ocdkit.plot.display``; this module focuses on the omnipose-specific
overlays for showing segmentation results alongside source images.
"""

import os

from .imports import *
from .. import io, transforms
from . import figure, imshow, colorize, normalize99, masks_to_outlines, outline_view
from .overlay import mask_outline_overlay


def image_to_rgb(img0, channels=None, channel_axis=-1):
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
            img[:, :, i] = normalize99(img[:, :, i])
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


def show_segmentation(fig, img, maski, flowi, bdi=None, channels=None, file_name=None,
                      seg_norm=False, bg_color=None, outline_color=[1, 0, 0], img_colors=None,
                      channel_axis=-1, figsize=(12, 3), dpi=300, hold=False,
                      interpolation="bilinear", **kwargs):
    """Plot segmentation results."""
    if fig is None:
        fig, ax = figure(figsize=figsize, dpi=dpi)

    if channels is None:
        channels = [0, 0]
    img0 = img.copy()

    if img0.ndim == 2:
        img0 = image_to_rgb(img0, channels=channels)
    else:
        if channel_axis is None:
            channel_axis = 0
        if img0.shape[0] == 3 and channel_axis != -1:
            img0 = np.transpose(img0, (1, 2, 0))
        if img0.shape[channel_axis] != 3:
            img0 = transforms.move_axis(img0, channel_axis, "first")
            img0 = colorize(img0, colors=img_colors)

    img0 = (normalize99(img0) * (2**8 - 1)).astype(np.uint8)

    if bdi is None or not bdi.shape:
        outlines = masks_to_outlines(maski)
    else:
        outlines = bdi

    if seg_norm:
        fg = 1 / 9
        p = np.clip(normalize99(img0), 0, 1)
        img1 = p ** (np.log(fg) / np.log(np.mean(p[maski > 0])))
    else:
        img1 = img0

    overlay = mask_outline_overlay(img1, maski, outlines)

    outli = outline_view(img0, maski, boundaries=outlines, color=outline_color)

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
