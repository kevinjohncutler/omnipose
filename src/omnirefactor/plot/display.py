import os

import numpy as np

from .defaults import figure


def set_outline(ax, outline_color=None, outline_width=0):
    """
    - Always hide axis ticks (ax.axis("off")).
    - If outline_color is not None and outline_width > 0,
        show spines with that color/width.
    - Otherwise, hide spines (no border).
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0)

    # Decide whether to draw spines:
    if outline_color is not None and outline_width > 0:
        for spine in ax.spines.values():
            spine.set_edgecolor(outline_color)
            spine.set_linewidth(outline_width)
    else:
        # Hide spines entirely
        for s in ax.spines.values():
            s.set_visible(False)


def imshow(imgs, figsize=2, ax=None, hold=False, titles=None, title_size=8, spacing=0.05,
           textcolor=[0.5] * 3, dpi=300, text_scale=1,
           outline_color=None,     # e.g. [0.5]*3
           outline_width=0.5,     # e.g. 0.5
           show=False,
           **kwargs):
    """
    Display one or more images. Optionally add an outline (colored border)
    around each image if outline_color is not None and outline_width > 0.
    Otherwise, axes ticks etc. remain off, as before.
    """

    # -------------------------------------------------------------
    # If imgs is a list, we display multiple images side by side
    # -------------------------------------------------------------
    if isinstance(imgs, list):
        if titles is None:
            titles = [None] * len(imgs)
        if title_size is None:
            title_size = figsize / len(imgs) * text_scale

        # normalize figsize so figure() gets a 2-tuple of numbers
        if isinstance(figsize, (list, tuple, np.ndarray)):
            if len(figsize) >= 2:
                fig_w, fig_h = float(figsize[0]), float(figsize[1])
            elif len(figsize) == 1:
                fig_w = fig_h = float(figsize[0])
            else:
                fig_w = fig_h = 2.0
        else:
            # scalar: scale width by number of panels like original behavior
            fig_w = float(figsize) * len(imgs)
            fig_h = float(figsize)
        figsize_list = (fig_w, fig_h)

        # Create figure + subplots for multiple images
        fig, axes = figure(
            nrow=1, ncol=len(imgs),
            figsize=figsize_list,
            dpi=dpi,
            frameon=False,
            facecolor=[0, 0, 0, 0]
        )

        for this_ax, img, ttl in zip(axes, imgs, titles):
            this_ax.imshow(img, **kwargs)
            set_outline(this_ax, outline_color, outline_width)
            this_ax.set_facecolor([0, 0, 0, 0])

            if ttl is not None:
                this_ax.set_title(ttl, fontsize=title_size, color=textcolor)

    # -------------------------------------------------------------
    # Otherwise, just one image
    # -------------------------------------------------------------
    else:
        if not isinstance(figsize, (list, tuple, np.ndarray)):
            figsize = (figsize, figsize)
        elif len(figsize) == 2:
            figsize = (figsize[0], figsize[1])
        else:
            figsize = (figsize[0], figsize[0])
        if title_size is None:
            title_size = figsize[0] * text_scale

        if ax is None:
            subplot_args = {
                'frameon': False,
                'figsize': figsize,
                'facecolor': [0, 0, 0, 0],
                'dpi': dpi
            }
            fig, ax = figure(**subplot_args)
        else:
            hold = True
            fig = ax.get_figure()

        ax.imshow(imgs, **kwargs)
        set_outline(ax, outline_color, outline_width)
        ax.set_facecolor([0, 0, 0, 0])

        if titles is not None:
            ax.set_title(titles, fontsize=title_size, color=textcolor)

    if not hold:
        display(fig)
    else:
        return fig


def image_to_rgb(img0, channels=None, channel_axis=-1, omni=False):
    """Convert image to RGB for visualization."""
    from .. import transforms

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
            img[:, :, i] = transforms.normalize99(img[:, :, i], omni=omni)
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
    from .. import transforms, utils
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
            outlines = utils.masks_to_outlines(maski, mode=mode)
    else:
        outlines = boundaries

    out_y, out_x = np.nonzero(outlines)
    imgout = img0.copy()
    imgout[out_y, out_x] = color
    return imgout


def mask_rgb(masks, colors=None):
    """Render masks in random RGB colors."""
    from .. import utils

    if colors is not None:
        if colors.max() > 1:
            colors = np.float32(colors)
            colors /= 255
        colors = utils.rgb_to_hsv(colors)

    hsv = np.zeros((masks.shape[0], masks.shape[1], 3), np.float32)
    hsv[:, :, 2] = 1.0
    for n in range(int(masks.max())):
        ipix = (masks == n + 1).nonzero()
        if colors is None:
            hsv[ipix[0], ipix[1], 0] = np.random.rand()
        else:
            hsv[ipix[0], ipix[1], 0] = colors[n, 0]
        hsv[ipix[0], ipix[1], 1] = np.random.rand() * 0.5 + 0.5
        hsv[ipix[0], ipix[1], 2] = np.random.rand() * 0.5 + 0.5
    rgb = (utils.hsv_to_rgb(hsv) * 255).astype(np.uint8)
    return rgb


def mask_overlay(img, masks, colors=None, omni=False):
    """Overlay masks on grayscale image."""
    from .. import transforms, utils

    if colors is not None:
        if colors.max() > 1:
            colors = np.float32(colors)
            colors /= 255
        colors = utils.rgb_to_hsv(colors)

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
    rgb = (utils.hsv_to_rgb(hsv) * 255).astype(np.uint8)
    return rgb


def show_segmentation(fig, img, maski, flowi, bdi=None, channels=None, file_name=None, omni=False,
                      seg_norm=False, bg_color=None, outline_color=[1, 0, 0], img_colors=None,
                      channel_axis=-1, figsize=(12, 3), dpi=300, hold=False,
                      interpolation="bilinear"):
    """Plot segmentation results."""
    from .. import io, transforms, utils
    try:
        from skimage import color as _skcolor  # noqa: F401
        skimage_enabled = True
    except Exception:
        skimage_enabled = False
    from .colorize import colorize
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

    img0 = (transforms.normalize99(img0, omni=omni) * (2**8 - 1)).astype(np.uint8)

    if bdi is None or not bdi.shape:
        outlines = utils.masks_to_outlines(maski, omni)
    else:
        outlines = bdi

    if seg_norm:
        fg = 1 / 9
        p = np.clip(transforms.normalize99(img0, omni=omni), 0, 1)
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
        io.imsave(save_path + "_overlay.jpg", overlay)
        io.imsave(save_path + "_outlines.jpg", outli)
        io.imsave(save_path + "_flows.jpg", flowi)

    if hold:
        return fig, img1, outlines, overlay
