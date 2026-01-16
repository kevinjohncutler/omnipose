import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.transforms import Bbox
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.path import Path


LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _coerce_pad(pad):
    if pad is None:
        return 0.0, 0.0
    if isinstance(pad, (list, tuple, np.ndarray)):
        if len(pad) == 0:
            return 0.0, 0.0
        if len(pad) == 1:
            val = float(pad[0])
            return val, val
        return float(pad[0]), float(pad[1])
    val = float(pad)
    return val, val


def _get_renderer(fig):
    if fig is None:
        return None
    canvas = getattr(fig, 'canvas', None)
    if canvas is None:
        try:
            canvas = FigureCanvas(fig)
            fig.set_canvas(canvas)
        except Exception:
            return None
    try:
        renderer = canvas.get_renderer()
    except Exception:
        renderer = None
    if renderer is not None:
        return renderer
    try:
        canvas.draw()
        return canvas.get_renderer()
    except Exception:
        return getattr(canvas, '_renderer', None)


def _ensure_figure_rendered(fig):
    if fig is None:
        return
    canvas = getattr(fig, 'canvas', None)
    if canvas is None:
        try:
            canvas = FigureCanvas(fig)
            fig.set_canvas(canvas)
        except Exception:
            return
    try:
        canvas.draw()
    except Exception:
        pass


def _get_display_rgb(image_artist, cache=None):
    if image_artist is None:
        return None

    cache_key = id(image_artist)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    raw = image_artist.get_array()
    if raw is None:
        if cache is not None:
            cache[cache_key] = None
        return None

    arr = np.ma.asarray(raw)
    if arr.size == 0:
        if cache is not None:
            cache[cache_key] = None
        return None

    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 1):
        data = np.squeeze(arr)
        norm = image_artist.norm
        if norm is None:
            norm = mpl.colors.Normalize()
            norm.autoscale(data)
        mapped = np.ma.asarray(image_artist.cmap(norm(data)))[..., :3]
        if isinstance(mapped, np.ma.MaskedArray):
            rgb = mapped.astype(np.float32).filled(np.nan)
        else:
            rgb = np.asarray(mapped, dtype=np.float32)
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        data = arr.astype(np.float32, copy=False)
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            rng = info.max - info.min
            if rng > 0:
                data = (data - info.min) / float(rng)
        clipped = np.clip(data, 0.0, 1.0)[..., :3]
        if isinstance(clipped, np.ma.MaskedArray):
            rgb = clipped.astype(np.float32).filled(np.nan)
        else:
            rgb = np.asarray(clipped, dtype=np.float32)
    else:
        rgb = None

    if cache is not None:
        cache[cache_key] = rgb
    return rgb


def _get_text_bbox_display(text, pad=(1.05, 1.25)):
    if text is None:
        return None

    fig = text.figure
    _ensure_figure_rendered(fig)
    canvas = getattr(fig, 'canvas', None)
    if canvas is None:
        canvas = FigureCanvas(fig)
        fig.set_canvas(canvas)
    try:
        canvas.draw()
        renderer = canvas.get_renderer()
    except Exception:
        renderer = None

    bbox = None
    if renderer is not None:
        try:
            bbox = text.get_window_extent(renderer=renderer)
        except Exception:
            bbox = None

    if bbox is not None and fig is not None:
        text_str = text.get_text() or ""
        if text_str:
            try:
                prop = text.get_fontproperties()
                if prop is None:
                    prop = FontProperties()
                else:
                    prop = prop.copy()
                prop.set_size(text.get_fontsize())
                path = TextPath((0, 0), text_str, prop=prop, usetex=False)
                path_bbox = path.get_extents()
            except Exception:
                path_bbox = None
            if path_bbox is not None:
                scale = (fig.dpi or 72.0) / 72.0
                target_width = path_bbox.width * scale
                target_height = path_bbox.height * scale
                x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
                width = x1 - x0
                height = y1 - y0
                ha = text.get_ha()
                va = text.get_va()
                eps = 0.5
                if target_width - width > eps:
                    if ha == 'left':
                        x1 = x0 + target_width
                    elif ha == 'right':
                        x0 = x1 - target_width
                    else:
                        cx = 0.5 * (x0 + x1)
                        x0 = cx - target_width * 0.5
                        x1 = cx + target_width * 0.5
                if target_height - height > eps:
                    if va in ('bottom', 'baseline'):
                        y1 = y0 + target_height
                    elif va == 'top':
                        y0 = y1 - target_height
                    else:
                        cy = 0.5 * (y0 + y1)
                        y0 = cy - target_height * 0.5
                        y1 = cy + target_height * 0.5
                bbox = Bbox.from_extents(x0, y0, x1, y1)
                fudge = max(1.0, (fig.dpi or 72.0) * 0.002)
                bbox = Bbox.from_extents(bbox.x0, bbox.y0,
                                         bbox.x1 + fudge,
                                         bbox.y1 + fudge)

    if bbox is None:
        pos = text.get_position()
        transform = text.get_transform()
        if fig is None or transform is None or pos is None:
            return None
        try:
            anchor = transform.transform(pos)
        except Exception:
            return None
        text_str = text.get_text() or ""
        fontsize_pt = text.get_fontsize() or 0
        dpi = getattr(fig, 'dpi', 72) or 72
        base_font_px = max(fontsize_pt, 1e-3) * (dpi / 72.0)
        approx_char_px = 0.6 * base_font_px
        lines = text_str.splitlines() or [text_str]
        n_lines = max(1, len(lines))
        max_line_len = max((len(line) for line in lines), default=1)
        text_width_px = max(max_line_len * approx_char_px, base_font_px)
        text_height_px = max(n_lines * base_font_px, base_font_px)
        half_w = max(2.0, 0.5 * text_width_px)
        half_h = max(2.0, 0.5 * text_height_px)
        ha = text.get_ha()
        va = text.get_va()
        center_x, center_y = anchor
        if ha == 'left':
            center_x += half_w
        elif ha == 'right':
            center_x -= half_w
        if va in ('bottom', 'baseline'):
            center_y += half_h
        elif va == 'top':
            center_y -= half_h
        bbox = Bbox.from_extents(center_x - half_w, center_y - half_h,
                                 center_x + half_w, center_y + half_h)

    if bbox is None:
        return None

    if pad is not None and len(pad) == 2:
        bbox = bbox.expanded(pad[0], pad[1])
    return bbox


def _sample_label_lightness(image_artist, ax, text, cache=None):
    rgb = _get_display_rgb(image_artist, cache=cache)
    if rgb is None or ax is None or text is None:
        return None

    bbox = _get_text_bbox_display(text)
    if bbox is None:
        return None

    axis_bbox = getattr(ax, 'bbox', None)
    if axis_bbox is None:
        return None

    disp_min = np.array([max(bbox.x0, axis_bbox.x0), max(bbox.y0, axis_bbox.y0)])
    disp_max = np.array([min(bbox.x1, axis_bbox.x1), min(bbox.y1, axis_bbox.y1)])

    if disp_min[0] >= disp_max[0] or disp_min[1] >= disp_max[1]:
        return None

    try:
        inv = ax.transData.inverted()
        data_corners = inv.transform(np.vstack((disp_min, disp_max)))
    except Exception:
        return None

    data_x = np.sort(data_corners[:, 0])
    data_y = np.sort(data_corners[:, 1])

    try:
        x0, x1, y0, y1 = image_artist.get_extent()
    except Exception:
        return None

    nx = rgb.shape[1]
    ny = rgb.shape[0]
    if nx == 0 or ny == 0:
        return None

    denom_x = x1 - x0
    denom_y = y1 - y0
    if denom_x == 0 or denom_y == 0:
        return None

    col_fracs = np.sort((np.array(data_x) - x0) / denom_x)
    row_fracs = np.sort((np.array(data_y) - y0) / denom_y)

    def _index_range(fracs, length):
        if length <= 1:
            return 0, 0
        scale = length - 1
        start = int(np.floor(np.clip(fracs[0], 0.0, 1.0) * scale))
        end = int(np.ceil(np.clip(fracs[1], 0.0, 1.0) * scale))
        start = min(max(start, 0), length - 1)
        end = min(max(end, 0), length - 1)
        return start, end

    col_start, col_end = _index_range(col_fracs, nx)
    row_start, row_end = _index_range(row_fracs, ny)

    if col_end < col_start or row_end < row_start:
        return None

    patch = rgb[row_start:row_end + 1, col_start:col_end + 1]
    if patch.size == 0:
        return None

    luminance = np.tensordot(patch, LUMA_WEIGHTS, axes=([-1], [0]))
    value = float(np.nanmean(luminance))
    if not np.isfinite(value):
        return None
    return value


def recolor_label(ax, text, image_artist=None,
                  threshold=0.6,
                  light_color=(0.8, 0.8, 0.8),
                  dark_color=(0.2, 0.2, 0.2),
                  cache=None):
    """Update a label's color based on the lightness of the underlying image."""

    if text is None:
        return None

    if ax is None:
        ax = text.axes

    if image_artist is None and ax is not None and ax.images:
        image_artist = ax.images[-1]

    if ax is None or image_artist is None:
        return None

    lightness = _sample_label_lightness(image_artist, ax, text, cache=cache)
    if lightness is None:
        return None

    if lightness >= threshold:
        text.set_color(dark_color)
    else:
        text.set_color(light_color)

    return lightness


def _create_pill_patch(ax, rect_x, rect_y, rect_w, rect_h,
                       orientation, facecolor, zorder):
    if rect_w <= 0 or rect_h <= 0:
        return None

    x0, y0 = rect_x, rect_y
    x1, y1 = rect_x + rect_w, rect_y + rect_h
    verts = []
    codes = []
    N = 20

    def add(pt, code):
        verts.append(pt)
        codes.append(code)

    if orientation in ("right", "left"):
        r = min(rect_h / 2.0, rect_w)
        if r <= 0:
            return None
        if orientation == "right":
            cx = x1 - r
            cy = y0 + rect_h / 2.0
            theta = np.linspace(-np.pi / 2, np.pi / 2, N)
            add((x0, y0), Path.MOVETO)
            add((cx, y0), Path.LINETO)
            for t in theta[1:-1]:
                add((cx + r * np.cos(t), cy + r * np.sin(t)), Path.LINETO)
            add((cx, y1), Path.LINETO)
            add((x0, y1), Path.LINETO)
        else:  # left rounded
            cx = x0 + r
            cy = y0 + rect_h / 2.0
            theta = np.linspace(np.pi / 2, 3 * np.pi / 2, N)
            add((x1, y0), Path.MOVETO)
            add((cx, y0), Path.LINETO)
            for t in theta[1:-1]:
                add((cx + r * np.cos(t), cy + r * np.sin(t)), Path.LINETO)
            add((cx, y1), Path.LINETO)
            add((x1, y1), Path.LINETO)
    elif orientation in ("top", "bottom"):
        r = min(rect_w / 2.0, rect_h)
        if r <= 0:
            return None
        cx = x0 + rect_w / 2.0
        if orientation == "top":
            cy = y1 - r
            theta = np.linspace(0, np.pi, N)
            add((x0, y0), Path.MOVETO)
            add((x1, y0), Path.LINETO)
            add((x1, cy), Path.LINETO)
            for t in theta[1:-1]:
                add((cx + r * np.cos(t), cy + r * np.sin(t)), Path.LINETO)
            add((x0, cy), Path.LINETO)
            add((x0, y0), Path.LINETO)
        else:
            cy = y0 + r
            theta = np.linspace(np.pi, 2 * np.pi, N)
            add((x0, y1), Path.MOVETO)
            add((x1, y1), Path.LINETO)
            add((x1, cy), Path.LINETO)
            for t in theta[1:-1]:
                add((cx + r * np.cos(t), cy + r * np.sin(t)), Path.LINETO)
            add((x0, cy), Path.LINETO)
            add((x0, y1), Path.LINETO)
    else:
        return None

    add(verts[0], Path.CLOSEPOLY)
    path = Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=facecolor,
                               edgecolor='none', linewidth=0,
                               transform=ax.transAxes)
    patch.set_zorder(zorder)
    return patch


def add_label_background(ax, text, width_px=None, pad_px=2.0,
                         color=None, alpha=1.0,
                         align_to_axes=True, cache_bbox=None,
                         zorder_offset=-0.3,
                         use_axis_width=False,
                         debug_bbox=False,
                         style='full'):
    """Add a rectangular background patch behind ``text`` within ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for the label.
    text : matplotlib.text.Text
        The text artist to backfill.
    width_px : float, optional
        Desired width of the panel in display pixels.  If ``None`` the
        text width (plus padding) is used.
    pad_px : float or (float, float)
        Padding in display pixels along (x, y).
    color : color-like or ``None``
        Fill color for the panel. ``None``/``'auto'`` uses white.
    alpha : float
        Opacity of the panel fill.
    align_to_axes : bool
        If ``True`` the panel edge nearest the axes (based on the text
        alignment) is locked to the axes boundary.
    cache_bbox : matplotlib.transforms.Bbox, optional
        Precomputed display-space bbox for the text.
    zorder_offset : float
        Offset applied so the background sits behind the text.
    use_axis_width : bool
        If ``True`` the rectangle spans the full axis width.
    """

    if text is None:
        return None

    if ax is None:
        ax = text.axes

    if ax is None:
        return None

    fig = ax.figure

    bbox = cache_bbox
    if bbox is None:
        bbox = _get_text_bbox_display(text, pad=None)

    if bbox is None:
        return None

    axis_bbox = getattr(ax, 'bbox', None)
    if axis_bbox is None:
        return None

    pad_x, pad_y = _coerce_pad(pad_px)
    symmetric_x = pad_x < 0
    symmetric_y = pad_y < 0
    if symmetric_x:
        pad_x = 0.0
    if symmetric_y:
        pad_y = 0.0
    axis_width = axis_bbox.width
    axis_height = axis_bbox.height
    if axis_width <= 0 or axis_height <= 0:
        return None

    left = bbox.x0 - pad_x
    right = bbox.x1 + pad_x
    bottom = bbox.y0 - pad_y
    top = bbox.y1 + pad_y

    ha = text.get_ha()
    va = text.get_va()
    anchor_left = align_to_axes and ha == 'left'
    anchor_right = align_to_axes and ha == 'right'
    anchor_bottom = align_to_axes and va in ('bottom', 'baseline')
    anchor_top = align_to_axes and va == 'top'

    pill_mode = (style == 'pill')
    epsilon = 1.0
    if pill_mode:
        if anchor_left and not anchor_right:
            right += epsilon
        elif anchor_right and not anchor_left:
            left -= epsilon
        elif anchor_top and not anchor_bottom:
            bottom -= epsilon
        elif anchor_bottom and not anchor_top:
            top += epsilon
    symmetric_x = pad_x < 0
    symmetric_y = pad_y < 0
    if symmetric_x:
        pad_x = 0.0
    if symmetric_y:
        pad_y = 0.0
    axis_width = axis_bbox.width
    axis_height = axis_bbox.height
    if axis_width <= 0 or axis_height <= 0:
        return None

    left = bbox.x0 - pad_x
    right = bbox.x1 + pad_x
    bottom = bbox.y0 - pad_y
    top = bbox.y1 + pad_y

    ha = text.get_ha()
    va = text.get_va()
    anchor_left = align_to_axes and ha == 'left'
    anchor_right = align_to_axes and ha == 'right'
    anchor_bottom = align_to_axes and va in ('bottom', 'baseline')
    anchor_top = align_to_axes and va == 'top'

    pill_mode = (style == 'pill')

    if pill_mode and align_to_axes:
        if ha == 'left':
            anchor_left, anchor_right = True, False
            anchor_top = anchor_bottom = False
        elif ha == 'right':
            anchor_right, anchor_left = True, False
            anchor_top = anchor_bottom = False
        elif va == 'top':
            anchor_top = True
            anchor_bottom = anchor_left = anchor_right = False
        elif va in ('bottom', 'baseline'):
            anchor_bottom = True
            anchor_top = anchor_left = anchor_right = False
        else:
            anchor_left = anchor_right = anchor_top = anchor_bottom = False

    base_left, base_right = left, right
    base_bottom, base_top = bottom, top

    left_adjust = 0.0
    right_adjust = 0.0
    bottom_adjust = 0.0
    top_adjust = 0.0

    if anchor_left:
        left_adjust = base_left - axis_bbox.x0
        left = axis_bbox.x0
    if anchor_right:
        right_adjust = axis_bbox.x1 - base_right
        right = axis_bbox.x1
    if anchor_bottom:
        bottom_adjust = base_bottom - axis_bbox.y0
        bottom = axis_bbox.y0
    if anchor_top:
        top_adjust = axis_bbox.y1 - base_top
        top = axis_bbox.y1

    desired_width = right - left
    if desired_width <= 0:
        desired_width = (bbox.width + 2 * pad_x)

    if use_axis_width:
        target_width = axis_width
    elif width_px is not None:
        target_width = min(max(width_px, desired_width), axis_width)
    else:
        target_width = desired_width

    if target_width > desired_width:
        extra = target_width - desired_width
        if anchor_left and not anchor_right:
            right = min(axis_bbox.x1, right + extra)
        elif anchor_right and not anchor_left:
            left = max(axis_bbox.x0, left - extra)
        else:
            left -= 0.5 * extra
            right += 0.5 * extra

    if symmetric_x:
        if anchor_left and not anchor_right:
            right = min(axis_bbox.x1, right + left_adjust)
        elif anchor_right and not anchor_left:
            left = max(axis_bbox.x0, left - right_adjust)
        else:
            left -= 0.5 * (left_adjust + right_adjust)
            right += 0.5 * (left_adjust + right_adjust)

    if symmetric_y:
        if anchor_bottom and not anchor_top:
            top = min(axis_bbox.y1, top + bottom_adjust)
        elif anchor_top and not anchor_bottom:
            bottom = max(axis_bbox.y0, bottom - top_adjust)
        else:
            bottom -= 0.5 * (bottom_adjust + top_adjust)
            top += 0.5 * (bottom_adjust + top_adjust)
    # Extend capsules so the rounded end does not intrude into the text region
    if style == 'pill' and top > bottom and right > left:
        s = .25
        height = top - bottom
        width = right - left
        if anchor_left and not anchor_right:
            right += s * height
        elif anchor_right and not anchor_left:
            left -= s * height
        elif anchor_top and not anchor_bottom:
            bottom -= s * width
        elif anchor_bottom and not anchor_top:
            top += s * width

    # Ensure the rectangle fully covers the text even when anchored to axes
    right = max(right, bbox.x1)
    left = min(left, bbox.x0)
    top = max(top, bbox.y1)
    bottom = min(bottom, bbox.y0)

    left = max(axis_bbox.x0, min(axis_bbox.x1, left))
    right = max(axis_bbox.x0, min(axis_bbox.x1, right))
    bottom = max(axis_bbox.y0, min(axis_bbox.y1, bottom))
    top = max(axis_bbox.y0, min(axis_bbox.y1, top))

    if right <= left or top <= bottom:
        return None

    if debug_bbox and fig is not None:
        try:
            prop = text.get_fontproperties()
            if prop is None:
                prop = FontProperties()
            else:
                prop = prop.copy()
            prop.set_size(text.get_fontsize())
            path = TextPath((0, 0), text.get_text() or "", prop=prop, usetex=False)
            path_bbox = path.get_extents()
            path_width = path_bbox.width * (fig.dpi or 72.0) / 72.0
        except Exception:
            path_width = None
        win_width = (bbox.x1 - bbox.x0)
        if path_width is not None:
            print(f"[bbox debug] label={text.get_text()} window_width={win_width:.2f} path_width={path_width:.2f}")
        else:
            print(f"[bbox debug] label={text.get_text()} window_width={win_width:.2f}")

    rect_x = (left - axis_bbox.x0) / axis_width
    rect_y = (bottom - axis_bbox.y0) / axis_height
    rect_w = (right - left) / axis_width
    rect_h = (top - bottom) / axis_height
    if rect_w <= 0 or rect_h <= 0:
        return None

    if color is None or color == 'auto':
        base_color = 'white'
    else:
        base_color = color

    facecolor = mpl.colors.to_rgba(base_color, alpha)

    orientation = None
    if style == 'pill':
        if anchor_left and not anchor_right:
            orientation = 'right'
        elif anchor_right and not anchor_left:
            orientation = 'left'
        elif anchor_top and not anchor_bottom:
            orientation = 'bottom'
        elif anchor_bottom and not anchor_top:
            orientation = 'top'

    if style == 'pill' and orientation is not None:
        patch = _create_pill_patch(ax, rect_x, rect_y, rect_w, rect_h,
                                   orientation, facecolor, text.get_zorder() + zorder_offset)
        if patch is None:
            patch = mpatches.Rectangle((rect_x, rect_y), rect_w, rect_h,
                                       facecolor=facecolor, edgecolor='none',
                                       linewidth=0, transform=ax.transAxes)
    else:
        patch = mpatches.Rectangle((rect_x, rect_y), rect_w, rect_h,
                                   facecolor=facecolor, edgecolor='none',
                                   linewidth=0, transform=ax.transAxes)
    patch.set_zorder(text.get_zorder() + zorder_offset)
    ax.add_patch(patch)
    text.set_zorder(max(text.get_zorder(), patch.get_zorder() + 0.1))

    if debug_bbox:
        inv_axes = ax.transAxes.inverted()
        raw_x0, raw_y0 = inv_axes.transform((bbox.x0, bbox.y0))
        raw_x1, raw_y1 = inv_axes.transform((bbox.x1, bbox.y1))
        dbg_x = min(raw_x0, raw_x1)
        dbg_y = min(raw_y0, raw_y1)
        dbg_w = abs(raw_x1 - raw_x0)
        dbg_h = abs(raw_y1 - raw_y0)
        dbg_patch = mpatches.Rectangle((dbg_x, dbg_y), dbg_w, dbg_h,
                                       facecolor='none', edgecolor='red',
                                       linewidth=0.5, linestyle='--',
                                       transform=ax.transAxes)
        dbg_patch.set_zorder(patch.get_zorder() + 0.1)
        ax.add_patch(dbg_patch)

    return patch


def apply_label_backgrounds(texts, pad_px=2.0, color='outline', alpha=1.0,
                            align_to_axes=True, default_color=None,
                            full_width=False, debug_bbox=False,
                            style='full'):
    """Apply uniform label backgrounds to a list of text artists."""

    if not texts:
        return

    processed = []
    widths = []
    pad_x, _ = _coerce_pad(pad_px)

    for text in texts:
        if text is None:
            continue
        bbox = _get_text_bbox_display(text, pad=None)
        if bbox is None:
            continue
        processed.append((text, bbox))
        widths.append(bbox.width)

    if not processed:
        return

    target_width = None
    if not full_width:
        target_width = max(widths) + 2 * pad_x

    for text, bbox in processed:
        ax = text.axes
        if ax is None:
            continue
        facecolor = color
        if facecolor in (None, 'outline', 'auto'):
            facecolor = default_color
        add_label_background(
            ax,
            text,
            width_px=target_width,
            pad_px=pad_px,
            color=facecolor,
            alpha=alpha,
            align_to_axes=align_to_axes,
            cache_bbox=bbox,
            use_axis_width=full_width,
            debug_bbox=debug_bbox,
            style=style,
        )
