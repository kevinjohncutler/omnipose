import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .label import apply_label_backgrounds, recolor_label
from .misc import split_list


def image_grid(images, column_titles=None, row_titles=None,
               plot_labels=None,
               xticks=[], yticks=[],
               outline=False, outline_color=[0.5] * 3, outline_width=.5,
               padding=0.05, interset_padding=0.1,
               fontsize=8, fontcolor=[0.5] * 3,
               adaptive_label_color=False,
               label_lightness_threshold=0.6,
               light_label_color=[0.8] * 3,
               dark_label_color=[0.2] * 3,
               label_background=False,
               label_background_color='outline',
               label_background_alpha=1.0,
               label_background_pad=2.0,
               label_background_align_axes=True,
               label_background_full_width=False,
               label_background_debug=False,
               label_background_style='full',
               facecolor=None,
               figsize=6,
               dpi=300,
               order='ij',
               reverse_row=False,
               stack_direction='horizontal',
               lpad=0.05,
               lpos='top_middle',
               return_axes=False,
               fig=None,
               offset=[0, 0],
               supcolor=None,
               right_justify_rows=False,  # New flag for right justification
               **kwargs):

    if supcolor is None:
        supcolor = fontcolor

    label_positions = {
        'top_middle': {'coords': (0.5, 1 - lpad), 'va': 'top', 'ha': 'center'},
        'bottom_left': {'coords': (lpad, lpad), 'va': 'bottom', 'ha': 'left'},
        'bottom_middle': {'coords': (0.5, lpad), 'va': 'bottom', 'ha': 'center'},
        'top_left': {'coords': (lpad, 1 - lpad), 'va': 'top', 'ha': 'left'},
        'above_middle': {'coords': (.5, 1 + lpad), 'va': 'bottom', 'ha': 'center'},

    }

    rgba_cache = {} if adaptive_label_color else None
    label_texts = [] if label_background else None

    # Check if 'images' is a list of lists of lists, meaning multiple image sets
    if isinstance(images[0][0], list):
        multiple_sets = True
    else:
        multiple_sets = False
        images = [images]  # Treat single set as a list of one
        plot_labels = [plot_labels] if plot_labels is not None else None

    n_sets = len(images)
    ij = order == 'ij'

    # ── swap the title lists when using column-major order ─────────────────
    if not ij:
        column_titles, row_titles = row_titles, column_titles

    # Initialize lists to hold positions and sizes
    all_left = []
    all_bottom = []
    all_width = []
    all_height = []

    # Initialize offset for stacking
    total_offset_x = 0
    total_offset_y = 0

    for set_idx, image_set in enumerate(images):
        # ───────────────────── grid dimensions ───────────────────────────
        if ij:
            nrows = len(image_set)
            ncols = max(len(row) for row in image_set)
        else:
            ncols = len(image_set)
            nrows = max(len(col) for col in image_set)

        # ───────────────────── constant-size axis setup ──────────────────
        p = padding  # gap between axes
        base = 1.0  # fixed width (ij) or height (!ij)
        positions = []

        if ij:  # constant widths → variable heights
            cur_bottom = total_offset_y
            for r, row in enumerate(image_set):
                rep = next((im for im in row if im is not None), None)
                ratio = (rep.shape[0] / rep.shape[1]) if rep is not None else 1.0
                h = ratio * base

                row_offset = ((ncols - len(row)) * (base + p)) if right_justify_rows else 0
                for c, _ in enumerate(row):
                    left = total_offset_x + row_offset + c * (base + p)
                    bottom = cur_bottom
                    positions.append((left, bottom, base, h))

                cur_bottom += h + p

            set_span_x = (base + p) * ncols - p
            set_span_y = cur_bottom - total_offset_y - p

        else:  # constant heights → variable widths
            cur_left = total_offset_x
            for c, col in enumerate(image_set):
                rep = next((im for im in col if im is not None), None)
                aspect = (rep.shape[1] / rep.shape[0]) if rep is not None else 1.0
                w = aspect * base

                for r, _ in enumerate(col):
                    left = cur_left
                    bottom = total_offset_y + r * (base + p)
                    positions.append((left, bottom, w, base))

                cur_left += w + p

            set_span_x = cur_left - total_offset_x - p
            set_span_y = (base + p) * nrows - p

        # ───────────────────── collect positions ─────────────────────────
        lefts, bottoms, widths, heights = zip(*positions)
        all_left.extend(lefts)
        all_bottom.extend(bottoms)
        all_width.extend(widths)
        all_height.extend(heights)

        # ───────────────────── inter-set stacking ────────────────────────
        if multiple_sets and set_idx < n_sets - 1:
            if stack_direction == 'horizontal':
                total_offset_x += set_span_x + interset_padding
            elif stack_direction == 'vertical':
                total_offset_y += set_span_y + interset_padding

    # Normalize positions
    lefts = np.array(all_left)
    bottoms = np.array(all_bottom)
    widths = np.array(all_width)
    heights = np.array(all_height)

    max_w = max(lefts + widths)
    max_h = max(bottoms + heights)
    lefts /= max_w
    widths /= max_w

    # Adjust bottoms for top-down layout
    bottoms = (max_h - bottoms - heights) / max_h
    heights /= max_h

    # Use the existing figure if provided; otherwise, create a new one
    if fig is None:
        figsize = (figsize, figsize * max_h / max_w) if ij else (figsize * max_w / max_h, figsize)

        fig = Figure(figsize=figsize,
                     frameon=False if facecolor is None else True,
                     facecolor=[0] * 4 if facecolor is None else facecolor,
                     dpi=dpi)
        FigureCanvas(fig)

    # Apply offsets to the left and bottom positions
    lefts += offset[0]
    bottoms += offset[1]

    # Add the subplots
    axes = []
    for idx, (left, bottom, width, height) in enumerate(zip(lefts, bottoms, widths, heights)):
        ax = fig.add_axes([left, bottom, width, height])
        axes.append(ax)

    # Add images to the subplots
    idx = 0
    for set_idx, image_set in enumerate(images):
        for row_idx, row in enumerate(image_set):
            for col_idx, img in enumerate(row):
                ax = axes[idx]
                idx += 1

                ax.set_xticks(xticks)
                ax.set_yticks(yticks)
                ax.patch.set_alpha(0)

                image_artist = None
                if img is not None:
                    image_artist = ax.imshow(img, **kwargs)

                # Add plot labels
                if plot_labels is not None:
                    try:
                        label = plot_labels[set_idx][row_idx][col_idx]
                    except IndexError:
                        label = None

                    if label is not None:
                        coords = label_positions[lpos]['coords']
                        va = label_positions[lpos]['va']
                        ha = label_positions[lpos]['ha']

                        text = ax.text(coords[0], coords[1], label,
                                       fontsize=fontsize, color=fontcolor, va=va, ha=ha, transform=ax.transAxes)

                        if adaptive_label_color and image_artist is not None:
                            recolor_label(
                                ax,
                                text,
                                image_artist=image_artist,
                                threshold=label_lightness_threshold,
                                light_color=light_label_color,
                                dark_color=dark_label_color,
                                cache=rgba_cache,
                            )

                        if label_background:
                            label_texts.append(text)

                        if img is None:
                            text.set_color([.5] * 4)

                # ── column titles ──────────────────────────────────────────
                if column_titles is not None:
                    want_title = (
                        (ij and row_idx == 0 and col_idx < len(column_titles)) or
                        (not ij and col_idx == 0 and row_idx < len(column_titles))
                    )
                    if want_title and (stack_direction != 'vertical' or set_idx == 0):
                        title_idx = col_idx if ij else row_idx
                        ax.text(0.5, 1 + p,
                                column_titles[title_idx],
                                rotation=0, fontsize=fontsize, color=supcolor,
                                va='bottom', ha='center', transform=ax.transAxes)

                # ── row titles ─────────────────────────────────────────────
                if row_titles is not None:
                    want_title = (
                        (ij and col_idx == 0 and row_idx < len(row_titles)) or
                        (not ij and row_idx == 0 and col_idx < len(row_titles))
                    )
                    if want_title and (stack_direction != 'horizontal' or set_idx == 0):
                        title_idx = row_idx if ij else col_idx
                        ax.text(-p, 0.5,
                                row_titles[title_idx],
                                rotation=0, fontsize=fontsize, color=supcolor,
                                va='center', ha='right', transform=ax.transAxes)

                # Add outline if needed
                if outline:
                    for s in ax.spines.values():
                        s.set_color(outline_color)
                        s.set_linewidth(outline_width)
                else:
                    for s in ax.spines.values():
                        s.set_visible(False)

    if label_background and label_texts:
        apply_label_backgrounds(
            label_texts,
            pad_px=label_background_pad,
            color=label_background_color,
            alpha=label_background_alpha,
            align_to_axes=label_background_align_axes,
            default_color=outline_color,
            full_width=label_background_full_width,
            debug_bbox=label_background_debug,
            style=label_background_style,
        )

    if return_axes:
        pos = [lefts, bottoms, widths, heights]
        return fig, axes, pos
    else:
        return fig


def color_grid(colors, **kwargs):
    # Convert colors to a numpy array
    colors = np.array(colors)

    # If colors is a 1D array (single color), reshape it to a 2D array
    if colors.ndim == 1:
        colors = colors.reshape(1, -1)

    # Ensure colors have 3 components (RGB)
    if colors.shape[-1] == 4:
        # If colors have 4 components (RGBA), remove the alpha component
        colors = colors[:, :3]

    # Create a list of 1x1 images
    images = [[np.full((1, 1, 3), color, dtype=np.float32)] for color in colors]

    # Display the image grid
    return image_grid(images, **kwargs)


def color_swatches(colors, figsize=0.5, dpi=150, fontsize=5, fontcolor='w', padding=0.05,
                   titles=None, ncol=None):
    if ncol is None:
        ncol = len(colors)
    # Convert colors to a numpy array
    colors = np.array(colors)

    # If colors is a 1D array (single color), reshape it to a 2D array
    if colors.ndim == 1:
        colors = colors.reshape(1, -1)

    # Create a list of swatches
    swatches = [np.full((1, 1, 3), color, dtype=np.float32) for color in colors]

    return image_grid(split_list(swatches, ncol),
                      plot_labels=split_list(titles, ncol) if titles is not None else None,
                      padding=0.05, fontsize=fontsize,
                      fontcolor=fontcolor,
                      facecolor=[0] * 4, figsize=figsize * ncol, dpi=dpi)
