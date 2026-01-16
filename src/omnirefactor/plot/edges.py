import types

import numpy as np
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.backend_bases import GraphicsContextBase, RendererBase
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class GC(GraphicsContextBase):
    def __init__(self):
        super().__init__()
        self._capstyle = 'round'


def custom_new_gc(self):
    return GC()


def plot_edges(
        shape,
        affinity_graph,
        neighbors,
        coords,
        figsize=1,
        fig=None,
        ax=None,
        extent=None,
        slc=None,
        pic=None,
        edgecol=[.75] * 3 + [.5],
        linewidth=0.15,
        step_inds=None,
        cmap='inferno',
        origin='lower',
        bounds=None,
):
    """
    Render an affinity graph as line segments laid over an optional image.

    Boundary pixels (including linear index 0) are handled explicitly, so every
    valid edge appears—even when its target lies on the image border.
    """
    # ——————————————————————————————————————————— imports that take time kept local
    from ..utils import get_neigh_inds

    nstep, npix = affinity_graph.shape
    coords = tuple(coords)

    # build lookup tables for neighbours
    indexes, neigh_inds, ind_matrix = get_neigh_inds(tuple(neighbors), coords, shape)

    # default to all steps if none supplied
    if step_inds is None:
        step_inds = np.arange(nstep)

    px_inds = np.arange(npix)

    # -------------------------------------------------------------------------
    # Build edge list manually so edges touching the border are never lost
    # -------------------------------------------------------------------------
    aff_coords = np.array(coords).T  # (2, N) -> (y, x)
    segments = []

    for s in step_inds:
        mask = affinity_graph[s].astype(bool)  # where an edge exists
        if not mask.any():
            continue

        src_idx = px_inds[mask]
        dst_idx = neigh_inds[s, mask]

        valid = dst_idx >= 0  # drop out-of-bounds neighbours
        src_idx = src_idx[valid]
        dst_idx = dst_idx[valid]

        for a, b in zip(src_idx, dst_idx):
            # flip Y/X order for imshow coords and shift to pixel-centres (+0.5)
            segments.append(aff_coords[:, ::-1][[a, b]] + 0.5)

    if not segments:
        raise ValueError("No edges found to plot; check affinity_graph and neighbours.")

    segments = np.stack(segments)

    # -------------------------------------------------------------------------
    # Figure / axes handling
    # -------------------------------------------------------------------------
    RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)

    newfig = fig is None and ax is None
    if newfig:
        if not isinstance(figsize, (list, tuple)):
            figsize = (figsize, figsize)
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)

    if extent is None:
        extent = np.array([0, shape[1], 0, shape[0]])

    # -------------------------------------------------------------------------
    # Background image (affinity heat-map) - create if not supplied
    # -------------------------------------------------------------------------
    nopic = pic is None
    if nopic:
        summed_affinity = np.zeros(shape, dtype=int)
        summed_affinity[coords] = np.sum(affinity_graph, axis=0)

        # build a visually pleasing reversed colormap
        colors = mpl.colormaps.get_cmap(cmap).reversed()(np.linspace(0, 1, 9))
        colors = np.vstack((np.array([0] * 4), colors))  # prepend transparent/black
        affinity_cmap = mpl.colors.ListedColormap(colors)
        pic = affinity_cmap(summed_affinity)

    ax.imshow(pic[slc] if slc is not None else pic,
              extent=extent,
              origin=origin)

    # -------------------------------------------------------------------------
    # Draw edges
    # -------------------------------------------------------------------------
    line_segments = LineCollection(segments, color=edgecol, linewidths=linewidth)
    ax.add_collection(line_segments)

    if newfig:
        ax.set_axis_off()
        ax.invert_yaxis()
        canvas = FigureCanvas(fig)
        canvas.draw()

    # -------------------------------------------------------------------------
    # Return values mirror original signature
    # -------------------------------------------------------------------------
    if nopic:
        return summed_affinity, affinity_cmap
    return None, None
