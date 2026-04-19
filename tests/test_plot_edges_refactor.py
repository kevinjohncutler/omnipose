"""Tests for omnirefactor.plot.edges.plot_edges.

Builds a minimal affinity graph from a small labeled mask (pattern taken from
docs/affinity.ipynb), then verifies rendering produces LineCollection segments
and a summed-affinity heatmap.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.collections import LineCollection

from ocdkit.spatial import kernel_setup, masks_to_affinity, get_neighbors
from omnirefactor.plot.edges import plot_edges


def _build_affinity(mask):
    """Build (shape, affinity_graph, neighbors, coords) for a 2D label mask."""
    dim = mask.ndim
    shape = mask.shape
    coords = np.nonzero(mask)
    steps, inds, idx, fact, sign = kernel_setup(dim)
    neighbors = get_neighbors(coords, steps, dim, shape)
    affinity_graph = masks_to_affinity(
        mask, coords, steps, inds, idx, fact, sign, dim, neighbors=neighbors,
    )
    return shape, affinity_graph, neighbors, coords


class TestPlotEdges:
    def _mask(self):
        m = np.zeros((12, 12), dtype=np.int32)
        m[2:8, 2:8] = 1
        m[9:11, 9:11] = 2
        return m

    def test_returns_heatmap_with_new_fig(self):
        mask = self._mask()
        shape, aff, neigh, coords = _build_affinity(mask)

        summed, cmap = plot_edges(shape, aff, neigh, coords)
        assert summed.shape == shape
        assert summed.sum() > 0
        assert cmap is not None

    def test_draws_line_segments(self):
        mask = self._mask()
        shape, aff, neigh, coords = _build_affinity(mask)

        fig = Figure(figsize=(2, 2))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        summed, cmap = plot_edges(shape, aff, neigh, coords, fig=fig, ax=ax)
        assert summed.shape == shape
        assert cmap is not None

        # Should have added exactly one LineCollection with segments
        line_collections = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(line_collections) == 1
        assert len(line_collections[0].get_segments()) > 0

    def test_step_inds_subset(self):
        """Restricting step_inds should still produce a valid render."""
        mask = self._mask()
        shape, aff, neigh, coords = _build_affinity(mask)

        # Use only cardinal steps (roughly half of the neighborhood)
        nstep = aff.shape[0]
        step_inds = np.arange(nstep)[: nstep // 2]
        summed, cmap = plot_edges(shape, aff, neigh, coords, step_inds=step_inds)
        assert summed.shape == shape

    def test_custom_pic(self):
        """Passing a pre-rendered background image should skip heatmap generation."""
        mask = self._mask()
        shape, aff, neigh, coords = _build_affinity(mask)

        pic = np.zeros(shape + (4,), dtype=np.float32)
        pic[..., 3] = 1.0  # opaque

        fig = Figure(figsize=(2, 2))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        out = plot_edges(shape, aff, neigh, coords, fig=fig, ax=ax, pic=pic)
        assert out == (None, None)

    def test_custom_extent(self):
        mask = self._mask()
        shape, aff, neigh, coords = _build_affinity(mask)
        extent = np.array([0, shape[1] * 2, 0, shape[0] * 2])

        fig = Figure(figsize=(2, 2))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        plot_edges(shape, aff, neigh, coords, fig=fig, ax=ax, extent=extent)

    def test_empty_affinity_raises(self):
        """All-zero affinity should raise rather than silently draw nothing."""
        mask = self._mask()
        shape, aff, neigh, coords = _build_affinity(mask)

        empty = np.zeros_like(aff)
        with pytest.raises(ValueError, match="No edges found"):
            plot_edges(shape, empty, neigh, coords)
