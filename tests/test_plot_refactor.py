"""Tests for omnirefactor.plot — display, overlay, and edges."""

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from omnirefactor.plot.display import image_to_rgb, outline_view
from omnirefactor.plot.overlay import channel_overlay, mask_outline_overlay
from omnirefactor.plot.edges import GC, custom_new_gc, plot_edges


def _make_img_and_masks():
    img = np.random.rand(32, 32).astype(np.float32)
    masks = np.zeros((32, 32), dtype=np.int32)
    masks[4:14, 4:14] = 1
    masks[18:28, 18:28] = 2
    return img, masks


# ---------------------------------------------------------------------------
# image_to_rgb
# ---------------------------------------------------------------------------

class TestImageToRgb:
    def test_grayscale(self):
        img = np.random.rand(16, 16).astype(np.float32)
        rgb = image_to_rgb(img)
        assert rgb.shape == (16, 16, 3)
        assert rgb.dtype == np.uint8

    def test_multichannel(self):
        img = np.random.rand(16, 16, 2).astype(np.float32)
        rgb = image_to_rgb(img, channels=[1, 2])
        assert rgb.shape == (16, 16, 3)

    def test_channel_first(self):
        img = np.random.rand(3, 16, 16).astype(np.float32)
        rgb = image_to_rgb(img, channels=[1, 2])
        assert rgb.shape == (16, 16, 3)


# ---------------------------------------------------------------------------
# outline_view
# ---------------------------------------------------------------------------

class TestOutlineView:
    def test_basic(self):
        img, masks = _make_img_and_masks()
        result = outline_view(img, masks)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_custom_color(self):
        img, masks = _make_img_and_masks()
        result = outline_view(img, masks, color=[0, 1, 0])
        assert result.shape == (32, 32, 3)

    def test_with_boundaries(self):
        img, masks = _make_img_and_masks()
        bd = np.zeros_like(masks, dtype=np.uint8)
        bd[4, 4:14] = 1
        result = outline_view(img, masks, boundaries=bd)
        assert result.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# channel_overlay
# ---------------------------------------------------------------------------

class TestChannelOverlay:
    def test_basic(self):
        channels = np.random.rand(4, 16, 16).astype(np.float32)
        rgb = channel_overlay(channels, color_indexes=[1])
        assert rgb.shape == (16, 16, 3)
        assert np.isfinite(rgb).all()

    def test_multiple_colors(self):
        channels = np.random.rand(4, 16, 16).astype(np.float32)
        rgb = channel_overlay(channels, color_indexes=[0, 2])
        assert rgb.shape == (16, 16, 3)


# ---------------------------------------------------------------------------
# mask_outline_overlay
# ---------------------------------------------------------------------------

class TestMaskOutlineOverlay:
    def test_basic(self):
        img = np.random.rand(32, 32).astype(np.float32)
        masks = np.zeros((32, 32), dtype=np.int32)
        masks[5:15, 5:15] = 1
        outlines = np.zeros_like(masks, dtype=np.uint8)
        outlines[5, 5:15] = 1
        overlay = mask_outline_overlay(img, masks, outlines)
        assert overlay.shape == (32, 32, 3)
        assert np.isfinite(overlay).all()

    def test_rgb_input(self):
        img = np.random.rand(32, 32, 3).astype(np.float32)
        masks = np.zeros((32, 32), dtype=np.int32)
        masks[5:15, 5:15] = 1
        outlines = np.zeros_like(masks, dtype=np.uint8)
        outlines[5, 5:15] = 1
        overlay = mask_outline_overlay(img, masks, outlines)
        assert overlay.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# edges (GC, plot_edges)
# ---------------------------------------------------------------------------

class TestEdges:
    def test_gc(self):
        gc = GC()
        assert gc._capstyle == 'round'

    def test_custom_new_gc(self):
        from matplotlib.backends.backend_agg import RendererAgg
        fig = Figure()
        canvas = FigureCanvas(fig)
        renderer = canvas.get_renderer()
        gc = custom_new_gc(renderer)
        assert isinstance(gc, GC)
