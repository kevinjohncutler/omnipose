import numpy as np
import pytest

from omnipose.io import lists as io_lists
from omnipose.plot import overlay as io_overlay


def test_lists_roundtrip(tmp_path):
    nested = [np.arange(5), np.arange(6).reshape(2, 3)]
    path = tmp_path / "nested.npz"
    io_lists.save_nested_list(path, nested)
    loaded = io_lists.load_nested_list(path)
    assert len(loaded) == len(nested)
    assert np.array_equal(loaded[0], nested[0])
    assert np.array_equal(loaded[1], nested[1])


def test_channel_overlay_basic():
    channels = np.stack([np.zeros((4, 4)), np.ones((4, 4))], axis=0)
    rgb = io_overlay.channel_overlay(channels, color_indexes=[1])
    assert rgb.shape == (4, 4, 3)
    assert np.isfinite(rgb).all()


def test_mask_outline_overlay_basic():
    img = np.zeros((4, 4), dtype=np.float32)
    masks = np.zeros((4, 4), dtype=np.int32)
    masks[1:3, 1:3] = 1
    outlines = np.zeros_like(masks, dtype=np.uint8)
    outlines[1, 1:3] = 1
    overlay = io_overlay.mask_outline_overlay(img, masks, outlines)
    assert overlay.shape == (4, 4, 3)
    assert np.isfinite(overlay).all()


