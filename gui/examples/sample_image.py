"""Shared helpers for GUI comparison examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from imageio import v2 as imageio

from cellpose_omni.gui import PRELOAD_IMAGE

ArrayLike = np.ndarray

INSTANCE_COLOR_TABLE = np.array(
    [
        (0, 0, 0, 0),
        (255, 0, 0, 120),
        (0, 180, 255, 120),
        (0, 200, 0, 120),
        (255, 109, 0, 120),
        (255, 0, 200, 120),
        (150, 255, 0, 120),
        (255, 216, 0, 120),
        (137, 0, 255, 120),
        (0, 255, 164, 120),
    ],
    dtype=np.uint8,
)

DEFAULT_BRUSH_RADIUS = 6


def get_preload_image_path() -> Path:
    """Return the path to the default Omnipose sample image."""
    return Path(PRELOAD_IMAGE)


def _load_raw() -> ArrayLike:
    path = get_preload_image_path()
    return imageio.imread(path)


def _ensure_spatial_last(array: ArrayLike) -> ArrayLike:
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        return np.moveaxis(array, 0, -1)
    return array


def _normalize_uint8(array: ArrayLike) -> ArrayLike:
    array = np.asarray(array)
    array = array.astype(np.float32)
    array -= array.min()
    maxv = array.max()
    if maxv > 0:
        array /= maxv
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return array


def load_image_uint8(as_rgb: bool = False) -> ArrayLike:
    """Load the default sample image and normalize it to uint8."""
    data = _load_raw()
    data = _ensure_spatial_last(data)
    data = _normalize_uint8(data)
    if as_rgb:
        if data.ndim == 2:
            data = np.repeat(data[..., None], 3, axis=-1)
        elif data.ndim == 3 and data.shape[-1] == 1:
            data = np.repeat(data, 3, axis=-1)
    return data


def apply_gamma(image_uint8: ArrayLike, gamma: float) -> ArrayLike:
    """Apply gamma correction to an 8-bit image, preserving dtype."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    arr = np.asarray(image_uint8).astype(np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0) ** gamma
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def get_instance_color_table() -> ArrayLike:
    """Return the RGBA lookup table for instance mask visualization."""
    return INSTANCE_COLOR_TABLE.copy()
