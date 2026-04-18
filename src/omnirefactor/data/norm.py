"""Per-image per-channel normalization parameter computation and application.

Paired helpers for lazy/disk-backed training datasets:

- :func:`compute_norm_params` reads each image file once, computes percentiles,
  then discards the array. Returns ~2 floats per channel per image.
- :func:`apply_norm_params` applies stored params to an in-memory (C, H, W)
  array during DataLoader iteration.
"""

import numpy as np

from ..io import imread
from ..transforms.shape import reshape
from ..transforms.axes import move_min_dim


def compute_norm_params(image_paths, channel_axis=0, channels=None, normalize=True, dim=2, omni=False):
    """Single-pass precomputation of per-image per-channel normalization parameters.

    Reads each file, computes percentiles, then discards the array.
    Returns a list of ``[(lo_c0, hi_c0), (lo_c1, hi_c1), ...]`` per image.
    Total memory: ~2 floats per channel per image (negligible).

    ``channel_axis=None`` uses the move_min_dim heuristic: smallest dimension is
    treated as channels and moved to axis 0.
    """
    norm_params = []
    for path in image_paths:
        img = imread(path).astype(np.float32)
        if channels is not None:
            _ca = channel_axis if channel_axis is not None else 0
            img = reshape(img, channels=channels, chan_first=True, channel_axis=_ca)
        elif channel_axis is not None:
            img = np.moveaxis(img, channel_axis, 0)
        else:
            if img.ndim > dim:
                img = move_min_dim(img)          # moves min-dim to last
                img = np.moveaxis(img, -1, 0)    # then bring to front
        if img.ndim == dim:
            img = img[np.newaxis]                # add channel dim for 2D images
        per_channel = []
        for c in range(img.shape[0]):
            ch = img[c]
            lo = float(np.percentile(ch, 0.01))
            hi = float(np.percentile(ch, 99.99))
            per_channel.append((lo, hi))
        norm_params.append(per_channel)
    return norm_params


def apply_norm_params(img, params):
    """Apply stored per-channel ``(lo, hi)`` normalization params to ``(C, H, W)`` float32 array."""
    img = img.astype(np.float32)
    for c, (lo, hi) in enumerate(params):
        if hi - lo > 1e-3:
            img[c] = np.clip((img[c] - lo) / (hi - lo), 0, 1)
    return img
