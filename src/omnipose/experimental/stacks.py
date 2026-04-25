"""Experimental: stack normalization for time-series/volume data."""

import numpy as np
from scipy.ndimage import binary_erosion

from ..transforms.imports import rescale, safe_divide


def normalize_stack(vol, mask, bg=0.5, bright_foreground=None,
                    subtractive=False, iterations=1, equalize_foreground=1, quantiles=[0.01, 0.99]):
    """Adjust image stacks so background is
    (1) consistent in brightness and
    (2) brought to an even average via semantic gamma normalization.
    """
    vol = vol.copy()
    kwargs = {'iterations': iterations} if iterations > 1 else {}
    bg_mask = [binary_erosion(m == 0, **kwargs) for m in mask]
    bg_real = [np.nanmean(v[m]) for v, m in zip(vol, bg_mask)]

    if bright_foreground is None:
        bright_foreground = np.mean(vol[bg_mask]) < np.mean(vol[mask > 0])

    bg_min = np.min(bg_real)

    if subtractive:
        vol = np.stack([safe_divide(v - bg_r, bg_min) for v, bg_r in zip(vol, bg_real)])
    else:
        vol = np.stack([v * safe_divide(bg_min, bg_r) for v, bg_r in zip(vol, bg_real)])

    if equalize_foreground:
        q1, q2 = quantiles
        if bright_foreground:
            fg_real = [np.percentile(v[m > 0], 99.99) for v, m in zip(vol, mask)]
            floor = np.percentile(vol[bg_mask], 0.01)
            vol = [rescale(v, ceiling=f, floor=floor) for v, f in zip(vol, fg_real)]
        else:
            fg_real = [np.quantile(v[m > 0], q1) for v, m in zip(vol, mask)]
            ceiling = np.quantile(vol, q2, axis=(-2, -1))
            vol = [np.interp(v, (f, c), (0, 1)) for v, f, c in zip(vol, fg_real, ceiling)]

    vol = np.stack(vol)
    vol = np.stack([v ** (np.log(bg) / np.log(np.mean(v[bg_m]))) for v, bg_m in zip(vol, bg_mask)])
    return vol
