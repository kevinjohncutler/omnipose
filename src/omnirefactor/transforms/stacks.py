import numpy as np
import fastremap
from scipy.ndimage import binary_erosion

from .imports import rescale, safe_divide


def make_unique(masks):
    """Relabel stack of label matrices such that there is no repeated label across slices."""
    masks = masks.copy().astype(np.uint32)
    T = range(len(masks))
    offset = 0
    for t in T:
        fastremap.renumber(masks[t], in_place=True)
        masks[t][masks[t] > 0] += offset
        offset = masks[t].max()
    return masks


def normalize_stack(vol, mask, bg=0.5, bright_foreground=None,
                    subtractive=False, iterations=1, equalize_foreground=1, quantiles=[0.01, 0.99]):
    """
    Adjust image stacks so that background is
    (1) consistent in brightness and
    (2) brought to an even average via semantic gamma normalization.
    """
    vol = vol.copy()
    # binarize background mask, recede from foreground, slice-wise to not erode in time
    kwargs = {'iterations': iterations} if iterations > 1 else {}
    bg_mask = [binary_erosion(m == 0, **kwargs) for m in mask]
    # find mean background for each slice
    bg_real = [np.nanmean(v[m]) for v, m in zip(vol, bg_mask)]

    # automatically determine if foreground objects are bright or dark
    if bright_foreground is None:
        bright_foreground = np.mean(vol[bg_mask]) < np.mean(vol[mask > 0])

    bg_min = np.min(bg_real)  # get the minimum one, want to normalize by lowest one

    # normalize wrt background
    if subtractive:
        vol = np.stack([safe_divide(v - bg_r, bg_min) for v, bg_r in zip(vol, bg_real)])
    else:
        vol = np.stack([v * safe_divide(bg_min, bg_r) for v, bg_r in zip(vol, bg_real)])

    # equalize foreground signal
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

    # now can gamma normalize
    vol = np.stack([v ** (np.log(bg) / np.log(np.mean(v[bg_m]))) for v, bg_m in zip(vol, bg_mask)])
    return vol
