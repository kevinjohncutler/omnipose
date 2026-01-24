from .imports import *

# used in affinity notebook
def masks_to_outlines(masks, omni=False, mode="inner", connectivity=None):
    """Get outlines of masks as a 0-1 array."""
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError("masks_to_outlines takes 2D or 3D array, not %dD array" % masks.ndim)
    outlines = np.zeros(masks.shape, bool)

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i], omni=omni, mode=mode, connectivity=connectivity)
        return outlines

    if omni and SKIMAGE_ENABLED:
        if connectivity is None:
            connectivity = masks.ndim
        outlines = find_boundaries(masks, mode=mode, connectivity=connectivity)
        return outlines

    slices = find_objects(masks.astype(int))
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            mask = (masks[sr, sc] == (i + 1)).astype(np.uint8)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T
            vr, vc = pvr + sr.start, pvc + sc.start
            outlines[vr, vc] = 1
    return outlines
