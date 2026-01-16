from .imports import *

# this is probably really slow
# guessing a cellpose function
def distance_to_boundary(masks):
    """Get distance to boundary of mask pixels."""
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError("distance_to_boundary takes 2D or 3D array, not %dD array" % masks.ndim)
    dist_to_bound = np.zeros(masks.shape, np.float64)

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            dist_to_bound[i] = distance_to_boundary(masks[i])
        return dist_to_bound
    slices = find_objects(masks)
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            mask = (masks[sr, sc] == (i + 1)).astype(np.uint8)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T
            ypix, xpix = np.nonzero(mask)
            min_dist = ((ypix[:, np.newaxis] - pvr) ** 2 +
                        (xpix[:, np.newaxis] - pvc) ** 2).min(axis=1)
            dist_to_bound[ypix + sr.start, xpix + sc.start] = min_dist
    return dist_to_bound

