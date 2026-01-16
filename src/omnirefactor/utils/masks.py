from .imports import *
from ..measure.dist import distance_to_boundary


# masks to edges and remove_edge_masks maybe should be replaced by 
# omnipose clean_boundary 

# many functions in this file might be vestigial from cellpose

def masks_to_edges(masks, threshold=1.0):
    """Get edges of masks as a 0-1 array."""
    dist_to_bound = distance_to_boundary(masks)
    edges = (dist_to_bound < threshold) * (masks > 0)
    return edges


def remove_edge_masks(masks, change_index=True):
    """Remove masks with pixels on edge of image."""
    slices = find_objects(masks.astype(int))
    for i, si in enumerate(slices):
        remove = False
        if si is not None:
            for d, sid in enumerate(si):
                if sid.start == 0 or sid.stop == masks.shape[d]:
                    remove = True
                    break
            if remove:
                masks[si][masks[si] == i + 1] = 0
    shape = masks.shape
    if change_index:
        _, masks = np.unique(masks, return_inverse=True)
        masks = np.reshape(masks, shape).astype(np.int32)

    return masks


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


def outlines_list(masks):
    """Get outlines of masks as a list to loop over for plotting."""
    outl_stack = []
    for n in np.unique(masks)[1:]:
        mn = masks == n
        if mn.sum() > 0:
            contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_NONE)
            contours = contours[-2]
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix) > 4:
                outl_stack.append(pix)
            else:
                outl_stack.append(np.zeros((0, 2)))
    return outl_stack


def get_perimeter(points):
    """Perimeter of points - npoints x ndim."""
    if points.shape[0] > 4:
        points = np.append(points, points[:1], axis=0)
        return ((np.diff(points, axis=0) ** 2).sum(axis=1) ** 0.5).sum()
    return 0


def get_mask_compactness(masks):
    perimeters = get_mask_perimeters(masks)
    npoints = np.unique(masks, return_counts=True)[1][1:]
    areas = npoints
    compactness = 4 * np.pi * areas / perimeters ** 2
    compactness[perimeters == 0] = 0
    compactness[compactness > 1.0] = 1.0
    return compactness


def get_mask_perimeters(masks):
    """Get perimeters of masks."""
    perimeters = np.zeros(masks.max())
    for n in range(masks.max()):
        mn = masks == (n + 1)
        if mn.sum() > 0:
            contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_NONE)[-2]
            perimeters[n] = np.array([get_perimeter(c.astype(int).squeeze()) for c in contours]).sum()
    return perimeters


def circleMask(d0):
    """Create array with indices which are the radius of that x,y point."""
    dx = np.tile(np.arange(-d0[1], d0[1] + 1), (2 * d0[0] + 1, 1))
    dy = np.tile(np.arange(-d0[0], d0[0] + 1), (2 * d0[1] + 1, 1))
    dy = dy.transpose()
    rs = (dy ** 2 + dx ** 2) ** 0.5
    return rs, dx, dy


def get_mask_stats(masks_true):
    mask_perimeters = get_mask_perimeters(masks_true)

    rs, dy, dx = circleMask(np.array([100, 100]))
    rsort = np.sort(rs.flatten())

    npoints = np.unique(masks_true, return_counts=True)[1][1:]
    areas = npoints - mask_perimeters / 2 - 1

    compactness = np.zeros(masks_true.max())
    convexity = np.zeros(masks_true.max())
    solidity = np.zeros(masks_true.max())
    convex_perimeters = np.zeros(masks_true.max())
    convex_areas = np.zeros(masks_true.max())
    for ic in range(masks_true.max()):
        points = np.array(np.nonzero(masks_true == (ic + 1))).T
        if len(points) > 15 and mask_perimeters[ic] > 0:
            med = np.median(points, axis=0)
            r2 = ((points - med) ** 2).sum(axis=1) ** 0.5
            compactness[ic] = (rsort[:r2.size].mean() + 1e-10) / r2.mean()
            try:
                hull = ConvexHull(points)
                convex_perimeters[ic] = hull.area
                convex_areas[ic] = hull.volume
            except Exception:
                convex_perimeters[ic] = 0

    convexity[mask_perimeters > 0.0] = (convex_perimeters[mask_perimeters > 0.0] /
                                        mask_perimeters[mask_perimeters > 0.0])
    solidity[convex_areas > 0.0] = (areas[convex_areas > 0.0] /
                                    convex_areas[convex_areas > 0.0])
    convexity = np.clip(convexity, 0.0, 1.0)
    solidity = np.clip(solidity, 0.0, 1.0)
    compactness = np.clip(compactness, 0.0, 1.0)
    return convexity, solidity, compactness

