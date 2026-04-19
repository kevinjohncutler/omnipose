"""Experimental: morphology helpers not in active use."""

import numpy as np
import fastremap
from scipy.ndimage import binary_dilation


def get_edge_masks(labels, dists):
    """Find masks largely cut off by the image boundary.

    Loops over masks touching the image boundary and compares the maximum
    boundary distance-field value to the in-mask 75th percentile. Regions
    whose edges just skim the image edge are not flagged; masks cut off
    in their center (where distance is high) are returned.
    """
    border_mask = np.zeros(labels.shape, dtype=bool)
    border_mask = binary_dilation(border_mask, border_value=1, iterations=1)
    clean_labels = np.zeros_like(labels)

    for cell_ID in fastremap.unique(labels[border_mask])[1:]:
        mask = labels == cell_ID
        max_dist = np.max(dists[np.logical_and(mask, border_mask)])
        dist_thresh = np.percentile(dists[mask], 75)
        if max_dist >= dist_thresh:
            clean_labels[mask] = cell_ID

    return clean_labels
