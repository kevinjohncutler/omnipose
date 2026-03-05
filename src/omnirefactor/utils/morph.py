from .imports import *


# This function takes a few milliseconds for a typical image 
def get_edge_masks(labels,dists):
    """Finds and returns masks that are largely cut off by the edge of the image.
    
    This function loops over all masks touching the image boundary and compares the 
    maximum value of the distance field along the boundary to the top quartile of distance
    within the mask. Regions whose edges just skim the image edge will not be classified as 
    an "edge mask" by this criteria, whereas masks cut off in their center (where distance is high)
    will be returned as part of this output. 
    
    Parameters
    ----------
    labels: ND array, int
        label matrix
        
    dists: ND array, float
        distance field (calculated with reflection padding of labels)
    
    Returns
    --------------
    clean_labels: ND array, int
        label matrix of all cells qualifying as 'edge masks'
    
    """
    border_mask = np.zeros(labels.shape, dtype=bool)
    border_mask = binary_dilation(border_mask, border_value=1, iterations=1)
    clean_labels = np.zeros_like(labels)
    
    for cell_ID in fastremap.unique(labels[border_mask])[1:]:
        mask = labels==cell_ID 
        max_dist = np.max(dists[np.logical_and(mask, border_mask)])
        # mean_dist = np.mean(dists[mask])
        dist_thresh = np.percentile(dists[mask],75) 
        # sort of a way to say the skeleton isn't touching the boundary
        # top 25%

        if max_dist>=dist_thresh: # we only want to keep cells whose distance at the boundary is not too small
            clean_labels[mask] = cell_ID
            
    return clean_labels


#TODO: vectorize this function
def fill_holes_and_remove_small_masks(masks, min_size=None, max_size=None, hole_size=3, dim=2):
    """Fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)."""
    if min_size is None:
        min_size = 3 ** dim

    masks = ncolor.format_labels(masks, min_area=min_size)
    fill_holes = hole_size > 0

    slices = find_objects(masks)
    j = 0
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            npix = msk.sum()

            too_small = npix < min_size
            too_big = False if max_size is None else npix > max_size

            if (min_size > 0) and (too_small or too_big):
                masks[slc][msk] = 0
            elif fill_holes:
                hsz = np.count_nonzero(msk) * hole_size / 100
                if SKIMAGE_ENABLED:
                    pad = 1
                    unpad = tuple([slice(pad, -pad)] * dim)
                    # Keep legacy behavior (< hsz) while using the new max_size API (<= max_size).
                    hsz_legacy = np.nextafter(float(hsz), -np.inf)
                    padmsk = remove_small_holes(np.pad(msk, pad, mode='constant'), max_size=hsz_legacy)
                    msk = padmsk[unpad]
                else:
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = (j + 1)
                j += 1
    return masks

# Omnipose version of remove_edge_masks, need to merge (this one is more flexible)
def clean_boundary(labels, boundary_thickness=3, area_thresh=30, cutoff=0.5):
    """Delete boundary masks below a given size threshold within a certain distance from the boundary. 
    
    Parameters
    ----------
    boundary_thickness: int
        labels within a stripe of this thickness along the boundary will be candidates for removal. 
        
    area_thresh: int
        labels with area below this value will be removed. 
        
    cutoff: float
        Fraction from 0 to 1 of the overlap with the boundary before the mask is removed. Default 0.5. 
        Set cutoff to 0 and are_thresh to np.inf if you want any mask touching the boundary to be removed. 
    
    Returns
    --------------
    label matrix with small edge labels removed. 
    
    """
    border_mask = np.zeros(labels.shape, dtype=bool)
    border_mask = binary_dilation(border_mask, border_value=1, iterations=boundary_thickness)
    clean_labels = np.copy(labels)

    for cell_ID in fastremap.unique(labels[border_mask])[1:]:
        mask = labels==cell_ID 
        area = np.count_nonzero(mask)
        overlap = np.count_nonzero(np.logical_and(mask, border_mask))
        if overlap > 0 and area<area_thresh and overlap/area >= cutoff: #only remove cells that are X% or more edge px
            clean_labels[mask] = 0

    return clean_labels
