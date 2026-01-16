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


from sklearn.utils.extmath import cartesian

def get_spruepoints(bw):
    d = bw.ndim
    idx = (3**d)//2 # the index of the center pixel is placed here when considering the neighbor kernel 
    neigh = [[-1,0,1] for i in range(d)]
    steps = cartesian(neigh) # all the possible step sequences in ND
    sign = np.sum(np.abs(steps),axis=1) # signature distinguishing each kind of m-face via the number of steps 
    
    hits = np.zeros_like(bw)
    mid = tuple([1]*d) # kernel 3 wide in every axis, so middle is 1
    
    # alt
    substeps = np.array(list(set([tuple(s) for s in steps])-set([(0,)*d]))) # remove zero shift element 
    # substeps = steps.copy()
    for step in substeps:
        oppose = np.array([np.dot(step,s) for s in steps])
        
        sprue = np.zeros([3]*d,dtype=int) # allocate matrix
        sprue[tuple(mid-step)] = 1
        sprue[mid] = 1
        
        miss = np.zeros([3]*d,dtype=int)
        for idx in np.argwhere(np.logical_and(oppose>=0,sign!=0)).flatten():
            c = tuple(steps[idx]+1)
            miss[c] = 1

        
        hitmiss = 2 - 2*miss - sprue
        
        # mahotas hitmiss is far faster than ndimage 
        hm = mh_hitmiss(bw,hitmiss)

        hits = hits+hm
        
    return hits>0



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
                    padmsk = remove_small_holes(np.pad(msk, pad, mode='constant'), area_threshold=hsz)
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

