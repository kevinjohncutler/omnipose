from .imports import *


def fill_holes_and_remove_small_masks(masks, min_size=None, max_size=None, hole_size=3, dim=2):
    """Fill holes in masks (2D/3D) and discard masks smaller than min_size (2D).

    Vectorized version — no per-label loop.
    """
    if min_size is None:
        min_size = 3 ** dim

    masks = masks.copy()

    # Vectorized size filtering
    if min_size > 0 or max_size is not None:
        label_ids, counts = np.unique(masks[masks > 0], return_counts=True)
        remove = np.zeros(len(label_ids), dtype=bool)
        if min_size > 0:
            remove |= counts < min_size
        if max_size is not None:
            remove |= counts > max_size
        if remove.any():
            masks[np.isin(masks, label_ids[remove])] = 0

    # Vectorized hole filling
    fill_holes = hole_size > 0
    if fill_holes and masks.max() > 0:
        fg = masks > 0
        filled_fg = binary_fill_holes(fg)
        holes = filled_fg & ~fg
        if holes.any():
            from scipy.ndimage import distance_transform_edt
            _, nearest_idx = distance_transform_edt(~fg, return_distances=True, return_indices=True)
            masks[holes] = masks[tuple(nearest_idx[:, holes])]

    # Renumber consecutively
    if masks.max() > 0:
        unique_ids = np.unique(masks)
        remap = np.zeros(masks.max() + 1, dtype=masks.dtype)
        for new_id, old_id in enumerate(unique_ids):
            remap[old_id] = new_id
        masks = remap[masks]

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
