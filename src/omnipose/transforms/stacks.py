import numpy as np
import fastremap


def make_unique(masks):
    """Relabel stack of label matrices such that there is no repeated label across slices."""
    masks = masks.copy().astype(np.uint32)
    offset = 0
    for t in range(len(masks)):
        fastremap.renumber(masks[t], in_place=True)
        masks[t][masks[t] > 0] += offset
        offset = masks[t].max()
    return masks
