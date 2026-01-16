import numpy as np
from numba import njit

import fastremap
import ncolor
from skimage.segmentation import expand_labels


@njit
def most_frequent(neighbor_masks):
    return np.array([np.bincount(row).argmax() for row in neighbor_masks.T])


@njit('(int64[:,:], int32[:], int32[:], int64[:], int64[:,:], float64[:,:], boolean[:,:])', nogil=True)
def parametrize(steps, labs, unique_L, inds, ind_shift, values, step_ok):
    """Parametrize 2D boundaries."""
    sign = np.sum(np.abs(steps), axis=1)
    cardinal_mask = sign > 1  # limit to cardinal steps for traversing
    contours = []
    for l in unique_L:
        indices = np.argwhere(labs == l).flatten()  # boundary indices for this label
        index = indices[0]

        closed = 0
        contour = []
        n_iter = 0

        while not closed and n_iter < len(indices) + 1:
            contour.append(inds[index])

            neighbor_inds = ind_shift[:, index]
            step_ok_here = step_ok[:, index]
            seen = np.array([i in contour for i in neighbor_inds])
            step_mask = (seen + cardinal_mask + ~step_ok_here) > 0

            vals = values[:, index]
            vals[step_mask] = np.inf

            if np.sum(step_mask) < len(step_mask):
                select = np.argmin(vals)
                neighbor_idx = neighbor_inds[select]
                w = np.argwhere(inds[indices] == neighbor_idx)[0][0]
                index = indices[w]
                n_iter += 1
            else:
                closed = True
                contours.append(contour)

    return contours


@njit
def parametrize_contours(steps, labs, unique_L, neigh_inds, step_ok, csum):
    """Helper function to sort 2D contours into cyclic paths. See get_contour()."""
    sign = np.sum(np.abs(steps), axis=1)
    contours = []
    s0 = 4
    for l in unique_L:
        sel = labs == l
        indices = np.argwhere(sel).flatten()
        index = indices[np.argmin(csum[sel])]

        closed = 0
        contour = []
        n_iter = 0

        while not closed and n_iter < len(indices) + 1:
            contour.append(neigh_inds[s0, index])

            neighbor_inds = neigh_inds[:, index]
            step_ok_here = step_ok[:, index]
            seen = np.array([i in contour for i in neighbor_inds])
            possible_steps = np.logical_and(step_ok_here, ~seen)

            if np.sum(possible_steps) > 0:
                possible_step_indices = np.nonzero(possible_steps)[0]

                if len(possible_step_indices) == 1:
                    select = possible_step_indices[0]
                else:
                    consider_steps = steps[possible_step_indices]
                    best = np.argmin(np.array([np.sum(s * steps[3]) for s in consider_steps]))
                    select = possible_step_indices[best]

                neighbor_idx = neighbor_inds[select]
                index = neighbor_idx
                n_iter += 1
            else:
                closed = True
                contours.append(contour)

    return contours


@njit
def candidate_cleanup_idx(idx, connect, neigh_inds, cardinal, ordinal, dim, boundary, internal):
    """
    Jitted helper for per-candidate boundary cleanup.
    This function is meant to mimic the inner loop in the original _despur.
    All indices (e.g. from 'cardinal' and 'ordinal') are assumed to be 1D arrays of integers.
    It updates connect in place.
    """
    n_dirs = connect.shape[0]
    for i in range(cardinal.shape[0]):
        d = cardinal[i]
        if connect[d, idx] != 0:
            target = neigh_inds[d, idx]
            if target < 0:
                continue
            for j in range(ordinal.shape[0]):
                o = ordinal[j]
                if target < 0:
                    continue
                t = neigh_inds[o, target]
                found = False
                for k in range(cardinal.shape[0]):
                    d2 = cardinal[k]
                    if neigh_inds[d2, idx] == t:
                        found = True
                        break
                if found:
                    c_val = 0
                    if (connect[o, target] != 0 and connect[d, idx] != 0) and (t > -1 and target > -1):
                        c_val = 1
                    connect[o, target] = c_val
                    sym_index = -(o + 1)
                    if t > -1:
                        connect[sym_index, t] = c_val
    return


@njit()
def linker_label_to_links(maski, linker_label_list):
    linker_mask = np.zeros(maski.shape, bool)
    for l in linker_label_list:
        mask = maski == l
        linker_mask[mask] = 1

    link_masks = ncolor.format_labels(maski, clean=True)
    linker_labels = link_masks.copy()
    unlink_masks = link_masks.copy()
    linker_labels[linker_mask == 0] = 0
    unlink_masks[linker_mask] = 0

    dic = fastremap.inverse_component_map(expand_labels(linker_labels, 1), unlink_masks)
    links = {(x, z) for x, y in dic.items() if x != 0 for z in y if z != 0}
    return links
