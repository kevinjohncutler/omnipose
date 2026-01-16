import numpy as np

import fastremap
from skimage.morphology import remove_small_objects

from .. import utils
from .affinity import get_link_matrix
from .flows import masks_to_flows
from .njit import parametrize, parametrize_contours


def links_to_boundary(masks, links):
    """Deprecated. Use masks_to_affinity instead."""
    pad = 1
    d = masks.ndim
    shape = masks.shape
    masks_padded = np.pad(masks, pad)
    coords = np.nonzero(masks_padded)

    steps, inds, idx, fact, sign = utils.kernel_setup(d)
    coords = np.nonzero(masks)
    neighbors = utils.get_neighbors(coords, steps, d, shape)

    piece_masks = masks_padded[tuple(neighbors)]
    is_link = np.zeros(piece_masks.shape, dtype=np.bool_)
    is_link = get_link_matrix(links, piece_masks, np.concatenate(inds), idx, is_link)

    border_mask = np.pad(np.zeros(masks.shape, dtype=bool), pad, constant_values=1)
    isborder = border_mask[tuple(neighbors)]

    isneighbor = np.logical_or.reduce((piece_masks == piece_masks[idx],
                                       is_link,
                                       isborder))
    isboundary = ~isneighbor

    bd0 = np.zeros(masks_padded.shape, dtype=bool)
    masks0 = np.zeros_like(masks_padded)

    s_all = np.concatenate(inds[1:])
    flat_bd = np.any(isboundary[s_all], axis=0)
    bd0[coords] = flat_bd

    neighbor_bd = bd0[tuple(neighbors)]

    sel = inds[1]

    crit1 = np.sum(isneighbor[inds[1]], axis=0) >= 2
    crit2 = np.sum(isboundary[inds[2]], axis=0) >= 1
    crit3 = np.sum(isneighbor[inds[1]], axis=0) == 3
    crit12 = np.logical_and(crit1, crit2)
    flat_bd = np.logical_or(crit12, crit3)
    bd0[coords] = flat_bd

    masks0[coords] = piece_masks[idx] * crit1
    coords = np.nonzero(masks0)
    neighbors = np.array([[coords[k] + s[k] for s in steps] for k in range(d)])

    piece_masks = masks0[tuple(neighbors)]

    is_link = np.zeros(piece_masks.shape, dtype=np.bool_)
    is_link = get_link_matrix(links, piece_masks, np.concatenate(inds), idx, is_link)

    isborder = border_mask[tuple(neighbors)]

    isneighbor = np.logical_or.reduce((piece_masks == piece_masks[idx],
                                       is_link,
                                       isborder))
    isboundary = ~isneighbor

    sel = inds[1]
    neighbor_bd = np.logical_or(bd0[tuple(neighbors)], isborder)
    c1 = np.sum(neighbor_bd[sel], axis=0) >= 2
    bd0[coords] = np.logical_and(c1, bd0[coords])

    isboundary = bd0[tuple(neighbors)]
    bd0[coords] = np.any(isboundary[inds[0]], axis=0)

    unpad = tuple([slice(pad, -pad)] * d)
    return bd0[unpad], masks0[unpad], isboundary, neighbors - pad


def get_boundary(mu, mask, bd=None, affinity_graph=None, contour=False, use_gpu=False, device=None, desprue=False):
    """One way to get boundaries by considering flow dot products. Will be deprecated."""

    d = mu.shape[0]
    pad = 1
    pad_seq = [(0,) * 2] + [(pad,) * 2] * d
    unpad = tuple([slice(pad, -pad)] * d)

    mu_pad = utils.normalize_field(np.pad(mu, pad_seq))
    lab_pad = np.pad(mask, pad)

    steps = utils.get_steps(d)
    steps = np.array(list(set([tuple(s) for s in steps]) - set([(0,) * d])))

    if bd is None:
        bd_pad = np.zeros_like(lab_pad, dtype=bool)

        bd_pad = _get_bd(steps, np.int32(lab_pad), mu_pad, bd_pad)
        s_inter = 0
        while desprue and s_inter < np.sum(bd_pad):
            sp = utils.get_spruepoints(bd_pad)
            desprue = np.any(sp)
            bd_pad[sp] = False

        bd_pad = remove_small_objects(bd_pad, min_size=9)
    else:
        bd_pad = np.pad(bd, pad).astype(bool)

    if contour:
        T, mu_pad = masks_to_flows(lab_pad,
                                   affinity_graph=affinity_graph,
                                   use_gpu=use_gpu,
                                   device=device)[-2:]

        step_ok, ind_shift, cross, dot = _get_bd(steps, lab_pad, mu_pad, bd_pad)
        values = (-dot + cross)

        bd_coords = np.array(np.nonzero(bd_pad))
        bd_inds = np.ravel_multi_index(bd_coords, bd_pad.shape)
        labs = np.take(lab_pad, bd_inds)
        unique_L = fastremap.unique(labs)
        contours = parametrize(steps, np.int32(labs), np.int32(unique_L), bd_inds, ind_shift, values, step_ok)

        contour_map = np.zeros(bd_pad.shape, dtype=np.int32)
        for contour in contours:
            coords_t = np.unravel_index(contour, bd_pad.shape)
            contour_map[coords_t] = np.arange(1, len(contour) + 1)

        return contour_map[unpad], contours

    else:
        return bd_pad[unpad]


# numba does not work yet with this indexing...
# @njit('(int64[:,:], int32[:,:], float64[:,:,:], boolean[:,:])', nogil=True)
def _get_bd(steps, lab_pad, mu_pad, bd_pad):
    """Helper function to get_boundaries."""
    get_bd = np.all(~bd_pad)
    axes = range(mu_pad.shape[0])
    mask_pad = lab_pad > 0
    coord = np.nonzero(mask_pad)
    coords = np.argwhere(mask_pad).T
    A = mu_pad[(Ellipsis,) + coord]
    mag_pad = np.sqrt(np.sum(mu_pad ** 2, axis=0))
    mag_A = mag_pad[coord]

    if not get_bd:
        dot = []
        cross = []
        ind_shift = []
        step_ok = []
    else:
        angles1 = []
        angles2 = []
        cutoff1 = np.pi * (1 / 2.5)
        cutoff2 = np.pi * (3 / 4)

    for s in steps:
        mag_s = np.sqrt(np.sum(s ** 2, axis=0))

        if get_bd:
            neigh_opp = tuple(coords - s[np.newaxis].T)
            B = mu_pad[(Ellipsis,) + neigh_opp]
            mag_B = mag_pad[neigh_opp]
            dot1 = np.sum(np.multiply(A, B), axis=0)

            angle1 = np.arccos(dot1.clip(-1, 1))
            angle1[np.logical_and(mask_pad[coord], mask_pad[neigh_opp] == 0)] = np.pi

            dot2 = utils.safe_divide(np.sum([A[a] * (-s[a]) for a in axes], axis=0), mag_s * mag_A)
            angle2 = np.arccos(dot2.clip(-1, 1))

            angles1.append(angle1 > cutoff1)
            angles2.append(angle2 > cutoff2)

        else:
            neigh_bd = tuple(coords[:, bd_pad[coord]])
            neigh_step = tuple(coords[:, bd_pad[coord]] + s[np.newaxis].T)
            A = mu_pad[(Ellipsis,) + neigh_bd]
            mag_A = mag_pad[neigh_bd]
            B = mu_pad[(Ellipsis,) + neigh_step]
            mag_B = mag_pad[neigh_step]
            dot1 = utils.safe_divide(np.sum(np.multiply(A, B), axis=0), (mag_B * mag_A))
            dot.append(dot1)

            dot2 = utils.safe_divide(np.sum([B[a] * (s[a]) for a in axes], axis=0), mag_s * mag_B)
            cross.append(np.cross(A, s, axisa=0))
            x = np.ravel_multi_index(neigh_step, bd_pad.shape)
            ind_shift.append(x)
            step_ok.append(np.logical_and.reduce((bd_pad[neigh_step],
                                                  lab_pad[neigh_step] == lab_pad[neigh_bd],
                                                  )))

    if get_bd:
        is_bd = np.any([np.logical_and(a1, a2) for a1, a2 in zip(angles1, angles2)], axis=0)
        bd_pad = np.zeros_like(mask_pad)
        bd_pad[coord] = is_bd
        return bd_pad
    else:
        step_ok = np.stack(step_ok)
        ind_shift = np.array(ind_shift)
        cross = np.stack(cross)
        dot = np.stack(dot)

        return step_ok, ind_shift, cross, dot


def get_contour(labels, affinity_graph, coords=None, neighbors=None, cardinal_only=True):
    """Sort 2D boundaries into cyclic paths."""
    dim = labels.ndim
    steps, inds, idx, fact, sign = utils.kernel_setup(dim)

    if cardinal_only:
        allowed_inds = np.concatenate(inds[1:2])
    else:
        allowed_inds = np.concatenate(inds[1:])

    shape = labels.shape
    coords = np.nonzero(labels) if coords is None else coords
    neighbors = utils.get_neighbors(coords, steps, dim, shape) if neighbors is None else neighbors
    indexes, neigh_inds, ind_matrix = utils.get_neigh_inds(neighbors, coords, shape)

    csum = np.sum(affinity_graph, axis=0)

    step_ok = np.zeros(affinity_graph.shape, bool)

    for s in allowed_inds:
        step_ok[s] = np.logical_and.reduce((affinity_graph[s] > 0,
                                            csum[neigh_inds[s]] < (3 ** dim - 1),
                                            neigh_inds[s] > -1
                                            ))

    labs = labels[coords]
    unique_L = fastremap.unique(labs)

    contours = parametrize_contours(steps, np.int32(labs), np.int32(unique_L), neigh_inds, step_ok, csum)

    contour_map = np.zeros(shape, dtype=np.int32)
    for contour in contours:
        coords_t = tuple([c[contour] for c in coords])
        contour_map[coords_t] = np.arange(1, len(contour) + 1)

    return contour_map, contours, unique_L
