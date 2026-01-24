import numpy as np

import fastremap

from ..utils.neighbor import kernel_setup, get_neighbors, get_neigh_inds
from .njit import parametrize_contours


def get_contour(labels, affinity_graph, coords=None, neighbors=None, cardinal_only=True):
    """Sort 2D boundaries into cyclic paths."""
    dim = labels.ndim
    steps, inds, idx, fact, sign = kernel_setup(dim)

    if cardinal_only:
        allowed_inds = np.concatenate(inds[1:2])
    else:
        allowed_inds = np.concatenate(inds[1:])

    shape = labels.shape
    coords = np.nonzero(labels) if coords is None else coords
    neighbors = get_neighbors(coords, steps, dim, shape) if neighbors is None else neighbors
    indexes, neigh_inds, ind_matrix = get_neigh_inds(neighbors, coords, shape)

    csum = np.sum(affinity_graph, axis=0)

    step_ok = np.zeros(affinity_graph.shape, bool)

    for s in allowed_inds:
        step_ok[s] = np.logical_and.reduce(
            (
                affinity_graph[s] > 0,
                csum[neigh_inds[s]] < (3**dim - 1),
                neigh_inds[s] > -1,
            )
        )

    labs = labels[coords]
    unique_L = fastremap.unique(labs)

    contours = parametrize_contours(
        steps, np.int32(labs), np.int32(unique_L), neigh_inds, step_ok, csum
    )

    contour_map = np.zeros(shape, dtype=np.int32)
    for contour in contours:
        coords_t = tuple([c[contour] for c in coords])
        contour_map[coords_t] = np.arange(1, len(contour) + 1)

    return contour_map, contours, unique_L
