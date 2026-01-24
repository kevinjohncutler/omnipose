import numpy as np
from skimage.segmentation import expand_labels

from omnirefactor.core.contour import get_contour
from omnirefactor.core import affinity as affinity_mod
from omnirefactor.core.affinity import (
    affinity_to_boundary,
    affinity_to_masks,
    masks_to_affinity,
)
from omnirefactor.utils.neighbor import kernel_setup
from omnirefactor import utils


def _make_labels():
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[2:5, 2:5] = 1
    return labels


def test_get_contour_cardinal_and_full():
    labels = _make_labels()
    coords = np.nonzero(labels)
    dim = labels.ndim
    steps, inds, idx, fact, sign = kernel_setup(dim)
    affinity_graph = masks_to_affinity(labels, coords, steps, inds, idx, fact, sign, dim)

    contour_map, contours, unique_L = get_contour(labels, affinity_graph, coords=coords, cardinal_only=True)
    assert contour_map.shape == labels.shape
    assert len(contours) > 0
    assert unique_L.size == 1
    assert contour_map.max() > 0

    contour_map2, contours2, unique_L2 = get_contour(labels, affinity_graph, coords=coords, cardinal_only=False)
    assert contour_map2.shape == labels.shape
    assert len(contours2) > 0
    assert unique_L2.size == 1
    assert contour_map2.max() > 0


def test_get_contour_from_augmented_affinity():
    labels = _make_labels()
    dim = labels.ndim
    steps, inds, idx, fact, sign = kernel_setup(dim)
    coords = np.nonzero(labels)
    neighbors = utils.get_neighbors(coords, steps, dim, labels.shape)
    affinity_graph = masks_to_affinity(labels, coords, steps, inds, idx, fact, sign, dim, neighbors)

    augmented = np.concatenate([neighbors, affinity_graph[None, ...]], axis=0)
    aa = augmented
    neighbors = aa[:dim]
    affinity_graph = aa[dim]
    idx = affinity_graph.shape[0] // 2
    coords = tuple(neighbors[:, idx])

    bounds = affinity_to_boundary(labels, affinity_graph, coords)
    contour_map, contour_list, unique_L = get_contour(labels, affinity_graph, coords, cardinal_only=True)

    assert bounds.shape == labels.shape
    assert contour_map.shape == labels.shape
    assert len(contour_list) > 0


def test_masks_to_affinity_with_links(monkeypatch):
    monkeypatch.setattr(
        affinity_mod, "_get_link_matrix", affinity_mod._get_link_matrix.py_func
    )
    labels = np.zeros((9, 9), dtype=np.int32)
    labels[2:5, 2:4] = 1
    labels[2:5, 4:6] = 2
    labels[6:8, 6:8] = 3

    coords = np.nonzero(labels)
    dim = labels.ndim
    steps, inds, idx, fact, sign = kernel_setup(dim)
    neighbors = utils.get_neighbors(coords, steps, dim, labels.shape)

    affinity_no = masks_to_affinity(
        labels,
        coords,
        steps,
        inds,
        idx,
        fact,
        sign,
        dim,
        neighbors=neighbors,
        links=None,
    )
    affinity_yes = masks_to_affinity(
        labels,
        coords,
        steps,
        inds,
        idx,
        fact,
        sign,
        dim,
        neighbors=neighbors,
        links=[(1, 2)],
    )

    piece_masks = labels[tuple(neighbors)]
    adj12 = (piece_masks[idx] == 1) & (piece_masks == 2)
    adj21 = (piece_masks[idx] == 2) & (piece_masks == 1)
    adj = adj12 | adj21

    assert adj.any(), "expected adjacent 1<->2 label pairs"
    assert not affinity_no[adj].any()
    assert affinity_yes[adj].any()


def test_affinity_to_masks_exclude_interior_noncardinal(monkeypatch):
    monkeypatch.setattr(affinity_mod.ncolor, "expand_labels", expand_labels, raising=False)
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[4:6, 4:6] = 2

    coords = np.nonzero(labels)
    dim = labels.ndim
    steps, inds, idx, fact, sign = kernel_setup(dim)
    neighbors = utils.get_neighbors(coords, steps, dim, labels.shape)
    affinity_graph = masks_to_affinity(
        labels,
        coords,
        steps,
        inds,
        idx,
        fact,
        sign,
        dim,
        neighbors=neighbors,
    )

    indexes, neigh_inds, _ = utils.get_neigh_inds(neighbors, coords, labels.shape)
    labels_out, edge_list, coords_out, px_inds = affinity_to_masks(
        affinity_graph,
        neigh_inds,
        labels > 0,
        coords,
        cardinal=False,
        exclude_interior=True,
        return_edges=True,
        verbose=True,
    )

    assert labels_out.shape == labels.shape
    assert edge_list.shape[1] == 2
    assert coords_out.ndim == 2
    assert px_inds.ndim == 1
