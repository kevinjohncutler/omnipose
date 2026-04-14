import numpy as np
import pytest

from omnirefactor.core import affinity
from ocdkit import spatial as contour_mod
from omnirefactor.transforms import augment
from omnirefactor.core import njit as njit_mod
from omnirefactor import utils
from omnirefactor.utils.neighbor import kernel_setup
from ocdkit.morphology import masks_to_outlines


def test_despur_calls_candidate_cleanup_idx(monkeypatch):
    dim = 2
    steps, inds, idx, fact, sign = utils.kernel_setup(dim)
    nsteps = len(steps)

    # Build a tiny 2-pixel configuration that mimics a spur-like boundary case.
    npix = 2
    connect = np.zeros((nsteps, npix), dtype=np.int32)
    cardinal = np.concatenate(inds[1:2])
    ordinal = np.concatenate(inds[2:])

    # Pixel 0 has two cardinal connections (>= dim) to avoid external spur zeroing.
    connect[cardinal[0], 0] = 1
    connect[cardinal[1], 0] = 1

    neigh_inds = np.zeros((nsteps, npix), dtype=np.int32)
    # One cardinal neighbor points to pixel 0 (boundary True), the other to pixel 1 (boundary False).
    neigh_inds[cardinal[0], 0] = 0
    neigh_inds[cardinal[1], 0] = 1
    # Make pixel 1 invalid as a target so it does not zero out pixel 0 during spur cleanup.
    neigh_inds[:, 1] = -1

    indexes = np.arange(npix, dtype=np.int32)
    non_self = np.array(list(set(np.arange(nsteps)) - {idx}))

    called = {"count": 0}

    def stub(candidate, connect, neigh_inds, cardinal, ordinal, dim, boundary, internal):
        called["count"] += 1

    monkeypatch.setattr(affinity, "candidate_cleanup_idx", stub)

    affinity._despur(
        connect,
        neigh_inds,
        indexes,
        steps,
        non_self,
        cardinal,
        ordinal,
        dim,
        clean_bd_connections=True,
        iter_cutoff=2,
        skeletonize=False,
    )

    assert called["count"] > 0


def test_mode_filter_hits_most_frequent_py_func(monkeypatch):
    monkeypatch.setattr(augment, "most_frequent", njit_mod.most_frequent.py_func)
    masks = np.zeros((5, 5), dtype=np.int32)
    masks[2, 2] = 1
    result = augment.mode_filter(masks)
    assert result.shape == masks.shape


def test_get_contour_step_selection():
    labels = np.zeros((3, 3), dtype=np.int32)
    labels[1, 1] = 1
    labels[1, 2] = 1
    coords = np.nonzero(labels)

    dim = labels.ndim
    steps, inds, idx, fact, sign = kernel_setup(dim)
    nsteps = len(steps)
    npix = coords[0].shape[0]

    neighbors = np.zeros((dim, nsteps, npix), dtype=np.int64)
    for d in range(dim):
        neighbors[d, :, 0] = coords[d][0]
        neighbors[d, :, 1] = coords[d][1]

    cardinal = np.concatenate(inds[1:2])
    step_sel = cardinal[0]
    neighbors[0, step_sel, 0] = coords[0][1]
    neighbors[1, step_sel, 0] = coords[1][1]

    affinity_graph = np.zeros((nsteps, npix), dtype=np.int32)
    affinity_graph[step_sel, 0] = 1

    contour_map, contours, unique_L = contour_mod.get_contour(
        labels, affinity_graph, coords=coords, neighbors=neighbors, cardinal_only=True
    )
    assert contour_map.shape == labels.shape
    assert len(contours) > 0
    assert unique_L.size == 1


def test_get_contour_multistep():
    labels = np.zeros((9, 9), dtype=np.int32)
    # S-shaped region to create multiple boundary choices.
    labels[2, 2:7] = 1
    labels[3, 2] = 1
    labels[4, 2:7] = 1
    labels[5, 6] = 1
    labels[6, 2:7] = 1

    coords = np.nonzero(labels)
    dim = labels.ndim
    steps, inds, idx, fact, sign = kernel_setup(dim)
    neighbors = utils.get_neighbors(coords, steps, dim, labels.shape)
    indexes, neigh_inds, _ = utils.get_neigh_inds(neighbors, coords, labels.shape)

    npix = coords[0].shape[0]
    nsteps = len(steps)
    affinity_graph = np.zeros((nsteps, npix), dtype=np.int32)

    # Use the first pixel (top-most in coords) as the contour start.
    start_index = indexes[0]
    allowed_inds = np.concatenate(inds[1:])
    candidate_steps = [s for s in allowed_inds if neigh_inds[s, start_index] != start_index and neigh_inds[s, start_index] > -1]
    step_a, step_b = candidate_steps[:2]
    affinity_graph[step_a, start_index] = 1
    affinity_graph[step_b, start_index] = 1

    # Give all other pixels higher csum so the start index is selected.
    filler_steps = candidate_steps[:3]
    for col in range(npix):
        if col == start_index:
            continue
        for s in filler_steps:
            affinity_graph[s, col] = 1

    contour_map, contours, unique_L = contour_mod.get_contour(
        labels, affinity_graph, coords=coords, neighbors=neighbors, cardinal_only=False
    )
    assert contour_map.shape == labels.shape
    assert len(contours) > 0
    assert unique_L.size == 1

    # Verify that the initial parametrize step sees multiple possible steps.
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
    contour_seed = [neigh_inds[4, start_index]]
    neighbor_inds = neigh_inds[:, start_index]
    step_ok_here = step_ok[:, start_index]
    seen = np.array([i in contour_seed for i in neighbor_inds])
    possible_steps = np.logical_and(step_ok_here, ~seen)
    assert np.sum(possible_steps) > 1

    outlines = masks_to_outlines(labels)
    assert outlines.shape == labels.shape
    assert outlines.dtype == bool
    assert outlines.sum() > 0

    outlines_3d = masks_to_outlines(np.stack([labels, labels]), omni=True)
    assert outlines_3d.shape == (2,) + labels.shape
    assert outlines_3d.dtype == bool

    with pytest.raises(ValueError):
        masks_to_outlines(labels[0])


def test_despur_hits_candidate_cleanup_py_func_invalid_target(monkeypatch):
    monkeypatch.setattr(
        affinity, "candidate_cleanup_idx", njit_mod.candidate_cleanup_idx.py_func
    )
    dim = 2
    steps, inds, idx, fact, sign = utils.kernel_setup(dim)
    nsteps = len(steps)
    npix = 2

    connect = np.zeros((nsteps, npix), dtype=np.int32)
    cardinal = np.concatenate(inds[1:2])
    ordinal = np.concatenate(inds[2:])
    connect[cardinal[0], 0] = 1
    connect[cardinal[1], 0] = 1

    neigh_inds = np.zeros((nsteps, npix), dtype=np.int32)
    neigh_inds[cardinal[0], 0] = -1
    neigh_inds[cardinal[1], 0] = 1
    neigh_inds[:, 1] = -1

    indexes = np.arange(npix, dtype=np.int32)
    non_self = np.array(list(set(np.arange(nsteps)) - {idx}))

    affinity._despur(
        connect,
        neigh_inds,
        indexes,
        steps,
        non_self,
        cardinal,
        ordinal,
        dim,
        clean_bd_connections=True,
        iter_cutoff=2,
        skeletonize=False,
    )


def test_despur_hits_candidate_cleanup_py_func_full_path(monkeypatch):
    monkeypatch.setattr(
        affinity, "candidate_cleanup_idx", njit_mod.candidate_cleanup_idx.py_func
    )
    dim = 2
    steps, inds, idx, fact, sign = utils.kernel_setup(dim)
    nsteps = len(steps)
    npix = 3

    connect = np.zeros((nsteps, npix), dtype=np.int32)
    cardinal = np.concatenate(inds[1:2])
    ordinal = np.concatenate(inds[2:])

    c0, c1 = cardinal[0], cardinal[1]
    o0 = ordinal[0]

    connect[c0, 0] = 1
    connect[c1, 0] = 1
    connect[o0, 1] = 1

    neigh_inds = np.zeros((nsteps, npix), dtype=np.int32)
    neigh_inds[c0, 0] = 1
    neigh_inds[c1, 0] = 2
    neigh_inds[o0, 1] = 2

    indexes = np.arange(npix, dtype=np.int32)
    non_self = np.array(list(set(np.arange(nsteps)) - {idx}))

    affinity._despur(
        connect,
        neigh_inds,
        indexes,
        steps,
        non_self,
        cardinal,
        ordinal,
        dim,
        clean_bd_connections=True,
        iter_cutoff=2,
        skeletonize=False,
    )
