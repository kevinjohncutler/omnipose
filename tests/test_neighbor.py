import numpy as np

from omnipose.utils import neighbor


def test_precompute_valid_mask_center_and_edge():
    steps, inds, idx, fact, sign = neighbor.kernel_setup(2)
    valid = neighbor.precompute_valid_mask((3, 3), steps)
    assert valid.shape == (steps.shape[0], 1, 3, 3)
    assert valid[idx].all()
    step_idx = np.where((steps == np.array([1, 0])).all(axis=1))[0][0]
    assert not valid[step_idx, 0, -1, :].any()


def test_get_neighbors_center_step_matches_coords():
    shape = (4, 4)
    steps, inds, idx, fact, sign = neighbor.kernel_setup(2)
    coords = np.array([0, 3, 1]), np.array([0, 3, 2])
    neighbors = neighbor.get_neighbors(coords, steps, dim=2, shape=shape)
    assert neighbors.shape == (2, steps.shape[0], coords[0].shape[0])
    assert np.array_equal(neighbors[0, idx], coords[0])
    assert np.array_equal(neighbors[1, idx], coords[1])
    assert neighbors.min() >= 0
    assert neighbors[0].max() <= shape[0] - 1
    assert neighbors[1].max() <= shape[1] - 1


def test_get_neigh_inds_background_reflect():
    shape = (4, 4)
    steps, inds, idx, fact, sign = neighbor.kernel_setup(2)
    coords = np.array([1, 2]), np.array([1, 2])
    neighbors = neighbor.get_neighbors(coords, steps, dim=2, shape=shape)

    indexes, neigh_inds, ind_matrix = neighbor.get_neigh_inds(neighbors, coords, shape)
    assert neigh_inds.shape[1] == coords[0].shape[0]
    assert (neigh_inds == -1).any()

    indexes_reflect, neigh_inds_reflect, _ = neighbor.get_neigh_inds(
        neighbors, coords, shape, background_reflect=True
    )
    assert not (neigh_inds_reflect == -1).any()


def test_steps_and_supporting_inds():
    k = neighbor.kernel_setup(2)
    assert len(k.steps) == 9
    assert k.inds[0].size == 1
    assert k.idx == k.inds[0][0]

    pairs = neighbor.get_supporting_inds(k.steps)
    step_idx = np.where((k.steps == np.array([1, 0])).all(axis=1))[0][0]
    assert step_idx in pairs
    assert len(pairs[step_idx]) > 0
