import numpy as np

from omnirefactor.utils import inds as inds_mod


def test_ravel_unravel_index_roundtrip():
    shape = (3, 4, 5)
    coords = np.array([1, 2, 3])
    idx = inds_mod.ravel_index(coords, shape)
    out = inds_mod.unravel_index(idx, shape)
    assert out == tuple(coords)


def test_border_indices():
    arr = np.zeros((3, 4), dtype=np.int32)
    border = inds_mod.border_indices(arr.shape)
    assert border.size > 0
    flat = arr.ravel()
    flat[border] = 1
    assert flat.sum() == np.unique(border).size
