import numpy as np

from omnipose.core import diam as diam_mod


def test_dist_to_diam_basic():
    dt = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = diam_mod.dist_to_diam(dt, n=2)
    assert out > 0


def test_diameters_with_dt_and_return_length():
    masks = np.zeros((4, 4), dtype=np.int32)
    masks[1:3, 1:3] = 1
    dt = np.ones_like(masks, dtype=np.float32)
    diam, length = diam_mod.diameters(masks, dt=dt, return_length=True)
    assert diam > 0
    assert length > 0


def test_diameters_pill_and_empty():
    masks = np.zeros((4, 4), dtype=np.int32)
    masks[1:3, 1:3] = 1
    dt = np.ones_like(masks, dtype=np.float32)
    r, l = diam_mod.diameters(masks, dt=dt, pill=True)
    assert r > 0
    assert l > 0

    empty = np.zeros((4, 4), dtype=np.int32)
    assert diam_mod.diameters(empty, dt=None) == 0


def test_pill_decomposition_direct():
    r, l = diam_mod.pill_decomposition(10.0, 5.0)
    assert r > 0
    assert l > 0
