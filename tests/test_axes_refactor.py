import numpy as np

from omnirefactor.transforms import axes


def test_move_axis_first_last():
    x = np.zeros((2, 3, 4))
    out_first = axes.move_axis(x, axis=-1, pos="first")
    out_last = axes.move_axis(x, axis=0, pos="last")
    assert out_first.shape == (4, 2, 3)
    assert out_last.shape == (3, 4, 2)


def test_move_axis_pos_variants():
    x = np.zeros((2, 3, 4, 5))
    out_first = axes.move_axis(x, axis=2, pos=0)
    out_last = axes.move_axis(x, axis=1, pos=-1)
    assert out_first.shape == (4, 2, 3, 5)
    assert out_last.shape == (2, 4, 5, 3)


def test_move_min_dim_behavior():
    x = np.zeros((4, 5, 2))
    out = axes.move_min_dim(x)
    assert out.shape == (4, 5, 2)

    x_force = np.zeros((10, 2, 10))
    out_force = axes.move_min_dim(x_force, force=True)
    assert out_force.shape == (10, 10, 2)


def test_update_axis_with_squeeze():
    assert axes.update_axis(2, np.array([2]), 3) is None
    assert axes.update_axis(2, np.array([0]), 3) == 1
