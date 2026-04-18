"""Verify omnirefactor.transforms.axes re-exports from ocdkit."""

import numpy as np

from omnirefactor.transforms import axes


def test_move_axis_reexport():
    from ocdkit.array import move_axis
    assert axes.move_axis is move_axis


def test_move_min_dim_reexport():
    from ocdkit.array import move_min_dim
    assert axes.move_min_dim is move_min_dim


def test_update_axis_local():
    """update_axis stays in omnirefactor (not migrated)."""
    assert axes.update_axis(2, np.array([2]), 3) is None
    assert axes.update_axis(2, np.array([0]), 3) == 1
