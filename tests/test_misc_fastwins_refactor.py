import numpy as np
import torch

from omnirefactor.core import niter as cniter
from omnirefactor.misc import coords as mcoords
from omnirefactor.misc import cube as mcube
from omnirefactor.transforms import vector as tvector


def test_cubestats_basic():
    assert mcube.cubestats(0) == [1]
    assert mcube.cubestats(1) == [2, 1]
    assert mcube.cubestats(2) == [4, 4, 1]


def test_meshgrid_and_flat_coords():
    shape = (2, 3)
    yy, xx = mcoords.meshgrid(shape)
    assert yy.shape == shape
    assert xx.shape == shape
    assert np.array_equal(yy[:, 0], np.array([0, 1]))
    assert np.array_equal(xx[0], np.array([0, 1, 2]))

    flat_y, flat_x = mcoords.generate_flat_coordinates(shape)
    assert flat_y.shape == (6,)
    assert flat_x.shape == (6,)
    assert flat_y[0] == 0 and flat_x[0] == 0
    assert flat_y[-1] == 1 and flat_x[-1] == 2


def test_get_niter_numpy_and_torch():
    dists = np.array([[0.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    out = cniter.get_niter(dists)
    assert isinstance(out, np.integer)
    assert out >= 1

    tdists = torch.tensor(dists)
    tout = cniter.get_niter(tdists)
    assert isinstance(tout, torch.Tensor)
    assert torch.is_floating_point(tout) is False


def test_torch_norm_matches_linalg():
    x = torch.arange(6.0).reshape(2, 3)
    out = tvector.torch_norm(x, dim=1)
    expected = torch.linalg.norm(x, dim=1)
    assert torch.allclose(out, expected)


def test_vector_field_from_zero_divergence():
    divergence = np.zeros((4, 5), dtype=np.float32)
    vec = tvector.compute_vector_field_from_divergence(divergence, grid_spacing=1)
    assert vec.shape == (2, 4, 5)
    assert np.allclose(vec, 0.0)
