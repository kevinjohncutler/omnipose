import numpy as np
import torch

from omnirefactor.core import niter as cniter


def test_get_niter_numpy_and_torch():
    dists = np.array([[0.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    out = cniter.get_niter(dists)
    assert isinstance(out, np.integer)
    assert out >= 1

    tdists = torch.tensor(dists)
    tout = cniter.get_niter(tdists)
    assert isinstance(tout, torch.Tensor)
    assert torch.is_floating_point(tout) is False
