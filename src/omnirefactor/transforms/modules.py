from __future__ import annotations

import numpy as np
import torch
from dask import array as da


def get_module(x):
    if isinstance(x, (np.ndarray, tuple, int, float, da.Array)) or np.isscalar(x):
        return np
    if torch.is_tensor(x):
        return torch
    raise ValueError("Input must be a numpy array, a tuple, a torch tensor, an integer, or a float")


__all__ = ["get_module"]
