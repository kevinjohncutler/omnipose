"""Array utilities for type conversion, size, and manipulation."""

import numpy as np

try:
    import dask.array as da
    DASK_ENABLED = True
except ImportError:
    DASK_ENABLED = False

try:
    import torch
    TORCH_ENABLED = True
except ImportError:
    TORCH_ENABLED = False

from ..transforms.normalize import rescale


def to_16_bit(im):
    """Rescale image [0,2^16-1] and then cast to uint16."""
    return np.uint16(rescale(im) * (2**16 - 1))


def to_8_bit(im):
    """Rescale image [0,2^8-1] and then cast to uint8."""
    return np.uint8(rescale(im) * (2**8 - 1))


def is_integer(var):
    """
    Check if a variable is an integer or integer-like.

    Handles Python int, NumPy integers, NumPy arrays with integer dtype,
    Dask arrays with integer dtype, and PyTorch tensors with integer dtype.
    """
    # Check for Python integer
    if isinstance(var, int):
        return True
    # Check for NumPy integer
    elif isinstance(var, np.integer):
        return True
    # Check for NumPy array or memmap with integer dtype
    elif isinstance(var, (np.ndarray, np.memmap)) and np.issubdtype(var.dtype, np.integer):
        return True
    # Check for Dask array with integer dtype
    elif DASK_ENABLED and isinstance(var, da.Array) and np.issubdtype(var.dtype, np.integer):
        return True
    # Check for PyTorch tensor with integer type
    elif TORCH_ENABLED and isinstance(var, torch.Tensor) and not var.is_floating_point():
        return True
    # Not an integer or integer-like object
    return False


def get_size(var, unit='GB'):
    """
    Get the memory size of an array.

    Parameters:
        var: Array with nbytes attribute (numpy, dask, torch, etc.)
        unit: Size unit - 'B', 'KB', 'MB', or 'GB'

    Returns:
        float: Size in the specified unit
    """
    units = {'B': 0, 'KB': 1, 'MB': 2, 'GB': 3}
    return var.nbytes / (1024 ** units[unit])


def random_int(N, M=None, seed=None):
    """
    Generate random integers.

    Parameters:
        N: Upper bound (exclusive) for random integers
        M: Number of integers to generate (optional)
        seed: Random seed (optional, prints seed if None)

    Returns:
        int or ndarray: Random integer(s)
    """
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
        print(f'Seed: {seed}')
    else:
        np.random.seed(seed)
    # Generate a random integer between 0 and N-1
    return np.random.randint(0, N, M)
