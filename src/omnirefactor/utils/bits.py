from .imports import *
from ..transforms.normalize import rescale

def to_16_bit(im):
    """Rescale image [0,2^16-1] and then cast to uint16."""
    return np.uint16(rescale(im)*(2**16-1))

def to_8_bit(im):
    """Rescale image [0,2^8-1] and then cast to uint8."""
    return np.uint8(rescale(im)*(2**8-1))

def is_integer(var):
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
    elif isinstance(var, da.Array) and np.issubdtype(var.dtype, np.integer):
        return True
    # Check for PyTorch tensor with integer type
    elif isinstance(var, torch.Tensor) and not var.is_floating_point():
        return True
    # Not an integer or integer-like object
    return False
    
