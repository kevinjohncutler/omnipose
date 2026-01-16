from .device import ARM, gpu_logger

import torch

try:  # similar backward incompatibility where torch.mps does not even exist
    if ARM:
        from torch.mps import empty_cache
    else:
        from torch.cuda import empty_cache
except Exception as e:
    empty_cache = torch.cuda.empty_cache
    gpu_logger.info(e)
