"""GPU device management — thin wrapper over ``ocdkit.gpu``.

Adds the omnipose-specific ARM env-var workaround (OMP_NUM_THREADS) on
import, then re-exports everything from ``ocdkit.gpu``.
"""

import os
import platform

from ..logger import get_logger

gpu_logger = get_logger('gpu', color='#ff0055')

# ARM-specific GUI workaround (must run before torch import)
if 'arm' in platform.processor():
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('PARLAY_NUM_THREADS', '1')

from ocdkit.gpu import *

# networks/__init__.py calls assign_device which is get_device
assign_device = get_device
