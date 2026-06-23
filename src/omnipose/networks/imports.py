"""Centralized imports for the networks subpackage.

Houses ``UnetND`` and the device / precision helpers that wrap it. Depends
on Layer 0 (gpu) and the logger.
"""

import os
import logging

import numpy as np
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from contextlib import nullcontext

from ..gpu import use_gpu, get_device, ARM, torch_GPU, torch_CPU, empty_cache
from ..logger import TqdmToLogger

core_logger = logging.getLogger(__name__)
tqdm_out = TqdmToLogger(core_logger, level=logging.INFO)
