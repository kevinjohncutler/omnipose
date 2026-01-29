from asyncio.log import logger
from ..logger import TqdmToLogger
import os, sys, time, shutil, tempfile, datetime, pathlib
import logging
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import tempfile
import cv2
from scipy.stats import mode
from .. import core, data, loss, transforms, utils, metrics, io
from ..plot import rgb_flow
# from acb_mse import ACBLoss
# print('need to add acb_mse to deps')

from ..gpu import use_gpu, get_device
from contextlib import nullcontext

# from multiprocessing import Pool, cpu_count
# from functools import partial
# from focal_loss.focal_loss import FocalLoss

MXNET_ENABLED = False


import torch
from torch.amp import autocast, GradScaler
from torch import nn
from .network import torch_GPU, torch_CPU, UnetND, ARM, empty_cache
# torch.serialization.add_safe_globals(UnetND)



core_logger = logging.getLogger(__name__)
tqdm_out = TqdmToLogger(core_logger, level=logging.INFO)

_CUDA_PRECISION_LOCKED = False
_ALLOW_TF32_ENV = os.environ.get("CELLPOSE_OMNI_ALLOW_TF32", "").lower() in {"1", "true", "yes", "on"}


def _lock_cuda_precision(device):
    """Disable TF32/autotuned kernels so CUDA numerics stay aligned with CPU."""
    global _CUDA_PRECISION_LOCKED

    if _ALLOW_TF32_ENV:
        if not _CUDA_PRECISION_LOCKED:
            core_logger.info(
                "CELLPOSE_OMNI_ALLOW_TF32 set; leaving CUDA TF32/autotune enabled for benchmarking."
            )
            _CUDA_PRECISION_LOCKED = True
        return

    if _CUDA_PRECISION_LOCKED:
        return

    if not isinstance(device, torch.device) or device.type != 'cuda':
        return

    cudnn = getattr(torch.backends, 'cudnn', None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False
        if hasattr(cudnn, 'allow_tf32'):
            cudnn.allow_tf32 = False

    cuda_matmul = getattr(torch.backends, 'cuda', None)
    if cuda_matmul is not None:
        matmul = getattr(cuda_matmul, 'matmul', None)
        if matmul is not None and hasattr(matmul, 'allow_tf32'):
            matmul.allow_tf32 = False

    if hasattr(torch, 'set_float32_matmul_precision'):
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    _CUDA_PRECISION_LOCKED = True
    core_logger.info('Enforcing deterministic FP32 CUDA kernels for reproducibility.')

# nclasses now specified by user or by model type in models.py
def parse_model_string(pretrained_model):
    if isinstance(pretrained_model, list):
        model_str = os.path.split(pretrained_model[0])[-1]
    else:
        model_str = os.path.split(pretrained_model)[-1]
    if len(model_str)>3 and model_str[:4]=='unet':
        nclasses = max(2, int(model_str[4]))
    elif len(model_str)>7 and model_str[:8]=='cellpose':
        nclasses = 3
    else:
        return True, True, False
    ostrs = model_str.split('_')[2::2]
    residual_on = ostrs[0]=='on'
    style_on = ostrs[1]=='on'
    concatenation = ostrs[2]=='on'
    return residual_on, style_on, concatenation

def assign_device(gpu=True, gpu_number=None):
    device, gpu_available = get_device(gpu_number)
    if gpu and gpu_available:
        core_logger.info('Using GPU.')
        _lock_cuda_precision(device)
    elif gpu and not gpu_available:
        core_logger.info('No GPU available or pytorch not configured, using CPU.')
        device = torch_CPU
    else:
        core_logger.info('Using CPU.')
        device = torch_CPU
    return device, gpu_available
    
