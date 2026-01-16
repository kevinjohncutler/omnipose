import multiprocessing
import os
import platform

from ..logger import get_logger

gpu_logger = get_logger('gpu', color='#ff0055')

ARM = 'arm' in platform.processor()  # backend check for Apple Silicon
if ARM:
    nt = str(multiprocessing.cpu_count())
    nt = "1"  # helps with gui crashing on subprocess
    os.environ['OMP_NUM_THREADS'] = nt
    os.environ["PARLAY_NUM_THREADS"] = "1"

# import torch after setting env variables
import torch

try:  # backends not available in older versions of torch
    ARM = torch.backends.mps.is_available() and ARM
except Exception as e:
    ARM = False
    gpu_logger.warning('resnet_torch.py backend check.', e)

torch_GPU = torch.device('mps') if ARM else torch.device('cuda')
torch_CPU = torch.device('cpu')


def get_device(gpu_number=0, use_torch=True):
    """check if gpu works"""
    if use_torch:
        return _get_gpu_torch(gpu_number)
    raise ValueError('cellpose only runs with pytorch now')


def use_gpu(gpu_number=0, use_torch=True):
    """alias for get_device"""
    return get_device(gpu_number, use_torch)


def _get_gpu_torch(gpu_number=0):
    try:
        if gpu_number is None:
            gpu_number = 0
        device = torch.device(f'mps:{gpu_number}') if ARM else torch.device(f'cuda:{gpu_number}')
        _ = torch.zeros([1, 2, 3]).to(device)
        return device, True
    except Exception:
        gpu_logger.info('TORCH GPU version not installed/working.')
        device = torch_CPU
        return device, False
