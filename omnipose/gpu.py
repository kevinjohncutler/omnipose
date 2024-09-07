import platform  
import os

from .logger import setup_logger
gpu_logger = setup_logger('gpu')

ARM = 'arm' in platform.processor() # the backend chack for apple silicon does not work on intel macs
if ARM:
    gpu_logger.info('On ARM, OMP_NUM_THREADS set to 1')
    os.environ['OMP_NUM_THREADS'] = '1'

# import torch after setting env variables 
import torch


# ARM = torch.backends.mps.is_available() and ARM
# torch_GPU = torch.device('mps') if ARM else torch.device('cuda')
# torch_CPU = torch.device('cpu')
try: #backends not available in order versions of torch 
    ARM = torch.backends.mps.is_available() and ARM
except Exception as e:
    ARM = False
    gpu_logger.warning('resnet_torch.py backend check.',e)
torch_GPU = torch.device('mps') if ARM else torch.device('cuda')
torch_CPU = torch.device('cpu')

try: # similar backward incompatibility where torch.mps does not even exist 
    if ARM:
        from torch.mps import empty_cache
    else: 
        from torch.cuda import empty_cache
        
except Exception as e:
    empty_cache = torch.cuda.empty_cache
    gpu_logger.info(e)

def use_gpu(gpu_number=0, use_torch=True):
    """ check if gpu works """
    if use_torch:
        return _use_gpu_torch(gpu_number)
    else:
        raise ValueError('cellpose only runs with pytorch now')

def _use_gpu_torch(gpu_number=0):
    try:
        if gpu_number is None:
            gpu_number = 0
        device = torch.device(f'mps:{gpu_number}') if ARM else torch.device(f'cuda:{gpu_number}')
        _ = torch.zeros([1, 2, 3]).to(device)
        return device, True
    except:# Exception as e:
        gpu_logger.info('TORCH GPU version not installed/working.')#, e)
        device = torch_CPU
        return device, False


# @torch.jit.script
# def custom_nonzero_cuda(tensor):
#   """Returns a tuple of tensors containing the non-zero indices of the given tensor
#   and the corresponding elements of the tensor."""

#   indices = torch.empty_like(tensor, dtype=torch.long)
#   elements = torch.empty_like(tensor)

#   block_size = 32
#   num_blocks = (tensor.numel() + block_size - 1) // block_size

#   torch.cuda.launch_kernel(
#       'custom_nonzero_kernel', num_blocks, block_size,
#       tensor, indices, elements)

#   return indices, elements