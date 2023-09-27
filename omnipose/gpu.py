import torch
import platform  

ARM = 'arm' in platform.processor() # the backend chack for apple silicon does not work on intel macs
# ARM = torch.backends.mps.is_available() and ARM
# torch_GPU = torch.device('mps') if ARM else torch.device('cuda')
# torch_CPU = torch.device('cpu')
try: #backends not available in order versions of torch 
    ARM = torch.backends.mps.is_available() and ARM
except Exception as e:
    ARM = False
    print('resnet_torch.py backend check.',e)
torch_GPU = torch.device('mps') if ARM else torch.device('cuda')
torch_CPU = torch.device('cpu')

try: # similar backward incompatibility where torch.mps does not even exist 
    if ARM:
        from torch.mps import empty_cache
    else: 
        from torch.cuda import empty_cache
except Exception as e:
    empty_cache = torch.cuda.empty_cache
    print(e)


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