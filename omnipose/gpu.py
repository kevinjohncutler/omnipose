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