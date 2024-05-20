import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from scipy.ndimage import convolve1d, convolve, affine_transform
from skimage.morphology import remove_small_holes
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as im_shift
from skimage import color
import dask 
from .plot import sinebow

from skimage import measure 
import fastremap
import mahotas as mh
import math
from ncolor import format_labels # just in case I forgot to switch it out elsewhere
from pathlib import Path
import os
import re

import mgen
import fastremap

from numba import njit
import functools
import itertools

import dask
import dask.array as da

# import logging, sys
# LOGGER_FORMAT = "%(asctime)-20s\t[%(levelname)-5s]\t[%(filename)-10s %(lineno)-5d%(funcName)-18s]\t%(message)s"
# logging.basicConfig(
#     level=logging.INFO,
#     format=LOGGER_FORMAT,
#     handlers=[
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# omnipose_logger = logging.getLogger(__name__)
# logging.getLogger('xmlschema').setLevel(logging.WARNING) # get rid of that annoying xmlschema warning
# # logging.getLogger('qdarktheme').setLevel(logging.WARNING)

import sys
from .logger import setup_logger
omnipose_logger = setup_logger('utils')

# No reason to support anything but pytorch for omnipose
# I want it to fail here otherwise, much easier to debug 
import torch
TORCH_ENABLED = True 
# the following is duplicated but I cannot import cellpose, circular import issue
import platform  
ARM = 'arm' in platform.processor() # the backend chack for apple silicon does not work on intel macs
try: #backends not available in order versions of torch 
    ARM = torch.backends.mps.is_available() and ARM
except:
    ARM = False
torch_GPU = torch.device('mps') if ARM else torch.device('cuda')
torch_CPU = torch.device('cpu')


def find_files(directory, suffix, exclude_suffixes=[]):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            name, ext = os.path.splitext(basename)
            if name.endswith(suffix) and not any(name.endswith(exclude) for exclude in exclude_suffixes):
                filename = os.path.join(root, basename)
                yield filename
                

def findbetween(s,string1='[',string2=']'):
    """Find text between string1 and string2."""
    return re.findall(str(re.escape(string1))+"(.*)"+str(re.escape(string2)),s)[0]

def getname(path,prefix='',suffix='',padding=0):
    """Extract the file name."""
    return os.path.splitext(Path(path).name)[0].replace(prefix,'').replace(suffix,'').zfill(padding)

def to_16_bit(im):
    """Rescale image [0,2^16-1] and then cast to uint16."""
    return np.uint16(rescale(im)*(2**16-1))

def to_8_bit(im):
    """Rescale image [0,2^8-1] and then cast to uint8."""
    return np.uint8(rescale(im)*(2**8-1))
    

### This section defines the tiling functions 
def get_module(x):
    if isinstance(x, (np.ndarray, tuple, int, float, dask.array.Array)) or np.isscalar(x):
        return np
    elif torch.is_tensor(x):
        return torch
    else:
        raise ValueError("Input must be a numpy array, a tuple, a torch tensor, an integer, or a float")
        
def get_flip(idx):
    module = get_module(idx)
    return tuple([slice(None,None,None) if i%2 else 
                  slice(None,None,-1) for i in idx])


def _taper_mask_ND(shape=(224,224), sig=7.5):
    dim = len(shape)
    bsize = max(shape)
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    # 1D distribution 
    mask = 1/(1 + np.exp((xm - (bsize/2-20)) / sig)) 
    # extend to ND
    for j in range(dim-1):
        mask = mask * mask[..., np.newaxis]
    slc = tuple([slice(bsize//2-s//2,bsize//2+s//2+s%2) for s in shape])
    mask = mask[slc]
    return mask

def unaugment_tiles_ND(y, inds, unet=False):
    """ reverse test-time augmentations for averaging

    Parameters
    ----------

    y: float32
        array of shape (ntiles, nchan, *DIMS)
        where nchan = (*DP,distance) (and boundary if nlasses=3)

    unet: bool (optional, False)
        whether or not unet output or cellpose output
    
    Returns
    -------

    y: float32

    """
    module = get_module(y)
    dim = len(inds[0])
    for i,idx in enumerate(inds): 
        flip = get_flip(idx)
        factor = module.array([1 if i%2 else -1 for i in idx])
        y[i] = y[i][(Ellipsis,)+flip]
        if not unet:
            y[i][:dim] = [s*f for s,f in zip(y[i][:dim],factor)]
    return y

def average_tiles_ND(y,subs,shape):
    """ average results of network over tiles

    Parameters
    -------------

    y: float, [ntiles x nclasses x bsize x bsize]
        output of cellpose network for each tile

    subs : list
        list of slices for each subtile 

    shape : int, list or tuple
        shape of pre-tiled image (may be larger than original image if
        image size is less than bsize)

    Returns
    -------------

    yf: float32, [nclasses x Ly x Lx]
        network output averaged over tiles
    """
    module = get_module(y)
    is_torch = module.__name__ == 'torch'
    if is_torch:
        params = {'device':y.device,'dtype':torch.float32} 
    else:
        params = {'dtype':np.float32}
    
    Navg = module.zeros(shape,**params)
    yf = module.zeros((y.shape[1],)+shape, **params)
    mask = _taper_mask_ND(y.shape[-len(shape):])
    
    if is_torch:
        mask = torch.tensor(mask,device=y.device)
        
    for j,slc in enumerate(subs):
        yf[(Ellipsis,)+slc] += y[j] * mask
        Navg[slc] += mask
    yf /= Navg
    return yf

def make_tiles_ND(imgi, bsize=224, augment=False, tile_overlap=0.1, 
                  normalize=True, return_tiles=True):
    """ make tiles of image to run at test-time

    if augmented, tiles are flipped and tile_overlap=2.
        * original
        * flipped vertically
        * flipped horizontally
        * flipped vertically and horizontally

    Parameters
    ----------
    imgi : float32
        array that's nchan x Ly x Lx

    bsize : float (optional, default 224)
        size of tiles

    augment : bool (optional, default False)
        flip tiles and set tile_overlap=2.

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles

    Returns
    -------
    IMG : float32
        tensor of shape ntiles,nchan,bsize,bsize

    subs : list
        list of slices for each subtile

    shape : tuple
        shape of original image

    """
    module = get_module(imgi)
    nchan = imgi.shape[0]
    shape = imgi.shape[1:]
    dim = len(shape)
    inds = []
    if augment:
        bsize = int(bsize)
        pad_seq = [(0,0)]+[(0,max(0,bsize-s))for s in shape]
        imgi = module.pad(imgi,pad_seq)
        shape = imgi.shape[-dim:]
        ntyx = [max(2, int(module.ceil(2. * s / bsize))) for s in shape]
        start = [module.linspace(0, s-bsize, n).astype(int) for s,n in zip(shape,ntyx)]
        intervals = [[slice(si,si+bsize) for si in s] for s in start]
        subs = list(itertools.product(*intervals))
        indexes = [module.arange(len(s)) for s in start]
        inds = list(itertools.product(*indexes))
        IMG = []
        for slc,idx in zip(subs,inds):        
            flip = get_flip(idx)
            IMG.append(imgi[(Ellipsis,)+slc][(Ellipsis,)+flip])
        IMG = module.stack(IMG)
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        bbox = tuple([int(min(bsize,s)) for s in shape])
        ntyx = [1 if s<=bsize else int(np.ceil((1.+2*tile_overlap) * s / bsize)) 
                for s in shape]
        start = [np.linspace(0, s-b, n).astype(int) for s,b,n in zip(shape,bbox,ntyx)]
        intervals = [[slice(si,si+bsize) for si in s] for s in start]
        subs = list(itertools.product(*intervals))
        
        if return_tiles:
            IMG = module.stack([imgi[(Ellipsis,)+slc] for slc in subs])
            if normalize:
                omnipose_logger.info('Running on tiles. Now normalizing each tile separately.')
                IMG = normalize99(IMG,dim=0)
            else:
                omnipose_logger.info('rescaling stack as a whole')
                IMG = rescale(IMG)
        else:
            IMG = None
                
    return IMG, subs, shape, inds



def generate_slices(image_shape, crop_size):
    """Generate slices for cropping an image into crops of size crop_size."""
    num_crops = [math.ceil(s / crop_size) for s in image_shape]
    I,J = range(num_crops[0]),range(num_crops[1])
    slices = [[[] for j in  J] for i in I]
    for i in I:
        row_start = i * crop_size
        row_end = min((i + 1) * crop_size, image_shape[0])
        for j in J:
            col_start = j * crop_size
            col_end = min((j + 1) * crop_size, image_shape[1])
            # slices.append((slice(row_start, row_end), slice(col_start, col_end)))
            slices[i][j] = (slice(row_start, row_end), slice(col_start, col_end))
            
    return slices, num_crops

# def shifts_to_slice(shifts,shape):
#     """
#     Find the minimal crop box from time lapse registration shifts.
#     """
# #     max_shift = np.max(shifts,axis=0)
# #     min_shift = np.min(shifts,axis=0)
# #     slc = tuple([slice(np.maximum(0,0+int(mn)),np.minimum(s,s-int(mx))) for mx,mn,s in zip(np.flip(max_shift),np.flip(min_shift),shape)])
#     # slc = tuple([slice(np.maximum(0,0+int(mn)),np.minimum(s,s-int(mx))) for mx,mn,s in zip(max_shift,min_shift,shape)])
#     upper_shift = np.min(shifts,axis=0)
#     lower_shift = np.max(shifts,axis=0)
#     slc = tuple([slice(np.maximum(0,0+int(l)),np.minimum(s,s-int(u))) for u,l,s in zip(upper_shift,lower_shift,shape)])
#     return slc


import numpy as np

def shifts_to_slice(shifts, shape):
    """
    Find the minimal crop box from time lapse registration shifts.
    """    
    # Convert shifts to integers
    shifts = np.round(shifts).astype(int)
    
    # Create a slice for each dimension
    slices = tuple(slice(max(0, np.max(shifts[:, dim])), min(shape[dim], shape[dim] + np.min(shifts[:, dim])))
                   for dim in range(shifts.shape[1]))
    
    return slices
    
def make_unique(masks):
    """Relabel stack of label matrices such that there is no repeated label across slices."""
    masks = masks.copy().astype(np.uint32)
    T = range(len(masks))
    offset = 0 
    for t in T:
        # f = format_labels(masks[t],clean=True)
        fastremap.renumber(masks[t],in_place=True)
        masks[t][masks[t]>0]+=offset
        offset = masks[t].max()
    return masks
    
    
# import imreg_dft 
def cross_reg(imstack,upsample_factor=100,order=1,
              normalization=None,
              reverse=True, localnorm=True):
    """
    Find the transformation matrices for all images in a time series to align to the beginning frame. 
    """
    dim = imstack.ndim - 1 # dim is spatial, assume first dimension is t
    s = np.zeros(dim)
    shape = imstack.shape[-dim:]
    regstack = np.zeros_like(imstack)
    shifts = np.zeros((len(imstack),dim))
    
    
    images_to_register = imstack if not reverse else imstack[::-1]

    # Now images_to_register[i] is the sum of image_stack over the interval slices[i]
    if localnorm: 
        images_to_register = images_to_register/gaussian_filter(images_to_register,sigma=[0,1,1])
                    
    shift_vectors = [phase_cross_correlation(images_to_register[i], 
                                            images_to_register[i+1], 
                                            upsample_factor = upsample_factor,
                                            normalization=normalization)[0] for i in range(len(images_to_register)-1)]
    
    
    shift_vectors.insert(0, np.asarray([0.0,0.0]))  
    shift_vectors = np.stack(shift_vectors)

    shift_vectors = np.where(np.abs(shift_vectors) > 50, 0, shift_vectors)
    shift_vectors = np.cumsum(shift_vectors, axis=0)
        
 
    if reverse:
        shift_vectors = -shift_vectors[::-1]
        
    return shift_vectors

def shift_stack(imstack, shift_vectors, order=1, cval=None, prefilter=True):
    """
    Shift each time slice of imstack according to list of nD shifts. 
    """
    imstack = imstack.astype(np.float32)
    shift_vectors = shift_vectors.astype(np.float32)

    # delayed_images = da.from_array(imstack).to_delayed()
    
    ndim = imstack.ndim
    axes = tuple(range(-(ndim-1),0))
    cvals = np.nanmean(imstack,axis=axes) if cval is None else [cval]*len(shift_vectors)
    mode = 'nearest' if cval is None else 'constant'
    
    # Apply the shift to each image
    shifted_images = [dask.delayed(im_shift)(image,
                                            shift_vector, 
                                            order=order,
                                            prefilter=prefilter,
                                            mode=mode,
                                            cval=cv,
                                            ) for image, shift_vector, cv in zip(imstack, shift_vectors, cvals)]

    # Compute the shifted images in parallel
    shifted_images = dask.compute(*shifted_images)
    shifted_images = np.stack(shifted_images, axis=0)
    
    return shifted_images

# GPU version
import torch
import torch.fft

# def phase_cross_correlation_GPU(target, moving_images):

#     # Assuming target is a 2D tensor [height, width]
#     # and moving_images is a 3D tensor [num_images, height, width]

#     # Expand dims of target to match moving_images
#     target = target.unsqueeze(0)
#     # print(target.shape,moving_images.shape)
#     # Compute FFT of images
#     target_fft = torch.fft.fftn(target, dim=[-2, -1])
#     moving_fft = torch.fft.fftn(moving_images, dim=[-2, -1])
    
#     # print(target_fft.shape,moving_fft.shape)
    
#     # Compute cross-correlation by multiplying with complex conjugate
#     cross_corr = torch.fft.ifftn(target_fft * moving_fft.conj(), dim=[-2, -1]).real
    
#     # Find peak in cross-correlation
#     max_indices = torch.argmax(cross_corr.view(cross_corr.shape[0], -1), dim=1)
    
#     # Convert flat indices to 2D indices
#     height = cross_corr.shape[-2]
#     width = cross_corr.shape[-1]
#     shifts_y = max_indices // width
#     shifts_x = max_indices % width

#     # Adjust shifts to fall within the correct range
#     # make sure shift vector points in the right direction 
#     shifts_y =  height // 2 - (shifts_y + height // 2) % height
#     shifts_x =  width // 2 - (shifts_x + width // 2) % width

#     # Combine shifts along both dimensions into a single tensor
#     shifts = torch.stack([shifts_y, shifts_x], dim=-1)
#     return shifts

# def phase_cross_correlation_GPU(image_stack, target_index):
#     # Assuming image_stack is a 3D tensor [num_images, height, width]
#     # and target_index is an integer

#     target_image = image_stack[target_index].unsqueeze(0)
#     moving_images = torch.cat([image_stack[:target_index], image_stack[target_index+1:]])

#     target_fft = torch.fft.fftn(target_image, dim=[-2, -1])
#     moving_fft = torch.fft.fftn(moving_images, dim=[-2, -1])

#     cross_corr = torch.fft.ifftn(target_fft * moving_fft.conj(), dim=[-2, -1]).real

#     max_indices = torch.argmax(cross_corr.view(cross_corr.shape[0], -1), dim=1)

#     height = cross_corr.shape[-2]
#     width = cross_corr.shape[-1]
#     shifts_y = max_indices // width
#     shifts_x = max_indices % width

#     shifts_y =  height // 2 - (shifts_y + height // 2) % height
#     shifts_x =  width // 2 - (shifts_x + width // 2) % width

#     shifts = torch.stack([shifts_y, shifts_x], dim=-1)

#     # Insert a zero shift at the target index
#     zero_shift = torch.zeros(1, 2, device=image_stack.device)
#     shifts = torch.cat([shifts[:target_index], zero_shift, shifts[target_index:]])

#     return shifts.long()
    


# import torch.nn.functional as F

# def phase_cross_correlation_GPU(image_stack, target_index, upsample_factor=1):
#     # Assuming image_stack is a 3D tensor [num_images, height, width]
#     # and target_index is an integer

#     target_image = image_stack[target_index].unsqueeze(0)
#     moving_images = torch.cat([image_stack[:target_index], image_stack[target_index+1:]])

#     target_fft = torch.fft.fftn(target_image, dim=[-2, -1])
#     moving_fft = torch.fft.fftn(moving_images, dim=[-2, -1])

#     cross_corr = torch.fft.ifftn(target_fft * moving_fft.conj(), dim=[-2, -1]).real

#     # Upsample cross correlation to achieve subpixel precision
#     if upsample_factor > 1:
#         cross_corr = cross_corr.unsqueeze(1)
#         print('cc',cross_corr.shape)
#         cross_corr = F.interpolate(cross_corr, scale_factor=upsample_factor, 
#                                    mode='bilinear', align_corners=False)
        
#         print('cc',cross_corr.shape)
        
#         cross_corr = cross_corr.squeeze(1)

#     max_indices = torch.argmax(cross_corr.view(cross_corr.shape[0], -1), dim=1)

#     height = cross_corr.shape[-2]
#     width = cross_corr.shape[-1]
#     shifts_y = max_indices // width
#     shifts_x = max_indices % width
#     shifts_y =  height // 2 - (shifts_y + height // 2) % height
#     shifts_x =  width // 2 - (shifts_x + width // 2) % width

#     # Convert shifts back to original pixel grid
#     shifts_y = shifts_y / upsample_factor
#     shifts_x = shifts_x / upsample_factor

#     shifts = torch.stack([shifts_y, shifts_x], dim=-1)

#     # Insert a zero shift at the target index
#     zero_shift = torch.zeros(1, 2, device=image_stack.device)
#     shifts = torch.cat([shifts[:target_index], zero_shift, shifts[target_index:]])
#     return shifts

# def phase_cross_correlation_GPU(image_stack, target_index, upsample_factor=10):
#     # Assuming image_stack is a 3D tensor [num_images, height, width]
#     # and target_index is an integer

#     # Upsample the images
#     image_stack = F.interpolate(image_stack.unsqueeze(1).float(), scale_factor=upsample_factor, mode='bilinear', align_corners=False).squeeze(1)

#     target_image = image_stack[target_index].unsqueeze(0)
#     moving_images = torch.cat([image_stack[:target_index], image_stack[target_index+1:]])

#     target_fft = torch.fft.fftn(target_image, dim=[-2, -1])
#     moving_fft = torch.fft.fftn(moving_images, dim=[-2, -1])

#     cross_corr = torch.fft.ifftn(target_fft * moving_fft.conj(), dim=[-2, -1]).real

#     max_indices = torch.argmax(cross_corr.view(cross_corr.shape[0], -1), dim=1)

#     height = cross_corr.shape[-2]
#     width = cross_corr.shape[-1]
#     shifts_y = max_indices // width
#     shifts_x = max_indices % width

#     shifts_y =  height // 2 - (shifts_y + height // 2) % height
#     shifts_x =  width // 2 - (shifts_x + width // 2) % width

#     # Convert shifts back to original pixel grid
#     shifts_y = shifts_y / upsample_factor
#     shifts_x = shifts_x / upsample_factor

#     shifts = torch.stack([shifts_y, shifts_x], dim=-1)

#     # Insert a zero shift at the target index
#     zero_shift = torch.zeros(1, 2, device=image_stack.device)
#     shifts = torch.cat([shifts[:target_index], zero_shift, shifts[target_index:]])

#     return shifts.float()


def gaussian_kernel(size: int, sigma: float, device=torch_GPU):
    """Creates a 2D Gaussian kernel with mean 0.

    Args:
        size (int): The size of the kernel. Should be an odd number.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: The Gaussian kernel.
    """
    coords = torch.arange(size,device=device).float() - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.outer(g)

def apply_gaussian_blur(image, kernel_size, sigma, device=torch_GPU):
    """Applies a Gaussian blur to the image.

    Args:
        image (torch.Tensor): The image to blur.
        kernel_size (int): The size of the Gaussian kernel.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: The blurred image.
    """
    kernel = gaussian_kernel(kernel_size, sigma, device).unsqueeze(0).unsqueeze(0)
    image = image.unsqueeze(0).unsqueeze(0)

    # Apply 'reflect' padding to the image
    padding_size = kernel_size // 2
    image = F.pad(image, (padding_size, padding_size, padding_size, padding_size), mode='reflect')

    # Perform the convolution without additional padding
    blurred = F.conv2d(image, kernel, padding=0)

    return blurred.squeeze(0).squeeze(0)

def phase_cross_correlation_GPU_old(image_stack, target_index=None, upsample_factor=10, 
                                reverse=False,normalize=False):
    # Assuming image_stack is a 3D tensor [num_images, height, width]
    # and target_index is an integer or None for sequential registration

    
    im_to_reg = torch.stack([i/apply_gaussian_blur(i, 5, 1) for i in image_stack])

    # Upsample the images
    image_stack = F.interpolate(im_to_reg.unsqueeze(1).float(), 
                                scale_factor=upsample_factor, mode='bilinear', 
                                align_corners=False).squeeze(1)

    # Initialize shifts with a zero shift for the first image
    # shifts = [[0, 0]]
    shifts = []
    
    for i in range(1, len(image_stack)):
        if target_index is None:
            # Sequential registration
            # target_image = image_stack[i-1]
            if reverse:
                # Reverse registration
                target_image = image_stack[i+1] if i < len(image_stack) - 1 else image_stack[i]
            else:
                # Sequential registration
                target_image = image_stack[i-1] if i > 0 else image_stack[i]
        else:
            # Target registration
            target_image = image_stack[target_index]

        moving_image = image_stack[i]

        # target_fft = torch.fft.fftn(target_image.unsqueeze(0), dim=[-2, -1])
        # moving_fft = torch.fft.fftn(moving_image.unsqueeze(0), dim=[-2, -1])
        target_fft = torch.fft.fftn(target_image, dim=[-2, -1])
        moving_fft = torch.fft.fftn(moving_image, dim=[-2, -1])

        # Compute the cross-power spectrum
        cross_power_spectrum = target_fft * moving_fft.conj()

        # Normalize the cross-power spectrum if the normalize option is True
        if normalize:
            cross_power_spectrum /= torch.abs(cross_power_spectrum)
        
        cross_corr = torch.abs(torch.fft.ifftn(cross_power_spectrum, dim=[-2, -1]))
        print('cc',cross_corr.shape)
        max_index = torch.argmax(cross_corr.view(-1))

        height = cross_corr.shape[-2]
        width = cross_corr.shape[-1]
        shift_y = max_index // width
        shift_x = max_index % width

        shift_y =  height // 2 - (shift_y + height // 2) % height
        shift_x =  width // 2 - (shift_x + width // 2) % width

        # Convert shifts back to original pixel grid
        shift_y = shift_y / upsample_factor
        shift_x = shift_x / upsample_factor

        shifts.append([shift_y, shift_x])

    shifts.append([0,0])
    shifts = torch.tensor(shifts, device=image_stack.device)*(-2)

    # Subtract the average shift from all shifts to minimize the total shift
    # avg_shift = shifts.mean(dim=0)
    # shifts -= avg_shift
    # shifts = torch.cumsum(shifts,dim=0)
    
    return shifts.float()
    
#     return accumulated_shifts
def phase_cross_correlation_GPU(image_stack, 
                                upsample_factor=10, 
                                # normalization='phase'
                                normalization=None,
                                
                                ):
    # Assuming image_stack is a 3D tensor [num_images, height, width]
    
    # Upsample the images
    # image_stack = F.interpolate(image_stack.unsqueeze(1).float(), 
    #                             scale_factor=upsample_factor, mode='bilinear', 
    #                             align_corners=False).squeeze(1)
    
    # m = torch.nn.Upsample(scale_factor=tuple([upsample_factor,upsample_factor]),mode='bilinear')
    # image_stack = m(image_stack.float().unsqueeze(1)).squeeze(1)
    device = image_stack.device
    
    im_to_reg = torch.stack([i/apply_gaussian_blur(i, 9, 3, device=device) for i in image_stack.float()])
    # im_to_reg = image_stack
    # Compute the FFT of the images
    norm='backward'
    image_fft = torch.fft.fft2(im_to_reg,norm=norm)#, dim=[-2, -1])
    
    # Compute the cross-power spectrum for each pair of images
    cross_power_spectrum = image_fft[:-1] * image_fft[1:].conj()
    
    # Normalize the cross-power spectrum
    if normalization == 'phase':
        cross_power_spectrum /= torch.abs(cross_power_spectrum)#+1e-6
    
    # Compute the cross-correlation by taking the inverse FFT
    cross_corr = torch.abs(torch.fft.ifft2(cross_power_spectrum,norm=norm)) #, dim=[-2, -1])
    m = torch.nn.Upsample(scale_factor=upsample_factor,mode='bilinear')
    cross_corr = m(cross_corr.unsqueeze(1)).squeeze(1)
    
    # Find the shift for each pair of images
    max_indices = torch.argmax(cross_corr.view(cross_corr.shape[0], -1), dim=-1).float()
    shifts_y, shifts_x = (max_indices / cross_corr.shape[-1]).long(), (max_indices % cross_corr.shape[-1]).long()

    # Stack the shifts and append a [0, 0] shift at the beginning
    # shifts = torch.stack([shifts_y, shifts_x]).T
    shifts = 2*torch.stack([shifts_y, shifts_x]).T
    zero_shift = torch.zeros(1, 2, dtype=shifts.dtype, device=shifts.device)
    shifts = torch.cat([shifts,zero_shift], dim=0) / upsample_factor

    # Accumulate the shifts - SUPER important and was the cause of the bug 
    shifts = torch.cumsum(shifts.flip(dims=[0]),dim=0).flip(dims=[0])
    
    # Subtract the average shift from all shifts to minimize the total shift
    avg_shift = shifts.mean(dim=0)
    shifts -= avg_shift

    # should replace shift by making it so that the shifts are closest to pixel shifts? 

    return shifts

# ### below two functions an experiment 
# def pairwise_registration(image_stack, upsample_factor=10):

#     im_to_reg = torch.stack([i/apply_gaussian_blur(i, 5, 5) for i in image_stack])

#     # Upsample the images
#     image_stack = F.interpolate(im_to_reg.unsqueeze(1).float(), scale_factor=upsample_factor, mode='bilinear', align_corners=False).squeeze(1)

#     num_images = len(image_stack)
#     shifts = torch.zeros((num_images, num_images, 2), device=image_stack.device)

#     for i in range(num_images):
#         for j in range(i+1, num_images):
#             target_image = image_stack[i]
#             moving_image = image_stack[j]

#             target_fft = torch.fft.fftn(target_image.unsqueeze(0), dim=[-2, -1])
#             moving_fft = torch.fft.fftn(moving_image.unsqueeze(0), dim=[-2, -1])

#             cross_corr = torch.fft.ifftn(target_fft * moving_fft.conj(), dim=[-2, -1]).real

#             max_index = torch.argmax(cross_corr.view(-1))

#             height = cross_corr.shape[-2]
#             width = cross_corr.shape[-1]
#             shift_y = max_index // width
#             shift_x = max_index % width

#             shift_y =  height // 2 - (shift_y + height // 2) % height
#             shift_x =  width // 2 - (shift_x + width // 2) % width

#             # Convert shifts back to original pixel grid
#             shift_y = shift_y / upsample_factor
#             shift_x = shift_x / upsample_factor

#             shifts[i, j] = torch.tensor([shift_y, shift_x])
#             shifts[j, i] = torch.tensor([-shift_y, -shift_x])  # Reverse shift for the opposite direction

#     # return shifts
#     # Compute final shifts
#     final_shifts = compute_final_shifts(shifts)
#     final_shifts = torch.cumsum(final_shifts, dim=0)
#     return final_shifts
    
# import networkx as nx
# def compute_final_shifts(pairwise_shifts):
#     # Create a graph where each node is an image and each edge is a shift
#     G = nx.Graph()

#     num_images = pairwise_shifts.shape[0]
#     for i in range(num_images):
#         for j in range(i+1, num_images):
#             shift = pairwise_shifts[i, j]
#             # Add an edge between image i and image j with weight equal to the magnitude of the shift
#             G.add_edge(i, j, weight=torch.norm(shift), shift=shift)

#     # Compute the minimum spanning tree of the graph
#     mst = nx.minimum_spanning_tree(G)

#     # Initialize final shifts with zeros
#     final_shifts = torch.zeros((num_images, 2), device=pairwise_shifts.device)

#     # Use a DFS to compute the shifts of all images relative to the reference
#     for edge in nx.dfs_edges(mst, source=0):
#         i, j = edge
#         shift = mst.edges[i, j]['shift']
#         final_shifts[j] = final_shifts[i] + shift

#     return final_shifts
    
# ### 

# def apply_shifts(moving_images, shifts):
#     # Assuming moving_images is a 3D tensor [num_images, height, width]
#     # and shifts is a 2D tensor [num_images, 2] (y, x)
#     N, H, W = moving_images.shape
#     device = moving_images.device
    

#     # Normalize the shifts to be in the range [-1, 1]
#     shifts = shifts / torch.tensor([H, W]).to(device)

#     # Create a grid of indices
#     grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float), 
#                                     torch.arange(W, device=device, dtype=torch.float)) 

#     # Normalize the grid to be in the range [-1, 1]
#     grid_y = 2.0 * grid_y / (H - 1) - 1.0
#     grid_x = 2.0 * grid_x / (W - 1) - 1.0

#     # Apply the shifts to the grid of indices
#     grid_y = grid_y[None] + shifts[:, 0][:, None, None]
#     grid_x = grid_x[None] + shifts[:, 1][:, None, None]

#     # Stack the grids to create a [N, H, W, 2] grid
#     grid = torch.stack([grid_x, grid_y], dim=-1)

#     # Use the shifted grid of indices to index into moving_images
#     intersection = F.grid_sample(moving_images.unsqueeze(1), grid, align_corners=False)

#     return intersection.squeeze(1)


#turns out that looping over the shifts is faster than using grid_sample on the entire thing, at least on CPU
# @torch.jit.script
def apply_shifts(moving_images, shifts):
    # If shifts is a 1D tensor, add an extra dimension to make it 2D
    if len(shifts.shape) == 1:
        shifts = shifts.unsqueeze(0)

    # print('shifts',shifts.shape)

    N, H, W = moving_images.shape
    device = moving_images.device
    # Normalize the shifts to be in the range [-1, 1]
    shifts = shifts / torch.tensor([H, W]).to(device)

    # Create a grid of indices
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float), 
                                    torch.arange(W, device=device, dtype=torch.float),
                                    indexing='ij') 

    # Normalize the grid to be in the range [-1, 1]
    grid_y = 2.0 * grid_y / (H - 1) - 1.0
    grid_x = 2.0 * grid_x / (W - 1) - 1.0

    # Initialize tensor to hold the shifted images
    shifted_images = torch.empty_like(moving_images)

    # Find unique shifts and their indices
    unique_shifts, indices = torch.unique(shifts, dim=0, return_inverse=True)

    # Group the indices by their corresponding shifts
    bincounts = torch.bincount(indices)
    split_sizes = [bincounts[i].item() for i in range(bincounts.size(0))]
    grouped_indices = torch.split_with_sizes(indices, split_sizes)

    for i, group in enumerate(grouped_indices):
        # Get the shift for this group
        shift = unique_shifts[i]

        # Apply the shift to the grid of indices
        grid_y_shifted = grid_y[None] + shift[0]
        grid_x_shifted = grid_x[None] + shift[1]

        # Stack the grids to create a [1, H, W, 2] grid
        grid = torch.stack([grid_x_shifted, grid_y_shifted], dim=-1)

        # Use the shifted grid of indices to index into the slices
        shifted_slices = torch.nn.functional.grid_sample(moving_images[group].unsqueeze(1), 
                                                         grid.repeat(len(group),1,1,1), 
                                                         mode='bilinear', #default
                                                         align_corners=False #default
                                                         )

        # Store the shifted slices
        shifted_images[group] = shifted_slices.squeeze(1)

    return shifted_images
    
# from scipy.ndimage import map_coordinates

# def apply_shifts_numpy(moving_images: np.ndarray, shifts: np.ndarray) -> np.ndarray:
#     N, H, W = moving_images.shape

#     # Normalize the shifts to be in the range [-1, 1]
#     shifts = shifts / np.array([H, W])

#     # Create a grid of indices
#     grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

#     # Apply the shift to the grid of indices
#     grid_y_shifted = grid_y[None] + shifts[:, 0, None, None]
#     grid_x_shifted = grid_x[None] + shifts[:, 1, None, None]

#     # Use the shifted grid of indices to index into the slices
#     shifted_images = np.empty_like(moving_images)
#     for i, image in enumerate(moving_images):
#         shifted_images[i] = map_coordinates(image, [grid_y_shifted[i], grid_x_shifted[i]], order=1)

#     return shifted_images  


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def normalize_field(mu,use_torch=False,cutoff=0):
    """ normalize all nonzero field vectors to magnitude 1
    
    Parameters
    ----------
    mu: ndarray, float
        Component array of lenth N by L1 by L2 by ... by LN. 
    
    Returns
    --------------
    normalized component array of identical size. 
    """
    if use_torch:
        mag = torch_norm(mu,dim=0)
        # out = torch.zeros_like(mu)
        # sel = mag>cutoff
        # out[:,sel] = torch.div(mu[:,sel],mag[sel])
        # return out
        # return torch.where(mag>cutoff,mu/mag,torch.zeros_like(mu))
        return torch.where(mag>cutoff,mu/mag,mu)
        
    else:
        mag = np.sqrt(np.nansum(mu**2,axis=0))
        return safe_divide(mu,mag,cutoff)
    
# @torch.jit.script
def torch_norm(a,dim=0,keepdim=False):
    """ Wrapper for torch.linalg.norm to handle ARM architecture. """
    if ARM: 
        #torch.linalg.norm not implemented on MPS yet
        # this is the fastest I have tested but still slow in comparison 
        return a.square().sum(dim=dim,keepdim=keepdim).sqrt()
    else:
        return torch.linalg.norm(a,dim=dim,keepdim=keepdim)

# @njit
# def safe_divide(num,den,cutoff=0):
#     """ Division ignoring zeros and NaNs in the denominator.""" 
#     # module = get_module(num) # assume num and den are the same type

#     # return np.divide(num, den, out=np.zeros_like(num), 
#     #                  where=np.logical_and(den>cutoff,~np.isnan(den)))        
    
#     if isinstance(num, np.ndarray):
#         return np.divide(num, den, out=np.zeros_like(num), 
#                          where=np.logical_and(den>cutoff,~np.isnan(den)))
#     elif isinstance(num, torch.Tensor):
#         return torch.where((den > cutoff) & torch.isfinite(den), num / den, torch.zeros_like(num))
#     else:
#         raise TypeError("num must be a numpy array or a PyTorch tensor")

# def safe_divide(num, den, cutoff=0):
#     """ Division ignoring zeros and NaNs in the denominator.""" 
#     module = get_module(num)
#     return module.where((den > cutoff) & module.isfinite(den) & (den>0), 
#                         module.divide(num, den), 
#                         module.zeros_like(num))

# def safe_divide(num, den, cutoff=0):
#     """ Division ignoring zeros and NaNs in the denominator.""" 
#     module = get_module(num)
#     valid_den = (den > cutoff) & module.isfinite(den) #isfinite catches both nan and inf
#     return module.divide(num, den, out=module.zeros_like(num,dtype=module.float64), where=valid_den, rounding_mode='trunc')


# def safe_divide(num, den, cutoff=0):
#     """ Division ignoring zeros and NaNs in the denominator.""" 
#     module = get_module(num)
#     valid_den = (den > cutoff) & module.isfinite(den) #isfinite catches both nan and inf

#     if module == np:
#         r = num.astype(np.float32, copy=False)
#     elif module == torch:
#         r = num.float()
#     else:
#         raise TypeError("num must be a numpy array or a PyTorch tensor")

#     # module.divide(r, den, out=r, where=valid_den)
#     print('gg',r.shape,den.shape)
#     r[valid_den] /= den[valid_den]
#     r[~valid_den] = 0
#     return r 

def safe_divide(num, den, cutoff=0):
    """ Division ignoring zeros and NaNs in the denominator.""" 
    module = get_module(num)
    valid_den = (den > cutoff) & module.isfinite(den) #isfinite catches both nan and inf

    if module == np:
        r = num.astype(np.float32, copy=False)
        r = np.divide(r, den, out=np.zeros_like(r), where=valid_den)
    elif module == torch:
        r = num.float()
        den = den.float()
        small_val = torch.finfo(den.dtype).tiny  # smallest positive representable number
        safe_den = torch.where(valid_den, den, small_val)
        r = torch.div(r, safe_den)
    else:
        raise TypeError("num must be a numpy array or a PyTorch tensor")

    return r

def bin_counts(data, num_bins=256):
    """Compute the counts of values in bins.

    Parameters:
    data (np.ndarray): Input data.
    num_bins (int): Number of bins.

    Returns:
    np.ndarray: Counts of values in each bin.
    """
    unique_values, counts = fastremap.unique(data, return_counts=True) # this only works on integer, e.g. raw images 
    bin_edges = np.linspace(unique_values.min(), unique_values.max(), num_bins+1)
    # bin_indices = np.digitize(unique_values, bin_edges)
    # binned_counts = np.bincount(bin_indices, weights=counts, minlength=num_bins+1)


    bin_indices = np.digitize(unique_values, bin_edges) - 1
    binned_counts = np.bincount(bin_indices, weights=counts, minlength=num_bins)

    # print(binned_counts.shape, bin_edges.shape)
    bin_start = bin_edges[:-1]

    # Ensure the shapes match
    binned_counts = binned_counts[:-1]
    
    return binned_counts, bin_start
    

from scipy.stats import gaussian_kde
def compute_density(x, y, bw_method=None):
    """Compute the density of points along a curve.

    Parameters:
    x (np.ndarray): x-coordinates of the points on the curve.
    y (np.ndarray): y-coordinates of the points on the curve.

    Returns:
    np.ndarray: Density of the points along the curve.
    """
    # Combine the x and y coordinates into a 2D array
    points = np.vstack([x, y])

    # Compute the KDE for the original points
    kde = gaussian_kde(points,bw_method=bw_method)
    density = kde(points)

    # Compute the KDE for the inverted points
    inverted_points = np.vstack([-x, y])
    inverted_kde = gaussian_kde(inverted_points,bw_method=bw_method)
    inverted_density = inverted_kde(inverted_points)

    # Take the average of the two densities
    symmetric_density = (density + inverted_density) / 2
    symmetric_density = rescale(symmetric_density)

    return symmetric_density
    
    
def qnorm(X, 
            nbins=100,
            bw_method=2, 
            density_cutoff=None, 
            density_quantile=[.001,.999],
            debug=False, 
            log=False,eps=1):
    # make it into an integer form that fasrtremap can work on  
    if X.dtype not in [np.uint8,np.uint16]:
        X = to_16_bit(X)
    counts, unique = bin_counts(X,nbins)
    # print('uu',np.std(unique)/np.mean(unique)) # curious this is the same for all images at same nbin 
    sel = counts>0
    counts = counts[sel]
    unique = unique[sel]
    x = np.arange(len(counts))
    if log:
        # x = np.log(unique+(unique==0))
        # y = np.log(counts+(counts==0))
        
        
        # x = np.log(unique+eps)
        y = np.log(counts+eps)
    else:
        y = counts
    
    d = compute_density(x,y,bw_method=bw_method)
    
    if not isinstance(density_quantile,list):
        density_quantile = [density_quantile,density_quantile]
    
    if density_cutoff is None:
        density_cutoff = np.quantile(d,density_quantile) 
        if debug: 
            print('dc',density_cutoff)
    elif not isinstance(density_cutoff,list):
        density_cutoff = [density_cutoff,density_cutoff]
        
    
    imin = np.argwhere(d>density_cutoff[0])[0][0]
    imax = np.argwhere(d>density_cutoff[1])[-1][0]
    vmin, vmax = unique[imin], unique[imax]
    
    scale_factor = 1.0 / (vmax - vmin) if vmax>vmin else 1.0
    r = X.astype(np.float32, copy=False)
    np.multiply(r, scale_factor, out=r)
    np.clip(r, 0, 1, out=r)

    if debug:
        return r, x, y, d, imin, imax, vmin, vmax
    else:
        return r


def normalize99(Y, lower=0.01, upper=99.99, contrast_limits=None, dim=None):
    """ normalize array/tensor so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile 
    Upper and lower percentile ranges configurable. 
    
    Parameters
    ----------
    Y: ndarray/tensor, float
        Input array/tensor. 
    upper: float
        upper percentile above which pixels are sent to 1.0
    
    lower: float
        lower percentile below which pixels are sent to 0.0
        
    contrast_limits: list, float (optional, override computation)
        list of two floats, lower and upper contrast limits
    
    Returns
    --------------
    normalized array/tensor with a minimum of 0 and maximum of 1
    
    """
    
    module = get_module(Y)
    
    if contrast_limits is None:
    
        quantiles = np.array([lower, upper]) / 100
        if module == torch:
            quantiles = torch.tensor(quantiles, dtype=Y.dtype, device=Y.device)
   
        if dim is not None:
            # Reshape Y into a 2D tensor for quantile computation
            Y_flattened = Y.reshape(Y.shape[dim], -1)

            lower_val, upper_val = module.quantile(Y_flattened, quantiles, axis=-1)        
            
            # Reshape back into original shape for broadcasting
            if dim == 0:
                lower_val = lower_val.reshape(Y.shape[dim], *([1] * (len(Y.shape) - 1)))
                upper_val = upper_val.reshape(Y.shape[dim], *([1] * (len(Y.shape) - 1)))
            else:
                lower_val = lower_val.reshape(*Y.shape[:dim], *([1] * (len(Y.shape) - dim - 1)))
                upper_val = upper_val.reshape(*Y.shape[:dim], *([1] * (len(Y.shape) - dim - 1)))
        else:
            # lower_val, upper_val = module.quantile(Y, quantiles)
            try:
                lower_val, upper_val = module.quantile(Y, quantiles)
            except RuntimeError:
                lower_val, upper_val = auto_chunked_quantile(Y, quantiles)

    else:
        if module == np:
            contrast_limits = np.array(contrast_limits)
        elif module == torch:
            contrast_limits = torch.tensor(contrast_limits)

        lower_val, upper_val = contrast_limits
        
    
    # Y = module.clip(Y, lower_val, upper_val) # is this needed? 
    # Y -= lower_val
    # Y /= (upper_val - lower_val)
    
    # return Y
    # return (Y-lower_val)/(upper_val-lower_val)
    # return module.clip((Y-lower_val)/(upper_val-lower_val),0,1)
    # return module.clip((Y-lower_val)/(upper_val-lower_val),0,1)
    # in this case, since lower_val is not the absolute minimum, but the lowerr quanitle, 
    # Y-lower_val can be less than zero. Likewise for the upward scalimg being slightly >1. 
    return module.clip(safe_divide(Y-lower_val,upper_val-lower_val),0,1)

def searchsorted(tensor, value):
    """Find the indices where `value` should be inserted in `tensor` to maintain order."""
    return (tensor < value).sum()



def compute_quantiles(sorted_array, lower=0.01, upper=0.99):
    """Compute a pair of quantiles of a sorted array.

    Parameters:
    sorted_array (np.ndarray): Input array sorted in ascending order.
    lower (float): Lower quantile to compute, which must be between 0 and 1 inclusive.
    upper (float): Upper quantile to compute, which must be between 0 and 1 inclusive.

    Returns:
    tuple: The lower and upper quantiles of the input array.
    """
    assert 0 <= lower <= 1, "Lower quantile must be between 0 and 1"
    assert 0 <= upper <= 1, "Upper quantile must be between 0 and 1"
    lower_index = int(lower * (len(sorted_array) - 1))
    upper_index = int(upper * (len(sorted_array) - 1))
    return sorted_array[lower_index], sorted_array[upper_index]

def quantile_rescale(Y, lower=0.0001, upper=0.9999, contrast_limits=None, bins=None):
   
    sorted_array = np.sort(Y.flatten(),kind='mergesort')
    lower_val, upper_val  = compute_quantiles(sorted_array, lower, upper)
    
    # return np.clip((Y - lower_val) / (upper_val - lower_val), 0, 1)
    
    # return np.clip(safe_divide(Y - lower_val, upper_val - lower_val), 0, 1)
    r = safe_divide(Y - lower_val, upper_val - lower_val)
    r [r<0] = 0
    r [r>1] = 1
    return r
    

    
    

def normalize99_hist(Y, lower=0.01, upper=99.99, contrast_limits=None, bins=None):
    """ normalize array/tensor using 1% and 99% quantiles
    
    Parameters
    ----------
    Y: ndarray/tensor, float
        Input array/tensor. 
    contrast_limits: list of float
        The lower and upper quantiles to use for normalization. Default is [0.01, 0.99].
    bins: int
        The number of bins to use for the histogram. Default is 1000.
    
    Returns
    --------------
    normalized array/tensor with values between 0 and 1
    
    """
    upper = upper/100
    lower = lower/100
    
    module = get_module(Y)
    if bins is None:
        if module == np:
            num_elements = Y.size
        elif module == torch:
            num_elements = Y.numel()
        bins = int(np.sqrt(num_elements))
        # bins = int(num_elements)
            
    # print(bins,num_elements,'bbv')
    if contrast_limits is None:
        # Estimate the quantiles using a histogram
        # if module == np:
        # elif module == torch:
        #     hist = torch.histc(Y, bins=bins)
        #     bin_edges = torch.linspace(Y.min(), Y.max(), steps=bins+1)

        hist, bin_edges = module.histogram(Y,bins=bins)
        # print(len(bin_edges))
        
        cdf = module.cumsum(hist, axis=0) / module.sum(hist)
        lower_val = bin_edges[searchsorted(cdf, lower)]
        upper_val = bin_edges[searchsorted(cdf, upper)]
    else:
        if module == np:
            contrast_limits = np.array(contrast_limits)
        elif module == torch:
            contrast_limits = torch.tensor(contrast_limits)

        lower_val, upper_val = contrast_limits
        
    # Normalize Y to the range [0, 1]
    # Y_normalized = module.clip((Y - lower_val) / (upper_val - lower_val), 0, 1)
    r = safe_divide(Y - lower_val, upper_val - lower_val)
    r [r<0] = 0
    r [r>1] = 1
    return r
    
    

# lol silent p, p-norm pun 
def pnormalize(Y, p_min=-1,p_max = 10):
    """ normalize array/tensor using p-norm
    
    Parameters
    ----------
    Y: ndarray/tensor, float
        Input array/tensor. 
    p: float
        The p value for the p-norm. Default is 2.
    
    Returns
    --------------
    normalized array/tensor with p-norm of 1
    
    """
    
    module = get_module(Y)
    
    # Compute the p-norm
    # upper_val = module.linalg.norm(Y, p_max)
    # lower_val = module.linalg.norm(Y, p_min)
    lower_val = (module.abs(Y*1.0)**p_min).sum()**(1./p_min)
    upper_val = (module.abs(Y*1.0)**p_max).sum()**(1./p_max)
        
    # print(upper_val,lower_val)
    
    return module.clip(safe_divide(Y-lower_val,upper_val-lower_val),0,1)



def auto_chunked_quantile(tensor, q):
    # Determine the maximum number of elements that can be handled by PyTorch's quantile function
    max_elements = 16e6 - 1  

    # Determine the number of elements in the tensor
    num_elements = tensor.nelement()

    # Determine the chunk size
    chunk_size = math.ceil(num_elements / max_elements)
    
    # Split the tensor into chunks
    chunks = torch.chunk(tensor, chunk_size)

    # Compute the quantile for each chunk
    return torch.stack([torch.quantile(chunk, q) for chunk in chunks]).mean(dim=0)

try:
    import numexpr as ne
except:
    pass
# from skimage.measure import label, regionprops_table

def normalize_image(im, mask, target=0.5, foreground=False, 
                    iterations=1, scale=1, channel_axis=0, per_channel=True):
    """
    Normalize image by rescaling from 0 to 1 and then adjusting gamma to bring 
    average background to specified value (0.5 by default).
    
    Parameters
    ----------
    im: ndarray, float
        input image or volume
        
    mask: ndarray, int or bool
        input labels or foreground mask
    
    target: float
        target background/foreground value in the range 0-1
    
    channel_axis: int
        the axis that contains the channels
    
    Returns
    --------------
    gamma-normalized array with a minimum of 0 and maximum of 1
    
    """
    # im = rescale(im) * scale
    # im = rescale(im).astype('float32') * scale
    im = im.astype('float32') * scale
    im_min = im.min()
    im_max = im.max()
    ne.evaluate("(im - im_min) / (im_max - im_min)",out=im)
    
    if im.ndim > 2:  # assume last axis is channel axis
        im = np.moveaxis(im, channel_axis, -1)  # move channels to last axis
    else:
        im = np.expand_dims(im, axis=-1)
        
    if not isinstance(mask, list):
        mask = np.expand_dims(mask, axis=-1)  # Add a new axis to mask
        mask = np.broadcast_to(mask, im.shape)  # Broadcast mask to the shape of im
        
    # for k in range(len(mask)):
    #     bin0 = binary_erosion(mask[k]>0 if foreground else mask[k] == 0, iterations=iterations) 
    #     source_target = np.mean(im[k][bin0])
    #     im[k] = im[k] ** (np.log(target) / np.log(source_target))
        
    bin0 = mask>0 if foreground else mask == 0
    if iterations > 0:
        # Create a structuring element that erodes only along the last two dimensions
        structure = np.ones((3,) * (im.ndim - 1) + (1,))
        structure[1, ...] = 0
        bin0 =  binary_erosion(bin0, structure=structure, iterations=iterations)
        
    # masked_im = np.ma.masked_array(im, mask=np.logical_not(bin0))
    # # source_target = np.ma.mean(masked_im, axis=(0,1) if per_channel else None) 
    # masked_im = im.copy()
    # masked_im[~bin0] = np.nan  # Replace masked values with nan

    # if per_channel:
    #     source_target = np.empty(im.shape[-1])  # Initialize array for mean values
    #     for i in range(im.shape[-1]):
    #         source_target[i] = np.nanmean(masked_im[..., i])
    # else:
    #     source_target = np.nanmean(masked_im)
    
    # Create a mask for the background
    # background_mask = ~bin0

    # Apply the mask to the image
    # masked_im = im.copy()
    # masked_im[bin0] = np.nan  # Replace background values with nan

    # # Compute the mean of the background values along the channel axis
    # source_target = np.apply_along_axis(np.nanmean, -1, masked_im)
        

    masked_im = im.copy()
    masked_im[~bin0] = np.nan
    source_target = np.nanmean(masked_im, axis=(0,1) if per_channel else None)
    source_target = source_target.astype('float32')
    target = np.array(target).astype('float32')
    # print(np.log(source_target).max(),'ss')
    # im = im ** (np.log(target) / np.log(source_target))
    # im **= (np.log(target) / np.log(source_target))
    ne.evaluate("im ** (log(target) / log(source_target))", out=im)
    # im = np.exp(np.log(im+1e-8) * np.log(target) / (np.log(source_target)))    
    # im = np.power(im,np.log(target) / np.log(source_target))
    return np.moveaxis(im, -1, channel_axis).squeeze()

import torch
from scipy.ndimage import binary_erosion

def gamma_normalize(im, mask, target=1.0, scale=1.0, foreground=True, iterations=0, per_channel=True, channel_axis=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    im = rescale(im) * scale
    if im.ndim > 2:  # assume last axis is channel axis
        im = np.moveaxis(im, channel_axis, -1)  # move channels to last axis
    else:
        im = np.expand_dims(im, axis=-1)
        
    if not isinstance(mask, list):
        mask = np.stack([mask] * im.shape[-1], axis=-1)
    im = torch.from_numpy(im).float().to(device)
    mask = torch.from_numpy(mask).float().to(device)


    bin0 = mask > 0 if foreground else mask == 0
    if iterations > 0:
        # Create a structuring element that erodes only along the last two dimensions
        structure = torch.ones((3,) * (im.ndim - 1) + (1,)).to(device)
        structure[1, ...] = 0
        bin0 = torch.from_numpy(binary_erosion(bin0.cpu().numpy(), structure=structure.cpu().numpy(), iterations=iterations)).to(device)

    masked_im = im.masked_fill(~bin0, float('nan'))
    source_target = torch.nanmean(masked_im, dim=(0,1) if per_channel else None)
    im **= (torch.log(target) / torch.log(source_target))

    return im.permute(*[channel_axis] + [i for i in range(im.ndim) if i != channel_axis]).squeeze().cpu().numpy()
    
import ncolor
def mask_outline_overlay(img,masks,outlines,mono=None):
    """
    Apply a color overlay to a grayscale image based on a label matrix.
    mono is a single color to use. Otherwise, N sinebow colors are used. 
    """
    if mono is None:
        m,n = ncolor.label(masks,max_depth=20,return_n=True)
        c = sinebow(n)
        colors = np.array(list(c.values()))[1:]
    else:
        colors = mono
        m = masks>0
    if img.ndim == 3:
        im = rescale(color.rgb2gray(img))
    else:
        im = img
    overlay = color.label2rgb(m,im,colors,
                              bg_label=0,
                              alpha=np.stack([((m>0)*1.+outlines*0.75)/3]*3,axis=-1))
    return overlay

def mono_mask_bd(masks,outlines,color=[1,0,0],a=0.25):
    m = masks>0
    alpha = (m>0)*a+outlines*(1-a)
    return np.stack([m*c for c in color]+[alpha],axis=-1)


def moving_average(x, w):
    return convolve1d(x,np.ones(w)/w,axis=0)


# def rescale(T,floor=None,ceiling=None):
#     """Rescale array between 0 and 1"""
#     if ceiling is None:
#         ceiling = T[:].max()
#     if floor is None:
#         floor = T[:].min()
#     T = np.interp(T, (floor, ceiling), (0, 1))
#     return T
    
def rescale(T, floor=None, ceiling=None, exclude_dims=None):
    """
    Rescale data between 0 and 1.
    exclude_dims is the axis or axes that will remain. 
    """
    
    module = get_module(T)
    if exclude_dims is not None:
        if isinstance(exclude_dims, int):
            exclude_dims = (exclude_dims,)
        order = [1 if i not in exclude_dims else -1 for i in range(T.ndim)]
        axes = tuple(i for i in range(T.ndim) if i not in exclude_dims)
    else:
        axes = None
                
    if ceiling is None:
        ceiling = module.amax(T, axis=axes)
        # print('\t',T.shape,ceiling,axes)
        if exclude_dims is not None:
            ceiling = ceiling.reshape(*order)
    if floor is None:
        floor = module.amin(T, axis=axes)
        if exclude_dims is not None:
            floor = floor.reshape(*order)
            
    T = safe_divide(T - floor, ceiling - floor)

    return T


def normalize_stack(vol,mask,bg=0.5,bright_foreground=None,
                    subtractive=False,iterations=1,equalize_foreground=1,quantiles=[0.01,0.99]):
    """
    Adjust image stacks so that background is 
    (1) consistent in brightness and 
    (2) brought to an even average via semantic gamma normalization.
    """
    # vol = rescale(vol)
    vol = vol.copy()
    # binarize background mask, recede from foreground, slice-wise to not erode in time
    kwargs = {'iterations':iterations} if iterations>1 else {}
    bg_mask = [binary_erosion(m==0,**kwargs) for m in mask] 
    # find mean backgroud for each slice
    bg_real = [np.nanmean(v[m]) for v,m in zip(vol,bg_mask)] 
    
    # automatically determine if foreground objects are bright or dark 
    if bright_foreground is None:
        bright_foreground = np.mean(vol[bg_mask]) < np.mean(vol[mask>0])
    
    # if smooth: 
    #     bg_real = moving_average(bg_real,5) 
    # some weird fluctuations happening with background being close to zero, but just on fluoresnece... might need to invert or go by foreground
    
    bg_min = np.min(bg_real) # get the minimum one, want to normalize by lowest one 
    
    # normalize wrt background
    if subtractive:
        vol = np.stack([safe_divide(v-bg_r,bg_min) for v,bg_r in zip(vol,bg_real)]) 
    else:
        vol = np.stack([v*safe_divide(bg_min,bg_r) for v,bg_r in zip(vol,bg_real)]) 
    # print('mm',vol.min(),vol.max(),bright_foreground)
    # equalize foreground signal
    if equalize_foreground:
        q1,q2 = quantiles
    
        if bright_foreground:
            fg_real = [np.percentile(v[m>0],99.99) for v,m in zip(vol,mask)]
            # fg_real = [v.max() for v,m in zip(vol,bg_mask)]    
            floor = np.percentile(vol[bg_mask],0.01)
            vol = [rescale(v,ceiling=f, floor=floor) for v,f in zip(vol,fg_real)]
        else:
            fg_real = [np.quantile(v[m>0],q1) for v,m in zip(vol,mask)]
            # fg_real = [.5]*(len(vol))
            # ceiling = np.percentile(vol[bg_mask],99.99)
            
            # print('hh',np.any(np.stack(fg_real)<0),np.any(np.stack(fg_real)>ceiling),ceiling,np.mean(fg_real))
            # vol = [rescale(v,ceiling=ceiling,floor=f) for v,f in zip(vol,fg_real)]
            # ceiling =  [np.percentile(v[m],99.99) for v,m in zip(vol,mask==0)]#bg_mask
            ceiling =  np.quantile(vol,q2,axis=(-2,-1))
            vol = [np.interp(v,(f, c), (0, 1)) for v,f,c in zip(vol,fg_real,ceiling)]
            
    # print([(np.max(v),np.min(v)) for v,bg_m in zip(vol,bg_mask)])
    vol = np.stack(vol)
    
    # vol = rescale(vol) # now rescale by overall min and max 
    vol = np.stack([v**(np.log(bg)/np.log(np.mean(v[bg_m]))) for v,bg_m in zip(vol,bg_mask)]) # now can gamma normalize 
    return vol

def is_integer(var):
    return isinstance(var, int) or isinstance(var, np.integer) or (isinstance(var, torch.Tensor) and var.is_integer())

def bbox_to_slice(bbox,shape,pad=0,im_pad=0):
    """
    return the tuple of slices for cropping an image based on the skimage.measure bounding box
    optional padding allows for the bounding box to be expanded, but not outside the original image dimensions 
    
    Parameters
    ----------
    bbox: ndarray, float
        input bounding box, e.g. [y0,x0,y1,x1]
        
    shape: array, tuple, or list, int
        shape of corresponding array to be sliced
    
    pad: array, tuple, or list, int
        padding to be applied to each axis of the bounding box
        can be a common padding (5 means 5 on every side) 
        or a list of each axis padding ([3,4] means 3 on y and 4 on x).
        N-volume requires an N-tuple. 
        
    im_pad: int
        region around the edges to avoid (pull back coordinate limits)
    
    Returns
    --------------
    tuple of slices 
    
    """
    dim = len(shape)
    # if type(pad) is int:
    if is_integer(pad):
        pad = [pad]*dim
    # if type(im_pad) is int:
    if is_integer(im_pad):
        im_pad = [im_pad]*dim
    # return tuple([slice(int(max(0,bbox[n]-pad[n])),int(min(bbox[n+dim]+pad[n],shape[n]))) for n in range(len(bbox)//2)])
    # added a +1 to stop, might be a necessary fix but not sure yet 
    # print('im_pad',im_pad, bbox, pad, shape)
    one = 0
    return tuple([slice(int(max(im_pad[n],bbox[n]-pad[n])),
                        int(min(bbox[n+dim]+pad[n]+one,shape[n]-im_pad[n]))) 
                  for n in range(len(bbox)//2)])
    

def crop_bbox(mask, pad=10, iterations=3, im_pad=0, area_cutoff=0, 
              max_dim=np.inf, get_biggest=False, binary=False):
    """Take a label matrix and return a list of bounding boxes identifying clusters of labels.
    
    Parameters
    --------------

    mask: matrix of integer labels
    pad: amount of space in pixels to add around the label (does not extend beyond image edges, will shrink for consistency)
    iterations: number of dilation iterations to merge labels separated by this number of pixel or less
    im_pad: amount of space to subtract off the label matrix edges
    area_cutoff: label clusters below this area in square pixels will be ignored
    max_dim: if a cluster is above this cutoff, quit and return the original image bounding box
    

    Returns
    ---------------

    slices: list of bounding box slices with padding 
    
    """
    bw = binary_dilation(mask>0,iterations=iterations) if iterations> 0 else mask>0
    clusters = measure.label(bw)
    regions = measure.regionprops(clusters)
    sz = mask.shape
    d = mask.ndim
    # ylim = [im_pad,sz[0]-im_pad]
    # xlim = [im_pad,sz[1]-im_pad]
    
    slices = []
    if get_biggest:
        w = np.argmax([props.area for props in regions])
        bbx = regions[w].bbox 
        minpad = min(pad,bbx[0],bbx[1],sz[0]-bbx[2],sz[1]-bbx[3])
        # print(pad,bbx[0],bbx[1],sz[0]-bbx[2],sz[1]-bbx[3])
        # print(minpad,sz,bbx)
        slices.append(bbox_to_slice(bbx,sz,pad=minpad,im_pad=im_pad))
        
    else:
        for props in regions:
            if props.area>area_cutoff:
                bbx = props.bbox 
                minpad = min(pad,bbx[0],bbx[1],sz[0]-bbx[2],sz[1]-bbx[3])
                # print(minpad,'m',im_pad)
                slices.append(bbox_to_slice(bbx,sz,pad=minpad,im_pad=im_pad))
    
    # merge into a single slice 
    if binary:
        start_xy = np.min([[slc[i].start for i in range(d)] for slc in slices],axis=0)
        stop_xy = np.max([[slc[i].stop for i in range(d)] for slc in slices],axis=0)
        slices = tuple([slice(start,stop) for start,stop in zip(start_xy,stop_xy)])
    
    return slices


def get_boundary(mask):
    """ND binary mask boundary using mahotas.
    
    Parameters
    ----------
    mask: ND array, bool
        binary mask
    
    Returns
    --------------
    Binary boundary map
    
    """
    return np.logical_xor(mask,mh.morph.erode(mask))

# Omnipose version of remove_edge_masks, need to merge (this one is more flexible)
def clean_boundary(labels, boundary_thickness=3, area_thresh=30, cutoff=0.5):
    """Delete boundary masks below a given size threshold within a certain distance from the boundary. 
    
    Parameters
    ----------
    boundary_thickness: int
        labels within a stripe of this thickness along the boundary will be candidates for removal. 
        
    area_thresh: int
        labels with area below this value will be removed. 
        
    cutoff: float
        Fraction from 0 to 1 of the overlap with the boundary before the mask is removed. Default 0.5. 
        Set to 0 if you want any mask touching the boundary to be removed. 
    
    Returns
    --------------
    label matrix with small edge labels removed. 
    
    """
    border_mask = np.zeros(labels.shape, dtype=bool)
    border_mask = binary_dilation(border_mask, border_value=1, iterations=boundary_thickness)
    clean_labels = np.copy(labels)

    for cell_ID in fastremap.unique(labels[border_mask])[1:]:
        mask = labels==cell_ID 
        area = np.count_nonzero(mask)
        overlap = np.count_nonzero(np.logical_and(mask, border_mask))
        if overlap > 0 and area<area_thresh and overlap/area >= cutoff: #only remove cells that are X% or more edge px
            clean_labels[mask] = 0

    return clean_labels

# This function takes a few milliseconds for a typical image 
def get_edge_masks(labels,dists):
    """Finds and returns masks that are largely cut off by the edge of the image.
    
    This function loops over all masks touching the image boundary and compares the 
    maximum value of the distance field along the boundary to the top quartile of distance
    within the mask. Regions whose edges just skim the image edge will not be classified as 
    an "edge mask" by this criteria, whereas masks cut off in their center (where distance is high)
    will be returned as part of this output. 
    
    Parameters
    ----------
    labels: ND array, int
        label matrix
        
    dists: ND array, float
        distance field (calculated with reflection padding of labels)
    
    Returns
    --------------
    clean_labels: ND array, int
        label matrix of all cells qualifying as 'edge masks'
    
    """
    border_mask = np.zeros(labels.shape, dtype=bool)
    border_mask = binary_dilation(border_mask, border_value=1, iterations=1)
    clean_labels = np.zeros_like(labels)
    
    for cell_ID in fastremap.unique(labels[border_mask])[1:]:
        mask = labels==cell_ID 
        max_dist = np.max(dists[np.logical_and(mask, border_mask)])
        # mean_dist = np.mean(dists[mask])
        dist_thresh = np.percentile(dists[mask],75) 
        # sort of a way to say the skeleton isn't touching the boundary
        # top 25%

        if max_dist>=dist_thresh: # we only want to keep cells whose distance at the boundary is not too small
            clean_labels[mask] = cell_ID
            
    return clean_labels



def border_indices(tyx):
    """Return flat indices of border values in ND. Use via A.flat[border_indices]."""
    dim_indices = [np.arange(dim_size) for dim_size in tyx]
    dim_indices = np.meshgrid(*dim_indices, indexing='ij')
    dim_indices = [indices.ravel() for indices in dim_indices]
    
    indices = []
    for i in range(len(tyx)):
        for j in [0, tyx[i] - 1]:
            mask = (dim_indices[i] == j)
            indices.append(np.where(mask)[0])
    return np.concatenate(indices)


# @njit 
# def get_neighbors(coords, steps, dim, shape, edges=None, pad=0):
#     print('this version actually a lot slower than below ')
#     if edges is None:
#         edges = [np.array([-1,s]) for s in shape]
        
#     npix = coords[0].shape[-1]
#     neighbors = np.empty((dim, len(steps), npix), dtype=np.int64)
#     for d in range(dim):
#         for i, s in enumerate(steps):
#             for j in range(npix):
#                 if  ((coords[d][j] + s[d]) in edges[d]) and ((coords[d][j] + 2*s[d]) not in edges[d]):     
#                     neighbors[d,i,j] = coords[d][j]
#                 else:
#                     neighbors[d,i,j] = coords[d][j] + s[d]
    
#     return neighbors


# much faster 
# @njit
# def isin_numba(x, y):
#     result = np.zeros(x.shape, dtype=np.bool_)
#     for i in range(x.size):
#         result[i] = x[i] in y
#     return result

# @njit
# def get_neighbors(coords, steps, dim, shape, edges=None):
#     if edges is None:
#         edges = [np.array([-1,s]) for s in shape]
        
#     npix = coords[0].shape[-1]
#     neighbors = np.empty((dim, len(steps), npix), dtype=np.int64)
    
#     for d in range(dim):
#         for i, s in enumerate(steps):
#             X = coords[d] + s[d]
#             mask = np.logical_and(isin_numba(X, edges[d]), ~isin_numba(X+s[d], edges[d]))
#             neighbors[d,i] = np.where(mask, coords[d], X)
#     return neighbors


# slightly faster than the jit code!
def get_neighbors(coords, steps, dim, shape, edges=None, pad=0):
    """
    Get the coordinates of all neighbor pixels. 
    Coordinates of pixels that are out-of-bounds get clipped. 
    """
    if edges is None:
        edges = [np.array([-1+pad,s-pad]) for s in shape]
        
    npix = coords[0].shape[-1]
    neighbors = np.empty((dim, len(steps), npix), dtype=np.int64)
    
    for d in range(dim):        
        S = steps[:,d].reshape(-1, 1)
        X = coords[d] + S
        # mask = np.logical_and(np.isin(X, edges[d]), ~np.isin(X+S, edges[d]))

        # out of bounds is where the shifted coordinate X is in the edges list
        # that second criterion might have been for my batched stuff 
        # oob = np.logical_and(np.isin(X, edges[d]), ~np.isin(X+S, edges[d]))
        oob = np.isin(X, edges[d])
        # print('checkme f', pad,np.sum(oob))

        C = np.broadcast_to(coords[d], X.shape)
        neighbors[d] = np.where(oob, C, X)
        # neighbors[d] = X

    return neighbors
    
def get_neighbors_torch(input, steps):
    """This version not yet used/tested."""
    # Get dimensions
    B, D, *DIMS = input.shape
    nsteps = steps.shape[0]

    # Compute coordinates
    coordinates = torch.stack(torch.meshgrid([torch.arange(dim) for dim in DIMS]), dim=0)
    coordinates = coordinates.unsqueeze(0).expand(B, *[-1]*(D+1))  # Add batch dimension and repeat for batch

    # Compute shifted coordinates
    steps = steps.unsqueeze(-1).unsqueeze(-1).expand(nsteps, D, *DIMS).to(input.device)
    shifted_coordinates = (coordinates.unsqueeze(1) + steps.unsqueeze(0))

    # Clamp shifted_coordinates in-place
    for d in range(D):
        shifted_coordinates[:, :, d].clamp_(min=0, max=DIMS[d]-1)

    return shifted_coordinates

# this version works without padding, should ultimately replace the other one in core
# @njit
def get_neigh_inds(neighbors,coords,shape,background_reflect=False):
    """
    For L pixels and S steps, find the neighboring pixel indexes 
    0,1,...,L for each step. Background index is -1. Returns:
    
    
    Parameters
    ----------
    coords: tuple, int
        coordinates of nonzero pixels, <dim>x<npix>

    shape: tuple, int
        shape of the image array

    Returns
    -------
    indexes: 1D array
        list of pixel indexes 0,1,...L-1
        
    neigh_inds: 2D array
        SxL array corresponding to affinity graph
    
    ind_matrix: ND array
        indexes inserted into the ND image volume
    """
    neighbors = tuple(neighbors) # just in case I pass it as ndarray

    npix = neighbors[0].shape[-1]
    indexes = np.arange(npix)
    ind_matrix = -np.ones(shape,int)
    
    ind_matrix[tuple(coords)] = indexes
    neigh_inds = ind_matrix[neighbors]
    
    # If needed, we can do a similar thing I do at boundaries and make neighbor
    # references to background redirect back to the edge pixel. However, this should 
    # not be default, since I rely on accurate neighbor indices later to test for background
    # So, probably better to do this sort of thing while contructing the affinity graph itself 
    if background_reflect:
        oob = np.nonzero(neigh_inds==-1) # 2 x nbad , pos 0 is the 0-step inds and pos 1 is the npix inds 
        neigh_inds[oob] = indexes[oob[1]] # reflect back to itself 
        ind_matrix[neighbors] = neigh_inds # update ind matrix as well

        # should I also update neighbor coordinate array? No, that's more fixed. 
        # index points to the correct coordinate. 
    
    # not sure if -1 is general enough, probbaly should be since other adjacent masks will be unlinked
    # can test it by adding some padding to the concatenation...
    
    # also, the reflections should be happening at edges of the image, but it is not? 
    
    return indexes, neigh_inds, ind_matrix


# This might need some reflection added in for it to work
# also might need generalization to include cleaned mask pixels getting dropped  
def subsample_affinity(augmented_affinity,slc,mask):
    """
    Helper function to subsample an affinity graph according to an image crop slice 
    and a foreground selection mask. 

    Parameters
    ----------
    augmented_affinity: NDarray, int64
        Stacked neighbor coordinate array and affinity graph. For dimension d, 
        augmented_affinity[:d] are the neighbor coordinates of shape (d,3**d,npix)
        and augmented_affinity[d] is the affinity graph of shape (3**d,npix). 

    slc: tuple, slice
        tuple of slices along each dimension defining the crop window
        
    mask: NDarray, bool
        foreground selection mask, in the image space of the original graph
        (i.e., not already sliced)

    Returns
    --------
    Augmented affinity graph corresponding to the cropped/masked region. 
    
    """

    # From the augmented affinity graph we can extract a lot
    nstep = augmented_affinity.shape[1]
    dim = len(slc) # dimension 
    neighbors = augmented_affinity[:dim]
    affinity_graph = augmented_affinity[dim]
    idx = nstep//2
    coords = neighbors[:,idx]
    in_bounds = np.all(np.vstack([[c<s.stop, c>=s.start] for c,s in zip(coords,slc)]),axis=0)
    in_mask = mask[tuple(coords)]>0
    
    in_mask_and_bounds = np.logical_and(in_bounds,in_mask)

    inds_crop = np.nonzero(in_mask_and_bounds)[0]
    
    # print('y',len(inds_crop),np.sum(in_mask_and_bounds), np.sum(in_bounds), np.sum(in_mask))

    if len(inds_crop):    
        crop_neighbors = neighbors[:,:,inds_crop]
        affinity_crop = affinity_graph[:,inds_crop]
    
        # shift coordinates back acording to the lower bound of the slice 
        # also refect at edges of the new bounding box
        edges = [np.array([-1,s.stop-s.start]) for s in slc]
        steps = get_steps(dim)
        
        # I should see if I can get this batched somehow... 
        for d in range(dim):        
            crop_coords = coords[d,inds_crop] - slc[d].start
            S = steps[:,d].reshape(-1, 1)
            X = crop_coords + S # cropped coordinates 
            # edgemask = np.logical_and(np.isin(X, edges[d]), ~np.isin(X+S, edges[d]))
            edgemask = np.isin(X, edges[d])
            # print('checkthisttoo')

            C = np.broadcast_to(crop_coords, X.shape)
            crop_neighbors[d] = np.where(edgemask, C, X)

        #return augmented affinity 
        return np.vstack((crop_neighbors,affinity_crop[np.newaxis]))
    else:
        e = np.empty((dim+1,nstep,0),dtype=augmented_affinity.dtype)
        return e, []

@functools.lru_cache(maxsize=None) 
def get_steps(dim):
    """
    Get a symmetrical list of all 3**N points in a hypercube represented
    by a list of all possible sequences of -1, 0, and 1 in ND.
    
    1D: [[-1],[0],[1]]
    2D: [[-1, -1],
         [-1,  0],
         [-1,  1],
         [ 0, -1],
         [ 0,  0],
         [ 0,  1],
         [ 1, -1],
         [ 1,  0],
         [ 1,  1]]
    
    The opposite pixel at index i is always found at index -(i+1). The number
    of possible face, edge, vertex, etc. connections grows exponentially with
    dimension: 3 steps in 1D, 9 steps in 3D, 3**N in ND. 
    """
    neigh = [[-1,0,1] for i in range(dim)]
    steps = cartesian(neigh) # all the possible step sequences in ND
    return steps

# @functools.lru_cache(maxsize=None)
def steps_to_indices(steps):
    """
    Get indices of the hupercubes sharing m-faces on the central n-cube. These
    are sorted by the connectivity (by center, face, edge, vertex, ...). I.e.,
    the central point index is first, followed by cardinal directions, ordinals,
    and so on. 
    """
     # each kind of m-face can be categorized by the number of steps to get there
    sign = np.sum(np.abs(steps),axis=1)
    
    # we want to bin them into groups 
    # E.g., in 2D: [4] (central), [1,3,5,7] (cardinal), [0,2,6,8] (ordinal)
    uniq = fastremap.unique(sign)
    inds = [np.where(sign==i)[0] for i in uniq] 
    
    # weighting factor for each hypercube group (distance from central point)
    fact = np.sqrt(uniq) 
    return inds, fact, sign

# [steps[:idx],steps[idx+1:]] can give the other steps 
@functools.lru_cache(maxsize=None) 
def kernel_setup(dim):
    """
    Get relevant kernel information for the hypercube of interest. 
    Calls get_steps(), steps_to_indices(). 
    
    Parameters
    ----------

    dim: int
        dimension (usually 2 or 3, but can be any positive integer)
    
    Returns
    -------
    steps: ndarray, int 
        list of steps to each kernal point
        see get_steps()
        
    idx: int
        index of the central point within the step list
        this is always (3**dim)//2
        
    inds: ndarray, int
        list of kernel points sorted by type
        see  steps_to_indices()
    
    fact: float
        list of face/edge/vertex/... distances 
        see steps_to_indices()
        
    sign: 1D array, int
        signature distinguishing each kind of m-face via the number of steps
        see steps_to_indices()

    
    """
    steps = get_steps(dim)
    inds, fact, sign = steps_to_indices(steps)
    idx = inds[0][0] # the central point is always first 
    return steps,inds,idx,fact,sign
    
    
    
# not acutally used in the code, typically use steps_to_indices etc. 
def cubestats(n):
    """
    Gets the number of m-dimensional hypercubes connected to the n-cube, including itself. 
    
    Parameters
    ----------
    n: int
        dimension of hypercube
    
    Returns
    -------
    List whose length tells us how many hypercube types there are (point/edge/pixel/voxel...) 
    connected to the central hypercube and whose entries denote many there in each group. 
    E.g., a square would be n=2, so cubestats returns [4, 4, 1] for four points (m=0), 
    four edges (m=1), and one face (the original square,m=n=2). 
    
    """
    faces = []
    for m in range(n+1):
          faces.append((2**(n-m))*math.comb(n,m))
    return faces


def curve_filter(im,filterWidth=1.5):
    """
    curveFilter : calculates the curvatures of an image.

     INPUT : 
           im : image to be filtered
           filterWidth : filter width
     OUTPUT : 
           M_ : Mean curvature of the image without negative values
           G_ : Gaussian curvature of the image without negative values
           C1_ : Principal curvature 1 of the image without negative values
           C2_ : Principal curvature 2 of the image without negative values
           M : Mean curvature of the ima ge
           G : Gaussian curvature of the image
           C1 : Principal curvature 1 of the image
           C2 : Principal curvature 2 of the image
           im_xx : \del^2 x / \del x^2
           im_yy : \del^2 x / \del y^2
           im_xy : \del^2 x / \del x \del y

    """
    shape = [np.floor(7*filterWidth) //2 *2 +1]*2 # minor modification is to make this odd
    
    m,n = [(s-1.)/2. for s in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    v = filterWidth**2
    gau = 1/(2*np.pi*v) * np.exp( -(x**2 + y**2) / (2.*v) )
    
    
    f_xx = ((x/v)**2-1/v)*gau
    f_yy = ((y/v)**2-1/v)*gau
    f_xy = y*x*gau/v**2
    
    im_xx = convolve(im, f_xx, mode='nearest')
    im_yy = convolve(im, f_yy, mode='nearest')
    im_xy = convolve(im, f_xy, mode='nearest')
    
    # gaussian curvature
    G = im_xx*im_yy-im_xy**2

    # mean curvature
    M = -(im_xx+im_yy)/2

    # compute principal curvatures
    C1 = (M-np.sqrt(np.abs(M**2-G)));
    C2 = (M+np.sqrt(np.abs(M**2-G)));

    
    # remove negative values
    G_ = G.copy()
    G_[G<0] = 0;

    M_ = M.copy()
    M_[M<0] = 0

    C1_ = C1.copy()
    C1_[C1<0] = 0

    C2_ = C2.copy()
    C2_[C2<0] = 0

    return M_, G_, C1_, C2_, M, G, C1, C2, im_xx, im_yy, im_xy


def rotate(V,theta,order=1,output_shape=None,center=None):
    
    dim = V.ndim
    v1 = np.array([0]*(dim-1)+[1])
    v2 = np.array([0]*(dim-2)+[1,0])

    s_in = V.shape
    if output_shape is None:
        s_out = s_in
    else:
        s_out = output_shape
    M = mgen.rotation_from_angle_and_plane(np.pi/2-theta,v2,v1)
    if center is None:
        c_in = 0.5 * np.array(s_in) 
    else:
        c_in = center
    c_out = 0.5 * np.array(s_out)
    offset = c_in  - np.dot(np.linalg.inv(M), c_out)
    V_rot = affine_transform(V, np.linalg.inv(M), offset=offset, 
                                           order=order, output_shape=output_shape)

    return V_rot



# make a list of all sprues 
from sklearn.utils.extmath import cartesian
from scipy.ndimage import binary_hit_or_miss


import mahotas as mh

def get_spruepoints(bw):
    d = bw.ndim
    idx = (3**d)//2 # the index of the center pixel is placed here when considering the neighbor kernel 
    neigh = [[-1,0,1] for i in range(d)]
    steps = cartesian(neigh) # all the possible step sequences in ND
    sign = np.sum(np.abs(steps),axis=1) # signature distinguishing each kind of m-face via the number of steps 
    
    hits = np.zeros_like(bw)
    mid = tuple([1]*d) # kernel 3 wide in every axis, so middle is 1
    
    # alt
    substeps = np.array(list(set([tuple(s) for s in steps])-set([(0,)*d]))) # remove zero shift element 
    # substeps = steps.copy()
    for step in substeps:
        oppose = np.array([np.dot(step,s) for s in steps])
        
        sprue = np.zeros([3]*d,dtype=int) # allocate matrix
        sprue[tuple(mid-step)] = 1
        sprue[mid] = 1
        
        miss = np.zeros([3]*d,dtype=int)
        for idx in np.argwhere(np.logical_and(oppose>=0,sign!=0)).flatten():
            c = tuple(steps[idx]+1)
            miss[c] = 1

        
        hitmiss = 2 - 2*miss - sprue
        
        # mahotas hitmiss is far faster than ndimage 
        hm = mh.morph.hitmiss(bw,hitmiss)

        hits = hits+hm
        
    return hits>0

def localnormalize(im,sigma1=2,sigma2=20):
    im = normalize99(im)
    blur1 = gaussian_filter(im,sigma=sigma1)
    num = im - blur1
    blur2 = gaussian_filter(num*num, sigma=sigma2)
    den = np.sqrt(blur2)
    
    return normalize99(num/den+1e-8)
    
import torchvision.transforms.functional as TF
def localnormalize_GPU(im, sigma1=2, sigma2=20):
    im = normalize99(im)
    kernel_size1 = round(sigma1 * 6)
    kernel_size1 += kernel_size1 % 2 == 0
    blur1 = TF.gaussian_blur(im, kernel_size1, sigma1)
    num = im - blur1
    kernel_size2 = round(sigma2 * 6)
    kernel_size2 += kernel_size2 % 2 == 0
    blur2 = TF.gaussian_blur(num*num, kernel_size2, sigma2)
    den = torch.sqrt(blur2)

    return normalize99(num/den+1e-8)

# from https://stackoverflow.com/questions/47370718/indexing-numpy-array-by-a-numpy-array-of-coordinates
def ravel_index(b, shp):
    return np.concatenate((np.asarray(shp[1:])[::-1].cumprod()[::-1],[1])).dot(b)

# https://stackoverflow.com/questions/31544129/extract-separate-non-zero-blocks-from-array
def find_nonzero_runs(a):
    # Create an array that is 1 where a is nonzero, and pad each end with an extra 0.
    isnonzero = np.concatenate(([0], (np.asarray(a) != 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isnonzero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

# @njit
# def remap_pairs(pairs: set[tuple[int, int]], mapping: dict[int, int]) -> set[tuple[int, int]]:
#     remapped_pairs = set()
#     for x, y in pairs:
#         remapped_x = mapping.get(x, x)
#         remapped_y = mapping.get(y, y)
#         remapped_pairs.add((remapped_x, remapped_y))
#     return remapped_pairs

# from numba import jit

# @jit(nopython=True)
# def remap_pairs(pairs, mapping):
#     remapped_pairs = set()
#     for x, y in pairs:
#         remapped_x = mapping.get(x, x)
#         remapped_y = mapping.get(y, y)
#         remapped_pairs.add((remapped_x, remapped_y))
#     return remapped_pairs

from numba import njit
from numba import types

@njit
def remap_pairs(pairs, replacements):
    remapped_pairs = set()
    for x, y in pairs:
        for a, b in replacements:
            if x == a:
                x = b
            if y == a:
                y = b
        remapped_pairs.add((x, y))
    return remapped_pairs

# need to comment out any njit code that I do not use...
@njit
def add_gaussian_noise(image, mean=0, var=0.01):
    shape = image.shape
    noise = np.random.normal(mean, var**0.5, shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)  # Clip values to [0, 1] range
    return noisy_image

def add_poisson_noise(image):
    noisy_image = np.random.poisson(image)
    noisy_image = np.clip(noisy_image, 0, 1)  # Clip values to [0, 1] range
    return noisy_image


from scipy.ndimage import gaussian_filter, convolve

def extract_skeleton(distance_field):
    # Smooth the distance field using Gaussian filter
    smoothed_field = gaussian_filter(distance_field, sigma=1)

    # Compute gradient of the smoothed distance field
    gradients = np.gradient(smoothed_field)

    # Compute Hessian matrix components
    hessians = []
    for i in range(len(distance_field.shape)):
        hessian = np.gradient(gradients[i])
        hessians.append(hessian)

    hessians = [gaussian_filter(hessian, sigma=1) for hessian in hessians]

    # Compute the Laplacian of Gaussian (LoG)
    log = np.sum(hessians, axis=0)

    # Find stationary points (zero-crossings) in the LoG
    zero_crossings = np.logical_and(log[:-1] * log[1:] < 0, np.abs(log[:-1] - log[1:]) > 0.02)

    # Thin the zero-crossings to get the skeleton
    skeleton = thin_skeleton(zero_crossings)

    return skeleton

def thin_skeleton(image):
    # DTS thinning algorithm
    dimensions = len(image.shape)
    neighbors = np.ones((3,) * dimensions, dtype=bool)
    neighbors[tuple([1] * dimensions)] = False

    while True:
        marker = np.zeros_like(image)

        # Convolve the image with the neighbors template
        convolution = convolve(image, neighbors, mode='constant')

        # Find the pixels where the convolution equals the number of neighbors
        marker[np.where(convolution == np.sum(neighbors))] = 1

        if np.sum(marker) == 0:
            break

        image = np.logical_and(image, np.logical_not(marker))

    return image

def save_nested_list(file_path, nested_list):
    """Helper function to save affinity graphs."""
    np.savez_compressed(file_path, *nested_list)

def load_nested_list(file_path):
    """Helper function to load affinity graphs."""
    loaded_data = np.load(file_path,allow_pickle=True)
    loaded_nested_list = []
    for key in loaded_data.keys():
        loaded_nested_list.append(loaded_data[key])
    return loaded_nested_list

import torch.nn.functional as F
def hysteresis_threshold(image, low, high):
    """
    Pytorch implementation of skimage.filters.apply_hysteresis_threshold(). 
    Discprepencies occur for very high thresholds/thin objects. 
    
    """
    # Ensure the image is a torch tensor
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)

    # Create masks for values greater than low and high thresholds
    mask_low = image > low
    mask_high = image > high

    # Initialize thresholded tensor
    thresholded = mask_low.clone()

    # Create hysteresis kernel
    spatial_dims = len(image.shape) - 2
    kernel_size = [3] * spatial_dims
    hysteresis_kernel = torch.ones([1, 1] + kernel_size, device=image.device, dtype=image.dtype)

    # Hysteresis thresholding
    thresholded_old = torch.zeros_like(thresholded)
    while (thresholded_old != thresholded).any():
        if spatial_dims == 2:
            hysteresis_magnitude = F.conv2d(thresholded.float(), hysteresis_kernel, padding=1)
        elif spatial_dims == 3:
            hysteresis_magnitude = F.conv3d(thresholded.float(), hysteresis_kernel, padding=1)
        else:
            raise ValueError(f'Unsupported number of spatial dimensions: {spatial_dims}')

        # thresholded_old = thresholded.clone()
        thresholded_old.copy_(thresholded)
        thresholded = ((hysteresis_magnitude > 0) & mask_low) | mask_high


    # sum_old = thresholded.sum()
    # while True:
    #     if spatial_dims == 2:
    #         hysteresis_magnitude = F.conv2d(thresholded.float(), hysteresis_kernel, padding=1)
    #     elif spatial_dims == 3:
    #         hysteresis_magnitude = F.conv3d(thresholded.float(), hysteresis_kernel, padding=1)
    #     else:
    #         raise ValueError(f'Unsupported number of spatial dimensions: {spatial_dims}')

    #     thresholded = ((hysteresis_magnitude > 0) & mask_low) | mask_high
    #     sum_new = thresholded.sum()

    #     if sum_new == sum_old:
    #         break

        # sum_old = sum_new
    return thresholded.bool()#, mask_low, mask_high


def correct_illumination(img,sigma=5):
    # Apply a Gaussian blur to the image
    blurred = gaussian_filter(img, sigma=sigma)

    # Normalize the image
    return (img - blurred) / np.std(blurred)