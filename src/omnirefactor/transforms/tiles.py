import itertools
import logging
import math

import numpy as np
import torch

from .imports import get_module, normalize99, rescale

omnipose_logger = logging.getLogger(__name__)


def pad_image_ND(img0, div=16, extra=1, dim=2):
    inds = [k for k in range(-dim, 0)]
    Lpad = [int(div * np.ceil(img0.shape[i] / div) - img0.shape[i]) for i in inds]
    pad1 = [extra * div // 2 + Lpad[k] // 2 for k in range(dim)]
    pad2 = [extra * div // 2 + Lpad[k] - Lpad[k] // 2 for k in range(dim)]

    emptypad = tuple([[0, 0]] * (img0.ndim - dim))
    pads = emptypad + tuple(np.stack((pad1, pad2), axis=1))
    I = np.pad(img0, pads, mode="reflect")

    shape = img0.shape[-dim:]
    subs = [np.arange(pad1[k], pad1[k] + shape[k]) for k in range(dim)]
    return I, subs

 

def get_flip(idx):
    """
    ND slices for flipping arrays along particular axes 
    based on the tile indices. Used in augment_tiles_ND()
    and unaugment_tiles_ND(). 
    """
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
    """Reverse test-time augmentations for averaging.

    Parameters
    ----------
    y : float32
        Array of shape ``(ntiles, nchan, *DIMS)``.
    unet : bool (optional, False)
        Whether output is from a plain U-Net (no flow unflipping needed).

    Returns
    -------
    y : float32
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
        network output for each tile

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
        mask = torch.tensor(mask,**params)
        
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
                omnipose_logger.info('[Tile] Now normalizing each tile separately.')
                IMG = normalize99(IMG,dim=0)
            else:
                omnipose_logger.info('[Tile] Rescaling stack as a whole.')
                IMG = rescale(IMG) # why not use percentile?
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
