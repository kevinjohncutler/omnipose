import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from scipy.ndimage import convolve1d, convolve, affine_transform
from skimage.morphology import remove_small_holes
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as im_shift
from skimage import color

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

def shifts_to_slice(shifts,shape):
    """
    Find the minimal crop box from time lapse registraton shifts.
    """
#     max_shift = np.max(shifts,axis=0)
#     min_shift = np.min(shifts,axis=0)
#     slc = tuple([slice(np.maximum(0,0+int(mn)),np.minimum(s,s-int(mx))) for mx,mn,s in zip(np.flip(max_shift),np.flip(min_shift),shape)])
    # slc = tuple([slice(np.maximum(0,0+int(mn)),np.minimum(s,s-int(mx))) for mx,mn,s in zip(max_shift,min_shift,shape)])
    upper_shift = np.min(shifts,axis=0)
    lower_shift = np.max(shifts,axis=0)
    slc = tuple([slice(np.maximum(0,0+int(l)),np.minimum(s,s-int(u))) for u,l,s in zip(upper_shift,lower_shift,shape)])
    return slc

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
              normalization=None,cval=None,prefilter=True,reverse=True):
    """
    Find the transformation matrices for all images in a time series to align to the beginning frame. 
    """
    dim = imstack.ndim - 1 # dim is spatial, assume first dimension is t
    s = np.zeros(dim)
    shape = imstack.shape[-dim:]
    regstack = np.zeros_like(imstack)
    shifts = np.zeros((len(imstack),dim))
    for i,im in enumerate(imstack[::-1] if reverse else imstack):
        ref = regstack[i-1] if i>0 else im 
        # reference_mask=~np.isnan(ref)
        # moving_mask=~np.isnan(im)
        # pad = 1
        # shift = phase_cross_correlation(np.pad(ref,pad), np.pad(im,pad), 
        shift = phase_cross_correlation(ref,im,
                                        upsample_factor=upsample_factor, 
                                        # return_error = False, 
                                        normalization=normalization)[0]
                                      # reference_mask=reference_mask,
                                      # moving_mask=moving_mask)
        
        # shift = imreg_dft.imreg.translation(ref,im)['tvec']
        
        shifts[i] = shift
        regstack[i] = im_shift(im, shift, order=order, prefilter=prefilter,
                               mode='nearest' if cval is None else 'constant',
                               cval=np.nanmean(imstack[i]) if cval is None else cval)   
    if reverse:
        return shifts[::-1], regstack[::-1]
    else:
        return shifts,regstack

def shift_stack(imstack, shifts, order=1, cval=None):
    """
    Shift each time slice of imstack according to list of 2D shifts. 
    """
    regstack = np.zeros_like(imstack)
    for i in range(len(shifts)):        
        regstack[i] = im_shift(imstack[i],shifts[i],order=order, 
                               mode='nearest' if cval is None else 'constant',
                               cval=np.nanmean(imstack[i]) if cval is None else cval)   
    return regstack

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
    if ARM: 
        #torch.linalg.norm not implemented on MPS yet
        # this is the fastest I have tested but still slow in comparison 
        return a.square().sum(dim=dim,keepdim=keepdim).sqrt()
    else:
        return torch.linalg.norm(a,dim=dim,keepdim=keepdim)

# @njit
def safe_divide(num,den,cutoff=0):
    """ Division ignoring zeros and NaNs in the denominator.""" 
    return np.divide(num, den, out=np.zeros_like(num), 
                     where=np.logical_and(den>cutoff,~np.isnan(den)))        


def normalize99(Y, lower=0.01, upper=99.99, dim=None):
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
    
    Returns
    --------------
    normalized array/tensor with a minimum of 0 and maximum of 1
    
    """
    quantiles = np.array([lower, upper]) / 100

    if isinstance(Y, np.ndarray):
        module = np
    elif torch.is_tensor(Y):
        module = torch
        quantiles = torch.tensor(quantiles, dtype=Y.dtype, device=Y.device)
    else:
        raise ValueError("Input should be either a numpy array or a torch tensor.")
    if dim is not None:
        # Reshape Y into a 2D tensor for quantile computation
        Y_reshaped = Y.transpose(0, dim).reshape(Y.shape[dim], -1)
        lower_val, upper_val = module.quantile(Y_reshaped, quantiles, dim=-1)
        
        # Reshape back into original shape for broadcasting
        if dim == 0:
            lower_val = lower_val.reshape(Y.shape[dim], *([1] * (len(Y.shape) - 1)))
            upper_val = upper_val.reshape(Y.shape[dim], *([1] * (len(Y.shape) - 1)))
        else:
            lower_val = lower_val.reshape(*Y.shape[:dim], *([1] * (len(Y.shape) - dim - 1)))
            upper_val = upper_val.reshape(*Y.shape[:dim], *([1] * (len(Y.shape) - dim - 1)))
    else:
        lower_val, upper_val = module.quantile(Y, quantiles)
        
    # Y = module.clip(Y, lower_val, upper_val) # is this needed? 
    # Y -= lower_val
    # Y /= (upper_val - lower_val)
    
    # return Y
    # return (Y-lower_val)/(upper_val-lower_val)
    return module.clip((Y-lower_val)/(upper_val-lower_val),0,1)


def normalize_image(im,mask,bg=0.5,dim=2,iterations=1,scale=1):
    """ Normalize image by rescaling from 0 to 1 and then adjusting gamma to bring 
    average background to specified value (0.5 by default).
    
    Parameters
    ----------
    im: ndarray, float
        input image or volume
        
    mask: ndarray, int or bool
        input labels or foreground mask
    
    bg: float
        target background value in the range 0-1
    
    dim: int
        dimension of image or volume
        (extra dims are channels, assume in front)
    
    Returns
    --------------
    gamma-normalized array with a minimum of 0 and maximum of 1
    
    """
    im = rescale(im)*scale
    if im.ndim>dim:#assume first axis is channel axis
        im = [chan for chan in im] # break into a list of channels
    else:
        im = [im]
        
    if mask is not list:
        mask = [mask]*len(im)

    for k in range(len(mask)):
        source_bg = np.mean(im[k][binary_erosion(mask[k]==0,iterations=iterations)])
        im[k] = im[k]**(np.log(bg)/np.log(source_bg))
        
    return np.stack(im,axis=0).squeeze()

import ncolor
def mask_outline_overlay(img,masks,outlines,mono=None):
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


def rescale(T,floor=None,ceiling=None):
    """Rescale array between 0 and 1"""
    if ceiling is None:
        ceiling = T[:].max()
    if floor is None:
        floor = T[:].min()
    T = np.interp(T, (floor, ceiling), (0, 1))
    return T


def normalize_stack(vol,mask,bg=0.5,bright_foreground=None,subtractive=False,iterations=1):
    """
    Adjust image stacks so that background is 
    (1) consistent in brightness and 
    (2) brought to an even average via semantic gamma normalization.
    """
    
    # vol = rescale(vol)
    vol = vol.copy()
    # binarize background mask, recede from foreground, slice-wise to not erode in time
    bg_mask = [binary_erosion(m==0,iterations=iterations) for m in mask] 
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
    
    # equalize foreground signal
    if bright_foreground:
        fg_real = [np.percentile(v[m>0],99.99) for v,m in zip(vol,mask)]
        # fg_real = [v.max() for v,m in zip(vol,bg_mask)]    
        floor = np.percentile(vol[bg_mask],0.01)
        vol = [rescale(v,ceiling=f, floor=floor) for v,f in zip(vol,fg_real)]
    else:
        fg_real = [np.percentile(v[m>0],1) for v,m in zip(vol,mask)]
        ceiling = np.percentile(vol[bg_mask],99.99)
        vol = [rescale(v,ceiling=ceiling,floor=f) for v,f in zip(vol,fg_real)]
        
    # vol = rescale(vol) # now rescale by overall min and max 
    vol = np.stack([v**(np.log(bg)/np.log(np.mean(v[bg_m]))) for v,bg_m in zip(vol,bg_mask)]) # now can gamma normalize 
    return vol


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
    if type(pad) is int:
        pad = [pad]*dim
    if type(im_pad) is int:
        im_pad = [im_pad]*dim
    # return tuple([slice(int(max(0,bbox[n]-pad[n])),int(min(bbox[n+dim]+pad[n],shape[n]))) for n in range(len(bbox)//2)])
    # added a +1 to stop, might be a necessary fix but not sure yet 
    print('im_pad',im_pad, bbox, pad, shape)
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
                print(minpad,'m',im_pad)
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
           im_xx :
           im_yy :
           im_xy :

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


    