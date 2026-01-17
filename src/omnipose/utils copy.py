import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from scipy.ndimage import convolve1d, convolve, affine_transform
from skimage.morphology import remove_small_holes
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as im_shift

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
    s = np.zeros(2)
    shape = imstack.shape[-2:]
    regstack = np.zeros_like(imstack)
    shifts = np.zeros((len(imstack),2))
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
@njit
def normalize99(Y,lower=0.01,upper=99.99):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile 
    Upper and lower percentile ranges configurable. 
    
    Parameters
    ----------
    Y: ndarray, float
        Component array of lenth N by L1 by L2 by ... by LN. 
    upper: float
        upper percentile above which pixels are sent to 1.0
    
    lower: float
        lower percentile below which pixels are sent to 0.0
    
    Returns
    --------------
    normalized array with a minimum of 0 and maximum of 1
    
    """
    # X = Y.copy()
    # return np.interp(Y, (np.percentile(Y, lower), np.percentile(Y, upper)), (0, 1))
    return np.interp(Y, np.percentile(Y, [lower,upper]), (0, 1)) # much faster to call both at once 
    

def normalize_image(im,mask,bg=0.5,dim=2):
    """ Normalize image by rescaling from 0 to 1 and then adjusting gamma to bring 
    average background to specified value (0.5 by default).
    
    Parameters
    ----------
    im: ndarray, float
        input image or volume
        
    mask: ndarray, int or bool
        input labels or foreground mask
    
    bg: float
        background value in the range 0-1
    
    dim: int
        dimension of image or volume
        (extra dims are channels, assume in front)
    
    Returns
    --------------
    gamma-normalized array with a minimum of 0 and maximum of 1
    
    """
    im = rescale(im)
    if im.ndim>dim:#assume first axis is channel axis
        im = [chan for chan in im] # break into a list of channels
    else:
        im = [im]
        
    if mask is not list:
        mask = [mask]*len(im)

    for k in range(len(mask)):
        im[k] = im[k]**(np.log(bg)/np.log(np.mean(im[k][binary_erosion(mask[k]==0)])))
        
    return np.stack(im,axis=0).squeeze()


from skimage import color
def mask_outline_overlay(img,masks,outlines,mono=None):
    if mono is None:
        m,n = ncolor.label(masks,max_depth=20,return_n=True)
        c = sinebow(n)
        colors = np.array(list(c.values()))[1:]
    else:
        colors = mono
        m = masks>0
    im = rescale(color.rgb2gray(img))
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
    return tuple([slice(int(max(im_pad[n],bbox[n]-pad[n])),int(min(bbox[n+dim]+pad[n],shape[n]-im_pad[n]))) for n in range(len(bbox)//2)])
    

def crop_bbox(mask, pad=10, iterations=3, im_pad=0, area_cutoff=0, max_dim=np.inf, get_biggest=False, binary=False):
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
    bw = binary_dilation(mask>0,iterations=iterations)
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
        slices.append(bbox_to_slice(bbx,sz,pad=minpad,im_pad=im_pad))
        
    else:
        for props in regions:
            if props.area>area_cutoff:
                bbx = props.bbox 
                minpad = min(pad,bbx[0],bbx[1],sz[0]-bbx[2],sz[1]-bbx[3])
                slices.append(bbox_to_slice(bbx,sz,pad=minpad,im_pad=im_pad))
    
    if binary:
        start_xy = np.min([[slc[i].start for i in range(d)] for slc in slices],axis=0)
        stop_xy = np.max([[slc[i].stop for i in range(d)] for slc in slices],axis=0)
        slices = tuple([slice(start,stop) for start,stop in zip(start_xy,stop_xy)])
    
    return slices

def sinebow(N,bg_color=[0,0,0,0], offset=0):
    """ Generate a color dictionary for use in visualizing N-colored labels. Background color 
    defaults to transparent black. 
    
    Parameters
    ----------
    N: int
        number of distinct colors to generate (excluding background)
        
    bg_color: ndarray, list, or tuple of length 4
        RGBA values specifying the background color at the front of the  dictionary.
    
    Returns
    --------------
    Dictionary with entries {int:RGBA array} to map integer labels to RGBA colors. 
    
    """
    colordict = {0:bg_color}
    for j in range(N): 
        k = j+offset
        angle = k*2*np.pi / (N) 
        r = ((np.cos(angle)+1)/2)
        g = ((np.cos(angle+2*np.pi/3)+1)/2)
        b = ((np.cos(angle+4*np.pi/3)+1)/2)
        colordict.update({j+1:[r,g,b,1]})
    return colordict

@njit
def colorize(im,offset=0):
    N = len(im)
    angle = np.arange(0,1,1/N)*2*np.pi+offset
    angles = np.stack((angle,angle+2*np.pi/3,angle+4*np.pi/3),axis=-1)
    colors = (np.cos(angles)+1)/2
    rgb = np.zeros((im.shape[1], im.shape[2], 3))
    for i in range(N):
        for j in range(3):
            rgb[..., j] += im[i] * colors[i, j]
    rgb /= N
    return rgb

import matplotlib as mpl
import ncolor
def apply_ncolor(masks):
    m,n = ncolor.label(masks,max_depth=20,return_n=True,conn=2)
    c = sinebow(n)
    colors = np.array(list(c.values()))
    cmap = mpl.colors.ListedColormap(colors)
    return cmap(m)

import matplotlib.pyplot as plt
def imshow(img,figsize=2,**kwargs):
    if type(figsize) is not (list or tuple):
        figsize = (figsize,figsize)
    plt.figure(frameon=False,figsize=figsize)
    plt.imshow(img,**kwargs)
    plt.axis("off")
    plt.show()

# def get_cmap(masks):
#     lut = ncolor.get_lut(masks)
#     c = sinebow(lut.max())
#     colors = [c[l] for l in lut]
#     cmap = mpl.colors.ListedColormap(colors)
#     return cmap
    
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


# @njit this version actually a lot slower than below 
# def get_neighbors(coords, steps, dim, shape, edges=None):
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
def get_neighbors(coords, steps, dim, shape, edges=None):
    """
    Get the coordinates of all neighbor pixels. 
    Coordinates of pixels that are out-of-bounds get clipped. 
    """
    if edges is None:
        edges = [np.array([-1,s]) for s in shape]
        
    npix = coords[0].shape[-1]
    neighbors = np.empty((dim, len(steps), npix), dtype=np.int64)
    
    for d in range(dim):        
        S = steps[:,d].reshape(-1, 1)
        X = coords[d] + S
        mask = np.logical_and(np.isin(X, edges[d]), ~np.isin(X+S, edges[d]))
        C = np.broadcast_to(coords[d], X.shape)
        neighbors[d] = np.where(mask, C, X)
    
    return neighbors

# this version works without padding, should ultimately replace the other one 
# @njit
def get_neigh_inds(neighbors,coords,shape):
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
    npix = neighbors[0].shape[-1]
    indexes = np.arange(npix)
    ind_matrix = -np.ones(shape,int)
    
    ind_matrix[coords] = indexes
    neigh_inds = ind_matrix[neighbors]

    oob = np.nonzero(neigh_inds==-1) # 2 x nbad , pos 0 is the 0-step inds and pos 1 is the npix inds 
    neigh_inds[oob] = indexes[oob[1]] # reflect back to itself 
    ind_matrix[neighbors] = neigh_inds # update ind matrix as well
    
    # not sure if -1 is general enough, probbaly should be since other adjacent masks will be unlinked
    # can test it by adding some padding to the concatenation...
    
    return indexes, neigh_inds, ind_matrix


# maybe neighbors and affinity graph should be packaged together...
def crop_affinity(affinity_graph,ind_matrix,slc):
    """
    Helper function to crop affinity graph according to an image crop slice slc. 
    The affinity graph must refer to the same hypervoxels as the neighbor array. 
    All entries in the affinity graph referring to out-of-bounds coordinates are untouched. 
    Entries corresponding to hypervoxels that are cropped need to be removed. However, order matters. 

    """

    # linear indexes change a lot when cropping. Need a mapping between old linear indexes and new.
    # ind_matrix is an intuitive way to do that. 
    # so just make ind_matrix for original, ind_matrix for cropped (or pixel-removed), then populate new
    # affinity graph using the mapping defined by slicing the old ind_matrix. The remapping will apply to all of 
    # the entries (center, cardinal, ordinal, etc.) simultaneously.
    # ind matrix itself can be cropped and thresholded to generate the replacement ind_matrix. This is very flexible,
    # as any other pixels we want to discard can be assined -1 in the ind matrix and will be gone from the affinity_graph. 

    ind_matrix_crop = ind_matrix[slc]
    cropmask = ind_matrix_crop>0
    d = cropmask.ndim
    shape = cropmask.shape
    npix = np.count_nonzero(cropmask)
    steps, inds, idx, fact, sign = kernel_setup(d)
    coords = np.nonzero(cropmask)
    neighbors = get_neighbors(coords,steps,d,shape) # shape (d,3**d,npix)   
    indexes, neigh_inds, ind_matrix = get_neigh_inds(tuple(neighbors),coords,shape)

    # affinity_graph_crop = np.zeros((3**d,npix))
    inds_crop = ind_matrix_crop[cropmask] #should all be nonnegative
    affinity_graph_crop = affinity_graph[:,inds_crop]

    mapping = fastremap.component_map(ind_matrix_crop,ind_matrix)
    affinity_graph_remap = fastremap.remap(affinity_graph_crop,mapping,in_place=True)
    return affinity_graph_remap


def get_steps(dim):
    """
    Get a symmetrical list of all 3**N points in a hypercube represented
    by a list of all possible sequences of -1, 0, and 1 in ND.

    Examples
    --------
    .. code-block:: text

        1D: [[-1], [0], [1]]
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
def kernel_setup(dim):
    """
    Get relevant kernel information for the hypercube of interest. 
    Calls get_steps(), steps_to_indices(). Input is the dimesion.
    Returns:
    
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
    
    
    
# not acutally used in the code, typically use  steps_to_indices etc. 
def cubestats(n):
    """
    Gets the number of m-dimensional hypercubes connected to the n-cube, including itself. 
    
    Parameters
    ----------
    n: int
        dimension of hypercube
    
    Returns
    --------------
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
    C1 = (M-np.sqrt(np.abs(M**2-G)))
    C2 = (M+np.sqrt(np.abs(M**2-G)))

    
    # remove negative values
    G_ = G.copy()
    G_[G<0] = 0

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
