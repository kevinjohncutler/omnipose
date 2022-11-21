import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import convolve1d, convolve
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

def findbetween(s,string1='[',string2=']'):
    return re.findall(str(re.escape(string1))+"(.*)"+str(re.escape(string2)),s)[0]

def getname(path,prefix='',suffix='',padding=0):
    return os.path.splitext(Path(path).name)[0].replace(prefix,'').replace(suffix,'').zfill(padding)

def to_16_bit(im):
    return np.uint16(rescale(im)*(2**16-1))

def to_8_bit(im):
    return np.uint8(rescale(im)*(2**8-1))

def shifts_to_slice(shifts,shape):
    """
    Find the minimal crop box from time lapse registraton shifts.
    """
    max_shift = np.max(shifts,axis=0)
    min_shift = np.min(shifts,axis=0)
    slc = tuple([slice(np.maximum(0,0-int(mn)),np.minimum(s,s-int(mx))) for mx,mn,s in zip(np.flip(max_shift),np.flip(min_shift),shape)])
    return slc

def cross_reg(imstack,upsample_factor=100,order=1,normalization=None,cval=None):
    """
    Find the transformation matrices for all images in a time series to align to the beginning frame. 
    """
    s = np.zeros(2)
    shape = imstack.shape[-2:]
    regstack = np.zeros_like(imstack)
    shifts = np.zeros((len(imstack),2))
    for i,im in enumerate(imstack):
        ref = regstack[i-1] if i>0 else im 
        # reference_mask=~np.isnan(ref)
        # moving_mask=~np.isnan(im)
        shift = phase_cross_correlation(ref, im, 
                                        upsample_factor=upsample_factor, 
                                        return_error = False, 
                                        normalization=normalization)
                                      # reference_mask=reference_mask,
                                      # moving_mask=moving_mask)
        # print(shift)
        shifts[i] = shift
        regstack[i] = im_shift(im, shift, order=order,
                               mode='nearest' if cval is None else 'constant',
                               cval=np.nanmean(imstack[i]) if cval is None else cval)   
    return shifts, regstack

def shift_stack(shifts, imstack, order=1, cval=None):
    """
    Shift each time slice of imstack according to list of 2D shifts. 
    """
    regstack = np.zeros_like(imstack)
    for i in range(len(shifts)):        
        regstack[i] = im_shift(imstack[i],shifts[i],order=order, 
                               mode='nearest' if cval is None else 'constant',
                               cval=np.nanmean(imstack[i]) if cval is None else cval)   
    return regstack

def moving_average(x, w, tmats):
    # return np.convolve(x, np.ones(w), 'valid') / w
    return scipy.ndimage.convolve1d(tmats,np.ones(w)/w,axis=0)

def normalize_field(mu):
    """ normalize all nonzero field vectors to magnitude 1
    
    Parameters
    ----------
    mu: ndarray, float
        Component array of lenth N by L1 by L2 by ... by LN. 
    
    Returns
    --------------
    normalized component array of identical size. 
    """
    mag = np.sqrt(np.nansum(mu**2,axis=0))
    # m = mag>0
    # mu = np.divide(mu, mag, out=np.zeros_like(mu), where=np.logical_and(mag!=0,~np.isnan(mag)))        
    # return mu
    return safe_divide(mu,mag)


def safe_divide(num,den):
    """ Division ignoring zeros and NaNs in the denominator.""" 
    return np.divide(num, den, out=np.zeros_like(num), 
                     where=np.logical_and(den!=0,~np.isnan(den)))        
    
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
    X = Y.copy()
    return np.interp(X, (np.percentile(X, lower), np.percentile(X, upper)), (0, 1))

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

def normalize_stack(vol,mask,bg=0.5,bright_foreground=None):
    """
    Adjust image stacks so that background is 
    (1) consistent in brightness and 
    (2) brought to an even average via semantic gamma normalization.
    """
    
    # vol = rescale(vol)
    vol = vol.copy()
    bg_mask = [binary_erosion(m==0) for m in mask] # binarize background mask, recede from foreground, slice-wise to not erode in time
    bg_real = [np.nanmean(v[m]) for v,m in zip(vol,bg_mask)] # find mean backgroud for each slice
    
    # automatically determine if foreground objects are bright or dark 
    if bright_foreground is None:
        bright_foreground = np.mean(vol[bg_mask]) < np.mean(vol[mask>0])
    
    # if smooth: 
    #     bg_real = moving_average(bg_real,5) 
    # some weird fluctuations happening with background being close to zero, but just on fluorescnece... might need to invert or go by foreground
    
    bg_min = np.min(bg_real) # git the minimum one, want to normalize by lowest one 
    vol = np.stack([v*safe_divide(bg_min,bg_r) for v,bg_r in zip(vol,bg_real)]) # normalize background
    
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
    

def crop_bbox(mask, pad=10, iterations=3, im_pad=0, area_cutoff=0, max_dim=np.inf):
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

    bbx: list of bounding boxes
    
    """
    bw = binary_dilation(mask>0,iterations=iterations)
    clusters = measure.label(bw)
    regions = measure.regionprops(clusters)
    sz = mask.shape
    # ylim = [im_pad,sz[0]-im_pad]
    # xlim = [im_pad,sz[1]-im_pad]
    
    bboxes = []
    for props in regions:
        if props.area>area_cutoff:
            bbx = props.bbox 
            minpad = min(pad,bbx[0],bbx[1],sz[0]-bbx[2],sz[1]-bbx[3])
#             y1 = max(bbx[0]-pad,ylim[0])
#             x1 = max(bbx[1]-pad,xlim[0])
#             y2 = min(bbx[2]+pad,ylim[1])
#             x2 = min(bbx[3]+pad,xlim[1])
#             w = x2-x1
#             h = y2-y1
            
#             if w>0 and h>0: 
#                 if w<max_dim and h<max_dim:
#                     bboxes.append([y1,y2,x1,x2])
#                     # m = maski[y1:y2,x1:x2].copy()
#             else:
#                 return [[0,ylim,0,xlim]]
            
            bboxes.append(bbox_to_slice(bbx,sz,pad=minpad,im_pad=im_pad))
    
    return bboxes

def sinebow(N,bg_color=[0,0,0,0]):
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
        angle = j*2*np.pi / (N)
        r = ((np.cos(angle)+1)/2)
        g = ((np.cos(angle+2*np.pi/3)+1)/2)
        b = ((np.cos(angle+4*np.pi/3)+1)/2)
        colordict.update({j+1:[r,g,b,1]})
    return colordict


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

# Kevin's version of remove_edge_masks, need to merge (this one is more flexible)
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

# not acutally used anymore 
def cubestats(n):
    """gets the number of m-dimensional hypercubes connected to the n-cube, including itself
    
    Parameters
    ----------
    n: int
        dimension of hypercube
    
    Returns
    --------------
    list whose length tells us how many hypercube types there are (point/edge/pixel/voxel...) connected 
    to the central hypercube and whose entries denote many there in each group. E.g., a square would be n=2, 
    so cubestats returns [4, 4, 1] for four points (m=0), four edges (m=1), and one face (the original square,m=n=2). 
    
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
    shape = [np.floor(7*filterWidth)]*2
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