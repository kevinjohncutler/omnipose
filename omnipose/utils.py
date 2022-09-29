import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import remove_small_holes
from skimage import measure 
import fastremap
import mahotas as mh
import math
from ncolor import format_labels # just in case I forgot to switch it out elsewhere

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
    pad: amount of space in pixels to add around the label (does not extend beyond image edges)
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

            bboxes.append(bbox_to_slice(bbx,sz,pad=pad,im_pad=im_pad))
    
    
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

def rescale(T):
    """Rescale array between 0 and 1"""
    T = np.interp(T, (T[:].min(), T[:].max()), (0, 1))
    return T

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
def clean_boundary(labels, boundary_thickness=3, area_thresh=30, dists=None):
    """Delete boundary masks below a given size threshold within a certain distance from the boundary. 
    
    Parameters
    ----------
    boundary_thickness: int
        labels within a stripe of this thickness along the boundary will be candidates for removal. 
        
    area_thresh: int
        labels with area below this value will be removed. 
    
    Returns
    --------------
    label matrix with small edge labels removed. 
    
    """
    border_mask = np.zeros(labels.shape, dtype=bool)
    border_mask = binary_dilation(border_mask, border_value=1, iterations=boundary_thickness)
    clean_labels = np.copy(labels)
    # bddist = dists[border_mask]
    # dist_thresh = np.mean(bddist[bddist>0])
    # print(dist_thresh)
    for cell_ID in fastremap.unique(labels[border_mask])[1:]:
        mask = labels==cell_ID 
        area = np.count_nonzero(mask)
        overlap = np.count_nonzero(np.logical_and(mask, border_mask))
        if overlap > 0 and area<area_thresh and overlap/area >= 0.5: #only remove cells that are 50% or more edge px
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


