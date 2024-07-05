from skimage import measure
from scipy.ndimage import binary_dilation
import numpy as np 
from .utils import is_integer

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