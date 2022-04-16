import numpy as np
# from cellpose.dynamics import SKIMAGE_ENABLED #circular import error 
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import remove_small_holes
from scipy.ndimage import label # used as alternative to skimage.measure.label, need to test speed and output...
import fastremap
import mahotas as mh


try:
    from skimage import measure
    SKIMAGE_ENABLED = True 
except:
    SKIMAGE_ENABLED = False

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

def normalize_image(im,mask,bg=0.5):
    """ Normalize image by rescaling from 0 to 1 and then adjusting gamma to bring 
    average background to specified value (0.5 by default).
    
    Parameters
    ----------
    im: ndarray, float
        input image or volume
        
    mask: ndarray, int or bool
        input labels or foreground mask
    
    Returns
    --------------
    gamma-normalized array with a minimum of 0 and maximum of 1
    
    """
    im = rescale(im)
    return im**(np.log(bg)/np.log(np.mean(im[binary_erosion(mask==0)])))

def bbox_to_slice(bbox,shape,pad=0):
    """
    return the tuple of slices for cropping an image based on the skimage.measure bounding box
    optional padding allows for the bounding box to be expanded, but not outside the original image dinensions 
    
    Parameters
    ----------
    bbox: ndarray, float
        input bounding box, e.g. [y0,x0,y1,x1]
        
    shape: array, tuple, or list, int
        shape of corresponding array to be sliced
    
    pad: array, tuple, or list, int
        padding to be applied to each edge of the bounding box
        can be a common padding or a list of each axis padding 
    
    Returns
    --------------
    tuple of slices 
    
    """
    if type(pad) is int:
        pad = [pad]*len(shape)
    return tuple([slice(max(0,bbox[n]-pad[n]),min(bbox[n+2]+pad[n],shape[n])) for n in range(len(bbox)//2)])

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

# Kevin's version of remove_edge_masks, need to merge (this one is more flexible)
def clean_boundary(labels,boundary_thickness=3,area_thresh=30):
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
    for cell_ID in np.unique(labels):
        mask = labels==cell_ID 
        area = np.count_nonzero(mask)
        overlap = np.count_nonzero(np.logical_and(mask, border_mask))
        if overlap > 0 and area<area_thresh and overlap/area >= 0.5: #only premove cells that are 50% or more edge px
            clean_labels[mask] = 0
    return clean_labels


# Should work for 3D too. Could put into usigned integer form at the end... 
# Also could use some parallelization 

def format_labels(labels, clean=False, min_area=9, despur=False, verbose=False):
    """
    Puts labels into 'standard form', i.e. background=0 and cells 1,2,3,...,N-1,N.
    Optional clean flag: disconnect and disjoint masks and discard small masks beflow min_area. 
    min_area default is 9px. 
    """
    
    # Labels are stored as a part of a float array in Cellpose, so it must be cast back here.
    # some people also use -1 as background, so we must cast to the signed integar class. We
    # can safely assume no 2D or 3D image will have more than 2^31 cells. Finally, cv2 does not
    # play well with unsigned integers (saves to default uint8), so we cast to uint32. 
    labels = labels.astype('int32') 
    labels -= np.min(labels) 
    labels = labels.astype('uint32') 
    
    # optional cleanup 
    if clean:
        inds = np.unique(labels)
        for j in inds[inds>0]:
            mask = labels==j
            if despur:
                mask = delete_spurs(mask) #needs updating for ND 
            
            if SKIMAGE_ENABLED:
                lbl = measure.label(mask)                       
                regions = measure.regionprops(lbl)
                regions.sort(key=lambda x: x.area, reverse=True)
                if len(regions) > 1:
                    if verbose:
                        print('Warning - found mask with disjoint label.')
                    for rg in regions[1:]:
                        if rg.area <= min_area:
                            labels[tuple(rg.coords.T)] = 0
                            if verbose:
                                print('secondary disjoint part smaller than min_area. Removing it.')
                        else:
                            if verbose:
                                print('secondary disjoint part bigger than min_area, relabeling. Area:',rg.area, 
                                        'Label value:',np.unique(labels[tuple(rg.coords.T)]))
                            labels[tuple(rg.coords.T)] = np.max(labels)+1
                            
                rg0 = regions[0]
                if rg0.area <= min_area:
                    labels[tuple(rg0.coords.T)] = 0
                    if verbose:
                        print('Warning - found mask area less than', min_area)
                        print('Removing it.')
            else:
                connectivity_shape = np.array([3 for i in range(mask.ndim)])
                lbl = label(mask, connectivity=np.ones(connectivity_shape))[0]
                labels = lbl
        
    fastremap.renumber(labels,in_place=True) # convenient to have unit increments from 1 to N cells
    labels = fastremap.refit(labels) # put into smaller data type if possible 
    return labels

# get the number of m-dimensional hypercubes connected to the n-cube
def cubestats(n):
    faces = []
    for m in range(n+1):
          faces.append((2**(n-m))*math.comb(n,m))
    return faces

def delete_spurs(mask):
    pad = 1
    #must fill single holes in image to avoid cusps causing issues. Will limit to holes of size ___
    skel = remove_small_holes(np.pad(mask,pad,mode='constant'),5)

    nbad = 1
    niter = 0
    while (nbad > 0):
        bad_points = endpoints(skel) 
        skel = np.logical_and(skel,np.logical_not(bad_points))
        nbad = np.sum(bad_points)
        niter+=1
    
    unpad =  tuple([slice(pad,-pad)]*skel.ndim)
    skel = skel[unpad] #unpad

    return skel

# this still  only works for 2D
def endpoints(skel):
    pad = 1 # appears to require padding to work properly....
    skel = np.pad(skel,pad)
    endpoint1=np.array([[0, 0, 0],
                        [0, 1, 0],
                        [2, 1, 2]])
    
    endpoint2=np.array([[0, 0, 0],
                        [0, 1, 2],
                        [0, 2, 1]])
    
    endpoint3=np.array([[0, 0, 2],
                        [0, 1, 1],
                        [0, 0, 2]])
    
    endpoint4=np.array([[0, 2, 1],
                        [0, 1, 2],
                        [0, 0, 0]])
    
    endpoint5=np.array([[2, 1, 2],
                        [0, 1, 0],
                        [0, 0, 0]])
    
    endpoint6=np.array([[1, 2, 0],
                        [2, 1, 0],
                        [0, 0, 0]])
    
    endpoint7=np.array([[2, 0, 0],
                        [1, 1, 0],
                        [2, 0, 0]])
    
    endpoint8=np.array([[0, 0, 0],
                        [2, 1, 0],
                        [1, 2, 0]])
    
    ep1=mh.morph.hitmiss(skel,endpoint1)
    ep2=mh.morph.hitmiss(skel,endpoint2)
    ep3=mh.morph.hitmiss(skel,endpoint3)
    ep4=mh.morph.hitmiss(skel,endpoint4)
    ep5=mh.morph.hitmiss(skel,endpoint5)
    ep6=mh.morph.hitmiss(skel,endpoint6)
    ep7=mh.morph.hitmiss(skel,endpoint7)
    ep8=mh.morph.hitmiss(skel,endpoint8)
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    unpad =  tuple([slice(pad,-pad)]*ep.ndim)
    ep = ep[unpad]
    return ep
