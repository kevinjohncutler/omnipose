import numpy as np
from numba import njit
import cv2
import edt
from scipy.ndimage import binary_dilation, binary_opening, binary_closing, label, shift # I need to test against skimage labelling
from skimage.morphology import remove_small_objects
from sklearn.utils.extmath import cartesian
from skimage.segmentation import find_boundaries


import fastremap
import os, tifffile
import time
import mgen #ND rotation matrix
from . import utils
from ncolor.format_labels import delete_spurs



# define the list of unqiue omnipose models 
OMNI_MODELS = ['bact_phase_cp',
               'bact_fluor_cp',
               'plant_cp', # 2D model
               'worm_cp',
               'cyto2_omni',
               'bact_phase_omni',
               'bact_fluor_omni',
               'plant_omni', #3D model 
               'worm_omni',
               'worm_bact_omni',
               'worm_high_res_omni']

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

# try:
#     from sklearn.cluster import DBSCAN
#     from sklearn.neighbors import NearestNeighbors
#     SKLEARN_ENABLED = True 
# except:
#     SKLEARN_ENABLED = False

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
SKLEARN_ENABLED = True 

try:
    from hdbscan import HDBSCAN
    HDBSCAN_ENABLED = True
    
except:
    HDBSCAN_ENABLED = False
    
import logging, sys
logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.StreamHandler(sys.stdout)
                    ]
                )
omnipose_logger = logging.getLogger(__name__)
# omnipose_logger.setLevel(logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler())


# We moved a bunch of dupicated code over here from cellpose_omni to revert back to the original bahavior. This flag is used
# within Cellpose only, but since I want to merge the shared code back together someday, I'll keep it around here. 
# Several '#'s denote locations where code needs to be changed if a remerger ever happens 
OMNI_INSTALLED = True

from tqdm import trange 
import ncolor, scipy
from scipy.ndimage.filters import maximum_filter1d
from scipy.ndimage import find_objects, gaussian_filter, generate_binary_structure, label, maximum_filter1d, binary_fill_holes, zoom

    
# try:
#     from skimage.morphology import remove_small_holes
#     from skimage.util import random_noise
#     from skimage.filters import gaussian
#     from skimage import measure
#     from skimage import filters
#     import skimage.io #for debugging only
#     SKIMAGE_ENABLED = True
# except:
#     from scipy.ndimage import gaussian_filter as gaussian
#     SKIMAGE_ENABLED = False

# skimage is necessary for Omnipose 
from skimage.morphology import remove_small_holes
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage import measure
from skimage import filters
import skimage.io #for debugging only
SKIMAGE_ENABLED = True

from scipy.ndimage import convolve, mean


### Section I: core utilities

# By testing for convergence across a range of superellipses, I found that the following
# ratio guarantees convergence. The edt() package gives a quick (but rough) distance field,
# and it allows us to find a least upper bound for the number of iterations needed for our
# smooth distance field computation. 
def get_niter(dists):
    """
    Get number of iterations. 
    
    Parameters
    --------------
    dists: ND array, float
        array of (nonnegative) distance field values
        
    Returns
    --------------
    niter: int
        number of iterations empirically found to be the lower bound for convergence 
        of the distance field relaxation method
    
    """
    return np.ceil(np.max(dists)*1.16).astype(int)+1

# minor modification to generalize to nD 
def dist_to_diam(dt_pos,n): 
    """
    Convert positive distance field values to a mean diameter. 
    
    Parameters
    --------------
    dt_pos: 1D array, float
        array of positive distance field values
    n: int
        dimension of volume. dt_pos is always 1D becasue only the positive values
        int he distance field are passed in. 
        
    Returns
    --------------
    mean diameter: float
        a single number that corresponds to the diameter of the N-sphere when
        dt_pos for a sphere is given to the function, holds constant for 
        extending rods of uniform width, much better than the diameter of a circle 
        of equivalent area for estimating the short-axis dimensions of objects
    
    """
    return 2*(n+1)*np.mean(dt_pos)
#     return np.exp(3/2)*gmean(dt_pos[dt_pos>=gmean(dt_pos)])

def diameters(masks, dt=None, dist_threshold=0):
    
    """
    Calculate the mean cell diameter from a label matrix. 
    
    Parameters
    --------------
    masks: ND array, float
        label matrix 0,...,N
    dt: ND array, float
        distance field
    dist_threshold: float
        cutoff below which all values in dt are set to 0. Must be >=0. 
        
    Returns
    --------------
    diam: float
        a single number that corresponds to the average diameter of labeled regions in the image, see dist_to_diam()
    
    """
    if dist_threshold<0:
        dist_threshold = 0
    
    if dt is None and np.any(masks):
        dt = edt.edt(np.int32(masks))
    dt_pos = np.abs(dt[dt>dist_threshold])
    if np.any(dt_pos):
        diam = dist_to_diam(np.abs(dt_pos),n=masks.ndim)
    else:
        diam = 0
    return diam

### Section II: ground-truth flow computation  

# It is possible that flows can be eliminated in place of the distance field. The current distance field may not be smooth 
# enough, or maybe the network really does require the flow field prediction to work well. But in 3D, it will be a huge
# advantage if the network could predict just the distance (and boudnary) classes and not 3 extra flow components. 
def labels_to_flows(labels, files=None, use_gpu=False, device=None, omni=True, redo_flows=False, dim=2):
    """ Convert labels (list of masks or flows) to flows for training model.

    if files is not None, flows are saved to files to be reused

    Parameters
    --------------
    labels: list of ND-arrays
        labels[k] can be 2D or 3D, if [3 x Ly x Lx] then it is assumed that flows were precomputed.
        Otherwise labels[k][0] or labels[k] (if 2D) is used to create flows.
    files: list of strings
        list of file names for the base images that are appended with '_flows.tif' for saving. 
    use_gpu: bool
        flag to use GPU for speedup. Note that Omnipose fixes some bugs that caused the Cellpose GPU implementation
        to have different behavior compared to the Cellpose CPU implementation. 
    device: torch device
        what compute hardware to use to run the code (GPU VS CPU)
    omni: bool
        flag to generate Omnipose flows instead of Cellpose flows
    redo_flows: bool
        flag to overwrite existing flows. This is necessary when changing over from cellpose_omni to Omnipose, 
        as the flows are very different.
    dim: int
        integer representing the intrinsic dimensionality of the data. This allows users to generate 3D flows
        for volumes. Some dependencies will need to be to be extended to allow for 4D, but the image and label
        loading is generalized to ND. 

    Returns
    --------------
    flows: list of [4 x Ly x Lx] arrays
        flows[k][0] is labels[k], flows[k][1] is cell distance transform, flows[k][2:2+dim] are the 
        (T)YX flow components, and flows[k][-1] is heat distribution / smooth distance 

    """
    
    
    nimg = len(labels)
    no_flow = labels[0].ndim != 3+dim # (6,Lt,Ly,Lx) for 3D, masks + dist + boundary + flow components, then image dimensions 
    
    if no_flow or redo_flows:
        
        omnipose_logger.info('NOTE: computing flows for labels (could be done before to save time)')
        
        # compute flows; labels are fixed in masks_to_flows, so they need to be passed back
        labels, dist, heat, veci = map(list,zip(*[masks_to_flows(labels[n],use_gpu=use_gpu, 
                                                                 device=device, omni=omni, dim=dim) 
                                                  for n in trange(nimg)])) 
        
        # concatenate labels, distance transform, vector flows, heat (boundary and mask are computed in augmentations)
        if omni and OMNI_INSTALLED:
            flows = [np.concatenate((labels[n][np.newaxis,:,:], 
                                     dist[n][np.newaxis,:,:], 
                                     veci[n], 
                                     heat[n][np.newaxis,:,:]), axis=0).astype(np.float32)
                        for n in range(nimg)] 
            # clean this up to swap heat and flowd and simplify code? would have to rerun all flow generation 
        else:
            flows = [np.concatenate((labels[n][np.newaxis,:,:], 
                                     labels[n][np.newaxis,:,:]>0.5, 
                                     veci[n]), axis=0).astype(np.float32)
                    for n in range(nimg)]
        if files is not None:
            for flow, file in zip(flows, files):
                file_name = os.path.splitext(file)[0]
                tifffile.imsave(file_name+'_flows.tif', flow)
    else:
        omnipose_logger.info('flows precomputed (in omnipose.core now)') 
        flows = [labels[n].astype(np.float32) for n in range(nimg)]

    return flows

def masks_to_flows(masks, dists=None, boundaries=None, use_gpu=False, device=None, omni=True, dim=2, smooth=True):
    """Convert masks to flows. 
    
    First, we find the scalar field. In Omnipose, this is the distance field. In Cellpose, 
    this is diffusion from center pixel. 
    Center of masks where diffusion starts is defined to be the 
    closest pixel to the median of all pixels that is inside the 
    mask.
    
    The flow components are then found as hthe gradient of the scalar field. 

    Parameters
    -------------
    masks: int, ND array
        labelled masks, 0 = background, 1,2,...,N = mask labels   
    dists: ND array, float
        array of (nonnegative) distance field values
    use_gpu: bool
        flag to use GPU for speedup. Note that Omnipose fixes some bugs that caused the Cellpose GPU implementation
        to have different behavior compared to the Cellpose CPU implementation. 
    device: torch device
        what compute hardware to use to run the code (GPU VS CPU)
    omni: bool
        flag to generate Omnipose flows instead of Cellpose flows
    dim: int
        dimensionality of image data

    Returns
    -------------
    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].
    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 

    """
    m_orig = masks.copy()
    if dists is None:
        masks = ncolor.format_labels(masks)
        dists = edt.edt(masks,parallel=8)
        
    if boundaries is None:
        boundaries = find_boundaries(masks,connectivity=dim) # does not find self-interesction boundaries of course 
        
    if device is None:
        if use_gpu:
            device = torch_GPU
        else:
            device = torch_CPU
    
    # No reason not to have pytorch installed. Running using CPU is still 2x faster
    # than the dedicated, jitted CPU code thanks to it being parallelized I think.
    masks_to_flows_device = masks_to_flows_torch
    
    if masks.ndim==3 and dim==2:
        # this branch preserves original 3D apprach 
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows_device(masks[z], dists[z], boundaries[z], 
                                        device=device, omni=omni)[0]
            mu[[1,2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows_device(masks[:,y], dists[:,y], boundaries[:,y], 
                                        device=device, omni=omni)[0]
            mu[[0,2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows_device(masks[:,:,x], dists[:,:,x], boundaries[:,:,x], 
                                        device=device, omni=omni)[0]
            mu[[0,1], :, :, x] += mu0
        return masks, dists, None, mu #consistency with below
    
    else:
        
        if omni and OMNI_INSTALLED: 
            # padding helps avoid edge artifacts from cut-off cells 
            # amount of padding should depend on how wide the cells are 
            pad = int(diameters(masks,dists)/2)
            unpad = tuple([slice(pad,-pad) if pad else slice(None,None)]*masks.ndim) # works in case pad is zero
            
            # reflect over those masks with high distance at the boundary, relevant when cropping 
            # step 1: remove any masks we do not want to reflect, then perform reflection padding
            edge_masks = utils.get_edge_masks(masks,dists=dists)
            edge_bd = boundaries*(edge_masks>0)
            masks_pad = np.pad(edge_masks,pad,mode='reflect') 
            bd_pad = np.pad(edge_bd,pad,mode='reflect') # note that the boundaries better not contain the edge of the image here... might need to crop the image in by 1px first 
            
            # step 2: restore the masks in the original area
            masks_pad[unpad] = masks
            bd_pad[unpad] = boundaries
            
            mu, T = masks_to_flows_device(masks_pad, dists, bd_pad, device=device, omni=omni, smooth=smooth)
            return masks, dists, T[unpad], mu[(Ellipsis,)+unpad]

        else: # reflection not a good idea for centroid model 
            mu, T = masks_to_flows_device(masks, dists=dists, boundaries=boundaries, device=device, omni=omni, smooth=smooth)
            return masks, dists, T, mu


#Now fully converted to work for ND.
def masks_to_flows_torch(masks, dists, boundaries, device=None, omni=True, smooth=True, niter=None):
    """Convert ND masks to flows. 
    
    Omnipose find distance field, Cellpose uses diffusion from center of mass.

    Parameters
    -------------

    masks: int, ND array
        labelled masks, 0 = background, 1,2,...,N = mask labels
    dists: ND array, float
        array of (nonnegative) distance field values
    device: torch device
        what compute hardware to use to run the code (GPU VS CPU)
    omni: bool
        flag to generate Omnipose flows instead of Cellpose flows
    smooth: bool
        use relaxation to smooth out distance and therby flow field
    niter: int
        override number of iterations 

    Returns
    -------------
    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z or T = mu[0].
    dist: float, 2D or 3D array
        scalar field representing temperature distribution (Cellpose)
        or the smooth distance field (Omnipose)

    """
    
    if device is None:
        device = torch.device('cuda')
        
    if np.any(masks):
        # the padding here is different than the padding added in masks_to_flows(); 
        # for omni, we reflect masks to extend skeletons to the boundary. Here we pad 
        # with 0 to ensure that edge pixels are not handled differently. 
        pad = 1
        masks_padded = np.pad(masks,pad)
        boundaries_padded = np.pad(boundaries,pad)

        centers = np.array([])
        if not omni: #do original centroid projection algrorithm
            # get mask centers
            centers = np.array(scipy.ndimage.center_of_mass(masks_padded, labels=masks_padded, 
                                                            index=np.arange(1, masks_padded.max()+1))).astype(int).T
            # (check mask center inside mask)
            valid = masks_padded[tuple(centers)] == np.arange(1, masks_padded.max()+1)
            for i in np.nonzero(~valid)[0]:
                coords = np.array(np.nonzero(masks_padded==(i+1)))
                meds = np.median(coords,axis=0)
                imin = np.argmin(np.sum((coords-meds)**2,axis=0))
                centers[:,i]=coords[:,imin]

        # set number of iterations
        if omni and OMNI_INSTALLED:
            # omni version requires fewer iterations 
            n_iter = get_niter(dists) ##### omnipose.core.get_niter
        else:
            slices = scipy.ndimage.find_objects(masks)
            ext = np.array([[s.stop - s.start + 1 for s in slc] for slc in slices])
            n_iter = 2 * (ext.sum(axis=1)).max()

        # run diffusion 
        mu, T = _extend_centers_torch(masks_padded, centers, boundaries_padded, 
                                      n_iter=n_iter, device=device, omni=omni, smooth=smooth)
        # normalize
        mu = utils.normalize_field(mu) ##### transforms.normalize_field(mu,omni) # maybe do not normalize the field with my computation now...

        # put into original image
        mu0 = np.zeros((mu.shape[0],)+masks.shape)
        mu0[(Ellipsis,)+np.nonzero(masks)] = mu
        unpad =  tuple([slice(pad,-pad)]*masks.ndim)
        dist = T[unpad] # mu_c now heat/distance
        return mu0, dist
    else:
        return np.zeros((masks.ndim,)+masks.shape),np.zeros(masks.shape)

# edited slightly to fix a 'bleeding' issue with the gradient; now identical to CPU version
def _extend_centers_torch(masks, centers, boundaries, n_iter=200, device=torch.device('cuda'), omni=True, smooth=True):
    """ runs diffusion on GPU to generate flows for training images or quality control
    PyTorch implementation is faster than jitted CPU implementation, therefore only the 
    GPU optimized code is being used moving forward. 
    
    Parameters
    -------------

    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels
    centers: int, 2D or 3D array
        array of center coordinates [[y0,x0],[x1,y1],...] or [[t0,y0,x0],...]
    n_inter: int
        number of iterations
    device: torch device
        what compute hardware to use to run the code (GPU VS CPU)  
    omni: bool
        whether to generate Omnipose field (solve Eikonal equation) 
        or the Cellpose field (solve heat equation from "center") 
        
    Returns
    -------------
    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z (or T) = mu[0].
    dist: float, 2D or 3D array
        the smooth distance field (Omnipose)
        or temperature distribution (Cellpose)
         
    """
        
    d = masks.ndim
    coords = np.nonzero(masks)
    idx = (3**d)//2 # the index of the center pixel is placed here when considering the neighbor kernel 

    neigh = [[-1,0,1] for i in range(d)]
    steps = cartesian(neigh) # all the possible step sequences in ND
    neighbors = np.array([np.add.outer(coords[i],steps[:,i]) for i in range(d)]).swapaxes(-1,-2)
    
    # get indices of the hupercubes sharing m-faces on the central n-cube
    sign = np.sum(np.abs(steps),axis=1) # signature distinguishing each kind of m-face via the number of steps 
    uniq = fastremap.unique(sign)
    inds = [np.where(sign==i)[0] for i in uniq] # 2D: [4], [1,3,5,7], [0,2,6,8]. 1-7 are y axis, 3-5 are x, etc. 
    fact = np.sqrt(uniq) # weighting factor for each hypercube group 
    # determine which pixels are neighbors. Pixels that are within reach (from step list) and the same label
    # are cosnidered neighbors. However, boundaries should not consider other boundaries neighbors. 
    # this means that the central pixel is not a boundary at the same time as the other. 
    neighbor_masks = masks[tuple(neighbors)] #extract list of label values, 
    neighbor_bd = boundaries[tuple(neighbors)] #extract list of boundary values 
    isneighbor = np.logical_and(neighbor_masks == neighbor_masks[idx], # must have the same label 
                                np.logical_or.reduce((
                                    # neighbor_bd != neighbor_bd[idx], # neighbor not the same as central 
                                    np.logical_and(neighbor_bd==0,neighbor_bd[idx]==0), # or the neighbor is not a boundary
                                    np.logical_and(neighbor_bd==1,neighbor_bd[idx]==0), #
                                    np.logical_and(neighbor_bd==0,neighbor_bd[idx]==1), #
                                ))
                               )

    # isneighbor = neighbor_masks == neighbor_masks[idx]
    
    nimg = neighbors.shape[1] // (3**d)
    pt = torch.from_numpy(neighbors).to(device)
    T = torch.zeros((nimg,)+masks.shape, dtype=torch.float, device=device)
    isneigh = torch.from_numpy(isneighbor).to(device) # isneigh is <3**d> x <number of points in mask>
    
    isneigh0 = torch.from_numpy(neighbor_masks == neighbor_masks[idx]).to(device)
    
    meds = torch.from_numpy(centers.astype(int)).to(device)

    mask_pix = (Ellipsis,)+tuple(pt[:,idx]) #indexing for the central coordinates 
    center_pix = (Ellipsis,)+tuple(meds)
    neigh_pix = (Ellipsis,)+tuple(pt)   
    
    for t in range(n_iter):
        if omni and OMNI_INSTALLED:
            T[mask_pix] = eikonal_update_torch(T,pt,isneigh,d,inds,fact) ##### omnipose.core.eikonal_update_torch
        else:
            T[center_pix] += 1
            
        if smooth or not omni:
            Tneigh = T[neigh_pix] # T is square, but Tneigh is nimg x <3**d> x <number of points in mask>
            Tneigh *= isneigh  #zeros out any elements that do not belong in convolution
            T[mask_pix] = Tneigh.mean(axis=1) # mean along the <3**d>-element column does the box convolution 

    # There is still a fade out effect on long cells, not enough iterations to diffuse far enough I think 
    # The log operation does not help much to alleviate it, would need a smaller constant inside. 
    if not omni:
        T = torch.log(1.+ T)
    
    Tcpy = T.clone()
    mu_torch = []
    # calculate gradient with contributions along cardinal, ordinal, etc. 
    for idx,f in zip(inds[1:],fact[1:]):
    # for idx,f in zip(inds[1:2],fact[1:2]):

    # for idx,f in zip(inds[2:3],fact[2:3]):
    
        # idx = inds[1] # cardinal points
        mask = isneigh0[idx]    
        cardinal_points = (Ellipsis,)+tuple(pt[:,idx]) 
        vals = (T[cardinal_points]*mask).cpu().squeeze() # prevent bleedover, big problem in stock Cellpose that got reverted! 
        
        # pairwise differences, e.g. cardinals of [1,3,5,7] pair up 1 and 7 (0,-1) and 3 and 5 (1,-2)
        diff = np.stack([(vals[-(i+1)] - vals[i]) / (2*f) for i in range(0,vals.shape[0]//2)])
        # unit vectors 
        vecs = steps[idx]
        uvecs = [(vecs[-(i+1)] - vecs[i]) / f for i in range(0,vecs.shape[0]//2)]
        # dot products
        # diff[0]*uvec[0][0] + diff[1]*uvec[0][1]+...
        diff_cardinal = np.stack([np.sum([d*u for d,u in zip(diff,uvecs[i])],axis=0) for i in range(d)])
    
        mu_torch.append(diff_cardinal)
        
    mu_torch = np.mean(mu_torch,axis=0)
    
    # I need a smoother way to compute the gradient, avergae out multiple directions
    # cardindals, ordinals, etc. I just need to rotate the field appropriately 
    
    
    return mu_torch, Tcpy.cpu().squeeze()

def eikonal_update_torch(T,pt,isneigh,d=None,index_list=None,factors=None):
    """Update for iterative solution of the eikonal equation on GPU."""
    # Flatten the zero out the non-neighbor elements so that they do not participate in min
    # Tneigh = T[:, pt[:,:,0], pt[:,:,1]] 
    # Flatten and zero out the non-neighbor elements so that they do not participate in min
    
    Tneigh = T[(Ellipsis,)+tuple(pt)]
    Tneigh *= isneigh
    # preallocate array to multiply into to do the geometric mean
    phi_total = torch.ones_like(Tneigh[0,0,:])
    # loop over each index list + weight factor 
    for inds,fact in zip(index_list[1:],factors[1:]):
        # find the minimum of each hypercube pair along each axis
        mins = [torch.minimum(Tneigh[:,inds[i],:],Tneigh[:,inds[-(i+1)],:]) for i in range(len(inds)//2)] 
        #apply update rule using the array of mins
        phi = update_torch(torch.cat(mins),fact)
        # multipy into storage array
        phi_total *= phi
        
        # # new: handle boundaries
        # phi_total[bd] = 1
        
    return phi_total**(1/d) #geometric mean of update along each connectivity set 

def update_torch(a,f):
    # Turns out we can just avoid a ton of individual if/else by evaluating the update function
    # for every upper limit on the sorted pairs. I do this by pieces using cumsum. The radicand
    # being nonegative sets the upper limit on the sorted pairs, so we simply select the largest 
    # upper limit that works. 
    """Update function for solving the Eikonal equation. """
    
    sum_a = torch.cumsum(a,dim=0)
    sum_a2 = torch.cumsum(a**2,dim=0)
    d = torch.cumsum(torch.ones_like(a),dim=0)
    radicand = sum_a**2-d*(sum_a2-f**2)
    mask = radicand>=0
    d = torch.count_nonzero(mask,dim=0)
    r = torch.arange(0,a.shape[-1])
    ad = sum_a[d-1,r]
    rd = radicand[d-1,r]
    return (1/d)*(ad+torch.sqrt(rd))


### Section II: mask recontruction

# Resize and rescale may appear redundant, but they are not. The resample option runs the Euler integation as usual but 
# the 
def compute_masks(dP, dist, bd=None, p=None, inds=None, niter=200, rescale=1.0, resize=None, 
                  mask_threshold=0.0, diam_threshold=12.,flow_threshold=0.4, 
                  interp=True, cluster=False, boundary_seg=False, do_3D=False, min_size=None, omni=True, 
                  calc_trace=False, verbose=False, use_gpu=False, device=None, nclasses=3, 
                  dim=2, eps=None, hdbscan=False, flow_factor=6, debug=False):
    """
    Compute masks using dynamics from dP, dist, and boundary outputs.
    
    Parameters
    -------------
    dP: float, ND array
        flow field components (2D: 2 x Ly x Lx, 3D: 3 x Lz x Ly x Lx)
    dist: float, ND array
        distance field (Ly x Lx)
    bd: float, ND array
        boundary field
    p: float32, ND array
        initial locations of each pixel before dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].  
    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x N]
    niter: int32
        number of iterations of dynamics to run
    rescale: float (optional, default None)
        resize factor for each image, if None, set to 1.0   
    resize: int, tuple
        shape of array (alternative to rescaling)  
    mask_threshold: float 
        all pixels with value above threshold kept for masks, decrease to find more and larger masks 
    flow_threshold: float 
        flow error threshold (all cells with errors below threshold are kept) (not used for Cellpose3D)
    interp: bool 
        interpolate during dynamics
    cluster: bool
        use sub-pixel DBSCAN clustering of pixel coordinates to find masks
    do_3D: bool (optional, default False)
        set to True to run 3D segmentation on 4D image input
    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1
    omni: bool 
        use omnipose mask recontruction features
    calc_trace: bool 
        calculate pixel traces and return as part of the flow
    verbose: bool 
        turn on additional output to logs for debugging 
    use_gpu: bool
        use GPU of flow_threshold>0 (computes flows from predicted masks on GPU)
    device: torch device
        what compute hardware to use to run the code (GPU VS CPU)
    nclasses:
        number of output classes of the network (Omnipose=4,Cellpose=3)
    dim: int
        dimensionality of data / model output
    eps: float
        internal epsilon parameter for (H)DBSCAN
    hdbscan: 
        use better, but much SLOWER, hdbscan clustering algorithm (experimental)
    flow_factor:
        multiple to increase flow magnitdue (used in 3D only, experimental)
    debug:
        option to return list of unique mask labels as a fourth output (for debugging only)

    Returns
    -------------
    mask: int, ND array
        label matrix
    p: float32, ND array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. 
    tr: float32, ND array
        intermediate locations of each pixel during dynamics,
        size [axis x niter x Ly x Lx] or [axis x niter x Lz x Ly x Lx]. 
        For debugging/paper figures, very slow. 
    
    """
    # Min size taken to be 9px or 27 voxels etc. if not specified 
    if min_size is None:
        min_size = 3**dim
    
    labels = None
    
    if verbose:
        omnipose_logger.info('mask_threshold is %f',mask_threshold)
        if omni and (not SKIMAGE_ENABLED):
             omnipose_logger.warning('Omni enabled but skimage not enabled')
    
    # inds very useful for debugging and figures; allows us to easily specify specific indices for Euler integration
    if inds is not None:
        mask = np.zeros_like(dist,dtype=np.int32)
        # print('info', mask.shape, inds.shape)
        mask[tuple(inds)] = 1
    else:
        if omni and SKIMAGE_ENABLED:
            if verbose:
                omnipose_logger.info('Using hysteresis threshold.')
            mask = filters.apply_hysteresis_threshold(dist, mask_threshold-1, mask_threshold) # good for thin features
            inds = np.array(np.nonzero(mask)).astype(np.int32)
        else:
            mask = dist > mask_threshold # analog to original iscell=(cellprob>cellprob_threshold)
            inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32) ### that dP[0] is a big bug... only first component!!!

    iscell = mask.copy()
    
    # mask at this point is a cell cluster binary map, not labels. If nclasses>1, we can do instance segmentation. 
    if np.any(mask) and nclasses>1: 
        
        #preprocess flows
        if omni and OMNI_INSTALLED:
            if bd is None:
                bd = np.ones_like(mask).astype(np.float)

            # the interpolated version of div_rescale is detrimental in 3D
            # the problem is thin sections where the
            if 1:#dim==2:
                dP_ = div_rescale(dP,mask) / rescale ##### omnipose.core.div_rescale
            else:
                dP_ = utils.normalize_field(dP)
            # print('rescaling with boundary output')
            # dP_ = bd_rescale(dP,mask, 4*bd) / rescale ##### omnipose.core.div_rescale

            # dP_ = dP.copy()
            if dim>2:
                dP_ *= flow_factor
                print('dP_ times {} for >2d, still experimenting'.format(flow_factor))

        else:
            dP_ = dP * mask / 5.
                
        if boundary_seg: # new tactic is to use flow to compute boundaries, including self-contact ones
            if verbose:
                omnipose_logger.info('doing new boundary seg')
            bd = get_boundary(dP,mask)
            mask, bounds, _ = boundary_to_masks(bd,mask)
            # mask = bounds # test to see if boundary multiplied masks could work 
            # compatibility 
            p = np.zeros([2,1,1])
            tr = []
            
        else: # do the ol' Euler-integration + clustering 


            # the clustering algorithm requires far fewer iterations because it 
            # can handle subpixel separation to define blobs, wheras the thresholding method
            # requires blobs to be separated by more than 1 pixel 
            if cluster:
                # niter = get_niter(dist)
                # niter = int(dist_to_diam(dist[dist>0],n=mask.ndim))
                niter = int(diameters(mask,dist))

            # follow flows
            if p is None:
                p, inds, tr = follow_flows(dP_, inds, niter=niter, interp=interp,
                                           use_gpu=use_gpu, device=device, omni=omni,
                                           calc_trace=calc_trace, verbose=verbose)
            else:
                tr = []
                inds = np.stack(np.nonzero(mask))
                if verbose:
                    omnipose_logger.info('p given')

            #calculate masks
            if omni and OMNI_INSTALLED:
                mask, labels = get_masks(p, bd, dist, mask, inds,nclasses, cluster=cluster,
                                         diam_threshold=diam_threshold, verbose=verbose, 
                                         eps=eps, hdbscan=hdbscan) ##### omnipose.core.get_masks
            else:
                mask = get_masks_cp(p, iscell=mask, flows=dP, use_gpu=use_gpu) ### just get_masks
        
            bounds = find_boundaries(mask)
            
        # flow thresholding factored out of get_masks
        # still could be useful for boundaries! Need to put in the self-contact boundaries as input <<<<<<
        # also can now turn on for do_3D... 
        if not do_3D: 
            shape0 = dP.shape[1:]
            flows = dP
            if mask.max()>0 and flow_threshold is not None and flow_threshold > 0 and flows is not None:
                mask = remove_bad_flow_masks(mask, flows, bounds, threshold=flow_threshold, use_gpu=use_gpu, device=device, omni=omni)
                _,mask = np.unique(mask, return_inverse=True)
                mask = np.reshape(mask, shape0).astype(np.int32)
        
        
    else: # nothing to compute, just make it compatible
        omnipose_logger.info('No cell pixels found.')
        p = np.zeros([2,1,1])
        tr = []
        bounds = mask
        # mask = np.zeros(resize,dtype=np.uint8) if resize is not None else np.zeros_like(dist) not necessary, would be zeros 
    
    
    
    # Resize mask, semantic or instance 
    if resize is not None:
        if verbose:
            omnipose_logger.info(f'resizing output with resize = {resize}')
        # mask = resize_image(mask, resize[0], resize[1], interpolation=cv2.INTER_NEAREST).astype(np.int32) 
        mask = zoom(mask, resize/np.array(mask.shape), order=0).astype(np.int32) 
    
    # need to reconsider this for self-contact 
    # could fill the region internal to boundaries, aka mask0
    mask = fill_holes_and_remove_small_masks(mask, min_size=min_size, dim=dim)*iscell ##### utils.fill_holes_and_remove_small_masks
        
    # print('warning, temp disable remove small masks')
    fastremap.renumber(mask,in_place=True) #convenient to guarantee non-skipped labels

    # print('maskinfo',mask.shape,len(np.unique(mask)))
    # moving the cleanup to the end helps avoid some bugs arising from scaling...
    # maybe better would be to rescale the min_size and hole_size parameters to do the
    # cleanup at the prediction scale, or switch depending on which one is bigger... 
    
    ret = [mask, p, tr, bounds]
    
    if debug:
        ret += [labels]
 
    return ret


# Omnipose requires (a) a special suppressed Euler step and (b) a special mask reconstruction algorithm. 

# no reason to use njit here except for compatibility with jitted fuctions that call it 
#this way, the same factor is used everywhere (CPU+-interp, GPU)
@njit()
def step_factor(t):
    """ Euler integration suppression factor.
    
    Conveneient wrapper function allowed me to test out several supression factors. 
    
    Parameters
    -------------
    t: int
        time step
    """
    return (1+t)

def div_rescale(dP,mask):
    """
    Normalize the flow magnitude to rescaled 0-1 divergence. 
    
    Parameters
    -------------
    dP: float, ND array
        flow field 
    mask: int, ND array
        label matrix
        
    Returns
    -------------
    dP: float, ND array
        rescaled flow field
    
    """
    dP = dP.copy()
    dP *= mask 
    dP = utils.normalize_field(dP)
    # div = utils.normalize99(likewise(dP))
    div = utils.normalize99(divergence(dP))
    dP *= div
    return dP

def sigmoid(x):
    """The sigmoid function."""
    return 1 / (1 + np.exp(-x))

# def bd_rescale(dP,mask,bd):
#     dP = dP.copy()
#     dP *= mask 
#     dP = utils.normalize_field(dP)
#     w = np.stack([bd]*mask.ndim)
#     dP *= sigmoid(bd)
#     return dP

def divergence(f,sp=None):
    """ Computes divergence of vector field
    
    Parameters
    -------------
    f: ND array, float
        vector field components [Fx,Fy,Fz,...]
    sp: ND array, float
        spacing between points in respecitve directions [spx, spy, spz,...]
        
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])


def get_masks(p, bd, dist, mask, inds, nclasses=4,cluster=False,
              diam_threshold=12., eps=None, hdbscan=False, verbose=False):
    """Omnipose mask recontruction algorithm.
    
    This function is called after dynamics are run. The final pixel coordinates are provided, 
    and cell labels are assigned to clusters found by labelling the pixel clusters after rounding
    the coordinates (snapping each pixel to the grid and labelling the resulting binary mask) or 
    by using DBSCAN or HDBSCAN for sub-pixel clustering. 
    
    Parameters
    -------------
    p: float32, ND array
        final locations of each pixel after dynamics
    bd: float, ND array
        boundary field
    dist: float, ND array
        distance field
    mask: bool, ND array
        binary cell mask
    inds: int, ND array 
        initial indices of pixels for the Euler integration [npixels x ndim]
    nclasses: int
        number of prediciton classes
    cluster: bool
        use DBSCAN clustering instead of coordinate thresholding
    diam_threshold: float
        mean diameter under which clustering will be turned on automatically
    eps: float
        internal espilon parameter for (H)DBSCAN
    hdbscan: bool
        use better, but much SLOWER, hdbscan clustering algorithm
    verbose: bool
        option to print more info to log file
    
    Returns
    -------------
    mask: int, ND array
        label matrix
    labels: int, list
        all unique labels 
    """
    if nclasses >= 4:
        dt = np.abs(dist[mask]) #abs needed if the threshold is negative
        d = dist_to_diam(dt,mask.ndim) 

    else: #backwards compatibility, doesn't help for *clusters* of thin/small cells
        d = diameters(mask,dist)
    
    if eps is None:
        eps = 2**0.5
    # The mean diameter can inform whether or not the cells are too small to form contiguous blobs.
    # My first solution was to upscale everything before Euler integration to give pixels 'room' to
    # stay together. My new solution is much better: use a clustering algorithm on the sub-pixel coordinates
    # to assign labels. It works just as well and is faster because it doesn't require increasing the 
    # number of points or taking time to upscale/downscale the data. Users can toggle cluster on manually or
    # by setting the diameter threshold higher than the average diameter of the cells. 
    if verbose:
        omnipose_logger.info('Mean diameter is %f'%d)

    if d <= diam_threshold: #diam_threshold needs to change for 3D
        cluster = True
        if verbose and not cluster:
            omnipose_logger.info('Turning on subpixel clustering for label continuity.')
    
    cell_px = tuple(inds)
    coords = np.nonzero(mask)
    newinds = p[(Ellipsis,)+cell_px].T
    mask = np.zeros(p.shape[1:],np.uint32)
    
    # the eps parameter needs to be opened as a parameter to the user
    if verbose:
        omnipose_logger.info('cluster: {}, SKLEARN_ENABLED: {}'.format(cluster,SKLEARN_ENABLED))
    if cluster and SKLEARN_ENABLED:
        startTime = time.time()
        if verbose:
            alg = ['','H']
            omnipose_logger.info('Doing {}DBSCAN clustering with eps={}'.format(alg[hdbscan],eps))
        
        if hdbscan and HDBSCAN_ENABLED:
            clusterer = HDBSCAN(cluster_selection_epsilon=eps,
                                # allow_single_cluster=True,
                                min_samples=3)
        else:
            clusterer = DBSCAN(eps=eps, min_samples=5, n_jobs=-1)
        
        clusterer.fit(newinds)
        labels = clusterer.labels_
        executionTime = (time.time() - startTime)
        
        if verbose:
            print('Execution time in seconds: ' + str(executionTime))
            print('{} unique labels found'.format(len(np.unique(labels))-1),newinds.shape)

        #### snapping outliers to nearest cluster 
        snap = 1
        if snap:
            nearest_neighbors = NearestNeighbors(n_neighbors=50)
            neighbors = nearest_neighbors.fit(newinds)
            o_inds= np.where(labels==-1)[0]
            if len(o_inds)>1:
                outliers = [newinds[i] for i in o_inds]
                distances, indices = neighbors.kneighbors(outliers)
                # indices,o_inds

                ns = labels[indices]
                # if len(ns)>0:
                l = [n[np.where(n!=-1)[0][0] if np.any(n!=-1) else 0] for n in ns]
                # l = [n[(np.where(n!=-1)+(0,))[0][0] ] for n in ns]
                labels[o_inds] = l

        ###
        mask[cell_px] = labels+1 # outliers have label -1
    else: #this branch can have issues near edges 
        newinds = np.rint(newinds.T).astype(int)
        new_px = tuple(newinds)
        skelmask = np.zeros_like(dist, dtype=bool)
        skelmask[new_px] = 1

        #disconnect skeletons at the edge, 5 pixels in 
        border_mask = np.zeros(skelmask.shape, dtype=bool)
        border_px =  border_mask.copy()
        border_mask = binary_dilation(border_mask, border_value=1, iterations=5)

        border_px[border_mask] = skelmask[border_mask]
        if verbose:
             omnipose_logger.info('nclasses: {}, mask.ndim: {}'.format(nclasses,mask.ndim))
        if nclasses == mask.ndim+2: #can use boundary to erase joined edge skelmasks 
            border_px[bd>-1] = 0
            if verbose:
                omnipose_logger.info('Using boundary output to split edge defects.')
        # else: #otherwise do morphological opening to attempt splitting 
        #     # border_px = binary_opening(border_px,border_value=0,iterations=3)
        #     print('BBBBB')

        skelmask[border_mask] = border_px[border_mask]

        if SKIMAGE_ENABLED:
            cnct = skelmask.ndim #-1
            labels = measure.label(skelmask,connectivity=cnct) #is this properly generalized to ND? seems like it works
        else:
            labels = label(skelmask)[0]
        mask[cell_px] = labels[new_px]
        
    if verbose:
        omnipose_logger.info('Done finding masks.')
    return mask, labels


# Generalizing to ND. Again, torch required but should be plenty fast on CPU too compared to jitted but non-explicitly-parallelized CPU code.
# also should just rescale to desired resolution HERE instead of rescaling the masks later... <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# grid_sample will only work for up to 5D tensors (3D segmentation). Will have to address this shortcoming if we ever do 4D. 
# I got rid of the map_coordinates branch, I tested execution times and pytorch implemtation seems as fast or faster
def steps_interp(p, dP, niter, use_gpu=True, device=None, omni=True, calc_trace=False, calc_bd=False, verbose=False):
    """Euler integration of pixel locations p subject to flow dP for niter steps in N dimensions. 
    
    Parameters
    ----------------
    p: float32, ND array
        pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid)
    dP: float32, ND array
        flows [axis x Lz x Ly x Lx]
    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------
    p: float32, ND array
        final locations of each pixel after dynamics

    """
    align_corners = True
    mode = 'bilinear'
    d = dP.shape[0] # number of components = number of dimensions 
    shape = dP.shape[1:] # shape of component array is the shape of the ambient volume 
    inds = list(range(d))[::-1] # grid_sample requires a particular ordering 
    
    if verbose:
        startTime = time.time()
    
    if device is None:
        if use_gpu:
            device = torch_GPU
        else:
            device = torch_CPU
    # for now, looks like grid_sampler_2d is not implemented for mps
    # so it is much faster to just default to CPU instead of allowing for fallback
    if ARM:
        device = torch_CPU
    shape = np.array(shape)[inds]-1.  # dP is d.Ly.Lx, inds flips this to flipped X-1, Y-1, ...

    # for grid_sample to work, we need im,pt to be (N,C,H,W),(N,H,W,2) or (N,C,D,H,W),(N,D,H,W,3). The 'image' getting interpolated
    # is the flow, which has d=2 channels in 2D and 3 in 3D (d vector components). Output has shape (N,C,H,W) or (N,C,D,H,W)
    pt = torch.from_numpy(p[inds].T).float().to(device)
    # print('pt shape',pt.shape)
    pt0 = pt.clone() # save first
    for k in range(d):
        pt = pt.unsqueeze(0) # get it in the right shape
    flow = torch.from_numpy(dP[inds]).float().to(device).unsqueeze(0) #covert flow numpy array to tensor on GPU, add dimension 
    # print('shapes',p.shape,dP.shape,pt.shape)

    # we want to normalize the coordinates between 0 and 1. To do this, 
    # we divide the coordinates by the shape along that dimension. To symmetrize,
    # we then multiply by 2 and subtract 1. I
    # We also need to rescale the flow by the same factor, but no shift of -1. 
    
    for k in range(d): 
        pt[...,k] = 2*pt[...,k]/shape[k] - 1
        flow[:,k] = 2*flow[:,k]/shape[k]
    
    # make an array to track the trajectories 
    if calc_trace:
        trace = torch.clone(pt).detach()
        # trace = torch.zeros((niter,)+pt.shape) # slower to preallocate...
        # print('trace shape',trace.shape)

    # init 
    if omni and OMNI_INSTALLED:
        dPt0 = torch.nn.functional.grid_sample(flow, pt, mode=mode, align_corners=align_corners)
        # r = torch.zeros_like(p)

    #here is where the stepping happens 
    for t in range(niter):
        if calc_trace:
            trace = torch.cat((trace,pt))
            # trace[t] = pt.detach()
        # align_corners default is False, just added to suppress warning
        dPt = torch.nn.functional.grid_sample(flow, pt, mode=mode, align_corners=align_corners)#see how nearest changes things 
        ### here is where I could add something for a potential, random step, etc. 

        # for k in range(d): 
        #     r[...,k] = pt[...,k].T - pt[...,k]
        # might be way too much or 100s of thousands of points. Instead, maybe could just smooth out an image of density
        # and then take the gradient to 


        if omni and OMNI_INSTALLED:
            dPt = (dPt+dPt0) / 2. # average with previous flow 
            dPt0 = dPt.clone() # update old flow 
            dPt /= step_factor(t) # suppression factor 

        for k in range(d): #clamp the final pixel locations
            pt[...,k] = torch.clamp(pt[...,k] + dPt[:,k], -1., 1.)
        
        # # differene gets rid pf 
        # r = (torch.sum((pt-pt0)**2,axis=-1))**0.5
        # r *= 0.5
        # for k in range(d): 
        #     r[...,k] *= shape[k]
        # # print(pt[:10,:10].cpu().numpy())
        # print('r', r.squeeze().cpu().numpy()[:10])
        
        # snapping to coordinate locations is no good... distance of points to all original 
    #undo the normalization from before, reverse order of operations 
    pt = (pt+1)*0.5
    for k in range(d): 
        pt[...,k] *= shape[k]

    if calc_trace:
        trace = (trace+1)*0.5
        for k in range(d): 
            trace[...,k] *= shape[k]
        # print('trace shape',trace.shape)

    #pass back to cpu
    if calc_trace:
        tr =  trace[...,inds].cpu().numpy().squeeze().T
    else:
        tr = None

    p =  pt[...,inds].cpu().numpy().squeeze().T

    if verbose:
        executionTime = (time.time() - startTime)
        omnipose_logger.info('steps_interp() execution time: {0:.3g} sec'.format(executionTime))
    return p, tr

@njit('(float32[:,:,:,:],float32[:,:,:,:], int32[:,:], int32)', nogil=True)
def steps3D(p, dP, inds, niter):
    """ Run dynamics of pixels to recover masks in 3D.
    
    Euler integration of dynamics dP for niter steps.

    Parameters
    ----------------
    p: float32, 4D array
        pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid)
    dP: float32, 4D array
        flows [axis x Lz x Ly x Lx]
    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x 3]
    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------
    p: float32, 4D array
        final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):
        #pi = p.astype(np.int32)
        for j in range(inds.shape[0]):
            z = inds[j,0]
            y = inds[j,1]
            x = inds[j,2]
            p0, p1, p2 = int(p[0,z,y,x]), int(p[1,z,y,x]), int(p[2,z,y,x])
            p[0,z,y,x] = min(shape[0]-1, max(0, p[0,z,y,x] + dP[0,p0,p1,p2]))
            p[1,z,y,x] = min(shape[1]-1, max(0, p[1,z,y,x] + dP[1,p0,p1,p2]))
            p[2,z,y,x] = min(shape[2]-1, max(0, p[2,z,y,x] + dP[2,p0,p1,p2]))
    return p, None

@njit('(float32[:,:,:], float32[:,:,:], int32[:,:], int32, boolean, boolean)', nogil=True)
def steps2D(p, dP, inds, niter, omni=True, calc_trace=False):
    """ Run dynamics of pixels to recover masks in 2D.
    
    Euler integration of dynamics dP for niter steps.

    Parameters
    ----------------
    p: float32, 3D array
        pixel locations [axis x Ly x Lx] (start at initial meshgrid)
    dP: float32, 3D array
        flows [axis x Ly x Lx]
    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x 2]
    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------
    p: float32, 3D array
        final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    if calc_trace:
        Ly = shape[0]
        Lx = shape[1]
        tr = np.zeros((niter,2,Ly,Lx))
    for t in range(niter):
        for j in range(inds.shape[0]):
            if calc_trace:
                tr[t] = p.copy()
            # starting coordinates
            y = inds[j,0]
            x = inds[j,1]
            p0, p1 = int(p[0,y,x]), int(p[1,y,x])
            step = dP[:,p0,p1]
            if omni and OMNI_INSTALLED:
                step /= step_factor(t)
            for k in range(p.shape[0]):
                p[k,y,x] = min(shape[k]-1, max(0, p[k,y,x] + step[k]))
    return p, tr

# now generalized and simplified. Will work for ND if dependencies are updated. 
def follow_flows(dP, inds, niter=200, interp=True, use_gpu=True, 
                 device=None, omni=True, calc_trace=False, verbose=False):
    """ define pixels and run dynamics to recover masks in 2D
    
    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------
    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
    inds: int, ND array 
        initial indices of pixels for the Euler integration 
    niter: int 
        number of iterations of dynamics to run
    interp: bool 
        interpolate during dynamics 
    use_gpu: bool 
        use GPU to run interpolated dynamics (faster than CPU)   
    omni: bool 
        flag to enable Omnipose suppressed Euler integration etc. 
    calc_trace: bool 
        flag to store and retrun all pixel coordinates during Euler integration (slow)

    Returns
    ---------------
    p: float32, ND array
        final locations of each pixel after dynamics
    inds: int, ND array
        initial indices of pixels for the Euler integration [npixels x ndim]
    tr: float32, ND array
        list of intermediate pixel coordinates for each step of the Euler integration

    """
    d = dP.shape[0] # dimension is the number of flow components 
    shape = np.array(dP.shape[1:]).astype(np.int32) # shape of masks is the shape of the component field
    niter = np.uint32(niter) 
    grid = [np.arange(shape[i]) for i in range(d)]
    p = np.meshgrid(*grid, indexing='ij')
    # not sure why, but I had changed this to float64 at some point... tests showed that map_coordinates expects float32
    # possible issues elsewhere? 
    p = np.array(p).astype(np.float32)
    # added inds for debugging while preserving backwards compatibility 
    
    if inds.ndim < 2 or inds.shape[0] < d:
        print(inds.shape,d)
        omnipose_logger.warning('WARNING: no mask pixels found')
        return p, inds, None

    cell_px = (Ellipsis,)+tuple(inds)


    if not interp:
        omnipose_logger.warning('WARNING: not interp')
        if d==2:
            p, tr = steps2D(p, dP.astype(np.float32), inds, niter,omni=omni,calc_trace=calc_trace)
        elif d==3:
            p, tr = steps3D(p, dP, inds, niter)
        else:
            omnipose_logger.warning('No non-interp code available for non-2D or -3D inputs.')

    else:
        p_interp, tr = steps_interp(p[cell_px], dP, niter, use_gpu=use_gpu,
                                    device=device, omni=omni, calc_trace=calc_trace, 
                                    verbose=verbose)
        p[cell_px] = p_interp
    return p, inds, tr

def remove_bad_flow_masks(masks, flows, bounds=None, threshold=0.4, use_gpu=False, device=None, omni=True):
    """ remove masks which have inconsistent flows 
    
    Uses metrics.flow_error to compute flows from predicted masks 
    and compare flows to predicted flows from network. Discards 
    masks with flow errors greater than the threshold.

    Parameters
    ----------------
    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    flows: float, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
    threshold: float
        masks with flow error greater than threshold are discarded

    Returns
    ---------------
    masks: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    merrors, _ =  flow_error(masks, flows, bounds, use_gpu, device, omni) ##### metrics.flow_error
    badi = 1+(merrors>threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks

def flow_error(maski, dP_net, bounds=None, use_gpu=False, device=None, omni=True):
    """ error in flows from predicted masks vs flows predicted by network run on image

    This function serves to benchmark the quality of masks, it works as follows
    1. The predicted masks are used to create a flow diagram
    2. The mask-flows are compared to the flows that the network predicted

    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
    Masks with flow_errors greater than 0.4 are discarded by default. Setting can be
    changed in Cellpose.eval or CellposeModel.eval.

    Parameters
    ------------
    maski: ND-array (int) 
        masks produced from running dynamics on dP_net, 
        where 0=NO masks; 1,2... are mask labels
    dP_net: ND-array (float) 
        ND flows where dP_net.shape[1:] = maski.shape

    Returns
    ------------
    flow_errors: float array with length maski.max()
        mean squared error between predicted flows and flows from masks
    dP_masks: ND-array (float)
        ND flows produced from the predicted masks
    
    """
    if dP_net.shape[1:] != maski.shape:
        omnipose_logger.info('ERROR: net flow is not same size as predicted masks')
        return

    # ensure unique masks
    # maski = np.reshape(np.unique(maski.astype(np.float32), return_inverse=True)[1], maski.shape)
    fastremap.renumber(maski,in_place=True)

    # flows predicted from estimated masks and boundaries
    idx = -1 # flows are the last thing returned now
    dP_masks = masks_to_flows(maski, boundaries=bounds, use_gpu=use_gpu, device=device, omni=omni)[idx] ##### dynamics.masks_to_flows
    # difference between predicted flows vs mask flows
    flow_errors = np.zeros(maski.max())
    for i in range(dP_masks.shape[0]):
        flow_errors += mean((dP_masks[i] - dP_net[i]/5.)**2, maski, #the /5 is to compensate for the *5 we do for training
                            index=np.arange(1, maski.max()+1))
    return flow_errors, dP_masks



### Section III: training

# Omnipose has special training settings. Loss function and augmentation. 
# Spacetime segmentation: augmentations need to treat time differently 
# Need to assume a particular axis is the temporal axis; most convenient is tyx. 
def random_rotate_and_resize(X, Y=None, scale_range=1., gamma_range=0.5, tyx = (224,224), 
                             do_flip=True, rescale=None, inds=None, nchan=1, nclasses=4):
    """ augmentation by random rotation and resizing

        X and Y are lists or arrays of length nimg, with channels x Lt x Ly x Lx (channels optional, Lt only in 3D)

        Parameters
        ----------
        X: float, list of ND arrays
            list of image arrays of size [nchan x Lt x Ly x Lx] or [Lt x Ly x Lx]
        Y: float, list of ND arrays
            list of image labels of size [nlabels x Lt x Ly x Lx] or [Lt x Ly x Lx]. The 1st channel
            of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
            If Y.shape[0]==3, then the labels are assumed to be [cell probability, T flow, Y flow, X flow]. 
        scale_range: float (optional, default 1.0)
            Range of resizing of images for augmentation. Images are resized by
            (1-scale_range/2) + scale_range * np.random.rand()
        gamma_range: float
           images are gamma-adjusted im**gamma for gamma in (1-gamma_range,1+gamma_range) 
        tyx: int, tuple
            size of transformed images to return, e.g. (Ly,Lx) or (Lt,Ly,Lx)
        do_flip: bool (optional, default True)
            whether or not to flip images horizontally
        rescale: float, array or list
            how much to resize images by before performing augmentations
        inds: int, list
            image indices (for debugging)
        nchan: int
            number of channels the images have 

        Returns
        -------
        imgi: float, ND array
            transformed images in array [nimg x nchan x xy[0] x xy[1]]
        lbl: float, ND array
            transformed labels in array [nimg x nchan x xy[0] x xy[1]]
        scale: float, 1D array
            scalar(s) by which each image was resized

    """
    dist_bg = 5 # background distance field was set to -dist_bg; now is variable 
    dim = len(tyx) # 2D will just have yx dimensions, 3D will be tyx
    
    nimg = len(X)
    imgi  = np.zeros((nimg, nchan)+tyx, np.float32)
        
    if Y is not None:
        for n in range(nimg):
            masks = Y[n] # now assume straight labels 
            iscell = masks>0
            if np.sum(iscell)==0:
                error_message = 'No cell pixels. Index is'+str(n)
                omnipose_logger.critical(error_message)
                raise ValueError(error_message)
            Y[n] = np.stack([masks,iscell])
    
    nt = 2 # instance seg (labels), semantic seg (cellprob)
    if nclasses==4:
        nt += 3+dim # add boundary, distance, weight, flow components
        
    lbl = np.zeros((nimg, nt)+tyx, np.float32)
    scale = np.zeros((nimg,dim), np.float32)
    
    for n in range(nimg):
        img = X[n].copy()
        y = None if Y is None else Y[n]
        # use recursive function here to pass back single image that was cropped appropriately 
        # # print(y.shape)
        # skimage.io.imsave('/home/kcutler/DataDrive/debug/img_orig.png',img[0])
        # skimage.io.imsave('/home/kcutler/DataDrive/debug/label_orig.tiff',y[n]) #so at this point the bad label is just fine 
        imgi[n], lbl[n], scale[n] = random_crop_warp(img, y, nt, tyx, nchan, scale[n], 
                                                     rescale is None if rescale is None else rescale[n], 
                                                     scale_range, gamma_range, do_flip, 
                                                     inds is None if inds is None else inds[n], dist_bg)
        
    return imgi, lbl, np.mean(scale) #for size training, must output scalar size (need to check this again)

# This function allows a more efficient implementation for recursively checking that the random crop includes cell pixels.
# Now it is rerun on a per-image basis if a crop fails to capture .1 percent cell pixels (minimum). 
def random_crop_warp(img, Y, nt, tyx, nchan, scale, rescale, scale_range, gamma_range, 
                     do_flip, ind, dist_bg, depth=0):
    """
    This sub-fuction of `random_rotate_and_resize()` recursively performs random cropping until 
    a minimum number of cell pixels are found, then proceeds with augemntations. 
    
    Parameters
    ----------
    X: float, list of ND arrays
        image array of size [nchan x Lt x Ly x Lx] or [Lt x Ly x Lx]
    Y: float, ND array
        image label array of size [nlabels x Lt x Ly x Lx] or [Lt x Ly x Lx].. The 1st channel
        of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
        If Y.shape[0]==3, then the labels are assumed to be [cell probability, T flow, Y flow, X flow]. 
    tyx: int, tuple
        size of transformed images to return, e.g. (Ly,Lx) or (Lt,Ly,Lx)
    nchan: int
        number of channels the images have 
    rescale: float, array or list
        how much to resize images by before performing augmentations
    scale_range: float
        Range of resizing of images for augmentation. Images are resized by
        (1-scale_range/2) + scale_range * np.random.rand()
    gamma_range: float
       images are gamma-adjusted im**gamma for gamma in (1-gamma_range,1+gamma_range) 
    do_flip: bool (optional, default True)
        whether or not to flip images horizontally
    ind: int
        image index (for debugging)
    dist_bg: float
        nonegative value X for assigning -X to where distance=0 (deprecated, now adapts to field values)
    depth: int
        how many time this function has been called on an image 

    Returns
    -------
    imgi: float, ND array
        transformed images in array [nchan x xy[0] x xy[1]]
    lbl: float, ND array
        transformed labels in array [nchan x xy[0] x xy[1]]
    scale: float, 1D array
        scalar by which the image was resized
    
    """
    
    dim = len(tyx)
    # np.random.seed(depth)
    if depth>100:
        error_message = 'Sparse or over-dense image detected. Problematic index is: '+str(ind)+' Image shape is: '+str(img.shape)+' tyx is: '+str(tyx)+' rescale is '+str(rescale)
        omnipose_logger.critical(error_message)
        skimage.io.imsave('/home/kcutler/DataDrive/debug/img'+str(depth)+'.png',img[0]) 
        raise ValueError(error_message)
    
    if depth>200:
        error_message = 'Recusion depth exceeded. Check that your images contain cells and background within a typical crop. Failed index is: '+str(ind)
        omnipose_logger.critical(error_message)
        raise ValueError(error_message)
        return
    
    # labels that will be passed to the loss function
    # 
    lbl = np.zeros((nt,)+tyx, np.float32)
    
    numpx = np.prod(tyx)
    if Y is not None:
        labels = Y.copy()
        # We want the scale distibution to have a mean of 1
        # There may be a better way to skew the distribution to
        # interpolate the parameter space without skewing the mean 
        ds = scale_range/2
        scale = np.random.uniform(low=1-ds,high=1+ds,size=dim) #anisotropic scaling 
        if rescale is not None:
            scale *= 1. / rescale

    # image dimensions are always the last <dim> in the stack (again, convention here is different)
    s = img.shape[-dim:]

    # generate random augmentation parameters
    dg = gamma_range/2 
    theta = np.random.rand() * np.pi * 2

    # first two basis vectors in any dimension 
    v1 = [0]*(dim-1)+[1]
    v2 = [0]*(dim-2)+[1,0]
    # M = mgen.rotation_from_angle_and_plane(theta,v1,v2) #not generalizing correctly to 3D? had -theta before  
    M = mgen.rotation_from_angle_and_plane(-theta,v2,v1).dot(np.diag(scale)) #equivalent
    # could define v3 and do another rotation here and compose them 

    axes = range(dim)
    s = img.shape[-dim:]
    rt = (np.random.rand(dim,) - .5) #random translation -.5 to .5
    dxy = [rt[a]*(np.maximum(0,s[a]-tyx[a])) for a in axes]
    
    c_in = 0.5 * np.array(s) + dxy
    c_out = 0.5 * np.array(tyx)
    offset = c_in - np.dot(np.linalg.inv(M), c_out)
    
    # M = np.vstack((M,offset))
    mode = 'reflect'
    if Y is not None:
        for k in [0,1]:#[i for i in range(nt) if i not in range(2,5)]: used to do first two and flows, now just first two
            l = labels[k].copy()
            if k==0:
                lbl[k] = do_warp(l, M, tyx, offset=offset, order=0, mode=mode) # order 0 is 'nearest neighbor'
                # check to make sure the region contains at enough cell pixels; if not, retry
                cellpx = np.sum(lbl[k]>0)
                cutoff = (numpx/10**(dim+1)) # .1 percent of pixels must be cells
                # print('after warp',len(np.unique(lbl[k])),np.max(lbl[k]),np.min(lbl[k]),cutoff,numpx, cellpx, theta)
                if cellpx<cutoff:# or cellpx==numpx: # had to disable the overdense feature for cyto2
                                #, may nto actually be a problem now anyway
                    # print('toosmall',nt)
                    # skimage.io.imsave('/home/kcutler/DataDrive/debug/img'+str(depth)+'.png',img[0])
                    # skimage.io.imsave('/home/kcutler/DataDrive/debug/training'+str(depth)+'.png',lbl[0])
                    return random_crop_warp(img, Y, nt, tyx, nchan, scale, rescale, scale_range, 
                                            gamma_range, do_flip, ind, dist_bg, depth=depth+1)
            else:
                lbl[k] = do_warp(l, M, tyx, offset=offset, mode=mode)
                # if k==1:
                #     print('fgd', np.sum(lbl[k]))
        
        # LABELS ARE NOW (masks,mask) for semantic seg + (bd,dist,weight,flows) for instance seg
        # semantic seg label transformations taken care of above, those are simple enough. Others
        # must be computed after mask transformations are made. 
        if nt > 2:
            l = lbl[0].astype(np.uint16)
            l, dist, T, mu = masks_to_flows(l,omni=True,dim=dim)
            cutoff = diameters(l,dist)/2
            lbl[2] = dist==1 # position 2 stores the boundary field
            smooth_dist = T
            smooth_dist[dist<=0] = - cutoff#-dist_bg
            lbl[3] = smooth_dist # position 3 stores the smooth distance field 
            lbl[-dim:] = mu*5.0 #oops, forgot this needs to be x5.0 for training
            # used to be that this put it in the same range as cellprob, but it still
            # puts it in the same range as the logits for the boundary, plus the magnitude makes 
            # for larger MSE
            
            # print('dists',np.max(dist),np.max(smooth_dist))
            # the black border may not be good in 3D, as it highlights a larger fraction? 
            mask = lbl[1] #binary mask 
            bg_edt = edt.edt(mask<0.5,black_border=True) #last arg gives weight to the border, which seems to always lose
            lbl[4] = (gaussian(1-np.clip(bg_edt,0,cutoff)/cutoff, 1)+0.5)


    # Makes more sense to spend time on image augmentations
    # after the label augmentation succeeds without triggering recursion 
    imgi  = np.zeros((nchan,)+tyx, np.float32)
    for k in range(nchan): # replace k with slice that handles when nchan=0
        I = do_warp(img[k], M, tyx, offset=offset, mode=mode)
        
        # gamma agumentation 
        gamma = np.random.uniform(low=1-dg,high=1+dg) 
        imgi[k] = I ** gamma
        
        # percentile clipping augmentation 
        dp = 10
        dpct = np.random.triangular(left=0, mode=0, right=dp, size=2) # weighted toward 0
        imgi[k] = utils.normalize99(imgi[k],upper=100-dpct[0],lower=dpct[1])
        
        # noise augmentation 
        if SKIMAGE_ENABLED:
            
            # imgi[k] = random_noise(utils.rescale(imgi[k]), mode="poisson")#, seed=None, clip=True)
            imgi[k] = random_noise(utils.rescale(imgi[k]), mode="poisson")#, seed=None, clip=True)
            
        else:
            #this is quite different
            # imgi[k] = np.random.poisson(imgi[k])
            print('warning,no randomnoise')
            
        # bit depth augmentation
        bit_shift = int(np.random.triangular(left=0, mode=8, right=16, size=1))
        im = (imgi[k]*(2**16-1)).astype(np.uint16)
        imgi[k] = utils.normalize99(im>>bit_shift)
    
    
    # Moved to the end because it conflicted with the recursion. 
    # Also, flipping the crop is ultimately equivalent and slightly faster.         
    # We now flip along every axis (randomly); could make do_flip a list to avoid some axes if needed
    if do_flip:
        for d in range(1,dim+1):
            flip = np.random.choice([0,1])
            if flip:
                imgi = np.flip(imgi,axis=-d) 
                if Y is not None:
                    lbl = np.flip(lbl,axis=-d)
                    if nt > 1:
                        lbl[-d] = -lbl[-d]        
        
    return imgi, lbl, scale

def do_warp(A,M,tyx,offset=0,order=1,mode='constant'):#,mode,method):
    """ Wrapper function for affine transformations during augmentation. 
    Uses scipy.ndimage.affine_transform().
        
    Parameters
    --------------
    A: NDarray, int or float
        input image to be transformed
    M: NDarray, float
        tranformation matrix
    order: int
        interpolation order, 1 is equivalent to 'nearest',
    """
    # dim = A.ndim'
    # if dim == 2:
    #     return cv2.warpAffine(A, M, rshape, borderMode=mode, flags=method)
    # else:
    #     return np.stack([cv2.warpAffine(A[k], M, rshape, borderMode=mode, flags=method) for k in range(A.shape[0])])
    # print('debug',A.shape,M.shape,tyx)
    
    return scipy.ndimage.affine_transform(A, np.linalg.inv(M), offset=offset, 
                                          output_shape=tyx, order=order, mode=mode)
    


def loss(self, lbl, y):
    """ Loss function for Omnipose.
    Parameters
    --------------
    lbl: ND-array, float
        transformed labels in array [nimg x nchan x xy[0] x xy[1]]
        lbl[:,0] cell masks
        lbl[:,1] thresholded mask layer
        lbl[:,2] boundary field
        lbl[:,3] smooth distance field 
        lbl[:,4] boundary-emphasized weights
        lbl[:,5:] flow components 
    
    y:  ND-tensor, float
        network predictions, with dimension D, these are:
        y[:,:D] flow field components at 0,1,...,D-1
        y[:,D] distance fields at D
        y[:,D+1] boundary fields at D+1
    
    """
    nt = lbl.shape[1]
    cellmask = lbl[:,1]

    cellmask = self._to_device(cellmask>0)#.bool()

    if nt==2: # semantic segmentation
        loss1 = self.criterion(y[:,0],cellmask) #MSE
        # loss1 = self.criterion17(y[:,0]*1.0,cellmask*1.0) 
        
        loss2 = self.criterion2(y[:,0],cellmask) #BCElogits 
        return loss1+loss2
        # return loss2
        
    else: #instance segmentation
        cellmask = cellmask.bool() #acts as a mask now, not output 
    
        # flow components are stored as the last self.dim slices 
        veci = self._to_device(lbl[:,5:]) 
        dist = lbl[:,3] # now distance transform replaces probability
        boundary =  lbl[:,2]

        w =  self._to_device(lbl[:,4])  
        dist = self._to_device(dist)
        boundary = self._to_device(boundary)
        flow = y[:,:self.dim] # 0,1,...self.dim-1
        dt = y[:,self.dim]
        bd = y[:,self.dim+1]
        a = 10.

        # stacked versions for weighting vector fields with scalars 
        wt = torch.stack([w]*self.dim,dim=1)
        ct = torch.stack([cellmask]*self.dim,dim=1) 

        #luckily, torch.gradient did exist after all and derivative loss was easy to implement. Could also fix divergenceloss, but I have not been using it. 
        # the rest seem good to go. 

        loss1 = 10.*self.criterion12(flow,veci,wt)  #weighted MSE 
        loss2 = self.criterion14(flow,veci,w,cellmask) #ArcCosDotLoss
        loss3 = self.criterion11(flow,veci,wt,ct)/a # DerivativeLoss
        loss4 = 2.*self.criterion2(bd,boundary) #BCElogits 
        loss5 = 2.*self.criterion15(flow,veci,w,cellmask) # loss on norm 
        loss6 = 2.*self.criterion12(dt,dist,w) #weighted MSE 
        loss7 = self.criterion11(dt.unsqueeze(1),dist.unsqueeze(1),w.unsqueeze(1),cellmask.unsqueeze(1))/a  
        loss8 = self.criterion16(flow,veci,cellmask) #divergence loss

        # print('loss1',loss1,loss1.type())
        # print('loss2',loss2,loss2.type())
        # print('loss3',loss3,loss3.type())
        # print('loss4',loss4,loss4.type())
        # print('loss5',loss5,loss5.type())
        # print('loss6',loss6,loss6.type())
        # print('loss7',loss7,loss7.type())

        return loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 +loss8


# used to recompute the smooth distance on transformed labels

#NOTE: in Omnipose, I do a pad-reflection to extend labels across the boundary so that partial cells are not
# as oddly distorted. This is not implemented here, so there is a discrepancy at image/volume edges. The 
# Omnipose variant is much closer to the edt edge behavior. A more sophisticated 'edge autofill' is really needed for
# a more robust approach (or just crop edges all the time). 
def smooth_distance(masks, dists=None, device=None):
    """
    A smooth fistance field generator implemented with pytorch. To reduce the effects of cut-off masks giving artifically 
    low distance values at image boundaries, masks are padded with reflection. 
    
    Parameters
    -------------
    masks: int, ND array
        labelled masks, 0 = background, 1,2,...,N = mask labels
    dists: ND array, float
        array of (nonnegative) distance field values
    device: torch device
        what compute hardware to use to run the code (GPU VS CPU)
        
    Returns
    -------------
    ND array, float
    
    """
    
    if device is None:
        device = torch.device('cuda')
    if dists is None:
        dists = edt.edt(masks)
        
    pad = 1
    
    masks_padded = np.pad(masks,pad)
    coords = np.nonzero(masks_padded)
    d = len(coords)
    idx = (3**d)//2 # center pixel index

    neigh = [[-1,0,1] for i in range(d)]
    steps = cartesian(neigh)
    neighbors = np.array([np.add.outer(coords[i],steps[:,i]) for i in range(d)]).swapaxes(-1,-2)
    # print('neighbors d', neighbors.shape)
    
    # get indices of the hupercubes sharing m-faces on the central n-cube
    sign = np.sum(np.abs(steps),axis=1) # signature distinguishing each kind of m-face via the number of steps 
    uniq = fastremap.unique(sign)
    inds = [np.where(sign==i)[0] for i in uniq] # 2D: [4], [1,3,5,7], [0,2,6,8]. 1-7 are y axis, 3-5 are x, etc. 
    fact = np.sqrt(uniq) # weighting factor for each hypercube group 
    
    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[tuple(neighbors)] #extract list of label values, 
    isneighbor = neighbor_masks == neighbor_masks[idx] 

    # set number of iterations
    n_iter = get_niter(dists)
    # n_iter = 20
    # print('n_iter',n_iter)
        
    nimg = neighbors.shape[1] // (3**d)
    pt = torch.from_numpy(neighbors).to(device)
    T = torch.zeros((nimg,)+masks_padded.shape, dtype=torch.float, device=device)#(nimg,)+
    isneigh = torch.from_numpy(isneighbor).to(device)
    for t in range(n_iter):
        T[(Ellipsis,)+tuple(pt[:,idx])] = eikonal_update_torch(T,pt,isneigh,d,inds,fact) 
        
    return T.cpu().squeeze().numpy()[tuple([slice(pad,-pad)]*d)]


### Section IV: duplicated mask recontruction

# this may still be in my local version of cellpose code

# I also have some edited trasnforms, namely 


### Section V: Helper functions duplicated from cellpose_omni, plan to find a way to merge them back without import loop

def get_masks_cp(p, iscell=None, rpad=20, flows=None, use_gpu=False, device=None):
    """ create masks using pixel convergence after running dynamics
    
    Makes a histogram of final pixel locations p, initializes masks 
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards 
    masks with flow errors greater than the threshold. 

    Parameters
    ----------------
    p: float32, 3D or 4D array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
    iscell: bool, 2D or 3D array
        if iscell is not None, set pixels that are 
        iscell False to stay in their original location.
    rpad: int (optional, default 20)
        histogram edge padding
    flows: float, 3D or 4D array (optional, default None)
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
        is not None, then masks with inconsistent flows are removed using 
        `remove_bad_flow_masks`.

    Returns
    ---------------
    M0: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims==3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                np.arange(shape0[2]), indexing='ij')
        elif dims==2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                     indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]
    
    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5-rpad, shape0[i]+.5+rpad, 1))

    h,_ = np.lib.histogramdd(pflows, bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]
    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims==3:
        expand = np.nonzero(np.ones((3,3,3)))
    else:
        expand = np.nonzero(np.ones((3,3)))
    for e in expand:
        e = np.expand_dims(e,1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter==0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i,e in enumerate(expand):
                epix = e[:,np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix>=0, epix<shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix]>2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter==4:
                pix[k] = tuple(pix[k])
    
    M = np.zeros(h.shape, np.int32)
    for k in range(len(pix)):
        M[pix[k]] = 1+k
        
    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]
    
    # remove big masks
    _,counts = np.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.4
    for i in np.nonzero(counts > big)[0]:
        M0[M0==i] = 0
    _,M0 = np.unique(M0, return_inverse=True)
    M0 = np.reshape(M0, shape0)

    # moved to compute masks
    # if M0.max()>0 and threshold is not None and threshold > 0 and flows is not None:
    #     M0 = remove_bad_flow_masks(M0, flows, threshold=threshold, use_gpu=use_gpu, device=device)
    #     _,M0 = np.unique(M0, return_inverse=True)
    #     M0 = np.reshape(M0, shape0).astype(np.int32)

    return M0


# duplicated from cellpose_omni temporarily, need to pass through spacetime before re-inserting 
def fill_holes_and_remove_small_masks(masks, min_size=15, hole_size=3, scale_factor=1, dim=2):
    """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)
    
    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes
    
    Parameters
    ----------------
    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1

    Returns
    ---------------
    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    # if masks.ndim==2 or dim>2:
    #     print('here')
        # formatting to integer is critical
        # need to test how it does with 3D
    masks = ncolor.format_labels(masks, min_area=min_size)#, clean=True)
        
    hole_size *= scale_factor
    
    slices = find_objects(masks)
    j = 0
    for i,slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i+1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            else:   
                hsz = np.count_nonzero(msk)*hole_size/100 #turn hole size into percentage
                #eventually the boundary output should be used to properly exclude real holes vs label gaps      
                if SKIMAGE_ENABLED: # Omnipose version (passes 2D tests)
                    pad = 1
                    unpad = tuple([slice(pad,-pad)]*msk.ndim) 
                    padmsk = remove_small_holes(np.pad(msk,pad,mode='constant'),hsz)
                    msk = padmsk[unpad]
                else: #Cellpose version
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = (j+1)
                j+=1
    return masks


    # if masks.ndim > 3 or masks.ndim < 2:
    #     raise ValueError('fill_holes_and_remove_small_masks takes 2D or 3D array, not %dD array'%masks.ndim)
    # slices = find_objects(masks)
    # j = 0
    # for i,slc in enumerate(slices):
    #     if slc is not None:
    #         msk = masks[slc] == (i+1)
    #         npix = msk.sum()
    #         if min_size > 0 and npix < min_size:
    #             masks[slc][msk] = 0
    #         else:    
    #             if msk.ndim==3:
    #                 for k in range(msk.shape[0]):
    #                     msk[k] = binary_fill_holes(msk[k])
    #             else:
    #                 msk = binary_fill_holes(msk)
    #             masks[slc][msk] = (j+1)
    #             j+=1
    # return masks


def get_boundary(mu,mask,contour=False,use_gpu=False,device=None):
    """
    mask can be binary 
    """
    d = mu.shape[0]
    pad = 1
    pad_seq = [(0,)*2]+[(pad,)*2]*d
    mu_pad = utils.normalize_field(np.pad(mu,pad_seq))
    lab_pad = np.pad(mask,pad)
    bd_pad = np.zeros_like(lab_pad,dtype=bool)
    # print(bd_pad.shape,bd_pad.dtype)
    # mu_pad = np.pad(mu,pad_seq)

    neigh = [[-1,0,1] for i in range(d)]
    steps = cartesian(neigh) # all the possible step sequences in ND    
    steps = np.array(list(set([tuple(s) for s in steps])-set([(0,)*d]))) # remove zero shift element 

    # first time to extract boundaries 
    bd_pad = _get_bd(steps, np.int32(lab_pad), mu_pad, bd_pad) 
    bd_pad = remove_small_objects(bd_pad,min_size=9)
    bd_pad[utils.get_spruepoints(bd_pad)] = False # remove spurs 
    
    unpad = tuple([slice(pad,-pad)]*d)
    
    #second time to parametrize
    # probably a way to do the boundary finding and stepping in the same step... 
    if contour:
        _,_,T,mu_pad = masks_to_flows(lab_pad,boundaries=bd_pad,
                              use_gpu=use_gpu,
                              device=device,smooth=1)
        
        step_ok, ind_shift, cross, dot = _get_bd(steps, lab_pad, mu_pad, bd_pad) 
        values = dot-cross # might be some cancellation here to leverage in computation earlier 
        bd_coords = np.array(np.nonzero(bd_pad))
        bd_inds = np.ravel_multi_index(bd_coords,bd_pad.shape)
        labs = np.take(lab_pad,bd_inds)
        unique_L = fastremap.unique(labs)
        contours = parametrize(steps,np.int32(labs),np.int32(unique_L),bd_inds,ind_shift,values,step_ok)

        contour_map = np.zeros(bd_pad.shape,dtype=np.int32)
        for contour in contours:
            coords_t = np.unravel_index(contour,bd_pad.shape)
            contour_map[coords_t] = np.arange(1,len(contour)+1)
            
        return contour_map[unpad]
    
    else:
        return bd_pad[unpad]


# numba does not wrok yet with this indexing... 
# @njit('(int64[:,:], int32[:,:], float64[:,:,:], boolean[:,:])', nogil=True)
def _get_bd(steps, lab_pad, mu_pad, bd_pad):
    
    get_bd = np.all(~bd_pad)
    axes = range(mu_pad.shape[0])
    mask_pad = lab_pad>0
    mag_pad = np.sqrt(np.sum(mu_pad**2,axis=0))
    coord = np.nonzero(mask_pad)
    coords = np.argwhere(mask_pad).T
    A = mu_pad[(Ellipsis,)+coord]
    
    if not get_bd:
        dot = []
        cross = []
        ind_shift = []
        step_ok = [] #whether or not this step will take you off the boundary 
    else:
        angles1 = []
        angles2 = []
        cutoff1 = np.pi*(1/3) # was 1/2, then 1/3, then 
        cutoff2 = np.pi*(3/4) # was 3/4, changed to 0.9, back to 3/4 
        # cutoff1 = np.pi*1/3
        # cutoff2 = np.pi*1/4 # too low finds sekeletons 

    for s in steps:
        # First see if the flow is parallel to the flow OPPOSITE the direction of the step 
        neigh_opp = tuple(coords-s[np.newaxis].T)
        B = mu_pad[(Ellipsis,)+neigh_opp]
        dot1 = np.sum(np.multiply(A,B),axis=0)
        
        if get_bd:
            angle1 = np.arccos(dot1.clip(-1,1))
            angle1[np.logical_and(mask_pad[coord],mask_pad[neigh_opp]==0)] = np.pi # consider all background pixels to be opposite

            # next see if the flow is parallel with the step itself 
            dot2 = np.sum([A[a]*(-s[a]) for a in axes],axis=0) #/ (mag_pad*mag_s)      
            angle2 = np.arccos(dot2.clip(-1,1))#*mag_pad[coord] # note the mag_pad multiplication here, attenuates 

            angles1.append(angle1>cutoff1)
            angles2.append(angle2>cutoff2)

        else:
            cross.append(np.cross(A,s,axisa=0))
            dot.append(dot1)
            coord_shift = tuple(coords[:,bd_pad[coord]]+s[np.newaxis].T)
            x = np.ravel_multi_index(coord_shift,bd_pad.shape)
            ind_shift.append(x)
            step_ok.append(np.logical_and(bd_pad[coord_shift],lab_pad[coord_shift]==lab_pad[bd_pad]))
    
    
    if get_bd:
        is_bd = np.any([np.logical_and(a1,a2) for a1,a2 in zip(angles1,angles2)],axis=0)
        bd_pad = np.zeros_like(mask_pad)
        bd_pad[coord] = is_bd
        return bd_pad
    else:
        step_ok = np.stack(step_ok)
        ind_shift = np.array(ind_shift)
        cross = np.stack([c[bd_pad[coord]] for c in cross])
        dot = np.stack([d[bd_pad[coord]] for d in dot])    
        return step_ok, ind_shift, cross, dot

@njit('(int64[:,:], int32[:], int32[:], int64[:], int64[:,:], float64[:,:], boolean[:,:])', nogil=True)
def parametrize(steps, labs, unique_L, inds, ind_shift, values, step_ok):
    sign = np.sum(np.abs(steps),axis=1)
    cardinal_mask = sign>1 # limit to cardinal steps fro traversing 
    contours = []
    for l in unique_L:
        indices = np.argwhere(labs==l).flatten() # which spots withing the inds list etc. are the boundary we want

        # just loop, manually calculate the best step, and proceed
        index = indices[0]
        closed = 0
        contour = []
        n_iter = 0
    
        while not closed and n_iter<len(indices)+1:
            contour.append(inds[index])

            # first step: find list of local points
            neighbor_inds = ind_shift[:,index]
            step_ok_here = step_ok[:,index]
            seen = np.array([i in contour for i in neighbor_inds])
            step_mask = (seen+cardinal_mask+~step_ok_here)>0 # save a smidge of time this way vs logical_or 
            
            vals = values[:,index]
            vals[step_mask] = np.inf # avoid these with min 
            
            if np.sum(step_mask)<len(step_mask): # 1.1 ms faster than np.any np.any(~step_mask)
                select = np.argmin(vals)
                neighbor_idx = neighbor_inds[select]
                w = np.argwhere(inds[indices]==neighbor_idx)[0][0] # find within limited list
                index = indices[w]
                n_iter += 1
            else:
                closed = True
                contours.append(contour)
    
    return contours  

from skimage.segmentation import expand_labels 
def boundary_to_masks(boundaries, binary_mask, min_size=9, dist=np.sqrt(2),connectivity=1):
    
    masks0 = remove_small_objects(measure.label((1-boundaries)*binary_mask,connectivity=connectivity),min_size=min_size)
    # bounds = find_boundaries(masks0,mode='outer')
    
    masks =  expand_labels(masks0,dist)
    # bounds = masks - masks0
    inner_bounds = (masks - masks0) > 0
    outer_bounds = find_boundaries(masks,mode='inner',connectivity=masks.ndim) #ensure that the mask interfaces are d-1-connected 
    bounds = np.logical_or(inner_bounds,outer_bounds) #restore the inner boundaries 
    return masks, bounds, masks0


import math, cv2
def get_midline(cell,img_stack,reference_point,debug=False):
    # plt.figure(figsize=(1,1))
    # plt.imshow(cell.image[0])
    # plt.axis('off')
    # plt.show()
    log = cell.image
    slc = cell.slice #TYX
    data = []
    segs = []
    T = range(slc[0].start,slc[0].stop)
    masks = np.zeros_like(img_stack,dtype=np.uint8)
    # print(masks.shape,cell.coords)
    masks[tuple(cell.coords.T)] = 1
    props = [measure.regionprops(masks[t])[0] for t in T]
    # angles = np.array([p.orientation for p in props])
    # angles = np.array([np.mod(np.pi-p.orientation,np.pi) for p in props])
    angles = np.array([np.mod(np.pi-p.orientation,2*np.pi) for p in props])
    

    if reference_point is None:
        print('starting with new ref point')
        # bd = find_boundaries(masks[0],mode='thick')
        mask = masks[0]
        y,x = np.nonzero(mask)
        contours = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print('contours',contours)
        x_,y_ = np.concatenate(contours[-2], axis=0).squeeze().T 
        ymed, xmed = props[0].centroid
        imin = np.argmax((x_-xmed)**2 + (y_-ymed)**2)
        reference_point = [y_[imin],x_[imin]]  # ok somehow using cv2 actually works for the furthest from center thing
        

        if debug:
            print('uop')
            # plt.figure(figsize=(2,2))
            # plt.imshow(img_stack[0])
            # plt.arrow(reference_point[1],reference_point[0],vectors[idx][1],vectors[idx][0])
            # plt.show()
            fig,ax = plt.subplots()
            ax.imshow(plot.outline_view(img_stack[0],masks[0]))
            y0, x0 = np.array(props[0].centroid)
            orientation = props[0].orientation
            x1 = x0 + math.cos(orientation) * 0.5 * props[0].axis_minor_length
            y1 = y0 - math.sin(orientation) * 0.5 * props[0].axis_minor_length
            x2 = x0 - math.sin(orientation) * 0.5 * props[0].axis_major_length
            y2 = y0 - math.cos(orientation) * 0.5 * props[0].axis_major_length

            ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
            ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
            ax.plot(x0, y0, '.g', markersize=15)
            
            ax.plot(reference_point[1], reference_point[0], '.y', markersize=5)

            plt.show()
        
        if angles[0]<0:
            angles*=-1

    # angles = [np.mod(a+np.pi/2,np.pi)-np.pi/2 for a in angles]
    
    old_pole = [reference_point]
    theta = angles[0]
    # centers = []
    angle_diffs = []
    for i, t in enumerate(T):
        center = np.array(props[i].centroid)
        mask = masks[t]        
        contours = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x_,y_ = np.concatenate(contours[-2], axis=0).squeeze().T 
        ymed, xmed = old_pole[-1]
        # yc, xc = props[i].centroid
        # dist_to_bound = np.sqrt((x_-xmed)**2 + (y_-ymed)**2) 
        # imin = np.argmin((x_-xmed)**2 + (y_-ymed)**2 - (x_-xc)**2 - (y_-yc)**2)
        # imin = np.argmin(np.dot())
        
        # instead of finding the pole position based on nearest point to last pole, should do it based on the direction?
        center = np.array(props[i].centroid)
        vectors = np.array([np.array([x,y])-center for x,y in zip(x_,y_)])
        # mag = np.sum((vectors)**2,axis=0)**0.5
        # units = vectors/mag
        uvec = [np.sin(angles[i]),np.cos(angles[i])]
        dot = [np.dot(u,uvec) for u in vectors]
        imin = np.argmax(dot) # furthest and most aligned
        
        new_ref = [y_[imin],x_[imin]] 
        
        
        old_pole.append(new_ref)
        d = center-new_ref # vector from pole to center
        thetaT = np.arctan2(d[0],d[1])

        # angles[i] = np.arctan2(d[1],d[0])
        angle_diffs.append(angles[i]-thetaT)
        # if cell.label==4:
        #     print(angles[i]-np.arctan2(d[0],d[1]),angles[i]-np.arctan2(d[0],d[1])+np.pi)
        if debug:
            fig,ax = plt.subplots(figsize=(2,2))
            # ax.imshow(img_stack[t])
            ax.imshow(plot.outline_view(img_stack[t],masks[t]))
            
            ax.arrow(new_ref[1],new_ref[0],d[1],d[0])
            ax.plot(reference_point[1], reference_point[0], '.y', markersize=5)
            ax.plot(new_ref[1], new_ref[0], '.c', markersize=5)
            plt.show()

    teststack = []
    for angle, prop, t in zip(angles,props,T):
        # angle = angles[t]
        img = img_stack[t]
        mask = masks[t]
        
        output_shape = [np.max(img.shape)]*2
        # output_shape = None
        
        # center = np.array([np.mean(c) for c in np.nonzero(mask)])
        center = np.array(prop.centroid)
        seg_rot = utils.rotate(mask,-angle,order=0,output_shape=output_shape,center=center)       
        img_rot = utils.rotate(img,-angle,output_shape=output_shape,center=center) 

        
        # weighted by distance version
        dt = smooth_distance(seg_rot,device=torch.device('cpu'))
        dt[seg_rot==0] = np.nan
        num = dt*img_rot
        l = np.nanmean(num,axis=0)/np.nanmean(dt,axis=0)
        teststack.append(l)
        forward =  np.argwhere(~np.isnan(l))
        first = forward[0][0] if len(forward) else 0
        backward =  np.argwhere(~np.isnan(np.flip(l)))
        last = backward[0][0] if len(backward) else 0
        strip = l[first:-(last+1)]
        data.append(strip)
        segs.append([cell.label for i in range(len(strip))])
        # print('ypoypo',l.shape,num.shape,dt.shape,np.nanmean(num,axis=0).shape,np.nanmean(dt,axis=0).shape)
#         plt.figure()
#         # plt.imshow(np.hstack([rescale(img_rot),rescale(dt)]))
#         plt.imshow(l[np.newaxis])
#         plt.show()

    # plt.figure()
    # # plt.imshow(np.hstack([rescale(img_rot),rescale(dt)]))
    # plt.imshow(np.stack(teststack))
    # plt.show()
    
    # center here is the last loop, the centroid of the last mask in the stack 
    # angle diff at the start is relevant to aligning pants 
    return data, segs, center, angles[0] 

def build_pants(node,cells,labels,img_stack,depth=0,reference_point=None, debug=False):
    tab = ''.join(['\t']*node.depth)

    idx = np.where(labels==node.name)[0][0]
    
    data, segs, reference_point, angle = get_midline(cells[idx], img_stack, reference_point, debug=debug)

    print(tab+'cell {}, angle {}'.format(node.name,angle))
    
    if node.is_leaf:
        padding = [[] for d in range(depth)]
        data = padding + data # pad it with veritcal empties so that it can be concatenated horizontally
        segs = padding + segs
        
        # print(tab+'leaf stack',len(data))
        return data, segs, reference_point, angle
    else:
        child_data, child_segs, child_angs = [], [], []
        for child in node.children:
            cdata, csegs, crefp, cangl = build_pants(child,cells,labels,img_stack,depth=depth+len(data),
                                                     reference_point=reference_point, debug=debug)
            # print(tab+'child',cangl, child.name, node.name)
            # print(tab+'intermediate',len(cdata))
            child_data.append(cdata)
            child_segs.append(csegs)
            # d = crefp - reference_point
            # child_angs.append(np.arctan2(d[0],d[1])) # these angles still need to be compared to the parent,
            d = crefp-reference_point
            rel_ang = np.arctan2(d[0],d[1])
            # child_angs.append(cangl) # these angles still need to be compared to the parent,
            child_angs.append(rel_ang) # these angles still need to be compared to the parent,
            
            # print(tab+'\trelative angle {}, or this angle {}'.format(angle-cangl,rel_ang))
            
        # sort = np.flip(np.argsort((angle-child_angs)))
        sort =  np.flip(np.argsort(child_angs))
        
        print(tab+'yo',angle-child_angs)
        child_data = [child_data[i] for i in sort]
        child_segs = [child_segs[i] for i in sort]
        print([len(c) for c in child_data])
        l = min([len(c) for c in child_data])
        child_stack = [np.hstack([c[i] for c in child_data]) for i in range(l)]
        child_masks = [np.hstack([c[i] for c in child_segs]) for i in range(l)]
        
        # print(tab+'child stack len',len(child_stack))
        padding = len(child_stack)-(len(data)+depth)
        parent_stack = [[] for d in range(depth)] + data + [[] for p in range(padding)]
        parent_masks = [[] for d in range(depth)] + segs + [[] for p in range(padding)]
        
        # print(tab+'parent_stack',len(parent_stack))
        return [np.hstack([p,c]) for p,c in zip(parent_stack,child_stack)], [np.hstack([p,c]) for p,c in zip(parent_masks,child_masks)], reference_point, angle
    
    
