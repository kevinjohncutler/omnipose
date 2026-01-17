import numpy as np
from numba import njit, prange
import edt
from scipy.ndimage import affine_transform, binary_dilation, binary_opening, binary_closing, label, shift, uniform_filter # I need to test against skimage labeling
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries
# networkit is optional; lazily imported only where needed to avoid
# import-time side effects and deprecation noise in environments that
# don't exercise those code paths during tests.

# import torch.nn.functional as F

import fastremap
import os, tifffile
import time
import mgen #ND rotation matrix
from . import utils
from .profiling import pyinstrument_profile
# from ncolor.format_labels import delete_spurs
# from .plot import rgb_flow

from .gpu import empty_cache # for clearing memory after follow_flows
from .misc import meshgrid, vector_to_arrow
from .stacks import shifts_to_slice
# from torchvf.losses import ivp_loss
# from typing import Any, Dict, List, Set, Tuple, Union, Callable
from typing import List


# define the lists of unique omnipose models 
# Some were trained with 2 channel input (C2)
# some were trained with a boundary field (BD)

C2_BD_MODELS = ['bact_phase_omni',
                'bact_fluor_omni',
                'worm_omni',
                'worm_bact_omni',
                'worm_high_res_omni',                    
                'cyto2_omni']          

C2_MODELS = ['bact_phase_cp',
            'bact_fluor_cp',
            'plant_cp', # 2D model for do_3D
            'worm_cp']    

C1_BD_MODELS = ['plant_omni']

# This will be the affinity seg models 
C1_MODELS = ['bact_phase_affinity']

import torch
# mse = torch.nn.MSELoss()
from .gpu import torch_GPU, torch_CPU, ARM

# try:
#     from sklearn.cluster import DBSCAN
#     from sklearn.neighbors import NearestNeighbors
#     SKLEARN_ENABLED = True 
# except:
#     SKLEARN_ENABLED = False

# from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
SKLEARN_ENABLED = True 

from dbscan import DBSCAN as new_DBSCAN # much faster 
import gc

try:
    from hdbscan import HDBSCAN
    HDBSCAN_ENABLED = True
except ModuleNotFoundError:
    HDBSCAN_ENABLED = False

try:
    from opensimplex import OpenSimplex
    OPEN_SIMPLEX_ENABLED = True
except ModuleNotFoundError:
    OpenSimplex = None
    OPEN_SIMPLEX_ENABLED = False

import sys
from .logger import setup_logger
omnipose_logger = setup_logger('core')

# omnipose_logger.setLevel(logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler())

# We moved a bunch of dupicated code over here from cellpose_omni to revert back to the original bahavior. This flag is used
# within Cellpose only, but since I want to merge the shared code back together someday, I'll keep it around here. 
# Several '#'s denote locations where code needs to be changed if a remerge ever happens 
OMNI_INSTALLED = True

from tqdm import trange 
import ncolor, scipy
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


# ## Section I: core utilities

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
    # isarray = type(dists) == np.ndarray
    # module = np if isarray else torch
    module = utils.get_module(dists)
    c = module.ceil(module.max(dists)*1.16)+1
    return c.astype(int) if module==np else c.int()
    
    # deprecated? only called during training flow generation it seems, not inference 

    # m = module.max(dists)
    # c = module.ceil(m*1.16)
    # # c = c.item()
    # i = c.to(torch.int32) + 1
    # return i


# minor modification to generalize to nD 
def dist_to_diam(dt_pos,n): 
    """
    Convert positive distance field values to a mean diameter. 
    
    Parameters
    --------------
    dt_pos: 1D array, float
        array of positive distance field values
    n: int
        dimension of volume. dt_pos is always 1D because only the positive values
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

def diameters(masks, dt=None, dist_threshold=0, pill=False, return_length=False):
    
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
          
    # omnipose_logger.info(dt_pos.shape)
    A = np.count_nonzero(dt_pos)
    D = np.sum(dt_pos)
    
    if np.any(dt_pos):
        if not pill:
            diam = dist_to_diam(np.abs(dt_pos),n=masks.ndim)
            if return_length:
                return diam, A/diam
        else:
            return pill_decomposition(A,D)
    else:
        diam = 0
        
    return diam
    
def pill_decomposition(A,D):
    R = np.sqrt((np.sqrt(A**2 + 24*np.pi*D) - A) / (2*np.pi))
    L = (3*D - np.pi*(R**4)) / (R**3)
    return R, L
    

    
# ## Section II: ground-truth flow computation  

# It is possible that flows can be eliminated in place of the distance field. The current distance field may not be smooth 
# enough, or maybe the network really does require the flow field prediction to work well. But in 3D, it will be a huge
# advantage if the network could predict just the distance (and boudnary) classes and not 3 extra flow components. 
def labels_to_flows(labels, links=None, files=None, use_gpu=False, device=None, 
                    omni=True, redo_flows=False, dim=2):
    """ Convert labels (list of masks or flows) to flows for training model.

    if files is not None, flows are saved to files to be reused

    Parameters
    --------------
    labels: list of ND-arrays
        labels[k] can be 2D or 3D, if [3 x Ly x Lx] then it is assumed that flows were precomputed.
        Otherwise labels[k][0] or labels[k] (if 2D) is used to create flows.
    links: list of label links
        These lists of label pairs define which labels are "linked",
        i.e. should be treated as part of the same object. This is how
        Omnipose handles internal/self-contact boundaries during training. 
    files: list of strings
        list of file names for the base images that are appended with '_flows.tif' for saving. 
    use_gpu: bool
        flag to use GPU for speedup. Note that Omnipose fixes some bugs that caused the Cellpose GPU 
        implementation to have different behavior compared to the Cellpose CPU implementation. 
    device: torch device
        what compute hardware to use to run the code (GPU VS CPU)
    omni: bool
        flag to generate Omnipose flows instead of Cellpose flows
    redo_flows: bool
        flag to overwrite existing flows. This is necessary when changing over from Cellpose to Omnipose, 
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
    if links is None:
        links = [None]*nimg # just for entering below 
    no_flow = labels[0].ndim != 3+dim # (6,Lt,Ly,Lx) for 3D, masks + dist + boundary + flow components, then image dimensions 
    
    if no_flow or redo_flows:
            
        # compute flows; labels are fixed in masks_to_flows, so they need to be passed back
        labels, dist, bd, heat, veci = map(list,zip(*[masks_to_flows(labels[n], links=links[n], use_gpu=use_gpu, 
                                                                 device=device, omni=omni, dim=dim) 
                                                  for n in trange(nimg)])) 
        
        # concatenate labels, distance transform, vector flows, heat (boundary and mask are computed in augmentations)
        if omni and OMNI_INSTALLED:
            flows = [np.concatenate((labels[n][np.newaxis,:,:], 
                                     dist[n][np.newaxis,:,:], 
                                     veci[n], 
                                     heat[n][np.newaxis,:,:]), axis=0).astype(np.float32)
                        for n in range(nimg)] 
            # clean this up to swap heat and flows and simplify code? would have to rerun all flow generation 
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

# @torch.no_grad() # try to solve memory leak in mps
def masks_to_flows(masks, affinity_graph=None, dists=None, coords=None, links=None, use_gpu=True, device=None, 
                   omni=True, dim=2, smooth=False, normalize=False, n_iter=None, verbose=False):
    """Convert masks to flows. 
    
    First, we find the scalar field. In Omnipose, this is the distance field. In Cellpose, 
    this is diffusion from center pixel. 
    Center of masks where diffusion starts is defined to be the 
    closest pixel to the median of all pixels that is inside the 
    mask.
    
    The flow components are then found as the gradient of the scalar field. 

    Parameters
    -------------
    masks: int, ND array
        labeled masks, 0 = background, 1,2,...,N = mask labels   
    dists: ND array, float
        array of (nonnegative) distance field values
    affinity_graph: ND array, bool
        hypervoxel affinity array, alternative to providing overseg labels and links
        the most general way to compute flows, and can represent internal boundaries 
    links: list of label links
        list of tuples used for treating label pairs as the same  
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
    if links is not None and dists is not None:
        print('Your dists are probably wrong...')
        
    if coords is None:
        coords = np.nonzero(masks) 
    # Generalize method of computing affinity graph for flow
    # as well as boundary, even with self-contact. Self-contact
    # requires mutilabel masks and link files. 
    
    steps, inds, idx, fact, sign = utils.kernel_setup(dim)
    
    case = [affinity_graph is None, 
             affinity_graph is not None and affinity_graph.shape[1] != len(coords[0])]
    if np.any(case):
        affinity_graph = masks_to_affinity(masks, coords, steps, inds, 
                                           idx, fact, sign, dim, links=links)
        if case[1]:
            print('Warning: passed affinity does not match mask coordinates. Recomputing.')

    boundaries = affinity_to_boundary(masks,affinity_graph,coords)
    
    if dists is None:
        # formatting reshuffles indices, so only do this
        # when no links are present 
        if (links is None or len(links)==0):# and (affinity_graph is None):
            masks = ncolor.format_labels(masks)
            dists = edt.edt(masks,parallel=-1)
        else:
            # this distance field is not completely accurate, but the point of it
            # is to estimate the number of iterations needed only, so close enough
            # better this than have self-contact boundaries mess up the distance field
            # and therefore completely overestimate the number of iterations required 
            # (Need to test to see if checking for convergence is faster...)
            dists = edt.edt(masks-boundaries,parallel=-1)+(masks>0)

    if device is None:
        if use_gpu:
            device = torch_GPU
        else:
            device = torch_CPU
    
    # masks_to_flows_device/cpu deprecated. Running using torch on CPU is still 2x faster
    # than the dedicated, jitted CPU code thanks to it being parallelized I think.
    
    if masks.ndim==3 and dim==2:
        # this branch preserves original 3D approach 
        print('Sorry, this branch has not yet been updated - do not use omnipiose for this')
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows_torch(masks[z], dists[z], boundaries[z], 
                                        device=device, omni=omni)[0]
            mu[[1,2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows_torch(masks[:,y], dists[:,y], boundaries[:,y], 
                                        device=device, omni=omni)[0]
            mu[[0,2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows_torch(masks[:,:,x], dists[:,:,x], boundaries[:,:,x], #<<< will want to fix this 
                                        device=device, omni=omni)[0]
            mu[[0,1], :, :, x] += mu0
        return masks, dists, None, mu #consistency with below
    
    else:
        T, mu = masks_to_flows_torch(masks, affinity_graph, coords, dists, device=device,
                                     omni=omni, smooth=smooth, normalize=normalize, n_iter=n_iter, 
                                     verbose=verbose)
        return masks, dists, boundaries, T, mu


# @torch.no_grad() # try to solve memory leak in mps
def masks_to_flows_batch(batch, links=[None], device=torch.device('cpu'), 
                         omni=True, dim=2, smooth=False, normalize=False, 
                         affinity_field=False, initialize=False, n_iter=None, 
                         verbose=False):
    """
    Batch process flows. This includes padding with relection to not have weird cutoff flows.
    
    Parameters
    -------------
    mask_batch: list, NDarray
        list of masks all of shape tyx
        
    Returns
    -------------
    concatenated labels, links, etc. and slices to extract them 
    """   
    
    # add an if statement to catch the case where all labels are empty 
    
    nsample = len(batch)
    final_flat,clinks,indices,final_shape,dL = concatenate_labels(batch,
                                                                  links=links,
                                                                  nsample=nsample)
    clabels = final_flat.reshape(final_shape)
    ccoords = np.unravel_index(indices,final_shape)
    # clabels,clinks,ccoords,dL = concatenate_labels(batch,links,nsample=nsample)
    
    # calculate affinity graph for the entire concatenated stack
    steps, inds, idx, fact, sign = utils.kernel_setup(dim)
    shape = batch[0].shape
    # edges = [np.concatenate([[i*dL,i*dL-1] for i in range(0,nsample+1)])]+[np.array([-1,0,s-1,s]) for s in shape[1:]]
    edges = [np.concatenate([[i*dL-1,i*dL] for i in range(0,nsample+1)])]+[np.array([-1,s]) for s in shape[1:]]
    
    # print('s',clabels.shape,[c.max() for c in ccoords])

    affinity_graph = masks_to_affinity(clabels, ccoords, steps, inds, idx, fact, sign, dim, 
                                       links=clinks, edges=edges)#, dists=cdists)
    
    # find boundary, flows 
    boundaries = affinity_to_boundary(clabels,affinity_graph,ccoords)
    
    # if I am do carry through the warped distance fields, I should probably use them here too to seed the iterations for faster convergence... have not doen that yet
    T, mu = masks_to_flows_torch(clabels, affinity_graph, ccoords,
                                 device=device, omni=omni, smooth=smooth,
                                 normalize=normalize, initialize=initialize, 
                                 affinity_field=affinity_field, n_iter=n_iter, 
                                 edges=edges, verbose=verbose)

    slices = [tuple([slice(i*dL,(i+1)*dL)]+[slice(None,None)]*(dim-1)) for i in range(nsample)]
    return torch.tensor(clabels.astype(int),device=device), torch.tensor(boundaries,device=device), T, mu, slices, clinks, ccoords, affinity_graph

# from numba import jit
# def concatenate_labels(masks,links,nsample):
# @njit #due to unravel_index
def concatenate_labels(masks: np.ndarray, links: list, nsample: int):
    # concatenate and increment both the masks and links 
    masks = masks.copy().astype(np.int64) # casting to int64 sped things up 10x???
    dtype = masks[0].dtype
    shape = masks[0].shape
    dL = shape[0]
    dim = len(shape)
    
    clinks = set()
    # clinks = []
    final_shape = (shape[0]*nsample,)+shape[1:]
    stride = np.prod(shape)
    length = np.prod(final_shape)
    # stride = 1
    # for s in shape:
    #     stride *=s
    # length = 1
    # for s in final_shape:
    #     length *= s
        
    # Preallocate flattened final array
    final_flat = np.empty(length, dtype=dtype)
    npix = np.array([np.count_nonzero(m>0) for m in masks],dtype)
    tpix = np.cumsum(np.hstack((0,npix)))
    # tpix = np.array([0]*(len(masks)+1),dtype)
    # for i,n in enumerate(npix):
    #     tpix[i+1:] += n
        
    indices = np.empty((tpix[-1],), dtype=np.int64)
    label_shift = 0 # shift labels of each tile outside the range of the last 
    for i,(masks,lnks) in enumerate(zip(masks,links)):
        mask_temp = np.ravel(masks)
        sel = np.nonzero(mask_temp)
        mask_temp[sel] = mask_temp[sel]+label_shift
        final_flat[(i*stride): (i+1)*stride] = mask_temp
        indices[tpix[i]:tpix[i]+npix[i]] = sel[0] + (i*stride)
        if lnks is not None:
            if len(lnks):
                for l in lnks:
                    clinks.add((l[0]+label_shift,l[1]+label_shift))
        label_shift += mask_temp.max()+1

    return final_flat,clinks,indices,final_shape,dL


# LABELS ARE NOW (masks,mask) for semantic seg with additional (bd,dist,weight,flows) for instance seg
# semantic seg label transformations taken care of above, those are simple enough. Others
# must be computed after mask transformations are made. Note that some of the labels are NOT used in training. Masks
# are never used, and boundary field is conditionally used. 
def batch_labels(masks,bd,T,mu,tyx,dim,nclasses,device,dist_bg=5):
    nimg = len(masks)
   
    nt = 2 # instance seg (labels), semantic seg (cellprob)
    if nclasses>1:
        nt += 3+dim # add boundary, distance, weight, flow components
    
    # preallocate 
    lbl = torch.zeros((nimg,nt,)+tyx, dtype=torch.float, device=device)
    
    lbl[:,0] = masks # probably do not need to store this here, but will keep it for now 
    lbl[:,1] = lbl[:,0]>0 # used to interpolate the mask, now thinking it is better to stay perfectly consistent 
    
    if nt>2:
        lbl[:,2] = bd # posisiton 2 store boundary, now returned as part of linked flow computation  
        lbl[:,3] = T # position 3 stores the smooth distance field 
        # lbl[:,3] = torch.log(lbl[:,3]+5) # try to reduce impact of large values 
        lbl[:,3][lbl[:,3]<=0] = -dist_bg # balance with boundary logits 
        
        lbl[:,-dim:] = mu*5.0 # *5 puts this in the same range as boundary logits
        lbl[:,4] = (1+lbl[:,1])/2 # position 4 stores the weighting image for weighted MSE 
        # lbl[:,4] = (1.+lbl[:,1]+lbl[:,2])/3. # position 4 stores the weighting image for weighted MSE 
        # uniform weight across cell appears to be best 
    return lbl

#Now fully converted to work for ND.
# @torch.no_grad() # try to solve memory leak in mps
def masks_to_flows_torch(masks, affinity_graph, coords=None, dists=None, device=torch.device('cpu'), omni=True,
                         affinity_field=False, smooth=False, normalize=False, n_iter=None, weight=1,
                         return_flows=True, edges=None, initialize=False, verbose=False):
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
    n_iter: int
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
    if np.any(masks):
        # the padding here is different than the padding added in masks_to_flows(); 
        # for omni, I used to reflect across the edge like a barbarian to simulate the mask extending past the edge, then crop
        # now I just use the affinity graph and force connections to the boundary!

        centers = np.array([])  
        if not omni: #do original centroid projection algrorithm
            unique_labels = fastremap.unique(masks)[1:]
            # get mask centers
            centers = np.array(scipy.ndimage.center_of_mass(masks, 
                                                            labels=masks, 
                                                            index=unique_labels)).astype(int).T
                        
            # check mask center inside mask
            valid = masks[tuple(centers)] == unique_labels
            for i in np.nonzero(~valid)[0]:
                crds = np.array(np.nonzero(masks==unique_labels[i]))
                meds = np.median(crds,axis=0)
                imin = np.argmin(np.sum((crds-meds)**2,axis=0))
                centers[:,i]=crds[:,imin]
                
        # set number of iterations
        if n_iter is None:
            if omni and OMNI_INSTALLED:
                if dists is not None:
                    # omni version requires fewer iterations 
                    n_iter = get_niter(dists) ##### omnipose.core.get_niter
            else:
                slices = scipy.ndimage.find_objects(masks)
                ext = np.array([[s.stop - s.start + 1 for s in slices[i-1]] for i in unique_labels])
                n_iter = 2 * (ext.sum(axis=1)).max()
            

        out = _extend_centers_torch(masks, centers, affinity_graph, coords,
                                    n_iter=n_iter, device=device, omni=omni, smooth=smooth, 
                                    weight=weight, return_flows=return_flows, affinity_field=affinity_field,
                                    edges=edges, initialize=initialize, verbose=verbose)
        
        if return_flows:
            T, mu = out
            if normalize:
                mu = utils.normalize_field(mu,use_torch=True,cutoff=0 if not smooth else 0.15) ##### transforms.normalize_field(mu,omni) 
                if verbose:
                    print('normalizing field')
            return T, mu
        else:
            return out
    else:
        return torch.zeros(masks.shape), torch.zeros((masks.ndim,)+masks.shape)

    
def get_links(masks,labels,bd,connectivity=1):   
    # Helper function. Might be unecessary now with the boundary_to_affinity function, which should be better. 
    # No, I still use it for multilabel data. 
    d = labels.ndim
    coords = np.nonzero(labels)

    steps, inds, idx, fact, sign = utils.kernel_setup(d)
    neighbors = utils.get_neighbors(coords,steps,d,labels.shape)

    # determine which pixels are neighbors. Pixels that are within reach (from step list) and the same label
    # are considered neighbors. However, boundaries should not consider other boundaries neighbors. 
    # this means that the central pixel is not a boundary at the same time as the other. 

    neighbor_masks = masks[tuple(neighbors)] #extract list of label values, here mmasks are the original, non-oversegged
    neighbor_bd = bd[tuple(neighbors)] #extract list of boundary values, here the original ones 
    isneighbor = np.logical_and(neighbor_masks == neighbor_masks[idx], # must have the same label 
                                np.logical_or.reduce((
                                    # neighbor_bd != neighbor_bd[idx], # neighbor not the same as central 
                                    np.logical_and(neighbor_bd==0,neighbor_bd[idx]==0), # or the neighbor is not a boundary
                                    np.logical_and(neighbor_bd==1,neighbor_bd[idx]==0), #
                                    np.logical_and(neighbor_bd==0,neighbor_bd[idx]==1), #                                    
                                ))
                               )

    piece_masks = labels[tuple(neighbors)] #extract list of label values from overseg 
    target = np.stack([piece_masks[idx]]*9)
    
    if connectivity==2:
        links = set([(a,b) for a,b in zip(target[isneighbor],piece_masks[isneighbor])]) #2-connected by default 
    else:
        #1-connected helps to avoid links I don't want
        sub_inds = np.concatenate(inds[:2])
        links = set([(a,b) for a,b in zip(target[sub_inds][isneighbor[sub_inds]],piece_masks[sub_inds][isneighbor[sub_inds]])]) 

    return links


# import networkx as nx
# def links_to_mask(masks,links):
#     """
#     Convert linked masks to stitched masks. 
#     """
#     G = nx.from_edgelist(links)
#     l = list(nx.connected_components(G))
#     # after that we create the map dict, for get the unique id for each nodes
#     mapdict={z:x for x, y in enumerate(l) for z in y }
#     # increment the dict keys to not conflict with any existing labels
#     m = np.max(masks)+1
#     mapdict = {k:v+m for k,v in mapdict.items()}
#     # remap
#     return fastremap.remap(masks,mapdict,preserve_missing_labels=True, in_place=False)



# this needs to be updaed... now a private jitted function, with a public wrapper below
# @njit(parallel=True) # cache not supported with parallel? 
# @njit(cache=True) # 2x as slow as paralle, but wirks in multiprocessing for training 
# def _get_link_matrix(links_arr, piece_masks, inds, idx, is_link):
#     for k in prange(len(inds)):
#         i = inds[k]
#         for j in range(len(piece_masks[i])):
#             a = piece_masks[i][j]
#             b = piece_masks[idx][j]
#             # check each link tuple in the array
#             for l in range(links_arr.shape[0]):
#                 if (links_arr[l, 0] == a and links_arr[l, 1] == b) or (links_arr[l, 1] == a and links_arr[l, 0] == b):
#                     is_link[i, j] = True
#                     break
#     return is_link
    
@njit(cache=True, fastmath=True)
def _get_link_matrix(links_arr, piece_masks, inds, idx, is_link):
    """
    Mark (i,j) as linked if (a,b) or (b,a) is found in links_arr.

    links_arr : (L,2) int64
    piece_masks : (S,N) int64   (S = 3**dim neighbours, N = #foreground px)
    inds : 1-D int64 indices of the neighbour planes you care about
    idx : int   index of the centre plane (inds[0] in your code)
    is_link : bool array to be filled in-place  (same shape as piece_masks)
    """
    # ---------- build an O(1) lookup table ----------
    max_label = links_arr.max() + 1          # for packing into one int
    link_set = set()                         # numba typed set[int64]
    for r in range(links_arr.shape[0]):
        a = links_arr[r, 0]
        b = links_arr[r, 1]
        if a > b:             # store unordered pair (min,max)
            a, b = b, a
        link_set.add(a * max_label + b)

    # ---------- mark links ----------
    for k in prange(len(inds)):
        i = inds[k]
        for j in range(piece_masks.shape[1]):
            a = piece_masks[i, j]
            b = piece_masks[idx, j]
            if a == b:                    # skip identical labels fast
                continue
            if a > b:
                a, b = b, a
            if a * max_label + b in link_set:
                is_link[i, j] = True
    return is_link

def get_link_matrix(links, piece_masks, inds, idx, is_link):
    """
    Public wrapper: convert an iterable of (a,b) link tuples into a 2D array
    and call the jitted helper.
    """
    # If no links provided, nothing to mark
    if not links:
        return is_link
    # Build an (N,2) int64 array of link pairs
    links_arr = np.array(list(links), dtype=np.int64)
    return _get_link_matrix(links_arr, piece_masks, inds, idx, is_link)

# @njit() cannot compute fingerprint of empty set
def masks_to_affinity(masks, coords, steps, inds, idx, fact, sign, dim,
                      neighbors=None,
                      links=None, edges=None, dists=None, cutoff=np.sqrt(2), 
                      spatial=False):
    """
    Convert label matrix to affinity graph. Here the affinity graph is an NxM matrix,
    where N is the number of possible hypercube connections (3**dimension) and M is the
    number of foreground hypervoxels. Self-connections are set to 0. 
    
    idx is the central index of the kernel, inds[0]. 
    edges is a list of tuples (y1,y2,y3,...),(x1,x2,x3,...) etc. to which all adjacent pixels should be connected
    concatenated masks should be paddedby 1 to make sure that doesn't cause unextpected label merging 
    dist can be used instead for edge connectivity 
    """

    # only reason to pad with edgemode  is to leverage duplicating labels to connect to boundary
    # must pad with 1 to allow for simple neighbor indexing 
    # There is much larger prior padding to handle edge artifacts, but we could avoid this with more sophisticated edge handling
    # need two things to ask the question: 1. is_background 2. is_edge 
    # if we are looking at an edge, we ask if we are connected to any background in any direction
    # if so, we do not connect to an edge 
    # that would leave single pixels connected to an edge, so need to check its neighbors for its edge connections
    
    shape = masks.shape
    # dim x steps x npix array of pixel coordinates 
    if neighbors is None: 
        
        neighbors = utils.get_neighbors(coords,steps,dim,shape,edges)
        
    # print('masks_to_affinity',masks.shape,coords[0].shape,neighbors.shape)
    
    # define where edges are, may be in the middle of concatenated images 
    is_edge = np.logical_and.reduce([neighbors[d]==neighbors[d][idx] for d in range(dim)]) 
    
    # extract list of neighbor label values
    piece_masks = masks[tuple(neighbors)]
    
    # see where the neighbor matches central pixel
    is_self = piece_masks == piece_masks[idx]

    # Pixels are linked if they share the same label or are next to an edge...
    conditions = [is_self,
                  is_edge
                 ] 
    # print([c.shape for c in conditions],len(links))
    # ...or they are connected via an explicit list of labels to be linked. 
    if links is not None and len(links)>0:
        is_link = np.zeros(piece_masks.shape, dtype=np.bool_)
        is_link = get_link_matrix(links, piece_masks, np.concatenate(inds), idx, is_link)
        conditions.append(is_link)
        
    affinity_graph = np.logical_or.reduce(conditions) 
    affinity_graph[idx] = 0 # no self connections
    
    # We may not want all masks to be reflected across the edge. Thresholding by distance field
    # is a good way to make sure that cells are not doubled up along their boundary. 
    if dists is not None:
        print('debug: check this')
        affinity_graph[is_edge] = dists[tuple(neighbors)][idx][np.nonzero(is_edge)[-1]]>cutoff
    
    return affinity_graph

# @njit() error 
def affinity_to_boundary(masks,affinity_graph,coords, dim=None):
    """Convert affinity graph to boundary map.
    
    Internal hypervoxels are those that are fully connected to all their 3^D-1 neighbors, 
    where D is the dimension. Boundary hypervoxels are those that are connected to fewer 
    than this number and at least 1 other hypervoxel. Correct boundaries should have >=D connections,
    but the lower bound here is set to 1. 
    
    Parameters:
    -----------
    masks: ND array, int or binary 
        label matrix or binary foreground mask
    
    affinity_graph: ND array, bool
        hypervoxel affinity array, <3^D> by <number of foreground hypervoxels>
    
    coords: tuple or ND array
        coordinates of foreground hypervoxels, <dim>x<npix>
    
    Returns:
    --------
    
    boundary
    """
    if dim is None:
        dim = masks.ndim       
    csum = np.sum(affinity_graph,axis=0)
    boundary = np.logical_and(csum<(3**dim-1),csum>0) # check this latter condition
    
    # check if spatial or npix
    # if spatial, no need to convert to mask coordinates 
    if boundary.shape == masks.shape:
        return boundary
    else:
        bd_matrix = np.zeros(masks.shape,int)
        bd_matrix[tuple(coords)] = boundary 
        return bd_matrix
    
def spatial_affinity(affinity_graph, coords, shape):
    """
    Convert affinity graph in (S,N) format to (S,*DIMS) format. 
    """
    nsteps,npix = affinity_graph.shape
    affinity = np.zeros((nsteps,)+shape)
    affinity[(Ellipsis,)+tuple(coords)] = affinity_graph
    return affinity

def links_to_boundary(masks,links):
    """Deprecated. Use masks_to_affinity instead."""
    pad = 1
    d = masks.ndim
    shape = masks.shape
    masks_padded = np.pad(masks,pad)
    coords = np.nonzero(masks_padded)
    # binary_dilation
    # coords = np.nonzero(binary_dilation(masks_padded>0))
    
    steps, inds, idx, fact, sign= utils.kernel_setup(d)
    coords = np.nonzero(masks)#bug??
    # neighbors = np.array([np.add.outer(coords[i],steps[:,i]) for i in range(d)]).swapaxes(-1,-2)
    neighbors = utils.get_neighbors(coords,steps,d,shape)

    piece_masks = masks_padded[tuple(neighbors)] #extract list of label values, 
    is_link = np.zeros(piece_masks.shape, dtype=np.bool_)
    is_link = get_link_matrix(links, piece_masks, np.concatenate(inds), idx, is_link)
    
    border_mask = np.pad(np.zeros(masks.shape,dtype=bool),pad,constant_values=1)
    isborder = border_mask[tuple(neighbors)] #extract list of border values
    
    # this tells us if a pixel in one of the 9 steps (0,0 included) is different
    # if so, the central pixel should be considered boundary, but that is determined later 

        # this version of neighbors does not rely on boundaries, ad the links are used to make the boundaries 
    isneighbor = np.logical_or.reduce((piece_masks == piece_masks[idx], # must have the same label 
                                       is_link,# or is linked 
                                       isborder)) 
    isboundary = ~isneighbor #equivalent to and of nots 
    
    bd0 = np.zeros(masks_padded.shape,dtype=bool)
    masks0 = np.zeros_like(masks_padded)
    
    s_all = np.concatenate(inds[1:])
    flat_bd = np.any(isboundary[s_all],axis=0)
    bd0[coords] = flat_bd
    
    
    neighbor_bd = bd0[tuple(neighbors)]

    # flat_bd = np.sum(is_neighbor_bd,axis=0)==2
    # sel = np.concatenate(inds[1:])
    sel = inds[1]
    # flat_bd = (np.sum(is_neighbor_bd[sel],axis=0)>1)*flat_bd
    # bd0[np.nonzero(masks_padded)] = flat_bd
    
    # implement below... does not quite work for pixels next to some that we want to remove 
    crit1 = np.sum(isneighbor[inds[1]],axis=0)>=2 # at least 2 edges touching linked pixels, REMOVES SPURS
    crit2 = np.sum(isboundary[inds[2]],axis=0)>=1 # at least 1 vertex touching unlinked pixels     
    crit3 = np.sum(isneighbor[inds[1]],axis=0)==3 # edges need 
    crit12 = np.logical_and(crit1,crit2)
    flat_bd = np.logical_or(crit12,crit3)
    bd0[coords] = flat_bd
    
    # delete the removed spurs 
    masks0[coords] = piece_masks[idx]*crit1
    coords = np.nonzero(masks0)
    # neighbors = np.array([np.add.outer(coords[i],steps[:,i]) for i in range(d)]).swapaxes(-1,-2) #recompute 
    neighbors = np.array([[coords[k] + s[k] for s in steps] for k in range(d)])
    # masks0[coords] = piece_masks[idx]
    # isneighbor[s_all]*=crit1
    # isboundary[s_all]*=crit1
    # piece_masks[(Ellipsis,)+np.where(crit1)] = 0
    
    #have to recompute?
    piece_masks = masks0[tuple(neighbors)] #extract list of label values, 
    
    is_link = np.zeros(piece_masks.shape, dtype=np.bool_)
    is_link = get_link_matrix(links,piece_masks, np.concatenate(inds), idx, is_link)

    isborder = border_mask[tuple(neighbors)] #extract list of border values
    
    isneighbor = np.logical_or.reduce((piece_masks == piece_masks[idx], # must have the same label 
                                       is_link, # or is linked 
                                       isborder)) 
    isboundary = ~isneighbor
    
    if 0:
        # ok, try the boundary cleanup way; given nearly perfect boundaries, clean up islands 
        # not equivalent to using flat_bd...
        neighbor_bd = np.logical_or(bd0[tuple(neighbors)],isborder)
        sel = inds[1]
        c1 = np.sum(np.logical_and(neighbor_bd[sel],isneighbor[sel]),axis=0)>=2 # at least two neighbors are linked 
        # c1*=flat_bd # might need to fix this 
        sel = inds[2]
        c2 =  np.sum(~isneighbor[sel],axis=0)>=1 # one vertex unlinked 
        a = np.logical_and(c1,c2)
        sel = s_all
        outside = np.any(np.logical_and(piece_masks[sel]==0,~isborder[sel])*piece_masks[idx],axis=0)
        bd0[coords] = np.logical_or(a,outside)
        # bd0[coords] = a

    # final cleanup of the spurs
    sel = inds[1]
    neighbor_bd = np.logical_or(bd0[tuple(neighbors)],isborder)
    c1 = np.sum(neighbor_bd[sel],axis=0)>=2
    bd0[coords] = np.logical_and(c1,bd0[coords])

    isboundary = bd0[tuple(neighbors)]
    # need to keep the boundary info as links throughout?
    bd0[coords] = np.any(isboundary[inds[0]],axis=0)
        
    unpad = tuple([slice(pad,-pad)]*d)
    return bd0[unpad], masks0[unpad], isboundary, neighbors-pad

def mode_filter(masks):
    """
    super fast mode filter (compared to scipy, idk about PIL) to clean up interpolated labels
    """
    pad = 1
    masks = np.pad(masks,pad).astype(int)
    d = masks.ndim
    shape = masks.shape
    coords = np.nonzero(masks)
    steps, inds, idx, fact, sign = utils.kernel_setup(d)
    
    # subinds = np.concatenate(inds[0:2]) # only consider center+cardinal 
    subinds = np.concatenate(inds)
    substeps = steps[subinds]
    # neighbors = np.array([np.add.outer(coords[i],substeps[:,i]) for i in range(d)]).swapaxes(-1,-2)
    neighbors = utils.get_neighbors(coords,substeps,d,shape) # good place to speed things up 
    
    neighbor_masks = masks[tuple(neighbors)]
    
    mask_filt = np.zeros_like(masks)
    # mask_filt[coords] = scipy.stats.mode(neighbor_masks,axis=0,keepdims=1)[0] # wayyyyyy tooo slow, nearly 500ms 

    # 30ms and identical output to mode, 16 now when I restrict to cardinal points of course  
    # most_f = np.array([np.bincount(row).argmax() for row in neighbor_masks.T])  
    # mask_filt[coords] = most_f
    most_f = most_frequent(neighbor_masks)
    z = most_f==0 
    most_f[z] = masks[coords][z]
    mask_filt[coords] = most_f
    
    unpad = tuple([slice(pad,-pad)]*d) 
    return mask_filt[unpad]

@njit # thanks to numba, this is down from 30ms to under 2ms and can keep the full kernel 
def most_frequent(neighbor_masks):
    return np.array([np.bincount(row).argmax() for row in neighbor_masks.T])  

# @torch.no_grad() # try to solve memory leak in mps
def _extend_centers_torch(masks, centers, affinity_graph, coords=None, n_iter=200, 
                          device=torch.device('cpu'), omni=True, smooth=False, 
                          weight=1, return_flows=True, affinity_field=False, 
                          edges=None, initialize=False, verbose=False):
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
    boundaries: bool, 2D or 3D array
        binary field representing 1-connected boundary 
         
    """
    d = masks.ndim
    shape = masks.shape
    npix = affinity_graph.shape[-1]
    steps, inds, idx, fact, sign = utils.kernel_setup(d)

    if coords is None:
        coords = np.nonzero(masks>0) # >0 to handle -1 labels at edge; do I use that anymore? check...
    else:
        coords = tuple(coords)
    # we want to index the flatened pixel list T will of shape (npix,)
    neighbors = utils.get_neighbors(coords,steps,d,shape,edges) # shape (d,3**d,npix)   
    indexes, neigh_inds, ind_matrix = utils.get_neigh_inds(tuple(neighbors),coords,shape)

    central_inds = ind_matrix[tuple(neighbors[:,idx])]
    centroid_inds = ind_matrix[tuple(centers)] if len(centers) else np.zeros(0)

    if verbose:
        print('affinity_graph',affinity_graph.shape,affinity_graph.dtype)
        print('index shape',indexes.shape)
        print('neighbors shape',neighbors.shape)
        print('neigh_inds shape',neigh_inds.shape)
        print('central_inds shape',central_inds.shape)
        print('centroid_inds shape',centroid_inds.shape)

    # previous neighbor-finding code has been replaced with affinity_graph code 
    # this is always precomputed by this stage 

    dtype = torch.float
    # T = torch.zeros(npix, dtype=dtype, device=device)
    T =  torch.ones(npix, dtype=dtype, device=device)

    d = torch.tensor(d)
    idx = torch.tensor(idx)
    fact = torch.tensor(fact)
    steps = torch.tensor(steps,device=device)        
    inds = tuple([torch.tensor(i) for i in inds])
    omni = torch.tensor(omni)
    smooth = torch.tensor(smooth)
    verbose = torch.tensor(verbose)

    isneigh = torch.tensor(affinity_graph,device=device,dtype=torch.bool) # isneigh shape (3**d,npix)
    neigh_inds = torch.tensor(neigh_inds,device=device)
    central_inds = torch.tensor(central_inds,device=device,dtype=torch.long)
    centroid_inds = torch.tensor(centroid_inds,device=device,dtype=torch.long)

    if affinity_field:
        # experimenting with using the connectivity graph to define the scalar field precition class
        T = torch.tensor(affinity_graph,device=device,dtype=dtype).sum(axis=0)
    else:
        if initialize and d<=3:
            T = torch.tensor(edt.edt(masks)[coords],device=device) 

        if n_iter is None:
            n_iter = torch.tensor(50)
        else:
            n_iter = torch.tensor(n_iter)

        T = _iterate(T,neigh_inds,central_inds,centroid_inds,
                     idx,d,inds,fact,isneigh,n_iter,omni,smooth,verbose)

    ret = []
    
    if return_flows:
        # calculate gradient with contributions along cardinal, ordinal, etc. 
        # new implementation is 30x faster than an earlier version 
        n_axes = len(fact)-1
        s = [n_axes,d,isneigh.shape[-1]]
        mu_ = torch.zeros((d,)+shape,device=device,dtype=dtype)
        mu_[(Ellipsis,)+coords] = _gradient(T,d,steps,fact,inds,isneigh,neigh_inds,central_inds,s)
        if verbose:
            print('mu',mu_.shape)
        ret += [mu_] # .detach() adds a lot of time? 
    
    # put back into ND
    T_ = torch.zeros(shape,device=device,dtype=dtype)
    T_[coords] = T
    
    # put it first 
    ret = [T_]+ret
    
    return (*ret,)


@torch.jit.script # saves maybe 10%
def update_torch(a,f,fsq):
    # Turns out we can just avoid a ton of individual if/else by evaluating the update function
    # for every upper limit on the sorted pairs. I do this by pieces using cumsum. The radicand
    # being nonegative sets the upper limit on the sorted pairs, so we simply select the largest 
    # upper limit that works. I also put a couple of the indexing tensors outside of the loop. 
    """Update function for solving the Eikonal equation. """
    a,_ = torch.sort(a,dim=0) # sorting was the source of the small artifact bug 
    am = a*((a-a[-1])<f)
    sum_a = am.sum(dim=0)
    sum_a2 = (am**2).sum(dim=0)
    # return (1/d)*(sum_a+torch.sqrt(torch.clamp((sum_a**2)-d*(sum_a2-fsq),min=0)))
    # return (1/d)*(sum_a+torch.clamp((sum_a**2)-d*(sum_a2-fsq),min=0)**0.5)
    # return (1/d)*(am.sum(dim=0)+torch.clamp((am.sum(dim=0)**2)-d*((am**2).sum(dim=0)-fsq),min=0)**0.5)
    # return (1/d)*(sum_a+torch.sqrt(torch.clamp((sum_a**2)-d*(sum_a2-fsq),min=0)))
    
    d = a.shape[0] # d acutally needed to be the number of elements being compared, not dimension 
    return (1/d)*(sum_a+torch.sqrt(torch.clamp((sum_a**2)-d*(sum_a2-fsq),min=0)))
    
    

@torch.jit.script
def eikonal_update_torch(Tneigh: torch.Tensor,
                         r: torch.Tensor,
                         d: torch.Tensor,
                         index_list: List[torch.Tensor],
                         factors: torch.Tensor):
    """Update for iterative solution of the eikonal equation on GPU."""
    # preallocate array to multiply into to do the geometric mean
    # Tneigh always has shape 1 x nconnections x npix
    geometric = 1
    phi_total = torch.ones_like(Tneigh[0,:]) if geometric else torch.zeros_like(Tneigh[0,:])
    
    # loop over each index list + weight factor 
    n = len(factors) - 1
    w = 0.

    for inds,f,fsq in zip(index_list[1:],factors[1:],factors[1:]**2):    
    
        # find the minimum of each hypercube pair along each axis
        npair = len(inds)//2
        
        # mins = torch.stack([torch.fmin(Tneigh[inds[i],:],Tneigh[inds[-(i+1)],:]) for i in range(npair)])
        mins = torch.stack([torch.minimum(Tneigh[inds[i],:],Tneigh[inds[-(i+1)],:]) for i in range(npair)])
        
        # apply update rule using the array of mins, 
        update = update_torch(mins,f,fsq)
        
        # put into storage array
        if geometric:
            phi_total *= update
        else:
            phi_total += update
            
    phi_total = torch.pow(phi_total,1/n) if geometric else phi_total/n
    
    return phi_total


@torch.jit.script
def _iterate(T: torch.Tensor, # 1D tensor of scalar values at each pixel
             neigh_inds: torch.Tensor, 
             central_inds: torch.Tensor, 
             centroid_inds: torch.Tensor, 
             idx: torch.Tensor,
             d: torch.Tensor,
             inds: List[torch.Tensor],
             fact: torch.Tensor,
             isneigh: torch.Tensor,
             n_iter: torch.Tensor,
             omni: torch.Tensor,
             smooth: torch.Tensor, 
             verbose: torch.Tensor):
    
    T0 = T.clone()
    eps = 1e-3 if not smooth else 1e-8
    # eps = 1e-5
    
    # n_iter = 200
    if verbose:
        print('eps is ', eps, 'n_iter is', n_iter)
    
    # I wonder if it is possible to reduce the update grid after points converge 
    t = torch.tensor(0)
    not_converged = torch.tensor(True)
    error = torch.tensor(1)
    npix = isneigh.shape[-1]

    
    # r = torch.arange(0,npix)
    r = central_inds 
    
    while not_converged:
        if omni:# and OMNI_INSTALLED:
            Tneigh = T[neigh_inds]
            Tneigh *= isneigh #zeros out any elements that do not belong in convolution
            T = eikonal_update_torch(Tneigh,r,d,inds,fact) # now central_inds = 0,1,2,3,...
        else:
            T[centroid_inds] += 1

        # error = mse(T,T0)
        error = (T-T0).square().mean() #faster than mse function
        
        if omni:
            not_converged = torch.logical_and(error>eps, t<n_iter)
            # not_converged = torch.logical_and(torch.tensor(error>eps), torch.tensor(t<n_iter))
            # not_converged = torch.logical_and(error>eps, torch.tensor(t<n_iter))
            
        else:
            not_converged = t<n_iter
            
        # helps to do a bit of smoothing to start get the signal propagated
        if not omni or t<1 or smooth:  #  or not not_converged
            Tneigh = T[neigh_inds]
            Tneigh *= isneigh
            T = Tneigh.mean(dim=0) # mean along the <3**d>-element column does the box convolution 
            
        # update the old one 
        T0.copy_(T) # faster than T0 = T.clone() or  T0[:] = T

        t+=1

    if verbose:
        print('iter: ',t,'{:.10f}'.format(error))
    # There is still a fade out effect on long cells, not enough iterations to diffuse far enough I think 
    # The log operation does not help much to alleviate it, would need a smaller constant inside. 
    if not omni:
        T = torch.log(1.+ T)

    return T


@torch.jit.script 
def _gradient(T,d,steps,fact,
              inds: List[torch.Tensor],
              isneigh,
              neigh_inds: torch.Tensor,
              central_inds: torch.Tensor, 
              s: List[int]
             ):

    finite_differences = torch.zeros(s,device=T.device,dtype=T.dtype)
    cvals = T[central_inds]
    for ax,(ind,f) in enumerate(zip(inds[1:],fact[1:])):

        vals = T[neigh_inds[ind]] # maybe go back to passing neigh_vals
        vals[~isneigh[ind]] = 0 # T[]*mask prevent bleedover / boundary issues, big problem in stock Cellpose that got reverted!

        mid = len(ind)//2
        r = torch.arange(mid)
        # unit vectors 
        vecs = steps[ind].float()
        uvecs = (vecs[-(r+1)] - vecs[r]).T #/(2*f) #move normalization to end for speed

        # calculate differences along each axis with directional pairs  
        diff = (vals[-(r+1)]-vals[r]) # /(2*f)

        # dot products, project differences onto cardinal coorinate system 
        finite_differences[ax] = torch.matmul(uvecs,diff) / (2*f)**2
        # finite_differences[ax] = torch.einsum('ij,jk->ik', uvecs, diff)  / (2*f)**2

        
    mu = torch.mean(finite_differences,dim=0) 

    # do some averaging with neighbors, but weighted by dot product so that magnitude does not fall off
    weight = torch.sum(mu[:,neigh_inds]*(mu[:,central_inds].unsqueeze(1)),dim=0).abs() # A.B
    weight[~isneigh] = 0
    wsum = weight.sum(dim=0)
    return torch.where(wsum!=0,
                       (mu[:,neigh_inds]*weight).sum(dim=1) / wsum,
                       torch.zeros_like(wsum))


# ## Section II: mask recontruction

def compute_masks(dP, dist, affinity_graph=None, bd=None, p=None, coords=None, iscell=None, niter=None, rescale=1.0, resize=None, 
                  mask_threshold=0.0, diam_threshold=12.,flow_threshold=0.4, 
                  interp=True, cluster=False, boundary_seg=False, affinity_seg=False, do_3D=False, 
                  min_size=None, max_size=None, hole_size=None, omni=True, 
                  calc_trace=False, verbose=False, use_gpu=False, device=None, nclasses=2, 
                  dim=2, eps=None, hdbscan=False, flow_factor=6, debug=False, override=False, suppress=None, despur=False):
    """
    Compute masks using dynamics from dP, dist, and boundary outputs.
    Called in cellpose.models(). 
    
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
    coords: int32, 2D array
        non-zero pixels to run dynamics on [npixels x D]
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
        number of output classes of the network (Omnipose=3,Cellpose=2)
    dim: int
        dimensionality of data / model output
    eps: float
        internal epsilon parameter for (H)DBSCAN
    hdbscan: 
        use better, but much SLOWER, hdbscan clustering algorithm (experimental)
    flow_factor:
        multiple to increase flow magnitude (used in 3D only, experimental)
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
    bd: float32, ND array
        boundary map
    augmented_affinity: float32, ND array
        concatenated coordinates and affinity graph, hence (d+1,3**d,npix)
    """

    # do everything in padded arrays for boundary/affinity functions - not needed anymore?
    pad = 0 ##
    # pad = 1 ##
    # print('pad',pad)
    if do_3D:
        dim = 3 
    pad_seq = [(0,)*2]+[(pad,)*2]*dim
    unpad = tuple([slice(pad,-pad) if pad else slice(None,None)]*dim) # works in case pad is zero

    if hole_size is None:
        hole_size = 3**(dim//2) # just a guess

    labels = None
    
    if verbose:
        startTime0 = time.time()
        omnipose_logger.info(f'mask_threshold is {mask_threshold}')
        if omni and (not SKIMAGE_ENABLED):
             omnipose_logger.warning('Omni enabled but skimage not enabled')
    
    # inds very useful for debugging and figures; allows us to easily specify specific indices for Euler integration
    if iscell is None:
        if coords is not None:
            iscell = np.zeros_like(dist,dtype=np.int32)
            iscell[tuple(coords)] = 1
        else:
            if (omni and SKIMAGE_ENABLED) or override:
                if verbose:
                    omnipose_logger.info('Using hysteresis threshold.')
                iscell = filters.apply_hysteresis_threshold(dist, mask_threshold-1, mask_threshold) # good for thin features

            else:
                iscell = dist > mask_threshold # analog to original iscell=(cellprob>cellprob_threshold)
    
    
    # if nclasses>1, we can do instance segmentation. 
    if np.any(iscell) and nclasses>1: 

        iscell_pad = np.pad(iscell,pad) # I should get rid of all padding commands, padding is zero now 
        coords = np.array(np.nonzero(iscell_pad)).astype(np.int32)       
        shape = iscell_pad.shape
        
        
        # for boundary later, also for affinity_seg option
        # steps = utils.get_steps(dim) # perhaps should factor this out of the function 

        if suppress is None:
            suppress = omni and not affinity_seg # Euler suppression ON with omni unless affinity seg 

        #preprocess flows
        if omni and OMNI_INSTALLED:

            # Euler suppression may be bad in 3D in general, fyi 
            if suppress:# and not affinity_seg:
                # dP_ = div_rescale(dP,iscell) / rescale ##### omnipose.core.div_rescale
                # print('testing something new')
                # dP_ = utils.normalize_field(dP)
                # dP_ *= (1-utils.rescale(dist))
                
                # this is the winner I think 
                dP_ = div_rescale(dP,iscell) / rescale ##### omnipose.core.div_rescale
                # dP_ /= np.clip(dist,1,np.inf) # this is a problem in some places, 06/13/2023
            else:
                dP_ = dP.copy()/5.
            

            # else:
            #     dP_ = utils.normalize_field(dP)
            # dP_ = bd_rescale(dP,mask, 4*bd) / rescale ##### omnipose.core.div_rescale
                
            # dP_ = dP.copy()
            if dim>2 and suppress:
                dP_ *= flow_factor
                print('dP_ times {} for >2d, still experimenting'.format(flow_factor))

        else:
            dP_ = dP * iscell / 5.
        
        dP_pad = np.pad(dP_,pad_seq)
        dt_pad = np.pad(dist,pad)
        bd_pad = np.pad(bd,pad)
        bounds = None        
        
        # boundary seg can be stupid fast but it is a little broken
        if boundary_seg: # new tactic is to use flow to compute boundaries, including self-contact ones
            if verbose:
                omnipose_logger.info('doing new boundary seg')
            bd = get_boundary(np.pad(dP,pad_seq),iscell_pad)
            labels, bounds, _ = boundary_to_masks(bd,iscell_pad) 

            hole_size = 0 # turn off small hole filling, still do area threhsolding 
            
            # compatibility 
            p = np.zeros([2,1,1])
            tr = []
            
        else: # do the ol' Euler-integration + clustering 

            # the clustering algorithm requires far fewer iterations because it 
            # can handle subpixel separation to define blobs, whereas the thresholding method
            # requires blobs to be separated by more than 1 pixel 
            # new affinity_seg does not do Euler supression and benefits from moderate point clustering 
            if (cluster or affinity_seg or not suppress) and niter is None:
                # niter = int(diameters(iscell,dist))
                # dividing by two is sometimes necessary, but it seems like it might be generally more harm than good
                niter = int(diameters(iscell,dist)/(1+affinity_seg))

                # if verbose:
                #     omnipose_logger.info('niter is now {}'.format(niter))
                
            if p is None:
                p, coords, tr = follow_flows(dP_pad, dt_pad, coords, niter=niter, interp=interp,
                                            use_gpu=use_gpu, device=device, omni=omni, 
                                            suppress= suppress,
                                            calc_trace=calc_trace, verbose=verbose)
            else:
                tr = []
                if verbose:
                    omnipose_logger.info('p given')
                # print('a2',shape,p.shape,coords.shape,p.max(), p[:,~iscell_pad].shape)
                
                # set the points that are background to not move
                p[:,~iscell_pad] = np.stack(np.nonzero(~iscell_pad))
                    

            #calculate masks
            if (omni and OMNI_INSTALLED) or override:
                steps, inds, idx, fact, sign = utils.kernel_setup(dim)
                if affinity_seg:
                    hole_size = 0 # turn off small hole filling, still do area threhsolding 
                    if affinity_graph is None:       
                        if verbose:
                            omnipose_logger.info('computing affinity graph')
                        # assuming we have no passed in the affinity graph, we need to compute it  
                        # affinity_graph, neighbors, neigh_inds = _get_affinity(steps,
                        #                                                     iscell_pad,
                        #                                                     dP_pad,
                        #                                                     dt_pad,
                        #                                                     p,
                        #                                                     coords, 
                        #                                                     pad=pad)
                        
                        initial_points = np.stack(meshgrid(iscell_pad.shape))
                        final_points = p
                        supporting_inds = utils.get_supporting_inds(steps)

                        # print('p',final_points.shape, initial_points.shape)
                        # if we can keep the points and predictions on GPU, we could make this a lot faster...
                        # especially if we can optmize euler+affinity together
                        affinity_graph = _get_affinity_torch(initial_points, 
                                                            final_points, 
                                                            dP_pad, #<<<<<<<<<<< add support for other options here 
                                                            dt_pad, 
                                                            iscell_pad, 
                                                            steps,
                                                            fact,
                                                            inds,
                                                            supporting_inds,
                                                            niter,
                                                            device=device # would default to torch GPU otherwise?
                                                            )
                        affinity_graph = affinity_graph.squeeze().cpu().numpy()
                        affinity_graph = affinity_graph[(Ellipsis,)+tuple(coords)]
                    
                    
                    # despur really not needed anymore, handled by the affinity grpah torch version
                    # elif despur:
                        # if it is passed in, we need the neigh_inds to compute masks 
                        # (though eventually we will want this to also be in parallel on GPU...)
                    
                    neighbors = utils.get_neighbors(tuple(coords),steps,dim,shape, pad=pad) # shape (d,3**d,npix)
                    indexes, neigh_inds, ind_matrix = utils.get_neigh_inds(tuple(neighbors),tuple(coords),shape)
                    
                    despur = dim==2 and despur # only do despur in 2D 
                    if verbose and not despur:
                        omnipose_logger.info('despur disabled')
                    
                    if despur:
                        non_self = np.array(list(set(np.arange(len(steps)))-{inds[0][0]})) # I need these to be in order
                        cardinal = np.concatenate(inds[1:2])
                        ordinal = np.concatenate(inds[2:])
                        
                        affinity_graph = _despur(affinity_graph, 
                                                neigh_inds, 
                                                indexes, 
                                                steps, 
                                                non_self, 
                                                cardinal, 
                                                ordinal, 
                                                dim)
                        
                        # I need to make sure that the masks/coords also get updated... that's what affinity_to_masks does 
                        # altertnatiely, the affinity_to_boundary then boundary_to_masks does this 
                  
                    bounds = affinity_to_boundary(iscell_pad,affinity_graph,tuple(coords))
                        
                    if cluster:
                        labels = affinity_to_masks(affinity_graph,neigh_inds,iscell_pad,coords,verbose=verbose)
                        # move bounds here, out of get affinity 
                    else:
                        # maybe faster version that skips connected components using the affinity graph
                        #  and instead uses the boundary output to define masks (implict connected components)
                        if verbose:
                            omnipose_logger.info('doing affinity seg without cluster.')
                        labels, bounds, _ = boundary_to_masks(bounds,iscell_pad) 
                                        
                else:
                    labels, _ = get_masks(p, bd_pad, dt_pad, iscell_pad, coords, nclasses, cluster=cluster,
                                             diam_threshold=diam_threshold, verbose=verbose, 
                                             eps=eps, hdbscan=hdbscan) ##### omnipose.core.get_masks
                    affinity_graph = None # could replace with masks to affinity 
                    coords = np.nonzero(labels)                
            else:
                labels = get_masks_cp(p, iscell=iscell_pad,
                                      flows = dP_pad if flow_threshold>0 else None, 
                                      use_gpu=use_gpu) ### just get_masks

        # flow thresholding factored out of get_masks
        # still could be useful for boundaries! TODO: Need to put in the self-contact boundaries as input <<<<<<
        # also can now turn on for do_3D... 
        
        if not do_3D: 
            flows = np.pad(dP,pad_seq) # original flow
            shape0 = flows.shape[1:]
            if labels.max()>0 and flow_threshold is not None and flow_threshold > 0 and flows is not None:
                # print('aaa',np.count_nonzero(labels),np.array(coords).shape,affinity_graph.shape)
                labels = remove_bad_flow_masks(labels, flows, 
                                               coords=coords, 
                                               affinity_graph=affinity_graph, 
                                               threshold=flow_threshold,
                                               use_gpu=use_gpu, 
                                               device=device, 
                                               omni=omni)
                _,labels = np.unique(labels, return_inverse=True)
                labels = np.reshape(labels, shape0).astype(np.int32)
        
        
        # need to reconsider this for self-contact... ended up just disabling with hole size 0
        # print('dd',iscell_pad.shape,labels.shape)
        masks = fill_holes_and_remove_small_masks(labels, min_size=min_size, max_size=max_size, ##### utils.fill_holes_and_remove_small_masks
                                                 hole_size=hole_size, dim=dim)*iscell_pad 
        # masks = labels
        # Resize mask, semantic or instance 
        resize_pad = np.array([r+2*pad for r in resize]) if resize is not None else labels.shape
        if tuple(resize_pad)!=labels.shape:
            if verbose:
                omnipose_logger.info(f'resizing output with resize = {resize_pad}')
            # mask = resize_image(mask, resize[0], resize[1], interpolation=0).astype(np.int32) 
            ratio = np.array(resize_pad)/np.array(labels.shape)
            masks = zoom(masks, ratio, order=0).astype(np.int32) 
            iscell_pad = masks>0
            dt_pad = zoom(dt_pad, ratio, order=1)
            dP_pad = zoom(dP_pad, np.concatenate([[1],ratio]), order=1) # for boundary 
            
            # affinity_seg not compatible with rescaling after euler integration
            # would need to upscale predcitons first 
            if verbose and affinity_seg:
                omnipose_logger.info('affinity_seg not compatible with rescaling, disabling')
            affinity_seg = False
            
        if not affinity_seg or boundary_seg:
            bounds = find_boundaries(masks,mode='inner',connectivity=dim)

        # If using default omnipose/cellpose for getting masks, still try to get accurate boundaries 
        if bounds is None:
            if verbose:
                print('Default clustering on, finding boundaries via affinity.')
            print('TO-DO: replace with _get_affinity_torch')
            affinity_graph, neighbors, neigh_inds, bounds = _get_affinity(steps,masks,dP_pad,dt_pad,p,inds, pad=pad)

            # boundary finder gets rid of some edge pixels, remove these from the mask 
            gone = neigh_inds[3**dim//2,np.sum(affinity_graph,axis=0)==0]
            # coords = np.argwhere(masks)
            crd = coords.T 
            masks[tuple(crd[gone].T)] = 0 
            iscell_pad[tuple(crd[gone].T)] = 0 
        else:
            # ensure that the boundaries are consistent with mask cleanup
            # only small masks would be deleted here, no changes otherwise to boundaries 
            bounds *= masks>0

        fastremap.renumber(masks,in_place=True) #convenient to guarantee non-skipped labels

        # moving the cleanup to the end helps avoid some bugs arising from scaling...
        # maybe better would be to rescale the min_size and hole_size parameters to do the
        # cleanup at the prediction scale, or switch depending on which one is bigger... 
        
        masks_unpad = masks[unpad] if pad else masks 
        bounds_unpad = bounds[unpad] if pad else bounds
        
        if affinity_seg:

            # I also want to return the raw affinity graph
            # the problem there is that it is computed on the padded array
            # besides unpadding, I need to delete columns for missing pixels 

            # Idea here is that I index everything corresponding to the affinity graph first


            # then I figure out which of these columns correspond to pixels that are in the final masks
            # this works by looking at an array of indices the same size as the image, and any pixels not part
            # of the original affinity graph do not participate, i.e. hole filling does not work 
            coords_remaining = np.nonzero(masks)
            inds_remaining = ind_matrix[coords_remaining]
            affinity_graph_unpad = affinity_graph[:,inds_remaining]
            neighbors_unpad = neighbors[...,inds_remaining] - pad

            # I also want to package the affinity graph with the pixel coordinates 
            # then there is no ambiguity and can extract a binary mask
            # thus the augmented affinity graph would be (d+1,3**d,npix)

            augmented_affinity = np.vstack((neighbors_unpad,affinity_graph_unpad[np.newaxis]))


            # # newer version that takes care of mask cleanup as well
            # # NOTE: without padding, this subsample affinity may be very overkill
            # # all I need to do is truncate the affinity graph so that neighbors and affinity are deleted where cleanup occurred 
            # slc = tuple([slice(pad,shape[d]-pad) for d in range(dim)]) 
            # augmented_affinity = np.vstack((neighbors,affinity_graph[np.newaxis]))
            # augmented_affinity = utils.subsample_affinity(augmented_affinity,slc,masks)


            # this also applied to the traced pixels
            if calc_trace:
                # tr = tr[:,inds_remaining]-pad
                print('warning calc trace not cropped')
                
        else:
            augmented_affinity = []
        
        ret = [masks_unpad, p, tr, bounds_unpad, augmented_affinity]
        
    else: # nothing to compute, just make it compatible
        omnipose_logger.info('No cell pixels found.')
        ret = [iscell, np.zeros([2,1,1]), [], iscell, []]
    
    if debug:
        ret += [labels] # also return the version of labels are prior to filling holes etc. 


    if verbose:
        executionTime0 = (time.time() - startTime0)
        omnipose_logger.info('compute_masks() execution time: {:.3g} sec'.format(executionTime0))
        if labels is not None:
            omnipose_logger.info('\texecution time per pixel: {:.6g} sec/px'.format(executionTime0/np.prod(labels.shape)))
            omnipose_logger.info('\texecution time per cell pixel: {:.6g} sec/px'.format(np.nan if not np.count_nonzero(labels) else executionTime0/np.count_nonzero(labels)))
        else:
            omnipose_logger.info('\tno objects found')

    return (*ret,)


# Omnipose requires (a) a special suppressed Euler step and (b) a special mask reconstruction algorithm. 

# no reason to use njit here except for compatibility with jitted fuctions that call it 
# this way, the same factor is used everywhere (CPU with/without interp, GPU)
# @njit()
def step_factor(t):
    """ Euler integration suppression factor.
    
    Conveneient wrapper function allowed me to test out several supression factors. 
    
    Parameters
    -------------
    t: int
        time step
    """
    return (1+t)

def div_rescale(dP,mask,p=1):
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
    if p>0:
        # div = utils.normalize99(likewise(dP))
        div = utils.normalize99(divergence(dP))**p
        dP *= div
    return dP

from scipy.special import expit
def sigmoid(x):
    """The sigmoid function."""
    expit(x) # this is the same as 1 / (1 + np.exp(-x))
    # return 1 / (1 + np.exp(-x))

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
    if any(f.shape[1+i] < 2 for i in range(num_dims-1)):
        return np.zeros_like(f[0])
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])


def get_masks(p, bd, dist, mask, inds, nclasses=2,cluster=False,
              diam_threshold=12., eps=None, min_samples=5, hdbscan=False, verbose=False):
    """Omnipose mask recontruction algorithm.
    
    This function is called after dynamics are run. The final pixel coordinates are provided, 
    and cell labels are assigned to clusters found by labeling the pixel clusters after rounding
    the coordinates (snapping each pixel to the grid and labeling the resulting binary mask) or 
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
    if nclasses > 1:
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
        if verbose:
            startTime = time.time()
            alg = ['','H']
            omnipose_logger.info('Doing {}DBSCAN clustering with eps={}, min_samples={}'.format(alg[hdbscan],eps,min_samples))

        if hdbscan and not HDBSCAN_ENABLED:
            omnipose_logger.warning('HDBSCAN clustering requested but not installed. Defaulting to DBSCAN')
        
        if hdbscan and HDBSCAN_ENABLED:
            #sklearn dbscan and hdbscan are really slow compared to the dbscan package below, 
            # cosndier depricating this option
            clusterer = HDBSCAN(cluster_selection_epsilon=eps,
                                # allow_single_cluster=True,
                                min_samples=min_samples)
            
            clusterer.fit(newinds)
            labels = clusterer.labels_
            
            # can try benchmarking this, but again, much slower that the dbscan package
            # import hdbscan
            # clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples)
            # labels = clusterer.fit_predict(newinds)
        else:
            labels, _ = new_DBSCAN(newinds, eps=eps, min_samples=min_samples)
            


        # filter out small clusters
        # unique_labels = set(labels) - {-1,0}
        # for l in unique_labels:
        #     hits = labels==l
        #     if np.sum(hits)<9:
        #         labels[hits] = -1 # make outliers
        
        if verbose:
            executionTime = (time.time() - startTime)
            omnipose_logger.info('Execution time in seconds: ' + str(executionTime))
            omnipose_logger.info('{} unique labels found'.format(len(np.unique(labels))-1))

        #### snapping outliers to nearest cluster
        # there was a bug where small clusters counted as outliers, and were snapped to a very distant cluster
        # I will fix this by enfocing a limit on distance to the nearest cluster
        # min_samples could also be reduced... 
        snap = 1
        if snap:
            nearest_neighbors = NearestNeighbors(n_neighbors=5) # maybe should be 5 instead of 50 
            neighbors = nearest_neighbors.fit(newinds)
            o_inds = np.where(labels==-1)[0]
            if len(o_inds):
                outliers = [newinds[i] for i in o_inds]
                nearest_dists, nearest_indices = neighbors.kneighbors(outliers)
                
                # get the labels of the nearest neighbors
                nearest_labels = labels[nearest_indices]
                
                # find the first instance that is not in reference to other ouliers
                nearest_idx = [np.where(n!=-1)[0][0] if np.any(n!=-1) else 0 for n in nearest_labels]
                dist_thresh = eps
                l = [nl[i] if nd[i]<dist_thresh else -1 for i,nl,nd in zip(nearest_idx,nearest_labels,nearest_dists)]
                labels[o_inds] = l
                if verbose:
                    omnipose_logger.info(f'Outlier cleanup with dist threshold {dist_thresh:.2f}:')
                    distances = [nd[i] for i,nd in zip(nearest_idx,nearest_dists)]
                    omnipose_logger.info(f'\tmin and max distance to nearest cluster: {np.min(distances):.2f},{np.max(distances):.2f}')
                    omnipose_logger.info('\tSnapped {} of {} outliers to nearest cluster'.format(np.sum(np.array(l)!=-1),len(o_inds)))

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
# grid_sample will only work for up to 5D tensors (3D segmentation). Will have to address this shortcoming if we ever do 4D. (see my pull request to torchvf for this
# I got rid of the map_coordinates branch, I tested execution times and pytorch implemtation seems as fast or faster
# @torch.jit.script

# deleted steps interp in favor of just using steps_batch as a unified function in ND.
# can use nearest interpolation if needed. 

def steps_batch(p, dP, niter, omni=True, suppress=True, interp=True, 
                 calc_trace=False, calc_bd=False, verbose=False):
    """Euler integration of pixel locations p subject to flow dP for niter steps in N dimensions. 
    
    Parameters
    ----------------
    p: float32, tensor
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
    # mode = 'nearest' if (omni and not suppress) else 'bilinear'
    
    # we want to use bilinear interpolation if using Euler suppression 
    # Affinity reconstruction does not require Euler suppression
    # and we want to also be able to toggle this globally with interp arg
    # (omni and and not suppress) is false when affinity is on
    interp = interp and not suppress
    mode = 'bilinear' if interp else 'nearest'
    if verbose:
        omnipose_logger.info(f'interp is {interp}, interpolation mode is {mode}')
    
    d = dP.shape[1] # number of components = number of dimensions 
    shape = dP.shape[2:] # shape of component array is the shape of the ambient volume 
    inds = list(range(d))[::-1] # grid_sample requires a particular ordering

    # print('inds', inds,p.shape, p.min(), p.max()) 

    device = dP.device # get the device from dP tensor

    shape = np.array(shape)[inds]-1.  # dP is d.Ly.Lx, inds flips this to flipped X-1, Y-1, ...
    # print('SHAPE',shape)
    B,D,I = p.shape
    # print('p...',p.shape,inds,shape)
    # pt = p[:,inds].permute(0,2,1).unsqueeze(1).float()
    pt = p[:,inds].permute(0,2,1).view([B]+[1]*(D-1)+[I,D]).float()
    
    # print('pt_new',pt.shape)

    pt0 = pt.clone() # save first
    flow = dP[:,inds] # inds is just flipping the spatial component ordering from TYX to XYT
    
    # print('point, flow shape',pt.shape,flow.shape)

    for k in range(d): 
        pt[...,k] = 2*pt[...,k]/shape[k] - 1 
        flow[:,k] = 2*flow[:,k]/shape[k]
    
    if calc_trace:
        dims = [-1,niter]+[-1]*(pt.ndim-1)
        trace = torch.clone(pt).detach().unsqueeze(1).expand(*dims) # add time 

    if omni and OMNI_INSTALLED and suppress:
        dPt0 = torch.nn.functional.grid_sample(flow, pt, mode=mode, align_corners=align_corners)

    for t in range(niter):
        if calc_trace and t>0:
            trace[:,t].copy_(pt)
            
        # print('aa',flow.shape,pt.shape)
        dPt = torch.nn.functional.grid_sample(flow, pt, mode=mode,
                                              align_corners=align_corners)

        if omni and OMNI_INSTALLED and suppress:
                dPt = (dPt+dPt0) / 2. # average with previous flow 
                dPt0.copy_(dPt) # update old flow 
                dPt /= step_factor(t) # suppression factor 
                
        for k in range(d): #clamp the final pixel locations
            pt[...,k] = torch.clamp(pt[...,k] + dPt[:,k], -1., 1.)
        
    pt = (pt+1)*0.5
    for k in range(d): 
        pt[...,k] *= shape[k]

    if calc_trace:
        trace = (trace+1)*0.5
        for k in range(d): 
            trace[...,k] *= shape[k]

    if calc_trace:
        # tr =  trace[...,inds].permute(0,1,-1,2,3)
        tr =  trace[...,inds].transpose(-1,1).contiguous()
        
    else:
        tr = None
    # p =  pt[...,inds].permute(0,-1,1,2)
    p =  pt[...,inds].transpose(-1,1).contiguous()
    
    empty_cache()
    return p, tr

# now generalized and simplified. Will work for ND if dependencies are updated. 
def follow_flows(dP, dist, inds, niter=None, interp=True, use_gpu=True, 
                 device=None, omni=True, suppress=False, calc_trace=False, verbose=False):
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
    if verbose:
        omnipose_logger.info(f'niter: {niter}, interp: {interp}, suppress: {suppress}, calc_trace: {calc_trace}')
    
    if niter is None:
        niter = 200
    
    niter = np.uint32(niter)
    cell_px = (Ellipsis,)+tuple(inds)

   # got rid of the interp vs not interp branch in favor of just using nearest interpolation in the
   # interp code; make single batch compatible with batched integrator with unsqueezing 
    
    flow_pred = torch.tensor(dP,device=device).unsqueeze(0) 
    shape = flow_pred.shape
    B = shape[0] # this should be 1 in this branch, from unsqueezing
    dim = shape[1]
    dims = shape[-dim:] #spatial dims

    coords = [torch.arange(0, l, device=device) for l in dims]
    mesh = torch.meshgrid(coords, indexing = "ij")
    init_shape = [B, 1] + ([1] * len(dims))
    initial_points = torch.stack(mesh, dim = 0) # torchvf flips with mesh[::-1]
    initial_points = initial_points.repeat(init_shape).float()

    final_points = initial_points.clone()

    
    
    if inds.ndim < 2 or inds.shape[0] < dim:
        omnipose_logger.warning('WARNING: no mask pixels found')
        tr = None
    else:
        final_p, tr = steps_batch(initial_points[cell_px], 
                                flow_pred, 
                                niter,
                                omni=omni, # omni controls the momentum term I have 
                                suppress=suppress, # Euler suppression can be independent, i.e. with agginity_seg 
                                interp=interp,
                                calc_trace=calc_trace, 
                                verbose=verbose)
        
        final_points[cell_px] = final_p.squeeze()
    
    p = final_points.squeeze().cpu().numpy()
    if verbose:
        omnipose_logger.info('done follow_flows')
    return p, inds, tr

def remove_bad_flow_masks(masks, flows, coords=None, affinity_graph=None, threshold=0.4, use_gpu=False, device=None, omni=True):
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
    merrors, _ =  flow_error(masks, flows, coords, affinity_graph, use_gpu, device, omni) ##### metrics.flow_error
    badi = 1+(merrors>threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks

def flow_error(maski, dP_net, coords=None, affinity_graph=None, use_gpu=False, device=None, omni=True):
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
    dim = maski.ndim
    dP_masks = masks_to_flows(maski, dim=dim, coords=coords, affinity_graph=affinity_graph, 
                              use_gpu=use_gpu, device=device, omni=omni)[idx].cpu().numpy() ##### dynamics.masks_to_flows
    # difference between predicted flows vs mask flows
    flow_errors = np.zeros(maski.max())
    
    for i in range(dP_masks.shape[0]):
        flow_errors += mean((dP_masks[i] - dP_net[i]/5.)**2, maski, #the /5 is to compensate for the *5 we do for training
                            index=np.arange(1, maski.max()+1))
        
    return flow_errors, dP_masks


# ## Section III: training

# Omnipose has special training settings. Loss function and augmentation. 
# Spacetime segmentation: augmentations need to treat time differently 
# Need to assume a particular axis is the temporal axis; most convenient is tyx. 
def random_rotate_and_resize(X, Y=None, scale_range=1., gamma_range=[.75,2.5], tyx = (224,224), 
                             do_flip=True, rescale=None, inds=None, nchan=1, allow_blank_masks=False):
    
    """ augmentation by random rotation and resizing

        X and Y are lists or arrays of length nimg, with channels x Lt x Ly x Lx (channels optional, Lt only in 3D)

        Parameters
        ----------
        X: float, list of ND arrays
            list of image arrays of size [nchan x Lt x Ly x Lx] or [Lt x Ly x Lx]
        Y: float, list of ND arrays
            list of image labels of size [nlabels x Lt x Ly x Lx] or [Lt x Ly x Lx]. The 1st channel
            of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
            If Y.shape[0]==3, then the labels are assumed to be [distance, T flow, Y flow, X flow]. 
        links: list of label links
            lists of label pairs linking parts of multi-label object together
            this is how omnipose gets around boudary artifacts druing image warps 
        scale_range: float (optional, default 1.0)
            Range of resizing of images for augmentation. Images are resized by
            (1-scale_range/2) + scale_range * np.random.rand()
        gamma_range: float, list
           images are gamma-adjusted im**gamma for gamma in [low,high]
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
        
    lbl = np.zeros((nimg,)+tyx, np.float32)
    scale = np.zeros((nimg,dim), np.float32)
    
    # first two basis vectors in any dimension, used to define rotation 
    v1 = [0]*(dim-1)+[1]
    v2 = [0]*(dim-2)+[1,0]
    
    for n in range(nimg):
        img = X[n].copy()
        y = None if Y is None else Y[n]
        # use recursive function here to pass back single image that was cropped appropriately 
        # skimage.io.imsave('/home/kcutler/DataDrive/debug/img_orig.png',img[0])
        # skimage.io.imsave('/home/kcutler/DataDrive/debug/label_orig.tiff',y[n]) #so at this point the bad label is just fine 
        imgi[n], lbl[n], scale[n] = random_crop_warp(img, y, tyx, v1, v2, nchan,
                                                     1 if rescale is None else rescale[n], 
                                                     scale_range, gamma_range, do_flip, 
                                                     inds is None if inds is None else inds[n], allow_blank_masks=allow_blank_masks)
        
    return imgi, lbl, np.mean(scale) #for size training, must output scalar size (need to check this again)

# This function allows a more efficient implementation for recursively checking that the random crop includes cell pixels.
# Now it is rerun on a per-image basis if a crop fails to capture .1 percent cell pixels (minimum). 
# scale is just a placeholder, the point to to figure out what the true rescaling facor is
def random_crop_warp(img, Y, tyx, v1, v2, nchan, rescale, scale_range, gamma_range, do_flip, ind, 
                     do_labels=True, depth=0, augment=True, allow_blank_masks=False):
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
    gamma_range: float, list
       images are gamma-adjusted im**gamma for gamma in [low,high]
    do_flip: bool (optional, default True)
        whether or not to flip images horizontally
    ind: int
        image index (for debugging)
    dist_bg: float
        nonegative value X for assigning -X to where distance=0 (deprecated, now adapts to field values)
    depth: int
        how many time this function has been called on an image 
    augment: bool
        whether or not to perform all non-morphological augmentations on the image (gamma, noise, etc.)

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
        error_message = """Sparse or over-dense image detected. 
        Problematic index is: {}. 
        Image shape is: {}. 
        tyx is: {}. 
        rescale is {}""".format(ind,img.shape,tyx,rescale)
        omnipose_logger.critical(error_message)
        # skimage.io.imsave('/home/kcutler/DataDrive/debug/img'+str(depth)+'.png',img[0]) 
        raise ValueError(error_message)
    
    if depth>200:
        error_message = """Recusion depth exceeded. Check that your images contain cells and 
                           background within a typical crop. 
                           Failed index is: {}.""".format(ind)
        omnipose_logger.critical(error_message)
        raise ValueError(error_message)
        return
    
    # labels that will be passed to the loss function
    # 
    # lbl = np.zeros((nt,)+tyx, np.float32) if do_labels else 
    
    numpx = np.prod(tyx) # number of pixels 
    if Y is not None:
        labels = Y.copy()
        # We want the scale distibution to have a mean of 1
        # There may be a better way to skew the distribution to
        # interpolate the parameter space without skewing the mean 
        # ds = scale_range/2
        # scale = np.random.uniform(low=1-ds,high=1+ds,size=dim) #anisotropic scaling 
        # scale = np.random.uniform(low=1/scale_range,high=scale_range,size=dim) #anisotropic scaling 
        
        eps = 1e-8
        # scale = np.random.triangular(left=1/(scale_range+eps), mode=1, right=scale_range+eps, size=dim) # weight to 1
        
        # this version gives much mro eweight to downsampling than upsampling
        mean_target = 1.0
        a = 1.0 / (scale_range + eps)
        b = scale_range + eps

        alpha = 1 # controls how skewed the distribution is 
        m = (mean_target - a) / (b - a)
        beta = alpha * (1.0 - m) / m
        scale = a + (b - a) * np.random.beta(alpha, beta, size=dim)


        # I need to make sure the scaling does not apply to time dimension...
        if rescale is not None:
            scale *= 1. / rescale
    else:
        scale = 1 # compatibility just in case
        
    # image dimensions are always the last <dim> in the stack (again, convention here is different)
    s = img.shape[-dim:]

    # generate random warp and crop 
    theta = np.random.rand() * np.pi * 2
    
    # M = mgen.rotation_from_angle_and_plane(theta,v1,v2) #not generalizing correctly to 3D? had -theta before
    rot = mgen.rotation_from_angle_and_plane(-theta,v2,v1)
    # M = rot.dot(np.diag(scale)) # we only need inverse matrix for warp 
    M_inv = np.diag(1./scale).dot(rot.T) # inverse of AB is (B_inv)(A_inv), and rot is orthogonal so transpose is inverse 

    # could define v3 and do another rotation here and compose them for more complicated 3D augmentations,
    # but usually the xy axes are distinct from z due to resolution limits, let alone t 

    axes = range(dim)
    s = img.shape[-dim:]
    rt = (np.random.rand(dim,) - .5) #random translation -.5 to .5
    dxy = [rt[a]*(np.maximum(0,s[a]-tyx[a])) for a in axes]
    
    # # replace this random translation with one biased toward cell density
    # wrap this in an if any foreground if I try it again 
    # foreground = labels>0
    
    # # Compute the projections and smooth them
    # # projections = [np.sum(foreground, axis=a) for a in axes]
    # # Compute the projections and smooth them
    # projections = [np.sum(foreground, axis=tuple(a for a in axes if a != ax)) for ax in axes]
    # # print('pp',projections[0].shape)
    # smoothed_projections = [uniform_filter(p, size=3) for p in projections]
    # # smoothed_projections = projections
    
    # # Normalize the smoothed projections to get probabilities
    # normalized_projections = [p / np.sum(p) for p in smoothed_projections]


    # # Replace the random translation with a weighted random choice based on these probabilities
    # rt = [np.random.choice(np.arange(len(p)), p=p)/len(p) - 0.5 for p in normalized_projections]
    # dxy = [rt[a]*(np.maximum(0,s[a]-tyx[a])) for a in axes]
    
    # print(dxy) 

    c_in = 0.5 * np.array(s) + dxy
    c_out = 0.5 * np.array(tyx)
    offset = c_in - np.dot(M_inv, c_out)
    
    # M = np.vstack((M,offset))
    # mode = 'reflect' 
    # should maybe alternate between reflect and extend and even cosntant 
    mode = np.random.choice(['constant','nearest','mirror'])
    
    lbl = do_warp(labels, M_inv, tyx, offset=offset, order=0, mode=mode) # order 0 is 'nearest neighbor'
    # check to make sure the region contains at enough cell pixels; if not, retry
    # cellpx = np.sum(lbl>0)
    
    # Past behavior was to recursively search for a crop that contained at least 10% cell pixels, 
    # to avoid the case where the crop is all background. This would mess up flow and I 
    # reasoned that we never want to train on just background. However, the new code handles
    # background just fine and so this is no longer necessary.
    
            # indented for readibility, will remove at some point
            # cutoff = (numpx/10**(dim+1)) # .1 percent of pixels must be cells
            # if cellpx<cutoff:# or cellpx==numpx: # had to disable the overdense feature for cyto2
            #                 # may not actually be a problem now anyway
            #     # skimage.io.imsave('/home/kcutler/DataDrive/debug/img'+str(depth)+'.png',img[0])
            #     # skimage.io.imsave('/home/kcutler/DataDrive/debug/training'+str(depth)+'.png',lbl[0])
            #     return random_crop_warp(img, Y, tyx, v1, v2, nchan, rescale, scale_range, 
            #                             gamma_range, do_flip, ind, do_labels, depth=depth+1)
            # else:
            #     # continue on, this filter helps get rid of orphaned pixels  (not perfect though)
            #     lbl = mode_filter(lbl)
            
    # boundary instead - fast way to check is number of unique labels 
    if len(fastremap.unique(lbl))<2 and not allow_blank_masks:# or cellpx==numpx: # had to disable the overdense feature for cyto2
                    # may not actually be a problem now anyway
        # skimage.io.imsave('/home/kcutler/DataDrive/debug/img'+str(depth)+'.png',img[0])
        # skimage.io.imsave('/home/kcutler/DataDrive/debug/training'+str(depth)+'.png',lbl[0])
        return random_crop_warp(img, Y, tyx, v1, v2, nchan, rescale, scale_range, 
                                gamma_range, do_flip, ind, do_labels, depth=depth+1, 
                                augment=augment, allow_blank_masks=allow_blank_masks)
        
        
    else:
        # continue on, this filter helps get rid of orphaned pixels  (not perfect though)
        lbl = mode_filter(lbl)
    
    
    # if np.any(lbl):
    #     lbl = mode_filter(lbl)  
            
    #flows now computed in parallel in masks_to_flows_batch
    # it occurs to me that maybe we could parallelize this image augmentation too if we compromise 
    # and just use the same parameters for all images, at least for cropping... since flows are done in parallel,
    # there is no longer an ussue if the patch has no cells 
    
    # each augmentation is now on 50% of the time to ensure that the network also gets to see "raw" images (raw meaning just warped)
    imgi = np.zeros((nchan,)+tyx, np.float32)

    # might need to think more carefully about augmentation for blank fovs with low contrast / bit depth
    # rescaling boosts banding or artifacts to the same range as real signal
    # however, preventing autoscaling at this stage only goes so far if the image has alrady been rescaled... which is often the case for annotations
    # What we can do isntead is detect if there is any foreground in the label image, and then also detect how many unique integers are being used, and use this information to guide our rescaling.

    for k in range(nchan):
        # print('aa', img[k].min(), img[k].max())
        # imgi[k] = do_warp(utils.rescale(img[k]), M_inv, tyx, order=1, offset=offset, mode=mode)
        has_foreground = np.any(lbl)
        if np.issubdtype(img[k].dtype, np.integer):
            nvals = len(fastremap.unique(img[k])) # number of unique values in the image 
        else:
            nvals = len(np.unique(img[k])) # floats must fall back to numpy.unique
        raw_bits = int(np.ceil(np.log2(max(nvals, 1))))

        if has_foreground:
            arr = utils.rescale(img[k])
        else:
            maxv = np.iinfo(img[k].dtype).max if np.issubdtype(img[k].dtype, np.integer) else 1.0
            # print('a',maxv)
            arr = img[k].astype(np.float32) / maxv
            if nvals < maxv // 100 and arr.max()==1.0:
                # low bit depth image without foreground
                # do not rescale to full 0-1 range, will exaggerate banding/artifacts
                arr *= (nvals / maxv) * 10 # boost the signal a bit, but not all the way to 1.0; guranteed not to exceed 1/10 here since 1/100th is the cutoff
                # print('low bit depth image detected, nvals:', nvals, 'maxv:', maxv, 'scaling to', arr.max(), img[k].max())


        imgi[k] = do_warp(arr, M_inv, tyx, order=1, offset=offset, mode=mode)
        
        # some augmentations I only want on half of the time
        # both for speed and because I want the network to see relatively raw images 
        aug_choices = np.random.choice([0,1],8) # faster to preallocate 
        
        if augment:
        # if 0:
            # TODO: make a tool to preview how a given dataset will be augmented with these settings
            # could be useful for debugging and for users to understand what is happening


            # defocus augmentation (inaccurate, but effective)
            if aug_choices[1]:
                imgi[k] = gaussian_filter(imgi[k],np.random.uniform(0,np.sqrt(2))) # was up to 2, but that was too extreme

            
            # percentile clipping augmentation
            if aug_choices[2] and has_foreground:
                dp = .1 # changed this from 10 to .1, as usual pipleine uses 0.01, 10 was way too high for some images 
                dpct = np.random.triangular(left=0, mode=0, right=dp, size=2) # weighted toward 0
                imgi[k] = utils.normalize99(imgi[k],upper=100-dpct[0],lower=dpct[1])


            # low-frequency illumination augmentation to mimic uneven backgrounds
            if aug_choices[0] and has_foreground: 
            # if 1:   

                if aug_choices[7] or not OPEN_SIMPLEX_ENABLED:
                # if 0: 
                    # extreme case: gradient illumination 
                    # choose a random axis 
                    axis = np.random.randint(0,dim)
                    coords = [np.arange(0, s, dtype=np.float32) for s in tyx[::-1]]
                    illum_field = coords[axis]
                    illum_field = (illum_field - illum_field.min()) / (illum_field.max() - illum_field.min())
                else:
                    simplex = OpenSimplex(seed=np.random.randint(0, 2**31)) # deterministic seed from global numpy RNG 
                    spatial_shape = tyx[-2:]
            
                    # get the average cell diamter in pixels to set frequency scale
                    # we don't want to automatially insert features smaller than the cells themselves
                    mean_obj_diam = 2*diameters(lbl) if np.any(lbl) else 1.0
                    freq_jitter = np.random.triangular(left=1, mode=1.0, right=10.0) # should turn this into user-configurable range
                    fs = mean_obj_diam * freq_jitter
                    coords = [np.arange(0, s, dtype=np.float32) / fs for s in spatial_shape[::-1]]
                    illum_field = utils.rescale( simplex.noise2array(*coords)) # get to 0-1

                # if this is allowed to go to 1, becomes flat field
                # take square root so that the background value itself is triangular distribution (background is roughtly min_factor**2)
                min_factor = np.random.triangular(left=0, mode=0, right=1) **.5
                # min_factor = (np.random.triangular(left=0, mode=.9, right=1)) **.5 


                # min_factor = .5 **.5 
                # min_factor = .01**.5

                multiplier = min_factor + (1.0 - min_factor) * illum_field # sets range to [min_factor, 1]
                # multiplier = illum_field


                if imgi[k].ndim > 2:
                    multiplier = multiplier[np.newaxis, ...]
                multiplier = np.broadcast_to(multiplier, imgi[k].shape).astype(np.float32)

                # interpolate between original image and illum field
                imgi[k] = (imgi[k] + min_factor) / (imgi[k].max() + min_factor) * multiplier # make the floor min_factor
  

                # print('fs', fs, mean_obj_diam, freq_jitter,min_factor, imgi[k].max(), imgi[k].min())
                # print(imgi[k].min(),imgi[k].max())
            # if has_foreground:
            #     imgi[k] = utils.adjust_contrast_masked(imgi[k]+.5, lbl, 1)[0] # rescale after illumination change


            # print('mm',imgi[k].min(), imgi[k].max())
            # noise augmentation
            if SKIMAGE_ENABLED and aug_choices[3]:
                var_range = 1e-2
                var = float(np.random.triangular(left=1e-8, mode=1e-8, right=var_range))
                # imgi[k] = random_noise(utils.rescale(imgi[k]), mode="poisson")#, seed=None, clip=True)
                # poisson is super slow... np.random.posson is faster
                # also poisson always gave the same noise, which is very bad...
                # but gaussian speckle is MUCH faster,<1ms vs >4ms 
                noise = np.random.normal(0.0, float(np.sqrt(var)), size=imgi[k].shape).astype(np.float32)
                imgi[k] = imgi[k] + imgi[k] * noise
                imgi[k] = np.clip(imgi[k], 0, 1)
                # imgi[k] = utils.add_gaussian_noise(imgi[k],0,var)
                
            # bit depth augmentation
            if aug_choices[4] and has_foreground:
                # we convert to 16 bit and then right shift by 0-14 bits
                # at the most extreme, that turns 16 bit into 2 bits, so we have 4 levels
                # changed the mode from 8 to 10 to push toward more typical low-quality images

                min_bits = 3  # keep ≥8 gray levels from the source
                max_shift = max(0, min(14, raw_bits - min_bits))
                if max_shift:
                    bit_shift = int(np.random.triangular(0, max_shift//2, max_shift))
                else:
                    bit_shift = 0
                # bit_shift = int(np.random.triangular(left=0, mode=10, right=14))
                im = utils.to_16_bit(imgi[k])
                # imgi[k] = utils.normalize99(im>>bit_shift)
                imgi[k] = utils.rescale(im>>bit_shift)
                
            # edge / line artifact augmentation
            # omnipose was hallucinating stuff at boundaries
            if aug_choices[5]:
                border_inds = utils.border_indices(tyx)
                imgi[k].flat[border_inds] *= np.random.uniform(0,1)
                
            
            # set some pixels randomly to 0 or 1         
            # much faster than random_noise s&p 
            if aug_choices[6]:
                indices = np.random.rand(*tyx) < 0.001
                imgi[k][indices] = np.random.choice([0, 1], size=np.count_nonzero(indices))


            # gamma agumentation - simulates different contrast, the most important and preserves fine structure 
            # now at the end so that other augmentations are also affected by it
            gamma = np.random.triangular(left=gamma_range[0], mode=1, right=gamma_range[1])
            imgi[k] = imgi[k] ** gamma


            # now augment overall normalization
            # this was a really bad idea! 
            # imgi[k] *= np.random.triangular(left=0.5, mode=1.0, right=1.5)


        
    # Moved to the end because it conflicted with the recursion. 
    # Also, flipping the crop is ultimately equivalent and slightly faster.         
    # We now flip along every axis (randomly); could make do_flip a list to avoid some axes if needed
    # also this seems unecessary, since the crop itself is random... but maybe not?
    if do_flip:
        for d in range(1,dim+1):
            if np.random.choice([0,1]):
                imgi = np.flip(imgi,axis=-d) 
                if Y is not None:
                    lbl = np.flip(lbl,axis=-d)
        
        # only flip the spatial dimensions now
        # reasoning is that time and even PSF in z are not symmetric
        # for d in range(1,2+1): 
        #     if np.random.choice([0,1]):
        #         imgi = np.flip(imgi,axis=-d) 
        #         if Y is not None:
        #             lbl = np.flip(lbl,axis=-d)
                        
    return imgi, lbl, scale


def do_warp(A,M_inv,tyx,offset=0,order=1,mode='constant',**kwargs):#,mode,method):
    """ Wrapper function for affine transformations during augmentation. 
    Uses scipy.ndimage.affine_transform().
        
    Parameters
    --------------
    A: NDarray, int or float
        input image to be transformed
    M_inv: NDarray, float
        inverse tranformation matrix
    order: int
        interpolation order, 1 is equivalent to 'nearest',
    """
    return affine_transform(A, M_inv, offset=offset, 
                                          output_shape=tyx, order=order, 
                                          mode=mode,**kwargs)


def scale_to_tenths(x):
    eps = 1e-12  # Small epsilon to prevent log issues
    scale_factor = 10 ** (-torch.floor(torch.log10(torch.abs(x) + eps)) - 1)
    return x * scale_factor
    
def scale_to_tenths(x, max_gain=10):
    eps = 1e-12
    sf  = 10 ** (-torch.floor(torch.log10(torch.abs(x)+eps)) - 1)
    sf  = torch.clamp(sf, 1/max_gain, max_gain)   # cap between 0.1× and 10×
    return x * sf

def loss(self, lbl, y, ext_loss=0):
    """Loss function for Omnipose.

    Parameters
    ----------
    lbl : ND-array, float
        Transformed labels in array [nimg x nchan x xy[0] x xy[1]].
        - ``lbl[:,0]`` cell masks
        - ``lbl[:,1]`` thresholded mask layer
        - ``lbl[:,2]`` boundary field
        - ``lbl[:,3]`` smooth distance field
        - ``lbl[:,4]`` boundary-emphasizing weights
        - ``lbl[:,5:]`` flow components

    y : ND-tensor, float
        Network predictions, with dimension D:
        - ``y[:,:D]`` flow field components at 0,1,...,D-1
        - ``y[:,D]`` distance fields at D
        - ``y[:,D+1]`` boundary fields at D+1
    """
    
    cellmask = lbl[:,1]>0
    if self.nclasses==1: # semantic segmentation, generalize to logits 
        cm = cellmask.float()
        flow_mse = self.MSELoss(y[:,0],cm) #MSE        
        BCE = self.BCELoss(y[:,0],cm) #BCElogits 
        return flow_mse+BCE/20
        
    
    else:   
        # flow components are stored as the last self.dim slices 
        veci = lbl[:,-self.dim:]
        dist = lbl[:,3] # now distance transform replaces probability
        boundary =  lbl[:,2]
        w =  lbl[:,4].detach()
        # w =  lbl[:,1].detach()
        wt = torch.stack([w]*self.dim,dim=1).detach()
        

        flow = y[:,:self.dim] # 0,1,...self.dim-1
        dt = y[:,self.dim]
        
        maxF, minF = flow.max(), flow.min()
        
        # flow = torch.clamp(flow, -5, 5) # clamp flow to [-5,5] range, this is the range we train on
    
        # experimenting with not having any boundary output 
        if self.nclasses==(self.dim+2):
            bd = y[:,self.dim+1]
            bd_loss = self.BCELoss(bd,boundary) #BCElogits 
        else:
            bd_loss = torch.tensor(0, device=self.device) 
        
        # placeholder     
        # loss3 = torch.tensor(0, device=self.device)    
        dist_loss = self.WeightedMSE(dt,dist,w) #WeightedMSELoss distance field, plain MSE does NOT work 
        
        # the distance field has weird stuff happening, I hope
        # that making its gradient explicitly equal to the GT flow will help
        # dims = [k for k in range(-self.dim,0)]
        
        # dims = [k for k in range(1,self.dim+1)]
        # grad = torch.stack(torch.gradient(dt,dim=dims),axis=1)
        # cross_loss = torch.mean(torch.sum(torch.square(veci/5.-grad),axis=0))
        
        # affinity loss, euler loss, boundary loss 
        lossA, lossE, lossB = self.AffinityLoss(flow,dt,veci,dist,
                                                mode = 'all',
                                                seed=0,
                                                
                                                ) 
        div = divergence_torch(veci)
        div_flow = divergence_torch(flow)

        # outer = torch.where(~cellmask)
        # outerDC = div_flow[outer].square().mean()
        # outerFL = [flow[:]] # mean flow in each component?

        bd_loss = self.DerivativeLoss(dt.unsqueeze(1),dist.unsqueeze(1),w.unsqueeze(1),cellmask.unsqueeze(1))
        SSL, norm_loss = self.SSNLoss(flow,veci,dist,w,boundary) #SineSquaredLoss, norm loss

        lossDC = self.MSELoss(div,div_flow) # also tried correlationloss

        # if torch.any(cellmask):
        # if 1: 
        if 0:

            inner = torch.where(cellmask) # torch.where(dist>2) alternative 
            lossDC = self.MSELoss(div[inner],div_flow[inner]) # also tried correlationloss

            # flow_mse = self.WeightedMSE(flow,veci,wt)  #weighted MSE, seems to still be useful 

        
        # else:
            # lossDC = torch.tensor(0, device=self.device)
            # bd_loss = torch.tensor(0, device=self.device)
            # SSL, norm_loss = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
            # lossA = torch.tensor(0, device=self.device)
            # lossE = torch.tensor(0, device=self.device)
            # lossB = torch.tensor(0, device=self.device)
            # flow_mse = torch.tensor(0, device=self.device)

        flow_mse = self.WeightedMSE(flow,veci,wt)  #weighted MSE, seems to still be useful 
        

        # flow_penalty = torch.relu(flow.abs() - 5).pow(2).mean()
        # flow_penalty = torch.nn.functional.softplus(flow.abs() - 5, beta=5).pow(2).mean()
        
        losses = [flow_mse, SSL, bd_loss, norm_loss, dist_loss, 
                  lossA, lossE, lossB, # B is particularly instable maybe, need to plot all these 
                  lossDC,
                #   flow_penalty
                # outerDC,
                
                  ] 
        raw_loss = sum(losses).detach() / len(losses)
        
        # print(', '.join([str(l.item()) for l in losses]))
        # print('flow', maxF, minF, veci.max(), veci.min())

        losses += ext_loss if isinstance(ext_loss, list) else [ext_loss] # add the external losses if any so they get scaled too
        
        # rescaling dynamically seems to work really well, but I might want to add momentum to it
        # that is, slowly change the weight factors by averaging over the last few batches
        losses = [scale_to_tenths(l, max_gain=1e12) for l in losses] 
        # capping gain appears to prevent feedback loop, and might be solving the flow offset issue
        return sum(losses), raw_loss

def bg_flow_corr_penalty(flow, cellmask, eps=1e-6):
    bg = flow * (~cellmask).unsqueeze(1)          # (B,C,H,W)
    B, C, H, W = bg.shape
    f = bg.flatten(2)                             # (B,C,N)
    μ = f.mean(-1, keepdim=True)
    f -= μ                                        # zero-centre

    var = (f.pow(2).mean(-1, keepdim=True) + eps) # (B,C,1)
    std = var.sqrt()
    f_norm = f / std                              # unit variance

    # Pearson correlation matrix; want off-diagonal ≈ 0
    corr = f_norm @ f_norm.transpose(-1, -2) / f.shape[-1]  # (B,C,C)
    off_diag = corr - torch.diag_embed(torch.diagonal(corr, dim1=-2, dim2=-1))
    return off_diag.pow(2).mean()                 # scalar, ≤1
     
import torch
import torch.fft as fft
import torch.nn.functional as F

def bg_flow_spectral_penalty(flow_pred, cellmask,
                             fc=0.10,      # cutoff fraction of Nyquist (0-0.5)
                             dim=2):       # 2→u,v ; 3→u,v,w
    """
    flow_pred : (B, dim, H, W)  predicted flow
    cellmask  : (B, H,  W)      True on cell pixels
    fc        : relative cutoff; 0.10 → remove first 10 % of freq band
    Returns scalar penalty.
    """

    bgmask = (~cellmask).float()                       # 1 on background
    B, C, H, W = flow_pred.shape
    eps = 1e-6

    # zero out foreground so only background contributes
    flow_bg = flow_pred * bgmask.unsqueeze(1)          # (B,C,H,W)

    # FFT: 2× forward real → complex
    spec = fft.rfftn(flow_bg, dim=(-2, -1), norm='forward')  # (B,C,H,W//2+1)
    power = spec.real.pow(2) + spec.imag.pow(2)        # magnitude²

    # radial frequency mask
    fy = torch.fft.fftfreq(H, d=1./H, device=flow_pred.device)  # (-0.5..0.5)
    fx = torch.fft.rfftfreq(W, d=1./W, device=flow_pred.device)
    fy2, fx2 = torch.meshgrid(fy, fx, indexing='ij')
    f_radius = torch.sqrt(fy2 ** 2 + fx2 ** 2)                 # (H,W//2+1)
    lowpass = (f_radius < fc).float()                          # 1 inside cutoff

    # energy below cutoff
    low_energy = (power * lowpass).sum(dim=(-2, -1))           # (B,C)
    # normalise by #bg pixels to keep scale independent
    bg_pix = bgmask.sum(dim=(-2, -1)).clamp_min(1.0).unsqueeze(1)
    penalty = (low_energy / bg_pix).mean()                     # scalar
    return penalty
    
def bg_flow_spec_penalty(flow, cellmask, fc=0.1, eps=1e-6):
    bg = flow * (~cellmask).unsqueeze(1)          # (B,C,H,W)
    power = torch.fft.rfftn(bg, dim=(-2, -1), norm='forward').abs().pow(2)

    fy = torch.fft.fftfreq(bg.size(-2), device=bg.device)
    fx = torch.fft.rfftfreq(bg.size(-1), device=bg.device)
    fy2, fx2 = torch.meshgrid(fy, fx, indexing='ij')
    rad = torch.sqrt(fy2**2 + fx2**2)

    lp_mask = (rad < fc).float()
    low = (power * lp_mask).sum(dim=(-2, -1))
    total = power.sum(dim=(-2, -1)) + eps
    frac = low / total                              # (B,C)
    return frac.mean()                              # scalar in [0,1]

# ## Section IV: Helper functions duplicated from cellpose_omni, plan to find a way to merge them back without import loop

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


def fill_holes_and_remove_small_masks(masks, min_size=None, max_size=None, hole_size=3, dim=2):
    """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)
    
    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes
    
    Parameters
    ----------------
    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    min_size: int (optional, default 3**dim)
        minimum number of pixels per mask (exclusive), can turn off with -1
    max_size: int (optional, default None)
        maximum number of pixels per mask (exclusive)
    hole_size: int (optional, default 3)
        holes bigger than this are NOT filled
    dim: int (optional, default 2)
        dimension of the masks

    Returns
    ---------------
    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    # Min size taken to be an N-cube in ND (9 pixels, 27 voxels, ...) if not specified 
    if min_size is None:
        min_size = 3**dim # N cube 

    # if masks.ndim==2 or dim>2:
        # formatting to integer is critical
        # need to test how it does with 3D
    masks = ncolor.format_labels(masks, min_area=min_size)#, clean=True)
    fill_holes = hole_size>0 # toggle off hole filling by setting hole size to 0
    
    slices = find_objects(masks)
    j = 0
    for i,slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i+1)
            npix = msk.sum()

            too_small = npix < min_size
            too_big = False if max_size is None else npix > max_size

            if (min_size > 0) and (too_small or too_big):
                masks[slc][msk] = 0
            elif fill_holes:   
                hsz = np.count_nonzero(msk)*hole_size/100 #turn hole size into percentage
                #eventually the boundary output should be used to properly exclude real holes vs label gaps
                # for not I just toggle it off 
                if SKIMAGE_ENABLED: # Omnipose version (passes 2D tests)
                    pad = 1
                    unpad = tuple([slice(pad,-pad)]*dim) 
                    padmsk = remove_small_holes(np.pad(msk,pad,mode='constant'),area_threshold=hsz)
                    msk = padmsk[unpad]
                else: #Cellpose version
                    msk = binary_fill_holes(msk)
                masks[slc][msk] = (j+1)
                j+=1
    return masks

def get_boundary(mu,mask,bd=None,affinity_graph=None,contour=False,use_gpu=False,device=None,desprue=False):
    """One way to get boundaries by considering flow dot products. Will be deprecated."""
    
    d = mu.shape[0]
    pad = 1
    pad_seq = [(0,)*2]+[(pad,)*2]*d
    unpad = tuple([slice(pad,-pad)]*d)
    
    mu_pad = utils.normalize_field(np.pad(mu,pad_seq))
    lab_pad = np.pad(mask,pad)

    steps = utils.get_steps(d)  
    steps = np.array(list(set([tuple(s) for s in steps])-set([(0,)*d]))) # remove zero shift element 

    # first time to extract boundaries
    # REPLACE THIS with affinity graph code? or if branch...
    if bd is None:
        bd_pad = np.zeros_like(lab_pad,dtype=bool)

        bd_pad = _get_bd(steps, np.int32(lab_pad), mu_pad, bd_pad) 
        # for k in range(2):
        s_inter = 0
        while desprue and s_inter<np.sum(bd_pad): 
        # for k in [0]:
            sp = utils.get_spruepoints(bd_pad)
            desprue = np.any(sp)
            bd_pad[sp] = False # remove spurs 

        bd_pad = remove_small_objects(bd_pad,min_size=9)
    else:
        bd_pad = np.pad(bd,pad).astype(bool)
        
    #second time to parametrize
    # probably a way to do the boundary finding and stepping in the same step... 
    if contour:
        T,mu_pad = masks_to_flows(lab_pad,
                                  affinity_graph=affinity_graph,
                                  use_gpu=use_gpu,
                                  device=device)[-2:]#,smooth=0,normalize=1)
        
        # utils.imshow(T,10)
        
        step_ok, ind_shift, cross, dot = _get_bd(steps, lab_pad, mu_pad, bd_pad) 
        # values = -(dot+cross) # clockwise 
        values = (-dot+cross) # anticlockwise

        bd_coords = np.array(np.nonzero(bd_pad))
        bd_inds = np.ravel_multi_index(bd_coords,bd_pad.shape)
        labs = np.take(lab_pad,bd_inds)
        unique_L = fastremap.unique(labs)
        contours = parametrize(steps,np.int32(labs),np.int32(unique_L),bd_inds,ind_shift,values,step_ok)

        # value_map = np.zeros(bd_pad.shape,dtype=np.float64)
        contour_map = np.zeros(bd_pad.shape,dtype=np.int32)
        for contour in contours:
            coords_t = np.unravel_index(contour,bd_pad.shape)
            contour_map[coords_t] = np.arange(1,len(contour)+1)
            # contour_map[coords_t] = contours
            
            
        return contour_map[unpad], contours
    
    else:
        return bd_pad[unpad]


# numba does not work yet with this indexing... 
# @njit('(int64[:,:], int32[:,:], float64[:,:,:], boolean[:,:])', nogil=True)
def _get_bd(steps, lab_pad, mu_pad, bd_pad):
    """Helper function to get_boundaries."""
    get_bd = np.all(~bd_pad)
    axes = range(mu_pad.shape[0])
    mask_pad = lab_pad>0
    coord = np.nonzero(mask_pad)
    coords = np.argwhere(mask_pad).T
    A = mu_pad[(Ellipsis,)+coord]
    mag_pad = np.sqrt(np.sum(mu_pad**2,axis=0))
    mag_A = mag_pad[coord]
    
    if not get_bd:
        dot = []
        cross = []
        ind_shift = []
        step_ok = [] #whether or not this step will take you off the boundary 
    else:
        angles1 = []
        angles2 = []
        cutoff1 = np.pi*(1/2.5) # was 1/2, then 1/3, then 1/2.5 or 2/5
        cutoff2 = np.pi*(3/4) # was 3/4, changed to 0.9, back to 3/4 

    for s in steps:
        mag_s = np.sqrt(np.sum(s**2,axis=0))

        if get_bd:
            # First see if the flow is parallel to the flow OPPOSITE the direction of the step 
            neigh_opp = tuple(coords-s[np.newaxis].T)
            B = mu_pad[(Ellipsis,)+neigh_opp]
            mag_B = mag_pad[neigh_opp]
            dot1 = np.sum(np.multiply(A,B),axis=0)
            
            angle1 = np.arccos(dot1.clip(-1,1))
            angle1[np.logical_and(mask_pad[coord],mask_pad[neigh_opp]==0)] = np.pi 
            # consider all background pixels to be opposite

            # next see if the flow is parallel with the step itself 
            dot2 = utils.safe_divide(np.sum([A[a]*(-s[a]) for a in axes],axis=0), mag_s * mag_A)
            angle2 = np.arccos(dot2.clip(-1,1))#*mag_A # note the mag_A multiplication here, attenuates 

            angles1.append(angle1>cutoff1)
            angles2.append(angle2>cutoff2)
            
            # alternate to this: get full affinities and then determine boundaries by connectivity after the fact

        else:
            # maybe I want the dot product with the fild at the step point, choose the most similar 
            # neigh_step = tuple(coords+s[np.newaxis].T)
            neigh_bd = tuple(coords[:,bd_pad[coord]])
            neigh_step = tuple(coords[:,bd_pad[coord]]+s[np.newaxis].T)
            A = mu_pad[(Ellipsis,)+neigh_bd]
            mag_A = mag_pad[neigh_bd]
            B = mu_pad[(Ellipsis,)+neigh_step]
            mag_B = mag_pad[neigh_step]
            dot1 = utils.safe_divide(np.sum(np.multiply(A,B),axis=0),(mag_B * mag_A))
            dot.append(dot1)
            
            dot2 = utils.safe_divide(np.sum([B[a]*(s[a]) for a in axes],axis=0), mag_s * mag_B)#/ (mag_A*mag_s)      
            # dot.append(np.sum((A.T*(s)).T,axis=0))
            cross.append(np.cross(A,s,axisa=0))
            x = np.ravel_multi_index(neigh_step,bd_pad.shape)
            ind_shift.append(x)
            step_ok.append(np.logical_and.reduce((bd_pad[neigh_step],
                                                  lab_pad[neigh_step]==lab_pad[neigh_bd],
                                                  # dot1[bd_pad[coord]]>0,
                                                  # dot2[bd_pad[coord]]>np.cos(3*np.pi/4),
                                                     
                                                 )))

    
    
    if get_bd:
        is_bd = np.any([np.logical_and(a1,a2) for a1,a2 in zip(angles1,angles2)],axis=0)
        bd_pad = np.zeros_like(mask_pad)
        bd_pad[coord] = is_bd
        return bd_pad
    else:
        step_ok = np.stack(step_ok)
        ind_shift = np.array(ind_shift)
        cross = np.stack(cross)
        dot = np.stack(dot)
        
        return step_ok, ind_shift, cross, dot



# possible optimization with ind_shift = np.ravel_multi_index(neighbors,mask.shape)
@njit('(int64[:,:], int32[:], int32[:], int64[:], int64[:,:], float64[:,:], boolean[:,:])', nogil=True)
def parametrize(steps, labs, unique_L, inds, ind_shift, values, step_ok):
    """Parametrize 2D boundaries."""
    sign = np.sum(np.abs(steps),axis=1)
    cardinal_mask = sign>1 # limit to cardinal steps for traversing
    contours = []
    for l in unique_L:
        indices = np.argwhere(labs==l).flatten() # which spots within the inds list etc. are the boundary we want

        # just loop, manually calculate the best step, and proceed
        index = indices[0] # starting point, this may not be best; should choose one that would be an endpoint of a skel

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

            vals[step_mask] = np.inf # avoid these points with min 

            if np.sum(step_mask)<len(step_mask): # 1.1 ms faster than np.any(~step_mask)
                select = np.argmin(vals)
                neighbor_idx = neighbor_inds[select]
                w = np.argwhere(inds[indices]==neighbor_idx)[0][0] # find within limited list
                index = indices[w]
                n_iter += 1
            else:
                closed = True
                contours.append(contour)
    
    return contours  

def get_contour(labels,affinity_graph,coords=None,neighbors=None,cardinal_only=True):
    """Sort 2D boundaries into cyclic paths.

    Parameters:
    -----------
    labels: 2D array, int
        label matrix 
    
    affinity_graph: 2D array, bool
        pixel affinity array, 9 by number of foreground pixels

    
    """
    dim = labels.ndim
    steps,inds,idx,fact,sign = utils.kernel_setup(dim)

    if cardinal_only:
        allowed_inds = np.concatenate(inds[1:2])
    else:
        allowed_inds = np.concatenate(inds[1:])

    shape = labels.shape
    coords = np.nonzero(labels) if coords is None else coords
    neighbors = utils.get_neighbors(coords,steps,dim,shape) if neighbors is None else neighbors
    indexes, neigh_inds, ind_matrix = utils.get_neigh_inds(neighbors,coords,shape)

    csum = np.sum(affinity_graph,axis=0)
    
    # determine what movements are allowed
    step_ok = np.zeros(affinity_graph.shape,bool)


    # print('AA',affinity_graph.dtype,csum.dtype,neigh_inds.dtype)
    # s = allowed_inds[0]
    # print('BB',affinity_graph[s].shape, csum[neigh_inds[s]].shape,neigh_inds[s].shape)
    
    for s in allowed_inds:
        step_ok[s] = np.logical_and.reduce((affinity_graph[s]>0, # must be connected 
                                            csum[neigh_inds[s]]<(3**dim-1), # but the target must also be a boundary ,
                                            neigh_inds[s]>-1 # must not be background, should NOT have to have this here?
                                   ))
        
    # bd_coords = np.array(np.nonzero(bd_pad))
    # bd_inds = np.ravel_multi_index(bd_coords,bd_pad.shape)
    # labs = np.take(lab_pad,bd_inds)
    # bd_inds = np.nonzero(csum<(3**dim-1))
    labs = labels[coords]
    unique_L = fastremap.unique(labs)

    # np.argmin(csum)
    # print('ff',len(fastremap.unique(labels)))
    
    contours = parametrize_contours(steps,np.int32(labs),np.int32(unique_L),neigh_inds,step_ok, csum)
    
    contour_map = np.zeros(shape,dtype=np.int32)
    for contour in contours:
        # coords_t = np.unravel_index(contour,shape)
        coords_t = tuple([c[contour] for c in coords])
        contour_map[coords_t] = np.arange(1,len(contour)+1)
        # contour_map[coords_t] = contour
            
            
    return contour_map, contours, unique_L


# @njit('(int64[:,:], int32[:], int32[:], int64[:,:], float64[:,:])', nogil=True)
@njit #
def parametrize_contours(steps, labs, unique_L, neigh_inds, step_ok, csum):
    """Helper function to sort 2D contours into cyclic paths. See get_contour()."""
    # print('enable njit for this')
    
    
    sign = np.sum(np.abs(steps),axis=1)
    contours = []
    s0 = 4
    for l in unique_L:
        sel = labs==l
        indices = np.argwhere(sel).flatten() # which spots within the inds list etc. are the boundary we want
        # just loop, manually calculate the best step, and proceed
        # index = indices[0] # starting point, this may not be best; should choose one that would be an endpoint of a skel
        index = indices[np.argmin(csum[sel])]

        closed = 0
        contour = []
        n_iter = 0
    
        while not closed and n_iter<len(indices)+1:
            contour.append(neigh_inds[s0,index]) #<<< might want to replace the 4

            # first step: find list of local points
            neighbor_inds = neigh_inds[:,index]
            step_ok_here = step_ok[:,index]
            seen = np.array([i in contour for i in neighbor_inds])
            possible_steps = np.logical_and(step_ok_here, ~seen)
            
            if np.sum(possible_steps)>0: 
                possible_step_indices = np.nonzero(possible_steps)[0]
                
                if len(possible_step_indices)==1:
                    select = possible_step_indices[0]
                else:
                    # There should only ever be multiple options at the start
                    # (maybe that could break down with "just boundary" sections... fix with persistence)
                    # break the tie with preferring counterclockwise
                    consider_steps = steps[possible_step_indices]
                    best = np.argmin(np.array([np.sum(s*steps[3]) for s in consider_steps]))
                    select = possible_step_indices[best]
                    
                neighbor_idx = neighbor_inds[select]
                index = neighbor_idx
                n_iter += 1
            else:
                closed = True
                contours.append(contour)
                
    return contours  




def divergence_torch_old(y):
    dim = y.shape[1]
    dims = [k for k in range(-dim,0)]
    return torch.stack([torch.gradient(y[:,k],dim=k)[0] for k in dims]).sum(dim=0)
    

def divergence_torch(y):
    """
    Divergence for a batched D-vector field stored as ``(B, D, *spatial)``.

    * **GPU / MPS** -> use a single call to ``torch.gradient`` (fast, parallel).
    * **CPU**       -> compute only the gradients actually needed, one component
      at a time, to avoid the unnecessary D^2 work that the vectorised call
      performs on the CPU.

    Returns
    -------
    div : torch.Tensor
        Shape ``(B, *spatial)`` - divergence of ``y``.
    """
    B, D, *spatial = y.shape

    # Guard against degenerate spatial dims that are too small for gradient
    # computation (edge_order >= 1). When any spatial dimension is < 2 the
    # divergence is undefined; return zeros to avoid runtime errors and keep
    # CPU/GPU parity checks focused on the network output.
    if any(s < 2 for s in spatial):
        return torch.zeros((B, *spatial), dtype=y.dtype, device=y.device)
    if y.device.type == 'cpu':
        # Allocate output once and fill in-place
        div = torch.zeros((B, *spatial), dtype=y.dtype, device=y.device)
        for d in range(D):                          # loop over spatial axes
            comp   = y[:, d]                        # (B, *spatial)
            axis   = d + 1                          # 0=batch, 1=first spatial …
            grad_d = torch.gradient(comp, dim=axis)[0]
            div   += grad_d
        return div
    else:
        spatial_axes = list(range(-len(spatial), 0))      # e.g. [-2,-1] in 2-D.
        grads = torch.gradient(y, dim=spatial_axes)       # tuple length == D
        div = sum(g[:, d] for d, g in enumerate(grads))   # pick aligned comps
        return div
    

def _ensure_torch(*arrays, device=None, dtype=torch.float32):
    """Convert numpy arrays to torch tensors if needed."""
    return tuple(
        torch.tensor(arr, dtype=dtype, device=device).unsqueeze(0) if isinstance(arr, np.ndarray) else arr
        for arr in arrays
    )

# def _ensure_torch(*arrays, device=None, dtype=torch.float32):
#     """Convert NumPy arrays to torch tensors on *device* with an extra batch dim."""
#     out = []
#     for arr in arrays:
#         if torch.is_tensor(arr):
#             out.append(arr.to(device=device, dtype=dtype, copy=False))
#         else:
#             # NumPy / list → Tensor and prepend batch axis
#             t = torch.as_tensor(arr, dtype=dtype, device=device).unsqueeze(0)
#             out.append(t)
#     return tuple(out)

# @pyinstrument_profile
def _get_affinity_torch(initial, final, flow, dist, iscell, steps, fact, inds, supporting_inds, 
                        niter,  euler_offset=None,
                        device=torch_GPU,
                        # angle_cutoff=np.pi/2):
                        # angle_cutoff=np.pi/2):
                        # angle_cutoff=np.pi/1.5):

                        angle_cutoff=np.pi/3):

                        # angle_cutoff=np.pi/10):

                        # angle_cutoff=np.pi/4):
    # print('using torch affinity - not equivalent YET, displacement vs flow field')
    # print('shapes',[arr.shape for arr in [initial, final, flow, dist, iscell]])
    # print([isinstance(arr, np.ndarray) for arr in [initial, final, flow, dist, iscell]])
    
    # adds batch dimension 
    initial, final, flow, dist, iscell = _ensure_torch(initial, final, flow, dist, iscell, device=device) 
    
    # compute the displacment vector field; repalcingflow with this does not seem to make a difference now
    # which means we could possibly forgo euler integration altogether 
    # using the displacmeent avoids some internal boundaries 
    mu = final - initial 
    # mu = flow 
    
    # Get the shape of the tensor
    B, D, *DIMS = mu.shape
    S = len(steps)
    
    # I think the new strategy is to fill in the arrays for each step
    # then take acos on the full cosine array for thresholding 
    div = divergence_torch(flow) 
    # div = divergence_torch(mu) # NOTE: my original code still uses the flow field prediciton as mu here, 
    # but easier to experiment here and indeed using displacemnet is much more robust without despurring 
    # thus mI might want to change the main loop as well somehow...
    # actually the thing here is that the scale might be all wrong... 
    
    # so divergence as computed now may be too crude, and I need a better metric for if there is inward flow
    # so that i can connect inner parts of the cell. 
    
    mag = utils.torch_norm(mu,dim=1,keepdim=True)
    # mag = torch.linalg.norm(mu,dim=1,keepdim=True)

    mu_norm = torch.where(mag>0,mu/mag,mu) # avoids dividing during loop
    cos = torch.stack([(mu_norm * mu_norm).sum(dim=1)]*S)
    # div = divergence_torch(mu_norm)
    # print('debug', torch.sum(iscell), torch.max(mag), torch.mean(mag.squeeze()[iscell]), torch.mean(utils.torch_norm(mu_norm,dim=1,keepdim=False)[iscell]))
    div_cutoff = 1/3 # this alone follows internal boundaries quite well 
    div_cutoff = 0    
    
    if euler_offset is None:
        euler_offset = 2*np.sqrt(D)
        # euler_offset = D
        
        
    # print('debug',niter, np.sqrt(niter), np.sqrt(niter/2),torch.mean(dist[dist>0]))
    use_flow = 0 # seems to work just fine without this option? saves time too 
    if use_flow:
        # print('using predicted flow for mag cutoff')
        mag_cutoff = .5
        mag = utils.torch_norm(flow,dim=1,keepdim=True) # alternate on real flow, better for catching boundary faults due to low mag flows 
    else:
        # mag_cutoff = np.sqrt(D) # could be higher or based on niter
        mag_cutoff = 3

    # not used anymore?
    # slow = mag<mag_cutoff
    
    sink = div<div_cutoff
    # sink = dist>D # this is actually much more rubust? 
    # sink = dist>np.sqrt(niter/2) # niter based on the mean distance field, no need to recompute that 
    # sink = dist>torch.mean(dist[dist>0])/2
    
    shape = cos.shape
    device = cos.device      
    is_sink = torch.zeros(shape,dtype=torch.bool,device=device)
    
    # define step slices 
    
    # this preallocation is another great example why using [[]*D]*S is a very bad idea 
    source_slices, target_slices = [[[[] for _ in range(D)] for _ in range(S)] for _ in range(2)]

    # instead of computing divergence with built-in gradient, I can do it manually
    # this is more precise, but still dodn't really show any improvement 
    # div = torch.zeros_like(div)
    
    # source and target slices are arranges so that the target is always in bounds
    # source is offset opposite the direciton of the step for this to be true 
        
    s1,s2,s3 = slice(1,None), slice(0,-1), slice(None,None) # this needs to be generalized to D dimensions
    for i in range(S):
        for j in range(D):
            s = steps[i][j]
            target_slices[i][j], source_slices[i][j] = (s1,s2) if s>0 else (s2,s1) if s<0 else (s3,s3)
            
    
    # print('target slices')
    # for ts,ss,step in zip(target_slices, source_slices, steps):
    #     print(f'source {ss},  target{ts}, {step} {vector_to_arrow(step)}')
        

    for i in range(S//2): # appears to work 

        # Create slices for the in-bounds region

        target_slc = (Ellipsis,)+tuple(target_slices[i])
        source_slc = (Ellipsis,)+tuple(source_slices[i])

        # Pairs that have one in a sink region  
        is_sink[i][source_slc] = is_sink[-(i+1)][target_slc] = torch.logical_or(sink[source_slc],sink[target_slc])
     
        # Compute the cosine of the angle between all pairs in this direction 
        cos[i][source_slc] = cos[-(i+1)][target_slc] = (mu_norm[target_slc] * mu_norm[source_slc]).sum(dim=1)

    # this criterion sets connectivity based on the angle between the two vectors 
    # I wonder if this angle should depend on cardinal vs ordinal...
    # is_parallel = torch.acos(cos.clamp(-1,1))<=angle_cutoff    
    # with torch.no_grad():
    # is_parallel = cos.clamp(-1, 1) >= np.cos(angle_cutoff) # still need a clamp here? Don't think so
    is_parallel = cos >= np.cos(angle_cutoff)
    
    # this is actually superior to my old method, the near condition can have poor behavior on Drad
    # The slow criterion is not used anymore? 
    connectivity = torch.logical_or(is_parallel, is_sink) 
    # print('c', connectivity.shape, is_parallel.shape)
    
    
    connectivity[S//2] = 0 # do not allow self connection via this criterion 
    
    # discard pixels with low connectivity  
    # also take care of background connections here
    csum = torch.sum(connectivity,axis=0)
    
    cutoff = D+2 # not sure if this will generalize to 3d.. those spurs will be connected to possibly 3x3 pixels
    cutoff = 3**(D-1) # + 1
    keep = csum>=cutoff   


    valid_mask = utils.precompute_valid_mask(DIMS,steps,device=keep.device)
    # print('valid',valid_mask.shape)
    # print(connectivity[~valid_mask])
    
    
    # self_idx = inds[0][0]
    # non_self = np.array(list(set(np.arange(len(steps)))-{self_idx})) # I need these to be in order
    # print('non self',non_self)  
    # print('supporting_inds',supporting_inds)
    # for i in non_self:
    for i in range(S//2):
    # for i in [0,1,2]:
    # for i in [3]:
     
        if 1:
            tuples = supporting_inds[i]
            # print('tuples',tuples)
            # source_support = []
            # target_support = []
            target_slc = (Ellipsis,)+tuple(target_slices[i])
            source_slc = (Ellipsis,)+tuple(source_slices[i])
            
            support = torch.zeros_like(keep[source_slc],dtype=torch.int32)
            # support = torch.zeros_like(keep,dtype=torch.int32)
            
            
            # as it tuns out, the corresponding connectivities are already in the right order 
            n_tuples = len(tuples)
            # now we loop over all possible paths from source to target 
            # some paths lead to oob zone, though 
            for j in range(n_tuples): 
                f_inds = tuples[j]
                b_inds = tuple(S-1-np.array(tuples[-(j+1)]))
                # could also do 
                # b_inds = tuple(S-1-np.array(f_inds[::-1]))
                
                # print(i, j, f_inds, b_inds, steps[i], [steps[k] for k in f_inds], [steps[k] for k in b_inds])
                # print(i, j, f_inds, b_inds, steps[i], vector_to_arrow(steps[i]), 
                #       vector_to_arrow([steps[k] for k in f_inds]), 
                #       vector_to_arrow([steps[k] for k in b_inds]))

                    
                for f,b in zip(f_inds,b_inds):
                    # connectivity in the forward direction at the source pixel
                    # supportive_connectivity.append(torch.logical_and(connectivity[f][source_slc], connectivity[b][target_slc])) 
                    # support.add_(torch.logical_and(connectivity[f][source_slc], connectivity[b][target_slc]))
                    # support+= torch.logical_and(connectivity[f][source_slc], connectivity[b][target_slc]))
                    # support = support.add(torch.logical_and(connectivity[f][source_slc], connectivity[b][target_slc]))
                    
                    support = support.add(torch_and([connectivity[f][source_slc], 
                                                     connectivity[b][target_slc],
                                                     valid_mask[f][source_slc],
                                                    #  valid_mask[b][target_slc], # no need to check backwards too, reference same point
                                                    
                                                     ]))
                    
                    
                    # source and target cannot be defined by the f ab b becasue those are different directions
                    # could do an intersection of the steps so that we only add to the support within directions
                    # step_intersect = np.sign(steps[f]+steps[b])
                    # idx_intersect = np.nonzero((steps==step_intersect).all(axis=1))[0][0]
                    # print(step_intersect, vector_to_arrow(step_intersect), idx_intersect) 
                    # target_slc_intersect = (Ellipsis,)+tuple(target_slices[idx_intersect])
                    # source_slc_intersect = (Ellipsis,)+tuple(source_slices[idx_intersect])
                    # print('ff',source_slc_intersect, target_slc_intersect)
                    
                    # print('\tddddddd',shifts_to_slice([steps[f],steps[b]],support.shape))
                    
                    # common_slc = (Ellipsis,)+ shifts_to_slice([steps[f],steps[b]],support.shape)
                    # print('common_slc',common_slc)
                    # support = support.add(torch.logical_and(connectivity[f][source_slc], connectivity[b][target_slc]))
                    # support[common_slc] = torch.logical_and(connectivity[f][source_slc][common_slc], connectivity[b][target_slc][common_slc])
                    
                    # one option: add an index check, leigh neigh inds array to see if >0 for oob 
                    
                    
                    
            # remove internal spurs 
            connectivity[i][source_slc] = connectivity[-(i+1)][target_slc] = torch.where(csum[source_slc]>=7, 1, connectivity[i][source_slc])
            # 1) Create a boolean mask, the same shape as connectivity[i][source_slc].
            # mask = (csum[source_slc] >= 7)

            # # 2) Use boolean indexing to set only those pixels to 1.
            # connectivity[i][source_slc][mask] = 1
            # connectivity[-(i+1)][target_slc][mask] = 1
     
            connectivity[i][source_slc] = connectivity[-(i+1)][target_slc] = torch_and([connectivity[i][source_slc],
                                                                                        connectivity[-(i+1)][target_slc],
                                                                                        
                                                                                        # connections should only exist if both hypervoxels are foreground
                                                                                        iscell[source_slc],
                                                                                        iscell[target_slc],
                                                                                        
                                                                                        # keep are those with "enoguh" connections to begin with 
                                                                                        keep[source_slc], 
                                                                                        keep[target_slc],
                                                                                        
                                                                                        # support connectiosn ensures that the hypervoxels
                                                                                        # are connected not just directly, but in a neighborhood
                                                                                        support>2 # Only keep connections that are supported in more than two routes 
                                                                                        # support[source_slc]>2,
                                                                                        # support[target_slc]>2 
                                                                                        ])
            

            

            

  
    # # I could also just delete all non-cardinal connections...
    return connectivity
    
    

from functools import reduce      
def torch_and_cpu(tensors):
    """
    Pair-wise logical AND using functools.reduce.
    Faster on CPU where kernel-launch overhead is negligible.
    """
    return reduce(torch.logical_and, tensors)

def torch_and_gpu(tensors):
    """
    Vectorized logical AND via torch.all after stacking.
    Single kernel makes it faster on GPU.
    """
    return torch.all(torch.stack(tuple(tensors), dim=0), dim=0)

def torch_and(tensors):
    """
    Dispatch to torch_and_cpu or torch_and_gpu depending on the
    device of the first tensor in *tensors*.
    """
    dev = tensors[0].device if tensors else torch.device('cpu')
    
    try:
        broadcasted = torch.broadcast_tensors(*tensors)
    except AttributeError:
        ref_shape = tensors[0].shape
        broadcasted = [
            t.expand(ref_shape) if t.shape != ref_shape else t
            for t in tensors
        ]
    
    if dev.type == 'cpu':
        return torch_and_cpu(broadcasted)
    else:
        return torch_and_gpu(broadcasted)
    


# padding the arrays makes "step indexing" really easy
# if this were not done, then the indexing would  get wierd for boundary pixels
def _get_affinity(steps, mask_pad, mu_pad, dt_pad, p, p0, 
                  acut=np.pi/2, euler_offset=None,
                  clean_bd_connections=True, pad=0):
    """
    Get the weights associated with the edges of the affinity graph. 
    Here pixels are connected (affinity 1) or disconnected (affinity 0). 
    The particular way I store this affinity graph may also be called an "adjacency list". 
    """

    axes = range(mu_pad.shape[0])
    # coord = np.nonzero(mask_pad) # should this not just be inds/p0?
    # print('coord',np.all(coord==p0),p.shape,p0.shape) yes it is 
    coord = tuple(p0)
    coords = np.stack(coord)

    div = divergence(mu_pad)

    # steps are laid out symmetrically the 0,0,0 in center, but I was getting off results
    d = mask_pad.ndim
    steps, inds, idx, fact, sign = utils.kernel_setup(d)

    # non_self = np.concatenate(inds[1:])
    non_self = np.array(list(set(np.arange(len(steps)))-{inds[0][0]})) # I need these to be in order

    if euler_offset is None:
        euler_offset = 2*np.sqrt(d)
        # euler_offset = d

    shape = mask_pad.shape

    # These functions are incredibly important, as they define neighbor coordinates everywhere
    # INCLUDING at boundaries. Before, I had to pad by 1 to ensure neighbor indexing would not go over. 
    neighbors = utils.get_neighbors(coord,steps,d,shape, pad=pad) # shape (d,3**d,npix)   
    indexes, neigh_inds, ind_matrix = utils.get_neigh_inds(tuple(neighbors),coord,shape)#,background_reflect=True)
    # indexes, neigh_inds, ind_matrix = get_neigh_inds(coords,shape,steps)

    S,L = neigh_inds.shape
    connect = np.zeros((S,L),dtype=bool)
    
    # cutoff for 
    flow_cutoff = 1
    div_cutoff = 0

    # central pixel operations factored out of the loop
    pix_A = p[(Ellipsis,)+coord]
    A = pix_A-p0[:, indexes] # displacement at each pixel 
    mag_A = np.sqrt(np.sum(A**2,axis=0))
    slow = mag_A<flow_cutoff
    sink = div[coord]<div_cutoff
    mask_A = mask_pad[coord]
    dt_pad_A = dt_pad[coord]


    # Including the [0,0] step gives 2-connected 
    # we unfortunately cannot use just half the steps because directionality is not symmetrical
    # i.e. self-referencing does not work here with -1 targets and using the neighbor in the opposing
    # direction to lookup the right index. Unfortunately, quite a lot of the computation is duplicated...
    # the point of this method is to stick to foreground pixels, but that adds complexity. Doing this in
    # torch over all pixels at once would probably be faster. 

    # for i in range(S//2):
    for i in non_self: # non-self 4x faster than range(S), barely slower than range(S//2)

        s = steps[i]
        neigh_indices = neigh_inds[i] # linear indices of pixel neighbors in this direction
        
        # earlier approach: -1 targets were excluded
        # this means that the number of pixels being considered changes depeding on direction
        sel = neigh_indices>-1 # non-foreground pixels have index -1, and that would mess up indexing
        source_inds = indexes[sel] # we therefore only deal with source pixels that have a valid target 
        target_inds = neigh_indices[source_inds] # and these are the corresponding valid targets 
        target = tuple(neighbors[:,i,source_inds])
   
        pix_B = pix_A[:,target_inds]
        B = pix_B - p0[:,target_inds] # displacement at neighbor
        cosAB = utils.safe_divide(np.sum(np.multiply(A[:,source_inds],B),axis=0), mag_A[source_inds] * mag_A[target_inds])

        angleAB = np.arccos(cosAB.clip(-1,1))
        # angleAB[np.logical_xor(mask_A[source_inds],mask_A[target_inds])] = np.pi # background is opposite
        angleAB[~mask_A[target_inds]] = np.pi # background is opposite

        # see if connected in forward direction by thresholding on squared distance of end location  
        sepAB = np.sum((pix_B - pix_A[:,source_inds])**2,axis=0) 

        # threshold determined by average of distance fields 
        # cutoff must be symmetrical
        scut = (euler_offset+np.mean((dt_pad_A[source_inds],dt_pad[target]),axis=0))**2 
        
        # We want pixels that do not move to be internal, connected everywhere
        is_slow = np.logical_or(slow[source_inds],slow[target_inds])
        is_sink = np.logical_or(sink[source_inds],sink[target_inds])


        # a slow pixel at the skeleton should be internal
        # or otherwise pixels that get closer together with somewhat parallel flows
        isconnectAB = np.logical_or(np.logical_and(is_slow,is_sink), 
                                    np.logical_and(sepAB<scut,np.logical_or(angleAB<=acut,is_sink))
                                   )

        # assign symmetrical connectivity 
        connect[i,source_inds] = connect[-(i+1),target_inds] = isconnectAB
        # Since this is overwriting, it is still not perfectly symmetrical...
        
 
    # for i in non_self:
    #     s = steps[i]
    #     neigh_indices = neigh_inds[i] # linear indices of pixel neighbors in this direction
        
    #     # earlier approach: -1 targets were excluded
    #     # this means that the number of pixels being considered changes depeding on direction
    #     sel = neigh_indices>-1 # non-foreground pixels have index -1, and that would mess up indexing
    #     source_inds = indexes[sel] # we therefore only deal with source pixels that have a valid target 
    #     target_inds = neigh_indices[source_inds] # and these are the corresponding valid targets 
    #     target = tuple(neighbors[:,i,source_inds])
   
    #     print(i,np.sum(connect[i,source_inds] != connect[-(i+1),target_inds]))
    # boundary cleanup 

    # discard pixels with low connectivity  
    csum = np.sum(connect,axis=0)
    crop = csum<d
    for i in non_self:
        target = neigh_inds[i,crop] # neighbors from which to delete connections 
        connect[i,crop] = 0 # delete connection from nbeighbor to self
        connect[-(i+1),target[target>-1]] = 0 # delete connection from self to neighbor

    return connect, neighbors, neigh_inds



# numba will require getting rid of stacking, summation, etc., super annoying... the number of pixels to fix is quite
# small in practice, so may not be worth it 
# @njit('(bool_[:,:], int64[:,:], int64[:], int64[:], int64[:],  int64[:], int64, bool_)')
def _despur(connect, neigh_inds, indexes, steps, non_self, 
            cardinal, ordinal, dim, clean_bd_connections=True, 
            iter_cutoff=100, skeletonize=False):
    """Critical cleanup function to get rid of spurious affinities."""
    count = 0
    delta = True
    s0 = len(non_self)//2 #<<<<<<<<<<<<<< idx 

    valid_neighs = neigh_inds > -1 # must avoid using -1 index to access array, could also do edges here maybe to avoid padding 

    while delta and count<iter_cutoff:    
        count+=1
        before = connect.copy()
        
        csum =  np.sum(connect,axis=0) # total number of connections for each hypervoxel
        internal = (csum==(3**dim-1)) # classify those hypervoxels that are "internal"
        csum_cardinal = np.sum(connect[cardinal],axis=0) # total connections in cardinal directions only 
        
        # 1st stage of processing removes spur pixels in parallel
        is_external_spur = csum_cardinal<dim

        # internal spurs are more subtle. I want to patch missing connections between internal pixels.
        # One idea is that usually internal pixels should be sandwiched between at leat two other internal pixels. 
        # This always is the case deep inside the graph, but not when close to boundary "folds" that partially surround intenral pixels 
        # However, any such pixels that are detected as spurs just get connected to everyone, so they don't change at all. They simply get 
        # caught every time as a spur. Since that might lead to extra processing, I could try to avoid it by also condiitoning on
        # the number of internal pixels that are cardinal neighbors. You need  at least two cardinal connections (since it always reduces to a line)
        

        is_internal = np.stack([internal[neigh_inds[s]] for s in cardinal]) # cardinal neighbor internal classification
        is_surround = np.sum(is_internal,axis=0)>1 # restrict to only those with 2+ internal cardinal neighbors
        is_sandwiched = np.any(np.logical_and(is_internal,is_internal[::-1]),axis=0) # flip and or for fast pairwise comparsion 
        is_internal_spur = np.logical_and(is_surround,is_sandwiched)
        # is_internal_spur = is_sandwiched
        
        for i in non_self:
            target = neigh_inds[i]
            valid_target = valid_neighs[i]
            
            # connection = 0 > remove pixels that are insufficiently connected by severing all connections
            # connection = 1 > remove internal spur boundary points by restoring all connections 
            for connection,spur in enumerate([is_external_spur,is_internal_spur]):
                sel = spur*valid_target
                connect[i,indexes[sel]] = connection # seems to actually be faster than  connect[i,sel]
                connect[-(i+1),target[sel]] = connection


        # must recompute after those operations were perfomed 
        csum = np.sum(connect,axis=0)
        internal = csum==(3**dim-1)
        csum_cardinal = np.sum(connect[cardinal],axis=0)
        
        # boundary = np.logical_and(csum<(3**dim-1),csum>0) # right now, boundary criteria more relaxed
        boundary = np.logical_and(csum<(3**dim-1),csum>=dim) # actually, may not be wise to do the above
        
        # the concept of internal-ish is useful for not eating away boundaries too much
        internal_ish = csum>=((3**dim - 1)//2 + 1)
        
        # in cardinal case, all but one cardinal connection 
        # internal_ish_cardinal = csum_cardinal>=((3**dim - 1)//2 + 1) 
        internal_ish_cardinal = csum_cardinal>=(dim + 1) 
        
        connect_boundary_cardinal = np.stack([np.logical_and(cn,boundary[ni]) for cn,ni
                                              in zip(connect[cardinal],neigh_inds[cardinal])])
        
        csum_boundary_cardinal = np.sum(connect_boundary_cardinal,axis=0)

        # the remaining problematic pixels come from boundary points that are insufficiently connected 
        bad = np.logical_and(boundary,csum_boundary_cardinal<dim) 
        
        # decide what kind of pixel removal to do
        if skeletonize: # skeletonize the graph
            # we want to remove all non-internal pixels as long as they are connected to internal-ish pixels
            # unfinished 
            bad = 0
        else: # get rid of all boundary spurs
            # the remaining problematic pixels come from boundary points that are insufficiently connected 
            bad = np.logical_and(boundary,csum_boundary_cardinal<dim) 
            is_internal_ordinal = np.stack([internal[neigh_inds[s]] for s in ordinal])
            is_internal_spur_ordinal = np.any(np.logical_and(is_internal_ordinal,is_internal_ordinal[::-1]),axis=0) 
            bad = np.logical_or(bad,np.logical_and(boundary,is_internal_spur_ordinal) )
            

        candidate_indexes = indexes[bad] 
            
        # candidate_indexes = []
        for idx in candidate_indexes:   
            
            check_inds = [neigh_inds[i,idx] for i in non_self] # find the axis 1 indices of these connected pixels to check 

            if clean_bd_connections:
                
                connect_inds = []
                connect_inds_cardinal = []
                # connect_inds = np.nonzero(connect[:,idx])[0] # get the axis 0 indices the pixel is connected to
                for i in np.nonzero(connect[:,idx])[0]:
                    connect_inds.append(i)
                    if i in cardinal:
                        connect_inds_cardinal.append(i)
                check_inds = [neigh_inds[i,idx] for i in connect_inds] 
                check_inds_cardinal = [neigh_inds[i,idx] for i in connect_inds_cardinal]
                
                boundary_connect = np.sum(np.array([boundary[i] for i in check_inds_cardinal]))
                internal_connect = np.sum(np.array([internal[i] for i in check_inds]))
                
                is_bad_bd = boundary_connect<dim #or internal_connect>3**(dim-1)
                                
                # reconnect or disconnect pixels based on shared connections
                if is_bad_bd:
                    
                    # for ax0,ax1 in [[cardinal,ordinal],[ordinal,cardinal]]:
                    for ax0,ax1 in [[cardinal,ordinal]]:
                    
                        neigh = neigh_inds[ax0,idx] # cardinal neighbors of the current pixel 

                        for i in ax0:
                            if connect[i,idx]: # if connected to a cardinal point
                                target = neigh_inds[i,idx] # index of the pixel we are pointing to 
                                for o in ax1:
                                    t = neigh_inds[o,target] # ordinal neighbor of this neighbor pixel
                                    if t in neigh:
                                    # if np.any(neigh==t): # slower 

                                        # w = np.argwhere(neigh==t)[0][0]
                                        w = np.flatnonzero(neigh==t)[0]
                                        k = ax0[w]
                                        c = (connect[o,target] and connect[k,idx]) and t>-1 and target>-1

                                        connect[o,target] = c # then disconnect the target pixel from it
                                        connect[-(o+1),t] = c # and disconnect this other pixel from the target pixel 


            # fascinatingly, this boundary cleanup makes the distance thresholding much less important
            # it tends to throw away any spurious boundaries anyhow; some edge cells can look a bit strange though 
            # plus you are processing more pixels 
                    
        after = connect.copy()
        delta = np.any(before!=after)
        if count>=iter_cutoff-1:
            print('run over iterations',count)
    return connect

import numpy as np
from numba import njit

@njit
def candidate_cleanup_idx(idx, connect, neigh_inds, cardinal, ordinal, dim, boundary, internal):
    """
    Jitted helper for per-candidate boundary cleanup.
    This function is meant to mimic the inner loop in the original _despur.
    All indices (e.g. from 'cardinal' and 'ordinal') are assumed to be 1D arrays of integers.
    It updates connect in place.
    """
    n_dirs = connect.shape[0]
    # Loop over all cardinal directions for candidate idx.
    for i in range(cardinal.shape[0]):
        d = cardinal[i]
        if connect[d, idx] != 0:
            target = neigh_inds[d, idx]
            if target < 0:
                continue
            # For each ordinal direction, try to “repair” the connection.
            for j in range(ordinal.shape[0]):
                o = ordinal[j]
                # Skip if target is not valid.
                if target < 0:
                    continue
                t = neigh_inds[o, target]
                # Check whether t is among the candidate's cardinal neighbors.
                found = False
                for k in range(cardinal.shape[0]):
                    d2 = cardinal[k]
                    if neigh_inds[d2, idx] == t:
                        found = True
                        break
                if found:
                    c_val = 0
                    # If both the ordinal connection at target and the cardinal connection at idx are present, restore (set to 1)
                    if (connect[o, target] != 0 and connect[d, idx] != 0) and (t > -1 and target > -1):
                        c_val = 1
                    connect[o, target] = c_val
                    # Also enforce symmetry: update the mirrored connection.
                    sym_index = -(o + 1)
                    if t > -1:
                        connect[sym_index, t] = c_val
    return

def _despur(connect, neigh_inds, indexes, steps, non_self, 
            cardinal, ordinal, dim, clean_bd_connections=True, 
            iter_cutoff=100, skeletonize=False):
    """
    Critical cleanup function to get rid of spurious affinities.
    This drop-in replacement has the same header.
    
    It uses vectorized operations for most of the bulk updates and calls a njit-accelerated helper
    (candidate_cleanup_idx) for the per-candidate boundary cleanup.
    
    Note: Due to the sequential nature of the original logic, even this hybrid version may yield
    slight differences. You may need to adjust conditions to exactly match your original behavior.
    """
    count = 0
    delta = True
    s0 = len(non_self) // 2  # preserved for compatibility

    valid_neighs = (neigh_inds > -1)

    while delta and count < iter_cutoff:
        count += 1
        before = connect.copy()

        #--- Stage 1: Spur removal (bulk update) ---#
        csum = np.sum(connect, axis=0)  # total connections per hypervoxel
        internal = (csum == (3**dim - 1))
        csum_cardinal = np.sum(connect[cardinal], axis=0)
        is_external_spur = csum_cardinal < dim

        # Internal spur detection via cardinal neighbors
        internal_neighbors = np.stack([internal[neigh_inds[s]] for s in cardinal])
        is_surround = np.sum(internal_neighbors, axis=0) > 1
        is_sandwiched = np.any(np.logical_and(internal_neighbors, internal_neighbors[::-1]), axis=0)
        is_internal_spur = np.logical_and(is_surround, is_sandwiched)

        # For each direction in non_self, update connection values in bulk.
        for i in non_self:
            target = neigh_inds[i]
            valid_target = valid_neighs[i]
            for connection, spur in enumerate([is_external_spur, is_internal_spur]):
                sel = spur & valid_target
                sel_indexes = indexes[sel]
                connect[i, sel_indexes] = connection
                connect[-(i + 1), target[sel]] = connection

        #--- Stage 2: Boundary cleanup ---#
        csum = np.sum(connect, axis=0)
        internal = (csum == (3**dim - 1))
        csum_cardinal = np.sum(connect[cardinal], axis=0)
        boundary = (csum < (3**dim - 1)) & (csum >= dim)
        # The following two variables are computed but not further used in this candidate cleanup.
        internal_ish = csum >= (((3**dim - 1) // 2) + 1)
        internal_ish_cardinal = csum_cardinal >= (dim + 1)

        # Determine boundary connections in cardinal directions.
        connect_boundary_cardinal = np.stack([connect[s] & boundary[neigh_inds[s]] for s in cardinal])
        csum_boundary_cardinal = np.sum(connect_boundary_cardinal, axis=0)
        bad = boundary & (csum_boundary_cardinal < dim)
        if not skeletonize:
            internal_ordinal = np.stack([internal[neigh_inds[s]] for s in ordinal])
            is_internal_spur_ordinal = np.any(np.logical_and(internal_ordinal, internal_ordinal[::-1]), axis=0)
            bad = bad | (boundary & is_internal_spur_ordinal)
        else:
            bad = np.zeros_like(bad, dtype=bool)

        candidate_indexes = indexes[bad]

        #--- Stage 3: Per-candidate cleanup with njit helper ---#
        if clean_bd_connections:
            for candidate in candidate_indexes:
                candidate_cleanup_idx(candidate, connect, neigh_inds, cardinal, ordinal, dim, boundary, internal)

        after = connect.copy()
        delta = np.any(before != after)
        if count >= iter_cutoff - 1:
            print('run over iterations', count)
    return connect

# this version is a lot faster.
@njit()
def affinity_to_edges(affinity_graph,neigh_inds,step_inds,px_inds):
    """Convert symmetric affinity graph to list of edge tuples for connected components labeling."""
    n_edges = len(step_inds) * len(px_inds)
    edge_list = np.empty((n_edges, 2), dtype=np.int64)
    # edge_list = [(-1,-1)] * n_edges  # Preallocate list with placeholder tuples

    idx = 0
    for s in step_inds:
        for p in px_inds:
            if p <= neigh_inds[s][p] and affinity_graph[s,p]:  # upper triangular 
                edge_list[idx] = (p,neigh_inds[s][p])
                idx += 1
    return edge_list[:idx] # return only the portion edge_list that contins edges 




def affinity_to_masks(affinity_graph,neigh_inds,iscell, coords,
                      cardinal=True,
                      exclude_interior=False,
                      return_edges=False, 
                      verbose=False):
    """ Convert affinity graph to label matrix using connected components."""
    
    if verbose:
        startTime = time.time()
    
    nstep,npix = affinity_graph.shape 
   
    # just run on the edges 
    csum = np.sum(affinity_graph,axis=0)
    dim = iscell.ndim
    boundary = np.logical_and(csum<(3**dim-1),csum>=dim)
    
    if exclude_interior:
        px_inds = np.nonzero(boundary)[0]
    else:
        px_inds = np.arange(npix)
    
    if cardinal and not exclude_interior:
        step_inds = utils.kernel_setup(dim)[1][1] # get the cardinal indices 
    else:
        print('yo')
        # step_inds = np.concatenate(utils.kernel_setup(dim)[1])
        step_inds = np.arange(nstep)
        
    edge_list = affinity_to_edges(affinity_graph,neigh_inds,step_inds,px_inds)
    # print(edge_list[0].shape,edge_list[1].shape)
    # Lazily import networkit here to avoid import-time side effects
    try:
        import networkit as nk  # for connected components
        np.ulong = np.uint64    # restore the old alias
    except Exception as e:
        raise ImportError("networkit is required for affinity_to_masks; please install networkit") from e

    # Create a Networkit graph from the edge list
    g = nk.graph.Graph(n=npix, weighted=False)
    
    # I benchmarked two methods of adding edges:
    # addEdges with a tuple of
    
    # edge_list = (np.array(edge_list[:,0]), np.array(edge_list[:,1]))
    # g.addEdges(edge_list)
    u = np.ascontiguousarray(edge_list[:, 0], dtype=np.uint64)
    v = np.ascontiguousarray(edge_list[:, 1], dtype=np.uint64)
    g.addEdges((u, v))



    # # Assume edge_list is a 2D NumPy array with shape (num_edges, 2)
    # num_edges, _ = edge_list.shape
    # # For an unweighted graph, assign a constant weight (e.g., 1.0) for each edge.
    # data = np.ones(num_edges, dtype=np.float64)
    # # Create a COO matrix from the edge list. Ensure the shape matches your total number of nodes.
    # coo = coo_matrix((data, (edge_list[:, 0], edge_list[:, 1])), shape=(npix, npix))
    # nk.setNumberOfThreads(4)  # e.g. on a 16-core system

    # # Create a graph and add edges in one go.
    # # g = nk.Graph(n=npix, weighted=False, directed=False)
    # # g.addEdges(coo)
    # g = GraphFromCoo(coo, weighted=False, directed=False)


    # Find the connected components
    cc = nk.components.ConnectedComponents(g).run()
    components = cc.getComponents()

    labels = np.zeros(iscell.shape,dtype=int)
    # for i,nodes in enumerate(components):
    #      labels[tuple([c[nodes] for c in coords])] = i+1 if len(nodes)>1 else 0
    comp_id = np.zeros(npix, dtype=np.int32)
    for i, nodes in enumerate(components):
        # Skip singletons or give them label 0
        if len(nodes) > 1:
            comp_id[nodes] = i + 1

    # 'coords' is shape (dim, npix); 
    # 'labels' is your ND array; 
    # we do one vectorized assignment:
    labels[tuple(coords)] = comp_id

    if exclude_interior:
        labels = ncolor.expand_labels(labels)*iscell
    
    coords = np.stack(coords).T
    gone = neigh_inds[(3**dim)//2,csum<dim]
    labels[tuple(coords[gone].T)] = 0 

    if verbose:
        executionTime = (time.time() - startTime)
        omnipose_logger.info('affinity_to_masks(cardinal={}) execution time: {:.3g} sec'.format(cardinal,executionTime))
        
    if return_edges:
        return labels, edge_list, coords, px_inds
    else:
        return labels
        
        
import numpy as np

class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=np.int32)

    def find(self, x):
        """ Path-compressing find. """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            # Union by rank
            if self.rank[rx] < self.rank[ry]:
                self.parent[rx] = ry
            elif self.rank[rx] > self.rank[ry]:
                self.parent[ry] = rx
            else:
                self.parent[ry] = rx
                self.rank[rx] += 1

def affinity_to_masks2(affinity_graph,neigh_inds,iscell, coords,
                      cardinal=True,
                      exclude_interior=False,
                      return_edges=False, 
                      verbose=False):
    """
    Faster replacement for affinity_to_masks using union-find.
    """
    # 1) Basic setup
    nstep, npix = affinity_graph.shape
    dim = iscell.ndim
    csum = np.sum(affinity_graph, axis=0)
    boundary = np.logical_and(csum < (3**dim - 1), csum >= dim)

    if exclude_interior:
        px_inds = np.nonzero(boundary)[0]
    else:
        px_inds = np.arange(npix)

    # Either use only cardinal steps or all steps
    if cardinal and not exclude_interior:
        step_inds = utils.kernel_setup(dim)[1][1]  # cardinal indices only
    else:
        step_inds = np.arange(nstep)

    # 2) Build and populate a union-find over all pixels
    uf = UnionFind(npix)

    for s in step_inds:
        # Find all columns where this step is True
        # (i.e. pixel col is connected to pixel neigh_inds[s, col]).
        cols = np.where(affinity_graph[s])[0]
        # Restrict to boundary or full interior if needed
        #   e.g. cols = np.intersect1d(cols, px_inds) if needed
        #   or just do it outside if exclude_interior is True
        for c in cols:
            if c in px_inds:
                neighbor = neigh_inds[s, c]
                uf.union(c, neighbor)

    # 3) For each pixel, find the “root” and map each root → label
    #    then assign those labels into `iscell.shape`.
    roots = np.array([uf.find(i) for i in range(npix)], dtype=int)
    unique_roots, inv = np.unique(roots, return_inverse=True)

    # If you want singletons to be labeled 0, all others to be labeled 1..N:
    counts = np.bincount(inv)
    label_of_root = np.zeros_like(unique_roots)  # 0 for singletons

    label_id = 1
    for r_idx, ccount in enumerate(counts):
        if ccount > 1:      # skip singletons
            label_of_root[r_idx] = label_id
            label_id += 1

    # final label for each pixel
    pix_labels = label_of_root[inv]

    # 4) Reshape to the original ND layout
    labels = np.zeros(iscell.shape, dtype=int)
    # coords is typically (dim, npix).T, so coords[d][col] is the d-th coordinate
    # You can do:
    labels[tuple(coords)] = pix_labels

    # 5) Optionally expand interior
    if exclude_interior:
        labels = ncolor.expand_labels(labels) * iscell

    # E.g. zero out certain boundary points if you want:
    gone = neigh_inds[(3**dim)//2, csum < dim]
    coords_t = np.stack(coords).T
    labels[tuple(coords_t[gone].T)] = 0

    return labels


def boundary_to_affinity(masks,boundaries):
    """
    This function converts boundary+interior labels to an affinity graph. 
    Boundaries are taken to have label 1,2,...,N and interior pixels have
    some value M>N. This format is the best way I have found to annotate 
    self-contact cells. 
    
    """
    d = masks.ndim
    steps, inds, idx, fact, sign = utils.kernel_setup(d)
    coords = np.nonzero(masks)
    neighbors = utils.get_neighbors(coords,steps,d,masks.shape)


#     # get indices of the hupercubes sharing m-faces on the central n-cube
#     sign = np.sum(np.abs(steps),axis=1) # signature distinguishing each kind of m-face via the number of steps 
#     uniq = fastremap.unique(sign)
#     inds = [np.where(sign==i)[0] for i in uniq] # 2D: [4], [1,3,5,7], [0,2,6,8]. 1-7 are y axis, 3-5 are x, etc. 
#     fact = np.sqrt(uniq) # weighting factor for each hypercube group 

    # Determine Neighbors 
    # We need to construct an "affinity graph", a matrix if N pixels by M neighbors defined by `steps` above.
    # Pixels fall into three categories: interior, exterior, and boundary. Boundary points need need to be
    # connected to interior points, but also be connected to each other along a contour. This code assumes that
    # a correct boundary has been generated.

    neighbor_masks = masks[tuple(neighbors)] #extract list of label values, 

    coords = np.nonzero(masks)
    neighbor_bd = boundaries[tuple(neighbors)] #extract list of boundary values 
    neighbor_int = np.logical_xor(neighbor_masks,neighbor_bd) #internal pixels 
    isneighbor = np.stack([neighbor_int[idx]]*len(steps)) # initialize with all internal pixels connected 

    subinds = np.concatenate(inds[1:])
    mags = np.array([np.linalg.norm(s) for s in steps])
    
    for i,step,sgn in zip(subinds,steps[subinds],sign[subinds]):
        # I basically do a bindary hit-miss operator here, defining a set of internal pixels relative to each step.
        # At least one of these pixels needs to be present in order for the connection in that step to be True.
        # This allows pixels on one side of a 2-px boundary to be connected while not connecting to pixels on the other side. 
        # I should do a bit more testing to see if the additonal ORs are necessary. 
        sm = mags[i]
        dot = np.array([np.dot(step,s)/(m*sm) if m>0 else 0 for s,m in zip(steps,mags)]) #dot of normalized vectors 

        u = np.sqrt(d)
        dot_cutoff = sm / np.sqrt( sm**2 + u**2 ) 
        dottest = np.logical_and(dot-dot_cutoff>=-1e-4,dot<=1)
        indices =  np.argwhere(np.logical_or(dottest, # either inside the forward cone 
                                     np.logical_and(sign==1,dot>=0) # or perpendicular in cardinal direction 
                                    )).flatten()
        
        isneighbor[i] = np.logical_or.reduce((np.any(neighbor_int[indices],axis=0), # if a qualifying adjacent pixel is internal
                                              neighbor_int[i], # target is internal
                                              isneighbor[i] # or the source is internal
                                             ))
    
    return isneighbor

from skimage.segmentation import expand_labels 

# hmm so in fact binary internal masks would work too
# the assumption is simply that the inner masks are separated by 2px boundaries 
def boundary_to_masks(boundaries, binary_mask=None, min_size=9, dist=np.sqrt(2),connectivity=1):
    
    nlab = len(fastremap.unique(np.uint32(boundaries)))
    # 0-1-2 format can also work here 
    if binary_mask is None:
        if nlab==3:
            inner_mask = boundaries==1
        else:
            omnipose_logger.warning('boundary labels improperly formatted')
    else:
        inner_mask = remove_small_objects(measure.label((1-boundaries)*binary_mask,connectivity=connectivity),min_size=min_size)
    # bounds = find_boundaries(masks0,mode='outer')
    
    masks = expand_labels(inner_mask,dist) # need to generalize dist to fact in ND <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # bounds = masks - inner_mask
    inner_bounds = (masks - inner_mask) > 0
    outer_bounds = find_boundaries(masks,mode='inner',connectivity=masks.ndim) #ensure that the mask interfaces are d-1-connected 
    bounds = np.logical_or(inner_bounds,outer_bounds) #restore the inner boundaries 
    return masks, bounds, inner_mask

def linker_label_to_links(maski,linker_label_list):

    linker_mask = np.zeros(maski.shape,bool)
    for l in linker_label_list:
        mask = maski==l
        linker_mask[mask] = 1

    link_masks = ncolor.format_labels(maski,clean=True)
    linker_labels = link_masks.copy()
    unlink_masks = link_masks.copy()
    linker_labels[linker_mask==0] = 0
    unlink_masks[linker_mask] = 0

    dic = fastremap.inverse_component_map(expand_labels(linker_labels,1),unlink_masks)
    links = {(x,z)  for x,y in dic.items() if x!=0 for z in y if z!=0}
    return links



def split_spacetime(augmented_affinity,mask,verbose=False):
    """
    Split lineage labels into frame-by-frame labels and Cell ID / spacetime labeling. 
    """
    shape = mask.shape
    dim = mask.ndim
    neighbors = augmented_affinity[:dim]
    affinity_graph = augmented_affinity[dim] 
    idx = affinity_graph.shape[0]//2
    coords = tuple(neighbors[:,idx])
    
    steps, inds, idx, fact, sign = utils.kernel_setup(dim)
    step_inds = inds[1] # cardinal only 

    npix = augmented_affinity.shape[-1]
    px_inds = np.arange(npix)

    
    sidx = np.nonzero(steps[:,0]==0)[0] # which indexes correspond to spatial-only steps
    tidx = np.nonzero(steps[:,0])[0] # which indexes correspond to steps in (space)time

    prun_ag = affinity_graph.copy()
    prun_ag[tidx] = 0 # zero out all connections to timelike steps

    indexes, neigh_inds, ind_matrix = utils.get_neigh_inds(tuple(neighbors),
                                                           tuple(coords),
                                                           shape)

    lbl = affinity_to_masks(prun_ag, neigh_inds, mask>0, coords, verbose=verbose)
    label_list = lbl[coords]



    # time_steps = np.nonzero([np.all(s == [0,0]) for s in steps[:,-(dim-1):]])[0] # no spatial component
    time_steps = np.nonzero(np.all(steps ==[1,0,0],axis=1))[0] # only the forward step, fewer links to handle  

    edge_list = affinity_to_edges(affinity_graph,
                                  neigh_inds,
                                  time_steps, # if we used step_inds, i.e. all steps, we would get spatial connections too
                                    px_inds)


    link_inds = np.nonzero(edge_list[:,0]!=edge_list[:,1])[0] # non-self links
    links = np.take(label_list, edge_list[link_inds]) # these are the frame-to-frame label links
    # get rid of connections to zero?
    sel = np.nonzero(np.logical_and(links[:,0]!=0,links[:,1]!=0))[0]
    links = links[sel]
    edge_list = edge_list[sel]
    

    unique_pairs,link_counts = fastremap.unique(links,axis=0,return_counts=True)
    uniq,cts = fastremap.unique(unique_pairs[:,0],return_counts=True)
    division_inds = np.nonzero(cts==2)[0]
    mothers = uniq[division_inds]
    mothers,len(link_counts)

    # now that I know where division happens in my link list, I can use this to prune the original affinity grpah to create logs
    # for eahc division, simply remove all connections with a negative time step component 
    # but this will need to be done symmetrically, of course...

    t_fwd = np.nonzero(steps[:,0]==1)[0]
    t_bwd = np.nonzero(steps[:,0]==-1)[0]

    log_affinity_graph = affinity_graph.copy()
    # I suspect there is some spur funny business going on
    # th


    for mother in mothers:
        # find the daugheters
        mother_inds =  np.nonzero(unique_pairs[:,0]==mother)[0] # should be exactly two here
        daughters = np.array([unique_pairs[k][1] for k in mother_inds])
        daughter_counts = np.array([link_counts[k] for k in mother_inds]) # links from mother to daughter

        # but the daughter could also be connected to mother, as there was no symmetry check?
        # daughter_inds = [np.nonzero(unique_pairs[:,0]==daughter)[0] for daughter in daughters]
        # mother_counts = [np.array([link_counts[k] for k in di]) for di in daughter_inds] 
        # print(daughter_inds,mother_counts)
        if verbose:
            print('mother {}, daughters {}, daughter counts {}'.format(mother,daughters,daughter_counts))

        midx = np.nonzero(label_list==mother)[0]
        didx = [np.nonzero(label_list==d)[0] for d in daughters]
        # print(didx)

        # if np.all([x>timelike_cutoff for x in daughter_counts]):
        dmin = daughter_counts.min()
        dmax = daughter_counts.max()


    #     for di in didx:
    #         # delete connections from daughter to mother 
    #         hits = np.isin(neigh_inds[:,di],midx)
    #         log_affinity_graph[:,di] = np.where(hits, 0, log_affinity_graph[:,di]) #this is one way

    #         # delete connections from mother to daughter 
    #         hits = np.isin(neigh_inds[:,midx],di)
    #         log_affinity_graph[:,midx] = np.where(hits, 0, log_affinity_graph[:,midx])


        if dmin/dmax>0.1: # a generous fraction for binary fission or splitting into multiple roughly equal cells 
            if verbose: print('real\n')
            # print(label_list[midx])

            # delete connections forward in time for the mother
            sel = np.ix_(t_fwd,midx)
            log_affinity_graph[sel] = 0

            # I forgot to do this symmetically
            hits = np.isin(neigh_inds[t_bwd],midx)
            log_affinity_graph[t_bwd] = np.where(hits, 0, log_affinity_graph[t_bwd]) 

            # delete connections backward in time for the daughters 
            for di in didx:
                # print(label_list[di])
                sel = np.ix_(t_bwd,di)
                log_affinity_graph[sel] = 0

                # I forgot to do this symmetically
                hits = np.isin(neigh_inds[t_fwd],di)
                log_affinity_graph[t_fwd] = np.where(hits, 0, log_affinity_graph[t_fwd]) 


        else:        
            # otherwsie delete the spurious connections, not every single connection 
            # unfortunately not just pruning entire  
            not_real = np.nonzero(daughter_counts<=dmin)[0]
            print('insufficient temporal connection inds:',not_real)
            for k in not_real:
                di = didx[k]
                daughter = daughters[k]
                print('info',len(midx),len(di),'daughter',daughter)
                # delete backward connections
                sel = np.ix_(t_bwd,di)
                hits = np.isin(neigh_inds[sel],midx)
                log_affinity_graph[sel] = np.where(hits, 0, log_affinity_graph[sel])

                # delete forward connections
                sel = np.ix_(t_fwd,midx)
                hits = np.isin(neigh_inds[sel],di)
                log_affinity_graph[sel] = np.where(hits, 0, log_affinity_graph[sel])

                print('\n')

            # should also handle removal from mother tracking links 


    logs = affinity_to_masks(log_affinity_graph,neigh_inds,mask>0,
                             coords,verbose=verbose)

    
    return lbl, logs

   
