import numpy as np
from numba import njit
import cv2
import edt
from scipy.ndimage import binary_dilation, binary_opening, label #again I need to test against skimage labelling
from sklearn.utils.extmath import cartesian
import fastremap

from . import utils

try:
    import torch
    TORCH_ENABLED = True 
    torch_GPU = torch.device('cuda')
    torch_CPU = torch.device('cpu')
except:
    TORCH_ENABLED = False

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_ENABLED = True 
except:
    SKLEARN_ENABLED = False

try:
    from skimage.util import random_noise
    from skimage.filters import gaussian
    from skimage import measure
    import skimage.io #for debugging only
    SKIMAGE_ENABLED = True 
except:
    from scipy.ndimage import gaussian_filter as gaussian
    SKIMAGE_ENABLED = False
    
import logging
omnipose_logger = logging.getLogger(__name__)
omnipose_logger.setLevel(logging.DEBUG)

# We moved a bunch of dupicated code over here from Cellpose to revert back to the original bahavior. This flag is used
# within Cellpose only, but since I want to merge the shared code back together, I'll keep it around here. 
OMNI_INSTALLED = True
#only used in cellpose dynamics, remove when ported back 
    ##
from tqdm import trange 
import ncolor, scipy
from scipy.ndimage.filters import maximum_filter1d
from scipy.ndimage import find_objects, gaussian_filter, generate_binary_structure, label, maximum_filter1d, binary_fill_holes
try:
    from skimage.morphology import remove_small_holes
    SKIMAGE_ENABLED = True
except:
    SKIMAGE_ENABLED = False
    
try:
    from skimage import filters
    SKIMAGE_ENABLED = True
except:
    SKIMAGE_ENABLED = False

from scipy.ndimage import convolve, mean
    ##

### Section I: utilities

# By testing for convergence across a range of superellipses, I found that the following
# ratio guarantees convergence. The edt() package gives a quick (but rough) distance field,
# and it allows us to find a least upper bound for the number of iterations needed for our
# smooth distance field computation. 
def get_niter(dists):
    return np.ceil(np.max(dists)*1.16).astype(int)+1

def dist_to_diam(dt_pos):
    return 6*np.mean(dt_pos)
#     return np.exp(3/2)*gmean(dt_pos[dt_pos>=gmean(dt_pos)])

def diameters(masks,dist_threshold=0):
    dt = edt.edt(np.int32(masks))
    dt_pos = np.abs(dt[dt>dist_threshold])
    return dist_to_diam(np.abs(dt_pos))

### Section II: ground-truth flow computation  

# It is possible that flows can be eliminated in place of the distance field. The current distance field may not be smooth 
# enough, or maybe the network really does require the flow field prediction to work well. But in 3D, it will be a huge
# advantage if the network could predict just the distance (and boudnary) classes and not 3 extra flow components. 
def labels_to_flows(labels, files=None, use_gpu=False, device=None, omni=True,redo_flows=False):
    """ convert labels (list of masks or flows) to flows for training model 

    if files is not None, flows are saved to files to be reused

    Parameters
    --------------

    labels: list of ND-arrays
        labels[k] can be 2D or 3D, if [3 x Ly x Lx] then it is assumed that flows were precomputed.
        Otherwise labels[k][0] or labels[k] (if 2D) is used to create flows and cell probabilities.

    Returns
    --------------

    flows: list of [4 x Ly x Lx] arrays
        flows[k][0] is labels[k], flows[k][1] is cell distance transform, flows[k][2] is Y flow,
        flows[k][3] is X flow, and flows[k][4] is heat distribution

    """
    nimg = len(labels)
    if labels[0].ndim < 3:
        labels = [labels[n][np.newaxis,:,:] for n in range(nimg)]

    if labels[0].shape[0] == 1 or labels[0].ndim < 3 or redo_flows: # flows need to be recomputed
        
        omnipose_logger.info('NOTE: computing flows for labels (could be done before to save time)')
        
        # compute flows; labels are fixed in masks_to_flows, so they need to be passed back
        labels, dist, heat, veci = map(list,zip(*[masks_to_flows(labels[n][0],use_gpu=use_gpu, device=device, omni=omni) for n in trange(nimg)]))
        # concatenate labels, distance transform, vector flows, heat (boundary and mask are computed in augmentations)
        if omni and OMNI_INSTALLED:
            flows = [np.concatenate((labels[n][np.newaxis,:,:], dist[n][np.newaxis,:,:], veci[n], heat[n][np.newaxis,:,:]), axis=0).astype(np.float32)
                        for n in range(nimg)]
        else:
            flows = [np.concatenate((labels[n][np.newaxis,:,:], labels[n][np.newaxis,:,:]>0.5, veci[n]), axis=0).astype(np.float32)
                    for n in range(nimg)]
        if files is not None:
            for flow, file in zip(flows, files):
                file_name = os.path.splitext(file)[0]
                tifffile.imsave(file_name+'_flows.tif', flow)
    else:
        omnipose_logger.info('flows precomputed')
        flows = [labels[n].astype(np.float32) for n in range(nimg)]
    return flows

def masks_to_flows(masks, dists=None, use_gpu=False, device=None, omni=True, spacetime=False):
    """ convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined to be the 
    closest pixel to the median of all pixels that is inside the 
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map. 

    Parameters
    -------------

    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels

    Returns
    -------------

    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].

    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 

    """

    if dists is None:
        masks = ncolor.format_labels(masks)
        dists = edt.edt(masks)
        
    if device is None:
        if use_gpu:
            device = torch_GPU
        else:
            device = torch_CPU
    
    # No reason not to have pytorch installed. Running using CPU is still 2x faster
    # than the dedicated, jitted CPU code thanks to it being parallelized I think.
    masks_to_flows_device = masks_to_flows_torch
        
    if masks.ndim==3 and not spacetime:
        #this branch preserves original 3D apprach with the spacetime flag
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows_device(masks[z], dists, device=device, omni=omni)[0]
            mu[[1,2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows_device(masks[:,y], dists, device=device, omni=omni)[0]
            mu[[0,2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows_device(masks[:,:,x], dists, device=device, omni=omni)[0]
            mu[[0,1], :, :, x] += mu0
        return masks, dists, None, mu #consistency with below
    
    elif masks.ndim==2 or spacetime:
        # this branch needs to
        
        if omni and OMNI_INSTALLED: # padding helps avoid edge artifacts from cut-off cells 
            pad = 15 
            masks_pad = np.pad(masks,pad,mode='reflect')
            dists_pad = np.pad(dists,pad,mode='reflect')
            mu, T = masks_to_flows_device(masks_pad, dists_pad, device=device, omni=omni)
            unpad =  tuple([slice(pad,-pad)]*masks.ndim)
            return masks, dists, T[unpad], mu[(Ellipsis,)+unpad]
            # return masks, dists, T[pad:-pad,pad:-pad], mu[:,pad:-pad,pad:-pad]

        else: # reflection not a good idea for centroid model 
            mu, T = masks_to_flows_device(masks, dists=dists, device=device, omni=omni)
            return masks, dists, T, mu

    else:
        raise ValueError('masks_to_flows only takes 2D or 3D arrays')

#Now fully converted to work for ND, however, the 3D cellpose flow doens't look right. 2D is as expected.  
def masks_to_flows_torch(masks, dists, device=None, omni=True):
    """ convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined using COM

    Parameters
    -------------

    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels

    Returns
    -------------

    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].

    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 

    """
    
    if device is None:
        device = torch.device('cuda')
    
    # the padding here is different than the padding added in masks_to_flows(); 
    # for omni, we reflect masks to extend skeletons to the boundary. Here we pad 
    # with 0 to ensure that edge pixels are not handled differently. 
    pad = 1
    masks_padded = np.pad(masks,pad)

    centers = np.array([])
    if not omni: #do original centroid projection algrorithm
        # WANT TO GENERALIZE TO 3D
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
        # omniversion requires fewer
        n_iter = get_niter(dists) ##### omnipose.core.get_niter
    else:
        slices = scipy.ndimage.find_objects(masks)
        ext = np.array([[s.stop - s.start + 1 for s in slc] for slc in slices])
        n_iter = 2 * (ext.sum(axis=1)).max()
   
    # run diffusion 
    mu, T = _extend_centers_torch(masks_padded, centers, n_iter=n_iter, device=device, omni=omni)
    # normalize
    mu = utils.normalize_field(mu) ##### transforms.normalize_field(mu,omni)
    
    # put into original image
    mu0 = np.zeros((mu.shape[0],)+masks.shape)
    mu0[(Ellipsis,)+np.nonzero(masks)] = mu
    unpad =  tuple([slice(pad,-pad)]*masks.ndim)
    mu_c = T[unpad] # mu_c now heat/distance
    return mu0, mu_c

# edited slightly to fix a 'bleeding' issue with the gradient; now identical to CPU version
def _extend_centers_torch(masks, centers, n_iter=200, device=torch.device('cuda'), omni=True):
    """ runs diffusion on GPU to generate flows for training images or quality control
    
    ...
    
    """
    
    d = masks.ndim
    coords = np.nonzero(masks)
    idx = (3**d)//2 # center pixel index

    neigh = [[-1,0,1] for i in range(d)]
    steps = cartesian(neigh)
    neighbors = np.array([np.add.outer(coords[i],steps[:,i]) for i in range(d)]).swapaxes(-1,-2)
    
    # get indices of the hupercubes sharing m-faces on the central n-cube
    sign = np.sum(np.abs(steps),axis=1) # signature distinguishing each kind of m-face via the number of steps 
    uniq = fastremap.unique(sign)
    inds = [np.where(sign==i)[0] for i in uniq] # 2D: [4], [1,3,5,7], [0,2,6,8]. 1-7 are y axis, 3-5 are x, etc. 
    fact = np.sqrt(uniq) # weighting factor for each hypercube group 
    
    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks[tuple(neighbors)] #extract list of label values, 
    isneighbor = neighbor_masks == neighbor_masks[idx] 
    
    nimg = neighbors.shape[1] // (3**d)
    pt = torch.from_numpy(neighbors).to(device)
    T = torch.zeros((nimg,)+masks.shape, dtype=torch.double, device=device)
    isneigh = torch.from_numpy(isneighbor).to(device)
    
    meds = torch.from_numpy(centers.astype(int)).to(device)

    mask_pix = (Ellipsis,)+tuple(pt[:,idx]) #indexing for the central coordinates 
    center_pix = (Ellipsis,)+tuple(meds)
    neigh_pix = (Ellipsis,)+tuple(pt)    
    for t in range(n_iter):
        if omni and OMNI_INSTALLED:
             T[mask_pix] = eikonal_update_torch(T,pt,isneigh,d,inds,fact) ##### omnipose.core.eikonal_update_torch
        else:
            T[center_pix] += 1
            Tneigh = T[neigh_pix] # T is square, but Tneigh is nimg x <3**d> x <number of points in mask>
            Tneigh *= isneigh # isneigh is <3**d> x <number of points in mask>, zeros out any elements that do not belong in convolution
            T[mask_pix] = Tneigh.mean(axis=1) # mean along the <3**d>-element column does the box convolution 

    # There is still a fade out effect on long cells, not enough iterations to diffuse far enough I think 
    # The log operation does not help much to alleviate it, would need a smaller constant inside. 
    if not omni:
        T = torch.log(1.+ T)
    
    Tcpy = T.clone()
    idx = inds[1]
    mask = isneigh[idx]
    # grads = T[:, pt[idx,:,0], pt[idx,:,1]]*mask # prevent bleedover
    cardinal_points = (Ellipsis,)+tuple(pt[:,idx]) 
    grads = T[cardinal_points]*mask # prevent bleedover
    
    mu_torch = np.stack([(grads[:,-(i+1)]-grads[:,i]).cpu().squeeze() for i in range(0,grads.shape[1]//2)])/2

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
    return phi_total**(1/d) #geometric mean of update along each connectivity set 

def update_torch(a,f):
    # Turns out we can just avoid a ton of infividual if/else by evaluating the update function
    # for every upper limit on the sorted pairs. I do this by piecies using cumsum. The radicand
    # neing nonegative sets the opper limit on the sorted pairs, so we simply select the largest 
    # upper limit that works. 
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

def compute_masks(dP, dist, bd=None, p=None, inds=None, niter=200, mask_threshold=0.0, diam_threshold=12.,
                   flow_threshold=0.4, interp=True, cluster=False, do_3D=False, 
                   min_size=15, resize=None, omni=True, calc_trace=False, verbose=False,
                   use_gpu=False,device=None,nclasses=3):
    """ compute masks using dynamics from dP, dist, and boundary """
    if verbose:
         omnipose_logger.info('mask_threshold is %f',mask_threshold)
    
    if (omni or (inds is not None)) and SKIMAGE_ENABLED:
        if verbose:
            omnipose_logger.info('Using hysteresis threshold.')
        mask = filters.apply_hysteresis_threshold(dist, mask_threshold-1, mask_threshold) # good for thin features
    else:
        mask = dist > mask_threshold # analog to original iscell=(cellprob>cellprob_threshold)
    
    if np.any(mask): #mask at this point is a cell cluster binary map, not labels 
        
        #preprocess flows
        if omni and OMNI_INSTALLED:
            dP_ = div_rescale(dP,mask) ##### omnipose.core.div_rescale
            # dP_ = dP.copy() # need to generalize the divergence code
        else:
            dP_ = dP * mask / 5.
        
        # follow flows
        if p is None:
            p , inds, tr = follow_flows(dP_, mask=mask, inds=inds, niter=niter, interp=interp, 
                                        use_gpu=use_gpu, device=device, omni=omni, calc_trace=calc_trace)
        else: 
            tr = []
            inds = np.stack(np.nonzero(mask)).T
            if verbose:
                omnipose_logger.info('p given')
                
        #calculate masks
        if omni and OMNI_INSTALLED:
            mask = get_masks(p,bd,dist,mask,inds,nclasses,cluster=cluster,
                             diam_threshold=diam_threshold,verbose=verbose) ##### omnipose.core.get_masks
        else:
            mask = get_masks_cp(p, iscell=mask, flows=dP, use_gpu=use_gpu) ### just get_masks
            
        # flow thresholding factored out of get_masks
        if not do_3D:
            shape0 = p.shape[1:]
            flows = dP
            if mask.max()>0 and flow_threshold is not None and flow_threshold > 0 and flows is not None:
                mask = remove_bad_flow_masks(mask, flows, threshold=flow_threshold, use_gpu=use_gpu, device=device, omni=omni)
                _,mask = np.unique(mask, return_inverse=True)
                mask = np.reshape(mask, shape0).astype(np.int32)

        if resize is not None:
            if verbose:
                omnipose_logger.info(f'resizing output with resize = {resize}')
            mask = resize_image(mask, resize[0], resize[1], interpolation=cv2.INTER_NEAREST) ##### transforms.resize_image need to generalize <<<<<<<<<<<<<<<<<<
            Ly,Lx = mask.shape

    else: # nothing to compute, just make it compatible
        omnipose_logger.info('No cell pixels found.')
        p = np.zeros([2,1,1])
        tr = []
        mask = np.zeros(resize)

    # moving the cleanup to the end helps avoid some bugs arising from scaling...
    # maybe better would be to rescale the min_size and hole_size parameters to do the
    # cleanup at the prediction scale, or switch depending on which one is bigger... 
    mask = fill_holes_and_remove_small_masks(mask, min_size=min_size) ##### utils.fill_holes_and_remove_small_masks
    fastremap.renumber(mask,in_place=True) #convenient to guarantee non-skipped labels
    return mask, p, tr


# Omnipose requires (a) a special suppressed Euler step and (b) a special mask reconstruction algorithm. 

# no reason to use njit here except for compatibility with jitted fuctions that call it 
#this way, the same factor is used everywhere (CPU+-interp, GPU)
@njit()
def step_factor(t):
    """ Euler integration suppression factor."""
    return (1+t)

def div_rescale(dP,mask):
    dP = dP.copy()
    dP *= mask 
    dP = utils.normalize_field(dP)
    div = utils.normalize99(divergence(dP))
    dP *= div
    return dP

def divergence(f,sp=None):
    """ Computes divergence of vector field 
    f: array -> vector field components [Fx,Fy,Fz,...]
    sp: array -> spacing between points in respecitve directions [spx, spy,spz,...]
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])

def get_masks(p,bd,dist,mask,inds,nclasses=4,cluster=False,diam_threshold=12.,verbose=False):
    """Omnipose mask recontruction algorithm.
    p: list of 
    """
    if nclasses == 4:
        dt = np.abs(dist[mask]) #abs needed if the threshold is negative
        d = dist_to_diam(dt)
        eps = 1+1/3
        # eps = np.sqrt(2)

    else: #backwards compatibility, doesn't help for *clusters* of thin/small cells
        d = diameters(mask)
        eps = np.sqrt(2)
    
    print('dddddd',nclasses)
    # The mean diameter can inform whether or not the cells are too small to form contiguous blobs.
    # My first solution was to upscale everything before Euler integration to give pixels 'room' to
    # stay together. My new solution is much better: use a clustering algorithm on the sub-pixel coordinates
    # to assign labels. It works just as well and is faster because it doesn't require increasing the 
    # number of points or taking time to upscale/downscale the data. Users can toggle cluster on manually or
    # by setting the diameter threshold higher than the average diameter of the cells. 
    if verbose:
        omnipose_logger.info('Mean diameter is %f'%d)

    if d <= diam_threshold:
        cluster = True
        if verbose:
            omnipose_logger.info('Turning on subpixel clustering for label continuity.')
    
    cell_px = tuple(inds.T)
    coords = np.nonzero(mask)
    newinds = p[(Ellipsis,)+cell_px].T
    mask = np.zeros(p.shape[1:])
    
    print('yeh',SKIMAGE_ENABLED,SKLEARN_ENABLED,cluster,d)
    # the eps parameter needs to be opened as a parameter to the user
    if cluster and SKLEARN_ENABLED:
    # if 0:
        if verbose:
            omnipose_logger.info('Doing DBSCAN clustering with eps=%f'%eps)
        db = DBSCAN(eps=eps, min_samples=3, n_jobs=8).fit(newinds)
        labels = db.labels_
        mask[cell_px] = labels+1
    else:
        newinds = np.rint(newinds).astype(int)
        new_px = tuple(newinds.T)
        skelmask = np.zeros_like(dist, dtype=bool)
        skelmask[new_px] = 1

        #disconnect skeletons at the edge, 5 pixels in 
        border_mask = np.zeros(skelmask.shape, dtype=bool)
        border_px =  border_mask.copy()
        border_mask = binary_dilation(border_mask, border_value=1, iterations=5)

        border_px[border_mask] = skelmask[border_mask]
        if nclasses == 4: #can use boundary to erase joined edge skelmasks 
            border_px[bd>-1] = 0
            if verbose:
                omnipose_logger.info('Using boundary output to split edge defects')
        else: #otherwise do morphological opening to attempt splitting 
            border_px = binary_opening(border_px,border_value=0,iterations=3)

        skelmask[border_mask] = border_px[border_mask]

        if SKIMAGE_ENABLED:
            print('connectivity',skelmask.ndim-1)
            LL = measure.label(skelmask,connectivity=skelmask.ndim-1) #<<<< connectivity may need to be generalized to higher dimensions
        else:
            LL = label(skelmask)[0]
        mask[cell_px] = LL[new_px]
    
    return mask


@njit(['(int16[:,:,:], float32[:], float32[:], float32[:,:])', 
        '(float32[:,:,:], float32[:], float32[:], float32[:,:])'], cache=True)
def map_coordinates(I, yc, xc, Y):
    """
    bilinear interpolation of image 'I' in-place with ycoordinates yc and xcoordinates xc to Y
    
    Parameters
    -------------
    I : C x Ly x Lx
    yc : ni
        new y coordinates
    xc : ni
        new x coordinates
    Y : C x ni
        I sampled at (yc,xc)
    """
    C,Ly,Lx = I.shape
    yc_floor = yc.astype(np.int32)
    xc_floor = xc.astype(np.int32)
    yc = yc - yc_floor
    xc = xc - xc_floor
    for i in range(yc_floor.shape[0]):
        yf = min(Ly-1, max(0, yc_floor[i]))
        xf = min(Lx-1, max(0, xc_floor[i]))
        yf1 = min(Ly-1, yf+1)
        xf1 = min(Lx-1, xf+1)
        y = yc[i]
        x = xc[i]
        for c in range(C):
            Y[c,i] = (np.float32(I[c, yf, xf]) * (1 - y) * (1 - x) +
                      np.float32(I[c, yf, xf1]) * (1 - y) * x +
                      np.float32(I[c, yf1, xf]) * y * (1 - x) +
                      np.float32(I[c, yf1, xf1]) * y * x )

# Generalizing to ND. Again, torch required but should be plenty fast on CPU too compared to jitted but non-explicitly-parallelized CPU code.
# also should just rescale to desired resolution HERE instead of rescaling the masks later... <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# grid_sample will only work for up to 5D tensors (3D segmentation). Will have to address this shortcoming if we ever do 4D. 
def steps_interp(p, dP, niter, use_gpu=True, device=None, omni=True, calc_trace=False):
    d = dP.shape[0]
    shape = dP.shape[1:]
    inds = list(range(d))[::-1] # grid_sample requires a particular ordering 
    if use_gpu and TORCH_ENABLED:
        if device is None:
            device = torch_GPU
        shape = np.array(shape)[inds]-1.  # dP is d.Ly.Lx, inds flips this to flipped X-1, Y-1, ...
        
        # for grid_sample to work, we need im,pt to be (N,C,H,W),(N,H,W,2) or (N,C,D,H,W),(N,D,H,W,3). The 'image' getting interpolated
        # is the flow, which has d=2 channels in 2D and 3 in 3D (d vector components). Output has shape (N,C,H,W) or (N,C,D,H,W)
        pt = torch.from_numpy(p[inds].T).double().to(device)
        for k in range(d):
            pt = pt.unsqueeze(0) # get it in the right shape
        im = torch.from_numpy(dP[inds]).double().to(device).unsqueeze(0) #covert flow numpy array to tensor on GPU, add dimension 
        print('shapes',p.shape,dP.shape,im.shape,pt.shape)
        
        # normalize pt between  0 and  1, normalize the flow
        for k in range(d): 
            im[:,k] *= 2./shape[k]
            pt[...,k] /= shape[k]
            
        # normalize to between -1 and 1
        pt = pt*2-1 
        
        # make an array to track the trajectories 
        if calc_trace:
            trace = torch.clone(pt).detach()
        
        #here is where the stepping happens
        for t in range(niter):
            if calc_trace:
                trace = torch.cat((trace,pt))
            # align_corners default is False, just added to suppress warning
            dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
            if omni and OMNI_INSTALLED:
                dPt /= step_factor(t)
            
            for k in range(d): #clamp the final pixel locations
                pt[...,k] = torch.clamp(pt[...,k] + dPt[:,k], -1., 1.)
            
        #undo the normalization from before, reverse order of operations 
        # <<<< should scale to the correct resolution right here
        pt = (pt+1)*0.5
        for k in range(d): 
            pt[...,k] *= shape[k]
            
        if calc_trace:
            trace = (trace+1)*0.5
            for k in range(d): 
                trace[...,k] *= shape[k]
                
        #pass back to cpu
        if calc_trace:
            tr =  trace[...,inds].cpu().numpy().squeeze().T
        else:
            tr = None
        
        p =  pt[...,inds].cpu().numpy().squeeze().T
        return p, tr
    
    else:
        dPt = np.zeros(p.shape, np.float32)
        if calc_trace:
            tr = np.zeros((p.shape[0],p.shape[1],niter))
        else:
            tr = None
            
        for t in range(niter):
            if calc_trace:
                tr[...,t] = p.copy()
            map_coordinates(dP.astype(np.float32), p[0], p[1], dPt)
            if omni and OMNI_INSTALLED:
                dPt /= step_factor(t)
            for k in range(len(p)):
                p[k] = np.minimum(shape[k]-1, np.maximum(0, p[k] + dPt[k]))
        return p, tr


@njit('(float32[:,:,:,:],float32[:,:,:,:], int32[:,:], int32)', nogil=True)
def steps3D(p, dP, inds, niter):
    """ run dynamics of pixels to recover masks in 3D
    
    Euler integration of dynamics dP for niter steps

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
    """ run dynamics of pixels to recover masks in 2D
    
    Euler integration of dynamics dP for niter steps

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
def follow_flows(dP, mask=None, inds=None, niter=200, interp=True, use_gpu=True, device=None, omni=True, calc_trace=False):
    """ define pixels and run dynamics to recover masks in 2D
    
    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------

    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
    
    mask: (optional, default None)
        pixel mask to seed masks. Useful when flows have low magnitudes.

    niter: int (optional, default 200)
        number of iterations of dynamics to run

    interp: bool (optional, default True)
        interpolate during 2D dynamics (not available in 3D) 
        (in previous versions + paper it was False)

    use_gpu: bool (optional, default False)
        use GPU to run interpolated dynamics (faster than CPU)


    Returns
    ---------------

    p: float32, 3D array
        final locations of each pixel after dynamics

    """
    d = dP.shape[0]
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)
    grid = [np.arange(shape[i]) for i in range(d)]
    p = np.meshgrid(*grid, indexing='ij')
    # not sure why, but I had changed this to float64 at some point... tests showed that map_coordinates expects float32
    # possible issues elsewhere? 
    p = np.array(p).astype(np.float32)

    # added inds for debugging while preserving backwards compatibility 
    if inds is None:
        if omni and (mask is not None):
            inds = np.array(np.nonzero(np.logical_or(mask,np.abs(dP[0])>1e-3))).astype(np.int32).T
        else:
            inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T

    cell_px = (Ellipsis,)+tuple(inds.T)

    if inds.ndim < 2 or inds.shape[0] < 5:
        omnipose_logger.warning('WARNING: no mask pixels found')
        return p, inds, None

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
                                    device=device, omni=omni, calc_trace=calc_trace)
        p[cell_px] = p_interp
    return p, inds, tr

def remove_bad_flow_masks(masks, flows, threshold=0.4, use_gpu=False, device=None, omni=True):
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

    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded.

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    merrors, _ =  flow_error(masks, flows, use_gpu, device, omni) ##### metrics.flow_error
    badi = 1+(merrors>threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks

def flow_error(maski, dP_net, use_gpu=False, device=None, omni=True):
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
        print('ERROR: net flow is not same size as predicted masks')
        return

    # ensure unique masks
    maski = np.reshape(np.unique(maski.astype(np.float32), return_inverse=True)[1], maski.shape)

    # flows predicted from estimated masks
    idx = -1 # flows are the last thing returned now
    dP_masks = masks_to_flows(maski, use_gpu=use_gpu, device=device, omni=omni)[idx] ##### dynamics.masks_to_flows
    # difference between predicted flows vs mask flows
    flow_errors=np.zeros(maski.max())
    for i in range(dP_masks.shape[0]):
        flow_errors += mean((dP_masks[i] - dP_net[i]/5.)**2, maski,
                            index=np.arange(1, maski.max()+1))

    return flow_errors, dP_masks



### Section III: training

# Omnipose has special training settings. Loss function and augmentation. 

def random_rotate_and_resize(X, Y=None, scale_range=1., gamma_range=0.5, xy = (224,224), 
                             do_flip=True, rescale=None, inds=None):
    """ augmentation by random rotation and resizing

        X and Y are lists or arrays of length nimg, with dims channels x Ly x Lx (channels optional)

        Parameters
        ----------
        X: LIST of ND-arrays, float
            list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]

        Y: LIST of ND-arrays, float (optional, default None)
            list of image labels of size [nlabels x Ly x Lx] or [Ly x Lx]. The 1st channel
            of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
            If Y.shape[0]==3, then the labels are assumed to be [cell probability, Y flow, X flow]. 

        scale_range: float (optional, default 1.0)
            Range of resizing of images for augmentation. Images are resized by
            (1-scale_range/2) + scale_range * np.random.rand()
            
        gamma_range: float (optional, default 0.5)
           Images are gamma-adjusted im**gamma for gamma in (1-gamma_range,1+gamma_range) 

        xy: tuple, int (optional, default (224,224))
            size of transformed images to return

        do_flip: bool (optional, default True)
            whether or not to flip images horizontally

        rescale: array, float (optional, default None)
            how much to resize images by before performing augmentations

        Returns
        -------
        imgi: ND-array, float
            transformed images in array [nimg x nchan x xy[0] x xy[1]]

        lbl: ND-array, float
            transformed labels in array [nimg x nchan x xy[0] x xy[1]]

        scale: array, float
            amount each image was resized by

    """
    dist_bg = 5 # background distance field is set to -dist_bg 

    # While in other parts of Cellpose channels are put last by default, here we have chan x Ly x Lx 
    if X[0].ndim>2:
        nchan = X[0].shape[0] 
    else:
        nchan = 1 
    
    nimg = len(X)
    imgi  = np.zeros((nimg, nchan, xy[0], xy[1]), np.float32)
        
    if Y is not None:
        for n in range(nimg):
            labels = Y[n].copy()
            if labels.ndim<3:
                labels = labels[np.newaxis,:,:]
            dist = labels[1]
            dist[dist==0] = - dist_bg
            if labels.shape[0]<6:
                bd = 5.*(labels[1]==1)
                bd[bd==0] = -5.
                labels = np.concatenate((labels, bd[np.newaxis,:]))# add a boundary layer
            if labels.shape[0]<7:
                mask = labels[0]>0
                if np.sum(mask)==0:
                    error_message = 'No cell pixels. Index is'+str(n)
                    omnipose_logger.critical(error_message)
                    raise ValueError(error_message)
                labels = np.concatenate((labels, mask[np.newaxis,:])) # add a mask layer
            Y[n] = labels

        if Y[0].ndim>2:
            nt = Y[0].shape[0] +1 #(added one for weight array)
        else:
            nt = 1
    else:
        nt = 1
    lbl = np.zeros((nimg, nt, xy[0], xy[1]), np.float32)
    

    scale = np.zeros((nimg,2), np.float32)
    for n in range(nimg):
        img = X[n].copy()
        y = None if Y is None else Y[n]
        # use recursive function here to pass back single image that was cropped appropriately 
        # # print(y.shape)
        # skimage.io.imsave('/home/kcutler/DataDrive/debug/img_orig.png',img[0])
        # skimage.io.imsave('/home/kcutler/DataDrive/debug/label_orig.tiff',y[n]) #so at this point the bad label is just fine 
        imgi[n], lbl[n], scale[n] = random_crop_warp(img, y, nt, xy, nchan, scale[n], 
                                                     rescale is None if rescale is None else rescale[n], 
                                                     scale_range, gamma_range, do_flip, 
                                                     inds is None if inds is None else inds[n], dist_bg)
        
    return imgi, lbl, np.mean(scale) #for size training, must output scalar size (need to check this again)

# This function allows a more efficient implementation for recursively checking that the random crop includes cell pixels.
# Now it is rerun on a per-image basis if a crop fails to capture .1 percent cell pixels (minimum). 
def random_crop_warp(img, Y, nt, xy, nchan, scale, rescale, scale_range, gamma_range, do_flip, ind, dist_bg, depth=0):
    # np.random.seed(depth)

    if depth>100:
        error_message = 'Sparse or over-dense image detected. Problematic index is: '+str(ind)+' Image shape is: '+str(img.shape)+' xy is: '+str(xy)
        omnipose_logger.critical(error_message)
        # skimage.io.imsave('/home/kcutler/DataDrive/debug/img'+str(depth)+'.png',img[0])
        raise ValueError(error_message)
    
    if depth>200:
        error_message = 'Recusion depth exceeded. Check that your images contain cells and background within a typical crop. Failed index is: '+str(ind)
        omnipose_logger.critical(error_message)
        raise ValueError(error_message)
        return
    
    do_old = True # Recomputing flow will never work because labels are jagged...
    lbl = np.zeros((nt, xy[0], xy[1]), np.float32)
    numpx = xy[0]*xy[1]
    if Y is not None:
        labels = Y.copy()
        # We want the scale distibution to have a mean of 1
        # There may be a better way to skew the distribution to
        # interpolate the parameter space without skewing the mean 
        ds = scale_range/2
        if do_old:
            scale = np.random.uniform(low=1-ds,high=1+ds,size=2) #anisotropic
        else:
            scale = [np.random.uniform(low=1-ds,high=1+ds,size=1)]*2 # isotropic
        if rescale is not None:
            scale *= 1. / rescale

    # image dimensions are always the last two in the stack (again, convention here is different)
    Ly, Lx = img.shape[-2:]

    # generate random augmentation parameters
    dg = gamma_range/2 
    flip = np.random.choice([0,1])

    if do_old:
        theta = np.random.rand() * np.pi * 2
    else:
        theta = np.random.choice([0, np.pi/4, np.pi/2, 3*np.pi/4]) 

    # random translation, take the difference between the scaled dimensions and the crop dimensions
    dxy = np.maximum(0, np.array([Lx*scale[1]-xy[1],Ly*scale[0]-xy[0]]))
    # multiplies by a pair of random numbers from -.5 to .5 (different for each dimension) 
    dxy = (np.random.rand(2,) - .5) * dxy 

    # create affine transform
    cc = np.array([Lx/2, Ly/2])
    # xy are the sizes of the cropped image, so this is the center coordinates minus half the difference
    cc1 = cc - np.array([Lx-xy[1], Ly-xy[0]])/2 + dxy
    # unit vectors from the center
    pts1 = np.float32([cc,cc + np.array([1,0]), cc + np.array([0,1])])
    # transformed unit vectors
    pts2 = np.float32([cc1,
            cc1 + scale*np.array([np.cos(theta), np.sin(theta)]),
            cc1 + scale*np.array([np.cos(np.pi/2+theta), np.sin(np.pi/2+theta)])])
    M = cv2.getAffineTransform(pts1,pts2)


    method = cv2.INTER_LINEAR
    # the mode determines what happens with out of bounds regions. If we recompute the flow, we can
    # reflect all the scalar quantities then take the derivative. If we just rotate the field, then
    # the reflection messes up the directions. For now, we are returning to the default of padding
    # with zeros. In the future, we may only predict a scalar field and can use reflection to fill
    # the entire FoV with data - or we can work out how to properly extend the flow field. 
    if do_old:
        mode = 0
    else:
        mode = cv2.BORDER_DEFAULT # Does reflection 
        
    label_method = cv2.INTER_NEAREST
    
    imgi  = np.zeros((nchan, xy[0], xy[1]), np.float32)
    for k in range(nchan):
        I = cv2.warpAffine(img[k], M, (xy[1],xy[0]),borderMode=mode, flags=method)
        
        # gamma agumentation 
        gamma = np.random.uniform(low=1-dg,high=1+dg) 
        imgi[k] = I ** gamma
        
        # percentile clipping augmentation 
        dp = 10
        dpct = np.random.triangular(left=0, mode=0, right=dp, size=2) # weighted toward 0
        imgi[k] = utils.normalize99(imgi[k],upper=100-dpct[0],lower=dpct[1])
        
        # noise augmentation 
        if SKIMAGE_ENABLED:
            imgi[k] = random_noise(imgi[k], mode="poisson")
        else:
            #this is quite different
            imgi[k] = np.random.poisson(imgi[k])
            
        # bit depth augmentation
        bit_shift = int(np.random.triangular(left=0, mode=8, right=16, size=1))
        im = (imgi[k]*(2**16-1)).astype(np.uint16)
        imgi[k] = utils.normalize99(im>>bit_shift)

    if Y is not None:
        for k in [0,1,2,3,4,5,6]: # was skipping 2 and 3, now not 
            
            if k==0:
                l = labels[k]
                # print('cellpix',np.sum(l),l.shape,mode,label_method)
                lbl[k] = cv2.warpAffine(l, M, (xy[1],xy[0]), borderMode=mode, flags=label_method)

                # check to make sure the region contains at enough cell pixels; if not, retry
                cellpx = np.sum(lbl[0]>0)
                cutoff = (numpx/1000) # .1 percent of pixels must be cells
                if cellpx<cutoff or cellpx==numpx:
                    # print('YOYO',cellpx,cellpx<cutoff,cellpx==numpx)
                    # skimage.io.imsave('/home/kcutler/DataDrive/debug/img'+str(depth)+'.png',img[0])
                    # skimage.io.imsave('/home/kcutler/DataDrive/debug/training'+str(depth)+'.png',lbl[0])
                    return random_crop_warp(img, Y, nt, xy, nchan, scale, rescale, scale_range, gamma_range, do_flip, ind, dist_bg, depth=depth+1)
            else:
                lbl[k] = cv2.warpAffine(labels[k], M, (xy[1],xy[0]), borderMode=mode, flags=method)
        
        if nt > 1:
            
            mask = lbl[6]
            l = lbl[0].astype(int)
#                 smooth_dist = lbl[n,4].copy()
            dist = edt.edt(l,parallel=8) # raplace with smooth dist function 
            lbl[5] = dist==1 # boundary 

            if do_old:
                v1 = lbl[3].copy() # x component
                v2 = lbl[2].copy() # y component 
                dy = (-v1 * np.sin(-theta) + v2*np.cos(-theta))
                dx = (v1 * np.cos(-theta) + v2*np.sin(-theta))

                lbl[3] = 5.*dx*mask # factor of 5 is applied here to rescale flow components to [-5,5] range 
                lbl[2] = 5.*dy*mask
                
                smooth_dist = smooth_distance(l,dist)
                smooth_dist[dist<=0] = -dist_bg
                lbl[1] = smooth_dist
#                 dist[dist<=0] = -dist_bg
#                 lbl[1] = dist
            else:
#                 _, _, smooth_dist, mu = dynamics.masks_to_flows_torch(l,dists=dist,omni=omni) #would want to replace this with a dedicated dist-only function
                lbl[3] = 5.*mu[1]
                lbl[2] = 5.*mu[0]

                smooth_dist[smooth_dist<=0] = -dist_bg
                lbl[1] = smooth_dist

            bg_edt = edt.edt(mask<0.5,black_border=True) #last arg gives weight to the border, which seems to always lose
            cutoff = 9
            lbl[7] = (gaussian(1-np.clip(bg_edt,0,cutoff)/cutoff, 1)+0.5)
    else:
        lbl = np.zeros((nt,imgi.shape[-2], imgi.shape[-1]))
    
    # Moved to the end because it conflicted with the recursion. Also, flipping the crop is ultimately equivalent and slightly faster. 
    if flip and do_flip:
        imgi = imgi[..., ::-1]
        if Y is not None:
            lbl = lbl[..., ::-1]
            if nt > 1:
                lbl[3] = -lbl[3]
    return imgi, lbl, scale

def loss(self, lbl, y):
    """ Loss function for Omnipose.
    
    Parameters
    --------------
    lbl: ND-array, float
        transformed labels in array [nimg x nchan x xy[0] x xy[1]]
        lbl[:,0] cell masks
        lbl[:,1] distance fields
        lbl[:,2:4] flow fields
        lbl[:,4] distance fields
        lbl[:,5] boundary fields    
        lbl[:,6] thresholded mask layer 
        lbl[:,7] boundary-emphasized weights        
    
    y:  ND-tensor, float
        network predictions
        y[:,:2] flow fields
        y[:,2] distance fields
        y[:,3] boundary fields
    
    """
    
    veci = self._to_device(lbl[:,2:4]) #scaled to 5 in augmentation 
    dist = lbl[:,1] # now distance transform replaces probability
    boundary =  lbl[:,5]
    cellmask = dist>0
    w =  self._to_device(lbl[:,7])  
    dist = self._to_device(dist)
    boundary = self._to_device(boundary)
    cellmask = self._to_device(cellmask).bool()
    flow = y[:,:2] # 0,1
    dt = y[:,2]
    bd = y[:,3]
    a = 10.

    wt = torch.stack((w,w),dim=1)
    ct = torch.stack((cellmask,cellmask),dim=1) 

    loss1 = 10.*self.criterion12(flow,veci,wt)  #weighted MSE 
    loss2 = self.criterion14(flow,veci,w,cellmask) #ArcCosDotLoss
    loss3 = self.criterion11(flow,veci,wt,ct)/a # DerivativeLoss
    loss4 = 2.*self.criterion2(bd,boundary)
    loss5 = 2.*self.criterion15(flow,veci,w,cellmask) # loss on norm 
    loss6 = 2.*self.criterion12(dt,dist,w) #weighted MSE 
    loss7 = self.criterion11(dt.unsqueeze(1),dist.unsqueeze(1),w.unsqueeze(1),cellmask.unsqueeze(1))/a  

    return loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7


# used to recompute the smooth distance on transformed labels

#NOTE: in Omnipose, I do a pad-reflection to extend labels across the boundary so that partial cells are not
# as oddly distorted. This is not implemented here, so there is a discrepancy at image/volume edges. The 
# Omnipose variant is much closer to the edt edge behavior. A more sophisticated 'edge autofill' is really needed for
# a more robust approach (or just crop edges all the time). 
def smooth_distance(masks, dists=None, device=None):
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
    T = torch.zeros((nimg,)+masks_padded.shape, dtype=torch.double, device=device)#(nimg,)+
    isneigh = torch.from_numpy(isneighbor).to(device)
    for t in range(n_iter):
        T[(Ellipsis,)+tuple(pt[:,idx])] = eikonal_update_torch(T,pt,isneigh,d,inds,fact) 
        
    return T.cpu().squeeze().numpy()[tuple([slice(pad,-pad)]*d)]


### Section IV: duplicated mask recontruction


### Section V: Duplicated

def resize_image(img0, Ly=None, Lx=None, rsz=None, interpolation=cv2.INTER_LINEAR, no_channels=False):
    """ resize image for computing flows / unresize for computing dynamics

    Parameters
    -------------

    img0: ND-array
        image of size [Y x X x nchan] or [Lz x Y x X x nchan] or [Lz x Y x X]

    Ly: int, optional

    Lx: int, optional

    rsz: float, optional
        resize coefficient(s) for image; if Ly is None then rsz is used

    interpolation: cv2 interp method (optional, default cv2.INTER_LINEAR)

    Returns
    --------------

    imgs: ND-array 
        image of size [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

    """
    if Ly is None and rsz is None:
        error_message = 'must give size to resize to or factor to use for resizing'
        omnipose_logger.critical(error_message)
        raise ValueError(error_message)

    if Ly is None:
        # determine Ly and Lx using rsz
        if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
            rsz = [rsz, rsz]
        if no_channels:
            Ly = int(img0.shape[-2] * rsz[-2])
            Lx = int(img0.shape[-1] * rsz[-1])
        else:
            Ly = int(img0.shape[-3] * rsz[-2])
            Lx = int(img0.shape[-2] * rsz[-1])
    
    # no_channels useful for z-stacks, sot he third dimension is not treated as a channel
    # but if this is called for grayscale images, they first become [Ly,Lx,2] so ndim=3 but 
    if (img0.ndim>2 and no_channels) or (img0.ndim==4 and not no_channels):
        if no_channels:
            imgs = np.zeros((img0.shape[0], Ly, Lx), np.float32)
        else:
            imgs = np.zeros((img0.shape[0], Ly, Lx, img0.shape[-1]), np.float32)
        for i,img in enumerate(img0):
            imgs[i] = cv2.resize(img, (Lx, Ly), interpolation=interpolation)
    else:
        imgs = cv2.resize(img0, (Lx, Ly), interpolation=interpolation)
    return imgs


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


# duplicated from cellpose temporarily, neec to pass through spacetime before re-inseting 
def fill_holes_and_remove_small_masks(masks, min_size=15, hole_size=3, scale_factor=1, spacetime=1):
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

    if masks.ndim==2 or spacetime:
        # formatting to integer is critical
        # need to test how it does with 3D
        masks = ncolor.format_labels(masks, min_area=min_size)
        
    hole_size *= scale_factor
        
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)
    
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
                if msk.ndim==3:
                    for k in range(msk.shape[0]):
                        # Omnipose version (breaks 3D tests)
                        # padmsk = remove_small_holes(np.pad(msk[k],1,mode='constant'),hsz)
                        # msk[k] = padmsk[1:-1,1:-1]
                        
                        #Cellpose version
                        msk[k] = binary_fill_holes(msk[k])

                else:          
                    if SKIMAGE_ENABLED: # Omnipose version (passes 2D tests)
                        padmsk = remove_small_holes(np.pad(msk,1,mode='constant'),hsz)
                        msk = padmsk[1:-1,1:-1]
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
