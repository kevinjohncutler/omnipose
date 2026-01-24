from .imports import *

from .affinity import masks_to_affinity, affinity_to_boundary
from .fields import _gradient, _iterate
from .centers import _extend_centers, _extend_centers_gpu
from ..transforms.normalize import normalize_field

omnipose_logger = logging.getLogger(__name__)


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
        if omni:
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
            if omni:
                if dists is not None:
                    from .niter import get_niter
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
                mu = normalize_field(mu, use_torch=True, cutoff=0 if not smooth else 0.15)  ##### transforms.normalize_field(mu,omni) 
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
