import logging
import time
import numpy as np
import torch
from numba import njit, prange

import fastremap
import ncolor
from skimage import measure
from skimage.morphology import remove_small_objects
from skimage.segmentation import expand_labels, find_boundaries

from .. import utils
from ..transforms.normalize import safe_divide
from ..transforms.vector import torch_norm
from ..gpu import torch_GPU
from .fields import _ensure_torch, torch_and, divergence_torch, divergence
from .njit import candidate_cleanup_idx

omnipose_logger = logging.getLogger(__name__)


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
    max_label = links_arr.max() + 1
    link_set = set()
    for r in range(links_arr.shape[0]):
        a = links_arr[r, 0]
        b = links_arr[r, 1]
        if a > b:
            a, b = b, a
        link_set.add(a * max_label + b)

    for k in prange(len(inds)):
        i = inds[k]
        for j in range(piece_masks.shape[1]):
            a = piece_masks[i, j]
            b = piece_masks[idx, j]
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if a * max_label + b in link_set:
                is_link[i, j] = True
    return is_link


def get_link_matrix(links, piece_masks, inds, idx, is_link):
    """Convert an iterable of (a,b) link tuples into a 2D array and mark links."""
    if not links:
        return is_link
    links_arr = np.array(list(links), dtype=np.int64)
    return _get_link_matrix(links_arr, piece_masks, inds, idx, is_link)

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
    if dists is not None: # pragma: no cover
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
    
    # compute the displacement vector field; replacing flow with this does not seem to make a difference now
    # which means we could possibly forgo euler integration altogether 
    # using the displacement avoids some internal boundaries 
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
    
    mag = torch_norm(mu, dim=1, keepdim=True)
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
        mag = torch_norm(flow, dim=1, keepdim=True) # alternate on real flow, better for catching boundary faults due to low mag flows 
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
    
    


# numba will require getting rid of stacking, summation, etc., super annoying... the number of pixels to fix is quite
# small in practice, so may not be worth it 
# @njit('(bool_[:,:], int64[:,:], int64[:], int64[:], int64[:],  int64[:], int64, bool_)')

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
        


def boundary_to_affinity(masks,boundaries):  # pragma: no cover
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

# hmm so in fact binary internal masks would work too
# the assumption is simply that the inner masks are separated by 2px boundaries 

def boundary_to_masks(boundaries, binary_mask=None, min_size=9, dist=np.sqrt(2),connectivity=1):  # pragma: no cover
    
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


def _despur(connect, neigh_inds, indexes, steps, non_self,
            cardinal, ordinal, dim, clean_bd_connections=True,
            iter_cutoff=100, skeletonize=False):
    """
    Critical cleanup function to get rid of spurious affinities.

    This drop-in replacement has the same header.

    It uses vectorized operations for most of the bulk updates and calls a njit-accelerated helper
    (candidate_cleanup_idx) for the per-candidate boundary cleanup.
    """
    count = 0
    delta = True
    s0 = len(non_self) // 2

    valid_neighs = (neigh_inds > -1)

    while delta and count < iter_cutoff:
        count += 1
        before = connect.copy()

        csum = np.sum(connect, axis=0)
        internal = (csum == (3 ** dim - 1))
        csum_cardinal = np.sum(connect[cardinal], axis=0)
        is_external_spur = csum_cardinal < dim

        internal_neighbors = np.stack([internal[neigh_inds[s]] for s in cardinal])
        is_surround = np.sum(internal_neighbors, axis=0) > 1
        is_sandwiched = np.any(np.logical_and(internal_neighbors, internal_neighbors[::-1]), axis=0)
        is_internal_spur = np.logical_and(is_surround, is_sandwiched)

        for i in non_self:
            target = neigh_inds[i]
            valid_target = valid_neighs[i]
            for connection, spur in enumerate([is_external_spur, is_internal_spur]):
                sel = spur & valid_target
                sel_indexes = indexes[sel]
                connect[i, sel_indexes] = connection
                connect[-(i + 1), target[sel]] = connection

        csum = np.sum(connect, axis=0)
        internal = (csum == (3 ** dim - 1))
        csum_cardinal = np.sum(connect[cardinal], axis=0)
        boundary = (csum < (3 ** dim - 1)) & (csum >= dim)

        internal_ish = csum >= (((3 ** dim - 1) // 2) + 1)
        internal_ish_cardinal = csum_cardinal >= (dim + 1)

        connect_boundary_cardinal = np.stack([connect[s] & boundary[neigh_inds[s]] for s in cardinal])
        csum_boundary_cardinal = np.sum(connect_boundary_cardinal, axis=0)
        bad = boundary & (csum_boundary_cardinal < dim)
        if not skeletonize:
            internal_ordinal = np.stack([internal[neigh_inds[s]] for s in ordinal])
            is_internal_spur_ordinal = np.any(np.logical_and(internal_ordinal, internal_ordinal[::-1]), axis=0)
            bad = bad | (boundary & is_internal_spur_ordinal)
        else: # pragma: no cover
            bad = np.zeros_like(bad, dtype=bool)

        candidate_indexes = indexes[bad]

        if clean_bd_connections:
            for candidate in candidate_indexes:
                candidate_cleanup_idx(candidate, connect, neigh_inds, cardinal, ordinal, dim, boundary, internal)

        after = connect.copy()
        delta = np.any(before != after)
        if count >= iter_cutoff - 1:
            print('run over iterations', count)
    return connect


def split_spacetime(augmented_affinity, mask, verbose=False):  # pragma: no cover
    """
    Split lineage labels into frame-by-frame labels and Cell ID / spacetime labeling.
    """
    shape = mask.shape
    dim = mask.ndim
    neighbors = augmented_affinity[:dim]
    affinity_graph = augmented_affinity[dim]
    idx = affinity_graph.shape[0] // 2
    coords = tuple(neighbors[:, idx])

    steps, inds, idx, fact, sign = utils.kernel_setup(dim)
    step_inds = inds[1]

    npix = augmented_affinity.shape[-1]
    px_inds = np.arange(npix)

    sidx = np.nonzero(steps[:, 0] == 0)[0]
    tidx = np.nonzero(steps[:, 0])[0]

    prun_ag = affinity_graph.copy()
    prun_ag[tidx] = 0

    indexes, neigh_inds, ind_matrix = utils.get_neigh_inds(tuple(neighbors),
                                                           tuple(coords),
                                                           shape)

    lbl = affinity_to_masks(prun_ag, neigh_inds, mask > 0, coords, verbose=verbose)
    label_list = lbl[coords]

    time_steps = np.nonzero(np.all(steps == [1, 0, 0], axis=1))[0]

    edge_list = affinity_to_edges(affinity_graph,
                                  neigh_inds,
                                  time_steps,
                                  px_inds)

    link_inds = np.nonzero(edge_list[:, 0] != edge_list[:, 1])[0]
    links = np.take(label_list, edge_list[link_inds])
    sel = np.nonzero(np.logical_and(links[:, 0] != 0, links[:, 1] != 0))[0]
    links = links[sel]
    edge_list = edge_list[sel]

    unique_pairs, link_counts = fastremap.unique(links, axis=0, return_counts=True)
    uniq, cts = fastremap.unique(unique_pairs[:, 0], return_counts=True)
    division_inds = np.nonzero(cts == 2)[0]
    mothers = uniq[division_inds]
    mothers, len(link_counts)

    t_fwd = np.nonzero(steps[:, 0] == 1)[0]
    t_bwd = np.nonzero(steps[:, 0] == -1)[0]

    log_affinity_graph = affinity_graph.copy()

    for mother in mothers:
        mother_inds = np.nonzero(unique_pairs[:, 0] == mother)[0]
        daughters = np.array([unique_pairs[k][1] for k in mother_inds])
        daughter_counts = np.array([link_counts[k] for k in mother_inds])

        if verbose:
            print('mother {}, daughters {}, daughter counts {}'.format(mother, daughters, daughter_counts))

        midx = np.nonzero(label_list == mother)[0]
        didx = [np.nonzero(label_list == d)[0] for d in daughters]

        dmin = daughter_counts.min()
        dmax = daughter_counts.max()

        if dmin / dmax > 0.1:
            if verbose:
                print('real')

            sel = np.ix_(t_fwd, midx)
            log_affinity_graph[sel] = 0

            hits = np.isin(neigh_inds[t_bwd], midx)
            log_affinity_graph[t_bwd] = np.where(hits, 0, log_affinity_graph[t_bwd])

            for di in didx:
                sel = np.ix_(t_bwd, di)
                log_affinity_graph[sel] = 0

                hits = np.isin(neigh_inds[t_fwd], di)
                log_affinity_graph[t_fwd] = np.where(hits, 0, log_affinity_graph[t_fwd])

        else:
            not_real = np.nonzero(daughter_counts <= dmin)[0]
            print('insufficient temporal connection inds:', not_real)
            for k in not_real:
                di = didx[k]
                daughter = daughters[k]
                print('info', len(midx), len(di), 'daughter', daughter)
                sel = np.ix_(t_bwd, di)
                hits = np.isin(neigh_inds[sel], midx)
                log_affinity_graph[sel] = np.where(hits, 0, log_affinity_graph[sel])

                sel = np.ix_(t_fwd, midx)
                hits = np.isin(neigh_inds[sel], di)
                log_affinity_graph[sel] = np.where(hits, 0, log_affinity_graph[sel])

                print()

    logs = affinity_to_masks(log_affinity_graph, neigh_inds, mask > 0,
                             coords, verbose=verbose)

    return lbl, logs
