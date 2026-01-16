from .imports import *



def precompute_valid_mask(shape, steps, device=None):
    """
    Boolean mask telling whether both a pixel and its neighbour at offset
    `steps[k]` are inside an N-D volume.

    Parameters
    ----------
    shape : tuple[int]
        Spatial dimensions, e.g. (Y, X)   or   (Z, Y, X).
    steps : list[tuple[int]]
        Offset table returned by `kernel_setup(dim)`.
    device : torch.device or None
        Device for the returned tensor.

    Returns
    -------
    valid : torch.BoolTensor        # shape (K, 1, *shape)
    """
    dim   = len(shape)
    step  = torch.as_tensor(steps, dtype=torch.int64, device=device)  # (K, dim)
    K     = step.shape[0]

    # start with all-true, then clear forbidden strips axis by axis
    valid = torch.ones((K,)+tuple(shape), dtype=torch.bool, device=device)

    for ax, size in enumerate(shape):
        coord = torch.arange(size, device=device)                                   # (size,)
        coord = coord.reshape((1,)+ (1,)*ax + (size,) + (1,)*(dim-ax-1))            # (1,…,size,…,1)

        neg = torch.clamp(-step[:, ax], min=0).reshape((K,)+ (1,)*dim)              # (K,1,1,…)
        pos = torch.clamp( step[:, ax], min=0).reshape((K,)+ (1,)*dim)

        valid &= (coord >= neg) & (coord < size - pos)

    # Insert singleton channel dimension so output matches connectivity (K,1,*shape)
    valid = valid.unsqueeze(1)
    return valid


# @njit 
# def get_neighbors(coords, steps, dim, shape, edges=None, pad=0):
#     print('this version actually a lot slower than below ')
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
# def get_neighbors(coords, steps, dim, shape, edges=None, pad=0):
#     """
#     Get the coordinates of all neighbor pixels. 
#     Coordinates of pixels that are out-of-bounds get clipped. 
#     """
#     if edges is None:
#         edges = [np.array([-1+pad,s-pad]) for s in shape]
        
#     # print('edges',edges,'\n')
        
#     npix = coords[0].shape[-1]
#     neighbors = np.empty((dim, len(steps), npix), dtype=np.int64)
    
#     for d in range(dim):        
#         S = steps[:,d].reshape(-1, 1)
#         X = coords[d] + S
#         # mask = np.logical_and(np.isin(X, edges[d]), ~np.isin(X+S, edges[d]))

#         # out of bounds is where the shifted coordinate X is in the edges list
#         # that second criterion might have been for my batched stuff 
#         oob = np.logical_and(np.isin(X, edges[d]), ~np.isin(X+S, edges[d]))
#         # above check was compeltelty necessary for batched 
#         # print('debug before release, there is probably a way to map into bool array to filter edge connections')
        
#         # oob = np.isin(X, edges[d])
#         # print('checkme f', pad,np.sum(oob))

#         C = np.broadcast_to(coords[d], X.shape)
#         neighbors[d] = np.where(oob, C, X)
#         # neighbors[d] = X

#     return neighbors



# 2x as fast as the above 
def get_neighbors(coords, steps, dim, shape, edges=None, pad=0):
    """
    Get the coordinates of all neighbor pixels.
    Coordinates of pixels that are out-of-bounds get clipped.
    """    
    if edges is None:
        edges = [np.array([-1+pad, s-pad]) for s in shape]

    npix = coords[0].shape[-1]
    neighbors = np.empty((dim, len(steps), npix), dtype=np.int64)

    # Create edge masks for each dimension
    edge_masks = []
    for d in range(dim):
        mask = np.zeros(shape[d], dtype=bool)
        valid_edges = edges[d][(edges[d] >= 0) & (edges[d] < shape[d])]
        mask[valid_edges] = True
        edge_masks.append(mask)

    for d in range(dim):
        S = steps[:, d].reshape(-1, 1)
        X = coords[d] + S

        # Ensure that both X and X + S do not exceed the bounds
        X_clipped = np.clip(X, 0, shape[d] - 1)
        X_shifted_clipped = np.clip(X + S, 0, shape[d] - 1)

        # Use the edge mask to determine out-of-bounds coordinates
        current_mask = edge_masks[d]
        oob = np.logical_and(current_mask[X_clipped], ~current_mask[X_shifted_clipped])

        C = np.broadcast_to(coords[d], X.shape)
        neighbors[d] = np.where(oob, C, X_clipped)

    return neighbors
    
# a tiny bit faster than the above
def get_neighbors(coords, steps, dim, shape, edges=None, pad=0):
    """
    Get the neighbor coordinates for each pixel in `coords` for each offset in `steps`.
    Out-of-bounds neighbors get clipped or replaced with original coords (depending on `edges`).
    """
    if edges is None:
        edges = [np.array([-1+pad, s-pad]) for s in shape]

    npix = coords[0].shape[-1]
    nsteps = len(steps)  # e.g. 8 (2D) or 26 (3D)
    
    # neighbors.shape = (dim, nsteps, npix)
    neighbors = np.empty((dim, nsteps, npix), dtype=np.int64)
    
    # Precompute edge_masks for each dimension
    edge_masks = []
    for d in range(dim):
        mask = np.zeros(shape[d], dtype=bool)
        valid_edges = edges[d][(edges[d] >= 0) & (edges[d] < shape[d])]
        mask[valid_edges] = True
        edge_masks.append(mask)
    
    # For each dimension d, process each step offset one by one
    for d in range(dim):
        current_mask = edge_masks[d]
        size_d = shape[d]
        
        for n, step_d in enumerate(steps[:, d]):
            # X is just 1D, shape: (npix,)
            X = coords[d] + step_d
            
            # clip in-place (avoid creating a second large array)
            # You can do: np.clip(X, 0, size_d - 1, out=X), but that modifies coords[d]! 
            # so we copy first:
            Xc = X.copy()
            np.clip(Xc, 0, size_d - 1, out=Xc)
            
            # shift also clipped in place, if you need it:
            Xs = X + step_d
            np.clip(Xs, 0, size_d - 1, out=Xs)
            
            # Out-of-bounds condition: 
            # "oob if current_mask[Xc] == True and current_mask[Xs] == False"
            # We'll do it only where Xc is within [0, size_d -1].
            # NB: Xc is an array of indices, we can check current_mask at those indices:
            oob = np.logical_and(current_mask[Xc], ~current_mask[Xs])
            
            # Now pick either coords[d] or the clipped coordinate.
            # Instead of np.where(...), we can do in-place assignment:
            out = Xc  # start with clipped
            out[oob] = coords[d][oob]  # revert out-of-bounds neighbors

            neighbors[d, n] = out  # store in final array

    return neighbors

    
def get_neighbors_torch(input, steps):
    """This version not yet used/tested."""
    # Get dimensions
    B, D, *DIMS = input.shape
    nsteps = steps.shape[0]

    # Compute coordinates
    coordinates = torch.stack(torch.meshgrid([torch.arange(dim) for dim in DIMS]), dim=0)
    coordinates = coordinates.unsqueeze(0).expand(B, *[-1]*(D+1))  # Add batch dimension and repeat for batch

    # Compute shifted coordinates
    steps = steps.unsqueeze(-1).unsqueeze(-1).expand(nsteps, D, *DIMS).to(input.device)
    shifted_coordinates = (coordinates.unsqueeze(1) + steps.unsqueeze(0))

    # Clamp shifted_coordinates in-place
    for d in range(D):
        shifted_coordinates[:, :, d].clamp_(min=0, max=DIMS[d]-1)

    return shifted_coordinates

# this version works without padding, should ultimately replace the other one in core
# @njit
def get_neigh_inds(neighbors,coords,shape,background_reflect=False):
    """
    For L pixels and S steps, find the neighboring pixel indexes 
    0,1,...,L for each step. Background index is -1. Returns:
    
    
    Parameters
    ----------
    neighbors: ND array, int
        ndim x nsteps x npix array of neighbor coordinates
    
    coords: tuple, int
        coordinates of nonzero pixels, <ndim>x<npix>

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
    neighbors = tuple(neighbors) # just in case I pass it as ndarray

    npix = neighbors[0].shape[-1]
    indexes = np.arange(npix)
    ind_matrix = -np.ones(shape,int)
    
    ind_matrix[tuple(coords)] = indexes
    neigh_inds = ind_matrix[neighbors]
    
    # If needed, we can do a similar thing I do at boundaries and make neighbor
    # references to background redirect back to the edge pixel. However, this should 
    # not be default, since I rely on accurate neighbor indices later to test for background
    # So, probably better to do this sort of thing while contructing the affinity graph itself 
    if background_reflect:
        oob = np.nonzero(neigh_inds==-1) # 2 x nbad , pos 0 is the 0-step inds and pos 1 is the npix inds 
        neigh_inds[oob] = indexes[oob[1]] # reflect back to itself 
        ind_matrix[neighbors] = neigh_inds # update ind matrix as well

        # should I also update neighbor coordinate array? No, that's more fixed. 
        # index points to the correct coordinate. 
    
    # not sure if -1 is general enough, probbaly should be since other adjacent masks will be unlinked
    # can test it by adding some padding to the concatenation...
    
    # also, the reflections should be happening at edges of the image, but it is not? 
    
    return indexes, neigh_inds, ind_matrix


# This might need some reflection added in for it to work
# also might need generalization to include cleaned mask pixels getting dropped  
def subsample_affinity(augmented_affinity,slc,mask):
    """
    Helper function to subsample an affinity graph according to an image crop slice 
    and a foreground selection mask. 

    Parameters
    ----------
    augmented_affinity: NDarray, int64
        Stacked neighbor coordinate array and affinity graph. For dimension d, 
        augmented_affinity[:d] are the neighbor coordinates of shape (d,3**d,npix)
        and augmented_affinity[d] is the affinity graph of shape (3**d,npix). 

    slc: tuple, slice
        tuple of slices along each dimension defining the crop window
        
    mask: NDarray, bool
        foreground selection mask, in the image space of the original graph
        (i.e., not already sliced)

    Returns
    --------
    Augmented affinity graph corresponding to the cropped/masked region. 
    
    """

    # From the augmented affinity graph we can extract a lot
    nstep = augmented_affinity.shape[1]
    dim = len(slc) # dimension 
    neighbors = augmented_affinity[:dim]
    affinity_graph = augmented_affinity[dim]
    idx = nstep//2
    coords = neighbors[:,idx]
    in_bounds = np.all(np.vstack([[c<s.stop, c>=s.start] for c,s in zip(coords,slc)]),axis=0)
    in_mask = mask[tuple(coords)]>0
    
    in_mask_and_bounds = np.logical_and(in_bounds,in_mask)

    inds_crop = np.nonzero(in_mask_and_bounds)[0]
    
    # print('y',len(inds_crop),np.sum(in_mask_and_bounds), np.sum(in_bounds), np.sum(in_mask))

    if len(inds_crop):    
        crop_neighbors = neighbors[:,:,inds_crop]
        affinity_crop = affinity_graph[:,inds_crop]
    
        # shift coordinates back acording to the lower bound of the slice 
        # also refect at edges of the new bounding box
        edges = [np.array([-1,s.stop-s.start]) for s in slc]
        steps = get_steps(dim)
        
        # I should see if I can get this batched somehow... 
        for d in range(dim):        
            crop_coords = coords[d,inds_crop] - slc[d].start
            S = steps[:,d].reshape(-1, 1)
            X = crop_coords + S # cropped coordinates 
            # edgemask = np.logical_and(np.isin(X, edges[d]), ~np.isin(X+S, edges[d]))
            edgemask = np.isin(X, edges[d])
            # print('checkthisttoo')

            C = np.broadcast_to(crop_coords, X.shape)
            crop_neighbors[d] = np.where(edgemask, C, X)

        #return augmented affinity 
        return np.vstack((crop_neighbors,affinity_crop[np.newaxis]))
    else:
        e = np.empty((dim+1,nstep,0),dtype=augmented_affinity.dtype)
        return e, []

@functools.lru_cache(maxsize=None) 
def get_steps(dim):
    """
    Get a symmetrical list of all 3**N points in a hypercube represented
    by a list of all possible sequences of -1, 0, and 1 in ND.
    
    1D: [[-1],[0],[1]]
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
    
    # a new function I learned about, np.ndindex, could be used here instead, 
    # np.stack([s for s in np.ndindex(*(3,) * ndim)])-1, 
    # but it runs in microseconds rather than nanoseconds... 
    return steps

# @functools.lru_cache(maxsize=None)
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
@functools.lru_cache(maxsize=None) 
def kernel_setup(dim):
    """
    Get relevant kernel information for the hypercube of interest. 
    Calls get_steps(), steps_to_indices(). 
    
    Parameters
    ----------

    dim: int
        dimension (usually 2 or 3, but can be any positive integer)
    
    Returns
    -------
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


from collections import defaultdict
def get_supporting_inds(steps):
    """
    For each step 'v', find all pairs (i, j) such that steps[i] + steps[j] == steps[v],
    excluding the center index.

    Steps shape: (S, d), with a 'center_index' = S//2 by default.
    """
    steps = np.array(steps, copy=False)
    S, d = steps.shape
    center_index = S // 2
    
    # Create a mask that excludes the center
    mask = np.arange(S) != center_index
    # Steps without the center
    steps_nocenter = steps[mask]  # shape: (S-1, d)
    orig_indices = np.nonzero(mask)[0]  # original indices in [0..S-1], skipping center
    
    N = S - 1  # number of non-center steps
    
    # Pairwise sums: shape (N, N, d)
    pair_sums = steps_nocenter[:, None, :] + steps_nocenter[None, :, :]
    # Flatten to (N*N, d)
    pair_sums_2d = pair_sums.reshape(-1, d)

    # We'll keep track of which (i,j) generated each sum
    # i_list, j_list are each of length N*N
    i_list = np.repeat(orig_indices, N)
    j_list = np.tile(orig_indices, N)

    # Build a dictionary: sum_map[ tuple_of_coords ] -> list of (i,j)
    sum_map = defaultdict(list)
    for k in range(N * N):
        key = tuple(pair_sums_2d[k])
        sum_map[key].append((i_list[k], j_list[k]))

    # Now for each v != center, look up tuple(steps[v]) in sum_map
    pairs = {}
    for v in range(S):
        if v == center_index:
            continue
        key = tuple(steps[v])
        pairs[v] = sum_map.get(key, [])

    return pairs
