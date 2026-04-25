from .imports import *



def precompute_valid_mask(shape, steps, device=None):
    """Boolean mask for valid pixel-neighbor pairs within an N-D volume.

    Parameters
    ----------
    shape : tuple of int
        Spatial dimensions, e.g. ``(Y, X)`` or ``(Z, Y, X)``.
    steps : list of tuple of int
        Offset table returned by ``kernel_setup(dim)``.
    device : torch.device or None
        Device for the returned tensor.

    Returns
    -------
    valid : torch.BoolTensor
        Shape ``(K, 1, *shape)``.
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
