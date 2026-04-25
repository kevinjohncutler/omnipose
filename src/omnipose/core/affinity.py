from __future__ import annotations
from .imports import *

import logging
from typing import Sequence
from numba import njit

from .fields import divergence_torch, _pad_and_stack_neighbors, _make_seam_mask
from .njit import candidate_cleanup_idx


from .imports import _cc_union_find

omnipose_logger = logging.getLogger(__name__)



def compute_affinity_gpu(clabels: torch.Tensor, steps: np.ndarray,
                          seam_ends: Sequence[int] = (),
                          seam_starts: Sequence[int] = (),
                          links=None) -> torch.Tensor:
    """GPU replacement for masks_to_affinity + get_neighbors.

    Computes the ``(nsteps, *spatial)`` affinity mask entirely on-device using
    clamped torch.roll.  No CPU round-trips, no neigh_inds array.

    Boundary semantics (matching the CPU get_neighbors clamp):

    - At true grid boundaries the neighbour coordinate is clamped to the
      boundary pixel itself, so ``clabels == rolled_clabels`` is True for
      foreground, giving affinity True (identical to CPU ``is_edge`` / Neumann BC).
    - At stitching seams (rows seam_ends / seam_starts) the same clamping is
      applied so that seam pixels report affinity True (connecting to self),
      matching the CPU behaviour where ``edges`` causes is_edge=True at seams.
      Without this fix the roll crosses into the adjacent tile, finds a
      different label, and returns affinity False.

    Args:
        clabels:     ``(*spatial)`` int tensor on the target device (0 = background).
        steps:       ``(nsteps, dim)`` NumPy array of neighbour offsets.
        seam_ends:   row indices (spatial dim 0) that are the last row of a
                     tile — clamp when ``step[0] > 0``.
        seam_starts: row indices that are the first row of the next tile —
                     clamp when ``step[0] < 0``.
        links:       iterable of (a, b) label pairs to treat as connected
                     (e.g. self-contact labels). None or empty = no links.

    Returns:
        affinity: ``(nsteps, *spatial)`` bool tensor on same device as clabels.
    """
    nsteps = len(steps)
    idx_center = nsteps // 2          # the (0, …, 0) step
    foreground = clabels > 0
    device = clabels.device

    # Vectorized: extract all nsteps neighbour label maps at once.
    steps_list = [list(s) for s in steps]
    rolled_cl = _pad_and_stack_neighbors(
        clabels.float(), steps_list,
    ).long()                                          # (nsteps, *spatial)

    # Seam correction: at tile-boundary rows, steps crossing the seam would
    # read the adjacent tile's labels.  Replace with self-label so affinity
    # reports True (= connected to self) at seam rows, matching CPU is_edge BC.
    if seam_ends or seam_starts:
        seam_mask = _make_seam_mask(nsteps, clabels.shape, steps_list,
                                    seam_ends, seam_starts, device)
        rolled_cl = torch.where(seam_mask, clabels.long().unsqueeze(0), rolled_cl)

    affinity = (rolled_cl == clabels.unsqueeze(0)) & foreground.unsqueeze(0)

    # Link-based affinities: pixels of linked label pairs are also connected.
    # Build a (max_label+1, max_label+1) bool matrix on-device, then use
    # vectorized 2D fancy-indexing across all steps in one call.
    if links:
        max_label = int(clabels.max().item())   # one GPU-CPU sync, only when links exist
        link_matrix = torch.zeros(max_label + 1, max_label + 1,
                                  dtype=torch.bool, device=device)
        # Vectorized link_matrix fill: convert the set of (a,b) pairs to a
        # (L,2) int tensor and use advanced indexing — avoids a slow Python loop.
        links_arr = torch.tensor(list(links), dtype=torch.long, device=device)  # (L, 2)
        valid = (links_arr[:, 0] <= max_label) & (links_arr[:, 1] <= max_label)
        a_idx = links_arr[valid, 0]
        b_idx = links_arr[valid, 1]
        link_matrix[a_idx, b_idx] = True
        link_matrix[b_idx, a_idx] = True
        # rolled_cl: (nsteps, *spatial); clabels: (*spatial)
        # Expand clabels to match rolled_cl shape for paired indexing.
        cl_exp = clabels.unsqueeze(0).expand_as(rolled_cl)      # (nsteps, *spatial)
        linked = link_matrix[cl_exp.reshape(nsteps, -1),
                             rolled_cl.reshape(nsteps, -1)      # (nsteps, npix)
                            ].view(nsteps, *clabels.shape)
        affinity |= linked & foreground.unsqueeze(0)

    affinity[idx_center] = False                      # center step: no self-connection
    return affinity


def affinity_to_boundary_gpu(affinity: torch.Tensor,
                              foreground: torch.Tensor) -> torch.Tensor:
    """GPU version of affinity_to_boundary.

    A pixel is a boundary pixel if it has at least one active connection and
    fewer than ``nsteps - 1`` active connections (not fully internal).

    Args:
        affinity:   ``(nsteps, *spatial)`` bool tensor.
        foreground: ``(*spatial)`` bool tensor, True where clabels > 0.

    Returns:
        boundary: ``(*spatial)`` bool tensor on same device.
    """
    nsteps = affinity.shape[0]
    csum = affinity.sum(0)
    return (csum < (nsteps - 1)) & (csum > 0) & foreground



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
                        niter, euler_offset=None,
                        device=torch_GPU,
                        angle_cutoff=np.pi/3):
    initial, final, flow, dist, iscell = ensure_torch(initial, final, flow, dist, iscell, device=device)

    mu = final - initial
    B, D, *DIMS = mu.shape
    S = len(steps)
    spatial_dims = tuple(range(-D, 0))

    div = divergence_torch(flow)
    mag = torch_norm(mu, dim=1, keepdim=True)
    mu_norm = torch.where(mag > 0, mu / mag, mu)

    sink = div < 0  # (B, *DIMS)
    valid_mask = utils.precompute_valid_mask(DIMS, steps, device=device)  # (S, 1, *DIMS)
    cutoff = 3 ** (D - 1)

    # Phase 1: pairwise cosines + sink via torch.roll -- works for any D
    cos = torch.stack([
        (mu_norm * torch.roll(mu_norm,
                              shifts=tuple(-int(s[j]) for j in range(D)),
                              dims=spatial_dims)).sum(dim=1)
        for s in steps])  # (S, B, *DIMS)
    is_sink = torch.stack([
        sink | torch.roll(sink,
                          shifts=tuple(-int(s[j]) for j in range(D)),
                          dims=spatial_dims)
        for s in steps])  # (S, B, *DIMS)

    # Fix OOB: cos -> self-dot, is_sink -> False
    self_cos = (mu_norm * mu_norm).sum(dim=1)  # (B, *DIMS)
    cos = torch.where(valid_mask, cos, self_cos.unsqueeze(0))
    is_sink = is_sink & valid_mask

    connectivity = (cos >= np.cos(angle_cutoff)) | is_sink
    connectivity[S // 2] = False
    csum = connectivity.sum(0)  # (B, *DIMS)
    keep = csum >= cutoff

    # Phases 4-6: support filtering + interior connectivity restoration
    # iscell_bool, keep_bool, conn_sq keep the full batch dim (B, *DIMS) / (S, B, *DIMS)
    iscell_bool = iscell.bool()   # (B, *DIMS)
    keep_bool   = keep            # (B, *DIMS)

    pairs_per_dir = []
    for i in range(S // 2):
        tuples = supporting_inds[i]
        all_f, all_b = [], []
        for j in range(len(tuples)):
            f_inds = tuples[j]
            b_inds = (S - 1 - np.array(tuples[-(j + 1)])).tolist()
            for f, b in zip(f_inds, b_inds):
                all_f.append(f)
                all_b.append(b)
        pairs_per_dir.append((all_f, all_b))

    for i in range(S // 2):
        shifts_neg = tuple(-int(steps[i][j]) for j in range(D))
        shifts_fwd = tuple( int(steps[i][j]) for j in range(D))
        vm_i   = valid_mask[i,      0]   # (*DIMS) — broadcasts over B
        vm_opp = valid_mask[-(i+1), 0]
        all_f, all_b = pairs_per_dir[i]
        conn_sq = connectivity  # (S, B, *DIMS) view — no [0] batch drop

        # Phase 4: vectorized support count — (len, B, *DIMS) after indexing
        cf = conn_sq[all_f]
        cb = torch.roll(connectivity[all_b], shifts=shifts_neg, dims=spatial_dims)
        vf = valid_mask[all_f]  # (len, 1, *DIMS) — broadcasts over B
        support = (cf & cb & vf).sum(0, dtype=torch.int32)  # (B, *DIMS)

        # Phase 5: interior connectivity restoration
        # Nearly fully connected: all neighbors minus one (generalizes 2D's >= 7)
        restore = (csum >= S - 2) & vm_i   # (B, *DIMS)
        conn_sq[i] = conn_sq[i] | restore
        conn_sq[-(i+1)] = conn_sq[-(i+1)] | (torch.roll(restore, shifts=shifts_fwd, dims=spatial_dims) & vm_opp)

        # Phase 6: final mask
        opp_at_tgt    = torch.roll(conn_sq[-(i+1)], shifts=shifts_neg, dims=spatial_dims)
        iscell_at_tgt = torch.roll(iscell_bool,      shifts=shifts_neg, dims=spatial_dims)
        keep_at_tgt   = torch.roll(keep_bool,        shifts=shifts_neg, dims=spatial_dims)
        _c = (conn_sq[i] & opp_at_tgt & iscell_bool & iscell_at_tgt
              & keep_bool & keep_at_tgt & (support > 2))
        conn_sq[i]      = torch.where(vm_i,   _c,
                                      conn_sq[i])
        conn_sq[-(i+1)] = torch.where(vm_opp, torch.roll(_c, shifts=shifts_fwd, dims=spatial_dims),
                                      conn_sq[-(i+1)])

    return connectivity



# numba will require getting rid of stacking, summation, etc., super annoying... the number of pixels to fix is quite
# small in practice, so may not be worth it 
# @njit('(bool_[:,:], int64[:,:], int64[:], int64[:], int64[:],  int64[:], int64, bool_)')

def affinity_to_edges(affinity_graph, neigh_inds, step_inds, px_inds):
    """Convert symmetric affinity graph to edge list (vectorized)."""
    rows_list = []
    cols_list = []
    for s in step_inds:
        ni = neigh_inds[s][px_inds]
        valid = (px_inds <= ni) & affinity_graph[s, px_inds].astype(bool)
        rows_list.append(px_inds[valid])
        cols_list.append(ni[valid])
    if not rows_list:
        return np.empty((0, 2), dtype=np.int64)
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    return np.stack([rows, cols], axis=1)


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
    # Connected components via numba union-find
    rows = edge_list[:, 0]
    cols = edge_list[:, 1]
    raw_labels = _cc_union_find(rows, cols, npix)
    # Zero out singletons (isolated foreground pixels with no edges)
    has_edge = np.zeros(npix, dtype=bool)
    has_edge[rows] = True
    has_edge[cols] = True
    comp_id = np.where(has_edge, raw_labels, 0).astype(np.int32)

    labels = np.zeros(iscell.shape, dtype=int)
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


