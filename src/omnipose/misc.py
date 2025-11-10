from .core import *
from .utils import *
import subprocess

# a bunch of development functions 

from scipy.interpolate import splprep, splev

import cv2 # still used here for finding contours...
import edt
import torch
import fastremap

from ncolor import unique_nonzero # for backward compatibility, used to have a copy here 

import numpy as np
from typing import Sequence

def nd_grid_hypercube_labels(shape: Sequence[int],
                             side: int,
                             *,
                             center: bool = True,
                             dtype=np.int32) -> np.ndarray:
    """
    Label an ND array with equal-side hypercubes of edge length `side` pixels.

    Parameters
    ----------
    shape : sequence of int
        Target array shape (H, W, D, ...).
    side : int
        Edge length of each hypercube in pixels (same along all axes).
    center : bool, default True
        Center the grid inside the array; leftover margins get label 0.
    dtype : numpy dtype, default np.int32
        Output dtype.

    Returns
    -------
    labels : ndarray
        Integer label map of shape `shape` with values in {0, 1..K}.
    """
    shape = np.asarray(shape, dtype=int)
    if shape.ndim != 1:
        raise ValueError("shape must be 1D sequence of ints")
    if not isinstance(side, (int, np.integer)) or side < 1:
        raise ValueError("side must be a positive integer")

    counts = shape // side
    if np.any(counts <= 0):
        raise ValueError("side too large for at least one axis")

    grid_span = counts * side
    offsets = ((shape - grid_span) // 2) if center else np.zeros_like(shape)

    # Build per-axis indices and in-bounds mask
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    idx_axes = []
    in_bounds = np.ones(tuple(shape), dtype=bool)
    for ax, g in enumerate(grids):
        rel = g - offsets[ax]
        mask = (rel >= 0) & (rel < grid_span[ax])
        in_bounds &= mask
        idx_axes.append((rel // side).astype(int))

    # Row-major linearization
    lin = np.zeros(tuple(shape), dtype=int)
    stride = 1
    for ax in range(shape.size - 1, -1, -1):
        lin += idx_axes[ax] * stride
        stride *= counts[ax]

    labels = np.zeros(tuple(shape), dtype=dtype)
    labels[in_bounds] = lin[in_bounds] + 1
    return labels

def make_label_matrix(N: int, M: int) -> np.ndarray:
    """
    General ND label matrix.
    
    Shape = (2*M,)*N
    Each axis is split into two halves of length M.
    The label is the binary code of the half-indices.
    
    Example:
      N=1 → [0...0,1...1]
      N=2 → quadrants labeled 0..3
      N=3 → octants labeled 0..7
      N=4 → 16 hyper-quadrants labeled 0..15
    """
    if N < 1:
        raise ValueError("N must be >=1")
    # build index grids
    grids = np.ogrid[tuple(slice(0,2*M) for _ in range(N))]
    labels = np.zeros((2*M,)*N, dtype=int)
    for axis, g in enumerate(grids):
        half = (g // M).astype(int)     # 0 or 1
        labels += half << axis          # bit-shift
    return labels

def vector_to_arrow(vectors, flip_y: bool = False):
    """
    Convert one or many 2-D vectors (in y,x order) into Unicode arrow glyphs.

    Parameters
    ----------
    vectors : array-like
        Either a single (dy, dx) pair or an iterable of such pairs.
    flip_y : bool, default False
        If True, invert the y-sign (dy) before mapping.

    Returns
    -------
    str | list[str]
        A single glyph when a single vector is provided,
        otherwise a list of glyphs.
    """
    arr = np.asarray(vectors, dtype=float)

    # Ensure shape (N, 2)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError("Expected a single 2-vector.")
        arr = arr.reshape(1, 2)
    elif arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Each vector must have exactly two components.")

    # Extract signed directions
    dy = np.sign(arr[:, 0]).astype(int)
    dx = np.sign(arr[:, 1]).astype(int)

    # Optionally flip only the y sign
    if flip_y:
        dy = -dy

    signs = list(zip(dy, dx))

    arrow_map = {
        ( 0,  0): '•',
        ( 0,  1): '→',
        ( 0, -1): '←',
        (-1,  0): '↓',
        ( 1,  0): '↑',
        (-1,  1): '↘',
        ( 1,  1): '↗',
        (-1, -1): '↙',
        ( 1, -1): '↖',
    }

    glyphs = [arrow_map.get(sig, '?') for sig in signs]
    return glyphs[0] if len(glyphs) == 1 else glyphs



def random_int(N, M=None, seed=None):

    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
        print(f'Seed: {seed}')
    else:
        np.random.seed(seed)
    # Generate a random integer between 0 and N-1
    return np.random.randint(0, N, M)

def meshgrid(shape):
    """
    Generate a tuple of coordinate grids for an ND array of a given shape.
    
    Parameters:
        shape (tuple): Shape of the ND array (e.g., (Y, X) for 2D, (Z, Y, X) for 3D, etc.).
    
    Returns:
        tuple: A tuple of N coordinate arrays, one per dimension.
    """
    ranges = [np.arange(dim) for dim in shape]  # Create a range for each dimension
    coords = np.meshgrid(*ranges, indexing='ij')  # Generate coordinate arrays
    return coords  # Returns a tuple of N arrays
    
def generate_flat_coordinates(shape):
    """
    Generate flat coordinate arrays for an ND array.
    
    Parameters:
        shape (tuple): Shape of the array (e.g., (Y, X) for 2D).
        
    Returns:
        tuple: A tuple of flat arrays representing the coordinates.
    """
    grids = meshgrid(shape)  # Generate the meshgrid
    return tuple(grid.ravel() for grid in grids)  # Flatten each grid
    
    
# def argmin_cdist(X, labels):
#     # Get unique labels and determine the boundaries where each label starts and ends
#     unique_labels, label_starts = torch.unique_consecutive(labels, return_inverse=False, return_counts=True)
#     label_ends = torch.cumsum(label_starts, dim=0)
#     label_starts = label_ends - label_starts

#     # Prepare a list to store the indices of the minimum summed distances
#     argmin_indices = []
#     summed_distances_all = torch.full((len(X),), float('nan'), device=X.device)
    
#     for i, label in enumerate(unique_labels):
#         # Use the precomputed start and end indices
#         start_idx = label_starts[i]
#         end_idx = label_ends[i]
#         label_indices = torch.arange(start_idx, end_idx, device=X.device)
        
#         # Extract the relevant coordinates
#         X_label = X[label_indices]
        
#         if X_label.ndim > 1:  # Only compute if there is more than one point
#             # Compute pairwise distances using cdist
#             distances = torch.cdist(X_label, X_label)
            
#             # Sum the distances for each point within the block
#             summed_distances = torch.nansum(distances, dim=1)
            
#             # Store the summed distances for all entries
#             summed_distances_all[label_indices] = summed_distances
            
#             # Find the index of the minimum summed distance within the block
#             argmin_index_in_block = torch.argmin(summed_distances)
            
#             # Map this index back to the original index in the contact map
#             argmin_indices.append(label_indices[argmin_index_in_block])
    
#     return torch.tensor(argmin_indices, device=X.device), summed_distances_all

import torch
import numpy as np
import edt  # Ensure this package is installed: pip install edt

def argmin_cdist(X, labels, distance_values):
    # Get unique labels and counts
    unique_labels, label_counts = torch.unique_consecutive(labels, return_counts=True)
    label_starts = torch.cumsum(torch.cat([torch.tensor([0], device=labels.device), label_counts[:-1]]), dim=0)
    label_ends = torch.cumsum(label_counts, dim=0)
    
    # Prepare a list to store the indices of the minimum adjusted summed distances
    argmin_indices = []
    adjusted_summed_distances_all = torch.full((len(X),), float('nan'), device=X.device)
    
    for i in range(len(unique_labels)):
        # Use the precomputed start and end indices
        start_idx = label_starts[i]
        end_idx = label_ends[i]
        label_indices = torch.arange(start_idx, end_idx, device=X.device)
        
        # Extract the relevant coordinates and distance values
        X_label = X[label_indices]
        distance_values_label = distance_values[label_indices]
        
        if X_label.shape[0] > 1:  # Only compute if there is more than one point
            # Compute pairwise distances using cdist
            distances = torch.cdist(X_label, X_label)
            
            # Sum the distances for each point within the label
            summed_distances = torch.sum(distances, dim=1)
            
            # Adjust summed distances by subtracting the distance field values
            adjusted_summed_distances = summed_distances/(distance_values_label**0.05) # the most centered 
            # adjusted_summed_distances = summed_distances/(1+distance_values_label*summed_distances) # nice and centered, but dominated by distance values
            # adjusted_summed_distances = summed_distances/distance_values_label # similar to above
            # adjusted_summed_distances = 1/distance_values_label*summed_distances#/(1+distance_values_label/summed_distances) # nice and centered
            
            adjusted_summed_distances = summed_distances/torch.sqrt(distance_values_label)
            # adjusted_summed_distances = (summed_distances/distance_values_label)**0.5
            
            adjusted_summed_distances = summed_distances*(1+1/distance_values_label)
            
       
            
            # Store the adjusted summed distances for all entries
            adjusted_summed_distances_all[label_indices] = adjusted_summed_distances
                
            # Find the index of the minimum adjusted summed distance within the label
            argmin_index_in_label = torch.argmin(adjusted_summed_distances)
                
            # Map this index back to the original index in X
            argmin_indices.append(label_indices[argmin_index_in_label])
        else:
            # If there's only one point, it's the medoid by default
            argmin_indices.append(label_indices[0])
            adjusted_summed_distances_all[label_indices] = 0  # Or the distance value itself
    
    return torch.tensor(argmin_indices, device=X.device), adjusted_summed_distances_all

def get_medoids(labels,do_skel=True,return_dists=False):
    """Get medoid coordinates and labels from label mask.
    """
    # TODO: see if this can be sped up

    if do_skel:
        masks = skeletonize(labels)
        dists = np.ones_like(labels)        
    else:
        masks = labels
        dists = edt.edt(labels)#,black_border=False)
    
    coords = np.argwhere(masks>0)
    slc = tuple(coords.T)
    labels = masks[slc]
    sort = np.argsort(labels)
    sort_coords = coords[sort]
    sort_labels = labels[sort]
    sort_dists = dists[slc][sort]
    
    # get torch Tensors for distance-based indexing
    inds_tensor, dists_tensor = argmin_cdist(
        torch.tensor(sort_coords).float(),
        torch.tensor(sort_labels).float(),
        torch.tensor(sort_dists).float()
    )

    # Convert to NumPy arrays
    inds = inds_tensor.cpu().numpy()
    dists_arr = dists_tensor.cpu().numpy()

    # Ensure inds is at least 1D
    if len(inds):
        inds = np.atleast_1d(inds)
        
        
        medoids = sort_coords[inds]
        mlabels = sort_labels[inds]
        
        if medoids.ndim==1:
            medoids = medoids[None]
            
        
        if return_dists:
            inner_dists = np.zeros(masks.shape,dtype=dists_arr.dtype)
            inner_dists[tuple(sort_coords.T)] = dists_arr
            return medoids, mlabels, inner_dists
        else:
            return medoids, mlabels
    else:
        return None, None



def project_to_skeletons(images,labels,augmented_affinity, device, interp, 
                         use_gpu, omni, reference, interp_skel=0, n_step=None,log=True):

    shape = labels.shape
    d = labels.ndim
    neighbors = augmented_affinity[:d]
    affinity_graph = augmented_affinity[d] #.astype(bool) VERY important to cast to bool, now done internally 
    idx = affinity_graph.shape[0]//2
    coords = neighbors[:,idx]
    
    # need ind_matrix to select
    npix = neighbors.shape[-1]
    indexes = np.arange(npix)
    ind_matrix = -np.ones(shape,int)
    ind_matrix[tuple(coords)] = indexes

    # T = masks_to_flows_torch(labels, 
    #                         affinity_graph=affinity_graph, 
    #                         coords=tuple(coords), 
    #                         device=device,
    #                         return_flows=False)[0]

    T, mu = masks_to_flows_torch(labels, 
                            affinity_graph=affinity_graph, 
                            coords=tuple(coords), 
                            device=device)
    
    dt = T.cpu().numpy()
    niter = int(diameters(labels,dt))
    inds = np.array(coords).astype(np.int32)
    p, inds, _ = follow_flows(mu, dt, inds, niter=niter, interp=interp,
                                use_gpu=use_gpu, device=device, omni=omni,
                                suppress=1, calc_trace=0, verbose=0)


    initial = inds
    # final = np.round(p[(Ellipsis,)+tuple(inds)]).astype(int)
    final = p[(Ellipsis,)+tuple(inds)]
        

    # get the skeletons
    # inner = dt>2
    # skel = skeletonize(inner, method='lee')
    skel = skeletonize(labels, dt_thresh=2, dt=dt)
    

    # label skeletons and compute affinity for parametrization
    skel_labels = (skel>0)*labels    
    skel_coords = np.nonzero(skel_labels)
    dim = skel.ndim
    steps, inds, idx, fact, sign = kernel_setup(dim)
    skel_affinity = masks_to_affinity(skel_labels, skel_coords, 
                                      steps, inds, idx, fact, sign, dim)

    # print('00 ',skel_labels.shape,skel_affinity.shape)
    
    # parametrize the skeletons 
    contour_map, contour_list, unique_L = get_contour(skel_labels,
                                                      skel_affinity,
                                                      cardinal_only=0)

    
    # generate mapping from pixel clusters to skeleton 
    # doing per-cell is probably combinatorially faster than all clusters to all skels
    # in which case, clustering step not needed 
    
    images += [dt] # add distance field as a channel
    nchan = len(images)

    N = []
    for c in contour_list:
        if log and n_step is not None:
            n = int(np.ceil(np.log(np.count_nonzero(len(c)))*n_step/np.log(n_step)))+1
        else:
            n = len(c) if n_step is None else n_step
        N.append(n)
    
    projections = [np.zeros((nchan,n)) for c,n in zip(contour_list,N)] # 


    print('c0',len(contour_list),unique_L)
    for contour, L, proj in zip(contour_list, unique_L, projections):
        N = proj.shape[-1]
        # target = np.nonzero(skel_labels==L)
        target = np.array([c[contour] for c in skel_coords])

        # alt: make intermp
        # pts = np.stack([c[contour] for c in skel_coords]).T #<<< using skel coords here
        if interp_skel:
            pts = target.T
            tck, u = splprep(pts.T, u=None, s=len(pts)/6, per=0) # per=1 would be cyclic, not the case here 
            u_new = np.linspace(u.min(), u.max(), N)
            new_pts = splev(u_new, tck, der=0)
            target = np.stack(new_pts)
    
        
        # fix orientation by tracking a pole... 
        # this could break with fast pole movement
        start = target[:,0]
        stop = target[:,-1]
        if reference is not None:
            if np.sum((start-reference)**2,axis=0) > np.sum((stop-reference)**2,axis=0):
                target = target[:,::-1]
                start = stop
        
        # mask_coords = np.nonzero(np.logical_and(labels==L,inner))
        # mask_coords = np.nonzero(labels==L)
        # source_inds = ind_matrix[mask_coords]
        source_inds = np.nonzero(labels[tuple(coords)] == L)[0]
        mask_coords = coords[:,source_inds]

        
        print('AA',np.any(source_inds<0),mask_coords.shape,coords.shape, 
              source_inds.shape,labels.shape)
        
        source = tuple(final[:,source_inds])        
        mapping = project_points(source,target)

        
        # print(source.shape,mapping.shape)
        print('cc',target[0].shape,N)
        
        for c in range(nchan):
            projection = np.zeros(N)
            data = images[c][tuple(mask_coords)]
            print('bb',projection.shape,mapping.shape,data.shape)
            np.add.at(projection,mapping,data)
            counts = np.bincount(mapping, minlength=N)
            projection = safe_divide(projection,counts)
            proj[c] = projection
            # print(np.min(counts),np.max(counts),N)
            # print(np.median(counts))
            # proj[c] = counts[contour]
    
            
    return projections, contour_map, contour_list, skel_coords, start
    

from sklearn.neighbors import NearestNeighbors
def project_points(source, target):
    target = np.array(target)
    source = np.array(source)
    
    source_count = source.shape[1]
    target_count = target.shape[1]
    result = np.empty(source_count, dtype=np.int64)
    
    # Create a k-d tree from the target points
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target.T)
    
    # Query the k-d tree for the closest target point for each source point
    _, indices = nbrs.kneighbors(source.T)
    
    # Assign the indices of the closest target points to the result array
    result = np.squeeze(indices)
    
    return result


def ncolor_contour(contour_map,contour_list,pad=1):
    
    contour_ncolor = np.zeros(np.array(contour_map.shape)+2*pad,np.uint32)
    for contour in contour_list:
    # for contour in [contour_list[0]]:
        ll,mapping = fastremap.renumber(np.array(contour))
        lab = np.zeros(np.array(contour_map.shape)+2*pad,np.uint32)
        lab_ncolor = lab.copy() # preallocate ncolor array
        coords_t = np.unravel_index(contour,np.pad(contour_map,pad).shape)
        lab[coords_t] = ll # this is actually equivalent to contour_map already, optimize later
        # adjacent points (1 step), diagonal points (2 step) and endpoints

         # all pairs 1 apart, includes ll[-1],ll[0] but not ll[0],ll[-1]
        contour_connect = [(ll[i],ll[np.mod(i+1,len(ll))]) for i in range(0,len(ll))]

         # all pairs 2 apart, includes ll[-1],l[1] but not (ll[1],ll[-1]), (ll[-2,],ll[0]), (ll[0],ll[-2])
        contour_connect += [(ll[i-1],ll[np.mod(i+1,len(ll))]) for i in range(0,len(ll))]

        # fill in missing endpoint connections
        contour_connect += [(ll[0],ll[-1]),(ll[1],ll[-1]),(ll[-2],ll[0]),(ll[0],ll[-2])]

        label_connect = ncolor.connect(lab,conn=2)
        A = set([tuple(m) for m in label_connect])
        B = set(contour_connect)
        C = A-B # set of all nontrivial connections
        # D = SymDict(C)
        D = dict([c for c in C])
        D2 = dict([c[::-1] for c in C])
        D.update(D2)
        # print(B)
        self_connected = list(D.values())
        current_label = 1

        coords_t = np.array(coords_t).T

        for t,l in enumerate(ll):
            coord = coords_t[t]
            if l in self_connected:
                cc = coords_t[D[l]-1] # get coordinate of self-contact pix
                vc = lab_ncolor[tuple(cc)] # value of self-contact pix

                # when the previous pixel in contour has the same number as
                # the self-contact contour, then we need to choose a new color 

                if vc==current_label: #nonzero means we have seen it before
                    current_label+=1
            lab_ncolor[tuple(coord)] = current_label
        lab_ncolor[lab_ncolor>0] += np.max(contour_ncolor)
        contour_ncolor += lab_ncolor  
    
    unpad = tuple([slice(pad,-pad)]*lab.ndim)
    return contour_ncolor[unpad]


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
    
    
from skimage import filters
from skimage.feature import peak_local_max, corner_peaks
from omnipose.utils import rescale
from scipy.ndimage import center_of_mass, binary_erosion, binary_dilation
from skimage import measure

def overseg_seeds(msk, bd, mu, T, ks=1.5, 
                  rskel=True,extra_peaks=None):
    from skimage.morphology import skeletonize, medial_axis

    skel = skeletonize(np.logical_xor(msk,bd))
    
    # div = divergence(mu.unsqueeze(0)).squeeze().cpu().numpy()
    div = divergence(mu) #from core
    # cf = utils.curve_filter(div,2.5)
    # cf = utils.curve_filter(skel*1.,2.5)
    # imgin = gaussian(bd-(msk>0)*1.,3)
    # imgin = skel*1.
    # imgin = gaussian(skel*1.,1) # potential best 
    
    # imgin = bd-(msk>0)*1.
    # cf = utils.curve_filter(imgin,2.5)
    
    imgin = T # no I think this is the best... 
    # imgin = div
    # ks = 2
    
    cf = utils.curve_filter(imgin,ks)
    
    # if rskel:
    if 1:
        image1 = np.abs(cf[-1]) #xy second derivative 
        
        cf = utils.curve_filter(image1,ks)
        # image = np.abs(cf[5])
        image = cf[5]
    else:
        # image = cf[2]
        # image = np.abs(div)
        cfx = utils.curve_filter(mu[1],ks)
        cfy = utils.curve_filter(mu[0],ks)
        image1 = cfx[-3]+cfy[-2]

        # image1 = np.abs(cf[-1]) #xy second derivative 
        
        cf = utils.curve_filter(image1,ks)
        # image = np.abs(cf[5])
        image = cf[5]
        
        

    image = utils.rescale(image)
    # skel = binary_erosion(np.logical_xor(msk,bd),iterations=1)
    # skel = binary_dilation(skel,iterations=1)
    
    if rskel:
        restriction=skel
    else:
        restriction = np.logical_xor(msk,bd)
        # restriction = image>.1
    # restriction=r1
    
    min_dist = 2
    # peaks = corner_peaks((1-utils.rescale(image))*restriction,min_distance=min_dist)#,footprint=np.ones((3, 3)))

    peaks = corner_peaks((image)*restriction,min_distance=min_dist)#,footprint=np.ones((3, 3)))
    
    is_peak = np.zeros(image.shape,dtype=bool)
    is_peak[tuple(peaks.T)] = True
    
    if extra_peaks is not None: # add in more 
        is_peak = np.logical_or(is_peak,extra_peaks)
    
    labels = measure.label(is_peak,connectivity=2)
    merged_peaks = center_of_mass(is_peak, labels, range(1, np.max(labels)+1))
    peaks = np.array(merged_peaks).astype(int)
    
    return peaks, image

def turn_overseg(maski,bdi):
    """
    This function works by detecting turns in boundary labels. First, the boundary
    is parametrized. Then, changes in boundary label are detected. For ND compatibility,
    this should be replaced with a version that detects these turns while rejecting other
    points of self-contact (where the boundary label is different) by another metric. 
    In particular, the flow should be more or less parallel at these turn points, at least
    not antiparallel. This is how the contour finding works. 
    
    An advantage of using contours is that they are closed, such that the labels can cycle back. 
    Contours provide the necessary ordering. In ND, there is no such ordering, and so I must 
    devise an alternative way to ensure that labels from different internal boundaries are still linked. 
    Currently, adjacent boundary labels get the same integer. 
    """
#     T, mu = masks_to_flows(masks,
#                            boundaries=boundaries,
#                            use_gpu=0,omni=1,
#                            smooth=0,normalize=0)[-2:]
    
#     contour_map,contour_list = get_boundary(mu,masks,contour=contour,desprue=False)

    bdi_label = ncolor.label(bdi)
    
    agi = boundary_to_affinity(maski,bdi_label>0)
    contour_map, contour_list = get_contour(maski,agi)
        
    pad = 1
    pad_bdi_lab = np.pad(bdi_label,1)
    contour_map_pad = np.pad(contour_map,1)
    maski_pad = np.pad(maski,1)
    bd_dumb_pad = find_boundaries(maski_pad,mode='inner',connectivity=2)

    turn_map = np.zeros_like(pad_bdi_lab)
    repl_map = np.zeros_like(pad_bdi_lab)

    turnpoints = []
    offset = 0
    turnlabels = []
    links = set()
    
    coords = np.nonzero(maski_pad)
    
    for c,contour in enumerate(contour_list):

        # coords_t = np.unravel_index(contour,contour_map_pad.shape)
        coords_unpad = np.nonzero(maski)
        coords_t = tuple([crd[contour] for crd in coords_unpad])
        coords_t_pad = tuple([coords_t[i]+pad for i in range(2)])
        u = bdi_label[coords_t].astype(int)
        label = np.unique(maski[coords_t])[0]
        # print(label,coords_t)
        d = np.diff(u,append=u[0])
        turns = np.nonzero(d)[0]

        bd_interior_pad = np.logical_xor(pad_bdi_lab[coords_t_pad],bd_dumb_pad[coords_t_pad]) 
        bd_interior_pad_cpy = bd_interior_pad.copy()

        for turn in turns:
            bd_interior_pad[slice(turn-1,turn+1)] = True

        nturn = len(turns)
        labels = []
        # print('nturn',nturn)
        if nturn:
            runs = utils.find_nonzero_runs(bd_interior_pad)

            # generalize to any number of turns
            labels = [[i,2,i+2] for i in range(1,2*nturn,2)]
            if nturn>1: #make cyclic 
                labels[-1][-1] = labels[0][0]
            labels = np.array(labels)+offset

            # keep track of which labels correspond to turns 
            turnlabels.append(labels[0][1]) 
        
            # create links
            [links.add((lnk[0],lnk[1])) for lnk in labels]
            if nturn>2: # make sure it loops around 
                [links.add((lnk[-1],lnk[1])) for lnk in [labels[-1]]]

            r = runs.flatten()
            intervals = [np.abs(r.take(i,mode='wrap')-r.take(i+1,mode='wrap')) for i in range(1,len(r),2)]
            endpoints = [0]+[r[1] for r in runs[:-1]]+[len(u)]
            for j,(run,turn,labs) in enumerate(zip(runs,turns,labels)):
                mid = slice(turn,turn+2)
                skip = np.sum(bd_interior_pad_cpy[mid])<2 # these are the joins along external boundaries 

                # replace with cyclic take 
                pads = [intervals[i%len(intervals)]//2 for i in [j,j+1]]
                inds = [range(turn-pads[0],turn),range(turn,turn+2),range(turn+2,turn+2+pads[1])]

                for l,i in zip(labs,inds):
                    turn_map[tuple([ct.take(i,mode='wrap') for ct in coords_t_pad])] = labs[1] if skip else l

                if not skip:  # put in the label to either side
                    repl_map[tuple([ct.take(inds[1],mode='wrap') for ct in coords_t_pad])] = [labs[i] for i in [0,-1]]

                offset+=3

        else:
            turn_map[coords_t_pad] = offset+1
            offset += 1

        vals = contour_map_pad[coords_t_pad]
        # print(len(vals),'vals')
        p = [[vals[t],vals.take(t+1,mode='wrap')] for t in turns]
        if len(p):
            turnpoints.append([label,p])
                
    
    result = np.zeros_like(maski_pad)
    for l in fastremap.unique(maski_pad)[1:]:
        mask = maski_pad==l
        # seeds = turn_map*bd_interior_pad*mask
        seeds = turn_map*mask

        if np.any(seeds):
            exp = ncolor.expand_labels(seeds)*mask
        
        result[mask] = exp[mask]
        
        
    # remove turnlabels, expand the remaining labels, then put the turnlabels back in the remaining space
    # turn_mask = np.zeros_like(turn_map)
    r2 = result.copy()
    for l in turnlabels:
        r2[np.nonzero(result==l)] = 0

    for l in fastremap.unique(maski_pad)[1:]:
        mask = maski_pad==l
        seeds = r2*mask

        if np.any(seeds):
            exp = expand_labels(seeds,1)*mask

        r2[mask] = exp[mask] # put in texpanded labels 
        r2[np.logical_and(mask,r2==0)] = result[np.logical_and(mask,r2==0)] # put back linker 

    # restore tips; expansion can mess this up a bit 
    r2[repl_map>0] = repl_map[repl_map>0]
    
    # unpad things and return split masks and corresponding links 
    unpad = tuple([slice(pad,-pad)]*maski.ndim)
    return r2[unpad], links


from scipy.signal import find_peaks

def split_contour(masks,contour_map,contour_list,bd_label=None):
    """
    Split contours at turns. Uses my own special metric for "curvature" by default.
    Can alternately use transitions between boundary labels as split points. 
    
    """
    seed_map = np.zeros(np.array(contour_map.shape),float)
    clabel_map = np.zeros(np.array(contour_map.shape),int)
    peaks = []
    inds = []
    crds = []
    
    diam = diameters(masks)
    coords = np.nonzero(masks)
    

    for contour in contour_list:
    # for contour in [contour_list[0]]:
        ll,mapping = fastremap.renumber(np.array(contour))
        lab = np.zeros(np.array(contour_map.shape),np.uint32)
        lab_ncolor = lab.copy() # preallocate ncolor array
        coords_t = tuple([c[contour] for c in coords])
        crds.append(coords_t)
        
        L = len(contour)
        Lpad = L
        
        if bd_label is None:
            coord_array = np.array(coords_t)
            step = coord_array - np.roll(coord_array,axis=1,shift=-1)
            csum = np.zeros(L,float)
            for d in range(1,int(diam)):
                c = 0.5
                d1 = np.sum((np.roll(coord_array,shift=d,axis=1)-np.roll(coord_array,shift=-d,axis=1))**2,axis=0)**c
                d2 = np.sum((np.roll(coord_array,shift=(d+1),axis=1)-np.roll(coord_array,shift=-d,axis=1))**2,axis=0)**c
                d3 = np.sum((np.roll(coord_array,shift=d,axis=1)-np.roll(coord_array,shift=-(d+1),axis=1))**2,axis=0)**c

                csum -= np.mean(np.stack([np.sum(np.roll(step,shift=d,axis=1)*np.roll(step,shift=-d,axis=1),axis=0)/d1,
                                         np.sum(np.roll(step,shift=(d+1),axis=1)*np.roll(step,shift=-d,axis=1),axis=0)/d2,
                                         np.sum(np.roll(step,shift=d,axis=1)*np.roll(step,shift=-(d+1),axis=1),axis=0)/d3,
                                         ])
                                ,axis=0)

            seed_map[coords_t] = utils.rescale(csum)
            X = np.concatenate([csum[::-1][:Lpad+1],csum,csum[::-1][:Lpad+1]])
            peaks, _ = find_peaks(X, height=1, distance=int(diam))
        
        else:
            values = bd_label[coords_t]
            Y = np.concatenate([values[::-1][:Lpad+1],values,values[::-1][:Lpad+1]])
            # X = np.logical_or(Y!=np.roll(Y,shift=1),Y!=np.roll(Y,shift=-1))*1.
            X = Y!=np.roll(Y,shift=-1)
            pks = [[[p,1] for p in np.nonzero(X)[0]]]
        

        indexes = []
        peak = []
        for peak_list in pks:
            for p in peak_list:
                idx = p[0]
                val = p[1]
                if idx>=Lpad and idx<(L+Lpad) and val>0: # deal with mirroring  
                    indexes.append(idx-Lpad)
                    peak.append([c[p[0]-Lpad] for c in coords_t])
                    peak.append([c[p[0]-Lpad+1] for c in coords_t])
                    


        
        ind = []
        I = len(indexes)
        clabel = np.ones_like(contour)
        # clabel = np.zeros_like(contour) if I else  np.ones_like(contour)
        # to change this properly, i should have an option to block splits along the exterior boundary and allow for some interval
        # Or fill this with the linker label right away 
        # otherwise default to normal
        
        
        # print(indexes)
        for i in range(I):
            start = indexes[i%I]+1
            stop = indexes[(i+1)%I]+1
            
            w =L
            # if start>stop:
            #     stop = start+w
            # else:
            #     start = stop-w
            # print(start,stop,'augmented_affinity')
            
            # 
            
            clabel[start:stop] = (i%I) + 2
            
            # clabel[start:stop] = 0
            
            # clabel[start:start+w] = (i%I) + 2
            # clabel[stop-w:stop] = (i%I) + 2
            
            # clabel[start+w:start] = 0
            # clabel[stop:stop-w] = 0
            
            ind.append(start)
            
        # clabel_map[coords_t] = clabel+clabel_map.max()*(clabel>0)
        clabel_map[coords_t] = clabel+clabel_map.max()*(clabel>0)
        
        
        inds.append(ind)
        peaks.append(peak)
    # peaks = np.stack(peaks) if len(peaks) else None
    return peaks, inds, crds, clabel_map, seed_map


# def channel_overlay(ch0, ch1, axis=1, a=1):
#     rgb = np.stack([ch0]*3,axis=-1)
#     print(rgb.shape)
#     rgb[Ellipsis,axis] = a*ch1+(ch0-a*ch1*ch0)
#     return rgb

# def channel_overlay(ch0, ch1, color=(1, 1, 0), a=1):
#     """Overlay ch1 as a color onto ch0 as grayscale."""
#     rgb = np.stack([ch0] * 3, axis=-1)
#     overlay = a * ch1 + (ch0 - a * ch1 * ch0)
#     for i in range(3):
#         rgb[..., i] = (1 - color[i]) * ch0 + color[i] * overlay
#     return rgb


def channel_overlay(channels, color_indexes, colors=None, a=1, cmaps=None):
    """Overlay selected channels as colors onto the remaining channels as grayscale."""
    N = len(channels)
    n = len(color_indexes)
    
    # Identify the grayscale channels
    grayscale_indexes = [i for i in range(N) if i not in color_indexes]
    
    # Calculate the grayscale image
    grayscale = np.mean(np.take(channels, grayscale_indexes, axis=0), axis=0) if len(grayscale_indexes) else np.zeros_like(channels[0])

    # If colors are not provided, generate them
    if colors is None:
        angle = np.arange(0, 1, 1/n) * 2 * np.pi
        angles = np.stack((angle, angle + 2*np.pi/3, angle + 4*np.pi/3), axis=-1)
        colors = (np.cos(angles) + 1) / 2
        
    else:
        colors = np.stack(colors)
        
        if colors.ndim==1:
            colors = np.expand_dims(colors, axis=0)
    
    # if there is an alpha channel to colors, mostly for color map
    nchan = colors.shape[1] if cmaps is None else 4
    
    # Create an array to hold the RGB image
    rgb = np.zeros(channels[0].shape+(nchan,))
    
    # Apply the overlays to each color channel
    for i,idx in enumerate(color_indexes):
        mapped_chan = None if cmaps is None else cmaps[i](channels[idx])
        for j in range(nchan):
            if cmaps is None:
                cc =  a * channels[idx] * colors[i,j] # color contribution 
            else:
                cc = a * mapped_chan[...,j]
            rgb[..., j] += (1 - cc) * grayscale + cc
        
    rgb /= n
    
    return rgb


# import torch
# def divergence(y):
#     axes = [k for k in range(len(y[0]))] #note that this only works when there are at least two images in batch 
#     dim = y.shape[1]
#     # print('divy',y.shape,y[:,0].shape)

#     # return torch.stack([torch.gradient(y[:,-k],dim=k)[0] for k in dims]).sum(dim=0)
#     return torch.stack([torch.gradient(y[:,ax],dim=ax-dim)[0] for ax in axes]).sum(dim=0)
    


# @njit(parallel=True)
# def project_points(source, target):
#     source_count = source.shape[1]
#     target_count = target.shape[1]
#     result = np.empty(source_count, dtype=np.int64)
    
#     for i in prange(source_count):
#         min_distance = np.inf
#         closest_point = -1
        
#         for j in range(target_count):
#             distance = np.sum((source[:, i] - target[:, j]) ** 2)
            
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_point = j
        
#         result[i] = closest_point
    
#     return result


# def export_movie(frames,basename,basedir,scale=1,fps=15):
#     frame_width, frame_height, nchan = frames.shape[-3:]
#     if nchan==3:
#         pixel_format = 'rgb48le'
#     else:
#         pixel_format = 'rgba64le'
        
#     file = os.path.join(basedir,basename+'_subprocess_lossless_h264_{}_fps.mp4'.format(fps))

#     p = subprocess.Popen(['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec',
#                           'rawvideo', '-s', '{}x{}'.format(frame_height,frame_width), '-pix_fmt', pixel_format,
#                           '-r', str(fps), '-i', '-', '-vf', 'scale=iw*{}:ih*{}:flags=neighbor'.format(scale,scale), 
#                           '-an', '-vcodec', 'libx264','-crf','0','-preset','ultrafast',
#                           file], stdin=subprocess.PIPE)


#     # loop over the frames
#     for frame in to_16_bit(frames): # I wonder if 8 bit would be interpolated too 
#         # write frame to pipe
#         p.stdin.write(frame.tostring())

#     # close the pipe
#     p.stdin.close()
#     p.wait()

# def export_movie(frames,basename,basedir,scale=1,fps=15):
#     frame_width, frame_height, nchan = frames.shape[-3:]
#     if nchan==3:
#         pixel_format = 'rgb48le'
#     else:
#         pixel_format = 'rgba64le'
        
#     file = os.path.join(basedir,basename+'_scale_{}_fps_{}.mov'.format(scale,fps))

#     p = subprocess.Popen(['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec',
#                           'rawvideo', '-s', '{}x{}'.format(frame_height,frame_width), '-pix_fmt', pixel_format,
#                           '-r', str(fps), '-i', '-', '-vf', 'scale=iw*{}:ih*{}:flags=neighbor'.format(scale,scale), 
#                           '-an', '-vcodec', 'prores_ks',
#                           file], stdin=subprocess.PIPE)

#     # loop over the frames
#     for frame in to_16_bit(frames): 
#         # write frame to pipe
#         p.stdin.write(frame.tobytes())

#     # close the pipe
#     p.stdin.close()
#     p.wait()
    




# def export_movie(frames,basename,basedir,scale=1,fps=15):
#     frame_width, frame_height, nchan = frames.shape[-3:]
#     if nchan==3:
#         pixel_format = 'rgb48le'
#     else:
#         pixel_format = 'rgba64le'
        
#     file = os.path.join(basedir,basename+'_{}_fps.mp4'.format(fps))

#     p = subprocess.Popen(['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec',
#                           'rawvideo', '-s', '{}x{}'.format(frame_height,frame_width), '-pix_fmt', pixel_format,
#                           '-r', str(fps), '-i', '-', '-vf', 'scale=iw*{}:ih*{}:flags=neighbor'.format(scale,scale), 
#                           '-an', '-vcodec', 'libx264','-preset','ultrafast',
#                           file], stdin=subprocess.PIPE)

#     # loop over the frames
#     for frame in to_16_bit(frames): 
#         # write frame to pipe
#         p.stdin.write(frame.tostring())

#     # close the pipe
#     p.stdin.close()
#     p.wait()
def export_movie(frames, basename, basedir, scale=1, fps=15):
    frame_width, frame_height, nchan = frames.shape[-3:]
    if nchan == 3:
        pixel_format = 'rgb48le'
    else:
        pixel_format = 'rgba64le'

    file = os.path.join(basedir, basename + '_{}_fps.mp4'.format(fps))

    p = subprocess.Popen(['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec',
                          'rawvideo', '-s', '{}x{}'.format(frame_height, frame_width), '-pix_fmt', pixel_format,
                          '-r', str(fps), '-i', '-', '-f', 'lavfi', '-i', 'anullsrc', '-vf', 'scale=iw*{}:ih*{}:flags=neighbor'.format(scale, scale),
                          '-shortest', '-c:v', 'mpeg4', '-q:v', '0',
                          file], stdin=subprocess.PIPE)

    # loop over the frames
    for frame in to_16_bit(frames):
        # write frame to pipe
        p.stdin.write(frame.tostring())

    # close the pipe
    p.stdin.close()
    p.wait()
    
    
# def export_gif(frames,basename,basedir,scale=1,fps=15, loop=0, bounce=True):
#     try:
#         frame_width, frame_height, nchan = frames.shape[-3:]
#         if nchan==3:
#             pixel_format = 'rgb24'
#         else:
#             pixel_format = 'rgba'
            
#         file = os.path.join(basedir,basename+'_{}_fps.gif'.format(fps))

#         p = subprocess.Popen(['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec',
#                               'rawvideo', '-s', '{}x{}'.format(frame_height,frame_width), '-pix_fmt', pixel_format,
#                               '-r', str(fps), '-i', '-', '-vf', 'scale=iw*{}:ih*{}:flags=neighbor'.format(scale,scale), 
#                               '-an', '-vcodec', 'gif', '-loop', str(loop),
#                               file], stdin=subprocess.PIPE)

#         # loop over the frames
#         frames_8_bit = to_8_bit(frames)
#         if bounce:
#             frames_8_bit = np.concatenate((frames_8_bit, frames_8_bit[::-1]), axis=0)
#         for frame in frames_8_bit: 
#             # write frame to pipe
#             p.stdin.write(frame.tobytes())
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         # close the pipe
#         p.stdin.close()
#         p.wait()

from scipy import ndimage
def export_gif(frames, basename, basedir, scale=1, fps=15, loop=0, bounce=True):
    if scale !=1:
        frames = ndimage.zoom(frames,[1,scale,scale,1],order=0)
        # scaling is not working with ffmpeg, so I will just scale the frames with ndimage
        # should be timepoints x Y x X x channels
    try:
        if frames.ndim==4:
            frame_width, frame_height, nchan = frames.shape[-3:]
            if nchan==3:
                pixel_format = 'rgb24'
            else:
                pixel_format = 'rgba'
        else:
            frame_width, frame_height = frames.shape[-2:]
            pixel_format = 'gray'
            # turns out this gives the same size, maybe I would need to specify the palette too
            #
            
        file = os.path.join(basedir, basename+'_{}_fps_scale_{}.gif'.format(fps,scale))

        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', 
                              '-f', 'rawvideo', '-vcodec',
                              'rawvideo', '-s', '{}x{}'.format(frame_height,frame_width), '-pix_fmt', pixel_format,
                              '-r', str(fps), '-i', '-', '-an', 
                            #   '-filter_complex', '[0:v]palettegen=stats_mode=single[pal],[0:v][pal]paletteuse=dither=none',
                               '-filter_complex', '[0:v]palettegen=stats_mode=full[pal],[0:v][pal]paletteuse=dither=none',

                                '-vcodec', 'gif', '-loop', str(loop),
                              file], stdin=subprocess.PIPE)

        # loop over the frames
        frames_8_bit = to_8_bit(frames)
        if bounce:
            frames_8_bit = np.concatenate((frames_8_bit, frames_8_bit[::-1]), axis=0)
        for frame in frames_8_bit: 
            # write frame to pipe
            p.stdin.write(frame.tobytes())
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # close the pipe
        p.stdin.close()
        p.wait()


def get_size(var, unit='GB'):
    units = {'B': 0, 'KB': 1, 'MB': 2, 'GB': 3}
    return var.nbytes / (1024 ** units[unit])
    
from collections.abc import Iterable

def get_slice_tuple(start, stop, shape, axis=None):
    ndim = len(shape)

    # Create a list of slices for each axis
    slices = [slice(None)] * ndim 


    # Check if start and stop are iterable
    if isinstance(start, Iterable) and isinstance(stop, Iterable):
        if axis is None:
            axis = list(range(ndim))
    
        # Check that start and stop are the same length
        if len(start) != len(stop):
            raise ValueError("start and stop must be the same length")

        # Check if axis is iterable
        if isinstance(axis, Iterable):
            # Check that axis is the same length as start and stop
            if len(axis) != len(start):
                raise ValueError("axis must be the same length as start and stop")
        else:
            # If axis is not iterable, use it for all slices
            axis = [axis] * len(start)

        # Replace the slice at each axis index
        for a, s, e in zip(axis, start, stop):
            slices[a] = slice(s, e, None)
    else:
        if axis is None:
            axis = 0
        # If start and stop are not iterable, use them as integers
        slices[axis] = slice(start, stop, None)

    # Convert the list to a tuple
    return tuple(slices)
    

import subprocess
def explore_object(obj):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets is not installed. Installing now...")
        subprocess.check_call(["pip", "install", "ipywidgets"])
        import ipywidgets as widgets
        from IPython.display import display

    output = widgets.Output()

    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            output.clear_output()
            with output:
                try:
                    next_obj = getattr(obj, change['new'])
                    print(f"Selected: {change['new']}")
                    print(dir(next_obj))
                    if hasattr(next_obj, '__dict__'):
                        explore_object(next_obj)
                except Exception as e:
                    print(str(e))

    dropdown = widgets.Dropdown(
        options=[attr for attr in dir(obj) if not attr.startswith("__")],
        description='Attributes:',
    )

    dropdown.observe(on_change)
    display(widgets.HBox([dropdown, output]))

# from scipy.ndimage import uniform_filter
# def find_highest_density_box(label_matrix, box_size, mode='constant'):
#     if box_size == -1:
#         # return tuple([slice(None)]*label_matrix.ndim)
#         return tuple([slice(0,s) for s in label_matrix.shape])
#     else:
#         # Compute the cell density for each box in the image
#         cell_density = uniform_filter((label_matrix > 0).astype(float), size=box_size, mode=mode)

#         # Find the coordinates of the box with the highest cell density
#         max_density_coords = np.unravel_index(np.argmax(cell_density), cell_density.shape)

#         # Compute the coordinates of the box
#         return tuple(slice(max_coord - box_size // 2, max_coord + box_size // 2) for max_coord in max_density_coords), cell_density
    
    
from scipy.signal import fftconvolve
def bartlett_nd(size):
    """
    Create an N-dimensional Bartlett (triangular) window with shape `size`.
    If size is an integer, we treat it as (size,).
    """
    if isinstance(size, int):
        size = (size,)
    # Create a 1D Bartlett window for each dimension
    windows = [np.bartlett(s) for s in size]

    # Use np.ix_ to create broadcastable ND grids
    grids = np.ix_(*windows)  # e.g., for 2D => shape(7,1), shape(1,7)

    # Multiply them elementwise to get an ND array
    # Instead of in-place, build the kernel in a loop
    kernel = grids[0].astype(float)
    for g in grids[1:]:
        kernel = kernel * g  # shape updates from (7,1)->(7,7) in 2D, etc.

    # Normalize so the total sum = 1
    kernel /= kernel.sum()

    return kernel

def find_highest_density_box(label_matrix, box_size):
    """
    Convolve a binary mask with an N-D Bartlett kernel of shape `box_size`,
    then find the sub-box of shape `box_size` around the maximum, ensuring 
    the box stays within bounds.
    """
    if box_size == -1:
        return tuple(slice(0, s) for s in label_matrix.shape)

    # Handle scalar box_size for all dimensions
    if isinstance(box_size, int):
        box_size = (box_size,) * label_matrix.ndim

    # Binary mask
    mask = (label_matrix > 0).astype(np.float32)

    # Build the N-D Bartlett (triangular) kernel
    kernel = bartlett_nd(box_size)

    # FFT-based convolution
    density_map = fftconvolve(mask, kernel, mode='same')

    # Find the coordinates of the box with the highest cell density
    max_density_coords = np.unravel_index(np.argmax(density_map), density_map.shape)

    # Compute the box bounds while ensuring no negative indices
    slices = []
    for max_coord, size, dim_size in zip(max_density_coords, box_size, label_matrix.shape):
        start = max(0, max_coord - size // 2)
        stop = min(dim_size, start + size)  # Ensure within bounds
        start = max(0, stop - size)  # Adjust start if necessary to maintain box size
        slices.append(slice(start, stop))

    return tuple(slices)
    
# def create_pill_mask(R, L, f = np.sqrt(2)):
def create_pill_mask(R, L, f = 1):

    # Determine the size of the image
    height = 2 * R# +2 for 1px boundary at top and bottom
    width = L + 2*R  # +2 for 1px boundary on left and right
    
    # Create an empty image
    pad = 3
    imh = height+2*pad + 1
    imw = width+2*pad +1
    # imh = 2*(imh//2)+1 # make odd
    # imw = 2*(imw//2)+1
    
    mask = np.zeros((imh,imw), dtype=np.uint8)
    
    # Calculate the center of the pill shape
    center_x = imw // 2
    center_y = imh // 2
    
    # Draw the rectangular part of the pill
    mask[center_y - R:center_y + R+1, R+pad:L+R+pad+1] = 1
    
    # Create a grid of coordinates
    y, x = np.ogrid[:imh, :imw]
    
    # Draw the left semicircle
    left_center_x = R+pad
    left_circle = (x - left_center_x) ** 2 + (y - center_y) ** 2 <= f*(R ** 2)
    mask[left_circle] = 1
    
    # Draw the right semicircle
    right_center_x = L+R+pad
    right_circle = (x - right_center_x) ** 2 + (y - center_y) ** 2 <= f*(R ** 2)
    mask[right_circle] = 1
    
    return mask
    
    
import numpy as np
from scipy.fft import dstn, idstn

def compute_vector_field_from_divergence(divergence, grid_spacing=1):
    """
    Compute the vector field from its divergence using DST-based Poisson solver.
    
    Parameters:
        divergence (ndarray): The divergence array (2D or 3D).
        grid_spacing (float or sequence of floats): Grid spacing in each dimension.
        
    Returns:
        vector_field (list of ndarrays): Components of the vector field.
    """
    # Ensure grid_spacing is a list for multi-dimensional grids
    if np.isscalar(grid_spacing):
        grid_spacing = [grid_spacing] * divergence.ndim
    
    # Get the shape of the divergence array
    shape = divergence.shape
    ndim = divergence.ndim
    
    # Grid spacings
    d = grid_spacing
    
    # Create k vectors for each dimension
    k = []
    for n, delta in zip(shape, d):
        k.append(np.pi * np.arange(1, n+1) / (n+1) / delta)
    mesh = np.meshgrid(*k, indexing='ij')
    
    # Compute K squared
    K_squared = sum((k_i)**2 for k_i in mesh)
    
    # Compute the DST of the divergence with normalization
    divergence_dst = dstn(divergence, type=1, norm='ortho')
    
    # Solve Poisson's equation in DST space
    Phi_dst = divergence_dst / K_squared
    
    # Handle division by zero (should not happen since k ranges from 1 to n)
    # However, just in case, set any zero K_squared to a very large number
    Phi_dst = np.nan_to_num(Phi_dst, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Inverse DST to get the scalar potential Phi with normalization
    Phi = idstn(Phi_dst, type=1, norm='ortho')
    
    # Compute the gradient of Phi to get the vector field components
    gradient = np.gradient(Phi, *d, edge_order=2)
    
    # The vector field V = -∇Phi
    vector_field = [-g for g in gradient]
    
    return np.stack(vector_field)
    
    
import numpy as np

def split_array(array, parts, axes=None):
    """
    Split an ndarray into parts along specified axes.
    
    Parameters:
    - array: ndarray
        The array to split.
    - parts: int or tuple of ints
        Number of parts to split along each axis.
        If an integer, applies to all axes. If a tuple, it specifies parts for each axis.
    - axes: int or tuple of ints, optional
        The axes to split. If None, splits across all axes.
    
    Returns:
    - List of ndarrays
        A nested list of sub-arrays after splitting.
    """
    if isinstance(parts, int):
        parts = (parts,) * array.ndim  # Apply same number of parts to all axes

    if axes is None:
        axes = tuple(range(array.ndim))  # Apply to all axes
    elif isinstance(axes, int):
        axes = (axes,)  # Make it a tuple for consistency

    if len(parts) != len(axes):
        raise ValueError("Length of 'parts' must match the number of axes specified.")

    splits = []  # Store split slices
    warnings = []  # Store warnings for uneven splits

    for ax, num_parts in zip(axes, parts):
        dim_size = array.shape[ax]
        chunk_sizes = [dim_size // num_parts + (1 if i < dim_size % num_parts else 0) for i in range(num_parts)]
        if dim_size % num_parts != 0:
            warnings.append(f"Axis {ax} ({dim_size}) is not evenly divisible by {num_parts}.")
        split_indices = np.cumsum(chunk_sizes[:-1])
        splits.append(np.split(np.arange(dim_size), split_indices))

    # Print warnings if any
    for warning in warnings:
        print("Warning:", warning)

    # Use the slices to split the array recursively
    def recursive_split(array, splits, axes):
        if not splits:
            return array
        ax = axes[0]
        subarrays = []
        for idxs in splits[0]:
            sliced = np.take(array, idxs, axis=ax)
            subarrays.append(recursive_split(sliced, splits[1:], axes[1:]))
        return subarrays

    return recursive_split(array, splits, axes)
    
def reconstruct_array(nested_list, axes=None):
    """
    Reconstruct an ndarray from a nested list of sub-arrays.
    
    Parameters:
    - nested_list: list of ndarrays
        The nested list of sub-arrays to reconstruct.
    - axes: int or tuple of ints, optional
        The axes used for splitting. If None, assumes all axes.

    Returns:
    - ndarray
        The reconstructed array.
    """
    if axes is None:
        axes = tuple(range(len(nested_list[0].shape) if isinstance(nested_list[0], np.ndarray) else len(nested_list)))
    elif isinstance(axes, int):
        axes = (axes,)

    def recursive_reconstruct(nested, level):
        """
        Recursively reconstruct the array along specified axes.
        """
        if isinstance(nested, np.ndarray):
            return nested
        if level == len(axes):
            return np.array(nested)
        return np.concatenate(
            [recursive_reconstruct(sub, level + 1) for sub in nested], axis=axes[level]
        )

    return recursive_reconstruct(nested_list, 0)


def enumerate_nested(*lists, parent_indices=None):
    """
    Traverse one or more matching nested lists and yield their indices and corresponding values.

    Parameters:
    - *lists: list(s)
        One or more nested lists to traverse. All lists must match in structure.
    - parent_indices: list, optional
        The list of indices leading to the current level (used internally).

    Yields:
    - tuple: (indices, values...)
        The indices and corresponding values from all input lists.
    """
    if parent_indices is None:
        parent_indices = []

    # Check if elements are lists at this level
    if all(isinstance(lst[0], list) for lst in lists):
        for i, sublists in enumerate(zip(*lists)):
            current_indices = parent_indices + [i]
            yield from enumerate_nested(*sublists, parent_indices=current_indices)
    else:  # Base case: elements are not lists
        for i, values in enumerate(zip(*lists)):
            current_indices = parent_indices + [i]
            yield current_indices, *values