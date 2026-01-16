import numpy as np
import fastremap
from scipy.ndimage import gaussian_filter, convolve
from skimage.morphology import skeletonize as skimage_skeletonize
from ..utils.neighbor import kernel_setup

def find_boundaries(labels, connectivity=1, use_symmetry=False):
    """
    Compute boundaries of labeled instances in an N-dimensional array.
    Replicates the behavior of skimage.segmentation.find_boundaries with mode='inner', but is much faster. 
    
    Decreasing the steps by leveraging symmetry seems not to matter, as we still end up with two logical operations 
    and two updates to the boundary matrix. Keeping for further testing. 
    """
    boundaries = np.zeros_like(labels, dtype=bool)
    ndim = labels.ndim
    shape = labels.shape

    # Generate all possible shifts based on connectivity
    steps, inds, idx, fact, sign = kernel_setup(ndim)
    
    if use_symmetry:
        allowed_inds = []
        for i in range(1,1+connectivity):
            j = inds[i][:len(inds[i]) // 2]
            allowed_inds.append(j)
        
        allowed_inds = np.concatenate(allowed_inds)
    else:
        allowed_inds = np.concatenate(inds[1:1+connectivity])
    
    
    shifts = steps[allowed_inds]
  
    if use_symmetry:
        # Process each shift
        for shift in shifts:
            slices_main = tuple(slice(max(-s, 0), min(shape[d] - s, shape[d])) for d, s in enumerate(shift))
            slices_shifted = tuple(slice(max(s, 0), min(shape[d] + s, shape[d])) for d, s in enumerate(shift))

            # Detect boundaries using symmetric property
            boundary_main = (labels[slices_main] != labels[slices_shifted]) & (labels[slices_main] != 0)
            boundary_shifted = (labels[slices_shifted] != labels[slices_main]) & (labels[slices_shifted] != 0)
            
            # Apply boundary detection symmetrically
            boundaries[slices_main] |= boundary_main
            boundaries[slices_shifted] |= boundary_shifted
    else:
        # Process each shift
        for shift in shifts:
            slices_main = tuple(slice(max(-s, 0), min(shape[d] - s, shape[d])) for d, s in enumerate(shift))
            slices_shifted = tuple(slice(max(s, 0), min(shape[d] + s, shape[d])) for d, s in enumerate(shift))

            # Detect boundaries in the valid region defined by the slices
            boundaries[slices_main] |= (labels[slices_main] != labels[slices_shifted]) & (labels[slices_main] != 0)

    return boundaries.astype(np.uint8)

def label_skeletonize(labels,method='zhang'):
    # Find boundaries
    bd = find_boundaries(labels, connectivity=2)
    
    # Remove boundaries from labels to get inner regions
    inner = np.logical_xor(labels > 0, bd)
    # inner = (labels > 0) - bd
    
    skel = skimage_skeletonize(inner, method=method)
    
    # Retain original labels on the skeleton
    skeleton = skel * labels
            
    # Identify labels present in the original labels
    original_labels = fastremap.unique(labels)
    original_labels = original_labels[original_labels != 0]  # Exclude background
    
    # Identify labels present in the skeletonized image
    skeleton_labels = fastremap.unique(skeleton)
    skeleton_labels = skeleton_labels[skeleton_labels != 0]  # Exclude background
    
    # Find missing labels
    missing_labels = np.setdiff1d(original_labels, skeleton_labels)
        
    # Create a mask for missing labels
    missing_labels_mask = np.isin(labels, missing_labels)
    missing_labels_mask = fastremap.mask_except(labels, list(missing_labels))
        
    # Add back missing labels to the skeleton
    # skeleton += missing_labels_mask * labels using isin 
    skeleton += missing_labels_mask 
    
    return skeleton

def skeletonize(labels, dt_thresh=1, dt=None, method='zhang'):
    """Skeletonize labels, optionally using a distance transform threshold."""
    if dt is not None:
        inner = dt > dt_thresh
        skel = skimage_skeletonize(inner, method=method)
        return skel * labels
    return label_skeletonize(labels, method=method)





def extract_skeleton(distance_field):
    # Smooth the distance field using Gaussian filter
    smoothed_field = gaussian_filter(distance_field, sigma=1)

    # Compute gradient of the smoothed distance field
    gradients = np.gradient(smoothed_field)

    # Compute Hessian matrix components
    hessians = []
    for i in range(len(distance_field.shape)):
        hessian = np.gradient(gradients[i])
        hessians.append(hessian)

    hessians = [gaussian_filter(hessian, sigma=1) for hessian in hessians]

    # Compute the Laplacian of Gaussian (LoG)
    log = np.sum(hessians, axis=0)

    # Find stationary points (zero-crossings) in the LoG
    zero_crossings = np.logical_and(log[:-1] * log[1:] < 0, np.abs(log[:-1] - log[1:]) > 0.02)

    # Thin the zero-crossings to get the skeleton
    skeleton = thin_skeleton(zero_crossings)

    return skeleton

def thin_skeleton(image):
    # DTS thinning algorithm
    dimensions = len(image.shape)
    neighbors = np.ones((3,) * dimensions, dtype=bool)
    neighbors[tuple([1] * dimensions)] = False

    while True:
        marker = np.zeros_like(image)

        # Convolve the image with the neighbors template
        convolution = convolve(image, neighbors, mode='constant')

        # Find the pixels where the convolution equals the number of neighbors
        marker[np.where(convolution == np.sum(neighbors))] = 1

        if np.sum(marker) == 0:
            break

        image = np.logical_and(image, np.logical_not(marker))

    return image
