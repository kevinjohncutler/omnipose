from .imports import *
from ..measure.skeleton import skeletonize

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
