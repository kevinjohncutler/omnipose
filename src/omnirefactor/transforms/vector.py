import numpy as np
import torch
from scipy.fft import dstn, idstn


def torch_norm(a, dim=0, keepdim=False):
    """Wrapper for torch.linalg.norm with minimal intermediate allocations."""
    norm_sq = (a * a).sum(dim=dim, keepdim=keepdim)
    return norm_sq.sqrt_() if not norm_sq.requires_grad else norm_sq.sqrt()

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
