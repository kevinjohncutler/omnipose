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
    """
    if np.isscalar(grid_spacing):
        grid_spacing = [grid_spacing] * divergence.ndim

    shape = divergence.shape
    ndim = divergence.ndim

    k = []
    for n, delta in zip(shape, grid_spacing):
        k.append(np.pi * np.arange(1, n + 1) / (n + 1) / delta)
    mesh = np.meshgrid(*k, indexing="ij")

    K_squared = sum((k_i) ** 2 for k_i in mesh)
    divergence_dst = dstn(divergence, type=1, norm="ortho")

    Phi_dst = divergence_dst / K_squared
    Phi_dst = np.nan_to_num(Phi_dst, nan=0.0, posinf=0.0, neginf=0.0)
    Phi = idstn(Phi_dst, type=1, norm="ortho")

    gradient = np.gradient(Phi, *grid_spacing, edge_order=2)
    vector_field = [-g for g in gradient]

    return np.stack(vector_field)


__all__ = ["torch_norm", "compute_vector_field_from_divergence"]
