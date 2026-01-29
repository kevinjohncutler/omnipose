from functools import reduce
from typing import List

import numpy as np
import torch
from scipy.special import expit

from ..transforms.normalize import normalize_field, normalize99


# @torch.no_grad() # try to solve memory leak in mps
def update_torch(a, f, fsq):
    # Turns out we can just avoid a ton of individual if/else by evaluating the update function
    # for every upper limit on the sorted pairs. I do this by pieces using cumsum. The radicand
    # being nonegative sets the upper limit on the sorted pairs, so we simply select the largest
    # upper limit that works. I also put a couple of the indexing tensors outside of the loop.
    """Update function for solving the Eikonal equation."""
    a, _ = torch.sort(a, dim=0)  # sorting was the source of the small artifact bug
    am = a * ((a - a[-1]) < f)
    sum_a = am.sum(dim=0)
    sum_a2 = (am ** 2).sum(dim=0)

    d = a.shape[0]  # d acutally needed to be the number of elements being compared, not dimension
    return (1 / d) * (sum_a + torch.sqrt(torch.clamp((sum_a ** 2) - d * (sum_a2 - fsq), min=0)))


def _iterate(T: torch.Tensor,
             neigh_inds: torch.Tensor,
             central_inds: torch.Tensor,
             centroid_inds: torch.Tensor,
             idx: torch.Tensor,
             d: torch.Tensor,
             inds: List[torch.Tensor],
             fact: torch.Tensor,
             isneigh: torch.Tensor,
             n_iter: torch.Tensor,
             omni: torch.Tensor,
             verbose: torch.Tensor):

    T0 = T.clone()
    eps = 1e-3

    if verbose:
        print('eps is ', eps, 'n_iter is', n_iter)

    t = torch.tensor(0)
    not_converged = torch.tensor(True)
    error = torch.tensor(1)

    r = central_inds

    while not_converged:
        if omni:
            Tneigh = T[neigh_inds]
            Tneigh *= isneigh
            T = eikonal_update_torch(Tneigh, r, d, inds, fact)
        else:
            T[centroid_inds] += 1

        error = (T - T0).square().mean()

        if omni:
            not_converged = torch.logical_and(error > eps, t < n_iter)
        else:
            not_converged = t < n_iter

        if not omni or t < 1:
            Tneigh = T[neigh_inds]
            Tneigh *= isneigh
            T = Tneigh.mean(dim=0)

        T0.copy_(T)
        t += 1

    if verbose:
        print('iter: ', t, '{:.10f}'.format(error))

    if not omni:
        T = torch.log(1. + T)

    return T


def _gradient(T, d, steps, fact,
              inds: List[torch.Tensor],
              isneigh,
              neigh_inds: torch.Tensor,
              central_inds: torch.Tensor,
              s: List[int]):

    finite_differences = torch.zeros(s, device=T.device, dtype=T.dtype)
    cvals = T[central_inds]
    for ax, (ind, f) in enumerate(zip(inds[1:], fact[1:])):

        vals = T[neigh_inds[ind]]
        vals[~isneigh[ind]] = 0

        mid = len(ind) // 2
        r = torch.arange(mid)
        vecs = steps[ind].float()
        uvecs = (vecs[-(r + 1)] - vecs[r]).T

        diff = (vals[-(r + 1)] - vals[r])

        finite_differences[ax] = torch.matmul(uvecs, diff) / (2 * f) ** 2

    mu = torch.mean(finite_differences, dim=0)

    weight = torch.sum(mu[:, neigh_inds] * (mu[:, central_inds].unsqueeze(1)), dim=0).abs()
    weight[~isneigh] = 0
    wsum = weight.sum(dim=0)
    return torch.where(wsum != 0,
                       (mu[:, neigh_inds] * weight).sum(dim=1) / wsum,
                       torch.zeros_like(wsum))

@torch.compile
def eikonal_update_torch(Tneigh: torch.Tensor,
                             r: torch.Tensor,
                             d: torch.Tensor,
                             index_list: List[torch.Tensor],
                             factors: torch.Tensor):
    """Vectorized variant of eikonal update for better compile performance."""
    geometric = 1
    phi_total = torch.ones_like(Tneigh[0, :]) if geometric else torch.zeros_like(Tneigh[0, :])

    n = len(factors) - 1

    for inds, f, fsq in zip(index_list[1:], factors[1:], factors[1:] ** 2):
        npair = len(inds) // 2
        left = inds[:npair]
        right = torch.flip(inds, dims=[0])[:npair]
        mins = torch.minimum(Tneigh[left], Tneigh[right])

        update = update_torch(mins, f, fsq)

        if geometric:
            phi_total *= update
        else:
            phi_total += update

    phi_total = torch.pow(phi_total, 1 / n) if geometric else phi_total / n
    return phi_total


    


# Omnipose requires (a) a special suppressed Euler step and (b) a special mask reconstruction algorithm.

# no reason to use njit here except for compatibility with jitted functions that call it
# this way, the same factor is used everywhere (CPU with/without interp, GPU)
# @njit()
def step_factor(t):
    """Euler integration suppression factor."""
    return (1 + t)


def div_rescale(dP, mask, p=1):
    """
    Normalize the flow magnitude to rescaled 0-1 divergence.

    Parameters
    -------------
    dP: float, ND array
        flow field
    mask: int, ND array
        label matrix

    Returns
    -------------
    dP: float, ND array
        rescaled flow field

    """
    dP = dP.copy()
    dP *= mask
    dP = normalize_field(dP)
    if p > 0:
        div = normalize99(divergence(dP)) ** p
        dP *= div
    return dP


def sigmoid(x): #  pragma: no cover
    """The sigmoid function."""
    return expit(x)


def divergence(f, sp=None):
    """Computes divergence of vector field."""
    num_dims = len(f)
    if any(f.shape[1 + i] < 2 for i in range(num_dims - 1)):
        return np.zeros_like(f[0])
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])


def divergence_torch_old(y):
    dim = y.shape[1]
    dims = [k for k in range(-dim, 0)]
    return torch.stack([torch.gradient(y[:, k], dim=k)[0] for k in dims]).sum(dim=0)


def divergence_torch(y):
    """
    Divergence for a batched D-vector field stored as ``(B, D, *spatial)``.

    * **GPU / MPS** -> use a single call to ``torch.gradient`` (fast, parallel).
    * **CPU**       -> compute only the gradients actually needed, one component
      at a time, to avoid the unnecessary D^2 work that the vectorised call
      performs on the CPU.

    Returns
    -------
    div : torch.Tensor
        Shape ``(B, *spatial)`` - divergence of ``y``.
    """
    B, D, *spatial = y.shape

    if any(s < 2 for s in spatial): 
        return torch.zeros((B, *spatial), dtype=y.dtype, device=y.device)
    if y.device.type == 'cpu':
        div = torch.zeros((B, *spatial), dtype=y.dtype, device=y.device)
        for d in range(D):
            comp = y[:, d]
            axis = d + 1
            grad_d = torch.gradient(comp, dim=axis)[0]
            div += grad_d
        return div
    else:
        spatial_axes = list(range(-len(spatial), 0))
        grads = torch.gradient(y, dim=spatial_axes)
        div = sum(g[:, d] for d, g in enumerate(grads))
        return div


def _ensure_torch(*arrays, device=None, dtype=torch.float32):
    """Convert numpy arrays to torch tensors if needed."""
    return tuple(
        torch.tensor(arr, dtype=dtype, device=device).unsqueeze(0) if isinstance(arr, np.ndarray) else arr
        for arr in arrays
    )


def torch_and_cpu(tensors):
    """
    Pair-wise logical AND using functools.reduce.
    Faster on CPU where kernel-launch overhead is negligible.
    """
    return reduce(torch.logical_and, tensors)


def torch_and_gpu(tensors):
    """
    Vectorized logical AND via torch.all after stacking.
    Single kernel makes it faster on GPU.
    """
    return torch.all(torch.stack(tuple(tensors), dim=0), dim=0)


def torch_and(tensors):
    """
    Dispatch to torch_and_cpu or torch_and_gpu depending on the
    device of the first tensor in *tensors*.
    """
    dev = tensors[0].device if tensors else torch.device('cpu')

    try:
        broadcasted = torch.broadcast_tensors(*tensors)
    except AttributeError:
        ref_shape = tensors[0].shape
        broadcasted = [
            t.expand(ref_shape) if t.shape != ref_shape else t
            for t in tensors
        ]

    if dev.type == 'cpu':
        return torch_and_cpu(broadcasted)
    else:
        return torch_and_gpu(broadcasted)
