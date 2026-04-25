from __future__ import annotations
from .imports import *

from typing import List, Sequence
import torch.nn.functional as F
from scipy.special import expit


def _maybe_compile(fn):
    """Apply torch.compile only when the Inductor backend is available (CUDA).
    MPS and some CPU builds don't support it, so we fall back to eager."""
    try:
        if torch.cuda.is_available():
            return torch.compile(fn)
    except Exception:
        pass
    return fn



def roll_and_clamp(x: torch.Tensor,
                   step: Sequence[int],
                   dim_offset: int = 0,
                   seam_ends: Sequence[int] = (),
                   seam_starts: Sequence[int] = ()) -> torch.Tensor:
    """Roll x by -step in each spatial dimension, clamping at boundaries.

    At true grid boundaries the circular-roll wraparound is replaced by the
    original boundary value (Neumann no-flux condition).

    ``seam_ends`` / ``seam_starts`` extend this clamping to the internal
    stitching seams produced by ``concatenate_labels``.  The CPU
    ``get_neighbors`` implementation clips neighbours that cross a seam back
    to the seam pixel itself (``is_edge = True``); these arguments replicate
    that behaviour on the GPU.

    Args:
        x:           tensor to roll.  Spatial dims start at ``dim_offset``.
        step:        per-spatial-dimension shift amounts.
        dim_offset:  leading non-spatial dims (0 for scalar, 1 for vector).
        seam_ends:   rows in spatial-dim 0 that are the *last* row of a tile
                     (clamp when step[0] > 0, i.e. stepping across the seam).
        seam_starts: rows in spatial-dim 0 that are the *first* row of the
                     next tile (clamp when step[0] < 0).
    """
    out = x
    for d, s in enumerate(step):
        if s != 0:
            real_d = d + dim_offset
            prev = out
            out = torch.roll(prev, -s, real_d)

            # True boundary clamp (prevents wrap-around at grid edges).
            idx = [slice(None)] * out.ndim
            idx[real_d] = -1 if s > 0 else 0
            out[tuple(idx)] = prev[tuple(idx)]

            # Seam clamp: treat stitching-seam rows the same as true
            # boundaries so the Eikonal field does not propagate across them.
            # Only needed for spatial dim 0 (the concatenation axis).
            if d == 0:
                seam_rows = seam_ends if s > 0 else seam_starts
                # Scalar loop avoids list/fancy indexing (can fail on some backends).
                # out and prev are different tensors (prev=before roll, out=after),
                # so there is no aliasing here.
                for sr in seam_rows:
                    sr_idx = [slice(None)] * out.ndim
                    sr_idx[real_d] = sr
                    out[tuple(sr_idx)] = prev[tuple(sr_idx)]
    return out


def _pad_and_stack_neighbors(x: torch.Tensor,
                              steps: Sequence[Sequence[int]]) -> torch.Tensor:
    """Vectorized neighbor extraction via F.pad + slice stack (no Python loop per step).

    Replaces the ``roll_and_clamp`` Python loop in ``_iterate_grid`` /
    ``_gradient_grid`` / ``compute_affinity_gpu``.  Runs one GPU pad kernel
    + one stack instead of ``nsteps`` roll+clamp calls.

    Seam handling is NOT done here — callers use ``_make_seam_mask`` +
    ``torch.where`` to apply the correction after stacking.  This avoids
    in-place writes to ``x_pad`` (which can cause aliasing issues).

    Args:
        x:     (*spatial) or (C, *spatial) tensor.
               Spatial dims are the LAST ``len(steps[0])`` dims.
        steps: iterable of int sequences, length = n_spatial_dims.

    Returns:
        (nsteps, *x.shape) tensor — x shifted by each step.
    """
    ndim_spatial = len(steps[0])
    x_shape = x.shape                          # (*spatial) or (C, *spatial)
    spatial = x_shape[-ndim_spatial:]          # spatial dims only
    non_sp = x.ndim - ndim_spatial            # leading non-spatial dims (0 scalar, 1 vector)

    # Pad spatial dims with replicate BC (+1 on every side).
    # F.pad pads from the *last* dim inward.
    pad_widths = (1, 1) * ndim_spatial
    x_pad = F.pad(x.unsqueeze(0), pad_widths, mode='replicate').squeeze(0)

    slices_list = []
    for step in steps:
        idx = ([slice(None)] * non_sp
               + [slice(1 + s, sh + 1 + s) for s, sh in zip(step, spatial)])
        slices_list.append(x_pad[tuple(idx)])

    return torch.stack(slices_list)   # (nsteps, *x_shape)


def _make_seam_mask(nsteps: int,
                    spatial,
                    steps_list: List[Sequence[int]],
                    seam_ends: Sequence[int],
                    seam_starts: Sequence[int],
                    device) -> torch.Tensor:
    """Build a (nsteps, *spatial) bool mask for seam-row correction.

    Returns True at position (k, row, ...) when step k crosses a concatenation
    seam at that row, so the caller can replace the pad+stack value with the
    self-value via ``torch.where``.

    Uses only scalar indexing (no list / fancy indexing that
    would trigger Metal Internal Errors).

    Args:
        nsteps:      number of neighbour steps.
        spatial:     spatial shape tuple, e.g. (H, W).
        steps_list:  list of step tuples, length nsteps.
        seam_ends:   row indices that are the *last* row of a tile.
                     Downward steps (step[0]>0) at these rows need clamping.
        seam_starts: row indices that are the *first* row of the next tile.
                     Upward steps (step[0]<0) at these rows need clamping.
        device:      torch device.

    Returns:
        mask: (nsteps, *spatial) bool tensor.
    """
    mask = torch.zeros((nsteps,) + tuple(spatial), dtype=torch.bool, device=device)
    for k, step in enumerate(steps_list):
        s0 = step[0]
        if s0 > 0:          # downward step: last row of each tile crosses seam
            for se in seam_ends:
                mask[k, se] = True
        elif s0 < 0:        # upward step: first row of next tile crosses seam
            for ss in seam_starts:
                mask[k, ss] = True
    return mask


# @torch.no_grad()
def update_torch(a, f, fsq):
    # Turns out we can just avoid a ton of individual if/else by evaluating the update function
    # for every upper limit on the sorted pairs. I do this by pieces using cumsum. The radicand
    # being nonegative sets the upper limit on the sorted pairs, so we simply select the largest
    # upper limit that works. I also put a couple of the indexing tensors outside of the loop.
    """Update function for solving the Eikonal equation."""
    d = a.shape[0]  # d acutally needed to be the number of elements being compared, not dimension
    if d == 2:
        # Special-case: for exactly 2 inputs, torch.sort is ~70x slower than min/max.
        # For sorted [a0, a1]: mask (a - a[-1]) < f is always True (a[0]<=a[1], f>0),
        # so am=a, sum_a=a0+a1, sum_a2=a0^2+a1^2 — identical to the general path.
        a0 = torch.minimum(a[0], a[1])
        a1 = torch.maximum(a[0], a[1])
        sum_a = a0 + a1
        sum_a2 = a0 * a0 + a1 * a1
        return 0.5 * (sum_a + torch.sqrt(torch.clamp(sum_a * sum_a - 2 * (sum_a2 - fsq), min=0)))
    a, _ = torch.sort(a, dim=0)  # sorting was the source of the small artifact bug
    am = a * ((a - a[-1]) < f)
    sum_a = am.sum(dim=0)
    sum_a2 = (am ** 2).sum(dim=0)
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
             n_iter: int,
             omni: bool,
             verbose: bool):
    """Update function for iterating the Eikonal/diffusion equation.

    Early-exit on convergence: check err every CHECK_EVERY iterations via
    .item() to avoid per-iteration CPU-GPU syncs while still exiting early.
    """

    eps = 1e-3
    CHECK_EVERY = 10  # .item() forces a GPU sync
    r = central_inds

    if verbose:
        print('n_iter is', n_iter)

    if omni:
        T0 = T.clone()

        for t in range(n_iter):
            Tneigh = T[neigh_inds]
            Tneigh *= isneigh
            T = eikonal_update_torch(Tneigh, r, d, inds, fact)

            if t < 1:   # one-time initial smoothing on first iteration
                Tneigh = T[neigh_inds]
                Tneigh *= isneigh
                T = Tneigh.mean(dim=0)

            err = (T - T0).square().mean()
            T0.copy_(T)

            if (t % CHECK_EVERY) == (CHECK_EVERY - 1) and err.item() < eps:
                break

    else:
        for t in range(n_iter):
            T[centroid_inds] += 1
            Tneigh = T[neigh_inds]
            Tneigh *= isneigh
            T = Tneigh.mean(dim=0)
        T = torch.log(1. + T)

    if verbose:
        print(f'_iterate ran {t + 1} iterations')

    return T


@_maybe_compile
def _eikonal_step(T: torch.Tensor,
                  affinity: torch.Tensor,
                  seam_mask,          # Tensor or None — compile specialises per case
                  fg_float: torch.Tensor,
                  d: torch.Tensor,
                  inds: List[torch.Tensor],
                  fact: torch.Tensor,
                  steps_list) -> torch.Tensor:
    """One compiled Eikonal grid step (no seam / smoothing logic).

    Called in a tight Python loop from ``_iterate_grid``.  All Python control
    flow (seam_mask check, steps_list loop inside _pad_and_stack_neighbors) is
    resolved at compile time, so the runtime inner loop is pure GPU with no
    Python overhead.
    """
    Tneigh = _pad_and_stack_neighbors(T, steps_list)
    if seam_mask is not None:
        Tneigh = torch.where(seam_mask, T.unsqueeze(0), Tneigh)
    Tneigh.mul_(affinity)
    T_new = eikonal_update_torch(Tneigh, None, d, inds, fact)
    T_new.mul_(fg_float)
    return T_new


def _iterate_grid(T: torch.Tensor,
                  affinity: torch.Tensor,
                  steps_list: List[Sequence[int]],
                  idx_center: int,
                  d: torch.Tensor,
                  inds: List[torch.Tensor],
                  fact: torch.Tensor,
                  n_iter: int,
                  omni: bool,
                  alpha: float = 1.0,
                  seam_ends: Sequence[int] = (),
                  seam_starts: Sequence[int] = (),
                  verbose: bool = False) -> torch.Tensor:
    """Full-grid Eikonal iteration — zero .item() calls.

    The Eikonal update is a monotone contraction: T increases toward the
    distance field and stabilises at the fixed point.  Running past convergence
    leaves T unchanged, so we simply iterate n_iter times and return T.

    No convergence check, no snap_T, no T0 clone.  Error is kept as a GPU
    tensor (never read in the loop) and printed once after the loop if verbose.

    Args:
        T:          (*spatial) float tensor, 1.0 inside masks, 0.0 outside.
        affinity:   (nsteps, *spatial) bool tensor from compute_affinity_gpu.
        steps_list: Python list of step tuples, length nsteps.
        idx_center: index of the (0,…,0) step — no self-connections.
        d:          scalar tensor, spatial dimensionality.
        inds:       connectivity index lists from kernel_setup.
        fact:       distance factors from kernel_setup.
        n_iter:     number of iterations to run.
        omni:       True → Eikonal, False → not implemented here.
        alpha:      unused (kept for API compat); damping removed since the
                    Eikonal update is inherently monotone.
    """
    if not omni:
        raise NotImplementedError("_iterate_grid only supports omni=True")

    fg_float = affinity.any(0).float()   # (*spatial) float

    # Seam mask: precomputed once, resolved at compile time inside _eikonal_step.
    seam_mask = None
    if seam_ends or seam_starts:
        seam_mask = _make_seam_mask(len(steps_list), T.shape, steps_list,
                                    seam_ends, seam_starts, T.device)

    # ── t = 0: one Eikonal step then smooth with neighbour mean ─────────────
    # Done outside the main loop so the loop body has no Python conditionals
    # and torch.compile can fuse the entire iteration into a single kernel.
    Tneigh = _pad_and_stack_neighbors(T, steps_list)
    if seam_mask is not None:
        Tneigh = torch.where(seam_mask, T.unsqueeze(0), Tneigh)
    Tneigh.mul_(affinity)
    T_new = eikonal_update_torch(Tneigh, None, d, inds, fact)

    Tneigh2 = _pad_and_stack_neighbors(T_new, steps_list)
    if seam_mask is not None:
        Tneigh2 = torch.where(seam_mask, T_new.unsqueeze(0), Tneigh2)
    Tneigh2.mul_(affinity)
    T = Tneigh2.mean(dim=0).mul_(fg_float)

    # ── t = 1 … n_iter-1: fixed-point loop with early convergence check ─────
    # Match omnipose's sparse _iterate: stop when MSE < eps (1e-3).
    # Check every CHECK_EVERY iterations to amortise the GPU sync cost.
    eps = 1e-3
    CHECK_EVERY = 10
    T0 = T.clone()
    for t in range(1, n_iter):
        T_new = _eikonal_step(T, affinity, seam_mask, fg_float, d, inds, fact, steps_list)
        T = T_new

        if (t % CHECK_EVERY) == (CHECK_EVERY - 1):
            err = (T - T0).square().mean()
            if err.item() < eps:
                break
            T0.copy_(T)

    if verbose:
        final_err = (T - T0).square().mean().item() if t >= CHECK_EVERY else 0.0
        print(f'_iterate_grid: {t + 1} iters, final MSE={final_err:.2e}')
    return T


def _gradient_grid(T: torch.Tensor,
                   affinity: torch.Tensor,
                   steps: torch.Tensor,
                   fact: torch.Tensor,
                   inds: List[torch.Tensor],
                   seam_ends: Sequence[int] = (),
                   seam_starts: Sequence[int] = (),
                   steps_list=None) -> torch.Tensor:
    """Full-grid gradient of T using torch.roll, matching _gradient semantics.

    Args:
        T:          (*spatial) float tensor (solved distance field).
        affinity:   (nsteps, *spatial) bool tensor.
        steps:      (nsteps, ndim) int tensor.
        fact:       (nfact,) float tensor from kernel_setup.
        inds:       connectivity index lists from kernel_setup.
        steps_list: pre-converted list-of-tuples for steps (avoids GPU sync).

    Returns:
        mu: (ndim, *spatial) flow field tensor.
    """
    spatial = T.shape
    ndim = len(spatial)
    if steps_list is None:
        steps_list = steps.tolist()  # GPU→CPU sync; pass steps_list to avoid
    n_axes = len(fact) - 1

    seam_mask = None
    if seam_ends or seam_starts:
        seam_mask = _make_seam_mask(len(steps_list), spatial, steps_list,
                                    seam_ends, seam_starts, T.device)

    # Vectorized neighbour gather for T: one F.pad + slice-stack.
    rolled_T = _pad_and_stack_neighbors(T, steps_list)
    if seam_mask is not None:
        rolled_T = torch.where(seam_mask, T.unsqueeze(0), rolled_T)
    rolled_T = rolled_T * affinity                   # (nsteps, *spatial)

    finite_differences = torch.zeros(
        (n_axes, ndim) + spatial, device=T.device, dtype=T.dtype
    )

    for ax, (ind, f) in enumerate(zip(inds[1:], fact[1:])):
        vals = rolled_T[ind]                         # (len(ind), *spatial)
        mid = len(ind) // 2
        r = torch.arange(mid, device=T.device)
        vecs = steps[ind].float()                    # (len(ind), ndim)
        uvecs = (vecs[-(r + 1)] - vecs[r]).T         # (ndim, mid)

        diff = vals[-(r + 1)] - vals[r]              # (mid, *spatial)
        flat_diff = diff.flatten(1)                  # (mid, prod_spatial)
        result = torch.matmul(uvecs, flat_diff) / (2.0 * f) ** 2
        finite_differences[ax] = result.view(ndim, *spatial)

    mu = finite_differences.mean(dim=0)              # (ndim, *spatial)

    # Coherence weighting: favour neighbour flows aligned with local flow.
    rolled_mu = _pad_and_stack_neighbors(mu, steps_list)
    if seam_mask is not None:
        # seam_mask is (nsteps, *spatial); unsqueeze(1) broadcasts over ndim.
        rolled_mu = torch.where(seam_mask.unsqueeze(1), mu.unsqueeze(0), rolled_mu)
    # (nsteps, ndim, *spatial)

    weight = (rolled_mu * mu.unsqueeze(0)).sum(dim=1).abs()  # (nsteps, *spatial)
    weight = weight * affinity
    wsum = weight.sum(dim=0)                         # (*spatial)

    numerator = (rolled_mu * weight.unsqueeze(1)).sum(dim=0)  # (ndim, *spatial)
    return torch.where(
        wsum.unsqueeze(0) != 0,
        numerator / wsum.unsqueeze(0),
        torch.zeros_like(mu),
    )


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

@_maybe_compile
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


divergence_torch = divergence  # backward-compat alias
