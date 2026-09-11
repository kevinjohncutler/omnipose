"""Shared GT-pipeline + per-voxel loss machinery for the 2D/3D loss verification.

This module deliberately calls the *shipped* omnipose entry points
(`masks_to_flows_batch`, `batch_labels`, `omnipose.core.loss.loss`,
`metrics.loss.AffinityLoss`, `divergence_torch`, `torch_norm`) so that the
figures and numbers reflect exactly what training sees. The only thing the
harness re-implements is the *per-voxel* (pre-reduction) form of each loss
term; every such map is validated by checking that its reduction reproduces
the authoritative scalar returned by the shipped loss to ~1e-5.
"""
from __future__ import annotations
import numpy as np
import torch

from omnipose.core.flows import masks_to_flows_batch, batch_labels
from omnipose.core.fields import divergence_torch
from omnipose.core.imports import torch_norm
import omnipose.metrics.loss as oloss
from omnipose.utils import kernel_setup, get_supporting_inds


# --------------------------------------------------------------------------
# Geometry: two cubes / squares sharing exactly one face / edge
# --------------------------------------------------------------------------
def two_cubes(dim, side=16, margin=10):
    """Label volume with two `side`-wide cubes touching along axis 0.

    label 1 occupies axis0 in [margin, margin+side)
    label 2 occupies axis0 in [margin+side, margin+2*side)
    both share the same extent [margin, margin+side) on the remaining axes.
    Contact face sits between index margin+side-1 and margin+side on axis 0.
    """
    L = margin * 2 + side          # transverse extent
    L0 = margin * 2 + 2 * side     # axis-0 extent (fits both cubes)
    shape = (L0,) + (L,) * (dim - 1)
    masks = np.zeros(shape, dtype=np.int32)
    sl_t = tuple(slice(margin, margin + side) for _ in range(dim - 1))
    masks[(slice(margin, margin + side),) + sl_t] = 1
    masks[(slice(margin + side, margin + 2 * side),) + sl_t] = 2
    contact = margin + side        # first axis-0 index of cube 2
    return masks, contact


def four_cubes(dim, side=14, margin=8):
    """Four identical cubes/squares in a 2x2 grid (labels 1..4), all sharing
    faces, with a 4-way junction at the centre.

    2D: 2x2 grid in (axis0, axis1).
    3D: 2x2x1 block — grid in (axis1, axis2), single slab of thickness `side`
        on axis0 (so every tile is a true cube).
    The four tiles are reflections of one another, so a symmetric perturbation
    must produce identical per-label loss iff label handling is symmetric.
    """
    quad = [(0, 0), (0, 1), (1, 0), (1, 1)]    # label = index+1
    if dim == 2:
        Lt = margin * 2 + 2 * side
        masks = np.zeros((Lt, Lt), dtype=np.int32)
        for li, (i, j) in enumerate(quad, start=1):
            masks[margin + i * side:margin + (i + 1) * side,
                  margin + j * side:margin + (j + 1) * side] = li
        center_plane = None
    else:
        L0 = margin * 2 + side
        Lt = margin * 2 + 2 * side
        masks = np.zeros((L0, Lt, Lt), dtype=np.int32)
        for li, (i, j) in enumerate(quad, start=1):
            masks[margin:margin + side,
                  margin + i * side:margin + (i + 1) * side,
                  margin + j * side:margin + (j + 1) * side] = li
        center_plane = margin + side // 2      # axis-0 plane through cube centres
    return masks, center_plane


# --------------------------------------------------------------------------
# Ground-truth label tensor via the real pipeline
# --------------------------------------------------------------------------
def build_gt(masks, dim, device, nclasses=None):
    """Return (lbl, info) using the exact training GT path.

    lbl channels (see core.flows.batch_labels):
        0 mask labels, 1 cellmask, 2 boundary, 3 distance (bg=-5),
        4 weight, [-dim:] flow*5
    """
    if nclasses is None:
        nclasses = dim + 2
    masks_np = masks[None].astype(np.int64)          # (B=1, *spatial)
    tyx = masks.shape
    out = masks_to_flows_batch(masks_np, links=[None], device=device,
                               omni=True, dim=dim, affinity_field=False)
    X = out[:-4]                                     # labels, boundaries, T, mu
    slices = out[-4]
    m, bd, T, mu = [torch.stack([x[(Ellipsis,) + slc] for slc in slices]) for x in X]
    lbl = batch_labels(m, bd, T, mu, tyx, dim=dim, nclasses=nclasses, device=device)
    info = dict(nclasses=nclasses, dim=dim, tyx=tyx)
    return lbl, info


def unpack_gt(lbl, dim):
    """Pull the named GT fields out of the lbl tensor."""
    return dict(
        cellmask=(lbl[:, 1] > 0),
        boundary=lbl[:, 2],
        dist=lbl[:, 3],
        weight=lbl[:, 4].detach(),
        veci=lbl[:, -dim:],
    )


def make_pred(lbl, dim, scenario, device):
    """Construct a predicted network output `y` (B, dim+2, *spatial).

    y[:, :dim]    flow      y[:, dim] distance      y[:, dim+1] boundary logit
    """
    g = unpack_gt(lbl, dim)
    veci, dist, boundary, cm = g['veci'], g['dist'], g['boundary'], g['cellmask']
    B = lbl.shape[0]
    spatial = lbl.shape[2:]
    y = torch.zeros((B, dim + 2) + tuple(spatial), device=device, dtype=lbl.dtype)
    flow = veci.clone()
    dt = dist.clone()
    BIG = 20.0
    bd_logit = (boundary * 2 - 1) * BIG               # sigmoid -> boundary

    # cube-2 selector (label==2) for the spatially-localised perturbations
    cube2 = (lbl[:, 0] == 2)
    cube1 = (lbl[:, 0] == 1)

    if scenario == 'perfect':
        pass
    elif scenario == 'flow_zero_all':
        # zero predicted flow inside EVERY cell -> must light all labels equally
        flow = flow * (~cm).unsqueeze(1)
    elif scenario.startswith('flow_zero_label'):
        k = int(scenario.rsplit('label', 1)[1])
        sel = (lbl[:, 0] == k)
        flow = flow * (~sel).unsqueeze(1)
    elif scenario == 'flow_zero_cube2':
        flow = flow * (~cube2).unsqueeze(1)
    elif scenario == 'flow_flip_cube2':
        flow = torch.where(cube2.unsqueeze(1), -flow, flow)
    elif scenario == 'dist_offset':
        dt = dt + cm.float() * 2.0                    # +2 inside all cells
    elif scenario == 'dist_bump':
        # localised gaussian bump centred in cube 1 -> nonzero gradient at its edges
        coords = torch.meshgrid(*[torch.arange(s, device=device, dtype=lbl.dtype)
                                  for s in spatial], indexing='ij')
        c = [s / 2 for s in spatial]
        c[0] = c[0] * 0.5                             # push toward cube 1
        r2 = sum((coords[i] - c[i]) ** 2 for i in range(dim))
        bump = 3.0 * torch.exp(-r2 / (2 * (spatial[-1] * 0.12) ** 2))
        dt = dt + bump.unsqueeze(0) * cm.float()
    else:
        raise ValueError(scenario)

    y[:, :dim] = flow
    y[:, dim] = dt
    y[:, dim + 1] = bd_logit
    return y


# --------------------------------------------------------------------------
# Per-voxel (pre-reduction) loss maps. Each returns a (B, *spatial) tensor.
# Reductions are checked against the shipped scalar in run_all().
# --------------------------------------------------------------------------
def flow_mse_map(y, lbl, dim, g):
    flow = y[:, :dim]
    err = (flow - g['veci']) ** 2                      # (B,dim,*sp)
    return g['weight'].unsqueeze(1).expand_as(err) * err  # keep component axis


def dist_map(y, lbl, dim, g):
    dt = y[:, dim]
    return g['weight'] * (dt - g['dist']) ** 2


def ssl_map(y, lbl, dim, g, eps=1e-12):
    x = y[:, :dim]
    yv = g['veci']
    magX = torch_norm(x, dim=1)
    magY = torch_norm(yv, dim=1)
    denom = magX * magY
    dot = (x * yv).sum(dim=1)
    mask = g['dist'] > 0
    cossq = torch.where(mask, dot / (denom + eps), torch.ones_like(dot)) ** 2
    return g['weight'] * (cossq - 1.0) ** 2            # WMSE(cossq, 1, w)


def norm_map(y, lbl, dim, g):
    magX = torch_norm(y[:, :dim], dim=1)
    magY = torch_norm(g['veci'], dim=1)
    return (magX - magY) ** 2


def derivative_map(y, lbl, dim, g):
    """Per-voxel weighted grad error (the bd_loss term)."""
    dt = y[:, dim].unsqueeze(1)
    Y = g['dist'].unsqueeze(1)
    w = g['weight'].unsqueeze(1)
    mask = g['cellmask'].unsqueeze(1)
    spatial_dims = dt.ndim - 2
    axes = list(range(-spatial_dims, 0))
    dy = torch.stack(torch.gradient(dt, dim=axes)).transpose(0, 1)
    dY = torch.stack(torch.gradient(Y, dim=axes)).transpose(0, 1)
    grad_err = torch.sum(((dy - dY) / 5.0) ** 2, dim=1)   # (B,*sp)
    wmap = grad_err * w.expand_as(grad_err)
    valid = mask.expand_as(grad_err).bool()
    return wmap.masked_fill(~valid, 0.0), valid


def divcorr_map(y, lbl, dim, g):
    div_gt = divergence_torch(g['veci'])
    div_pred = divergence_torch(y[:, :dim])
    return (div_gt - div_pred) ** 2


def affinity_maps(y, lbl, dim, g, device):
    """Replicate AffinityLoss.forward(mode='all') keeping per-voxel maps.

    Returns dict with 'E' (endpoint err, B*sp), 'A' (affinity-graph err, B*sp),
    plus scalars 'A','E','B' for cross-check against the shipped loss.
    """
    steps, inds, idx, fact, sign = kernel_setup(dim)
    supporting_inds = get_supporting_inds(steps)
    flow_pred = y[:, :dim]
    dist_pred = y[:, dim]
    flow_gt = g['veci']
    dist_gt = g['dist']

    foreground = torch.ones_like(dist_pred, dtype=torch.bool)
    shape = flow_pred.shape
    B = shape[0]
    dims = shape[-dim:]
    coords = [torch.arange(0, l, device=device) for l in dims]
    mesh = torch.meshgrid(coords, indexing="ij")
    init_shape = [B, 1] + ([1] * len(dims))
    initial_points = torch.stack(mesh, dim=0).repeat(init_shape).float()
    niter = 10

    flow_all = torch.cat([flow_pred, flow_gt], dim=0)
    ip_all = torch.cat([initial_points, initial_points], dim=0)
    # use the shipped (corrected) flow-following so maps match the real loss
    fp_all = oloss.ivp_euler_batched(flow_all, ip_all, dx=np.sqrt(dim) / 5, n_steps=2)
    fp_pred, fp_gt = torch.chunk(fp_all, 2, dim=0)

    ags = []
    for f, d, fp in zip([flow_pred, flow_gt], [dist_pred, dist_gt], [fp_pred, fp_gt]):
        ag = oloss._get_affinity_torch(initial_points, fp, f / 5., d, foreground,
                                       steps, fact, inds, supporting_inds, niter,
                                       device=device)
        ags.append(ag * 1.0)

    ag_pred, ag_gt = ags                                # (nsteps, B, *sp)
    nsteps = ag_pred.shape[0]
    # endpoint error per voxel: sum over coordinate dim (fp is (B, dim, *sp))
    E_map = ((fp_pred - fp_gt) ** 2).sum(dim=1)         # (B,*sp)
    # affinity-graph error per voxel: sum over step axis (axis 0)
    A_map = ((ag_pred - ag_gt) ** 2).sum(dim=0)         # (B,*sp)
    return dict(E_map=E_map, A_map=A_map, nsteps=nsteps)
