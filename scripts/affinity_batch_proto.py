#!/usr/bin/env python
"""
Prototype batched affinity computation using vmap over _get_affinity_torch.
This does NOT change production code; it's a standalone experiment.
"""
from __future__ import annotations

import time
from typing import Tuple

import torch

from omnirefactor.core.affinity import _get_affinity_torch
from omnirefactor.utils.neighbor import kernel_setup, get_supporting_inds
from ocdkit.array import torch_norm
from torchvf.numerics import interp_vf, ivp_solver


def _make_inputs(batch: int, dim: int, size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
    # flow/dist tensors shaped (B, D, H, W) or (B, D, *dims)
    dims = (size,) * dim
    flow = torch.randn((batch, dim, *dims), device=device)
    dist = torch.rand((batch, *dims), device=device)
    iscell = dist > 0.2

    coords = [torch.arange(0, l, device=device) for l in dims]
    mesh = torch.meshgrid(coords, indexing="ij")
    initial = torch.stack(mesh, dim=0).float()
    initial = initial.unsqueeze(0).repeat(batch, 1, *([1] * dim))

    # small displacement toward flow direction
    mag = torch_norm(flow, dim=1, keepdim=True)
    flow_norm = torch.where(mag > 0, flow / mag, flow)
    final = initial + 0.5 * flow_norm
    return initial, final, flow, dist, iscell


def affinity_loop(initial, final, flow, dist, iscell, steps, fact, inds, supporting_inds, niter, device):
    out = []
    for b in range(flow.shape[0]):
        out.append(
            _get_affinity_torch(
                initial[b : b + 1],
                final[b : b + 1],
                flow[b : b + 1],
                dist[b : b + 1],
                iscell[b : b + 1],
                steps,
                fact,
                inds,
                supporting_inds,
                niter,
                device=device,
            ).squeeze(0)
        )
    return torch.stack(out, dim=0)


def affinity_pair_loop(initial, flow_pred, dist_pred, flow_gt, dist_gt, iscell,
                       steps, fact, inds, supporting_inds, niter, device):
    # baseline: compute pred/gt separately
    vf_pred = interp_vf(flow_pred, mode="nearest_batched")
    fp_pred = ivp_solver(vf_pred, initial, dx=0.2, n_steps=2, solver="euler")[-1]
    ag_pred = _get_affinity_torch(
        initial, fp_pred, flow_pred, dist_pred, iscell,
        steps, fact, inds, supporting_inds, niter, device=device,
    )

    vf_gt = interp_vf(flow_gt, mode="nearest_batched")
    fp_gt = ivp_solver(vf_gt, initial, dx=0.2, n_steps=2, solver="euler")[-1]
    ag_gt = _get_affinity_torch(
        initial, fp_gt, flow_gt, dist_gt, iscell,
        steps, fact, inds, supporting_inds, niter, device=device,
    )
    return (fp_pred, ag_pred), (fp_gt, ag_gt)


def affinity_pair_batched(initial, flow_pred, dist_pred, flow_gt, dist_gt, iscell,
                          steps, fact, inds, supporting_inds, niter, device):
    # concatenate pred/gt along batch dim (dim=0)
    flow_all = torch.cat([flow_pred, flow_gt], dim=0)
    dist_all = torch.cat([dist_pred, dist_gt], dim=0)
    iscell_all = torch.cat([iscell, iscell], dim=0)
    init_all = initial.repeat(2, 1, *([1] * (initial.ndim - 2)))

    vf_all = interp_vf(flow_all, mode="nearest_batched")
    fp_all = ivp_solver(vf_all, init_all, dx=0.2, n_steps=2, solver="euler")[-1]
    ag_all = _get_affinity_torch(
        init_all, fp_all, flow_all, dist_all, iscell_all,
        steps, fact, inds, supporting_inds, niter, device=device,
    )

    fp_pred, fp_gt = torch.chunk(fp_all, 2, dim=0)
    ag_pred, ag_gt = torch.chunk(ag_all, 2, dim=1)
    return (fp_pred, ag_pred), (fp_gt, ag_gt)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(1)

    batch = 2
    dim = 2
    size = 32
    niter = 10

    steps, inds, idx, fact, sign = kernel_setup(dim)
    supporting_inds = get_supporting_inds(steps)

    initial, _, flow_pred, dist_pred, iscell = _make_inputs(batch, dim, size, device)
    _, _, flow_gt, dist_gt, _ = _make_inputs(batch, dim, size, device)

    t0 = time.time()
    (fp_pred_ref, ag_pred_ref), (fp_gt_ref, ag_gt_ref) = affinity_pair_loop(
        initial, flow_pred, dist_pred, flow_gt, dist_gt, iscell,
        steps, fact, inds, supporting_inds, niter, device
    )
    t1 = time.time()

    t2 = time.time()
    (fp_pred_b, ag_pred_b), (fp_gt_b, ag_gt_b) = affinity_pair_batched(
        initial, flow_pred, dist_pred, flow_gt, dist_gt, iscell,
        steps, fact, inds, supporting_inds, niter, device
    )
    t3 = time.time()

    def _report(name, a, b):
        if a.dtype == torch.bool and b.dtype == torch.bool:
            diff = (a ^ b).float()
        else:
            diff = (a - b).abs()
        print(f"{name} max abs diff:", diff.max().item())
        print(f"{name} mean abs diff:", diff.mean().item())

    print("device:", device)
    _report("fp_pred", fp_pred_ref, fp_pred_b)
    _report("ag_pred", ag_pred_ref, ag_pred_b)
    _report("fp_gt", fp_gt_ref, fp_gt_b)
    _report("ag_gt", ag_gt_ref, ag_gt_b)
    print(f"loop:  {(t1 - t0)*1000:.2f} ms")
    print(f"batched:  {(t3 - t2)*1000:.2f} ms")


if __name__ == "__main__":
    main()
