#!/usr/bin/env python
"""Compare loop vs batched affinity path on real training data."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from omnirefactor.core.affinity import _get_affinity_torch
from omnirefactor.utils.neighbor import kernel_setup, get_supporting_inds
from omnirefactor.transforms.vector import torch_norm
from torchvf.numerics import interp_vf, ivp_solver


def _load_first_sample(data_dir: Path):
    # Expect images and masks with suffix filters used in CLI
    imgs = sorted(data_dir.glob("*_img.*"))
    masks = sorted(data_dir.glob("*_masks.*"))
    if not imgs or not masks:
        raise RuntimeError("No *_img.* or *_masks.* files found in dataset")

    # simple tif/png loader via imageio
    import imageio.v3 as iio

    img = iio.imread(imgs[0])
    msk = iio.imread(masks[0])
    return img, msk


def _make_tensors(img, mask, device):
    # Flow/distance placeholders: mimic shapes from training pipeline
    # Use mask to create dist/flow-like tensors
    mask = (mask > 0).astype(np.float32)
    dist = torch.from_numpy(mask).to(device)
    dist = dist.unsqueeze(0)  # B=1

    # fake flow: gradients of mask
    gy, gx = np.gradient(mask)
    flow = np.stack([gy, gx], axis=0).astype(np.float32)
    flow = torch.from_numpy(flow).to(device)
    flow = flow.unsqueeze(0)  # B=1, D=2

    iscell = dist > 0

    # initial points
    dims = flow.shape[-2:]
    coords = [torch.arange(0, l, device=device) for l in dims]
    mesh = torch.meshgrid(coords, indexing="ij")
    initial = torch.stack(mesh, dim=0).float()
    initial = initial.unsqueeze(0)

    return flow, dist, iscell, initial


def affinity_loop(initial, flow_pred, dist_pred, flow_gt, dist_gt, iscell,
                  steps, fact, inds, supporting_inds, niter, device):
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


def affinity_batched(initial, flow_pred, dist_pred, flow_gt, dist_gt, iscell,
                     steps, fact, inds, supporting_inds, niter, device):
    flow_all = torch.cat([flow_pred, flow_gt], dim=0)
    dist_all = torch.cat([dist_pred, dist_gt], dim=0)
    iscell_all = torch.cat([iscell, iscell], dim=0)
    init_all = torch.cat([initial, initial], dim=0)

    vf_all = interp_vf(flow_all, mode="nearest_batched")
    fp_all = ivp_solver(vf_all, init_all, dx=0.2, n_steps=2, solver="euler")[-1]
    ag_all = _get_affinity_torch(
        init_all, fp_all, flow_all, dist_all, iscell_all,
        steps, fact, inds, supporting_inds, niter, device=device,
    )

    fp_pred, fp_gt = torch.chunk(fp_all, 2, dim=0)
    ag_pred, ag_gt = torch.chunk(ag_all, 2, dim=1)
    return (fp_pred, ag_pred), (fp_gt, ag_gt)


def _report(name, a, b):
    if a.dtype == torch.bool and b.dtype == torch.bool:
        diff = (a ^ b).float()
    else:
        diff = (a - b).abs()
    print(f"{name} max abs diff: {diff.max().item()}")
    print(f"{name} mean abs diff: {diff.mean().item()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dim = 2
    niter = 10
    steps, inds, idx, fact, sign = kernel_setup(dim)
    supporting_inds = get_supporting_inds(steps)

    img, mask = _load_first_sample(Path(args.data))
    flow_pred, dist_pred, iscell, initial = _make_tensors(img, mask, device)
    # use different tensor for gt
    flow_gt = flow_pred * 0.9
    dist_gt = dist_pred * 1.05

    t0 = time.time()
    (fp_pred_ref, ag_pred_ref), (fp_gt_ref, ag_gt_ref) = affinity_loop(
        initial, flow_pred, dist_pred, flow_gt, dist_gt, iscell,
        steps, fact, inds, supporting_inds, niter, device
    )
    t1 = time.time()

    t2 = time.time()
    (fp_pred_b, ag_pred_b), (fp_gt_b, ag_gt_b) = affinity_batched(
        initial, flow_pred, dist_pred, flow_gt, dist_gt, iscell,
        steps, fact, inds, supporting_inds, niter, device
    )
    t3 = time.time()

    print("device:", device)
    _report("fp_pred", fp_pred_ref, fp_pred_b)
    _report("ag_pred", ag_pred_ref, ag_pred_b)
    _report("fp_gt", fp_gt_ref, fp_gt_b)
    _report("ag_gt", ag_gt_ref, ag_gt_b)
    print(f"loop:  {(t1 - t0)*1000:.2f} ms")
    print(f"batched:  {(t3 - t2)*1000:.2f} ms")


if __name__ == "__main__":
    main()
