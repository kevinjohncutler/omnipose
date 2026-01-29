#!/usr/bin/env python
"""Compare loop vs batched affinity inside real training batch (omnirefactor)."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from omnirefactor import core
from omnirefactor.models import OmniModel
from omnirefactor.utils.neighbor import kernel_setup, get_supporting_inds
from omnirefactor.metrics.loss import BatchMeanMSE, BatchMeanBSE
from omnirefactor import io as io_mod
from torchvf.numerics import interp_vf, ivp_solver


def affinity_loop(initial_points, flow_pred, dist_pred, flow_gt, dist_gt, foreground,
                  steps, fact, inds, supporting_inds, niter, device, dim):
    ags, fps, bds = [], [], []
    for f, d in zip([flow_pred, flow_gt], [dist_pred, dist_gt]):
        vf = interp_vf(f, mode="nearest_batched")
        final_points = ivp_solver(
            vf, initial_points, dx=(dim ** 0.5) / 5, n_steps=2, solver="euler"
        )[-1]
        fps.append(final_points)
        affinity_graph = core.affinity._get_affinity_torch(
            initial_points, final_points, f / 5.0, d, foreground,
            steps, fact, inds, supporting_inds, niter, device=device
        )
        ags.append(affinity_graph * 1.0)
        csum = torch.sum(affinity_graph, axis=1)
        bds.append(1.0 * torch.logical_and(csum < (3 ** dim - 1), csum >= dim))
    return ags, fps, bds


def affinity_batched(initial_points, flow_pred, dist_pred, flow_gt, dist_gt, foreground,
                     steps, fact, inds, supporting_inds, niter, device, dim):
    flow_all = torch.cat([flow_pred, flow_gt], dim=0)
    dist_all = torch.cat([dist_pred, dist_gt], dim=0)
    foreground_all = torch.cat([foreground, foreground], dim=0)
    initial_all = torch.cat([initial_points, initial_points], dim=0)

    vf_all = interp_vf(flow_all, mode="nearest_batched")
    final_all = ivp_solver(
        vf_all, initial_all, dx=(dim ** 0.5) / 5, n_steps=2, solver="euler"
    )[-1]
    affinity_all = core.affinity._get_affinity_torch(
        initial_all, final_all, flow_all / 5.0, dist_all, foreground_all,
        steps, fact, inds, supporting_inds, niter, device=device
    )
    final_pred, final_gt = torch.chunk(final_all, 2, dim=0)
    affinity_pred, affinity_gt = torch.chunk(affinity_all, 2, dim=1)
    ags = [affinity_pred * 1.0, affinity_gt * 1.0]
    fps = [final_pred, final_gt]
    bds = []
    for affinity_graph in ags:
        csum = torch.sum(affinity_graph, axis=1)
        bds.append(1.0 * torch.logical_and(csum < (3 ** dim - 1), csum >= dim))
    return ags, fps, bds


def report(tag, a, b):
    if a.dtype == torch.bool and b.dtype == torch.bool:
        diff = (a ^ b).float()
    else:
        diff = (a - b).abs()
    print(f"{tag} max abs diff: {diff.max().item()}")
    print(f"{tag} mean abs diff: {diff.mean().item()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    # Use CPU to avoid torch.compile/MPS float64 limitations in labels_to_flows
    device = torch.device("cpu")
    dim = 2
    steps, inds, idx, fact, sign = kernel_setup(dim)
    supporting_inds = get_supporting_inds(steps)

    # Build a minimal model and training batch (uses same pipeline as training)
    model = OmniModel(gpu=device.type != "cpu", omni=True, use_torch=True, nclasses=2, dim=2)
    data, labels, links, files, _test_images, _test_labels, _test_links, _test_names = (
        io_mod.load_train_test_data(
            args.dir, image_filter="_img", mask_filter="_masks", omni=True, do_links=False
        )
    )

    # Use the first training batch
    img = torch.from_numpy(data[0]).float().to(device)
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        img = img.unsqueeze(0)

    # forward pass
    y = model.net(img)[0]

    # fake GT tensors from labels
    lbl = labels[0]
    flows = core.labels_to_flows([lbl], device=device, omni=True)
    flows = torch.from_numpy(flows[0]).float().to(device)
    flow_gt = flows[:dim].unsqueeze(0)
    dist_gt = flows[dim].unsqueeze(0)

    # pred tensors
    flow_pred = y[:, :dim]
    dist_pred = y[:, dim:dim+1].squeeze(1)

    foreground = torch.logical_or(dist_pred >= 0, dist_gt >= 0)

    # initial points
    dims = flow_pred.shape[-dim:]
    coords = [torch.arange(0, l, device=device) for l in dims]
    mesh = torch.meshgrid(coords, indexing="ij")
    initial = torch.stack(mesh, dim=0).float()
    initial = initial.unsqueeze(0)

    niter = 10

    ags_loop, fps_loop, bds_loop = affinity_loop(
        initial, flow_pred, dist_pred, flow_gt, dist_gt, foreground,
        steps, fact, inds, supporting_inds, niter, device, dim
    )
    ags_b, fps_b, bds_b = affinity_batched(
        initial, flow_pred, dist_pred, flow_gt, dist_gt, foreground,
        steps, fact, inds, supporting_inds, niter, device, dim
    )

    report("fp_pred", fps_loop[0], fps_b[0])
    report("fp_gt", fps_loop[1], fps_b[1])
    report("ag_pred", ags_loop[0], ags_b[0])
    report("ag_gt", ags_loop[1], ags_b[1])
    report("bd_pred", bds_loop[0], bds_b[0])
    report("bd_gt", bds_loop[1], bds_b[1])

    # loss parity
    mse = BatchMeanMSE()
    bce = BatchMeanBSE()
    lossA_loop = mse(*ags_loop)
    lossE_loop = mse(*fps_loop)
    lossB_loop = bce(*bds_loop)

    lossA_b = mse(*ags_b)
    lossE_b = mse(*fps_b)
    lossB_b = bce(*bds_b)

    print("lossA diff:", (lossA_loop - lossA_b).abs().item())
    print("lossE diff:", (lossE_loop - lossE_b).abs().item())
    print("lossB diff:", (lossB_loop - lossB_b).abs().item())


if __name__ == "__main__":
    main()
