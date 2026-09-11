"""Diagnose + demonstrate the nearest_interpolation_batched axis-collapse bug
that drives the streaked lossE / lossA maps, and show the fix removes it.

Root cause (torchvf/numerics/interpolation/functional.py):

    def nearest_interpolation_batched(vf, points):
        B, D, *dims = vf.shape
        points = stack([clamp(points[:, i], 0, d-1) for i, d in enumerate(dims)], 1)
        points = points.round_().long()
        return vf.gather(-1, points)          # <-- only indexes the LAST axis

`gather(-1, points)` sets out[b,c, i, j, ...] = vf[b,c, i, j, ..., points[b,c,...]],
i.e. it replaces ONLY the last spatial index and keeps every other output axis at
its own position. So only the last flow component is sampled correctly; all others
are smeared along the last axis -> the streak.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import importlib
IVF = importlib.import_module('torchvf.numerics.interpolation.interp_vf')
from torchvf.numerics import interp_vf, ivp_solver
import omnipose.metrics.loss as oloss
from omnipose.utils import kernel_setup, get_supporting_inds
import _pipeline as P
from make_figures import add_panel

HERE = os.path.dirname(os.path.abspath(__file__))


def nearest_batched_fixed(vf, points):
    """Correct ND nearest-neighbour sampling: out[b,c,*t] = vf[b,c, round(points[b,:,*t])]."""
    B, D, *dims = vf.shape
    idx = [torch.clamp(points[:, k], 0, dims[k] - 1).round().long() for k in range(D)]
    strides = [1] * D
    for k in range(D - 2, -1, -1):
        strides[k] = strides[k + 1] * dims[k + 1]
    lin = sum(idx[k] * strides[k] for k in range(D))        # (B, *T)
    Tshape = lin.shape[1:]
    vf_flat = vf.reshape(B, D, -1)
    lin_exp = lin.unsqueeze(1).expand(B, D, *Tshape).reshape(B, D, -1)
    return torch.gather(vf_flat, -1, lin_exp).reshape(B, D, *Tshape)


def identity_check():
    print("Identity-grid sampling should return the field unchanged:")
    for dim, shape in [(2, (5, 7)), (3, (4, 5, 6))]:
        coords = torch.meshgrid(*[torch.arange(s) for s in shape], indexing='ij')
        vf = torch.stack([(c + 1) * 1000.0 + sum(coords[k] * (10 ** k) for k in range(dim))
                          for c in range(dim)], 0)[None].float()
        ip = torch.stack(coords, 0)[None].float()
        broken = interp_vf(vf, mode="nearest_batched")(ip)
        fixed = nearest_batched_fixed(vf, ip)
        print(f"  dim={dim}: broken maxdiff={float((broken-vf).abs().max()):8.1f}   "
              f"fixed maxdiff={float((fixed-vf).abs().max()):.1f}")


def affinity_maps_with(interp_fn, y, lbl, dim, g, dev):
    """Replicate AffinityLoss endpoint/affinity maps using a given interp fn."""
    steps, inds, idx, fact, sign = kernel_setup(dim)
    supp = get_supporting_inds(steps)
    flow_pred, dist_pred = y[:, :dim], y[:, dim]
    flow_gt, dist_gt = g['veci'], g['dist']
    foreground = torch.ones_like(dist_pred, dtype=torch.bool)
    B = flow_pred.shape[0]; dims = flow_pred.shape[-dim:]
    mesh = torch.meshgrid([torch.arange(0, l, device=dev) for l in dims], indexing="ij")
    ip = torch.stack(mesh, 0).repeat([B, 1] + [1] * len(dims)).float()

    flow_all = torch.cat([flow_pred, flow_gt], 0)
    ip_all = torch.cat([ip, ip], 0)

    # temporarily install the chosen interpolator
    orig = IVF.nearest_interpolation_batched
    IVF.nearest_interpolation_batched = interp_fn
    try:
        vf_all = interp_vf(flow_all, mode="nearest_batched")
        fp_all = ivp_solver(vf_all, ip_all, dx=np.sqrt(dim) / 5, n_steps=2, solver="euler")[-1]
    finally:
        IVF.nearest_interpolation_batched = orig
    fpp, fpg = torch.chunk(fp_all, 2, 0)

    ags = []
    for f, d, fp in zip([flow_pred, flow_gt], [dist_pred, dist_gt], [fpp, fpg]):
        ag = oloss._get_affinity_torch(ip, fp, f / 5., d, foreground, steps, fact, inds,
                                       supp, 10, device=dev)
        ags.append(ag * 1.0)
    E = ((fpp - fpg) ** 2).sum(1)[0]
    A = ((ags[0] - ags[1]) ** 2).sum(0)[0]
    return E, A


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    identity_check()

    fig, axes = plt.subplots(2, 4, figsize=(3.15 * 4, 3.35 * 2),
                             squeeze=False, constrained_layout=True)
    print("\nlossE / lossA scalar means (flow set to 0 in cube 2):")
    for ri, dim in enumerate((2, 3)):
        masks, contact = P.two_cubes(dim)
        lbl, _ = P.build_gt(masks, dim, dev, nclasses=dim + 2)
        g = P.unpack_gt(lbl, dim)
        y = P.make_pred(lbl, dim, 'flow_zero_cube2', dev)
        cp = None if dim == 2 else masks.shape[-2] // 2
        def sl(a):
            a = a.detach().cpu().numpy()
            return a if dim == 2 else a[:, cp, :]
        Eb, Ab = affinity_maps_with(IVF.nearest_interpolation_batched, y, lbl, dim, g, dev)
        Ef, Af = affinity_maps_with(nearest_batched_fixed, y, lbl, dim, g, dev)
        print(f"  dim={dim}: lossE broken={float(Eb.mean()/dim):.4g} fixed={float(Ef.mean()/dim):.4g} | "
              f"lossA broken={float(Ab.mean()/3**dim):.4g} fixed={float(Af.mean()/3**dim):.4g}")
        add_panel(axes[ri, 0], sl(Eb), f'lossE map — CURRENT (buggy)\ndim={dim}', cmap='inferno')
        add_panel(axes[ri, 1], sl(Ef), f'lossE map — FIXED\ndim={dim}', cmap='inferno')
        add_panel(axes[ri, 2], sl(Ab), f'lossA map — CURRENT (buggy)\ndim={dim}', cmap='inferno')
        add_panel(axes[ri, 3], sl(Af), f'lossA map — FIXED\ndim={dim}', cmap='inferno')
    fig.suptitle('nearest_interpolation_batched axis-collapse: current maps streak along the '
                 'last axis; the fixed ND interpolator recovers the flow skeleton.', color='gray')
    out = os.path.join(HERE, 'interp_bug_comparison.png')
    fig.savefig(out, dpi=150, transparent=True)
    plt.close(fig)
    print("\nWROTE:", out)


if __name__ == '__main__':
    main()
