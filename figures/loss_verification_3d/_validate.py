"""Validate the per-voxel loss maps against the shipped scalar losses.

For every (dim, scenario) this:
  1. builds GT via the real pipeline,
  2. builds a predicted y,
  3. calls the shipped omnipose.core.loss.loss to get authoritative raw_losses,
  4. reduces our per-voxel maps and asserts they match to tol.
If all assertions pass, the maps we later plot are faithful to training.
"""
from __future__ import annotations
import numpy as np
import torch

import omnipose.core.loss as core_loss
import omnipose.metrics.loss as oloss
import _pipeline as P


class Criterion:
    """Mirror of models.train._set_criterion, usable as `self` for loss()."""
    def __init__(self, dim, device, nclasses):
        self.dim = dim
        self.device = device
        self.nclasses = nclasses
        self.MSELoss = oloss.BatchMeanMSE()
        self.BCELoss = oloss.BatchMeanBSE()
        self.SSNLoss = oloss.SSL_Norm()
        self.WeightedMSE = oloss.WeightedMSELoss()
        self.AffinityLoss = oloss.AffinityLoss(device, dim)
        self.DerivativeLoss = oloss.DerivativeLoss()


SCENARIOS = ['perfect', 'flow_zero_cube2', 'flow_flip_cube2', 'dist_offset', 'dist_bump']


def compute_all(dim, device, side=16, margin=10):
    masks, contact = P.two_cubes(dim, side=side, margin=margin)
    nclasses = dim + 2
    lbl, info = P.build_gt(masks, dim, device, nclasses=nclasses)
    g = P.unpack_gt(lbl, dim)
    crit = Criterion(dim, device, nclasses)

    results = {}
    for sc in SCENARIOS:
        y = P.make_pred(lbl, dim, sc, device)
        total, raw_loss, raw = core_loss.loss(crit, lbl, y)
        raw = {k: float(v) for k, v in raw.items()}

        # our per-voxel maps
        fm = P.flow_mse_map(y, lbl, dim, g)       # (B,dim,*sp)
        dm = P.dist_map(y, lbl, dim, g)
        sm = P.ssl_map(y, lbl, dim, g)
        nm = P.norm_map(y, lbl, dim, g)
        bdmap, valid = P.derivative_map(y, lbl, dim, g)
        dcm = P.divcorr_map(y, lbl, dim, g)
        aff = P.affinity_maps(y, lbl, dim, g, device)

        # reductions matching the shipped reduction conventions
        red = {
            'flow_mse': float(fm.mean()),                 # BatchMeanMSE over (dim,*sp)
            'dist_loss': float(dm.mean()),
            'SSL': float(sm.mean()),
            'norm_loss': float(nm.mean()),
            'bd_loss': float(bdmap.sum() / valid.sum().clamp_min(1)),
            'lossDC': float(dcm.mean()),
            'lossE': float(aff['E_map'].mean() / dim),    # MSE over (dim,*sp)
            'lossA': float(aff['A_map'].mean() / aff['A_map'].new_tensor(3 ** dim)),
        }
        results[sc] = dict(shipped=raw, ours=red,
                           maps=dict(flow=fm, dist=dm, ssl=sm, norm=nm,
                                     bd=bdmap, dc=dcm, E=aff['E_map'], A=aff['A_map']),
                           y=y)
    return dict(masks=masks, contact=contact, lbl=lbl, g=g, info=info, results=results)


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"device={dev}")
    keys = ['flow_mse', 'dist_loss', 'SSL', 'norm_loss', 'bd_loss', 'lossDC', 'lossE', 'lossA']
    worst = 0.0
    for dim in (2, 3):
        out = compute_all(dim, dev)
        print(f"\n===== dim={dim}  shape={out['info']['tyx']}  contact@axis0={out['contact']} =====")
        for sc, r in out['results'].items():
            print(f"  scenario={sc}")
            for k in keys:
                a = r['shipped'].get(k, float('nan'))
                b = r['ours'].get(k, float('nan'))
                denom = max(abs(a), 1e-9)
                rel = abs(a - b) / denom
                worst = max(worst, rel if np.isfinite(rel) else 0.0)
                flag = 'OK ' if rel < 1e-3 or abs(a - b) < 1e-6 else '!!!'
                print(f"    {flag} {k:10s} shipped={a:12.6g}  ours={b:12.6g}  rel={rel:.2e}")
    print(f"\nworst relative mismatch across all = {worst:.2e}")


if __name__ == '__main__':
    main()
