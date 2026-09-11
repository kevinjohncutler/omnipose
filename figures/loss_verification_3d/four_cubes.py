"""Four-cube (2x2) label-equivariance experiment.

Goal: prove no label (esp. label 1) is treated differently.
  * flow_zero_all   -> identical per-voxel loss in all four tiles; per-label
                       integrated losses equal across labels 1..4.
  * flow_zero_label1 -> only the label-1 tile lights up (label 1 is NOT ignored).
"""
from __future__ import annotations
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from omnipose.core.fields import divergence_torch
from omnipose.core.imports import torch_norm
import omnipose.core.loss as core_loss
import _pipeline as P
from _validate import Criterion
from make_figures import add_panel          # also applies dark-mode rcParams

HERE = os.path.dirname(os.path.abspath(__file__))


def get2d(arr_sp, dim, cp):
    return arr_sp if dim == 2 else arr_sp[cp]


def per_voxel_maps(y, lbl, dim, g, dev):
    """Return dict of per-voxel scalar maps (B,*sp) for every loss term."""
    fm = P.flow_mse_map(y, lbl, dim, g)[0].sum(0)        # (*sp)
    dm = P.dist_map(y, lbl, dim, g)[0]
    sm = P.ssl_map(y, lbl, dim, g)[0]
    nm = P.norm_map(y, lbl, dim, g)[0]
    bdmap, _ = P.derivative_map(y, lbl, dim, g)
    bd = bdmap[0, 0]
    dc = P.divcorr_map(y, lbl, dim, g)[0]
    aff = P.affinity_maps(y, lbl, dim, g, dev)
    return dict(flow_mse=fm, dist_loss=dm, SSL=sm, norm_loss=nm, bd_loss=bd,
                lossDC=dc, lossE=aff['E_map'][0], lossA=aff['A_map'][0])


def run(dim, dev):
    masks, cp = P.four_cubes(dim)
    lbl, _ = P.build_gt(masks, dim, dev, nclasses=dim + 2)
    g = P.unpack_gt(lbl, dim)
    labmap = lbl[0, 0].cpu().numpy()
    dist = g['dist'][0].cpu().numpy()

    # GT per-label symmetry
    maxd = {k: float(dist[labmap == k].max()) for k in (1, 2, 3, 4)}

    # per-label loss contributions for flow_zero_all
    y_all = P.make_pred(lbl, dim, 'flow_zero_all', dev)
    maps_all = per_voxel_maps(y_all, lbl, dim, g, dev)
    terms = list(maps_all.keys())
    per_label = {t: {} for t in terms}
    for t in terms:
        m = maps_all[t].detach().cpu().numpy()
        for k in (1, 2, 3, 4):
            per_label[t][k] = float(m[labmap == k].sum())

    y_l1 = P.make_pred(lbl, dim, 'flow_zero_label1', dev)
    maps_l1 = per_voxel_maps(y_l1, lbl, dim, g, dev)

    return dict(masks=masks, cp=cp, lbl=lbl, g=g, labmap=labmap, maxd=maxd,
                maps_all=maps_all, maps_l1=maps_l1, terms=terms,
                per_label=per_label, y_all=y_all)


def figure(dim, res, dev):
    masks, cp = res['masks'], res['cp']
    g = res['g']
    veci = g['veci'].detach().cpu().numpy()[0]
    lab = masks.astype(float)
    dist = g['dist'][0].cpu().numpy()
    mag = torch_norm(g['veci'], dim=1)[0].cpu().numpy()
    div = divergence_torch(g['veci'])[0].cpu().numpy()

    # quiver components for the shown plane
    if dim == 2:
        ra, ca = 0, 1
    else:
        ra, ca = 1, 2                       # center plane spans axis1,axis2
    u = get2d(veci[ca], dim, cp)
    v = get2d(veci[ra], dim, cp)

    flow_terms = ['flow_mse', 'SSL', 'norm_loss', 'lossDC', 'lossE', 'lossA']
    fig, axes = plt.subplots(3, 6, figsize=(3.15 * 6, 3.35 * 3),
                             squeeze=False, constrained_layout=True)

    # row 0: GT context
    add_panel(axes[0, 0], get2d(lab, dim, cp), 'labels 1..4', cmap='tab10', vmin=0, vmax=4)
    add_panel(axes[0, 1], np.where(get2d(dist, dim, cp) > -4, get2d(dist, dim, cp), np.nan),
              'distance T', cmap='viridis')
    add_panel(axes[0, 2], get2d(mag, dim, cp), '|flow| + dirs', cmap='magma', quiver=(u, v))
    add_panel(axes[0, 3], get2d(div, dim, cp), 'divergence(flow)', diverging=True)
    axes[0, 4].axis('off'); axes[0, 5].axis('off')
    axes[0, 0].set_ylabel('ground truth', color='gray')

    # rows 1,2: loss maps for the two scenarios
    for ri, (scen, maps, lblrow) in enumerate(
            [('flow_zero_all', res['maps_all'], 'flow→0 in ALL cells'),
             ('flow_zero_label1', res['maps_l1'], 'flow→0 in LABEL 1 only')], start=1):
        crit = Criterion(dim, dev, dim + 2)
        y = res['y_all'] if scen == 'flow_zero_all' else P.make_pred(res['lbl'], dim, scen, dev)
        _, _, raw = core_loss.loss(crit, res['lbl'], y)
        for ci, t in enumerate(flow_terms):
            img = get2d(maps[t].detach().cpu().numpy(), dim, cp)
            add_panel(axes[ri, ci], img, f'{t}\nΣ={float(raw[t]):.3g}',
                      cmap='inferno', zero_floor=1e-9)
        axes[ri, 0].set_ylabel(lblrow, color='gray')

    fig.suptitle(f'Four cubes (2x2), dim={dim} — label-equivariance check. '
                 f'Row 2: all tiles light identically. Row 3: only label-1 tile lights '
                 f'(label 1 is not ignored).', color='gray')
    out = os.path.join(HERE, f'four_cubes_{dim}d.png')
    fig.savefig(out, dpi=150, transparent=True)
    plt.close(fig)
    return out


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    outs = []
    print('Per-label equivariance (flow_zero_all): contribution of each label to each term.')
    print('If label handling is symmetric, labels 1..4 are equal (the four tiles are reflections).\n')
    for dim in (2, 3):
        res = run(dim, dev)
        print(f'===== dim={dim}  shape={res["masks"].shape} =====')
        md = res['maxd']
        spread_d = (max(md.values()) - min(md.values())) / (sum(md.values()) / 4)
        print(f'  GT max distance per label: '
              + ' '.join(f'L{k}={md[k]:.4f}' for k in (1, 2, 3, 4))
              + f'   spread={spread_d:.2e}')
        print(f'  {"term":10s} ' + ' '.join(f'{"L"+str(k):>12s}' for k in (1, 2, 3, 4)) + '   max_spread')
        for t in res['terms']:
            vals = [res['per_label'][t][k] for k in (1, 2, 3, 4)]
            mean = sum(vals) / 4
            spread = (max(vals) - min(vals)) / mean if mean > 1e-12 else 0.0
            print(f'  {t:10s} ' + ' '.join(f'{v:12.6g}' for v in vals) + f'   {spread:.2e}')
        outs.append(figure(dim, res, dev))
        print()
    print('WROTE:')
    for o in outs:
        print('  ' + o)


if __name__ == '__main__':
    main()
