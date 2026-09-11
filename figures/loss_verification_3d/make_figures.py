"""Generate the 2D/3D loss-verification figures for omnipose.

Two cubes (3D) / two squares (2D) sharing exactly one face/edge. We plot:
  * gt_fields_{2,3}d.png  — the ground-truth fields the pipeline produces
  * loss_maps_{2,3}d.png  — per-voxel maps of every loss term, one row per
                            controlled prediction error.

Every number / map is the shipped omnipose loss (see _validate.py, which
checks the maps reduce to the shipped scalars to ~1e-7).

Dark-mode figures: transparent canvas, gray annotations (saved transparent).
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
import _pipeline as P
from _validate import compute_all

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'figure.facecolor': 'none', 'axes.facecolor': 'none', 'savefig.facecolor': 'none',
    'text.color': 'gray', 'axes.edgecolor': 'gray', 'axes.labelcolor': 'gray',
    'xtick.color': 'gray', 'ytick.color': 'gray', 'axes.titlecolor': 'gray',
    'font.size': 8, 'axes.titlesize': 8,
})


def np0(t):
    return t.detach().cpu().numpy()[0] if t.ndim and t.shape[0] == 1 else t.detach().cpu().numpy()


# ----- cross-section extraction --------------------------------------------
def slice2d(field, dim, plane, centers):
    """field: numpy (*spatial). Return a 2D slice + axis labels + (row_axis,col_axis)."""
    if dim == 2:
        return field, ('axis0', 'axis1'), (0, 1)
    cz1, cy, cx = centers
    if plane == 'xz':                     # through both cube centres, shows contact
        return field[:, cy, :], ('axis0 (stacking)', 'axis2'), (0, 2)
    if plane == 'xy':                     # in-cube1 transverse cross-section
        return field[cz1, :, :], ('axis1', 'axis2'), (1, 2)
    raise ValueError(plane)


def add_panel(ax, img, title, cmap='magma', diverging=False, vmax=None, vmin=None,
              contact_row=None, quiver=None, step=2, zero_floor=None):
    near_zero = False
    if zero_floor is not None and np.nanmax(np.abs(img)) <= zero_floor:
        # map is numerically zero — render flat dark, don't autoscale noise
        near_zero = True
        vmin, vmax = 0.0, 1.0
    if diverging:
        m = np.nanmax(np.abs(img)) if vmax is None else vmax
        vmin, vmax = -m, m
        cmap = 'RdBu_r'
    else:
        if vmax is None:
            vmax = np.nanpercentile(img, 99.5)
            if vmax <= 0:
                vmax = np.nanmax(img) if np.nanmax(img) > 0 else 1.0
        if vmin is None:
            vmin = np.nanmin(img)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper',
                   interpolation='nearest')
    if near_zero:
        ax.text(0.5, 0.5, '≈ 0', transform=ax.transAxes, ha='center', va='center',
                color='gray', fontsize=11, alpha=0.8)
    if contact_row is not None:
        ax.axhline(contact_row - 0.5, color='lime', lw=0.8, ls='--', alpha=0.7)
    if quiver is not None:
        u, v = quiver                      # u: column comp, v: row comp
        H, W = img.shape
        ys, xs = np.mgrid[0:H:step, 0:W:step]
        ax.quiver(xs, ys, u[::step, ::step], v[::step, ::step],
                  color='cyan', scale=70, width=0.004, alpha=0.9)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6, color='gray', labelcolor='gray')
    cb.outline.set_edgecolor('gray')
    return im


# ----- GT field figure ------------------------------------------------------
def fig_gt(dim, dev):
    masks, contact = P.two_cubes(dim)
    lbl, info = P.build_gt(masks, dim, dev, nclasses=dim + 2)
    g = P.unpack_gt(lbl, dim)
    centers = (10 + 8, masks.shape[-2] // 2, masks.shape[-1] // 2)  # cube1 center z, y, x

    dist = np0(g['dist']); bd = np0(g['boundary']); cm = np0(g['cellmask']).astype(float)
    wgt = np0(g['weight'])
    mag = np0(torch_norm(g['veci'], dim=1))
    div = np0(divergence_torch(g['veci']))
    veci = g['veci'].detach().cpu().numpy()[0]    # (dim, *spatial)
    lab = masks.astype(float)

    planes = ['xz'] if dim == 3 else [None]
    if dim == 3:
        planes = ['xz', 'xy']
    cols = ['labels', 'distance T', '|flow| + flow dirs', 'divergence(flow)', 'boundary', 'weight']
    nrow = len(planes)
    fig, axes = plt.subplots(nrow, len(cols), figsize=(3.0 * len(cols), 3.1 * nrow),
                             squeeze=False, constrained_layout=True)
    for ri, plane in enumerate(planes):
        L, _, ax_ids = slice2d(lab, dim, plane, centers)
        D, _, _ = slice2d(dist, dim, plane, centers)
        M, _, _ = slice2d(mag, dim, plane, centers)
        DV, _, _ = slice2d(div, dim, plane, centers)
        BD, _, _ = slice2d(bd, dim, plane, centers)
        W, _, _ = slice2d(wgt, dim, plane, centers)
        ra, ca = ax_ids
        u = slice2d(veci[ca], dim, plane, centers)[0]
        v = slice2d(veci[ra], dim, plane, centers)[0]
        crow = contact if (dim == 2 or plane == 'xz') else None
        add_panel(axes[ri, 0], L, f'labels ({plane or "2D"})', cmap='tab10', vmin=0, vmax=3, contact_row=crow)
        add_panel(axes[ri, 1], np.where(D > -4, D, np.nan), 'distance field T', cmap='viridis', contact_row=crow)
        add_panel(axes[ri, 2], M, '|flow| (=5·|∇T|) + dirs', cmap='magma', contact_row=crow,
                  quiver=(u, v))
        add_panel(axes[ri, 3], DV, 'divergence(flow)', diverging=True, contact_row=crow)
        add_panel(axes[ri, 4], BD, 'boundary', cmap='gray', vmin=0, vmax=1, contact_row=crow)
        add_panel(axes[ri, 5], W, 'weight', cmap='cividis', vmin=0, vmax=1, contact_row=crow)
    fig.suptitle(f'Ground-truth fields — two {"cubes" if dim==3 else "squares"} sharing one '
                 f'{"face" if dim==3 else "edge"} (dim={dim}). Green dashes = contact plane.',
                 color='gray')
    out = os.path.join(HERE, f'gt_fields_{dim}d.png')
    fig.savefig(out, dpi=150, transparent=True)
    plt.close(fig)
    return out


# ----- loss-map figure ------------------------------------------------------
def fig_loss(dim, dev):
    out_all = compute_all(dim, dev)
    masks, contact = out_all['masks'], out_all['contact']
    centers = (10 + 8, masks.shape[-2] // 2, masks.shape[-1] // 2)
    plane = 'xz' if dim == 3 else None
    crow = contact

    scen = ['flow_zero_cube2', 'flow_flip_cube2', 'dist_offset', 'dist_bump']
    scen_lbl = {'flow_zero_cube2': 'flow→0 in cube 2',
                'flow_flip_cube2': 'flow→ −flow in cube 2',
                'dist_offset': 'dist += 2 in cells',
                'dist_bump': 'gaussian dist bump'}
    # (key in maps dict, display title, shipped raw_losses key)
    cols = [('flow', 'flow_mse', 'flow_mse'),
            ('dist', 'dist_loss', 'dist_loss'),
            ('ssl', 'SSL (orient.)', 'SSL'),
            ('norm', 'norm_loss', 'norm_loss'),
            ('bd', 'bd_loss (deriv)', 'bd_loss'),
            ('dc', 'lossDC (div)', 'lossDC'),
            ('E', 'lossE (euler)', 'lossE'),
            ('A', 'lossA (affinity)', 'lossA')]

    fig, axes = plt.subplots(len(scen), len(cols),
                             figsize=(3.15 * len(cols), 3.35 * len(scen)),
                             squeeze=False, constrained_layout=True)
    for ri, sc in enumerate(scen):
        r = out_all['results'][sc]
        maps = r['maps']
        for ci, (mk, title, rawk) in enumerate(cols):
            mp = maps[mk]
            arr = mp.detach().cpu().numpy()
            if mk == 'flow':                 # (B,dim,*sp) -> per-voxel vector sq err
                arr = arr[0].sum(0)
            elif mk == 'bd':                 # (B,1,*sp) -> drop channel axis
                arr = arr[0, 0]
            else:
                arr = arr[0]
            img = slice2d(arr, dim, plane, centers)[0]
            scalar = r['shipped'].get(rawk, float('nan'))
            ax = axes[ri, ci]
            add_panel(ax, img, f'{title}\nΣ={scalar:.3g}', cmap='inferno',
                      contact_row=crow, zero_floor=1e-9)
            if ci == 0:
                ax.set_ylabel(scen_lbl[sc], color='gray', fontsize=8)
    fig.suptitle(f'Per-voxel loss maps (dim={dim}) — cross-section'
                 f'{" (xz mid-plane)" if dim==3 else ""}. Σ = shipped scalar loss. '
                 f'Green dashes = contact plane.', color='gray')
    out = os.path.join(HERE, f'loss_maps_{dim}d.png')
    fig.savefig(out, dpi=150, transparent=True)
    plt.close(fig)
    return out


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    outs = []
    for dim in (2, 3):
        outs.append(fig_gt(dim, dev))
        outs.append(fig_loss(dim, dev))
    print("WROTE:")
    for o in outs:
        print("  " + o)


if __name__ == '__main__':
    main()
