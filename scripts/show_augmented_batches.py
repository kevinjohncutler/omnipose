#!/usr/bin/env python3
"""
Save examples of augmented training batches (inputs + label channels).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from omnirefactor import io, core, transforms
from omnirefactor.data.train import train_set, CyclingRandomBatchSampler
from omnirefactor.transforms.augment import random_rotate_and_resize
from omnirefactor.core.flows import masks_to_flows_batch, batch_labels


def _save_grid(imgs: np.ndarray, out_path: Path, ncols: int = 4, cmap: str | None = None) -> None:
    """Save a simple image grid for (N,H,W) arrays."""
    n = imgs.shape[0]
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])
    axes = axes.reshape(nrows, ncols)
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i], cmap=cmap)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """flow: (2,H,W) -> RGB image"""
    fy, fx = flow[0], flow[1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    mag_norm = mag / (mag.max() + 1e-8)
    ang = np.arctan2(fy, fx)  # [-pi,pi]
    hue = (ang + np.pi) / (2 * np.pi)
    h = hue
    s = np.ones_like(h)
    v = mag_norm
    c = v * s
    hp = h * 6.0
    x = c * (1 - np.abs((hp % 2) - 1))
    z = np.zeros_like(h)
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    cond = (0 <= hp) & (hp < 1)
    r[cond], g[cond], b[cond] = c[cond], x[cond], z[cond]
    cond = (1 <= hp) & (hp < 2)
    r[cond], g[cond], b[cond] = x[cond], c[cond], z[cond]
    cond = (2 <= hp) & (hp < 3)
    r[cond], g[cond], b[cond] = z[cond], c[cond], x[cond]
    cond = (3 <= hp) & (hp < 4)
    r[cond], g[cond], b[cond] = z[cond], x[cond], c[cond]
    cond = (4 <= hp) & (hp < 5)
    r[cond], g[cond], b[cond] = x[cond], z[cond], c[cond]
    cond = (5 <= hp) & (hp < 6)
    r[cond], g[cond], b[cond] = c[cond], z[cond], x[cond]
    m = v - c
    return np.stack([r + m, g + m, b + m], axis=-1)


def _normalize_for_display(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.integer):
        maxv = np.iinfo(img.dtype).max
        return (img.astype(np.float32) / max(maxv, 1)).clip(0, 1)
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - vmin) / (vmax - vmin)).astype(np.float32)

def _order_polygon_corners(corners: np.ndarray) -> np.ndarray:
    """Order 2D corners (y,x) around centroid to avoid self-intersections."""
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 0] - center[0], corners[:, 1] - center[1])
    order = np.argsort(angles)
    return corners[order]

def _draw_colored_edges(ax, corners: np.ndarray, colors: list[str], lw: float = 2.0) -> None:
    """Draw polygon edges with per-edge colors. corners expected in (y,x)."""
    n = corners.shape[0]
    for i in range(n):
        p0 = corners[i]
        p1 = corners[(i + 1) % n]
        ax.plot([p0[1], p1[1]], [p0[0], p1[0]], color=colors[i % len(colors)], linewidth=lw)

def main() -> None:
    parser = argparse.ArgumentParser(description="Save augmented training batches.")
    parser.add_argument("--dir", required=True, type=str, help="training directory")
    parser.add_argument("--look_one_level_down", action="store_true", help="scan subfolders")
    parser.add_argument("--n_epochs", default=1, type=int, help="number of epochs for sampler")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--dim", default=2, type=int)
    parser.add_argument("--nchan", default=1, type=int)
    parser.add_argument("--nclasses", default=2, type=int)
    parser.add_argument("--omni", action="store_true", default=True)
    parser.add_argument("--links", action="store_true")
    parser.add_argument("--affinity_field", action="store_true")
    parser.add_argument("--num_batches", default=2, type=int)
    parser.add_argument("--out_dir", default="omnirefactor/tmp/augmented_batches", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--do_rescale", action="store_true")
    parser.add_argument("--diam_mean", default=30.0, type=float)
    parser.add_argument("--device", default="cpu", type=str, help="cpu or cuda")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, labels, links, *_ = io.load_train_test_data(
        args.dir,
        test_dir=None,
        look_one_level_down=args.look_one_level_down,
        omni=args.omni,
        do_links=args.links,
    )
    orig_images = [img.copy() for img in images]

    # Match training preprocessing (reshape + normalize)
    images, labels, _, _, _ = transforms.reshape_train_test(
        images,
        labels,
        None,
        None,
        channels=None,
        channel_axis=None,
        normalize=True,
        dim=args.dim,
        omni=args.omni,
    )

    if args.do_rescale:
        diam_train = np.array([core.diam.diameters(lbl, omni=args.omni) for lbl in labels])
        diam_train[diam_train < 5] = 5.0
    else:
        diam_train = np.ones(len(labels), np.float32)

    device = torch.device("cuda") if (args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")
    kwargs = {
        "do_rescale": bool(args.do_rescale),
        "diam_train": diam_train,
        "diam_mean": args.diam_mean,
        "tyx": None,
        "scale_range": 1.5,
        "omni": args.omni,
        "dim": args.dim,
        "nchan": args.nchan,
        "nclasses": args.nclasses,
        "device": torch.device("cpu"),
        "affinity_field": args.affinity_field,
        "allow_blank_masks": False,
        "timing": False,
    }

    dataset = train_set(images, labels, links, **kwargs)
    sampler = CyclingRandomBatchSampler(
        dataset,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        nimg_per_epoch=len(dataset),
    )
    sampler_iter = iter(sampler)

    for b in range(args.num_batches):
        batch_inds = next(sampler_iter)
        inds = list(batch_inds)

        links_batch = [links[idx] for idx in inds]
        rsc = np.array([dataset.rescale_factor[idx] for idx in inds]) if dataset.do_rescale else None

        imgi, labels_aug, _scale, metas = random_rotate_and_resize(
            X=[images[idx] for idx in inds],
            Y=[labels[idx] for idx in inds],
            scale_range=dataset.scale_range,
            gamma_range=dataset.gamma_range,
            tyx=dataset.tyx,
            do_flip=dataset.do_flip,
            rescale_factor=rsc,
            inds=inds,
            nchan=dataset.nchan,
            allow_blank_masks=dataset.allow_blank_masks,
            return_meta=True,
        )

        out = masks_to_flows_batch(
            labels_aug,
            links_batch,
            device=dataset.device,
            omni=dataset.omni,
            dim=dataset.dim,
            affinity_field=dataset.affinity_field,
        )
        X = out[:-4]
        slices = out[-4]
        masks, bd, T, mu = [torch.stack([x[(Ellipsis,) + slc] for slc in slices]) for x in X]

        lbl = batch_labels(
            masks,
            bd,
            T,
            mu,
            dataset.tyx,
            dim=dataset.dim,
            nclasses=dataset.nclasses,
            device=dataset.device,
        )
        x = torch.tensor(imgi, device=dataset.device, dtype=torch.float32)

        # x: (B,C,H,W)
        x_np = x.detach().cpu().numpy()
        lbl_np = lbl.detach().cpu().numpy()

        # Optional move to GPU to reflect training input device
        if device.type == "cuda":
            x_gpu = x.to(device, non_blocking=True)
            lbl_gpu = lbl.to(device, non_blocking=True)
            _ = x_gpu, lbl_gpu

        batch_dir = out_dir / f"batch_{b:02d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Save inputs (first channel)
        _save_grid(x_np[:, 0], batch_dir / "input_c0.png", cmap="gray")

        # Save inputs with colored crop edges (output space)
        edge_colors = ["red", "blue", "yellow", "green"]
        n = x_np.shape[0]
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = np.array([[ax] for ax in axes])
        axes = axes.reshape(nrows, ncols)
        for i in range(nrows * ncols):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            ax.axis("off")
            if i < n:
                ax.imshow(x_np[i, 0], cmap="gray")
                ty, tx = x_np.shape[-2], x_np.shape[-1]
                corners_out = np.array(
                    [
                        [0, 0],
                        [0, tx - 1],
                        [ty - 1, tx - 1],
                        [ty - 1, 0],
                    ],
                    dtype=np.float32,
                )
                _draw_colored_edges(ax, corners_out, edge_colors, lw=2.0)
        fig.tight_layout()
        fig.savefig(batch_dir / "input_c0_edges.png", dpi=150)
        plt.close(fig)

        # Save original images with crop boxes
        if args.dim == 2 and metas:
            n = len(inds)
            ncols = min(4, n)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
            if nrows == 1 and ncols == 1:
                axes = np.array([[axes]])
            elif nrows == 1:
                axes = np.array([axes])
            elif ncols == 1:
                axes = np.array([[ax] for ax in axes])
            axes = axes.reshape(nrows, ncols)
            for i in range(nrows * ncols):
                r, c = divmod(i, ncols)
                ax = axes[r, c]
                ax.axis("off")
                if i < n:
                    img = orig_images[inds[i]]
                    if img.ndim > 2:
                        img_disp = _normalize_for_display(img[0])
                    else:
                        img_disp = _normalize_for_display(img)
                    ax.imshow(img_disp, cmap="gray")
                    corners = metas[i].get("corners_in")
                    if corners is not None:
                        corners = _order_polygon_corners(corners)
                        _draw_colored_edges(ax, corners, edge_colors, lw=2.0)
            fig.tight_layout()
            fig.savefig(batch_dir / "orig_with_crop.png", dpi=150)
            plt.close(fig)

        # Save key label channels when available
        if lbl_np.shape[1] >= 1:
            _save_grid(lbl_np[:, 0], batch_dir / "lbl_masks.png", cmap="gray")
        if lbl_np.shape[1] >= 2:
            _save_grid(lbl_np[:, 1], batch_dir / "lbl_cellprob.png", cmap="gray")
        if lbl_np.shape[1] >= 3:
            _save_grid(lbl_np[:, 2], batch_dir / "lbl_boundary.png", cmap="gray")
        if lbl_np.shape[1] >= 4:
            _save_grid(lbl_np[:, 3], batch_dir / "lbl_distance.png", cmap="gray")
        if lbl_np.shape[1] >= 5:
            _save_grid(lbl_np[:, 4], batch_dir / "lbl_weight.png", cmap="gray")

        # Flow visualization (last 2 channels for 2D)
        if args.dim == 2 and lbl_np.shape[1] >= 2:
            flow = lbl_np[:, -2:, :, :]
            flow_rgb = np.stack([_flow_to_rgb(flow[i]) for i in range(flow.shape[0])], axis=0)
            # save as grid
            n = flow_rgb.shape[0]
            ncols = min(4, n)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
            if nrows == 1 and ncols == 1:
                axes = np.array([[axes]])
            elif nrows == 1:
                axes = np.array([axes])
            elif ncols == 1:
                axes = np.array([[ax] for ax in axes])
            axes = axes.reshape(nrows, ncols)
            for i in range(nrows * ncols):
                r, c = divmod(i, ncols)
                ax = axes[r, c]
                ax.axis("off")
                if i < n:
                    ax.imshow(flow_rgb[i])
            fig.tight_layout()
            fig.savefig(batch_dir / "lbl_flow.png", dpi=150)
            plt.close(fig)

        # Summary stats
        stats_path = batch_dir / "stats.txt"
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(f"indices: {inds}\n")
            f.write(f"input shape: {tuple(x.shape)}\n")
            f.write(f"label shape: {tuple(lbl.shape)}\n")
            f.write(f"device: {device}\n")
            f.write(f"input min/max: {x.min().item():.4f} / {x.max().item():.4f}\n")
            f.write(f"label min/max: {lbl.min().item():.4f} / {lbl.max().item():.4f}\n")

        print(f"saved batch {b} to {batch_dir}")


if __name__ == "__main__":
    main()
