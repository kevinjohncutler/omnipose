#!/usr/bin/env python3
"""
GPU grid vs GPU sparse benchmark for the Eikonal solver.
Both paths run on the same device (GPU) for a fair comparison.

Usage:
    python test_grid_local.py                    # grid path on GPU
    python test_grid_local.py --sparse           # sparse path on GPU (old approach)
    python test_grid_local.py --save-ref         # save GPU sparse reference
    python test_grid_local.py --compare          # compare grid vs sparse reference
    python test_grid_local.py --both             # run both and print side-by-side
    python test_grid_local.py --diag             # dump iteration count
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from omnirefactor.core.flows import masks_to_flows_batch
from omnirefactor.gpu import torch_GPU, torch_CPU

# ── config ────────────────────────────────────────────────────────────────────
N_MASKS   = 4
IMG_SIZE  = 256          # each synthetic mask is IMG_SIZE × IMG_SIZE
SEED      = 42
REF_FILE  = Path('/tmp/test_grid_local_ref.npz')
N_WARMUP  = 3
N_BENCH   = 10
# ─────────────────────────────────────────────────────────────────────────────


def make_synthetic_masks(n: int, size: int, seed: int = 42) -> np.ndarray:
    """Create simple synthetic label masks (square blobs)."""
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(n):
        m = np.zeros((size, size), dtype=np.int32)
        label = 1
        for _ in range(20):
            cy, cx = rng.integers(20, size - 20, size=2)
            ry, rx = rng.integers(8, 30, size=2)
            ys = np.clip(np.arange(cy - ry, cy + ry), 0, size - 1)
            xs = np.clip(np.arange(cx - rx, cx + rx), 0, size - 1)
            yy, xx = np.meshgrid(ys, xs, indexing='ij')
            m[yy, xx] = label
            label += 1
        masks.append(m)
    return np.stack(masks)


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_once(masks, device, n_iter=None, use_grid=True):
    t0 = time.perf_counter()
    with torch.no_grad():
        out = masks_to_flows_batch(
            masks,
            links=[None] * len(masks),
            device=device,
            omni=True,
            dim=2,
            n_iter=n_iter,
            use_grid=use_grid,
        )
    sync(device)
    elapsed = time.perf_counter() - t0
    T_np  = out[2].cpu().numpy()
    mu_np = out[3].cpu().numpy()
    return T_np, mu_np, elapsed


def benchmark(masks, device, label, n_iter=None, use_grid=True):
    n_str = f' n_iter={n_iter}' if n_iter is not None else ' n_iter=default'
    print(f'\n[{label}{n_str}]  warming up ({N_WARMUP} calls) …', flush=True)
    for _ in range(N_WARMUP):
        run_once(masks, device, n_iter=n_iter, use_grid=use_grid)

    print(f'[{label}{n_str}]  benchmarking ({N_BENCH} calls) …', flush=True)
    times, T_out, mu_out = [], None, None
    for _ in range(N_BENCH):
        T_out, mu_out, t = run_once(masks, device, n_iter=n_iter, use_grid=use_grid)
        times.append(t)

    arr = np.array(times)
    print(f'[{label}{n_str}]  {arr.mean()*1e3:.1f} ms/call'
          f'  ± {arr.std()*1e3:.1f}'
          f'  (min {arr.min()*1e3:.1f}  max {arr.max()*1e3:.1f})')
    return T_out, mu_out, arr


def compare(T, mu, ref_file: Path):
    """Compare T and mu vs saved reference."""
    if not ref_file.exists():
        print(f'  [accuracy] no reference at {ref_file} — skipping')
        return
    ref = np.load(str(ref_file))
    T_r, mu_r = ref['T'], ref['mu']
    T_abs  = np.abs(T  - T_r).max()
    T_rel  = T_abs  / (np.abs(T_r ).max() + 1e-10)

    fg = ref['fg'] if 'fg' in ref else (T_r > 0)
    mu_d = np.abs(mu - mu_r)
    mu_mean = mu_d[:, fg].mean()  if fg.any() else mu_d.mean()
    mu_max  = mu_d[:, fg].max()   if fg.any() else mu_d.max()
    n_bad   = int((mu_d[:, fg] > 0.1).sum()) if fg.any() else int((mu_d > 0.1).sum())
    n_fg    = int(fg.sum())

    print(f'  [accuracy]  T   max-abs {T_abs:.3e}  rel {T_rel:.3e}')
    print(f'  [accuracy]  mu  mean    {mu_mean:.3e}  max {mu_max:.3e}  '
          f'(>0.1 err: {n_bad}/{n_fg} fg pixels)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--save-ref', action='store_true',
                   help='Save GPU sparse reference (use_grid=False)')
    p.add_argument('--compare',  action='store_true',
                   help='Compare grid output against saved sparse reference')
    p.add_argument('--sparse',   action='store_true',
                   help='Run the old GPU sparse path (use_grid=False)')
    p.add_argument('--both',     action='store_true',
                   help='Benchmark both paths on GPU and compare')
    p.add_argument('--n50',      action='store_true',
                   help='Force n_iter=50')
    p.add_argument('--diag',     action='store_true',
                   help='One verbose call, then exit')
    p.add_argument('--label',    default=None)
    args = p.parse_args()

    device = torch_GPU
    print(f'Device: {device}')

    masks = make_synthetic_masks(N_MASKS, IMG_SIZE, SEED)
    print(f'Masks: {N_MASKS} × {IMG_SIZE}×{IMG_SIZE}, '
          f'max labels {[masks[i].max() for i in range(N_MASKS)]}')

    if args.diag:
        print('\n--- verbose diagnostic ---')
        with torch.no_grad():
            masks_to_flows_batch(masks, links=[None]*N_MASKS,
                                 device=device, omni=True, dim=2, verbose=True)
        return

    n_iter = 50 if args.n50 else None

    if args.both:
        T_sparse, mu_sparse, _ = benchmark(masks, device, 'sparse', n_iter=n_iter, use_grid=False)
        T_grid,   mu_grid,   _ = benchmark(masks, device, 'grid',   n_iter=n_iter, use_grid=True)
        # cross-compare
        T_abs = np.abs(T_grid - T_sparse).max()
        T_rel = T_abs / (np.abs(T_sparse).max() + 1e-10)
        fg = T_sparse > 0
        mu_d = np.abs(mu_grid - mu_sparse)
        mu_mean = mu_d[:, fg].mean()
        n_bad = int((mu_d[:, fg] > 0.1).sum())
        n_fg = int(fg.sum())
        print(f'\n[grid vs sparse]  T rel {T_rel:.3e}  mu mean {mu_mean:.3e}  '
              f'(>0.1 err: {n_bad}/{n_fg} fg pixels)')
        return

    use_grid = not args.sparse
    label = args.label or ('sparse' if args.sparse else 'grid')
    T, mu, _ = benchmark(masks, device, label, n_iter=n_iter, use_grid=use_grid)

    if args.save_ref:
        fg = T > 0
        np.savez(str(REF_FILE), T=T, mu=mu, fg=fg)
        print(f'\nReference saved → {REF_FILE}')
        print('Re-run with --compare after code changes.')
    elif args.compare:
        compare(T, mu, REF_FILE)


if __name__ == '__main__':
    main()
