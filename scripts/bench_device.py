#!/usr/bin/env python
"""Benchmark omnipose segmentation (GPU/MPS vs CPU) on the demo test images.

Replicates docs/examples/mono_channel_bact-gpu-v-cpu.ipynb preprocessing and
params. Warms up once per device (MPS/torch JIT), then times N repeats.

Usage: python scripts/bench_device.py [--repeats 3] [--devices gpu,cpu]
Prints a JSON blob (machine-readable) and a human table.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

import omnipose
from omnipose import io, transforms, models
from omnipose.transforms import normalize99


MODEL_NAME = "bact_phase_affinity"

PARAMS = {
    "channels": None,
    "rescale": None,
    "mask_threshold": -2,
    "flow_threshold": 0,
    "transparency": True,
    "omni": True,
    "cluster": True,
    "resample": True,
    "verbose": False,
    "tile": False,
    "niter": None,
    "augment": False,
    "affinity_seg": True,
}


def load_images():
    omnidir = Path(omnipose.__file__).parent.parent.parent
    basedir = os.path.join(omnidir, "docs", "test_files")
    files = io.get_image_files(basedir)
    imgs = []
    for f in files:
        img = io.imread(f)
        img = transforms.move_min_dim(img)
        if img.ndim > 2:
            img = np.mean(img, axis=-1)
        imgs.append(normalize99(img))
    return files, imgs


def hw_info():
    info = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import subprocess
        info["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        info["ncpu"] = int(subprocess.check_output(
            ["sysctl", "-n", "hw.logicalcpu"]).decode().strip())
    except Exception:
        info["cpu"] = platform.processor()
    try:
        import torch
        info["torch"] = torch.__version__
        info["mps"] = bool(torch.backends.mps.is_available())
        info["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    return info


def bench_device(use_gpu, imgs, repeats):
    import torch
    model = models.OmniModel(gpu=use_gpu, model_type=MODEL_NAME)

    def sync():
        if use_gpu and torch.backends.mps.is_available():
            torch.mps.synchronize()

    # warmup (downloads weights if needed, triggers JIT/compile)
    res = model.eval(imgs, **PARAMS)
    sync()

    times = []
    nmasks_last = None
    for _ in range(repeats):
        tic = time.perf_counter()
        res = model.eval(imgs, **PARAMS)
        sync()
        times.append(time.perf_counter() - tic)
        nmasks_last = [int(m.max()) for m in res.masks]

    times = np.array(times)
    return {
        "device": "gpu/mps" if use_gpu else "cpu",
        "repeats": repeats,
        "total_best_s": float(times.min()),
        "total_mean_s": float(times.mean()),
        "total_std_s": float(times.std()),
        "per_image_best_ms": [float(1000 * times.min() / len(imgs))],
        "nmasks": nmasks_last,
        "all_times_s": times.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--devices", default="gpu,cpu")
    args = ap.parse_args()

    files, imgs = load_images()
    result = {
        "hw": hw_info(),
        "model": MODEL_NAME,
        "n_images": len(imgs),
        "image_shapes": [list(im.shape) for im in imgs],
        "files": [os.path.basename(str(f)) for f in files],
        "runs": [],
    }

    for d in args.devices.split(","):
        use_gpu = d.strip() == "gpu"
        result["runs"].append(bench_device(use_gpu, imgs, args.repeats))

    print("\n===JSON===")
    print(json.dumps(result))
    print("===ENDJSON===\n")

    hw = result["hw"]
    print(f"Host: {hw.get('hostname')}  CPU: {hw.get('cpu')}  "
          f"cores: {hw.get('ncpu')}  torch: {hw.get('torch')}  "
          f"py: {hw.get('python')}")
    print(f"Images: {result['n_images']}  model: {MODEL_NAME}")
    print(f"{'device':10} {'best total (s)':>15} {'mean total (s)':>15} "
          f"{'per-img best (ms)':>18}")
    for r in result["runs"]:
        print(f"{r['device']:10} {r['total_best_s']:15.3f} "
              f"{r['total_mean_s']:15.3f} {r['per_image_best_ms'][0]:18.1f}")


if __name__ == "__main__":
    main()
