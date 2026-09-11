"""bench_seg_times — compare segmentation timing in-process vs through the viewer HTTP API.

Discovers images in a directory, runs each through the CLI-equivalent
in-process `Segmenter.segment()` and the GUI-equivalent `/api/segment`
HTTP endpoint, and reports per-cell mean/median/min over N runs (after a
warmup pass that's discarded).

Server-side timing comes from the `timing_ms` field on the segment response
(plugin time + ncolor time, separated). Client-side `roundtrip_ms` is the
full request wall clock. The gap is FastAPI/Starlette/middleware overhead.

Example:
    pyenv exec python scripts/bench_seg_times.py \\
        --images-dir docs/test_files \\
        --models bact_phase_affinity bact_fluor_affinity \\
        --gui-url https://127.0.0.1:8766 \\
        --runs 5 --warmup 1 \\
        --csv-out /tmp/bench.csv

The HTTP path skips TLS verification (the dev viewer uses a self-signed
cert via --https-auto). Use --no-gui to skip the GUI half if no server is
running, or --no-cli to skip the in-process half.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _format_ms(values: list[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f} (med {statistics.median(values):.1f}, min {min(values):.1f})"


def _discover_images(folder: Path) -> list[Path]:
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    out = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file())
    if not out:
        raise SystemExit(f"no images found in {folder}")
    return out


def _load_image(path: Path) -> np.ndarray:
    """Load via skimage (per project policy: never PIL/cv2)."""
    from skimage import io as skio
    img = skio.imread(str(path))
    return img


# ---------------------------------------------------------------------------
# CLI / in-process path
# ---------------------------------------------------------------------------


def bench_cli(images: list[Path], models: list[str], runs: int, warmup: int) -> list[dict]:
    """Run Segmenter.segment in-process; one Segmenter shared across (image, model) pairs."""
    from omnipose.gui._segmenter import Segmenter

    print(f"[cli] importing Segmenter… (this triggers torch/omnipose load)")
    seg = Segmenter()
    print(f"[cli] ready")

    results: list[dict] = []
    for model in models:
        for img_path in images:
            image = _load_image(img_path)
            settings = {"model": model}
            timings: list[float] = []
            for i in range(warmup + runs):
                t0 = time.perf_counter()
                _ = seg.segment(image, settings=settings)
                ms = (time.perf_counter() - t0) * 1000.0
                if i >= warmup:
                    timings.append(ms)
                tag = "warm" if i >= warmup else "cold"
                print(f"[cli] {model:<22} {img_path.name:<28} run={i:<2} {tag:<4} {ms:8.1f} ms")
            results.append({
                "path": "cli",
                "model": model,
                "image": img_path.name,
                "shape": _safe_shape(image),
                "timings_ms": timings,
            })
    return results


# ---------------------------------------------------------------------------
# GUI / HTTP path
# ---------------------------------------------------------------------------


def bench_gui(images: list[Path], models: list[str], runs: int, warmup: int, base_url: str) -> list[dict]:
    """Drive the running viewer via HTTP; one persistent session for cache locality."""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    import requests
    s = requests.Session()
    s.verify = False  # dev cert
    base = base_url.rstrip("/")

    # Mint a session id (the viewer sets it as a cookie on GET /).
    r = s.get(f"{base}/", timeout=30)
    r.raise_for_status()
    session_id = s.cookies.get("ocdkit_session") or s.cookies.get("session") or s.cookies.get("ocdkit-session")
    if not session_id:
        for c in s.cookies:
            session_id = c.value
            break
    if not session_id:
        raise SystemExit("could not obtain session id from GET /")

    # Make sure the omnipose plugin is active.
    r = s.post(f"{base}/api/plugin/select", json={"name": "omnipose"}, timeout=30)
    r.raise_for_status()

    results: list[dict] = []
    for model in models:
        for img_path in images:
            r = s.post(f"{base}/api/open_image",
                       json={"sessionId": session_id, "path": str(img_path.resolve())},
                       timeout=30)
            r.raise_for_status()

            timings_total: list[float] = []
            timings_plugin: list[float] = []
            timings_ncolor: list[float] = []
            roundtrips: list[float] = []
            shape: Optional[str] = None
            for i in range(warmup + runs):
                t0 = time.perf_counter()
                r = s.post(f"{base}/api/segment",
                           json={"sessionId": session_id, "model": model},
                           timeout=300)
                rt = (time.perf_counter() - t0) * 1000.0
                if not r.ok:
                    raise SystemExit(f"/api/segment HTTP {r.status_code}: {r.text[:300]}")
                payload = r.json()
                timing = payload.get("timing_ms") or {}
                if shape is None:
                    shape = str(payload.get("shape") or "")
                tag = "warm" if i >= warmup else "cold"
                if i >= warmup:
                    timings_total.append(float(timing.get("total", 0.0)))
                    timings_plugin.append(float(timing.get("plugin", 0.0)))
                    timings_ncolor.append(float(timing.get("ncolor", 0.0)))
                    roundtrips.append(rt)
                print(f"[gui] {model:<22} {img_path.name:<28} run={i:<2} {tag:<4} "
                      f"plugin={timing.get('plugin', 0.0):8.1f} ncolor={timing.get('ncolor', 0.0):6.1f} "
                      f"server={timing.get('total', 0.0):8.1f} roundtrip={rt:8.1f} ms")
            results.append({
                "path": "gui",
                "model": model,
                "image": img_path.name,
                "shape": shape,
                "timings_ms": timings_total,
                "timings_plugin_ms": timings_plugin,
                "timings_ncolor_ms": timings_ncolor,
                "timings_roundtrip_ms": roundtrips,
            })
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _safe_shape(arr: np.ndarray) -> str:
    try:
        return "x".join(str(d) for d in arr.shape)
    except Exception:
        return ""


def write_csv(rows: list[dict], path: Path) -> None:
    """One row per (path, model, image, run-index)."""
    fieldnames = ["path", "model", "image", "shape", "run", "ms", "plugin_ms", "ncolor_ms", "roundtrip_ms"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            timings = r["timings_ms"]
            for i, ms in enumerate(timings):
                w.writerow({
                    "path": r["path"],
                    "model": r["model"],
                    "image": r["image"],
                    "shape": r.get("shape", ""),
                    "run": i,
                    "ms": f"{ms:.2f}",
                    "plugin_ms": f"{r.get('timings_plugin_ms', [0.0]*len(timings))[i]:.2f}" if r.get("timings_plugin_ms") else "",
                    "ncolor_ms": f"{r.get('timings_ncolor_ms', [0.0]*len(timings))[i]:.2f}" if r.get("timings_ncolor_ms") else "",
                    "roundtrip_ms": f"{r.get('timings_roundtrip_ms', [0.0]*len(timings))[i]:.2f}" if r.get("timings_roundtrip_ms") else "",
                })
    print(f"\nwrote {path}")


def print_summary(rows: list[dict]) -> None:
    print("\n=== summary (mean (med, min) ms) ===")
    print(f"{'path':<5} {'model':<22} {'image':<28} {'shape':<14} {'total':<28}")
    for r in rows:
        print(f"{r['path']:<5} {r['model']:<22} {r['image']:<28} {r.get('shape', ''):<14} {_format_ms(r['timings_ms']):<28}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compare seg timing CLI vs GUI for a directory of images.")
    p.add_argument("--images-dir", type=Path, default=Path("docs/test_files"),
                   help="directory of images to bench (default: docs/test_files)")
    p.add_argument("--models", nargs="+", default=["bact_phase_affinity"])
    p.add_argument("--runs", type=int, default=5, help="timed runs per cell (default: 5)")
    p.add_argument("--warmup", type=int, default=1, help="warmup runs to discard (default: 1)")
    p.add_argument("--gui-url", default="https://127.0.0.1:8766", help="base URL of running viewer")
    p.add_argument("--csv-out", type=Path, default=Path("/tmp/bench_seg_times.csv"))
    p.add_argument("--no-cli", action="store_true", help="skip in-process path")
    p.add_argument("--no-gui", action="store_true", help="skip HTTP path")
    args = p.parse_args(list(argv) if argv is not None else None)

    images = _discover_images(args.images_dir.resolve())
    print(f"[bench] {len(images)} images, {len(args.models)} model(s), runs={args.runs} (+warmup={args.warmup})")
    for img in images:
        print(f"  - {img}")

    rows: list[dict] = []
    if not args.no_cli:
        rows += bench_cli(images, args.models, runs=args.runs, warmup=args.warmup)
    if not args.no_gui:
        rows += bench_gui(images, args.models, runs=args.runs, warmup=args.warmup, base_url=args.gui_url)

    print_summary(rows)
    write_csv(rows, args.csv_out)
    print(f"\nfull CSV: {args.csv_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
