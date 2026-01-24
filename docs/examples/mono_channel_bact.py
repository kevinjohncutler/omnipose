#!/usr/bin/env python3
"""Monochannel bacteria example (script version of mono_channel_bact.ipynb)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from omnirefactor.io.imio import get_image_files, imread
from omnirefactor.models import OmniModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run omnirefactor on mono-channel images.")
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--model-type",
        default="bact_phase_affinity",
        help="Model name to load (default: bact_phase_affinity).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available.",
    )
    parser.add_argument(
        "--img-filter",
        default="",
        help="Suffix filter for image filenames (default: none).",
    )
    parser.add_argument(
        "--channels",
        default=None,
        help="Channel specification string, e.g. '0,0'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = Path(args.dir)
    images = get_image_files(str(image_dir), img_filter=args.img_filter)
    if not images:
        raise SystemExit(f"No images found in {image_dir}")

    model = OmniModel(gpu=args.gpu, model_type=args.model_type, use_torch=True, omni=True)

    # Use the first image to mirror the notebook flow.
    img = imread(images[0])
    chans = None
    if args.channels:
        chans = tuple(int(c) for c in args.channels.split(","))

    masks, flows, styles = model.eval([img], channels=chans, channel_axis=-1)
    print(f"Processed {images[0]}")
    print(f"masks shape: {masks[0].shape}")
    print(f"flows entries: {len(flows[0])}")
    print(f"styles shape: {np.asarray(styles[0]).shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
