#!/usr/bin/env python3
"""Multi-channel cytoplasm example (script version of multi_channel_cyto.ipynb)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from omnirefactor.io.imio import get_image_files, imread
from omnirefactor.models import OmniModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run omnirefactor on multi-channel images.")
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--model-type",
        default="cyto",
        help="Model name to load (default: cyto).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available.",
    )
    parser.add_argument(
        "--channels",
        default="0,0",
        help="Channels to use, e.g. '0,1' (default: 0,0).",
    )
    parser.add_argument(
        "--channel-axis",
        type=int,
        default=-1,
        help="Channel axis (default: -1).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = Path(args.dir)
    images = get_image_files(str(image_dir))
    if not images:
        raise SystemExit(f"No images found in {image_dir}")

    model = OmniModel(gpu=args.gpu, model_type=args.model_type, use_torch=True, omni=False)

    img = imread(images[0])
    chans = tuple(int(c) for c in args.channels.split(","))
    masks, flows, styles = model.eval([img], channels=chans, channel_axis=args.channel_axis)

    print(f"Processed {images[0]}")
    print(f"masks shape: {masks[0].shape}")
    print(f"flows entries: {len(flows[0])}")
    print(f"styles shape: {np.asarray(styles[0]).shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
