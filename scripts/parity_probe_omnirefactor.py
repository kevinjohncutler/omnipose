#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import imageio.v3 as iio

from omnirefactor.models import OmniModel


def tensor_hash(t: torch.Tensor) -> str:
    data = t.detach().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def state_dict_hash(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        v = sd[k]
        if torch.is_tensor(v):
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data)
    imgs = sorted(data_dir.glob("*_img.*"))
    if not imgs:
        raise SystemExit("No *_img.* files found")

    img = iio.imread(imgs[0]).astype(np.float32)
    if img.ndim == 2:
        img = img[None, None]
    elif img.ndim == 3:
        img = img.transpose(2, 0, 1)
        img = img[None]
    x = torch.from_numpy(img)

    model = OmniModel(gpu=False, omni=True, use_torch=True, nclasses=2, dim=2)
    with torch.no_grad():
        out = model.net(x)[0]

    print("img_path:", imgs[0])
    print("input_hash:", tensor_hash(x))
    print("output_hash:", tensor_hash(out))
    print("state_hash:", state_dict_hash(model.net.state_dict()))


if __name__ == "__main__":
    main()
