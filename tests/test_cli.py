from pathlib import Path
import subprocess
import sys

import numpy as np
import tifffile


def _write_training_pair(directory: Path, stem: str) -> None:
    image = (np.random.rand(32, 32) * 255).astype(np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[8:16, 8:16] = 1

    tifffile.imwrite(directory / f"{stem}.tif", image)
    tifffile.imwrite(directory / f"{stem}_masks.tif", mask)


def _latest_model_file(model_dir: Path) -> Path:
    candidates = [p for p in model_dir.iterdir() if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def test_cli_train_then_eval_with_random_init_model(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _write_training_pair(train_dir, "sample")

    train_cmd = [
        sys.executable,
        "-m",
        "omnipose",
        "--dir",
        str(train_dir),
        "--train",
        "--pretrained_model",
        "None",
        "--n_epochs",
        "1",
        "--batch_size",
        "1",
        "--learning_rate",
        "0.01",
        "--min_train_masks",
        "1",
        "--tyx",
        "32,32",
        "--testing",
    ]
    subprocess.run(train_cmd, check=True, capture_output=True, text=True)

    model_dir = train_dir / "models"
    model_path = _latest_model_file(model_dir)

    eval_cmd = [
        sys.executable,
        "-m",
        "omnipose",
        "--dir",
        str(train_dir),
        "--pretrained_model",
        str(model_path),
        "--testing",
        "--no_npy",
    ]
    subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
