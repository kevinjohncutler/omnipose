import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure refactor package is importable
REFRACTOR_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REFRACTOR_SRC))

from omnirefactor import io as cpio
from omnirefactor import models as omnimodels
from omnirefactor.gpu import use_gpu


def _prepare_overlay(img):
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        # Convert RGB/RGBA to grayscale for overlay.
        rgb = img[..., :3].astype(np.float32)
        return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    return img.squeeze()


def _to_grayscale(img):
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        rgb = img[..., :3].astype(np.float32)
        return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    return img


def _normalize_mask(mask, target_shape):
    arr = np.asarray(mask)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        elif arr.shape[1:] == target_shape:
            # Assume Z,Y,X and take middle slice for 2D comparison.
            arr = arr[arr.shape[0] // 2]
        elif arr.shape[:2] == target_shape:
            arr = arr[..., 0]
    elif arr.ndim == 1 and arr.size == np.prod(target_shape):
        arr = arr.reshape(target_shape)
    elif arr.ndim == 2 and arr.shape != target_shape and arr.T.shape == target_shape:
        arr = arr.T
    return arr


def _labels_to_rgba(labels, alpha=0.4):
    import ncolor
    from ocdkit.plot import sinebow

    arr = np.asarray(labels)
    if arr.size == 0:
        return np.zeros((*arr.shape, 4), dtype=np.float32)
    ncolor_labels, ncolors = ncolor.label(arr, max_depth=20, return_n=True)
    ncolors = int(max(ncolors, 1))
    color_dict = sinebow(ncolors)
    cmap = np.zeros((ncolors + 1, 4), dtype=np.float32)
    for idx, rgba in color_dict.items():
        if idx <= ncolors:
            cmap[idx] = rgba
    rgba = cmap[ncolor_labels]
    rgba[..., 3] *= alpha
    return rgba


def _plot_comparison(image_path, img, base_mask, ref_mask, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlay = _prepare_overlay(img)
    base_mask = _normalize_mask(base_mask, overlay.shape)
    ref_mask = _normalize_mask(ref_mask, overlay.shape)
    diff = (base_mask != ref_mask).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    axes[0].imshow(overlay, cmap="gray")
    axes[0].set_title("Input")

    axes[1].imshow(overlay, cmap="gray")
    axes[1].imshow(_labels_to_rgba(base_mask), interpolation="nearest")
    axes[1].set_title("Base mask")

    axes[2].imshow(overlay, cmap="gray")
    axes[2].imshow(_labels_to_rgba(ref_mask), interpolation="nearest")
    axes[2].set_title("Refactor mask")

    axes[3].imshow(diff, cmap="gray", interpolation="nearest")
    axes[3].set_title("Diff (base != ref)")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Parity comparison: {Path(image_path).name}")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_dataset(model, img, params):
    from omnirefactor.data.eval import eval_set
    dataset = eval_set([img], dim=2, channel_axis=None, rescale=1.0, tile=False)
    masks, _ = model.eval(dataset, **params)
    return masks


def main():
    import omnirefactor
    print(f"omnirefactor_module: {omnirefactor.__file__}")
    repo_root = Path(__file__).resolve().parents[2]
    test_dir = repo_root / "docs" / "test_files"
    files = cpio.get_image_files(str(test_dir))
    image_path = files[0]

    img = cpio.imread(image_path)
    img = _to_grayscale(img)
    model_name = "bact_phase_affinity"

    channel_axis = -1 if (img.ndim == 3 and img.shape[-1] in (3, 4)) else None
    if channel_axis is not None:
        img = img[..., 0]
        channel_axis = None

    params = {
        "channels": None,
        "channel_axis": channel_axis,
        "rescale": None,
        "mask_threshold": -2,
        "flow_threshold": 0,
        "transparency": True,
        "omni": True,
        "cluster": True,
        "resample": True,
        "verbose": False,
        "tile": False,
        "augment": False,
        "affinity_seg": True,
        "show_progress": False,
    }

    device, gpu_available = use_gpu()
    model = omnimodels.OmniModel(
        gpu=gpu_available,
        device=device if gpu_available else None,
        model_type=model_name,
    )

    t0 = time.time()
    masks, _ = model.eval(img, **params)
    t1 = time.time()

    mask = masks[0] if isinstance(masks, list) else masks

    ref_path = repo_root / "docs" / "test_files" / "masks" / "Sample000033_cp_masks.tif"
    ref_mask = cpio.imread(str(ref_path))

    dataset_masks = _run_dataset(model, img, params)
    dataset_mask = dataset_masks[0] if isinstance(dataset_masks, list) else dataset_masks
    dataset_mask = _normalize_mask(dataset_mask, ref_mask.shape)
    dataset_identical = np.array_equal(dataset_mask, ref_mask)
    dataset_diff = np.abs(dataset_mask.astype(np.int64) - ref_mask.astype(np.int64))
    dataset_max_diff = int(dataset_diff.max()) if dataset_diff.size else 0

    mask = _normalize_mask(mask, ref_mask.shape)
    identical = np.array_equal(mask, ref_mask)
    diff = np.abs(mask.astype(np.int64) - ref_mask.astype(np.int64))
    max_diff = int(diff.max()) if diff.size else 0

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "parity_overlay_Sample000033_bact_phase_affinity.png"
    _plot_comparison(image_path, img, ref_mask, mask, out_path)

    print("Segmentation parity run")
    print(f"image: {image_path}")
    print(f"model_type: {model_name}")
    print(f"eval_s: {t1 - t0:.3f}")
    print(f"mask_equal: {identical}")
    print(f"mask_max_abs_diff: {max_diff}")
    print(f"dataset_mask_equal: {dataset_identical}")
    print(f"dataset_mask_max_abs_diff: {dataset_max_diff}")
    print(f"overlay_path: {out_path}")

    if not identical:
        raise SystemExit("Parity check failed: masks differ")
    if not dataset_identical:
        raise SystemExit("Dataset parity check failed: masks differ")


if __name__ == "__main__":
    main()
