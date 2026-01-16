import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_SRC = REPO_ROOT / "src"
REFACTOR_SRC = REPO_ROOT / "refactor" / "src"
sys.path.insert(0, str(BASE_SRC))
sys.path.insert(0, str(REFACTOR_SRC))

from omnirefactor import io as cpio
from omnirefactor import models as omnimodels
from omnirefactor.gpu import use_gpu
from cellpose_omni import models as base_models


def _normalize_mask(mask, target_shape):
    arr = np.asarray(mask)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        elif arr.shape[1:] == target_shape:
            arr = arr[arr.shape[0] // 2]
        elif arr.shape[:2] == target_shape:
            arr = arr[..., 0]
    elif arr.ndim == 1 and arr.size == np.prod(target_shape):
        arr = arr.reshape(target_shape)
    elif arr.ndim == 2 and arr.shape != target_shape and arr.T.shape == target_shape:
        arr = arr.T
    return arr


def _to_grayscale(img):
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        rgb = img[..., :3].astype(np.float32)
        return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    return img


def _param_table_row(params, equal, max_diff):
    return (
        f"| {params['flow_threshold']} | {params['mask_threshold']} | {params['niter']} | "
        f"{params['omni']} | {params['cluster']} | {params['resample']} | {params['tile']} | "
        f"{params['affinity_seg']} | {equal} | {max_diff} |"
    )


def main():
    import omnirefactor
    print(f"omnirefactor_module: {omnirefactor.__file__}")
    test_dir = REPO_ROOT / "docs" / "test_files"
    files = cpio.get_image_files(str(test_dir))
    image_path = files[0]

    img = cpio.imread(image_path)
    img = _to_grayscale(img)
    channel_axis = -1 if (img.ndim == 3 and img.shape[-1] in (3, 4)) else None
    if channel_axis is not None:
        img = img[..., 0]
        channel_axis = None

    model_name = "bact_phase_affinity"
    device, gpu_available = use_gpu()
    ref_model = omnimodels.OmniModel(
        gpu=gpu_available,
        device=device if gpu_available else None,
        model_type=model_name,
    )
    base_model = base_models.CellposeModel(
        gpu=gpu_available,
        device=device if gpu_available else None,
        model_type=model_name,
    )
    ref_path = REPO_ROOT / "docs" / "test_files" / "masks" / "Sample000033_cp_masks.tif"
    ref_mask = cpio.imread(str(ref_path))

    float_params = {
        "flow_threshold": [0.0],
        "mask_threshold": [-2.0],
        "niter": [None, 10],
    }

    bool_params = {
        "omni": [True],
        "cluster": [False, True],
        "resample": [False],
        "affinity_seg": [False, True],
        "tile": [False, True],
    }

    out_path = Path(__file__).resolve().parent / "segmentation_sweep_log.md"
    header = [
        "# Segmentation Sweep Log",
        f"- image: {image_path}",
        f"- reference_mask: {ref_path}",
        f"- model_type: {model_name}",
        f"- channel_axis: {channel_axis}",
        f"- gpu_available: {gpu_available}",
        "",
        "| flow_threshold | mask_threshold | niter | omni | cluster | resample | tile | affinity_seg | equal | max_diff |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    out_path.write_text("\n".join(header) + "\n")

    mismatch_count = 0
    rows = []
    t0 = time.time()

    for flow_th, mask_th, niter in product(
        float_params["flow_threshold"],
        float_params["mask_threshold"],
        float_params["niter"],
    ):
        for omni, cluster, resample, tile, affinity_seg in product(
            bool_params["omni"],
            bool_params["cluster"],
            bool_params["resample"],
            bool_params["tile"],
            bool_params["affinity_seg"],
        ):
            params = {
                "channels": None,
                "channel_axis": channel_axis,
                "rescale": None,
                "mask_threshold": mask_th,
                "flow_threshold": flow_th,
                "transparency": True,
                "omni": omni,
                "cluster": cluster,
                "resample": resample,
                "niter": niter,
                "tile": tile,
                "affinity_seg": affinity_seg,
                "show_progress": False,
            }

            key = (
                str(flow_th),
                str(mask_th),
                str(niter),
                str(omni),
                str(cluster),
                str(resample),
                str(tile),
                str(affinity_seg),
            )
            base_masks, _, _ = base_model.eval(img, **params)
            base_mask = base_masks[0] if isinstance(base_masks, list) else base_masks

            ref_masks, _, _ = ref_model.eval(img, **params)
            ref_mask_eval = ref_masks[0] if isinstance(ref_masks, list) else ref_masks

            base_mask = _normalize_mask(base_mask, ref_mask.shape)
            ref_mask_eval = _normalize_mask(ref_mask_eval, ref_mask.shape)
            equal = np.array_equal(base_mask, ref_mask_eval)
            diff = np.abs(base_mask.astype(np.int64) - ref_mask_eval.astype(np.int64))
            max_diff = int(diff.max()) if diff.size else 0
            row = _param_table_row(params, equal, max_diff)
            rows.append(row)

            if not equal:
                mismatch_count += 1

    elapsed = time.time() - t0
    summary = [
        "",
        "## Summary",
        f"- elapsed_s: {elapsed:.2f}",
        f"- mismatches: {mismatch_count}",
    ]
    with out_path.open("a") as handle:
        handle.write("\n".join(rows) + "\n")
        handle.write("\n".join(summary) + "\n")

    print(f"Sweep complete, log written to: {out_path}")


if __name__ == "__main__":
    main()
