import sys
from pathlib import Path

import numpy as np
import torch

import random


def _seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

repo_root = Path(__file__).resolve().parents[2]

# refactor package
refactor_src = repo_root / "refactor" / "src"
sys.path.insert(0, str(refactor_src))

# original package
sys.path.insert(0, str(repo_root / "src"))

from omnipose import io as ref_io
from omnipose import models as ref_models
import cellpose_omni.models as base_models

def _patch_random_noise():
    try:
        import omnipose.core as base_core
        from omnipose.core import augment as ref_aug
    except Exception:
        return
    base_core.random_noise = _deterministic_random_noise
    ref_aug.random_noise = _deterministic_random_noise



def _load_data(names):
    imgs = []
    masks = []
    links = []
    for name in names:
        img_path = repo_root / "docs" / "test_files" / f"{name}.png"
        if not img_path.exists():
            img_path = repo_root / "docs" / "test_files" / f"{name}.tif"
        if not img_path.exists():
            img_path = repo_root / "docs" / "test_files" / f"{name}.tiff"
        mask_path = repo_root / "docs" / "test_files" / "masks" / f"{name}_cp_masks.tif"
        img = ref_io.imread(str(img_path))
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img[..., 0]
        mask = ref_io.imread(str(mask_path))
        imgs.append(img)
        masks.append(mask)
        links.append(None)
    return imgs, masks, links


def _run_train(model, name, names, train_kwargs):
    _seed_all(0)
    losses = []
    orig = model._train_step

    def _wrap(x, lbl):
        loss = orig(x, lbl)
        losses.append(float(loss))
        return loss

    model._train_step = _wrap
    train_data, train_labels, train_links = _load_data(names)
    model.train(
        train_data,
        train_labels,
        train_links=train_links,
        channels=None,
        normalize=True,
        save_path=None,
        save_every=10000,
        save_each=False,
        momentum=0.9,
        SGD=True,
        weight_decay=0.0,
        dataloader=False,
        num_workers=0,
        rescale=False,
        min_train_masks=1,
        timing=False,
        do_autocast=False,
        affinity_field=False,
        **train_kwargs,
    )
    model._train_step = orig
    print(f"{name}: steps={len(losses)} first={losses[0]:.6f} last={losses[-1]:.6f}")
    if not np.isfinite(losses).all():
        raise SystemExit(f"{name}: non-finite loss")
    return losses


def _format_names(names):
    return "+".join(names)


def _write_log(rows):
    log_path = repo_root / "refactor" / "tests_parity" / "train_sweep_log.md"
    header = (
        "| dataset | lr | batch | epochs | nimg_per_epoch | ref_first | ref_last | base_first | base_last | max_abs_diff | match |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {dataset} | {lr:.3g} | {batch} | {epochs} | {nimg} | {ref_first:.6f} | {ref_last:.6f} | "
            "{base_first:.6f} | {base_last:.6f} | {max_diff:.6f} | {match} |\n".format(**row)
        )
    log_path.write_text("".join(lines))
    return log_path


if __name__ == "__main__":
    _seed_all(0)

    datasets = [
        ("Sample000033",),
        ("Sample000033", "Sample000193"),
    ]
    lrs = [0.05, 0.1]
    batch_sizes = [1, 2]
    epochs = 2

    rows = []
    for names in datasets:
        for lr in lrs:
            for batch in batch_sizes:
                train_kwargs = {
                    "learning_rate": lr,
                    "n_epochs": epochs,
                    "batch_size": batch,
                    "nimg_per_epoch": min(len(names), 2),
                }
                _seed_all(0)
                ref_model = ref_models.OmniModel(gpu=False, model_type="bact_phase_affinity")
                _seed_all(0)
                base_model = base_models.CellposeModel(gpu=False, model_type="bact_phase_affinity", omni=True)
                # align initial weights for strict parity
                base_model.net.load_state_dict(ref_model.net.state_dict())

                ref_losses = _run_train(ref_model, "refactor", names, train_kwargs)
                base_losses = _run_train(base_model, "base", names, train_kwargs)
                max_diff = float(np.max(np.abs(np.array(ref_losses) - np.array(base_losses))))
                rows.append({
                    "dataset": _format_names(names),
                    "lr": lr,
                    "batch": batch,
                    "epochs": epochs,
                    "nimg": train_kwargs["nimg_per_epoch"],
                    "ref_first": ref_losses[0],
                    "ref_last": ref_losses[-1],
                    "base_first": base_losses[0],
                    "base_last": base_losses[-1],
                    "max_diff": max_diff,
                    "match": "yes" if max_diff < 1e-5 else "no",
                })

    log_path = _write_log(rows)
    print(f"Wrote sweep log: {log_path}")
