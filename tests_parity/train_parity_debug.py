import sys
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[2]
ref_src = repo_root / "refactor" / "src"
sys.path.insert(0, str(ref_src))
sys.path.insert(0, str(repo_root / "src"))

from omnirefactor import io as ref_io
from omnirefactor import core as ref_core
from omnirefactor import transforms as ref_transforms
import omnipose.core as base_core
import cellpose_omni.transforms as base_transforms


def _load_data(names):
    imgs = []
    masks = []
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
    return imgs, masks


def _max_abs(a, b):
    if isinstance(a, slice) and isinstance(b, slice):
        return 0.0
    if isinstance(a, set) and isinstance(b, set):
        return 0.0 if a == b else 1.0
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return max(_max_abs(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    if getattr(a, "dtype", None) == np.bool_ or getattr(b, "dtype", None) == np.bool_:
        return float(np.max(np.logical_xor(a, b)))
    return float(np.max(np.abs(a - b)))


def _iter_pairs(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        for item_a, item_b in zip(a, b):
            yield item_a, item_b
    else:
        yield a, b


def compare(names=("Sample000033",)):
    imgs, masks = _load_data(names)
    X = [m[np.newaxis, ...] for m in imgs]  # add channel axis
    Y = masks
    tyx = (224, 224)
    scale_range = 1.5
    rescale = np.ones(len(X), np.float32)

    np.random.seed(0)
    r_imgi, r_lbl, r_scale = ref_transforms.random_rotate_and_resize(
        X,
        Y=Y,
        rescale=rescale,
        scale_range=scale_range,
        unet=False,
        tyx=tyx,
        inds=list(range(len(X))),
        omni=True,
        dim=2,
        nchan=1,
        nclasses=3,
        device=torch.device("cpu"),
        allow_blank_masks=False,
    )

    np.random.seed(0)
    b_imgi, b_lbl, b_scale = base_transforms.random_rotate_and_resize(
        X,
        Y=Y,
        rescale=rescale,
        scale_range=scale_range,
        unet=False,
        tyx=tyx,
        inds=list(range(len(X))),
        omni=True,
        dim=2,
        nchan=1,
        nclasses=3,
        device=torch.device("cpu"),
        allow_blank_masks=False,
    )

    print("rrr imgi max diff", _max_abs(r_imgi, b_imgi))
    print("rrr lbl max diff", _max_abs(r_lbl, b_lbl))
    print("rrr scale diff", float(abs(r_scale - b_scale)))

    np.random.seed(0)
    r_out = ref_core.masks_to_flows_batch(r_lbl, [None] * len(r_lbl), device=torch.device("cpu"), omni=True, dim=2, affinity_field=False)
    np.random.seed(0)
    b_out = base_core.masks_to_flows_batch(b_lbl, [None] * len(b_lbl), device=torch.device("cpu"), omni=True, dim=2, affinity_field=False)

    for idx, (ra, ba) in enumerate(zip(r_out, b_out)):
        if isinstance(ra, list) and ra and isinstance(ra[0], slice):
            continue
        for j, (ra_i, ba_i) in enumerate(_iter_pairs(ra, ba)):
            suffix = f"[{j}]" if isinstance(ra, (list, tuple)) else ""
            print(f"masks_to_flows_batch[{idx}]{suffix} max diff", _max_abs(ra_i, ba_i))

    # build batch labels
    Xr = r_out[:-4]
    slices_r = r_out[-4]
    masks_r, bd_r, T_r, mu_r = [torch.stack([x[(Ellipsis,) + slc] for slc in slices_r]) for x in Xr]
    rb = ref_core.batch_labels(masks_r, bd_r, T_r, mu_r, tyx, dim=2, nclasses=3, device=torch.device("cpu"))

    Xb = b_out[:-4]
    slices_b = b_out[-4]
    masks_b, bd_b, T_b, mu_b = [torch.stack([x[(Ellipsis,) + slc] for slc in slices_b]) for x in Xb]
    bb = base_core.batch_labels(masks_b, bd_b, T_b, mu_b, tyx, dim=2, nclasses=3, device=torch.device("cpu"))

    print("batch_labels max diff", _max_abs(rb, bb))


if __name__ == "__main__":
    compare()
