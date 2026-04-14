import sys
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[2]
ref_src = repo_root / "refactor" / "src"
sys.path.insert(0, str(ref_src))
sys.path.insert(0, str(repo_root / "src"))

from omnirefactor import io as ref_io
from omnirefactor import models as ref_models
from omnirefactor import core as ref_core
from omnirefactor.core import loss as ref_loss_module
from omnirefactor import transforms as ref_transforms
import cellpose_omni.models as base_models
import omnipose.core as base_core


def _seed_all(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _load_data():
    img_path = repo_root / "docs" / "test_files" / "Sample000033.png"
    mask_path = repo_root / "docs" / "test_files" / "masks" / "Sample000033_cp_masks.tif"
    img = ref_io.imread(str(img_path))
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img = img[..., 0]
    mask = ref_io.imread(str(mask_path))
    return [img], [mask], [None]


def _make_batch():
    imgs, masks, links = _load_data()
    X = [img[np.newaxis, ...] for img in imgs]
    Y = masks
    tyx = (224, 224)
    rsc = np.ones(len(X), np.float32)
    imgi, lbl, _, _ = ref_transforms.random_rotate_and_resize(
        X,
        Y=Y,
        rescale=rsc,
        scale_range=1.5,
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
    out = ref_core.masks_to_flows_batch(lbl, links, device=torch.device("cpu"), omni=True, dim=2, affinity_field=False)
    Xf = out[:-4]
    slices = out[-4]
    masks_t, bd_t, T_t, mu_t = [torch.stack([x[(Ellipsis,) + slc] for slc in slices]) for x in Xf]
    lbl_t = ref_core.batch_labels(masks_t, bd_t, T_t, mu_t, tyx, dim=2, nclasses=3, device=torch.device("cpu"))
    imgi_t = torch.tensor(np.stack(imgi))
    return imgi_t, lbl_t


if __name__ == "__main__":
    _seed_all(0)

    # disable skimage noise to remove nondeterminism
    base_core.SKIMAGE_ENABLED = False
    from omnirefactor.core import augment as ref_aug
    ref_aug.SKIMAGE_ENABLED = False

    ref_model = ref_models.OmniModel(gpu=False, model_type="bact_phase_affinity")
    base_model = base_models.CellposeModel(gpu=False, model_type="bact_phase_affinity", omni=True)
    base_model.net.load_state_dict(ref_model.net.state_dict())
    ref_model._set_criterion()
    base_model._set_criterion()

    imgi, lbl = _make_batch()

    with torch.no_grad():
        y_ref, _ = ref_model.net(imgi)
        y_base, _ = base_model.net(imgi)

    max_diff = float(torch.max(torch.abs(y_ref - y_base)).item())
    print(f"max pred diff: {max_diff:.6f}")
