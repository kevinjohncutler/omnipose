import numpy as np
from scipy.ndimage import distance_transform_edt

from omnirefactor.core import masks as masks_module


def _make_synthetic_masks():
    masks = np.zeros((64, 64), dtype=np.int32)
    masks[8:28, 10:30] = 1
    masks[36:52, 34:54] = 2
    return masks


def _make_flows(masks):
    dist = distance_transform_edt(masks > 0).astype(np.float32)
    gy, gx = np.gradient(dist)
    dP = np.stack([gy, gx]).astype(np.float32)
    return dP, dist


def test_compute_masks_basic():
    masks = _make_synthetic_masks()
    dP, dist = _make_flows(masks)

    iscell = masks > 0
    coords = np.nonzero(iscell)
    out_masks, p, tr, bounds, affinity = masks_module.compute_masks(
        dP,
        dist,
        iscell=iscell,
        coords=coords,
        affinity_seg=False,
        cluster=False,
        suppress=False,
        flow_threshold=0,
        omni=True,
        use_gpu=False,
        nclasses=2,
        dim=2,
    )

    assert out_masks.shape == masks.shape
    assert bounds.shape == masks.shape
    assert affinity == []


def test_compute_masks_affinity_seg_cluster_false():
    masks = _make_synthetic_masks()
    dP, dist = _make_flows(masks)

    iscell = masks > 0
    coords = np.nonzero(iscell)
    out_masks, p, tr, bounds, affinity = masks_module.compute_masks(
        dP,
        dist,
        iscell=iscell,
        coords=coords,
        affinity_seg=True,
        cluster=False,
        despur=True,
        suppress=False,
        flow_threshold=0,
        omni=True,
        use_gpu=False,
        nclasses=2,
        dim=2,
    )

    assert out_masks.shape == masks.shape
    assert bounds.shape == masks.shape
    assert isinstance(affinity, np.ndarray)


def test_compute_masks_affinity_seg_cluster_true():
    masks = _make_synthetic_masks()
    dP, dist = _make_flows(masks)

    iscell = masks > 0
    coords = np.nonzero(iscell)
    out_masks, p, tr, bounds, affinity = masks_module.compute_masks(
        dP,
        dist,
        iscell=iscell,
        coords=coords,
        affinity_seg=True,
        cluster=True,
        despur=False,
        suppress=False,
        flow_threshold=0,
        omni=True,
        use_gpu=False,
        nclasses=2,
        dim=2,
    )

    assert out_masks.shape == masks.shape
    assert bounds.shape == masks.shape
    assert isinstance(affinity, np.ndarray)
