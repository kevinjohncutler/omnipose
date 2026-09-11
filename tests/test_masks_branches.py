import numpy as np
import pytest
import torch
from scipy.ndimage import distance_transform_edt

from omnipose.core import masks as masks_module
from omnipose import utils


def _make_2d_case():
    masks = np.zeros((8, 8), dtype=np.int32)
    masks[2:5, 2:5] = 1
    dist = distance_transform_edt(masks > 0).astype(np.float32)
    gy, gx = np.gradient(dist)
    dP = np.stack([gy, gx]).astype(np.float32)
    bd = np.zeros_like(dist, dtype=np.float32)
    return masks, dP, dist, bd


def test_compute_masks_coords_only():
    masks, dP, dist, bd = _make_2d_case()
    coords = np.nonzero(masks)
    out = masks_module.compute_masks(
        dP,
        dist,
        bd=bd,
        iscell=None,
        coords=coords,
        flow_threshold=0,
        suppress=False,
        affinity_seg=False,
        omni=True,
        verbose=True,
        use_gpu=False,
        nclasses=2,
        dim=2,
    )
    assert out[0].shape == masks.shape


def test_compute_masks_p_branch_debug():
    masks, dP, dist, bd = _make_2d_case()
    iscell = masks > 0
    p = np.zeros((2,) + masks.shape, dtype=np.float32)

    out = masks_module.compute_masks(
        dP,
        dist,
        bd=bd,
        iscell=iscell,
        coords=None,
        p=p,
        flow_threshold=0,
        suppress=False,
        affinity_seg=False,
        omni=True,
        verbose=True,
        debug=True,
        use_gpu=False,
        nclasses=2,
        dim=2,
    )
    assert out[0].shape == masks.shape
    assert len(out) >= 6


def test_compute_masks_do3d_suppress_flowfactor():
    masks = np.zeros((2, 4, 4), dtype=np.int32)
    masks[:, 1:3, 1:3] = 1
    dist = np.ones_like(masks, dtype=np.float32)
    dP = np.zeros((3,) + masks.shape, dtype=np.float32)
    bd = np.zeros_like(masks, dtype=np.float32)
    coords = np.nonzero(masks)
    p = np.zeros((3,) + masks.shape, dtype=np.float32)

    out = masks_module.compute_masks(
        dP,
        dist,
        bd=bd,
        iscell=masks > 0,
        coords=coords,
        p=p,
        flow_threshold=0,
        suppress=True,
        affinity_seg=False,
        omni=True,
        do_3D=True,
        use_gpu=False,
        nclasses=2,
        dim=3,
    )
    assert out[0].ndim == 3


def test_compute_masks_affinity_resize_calc_trace():
    masks, dP, dist, bd = _make_2d_case()
    iscell = masks > 0
    coords = np.nonzero(iscell)
    steps, _, _, _, _ = utils.kernel_setup(2)
    affinity_graph = np.ones((len(steps), coords[0].size), dtype=bool)

    out = masks_module.compute_masks(
        dP,
        dist,
        bd=bd,
        iscell=iscell,
        coords=coords,
        affinity_graph=affinity_graph,
        affinity_seg=True,
        cluster=False,
        calc_trace=True,
        resize=(masks.shape[0] + 1, masks.shape[1] + 1),
        flow_threshold=0,
        suppress=False,
        omni=True,
        use_gpu=False,
        nclasses=2,
        dim=2,
        verbose=True,
    )
    assert out[0].shape == (masks.shape[0] + 1, masks.shape[1] + 1)


def test_compute_masks_affinity_calc_trace_no_resize():
    masks, dP, dist, bd = _make_2d_case()
    iscell = masks > 0
    coords = np.nonzero(iscell)
    steps, _, _, _, _ = utils.kernel_setup(2)
    affinity_graph = np.ones((len(steps), coords[0].size), dtype=bool)

    out = masks_module.compute_masks(
        dP,
        dist,
        bd=bd,
        iscell=iscell,
        coords=coords,
        affinity_graph=affinity_graph,
        affinity_seg=True,
        cluster=False,
        calc_trace=True,
        flow_threshold=0,
        suppress=False,
        omni=True,
        use_gpu=False,
        nclasses=2,
        dim=2,
    )
    assert out[0].shape == masks.shape


def test_get_masks_border_nclasses():
    masks, dP, dist, bd = _make_2d_case()
    iscell = masks > 0
    coords = np.nonzero(iscell)
    p = np.zeros((2,) + masks.shape, dtype=np.float32)
    inds = coords

    masks_module.get_masks(
        p,
        bd,
        dist,
        iscell,
        inds,
        nclasses=iscell.ndim + 2,
        cluster=False,
        diam_threshold=0.0,
        verbose=True,
    )



def test_get_masks_nclasses_one():
    masks, dP, dist, bd = _make_2d_case()
    iscell = masks > 0
    coords = np.nonzero(iscell)
    p = np.zeros((2,) + masks.shape, dtype=np.float32)
    masks_module.get_masks(
        p,
        bd,
        dist,
        iscell,
        coords,
        nclasses=1,
        cluster=False,
    )


def test_follow_flows_niter_and_empty_inds():
    masks, dP, dist, bd = _make_2d_case()
    inds = np.array([[]], dtype=np.int64)
    p, coords, tr = masks_module.follow_flows(
        dP,
        dist,
        inds,
        niter=None,
        use_gpu=False,
        device=torch.device("cpu"),
        omni=True,
    )
    assert tr is None


def test_flow_error_shape_mismatch():
    mask = np.zeros((4, 4), dtype=np.int32)
    dP_net = np.zeros((2, 5, 5), dtype=np.float32)
    assert masks_module.flow_error(mask, dP_net) is None


def test_get_masks_cp_dim3():
    p = np.zeros((3, 2, 3, 3), dtype=np.float32)
    p[:, :, 1, 1] = 2.0
    iscell = np.zeros((2, 3, 3), dtype=bool)
    iscell[:, 1, 1] = True
    out = masks_module.get_masks_cp(p, iscell=iscell)
    assert out.shape == p.shape[1:]




def test_compute_masks_empty_result_is_numpy():
    """The 'no cell pixels' branch must not leak the torch `iscell` tensor.

    On the GPU hysteresis path `iscell` is a tensor; callers cast the returned
    labels with `.astype(...)`, so returning it directly raised
    ``AttributeError: 'Tensor' object has no attribute 'astype'`` whenever a
    threshold/rescale combination left no foreground.
    """
    import numpy as np
    import torch
    from omnipose.core import compute_masks

    dP = np.zeros((2, 24, 24), np.float32)
    dist = np.full((24, 24), -5.0, np.float32)  # nothing clears the threshold
    bd = np.zeros((24, 24), np.float32)

    for use_gpu, device in _empty_branch_devices():
        out = compute_masks(dP=dP, dist=dist, bd=bd, mask_threshold=0.0,
                            affinity_seg=True, omni=True, nclasses=3, dim=2,
                            use_gpu=use_gpu, device=device)
        mask = out[0]
        assert isinstance(mask, np.ndarray), f"tensor leaked with use_gpu={use_gpu}"
        assert int(mask.sum()) == 0
        mask.astype(np.uint32)  # the call that used to raise


def _empty_branch_devices():
    """(use_gpu, device) pairs to exercise: always CPU, plus GPU when present."""
    cases = [(False, None)]
    try:
        from omnipose.gpu import get_device
        device, available = get_device(gpu_number=0)
        if available:
            cases.append((True, device))
    except Exception:
        pass
    return cases
