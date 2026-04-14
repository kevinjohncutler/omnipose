import numpy as np
import pytest

from omnirefactor.core import flows as flow_mod
from omnirefactor.core.affinity import masks_to_affinity


def test_labels_to_flows_branches(monkeypatch):
    masks = [
        np.array([[0, 1], [0, 0]], dtype=np.uint16),
        np.array([[0, 0], [2, 0]], dtype=np.uint16),
    ]

    def fake_mtf(mask, **_):
        dist = np.ones_like(mask, dtype=np.float32)
        bd = np.zeros_like(mask, dtype=np.float32)
        heat = np.ones_like(mask, dtype=np.float32)
        veci = np.zeros((2,) + mask.shape, dtype=np.float32)
        return mask, dist, bd, heat, veci

    monkeypatch.setattr(flow_mod, "masks_to_flows", fake_mtf)

    flows = flow_mod.labels_to_flows(masks, links=None, omni=True, dim=2)
    assert flows[0].shape[0] == 5

    flows2 = flow_mod.labels_to_flows(masks, links=None, omni=False, dim=2)
    assert flows2[0].shape[0] == 4

    with pytest.raises(ValueError):
        flow_mod.labels_to_flows([np.zeros((1, 1, 1, 1, 1), dtype=np.float32)], dim=2)


def test_masks_to_flows_links_dists_and_recompute():
    masks = np.zeros((6, 6), dtype=np.int32)
    masks[2:4, 2:4] = 1
    dists = np.ones_like(masks, dtype=np.float32)

    flow_mod.masks_to_flows(
        masks,
        dists=dists,
        links=[(1, 2)],
        use_gpu=False,
        dim=2,
    )

    # mismatched affinity graph triggers recompute warning branch
    affinity_graph = np.ones((1, 1), dtype=bool)
    flow_mod.masks_to_flows(
        masks,
        affinity_graph=affinity_graph,
        dists=None,
        links=[(1, 2)],
        use_gpu=False,
        dim=2,
    )


def test_masks_to_flows_legacy_3d_dim2(monkeypatch):
    masks = np.zeros((2, 4, 4), dtype=np.int32)
    masks[:, 1:3, 1:3] = 1
    coords = np.nonzero(masks)
    affinity_graph = np.ones((9, len(coords[0])), dtype=bool)

    def fake_mtf(mask, *args, **kwargs):
        return np.zeros((2,) + mask.shape, dtype=np.float32), None

    monkeypatch.setattr(flow_mod, "masks_to_flows_torch", fake_mtf)
    result = flow_mod.masks_to_flows(
        masks,
        affinity_graph=affinity_graph,
        coords=coords,
        dim=2,
        omni=True,
        use_gpu=False,
    )
    assert result.mu.shape[0] == 3


def test_masks_to_flows_torch_center_normalize_and_return():
    masks = np.zeros((4, 4), dtype=np.int32)
    masks[0, 0] = 1
    masks[0, 2] = 1
    coords = np.nonzero(masks)
    affinity_graph = np.ones((9, len(coords[0])), dtype=bool)

    T, mu = flow_mod.masks_to_flows_torch(
        masks,
        affinity_graph,
        coords=coords,
        dists=np.ones_like(masks, dtype=np.float32),
        omni=False,
        normalize=True,
        n_iter=2,
    )
    assert T.shape == masks.shape
    assert mu.shape[0] == masks.ndim

    out = flow_mod.masks_to_flows_torch(
        masks,
        affinity_graph,
        coords=coords,
        dists=np.ones_like(masks, dtype=np.float32),
        omni=False,
        return_flows=False,
        n_iter=1,
    )
    assert out is not None


def test_masks_to_flows_torch_get_niter_and_initialize():
    masks = np.zeros((4, 4), dtype=np.int32)
    masks[1:3, 1:3] = 1
    coords = np.nonzero(masks)
    affinity_graph = np.ones((9, len(coords[0])), dtype=bool)

    # n_iter None + dists present triggers get_niter path
    T, mu = flow_mod.masks_to_flows_torch(
        masks,
        affinity_graph,
        coords=coords,
        dists=np.ones_like(masks, dtype=np.float32),
        omni=True,
        n_iter=None,
    )
    assert T.shape == masks.shape

    # coords None + initialize True triggers edt init branch
    T2, mu2 = flow_mod.masks_to_flows_torch(
        masks,
        affinity_graph,
        coords=None,
        dists=np.ones_like(masks, dtype=np.float32),
        omni=True,
        initialize=True,
        n_iter=1,
    )
    assert T2.shape == masks.shape
