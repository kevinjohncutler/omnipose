import numpy as np

from omnirefactor.utils import morph


def test_get_edge_masks_selective():
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[0:4, 0:4] = 1  # touches boundary with interior
    labels[3:5, 4:6] = 2  # touches boundary

    dists = np.zeros_like(labels, dtype=np.float32)
    dists[labels == 1] = 1.0
    dists[labels == 2] = 10.0
    dists[2:4, 2:4] = 10.0  # higher interior distances for label 1

    out = morph.get_edge_masks(labels, dists)
    assert np.any(out == 2)
    assert not np.any(out == 1)


def test_fill_holes_and_remove_small_masks_no_skimage(monkeypatch):
    monkeypatch.setattr(morph, "SKIMAGE_ENABLED", False, raising=False)
    masks = np.zeros((6, 6), dtype=np.int32)
    masks[1:5, 1:5] = 1
    masks[2:4, 2:4] = 0  # hole
    masks[0, 5] = 2  # tiny

    out = morph.fill_holes_and_remove_small_masks(masks, min_size=4, hole_size=10, dim=2)
    assert np.any(out == 1)
    assert out[0, 5] == 0
    assert out[2, 2] == 1


def test_clean_boundary_removes_small_edge():
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[0:2, 0:2] = 1  # small edge
    labels[2:5, 2:5] = 2  # interior

    out = morph.clean_boundary(labels, boundary_thickness=1, area_thresh=10, cutoff=0.5)
    assert not np.any(out == 1)
    assert np.any(out == 2)
