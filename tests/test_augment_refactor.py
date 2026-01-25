import numpy as np

from omnirefactor.transforms import augment as aug


def test_mode_filter_hits_most_frequent():
    masks = np.zeros((8, 8), dtype=np.int32)
    masks[2:6, 2:6] = 1
    masks[3:5, 3:5] = 2
    out = aug.mode_filter(masks)
    assert out.shape == masks.shape
    assert set(np.unique(out)) <= {0, 1, 2}


def test_rotate_and_do_warp_smoke():
    img = np.arange(25, dtype=np.float32).reshape(5, 5)
    out = aug.rotate(img, theta=0.0)
    assert out.shape == img.shape
    assert np.isfinite(out).all()

    M_inv = np.eye(2, dtype=np.float32)
    warped = aug.do_warp(img, M_inv, tyx=img.shape, offset=0, order=1, mode="nearest")
    assert warped.shape == img.shape
    assert np.isfinite(warped).all()
