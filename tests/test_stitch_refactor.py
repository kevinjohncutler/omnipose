import numpy as np

from omnirefactor.utils import stitch as stitch_mod


def _make_label(shape, loc, label):
    masks = np.zeros(shape, dtype=np.int32)
    y, x = loc
    masks[y:y + 2, x:x + 2] = label
    return masks


def test_stitch3d_branches():
    m0 = _make_label((6, 6), (1, 1), 1)
    m1 = _make_label((6, 6), (1, 1), 1)  # overlap with m0 -> iou branch
    m2 = np.zeros((6, 6), dtype=np.int32)  # empty triggers empty==1 branch later
    masks = [m0.copy(), m1.copy(), m2.copy()]
    out = stitch_mod.stitch3D(masks, stitch_threshold=0.25)
    assert len(out) == 3
    assert out[1].max() >= 1
    assert out[2].max() >= 0
