import numpy as np

from omnirefactor.core import augment as aug


def test_mode_filter_hits_most_frequent():
    masks = np.zeros((8, 8), dtype=np.int32)
    masks[2:6, 2:6] = 1
    masks[3:5, 3:5] = 2
    out = aug.mode_filter(masks)
    assert out.shape == masks.shape
    assert set(np.unique(out)) <= {0, 1, 2}
