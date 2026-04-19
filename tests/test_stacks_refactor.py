import numpy as np

from omnirefactor.transforms import stacks


def test_make_unique_offsets_labels():
    masks = np.zeros((2, 4, 4), dtype=np.int32)
    masks[0, 1:3, 1:3] = 1
    masks[1, 1:3, 1:3] = 1

    unique = stacks.make_unique(masks)
    labels0 = set(np.unique(unique[0])) - {0}
    labels1 = set(np.unique(unique[1])) - {0}

    assert labels0 == {1}
    assert labels1 and labels0.isdisjoint(labels1)
