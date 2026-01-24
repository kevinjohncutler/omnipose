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


def test_normalize_stack_bright_foreground():
    vol = np.ones((2, 16, 16), dtype=np.float32)
    mask = np.zeros_like(vol, dtype=np.int32)
    mask[:, 6:10, 6:10] = 1
    vol[mask > 0] = 5.0

    out = stacks.normalize_stack(vol, mask, bright_foreground=True, equalize_foreground=1)
    assert out.shape == vol.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0


def test_normalize_stack_dark_foreground_subtractive():
    vol = np.ones((2, 16, 16), dtype=np.float32) * 5.0
    mask = np.zeros_like(vol, dtype=np.int32)
    mask[:, 6:10, 6:10] = 1
    vol[mask > 0] = 0.5

    out = stacks.normalize_stack(
        vol,
        mask,
        bright_foreground=False,
        subtractive=True,
        equalize_foreground=1,
        quantiles=[0.1, 0.9],
    )
    assert out.shape == vol.shape
