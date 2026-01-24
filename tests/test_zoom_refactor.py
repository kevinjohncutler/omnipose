import numpy as np
import torch

from omnirefactor.transforms import zoom


def test_torch_zoom_scale_and_size():
    x = torch.zeros((1, 1, 4, 6))
    y = zoom.torch_zoom(x, scale_factor=0.5, dim=2)
    assert y.shape[-2:] == (2, 3)

    z = zoom.torch_zoom(x, size=(5, 5))
    assert z.shape[-2:] == (5, 5)


def test_resize_image_errors_without_size():
    img = np.zeros((4, 4), dtype=np.float32)
    try:
        zoom.resize_image(img)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when no size or rsz is provided.")


def test_resize_image_no_channels_stack():
    img = np.zeros((2, 4, 4), dtype=np.float32)
    out = zoom.resize_image(img, rsz=0.5, no_channels=True)
    assert out.shape == (2, 2, 2)


def test_resize_image_with_channels():
    img = np.zeros((2, 4, 4, 2), dtype=np.float32)
    out = zoom.resize_image(img, rsz=[0.5, 0.5], no_channels=False)
    assert out.shape == (2, 2, 2, 2)


def test_resize_image_2d():
    img = np.zeros((4, 6), dtype=np.float32)
    out = zoom.resize_image(img, rsz=0.5, no_channels=True)
    assert out.shape == (2, 3)
