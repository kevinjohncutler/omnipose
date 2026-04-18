"""Verify omnirefactor.transforms.zoom re-exports from ocdkit."""

from omnirefactor.transforms import zoom


def test_torch_zoom_reexport():
    from ocdkit.gpu import torch_zoom
    assert zoom.torch_zoom is torch_zoom


def test_resize_image_reexport():
    from ocdkit.array import resize_image
    assert zoom.resize_image is resize_image
