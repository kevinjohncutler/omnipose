"""Tests for augment.py coverage — CPU fallback path and uncovered branches."""
import numpy as np
import pytest
import torch

from omnirefactor.transforms.augment import (
    random_rotate_and_resize,
    random_crop_warp,
    _supports_grid3d,
    _mode_filter_gpu,
    mode_filter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(nchan=1, size=64, dim=2, with_mask=True):
    """Create a simple test image with a centered object."""
    shape = (size,) * dim
    img = np.random.rand(nchan, *shape).astype(np.float32)
    if with_mask:
        lbl = np.zeros(shape, dtype=np.int32)
        slc = tuple(slice(size // 4, 3 * size // 4) for _ in range(dim))
        lbl[slc] = 1
    else:
        lbl = np.zeros(shape, dtype=np.int32)
    return img, lbl


# ---------------------------------------------------------------------------
# CPU fallback path (device=None) — lines 690-772
# ---------------------------------------------------------------------------

class TestCPUAugmentationPath:
    """The CPU numpy path triggers when device=None (DataLoader workers)."""

    def test_cpu_path_returns_numpy(self):
        np.random.seed(0)
        img, lbl = _make_image()
        r = random_rotate_and_resize([img], Y=[lbl], tyx=(32, 32),
                                     device=None, nchan=1)
        assert isinstance(r.images, np.ndarray)
        assert isinstance(r.labels, np.ndarray)

    def test_cpu_path_basic_shape(self):
        np.random.seed(1)
        img, lbl = _make_image(nchan=2)
        r = random_rotate_and_resize([img], Y=[lbl], tyx=(32, 32),
                                     device=None, nchan=2)
        assert r.images.shape == (1, 2, 32, 32)
        assert r.labels.shape == (1, 32, 32)

    def test_cpu_path_with_augmentation(self):
        """Run enough seeds to hit most augmentation branches."""
        img, lbl = _make_image()
        for seed in range(20):
            np.random.seed(seed)
            r = random_rotate_and_resize([img], Y=[lbl], tyx=(32, 32),
                                         device=None, nchan=1)
            assert r.images.shape == (1, 1, 32, 32)
            assert np.isfinite(r.images).all()

    def test_cpu_path_no_foreground(self):
        """CPU path with blank mask — special rescaling branch."""
        np.random.seed(42)
        img, _ = _make_image(with_mask=False)
        lbl = np.zeros((64, 64), dtype=np.int32)
        r = random_rotate_and_resize([img], Y=[lbl], tyx=(32, 32),
                                     device=None, nchan=1,
                                     allow_blank_masks=True)
        assert r.images.shape == (1, 1, 32, 32)

    def test_cpu_path_integer_image(self):
        """CPU path with integer input — triggers np.iinfo branch."""
        np.random.seed(0)
        img = (np.random.rand(1, 64, 64) * 255).astype(np.uint8)
        lbl = np.zeros((64, 64), dtype=np.int32)
        lbl[20:40, 20:40] = 1
        r = random_rotate_and_resize([img], Y=[lbl], tyx=(32, 32),
                                     device=None, nchan=1)
        assert r.images.shape == (1, 1, 32, 32)

    def test_cpu_path_flips(self):
        """CPU flip path (lines 784-786)."""
        np.random.seed(7)
        img, lbl = _make_image()
        r = random_rotate_and_resize([img], Y=[lbl], tyx=(32, 32),
                                     device=None, nchan=1, do_flip=True)
        assert r.images.shape == (1, 1, 32, 32)

    def test_cpu_path_3d(self):
        """3D CPU path — device=None with 3D data."""
        np.random.seed(0)
        img, lbl = _make_image(dim=3, size=16)
        r = random_rotate_and_resize([img], Y=[lbl], tyx=(12, 12, 12),
                                     device=None, nchan=1)
        assert r.images.shape == (1, 1, 12, 12, 12)

    def test_cpu_path_return_meta(self):
        """CPU return_meta path (lines 789-802)."""
        np.random.seed(0)
        img, lbl = _make_image()
        r = random_rotate_and_resize([img], Y=[lbl], tyx=(32, 32),
                                     device=None, nchan=1, return_meta=True)
        assert r.meta is not None
        assert len(r.meta) == 1
        assert 'M_inv' in r.meta[0]
        assert 'bbox_in' in r.meta[0]


# ---------------------------------------------------------------------------
# Torch path coverage gaps
# ---------------------------------------------------------------------------

class TestTorchAugmentationBranches:
    """Cover specific torch-path branches that are currently missed."""

    def test_torch_path_no_foreground(self):
        """Torch path with blank mask — rare rescaling branch."""
        np.random.seed(42)
        img = np.random.rand(1, 64, 64).astype(np.float32)
        lbl = np.zeros((64, 64), dtype=np.int32)
        r = random_rotate_and_resize([img], Y=[lbl], tyx=(32, 32),
                                     device=torch.device('cpu'), nchan=1,
                                     allow_blank_masks=True)
        assert isinstance(r.images, torch.Tensor)

    def test_torch_path_return_meta(self):
        """Torch return_meta path (line 801)."""
        np.random.seed(0)
        img, lbl = _make_image()
        r = random_rotate_and_resize([img], Y=[lbl], tyx=(32, 32),
                                     device=torch.device('cpu'), nchan=1,
                                     return_meta=True)
        assert r.meta is not None
        assert 'bbox_in' in r.meta[0]

    def test_torch_path_multichannel(self):
        """Multichannel torch path."""
        np.random.seed(0)
        img, lbl = _make_image(nchan=3)
        r = random_rotate_and_resize([img], Y=[lbl], tyx=(32, 32),
                                     device=torch.device('cpu'), nchan=3)
        assert r.images.shape == (1, 3, 32, 32)

    def test_torch_path_3d(self):
        """3D torch path — CPU supports 3D grid_sample."""
        np.random.seed(0)
        img, lbl = _make_image(dim=3, size=16)
        r = random_rotate_and_resize([img], Y=[lbl], tyx=(12, 12, 12),
                                     device=torch.device('cpu'), nchan=1)
        assert isinstance(r.images, torch.Tensor)
        assert r.images.shape == (1, 1, 12, 12, 12)

    def test_torch_deferred_batch_aug(self):
        """When _defer_batch_aug=True, gamma/noise/S&P are batch-level."""
        np.random.seed(0)
        img, lbl = _make_image()
        # _defer_batch_aug is internal; triggered via random_rotate_and_resize
        # when use_torch=True and return_meta=False (the default training path)
        r = random_rotate_and_resize([img, img], Y=[lbl, lbl], tyx=(32, 32),
                                     device=torch.device('cpu'), nchan=1)
        assert r.images.shape[0] == 2


# ---------------------------------------------------------------------------
# _supports_grid3d
# ---------------------------------------------------------------------------

class TestSupportsGrid3d:

    def test_cpu_supports_3d(self):
        assert _supports_grid3d(torch.device('cpu')) is True

    def test_caching(self):
        """Second call should hit cache."""
        from omnirefactor.transforms.augment import _grid3d_cap
        _grid3d_cap.clear()
        _supports_grid3d(torch.device('cpu'))
        assert ('cpu', 'bilinear') in _grid3d_cap
        # Second call
        assert _supports_grid3d(torch.device('cpu')) is True

    def test_unsupported_device_returns_false(self, monkeypatch):
        """Simulate a device that doesn't support 3D grid_sample."""
        from omnirefactor.transforms import augment as aug_mod
        orig = aug_mod._grid3d_cap
        aug_mod._grid3d_cap = {}
        monkeypatch.setattr(aug_mod._F, 'grid_sample',
                            lambda *a, **kw: (_ for _ in ()).throw(NotImplementedError))
        result = _supports_grid3d(torch.device('cpu'))
        aug_mod._grid3d_cap = orig
        assert result is False


# ---------------------------------------------------------------------------
# mode_filter CPU vs GPU consistency
# ---------------------------------------------------------------------------

class TestModeFilter:

    def test_mode_filter_cpu_preserves_shape(self):
        lbl = np.zeros((16, 16), dtype=np.float32)
        lbl[4:12, 4:12] = 1
        filtered = mode_filter(lbl)
        assert filtered.shape == lbl.shape

    def test_mode_filter_gpu_preserves_shape(self):
        lbl = torch.zeros((16, 16), dtype=torch.float32)
        lbl[4:12, 4:12] = 1
        filtered = _mode_filter_gpu(lbl)
        assert filtered.shape == lbl.shape
