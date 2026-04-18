"""Tests for omnirefactor.data.norm — per-image per-channel normalization."""

import numpy as np
import tifffile

from omnirefactor.data.norm import compute_norm_params, apply_norm_params


def _write_tiff(path, arr):
    tifffile.imwrite(str(path), arr)


# ---------------------------------------------------------------------------
# compute_norm_params
# ---------------------------------------------------------------------------

class TestComputeNormParams:
    def test_2d_single_channel(self, tmp_path):
        p = tmp_path / "a.tif"
        _write_tiff(p, (np.random.rand(32, 32) * 100).astype(np.float32))
        params = compute_norm_params([str(p)], dim=2, channel_axis=None)
        assert len(params) == 1
        assert len(params[0]) == 1  # one channel added
        lo, hi = params[0][0]
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo < hi

    def test_multichannel_first_axis(self, tmp_path):
        p = tmp_path / "b.tif"
        _write_tiff(p, (np.random.rand(3, 32, 32) * 100).astype(np.float32))
        params = compute_norm_params([str(p)], dim=2, channel_axis=0)
        assert len(params[0]) == 3

    def test_multichannel_auto_detect(self, tmp_path):
        """channel_axis=None uses move_min_dim heuristic."""
        p = tmp_path / "c.tif"
        # (H, W, C) with C=3 (smallest dim)
        _write_tiff(p, (np.random.rand(32, 32, 3) * 100).astype(np.float32))
        params = compute_norm_params([str(p)], dim=2, channel_axis=None)
        assert len(params[0]) == 3

    def test_multiple_images(self, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"im_{i}.tif"
            _write_tiff(p, (np.random.rand(16, 16) * 100).astype(np.float32))
            paths.append(str(p))
        params = compute_norm_params(paths, dim=2, channel_axis=None)
        assert len(params) == 3

    def test_uses_percentiles(self, tmp_path):
        """Percentile bounds should be well inside min/max for noisy input."""
        p = tmp_path / "d.tif"
        arr = np.random.rand(64, 64).astype(np.float32) * 100
        _write_tiff(p, arr)
        params = compute_norm_params([str(p)], dim=2, channel_axis=None)
        lo, hi = params[0][0]
        # 0.01 and 99.99 percentiles should be strictly inside min/max
        assert lo >= arr.min()
        assert hi <= arr.max()
        assert lo > arr.min() or hi < arr.max()  # at least one is clipped


# ---------------------------------------------------------------------------
# apply_norm_params
# ---------------------------------------------------------------------------

class TestApplyNormParams:
    def test_basic(self):
        img = np.array([[[0.0, 5.0, 10.0]]], dtype=np.float32)  # (C=1, H=1, W=3)
        params = [(0.0, 10.0)]
        out = apply_norm_params(img.copy(), params)
        np.testing.assert_allclose(out, [[[0.0, 0.5, 1.0]]])

    def test_clips_out_of_range(self):
        img = np.array([[[-5.0, 0.0, 15.0]]], dtype=np.float32)
        params = [(0.0, 10.0)]
        out = apply_norm_params(img.copy(), params)
        assert out.min() == 0.0
        assert out.max() == 1.0

    def test_skips_degenerate_range(self):
        """If hi - lo < 1e-3, channel is left as-is (avoids div-by-zero)."""
        img = np.array([[[5.0, 5.0, 5.0]]], dtype=np.float32)
        params = [(5.0, 5.0)]  # zero range
        out = apply_norm_params(img.copy(), params)
        # Should not blow up; output should be the input (float32)
        np.testing.assert_allclose(out, img.astype(np.float32))

    def test_multichannel(self):
        img = np.stack([
            np.full((2, 2), 5.0),
            np.full((2, 2), 20.0),
        ]).astype(np.float32)
        params = [(0.0, 10.0), (0.0, 40.0)]
        out = apply_norm_params(img.copy(), params)
        np.testing.assert_allclose(out[0], 0.5)
        np.testing.assert_allclose(out[1], 0.5)

    def test_casts_to_float32(self):
        img = np.zeros((1, 2, 2), dtype=np.uint8)
        params = [(0.0, 1.0)]
        out = apply_norm_params(img, params)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# roundtrip: compute then apply
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_roundtrip_2d(self, tmp_path):
        p = tmp_path / "rt.tif"
        arr = (np.random.rand(16, 16) * 255).astype(np.float32)
        _write_tiff(p, arr)
        params = compute_norm_params([str(p)], dim=2, channel_axis=None)

        loaded = arr[np.newaxis].astype(np.float32)  # (1, H, W)
        normalized = apply_norm_params(loaded, params[0])
        assert normalized.shape == loaded.shape
        assert 0.0 <= normalized.min()
        assert normalized.max() <= 1.0
