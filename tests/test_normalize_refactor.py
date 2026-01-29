import numpy as np
import pytest
import torch

from omnirefactor.transforms import normalize as tnorm


def test_bin_counts_basic():
    data = np.array([0, 0, 1, 1, 2, 2], dtype=np.uint16)
    counts, starts = tnorm.bin_counts(data, num_bins=3)
    assert counts.sum() <= data.size
    assert len(starts) == len(counts)


def test_compute_density_shape():
    rng = np.random.RandomState(0)
    x = rng.normal(size=50)
    y = rng.normal(scale=2.0, size=50)
    d = tnorm.compute_density(x, y)
    assert d.shape == x.shape
    assert np.isfinite(d).all()


def test_normalize99_hist_bounds():
    arr = np.linspace(0, 100, 101, dtype=np.float32)
    out = tnorm.normalize99_hist(arr, lower=0, upper=100)
    assert out.min() == 0.0
    assert out.max() == 1.0


def test_pnormalize_range():
    arr = np.linspace(0, 5, 6, dtype=np.float32)
    out = tnorm.pnormalize(arr, p_min=-1, p_max=2)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_auto_chunked_quantile():
    x = torch.linspace(0, 1, 100)
    q = tnorm.auto_chunked_quantile(x, torch.tensor([0.25, 0.75]))
    assert torch.allclose(q, torch.tensor([0.25, 0.75]), atol=0.05)


def test_normalize_image_foreground():
    rng = np.random.RandomState(1)
    img = rng.uniform(0.1, 2.0, size=(8, 8)).astype(np.float32)
    mask = np.ones_like(img, dtype=np.uint8)
    out = tnorm.normalize_image(img, mask, target=0.5, foreground=True, iterations=0)
    assert out.shape == img.shape
    assert np.isfinite(out).all()


def test_adjust_contrast_masked():
    img = np.zeros((8, 8), dtype=np.float32)
    img[2:6, 2:6] = 1.0
    masks = np.zeros_like(img, dtype=np.uint8)
    masks[2:6, 2:6] = 1
    out, gamma, limits = tnorm.adjust_contrast_masked(img, masks)
    assert out.shape == img.shape
    assert 0.2 <= gamma <= 5.0
    assert np.isfinite(out).all()
    assert limits[0] <= limits[1]


def test_gamma_normalize_cpu():
    img = np.zeros((1, 8, 8), dtype=np.float32)
    vals = np.linspace(0.2, 1.0, 16, dtype=np.float32).reshape(4, 4)
    img[:, 2:6, 2:6] = vals
    mask = (img > 0).astype(np.uint8)
    out = tnorm.gamma_normalize(
        img,
        mask,
        target=torch.tensor(0.5),
        foreground=True,
        iterations=0,
        channel_axis=0,
    )
    assert out.shape == img.shape[1:]
    assert np.isfinite(out[mask[0] > 0]).all()


def test_localnormalize_cpu():
    img = np.random.RandomState(0).rand(16, 16).astype(np.float32)
    out = tnorm.localnormalize(img, sigma1=1, sigma2=2)
    assert out.shape == img.shape
    assert np.isfinite(out).all()


def test_localnormalize_gpu_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    img = torch.rand(1, 16, 16)
    out = tnorm.localnormalize_GPU(img, sigma1=1, sigma2=2)
    assert out.shape == img.shape
    assert torch.isfinite(out).all()


def test_safe_divide_dask_branch():
    da = pytest.importorskip("dask.array")
    num = da.from_array(np.array([1.0, 2.0]))
    den = da.from_array(np.array([0.0, 2.0]))
    out = tnorm.safe_divide(num, den)
    assert np.allclose(out.compute(), np.array([0.0, 1.0]))


def test_safe_divide_type_error(monkeypatch):
    class DummyModule:
        @staticmethod
        def isfinite(x):
            return np.isfinite(x)

    monkeypatch.setattr(tnorm, "get_module", lambda _x: DummyModule())
    with pytest.raises(TypeError):
        tnorm.safe_divide(np.array([1.0]), np.array([1.0]))


def test_qnorm_branches(monkeypatch):
    rng = np.random.RandomState(0)
    Y = rng.rand(4, 8, 8).astype(np.float32)
    monkeypatch.setattr(tnorm, "ne", None)
    r, x, y, d, imin, imax, vmin, vmax = tnorm.qnorm(
        Y,
        nbins=16,
        dx=2,
        log=True,
        debug=True,
        density_quantile=0.5,
        density_cutoff=0.5,
    )
    assert r.shape == (4, 4, 4)
    assert imin <= imax


def test_qnorm_constant_vmax_branch(monkeypatch):
    monkeypatch.setattr(tnorm, "ne", None)

    def fake_density(x, y, bw_method=None):
        d = np.zeros_like(x, dtype=np.float64)
        if d.size:
            d[0] = 1.0
        return d

    monkeypatch.setattr(tnorm, "compute_density", fake_density)
    Y = np.linspace(0, 1, 64, dtype=np.float32).reshape(2, 8, 4)
    r = tnorm.qnorm(Y, nbins=8, debug=False, density_cutoff=0.5)
    assert r.shape == Y.shape


def test_normalize99_dim_and_contrast_limits(monkeypatch):
    Y = np.arange(9, dtype=np.float32).reshape(3, 3)
    out = tnorm.normalize99(Y, dim=1)
    assert out.shape == Y.shape
    out2 = tnorm.normalize99(Y, contrast_limits=(0.0, 10.0))
    assert out2.min() >= 0.0

    t = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(torch, "quantile", _raise)
    monkeypatch.setattr(tnorm, "auto_chunked_quantile", lambda _t, _q: torch.tensor([0.0, 1.0]))
    out3 = tnorm.normalize99(t, dim=None)
    assert out3.shape == t.shape


def test_normalize_field_torch():
    mu = torch.randn(2, 4, 4)
    out = tnorm.normalize_field(mu, use_torch=True, cutoff=0.0)
    assert out.shape == mu.shape


def test_normalize99_hist_contrast_limits_torch():
    t = torch.linspace(0, 1, 10)
    out = tnorm.normalize99_hist(t, contrast_limits=(0.0, 1.0))
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_normalize_image_branches(monkeypatch):
    monkeypatch.setattr(tnorm, "ne", None)
    img = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    mask = np.ones((3, 3), dtype=np.uint8)
    out = tnorm.normalize_image(img, mask, iterations=1, channel_axis=0)
    assert out.shape == img.shape


def test_adjust_contrast_masked_edge_cases():
    img = np.zeros((4, 4), dtype=np.float32)
    masks = np.zeros_like(img, dtype=np.uint8)
    out, gamma, limits = tnorm.adjust_contrast_masked(img, masks)
    assert gamma == 1.0

    img[:] = 1.0
    masks[0:2, 0:2] = 1
    out2, gamma2, _ = tnorm.adjust_contrast_masked(img, masks, r_target=0.5)
    assert 0.2 <= gamma2 <= 5.0
    assert np.isfinite(out2).all()


def test_gamma_normalize_branches():
    img = np.linspace(0.1, 1.0, 9, dtype=np.float32).reshape(3, 3)
    mask = (img > 0.5).astype(np.uint8)
    out = tnorm.gamma_normalize(img, mask, iterations=1, channel_axis=0, target=torch.tensor(1.0))
    assert out.shape == img.shape
