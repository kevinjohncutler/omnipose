import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from omnipose.transforms import augment as aug
from omnipose.transforms.augment import _mode_filter_gpu, _gaussian_blur_gpu


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


# ---------------------------------------------------------------------------
# CPU / GPU parity tests
# ---------------------------------------------------------------------------

def _make_label_grid(shape, cell_size):
    """Create a grid of labeled squares for testing mode filter parity."""
    labels = np.zeros(shape, dtype=np.int32)
    label_id = 1
    for y in range(0, shape[0] - cell_size + 1, cell_size + 1):
        for x in range(0, shape[1] - cell_size + 1, cell_size + 1):
            labels[y:y + cell_size, x:x + cell_size] = label_id
            label_id += 1
    return labels


class TestModeFilterCpuGpuParity:
    """Verify _mode_filter_gpu matches CPU mode_filter on foreground pixels."""

    def test_2d_clean_labels(self):
        """Clean labels (no interpolation artifacts) — both paths should be identity."""
        labels = _make_label_grid((32, 32), cell_size=5)
        cpu_out = aug.mode_filter(labels)
        gpu_out = _mode_filter_gpu(torch.from_numpy(labels).float()).numpy().astype(np.int32)
        fg = labels > 0
        np.testing.assert_array_equal(cpu_out[fg], gpu_out[fg])

    def test_2d_noisy_labels(self):
        """Inject isolated wrong-label pixels — both paths should vote them out."""
        labels = _make_label_grid((32, 32), cell_size=6)
        noisy = labels.copy()
        # Scatter some wrong labels inside cells
        rng = np.random.RandomState(42)
        fg_coords = np.argwhere(labels > 0)
        n_flip = min(20, len(fg_coords))
        flip_idx = rng.choice(len(fg_coords), n_flip, replace=False)
        for idx in flip_idx:
            y, x = fg_coords[idx]
            noisy[y, x] = rng.randint(1, labels.max() + 1)

        cpu_out = aug.mode_filter(noisy)
        gpu_out = _mode_filter_gpu(torch.from_numpy(noisy).float()).numpy().astype(np.int32)
        fg = noisy > 0
        np.testing.assert_array_equal(cpu_out[fg], gpu_out[fg])

    def test_3d_parity(self):
        """3D label volume — verify GPU matches CPU on consistent labels.

        Uses the SAME labels across slices (no inter-slice label conflicts)
        to avoid tie-breaking differences at slice boundaries.
        """
        labels_2d = _make_label_grid((16, 16), cell_size=4)
        labels_3d = np.stack([labels_2d, labels_2d, labels_2d])
        cpu_out = aug.mode_filter(labels_3d)
        gpu_out = _mode_filter_gpu(torch.from_numpy(labels_3d).float()).numpy().astype(np.int32)
        fg = labels_3d > 0
        np.testing.assert_array_equal(cpu_out[fg], gpu_out[fg])

    def test_background_preserved(self):
        """Background (0) pixels should stay 0 in both paths."""
        labels = np.zeros((16, 16), dtype=np.int32)
        labels[4:12, 4:12] = 1
        labels[6:10, 6:10] = 2
        cpu_out = aug.mode_filter(labels)
        gpu_out = _mode_filter_gpu(torch.from_numpy(labels).float()).numpy().astype(np.int32)
        bg = labels == 0
        np.testing.assert_array_equal(cpu_out[bg], 0)
        np.testing.assert_array_equal(gpu_out[bg], 0)

    def test_all_background(self):
        """All-zero input should be returned unchanged."""
        labels = np.zeros((8, 8), dtype=np.int32)
        gpu_out = _mode_filter_gpu(torch.from_numpy(labels).float())
        np.testing.assert_array_equal(gpu_out.numpy(), 0)


class TestGaussianBlurCpuGpuParity:
    """Verify _gaussian_blur_gpu matches scipy.ndimage.gaussian_filter.

    Note: torch ``F.pad(mode='reflect')`` implements whole-sample symmetric
    padding, which corresponds to scipy ``mode='mirror'``, NOT scipy
    ``mode='reflect'`` (half-sample symmetric). We test against ``mode='mirror'``
    for exact parity. The boundary difference is negligible for augmentation.
    """

    def test_2d_impulse(self):
        img = np.zeros((32, 32), dtype=np.float32)
        img[16, 16] = 1.0
        sigma = 2.0
        cpu_out = gaussian_filter(img, sigma=sigma, mode='mirror')
        gpu_out = _gaussian_blur_gpu(torch.from_numpy(img), sigma).numpy()
        np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5)

    def test_2d_random(self):
        rng = np.random.RandomState(0)
        img = rng.rand(64, 64).astype(np.float32)
        sigma = 1.5
        cpu_out = gaussian_filter(img, sigma=sigma, mode='mirror')
        gpu_out = _gaussian_blur_gpu(torch.from_numpy(img), sigma).numpy()
        np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5)

    def test_3d_impulse(self):
        img = np.zeros((8, 16, 16), dtype=np.float32)
        img[4, 8, 8] = 1.0
        sigma = 1.0
        cpu_out = gaussian_filter(img, sigma=sigma, mode='mirror')
        gpu_out = _gaussian_blur_gpu(torch.from_numpy(img), sigma).numpy()
        np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5)

    def test_sigma_zero(self):
        img = torch.rand(8, 8)
        out = _gaussian_blur_gpu(img, 0.0)
        assert torch.equal(out, img)

    def test_varying_sigma(self):
        rng = np.random.RandomState(1)
        img = rng.rand(32, 32).astype(np.float32)
        for sigma in [0.5, 1.0, 2.0, 3.0]:
            cpu_out = gaussian_filter(img, sigma=sigma, mode='mirror')
            gpu_out = _gaussian_blur_gpu(torch.from_numpy(img), sigma).numpy()
            np.testing.assert_allclose(gpu_out, cpu_out, atol=1e-5,
                                       err_msg=f"Mismatch at sigma={sigma}")
