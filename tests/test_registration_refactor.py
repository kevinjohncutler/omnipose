import numpy as np
import torch
from scipy.ndimage import shift as im_shift
from scipy.ndimage import center_of_mass

from omnirefactor.utils import registration


def _circle_image(shape, center, radius):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    mask = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2
    img = np.zeros(shape, dtype=np.float32)
    img[mask] = 1.0
    return img


def test_cross_reg_constant_shift():
    shape = (64, 64)
    nframes = 5
    dx = 3
    base = _circle_image(shape, center=(32, 20), radius=6)
    stack = np.stack([im_shift(base, shift=(0, t * dx), order=1, mode="nearest")
                      for t in range(nframes)], axis=0)

    shifts = registration.cross_reg(stack, upsample_factor=1, localnorm=False, max_shift=20)

    expected = np.zeros_like(shifts)
    expected[:, 1] = -dx * np.arange(nframes)
    expected -= expected.mean(axis=0)

    assert np.allclose(shifts[:, 0], 0.0, atol=0.5)
    assert np.allclose(shifts[:, 1], expected[:, 1], atol=0.75)


def test_shift_stack_aligns_frames():
    shape = (64, 64)
    nframes = 5
    dx = 2
    base = _circle_image(shape, center=(32, 22), radius=5)
    stack = np.stack([im_shift(base, shift=(0, t * dx), order=1, mode="nearest")
                      for t in range(nframes)], axis=0)

    shifts = registration.cross_reg(stack, upsample_factor=1, localnorm=False, max_shift=20)
    aligned = registration.shift_stack(stack, shifts, order=1, prefilter=False)

    centers = np.array([center_of_mass(frame) for frame in aligned])
    assert np.allclose(centers[:, 1], centers[0, 1], atol=0.75)


def test_shifts_to_slice_basic():
    shifts = np.array([[2, -1], [-3, 4]], dtype=float)
    shape = (10, 12)
    slices = registration.shifts_to_slice(shifts, shape)
    assert slices == (slice(2, 7), slice(4, 11))


def test_cross_reg_reverse_sign():
    shape = (48, 48)
    nframes = 4
    dx = 2
    base = _circle_image(shape, center=(24, 16), radius=5)
    stack = np.stack([im_shift(base, shift=(0, t * dx), order=1, mode="nearest")
                      for t in range(nframes)], axis=0)

    shifts_fwd = registration.cross_reg(stack, upsample_factor=1, localnorm=False, max_shift=20, reverse=False)
    shifts_rev = registration.cross_reg(stack, upsample_factor=1, localnorm=False, max_shift=20, reverse=True)

    assert shifts_fwd.shape == shifts_rev.shape
    assert np.allclose(shifts_fwd.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(shifts_rev.mean(axis=0), 0.0, atol=1e-6)
    assert not np.allclose(shifts_fwd, shifts_rev)


def test_cross_reg_moving_reference_max_shift_clips():
    shape = (48, 48)
    nframes = 4
    dx = 5
    base = _circle_image(shape, center=(24, 16), radius=5) + 0.1
    stack = np.stack([im_shift(base, shift=(0, t * dx), order=1, mode="nearest")
                      for t in range(nframes)], axis=0)

    shifts = registration.cross_reg(
        stack,
        upsample_factor=1,
        localnorm=True,
        max_shift=1,
        moving_reference=True,
    )
    assert np.all(np.linalg.norm(shifts, axis=1) <= 1.0 + 1e-3)


def test_shift_stack_constant_cval():
    shape = (32, 32)
    base = _circle_image(shape, center=(16, 12), radius=4)
    stack = np.stack([base, im_shift(base, shift=(0, 2), order=1, mode="nearest")], axis=0)
    shifts = np.array([[0.0, 0.0], [0.0, -2.0]], dtype=np.float32)
    aligned = registration.shift_stack(stack, shifts, order=1, cval=0.0, prefilter=False)
    assert aligned.shape == stack.shape
    assert not np.isnan(aligned).any()


def test_gaussian_kernel_and_blur():
    device = torch.device("cpu")
    kernel = registration.gaussian_kernel(5, 1.0, device=device)
    assert kernel.shape == (5, 5)
    assert torch.allclose(kernel.sum(), torch.tensor(1.0), atol=1e-5)

    image = torch.zeros((9, 9), device=device)
    image[4, 4] = 1.0
    blurred = registration.apply_gaussian_blur(image, 5, 1.0, device=device)
    assert blurred.shape == image.shape
    assert blurred.max() < 1.0


def test_phase_cross_correlation_gpu():
    device = torch.device("cpu")
    base = _circle_image((32, 32), center=(16, 12), radius=4)
    stack = torch.from_numpy(np.stack([base, im_shift(base, shift=(0, 3), order=1, mode="nearest"),
                                       im_shift(base, shift=(0, 6), order=1, mode="nearest")], axis=0)).to(device)
    shifts = registration.phase_cross_correlation_GPU(stack, upsample_factor=2, normalization="phase")
    assert shifts.shape == (3, 2)
    assert torch.isfinite(shifts).all()


def test_phase_cross_correlation_gpu_old(monkeypatch):
    device = torch.device("cpu")

    orig_blur = registration.apply_gaussian_blur

    def blur_cpu(image, kernel_size, sigma, device=device):
        return orig_blur(image, kernel_size, sigma, device=device)

    monkeypatch.setattr(registration, "apply_gaussian_blur", blur_cpu)

    base = _circle_image((24, 24), center=(12, 10), radius=3)
    stack = torch.from_numpy(np.stack([base,
                                       im_shift(base, shift=(0, 2), order=1, mode="nearest"),
                                       im_shift(base, shift=(0, 4), order=1, mode="nearest")], axis=0)).to(device)

    shifts = registration.phase_cross_correlation_GPU_old(
        stack,
        upsample_factor=2,
        reverse=True,
        normalize=True,
    )
    assert shifts.shape == (3, 2)
    assert torch.isfinite(shifts).all()


def test_apply_shifts_grouping_and_single():
    device = torch.device("cpu")
    base = torch.zeros((3, 16, 16), device=device)
    base[0, 6:10, 6:10] = 1.0
    base[1, 6:10, 6:10] = 1.0
    base[2, 6:10, 6:10] = 1.0

    shifts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, -1.0]], device=device)
    shifted = registration.apply_shifts(base, shifts)
    assert shifted.shape == base.shape
    assert torch.isfinite(shifted).all()

    single = registration.apply_shifts(base[:1], torch.tensor([1.0, 0.0], device=device))
    assert single.shape == base[:1].shape
