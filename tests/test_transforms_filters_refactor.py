import numpy as np
import torch

from omnirefactor.transforms import filters as tfilters


def test_moving_average_simple():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = tfilters.moving_average(x, 3)
    assert out.shape == x.shape
    assert np.isfinite(out).all()


def test_curve_filter_shapes():
    img = np.zeros((32, 32), dtype=np.float32)
    img[16, 16] = 1.0
    outputs = tfilters.curve_filter(img, filterWidth=1.2)
    for out in outputs:
        assert out.shape == img.shape
        assert np.isfinite(out).all()
    # nonnegative variants should be >= 0
    for out in outputs[:4]:
        assert (out >= 0).all()


def test_hysteresis_threshold_2d():
    img = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    img[0, 0, 2, 2] = 1.0
    img[0, 0, 2, 3] = 0.6
    img[0, 0, 2, 4] = 0.2
    mask = tfilters.hysteresis_threshold(img, low=0.5, high=0.9)
    assert mask.shape == img.shape
    assert mask[0, 0, 2, 2]
    assert mask[0, 0, 2, 3]
    assert not mask[0, 0, 2, 4]


def test_add_poisson_noise_bounds():
    img = np.ones((8, 8), dtype=np.float32) * 0.5
    noisy = tfilters.add_poisson_noise(img)
    assert noisy.shape == img.shape
    assert noisy.min() >= 0.0
    assert noisy.max() <= 1.0


def test_correct_illumination_finite():
    img = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    out = tfilters.correct_illumination(img, sigma=1)
    assert out.shape == img.shape
    assert np.isfinite(out).all()
