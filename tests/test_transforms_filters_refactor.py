"""Verify omnirefactor.transforms.filters re-exports from ocdkit."""

import numpy as np
import torch

from omnirefactor.transforms import filters as tfilters


def test_moving_average_reexport():
    from ocdkit.array import moving_average
    assert tfilters.moving_average is moving_average


def test_curve_filter_reexport():
    from ocdkit.measure import curve_filter
    assert tfilters.curve_filter is curve_filter


def test_hysteresis_threshold_reexport():
    from ocdkit.morphology import hysteresis_threshold
    assert tfilters.hysteresis_threshold is hysteresis_threshold


def test_add_poisson_noise_reexport():
    from ocdkit.array import add_poisson_noise
    assert tfilters.add_poisson_noise is add_poisson_noise


def test_correct_illumination_reexport():
    from ocdkit.array import correct_illumination
    assert tfilters.correct_illumination is correct_illumination
