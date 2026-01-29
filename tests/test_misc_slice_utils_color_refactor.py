import numpy as np

from omnirefactor.misc import slice as mslice
from omnirefactor.misc import utils as mutils
from omnirefactor.utils import color as ucolor


def test_get_slice_tuple_scalar_axis():
    shape = (5, 6)
    slc = mslice.get_slice_tuple(1, 4, shape)
    assert slc == (slice(1, 4, None), slice(None))

    slc_axis = mslice.get_slice_tuple(0, 2, shape, axis=1)
    assert slc_axis == (slice(None), slice(0, 2, None))


def test_get_slice_tuple_iterable():
    shape = (10, 12, 14)
    slc = mslice.get_slice_tuple([1, 2, 3], [4, 6, 9], shape)
    assert slc == (slice(1, 4, None), slice(2, 6, None), slice(3, 9, None))

    slc_axis = mslice.get_slice_tuple([1, 2], [4, 6], shape, axis=[0, 2])
    assert slc_axis == (slice(1, 4, None), slice(None), slice(2, 6, None))


def test_get_slice_tuple_mismatch_errors():
    shape = (5, 6)
    try:
        mslice.get_slice_tuple([1, 2], [3], shape)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched start/stop lengths")

    try:
        mslice.get_slice_tuple([1, 2], [3, 4], shape, axis=[0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched axis length")


def test_get_size_units():
    arr = np.zeros((10,), dtype=np.uint8)
    assert mutils.get_size(arr, unit="B") == arr.nbytes
    assert np.isclose(mutils.get_size(arr, unit="KB"), arr.nbytes / 1024)
    assert np.isclose(mutils.get_size(arr, unit="MB"), arr.nbytes / (1024**2))


def test_random_int_seeded():
    N = 10
    M = 5
    seed = 123
    rng = np.random.RandomState(seed)
    expected = rng.randint(0, N, M)
    out = mutils.random_int(N, M, seed=seed)
    assert np.array_equal(out, expected)

    rng = np.random.RandomState(seed)
    expected_scalar = rng.randint(0, N)
    out_scalar = mutils.random_int(N, seed=seed)
    assert out_scalar == expected_scalar


def test_rgb_hsv_roundtrip():
    rgb = np.array(
        [
            [[0.1, 0.2, 0.3], [0.8, 0.1, 0.2]],
            [[0.9, 0.9, 0.1], [0.0, 0.4, 0.7]],
        ],
        dtype=np.float32,
    )
    hsv = ucolor.rgb_to_hsv(rgb)
    out = ucolor.hsv_to_rgb(hsv)
    assert np.allclose(out, rgb, atol=1e-6)


def test_sinebow_palette():
    bg = [0.1, 0.2, 0.3, 0.4]
    palette = ucolor.sinebow(3, bg_color=bg, offset=1)
    assert palette[0] == bg
    assert len(palette) == 4
    for k in range(1, 4):
        r, g, b, a = palette[k]
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0
        assert a == 1
