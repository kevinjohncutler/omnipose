import numpy as np

from ocdkit import array as tnorm
from omnipose import transforms as tbase
from omnipose.transforms import tiles as ttiles


def test_rescale_exclude_dims():
    arr = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)
    out = tnorm.rescale(arr, exclude_dims=0)
    assert out.shape == arr.shape
    assert np.allclose(out[0].min(), 0.0)
    assert np.allclose(out[0].max(), 1.0)
    assert np.allclose(out[1].min(), 0.0)
    assert np.allclose(out[1].max(), 1.0)


def test_safe_divide_handles_zeros():
    num = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    den = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    out = tnorm.safe_divide(num, den)
    assert np.isfinite(out).all()
    assert np.allclose(out, np.array([1.0, 0.0, 1.5], dtype=np.float32))


def test_normalize99_bounds():
    arr = np.linspace(0, 100, 101, dtype=np.float32)
    out = tnorm.normalize99(arr, lower=0, upper=100)
    assert out.min() == 0.0
    assert out.max() == 1.0


def test_quantile_rescale_clamps():
    arr = np.arange(10, dtype=np.float32)
    out = tnorm.quantile_rescale(arr, lower=0.2, upper=0.8)
    assert out.min() == 0.0
    assert out.max() == 1.0
    assert np.all(out[:2] == 0.0)
    assert np.all(out[-2:] == 1.0)


def test_make_tiles_nd_shapes():
    imgi = np.zeros((1, 64, 64), dtype=np.float32)
    img_tiles, subs, shape, inds = ttiles.make_tiles_ND(imgi, bsize=32, augment=False,
                                                        tile_overlap=0.1, normalize=False)
    assert shape == (64, 64)
    assert img_tiles.shape[1:] == (1, 32, 32)
    assert len(subs) == img_tiles.shape[0]
    assert len(inds) == 0
    assert img_tiles.shape[0] == 9  # 3x3 tiling for 64x64 with overlap


def test_average_tiles_nd_constant():
    imgi = np.ones((1, 64, 64), dtype=np.float32)
    img_tiles, subs, shape, _ = ttiles.make_tiles_ND(imgi, bsize=32, augment=False,
                                                     tile_overlap=0.1, normalize=False)
    y = np.ones((img_tiles.shape[0], 3, 32, 32), dtype=np.float32)
    averaged = ttiles.average_tiles_ND(y, subs, shape)
    assert averaged.shape == (3, 64, 64)
    assert np.allclose(averaged, 1.0, atol=1e-6)


def test_make_tiles_nd_reexport():
    imgi = np.zeros((1, 64, 64), dtype=np.float32)
    tiles, subs, shape, inds = tbase.make_tiles_ND(
        imgi, bsize=32, augment=False, tile_overlap=0.1, normalize=False
    )
    assert tiles.shape[1:] == (1, 32, 32)
    assert shape == (64, 64)
    assert len(subs) == tiles.shape[0]
    assert len(inds) == 0


def test_unaugment_tiles_nd_sign_flip():
    imgi = np.zeros((1, 32, 32), dtype=np.float32)
    tiles, subs, shape, inds = ttiles.make_tiles_ND(
        imgi, bsize=16, augment=True, tile_overlap=0.1, normalize=False
    )
    y = np.ones((tiles.shape[0], 3, 16, 16), dtype=np.float32)
    out = ttiles.unaugment_tiles_ND(y.copy(), inds, unet=False)
    idx0 = inds.index((0, 0))
    assert np.all(out[idx0][0] == -1.0)
    assert np.all(out[idx0][1] == -1.0)
