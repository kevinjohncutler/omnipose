import numpy as np
import tifffile

from omnipose import io


def test_imread_npy_npz_tif(tmp_path):
    arr = np.arange(16, dtype=np.uint16).reshape(4, 4)

    npy_path = tmp_path / "a.npy"
    np.save(npy_path, arr)
    assert np.array_equal(io.imread(str(npy_path)), arr)

    npz_path = tmp_path / "a.npz"
    np.savez(npz_path, arr)
    assert np.array_equal(io.imread(str(npz_path)), arr)

    tif_path = tmp_path / "a.tif"
    tifffile.imwrite(tif_path, arr)
    assert np.array_equal(io.imread(str(tif_path)), arr)


def test_imwrite_then_imread_roundtrip(tmp_path):
    arr = (np.random.rand(4, 4, 3) * 255).astype(np.uint8)
    png_path = tmp_path / "a.png"
    io.imwrite(str(png_path), arr)
    out = io.imread(str(png_path))
    assert out is not None
    assert out.shape[:2] == arr.shape[:2]
