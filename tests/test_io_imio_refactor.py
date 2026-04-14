import numpy as np

from omnirefactor import io


def test_imio_read_write_npy(tmp_path):
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    path = tmp_path / "arr.npy"
    io.imwrite(str(path), arr)
    out = io.imread(str(path))
    assert np.array_equal(out, arr)


def test_get_image_files_filters_masks(tmp_path):
    img = np.zeros((8, 8), dtype=np.uint8)
    io.imwrite(str(tmp_path / "sample.tif"), img)
    io.imwrite(str(tmp_path / "sample_masks.tif"), img)
    io.imwrite(str(tmp_path / "other_img.tif"), img)

    files = io.get_image_files(str(tmp_path), mask_filter="_masks", img_filter="")
    names = {p.split("/")[-1] for p in files}
    assert "sample.tif" in names
    assert "sample_masks.tif" not in names
    assert "other_img.tif" in names


def test_get_image_files_pattern(tmp_path):
    img = np.zeros((8, 8), dtype=np.uint8)
    io.imwrite(str(tmp_path / "alpha_img.tif"), img)
    io.imwrite(str(tmp_path / "beta_img.tif"), img)

    files = io.get_image_files(str(tmp_path), img_filter="_img", pattern="alpha_img")
    assert len(files) == 1
    assert files[0].endswith("alpha_img.tif")
