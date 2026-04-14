import os
import shutil
import tempfile

import ocdkit.io as ocdkit_io


def test_check_dir_creates():
    root = tempfile.mkdtemp()
    try:
        target = os.path.join(root, "a", "b")
        ocdkit_io.check_dir(target)
        assert os.path.isdir(target)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_findbetween_and_getname():
    assert ocdkit_io.findbetween("a[bc]d") == "bc"
    assert ocdkit_io.getname("/tmp/prefix_file_suffix.tif", prefix="prefix_", suffix="_suffix") == "file"
    assert ocdkit_io.getname("/tmp/file.tif", padding=4) == "file".zfill(4)


def test_adjust_file_path_platforms(monkeypatch):
    monkeypatch.setattr(ocdkit_io.platform, "system", lambda: "Darwin")
    assert ocdkit_io.adjust_file_path("/home/user/data") == "/Volumes/data"

    monkeypatch.setattr(ocdkit_io.platform, "system", lambda: "Linux")
    home = os.path.expanduser("~")
    assert ocdkit_io.adjust_file_path("/Volumes/data").startswith(home)

    monkeypatch.setattr(ocdkit_io.platform, "system", lambda: "Windows")
    out = ocdkit_io.adjust_file_path("/home/user/data")
    assert os.path.normpath(out) == out

    monkeypatch.setattr(ocdkit_io.platform, "system", lambda: "UnknownOS")
    assert ocdkit_io.adjust_file_path("/custom/path") == "/custom/path"


def test_find_files_exclude_suffixes():
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "sub"), exist_ok=True)
        open(os.path.join(root, "img_mask.tif"), "w").close()
        open(os.path.join(root, "img_mask_backup.tif"), "w").close()
        open(os.path.join(root, "sub", "img_mask.tif"), "w").close()
        matches = ocdkit_io.find_files(root, "_mask", exclude_suffixes=["_mask_backup"])
        assert len(matches) == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)
