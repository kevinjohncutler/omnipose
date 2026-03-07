import os
from pathlib import Path

import numpy as np
import tifffile

from omnirefactor.io import gif as gif_mod
from omnirefactor.io import imio


def test_imread_npy_npz_tif(tmp_path):
    arr = (np.arange(16, dtype=np.uint16).reshape(4, 4))

    npy_path = tmp_path / "a.npy"
    np.save(npy_path, arr)
    assert np.array_equal(imio.imread(str(npy_path)), arr)

    npz_path = tmp_path / "a.npz"
    np.savez(npz_path, arr)
    assert np.array_equal(imio.imread(str(npz_path)), arr)

    tif_path = tmp_path / "a.tif"
    tifffile.imwrite(tif_path, arr)
    assert np.array_equal(imio.imread(str(tif_path)), arr)


def test_imread_czi_monkeypatch(tmp_path, monkeypatch):
    class DummyAICS:
        def __init__(self, _path):
            self.data = np.ones((2, 2), dtype=np.uint8)

    monkeypatch.setattr(imio, "AICSImage", DummyAICS)
    czi_path = tmp_path / "a.czi"
    czi_path.write_bytes(b"dummy")
    assert np.array_equal(imio.imread(str(czi_path)), np.ones((2, 2), dtype=np.uint8))


def test_imread_cv2_failure_returns_none(tmp_path, monkeypatch):
    png_path = tmp_path / "bad.png"
    png_path.write_bytes(b"not an image")
    monkeypatch.setattr(imio.cv2, "imread", lambda *_args, **_kwargs: None)
    assert imio.imread(str(png_path)) is None


def test_imwrite_branches_with_monkeypatched_encoders(tmp_path, monkeypatch):
    if not hasattr(imio.imagecodecs, "bmp_encode"):
        import pytest
        pytest.skip("imagecodecs.bmp_encode not available")
    arr = (np.random.rand(4, 4, 3) * 255).astype(np.uint8)

    def _fake_encode(_arr, **_kwargs):
        return b"fake"

    monkeypatch.setattr(imio.imagecodecs, "png_encode", _fake_encode)
    monkeypatch.setattr(imio.imagecodecs, "jpeg_encode", _fake_encode)
    monkeypatch.setattr(imio.imagecodecs, "webp_encode", _fake_encode)
    monkeypatch.setattr(imio.imagecodecs, "jpegxl_encode", _fake_encode)
    monkeypatch.setattr(imio.imagecodecs, "bmp_encode", _fake_encode)

    paths = [
        tmp_path / "a.png",
        tmp_path / "a.jpg",
        tmp_path / "a.webp",
        tmp_path / "a.jxl",
        tmp_path / "a.bmp",
        tmp_path / "a.unknown",
    ]
    for path in paths:
        imio.imwrite(str(path), arr, quality=90)
        assert path.exists()
        assert path.stat().st_size > 0


def test_imwrite_then_imread_cv2_roundtrip(tmp_path):
    arr = (np.random.rand(4, 4, 3) * 255).astype(np.uint8)
    png_path = tmp_path / "a.png"
    imio.imwrite(str(png_path), arr)
    out = imio.imread(str(png_path))
    assert out is not None
    assert out.shape[:2] == arr.shape[:2]


def test_export_gif_and_movie_smoke(tmp_path, monkeypatch):
    calls = []

    class DummyStdin:
        def __init__(self):
            self.buf = bytearray()

        def write(self, data):
            self.buf.extend(data)

        def close(self):
            return None

    class DummyProc:
        def __init__(self, args, **_kwargs):
            calls.append(args)
            self.stdin = DummyStdin()

        def wait(self):
            return 0

    monkeypatch.setattr(gif_mod.subprocess, "Popen", DummyProc)

    frames_rgb = (np.random.rand(2, 4, 4, 3) * 255).astype(np.uint8)
    frames_gray = (np.random.rand(2, 4, 4) * 255).astype(np.uint8)

    gif_mod.export_gif(frames_rgb, "rgb", str(tmp_path), fps=5, scale=1, bounce=True)
    gif_mod.export_gif(frames_gray, "gray", str(tmp_path), fps=5, scale=1, bounce=False)
    gif_mod.export_movie(frames_rgb, "movie", str(tmp_path), fps=5)

    assert len(calls) >= 3
