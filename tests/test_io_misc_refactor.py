import io as pyio

import numpy as np
import pytest

from omnirefactor.io import gif as io_gif
from omnirefactor.io import lists as io_lists
from omnirefactor.plot import overlay as io_overlay
from omnirefactor.io import url as io_url


def test_lists_roundtrip(tmp_path):
    nested = [np.arange(5), np.arange(6).reshape(2, 3)]
    path = tmp_path / "nested.npz"
    io_lists.save_nested_list(path, nested)
    loaded = io_lists.load_nested_list(path)
    assert len(loaded) == len(nested)
    assert np.array_equal(loaded[0], nested[0])
    assert np.array_equal(loaded[1], nested[1])


def test_channel_overlay_basic():
    channels = np.stack([np.zeros((4, 4)), np.ones((4, 4))], axis=0)
    rgb = io_overlay.channel_overlay(channels, color_indexes=[1])
    assert rgb.shape == (4, 4, 3)
    assert np.isfinite(rgb).all()


def test_mask_outline_overlay_basic():
    img = np.zeros((4, 4), dtype=np.float32)
    masks = np.zeros((4, 4), dtype=np.int32)
    masks[1:3, 1:3] = 1
    outlines = np.zeros_like(masks, dtype=np.uint8)
    outlines[1, 1:3] = 1
    overlay = io_overlay.mask_outline_overlay(img, masks, outlines)
    assert overlay.shape == (4, 4, 3)
    assert np.isfinite(overlay).all()


def test_download_url_to_file(monkeypatch, tmp_path):
    data = b"hello world"

    class DummyResponse:
        def __init__(self, payload):
            self._buf = pyio.BytesIO(payload)

        def read(self, size=-1):
            return self._buf.read(size)

        def info(self):
            class Info:
                def get_all(self, name):
                    return [str(len(data))]
            return Info()

    monkeypatch.setattr(io_url, "urlopen", lambda _: DummyResponse(data))
    dst = tmp_path / "out.bin"
    io_url.download_url_to_file("http://example.invalid/data", dst, progress=False)
    assert dst.read_bytes() == data


def test_export_gif_mocks_ffmpeg(monkeypatch, tmp_path):
    calls = {}

    class DummyPopen:
        def __init__(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            self.stdin = pyio.BytesIO()

        def wait(self):
            return 0

    monkeypatch.setattr(io_gif.subprocess, "Popen", DummyPopen)
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    io_gif.export_gif(frames, "demo", str(tmp_path), scale=1, fps=5, loop=0, bounce=False)
    assert "args" in calls
    assert calls["args"][0][0] == "ffmpeg"
