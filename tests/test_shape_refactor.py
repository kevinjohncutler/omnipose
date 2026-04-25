import numpy as np
import pytest

from omnipose.transforms import shape


def test_convert_image_squeeze_and_channel_axis():
    x = np.zeros((1, 4, 5, 1), dtype=np.float32)
    out = shape.convert_image(x, channels=None, channel_axis=1, z_axis=None, nchan=2, dim=2)
    assert out.ndim == 3


def test_convert_image_with_z_axis_adds_channel():
    x = np.zeros((2, 3, 4), dtype=np.float32)
    out = shape.convert_image(x, channels=None, channel_axis=None, z_axis=0, dim=3)
    assert out.ndim == 4


def test_convert_image_do3d_adds_channel():
    x = np.zeros((2, 3, 4), dtype=np.float32)
    out = shape.convert_image(x, channels=None, do_3D=True, dim=3)
    assert out.ndim == 4


def test_convert_image_do3d_error():
    x = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        shape.convert_image(x, channels=None, do_3D=True, dim=3)


def test_convert_image_channels_error():
    x = np.zeros((4, 4, 2), dtype=np.float32)
    with pytest.raises(TypeError):
        shape.convert_image(x, channels=[0], channel_axis=-1)


def test_convert_image_channels_too_short():
    x = np.zeros((4, 4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        shape.convert_image(x, channels=[[1]], channel_axis=-1)


def test_convert_image_channel_trim_and_pad():
    x = np.zeros((4, 4, 4), dtype=np.float32)
    out = shape.convert_image(x, channels=None, channel_axis=-1, nchan=2)
    assert out.shape[-1] == 2

    x = np.zeros((4, 4, 1), dtype=np.float32)
    out = shape.convert_image(x, channels=None, channel_axis=-1, nchan=2)
    assert out.shape[-1] == 2


def test_convert_image_4d_dim2_error():
    x = np.zeros((2, 3, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        shape.convert_image(x, channels=None, channel_axis=-1, dim=2, do_3D=False)


def test_reshape_channel_modes():
    x = np.zeros((5, 5), dtype=np.float32)
    out = shape.reshape(x, channels=(0, 0))
    assert out.shape[-1] == 2

    x = np.zeros((3, 5, 5), dtype=np.float32)
    out = shape.reshape(x, channels=(1, 0), channel_axis=0, chan_first=True)
    assert out.ndim == 3

    x = np.zeros((5, 5, 2), dtype=np.float32)
    out = shape.reshape(x, channels=(0, 0), channel_axis=-1)
    assert out.shape[-1] == 2

    x = np.zeros((2, 3, 4, 2), dtype=np.float32)
    out = shape.reshape(x, channels=(1, 0), channel_axis=-1, chan_first=True)
    assert out.ndim == 4


def test_reshape_channel_warnings():
    x = np.zeros((5, 5, 2), dtype=np.float32)
    with pytest.warns(UserWarning):
        shape.reshape(x, channels=(1, 2), channel_axis=-1)


def test_normalize_img_error_and_invert():
    x = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        shape.normalize_img(x)

    x = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4, 1)
    out = shape.normalize_img(x, invert=True)
    assert out.shape == x.shape


def test_reshape_train_test_errors():
    x = [np.zeros((4, 4), dtype=np.float32)]
    y = [np.zeros((4, 4), dtype=np.float32), np.zeros((4, 4), dtype=np.float32)]
    with pytest.raises(ValueError):
        shape.reshape_train_test(x, y, None, None, channels=(0, 0))

    x = [np.zeros((1,), dtype=np.float32)]
    y = [np.zeros((1,), dtype=np.float32)]
    with pytest.raises(ValueError):
        shape.reshape_train_test(x, y, None, None, channels=(0, 0))

    x = [np.zeros((2, 2, 2, 2), dtype=np.float32)]
    y = [np.zeros((2, 2), dtype=np.float32)]
    with pytest.raises(ValueError):
        shape.reshape_train_test(x, y, None, None, channels=(0, 0))


def test_reshape_train_test_train_data_none(monkeypatch):
    x = [np.zeros((4, 4), dtype=np.float32)]
    y = [np.zeros((4, 4), dtype=np.float32)]
    monkeypatch.setattr(shape, "reshape_and_normalize_data", lambda *a, **k: (None, None, True))
    with pytest.raises(ValueError):
        shape.reshape_train_test(x, y, None, None, channels=(0, 0))


def test_reshape_train_test_shape_mismatch():
    x = [np.zeros((4, 4), dtype=np.float32)]
    y = [np.zeros((3, 3), dtype=np.float32)]
    with pytest.raises(ValueError):
        shape.reshape_train_test(x, y, None, None, channels=(0, 0))


def test_reshape_and_normalize_data_warns_no_channel_axis(caplog):
    x = [np.zeros((4, 4, 2), dtype=np.float32)]
    shape.reshape_and_normalize_data(x, test_data=None, channels=None, channel_axis=None)
    assert any("No channel axis specified" in rec.message for rec in caplog.records)


def test_reshape_and_normalize_data_with_channel_axis():
    x = [np.zeros((4, 4, 2), dtype=np.float32)]
    test_x = [np.zeros((4, 4, 2), dtype=np.float32)]
    train_out, test_out, run_test = shape.reshape_and_normalize_data(
        x, test_data=test_x, channels=None, channel_axis=0, normalize=False, dim=2
    )
    assert run_test is True
