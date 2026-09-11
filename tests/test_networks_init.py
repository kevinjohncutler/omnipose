import importlib

import torch


def test_parse_model_string_variants():
    nets = importlib.import_module("omnipose.networks")
    assert nets.parse_model_string("unet3_residual_on_style_on_concatenation_on") == (True, True, True)
    assert nets.parse_model_string("cellpose_residual_on_style_on_concatenation_off") == (True, True, False)
    assert nets.parse_model_string("custom_model") == (True, True, False)


def test_assign_device_cpu(monkeypatch):
    precision = importlib.import_module("omnipose.networks.precision")

    def fake_get_device(_):
        return torch.device("cpu"), False

    monkeypatch.setattr(precision, "get_device", fake_get_device)
    device, available = precision.assign_device(gpu=False)
    assert device.type == "cpu"
    assert available is False


def test_assign_device_gpu_calls_lock(monkeypatch):
    precision = importlib.import_module("omnipose.networks.precision")
    called = {}

    def fake_get_device(_):
        return torch.device("cuda"), True

    def fake_lock(device):
        called["lock"] = device

    monkeypatch.setattr(precision, "get_device", fake_get_device)
    monkeypatch.setattr(precision, "_lock_cuda_precision", fake_lock)
    device, available = precision.assign_device(gpu=True)
    assert device.type == "cuda"
    assert called.get("lock") is not None


def test_lock_cuda_precision_tf32_env(monkeypatch):
    precision = importlib.import_module("omnipose.networks.precision")
    monkeypatch.setattr(precision, "_ALLOW_TF32_ENV", True)
    monkeypatch.setattr(precision, "_CUDA_PRECISION_LOCKED", False)
    precision._lock_cuda_precision(torch.device("cuda"))
    assert precision._CUDA_PRECISION_LOCKED is True


def test_lock_cuda_precision_cuda_branch(monkeypatch):
    precision = importlib.import_module("omnipose.networks.precision")
    monkeypatch.setattr(precision, "_ALLOW_TF32_ENV", False)
    monkeypatch.setattr(precision, "_CUDA_PRECISION_LOCKED", False)
    precision._lock_cuda_precision(torch.device("cuda"))
    assert precision._CUDA_PRECISION_LOCKED is True


def test_lock_cuda_precision_locked_noop(monkeypatch):
    precision = importlib.import_module("omnipose.networks.precision")
    monkeypatch.setattr(precision, "_CUDA_PRECISION_LOCKED", True)
    precision._lock_cuda_precision(torch.device("cuda"))
    assert precision._CUDA_PRECISION_LOCKED is True
