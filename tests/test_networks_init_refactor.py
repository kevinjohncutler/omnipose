import importlib

import torch


def test_parse_model_string_variants():
    nets = importlib.import_module("omnipose.networks")
    assert nets.parse_model_string("unet3_residual_on_style_on_concatenation_on") == (True, True, True)
    assert nets.parse_model_string("cellpose_residual_on_style_on_concatenation_off") == (True, True, False)
    assert nets.parse_model_string("custom_model") == (True, True, False)


def test_assign_device_cpu(monkeypatch):
    nets = importlib.import_module("omnipose.networks")

    def fake_get_device(_):
        return torch.device("cpu"), False

    monkeypatch.setattr(nets, "get_device", fake_get_device)
    device, available = nets.assign_device(gpu=False)
    assert device.type == "cpu"
    assert available is False


def test_assign_device_gpu_calls_lock(monkeypatch):
    nets = importlib.import_module("omnipose.networks")
    called = {}

    def fake_get_device(_):
        return torch.device("cuda"), True

    def fake_lock(device):
        called["lock"] = device

    monkeypatch.setattr(nets, "get_device", fake_get_device)
    monkeypatch.setattr(nets, "_lock_cuda_precision", fake_lock)
    device, available = nets.assign_device(gpu=True)
    assert device.type == "cuda"
    assert called.get("lock") is not None


def test_lock_cuda_precision_tf32_env(monkeypatch):
    nets = importlib.import_module("omnipose.networks")
    monkeypatch.setattr(nets, "_ALLOW_TF32_ENV", True)
    monkeypatch.setattr(nets, "_CUDA_PRECISION_LOCKED", False)
    nets._lock_cuda_precision(torch.device("cuda"))
    assert nets._CUDA_PRECISION_LOCKED is True


def test_lock_cuda_precision_cuda_branch(monkeypatch):
    nets = importlib.import_module("omnipose.networks")
    monkeypatch.setattr(nets, "_ALLOW_TF32_ENV", False)
    monkeypatch.setattr(nets, "_CUDA_PRECISION_LOCKED", False)
    nets._lock_cuda_precision(torch.device("cuda"))
    assert nets._CUDA_PRECISION_LOCKED is True


def test_lock_cuda_precision_locked_noop(monkeypatch):
    nets = importlib.import_module("omnipose.networks")
    monkeypatch.setattr(nets, "_CUDA_PRECISION_LOCKED", True)
    nets._lock_cuda_precision(torch.device("cuda"))
    assert nets._CUDA_PRECISION_LOCKED is True
