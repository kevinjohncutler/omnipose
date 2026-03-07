import importlib
import os
import sys
import types

import torch


def _reload_device(monkeypatch, processor_value, mps_available=True, mps_raises=False):
    import omnirefactor.gpu.device as device

    # Save current torch_GPU so monkeypatch restores module state after the test.
    # importlib.reload() changes module globals in ways monkeypatch won't undo on its own.
    monkeypatch.setattr(device, "torch_GPU", device.torch_GPU, raising=False)

    # patch platform.processor before reload
    monkeypatch.setattr(device.platform, "processor", lambda: processor_value)
    if mps_raises:
        monkeypatch.setattr(device.gpu_logger, "warning", lambda *args, **kwargs: None)
    # patch torch.backends.mps.is_available
    if hasattr(torch.backends, "mps"):
        if mps_raises:
            def _raise():
                raise RuntimeError("boom")
            monkeypatch.setattr(torch.backends.mps, "is_available", _raise, raising=False)
        else:
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available, raising=False)
    return importlib.reload(device)


def test_device_arm_env_set(monkeypatch):
    device = _reload_device(monkeypatch, processor_value="arm64", mps_available=True)
    assert device.ARM is True
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("PARLAY_NUM_THREADS") == "1"


def test_device_mps_check_exception(monkeypatch):
    device = _reload_device(monkeypatch, processor_value="arm64", mps_raises=True)
    assert device.ARM is False


def test_get_gpu_torch_failure_path(monkeypatch):
    import omnirefactor.gpu.device as device
    monkeypatch.setattr(device, "ARM", False)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no gpu")

    monkeypatch.setattr(torch, "zeros", _boom)
    dev, ok = device._get_gpu_torch(0)
    assert ok is False
    assert dev.type == "cpu"


def _reload_cache(monkeypatch, arm_value, stub_mps=False):
    import omnirefactor.gpu.cache as cache
    import omnirefactor.gpu.device as device

    monkeypatch.setattr(device, "ARM", arm_value)
    if stub_mps:
        # force ImportError from `from torch.mps import empty_cache`
        stub = types.ModuleType("torch.mps")
        sys.modules["torch.mps"] = stub
    return importlib.reload(cache)


def test_cache_uses_cuda_on_non_arm(monkeypatch):
    cache = _reload_cache(monkeypatch, arm_value=False)
    assert cache.empty_cache is torch.cuda.empty_cache


def test_cache_fallback_on_missing_mps(monkeypatch):
    cache = _reload_cache(monkeypatch, arm_value=True, stub_mps=True)
    assert cache.empty_cache is torch.cuda.empty_cache
