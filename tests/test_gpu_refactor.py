import importlib
import os

import torch


def _reload_gpu(monkeypatch, processor_value, mps_available=True, mps_raises=False):
    import ocdkit.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "torch_GPU", gpu_mod.torch_GPU, raising=False)
    monkeypatch.setattr(gpu_mod.platform, "processor", lambda: processor_value)
    if hasattr(torch.backends, "mps"):
        if mps_raises:
            def _raise():
                raise RuntimeError("boom")
            monkeypatch.setattr(torch.backends.mps, "is_available", _raise, raising=False)
        else:
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available, raising=False)
    return importlib.reload(gpu_mod)


def test_device_arm_detection(monkeypatch):
    gpu_mod = _reload_gpu(monkeypatch, processor_value="arm64", mps_available=True)
    assert gpu_mod.ARM is True


def test_device_mps_check_exception(monkeypatch):
    gpu_mod = _reload_gpu(monkeypatch, processor_value="arm64", mps_raises=True)
    assert gpu_mod.ARM is False


def test_get_device_failure_path(monkeypatch):
    import ocdkit.gpu as gpu_mod
    monkeypatch.setattr(gpu_mod, "ARM", False)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no gpu")

    monkeypatch.setattr(torch, "zeros", _boom)
    dev, ok = gpu_mod.get_device(0)
    assert ok is False
    assert dev.type == "cpu"


def test_omnirefactor_gpu_env_vars():
    """omnirefactor.gpu sets OMP_NUM_THREADS on ARM."""
    import omnirefactor.gpu
    # If we're on ARM, the env vars should be set
    if omnirefactor.gpu.ARM:
        assert os.environ.get("OMP_NUM_THREADS") == "1"
        assert os.environ.get("PARLAY_NUM_THREADS") == "1"
