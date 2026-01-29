import importlib
import types

import torch


def _reload_cache(monkeypatch, arm_value, mps_empty_cache):
    import omnirefactor.gpu.device as device

    monkeypatch.setattr(device, "ARM", arm_value, raising=False)
    if mps_empty_cache is not None:
        if not hasattr(torch, "mps"):
            torch.mps = types.SimpleNamespace()
        torch.mps.empty_cache = mps_empty_cache
    return importlib.reload(importlib.import_module("omnirefactor.gpu.cache"))


def test_cache_uses_cuda_empty_cache(monkeypatch):
    cache = _reload_cache(monkeypatch, arm_value=False, mps_empty_cache=None)
    assert cache.empty_cache is torch.cuda.empty_cache


def test_cache_uses_mps_empty_cache(monkeypatch):
    sentinel = object()

    def _mps_cache():
        return sentinel

    cache = _reload_cache(monkeypatch, arm_value=True, mps_empty_cache=_mps_cache)
    assert cache.empty_cache is _mps_cache
