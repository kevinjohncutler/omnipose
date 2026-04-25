"""Verify omnipose.gpu re-exports from ocdkit and sets ARM env vars."""

import os
import omnipose.gpu as gpu_mod


def test_reexports_ocdkit():
    from ocdkit.utils.gpu import get_device, empty_cache, torch_GPU
    assert gpu_mod.get_device is get_device
    assert gpu_mod.empty_cache is empty_cache
    assert gpu_mod.torch_GPU is torch_GPU


def test_omnipose_gpu_env_vars():
    """omnipose.gpu sets OMP_NUM_THREADS on ARM."""
    if gpu_mod.ARM:
        assert os.environ.get("OMP_NUM_THREADS") == "1"
        assert os.environ.get("PARLAY_NUM_THREADS") == "1"
