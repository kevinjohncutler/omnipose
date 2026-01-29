import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def net_module():
    pkg_name = "omnirefactor.networks"
    saved_pkg = sys.modules.get(pkg_name)
    saved_sub = sys.modules.get(f"{pkg_name}.network")
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "omnirefactor" / "networks")]
        sys.modules[pkg_name] = pkg
    path = Path(__file__).resolve().parents[1] / "src" / "omnirefactor" / "networks" / "network.py"
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.network", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        if saved_pkg is None:
            sys.modules.pop(pkg_name, None)
        else:
            sys.modules[pkg_name] = saved_pkg
        if saved_sub is None:
            sys.modules.pop(f"{pkg_name}.network", None)
        else:
            sys.modules[f"{pkg_name}.network"] = saved_sub


def test_norm_helpers_and_dilation_list(net_module):
    net = net_module
    net.set_norm_type("group")
    norm = net._make_norm(8, dim=2)
    assert isinstance(norm, torch.nn.GroupNorm)

    net.set_norm_type("batch")
    norm = net._make_norm(8, dim=3)
    assert isinstance(norm, torch.nn.BatchNorm3d)

    with pytest.raises(ValueError):
        net.set_norm_type("invalid")
        net._make_norm(8, dim=2)

    assert net._select_group_count(24) == 8
    assert net._select_group_count(7) == 1
    assert net.dilation_list(5, 3) == [1, 3, 5]


def test_batchconv_and_unet_forward_paths(net_module, tmp_path):
    net = net_module
    net.set_norm_type("batch")
    block = net.batchconv(1, 2, kernel_size=3, dim=2, dilation=1, relu=False)
    assert isinstance(block[-1], torch.nn.Conv2d)

    model = net.UnetND(
        nbase=[1, 2, 4],
        nout=3,
        sz=3,
        residual_on=True,
        style_on=False,
        concatenation=False,
        dim=2,
        checkpoint=False,
        dropout=True,
        kernel_size=2,
        scale_factor=2,
        dilation=1,
        norm_type="batch",
    )
    x = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
    y, style = model(x)
    assert y.shape[1] == 3
    assert style.shape[-1] == 4

    _ = net.UnetND(
        nbase=[1, 2, 4],
        nout=3,
        sz=3,
        residual_on=False,
        style_on=True,
        concatenation=False,
        dim=2,
        checkpoint=False,
        dropout=False,
        kernel_size=2,
        scale_factor=2,
        dilation=1,
        norm_type="batch",
    )

    # cover cpu load_model path
    model_path = tmp_path / "net.pt"
    torch.save(model.state_dict(), model_path)
    model.load_model(str(model_path), cpu=True)


def test_unet_load_model_non_cpu(net_module, monkeypatch, tmp_path):
    net = net_module
    model = net.UnetND(
        nbase=[1, 2, 4],
        nout=3,
        sz=3,
        residual_on=True,
        style_on=True,
        concatenation=False,
        dim=2,
        checkpoint=False,
        dropout=False,
        kernel_size=2,
        scale_factor=2,
        dilation=1,
        norm_type="batch",
    )
    model_path = tmp_path / "net_gpu.pt"
    torch.save(model.state_dict(), model_path)
    monkeypatch.setattr(net, "torch_GPU", torch.device("cpu"), raising=False)
    model.load_model(str(model_path), cpu=False)


def test_convdown_convup_and_batchconvstyle(net_module):
    net = net_module
    conv = net.convdown(1, 2, sz=3, dim=2, dilation=1)
    x = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    y = conv(x)
    assert y.shape[1] == 2

    class Parent:
        sz = 3
        concatenation = False
        dim = 2
        dilation = 1

    cu = net.convup(4, 2, style_channels=3, parent=Parent())
    x = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    y = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    style = torch.zeros((1, 3), dtype=torch.float32)
    out = cu(x, y, style)
    assert out.shape[1] == 2

    _ = net.batchconvstyle(2, 2, style_channels=3, sz=3, dim=2, dilation=1, concatenation=True)
    bcs = net.batchconvstyle(2, 2, style_channels=3, sz=3, dim=2, dilation=1, concatenation=False)
    x = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    style = torch.zeros((1, 3), dtype=torch.float32)
    out = bcs(style, x, y=x)
    assert out.shape == x.shape
