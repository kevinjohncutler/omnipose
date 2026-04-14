import numpy as np
import pytest
import torch

from omnirefactor import models
from omnirefactor.networks import UnetND


def _detect_gpu_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


GPU_DEVICE = _detect_gpu_device()

if GPU_DEVICE is None:
    pytest.skip("GPU backend is required for parity tests", allow_module_level=True)


def _load_state_dict_on_target(target_module, state_dict):
    module = target_module.module if isinstance(target_module, torch.nn.DataParallel) else target_module
    module.load_state_dict(state_dict)


def test_unetnd_matches_cpu_and_gpu_forward_pass():
    torch.manual_seed(0)
    nbase = [2, 4, 8]
    nout = 4
    cpu_net = UnetND(nbase, nout, sz=3, residual_on=True, style_on=True,
                     concatenation=False, dim=2)
    gpu_net = UnetND(nbase, nout, sz=3, residual_on=True, style_on=True,
                     concatenation=False, dim=2).to(GPU_DEVICE)
    gpu_net.load_state_dict(cpu_net.state_dict())
    cpu_net.eval()
    gpu_net.eval()

    data = torch.randn(2, 2, 32, 32)
    cpu_out, cpu_style = cpu_net(data.clone())
    gpu_out, gpu_style = gpu_net(data.to(GPU_DEVICE))

    torch.testing.assert_close(cpu_out, gpu_out.cpu(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(cpu_style, gpu_style.cpu(), atol=1e-4, rtol=1e-4)


def test_omnimodel_pipeline_cpu_gpu_outputs_match():
    torch.manual_seed(1)
    np.random.seed(1)

    cpu_model = models.OmniModel(gpu=False, pretrained_model=False,
                                 nclasses=3, nchan=2, omni=False)

    torch.manual_seed(1)
    gpu_model = models.OmniModel(gpu=True, pretrained_model=False,
                                 nclasses=3, nchan=2, omni=False)

    _load_state_dict_on_target(gpu_model.net, cpu_model.net.state_dict())

    image = np.random.RandomState(7).rand(40, 40, 2).astype(np.float32)
    eval_args = dict(
        compute_masks=False,
        normalize=False,
        invert=False,
        rescale_factor=1.0,
        net_avg=False,
        resample=True,
        augment=False,
        tile=False,
        tile_overlap=0.1,
        bsize=64,
        omni=False,
        show_progress=False,
        verbose=False,
    )

    cpu_masks, cpu_flows = cpu_model.eval([image.copy()], **eval_args)
    gpu_masks, gpu_flows = gpu_model.eval([image.copy()], **eval_args)

    # flows[0]: [rgb, dP, dist, p, bd, tr, affinity, bounds]
    cpu_flow = cpu_flows[0][1]  # dP
    gpu_flow = gpu_flows[0][1]
    cpu_prob = cpu_flows[0][2]  # dist
    gpu_prob = gpu_flows[0][2]

    np.testing.assert_allclose(cpu_flow, gpu_flow, atol=1e-4)
    np.testing.assert_allclose(cpu_prob, gpu_prob, atol=1e-4)
