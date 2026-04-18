"""GPU/CPU parity tests for core omnirefactor functions.

These tests run on real hardware — no monkeypatching. They compare GPU
outputs against CPU outputs to verify device-independent correctness.

On CPU-only systems, GPU tests are skipped (not failed).
"""

import numpy as np
import pytest
import torch

from omnirefactor import models
from omnirefactor.networks import UnetND
from omnirefactor.gpu import torch_GPU, torch_CPU


def _detect_gpu_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


GPU_DEVICE = _detect_gpu_device()
requires_gpu = pytest.mark.skipif(GPU_DEVICE is None, reason="No GPU available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_mask_2d(shape=(32, 32)):
    """Two non-overlapping square cells."""
    mask = np.zeros(shape, dtype=np.int32)
    mask[4:14, 4:14] = 1
    mask[18:28, 18:28] = 2
    return mask


def _simple_mask_batch(n=2, shape=(32, 32)):
    """Batch of simple masks."""
    return [_simple_mask_2d(shape) for _ in range(n)]


# ---------------------------------------------------------------------------
# Network forward pass
# ---------------------------------------------------------------------------

@requires_gpu
class TestUnetNDParity:
    def test_forward_pass(self):
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


# ---------------------------------------------------------------------------
# Full model eval pipeline
# ---------------------------------------------------------------------------

@requires_gpu
class TestOmniModelParity:
    def test_eval_pipeline(self):
        torch.manual_seed(1)
        np.random.seed(1)

        cpu_model = models.OmniModel(gpu=False, pretrained_model=False,
                                     nclasses=3, nchan=2, omni=False)
        torch.manual_seed(1)
        gpu_model = models.OmniModel(gpu=True, pretrained_model=False,
                                     nclasses=3, nchan=2, omni=False)

        module = gpu_model.net.module if isinstance(gpu_model.net, torch.nn.DataParallel) else gpu_model.net
        module.load_state_dict(cpu_model.net.state_dict())

        image = np.random.RandomState(7).rand(40, 40, 2).astype(np.float32)
        eval_args = dict(
            compute_masks=False, normalize=False, invert=False,
            rescale_factor=1.0, net_avg=False, resample=True,
            augment=False, tile=False, tile_overlap=0.1, bsize=64,
            omni=False, show_progress=False, verbose=False,
        )

        cpu_masks, cpu_flows = cpu_model.eval([image.copy()], **eval_args)
        gpu_masks, gpu_flows = gpu_model.eval([image.copy()], **eval_args)

        cpu_flow = cpu_flows[0][1]
        gpu_flow = gpu_flows[0][1]
        cpu_prob = cpu_flows[0][2]
        gpu_prob = gpu_flows[0][2]

        np.testing.assert_allclose(cpu_flow, gpu_flow, atol=1e-4)
        np.testing.assert_allclose(cpu_prob, gpu_prob, atol=1e-4)


# ---------------------------------------------------------------------------
# masks_to_flows: CPU vs GPU
# ---------------------------------------------------------------------------

@requires_gpu
class TestMasksToFlowsParity:
    def test_2d(self):
        from omnirefactor.core.flows import masks_to_flows

        mask = _simple_mask_2d()

        cpu_result = masks_to_flows(mask.copy(), use_gpu=False, device=torch_CPU, omni=True)
        gpu_result = masks_to_flows(mask.copy(), use_gpu=True, device=GPU_DEVICE, omni=True)

        cpu_mu = cpu_result.mu.cpu().numpy() if isinstance(cpu_result.mu, torch.Tensor) else cpu_result.mu
        gpu_mu = gpu_result.mu.cpu().numpy() if isinstance(gpu_result.mu, torch.Tensor) else gpu_result.mu
        cpu_T = cpu_result.T.cpu().numpy() if isinstance(cpu_result.T, torch.Tensor) else cpu_result.T
        gpu_T = gpu_result.T.cpu().numpy() if isinstance(gpu_result.T, torch.Tensor) else gpu_result.T

        np.testing.assert_allclose(cpu_mu, gpu_mu, atol=1e-4)
        np.testing.assert_allclose(cpu_T, gpu_T, atol=1e-4)

    def test_batch(self):
        from omnirefactor.core.flows import masks_to_flows_batch

        masks = np.stack(_simple_mask_batch(2)).astype(np.int64)
        links = [None] * len(masks)

        cpu_out = masks_to_flows_batch(masks, links=links, device=torch_CPU, omni=True)
        gpu_out = masks_to_flows_batch(masks, links=links, device=GPU_DEVICE, omni=True)

        # masks_to_flows_batch returns (masks, bd, T, mu, slices, ...)
        # mu (flows) should match closely; T (distance field) may differ
        # slightly due to eikonal solver convergence on different devices
        cpu_mu = cpu_out[3].cpu()
        gpu_mu = gpu_out[3].cpu()

        torch.testing.assert_close(cpu_mu, gpu_mu, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# follow_flows / steps: CPU vs GPU
# ---------------------------------------------------------------------------

@requires_gpu
class TestStepsParity:
    def test_follow_flows_batch(self):
        from omnirefactor.core.steps import follow_flows_batch

        B, D, H, W = 2, 2, 16, 16
        torch.manual_seed(42)
        dP = torch.randn(B, D, H, W) * 0.1

        cpu_p = follow_flows_batch(dP.clone().to(torch_CPU), niter=10, omni=True, suppress=True)
        gpu_p = follow_flows_batch(dP.clone().to(GPU_DEVICE), niter=10, omni=True, suppress=True)

        torch.testing.assert_close(cpu_p, gpu_p.cpu(), atol=1e-3, rtol=1e-3)

    def test_steps_batch(self):
        from omnirefactor.core.steps import steps_batch

        B, D, N = 2, 2, 64
        torch.manual_seed(7)
        p = torch.randn(B, D, N)
        dP = torch.randn(B, D, 8, 8) * 0.1

        cpu_final, _ = steps_batch(p.clone().to(torch_CPU), dP.clone().to(torch_CPU),
                                   niter=10, omni=True, suppress=True)
        gpu_final, _ = steps_batch(p.clone().to(GPU_DEVICE), dP.clone().to(GPU_DEVICE),
                                   niter=10, omni=True, suppress=True)

        torch.testing.assert_close(cpu_final, gpu_final.cpu(), atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# compute_masks: CPU vs GPU
# ---------------------------------------------------------------------------

@requires_gpu
class TestComputeMasksParity:
    def test_basic(self):
        from omnirefactor.core.flows import masks_to_flows
        from omnirefactor.core.masks import compute_masks

        mask = _simple_mask_2d()
        result = masks_to_flows(mask.copy(), use_gpu=False, device=torch_CPU, omni=True)
        dP = result.mu
        dist = result.T

        # Convert to numpy for compute_masks
        dP_np = dP.cpu().numpy() if isinstance(dP, torch.Tensor) else dP
        dist_np = dist.cpu().numpy() if isinstance(dist, torch.Tensor) else dist

        cpu_result = compute_masks(
            dP_np.copy(), dist_np.copy(), use_gpu=False, device=torch_CPU, omni=True,
        )
        gpu_result = compute_masks(
            dP_np.copy(), dist_np.copy(), use_gpu=True, device=GPU_DEVICE, omni=True,
        )

        cpu_masks = cpu_result[0]
        gpu_masks = gpu_result[0]

        # Mask labels may differ, but the number of cells should match
        assert cpu_masks.max() == gpu_masks.max(), (
            f"CPU found {cpu_masks.max()} cells, GPU found {gpu_masks.max()}"
        )


# ---------------------------------------------------------------------------
# Augmentation GPU paths
# ---------------------------------------------------------------------------

@requires_gpu
class TestAugmentParity:
    def test_mode_filter_gpu(self):
        from omnirefactor.transforms.augment import _mode_filter_gpu

        torch.manual_seed(0)
        labels = torch.zeros(1, 16, 16, dtype=torch.long)
        labels[0, 3:8, 3:8] = 1
        labels[0, 10:15, 10:15] = 2
        labels[0, 5, 5] = 2  # noise pixel

        cpu_out = _mode_filter_gpu(labels.clone().to(torch_CPU))
        gpu_out = _mode_filter_gpu(labels.clone().to(GPU_DEVICE))

        torch.testing.assert_close(cpu_out.cpu(), gpu_out.cpu())

    def test_gaussian_blur_gpu(self):
        from omnirefactor.transforms.augment import _gaussian_blur_gpu

        torch.manual_seed(0)
        # _gaussian_blur_gpu expects a single spatial array (H, W)
        img = torch.randn(32, 32)

        cpu_out = _gaussian_blur_gpu(img.clone().to(torch_CPU), sigma=1.0)
        gpu_out = _gaussian_blur_gpu(img.clone().to(GPU_DEVICE), sigma=1.0)

        torch.testing.assert_close(cpu_out.cpu(), gpu_out.cpu(), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# compute_flows_gpu (training pipeline)
# ---------------------------------------------------------------------------

@requires_gpu
class TestComputeFlowsGpu:
    def test_basic(self):
        from omnirefactor.data.train import train_set

        masks = _simple_mask_batch(2)
        images = [np.random.rand(1, 32, 32).astype(np.float32) for _ in range(2)]
        links = [None, None]

        ds = train_set(
            data=images, labels=masks, links=links,
            dim=2, nchan=1, nclasses=3, tyx=(32, 32),
            scale_range=0, omni=True, affinity_field=False,
            device=GPU_DEVICE, allow_blank_masks=False,
            do_rescale=False, diam_train=np.ones(2), diam_mean=np.ones(2),
        )

        # compute_flows_gpu takes raw masks and produces flow labels on GPU
        dummy_imgs = torch.randn(2, 1, 32, 32)
        imgi, lbl = ds.compute_flows_gpu(dummy_imgs, np.stack(masks), links, GPU_DEVICE)

        assert imgi.device.type == GPU_DEVICE.type
        assert lbl.device.type == GPU_DEVICE.type
        assert lbl.shape[0] == 2  # batch size
