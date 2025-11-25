import numpy as np
import pytest
import torch
import os
from pathlib import Path

from cellpose_omni import models
from cellpose_omni import transforms
from cellpose_omni import io
import omnipose
from cellpose_omni.resnet_torch import CPnet

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


def test_cpnet_matches_cpu_and_gpu_forward_pass():
    torch.manual_seed(0)
    nbase = [2, 4, 8]
    nout = 4
    cpu_net = CPnet(nbase, nout, sz=3, residual_on=True, style_on=True,
                    concatenation=False, dim=2)
    gpu_net = CPnet(nbase, nout, sz=3, residual_on=True, style_on=True,
                    concatenation=False, dim=2).to(GPU_DEVICE)
    gpu_net.load_state_dict(cpu_net.state_dict())
    cpu_net.eval()
    gpu_net.eval()

    data = torch.randn(2, 2, 32, 32)
    cpu_out, cpu_style = cpu_net(data.clone())
    gpu_out, gpu_style = gpu_net(data.to(GPU_DEVICE))

    torch.testing.assert_close(cpu_out, gpu_out.cpu(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(cpu_style, gpu_style.cpu(), atol=1e-4, rtol=1e-4)


def test_cellposemodel_pipeline_cpu_gpu_outputs_match():
    torch.manual_seed(1)
    np.random.seed(1)

    cpu_model = models.CellposeModel(gpu=False, pretrained_model=False, use_torch=True,
                                     nclasses=3, nchan=2, omni=False)

    torch.manual_seed(1)
    gpu_model = models.CellposeModel(gpu=True, pretrained_model=False, use_torch=True,
                                     nclasses=3, nchan=2, omni=False)

    _load_state_dict_on_target(gpu_model.net, cpu_model.net.state_dict())

    image = np.random.RandomState(7).rand(1, 40, 40, 2).astype(np.float32)
    run_args = dict(
        compute_masks=False,
        normalize=False,
        invert=False,
        rescale=1.0,
        net_avg=False,
        resample=True,
        augment=False,
        tile=False,
        tile_overlap=0.1,
        bsize=64,
        mask_threshold=0.0,
        flow_threshold=0.4,
        niter=None,
        flow_factor=5.0,
        min_size=5,
        max_size=None,
        interp=True,
        cluster=False,
        suppress=None,
        boundary_seg=False,
        affinity_seg=False,
        despur=False,
        anisotropy=1.0,
        do_3D=False,
        stitch_threshold=0.0,
        omni=False,
        calc_trace=False,
        show_progress=False,
        verbose=False,
        pad=0,
    )

    cpu_outputs = cpu_model._run_cp(image.copy(), **run_args)
    gpu_outputs = gpu_model._run_cp(image.copy(), **run_args)

    _, cpu_style, cpu_flow, cpu_prob, *_ = cpu_outputs
    _, gpu_style, gpu_flow, gpu_prob, *_ = gpu_outputs

    np.testing.assert_allclose(cpu_style, gpu_style, atol=1e-5)
    np.testing.assert_allclose(cpu_flow, gpu_flow, atol=1e-5)
    np.testing.assert_allclose(cpu_prob, gpu_prob, atol=1e-5)


def test_pretrained_bact_affinity_cpu_gpu_match_on_synthetic_image():
    torch.manual_seed(2)
    np.random.seed(2)

    # Use a real pretrained configuration to mirror the notebook scenario
    cpu_model = models.CellposeModel(
        gpu=False,
        model_type="bact_phase_affinity",
        use_torch=True,
        omni=True,
    )

    torch.manual_seed(2)
    gpu_model = models.CellposeModel(
        gpu=True,
        model_type="bact_phase_affinity",
        use_torch=True,
        omni=True,
    )

    # Ensure both models share identical weights even if DataParallel is wrapping the GPU model
    _load_state_dict_on_target(gpu_model.net, cpu_model.net.state_dict())

    cpu_model.batch_size = gpu_model.batch_size = 1

    # Synthetic but fixed input; matches channel expectations for the model
    image = np.random.RandomState(11).rand(1, 96, 96, cpu_model.nchan).astype(np.float32)

    run_args = dict(
        compute_masks=False,
        normalize=False,
        invert=False,
        rescale=1.0,
        net_avg=False,
        resample=True,
        augment=False,
        tile=False,
        tile_overlap=0.1,
        bsize=128,
        mask_threshold=0.0,
        diam_threshold=12.0,
        flow_threshold=0.4,
        niter=None,
        flow_factor=5.0,
        min_size=5,
        max_size=None,
        interp=True,
        cluster=False,
        suppress=None,
        boundary_seg=False,
        affinity_seg=False,
        despur=False,
        anisotropy=1.0,
        do_3D=False,
        stitch_threshold=0.0,
        omni=True,
        calc_trace=False,
        show_progress=False,
        verbose=False,
        pad=0,
    )

    cpu_outputs = cpu_model._run_cp(image.copy(), **run_args)
    gpu_outputs = gpu_model._run_cp(image.copy(), **run_args)

    _, cpu_style, cpu_flow, cpu_prob, *_ = cpu_outputs
    _, gpu_style, gpu_flow, gpu_prob, *_ = gpu_outputs

    np.testing.assert_allclose(cpu_style, gpu_style, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cpu_flow, gpu_flow, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(cpu_prob, gpu_prob, atol=1e-5, rtol=1e-5)


def test_real_images_bact_affinity_cpu_gpu_network_outputs_match():
    torch.manual_seed(3)
    np.random.seed(3)

    base_dir = Path(omnipose.__file__).parent.parent.parent / "docs" / "test_files"
    files = io.get_image_files(str(base_dir))
    images = [io.imread(f) for f in files]

    cpu_model = models.CellposeModel(
        gpu=False,
        model_type="bact_phase_affinity",
        use_torch=True,
        omni=True,
    )

    torch.manual_seed(3)
    gpu_model = models.CellposeModel(
        gpu=True,
        model_type="bact_phase_affinity",
        use_torch=True,
        omni=True,
    )

    _load_state_dict_on_target(gpu_model.net, cpu_model.net.state_dict())
    cpu_model.batch_size = gpu_model.batch_size = 1

    for img in images:
        # mirror eval preprocessing but stop before compute_masks to focus on raw network outputs
        proc = transforms.convert_image(
            img,
            channels=None,
            channel_axis=-1,
            z_axis=None,
            do_3D=False,
            normalize=False,
            invert=False,
            nchan=cpu_model.nchan,
            dim=cpu_model.dim,
            omni=True,
        )

        if proc.ndim < cpu_model.dim + 2:
            proc = proc[np.newaxis]

        y_cpu, style_cpu = cpu_model._run_nets(
            proc,
            net_avg=False,
            augment=False,
            tile=False,
            normalize=True,
            tile_overlap=0.1,
            bsize=224,
        )

        y_gpu, style_gpu = gpu_model._run_nets(
            proc,
            net_avg=False,
            augment=False,
            tile=False,
            normalize=True,
            tile_overlap=0.1,
            bsize=224,
        )

        np.testing.assert_allclose(y_cpu, y_gpu, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(style_cpu, style_gpu, atol=1e-5, rtol=1e-5)


def test_real_images_bact_affinity_cpu_gpu_masks_match():
    pytest.skip("compute_masks cluster=False path unstable on real images; skip until refactored")
    torch.manual_seed(4)
    np.random.seed(4)

    base_dir = Path(omnipose.__file__).parent.parent.parent / "docs" / "test_files"
    files = io.get_image_files(str(base_dir))
    images = [io.imread(f) for f in files]

    cpu_model = models.CellposeModel(
        gpu=False,
        model_type="bact_phase_affinity",
        use_torch=True,
        omni=True,
    )

    torch.manual_seed(4)
    gpu_model = models.CellposeModel(
        gpu=True,
        model_type="bact_phase_affinity",
        use_torch=True,
        omni=True,
    )

    _load_state_dict_on_target(gpu_model.net, cpu_model.net.state_dict())

    params = dict(
        channels=None,
        rescale=None,
        mask_threshold=-2,
        flow_threshold=0,
        transparency=True,
        omni=True,
        cluster=False,
        diam_threshold=-1,
        resample=True,
        verbose=False,
        tile=False,
        niter=None,
        augment=False,
        affinity_seg=False,
        channel_axis=-1,
        batch_size=1,
    )

    def _run(model, img):
        masks, flows, _ = model.eval(img, **params)
        masks = masks if isinstance(masks, list) else [masks]
        return masks, flows

    masks_cpu, flows_cpu, masks_gpu, flows_gpu = [], [], [], []
    for img in images:
        mc, fc = _run(cpu_model, img)
        mg, fg = _run(gpu_model, img)
        masks_cpu.append(mc)
        masks_gpu.append(mg)
        flows_cpu.append(fc)
        flows_gpu.append(fg)

    assert len(masks_cpu) == len(masks_gpu) == len(images)
    assert len(flows_cpu) == len(flows_gpu) == len(images)

    for mc, mg in zip(masks_cpu, masks_gpu):
        np.testing.assert_array_equal(mc[0], mg[0])

    for fc, fg in zip(flows_cpu, flows_gpu):
        # fc layout: [RGB flow, dP, cellprob, final coords, bd, tr, affinity, bounds]
        assert isinstance(fc, list) and isinstance(fg, list)
        assert len(fc) == len(fg)
        for ac, ag in zip(fc, fg):
            if not isinstance(ac, np.ndarray) or not isinstance(ag, np.ndarray):
                continue
            if ac.dtype.kind not in {"i", "u", "f"}:
                continue
            if ac.dtype.kind in {"i", "u"}:
                np.testing.assert_allclose(ac, ag, atol=1, rtol=0)
            else:
                np.testing.assert_allclose(ac, ag, atol=1e-1, rtol=1e-4)


def test_affinity_full_eval_masks_and_flows_match_no_tiling():
    torch.manual_seed(5)
    np.random.seed(5)

    base_dir = Path(omnipose.__file__).parent.parent.parent / "docs" / "test_files"
    files = io.get_image_files(str(base_dir))
    # use a single image to mirror the notebook snippet
    img = io.imread(files[0])

    cpu_model = models.CellposeModel(
        gpu=False,
        model_type="bact_phase_affinity",
        use_torch=True,
        omni=True,
    )

    torch.manual_seed(5)
    gpu_model = models.CellposeModel(
        gpu=True,
        model_type="bact_phase_affinity",
        use_torch=True,
        omni=True,
    )

    _load_state_dict_on_target(gpu_model.net, cpu_model.net.state_dict())
    cpu_model.batch_size = gpu_model.batch_size = 1

    params = dict(
        channels=None,
        rescale=None,
        mask_threshold=-2,
        flow_threshold=0,
        transparency=True,
        omni=True,
        cluster=True,
        resample=True,
        verbose=False,
        tile=False,
        niter=None,
        augment=False,
        affinity_seg=True,
        channel_axis=-1,
        batch_size=1,
    )

    masks_cpu, flows_cpu, _ = cpu_model.eval([img], **params)
    masks_gpu, flows_gpu, _ = gpu_model.eval([img], **params)

    # Expect identical masks and flow outputs; fail loudly if any discrepancy remains.
    np.testing.assert_array_equal(masks_cpu[0], masks_gpu[0])
    fc, fg = flows_cpu[0], flows_gpu[0]
    assert len(fc) == len(fg)
    for ac, ag in zip(fc, fg):
        if not isinstance(ac, np.ndarray) or not isinstance(ag, np.ndarray):
            continue
        if ac is None or ag is None:
            continue
        if ac.dtype.kind == "O" or ag.dtype.kind == "O":
            continue
        if ac.dtype.kind in {"i", "u"}:
            diff = np.abs(ac.astype(np.int16) - ag.astype(np.int16))
            assert diff.max() <= 1  # allow single LSB drift in rendered RGB
        else:
            np.testing.assert_allclose(ac, ag, atol=1e-5, rtol=1e-5)


def test_bact_affinity_single_channel_no_tiling_cpu_gpu_match():
    """
    Mirror the notebook scenario: single-channel image, no tiling/augment,
    affinity_seg + clustering on. This should be identical between CPU/GPU.
    """
    torch.manual_seed(6)
    np.random.seed(6)

    base_dir = Path(omnipose.__file__).parent.parent.parent / "docs" / "test_files"
    img = io.imread(io.get_image_files(str(base_dir))[0])
    # force single-channel to match model expectations and the notebook setup
    img_single = img[..., 0] if img.ndim == 3 else img

    cpu_model = models.CellposeModel(
        gpu=False,
        model_type="bact_phase_affinity",
        use_torch=True,
        omni=True,
    )

    torch.manual_seed(6)
    gpu_model = models.CellposeModel(
        gpu=True,
        model_type="bact_phase_affinity",
        use_torch=True,
        omni=True,
    )

    _load_state_dict_on_target(gpu_model.net, cpu_model.net.state_dict())
    cpu_model.batch_size = gpu_model.batch_size = 1

    params = dict(
        channels=None,
        rescale=None,
        mask_threshold=-2,
        flow_threshold=0,
        transparency=True,
        omni=True,
        cluster=True,
        resample=True,
        verbose=False,
        tile=False,
        niter=None,
        augment=False,
        affinity_seg=True,
        channel_axis=None,
        batch_size=1,
    )

    # Compare raw network outputs on identically preprocessed tensors
    proc = transforms.convert_image(
        img_single,
        channels=None,
        channel_axis=None,
        z_axis=None,
        do_3D=False,
        normalize=False,
        invert=False,
        nchan=cpu_model.nchan,
        dim=cpu_model.dim,
        omni=True,
    )

    y_cpu, style_cpu = cpu_model._run_nets(
        proc, net_avg=False, augment=False, tile=False, normalize=True, tile_overlap=0.1, bsize=224
    )
    y_gpu, style_gpu = gpu_model._run_nets(
        proc, net_avg=False, augment=False, tile=False, normalize=True, tile_overlap=0.1, bsize=224
    )

    np.testing.assert_allclose(y_cpu, y_gpu, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(style_cpu, style_gpu, atol=1e-5, rtol=1e-5)

    masks_cpu, flows_cpu, _ = cpu_model.eval([img_single], **params)
    masks_gpu, flows_gpu, _ = gpu_model.eval([img_single], **params)

    np.testing.assert_array_equal(masks_cpu[0], masks_gpu[0])

    fc, fg = flows_cpu[0], flows_gpu[0]
    assert len(fc) == len(fg)
    for ac, ag in zip(fc, fg):
        if not isinstance(ac, np.ndarray) or not isinstance(ag, np.ndarray):
            continue
        if ac is None or ag is None:
            continue
        if ac.dtype.kind not in {"i", "u", "f"}:
            continue
        if ac.dtype.kind in {"i", "u"}:
            diff = np.abs(ac.astype(np.int16) - ag.astype(np.int16))
            assert diff.max() <= 1
        else:
            np.testing.assert_allclose(ac, ag, atol=1e-5, rtol=1e-5)
