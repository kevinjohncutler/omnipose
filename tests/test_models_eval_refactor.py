import warnings

import numpy as np

from omnirefactor.data import eval as eval_mod
from omnirefactor.models import OmniModel, eval as meval
from omnirefactor.models import run as mrun


def make_model(*, dim=2, nchan=1, nclasses=3, omni=True, logits=True):
    model = OmniModel(
        gpu=False,
        pretrained_model=False,
        model_type=None,
        net_avg=False,
        use_torch=True,
        dim=dim,
        nchan=nchan,
        nclasses=nclasses,
        omni=omni,
        logits=logits,
    )
    return model


def stub_run_nets(model):
    def _run_nets(img, **_):
        h, w = img.shape[:2]
        y = np.zeros((h, w, model.nclasses), dtype=np.float32)
        style = np.ones((model.nbase[-1],), dtype=np.float32)
        return y, style

    model._run_nets = _run_nets
    return model


def stub_run_network(model):
    model._from_device = lambda x: x.detach().cpu().numpy()

    def run_network(batch, to_numpy=False):
        return (
            batch.new_zeros((batch.shape[0], model.nclasses, *batch.shape[-2:])),
            np.ones((batch.shape[0], model.nbase[-1]), dtype=np.float32),
        )

    model.run_network = run_network
    return model


def stub_run_3d(model):
    def _run_3D(img, **_):
        shape = img.shape[-3:]
        d0 = np.zeros(shape, dtype=np.float32)
        d1 = np.zeros(shape, dtype=np.float32)
        cellprob = np.ones(shape, dtype=np.float32)
        bd = np.zeros(shape, dtype=np.float32)
        yf = [(d0, d1, cellprob, bd) for _ in range(3)]
        styles = np.ones((model.nbase[-1],), dtype=np.float32)
        return yf, styles

    model._run_3D = _run_3D
    return model


def make_stub_model(*, dim=2, nchan=1, nclasses=3, omni=True, logits=True):
    model = make_model(dim=dim, nchan=nchan, nclasses=nclasses, omni=omni, logits=logits)
    stub_run_nets(model)
    stub_run_network(model)
    return model


def test_eval_top_level():
    model = make_stub_model()
    img = np.zeros((16, 16, 1), dtype=np.float32)
    masks, flows = meval.eval(
        model,
        [img],
        channels=[0, 0],
        normalize=False,
        invert=False,
        rescale_factor=1.0,
        compute_masks=False,
        net_avg=False,
        augment=False,
        tile=False,
        bsize=16,
        show_progress=False,
        verbose=False,
    )
    assert isinstance(flows, list)


def test_eval_dataset_branch(monkeypatch):
    model = make_stub_model()

    def fake_compute_masks(dP, dist, **_):
        mask = np.zeros(dist.shape, dtype=np.int32)
        p = np.zeros((model.dim,) + dist.shape, dtype=np.float32)
        tr = np.zeros_like(p)
        bounds = np.zeros(dist.shape, dtype=bool)
        affinity = np.zeros((1,), dtype=np.float32)
        return mask, p, tr, bounds, affinity

    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks)

    dataset = eval_mod.eval_set([np.zeros((16, 16), dtype=np.float32)], dim=2, normalize=False, invert=False)

    masks, flows = meval.eval(
        model,
        dataset,
        channels=[0, 0],
        normalize=False,
        invert=False,
        rescale_factor=1.0,
        compute_masks=True,
        net_avg=False,
        augment=False,
        tile=False,
        bsize=16,
        show_progress=False,
        verbose=False,
        num_workers=0,
    )
    assert isinstance(masks, np.ndarray)
    assert isinstance(flows, list)


def test_eval_dataset_branch_with_tile(monkeypatch):
    model = make_stub_model()
    model.net = lambda x: (x.new_zeros((x.shape[0], model.nclasses, *x.shape[-2:])),)

    def fake_compute_masks(dP, dist, **_):
        mask = np.zeros(dist.shape, dtype=np.int32)
        p = np.zeros((model.dim,) + dist.shape, dtype=np.float32)
        tr = np.zeros_like(p)
        bounds = np.zeros(dist.shape, dtype=bool)
        affinity = np.zeros((1,), dtype=np.float32)
        return mask, p, tr, bounds, affinity

    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks)

    dataset = eval_mod.eval_set([np.zeros((16, 16), dtype=np.float32)], dim=2, normalize=False, invert=False)

    masks, flows = meval.eval(
        model,
        dataset,
        channels=[0, 0],
        normalize=False,
        invert=False,
        rescale_factor=1.0,
        compute_masks=True,
        net_avg=False,
        augment=False,
        tile=True,
        bsize=8,
        show_progress=False,
        verbose=False,
        num_workers=0,
    )
    assert isinstance(masks, np.ndarray)
    assert isinstance(flows, list)


def test_eval_dataset_branch_no_masks(monkeypatch):
    model = make_stub_model()

    dataset = eval_mod.eval_set([np.zeros((16, 16), dtype=np.float32)], dim=2, normalize=False, invert=False)

    masks, flows = meval.eval(
        model,
        dataset,
        channels=[0, 0],
        normalize=False,
        invert=False,
        rescale_factor=1.0,
        compute_masks=False,
        net_avg=False,
        augment=False,
        tile=False,
        bsize=16,
        show_progress=False,
        verbose=False,
        num_workers=0,
    )
    assert isinstance(masks, np.ndarray)
    assert isinstance(flows, list)


def test_eval_dataset_branch_hysteresis_off(monkeypatch):
    model = make_stub_model()

    def fake_compute_masks(dP, dist, **_):
        mask = np.zeros(dist.shape, dtype=np.int32)
        p = np.zeros((model.dim,) + dist.shape, dtype=np.float32)
        tr = np.zeros_like(p)
        bounds = np.zeros(dist.shape, dtype=bool)
        affinity = np.zeros((1,), dtype=np.float32)
        return mask, p, tr, bounds, affinity

    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks)

    dataset = eval_mod.eval_set([np.zeros((16, 16), dtype=np.float32)], dim=2, normalize=False, invert=False)

    masks, flows = meval.eval(
        model,
        dataset,
        channels=[0, 0],
        normalize=False,
        invert=False,
        rescale_factor=1.0,
        compute_masks=True,
        net_avg=False,
        augment=False,
        tile=False,
        bsize=16,
        show_progress=False,
        verbose=False,
        num_workers=0,
    )
    assert isinstance(masks, np.ndarray)
    assert isinstance(flows, list)


def test_eval_pretrained_model_list_initialization():
    model = make_stub_model()
    model.pretrained_model = ["bact_phase_omni", "other_model"]
    model.gpu = False
    model.torch = True
    model.net.load_model = lambda *_args, **_kwargs: None

    img = np.zeros((16, 16, 1), dtype=np.float32)
    masks, flows = meval.eval(
        model,
        img,
        channels=[0, 0],
        normalize=False,
        invert=False,
        rescale_factor=1.0,
        compute_masks=False,
        net_avg=False,
        augment=False,
        tile=False,
        bsize=16,
        show_progress=False,
        verbose=False,
        loop_run=False,
        model_loaded=False,
    )
    assert isinstance(masks, np.ndarray)
    assert isinstance(flows, list)


def test_eval_rescale_kwarg_ignored():
    model = make_stub_model()
    img = np.zeros((16, 16, 1), dtype=np.float32)
    masks, flows = meval.eval(
        model,
        [img],
        channels=[0, 0],
        normalize=False,
        invert=False,
        rescale=2.0,
        compute_masks=False,
        net_avg=False,
        augment=False,
        tile=False,
        bsize=16,
        show_progress=False,
        verbose=False,
    )
    assert isinstance(masks, np.ndarray)
    assert isinstance(flows, list)


def test_eval_single_image_array_branch():
    model = make_stub_model()
    img = np.zeros((16, 16), dtype=np.float32)
    masks, flows = meval.eval(
        model,
        img,
        channels=[0, 0],
        normalize=False,
        invert=False,
        compute_masks=False,
        net_avg=False,
        augment=False,
        tile=False,
        bsize=16,
        show_progress=False,
        verbose=False,
    )
    assert isinstance(masks, np.ndarray)
    assert isinstance(flows, list)


def test_eval_dataset_branch_with_real_file(monkeypatch):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sample = root / "docs" / "test_files" / "Sample000033.png"
    if not sample.exists():
        import pytest

        pytest.skip("sample image not available")

    model = make_stub_model()

    def fake_compute_masks(dP, dist, **_):
        mask = np.zeros(dist.shape, dtype=np.int32)
        p = np.zeros((model.dim,) + dist.shape, dtype=np.float32)
        tr = np.zeros_like(p)
        bounds = np.zeros(dist.shape, dtype=bool)
        affinity = np.zeros((1,), dtype=np.float32)
        return mask, p, tr, bounds, affinity

    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks)

    dataset = eval_mod.eval_set(
        [str(sample)],
        dim=2,
        normalize=False,
        invert=False,
        pad_mode="constant",
        extra_pad=0,
    )

    masks, flows = meval.eval(
        model,
        dataset,
        channels=[0, 0],
        normalize=False,
        invert=False,
        rescale_factor=1.0,
        compute_masks=True,
        net_avg=False,
        augment=False,
        tile=False,
        bsize=16,
        show_progress=False,
        verbose=False,
        num_workers=0,
    )
    assert isinstance(masks, np.ndarray)
    assert isinstance(flows, list)


def test_eval_dataset_branch_rescale_resample(monkeypatch):
    model = make_stub_model()

    def fake_compute_masks(dP, dist, **_):
        mask = np.zeros(dist.shape, dtype=np.int32)
        p = np.zeros((model.dim,) + dist.shape, dtype=np.float32)
        tr = np.zeros_like(p)
        bounds = np.zeros(dist.shape, dtype=bool)
        affinity = np.zeros((1,), dtype=np.float32)
        return mask, p, tr, bounds, affinity

    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks)

    dataset = eval_mod.eval_set([np.zeros((32, 32), dtype=np.float32)], dim=2, normalize=False, invert=False)

    for resample in (True, False):
        masks, flows = meval.eval(
            model,
            dataset,
            channels=[0, 0],
            normalize=False,
            invert=False,
            rescale_factor=0.5,
            compute_masks=True,
            net_avg=False,
            augment=False,
            tile=False,
            bsize=16,
            show_progress=False,
            verbose=False,
            num_workers=0,
            resample=resample,
        )
        assert isinstance(masks, np.ndarray)
        assert isinstance(flows, list)
    


def test_eval_dataset_branch_normalize_invert_rescale_none(monkeypatch):
    model = make_stub_model()

    def fake_compute_masks(dP, dist, **_):
        mask = np.zeros(dist.shape, dtype=np.int32)
        p = np.zeros((model.dim,) + dist.shape, dtype=np.float32)
        tr = np.zeros_like(p)
        bounds = np.zeros(dist.shape, dtype=bool)
        affinity = np.zeros((1,), dtype=np.float32)
        return mask, p, tr, bounds, affinity

    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks)

    dataset = eval_mod.eval_set([np.zeros((16, 16), dtype=np.float32)], dim=2, normalize=False, invert=False)

    masks, flows = meval.eval(
        model,
        dataset,
        channels=[0, 0],
        normalize=True,
        invert=True,
        rescale_factor=None,
        compute_masks=True,
        net_avg=False,
        augment=False,
        tile=False,
        bsize=16,
        show_progress=False,
        verbose=False,
        num_workers=0,
    )
    assert isinstance(masks, np.ndarray)
    assert isinstance(flows, list)


def test_eval_dataparallel_verbose(monkeypatch):
    model = make_stub_model()
    model.gpu = True
    model.torch = True

    class DummyModule:
        def load_model(self, *_args, **_kwargs):
            return None

    class DummyNet:
        module = DummyModule()

        def load_model(self, *_args, **_kwargs):
            return None

    model.net = DummyNet()
    model.pretrained_model = ["dummy_path"]

    img = np.zeros((16, 16), dtype=np.float32)
    masks, flows = meval.eval(
        model,
        img,
        channels=[0, 0],
        normalize=False,
        invert=False,
        rescale_factor=1.0,
        compute_masks=False,
        net_avg=False,
        augment=False,
        tile=False,
        bsize=16,
        show_progress=False,
        verbose=True,
        loop_run=False,
        model_loaded=False,
        omni=False,
    )
    assert isinstance(masks, np.ndarray)
    assert isinstance(flows, list)


def test_eval_dataset_branch_boundary_output(monkeypatch):
    model = make_stub_model(nclasses=4, logits=True)
    model.nclasses = 4
    model.dim = 2

    def fake_compute_masks(dP, dist, **_):
        mask = np.zeros(dist.shape, dtype=np.int32)
        p = np.zeros((model.dim,) + dist.shape, dtype=np.float32)
        tr = np.zeros_like(p)
        bounds = np.zeros(dist.shape, dtype=bool)
        affinity = np.zeros((1,), dtype=np.float32)
        return mask, p, tr, bounds, affinity

    monkeypatch.setattr(meval.core, "compute_masks", fake_compute_masks)

    dataset = eval_mod.eval_set([np.zeros((16, 16), dtype=np.float32)], dim=2, normalize=False, invert=False)

    masks, flows = meval.eval(
        model,
        dataset,
        channels=[0, 0],
        normalize=False,
        invert=False,
        rescale_factor=1.0,
        compute_masks=True,
        net_avg=False,
        augment=False,
        tile=False,
        bsize=16,
        show_progress=False,
        verbose=False,
        num_workers=0,
    )
    assert isinstance(masks, np.ndarray)
    assert isinstance(flows, list)
