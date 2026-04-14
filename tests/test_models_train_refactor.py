import sys
import types
import warnings

import numpy as np
import pytest
import torch

from omnirefactor.models import OmniModel
from omnirefactor.models import train as train_mod


class TinyNet(torch.nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, nclasses, kernel_size=1)
        self.mkldnn = False
        self.saved = []

    def forward(self, x):
        y = self.conv(x)
        style = torch.zeros((1,), device=y.device)
        return y, style

    def save_model(self, path):
        self.saved.append(path)


def make_model(*, dim=2, nchan=1, nclasses=3, logits=True):
    model = OmniModel(
        gpu=False,
        pretrained_model=False,
        model_type=None,
        net_avg=False,
        use_torch=True,
        dim=dim,
        nchan=nchan,
        nclasses=nclasses,
        omni=True,
        logits=logits,
    )
    model.device = torch.device("cpu")
    model.gpu = False
    model.net = TinyNet(model.nclasses)
    return model


def _patch_manual_train_helpers(monkeypatch):
    def fake_random_rotate_and_resize(**kwargs):
        batch = len(kwargs["X"])
        tyx = kwargs["tyx"]
        imgi = [np.zeros((1, *tyx), np.float32) for _ in range(batch)]
        lbl = [np.zeros(tyx, np.int32) for _ in range(batch)]
        scale = np.ones((batch, len(tyx)), np.float32)
        return imgi, lbl, scale, None

    def fake_masks_to_flows_batch(lbl, links, **_):
        shape = lbl[0].shape
        x = torch.zeros(shape, dtype=torch.float32)
        slices = [(slice(0, shape[0]), slice(0, shape[1]))]
        affinity_graph = None
        return [x, x.clone(), x.clone(), x.clone(), slices, None, None, affinity_graph]

    def fake_batch_labels(*_args, **kwargs):
        tyx = _args[4]
        device = kwargs["device"]
        return torch.zeros((1, 2, *tyx), device=device)

    # Patch within the data.train module where these are actually called
    monkeypatch.setattr(train_mod.data.train, "random_rotate_and_resize", fake_random_rotate_and_resize)
    monkeypatch.setattr(train_mod.data.train, "masks_to_flows_batch", fake_masks_to_flows_batch)
    monkeypatch.setattr(train_mod.data.train, "batch_labels", fake_batch_labels)
    monkeypatch.setattr(train_mod.core.diam, "diameters", lambda *_args, **_kwargs: 10.0)


def test_train_calls_train_net_and_filters(monkeypatch, caplog):
    model = make_model()

    train_data = [np.zeros((1, 8, 8), np.float32)]
    train_labels = [np.zeros((8, 8), np.int32)]
    train_links = [None]

    def fake_reshape(train_data, train_labels, test_data, test_labels, channels, channel_axis, normalize, dim, omni):
        return train_data, train_labels, test_data, test_labels, True

    monkeypatch.setattr(train_mod.transforms, "reshape_train_test", fake_reshape)
    monkeypatch.setattr(train_mod.core, "labels_to_flows", lambda *args, **kwargs: [np.zeros((4, 8, 8))])
    monkeypatch.setattr(train_mod.ncolor, "format_labels", lambda x: x)

    called = {}

    def fake_train_net(**kwargs):
        called.update(kwargs)
        return "model-path"

    model._train_net = fake_train_net

    out = train_mod.train(
        model,
        train_data=train_data,
        train_labels=train_labels,
        train_links=train_links,
        test_data=[np.zeros((1, 8, 8), np.float32)],
        test_labels=[np.zeros((8, 8), np.int32)],
        test_links=[None],
        channels=None,
        channel_axis=0,
        normalize=True,
        min_train_masks=0,
        rescale_factor=0.5,
    )

    assert out == "model-path"
    assert called["do_rescale"] is True
    assert called["channels"] is None
    assert called["train_data"] == train_data


def test_train_run_test_false_and_remove(monkeypatch):
    model = make_model()
    train_data = [np.zeros((1, 8, 8), np.float32)]
    train_labels = [np.zeros((8, 8), np.int32)]
    train_links = [None]

    def fake_reshape(train_data, train_labels, test_data, test_labels, channels, channel_axis, normalize, dim, omni):
        return train_data, train_labels, test_data, test_labels, False

    monkeypatch.setattr(train_mod.transforms, "reshape_train_test", fake_reshape)
    monkeypatch.setattr(train_mod.ncolor, "format_labels", lambda x: x)

    called = {}

    def fake_train_net(**kwargs):
        called.update(kwargs)
        return "model-path"

    model._train_net = fake_train_net

    out = train_mod.train(
        model,
        train_data=train_data,
        train_labels=train_labels,
        train_links=train_links,
        test_data=None,
        test_labels=None,
        test_links=None,
        channels=[0, 0],
        channel_axis=0,
        normalize=False,
        min_train_masks=2,
        rescale=True,
    )

    assert out == "model-path"
    assert called["test_labels"] is None


def test_train_step_symmetry_and_grad_check(monkeypatch):
    model = make_model()
    model.autocast = False
    model.optimizer = torch.optim.SGD(model.net.parameters(), lr=0.01)
    model.MSELoss = torch.nn.MSELoss()

    def fake_core_loss(self, lbl, y, ext_loss=None):
        loss = y.mean()
        if ext_loss is not None:
            loss = loss + ext_loss
        return loss, loss.detach(), {}

    monkeypatch.setattr(train_mod, "core_loss", fake_core_loss)

    x = torch.zeros((1, 1, 8, 8))
    lbl = torch.zeros((1, 1, 8, 8))

    train_loss = train_mod._train_step(model, x, lbl, symmetry_weight=1)
    assert torch.isfinite(train_loss).all()

    def fake_clip(params, max_norm):
        for p in params:
            if p.grad is not None:
                p.grad[:] = float("nan")
        return 0.0

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip)
    with pytest.raises(RuntimeError, match="Non-finite gradient"):
        train_mod._train_step(model, x, lbl, symmetry_weight=0)


def test_train_step_autocast_branch(monkeypatch):
    model = make_model()
    model.autocast = True

    class DummyScaler:
        def scale(self, loss):
            return loss

        def unscale_(self, _optimizer):
            return None

        def step(self, _optimizer):
            return None

        def update(self):
            return None

    model.scaler = DummyScaler()
    model.optimizer = torch.optim.SGD(model.net.parameters(), lr=0.01)
    model.MSELoss = torch.nn.MSELoss()

    def fake_core_loss(self, lbl, y, ext_loss=None):
        loss = y.mean()
        return loss, loss.detach(), {}

    monkeypatch.setattr(train_mod, "core_loss", fake_core_loss)
    class DummyCtx:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(train_mod, "autocast", lambda: DummyCtx())

    # add a param with no grad to hit the None-grad branch
    for p in model.net.parameters():
        p.requires_grad_(False)
        break

    x = torch.zeros((1, 1, 8, 8))
    lbl = torch.zeros((1, 1, 8, 8))
    train_mod._train_step(model, x, lbl, symmetry_weight=0)


def test_train_step_nonfinite_loss(monkeypatch):
    model = make_model()
    model.autocast = False
    model.optimizer = torch.optim.SGD(model.net.parameters(), lr=0.01)
    model.MSELoss = torch.nn.MSELoss()

    def fake_core_loss(self, lbl, y, ext_loss=None):
        loss = torch.tensor(float("nan"), requires_grad=True)
        return loss, loss.detach(), {}

    monkeypatch.setattr(train_mod, "core_loss", fake_core_loss)

    x = torch.zeros((1, 1, 8, 8))
    lbl = torch.zeros((1, 1, 8, 8))
    with pytest.raises(RuntimeError, match="Non-finite loss"):
        train_mod._train_step(model, x, lbl, symmetry_weight=0)


def test_set_optimizer_and_learning_rate(monkeypatch):
    model = make_model()
    model.net = TinyNet(model.nclasses)

    train_mod._set_optimizer(model, learning_rate=0.01, momentum=0.9, weight_decay=0.0, SGD=True)
    assert isinstance(model.optimizer, torch.optim.SGD)

    dummy_mod = types.SimpleNamespace(
        RAdam=type(
            "DummyRAdam",
            (torch.optim.Optimizer,),
            {
                "__init__": lambda self, params, lr, betas, eps, weight_decay: torch.optim.Optimizer.__init__(
                    self, params, {"lr": lr}
                ),
                "step": lambda self, closure=None: None,
            },
        )
    )
    monkeypatch.setitem(sys.modules, "torch_optimizer", dummy_mod)

    train_mod._set_optimizer(model, learning_rate=0.02, momentum=0.9, weight_decay=0.0, SGD=False)
    assert hasattr(model.optimizer, "current_lr")

    train_mod._set_learning_rate(model, 0.123)
    assert model.optimizer.param_groups[0]["lr"] == 0.123



def test_train_net_manual_batching(monkeypatch, tmp_path):
    model = make_model()
    model.gpu = True
    model.net.module = types.SimpleNamespace(save_model=model.net.save_model)
    model.unet = True

    train_data = [np.zeros((1, 8, 8), np.float32)]
    train_labels = [np.zeros((8, 8), np.int32)]
    train_links = [None]
    test_data = [np.zeros((1, 8, 8), np.float32)]
    test_labels = [np.zeros((8, 8), np.int32)]
    test_links = [None]

    _patch_manual_train_helpers(monkeypatch)

    model._train_step = lambda *_args, **_kwargs: torch.tensor(1.0)

    out = train_mod._train_net(
        model,
        train_data=train_data,
        train_labels=train_labels,
        train_links=train_links,
        test_data=test_data,
        test_labels=test_labels,
        test_links=test_links,
        save_path=str(tmp_path),
        save_every=1,
        save_each=True,
        learning_rate=0.1,
        n_epochs=1,
        momentum=0.9,
        weight_decay=0.0,
        SGD=True,
        batch_size=1,
        num_workers=0,
        nimg_per_epoch=None,
        do_rescale=True,
        netstr=None,
        do_autocast=False,
        tyx=(9, 9),
        timing=True,
    )

    assert out is not None
    assert model.net.saved


def test_train_net_dataloader_branch(monkeypatch, tmp_path):
    model = make_model()

    class DummyDataset:
        def __init__(self, data, labels, links, **kwargs):
            self.data = data
            self.labels = labels
            self.links = links
            self.nimg = len(data)
            self.tyx = kwargs["tyx"]
            self.nclasses = kwargs.get("nclasses", 2)

        def __len__(self):
            return self.nimg

        def __getitem__(self, inds):
            if not isinstance(inds, list):
                inds = [inds]
            img = np.zeros((1, *self.tyx), dtype=np.float32)
            masks = [np.zeros(self.tyx, dtype=np.int32) for _ in inds]
            links = [None] * len(inds)
            return img, masks, links, inds

        def collate_fn(self, worker_data):
            return worker_data[0]

        def worker_init_fn(self, *_args):
            return None

        def compute_flows_gpu(self, imgi, masks_np, links, device):
            lbl = torch.zeros((1, self.nclasses, *self.tyx), device=device)
            if not isinstance(imgi, torch.Tensor):
                imgi = torch.tensor(np.array(imgi), device=device, dtype=torch.float32)
            return imgi, lbl

    class DummyLoader:
        def __init__(self, dataset, batch_sampler, collate_fn, **_kwargs):
            self.dataset = dataset
            self.batch_sampler = batch_sampler
            self.collate_fn = collate_fn

        def __iter__(self):
            for batch in self.batch_sampler:
                yield self.collate_fn([self.dataset[batch]])

    monkeypatch.setattr(train_mod.data.train, "train_set", DummyDataset)
    monkeypatch.setattr(torch.utils.data, "DataLoader", DummyLoader)
    monkeypatch.setattr(torch.multiprocessing, "set_start_method", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_mod.core.diam, "diameters", lambda *_args, **_kwargs: 10.0)

    model._train_step = lambda *_args, **_kwargs: torch.tensor(1.0)

    out = train_mod._train_net(
        model,
        train_data=[np.zeros((1, 8, 8), np.float32)],
        train_labels=[np.zeros((8, 8), np.int32)],
        train_links=[None],
        test_data=[np.zeros((1, 8, 8), np.float32)],
        test_labels=[np.zeros((8, 8), np.int32)],
        test_links=[None],
        save_path=str(tmp_path),
        save_every=1,
        save_each=False,
        learning_rate=0.1,
        n_epochs=1,
        momentum=0.9,
        weight_decay=0.0,
        SGD=True,
        batch_size=1,
        num_workers=1,
        nimg_per_epoch=1,
        do_rescale=False,
        netstr=None,
        do_autocast=True,
        tyx=(8, 8),
        timing=True,
    )

    assert out is not None


def test_train_net_lr_schedule_large_epochs(monkeypatch):
    model = make_model()
    _patch_manual_train_helpers(monkeypatch)
    model._train_step = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop"))

    with pytest.raises(RuntimeError, match="stop"):
        train_mod._train_net(
            model,
            train_data=[np.zeros((1, 8, 8), np.float32)],
            train_labels=[np.zeros((8, 8), np.int32)],
            train_links=[None],
            test_data=None,
            test_labels=None,
            test_links=None,
            save_path=None,
            save_every=1,
            save_each=False,
            learning_rate=0.1,
            n_epochs=251,
            momentum=0.9,
            weight_decay=0.0,
            SGD=True,
            batch_size=1,
            num_workers=0,
            nimg_per_epoch=1,
            do_rescale=False,
            netstr=None,
            do_autocast=False,
            tyx=(8, 8),
            timing=False,
        )

def test_train_net_lr_list(monkeypatch):
    model = make_model()
    _patch_manual_train_helpers(monkeypatch)
    model._train_step = lambda *_args, **_kwargs: torch.tensor(1.0)
    train_mod._train_net(
        model,
        train_data=[np.zeros((1, 8, 8), np.float32)],
        train_labels=[np.zeros((8, 8), np.int32)],
        train_links=[None],
        test_data=None,
        test_labels=None,
        test_links=None,
        save_path=None,
        save_every=1,
        save_each=False,
        learning_rate=[0.1, 0.1],
        n_epochs=2,
        momentum=0.9,
        weight_decay=0.0,
        SGD=False,
        batch_size=1,
        num_workers=0,
        nimg_per_epoch=1,
        do_rescale=False,
        netstr=None,
        do_autocast=False,
        tyx=None,
        timing=False,
    )


def test_train_net_lr_scalar_non_sgd(monkeypatch):
    model = make_model()
    _patch_manual_train_helpers(monkeypatch)
    model._train_step = lambda *_args, **_kwargs: torch.tensor(1.0)
    train_mod._train_net(
        model,
        train_data=[np.zeros((1, 8, 8), np.float32)],
        train_labels=[np.zeros((8, 8), np.int32)],
        train_links=[None],
        test_data=None,
        test_labels=None,
        test_links=None,
        save_path=None,
        save_every=1,
        save_each=False,
        learning_rate=0.1,
        n_epochs=1,
        momentum=0.9,
        weight_decay=0.0,
        SGD=False,
        batch_size=1,
        num_workers=0,
        nimg_per_epoch=1,
        do_rescale=False,
        netstr=None,
        do_autocast=False,
        tyx=(8, 8),
        timing=False,
    )


def test_train_net_tyx_default_3d(monkeypatch):
    model = make_model(dim=3)
    _patch_manual_train_helpers(monkeypatch)
    model._train_step = lambda *_args, **_kwargs: torch.tensor(1.0)
    train_mod._train_net(
        model,
        train_data=[np.zeros((1, 8, 8, 8), np.float32)],
        train_labels=[np.zeros((8, 8, 8), np.int32)],
        train_links=[None],
        test_data=None,
        test_labels=None,
        test_links=None,
        save_path=None,
        save_every=1,
        save_each=False,
        learning_rate=0.1,
        n_epochs=1,
        momentum=0.9,
        weight_decay=0.0,
        SGD=True,
        batch_size=1,
        num_workers=0,
        nimg_per_epoch=1,
        do_rescale=False,
        netstr=None,
        do_autocast=False,
        tyx=None,
        timing=False,
    )


def test_train_net_invalid_inputs():
    model = make_model()
    train_data = [np.zeros((1, 8, 8), np.float32)]
    train_labels = [np.zeros((8, 8), np.int32)]
    train_links = [None]

    with pytest.raises(ValueError, match="learning_rate.ndim"):
        train_mod._train_net(
            model,
            train_data=train_data,
            train_labels=train_labels,
            train_links=train_links,
            test_data=None,
            test_labels=None,
            test_links=None,
            save_path=None,
            save_every=1,
            save_each=False,
            learning_rate=np.ones((2, 2)),
            n_epochs=1,
            momentum=0.9,
            weight_decay=0.0,
            SGD=True,
            batch_size=1,
            num_workers=0,
            nimg_per_epoch=1,
            do_rescale=False,
            netstr=None,
            do_autocast=False,
            tyx=(8, 8),
            timing=False,
        )

    with pytest.raises(ValueError, match="length n_epochs"):
        train_mod._train_net(
            model,
            train_data=train_data,
            train_labels=train_labels,
            train_links=train_links,
            test_data=None,
            test_labels=None,
            test_links=None,
            save_path=None,
            save_every=1,
            save_each=False,
            learning_rate=[0.1, 0.2],
            n_epochs=3,
            momentum=0.9,
            weight_decay=0.0,
            SGD=True,
            batch_size=1,
            num_workers=0,
            nimg_per_epoch=1,
            do_rescale=False,
            netstr=None,
            do_autocast=False,
            tyx=(8, 8),
            timing=False,
        )

    with pytest.raises(ValueError, match="tyx must be a tuple"):
        train_mod._train_net(
            model,
            train_data=train_data,
            train_labels=train_labels,
            train_links=train_links,
            test_data=None,
            test_labels=None,
            test_links=None,
            save_path=None,
            save_every=1,
            save_each=False,
            learning_rate=0.1,
            n_epochs=1,
            momentum=0.9,
            weight_decay=0.0,
            SGD=True,
            batch_size=1,
            num_workers=0,
            nimg_per_epoch=1,
            do_rescale=False,
            netstr=None,
            do_autocast=False,
            tyx=[8, 8],
            timing=False,
        )
