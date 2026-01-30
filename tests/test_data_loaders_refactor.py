import numpy as np
import pytest
import torch

from omnirefactor.data import train as train_mod
from omnirefactor.data import eval as eval_mod


def test_train_set_getitem_with_monkeypatch(monkeypatch):
    tyx = (16, 16)
    data = [np.zeros((32, 32), dtype=np.float32) for _ in range(2)]
    labels = [np.zeros((32, 32), dtype=np.float32) for _ in range(2)]
    links = [None, None]

    def fake_random_crop_warp(img, Y, tyx, **_):
        return np.zeros(tyx, dtype=np.float32), np.zeros(tyx, dtype=np.float32), np.ones((2,), dtype=np.float32)

    def fake_masks_to_flows_batch(labels, links, **_):
        masks = torch.zeros(labels.shape[0], *labels.shape[1:], dtype=torch.float32)
        bd = torch.zeros_like(masks)
        T = torch.zeros_like(masks)
        mu = torch.zeros((labels.shape[0], 2, *labels.shape[1:]), dtype=torch.float32)
        slices = [(slice(0, tyx[0]), slice(0, tyx[1])) for _ in range(labels.shape[0])]
        return [masks, bd, T, mu, slices, None, None, None]

    def fake_batch_labels(masks, bd, T, mu, tyx, dim, nclasses, device):
        return torch.zeros((masks.shape[0], nclasses, *tyx), device=device)

    monkeypatch.setattr(train_mod, "random_crop_warp", fake_random_crop_warp)
    monkeypatch.setattr(train_mod, "masks_to_flows_batch", fake_masks_to_flows_batch)
    monkeypatch.setattr(train_mod, "batch_labels", fake_batch_labels)

    dataset = train_mod.train_set(
        data=data,
        labels=labels,
        links=links,
        dim=2,
        nchan=1,
        nclasses=4,
        device=torch.device("cpu"),
        diam_train=np.ones(2, dtype=np.float32),
        diam_mean=30.0,
        scale_range=1.0,
        tyx=tyx,
        allow_blank_masks=True,
        augment=False,
        affinity_field=False,
        omni=False,
    )

    imgi, lbl, inds = dataset[0]
    assert imgi.shape == (1, 1) + tyx
    assert lbl.shape == (1, 4) + tyx
    assert inds == [0]

    batch_imgs, batch_labels, batch_inds = dataset.collate_fn([dataset[0], dataset[1]])
    assert batch_imgs.shape[0] == 2
    assert batch_labels.shape[0] == 2
    assert batch_inds == [0, 1]


def test_eval_set_stack_basic():
    data = np.zeros((2, 32, 32), dtype=np.float32)
    dataset = eval_mod.eval_set(data, dim=2, normalize=True, invert=True, rescale_factor=1.0)
    imgs = dataset[0]
    assert isinstance(imgs, tuple)
    padded, inds, subs = imgs
    assert padded.ndim == 4
    assert inds == [0]
    assert len(subs) == 2


def test_eval_set_list_no_pad():
    data = [np.zeros((32, 32), dtype=np.float32)]
    dataset = eval_mod.eval_set(data, dim=2, normalize=False, invert=False, rescale_factor=None)
    imgs = dataset.__getitem__(0, no_pad=True)
    # Shape is (C, H, W) = (1, 32, 32) - channel dimension is preserved
    assert imgs.shape == (1, 32, 32)


def test_eval_set_run_tiled():
    data = np.zeros((1, 32, 32), dtype=np.float32)
    dataset = eval_mod.eval_set(data, dim=2, normalize=False, invert=False, rescale_factor=1.0)
    batch, _, _ = dataset[0]

    class DummyModel:
        nclasses = 2
        dim = 2
        unet = False

        def __init__(self):
            self.net = self

        def __call__(self, x):
            out = torch.ones((x.shape[0], self.nclasses, *x.shape[-2:]), device=x.device)
            return (out,)

    model = DummyModel()
    out = dataset._run_tiled(batch, model, batch_size=2, augment=False, bsize=16, normalize=False)
    assert out.shape[0] == batch.shape[0]
    assert out.shape[1] == model.nclasses
