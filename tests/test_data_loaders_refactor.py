import numpy as np
import pytest
import torch

from omnipose.data import train as train_mod
from omnipose.data import eval as eval_mod


def test_train_set_getitem_with_monkeypatch(monkeypatch):
    tyx = (16, 16)
    data = [np.zeros((1, 32, 32), dtype=np.float32) for _ in range(2)]
    labels = [np.zeros((32, 32), dtype=np.int32) for _ in range(2)]
    links = [None, None]

    def fake_random_rotate_and_resize(**kwargs):
        batch = len(kwargs["X"])
        t = kwargs["tyx"]
        imgs = [np.zeros((1,) + t, dtype=np.float32) for _ in range(batch)]
        lbls = [np.zeros(t, dtype=np.int32) for _ in range(batch)]
        return imgs, lbls, np.ones((batch, 2), dtype=np.float32), None

    def fake_masks_to_flows_batch(lbl, links, **_):
        nimg = len(lbl)
        shape = lbl[0].shape
        masks = torch.zeros((nimg,) + shape, dtype=torch.float32)
        bd = torch.zeros_like(masks)
        T = torch.zeros_like(masks)
        mu = torch.zeros((nimg, 2) + shape, dtype=torch.float32)
        slices = [tuple(slice(0, s) for s in shape) for _ in range(nimg)]
        return [masks, bd, T, mu, slices, None, None, None]

    def fake_batch_labels(masks, bd, T, mu, tyx, dim, nclasses, device):
        return torch.zeros((masks.shape[0], nclasses, *tyx), device=device)

    monkeypatch.setattr(train_mod, "random_rotate_and_resize", fake_random_rotate_and_resize)
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

    # __getitem__ now returns 4 values: (imgi, labels, links, inds)
    imgi, lbl, batch_links, inds = dataset[0]
    assert inds == [0]

    # collate_fn is a pass-through
    item = dataset[[0]]
    result = dataset.collate_fn([item])
    assert result is item


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
