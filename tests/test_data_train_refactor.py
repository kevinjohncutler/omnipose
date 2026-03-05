import os

import numpy as np
import torch

from omnirefactor.data import train as train_mod


def _make_train_set(**kwargs):
    data = [np.zeros((1, 8, 8), dtype=np.float32) for _ in range(2)]
    labels = [np.zeros((8, 8), dtype=np.int32) for _ in range(2)]
    links = [None, None]
    base = dict(
        dim=2,
        nchan=1,
        nclasses=3,
        device=torch.device("cpu"),
        diam_train=np.ones(2, np.float32),
        diam_mean=30.0,
        tyx=(8, 8),
        scale_range=1.0,
        allow_blank_masks=True,
        omni=True,
        affinity_field=False,
    )
    base.update(kwargs)
    return train_mod.train_set(data, labels, links, **base)


def test_train_set_iter_collate_and_worker_init(monkeypatch):
    dataset = _make_train_set()

    class WorkerInfo:
        num_workers = 2
        id = 1

    train_mod.mp.get_worker_info = lambda: WorkerInfo()
    items = list(iter(dataset))
    assert len(items) == 1

    imgs, labels, inds = dataset.collate_fn(items)
    assert imgs.shape[0] == 1
    assert labels.shape[0] == 1
    assert inds == [1]

    dataset.worker_init_fn(0)
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_train_set_defaults_and_iter_no_worker(monkeypatch):
    def fake_random_crop_warp(**kwargs):
        tyx = kwargs["tyx"]
        img = np.zeros((1,) + tyx, dtype=np.float32)
        lbl = np.zeros(tyx, dtype=np.float32)
        scale = np.ones((2,), dtype=np.float32)
        return img, lbl, scale

    def fake_masks_to_flows_batch(labels, links, **_):
        masks = torch.zeros((1,) + labels.shape[-2:])
        bd = torch.zeros_like(masks)
        T = torch.zeros((1, 2) + labels.shape[-2:])
        mu = torch.zeros_like(T)
        slices = [tuple(slice(0, t) for t in labels.shape[-2:])]
        return [masks, bd, T, mu, slices, None, None, None]

    def fake_batch_labels(*_args, **_kwargs):
        return torch.zeros((1, 1, 16, 16))

    monkeypatch.setattr(train_mod, "random_crop_warp", fake_random_crop_warp)
    monkeypatch.setattr(train_mod, "masks_to_flows_batch", fake_masks_to_flows_batch)
    monkeypatch.setattr(train_mod, "batch_labels", fake_batch_labels)

    dataset = _make_train_set(tyx=None, dim=2)
    train_mod.mp.get_worker_info = lambda: None

    items = list(iter(dataset))
    assert len(items) == len(dataset)


def test_train_set_getitem(monkeypatch):
    dataset = _make_train_set(augment=True, timing=True)

    def fake_random_crop_warp(**kwargs):
        tyx = kwargs["tyx"]
        img = np.zeros((1,) + tyx, dtype=np.float32)
        lbl = np.zeros(tyx, dtype=np.float32)
        scale = np.ones((dataset.dim,), dtype=np.float32)
        return img, lbl, scale

    def fake_masks_to_flows_batch(labels, links, **_):
        nimg = labels.shape[0]
        tyx = labels.shape[-2:]
        masks = torch.zeros((nimg,) + tyx)
        bd = torch.zeros_like(masks)
        T = torch.zeros((nimg, dataset.dim) + tyx)
        mu = torch.zeros_like(T)
        slices = [tuple(slice(0, t) for t in tyx) for _ in range(nimg)]
        affinity = torch.zeros((nimg,))
        return [masks, bd, T, mu, slices, None, None, affinity]

    def fake_batch_labels(*_args, **_kwargs):
        return torch.zeros((1, dataset.nclasses) + dataset.tyx)

    monkeypatch.setattr(train_mod, "random_crop_warp", fake_random_crop_warp)
    monkeypatch.setattr(train_mod, "masks_to_flows_batch", fake_masks_to_flows_batch)
    monkeypatch.setattr(train_mod, "batch_labels", fake_batch_labels)

    imgs, lbl, inds = dataset[0]
    assert imgs.shape[-2:] == dataset.tyx
    assert lbl.shape[-2:] == dataset.tyx
    assert inds == [0]


def test_cycling_random_batch_sampler():
    data = list(range(5))
    sampler = train_mod.CyclingRandomBatchSampler(data, batch_size=2, generator=torch.Generator().manual_seed(0))
    it = iter(sampler)
    first = next(it)
    second = next(it)
    assert len(first) <= 2
    assert len(second) <= 2
    assert len(sampler) == 3

    sampler = train_mod.CyclingRandomBatchSampler(list(range(3)), batch_size=2, generator=torch.Generator().manual_seed(1))
    it = iter(sampler)
    _ = next(it)
    _ = next(it)
    spill = next(it)
    assert len(spill) == 1
    _ = next(it)
