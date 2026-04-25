import os

import numpy as np
import pytest
import torch

from omnipose.data import train as train_mod


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
    assert len(items) == 1  # worker 1 of 2 gets 1 item

    # collate_fn is a pass-through that returns worker_data[0]
    result = dataset.collate_fn(items)
    assert result is items[0]

    dataset.worker_init_fn(0)
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_train_set_getitem(monkeypatch):
    dataset = _make_train_set(augment=True, timing=True)

    def fake_random_rotate_and_resize(**kwargs):
        tyx = kwargs["tyx"]
        batch = len(kwargs["X"])
        img = [np.zeros((1,) + tyx, dtype=np.float32) for _ in range(batch)]
        lbl = [np.zeros(tyx, dtype=np.int32) for _ in range(batch)]
        scale = np.ones((batch, len(tyx)), dtype=np.float32)
        return img, lbl, scale, None

    def fake_masks_to_flows_batch(labels, links, **_):
        nimg = len(labels)
        tyx = labels[0].shape[-2:]
        masks = torch.zeros((nimg,) + tyx)
        bd = torch.zeros_like(masks)
        T = torch.zeros((nimg, dataset.dim) + tyx)
        mu = torch.zeros_like(T)
        slices = [tuple(slice(0, t) for t in tyx) for _ in range(nimg)]
        affinity = torch.zeros((nimg,))
        return [masks, bd, T, mu, slices, None, None, affinity]

    def fake_batch_labels(*_args, **_kwargs):
        return torch.zeros((1, dataset.nclasses) + dataset.tyx)

    monkeypatch.setattr(train_mod, "random_rotate_and_resize", fake_random_rotate_and_resize)
    monkeypatch.setattr(train_mod, "masks_to_flows_batch", fake_masks_to_flows_batch)
    monkeypatch.setattr(train_mod, "batch_labels", fake_batch_labels)

    imgi, lbl, links, inds = dataset[0]
    assert inds == [0]


def test_train_set_defaults_no_tyx(monkeypatch):
    def fake_random_rotate_and_resize(**kwargs):
        tyx = kwargs["tyx"]
        batch = len(kwargs["X"])
        img = [np.zeros((1,) + tyx, dtype=np.float32) for _ in range(batch)]
        lbl = [np.zeros(tyx, dtype=np.int32) for _ in range(batch)]
        return img, lbl, np.ones((batch, 2), dtype=np.float32)

    def fake_masks_to_flows_batch(labels, links, **_):
        nimg = len(labels)
        tyx = (16, 16)
        masks = torch.zeros((nimg,) + tyx)
        bd = torch.zeros_like(masks)
        T = torch.zeros((nimg, 2) + tyx)
        mu = torch.zeros_like(T)
        slices = [tuple(slice(0, t) for t in tyx)] * nimg
        return [masks, bd, T, mu, slices, None, None, None]

    def fake_batch_labels(*_args, **_kwargs):
        return torch.zeros((1, 3, 16, 16))

    monkeypatch.setattr(train_mod, "random_rotate_and_resize", fake_random_rotate_and_resize)
    monkeypatch.setattr(train_mod, "masks_to_flows_batch", fake_masks_to_flows_batch)
    monkeypatch.setattr(train_mod, "batch_labels", fake_batch_labels)

    dataset = _make_train_set(tyx=None, dim=2)
    assert dataset.tyx is not None  # default was computed


def test_cycling_random_batch_sampler():
    data = list(range(5))
    sampler = train_mod.CyclingRandomBatchSampler(data, batch_size=2, n_epochs=2)
    it = iter(sampler)
    first = next(it)
    second = next(it)
    assert len(first) <= 2
    assert len(second) <= 2
    assert len(sampler) == 3  # ceil(5/2)

    # Sampler with 3 items, batch_size=2: epochs yield [2, 1] or similar batches
    sampler2 = train_mod.CyclingRandomBatchSampler(list(range(3)), batch_size=2, n_epochs=2)
    it2 = iter(sampler2)
    b1 = next(it2)
    b2 = next(it2)
    assert len(b1) + len(b2) == 3  # first epoch covers all 3 items
