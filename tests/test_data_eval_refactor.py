import numpy as np
import torch

from omnirefactor.data import eval as eval_mod


class DummyNet:
    def __call__(self, batch):
        # return tuple like torch model (y, aux)
        y = torch.zeros((batch.shape[0], 3, *batch.shape[-2:]), device=batch.device)
        return (y,)


class DummyModel:
    def __init__(self):
        self.nclasses = 3
        self.dim = 2
        self.unet = False
        self.net = DummyNet()


class DummyRunNetModel:
    def _run_net(self, batch):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return batch


def test_eval_set_stack_getitem_no_pad():
    data = np.zeros((2, 8, 8), dtype=np.float32)
    dataset = eval_mod.eval_set(data, dim=2, normalize=False, invert=False)
    img = dataset.__getitem__(0, no_pad=True)
    assert isinstance(img, torch.Tensor)
    assert img.shape == (8, 8)


def test_eval_set_list_channel_axis_and_rescale():
    data = [np.zeros((16, 32, 2), dtype=np.float32)]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        channel_axis=3,
        normalize=False,
        invert=False,
        rescale_factor=0.5,
        interp_mode="bilinear",
        pad_mode="constant",
        extra_pad=0,
    )
    img, inds, subs = dataset[0]
    assert img.shape[1] == 2  # channel axis moved to C
    assert len(inds) == 1
    assert len(subs) == 2


def test_eval_set_normalize_invert_with_contrast_limits():
    data = np.ones((1, 6, 6), dtype=np.float32)
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=True,
        invert=True,
        contrast_limits=(0.0, 1.0),
    )
    img = dataset.__getitem__(0, no_pad=True)
    assert img.min() >= 0
    assert img.max() <= 1


def test_eval_set_iterates():
    data = [np.zeros((5, 5), dtype=np.float32), np.zeros((5, 5), dtype=np.float32)]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        pad_mode="constant",
        extra_pad=0,
    )
    eval_mod.mp.get_worker_info = lambda: None
    items = list(iter(dataset))
    assert len(items) == 2


def test_eval_set_run_tiled_and_collate():
    data = np.zeros((1, 32, 32), dtype=np.float32)
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        tile=True,
        pad_mode="constant",
        extra_pad=0,
    )
    batch, inds, subs = dataset[0]
    dummy = DummyModel()
    out = dataset._run_tiled(batch, dummy, batch_size=1, augment=False, bsize=8, tile_overlap=0.1)
    assert out.shape[1] == dummy.nclasses

    batch2, inds2, subs2 = dataset[0]
    collated = dataset.collate_fn([(batch, inds, subs), (batch2, inds2, subs2)])
    assert isinstance(collated, tuple)
    assert collated[0].shape[0] == 2


def test_eval_loader_iter_and_sampler_len():
    sampler = eval_mod.sampler([0, 1, 2])
    assert len(sampler) == 3
    assert list(iter(sampler)) == [0, 1, 2]

    dataset = torch.utils.data.TensorDataset(torch.zeros((2, 1, 4, 4)))
    loader = eval_mod.eval_loader(dataset, DummyRunNetModel(), lambda x: x, batch_size=1)
    batches = list(iter(loader))
    assert len(batches) == 2


def test_eval_set_iter_worker_split():
    data = [np.zeros((32, 32), dtype=np.float32) for _ in range(5)]
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        pad_mode="constant",
        extra_pad=0,
    )

    class WorkerInfo:
        num_workers = 2
        id = 1

    eval_mod.mp.get_worker_info = lambda: WorkerInfo()
    items = list(iter(dataset))
    assert len(items) == 3


def test_eval_set_files_and_aics_branches():
    class DummyAICSImage:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_image_data(self, *_args, **_kwargs):
            return np.zeros((16, 16), dtype=np.float32)

    eval_mod.AICSImage = DummyAICSImage

    file_dataset = eval_mod.eval_set(
        ["fake.tif"],
        dim=2,
        normalize=False,
        invert=False,
        pad_mode="constant",
        extra_pad=0,
    )
    img, inds, subs = file_dataset[0]
    assert img.shape[1] == 1
    assert len(inds) == 1

    aics_dataset = eval_mod.eval_set(
        DummyAICSImage(),
        dim=2,
        normalize=False,
        invert=False,
        pad_mode="constant",
        extra_pad=0,
        aics_args={"slice_dim": "Z"},
    )
    img, inds, subs = aics_dataset[0]
    assert img.shape[1] == 1


def test_eval_set_run_tiled_augment(monkeypatch):
    data = np.zeros((1, 32, 32), dtype=np.float32)
    dataset = eval_mod.eval_set(
        data,
        dim=2,
        normalize=False,
        invert=False,
        tile=True,
        pad_mode="constant",
        extra_pad=0,
    )
    batch, inds, subs = dataset[0]
    dummy = DummyModel()
    called = {"unaugment": False}

    def fake_make_tiles(imgi, **_):
        IMG = torch.zeros((1, 1, 8, 8))
        subs = [slice(0, 32), slice(0, 32)]
        shape = (32, 32)
        inds = [0]
        return IMG, subs, shape, inds

    def fake_unaugment(y, inds, _unet):
        called["unaugment"] = True
        return y

    def fake_average_tiles(y, subs, shape):
        return torch.zeros((y.shape[1], *shape), device=y.device)

    monkeypatch.setattr(eval_mod, "make_tiles_ND", fake_make_tiles)
    monkeypatch.setattr(eval_mod, "unaugment_tiles_ND", fake_unaugment)
    monkeypatch.setattr(eval_mod, "average_tiles_ND", fake_average_tiles)

    out = dataset._run_tiled(batch, dummy, batch_size=1, augment=True, bsize=8, tile_overlap=0.1)
    assert out.shape[1] == dummy.nclasses
    assert called["unaugment"] is True
