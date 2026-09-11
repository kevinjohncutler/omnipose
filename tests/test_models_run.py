import numpy as np
import torch

from omnipose.models import run as mrun


class DummyNet:
    def __init__(self, nclasses=3, style_dim=8):
        self.nclasses = nclasses
        self.style_dim = style_dim

    def eval(self):
        return self

    def __call__(self, x):
        batch = x.shape[0]
        y = torch.zeros((batch, self.nclasses, *x.shape[-2:]), dtype=torch.float32)
        style = torch.ones((batch, self.style_dim), dtype=torch.float32)
        return y, style

    def load_model(self, *_args, **_kwargs):
        return None


class DummyModel:
    def __init__(self):
        self.torch = True
        self.net = DummyNet()
        self.nclasses = 3
        self.dim = 2
        self.unet = False
        self.batch_size = 2
        self.pretrained_model = "dummy"
        self.gpu = False
        self.device = "cpu"
        self.run_network = mrun.run_network.__get__(self, DummyModel)

    def _to_device(self, x):
        return torch.tensor(x).float()

    def _from_device(self, x):
        return x.detach().cpu().numpy()


def test_run_network_torch():
    model = DummyModel()
    x = np.zeros((2, 1, 8, 8), dtype=np.float32)
    y, style = mrun.run_network(model, x)
    assert isinstance(y, np.ndarray)
    assert y.shape == (2, 3, 8, 8)
    assert style.shape == (2, 8)


def test_run_network_non_torch():
    class NonTorchNet:
        def eval(self):
            return self

        def __call__(self, x):
            batch = x.shape[0]
            y = np.zeros((batch, 3, x.shape[-2], x.shape[-1]), dtype=np.float32)
            style = np.ones((batch, 8), dtype=np.float32)
            return y, style

    class NonTorchModel(DummyModel):
        def __init__(self):
            super().__init__()
            self.torch = False
            self.net = NonTorchNet()

        def _to_device(self, x):
            return x

        def _from_device(self, x):
            return x

    model = NonTorchModel()
    x = np.zeros((1, 1, 4, 4), dtype=np.float32)
    y, style = mrun.run_network(model, x)
    assert y.shape == (1, 3, 4, 4)
    assert style.shape == (1, 8)


def test_run_net_no_tile():
    model = DummyModel()
    img = np.zeros((8, 8, 1), dtype=np.float32)
    y, style = mrun._run_net(model, img, tile=False, augment=False, normalize=False, bsize=8)
    assert y.shape == (8, 8, 3)
    assert style.shape == (8,)


def test_run_tiled_nd_simple():
    model = DummyModel()
    model.run_network = lambda x, **_: (np.zeros((x.shape[0], 3, x.shape[-2], x.shape[-1]), dtype=np.float32),
                                        np.ones((x.shape[0], 8), dtype=np.float32))
    imgi = np.zeros((1, 16, 16), dtype=np.float32)
    y, style = mrun._run_tiled(model, imgi, augment=False, normalize=False, bsize=8, tile_overlap=0.1)
    assert y.shape == (3, 16, 16)
    assert style.shape == (8,)


def test_run_tiled_unaugment_branch(monkeypatch):
    model = DummyModel()
    model.run_network = lambda x, **_: (
        np.zeros((x.shape[0], 3, x.shape[-2], x.shape[-1]), dtype=np.float32),
        np.ones((x.shape[0], 8), dtype=np.float32),
    )
    monkeypatch.setattr(mrun.transforms, "unaugment_tiles_ND", lambda y, *_: y, raising=False)
    imgi = np.zeros((1, 16, 16), dtype=np.float32)
    y, style = mrun._run_tiled(model, imgi, augment=True, normalize=False, bsize=8, tile_overlap=0.1)
    assert y.shape == (3, 16, 16)

def test_run_tiled_4d_batch_and_augment():
    model = DummyModel()
    model.batch_size = 4
    model.run_network = lambda x, **_: (
        np.zeros((x.shape[0], 3, x.shape[-2], x.shape[-1]), dtype=np.float32),
        np.ones((x.shape[0], 8), dtype=np.float32),
    )
    imgi = np.zeros((2, 1, 8, 8), dtype=np.float32)
    y, style = mrun._run_tiled(model, imgi, augment=True, normalize=False, bsize=8, tile_overlap=0.1)
    assert y.shape == (2, 3, 8, 8)
    assert style.shape[-1] == 8


def test_run_nets_multi_model_avg():
    model = DummyModel()
    model.pretrained_model = ["a", "b"]
    counter = {"i": 0}

    def fake_run_net(_self, img, **_):
        counter["i"] += 1
        val = float(counter["i"])
        y = np.ones((8, 8, 3), dtype=np.float32) * val
        style = np.ones((8,), dtype=np.float32)
        return y, style

    model._run_net = fake_run_net.__get__(model, DummyModel)
    y, style = mrun._run_nets(model, np.zeros((8, 8, 1), dtype=np.float32), net_avg=True)
    assert np.allclose(y, 1.5)
    assert style.shape == (8,)


def test_run_nets_multi_model_gpu_progress():
    class Progress:
        def __init__(self):
            self.values = []

        def setValue(self, value):
            self.values.append(value)

    model = DummyModel()
    model.pretrained_model = ["a", "b"]
    model.gpu = True

    class ModuleNet(DummyNet):
        def load_model(self, *_args, **_kwargs):
            return None

    model.net = type("Wrapper", (), {"module": ModuleNet(), "load_model": lambda *a, **kw: None})()
    def fake_run_net(_self, img, **_):
        return np.zeros((8, 8, 3), dtype=np.float32), np.ones((8,), dtype=np.float32)

    model._run_net = fake_run_net.__get__(model, DummyModel)
    progress = Progress()
    y, style = mrun._run_nets(model, np.zeros((8, 8, 1), dtype=np.float32), net_avg=True, progress=progress)
    assert progress.values
    assert y.shape == (8, 8, 3)



def test_run_3d_basic():
    model = DummyModel()
    model.pretrained_model = "dummy"

    def fake_run_nets(_self, img, **_):
        z, y, x, _ = img.shape
        yout = np.zeros((z, y, x, model.nclasses), dtype=np.float32)
        style = np.ones((8,), dtype=np.float32)
        return yout, style

    model._run_nets = fake_run_nets.__get__(model, DummyModel)

    def identity_resize(arr, *_, **__):
        return arr

    mrun.transforms.resize_image = identity_resize
    imgs = np.zeros((2, 4, 4, 1), dtype=np.float32)
    yf, style = mrun._run_3D(model, imgs, rsz=1.0, anisotropy=None, net_avg=False, tile=False, normalize=False)
    assert yf.shape == (3, model.nclasses, 2, 4, 4)
    assert style.shape == (8,)


def test_run_3d_with_tiling():
    class TiledModel(DummyModel):
        def __init__(self):
            super().__init__()
            self._run_net = mrun._run_net.__get__(self, TiledModel)
            self._run_nets = mrun._run_nets.__get__(self, TiledModel)
            self._run_tiled = mrun._run_tiled.__get__(self, TiledModel)
            self.run_network = lambda x, **_: (
                np.zeros((x.shape[0], self.nclasses, x.shape[-2], x.shape[-1]), dtype=np.float32),
                np.ones((x.shape[0], 8), dtype=np.float32),
            )

    model = TiledModel()
    called = {"tiled": False}

    def wrapped_run_tiled(self, *args, **kwargs):
        called["tiled"] = True
        return mrun._run_tiled(self, *args, **kwargs)

    model._run_tiled = wrapped_run_tiled.__get__(model, TiledModel)

    imgs = np.zeros((2, 12, 12, 1), dtype=np.float32)
    yf, style = mrun._run_3D(model, imgs, rsz=1.0, anisotropy=None, net_avg=False, tile=True, normalize=False, bsize=8)
    assert called["tiled"] is True
    assert yf.shape == (3, model.nclasses, 2, 12, 12)
    assert style.shape[-1] == 8


def test_run_3d_rescaling_progress(monkeypatch):
    class Progress:
        def __init__(self):
            self.values = []

        def setValue(self, value):
            self.values.append(value)

    model = DummyModel()
    model._run_nets = mrun._run_nets.__get__(model, DummyModel)
    def fake_run_net(_self, img, **_):
        return np.zeros((img.shape[0], img.shape[1], img.shape[2], 3), dtype=np.float32), np.ones((8,), dtype=np.float32)

    model._run_net = fake_run_net.__get__(model, DummyModel)

    def identity_resize(arr, *_, **__):
        return arr

    monkeypatch.setattr(mrun.transforms, "resize_image", identity_resize, raising=False)
    imgs = np.zeros((2, 4, 4, 1), dtype=np.float32)
    progress = Progress()
    y, style = mrun._run_3D(
        model,
        imgs,
        rsz=0.5,
        anisotropy=2.0,
        net_avg=False,
        tile=False,
        normalize=False,
        progress=progress,
    )
    assert progress.values
    assert y.shape[1] == model.nclasses
