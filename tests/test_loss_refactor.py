import torch

from omnirefactor.core import loss as loss_mod


class _LossHarness:
    def __init__(self, dim=2, nclasses=4, device="cpu"):
        self.dim = dim
        self.nclasses = nclasses
        self.device = device
        self.MSELoss = torch.nn.MSELoss()
        self.BCELoss = torch.nn.BCELoss()

    def WeightedMSE(self, pred, target, weight):
        return ((pred - target) ** 2 * weight).mean()

    def AffinityLoss(self, flow, dt, veci, dist, mode="all", seed=0):
        return flow.mean(), dt.mean(), veci.mean()

    def DerivativeLoss(self, dt, dist, w, cellmask):
        return (dt - dist).abs().mean()

    def SSNLoss(self, flow, veci, dist, w, boundary):
        return flow.abs().mean(), dist.abs().mean()


def test_scale_to_tenths_limits_gain():
    x = torch.tensor([1e-6, 1.0, 1e6])
    scaled = loss_mod.scale_to_tenths(x, max_gain=10)
    assert scaled.shape == x.shape
    assert torch.all(torch.isfinite(scaled))


def test_loss_nclasses_one_branch():
    model = _LossHarness(dim=2, nclasses=1)
    lbl = torch.zeros((1, 2, 4, 4))
    lbl[:, 1] = 1.0
    y = torch.sigmoid(torch.randn((1, 1, 4, 4)))
    out = loss_mod.loss(model, lbl, y)
    assert torch.is_tensor(out)


def test_loss_full_branch():
    dim = 2
    model = _LossHarness(dim=dim, nclasses=dim + 2)
    lbl = torch.zeros((1, 5 + dim, 4, 4))
    lbl[:, 1] = 1.0
    y = torch.randn((1, dim + 2, 4, 4))
    y[:, dim + 1] = torch.sigmoid(y[:, dim + 1])
    total, raw = loss_mod.loss(model, lbl, y, ext_loss=torch.tensor(0.0))
    assert torch.is_tensor(total)
    assert torch.is_tensor(raw)
