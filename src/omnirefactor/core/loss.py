import torch
import torch.fft as fft
from .fields import divergence_torch


def scale_to_tenths(x, max_gain=10):
    eps = 1e-12
    sf = 10 ** (-torch.floor(torch.log10(torch.abs(x) + eps)) - 1)
    sf = torch.clamp(sf, 1 / max_gain, max_gain)
    return x * sf


def loss(self, lbl, y, ext_loss=0):
    """Loss function for Omnipose."""

    cellmask = lbl[:, 1] > 0
    if self.nclasses == 1:
        cm = cellmask.float()
        flow_mse = self.MSELoss(y[:, 0], cm)
        BCE = self.BCELoss(y[:, 0], cm)
        return flow_mse + BCE / 20

    else:
        veci = lbl[:, -self.dim:]
        dist = lbl[:, 3]
        boundary = lbl[:, 2]
        w = lbl[:, 4].detach()
        wt = torch.stack([w] * self.dim, dim=1).detach()

        flow = y[:, :self.dim]
        dt = y[:, self.dim]

        if self.nclasses == (self.dim + 2):
            bd = y[:, self.dim + 1]
            bd_loss = self.BCELoss(bd, boundary)
        else:
            bd_loss = torch.tensor(0, device=self.device)

        dist_loss = self.WeightedMSE(dt, dist, w)

        lossA, lossE, lossB = self.AffinityLoss(flow, dt, veci, dist,
                                                mode='all',
                                                seed=0,
                                                )
        div = divergence_torch(veci)
        div_flow = divergence_torch(flow)

        bd_loss = self.DerivativeLoss(dt.unsqueeze(1), dist.unsqueeze(1), w.unsqueeze(1), cellmask.unsqueeze(1))
        SSL, norm_loss = self.SSNLoss(flow, veci, dist, w, boundary)

        lossDC = self.MSELoss(div, div_flow)

        flow_mse = self.WeightedMSE(flow, veci, wt)

        losses = [flow_mse, SSL, bd_loss, norm_loss, dist_loss,
                  lossA, lossE, lossB,
                  lossDC,
                  ]
        raw_loss = sum(losses).detach() / len(losses)

        losses += ext_loss if isinstance(ext_loss, list) else [ext_loss]

        losses = [scale_to_tenths(l, max_gain=1e12) for l in losses]
        return sum(losses), raw_loss


def bg_flow_corr_penalty(flow, cellmask, eps=1e-6):  # pragma: no cover
    bg = flow * (~cellmask).unsqueeze(1)
    B, C, H, W = bg.shape
    f = bg.flatten(2)
    mu = f.mean(-1, keepdim=True)
    f -= mu

    var = (f.pow(2).mean(-1, keepdim=True) + eps)
    std = var.sqrt()
    f_norm = f / std

    corr = f_norm @ f_norm.transpose(-1, -2) / f.shape[-1]
    off_diag = corr - torch.diag_embed(torch.diagonal(corr, dim1=-2, dim2=-1))
    return off_diag.pow(2).mean()


def bg_flow_spectral_penalty(flow_pred, cellmask,
                             fc=0.10,
                             dim=2):  # pragma: no cover
    """
    flow_pred : (B, dim, H, W)  predicted flow
    cellmask  : (B, H,  W)      True on cell pixels
    fc        : relative cutoff; 0.10 -> remove first 10 % of freq band
    Returns scalar penalty.
    """

    bgmask = (~cellmask).float()
    B, C, H, W = flow_pred.shape
    eps = 1e-6

    flow_bg = flow_pred * bgmask.unsqueeze(1)

    spec = fft.rfftn(flow_bg, dim=(-2, -1), norm='forward')
    power = spec.real.pow(2) + spec.imag.pow(2)

    fy = torch.fft.fftfreq(H, d=1. / H, device=flow_pred.device)
    fx = torch.fft.rfftfreq(W, d=1. / W, device=flow_pred.device)
    fy2, fx2 = torch.meshgrid(fy, fx, indexing='ij')
    f_radius = torch.sqrt(fy2 ** 2 + fx2 ** 2)
    lowpass = (f_radius < fc).float()

    low_energy = (power * lowpass).sum(dim=(-2, -1))
    bg_pix = bgmask.sum(dim=(-2, -1)).clamp_min(1.0).unsqueeze(1)
    penalty = (low_energy / bg_pix).mean()
    return penalty


def bg_flow_spec_penalty(flow, cellmask, fc=0.1, eps=1e-6):  # pragma: no cover
    bg = flow * (~cellmask).unsqueeze(1)
    power = torch.fft.rfftn(bg, dim=(-2, -1), norm='forward').abs().pow(2)

    fy = torch.fft.fftfreq(bg.size(-2), device=bg.device)
    fx = torch.fft.rfftfreq(bg.size(-1), device=bg.device)
    fy2, fx2 = torch.meshgrid(fy, fx, indexing='ij')
    rad = torch.sqrt(fy2 ** 2 + fx2 ** 2)

    lp_mask = (rad < fc).float()
    low = (power * lp_mask).sum(dim=(-2, -1))
    total = power.sum(dim=(-2, -1)) + eps
    frac = low / total
    return frac.mean()
