"""Experimental loss functions — explored but not integrated into training."""

import numpy as np
import torch

from torchvf.losses import ivp_loss

from ocdkit.array import torch_norm


class EulerLoss(ivp_loss.IVPLoss):
    def __init__(self, device, dim):
        super().__init__(
            dx=np.sqrt(dim) / 5,
            n_steps=dim,
            device=device,
            mode='nearest_batched',
        )


class SineSquaredLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, w):
        eps = 1e-12
        magX = torch_norm(x, dim=1)
        magY = torch_norm(y, dim=1)
        denom = torch.multiply(magX, magY)
        dot = torch.sum(torch.stack([x[:, k] * y[:, k] for k in range(x.shape[1])], dim=1), dim=1)
        cos = torch.where(denom > eps, dot / (denom + eps), 1)
        return torch.mean((1 - cos ** 2) * w)


class NormLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, Y, w):
        return torch.mean(torch.square(torch_norm(y, dim=1) - torch_norm(Y, dim=1)) * w) / 25


class SSL_Norm_MSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, w, dist):
        eps = 1e-12
        magX = torch_norm(x, dim=1)
        magY = torch_norm(y, dim=1)
        denom = torch.multiply(magX, magY)
        dot = torch.sum(torch.stack([x[:, k] * y[:, k] for k in range(x.shape[1])], dim=1), dim=1)
        cos = torch.where(denom > eps, dot / (denom + eps), 1)
        err = torch.square((x - y) / 5.).sum(dim=1)

        w3 = torch.clip(dist, 0.5, 2) / 2
        w2 = 1. - w3
        cos_weighted = torch.sum((1 - cos ** 2) * w2) / torch.sum(w2)
        mse_weighted = torch.sum(err * w3) / torch.sum(w3)

        cos_weighted = torch.mean((1 - cos ** 2) * err)
        mse_weighted = 0

        cos_weighted = torch.mean(err / (cos ** 2 + 1))

        return mse_weighted, cos_weighted, torch.mean(torch.square(magX - magY) * w) / 25


class CorrelationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        vx = x - torch.mean(x, dim=0)
        vy = y - torch.mean(y, dim=0)
        num = torch.sum(vx * vy, dim=0)
        denom = torch.sum(vx ** 2, dim=0) * torch.sum(vy ** 2, dim=0)
        cost = torch.where(denom > 0, num / torch.sqrt(denom), -1)
        return -torch.mean(cost)


class TruncatedMSELoss(torch.nn.Module):
    def __init__(self, t=5.0):
        super().__init__()
        self.t = t

    def forward(self, pred, target):
        SE = torch.square(pred - target)
        loss = torch.where(SE < self.t, SE, self.t)
        return loss.mean()


class MeanAdjustedMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        mean_error = torch.mean(pred - target)
        adjusted_pred = pred - mean_error
        loss = torch.mean((adjusted_pred - target) ** 2)
        return loss


class GradNormLoss(torch.nn.Module):
    def __init__(self, num_losses, device, alpha=0.12):
        super().__init__()
        self.alpha = alpha
        self.loss_weights = torch.nn.Parameter(torch.ones(num_losses, device=device, dtype=torch.float32))

    def forward(self, losses, shared_params):
        if len(losses) == 0:
            raise RuntimeError("All loss terms are zero or don't require gradients.")

        weighted_losses = self.loss_weights[:len(losses)] * torch.stack(losses)
        total_loss = weighted_losses.sum()
        grads = torch.autograd.grad(total_loss, shared_params, retain_graph=True, create_graph=False)

        grad_norms = []
        for loss, grad in zip(losses, grads):
            if grad is not None:
                grad_norms.append(torch.norm(grad))

        grad_norms = torch.stack(grad_norms)
        mean_grad = grad_norms.mean().detach()
        target_scales = (grad_norms / mean_grad) ** self.alpha
        target_scales = target_scales.detach()
        weight_loss = torch.sum(torch.abs(self.loss_weights[:len(losses)] * grad_norms - target_scales * mean_grad))
        weight_grads = torch.autograd.grad(weight_loss, self.loss_weights[:len(losses)], allow_unused=True)[0]
        self.loss_weights.grad = weight_grads
        return total_loss
