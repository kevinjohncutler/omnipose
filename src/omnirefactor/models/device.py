from .imports import *


def _to_device(self, x):
    if isinstance(x, torch.Tensor):
        if self.device != x.device:
            return x.to(self.device)
        return x
    return torch.tensor(x, device=self.device, dtype=torch.float32)


def _from_device(self, X):
    x = X.detach().cpu().numpy()
    return x
