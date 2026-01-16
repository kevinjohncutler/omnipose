from .imports import *


def _to_device(self, x):
    if isinstance(x, torch.Tensor):
        if self.device != x.device:
            return x.to(self.device)
        return x
    if self.torch:
        return torch.tensor(x, device=self.device, dtype=torch.float32)
    return np.array(x.astype(np.float32), ctx=self.device)


def _from_device(self, X):
    if self.torch:
        x = X.detach().cpu().numpy()
        empty_cache()
    else:
        x = X.asnumpy()
    return x
