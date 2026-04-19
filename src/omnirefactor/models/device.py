from __future__ import annotations

from ocdkit.utils.gpu import to_device, from_device


def _to_device(self, x):
    return to_device(x, self.device)


def _from_device(self, X):
    return from_device(X)
