"""Batch samplers for training."""

import numpy as np
from torch.utils.data import BatchSampler


class CyclingRandomBatchSampler(BatchSampler):
    """Infinite stream of shuffled, non-overlapping batch indices.

    Pre-generates all indices upfront using ``np.random.seed(0)`` to match
    the omnipose manual batching path. This ensures identical index
    sequences for reproducibility.

    The batching follows omnipose's structure:
    - Generate indices for ``n_epochs * nimg_per_epoch``
    - For each epoch, slice ``nimg_per_epoch`` indices
    - Within each epoch, yield batches of ``batch_size``
    """

    def __init__(self, data_source, batch_size, n_epochs=None, nimg_per_epoch=None, generator=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.N = len(data_source)
        self.n_epochs = n_epochs or 500
        self.nimg_per_epoch = nimg_per_epoch if nimg_per_epoch is not None else self.N

        np.random.seed(0)
        inds_all = np.zeros((0,), 'int32')
        while len(inds_all) < self.n_epochs * self.nimg_per_epoch:
            rperm = np.random.permutation(self.N)
            inds_all = np.hstack((inds_all, rperm))
        self._inds_all = inds_all
        self.epoch = 0

    def __iter__(self):
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            start = epoch * self.nimg_per_epoch
            end = (epoch + 1) * self.nimg_per_epoch
            rperm = self._inds_all[start:end]
            for ibatch in range(0, self.nimg_per_epoch, self.batch_size):
                batch_end = min(ibatch + self.batch_size, self.nimg_per_epoch)
                yield rperm[ibatch:batch_end].tolist()

    def __len__(self):
        """Number of batches per epoch."""
        return (self.nimg_per_epoch + self.batch_size - 1) // self.batch_size
