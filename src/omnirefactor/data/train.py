import os
import time
import multiprocessing as mp

import numpy as np
import torch

from torch.utils.data import BatchSampler

from ..transforms.augment import random_rotate_and_resize
from ..core.flows import masks_to_flows_batch, batch_labels


class train_set(torch.utils.data.Dataset):
    def __init__(self, data, labels, links,
                 timing=False, **kwargs):
        self.__dict__.update(kwargs)

        self.data = data
        self.labels = labels
        self.links = links

        self.timing = timing

        if not hasattr(self, 'augment'):
            self.augment = True

        if self.tyx is None:
            n = 16
            kernel_size = 2
            base = kernel_size
            L = max(round(224 / (base**4)), 1) * (base**4)
            self.tyx = (L,) * self.dim if self.dim == 2 else (8 * n,) + (8 * n,) * (self.dim - 1)

        self.scale_range = max(0, min(2, float(self.scale_range)))

        self.do_flip = True
        self.dist_bg = 5
        self.normalize = False
        # gamma_range must match omnipose MANUAL path default [.5, 4] for parity
        # (the manual path doesn't pass gamma_range so uses the default from random_rotate_and_resize)
        self.gamma_range = [.5, 4]
        self.nimg = len(data)
        do_rescale = getattr(self, "do_rescale", getattr(self, "rescale", True))
        self.do_rescale = bool(do_rescale)
        self.rescale_factor = self.diam_train / self.diam_mean if self.do_rescale else np.ones(self.nimg, np.float32)

        self.v1 = [0] * (self.dim - 1) + [1]
        self.v2 = [0] * (self.dim - 2) + [1, 0]

    def __iter__(self):
        worker_info = mp.get_worker_info()

        if worker_info is None:
            start = 0
            end = len(self)
        else:
            total_samples = len(self)
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(total_samples / num_workers)
            leftover = total_samples % num_workers
            start = worker_id * per_worker
            end = start + per_worker

            if worker_id == num_workers - 1:
                end += leftover

        for index in range(start, end):
            yield self[index]

    def collate_fn(self, worker_data):
        worker_imgs, worker_labels, worker_inds = zip(*worker_data)

        batch_imgs = torch.cat(worker_imgs, dim=0)
        batch_labels = torch.cat(worker_labels, dim=0)
        batch_inds = [item for sublist in worker_inds for item in sublist]

        return batch_imgs, batch_labels, batch_inds

    def worker_init_fn(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    def __len__(self):
        return self.nimg

    def __getitems__(self, inds):
        """
        Batched getter that PyTorch DataLoader uses when available.

        This ensures all indices in a batch are processed together through
        masks_to_flows_batch, matching omnipose's manual batching behavior
        where flows are computed on the concatenated mask batch at once.

        Without this method, DataLoader calls __getitem__ per-index separately,
        then collates - which computes flows per-image instead of per-batch.
        """
        return [self.__getitem__(inds)]

    def __getitem__(self, inds):
        if isinstance(inds, int):
            inds = [inds]

        if self.timing:
            tic = time.time()

        links = [self.links[idx] for idx in inds]
        rsc = np.array([self.rescale_factor[idx] for idx in inds]) if self.do_rescale else None

        # Use random_rotate_and_resize on full batch (matches omnipose manual path)
        imgi, labels, scale = random_rotate_and_resize(
            X=[self.data[idx] for idx in inds],
            Y=[self.labels[idx] for idx in inds],
            scale_range=self.scale_range,
            gamma_range=self.gamma_range,
            tyx=self.tyx,
            do_flip=self.do_flip,
            rescale_factor=rsc,
            inds=inds,
            nchan=self.nchan,
            allow_blank_masks=self.allow_blank_masks
        )
        if self.timing:
            toc = time.time()
            print('image augmentation time: {:.2f}'.format(toc - tic))
            tic = toc

        out = masks_to_flows_batch(labels, links,
                                   device=self.device,
                                   omni=self.omni,
                                   dim=self.dim,
                                   affinity_field=self.affinity_field
                                   )
        if self.timing:
            toc = time.time()
            print('flow time: {:.2f}'.format(toc - tic))
            tic = toc

        X = out[:-4]
        slices = out[-4]
        masks, bd, T, mu = [torch.stack([x[(Ellipsis,) + slc] for slc in slices]) for x in X]

        lbl = batch_labels(masks,
                           bd,
                           T,
                           mu,
                           self.tyx,
                           dim=self.dim,
                           nclasses=self.nclasses,
                           device=self.device
                           )
        if self.timing:
            toc = time.time()
            print('batching time: {:.2f}'.format(toc - tic))
            tic = toc

        imgi = torch.tensor(imgi, device=self.device, dtype=torch.float32)

        if self.timing:
            print('inds', len(inds))

        return imgi, lbl, inds


class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self._preload()

    def _preload(self):
        try:
            batch = next(self.loader)
            self.next_data = batch[0]
            self.next_labels = batch[1]
            if len(batch) > 2:
                self.next_inds = batch[2]
            else:
                self.next_inds = None
        except StopIteration:
            self.next_data = None
            self.next_labels = None
            self.next_inds = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data = self.next_data.cuda(self.device, non_blocking=True)
            self.next_labels = self.next_labels.cuda(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        labels = self.next_labels
        inds = self.next_inds
        self._preload()
        return data, labels, inds


class CyclingRandomBatchSampler(BatchSampler):
    """
    Infinite stream of shuffled, non-overlapping batch indices.

    Pre-generates all indices upfront using np.random.seed(0) to match
    the omnipose manual batching path when num_workers=0. This ensures
    identical index sequences for reproducibility.

    The batching follows omnipose's structure exactly:
    - Generate indices for n_epochs * nimg_per_epoch
    - For each epoch, slice nimg_per_epoch indices
    - Within each epoch, yield batches of size batch_size
    """

    def __init__(self, data_source, batch_size, n_epochs=None, nimg_per_epoch=None, generator=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.N = len(data_source)
        self.n_epochs = n_epochs or 500  # default from training
        self.nimg_per_epoch = nimg_per_epoch if nimg_per_epoch is not None else self.N

        # Pre-generate ALL indices upfront (matches omnipose manual batching)
        # This uses np.random.seed(0) like the manual path for reproducibility
        np.random.seed(0)
        inds_all = np.zeros((0,), 'int32')
        while len(inds_all) < self.n_epochs * self.nimg_per_epoch:
            rperm = np.random.permutation(self.N)
            inds_all = np.hstack((inds_all, rperm))
        self._inds_all = inds_all
        self.epoch = 0

    def __iter__(self):
        # Match omnipose manual path batch generation:
        # for epoch in range(n_epochs):
        #     rperm = inds_all[epoch*nimg_per_epoch:(epoch+1)*nimg_per_epoch]
        #     for ibatch in range(0, nimg_per_epoch, batch_size):
        #         inds = rperm[ibatch:ibatch+batch_size]
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
