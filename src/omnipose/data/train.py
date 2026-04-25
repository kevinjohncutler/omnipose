from __future__ import annotations
"""Training dataset with random crop/warp augmentation."""

import os
import time
import multiprocessing as mp

import numpy as np
import torch

from .imports import *
from .sampler import CyclingRandomBatchSampler
from .shm import ShmPool

# Backward compat: scripts import _ShmPool from here
_ShmPool = ShmPool


class train_set(torch.utils.data.Dataset):
    """Training dataset that returns augmented images + raw masks.

    Flow labels are NOT computed here — that happens in the training loop
    on GPU via :meth:`compute_flows_gpu`, which is ~10x faster on CUDA
    than CPU.

    Parameters
    ----------
    data : list of ndarray
        Training images, each ``(C, *spatial)``.
    labels : list of ndarray
        Mask labels for each image.
    links : list
        Per-image label pair links for multi-label objects.
    timing : bool
        Print per-step timing info.
    **kwargs
        All remaining keyword arguments are set as attributes directly.
        Expected keys include ``dim``, ``nchan``, ``nclasses``, ``tyx``,
        ``scale_range``, ``omni``, ``affinity_field``, ``device``,
        ``allow_blank_masks``, ``do_rescale``, ``diam_train``, ``diam_mean``.
    """

    def __init__(self, data, labels, links, timing=False, **kwargs):
        self.__dict__.update(kwargs)

        self.data = data
        self.labels = labels
        self.links = links
        self.timing = timing
        self.nimg = len(data)

        # Defaults for optional attributes
        if not hasattr(self, 'augment'):
            self.augment = True

        # Compute tyx from dim if not provided
        if self.tyx is None:
            base = 2  # kernel_size
            L = max(round(224 / (base ** 4)), 1) * (base ** 4)
            self.tyx = (L,) * self.dim if self.dim == 2 else (8 * 16,) + (8 * 16,) * (self.dim - 1)

        self.scale_range = max(0, min(2, float(self.scale_range)))
        self.do_flip = True
        self.gamma_range = [0.75, 2.5]

        # Rescale factors
        do_rescale = getattr(self, 'do_rescale', getattr(self, 'rescale', True))
        self.do_rescale = bool(do_rescale)
        self.rescale_factor = (
            self.diam_train / self.diam_mean if self.do_rescale
            else np.ones(self.nimg, np.float32)
        )

        # Shared-memory pools (set externally when using num_workers > 0)
        if not hasattr(self, 'data_pool'):
            self.data_pool = None
        if not hasattr(self, 'label_pool'):
            self.label_pool = None

        # Lazy file-loading paths (set externally for disk-backed datasets)
        if not hasattr(self, 'image_paths'):
            self.image_paths = None
        if not hasattr(self, 'label_paths'):
            self.label_paths = None
        if not hasattr(self, 'norm_params'):
            self.norm_params = None

    # ------------------------------------------------------------------
    # Pickle: spawn workers only get paths/shm, never full arrays
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.image_paths is not None or self.data_pool is not None:
            state['data'] = None
            state['labels'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _get_batch_arrays(self, inds):
        """Return ``(images, labels)`` for the given indices.

        Priority: image_paths (lazy) > ShmPool (zero-copy) > in-memory.
        """
        if self.image_paths is not None:
            from .norm import apply_norm_params
            images, labels = [], []
            for i in inds:
                img = imread(self.image_paths[i]).astype(np.float32)
                _ch_ax = getattr(self, '_channel_axis', None)
                if _ch_ax is not None:
                    img = np.moveaxis(img, _ch_ax, 0)
                elif img.ndim > self.dim:
                    img = move_min_dim(img)
                    img = np.moveaxis(img, -1, 0)
                if img.ndim == self.dim:
                    img = img[np.newaxis]
                if self.norm_params is not None:
                    img = apply_norm_params(img, self.norm_params[i])
                images.append(img)
                lbl = imread(self.label_paths[i])
                if self.links[i] is None:
                    import ncolor
                    lbl = ncolor.format_labels(lbl)
                labels.append(lbl)
            return images, labels
        elif self.data_pool is not None:
            return ([self.data_pool.get(i) for i in inds],
                    [self.label_pool.get(i) for i in inds])
        else:
            return ([self.data[i] for i in inds],
                    [self.labels[i] for i in inds])

    # ------------------------------------------------------------------
    # DataLoader interface
    # ------------------------------------------------------------------

    def __iter__(self):
        worker_info = mp.get_worker_info()
        if worker_info is None:
            start, end = 0, len(self)
        else:
            total = len(self)
            per_worker = total // worker_info.num_workers
            leftover = total % worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker
            if worker_info.id == worker_info.num_workers - 1:
                end += leftover
        for index in range(start, end):
            yield self[index]

    def collate_fn(self, worker_data):
        """Pass through the single batch produced by __getitems__."""
        return worker_data[0]

    def worker_init_fn(self, worker_id):
        """Set thread counts and RNG seed for worker reproducibility."""
        np.random.seed(torch.initial_seed() % 2 ** 32)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[var] = '1'

    def __len__(self):
        return self.nimg

    def __getitems__(self, inds):
        """Batched getter — ensures all indices are processed together
        through a single augmentation call."""
        return [self.__getitem__(inds)]

    def __getitem__(self, inds):
        """Return augmented images + raw masks (no flow computation).

        Flow labels are computed later in the training loop on GPU via
        ``masks_to_flows_batch``, which is ~10x faster on CUDA than CPU.

        Returns
        -------
        imgi : Tensor or ndarray
            Augmented images ``(B, C, *tyx)``. Tensor on GPU if main
            process, numpy if in a DataLoader worker.
        labels : ndarray
            Augmented raw masks ``(B, *tyx)``. Always numpy.
        links : list
            Per-image label links.
        inds : list
            Original sample indices.
        """
        if isinstance(inds, int):
            inds = [inds]

        if self.timing:
            tic = time.time()

        links = [self.links[idx] for idx in inds]
        rsc = np.array([self.rescale_factor[idx] for idx in inds]) if self.do_rescale else None

        batch_images, batch_labels_raw = self._get_batch_arrays(inds)

        # Workers must not init CUDA/MPS — use CPU augmentation in workers,
        # GPU grid_sample in the main process only.
        in_worker = torch.utils.data.get_worker_info() is not None
        aug_device = None if in_worker else self.device

        imgi, labels, scale, _ = random_rotate_and_resize(
            X=batch_images,
            Y=batch_labels_raw,
            scale_range=self.scale_range,
            gamma_range=self.gamma_range,
            tyx=self.tyx,
            do_flip=self.do_flip,
            rescale_factor=rsc,
            inds=inds,
            nchan=self.nchan,
            allow_blank_masks=self.allow_blank_masks,
            device=aug_device,
        )
        if self.timing:
            print(f'augmentation: {time.time() - tic:.2f}s')

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        return imgi, labels, links, inds

    # ------------------------------------------------------------------
    # Flow computation (called from training loop, not from workers)
    # ------------------------------------------------------------------

    def compute_flows_gpu(self, imgi, masks_np, links, device):
        """Compute flow labels on GPU from raw augmented masks.

        Called from the training loop (main process only), NOT from
        DataLoader workers.

        Returns ``(imgi_gpu, lbl)`` — both on *device*.
        """
        with torch.no_grad():
            out = masks_to_flows_batch(
                masks_np, links,
                device=device,
                omni=self.omni,
                dim=self.dim,
                affinity_field=self.affinity_field,
            )
            X = out[:-4]
            slices = out[-4]
            masks, bd, T, mu = [
                torch.stack([x[(Ellipsis,) + slc] for slc in slices])
                for x in X
            ]
            lbl = batch_labels(masks, bd, T, mu, self.tyx,
                               dim=self.dim, nclasses=self.nclasses,
                               device=device)

            if isinstance(imgi, torch.Tensor):
                imgi = imgi.to(device, non_blocking=True)
            else:
                imgi = torch.tensor(imgi, device=device, dtype=torch.float32)
        return imgi, lbl
