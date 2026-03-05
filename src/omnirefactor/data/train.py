import os
import time
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch

from torch.utils.data import BatchSampler

from ..transforms.augment import random_rotate_and_resize
from ..core.flows import masks_to_flows_batch, batch_labels


class _ShmPool:
    """Pack a list of numpy arrays (heterogeneous shapes/dtypes) into a single
    POSIX shared memory segment.

    Uses only **one** file descriptor for the entire list, regardless of how
    many arrays it holds.  With 284 training images, this reduces open-fd
    usage from 568 → 2 (one pool for data, one for labels).

    Pickle-efficient: only the shm name + offset table (tiny) is serialized.
    Spawn workers attach to the existing segment by name — zero copies, zero
    extra RAM beyond what the main process already holds.

    Lifecycle:
      Main process:  pool = _ShmPool(list_of_arrays)  # creates shm, copies once
      Worker:        pool.get(i)                       # zero-copy view
      Cleanup:       pool.close(); pool.unlink()       # release then destroy
    """

    _ALIGN = 64  # align array starts to cache-line boundaries

    def __init__(self, arrays):
        meta, offset = [], 0
        for a in arrays:
            meta.append((offset, a.shape, a.dtype.str))
            offset += a.nbytes
            offset = (offset + self._ALIGN - 1) // self._ALIGN * self._ALIGN

        total = max(offset, 1)
        self._shm = SharedMemory(create=True, size=total)
        self._name = self._shm.name
        self._owner = True

        for (byte_off, shape, dtype_str), a in zip(meta, arrays):
            dst = np.ndarray(shape, dtype=np.dtype(dtype_str),
                             buffer=self._shm.buf, offset=byte_off)
            dst[:] = a

        self._meta = meta  # list of (int, tuple, str) — all serializable

    def __getstate__(self):
        return {'name': self._name, 'meta': self._meta}

    def __setstate__(self, state):
        self._name = state['name']
        self._meta = state['meta']
        # Attach to the existing shm segment without registering it in the
        # resource_tracker — we own the lifecycle (main process unlinks).
        # Monkey-patch register to a no-op for the duration of __init__
        # so no registration happens, and no unregister is ever needed.
        import multiprocessing.resource_tracker as _rt_mod
        _orig_register = _rt_mod.register
        _rt_mod.register = lambda *a, **kw: None
        try:
            self._shm = SharedMemory(name=self._name, create=False)
        finally:
            _rt_mod.register = _orig_register
        self._owner = False

    def get(self, i) -> np.ndarray:
        """Return a zero-copy numpy view of the i-th array."""
        byte_off, shape, dtype_str = self._meta[i]
        return np.ndarray(shape, dtype=np.dtype(dtype_str),
                          buffer=self._shm.buf, offset=byte_off)

    def close(self):
        try:
            self._shm.close()
        except Exception:
            pass

    def unlink(self):
        if getattr(self, '_owner', False):
            try:
                self._shm.unlink()
            except Exception:
                pass


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

        if not hasattr(self, 'defer_flows'):
            self.defer_flows = False

        # Shared-memory pools: single fd per role, zero extra RAM.
        if not hasattr(self, 'data_pool'):
            self.data_pool = None
        if not hasattr(self, 'label_pool'):
            self.label_pool = None

        # True lazy loading: workers imread + normalize from original files.
        # norm_params is a list of [(lo, hi), ...] per channel per image.
        if not hasattr(self, 'image_paths'):
            self.image_paths = None
        if not hasattr(self, 'label_paths'):
            self.label_paths = None
        if not hasattr(self, 'norm_params'):
            self.norm_params = None

    # ------------------------------------------------------------------
    # Pickle protocol: spawn workers only get paths + tiny norm params,
    # never the full arrays.
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.image_paths is not None or self.data_pool is not None:
            state['data'] = None
            state['labels'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_batch_arrays(self, inds):
        """Return (images, labels) for the given indices.

        Priority:
          1. image_paths + norm_params — imread from original files, zero main-
             process RAM, preprocessing on demand in each worker.
          2. _ShmPool — zero-copy shm, 2 fds total, preprocessed arrays shared.
          3. In-memory arrays — num_workers=0 / legacy path.
        """
        if self.image_paths is not None:
            from ..io.imio import imread
            from ..transforms.shape import apply_norm_params
            from ..transforms.axes import move_min_dim
            images, labels = [], []
            for i in inds:
                img = imread(self.image_paths[i]).astype(np.float32)
                # Restore the channel layout that compute_norm_params assumed.
                # Must use the same heuristic as compute_norm_params.
                _ch_ax = getattr(self, '_channel_axis', None)
                if _ch_ax is not None:
                    img = np.moveaxis(img, _ch_ax, 0)
                elif img.ndim > self.dim:
                    # channel_axis=None: move smallest dim to front (mirrors compute_norm_params)
                    img = move_min_dim(img)       # moves min-dim to last
                    img = np.moveaxis(img, -1, 0) # then bring to front
                if img.ndim == self.dim:
                    img = img[np.newaxis]
                if self.norm_params is not None:
                    img = apply_norm_params(img, self.norm_params[i])
                images.append(img)
                lbl = imread(self.label_paths[i])
                # format_labels (sequential integers) if no link file
                if self.links[i] is None:
                    import ncolor
                    lbl = ncolor.format_labels(lbl)
                labels.append(lbl)
            return images, labels
        elif self.data_pool is not None:
            images = [self.data_pool.get(i) for i in inds]
            labels = [self.label_pool.get(i) for i in inds]
        else:
            images = [self.data[i] for i in inds]
            labels = [self.labels[i] for i in inds]
        return images, labels

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

    def collate_fn_deferred(self, worker_data):
        """Collate for deferred flows mode - just pass through the single batch."""
        return worker_data[0]

    def compute_flows_gpu(self, imgi_np, masks_np, links, device):
        """Compute flows on GPU from raw augmented data (main process only)."""
        with torch.no_grad():
            out = masks_to_flows_batch(masks_np, links,
                                       device=device,
                                       omni=self.omni,
                                       dim=self.dim,
                                       affinity_field=self.affinity_field)
            X = out[:-4]
            slices = out[-4]
            masks, bd, T, mu = [torch.stack([x[(Ellipsis,) + slc] for slc in slices]) for x in X]

            lbl = batch_labels(masks, bd, T, mu, self.tyx,
                               dim=self.dim, nclasses=self.nclasses, device=device)

            imgi = torch.tensor(imgi_np, device=device, dtype=torch.float32)
        return imgi, lbl

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

        batch_images, batch_labels_raw = self._get_batch_arrays(inds)

        # Use random_rotate_and_resize on full batch (matches omnipose manual path)
        imgi, labels, scale = random_rotate_and_resize(
            X=batch_images,
            Y=batch_labels_raw,
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

        # Deferred mode: return raw augmented data for GPU flow computation in main process
        if self.defer_flows:
            return imgi, labels, links, inds

        with torch.no_grad():
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
