"""
Base dataset class shared between train_set and eval_set.

This provides common functionality for:
- Data storage and access
- Multi-worker support with proper seeding
- Collation of batches
- Normalization configuration
"""
import contextlib
import os
import multiprocessing as mp

import numpy as np
import torch


class base_set(torch.utils.data.Dataset):
    """
    Base class for Omnipose datasets.

    Handles common functionality:
    - Data storage (images as list or from files)
    - Multi-worker iteration with proper splitting
    - Collation for DataLoader
    - Worker initialization (thread settings, seeding)

    Subclasses implement:
    - __getitem__: How to retrieve/transform individual samples
    - Any mode-specific batching logic
    """

    def __init__(self, data, dim=2, nchan=1, normalize=False, invert=False,
                 channel_axis=None, device=None, **kwargs):
        """
        Parameters
        ----------
        data : list of arrays, array, or list of file paths
            Input images. Can be:
            - List of numpy arrays (varying shapes OK)
            - Single numpy array (uniform stack)
            - List of file paths to load lazily
        dim : int
            Spatial dimensionality (2 or 3)
        nchan : int
            Number of input channels expected by model
        normalize : bool
            Whether to apply percentile normalization
        invert : bool
            Whether to invert intensities
        channel_axis : int or None
            Axis containing channels, or None to auto-detect
        device : torch.device or None
            Device for tensor operations
        **kwargs : dict
            Additional attributes stored on instance
        """
        self.__dict__.update(kwargs)

        # Core attributes
        self.dim = dim
        self.nchan = nchan
        self.normalize = normalize
        self.invert = invert
        self.channel_axis = channel_axis
        self.device = device if device is not None else torch.device('cpu')

        # Store data - could be arrays or file paths
        self._init_data(data)

    def _init_data(self, data):
        """Initialize data storage. Override for lazy loading."""
        if isinstance(data, np.ndarray):
            # Single array - convert to list
            self.data = [data[i] for i in range(len(data))]
        elif isinstance(data, list):
            self.data = data
        else:
            raise TypeError(f"data must be list or array, got {type(data)}")

        self._nimg = len(self.data)

    def __len__(self):
        return self._nimg

    @property
    def n_images(self):
        """Number of images in the dataset."""
        return self._nimg

    def __iter__(self):
        """
        Iterate over indices, supporting multi-worker splitting.

        When used with DataLoader num_workers > 0, each worker
        iterates over a disjoint subset of indices.
        """
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process loading
            start, end = 0, len(self)
        else:
            # Multi-process: split indices across workers
            total = len(self)
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = total // num_workers
            leftover = total % num_workers

            start = worker_id * per_worker
            end = start + per_worker
            if worker_id == num_workers - 1:
                end += leftover

        for idx in range(start, end):
            yield self[idx]

    def collate_fn(self, worker_data):
        """
        Collate function for DataLoader.

        Default implementation stacks tensors along batch dimension.
        Override in subclasses for custom collation (e.g., padding).

        Parameters
        ----------
        worker_data : list of tuples
            Each tuple is (images, labels, indices) or (images, indices)

        Returns
        -------
        tuple
            Collated batch tensors
        """
        # Unzip the worker outputs
        items = list(zip(*worker_data))

        collated = []
        for item_list in items:
            if isinstance(item_list[0], torch.Tensor):
                # Try to stack tensors
                try:
                    collated.append(torch.cat(item_list, dim=0))
                except RuntimeError:
                    # Different shapes - return as list
                    collated.append(list(item_list))
            elif isinstance(item_list[0], (list, tuple)):
                # Flatten nested lists
                collated.append([x for sublist in item_list for x in sublist])
            else:
                # Keep as list
                collated.append(list(item_list))

        return tuple(collated)

    @staticmethod
    def worker_init_fn(worker_id):
        """
        Initialize worker process.

        Sets random seeds and thread counts for reproducibility
        and to avoid oversubscription.
        """
        # Seed numpy from torch's seed (set by DataLoader)
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

        # Limit threads to avoid oversubscription with multiple workers
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    def get_image(self, idx):
        """
        Get a single image by index.

        Override for lazy loading from files.

        Parameters
        ----------
        idx : int
            Image index

        Returns
        -------
        np.ndarray
            Image array
        """
        return self.data[idx]

    def __getitem__(self, idx):
        """
        Get item(s) by index.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement __getitem__")


class DataPrefetcher:
    """
    Async GPU data prefetcher.

    Overlaps data transfer with computation by loading the next
    batch to GPU while the current batch is being processed.
    Uses CUDA streams when available; otherwise uses non-blocking transfers.
    """

    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self._stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self._preload()

    def _preload(self):
        try:
            batch = next(self.loader)
            self.next_data = batch[0]
            self.next_labels = batch[1] if len(batch) > 1 else None
            self.next_extra = batch[2:] if len(batch) > 2 else ()
        except StopIteration:
            self.next_data = None
            self.next_labels = None
            self.next_extra = ()
            return

        ctx = torch.cuda.stream(self._stream) if self._stream is not None else contextlib.nullcontext()
        with ctx:
            self.next_data = self.next_data.to(self.device, non_blocking=True)
            if self.next_labels is not None:
                self.next_labels = self.next_labels.to(self.device, non_blocking=True)

    def next(self):
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)
        data = self.next_data
        labels = self.next_labels
        extra = self.next_extra
        self._preload()
        if labels is not None:
            return (data, labels) + extra
        return (data,) + extra

    def __iter__(self):
        return self

    def __next__(self):
        result = self.next()
        if result[0] is None:
            raise StopIteration
        return result


class CyclingRandomBatchSampler(torch.utils.data.BatchSampler):
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
        self._batch_idx = 0  # batch index within current epoch

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
