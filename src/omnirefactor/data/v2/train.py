"""
v2 train_set extending base_set.

This implementation mirrors the existing train_set but inherits
common functionality from base_set.
"""
import time
import numpy as np
import torch

from .base import base_set
from ...transforms.augment import random_crop_warp
from ...core.flows import masks_to_flows_batch, batch_labels


class train_set(base_set):
    """
    Training dataset with random crop/warp augmentation.

    All samples are cropped to the same tyx size, so batching is trivial.
    Flow labels are computed on-the-fly from mask labels.
    """

    def __init__(self, data, labels, links, timing=False, **kwargs):
        """
        Parameters
        ----------
        data : list of arrays
            Training images
        labels : list of arrays
            Mask labels for each image
        links : list
            Label pair links for multi-label objects
        timing : bool
            Print timing info
        **kwargs : dict
            Additional attributes including:
            - tyx: patch size tuple
            - scale_range: augmentation scale range
            - omni: use Omnipose mode
            - dim: spatial dimensionality
            - nchan: number of channels
            - nclasses: number of output classes
            - device: torch device
            - affinity_field: compute affinity graphs
            - allow_blank_masks: allow empty masks
            - do_rescale/rescale: whether to rescale
            - diam_train: per-image diameters
            - diam_mean: mean diameter
        """
        # Initialize base class
        super().__init__(data, **kwargs)

        # Training-specific data
        self.labels = labels
        self.links = links
        self.timing = timing

        # Set defaults for attributes that might not be in kwargs
        if not hasattr(self, 'augment'):
            self.augment = True

        # Compute tyx if not provided
        if getattr(self, 'tyx', None) is None:
            n = 16
            kernel_size = 2
            base = kernel_size
            L = max(round(224 / (base**4)), 1) * (base**4)
            self.tyx = (L,) * self.dim if self.dim == 2 else (8 * n,) + (8 * n,) * (self.dim - 1)

        # Clamp scale_range
        if hasattr(self, 'scale_range'):
            self.scale_range = max(0, min(2, float(self.scale_range)))
        else:
            self.scale_range = 1.0

        # Augmentation defaults
        self.do_flip = True
        self.dist_bg = 5
        self.normalize = False
        self.gamma_range = [0.75, 2.5]

        # Handle rescale parameter naming (do_rescale vs rescale)
        do_rescale = getattr(self, 'do_rescale', getattr(self, 'rescale', True))
        self.do_rescale = bool(do_rescale)

        # Compute per-image rescale factors
        diam_train = getattr(self, 'diam_train', None)
        diam_mean = getattr(self, 'diam_mean', 30.0)
        if self.do_rescale and diam_train is not None:
            self.rescale_factor = diam_train / diam_mean
        else:
            self.rescale_factor = np.ones(self._nimg, np.float32)

        # Flip axes for proper flow handling
        self.v1 = [0] * (self.dim - 1) + [1]
        self.v2 = [0] * (self.dim - 2) + [1, 0]

    def __getitem__(self, inds):
        """
        Get augmented training batch.

        Parameters
        ----------
        inds : int or list of int
            Sample indices

        Returns
        -------
        imgi : torch.Tensor
            Augmented images (B, C, *tyx)
        lbl : torch.Tensor
            Flow labels (B, nclasses, *tyx)
        inds : list
            Original indices
        """
        if isinstance(inds, int):
            inds = [inds]

        if self.timing:
            tic = time.time()

        nimg = len(inds)
        imgi = np.zeros((nimg, self.nchan) + self.tyx, np.float32)
        labels_crop = np.zeros((nimg,) + self.tyx, np.float32)
        scale = np.zeros((nimg, self.dim), np.float32)
        links = [self.links[idx] for idx in inds]

        # Random crop and augmentation
        for i, idx in enumerate(inds):
            imgi[i], labels_crop[i], scale[i] = random_crop_warp(
                img=self.data[idx],
                Y=self.labels[idx],
                tyx=self.tyx,
                v1=self.v1,
                v2=self.v2,
                nchan=self.nchan,
                rescale_factor=self.rescale_factor[idx],
                scale_range=self.scale_range,
                gamma_range=self.gamma_range,
                do_flip=self.do_flip,
                ind=idx,
                augment=self.augment,
                allow_blank_masks=getattr(self, 'allow_blank_masks', False),
            )

        if self.timing:
            toc = time.time()
            print(f'image augmentation time: {toc - tic:.2f}')
            tic = toc

        # Compute flows from masks
        with torch.no_grad():
            out = masks_to_flows_batch(
                labels_crop, links,
                device=self.device,
                omni=getattr(self, 'omni', True),
                dim=self.dim,
                affinity_field=getattr(self, 'affinity_field', False),
            )

            if self.timing:
                toc = time.time()
                print(f'flow time: {toc - tic:.2f}')
                tic = toc

            # Extract and batch labels
            X = out[:-4]
            slices = out[-4]
            masks, bd, T, mu = [
                torch.stack([x[(Ellipsis,) + slc] for slc in slices])
                for x in X
            ]

            lbl = batch_labels(
                masks, bd, T, mu, self.tyx,
                dim=self.dim,
                nclasses=getattr(self, 'nclasses', 4),
                device=self.device,
            )

            if self.timing:
                toc = time.time()
                print(f'batching time: {toc - tic:.2f}')
                print(f'inds {len(inds)}')

            imgi = torch.tensor(imgi, device=self.device)
        return imgi, lbl, inds

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

    @property
    def nimg(self):
        """Number of images (alias for base class _nimg)."""
        return self._nimg
