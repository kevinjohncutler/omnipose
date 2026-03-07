"""
v2 eval_set extending base_set.

This implementation mirrors the existing eval_set but inherits
common functionality from base_set.
"""
from collections import defaultdict

import numpy as np
import torch
from aicsimageio import AICSImage

from .base import base_set
from ...transforms.normalize import normalize99
from ...transforms.tiles import unaugment_tiles_ND, average_tiles_ND, make_tiles_ND
from ...transforms.zoom import torch_zoom


class eval_set(base_set):
    """
    Evaluation dataset with shape-aware batching.

    Supports three batch modes for handling images of different shapes:
    - 'group': Group images by shape, batch within groups (most efficient for mixed datasets)
    - 'pad': Pad all images to common shape (simpler, may waste memory)
    - 'single': Process one image at a time (fallback, no batching)
    - 'auto': Automatically choose best mode based on shape variance

    For uniform-shape datasets, all modes produce identical outputs.
    """

    def __init__(self, data, dim=2,
                 channel_axis=None,
                 device=None,
                 normalize_stack=True,
                 normalize=True,
                 invert=False,
                 rescale_factor=1.0,
                 pad_mode='reflect',
                 interp_mode='bilinear',
                 extra_pad=1,
                 projection=None,
                 tile=False,
                 aics_args=None,
                 contrast_limits=None,
                 batch_mode='auto',
                 max_batch_size=8,
                 min_batch_size=1,
                 **kwargs):
        """
        Parameters
        ----------
        data : array, list, or AICSImage
            Input images. Can be:
            - numpy array (stack of images)
            - list of numpy arrays (can have different shapes)
            - list of file paths
            - AICSImage object
        dim : int
            Spatial dimensionality (2 or 3)
        batch_mode : str, optional
            How to handle images of different shapes:
            - 'auto': Choose based on shape variance (default)
            - 'group': Group same-shape images for batching
            - 'pad': Pad all to common shape
            - 'single': Process one at a time
        max_batch_size : int, optional
            Maximum images per batch (default: 8)
        min_batch_size : int, optional
            Minimum images to form a batch in 'group' mode (default: 1)
        """
        # Detect data type before calling super().__init__
        self.stack = isinstance(data, np.ndarray)
        self.aics = isinstance(data, AICSImage)
        self.aics_args = aics_args if aics_args is not None else {}
        self.list = isinstance(data, list)
        self.files = self.list and len(data) > 0 and isinstance(data[0], str)

        # Initialize base class - pass data as-is
        # Override _init_data to handle different data types
        self.dim = dim
        self.channel_axis = channel_axis
        self.device = device if device is not None else torch.device('cpu')

        # Store data directly without base class conversion
        self.data = data
        if self.stack:
            self._nimg = len(data)
        elif self.list:
            self._nimg = len(data)
        elif self.aics:
            kwargs_copy = self.aics_args.copy()
            slice_dim = kwargs_copy.get('slice_dim', 'Z')
            self._nimg = data.dims.get(slice_dim, 1)
        else:
            self._nimg = 1

        # Store remaining attributes
        self.__dict__.update(kwargs)

        # Eval-specific attributes
        self.normalize_stack = normalize_stack
        self.normalize = normalize
        self.invert = invert
        self.rescale_factor = rescale_factor
        self.pad_mode = pad_mode
        self.interp_mode = interp_mode
        self.extra_pad = extra_pad
        self.tile = tile
        self.contrast_limits = contrast_limits

        # Batch mode parameters
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size

        # Analyze shapes and set batch mode
        self._analyze_shapes()
        self.batch_mode = self._resolve_batch_mode(batch_mode)

        # Build batch plan based on mode
        self._build_batch_plan()

    def _analyze_shapes(self):
        """Analyze spatial shapes of all images to plan batching."""
        self._shapes = []
        self._shape_groups = defaultdict(list)

        for idx in range(self._nimg):
            shape = self._get_spatial_shape(idx)
            self._shapes.append(shape)
            self._shape_groups[shape].append(idx)

    def _get_spatial_shape(self, idx):
        """Get the spatial shape of image at index (after rescaling)."""
        if self.stack:
            img = self.data[idx]
        elif self.list:
            if self.files:
                # For files, we need to peek at the shape
                img = AICSImage(self.data[idx]).get_image_data("YX", out_of_memory=True).squeeze()
            else:
                img = self.data[idx]
        elif self.aics:
            kwargs = self.aics_args.copy()
            slice_dim = kwargs.pop('slice_dim')
            kwargs[slice_dim] = [idx]
            img = self.data.get_image_data(**kwargs).squeeze()
        else:
            img = self.data

        # Get spatial dimensions (last dim dimensions)
        if hasattr(img, 'shape'):
            shape = img.shape[-self.dim:]
        else:
            shape = np.array(img).shape[-self.dim:]

        # Account for rescaling
        if self.rescale_factor is not None and self.rescale_factor != 1.0:
            shape = tuple(int(s * self.rescale_factor) for s in shape)

        return shape

    def _resolve_batch_mode(self, mode):
        """Resolve 'auto' batch mode to concrete mode."""
        if mode != 'auto':
            return mode

        n_shapes = len(self._shape_groups)
        n_images = self._nimg

        if n_images == 1:
            return 'single'
        elif n_shapes == 1:
            # All same shape - group is most efficient
            return 'group'
        elif n_shapes <= 5:
            # Few distinct shapes - grouping works well
            return 'group'
        else:
            # Many different shapes - padding may be more efficient
            return 'pad'

    def _build_batch_plan(self):
        """Build the batch plan based on batch mode."""
        self._batches = []

        if self.batch_mode == 'single':
            self._build_single_plan()
        elif self.batch_mode == 'group':
            self._build_grouped_plan()
        elif self.batch_mode == 'pad':
            self._build_padded_plan()
        else:
            raise ValueError(f"Unknown batch_mode: {self.batch_mode}")

    def _build_single_plan(self):
        """Build batch plan for single-image processing."""
        for idx in range(len(self._shapes)):
            shape = self._shapes[idx]
            pad_shape = self._compute_pad_shape(shape)
            self._batches.append(([idx], shape, pad_shape))

    def _build_grouped_plan(self):
        """Build batch plan by grouping same-shape images."""
        overflow = []

        for shape, indices in self._shape_groups.items():
            pad_shape = self._compute_pad_shape(shape)

            # Split into chunks of max_batch_size
            for i in range(0, len(indices), self.max_batch_size):
                batch_indices = indices[i:i + self.max_batch_size]
                if len(batch_indices) >= self.min_batch_size:
                    self._batches.append((batch_indices, shape, pad_shape))
                else:
                    # Small groups go to overflow
                    overflow.extend(batch_indices)

        # Handle overflow by processing individually
        for idx in overflow:
            shape = self._shapes[idx]
            pad_shape = self._compute_pad_shape(shape)
            self._batches.append(([idx], shape, pad_shape))

    def _build_padded_plan(self):
        """Build batch plan with all images padded to common shape."""
        # Find maximum shape across all images
        max_shape = tuple(
            max(s[i] for s in self._shapes)
            for i in range(self.dim)
        )
        common_pad_shape = self._compute_pad_shape(max_shape)

        # All images go in sequential batches, padded to common shape
        all_indices = list(range(len(self._shapes)))
        for i in range(0, len(all_indices), self.max_batch_size):
            batch_indices = all_indices[i:i + self.max_batch_size]
            # None for orig_shape indicates mixed shapes in batch
            self._batches.append((batch_indices, None, common_pad_shape))

    def _compute_pad_shape(self, shape):
        """Compute padded shape for network (16-divisible + extra_pad)."""
        div = 16
        extra = self.extra_pad
        pad_shape = tuple(
            int(div * np.ceil(s / div)) + extra * div
            for s in shape
        )
        return pad_shape

    @property
    def n_batches(self):
        """Number of batches in the current plan."""
        return len(self._batches)

    @property
    def shape_info(self):
        """Summary of shape distribution."""
        return {
            'n_images': self._nimg,
            'n_unique_shapes': len(self._shape_groups),
            'batch_mode': self.batch_mode,
            'n_batches': self.n_batches,
            'shapes': dict(self._shape_groups),
        }

    def __getitem__(self, inds, no_pad=False, no_rescale=False):
        """
        Get preprocessed images by index.

        Parameters
        ----------
        inds : int or list of int
            Image indices
        no_pad : bool
            If True, skip padding (return raw tensor)
        no_rescale : bool
            If True, skip rescaling

        Returns
        -------
        If no_pad=True:
            imgs : torch.Tensor (C, *spatial) or (B, C, *spatial)
        If no_pad=False:
            imgs : torch.Tensor (B, C, *padded_spatial)
            inds : list of int
            subs : list of slice arrays for extracting original region
        """
        if isinstance(inds, int):
            inds = [inds]

        # Load images
        if self.stack:
            imgs = torch.tensor(self.data[inds].astype(np.float32))
        elif self.list:
            imgs = []
            for index in inds:
                if self.files:
                    file = self.data[index]
                    img = AICSImage(file).get_image_data("YX", out_of_memory=True).squeeze()
                else:
                    img = self.data[index]
                imgs.append(torch.tensor(img.astype(np.float32)))
            imgs = torch.stack(imgs, dim=0)
        elif self.aics:
            kwargs = self.aics_args.copy()
            slice_dim = kwargs.pop('slice_dim')
            kwargs[slice_dim] = inds
            imgs = self.data.get_image_data(**kwargs).squeeze().astype(float)
            imgs = torch.tensor(imgs)

        # Handle dimensions
        if imgs.ndim == self.dim:
            imgs = imgs.unsqueeze(0)

        if self.channel_axis is not None:
            dims = [0, self.channel_axis] + list(range(1, self.channel_axis)) + list(range(self.channel_axis + 1, imgs.ndim))
            imgs = imgs.permute(dims)
        else:
            imgs = imgs.unsqueeze(1)

        # Normalize
        if self.normalize and not self.tile:
            for b in range(imgs.shape[0]):
                for c in range(imgs.shape[1]):
                    imgs[b, c] = normalize99(
                        imgs[b, c],
                        contrast_limits=self.contrast_limits,
                        dim=None,
                    )
                    if self.invert:
                        imgs[b, c] = -1 * imgs[b, c] + 1

        # Rescale
        if self.rescale_factor is not None and self.rescale_factor != 1.0 and not no_rescale:
            imgs = torch_zoom(imgs, self.rescale_factor, mode=self.interp_mode)

        if no_pad:
            # Squeeze only the batch dimension if singleton, keep channel dimension
            if imgs.shape[0] == 1:
                return imgs.squeeze(0)  # (1, C, *spatial) -> (C, *spatial)
            return imgs  # (B, C, *spatial)
        else:
            # Compute padding
            shape = imgs.shape[-self.dim:]
            div = 16
            extra = self.extra_pad
            idxs = [k for k in range(-self.dim, 0)]
            Lpad = [int(div * np.ceil(shape[i] / div) - shape[i]) for i in idxs]
            lower_pad = [extra * div // 2 + Lpad[k] // 2 for k in range(self.dim)]
            upper_pad = [extra * div // 2 + Lpad[k] - Lpad[k] // 2 for k in range(self.dim)]

            pads = tuple()
            for k in range(self.dim):
                pads += (lower_pad[-(k + 1)], upper_pad[-(k + 1)])

            subs = [np.arange(lower_pad[k], lower_pad[k] + shape[k]) for k in range(self.dim)]

            I = torch.nn.functional.pad(imgs, pads, mode=self.pad_mode, value=None)
            return I, inds, subs

    def _run_tiled(self, batch, model,
                   batch_size=8, augment=False, bsize=224,
                   normalize=True,
                   tile_overlap=0.1, return_conv=False):
        """Run tiled inference on large images."""
        B, *DIMS = batch.shape
        YF = torch.zeros((B, model.nclasses, *DIMS[1:]), device=batch.device)

        for b, imgi in enumerate(batch):
            IMG, subs, shape, inds = make_tiles_ND(
                imgi,
                bsize=bsize,
                augment=augment,
                normalize=normalize,
                tile_overlap=tile_overlap
            )

            niter = int(np.ceil(IMG.shape[0] / batch_size))
            nout = model.nclasses + 32 * return_conv
            y = torch.zeros((IMG.shape[0], nout) + tuple(IMG.shape[-model.dim:]), device=IMG.device)
            for k in range(niter):
                irange = np.arange(batch_size * k, min(IMG.shape[0], batch_size * k + batch_size))
                y0 = model.net(IMG[irange])[0]
                arg = (len(irange),) + y0.shape[-(model.dim + 1):]
                y[irange] = y0.reshape(arg)

            if augment:
                y = unaugment_tiles_ND(y, inds, model.unet)
            yf = average_tiles_ND(y, subs, shape)
            slc = tuple([slice(s) for s in shape])
            yf = yf[(Ellipsis,) + slc]

            YF[b] = yf
        return YF

    def collate_fn(self, worker_data):
        """Collate function for DataLoader."""
        worker_imgs, worker_inds, worker_subs = zip(*worker_data)

        batch_imgs = torch.cat(worker_imgs, dim=0)
        batch_inds = [item for sublist in worker_inds for item in sublist]
        batch_subs = [item for sublist in worker_subs for item in sublist]

        return batch_imgs.float(), batch_inds, batch_subs

    def collate_fn_batched(self, batch_data):
        """Collate function for batch-mode iteration.

        Handles padding images to common shape within a batch when needed.
        """
        all_imgs = []
        all_inds = []
        all_subs = []

        for imgs, inds, subs in batch_data:
            if isinstance(imgs, torch.Tensor):
                all_imgs.append(imgs)
            else:
                all_imgs.extend(imgs)
            all_inds.extend(inds if isinstance(inds, list) else [inds])
            all_subs.extend(subs if isinstance(subs[0], (list, np.ndarray)) else [subs])

        # Stack if all same shape, otherwise they should already be padded
        if all(img.shape == all_imgs[0].shape for img in all_imgs):
            batch_imgs = torch.stack(all_imgs, dim=0) if all_imgs[0].dim() == self.dim + 1 else torch.cat(all_imgs, dim=0)
        else:
            batch_imgs = torch.cat(all_imgs, dim=0)

        return batch_imgs.float(), all_inds, all_subs

    def get_batch(self, batch_idx):
        """Get a specific batch by index.

        Returns
        -------
        batch_imgs : torch.Tensor
            Batch of images, shape (N, C, *spatial)
        batch_inds : list
            Original image indices in this batch
        batch_subs : list
            Subscript slices to extract original region from each image
        """
        indices, orig_shape, target_shape = self._batches[batch_idx]

        batch_imgs = []
        batch_subs = []

        for idx in indices:
            # Get preprocessed image (no_pad=True returns just the tensor)
            img = self.__getitem__([idx], no_pad=True, no_rescale=False)

            # Handle shape: should be (B, C, *spatial) -> (C, *spatial)
            if img.dim() > self.dim + 1:
                img = img.squeeze(0)

            # Pad to target shape for this batch
            img_padded, subs = self._pad_to_target(img, target_shape)
            batch_imgs.append(img_padded)
            batch_subs.append(subs)

        batch = torch.stack(batch_imgs, dim=0)
        return batch, list(indices), batch_subs

    def _pad_to_target(self, img, target_shape):
        """Pad image to target shape with reflection padding.

        Parameters
        ----------
        img : torch.Tensor
            Image tensor of shape (C, *spatial)
        target_shape : tuple
            Target spatial shape

        Returns
        -------
        padded : torch.Tensor
            Padded image
        subs : list
            Subscript slices to extract original region
        """
        current_shape = img.shape[-self.dim:]

        pads = []
        subs = []
        for i in range(self.dim):
            diff = target_shape[i] - current_shape[i]
            pad_lo = diff // 2
            pad_hi = diff - pad_lo
            # torch.nn.functional.pad uses reverse order (last dim first)
            pads.extend([pad_lo, pad_hi])
            # Use slice for proper 2D extraction
            subs.append(slice(pad_lo, pad_lo + current_shape[i]))

        # Reverse pads for torch.nn.functional.pad (last dim first)
        pads = pads[::-1]

        if any(p > 0 for p in pads):
            padded = torch.nn.functional.pad(img.unsqueeze(0), pads, mode=self.pad_mode).squeeze(0)
        else:
            padded = img

        return padded, subs

    def iter_batches(self):
        """Generator that yields batches according to the batch plan.

        Yields
        ------
        batch_imgs : torch.Tensor
            Batch of images, shape (N, C, *spatial)
        batch_inds : list
            Original image indices in this batch
        batch_subs : list
            Subscript slices to extract original region from each image
        """
        for batch_idx in range(self.n_batches):
            yield self.get_batch(batch_idx)

    def __len__(self):
        return self._nimg

    @property
    def n_images(self):
        """Number of images in the dataset."""
        return self._nimg
