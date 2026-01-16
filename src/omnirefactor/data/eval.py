import os
import time
import multiprocessing as mp

import torch
import numpy as np
from aicsimageio import AICSImage

from ..utils import normalize99, unaugment_tiles_ND, average_tiles_ND, make_tiles_ND
from ..transforms import torch_zoom


class eval_loader(torch.utils.data.DataLoader):
    def __init__(self, dataset, model, postprocess_fn, **kwargs):
        super().__init__(dataset, **kwargs)
        self.model = model
        self.postprocess_fn = postprocess_fn

    def __iter__(self):
        for batch in super().__iter__():
            print(batch)
            predictions = self.model._run_net(batch)
            post_processed_predictions = self.postprocess_fn(predictions)
            yield post_processed_predictions


class sampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class eval_set(torch.utils.data.Dataset):
    def __init__(self, data, dim,
                 channel_axis=None,
                 device=torch.device('cpu'),
                 normalize_stack=True,
                 normalize=True,
                 invert=False,
                 rescale=1.0,
                 pad_mode='reflect',
                 interp_mode='bilinear',
                 extra_pad=1,
                 projection=None,
                 tile=False,
                 aics_args=None,
                 contrast_limits=None):
        self.data = data
        self.dim = dim
        self.channel_axis = channel_axis
        self.stack = isinstance(self.data, np.ndarray)
        self.aics = isinstance(self.data, AICSImage)
        self.aics_args = aics_args if aics_args is not None else {}
        self.list = isinstance(self.data, list)
        if self.list:
            self.files = isinstance(self.data[0], str)
        else:
            self.files = False

        self.device = device
        self.normalize_stack = normalize_stack
        self.normalize = normalize
        self.invert = invert
        self.rescale = rescale
        self.pad_mode = pad_mode
        self.interp_mode = interp_mode
        self.extra_pad = extra_pad
        self.tile = tile
        self.contrast_limits = contrast_limits

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

    def __getitem__(self, inds, no_pad=False, no_rescale=False):
        if isinstance(inds, int):
            inds = [inds]

        if self.stack:
            imgs = torch.tensor(self.data[inds].astype(np.float32))

        elif self.list:
            imgs = [[] for _ in inds]

            for i, index in enumerate(inds):
                if self.files:
                    file = self.data[index]
                    img = AICSImage(file).get_image_data("YX", out_of_memory=True).squeeze()
                else:
                    img = self.data[index]

                imgs[i] = torch.tensor(img.astype(np.float32))

            imgs = torch.stack(imgs, dim=0)

        elif self.aics:
            kwargs = self.aics_args.copy()
            slice_dim = kwargs.pop('slice_dim')
            kwargs[slice_dim] = inds
            imgs = self.data.get_image_data(**kwargs).squeeze().astype(float)
            imgs = torch.tensor(imgs)

        if imgs.ndim == self.dim:
            imgs = imgs.unsqueeze(0)
            print('adding channel dim')

        if self.channel_axis is not None:
            dims = [0, self.channel_axis] + list(range(1, self.channel_axis)) + list(range(self.channel_axis + 1, imgs.ndim))
            print('d', dims, len(dims), imgs.shape)
            imgs = imgs.permute(dims)
        else:
            imgs = imgs.unsqueeze(1)

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

        if self.rescale is not None and self.rescale != 1.0 and not no_rescale:
            imgs = torch_zoom(imgs, self.rescale, mode=self.interp_mode)

        if no_pad:
            return imgs.squeeze()
        else:
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

        B, *DIMS = batch.shape
        YF = torch.zeros((B, model.nclasses, *DIMS[1:]), device=batch.device)

        for b, imgi in enumerate(batch):
            IMG, subs, shape, inds = make_tiles_ND(imgi,
                                                   bsize=bsize,
                                                   augment=augment,
                                                   normalize=normalize,
                                                   tile_overlap=tile_overlap)

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
        worker_imgs, worker_inds, worker_subs = zip(*worker_data)

        batch_imgs = torch.cat(worker_imgs, dim=0)
        batch_inds = [item for sublist in worker_inds for item in sublist]
        batch_subs = [item for sublist in worker_subs for item in sublist]

        return batch_imgs.float(), batch_inds, batch_subs

    def __len__(self):
        return len(self.data)
