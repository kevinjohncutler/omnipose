from .imports import *

import math

import numpy as np
import torch
import torch.nn.functional as F
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as im_shift
from scipy.ndimage import gaussian_filter
import dask

from ..gpu import torch_GPU


def shifts_to_slice(shifts, shape, pad=0):
    """Find the minimal crop box from registration shifts."""
    shifts = np.asarray(shifts, dtype=float)

    slices = []
    for dim in range(shifts.shape[1]):
        max_shift = shifts[:, dim].max()
        min_shift = shifts[:, dim].min()

        start = max(0, math.ceil(max_shift - pad))
        stop = min(shape[dim], math.floor(shape[dim] + min_shift + pad))
        slices.append(slice(start, stop))

    return tuple(slices)


def cross_reg(imstack, upsample_factor=100,
              normalization=None,
              reverse=False,
              localnorm=True,
              max_shift=50,
              order=1,
              moving_reference=False):
    """Find shifts to align a time series to the first frame."""
    dim = imstack.ndim - 1  # dim is spatial, assume first dimension is t
    shape = imstack.shape[-dim:]

    images_to_register = imstack if not reverse else imstack[::-1]

    if localnorm:
        images_to_register = images_to_register / gaussian_filter(images_to_register, sigma=[0, 1, 1])

    if moving_reference:
        shift_vectors = [[]] * len(images_to_register)

        for i, im in enumerate(images_to_register):
            if i == 0:
                ref = im
                shift_vectors[i] = np.zeros(dim)
            else:
                shift = phase_cross_correlation(ref,
                                                im,
                                                upsample_factor=upsample_factor,
                                                normalization=normalization)[0]

                if np.linalg.norm(shift) > max_shift:
                    shift = np.zeros_like(shift)

                shift_vectors[i] = shift
                ref = im_shift(im, shift, cval=np.mean(im), order=order)

    else:
        shift_vectors = [phase_cross_correlation(images_to_register[i],
                                                 images_to_register[i + 1],
                                                 upsample_factor=upsample_factor,
                                                 normalization=normalization)[0]
                         for i in range(len(images_to_register) - 1)]

    if not moving_reference:
        shift_vectors.insert(0, np.asarray([0.0, 0.0]))

    shift_vectors = np.stack(shift_vectors)

    if reverse:
        shift_vectors = shift_vectors[::-1] * (-1 if not moving_reference else 1)

    if not moving_reference:
        shift_vectors = np.where(np.linalg.norm(shift_vectors, axis=1, keepdims=1) > max_shift, 0, shift_vectors)
        shift_vectors = np.cumsum(shift_vectors, axis=0)

    shift_vectors -= np.mean(shift_vectors, axis=0)

    return shift_vectors


def shift_stack(imstack, shift_vectors, order=1, cval=None, prefilter=True, mode='nearest'):
    """Shift each time slice of imstack according to list of nD shifts."""
    imstack = imstack.astype(np.float32)
    shift_vectors = shift_vectors.astype(np.float32)

    ndim = imstack.ndim
    axes = tuple(range(-(ndim - 1), 0))
    cvals = np.nanmean(imstack, axis=axes) if cval is None else [cval] * len(shift_vectors)
    mode = mode if cval is None else 'constant'

    shifted_images = [dask.delayed(im_shift)(image,
                                             shift_vector,
                                             order=order,
                                             prefilter=prefilter,
                                             mode=mode,
                                             cval=cv,
                                             ) for image, shift_vector, cv in zip(imstack, shift_vectors, cvals)]

    shifted_images = dask.compute(*shifted_images)
    shifted_images = np.stack(shifted_images, axis=0)

    return shifted_images


def gaussian_kernel(size: int, sigma: float, device=torch_GPU):
    """Creates a 2D Gaussian kernel with mean 0."""
    coords = torch.arange(size, device=device).float() - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.outer(g)


def apply_gaussian_blur(image, kernel_size, sigma, device=torch_GPU):
    """Applies a Gaussian blur to the image."""
    kernel = gaussian_kernel(kernel_size, sigma, device).unsqueeze(0).unsqueeze(0)
    image = image.unsqueeze(0).unsqueeze(0)

    padding_size = kernel_size // 2
    image = F.pad(image, (padding_size, padding_size, padding_size, padding_size), mode='reflect')

    blurred = F.conv2d(image, kernel, padding=0)

    return blurred.squeeze(0).squeeze(0)


def phase_cross_correlation_GPU_old(image_stack, target_index=None, upsample_factor=10,
                                reverse=False, normalize=False):
    # Assuming image_stack is a 3D tensor [num_images, height, width]
    im_to_reg = torch.stack([i / apply_gaussian_blur(i, 5, 1) for i in image_stack])

    image_stack = F.interpolate(im_to_reg.unsqueeze(1).float(),
                                scale_factor=upsample_factor, mode='bilinear',
                                align_corners=False).squeeze(1)

    shifts = []

    for i in range(1, len(image_stack)):
        if target_index is None:
            if reverse:
                target_image = image_stack[i + 1] if i < len(image_stack) - 1 else image_stack[i]
            else:
                target_image = image_stack[i - 1] if i > 0 else image_stack[i]
        else:
            target_image = image_stack[target_index]

        moving_image = image_stack[i]

        target_fft = torch.fft.fftn(target_image, dim=[-2, -1])
        moving_fft = torch.fft.fftn(moving_image, dim=[-2, -1])

        cross_power_spectrum = target_fft * moving_fft.conj()

        if normalize:
            cross_power_spectrum /= torch.abs(cross_power_spectrum)

        cross_corr = torch.abs(torch.fft.ifftn(cross_power_spectrum, dim=[-2, -1]))
        max_index = torch.argmax(cross_corr.view(-1))

        height = cross_corr.shape[-2]
        width = cross_corr.shape[-1]
        shift_y = max_index // width
        shift_x = max_index % width

        shift_y = height // 2 - (shift_y + height // 2) % height
        shift_x = width // 2 - (shift_x + width // 2) % width

        shift_y = shift_y / upsample_factor
        shift_x = shift_x / upsample_factor

        shifts.append([shift_y, shift_x])

    shifts.append([0, 0])
    shifts = torch.tensor(shifts, device=image_stack.device) * (-2)

    return shifts.float()


def phase_cross_correlation_GPU(image_stack,
                                upsample_factor=10,
                                normalization=None,
                                ):
    device = image_stack.device

    im_to_reg = torch.stack([i / apply_gaussian_blur(i, 9, 3, device=device) for i in image_stack.float()])
    norm = 'backward'
    image_fft = torch.fft.fft2(im_to_reg, norm=norm)

    cross_power_spectrum = image_fft[:-1] * image_fft[1:].conj()

    if normalization == 'phase':
        cross_power_spectrum /= torch.abs(cross_power_spectrum)

    cross_corr = torch.abs(torch.fft.ifft2(cross_power_spectrum, norm=norm))
    m = torch.nn.Upsample(scale_factor=upsample_factor, mode='bilinear')
    cross_corr = m(cross_corr.unsqueeze(1)).squeeze(1)

    max_indices = torch.argmax(cross_corr.view(cross_corr.shape[0], -1), dim=-1).float()
    shifts_y, shifts_x = (max_indices / cross_corr.shape[-1]).long(), (max_indices % cross_corr.shape[-1]).long()

    shifts = 2 * torch.stack([shifts_y, shifts_x]).T
    zero_shift = torch.zeros(1, 2, dtype=shifts.dtype, device=shifts.device)
    shifts = torch.cat([shifts, zero_shift], dim=0) / upsample_factor

    shifts = torch.cumsum(shifts.flip(dims=[0]), dim=0).flip(dims=[0])

    avg_shift = shifts.mean(dim=0)
    shifts -= avg_shift

    return shifts


def apply_shifts(moving_images, shifts):
    # If shifts is a 1D tensor, add an extra dimension to make it 2D
    if len(shifts.shape) == 1:
        shifts = shifts.unsqueeze(0)

    N, H, W = moving_images.shape
    device = moving_images.device
    shifts = shifts / torch.tensor([H, W]).to(device)

    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float),
                                    torch.arange(W, device=device, dtype=torch.float),
                                    indexing='ij')

    grid_y = 2.0 * grid_y / (H - 1) - 1.0
    grid_x = 2.0 * grid_x / (W - 1) - 1.0

    shifted_images = torch.empty_like(moving_images)

    unique_shifts, indices = torch.unique(shifts, dim=0, return_inverse=True)

    bincounts = torch.bincount(indices)
    split_sizes = [bincounts[i].item() for i in range(bincounts.size(0))]
    grouped_indices = torch.split_with_sizes(indices, split_sizes)

    for i, group in enumerate(grouped_indices):
        shift = unique_shifts[i]

        grid_y_shifted = grid_y[None] + shift[0]
        grid_x_shifted = grid_x[None] + shift[1]

        grid = torch.stack([grid_x_shifted, grid_y_shifted], dim=-1)

        shifted_slices = torch.nn.functional.grid_sample(moving_images[group].unsqueeze(1),
                                                         grid.repeat(len(group), 1, 1, 1),
                                                         mode='bilinear',
                                                         align_corners=False
                                                         )

        shifted_images[group] = shifted_slices.squeeze(1)

    return shifted_images
