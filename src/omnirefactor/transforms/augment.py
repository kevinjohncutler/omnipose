from __future__ import annotations

import numpy as np
import mgen
import fastremap
from scipy.ndimage import affine_transform, gaussian_filter

from .. import utils
from ..transforms.normalize import normalize99, rescale
from ..core.diam import diameters
from ..core.njit import most_frequent

try:
    from skimage.util import random_noise
    SKIMAGE_ENABLED = True
except ModuleNotFoundError:
    random_noise = None
    SKIMAGE_ENABLED = False

try:
    from opensimplex import OpenSimplex
    OPEN_SIMPLEX_ENABLED = True
except ModuleNotFoundError:
    OpenSimplex = None
    OPEN_SIMPLEX_ENABLED = False


def rotate(V, theta, order=1, output_shape=None, center=None):
    dim = V.ndim
    v1 = np.array([0] * (dim - 1) + [1])
    v2 = np.array([0] * (dim - 2) + [1, 0])

    s_in = V.shape
    s_out = s_in if output_shape is None else output_shape
    M = mgen.rotation_from_angle_and_plane(np.pi / 2 - theta, v2, v1)
    if center is None:
        c_in = 0.5 * np.array(s_in)
    else:
        c_in = center
    c_out = 0.5 * np.array(s_out)
    offset = c_in - np.dot(np.linalg.inv(M), c_out)
    V_rot = affine_transform(V, np.linalg.inv(M), offset=offset, order=order, output_shape=output_shape)
    return V_rot


# no reason to use njit here except for compatibility with jitted functions that call it
# this way, the same factor is used everywhere (CPU with/without interp, GPU)
def mode_filter(masks):
    """
    Super fast mode filter (compared to scipy, idk about PIL) to clean up interpolated labels.
    """
    pad = 1
    masks = np.pad(masks, pad).astype(int)
    d = masks.ndim
    shape = masks.shape
    coords = np.nonzero(masks)
    steps, inds, idx, fact, sign = utils.kernel_setup(d)

    subinds = np.concatenate(inds)
    substeps = steps[subinds]
    neighbors = utils.get_neighbors(coords, substeps, d, shape)

    neighbor_masks = masks[tuple(neighbors)]

    mask_filt = np.zeros_like(masks)
    most_f = most_frequent(neighbor_masks)
    z = most_f == 0
    most_f[z] = masks[coords][z]
    mask_filt[coords] = most_f

    unpad = tuple([slice(pad, -pad)] * d)
    return mask_filt[unpad]


# Omnipose has special training settings. Loss function and augmentation.
# Spacetime segmentation: augmentations need to treat time differently
# Need to assume a particular axis is the temporal axis; most convenient is tyx.
def random_rotate_and_resize(X, Y=None, scale_range=1., gamma_range=[.75, 2.5], tyx=(224, 224),
                             do_flip=True, rescale_factor=None, inds=None, nchan=1, allow_blank_masks=False):

    """ augmentation by random rotation and resizing

        X and Y are lists or arrays of length nimg, with channels x Lt x Ly x Lx (channels optional, Lt only in 3D)

        Parameters
        ----------
        X: float, list of ND arrays
            list of image arrays of size [nchan x Lt x Ly x Lx] or [Lt x Ly x Lx]
        Y: float, list of ND arrays
            list of image labels of size [nlabels x Lt x Ly x Lx] or [Lt x Ly x Lx]. The 1st channel
            of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
            If Y.shape[0]==3, then the labels are assumed to be [distance, T flow, Y flow, X flow].
        links: list of label links
            lists of label pairs linking parts of multi-label object together
            this is how omnipose gets around boudnary artifacts druing image warps
        scale_range: float (optional, default 1.0)
            Range of resizing of images for augmentation. Images are resized by
            (1-scale_range/2) + scale_range * np.random.rand()
        gamma_range: float, list
           images are gamma-adjusted im**gamma for gamma in [low,high]
        tyx: int, tuple
            size of transformed images to return, e.g. (Ly,Lx) or (Lt,Ly,Lx)
        do_flip: bool (optional, default True)
            whether or not to flip images horizontally
        rescale_factor: float, array or list
            how much to resize images by before performing augmentations
        inds: int, list
            image indices (for debugging)
        nchan: int
            number of channels the images have

        Returns
        -------
        imgi: float, ND array
            transformed images in array [nimg x nchan x xy[0] x xy[1]]
        lbl: float, ND array
            transformed labels in array [nimg x nchan x xy[0] x xy[1]]
        scale: float, 1D array
            scalar(s) by which each image was resized

    """
    dist_bg = 5
    dim = len(tyx)

    nimg = len(X)
    imgi = np.zeros((nimg, nchan) + tyx, np.float32)

    lbl = np.zeros((nimg,) + tyx, np.float32)
    scale = np.zeros((nimg, dim), np.float32)

    v1 = [0] * (dim - 1) + [1]
    v2 = [0] * (dim - 2) + [1, 0]

    if rescale_factor is None:
        rescale = np.ones(nimg, np.float32)
    elif np.isscalar(rescale_factor):
        rescale = np.ones(nimg, np.float32) * rescale_factor
    else:
        rescale = np.array(rescale_factor)

    for n in range(nimg):
        img = X[n].copy()
        y = None if Y is None else Y[n]
        imgi[n], lbl[n], scale[n] = random_crop_warp(img, y, tyx, v1, v2, nchan,
                                                     rescale[n],
                                                     scale_range, gamma_range, do_flip,
                                                     inds is None if inds is None else inds[n],
                                                     allow_blank_masks=allow_blank_masks)

    return imgi, lbl, np.mean(scale)


def random_crop_warp(img, Y, tyx, v1, v2, nchan, rescale_factor, scale_range, gamma_range, do_flip, ind,
                     do_labels=True, depth=0, augment=True, allow_blank_masks=False):
    """
    This sub-fuction of `random_rotate_and_resize()` recursively performs random cropping until
    a minimum number of cell pixels are found, then proceeds with augemntations.

    Parameters
    ----------
    X: float, list of ND arrays
        image array of size [nchan x Lt x Ly x Lx] or [Lt x Ly x Lx]
    Y: float, ND array
        image label array of size [nlabels x Lt x Ly x Lx] or [Lt x Ly x Lx].. The 1st channel
        of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
        If Y.shape[0]==3, then the labels are assumed to be [cell probability, T flow, Y flow, X flow].
    tyx: int, tuple
        size of transformed images to return, e.g. (Ly,Lx) or (Lt,Ly,Lx)
    nchan: int
        number of channels the images have
    rescale_factor: float, array or list
        how much to resize images by before performing augmentations
    scale_range: float
        Range of resizing of images for augmentation. Images are resized by
        (1-scale_range/2) + scale_range * np.random.rand()
    gamma_range: float, list
       images are gamma-adjusted im**gamma for gamma in [low,high]
    do_flip: bool (optional, default True)
        whether or not to flip images horizontally
    ind: int
        image index (for debugging)
    dist_bg: float
        nonegative value X for assigning -X to where distance=0 (deprecated, now adapts to field values)
    depth: int
        how many time this function has been called on an image
    augment: bool
        whether or not to perform all non-morphological augmentations on the image (gamma, noise, etc.)

    Returns
    -------
    imgi: float, ND array
        transformed images in array [nchan x xy[0] x xy[1]]
    lbl: float, ND array
        transformed labels in array [nchan x xy[0] x xy[1]]
    scale: float, 1D array
        scalar by which the image was resized

    """

    dim = len(tyx)
    if depth > 100:
        error_message = """Sparse or over-dense image detected.
        Problematic index is: {}.
        Image shape is: {}.
        tyx is: {}.
                               rescale_factor is {}""".format(ind, img.shape, tyx, rescale_factor)
        raise ValueError(error_message)

    if depth > 200:
        error_message = """Recusion depth exceeded. Check that your images contain cells and
                           background within a typical crop.
                           Failed index is: {}.""".format(ind)
        raise ValueError(error_message)
        return

    numpx = np.prod(tyx)
    if Y is not None:
        labels = Y.copy()

        eps = 1e-8
        mean_target = 1.0
        a = 1.0 / (scale_range + eps)
        b = scale_range + eps

        alpha = 1
        m = (mean_target - a) / (b - a)
        beta = alpha * (1.0 - m) / m
        scale = a + (b - a) * np.random.beta(alpha, beta, size=dim)

        if rescale_factor is not None:
            scale *= 1. / rescale_factor
    else:
        scale = 1

    s = img.shape[-dim:]

    theta = np.random.rand() * np.pi * 2

    rot = mgen.rotation_from_angle_and_plane(-theta, v2, v1)
    M_inv = np.diag(1. / scale).dot(rot.T)

    axes = range(dim)
    s = img.shape[-dim:]
    rt = (np.random.rand(dim,) - .5)
    dxy = [rt[a] * (np.maximum(0, s[a] - tyx[a])) for a in axes]

    c_in = 0.5 * np.array(s) + dxy
    c_out = 0.5 * np.array(tyx)
    offset = c_in - np.dot(M_inv, c_out)

    mode = np.random.choice(['constant', 'nearest', 'mirror'])

    lbl = do_warp(labels, M_inv, tyx, offset=offset, order=0, mode=mode)

    if len(fastremap.unique(lbl)) < 2 and not allow_blank_masks:
            return random_crop_warp(img, Y, tyx, v1, v2, nchan, rescale_factor, scale_range,
                                    gamma_range, do_flip, ind, do_labels, depth=depth + 1,
                                    augment=augment, allow_blank_masks=allow_blank_masks)
    else:
        lbl = mode_filter(lbl)

    imgi = np.zeros((nchan,) + tyx, np.float32)

    for k in range(nchan):
        has_foreground = np.any(lbl)
        if np.issubdtype(img[k].dtype, np.integer):
            nvals = len(fastremap.unique(img[k]))
        else:
            nvals = len(np.unique(img[k]))
        raw_bits = int(np.ceil(np.log2(max(nvals, 1))))

        if has_foreground:
            arr = rescale(img[k])
        else:
            maxv = np.iinfo(img[k].dtype).max if np.issubdtype(img[k].dtype, np.integer) else 1.0
            arr = img[k].astype(np.float32) / maxv
            if nvals < maxv // 100 and arr.max() == 1.0:
                arr *= (nvals / maxv) * 10

        imgi[k] = do_warp(arr, M_inv, tyx, order=1, offset=offset, mode=mode)

        aug_choices = np.random.choice([0, 1], 8)

        if augment:
            if aug_choices[1]:
                imgi[k] = gaussian_filter(imgi[k], np.random.uniform(0, np.sqrt(2)))

            if aug_choices[2] and has_foreground:
                dp = .1
                dpct = np.random.triangular(left=0, mode=0, right=dp, size=2)
                imgi[k] = normalize99(imgi[k], upper=100 - dpct[0], lower=dpct[1])

            if aug_choices[0] and has_foreground:
                if aug_choices[7] or not OPEN_SIMPLEX_ENABLED:
                    axis = np.random.randint(0, dim)
                    coords = [np.arange(0, s, dtype=np.float32) for s in tyx[::-1]]
                    illum_field = coords[axis]
                    illum_field = (illum_field - illum_field.min()) / (illum_field.max() - illum_field.min())
                else:
                    simplex = OpenSimplex(seed=np.random.randint(0, 2 ** 31))
                    spatial_shape = tyx[-2:]

                    mean_obj_diam = 2 * diameters(lbl) if np.any(lbl) else 1.0
                    freq_jitter = np.random.triangular(left=1, mode=1.0, right=10.0)
                    fs = mean_obj_diam * freq_jitter
                    coords = [np.arange(0, s, dtype=np.float32) / fs for s in spatial_shape[::-1]]
                    illum_field = rescale(simplex.noise2array(*coords))

                min_factor = np.random.triangular(left=0, mode=0, right=1) ** .5

                multiplier = min_factor + (1.0 - min_factor) * illum_field

                if imgi[k].ndim > 2:
                    multiplier = multiplier[np.newaxis, ...]
                multiplier = np.broadcast_to(multiplier, imgi[k].shape).astype(np.float32)

                imgi[k] = (imgi[k] + min_factor) / (imgi[k].max() + min_factor) * multiplier

            if SKIMAGE_ENABLED and aug_choices[3]:
                var_range = 1e-2
                var = np.random.triangular(left=1e-8, mode=1e-8, right=var_range, size=1)
                sigma = float(np.sqrt(var).item())
                noise = np.random.normal(0.0, sigma, size=imgi[k].shape).astype(np.float32)
                imgi[k] = imgi[k] + imgi[k] * noise
                imgi[k] = np.clip(imgi[k], 0, 1)

            if aug_choices[4] and has_foreground:
                min_bits = 3
                max_shift = max(0, min(14, raw_bits - min_bits))
                if max_shift:
                    bit_shift = int(np.random.triangular(0, max_shift // 2, max_shift))
                else:
                    bit_shift = 0
                im = utils.to_16_bit(imgi[k])
                imgi[k] = rescale(im >> bit_shift)

            if aug_choices[5]:
                border_inds = utils.border_indices(tyx)
                imgi[k].flat[border_inds] *= np.random.uniform(0, 1)

            if aug_choices[6]:
                indices = np.random.rand(*tyx) < 0.001
                imgi[k][indices] = np.random.choice([0, 1], size=np.count_nonzero(indices))

            gamma = np.random.triangular(left=gamma_range[0], mode=1, right=gamma_range[1])
            imgi[k] = imgi[k] ** gamma

    if do_flip:
        for d in range(1, dim + 1):
            if np.random.choice([0, 1]):
                imgi = np.flip(imgi, axis=-d)
                if Y is not None:
                    lbl = np.flip(lbl, axis=-d)

    return imgi, lbl, scale


def do_warp(A, M_inv, tyx, offset=0, order=1, mode='constant', **kwargs):
    """Wrapper function for affine transformations during augmentation."""
    return affine_transform(A, M_inv, offset=offset,
                            output_shape=tyx, order=order,
                            mode=mode, **kwargs)
