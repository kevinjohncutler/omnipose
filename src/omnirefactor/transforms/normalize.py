from __future__ import annotations

import math

import numpy as np
import torch
import fastremap
from dask import array as da
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.stats import gaussian_kde

from .modules import get_module
from .vector import torch_norm

try:
    import numexpr as ne
except Exception:  # pragma: no cover
    ne = None


def safe_divide(num, den, cutoff=0):
    """Divide while ignoring zeros and NaNs in the denominator."""
    module = get_module(num)
    valid_den = (den > cutoff) & module.isfinite(den)

    if isinstance(num, da.Array) or isinstance(den, da.Array):
        return da.where(valid_den, num / den, 0)

    if module == np:
        r = num.astype(np.float32, copy=False)
        return np.divide(r, den, out=np.zeros_like(r), where=valid_den)
    if module == torch:
        r = num.float()
        den = den.float()
        small_val = torch.finfo(den.dtype).tiny
        safe_den = torch.where(valid_den, den, small_val)
        return torch.div(r, safe_den)

    raise TypeError("num must be a numpy array or a PyTorch tensor")


def rescale(T, floor=None, ceiling=None, exclude_dims=None):
    module = get_module(T)
    if exclude_dims is not None:
        if isinstance(exclude_dims, int):
            exclude_dims = (exclude_dims,)
        axes = tuple(i for i in range(T.ndim) if i not in exclude_dims)
        newshape = [T.shape[i] if i in exclude_dims else 1 for i in range(T.ndim)]
    else:
        axes = None
        newshape = T.shape

    if ceiling is None:
        ceiling = module.amax(T, axis=axes)
        if exclude_dims is not None:
            ceiling = ceiling.reshape(*newshape)
    if floor is None:
        floor = module.amin(T, axis=axes)
        if exclude_dims is not None:
            floor = floor.reshape(*newshape)

    return safe_divide(T - floor, ceiling - floor)


def bin_counts(data, num_bins=256):
    """Compute counts of values in bins. Helper for qnorm"""
    unique_values, counts = fastremap.unique(data, return_counts=True)
    bin_edges = np.linspace(unique_values.min(), unique_values.max(), num_bins + 1)
    bin_indices = np.digitize(unique_values, bin_edges) - 1
    binned_counts = np.bincount(bin_indices, weights=counts, minlength=num_bins)
    bin_start = bin_edges[:-1]
    binned_counts = binned_counts[:-1]
    return binned_counts, bin_start


def compute_density(x, y, bw_method=None):
    points = np.vstack([x, y])
    kde = gaussian_kde(points, bw_method=bw_method)
    density = kde(points)
    inverted_points = np.vstack([-x, y])
    inverted_kde = gaussian_kde(inverted_points, bw_method=bw_method)
    inverted_density = inverted_kde(inverted_points)
    symmetric_density = (density + inverted_density) / 2
    symmetric_density = rescale(symmetric_density)
    return symmetric_density


def qnorm(
    Y,
    nbins=100,
    bw_method=2,
    density_cutoff=None,
    density_quantile=[0.001, 0.999],
    debug=False,
    dx=None,
    log=False,
    eps=1,
):
    if dx is not None:
        X = Y[:, ::dx, ::dx]
    else:
        X = Y

    if X.dtype not in [np.uint8, np.uint16, np.uint32, np.uint64]:
        X = (rescale(X) * (2**16 - 1)).astype(np.uint16)

    counts, unique = bin_counts(X, nbins)
    sel = counts > 0
    counts = counts[sel]
    unique = unique[sel]
    x = np.arange(len(counts))
    if log:
        y = np.log(counts + eps)
    else:
        y = counts

    d = compute_density(x, y, bw_method=bw_method)

    if not isinstance(density_quantile, list):
        density_quantile = [density_quantile, density_quantile]

    if density_cutoff is None:
        density_cutoff = np.quantile(d, density_quantile)  # pragma: no cover
        if debug:  # pragma: no cover
            print("dc", density_cutoff)
    elif not isinstance(density_cutoff, list):
        density_cutoff = [density_cutoff, density_cutoff]

    imin = np.argwhere(d > density_cutoff[0])[0][0]
    imax = np.argwhere(d > density_cutoff[1])[-1][0]
    vmin, vmax = unique[imin], unique[imax]

    if vmax > vmin:
        scale_factor = np.float16(1.0 / (vmax - vmin))
        if ne is None:
            r = X * scale_factor
            r[r > 1] = 1
        else:
            r = ne.evaluate("where(X * scale_factor > 1, 1, X * scale_factor)")  # pragma: no cover
    else:
        r = X

    if debug:
        return r, x, y, d, imin, imax, vmin, vmax
    return r


def normalize99(Y, lower=0.01, upper=99.99, contrast_limits=None, dim=None, omni=False):
    """Normalize array/tensor to percentile range."""
    module = get_module(Y)

    if contrast_limits is None:
        quantiles = np.array([lower, upper]) / 100
        if module == torch:
            quantiles = torch.tensor(quantiles, dtype=Y.dtype, device=Y.device)

        if dim is not None:
            Y_flattened = Y.reshape(Y.shape[dim], -1)
            lower_val, upper_val = module.quantile(Y_flattened, quantiles, axis=-1)
            if dim == 0:
                lower_val = lower_val.reshape(Y.shape[dim], *([1] * (len(Y.shape) - 1)))
                upper_val = upper_val.reshape(Y.shape[dim], *([1] * (len(Y.shape) - 1)))
            else:
                lower_val = lower_val.reshape(*Y.shape[:dim], *([1] * (len(Y.shape) - dim - 1)))
                upper_val = upper_val.reshape(*Y.shape[:dim], *([1] * (len(Y.shape) - dim - 1)))
        else:
            try:
                lower_val, upper_val = module.quantile(Y, quantiles)
            except RuntimeError:
                lower_val, upper_val = auto_chunked_quantile(Y, quantiles)
    else:
        if module == np:  # pragma: no cover
            contrast_limits = np.array(contrast_limits)  # pragma: no cover
        elif module == torch:
            contrast_limits = torch.tensor(contrast_limits)
        lower_val, upper_val = contrast_limits

    return module.clip(safe_divide(Y - lower_val, upper_val - lower_val), 0, 1)


def normalize_field(mu, use_torch=False, cutoff=0, omni=False):
    """Normalize all nonzero field vectors to magnitude 1."""
    if use_torch:
        mag = torch_norm(mu, dim=0)
        return torch.where(mag > cutoff, mu / mag, mu)
    mag = np.sqrt(np.nansum(mu**2, axis=0))
    return safe_divide(mu, mag, cutoff)


def searchsorted(tensor, value):
    """Find the indices where `value` should be inserted in `tensor` to maintain order."""
    return (tensor < value).sum()


def compute_quantiles(sorted_array, lower=0.01, upper=0.99):
    assert 0 <= lower <= 1, "Lower quantile must be between 0 and 1"
    assert 0 <= upper <= 1, "Upper quantile must be between 0 and 1"
    lower_index = int(lower * (len(sorted_array) - 1))
    upper_index = int(upper * (len(sorted_array) - 1))
    return sorted_array[lower_index], sorted_array[upper_index]


def quantile_rescale(Y, lower=0.0001, upper=0.9999, contrast_limits=None, bins=None):
    sorted_array = np.sort(Y.flatten(), kind="mergesort")
    lower_val, upper_val = compute_quantiles(sorted_array, lower, upper)
    r = safe_divide(Y - lower_val, upper_val - lower_val)
    r[r < 0] = 0
    r[r > 1] = 1
    return r


def normalize99_hist(Y, lower=0.01, upper=99.99, contrast_limits=None, bins=None):
    upper = upper / 100
    lower = lower / 100

    module = get_module(Y)
    if bins is None:
        num_elements = Y.size if module == np else Y.numel()
        bins = int(np.sqrt(num_elements))

    if contrast_limits is None:
        hist, bin_edges = module.histogram(Y, bins=bins)
        cdf = module.cumsum(hist, axis=0) / module.sum(hist)
        lower_val = bin_edges[searchsorted(cdf, lower)]
        upper_val = bin_edges[searchsorted(cdf, upper)]
    else:
        if module == np:
            contrast_limits = np.array(contrast_limits)
        elif module == torch:
            contrast_limits = torch.tensor(contrast_limits)
        lower_val, upper_val = contrast_limits

    r = safe_divide(Y - lower_val, upper_val - lower_val)
    r[r < 0] = 0
    r[r > 1] = 1
    return r


def pnormalize(Y, p_min=-1, p_max=10):
    module = get_module(Y)
    lower_val = (module.abs(Y * 1.0) ** p_min).sum() ** (1.0 / p_min)
    upper_val = (module.abs(Y * 1.0) ** p_max).sum() ** (1.0 / p_max)
    return module.clip(safe_divide(Y - lower_val, upper_val - lower_val), 0, 1)


def auto_chunked_quantile(tensor, q):
    max_elements = 16e6 - 1
    num_elements = tensor.nelement()
    chunk_size = math.ceil(num_elements / max_elements)
    chunks = torch.chunk(tensor, chunk_size)
    return torch.stack([torch.quantile(chunk, q) for chunk in chunks]).mean(dim=0)


def normalize_image(
    im,
    mask,
    target=0.5,
    foreground=False,
    iterations=1,
    scale=1,
    channel_axis=0,
    per_channel=True,
):
    im = im.astype("float32") * scale
    im_min = im.min()
    im_max = im.max()
    if ne is None:
        im = (im - im_min) / (im_max - im_min)
    else:
        ne.evaluate("(im - im_min) / (im_max - im_min)", out=im)

    if im.ndim > 2:
        im = np.moveaxis(im, channel_axis, -1)
    else:
        im = np.expand_dims(im, axis=-1)

    if not isinstance(mask, list):
        mask = np.expand_dims(mask, axis=-1)
        mask = np.broadcast_to(mask, im.shape)

    bin0 = mask > 0 if foreground else mask == 0
    if iterations > 0:
        structure = np.ones((3,) * (im.ndim - 1) + (1,))
        structure[1, ...] = 0
        bin0 = binary_erosion(bin0, structure=structure, iterations=iterations)

    masked_im = im.copy()
    masked_im[~bin0] = np.nan
    source_target = np.nanmean(masked_im, axis=(0, 1) if per_channel else None)
    source_target = source_target.astype("float32")
    target = np.array(target).astype("float32")
    if ne is None:
        im = im ** (np.log(target) / np.log(source_target))
    else:
        ne.evaluate("im ** (log(target) / log(source_target))", out=im)
    return np.moveaxis(im, -1, channel_axis).squeeze()


def adjust_contrast_masked(
    img: np.ndarray,
    masks: np.ndarray,
    r_target: float = 1.10,
    plo: float = 0.01,
    phi: float = 99.99,
    clip_output: bool = True,
):
    x = np.asarray(img, dtype=np.float32)
    m = np.asarray(masks).astype(bool)
    bg = ~m
    fg = m

    if fg.sum() == 0 or bg.sum() == 0:
        return x.copy(), 1.0, (float(np.min(x)), float(np.max(x)))

    a = np.percentile(x[bg], plo)
    b = np.percentile(x[fg], phi)
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        a = float(np.min(x))
        b = float(np.max(x))

    if not np.isfinite(b - a) or b <= a:
        return x.copy(), 1.0, (float(a), float(b))

    j = (x - a) / (b - a)
    j = np.clip(j, 0.0, 1.0)

    m_fg = float(j[fg].mean())
    m_bg = float(j[bg].mean() + 1e-12)
    r = m_fg / m_bg

    if (r >= 1.0 and r_target < 1.0) or (r <= 1.0 and r_target > 1.0):  # pragma: no cover
        return j.copy(), 1.0, (a, b)

    if abs(np.log(max(r, 1e-12))) < 1e-8 or abs((r - r_target) / max(r_target, 1e-12)) < 1e-3:  # pragma: no cover
        y = j
        gamma = 1.0
    else:
        gamma = float(np.log(max(r_target, 1e-12)) / np.log(max(r, 1e-12)))
        gamma = float(np.clip(gamma, 0.2, 5.0))
        y = np.power(j, gamma)

    if clip_output:
        y = np.clip(y, 0.0, 1.0)

    return y.astype(np.float32), gamma, (float(a), float(b))


def gamma_normalize(
    im,
    mask,
    target=1.0,
    scale=1.0,
    foreground=True,
    iterations=0,
    per_channel=True,
    channel_axis=-1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im = rescale(im) * scale
    if im.ndim > 2:
        im = np.moveaxis(im, channel_axis, -1)
    else:
        im = np.expand_dims(im, axis=-1)

    if not isinstance(mask, list):
        mask = np.stack([mask] * im.shape[-1], axis=-1)

    im = torch.from_numpy(im).float().to(device)
    mask = torch.from_numpy(mask).float().to(device)

    bin0 = mask > 0 if foreground else mask == 0
    if iterations > 0:
        structure = torch.ones((3,) * (im.ndim - 1) + (1,)).to(device)
        structure[1, ...] = 0
        bin0 = torch.from_numpy(
            binary_erosion(bin0.cpu().numpy(), structure=structure.cpu().numpy(), iterations=iterations)
        ).to(device)

    masked_im = im.masked_fill(~bin0, float("nan"))
    source_target = torch.nanmean(masked_im, dim=(0, 1) if per_channel else None)
    im **= (torch.log(target) / torch.log(source_target))

    return im.permute(*[channel_axis] + [i for i in range(im.ndim) if i != channel_axis]).squeeze().cpu().numpy()


def localnormalize(im, sigma1=2, sigma2=20):
    im = normalize99(im)
    blur1 = gaussian_filter(im, sigma=sigma1)
    num = im - blur1
    blur2 = gaussian_filter(num * num, sigma=sigma2)
    den = np.sqrt(blur2)
    return normalize99(num / den + 1e-8)


def localnormalize_GPU(im, sigma1=2, sigma2=20):
    import torchvision.transforms.functional as TF

    im = normalize99(im)
    kernel_size1 = round(sigma1 * 6)
    kernel_size1 += kernel_size1 % 2 == 0
    blur1 = TF.gaussian_blur(im, kernel_size1, sigma1)
    num = im - blur1
    kernel_size2 = round(sigma2 * 6)
    kernel_size2 += kernel_size2 % 2 == 0
    blur2 = TF.gaussian_blur(num * num, kernel_size2, sigma2)
    den = torch.sqrt(blur2)
    return normalize99(num / den + 1e-8)
