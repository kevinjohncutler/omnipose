from __future__ import annotations

import numpy as np
import mgen
import fastremap
from scipy.ndimage import affine_transform, gaussian_filter

import torch
import torch.nn.functional as _F

from .. import utils
from ..transforms.normalize import normalize99, rescale
from ..core.diam import diameters
from ..core.njit import most_frequent

# scipy boundary mode → torch grid_sample padding_mode
_SCIPY_TO_TORCH_PAD = {'constant': 'zeros', 'nearest': 'reflection', 'mirror': 'reflection'}
_MODE_CHOICES = ['constant', 'nearest', 'mirror']


# ── Torch random helpers (hardware-agnostic; use global torch RNG) ─────────────

def _trand(*size):
    """Uniform [0, 1) scalar or array — replaces np.random.rand."""
    t = torch.rand(size if size else (1,))
    return t.item() if not size else t.numpy()


def _trandint(low, high):
    """Integer in [low, high) — replaces np.random.randint."""
    return torch.randint(low, high, (1,)).item()


def _tchoice2():
    """Binary choice {0, 1} — replaces np.random.choice([0, 1])."""
    return torch.randint(0, 2, (1,)).item()


def _tchoice2_vec(n):
    """n independent binary choices — replaces np.random.choice([0, 1], n)."""
    return torch.randint(0, 2, (n,)).numpy()


def _tbeta(alpha, beta, size):
    """Beta distribution sample — replaces np.random.beta(alpha, beta, size=dim)."""
    a = torch.full((size,), float(alpha))
    b = torch.full((size,), float(beta))
    return torch.distributions.Beta(a, b).sample().numpy()


def _ttriangular(low, mode, high, size=None):
    """Triangular distribution via inverse CDF — replaces np.random.triangular.

    Works for scalars (size=None → returns float) and arrays (size=int → returns ndarray).
    """
    n = 1 if size is None else (size if isinstance(size, int) else size[0])
    u = torch.rand(n)
    fc = (mode - low) / (high - low)
    left  = low  + torch.sqrt(torch.clamp(u          * (high - low) * (mode - low), min=0.0))
    right = high - torch.sqrt(torch.clamp((1.0 - u)  * (high - low) * (high - mode), min=0.0))
    result = torch.where(u < fc, left, right)
    if size is None:
        return result.item()
    return result.numpy()


def _tnormal(sigma, size):
    """Gaussian noise array — replaces np.random.normal(0, sigma, size)."""
    return (torch.randn(size) * sigma).numpy()


def _tuniform(low, high):
    """Uniform [low, high) scalar — replaces np.random.uniform(low, high)."""
    return (low + (high - low) * torch.rand(1)).item()



# Cache for _build_grid_nd: (tyx, device_str) → out_coords (dim, npix)
_grid_coords_cache: dict = {}
# Cache for _mode_filter_gpu: conv kernel keyed by (ndim, device_str)
_mode_kernel_cache: dict = {}
# Cache for border indices GPU tensors: (tyx, device_str) → LongTensor
_border_inds_cache: dict = {}
# Cache for border bool mask tensors: (tyx, device_str) → (*tyx) bool
_border_mask_cache: dict = {}
# Cache for Gaussian blur kernels: (rounded_sigma, ndim, device_str) → kernel
_blur_kernel_cache: dict = {}

# Cached capability flags — probed once per device type on first use.
_grid3d_cap: dict = {}   # (device_type, mode) → bool

def _supports_grid3d(device, mode: str = 'bilinear') -> bool:
    """True if device supports 3D grid_sample with the given interpolation mode."""
    dtype = getattr(device, 'type', str(device))
    key = (dtype, mode)
    if key not in _grid3d_cap:
        try:
            dev = torch.device(dtype)
            _F.grid_sample(torch.zeros(1, 1, 2, 2, 2, device=dev),
                           torch.zeros(1, 2, 2, 2, 3, device=dev),
                           mode=mode, align_corners=True)
            _grid3d_cap[key] = True
        except (NotImplementedError, RuntimeError):
            _grid3d_cap[key] = False
    return _grid3d_cap[key]


def _build_grid_nd(M_inv, offset, s_in, tyx, device):
    """Build a (1, *tyx, dim) sampling grid for F.grid_sample (2D or 3D).

    Converts scipy affine_transform params (M_inv dim×dim, offset dim-vec) to
    PyTorch normalized coords in [-1, 1].

    scipy: in_coords = M_inv @ out_coords + offset
    PyTorch convention: last dim lists normalized coords in reverse spatial order
    (x_W, x_H) for 2D or (x_W, x_H, x_D) for 3D.

    The output-pixel coordinate grid is cached per (tyx, device).
    """
    dim = len(tyx)

    key = (tyx, str(device))
    if key not in _grid_coords_cache:
        ranges = [torch.arange(int(t), dtype=torch.float32, device=device) for t in tyx]
        grids = torch.meshgrid(*ranges, indexing='ij')          # each (*tyx)
        _grid_coords_cache[key] = torch.stack([g.reshape(-1) for g in grids])  # (dim, npix)
    out_coords = _grid_coords_cache[key]                        # (dim, npix)

    M_t   = torch.tensor(M_inv,  dtype=torch.float32, device=device)   # (dim, dim)
    off_t = torch.tensor(offset, dtype=torch.float32, device=device)   # (dim,)

    in_coords = M_t @ out_coords + off_t.unsqueeze(1)                  # (dim, npix)

    # Normalize each spatial dim to [-1, 1]
    norms = torch.tensor([max(int(s_in[i]) - 1, 1) for i in range(-dim, 0)],
                         dtype=torch.float32, device=device)
    norm_coords = 2.0 * in_coords / norms.unsqueeze(1) - 1.0          # (dim, npix)

    # PyTorch grid_sample wants last dim in reverse spatial order: (x_last, ..., x_first)
    norm_coords_rev = norm_coords.flip(0).T.reshape(*tyx, dim)         # (*tyx, dim)
    return norm_coords_rev.unsqueeze(0)                                 # (1, *tyx, dim)


def _mode_filter_gpu(lbl_t):
    """GPU mode filter for 2D or 3D label tensors.

    lbl_t: (*spatial) float32 tensor with integer label values — (H,W) or (D,H,W).
    Returns same shape float32 tensor with the modal label in each 3^ndim neighborhood.

    Uses conv2d/conv3d to count each label's presence — O(N_labels × pixels).
    Kernel is cached per (ndim, device).
    """
    device = lbl_t.device
    ndim = lbl_t.ndim
    key = (ndim, str(device))
    if key not in _mode_kernel_cache:
        if ndim == 2:
            _mode_kernel_cache[key] = torch.ones(1, 1, 3, 3, dtype=torch.float32, device=device)
        elif ndim == 3:
            _mode_kernel_cache[key] = torch.ones(1, 1, 3, 3, 3, dtype=torch.float32, device=device)
        else:
            raise ValueError(f"_mode_filter_gpu: unsupported ndim={ndim}, expected 2 or 3")
    kernel = _mode_kernel_cache[key]
    conv = _F.conv2d if ndim == 2 else _F.conv3d

    lbl_long = lbl_t.long()
    # Use max()+arange instead of unique() to avoid the sort-based GPU sync
    # (unique sort costs ~1ms; max+arange costs ~0.05ms).
    # For sparse crops some arange entries may not exist — that is fine;
    # their binary masks are all-zero and contribute nothing to counts.
    max_label = lbl_long.max().item()
    if max_label == 0:
        return lbl_t
    uniq = torch.arange(1, max_label + 1, device=lbl_t.device, dtype=torch.long)

    # (N_labels, *spatial) binary masks, one per label
    masks = (lbl_long.unsqueeze(0) == uniq.view(-1, *([1] * ndim))).float()
    # Count neighbors: (N_labels, 1, *spatial) → squeeze → (N_labels, *spatial)
    counts = conv(masks.unsqueeze(1), kernel, padding=1).squeeze(1)
    best = counts.argmax(dim=0)          # (*spatial) — index into uniq
    mode_map = uniq[best].float()

    fg = lbl_t > 0
    return torch.where(fg & (mode_map == 0), lbl_t, mode_map)


def _gaussian_blur_gpu(img_t, sigma):
    """Apply Gaussian blur to a (*spatial) tensor (2D or 3D).

    Matches scipy.ndimage.gaussian_filter with truncate=4.0.
    Kernel is cached by (quantized sigma, ndim, device).
    """
    if sigma <= 0:
        return img_t
    device = img_t.device
    ndim = img_t.ndim
    sigma_q = round(float(sigma), 2)
    if sigma_q <= 0:
        return img_t
    key = (sigma_q, ndim, str(device))
    if key not in _blur_kernel_cache:
        radius = int(4.0 * sigma_q + 0.5)
        ks = 2 * radius + 1
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        k1d = torch.exp(-0.5 * (x / sigma_q) ** 2)
        k1d /= k1d.sum()
        if ndim == 2:
            kernel = (k1d.unsqueeze(0) * k1d.unsqueeze(1)).view(1, 1, ks, ks)
        elif ndim == 3:
            kernel = (k1d.view(ks, 1, 1) * k1d.view(1, ks, 1) * k1d.view(1, 1, ks)).view(1, 1, ks, ks, ks)
        else:
            raise ValueError(f"_gaussian_blur_gpu: unsupported ndim={ndim}, expected 2 or 3")
        _blur_kernel_cache[key] = kernel
    kernel = _blur_kernel_cache[key]
    conv = _F.conv2d if ndim == 2 else _F.conv3d

    inp = img_t.unsqueeze(0).unsqueeze(0)                            # (1, 1, *spatial)
    return conv(inp, kernel, padding=kernel.shape[-1] // 2).squeeze(0).squeeze(0)

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
def random_rotate_and_resize(X, Y=None, scale_range=1., gamma_range=[.5, 4], tyx=(224, 224),
                             do_flip=True, rescale_factor=None, inds=None, nchan=1, allow_blank_masks=False,
                             return_meta=False, device=None):

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
    scale = np.zeros((nimg, dim), np.float32)

    # Fall back to CPU if the device doesn't support 3D grid_sample.
    use_torch = (device is not None) and (dim <= 2 or _supports_grid3d(device))

    if not use_torch:
        imgi = np.zeros((nimg, nchan) + tyx, np.float32)
        lbl  = np.zeros((nimg,) + tyx, np.float32)

    v1 = [0] * (dim - 1) + [1]
    v2 = [0] * (dim - 2) + [1, 0]

    if rescale_factor is None:
        rescale = np.ones(nimg, np.float32)
    elif np.isscalar(rescale_factor):
        rescale = np.ones(nimg, np.float32) * rescale_factor
    else:
        rescale = np.array(rescale_factor)

    meta_list = [] if return_meta else None
    imgi_list = [] if use_torch else None
    lbl_list  = [] if use_torch else None

    for n in range(nimg):
        img = X[n].copy()
        y = None if Y is None else Y[n]
        ind = None if inds is None else inds[n]
        if return_meta:
            out = random_crop_warp(
                img, y, tyx, v1, v2, nchan, rescale[n],
                scale_range, gamma_range, do_flip, ind,
                allow_blank_masks=allow_blank_masks,
                return_meta=True, device=device,
            )
            imgi_n, lbl_n, scale[n], meta = out
            meta_list.append(meta)
        else:
            imgi_n, lbl_n, scale[n] = random_crop_warp(
                img, y, tyx, v1, v2, nchan, rescale[n],
                scale_range, gamma_range, do_flip, ind,
                allow_blank_masks=allow_blank_masks,
                device=device,
                _defer_batch_aug=use_torch,
            )
        if use_torch:
            imgi_list.append(imgi_n)
            lbl_list.append(lbl_n)
        else:
            imgi[n] = imgi_n
            lbl[n]  = lbl_n

    if use_torch:
        imgi_t = torch.stack(imgi_list)   # (N, C, H, W)
        lbl_t  = torch.stack(lbl_list)    # (N, H, W)

        # ── Batch-level augmentations (only for non-meta deferred path) ─────────
        if not return_meta:
            N, C = imgi_t.shape[:2]
            nc = N * C
            trailing = [1] * dim   # (1,) × dim for broadcasting over spatial dims

            # Gamma — drawn FIRST to maintain the same CPU-RNG position as before
            # (extra noise/S&P draws below must not shift the gamma sequence).
            gammas_flat = _ttriangular(gamma_range[0], 1, gamma_range[1], size=nc)
            gammas_NC = torch.tensor(
                gammas_flat, dtype=torch.float32, device=device
            ).view(N, C, *trailing)
            imgi_t = imgi_t.clamp(min=0).pow(gammas_NC)

            # [3] Noise — one batched draw beats N*C per-channel draws.
            noise_aug = torch.from_numpy(_tchoice2_vec(nc).reshape(N, C)).to(device)
            noise_vars = torch.tensor(
                _ttriangular(1e-8, 1e-8, 1e-2, size=nc), dtype=torch.float32
            ).view(N, C, *trailing).to(device).sqrt()
            noise = torch.randn(*imgi_t.shape, device=device) * noise_vars
            imgi_t = torch.where(noise_aug.view(N, C, *trailing).bool(),
                                 (imgi_t + imgi_t * noise).clamp(0, 1), imgi_t)

            # [6] S&P — one batched draw beats N*C per-channel draws.
            sp_aug = torch.from_numpy(_tchoice2_vec(nc).reshape(N, C)).to(device)
            sp_mask = (torch.rand(*imgi_t.shape, device=device) < 0.001) & sp_aug.view(N, C, *trailing).bool()
            sp_vals = torch.randint(0, 2, imgi_t.shape, device=device).float()
            imgi_t = torch.where(sp_mask, sp_vals, imgi_t)

        imgi = imgi_t
        lbl  = lbl_t

    if return_meta:
        return imgi, lbl, np.mean(scale), meta_list
    return imgi, lbl, np.mean(scale)


def random_crop_warp(img, Y, tyx, v1, v2, nchan, rescale_factor, scale_range, gamma_range, do_flip, ind,
                     do_labels=True, depth=0, augment=True, allow_blank_masks=False, return_meta=False,
                     device=None, _defer_batch_aug=False):
    """
    Sub-function of `random_rotate_and_resize()` that recursively crops until a minimum number of
    cell pixels are found, then applies augmentations.

    When ``device`` is a GPU torch.device, all heavy compute (affine warp, mode filter,
    intensity ops, flips) runs on the GPU for both 2D and 3D data.  All numpy.random draws
    are preserved in identical order regardless of which path is used, so results are
    reproducible when seeded.

    Returns numpy arrays when device is None/CPU, torch tensors on ``device`` otherwise.
    """
    dim = len(tyx)
    # Fall back to CPU if the device doesn't support 3D grid_sample.
    use_torch = (device is not None) and (dim <= 2 or _supports_grid3d(device))

    if depth > 100:
        raise ValueError(
            "Sparse or over-dense image detected. "
            f"Problematic index is: {ind}. Image shape is: {img.shape}. "
            f"tyx is: {tyx}. rescale_factor is {rescale_factor}"
        )
    if depth > 200:
        raise ValueError(
            "Recursion depth exceeded. Check that your images contain cells and "
            f"background within a typical crop. Failed index is: {ind}."
        )

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
        if use_torch:
            scale = a + (b - a) * _tbeta(alpha, beta, size=dim)
        else:
            scale = a + (b - a) * np.random.beta(alpha, beta, size=dim)

        if rescale_factor is not None:
            scale *= 1. / rescale_factor
    else:
        scale = 1

    s = img.shape[-dim:]

    if use_torch:
        theta = _trand() * np.pi * 2
    else:
        theta = np.random.rand() * np.pi * 2

    rot = mgen.rotation_from_angle_and_plane(-theta, v2, v1)
    M_inv = np.diag(1. / scale).dot(rot.T)

    axes = range(dim)
    s = img.shape[-dim:]
    if use_torch:
        rt = _trand(dim) - 0.5
    else:
        rt = np.random.rand(dim,) - .5
    dxy = [rt[a] * (np.maximum(0, s[a] - tyx[a])) for a in axes]

    c_in = 0.5 * np.array(s) + dxy
    c_out = 0.5 * np.array(tyx)
    offset = c_in - np.dot(M_inv, c_out)

    if use_torch:
        mode = _MODE_CHOICES[torch.randint(0, 3, (1,)).item()]
    else:
        mode = np.random.choice(['constant', 'nearest', 'mirror'])

    # ── Warp label ────────────────────────────────────────────────────────
    if use_torch:
        grid = _build_grid_nd(M_inv, offset, s, tyx, device)
        padding_mode = _SCIPY_TO_TORCH_PAD[mode]
        lbl_t = torch.tensor(labels.astype(np.float32), device=device).unsqueeze(0).unsqueeze(0)
        # Use nearest for label warp; fall back to bilinear+round if nearest is unsupported.
        lbl_interp = 'nearest' if (dim <= 2 or _supports_grid3d(device, mode='nearest')) else 'bilinear'
        lbl_t = _F.grid_sample(lbl_t, grid, mode=lbl_interp,
                               padding_mode=padding_mode, align_corners=True)
        lbl_t = lbl_t.squeeze(0).squeeze(0)                    # (*tyx) float
        if lbl_interp == 'bilinear':
            lbl_t = lbl_t.round()  # restore integer label values
        # Blank-mask check via GPU scalar compare (avoids full H×W CPU transfer).
        if not allow_blank_masks and (lbl_t.max() == lbl_t.min()).item():
            return random_crop_warp(
                img, Y, tyx, v1, v2, nchan, rescale_factor, scale_range,
                gamma_range, do_flip, ind, do_labels, depth=depth + 1,
                augment=augment, allow_blank_masks=allow_blank_masks,
                return_meta=return_meta, device=device, _defer_batch_aug=_defer_batch_aug,
            )
        lbl_t = _mode_filter_gpu(lbl_t)
        # numpy lbl is only needed for diameters() in OpenSimplex illumination; compute lazily.
        _lbl_np = [None]

        def _get_lbl_np():
            if _lbl_np[0] is None:
                _lbl_np[0] = lbl_t.cpu().numpy().astype(np.int32)
            return _lbl_np[0]

    else:
        grid = None
        lbl = do_warp(labels, M_inv, tyx, offset=offset, order=0, mode=mode)
        # ── Blank-mask retry ─────────────────────────────────────────────
        if len(fastremap.unique(lbl)) < 2 and not allow_blank_masks:
            return random_crop_warp(
                img, Y, tyx, v1, v2, nchan, rescale_factor, scale_range,
                gamma_range, do_flip, ind, do_labels, depth=depth + 1,
                augment=augment, allow_blank_masks=allow_blank_masks,
                return_meta=return_meta, device=device, _defer_batch_aug=_defer_batch_aug,
            )
        lbl = mode_filter(lbl)

    # ── Warp + augment each channel ────────────────────────────────────────
    if use_torch:
        # Use original label to determine foreground — avoids GPU sync.
        # (Equivalent for typical crops; blank warps handled by the retry above.)
        has_foreground = bool(np.any(labels)) if Y is not None else False

        # Draw aug choices BEFORE arr_stack so we can skip nv when not needed.
        all_aug = _tchoice2_vec(8 * nchan).reshape(nchan, 8)  # (C, 8)

        # Rescale all channels on CPU, then warp ALL at once with one grid_sample call.
        # np.unique/fastremap.unique is skipped when has_foreground=True (most cases).
        arr_stack = np.empty((nchan,) + tuple(img.shape[-dim:]), dtype=np.float32)
        for k in range(nchan):
            if has_foreground:
                arr_stack[k] = rescale(img[k])
            else:
                # Rare path: no foreground — need nv for special low-bit scaling.
                if np.issubdtype(img[k].dtype, np.integer):
                    nv = len(fastremap.unique(img[k]))
                else:
                    nv = len(np.unique(img[k]))
                maxv = np.iinfo(img[k].dtype).max if np.issubdtype(img[k].dtype, np.integer) else 1.0
                a = img[k].astype(np.float32) / maxv
                if nv < maxv // 100 and a.max() == 1.0:
                    a = a * ((nv / maxv) * 10)
                arr_stack[k] = a

        # One batched grid_sample for all channels — replaces C separate calls.
        arr_t = torch.tensor(arr_stack, dtype=torch.float32, device=device).unsqueeze(1)  # (C,1,*s_in)
        grid_C = grid.expand(nchan, *grid.shape[1:])                                      # (C,*tyx,dim)
        imgi_t = _F.grid_sample(arr_t, grid_C, mode='bilinear',
                                padding_mode=padding_mode, align_corners=True).squeeze(1)  # (C,*tyx)

        # Collect per-channel gammas; apply vectorised after the loop.
        gammas = []

        # Pre-compute border index tensor (cached).
        _border_inds_np = None

        for k in range(nchan):
            aug_choices = all_aug[k]

            if augment:
                # [1] Gaussian blur
                if aug_choices[1]:
                    sigma = _tuniform(0, np.sqrt(2))
                    imgi_t[k] = _gaussian_blur_gpu(imgi_t[k], sigma)

                # [2] Contrast clipping
                if aug_choices[2] and has_foreground:
                    dp = .1
                    dpct = _ttriangular(0, 0, dp, size=2)
                    imgi_t[k] = normalize99(imgi_t[k], upper=100 - dpct[0], lower=dpct[1])

                # [0] Illumination field (OpenSimplex is CPU-only)
                if aug_choices[0] and has_foreground:
                    if aug_choices[7] or not OPEN_SIMPLEX_ENABLED:
                        axis = _trandint(0, dim)
                        axis_size = tyx[axis]
                        illum_1d = np.linspace(0, 1, axis_size, dtype=np.float32)
                        shape_nd = [1] * dim
                        shape_nd[axis] = axis_size
                        illum_field = illum_1d.reshape(shape_nd)  # broadcasts along correct axis
                    else:
                        simplex_seed = _trandint(0, 2 ** 31)
                        simplex = OpenSimplex(seed=simplex_seed)
                        spatial_shape = tyx[-2:]
                        lbl_np = _get_lbl_np()
                        mean_obj_diam = 2 * diameters(lbl_np) if np.any(lbl_np) else 1.0
                        freq_jitter = _ttriangular(1, 1.0, 10.0)
                        fs = mean_obj_diam * freq_jitter
                        coords = [np.arange(0, _s, dtype=np.float32) / fs for _s in spatial_shape[::-1]]
                        illum_field = rescale(simplex.noise2array(*coords))

                    min_factor = _ttriangular(0, 0, 1) ** .5
                    mult_t = torch.tensor(
                        (min_factor + (1.0 - min_factor) * illum_field).astype(np.float32),
                        device=device)
                    # Keep denom as GPU tensor — no .item() sync needed.
                    denom = imgi_t[k].max() + min_factor
                    imgi_t[k] = (imgi_t[k] + min_factor) / denom.clamp(min=1e-8) * mult_t

                # [3] Gaussian noise — deferred to batch level when _defer_batch_aug.
                if aug_choices[3] and not _defer_batch_aug:
                    var_range = 1e-2
                    var = _ttriangular(1e-8, 1e-8, var_range, size=1)
                    sigma_n = float(np.sqrt(var[0]))
                    noise_t = torch.randn(tyx, device=device, dtype=torch.float32) * sigma_n
                    imgi_t[k] = (imgi_t[k] + imgi_t[k] * noise_t).clamp(0, 1)

                # [4] Bit shifting — nv computed lazily (skipped ~50% of the time).
                if aug_choices[4] and has_foreground:
                    if np.issubdtype(img[k].dtype, np.integer):
                        nv = len(fastremap.unique(img[k]))
                    else:
                        nv = len(np.unique(img[k]))
                    raw_bits = int(np.ceil(np.log2(max(nv, 1))))
                    min_bits = 3
                    max_shift = max(0, min(14, raw_bits - min_bits))
                    bit_shift = int(_ttriangular(0, max_shift // 2, max_shift)) if max_shift else 0
                    if bit_shift:
                        im_int = (imgi_t[k] * 65535).clamp(0, 65535).to(torch.int32)
                        shifted = torch.div(im_int, 2 ** bit_shift, rounding_mode='floor').float()
                        mn, mx = shifted.min(), shifted.max()
                        imgi_t[k] = ((shifted - mn) / (mx - mn + 1e-8)).clamp(0, 1)

                # [5] Border darkening — precomputed bool mask avoids index_put_ scatter.
                if aug_choices[5]:
                    factor = _tuniform(0, 1)
                    bkey = (tyx, str(device))
                    if bkey not in _border_mask_cache:
                        if _border_inds_np is None:
                            _border_inds_np = utils.border_indices(tyx)
                        mask_flat = torch.zeros(int(np.prod(tyx)), dtype=torch.bool, device=device)
                        mask_flat[torch.tensor(_border_inds_np, dtype=torch.long, device=device)] = True
                        _border_mask_cache[bkey] = mask_flat.view(tyx)
                    border_mask = _border_mask_cache[bkey]
                    imgi_t[k] = torch.where(border_mask, imgi_t[k] * factor, imgi_t[k])

                # [6] S&P — deferred to batch level when _defer_batch_aug.
                if aug_choices[6] and not _defer_batch_aug:
                    sp_mask_t = torch.rand(tyx, device=device) < 0.001
                    sp_vals_t = torch.randint(0, 2, tyx, device=device).float()
                    imgi_t[k] = torch.where(sp_mask_t, sp_vals_t, imgi_t[k])

            # Gamma — deferred to batch level when _defer_batch_aug
            if not _defer_batch_aug:
                gammas.append(_ttriangular(gamma_range[0], 1, gamma_range[1]))

        # Gamma correction for ALL channels in one op (skipped when deferred).
        if not _defer_batch_aug:
            gammas_t = torch.tensor(gammas, dtype=torch.float32, device=device).view(nchan, *([1]*dim))
            imgi_t = imgi_t.clamp(min=0).pow(gammas_t)

    else:
        has_foreground = bool(np.any(lbl)) if Y is not None else False
        imgi = np.zeros((nchan,) + tyx, np.float32)
        _border_inds_np = None

        for k in range(img.shape[0]):
            has_foreground = bool(np.any(lbl))
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
                    sigma = np.random.uniform(0, np.sqrt(2))
                    imgi[k] = gaussian_filter(imgi[k], sigma)
                if aug_choices[2] and has_foreground:
                    dp = .1
                    dpct = np.random.triangular(left=0, mode=0, right=dp, size=2)
                    imgi[k] = normalize99(imgi[k], upper=100 - dpct[0], lower=dpct[1])
                if aug_choices[0] and has_foreground:
                    if aug_choices[7] or not OPEN_SIMPLEX_ENABLED:
                        axis = np.random.randint(0, dim)
                        axis_size = tyx[axis]
                        illum_1d = np.linspace(0, 1, axis_size, dtype=np.float32)
                        shape_nd = [1] * dim
                        shape_nd[axis] = axis_size
                        illum_field = illum_1d.reshape(shape_nd)  # broadcasts along correct axis
                    else:
                        simplex_seed = np.random.randint(0, 2 ** 31)
                        simplex = OpenSimplex(seed=simplex_seed)
                        spatial_shape = tyx[-2:]
                        mean_obj_diam = 2 * diameters(lbl) if np.any(lbl) else 1.0
                        freq_jitter = np.random.triangular(left=1, mode=1.0, right=10.0)
                        fs = mean_obj_diam * freq_jitter
                        coords = [np.arange(0, _s, dtype=np.float32) / fs for _s in spatial_shape[::-1]]
                        illum_field = rescale(simplex.noise2array(*coords))
                    min_factor = np.random.triangular(left=0, mode=0, right=1) ** .5
                    multiplier = (min_factor + (1.0 - min_factor) * illum_field).astype(np.float32)
                    imgi[k] = (imgi[k] + min_factor) / (imgi[k].max() + min_factor) * multiplier
                if aug_choices[3] and SKIMAGE_ENABLED:
                    var = np.random.triangular(left=1e-8, mode=1e-8, right=1e-2, size=1)
                    sigma_n = float(np.sqrt(var).item())
                    noise_np = np.random.normal(0.0, sigma_n, size=tyx).astype(np.float32)
                    imgi[k] = np.clip(imgi[k] + imgi[k] * noise_np, 0, 1)
                if aug_choices[4] and has_foreground:
                    min_bits = 3
                    max_shift = max(0, min(14, raw_bits - min_bits))
                    bit_shift = int(np.random.triangular(0, max_shift // 2, max_shift)) if max_shift else 0
                    if bit_shift:
                        im = utils.to_16_bit(imgi[k])
                        imgi[k] = rescale(im >> bit_shift)
                if aug_choices[5]:
                    factor = np.random.uniform(0, 1)
                    if _border_inds_np is None:
                        _border_inds_np = utils.border_indices(tyx)
                    imgi[k].flat[_border_inds_np] *= factor
                if aug_choices[6]:
                    sp_mask_np = np.random.rand(*tyx) < 0.001
                    n_sp = int(sp_mask_np.sum())
                    imgi[k][sp_mask_np] = np.random.choice([0, 1], size=n_sp).astype(np.float32)

            gamma = np.random.triangular(left=gamma_range[0], mode=1, right=gamma_range[1])
            imgi[k] = imgi[k] ** gamma

    # ── Flips ──────────────────────────────────────────────────────────────
    if do_flip:
        for d in range(1, dim + 1):
            flip = bool(_tchoice2()) if use_torch else bool(np.random.choice([0, 1]))
            if flip:
                if use_torch:
                    imgi_t = torch.flip(imgi_t, dims=[-d])
                    if Y is not None:
                        lbl_t = torch.flip(lbl_t, dims=[-d])
                else:
                    imgi = np.flip(imgi, axis=-d)
                    if Y is not None:
                        lbl = np.flip(lbl, axis=-d)

    if return_meta:
        meta = {"shape_in": s, "offset": offset, "M_inv": M_inv}
        if dim == 2:
            corners_out = np.array(
                [[0, 0], [0, tyx[1] - 1], [tyx[0] - 1, 0], [tyx[0] - 1, tyx[1] - 1]],
                dtype=np.float32,
            )
            corners_in = np.stack([M_inv.dot(c) + offset for c in corners_out], axis=0)
            ymin, xmin = corners_in.min(axis=0)
            ymax, xmax = corners_in.max(axis=0)
            meta["corners_in"] = corners_in
            meta["bbox_in"] = (float(ymin), float(xmin), float(ymax), float(xmax))
        if use_torch:
            return imgi_t, lbl_t, scale, meta
        return imgi, lbl, scale, meta

    if use_torch:
        return imgi_t, lbl_t, scale
    return imgi, lbl, scale


def do_warp(A, M_inv, tyx, offset=0, order=1, mode='constant', **kwargs):
    """Wrapper function for affine transformations during augmentation."""
    return affine_transform(A, M_inv, offset=offset,
                            output_shape=tyx, order=order,
                            mode=mode, **kwargs)
