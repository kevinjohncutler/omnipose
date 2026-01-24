import numpy as np
import matplotlib as mpl
from skimage import img_as_ubyte

from ..utils.color import sinebow
from ..transforms.normalize import rescale, safe_divide, normalize99
from ..transforms.vector import torch_norm


def colorize(im, colors=None, color_weights=None, offset=0, channel_axis=-1):
    N = len(im)
    if colors is None:
        angle = np.arange(0, 1, 1 / N) * 2 * np.pi + offset
        angles = np.stack((angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3), axis=-1)
        colors = (np.cos(angles) + 1) / 2

    if color_weights is not None:
        colors *= np.expand_dims(color_weights, -1)

    rgb_shape = im.shape[1:] + (colors.shape[1],)
    if channel_axis == 0:
        rgb_shape = rgb_shape[::-1]
    rgb = np.zeros(rgb_shape)

    # Use broadcasting to multiply im and colors and sum along the 0th dimension
    rgb = (np.expand_dims(im, axis=-1) * colors.reshape(colors.shape[0], 1, 1, colors.shape[1])).mean(axis=0)

    return rgb


def colorize_GPU(im, colors=None, color_weights=None, intervals=None, offset=0):
    import torch
    import string
    from opt_einsum import contract

    C = im.shape[0]  # Number of input channels
    device = im.device

    # Determine the number of spatial dimensions
    num_spatial_dims = im.ndim - 1  # Exclude the channel dimension

    # Generate index letters for einsum (excluding 'c' and 'l')
    idx_letters = ''.join(letter for letter in string.ascii_lowercase if letter not in {'c', 'l'})
    spatial_indices = idx_letters[:num_spatial_dims]

    # Build einsum indices dynamically
    im_indices = 'c' + spatial_indices
    aggregator_indices = 'cN'  # 'N' corresponds to the interval/bin dimension
    colors_indices = 'cl'  # Colors indexed by channel and RGB
    output_indices = 'N' + spatial_indices + 'l'
    einsum_eq = f'{im_indices},{aggregator_indices},{colors_indices}->{output_indices}'

    if intervals is None:
        intervals = [C]

    N = len(intervals)  # Number of intervals

    aggregator = torch.zeros(C, N, device=device)
    start = 0
    for i, length in enumerate(intervals):
        aggregator[start:start + length, i] = 1 / length
        start += length

    # Generate colors if not provided
    if colors is None:
        angle = torch.linspace(0, 1, C, device=device) * 2 * np.pi + offset
        angles = torch.stack((angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3), dim=-1)
        colors = (torch.cos(angles) + 1) / 2  # Generate RGB colors for intervals or channels

    # Apply color weights if provided
    if color_weights is not None:
        colors *= color_weights.unsqueeze(-1)

    # Perform einsum operation
    rgb = contract(einsum_eq, im.float(), aggregator.float(), colors.float())

    # Squeeze the interval dimension if no intervals are used
    if N == 1:
        rgb = rgb.squeeze(0)  # Shape: [spatial_dims..., 3]

    return rgb


def colorize_dask_fast(im_dask, colors=None, color_weights=None, intervals=None, offset=0):
    import dask.array as da

    C = im_dask.shape[0]
    spatial_shape = im_dask.shape[1:]

    if intervals is None:
        intervals = [C]
    N = len(intervals)

    # Precompute aggregator matrix (C, N)
    aggregator = np.zeros((C, N), dtype=np.float32)
    start = 0
    for i, size in enumerate(intervals):
        aggregator[start:start + size, i] = 1.0 / size
        start += size

    # Compute colors (C, 3)
    if colors is None:
        angle = np.linspace(0, 1, C, endpoint=False) * 2 * np.pi + offset
        angles = np.stack([angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3], axis=-1)
        colors = (np.cos(angles) + 1.0) / 2.0
    if color_weights is not None:
        colors *= color_weights[:, None]

    # Precompute final weighting matrix: (C, N, 3)
    weights = aggregator[..., None] * colors[:, None, :]  # shape (C, N, 3)

    # Collapse color dimensions early: (C, N*3)
    weights_flat = weights.reshape(C, N * 3)

    # Flatten input to shape (C, ZYX)
    im_flat = im_dask.reshape(C, -1)

    # Matrix multiplication: (N*3, ZYX)
    out_flat = da.dot(weights_flat.T, im_flat)

    # Reshape to (N, 3, Z, Y, X)
    out = out_flat.reshape(N, 3, *spatial_shape)

    # Move color channel to last axis: (N, Z, Y, X, 3)
    out = da.moveaxis(out, 1, -1)

    # If single interval, squeeze it out
    if N == 1:
        out = out.squeeze(axis=0)

    return out


def colorize_dask(im_dask, colors=None, color_weights=None, intervals=None, offset=0):
    import dask
    from opt_einsum import contract

    # Number of channels
    C = im_dask.shape[0]
    spatial_shape = im_dask.shape[1:]
    spatial_size = np.prod(spatial_shape)

    # Interval setup
    if intervals is None:
        intervals = [C]
    N = len(intervals)

    # Build aggregator: shape (C, N)
    aggregator = np.zeros((C, N), dtype=np.float32)
    start = 0
    for i, size in enumerate(intervals):
        aggregator[start:start + size, i] = 1.0 / size
        start += size

    # Default color generation: shape (C, 3)
    if colors is None:
        angle = np.linspace(0, 1, C, endpoint=False) * 2 * np.pi + offset
        angles = np.stack([angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3], axis=-1)
        colors = (np.cos(angles) + 1.0) / 2.0

    # Apply any color weights
    if color_weights is not None:
        colors *= color_weights[:, None]

    # Combine aggregator and colors: shape (C, N, 3)
    # Keep this as a NumPy array to avoid a big Dask overhead
    agg_colors = aggregator[..., None] * colors[:, None, :]

    # Flatten input from (C, Z, Y, X) -> (C, Z*Y*X)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        im_flat = im_dask.reshape(C, spatial_size)

    # Perform a single einsum contraction:
    # cX * cNr -> NXr
    # c -> channel axis, X -> flattened spatial axis, N-> interval groups, r -> RGB
    out_flat = contract('cX,cNr->NXr', im_flat, agg_colors)

    # Reshape from (N, X, 3) -> (N, Z, Y, X, 3)
    out = out_flat.reshape(N, *spatial_shape, 3)

    # For a single interval, remove the interval dimension
    if N == 1:
        out = out[0]  # shape (Z, Y, X, 3)

    return out


def colorize_dask_matmul(im_dask, colors=None, color_weights=None, intervals=None, offset=0):
    """
    A faster version of colorize_dask that uses a single matrix multiply instead
    of explicit loops or opt_einsum for the core contraction step.
    """
    import dask

    # Number of channels
    C = im_dask.shape[0]
    spatial_shape = im_dask.shape[1:]
    spatial_size = np.prod(spatial_shape)

    # Interval setup
    if intervals is None:
        intervals = [C]
    N = len(intervals)

    # Build aggregator: shape (C, N)
    aggregator = np.zeros((C, N), dtype=np.float32)
    start = 0
    for i, size in enumerate(intervals):
        aggregator[start:start + size, i] = 1.0 / size
        start += size

    # Default color generation: shape (C, 3)
    if colors is None:
        angle = np.linspace(0, 1, C, endpoint=False) * 2 * np.pi + offset
        angles = np.stack([angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3], axis=-1)
        colors = (np.cos(angles) + 1.0) / 2.0  # shape (C, 3)

    # Apply any color weights
    if color_weights is not None:
        colors = colors * color_weights[:, None]

    # aggregator (C, N), colors (C, 3)
    # aggregator[..., None] * colors => shape (C, N, 3)
    # then collapse to shape (C, N*3) so that a single matrix multiply can be used
    combined = (aggregator[..., None] * colors[:, None, :]).reshape(C, N * 3).astype(np.float32)

    # Flatten the input image: (C, Z, Y, X) -> (C, Z*Y*X)
    # Casting to float32 can help ensure everything matches for the matrix multiply
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        im_flat = im_dask.reshape(C, spatial_size, merge_chunks=True).astype(np.float32)

    # Matrix multiplication:
    #   im_flat^T is shape (X, C)
    #   combined is shape (C, N*3)
    # => result is shape (X, N*3)
    out_mat = im_flat.T @ combined

    # Reshape:
    #   out_mat:  (X, N*3)
    #   => (X, N, 3) => transpose to (N, X, 3)
    out_flat = out_mat.reshape(spatial_size, N, 3).transpose(1, 0, 2)

    # Finally shape it to (N, Z, Y, X, 3)
    out = out_flat.reshape(N, *spatial_shape, 3)

    # If only one interval, remove that dimension to keep the same behavior
    if N == 1:
        out = out[0]  # shape (Z, Y, X, 3)

    return out


def apply_ncolor(masks, offset=0, cmap=None, max_depth=20, expand=True, maxv=1, greedy=False):
    import ncolor
    from cmap import Colormap

    cmap = Colormap(cmap) if isinstance(cmap, str) else cmap

    m, n = ncolor.label(
        masks,
        max_depth=max_depth,
        return_n=True,
        conn=2,
        expand=expand,
        greedy=greedy,
    )
    if cmap is None:
        c = sinebow(n, offset=offset)
        colors = np.array(list(c.values()))
        cmap = mpl.colors.ListedColormap(colors)
        return cmap(m)
    else:
        return cmap(rescale(m) / maxv)


def rgb_flow(dP, transparency=True, mask=None, norm=True, device=None):
    """Meant for stacks of dP, unsqueeze if using on a single plane."""
    import torch
# normalize99 and safe_divide imported from transforms.normalize

    if device is None:
        device = torch.device('cpu')

    if isinstance(dP, torch.Tensor):
        device = dP.device
    else:
        dP = torch.from_numpy(dP).to(device)

    mag = torch_norm(dP, dim=1)
    dP = safe_divide(dP, mag.unsqueeze(1))  # Normalize the flow vectors

    if norm:
        mag = normalize99(mag)

    vecs = dP[:, 0] + dP[:, 1] * 1j
    roots = torch.exp(1j * np.pi * (2 * torch.arange(3, device=device) / 3 + 1))
    rgb = (torch.real(vecs.unsqueeze(-1) * roots.view(1, 1, 1, -1)) + 1) / 2

    if transparency:
        im = torch.cat((rgb, mag[..., None]), dim=-1)
    else:
        im = rgb * mag[..., None]

    im = (torch.clamp(im, 0, 1) * 255).type(torch.uint8)
    return im


def create_colormap(image, labels):
    """
    Create a colormap based on the average color of each label in the image.

    Parameters
    ----------
    image: ndarray
        An RGB image.
    labels: ndarray
        A 2D array of labels corresponding to the image.

    Returns
    -------
    colormap: ndarray
        A colormap where each row is the RGB color for the corresponding label.
    """
    # Ensure the image is in the range 0-255
    image = img_as_ubyte(image)

    # Initialize an array to hold the RGB color for each label
    colormap = np.zeros((labels.max() + 1, 3), dtype=np.uint8)

    # Calculate the average color for each label
    for label in np.unique(labels):
        mask = labels == label
        colormap[label] = image[mask].mean(axis=0)

    return colormap


def color_from_RGB(im, rgb, m, bd=None, mode='inner', connectivity=2):
    from skimage import color
    import ncolor

    if bd is None:
        from skimage.segmentation import find_boundaries
        bd = find_boundaries(m, mode=mode, connectivity=connectivity)

    alpha = (m > 0) * .5
    alpha[bd] = 1
    alpha = np.stack([alpha] * 3, axis=-1)
    m = ncolor.format_labels(m)
    cmap = create_colormap(rgb, m)
    clrs = rescale(cmap[1:])
    overlay = color.label2rgb(
        m,
        im,
        clrs,
        bg_label=0,
        alpha=alpha,
    )
    return overlay


def dx_to_circ(dP, transparency=False, mask=None, sinebow=1, iso=0, iso_map="oklch", offset=0, norm=True):
    """Convert flow field to an RGB(A) visualization."""
    import colorsys
    from .. import transforms

    dP = np.array(dP)
    mag = np.sqrt(np.sum(dP**2, axis=0))
    if norm:
        mag = np.clip(transforms.normalize99(mag, omni=True), 0, 1.0)[..., np.newaxis]

    angles = np.arctan2(dP[1], dP[0]) + np.pi
    angles = (angles + offset) % (2 * np.pi)

    if sinebow or iso:
        angles_shifted = np.stack(
            [angles, angles + 2 * np.pi / 3, angles + 4 * np.pi / 3],
            axis=-1,
        )
        rgb = (np.cos(angles_shifted) + 1) / 2
    else:
        r, g, b = colorsys.hsv_to_rgb(angles, 1, 1)
        rgb = np.stack((r, g, b), axis=0)

    if transparency:
        im = np.concatenate((rgb, mag), axis=-1)
    else:
        im = rgb * mag

    if mask is not None and transparency and dP.shape[0] < 3:
        im[:, :, -1] *= mask

    return (np.clip(im, 0, 1) * 255).astype(np.uint8)
