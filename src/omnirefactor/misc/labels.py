from .imports import *

def nd_grid_hypercube_labels(shape: Sequence[int],
                             side: int,
                             *,
                             center: bool = True,
                             dtype=np.int32) -> np.ndarray:
    """
    Label an ND array with equal-side hypercubes of edge length `side` pixels.

    Parameters
    ----------
    shape : sequence of int
        Target array shape (H, W, D, ...).
    side : int
        Edge length of each hypercube in pixels (same along all axes).
    center : bool, default True
        Center the grid inside the array; leftover margins get label 0.
    dtype : numpy dtype, default np.int32
        Output dtype.

    Returns
    -------
    labels : ndarray
        Integer label map of shape `shape` with values in {0, 1..K}.
    """
    shape = np.asarray(shape, dtype=int)
    if shape.ndim != 1:
        raise ValueError("shape must be 1D sequence of ints")
    if not isinstance(side, (int, np.integer)) or side < 1:
        raise ValueError("side must be a positive integer")

    counts = shape // side
    if np.any(counts <= 0):
        raise ValueError("side too large for at least one axis")

    grid_span = counts * side
    offsets = ((shape - grid_span) // 2) if center else np.zeros_like(shape)

    # Build per-axis indices and in-bounds mask
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    idx_axes = []
    in_bounds = np.ones(tuple(shape), dtype=bool)
    for ax, g in enumerate(grids):
        rel = g - offsets[ax]
        mask = (rel >= 0) & (rel < grid_span[ax])
        in_bounds &= mask
        idx_axes.append((rel // side).astype(int))

    # Row-major linearization
    lin = np.zeros(tuple(shape), dtype=int)
    stride = 1
    for ax in range(shape.size - 1, -1, -1):
        lin += idx_axes[ax] * stride
        stride *= counts[ax]

    labels = np.zeros(tuple(shape), dtype=dtype)
    labels[in_bounds] = lin[in_bounds] + 1
    return labels

def make_label_matrix(N: int, M: int) -> np.ndarray:
    """
    General ND label matrix.
    
    Shape = (2*M,)*N
    Each axis is split into two halves of length M.
    The label is the binary code of the half-indices.
    
    Example:
      N=1 → [0...0,1...1]
      N=2 → quadrants labeled 0..3
      N=3 → octants labeled 0..7
      N=4 → 16 hyper-quadrants labeled 0..15
    """
    if N < 1:
        raise ValueError("N must be >=1")
    # build index grids
    grids = np.ogrid[tuple(slice(0,2*M) for _ in range(N))]
    labels = np.zeros((2*M,)*N, dtype=int)
    for axis, g in enumerate(grids):
        half = (g // M).astype(int)     # 0 or 1
        labels += half << axis          # bit-shift
    return labels


# def create_pill_mask(R, L, f = np.sqrt(2)):
def create_pill_mask(R, L, f = 1):

    # Determine the size of the image
    height = 2 * R# +2 for 1px boundary at top and bottom
    width = L + 2*R  # +2 for 1px boundary on left and right
    
    # Create an empty image
    pad = 3
    imh = height+2*pad + 1
    imw = width+2*pad +1
    # imh = 2*(imh//2)+1 # make odd
    # imw = 2*(imw//2)+1
    
    mask = np.zeros((imh,imw), dtype=np.uint8)
    
    # Calculate the center of the pill shape
    center_x = imw // 2
    center_y = imh // 2
    
    # Draw the rectangular part of the pill
    mask[center_y - R:center_y + R+1, R+pad:L+R+pad+1] = 1
    
    # Create a grid of coordinates
    y, x = np.ogrid[:imh, :imw]
    
    # Draw the left semicircle
    left_center_x = R+pad
    left_circle = (x - left_center_x) ** 2 + (y - center_y) ** 2 <= f*(R ** 2)
    mask[left_circle] = 1
    
    # Draw the right semicircle
    right_center_x = L+R+pad
    right_circle = (x - right_center_x) ** 2 + (y - center_y) ** 2 <= f*(R ** 2)
    mask[right_circle] = 1
    
    return mask


def split_array(array, parts, axes=None):
    """
    Split an ndarray into parts along specified axes.
    
    Parameters:
    - array: ndarray
        The array to split.
    - parts: int or tuple of ints
        Number of parts to split along each axis.
        If an integer, applies to all axes. If a tuple, it specifies parts for each axis.
    - axes: int or tuple of ints, optional
        The axes to split. If None, splits across all axes.
    
    Returns:
    - List of ndarrays
        A nested list of sub-arrays after splitting.
    """
    if isinstance(parts, int):
        parts = (parts,) * array.ndim  # Apply same number of parts to all axes

    if axes is None:
        axes = tuple(range(array.ndim))  # Apply to all axes
    elif isinstance(axes, int):
        axes = (axes,)  # Make it a tuple for consistency

    if len(parts) != len(axes):
        raise ValueError("Length of 'parts' must match the number of axes specified.")

    splits = []  # Store split slices
    warnings = []  # Store warnings for uneven splits

    for ax, num_parts in zip(axes, parts):
        dim_size = array.shape[ax]
        chunk_sizes = [dim_size // num_parts + (1 if i < dim_size % num_parts else 0) for i in range(num_parts)]
        if dim_size % num_parts != 0:
            warnings.append(f"Axis {ax} ({dim_size}) is not evenly divisible by {num_parts}.")
        split_indices = np.cumsum(chunk_sizes[:-1])
        splits.append(np.split(np.arange(dim_size), split_indices))

    # Print warnings if any
    for warning in warnings:
        print("Warning:", warning)

    # Use the slices to split the array recursively
    def recursive_split(array, splits, axes):
        if not splits:
            return array
        ax = axes[0]
        subarrays = []
        for idxs in splits[0]:
            sliced = np.take(array, idxs, axis=ax)
            subarrays.append(recursive_split(sliced, splits[1:], axes[1:]))
        return subarrays

    return recursive_split(array, splits, axes)
    
def reconstruct_array(nested_list, axes=None):
    """
    Reconstruct an ndarray from a nested list of sub-arrays.
    
    Parameters:
    - nested_list: list of ndarrays
        The nested list of sub-arrays to reconstruct.
    - axes: int or tuple of ints, optional
        The axes used for splitting. If None, assumes all axes.

    Returns:
    - ndarray
        The reconstructed array.
    """
    if axes is None:
        axes = tuple(range(len(nested_list[0].shape) if isinstance(nested_list[0], np.ndarray) else len(nested_list)))
    elif isinstance(axes, int):
        axes = (axes,)

    def recursive_reconstruct(nested, level):
        """
        Recursively reconstruct the array along specified axes.
        """
        if isinstance(nested, np.ndarray):
            return nested
        if level == len(axes):
            return np.array(nested)
        return np.concatenate(
            [recursive_reconstruct(sub, level + 1) for sub in nested], axis=axes[level]
        )

    return recursive_reconstruct(nested_list, 0)


def enumerate_nested(*lists, parent_indices=None):
    """
    Traverse one or more matching nested lists and yield their indices and corresponding values.

    Parameters:
    - *lists: list(s)
        One or more nested lists to traverse. All lists must match in structure.
    - parent_indices: list, optional
        The list of indices leading to the current level (used internally).

    Yields:
    - tuple: (indices, values...)
        The indices and corresponding values from all input lists.
    """
    if parent_indices is None:
        parent_indices = []

    # Check if elements are lists at this level
    if all(isinstance(lst[0], list) for lst in lists):
        for i, sublists in enumerate(zip(*lists)):
            current_indices = parent_indices + [i]
            yield from enumerate_nested(*sublists, parent_indices=current_indices)
    else:  # Base case: elements are not lists
        for i, values in enumerate(zip(*lists)):
            current_indices = parent_indices + [i]
            yield current_indices, *values
