"""Array partitioning and reconstruction utilities."""

import numpy as np


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
