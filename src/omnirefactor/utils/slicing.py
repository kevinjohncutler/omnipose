"""Array slicing utilities."""

from collections.abc import Iterable


def get_slice_tuple(start, stop, shape, axis=None):
    """
    Generate a tuple of slice objects for array indexing.

    Parameters:
        start: Starting index(es) - can be int or iterable
        stop: Stopping index(es) - can be int or iterable
        shape: Shape of the array to slice
        axis: Axis or axes to apply slices to (default: all axes if start/stop are iterable, else 0)

    Returns:
        tuple: Tuple of slice objects for indexing
    """
    ndim = len(shape)

    # Create a list of slices for each axis
    slices = [slice(None)] * ndim


    # Check if start and stop are iterable
    if isinstance(start, Iterable) and isinstance(stop, Iterable):
        if axis is None:
            axis = list(range(ndim))

        # Check that start and stop are the same length
        if len(start) != len(stop):
            raise ValueError("start and stop must be the same length")

        # Check if axis is iterable
        if isinstance(axis, Iterable):
            # Check that axis is the same length as start and stop
            if len(axis) != len(start):
                raise ValueError("axis must be the same length as start and stop")
        else:
            # If axis is not iterable, use it for all slices
            axis = [axis] * len(start)

        # Replace the slice at each axis index
        for a, s, e in zip(axis, start, stop):
            slices[a] = slice(s, e, None)
    else:
        if axis is None:
            axis = 0
        # If start and stop are not iterable, use them as integers
        slices[axis] = slice(start, stop, None)

    # Convert the list to a tuple
    return tuple(slices)
