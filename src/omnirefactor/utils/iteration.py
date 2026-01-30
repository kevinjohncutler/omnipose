"""Iteration utilities for nested structures."""


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
