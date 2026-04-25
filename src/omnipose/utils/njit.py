"""Numba-jitted utility helpers used across subpackages."""

import numpy as np
from numba import njit


@njit
def most_frequent(neighbor_masks):
    """Column-wise mode: for each column, return the most common value."""
    return np.array([np.bincount(row).argmax() for row in neighbor_masks.T])
