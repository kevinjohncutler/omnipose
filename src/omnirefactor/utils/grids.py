"""Grid and label matrix generation utilities."""

from typing import Sequence

import numpy as np


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
      N=1 -> [0...0,1...1]
      N=2 -> quadrants labeled 0..3
      N=3 -> octants labeled 0..7
      N=4 -> 16 hyper-quadrants labeled 0..15
    """
    if N < 1:
        raise ValueError("N must be >=1")
    # build index grids
    grids = np.ogrid[tuple(slice(0, 2*M) for _ in range(N))]
    labels = np.zeros((2*M,)*N, dtype=int)
    for axis, g in enumerate(grids):
        half = (g // M).astype(int)     # 0 or 1
        labels += half << axis          # bit-shift
    return labels
