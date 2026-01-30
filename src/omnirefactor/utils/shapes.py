"""Synthetic shape/mask generation utilities."""

import numpy as np


def create_pill_mask(R, L, f=1):
    """
    Create a pill-shaped (stadium) binary mask.

    Parameters:
        R: Radius of the semicircular ends
        L: Length of the straight middle section
        f: Factor for circle radius (default 1)

    Returns:
        ndarray: Binary mask of the pill shape (uint8)
    """
    # Determine the size of the image
    height = 2 * R  # +2 for 1px boundary at top and bottom
    width = L + 2 * R  # +2 for 1px boundary on left and right

    # Create an empty image
    pad = 3
    imh = height + 2 * pad + 1
    imw = width + 2 * pad + 1

    mask = np.zeros((imh, imw), dtype=np.uint8)

    # Calculate the center of the pill shape
    center_x = imw // 2
    center_y = imh // 2

    # Draw the rectangular part of the pill
    mask[center_y - R:center_y + R + 1, R + pad:L + R + pad + 1] = 1

    # Create a grid of coordinates
    y, x = np.ogrid[:imh, :imw]

    # Draw the left semicircle
    left_center_x = R + pad
    left_circle = (x - left_center_x) ** 2 + (y - center_y) ** 2 <= f * (R ** 2)
    mask[left_circle] = 1

    # Draw the right semicircle
    right_center_x = L + R + pad
    right_circle = (x - right_center_x) ** 2 + (y - center_y) ** 2 <= f * (R ** 2)
    mask[right_circle] = 1

    return mask
