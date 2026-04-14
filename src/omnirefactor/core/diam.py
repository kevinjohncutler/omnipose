from ocdkit.measure import dist_to_diam, diameters

import numpy as np


def pill_decomposition(A, D):
    R = np.sqrt((np.sqrt(A ** 2 + 24 * np.pi * D) - A) / (2 * np.pi))
    L = (3 * D - np.pi * (R ** 4)) / (R ** 3)
    return R, L
