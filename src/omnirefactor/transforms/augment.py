from __future__ import annotations

import numpy as np
from scipy.ndimage import affine_transform
import mgen


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
