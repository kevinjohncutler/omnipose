import numpy as np
import edt


# By testing for convergence across a range of superellipses, I found that the following
# ratio guarantees convergence. The edt() package gives a quick (but rough) distance field,
# and it allows us to find a least upper bound for the number of iterations needed for our
# smooth distance field computation.
def dist_to_diam(dt_pos, n):
    """
    Convert positive distance field values to a mean diameter.

    Parameters
    --------------
    dt_pos: 1D array, float
        array of positive distance field values
    n: int
        dimension of volume. dt_pos is always 1D because only the positive values
        int he distance field are passed in.

    Returns
    --------------
    mean diameter: float
        a single number that corresponds to the diameter of the N-sphere when
        dt_pos for a sphere is given to the function, holds constant for
        extending rods of uniform width, much better than the diameter of a circle
        of equivalent area for estimating the short-axis dimensions of objects

    """
    return 2 * (n + 1) * np.mean(dt_pos)


def diameters(masks, dt=None, dist_threshold=0, pill=False, return_length=False):
    """
    Calculate the mean cell diameter from a label matrix.

    Parameters
    --------------
    masks: ND array, float
        label matrix 0,...,N
    dt: ND array, float
        distance field
    dist_threshold: float
        cutoff below which all values in dt are set to 0. Must be >=0.

    Returns
    --------------
    diam: float
        a single number that corresponds to the average diameter of labeled regions in the image, see dist_to_diam()

    """
    if dist_threshold < 0:
        dist_threshold = 0

    if dt is None and np.any(masks):
        dt = edt.edt(np.int32(masks))
    dt_pos = np.abs(dt[dt > dist_threshold])

    A = np.count_nonzero(dt_pos)
    D = np.sum(dt_pos)

    if np.any(dt_pos):
        if not pill:
            diam = dist_to_diam(np.abs(dt_pos), n=masks.ndim)
            if return_length:
                return diam, A / diam
        else:
            return pill_decomposition(A, D)
    else:
        diam = 0

    return diam


def pill_decomposition(A, D):
    R = np.sqrt((np.sqrt(A ** 2 + 24 * np.pi * D) - A) / (2 * np.pi))
    L = (3 * D - np.pi * (R ** 4)) / (R ** 3)
    return R, L
