from .imports import *

def get_niter(dists):
    """
    Get number of iterations.

    Parameters
    --------------
    dists: ND array, float
        array of (nonnegative) distance field values

    Returns
    --------------
    niter: int
        number of iterations empirically found to be the lower bound for convergence
        of the distance field relaxation method
    """
    module = utils.get_module(dists)
    c = module.ceil(module.max(dists) * 1.16) + 1
    return c.astype(int) if module == np else c.int()
