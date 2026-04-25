import numpy as np
from numba import njit


# might want to deprecate this and do all despur using the torch code
@njit
def candidate_cleanup_idx(idx, connect, neigh_inds, cardinal, ordinal, dim, boundary, internal):
    """
    Jitted helper for per-candidate boundary cleanup.
    This function is meant to mimic the inner loop in the original _despur.
    All indices (e.g. from 'cardinal' and 'ordinal') are assumed to be 1D arrays of integers.
    It updates connect in place.
    """
    n_dirs = connect.shape[0]
    for i in range(cardinal.shape[0]):
        d = cardinal[i]
        if connect[d, idx] != 0:
            target = neigh_inds[d, idx]
            if target < 0:
                continue
            for j in range(ordinal.shape[0]):
                o = ordinal[j]
                if target < 0:
                    continue
                t = neigh_inds[o, target]
                found = False
                for k in range(cardinal.shape[0]):
                    d2 = cardinal[k]
                    if neigh_inds[d2, idx] == t:
                        found = True
                        break
                if found:
                    c_val = 0
                    if (connect[o, target] != 0 and connect[d, idx] != 0) and (t > -1 and target > -1):
                        c_val = 1
                    connect[o, target] = c_val
                    sym_index = -(o + 1)
                    if t > -1:
                        connect[sym_index, t] = c_val
    return
