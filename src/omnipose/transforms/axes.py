"""Axis tracking helper (local). Generic axis manipulation lives in ocdkit.array
and is re-exported through ``transforms/imports.py``.
"""

import numpy as np


def update_axis(m_axis, to_squeeze, ndim):
    """Update an axis index after squeezing singleton dimensions."""
    if m_axis == -1:
        m_axis = ndim - 1
    if (to_squeeze == m_axis).sum() == 1:
        m_axis = None
    else:
        inds = np.ones(ndim, bool)
        inds[to_squeeze] = False
        m_axis = np.nonzero(np.arange(0, ndim)[inds] == m_axis)[0]
        if len(m_axis) > 0:
            m_axis = m_axis[0]
        else:
            m_axis = None
    return m_axis
