
import numpy as np

def move_axis(img, axis=-1, pos="last"):
    """Move ndarray axis to new location, preserving order of other axes."""
    if axis == -1:
        axis = img.ndim - 1
    axis = min(img.ndim - 1, axis)
    if pos in ("first", 0):
        pos = 0
    elif pos in ("last", -1):
        pos = img.ndim - 1
    perm = list(range(img.ndim))
    perm.pop(axis)
    perm.insert(pos, axis)
    return np.transpose(img, perm)

# This was edited to fix a bug where single-channel images of shape (y,x) would be 
# transposed to (x,y) if x<y, making the labels no longer correspond to the data. 
def move_min_dim(img, force=False):
    """ move minimum dimension last as channels if < 10, or force==True """
    if len(img.shape) > 2: #only makes sense to do this if channel axis is already present, not best for 3D though! 
        min_dim = min(img.shape)
        if min_dim < 10 or force:
            if img.shape[-1]==min_dim:
                channel_axis = -1
            else:
                channel_axis = (img.shape).index(min_dim)
            img = move_axis(img, axis=channel_axis, pos="last")
    return img

def update_axis(m_axis, to_squeeze, ndim):
    """Update an axis index after squeezing singleton dimensions."""
    if m_axis==-1:
        m_axis = ndim-1
    if (to_squeeze==m_axis).sum() == 1:
        m_axis = None
    else:
        inds = np.ones(ndim, bool)
        inds[to_squeeze] = False
        m_axis = np.nonzero(np.arange(0, ndim)[inds]==m_axis)[0]
        if len(m_axis) > 0:
            m_axis = m_axis[0]
        else:
            m_axis = None
    return m_axis
