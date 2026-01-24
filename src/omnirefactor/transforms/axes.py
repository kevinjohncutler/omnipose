
def move_axis(img, m_axis=-1, first=True):
    """ move axis m_axis to first or last position """
    if m_axis==-1:
        m_axis = img.ndim-1
    m_axis = min(img.ndim-1, m_axis)
    axes = np.arange(0, img.ndim)
    if first:
        axes[1:m_axis+1] = axes[:m_axis]
        axes[0] = m_axis
    else:
        axes[m_axis:-1] = axes[m_axis+1:]
        axes[-1] = m_axis
    img = img.transpose(tuple(axes))
    return img

# more flexible replacement 
def move_axis_new(a, axis, pos):
    """Move ndarray axis to new location, preserving order of other axes."""
    # Get the current shape of the array
    shape = a.shape
    
    # Create the permutation order for numpy.transpose()
    perm = list(range(len(shape)))
    perm.pop(axis)
    perm.insert(pos, axis)
    
    # Transpose the array based on the permutation order
    return np.transpose(a, perm)

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
            img = move_axis(img, m_axis=channel_axis, first=False)
    return img

def update_axis(m_axis, to_squeeze, ndim):
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
import numpy as np
