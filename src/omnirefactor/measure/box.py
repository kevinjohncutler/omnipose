from .imports import *

from scipy.signal import fftconvolve
def bartlett_nd(size):
    """
    Create an N-dimensional Bartlett (triangular) window with shape `size`.
    If size is an integer, we treat it as (size,).
    """
    if isinstance(size, int):
        size = (size,)
    # Create a 1D Bartlett window for each dimension
    windows = [np.bartlett(s) for s in size]

    # Use np.ix_ to create broadcastable ND grids
    grids = np.ix_(*windows)  # e.g., for 2D => shape(7,1), shape(1,7)

    # Multiply them elementwise to get an ND array
    # Instead of in-place, build the kernel in a loop
    kernel = grids[0].astype(float)
    for g in grids[1:]:
        kernel = kernel * g  # shape updates from (7,1)->(7,7) in 2D, etc.

    # Normalize so the total sum = 1
    kernel /= kernel.sum()

    return kernel

def find_highest_density_box(label_matrix, box_size):
    """
    Convolve a binary mask with an N-D Bartlett kernel of shape `box_size`,
    then find the sub-box of shape `box_size` around the maximum, ensuring 
    the box stays within bounds.
    """
    if box_size == -1:
        return tuple(slice(0, s) for s in label_matrix.shape)

    # Handle scalar box_size for all dimensions
    if isinstance(box_size, int):
        box_size = (box_size,) * label_matrix.ndim

    # Binary mask
    mask = (label_matrix > 0).astype(np.float32)

    # Build the N-D Bartlett (triangular) kernel
    kernel = bartlett_nd(box_size)

    # FFT-based convolution
    density_map = fftconvolve(mask, kernel, mode='same')

    # Find the coordinates of the box with the highest cell density
    max_density_coords = np.unravel_index(np.argmax(density_map), density_map.shape)

    # Compute the box bounds while ensuring no negative indices
    slices = []
    for max_coord, size, dim_size in zip(max_density_coords, box_size, label_matrix.shape):
        start = max(0, max_coord - size // 2)
        stop = min(dim_size, start + size)  # Ensure within bounds
        start = max(0, stop - size)  # Adjust start if necessary to maintain box size
        slices.append(slice(start, stop))

    return tuple(slices)

# from scipy.ndimage import uniform_filter
# def find_highest_density_box(label_matrix, box_size, mode='constant'):
#     if box_size == -1:
#         # return tuple([slice(None)]*label_matrix.ndim)
#         return tuple([slice(0,s) for s in label_matrix.shape])
#     else:
#         # Compute the cell density for each box in the image
#         cell_density = uniform_filter((label_matrix > 0).astype(float), size=box_size, mode=mode)

#         # Find the coordinates of the box with the highest cell density
#         max_density_coords = np.unravel_index(np.argmax(cell_density), cell_density.shape)

#         # Compute the coordinates of the box
#         return tuple(slice(max_coord - box_size // 2, max_coord + box_size // 2) for max_coord in max_density_coords), cell_density
    
    
