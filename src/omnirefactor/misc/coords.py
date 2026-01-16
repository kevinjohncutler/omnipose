from .imports import *

def meshgrid(shape):
    """
    Generate a tuple of coordinate grids for an ND array of a given shape.
    
    Parameters:
        shape (tuple): Shape of the ND array (e.g., (Y, X) for 2D, (Z, Y, X) for 3D, etc.).
    
    Returns:
        tuple: A tuple of N coordinate arrays, one per dimension.
    """
    ranges = [np.arange(dim) for dim in shape]  # Create a range for each dimension
    coords = np.meshgrid(*ranges, indexing='ij')  # Generate coordinate arrays
    return coords  # Returns a tuple of N arrays
    
def generate_flat_coordinates(shape):
    """
    Generate flat coordinate arrays for an ND array.
    
    Parameters:
        shape (tuple): Shape of the array (e.g., (Y, X) for 2D).
        
    Returns:
        tuple: A tuple of flat arrays representing the coordinates.
    """
    grids = meshgrid(shape)  # Generate the meshgrid
    return tuple(grid.ravel() for grid in grids)  # Flatten each grid