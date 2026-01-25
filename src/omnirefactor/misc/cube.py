from .imports import *
import math

# not acutally used in the code, typically use steps_to_indices etc. 
def cubestats(n):
    """
    Gets the number of m-dimensional hypercubes connected to the n-cube, including itself. 
    
    Parameters
    ----------
    n: int
        dimension of hypercube
    
    Returns
    -------
    List whose length tells us how many hypercube types there are (point/edge/pixel/voxel...) 
    connected to the central hypercube and whose entries denote many there in each group. 
    E.g., a square would be n=2, so cubestats returns [4, 4, 1] for four points (m=0), 
    four edges (m=1), and one face (the original square,m=n=2). 
    
    """
    faces = []
    for m in range(n+1):
          faces.append((2**(n-m))*math.comb(n,m))
    return faces
