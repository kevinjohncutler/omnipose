from skimage import measure
from scipy.ndimage import binary_dilation
import numpy as np 
from .utils import is_integer

def bbox_to_slice(bbox,shape,pad=0,im_pad=0):
    """
    return the tuple of slices for cropping an image based on the skimage.measure bounding box
    optional padding allows for the bounding box to be expanded, but not outside the original image dimensions 
    
    Parameters
    ----------
    bbox: ndarray, float
        input bounding box, e.g. [y0,x0,y1,x1]
        
    shape: array, tuple, or list, int
        shape of corresponding array to be sliced
    
    pad: array, tuple, or list, int
        padding to be applied to each axis of the bounding box
        can be a common padding (5 means 5 on every side) 
        or a list of each axis padding ([3,4] means 3 on y and 4 on x).
        N-volume requires an N-tuple. 
        
    im_pad: int
        region around the edges to avoid (pull back coordinate limits)
    
    Returns
    --------------
    tuple of slices 
    
    """
    dim = len(shape)
    # if type(pad) is int:
    if is_integer(pad):
        pad = [pad]*dim
    # if type(im_pad) is int:
    if is_integer(im_pad):
        im_pad = [im_pad]*dim
    # return tuple([slice(int(max(0,bbox[n]-pad[n])),int(min(bbox[n+dim]+pad[n],shape[n]))) for n in range(len(bbox)//2)])
    # added a +1 to stop, might be a necessary fix but not sure yet 
    # print('im_pad',im_pad, bbox, pad, shape)
    one = 0
    return tuple([slice(int(max(im_pad[n],bbox[n]-pad[n])),
                        int(min(bbox[n+dim]+pad[n]+one,shape[n]-im_pad[n]))) 
                  for n in range(len(bbox)//2)])
    
    
def make_square(bbox, shape):
    """
    Expand bbox to be square. bbox = (miny, minx, maxy, maxx).
    Clamps to image boundaries.
    """
    miny, minx, maxy, maxx = bbox
    height = maxy - miny
    width = maxx - minx
    side = max(height, width)

    # Extra space needed
    dy = side - height
    dx = side - width

    # Symmetric expansion
    miny = max(miny - dy // 2, 0)
    maxy = min(maxy + dy - dy // 2, shape[0])
    minx = max(minx - dx // 2, 0)
    maxx = min(maxx + dx - dx // 2, shape[1])

    return (miny, minx, maxy, maxx)

def crop_bbox(mask, pad=10, iterations=3, im_pad=0, area_cutoff=0,
              max_dim=np.inf, get_biggest=False, binary=False, square=False):
    """
    Take a label matrix and return bounding box slices. The `square` option 
    applies to both individual regions AND the merged bounding box if `binary`.
    """

    bw = binary_dilation(mask > 0, iterations=iterations) if iterations > 0 else (mask > 0)
    clusters = measure.label(bw)
    regions = measure.regionprops(clusters)
    sz = mask.shape
    d = mask.ndim

    def adjust_bbox(bbx):
        # Clamp the pad so we never go out of image bounds
        minpad = min(pad, bbx[0], bbx[1],
                     sz[0] - bbx[2], sz[1] - bbx[3])
        if square:
            bbx = make_square(bbx, sz)
        return bbox_to_slice(bbx, sz, pad=minpad, im_pad=im_pad)

    slices = []
    if get_biggest and regions:
        # Single largest region
        largest_idx = np.argmax([r.area for r in regions])
        bbx = regions[largest_idx].bbox
        slices.append(adjust_bbox(bbx))

    else:
        # All regions above area_cutoff
        for props in regions:
            if props.area > area_cutoff:
                bbx = props.bbox
                slices.append(adjust_bbox(bbx))

    # Merge into a single bounding box if binary=True
    if binary and slices:
        # Convert list of slices -> overall bounding box
        start_xy = np.min([[slc[i].start for i in range(d)] for slc in slices], axis=0)
        stop_xy  = np.max([[slc[i].stop  for i in range(d)] for slc in slices], axis=0)
        union_bbox = (start_xy[0], start_xy[1], stop_xy[0], stop_xy[1])

        # Build a single slice from union bbox
        merged_slice = adjust_bbox(union_bbox)
        return merged_slice

    return slices

def extract_patches(image, points, box_size, fill_value=0, point_order='yx'):
    """
    Extract patches centered around points from an image, even if the points are at the edge.
    Out-of-bounds areas are filled with the given fill_value.
    Works for both grayscale (yx) and RGB (yxc) images.

    Args:
    - image: 2D (grayscale) or 3D (RGB) numpy array representing the source image.
    - points: List or array of (x, y) or (y, x) tuples representing the center points of each patch.
    - box_size: Integer for square patches or tuple (height, width) for rectangular patches.
    - fill_value: The value to fill for out-of-bounds areas (default is 0).
    - point_order: String specifying whether the points are in 'yx' (default) or 'xy' order.

    Returns:
    - patches: A 4D (if RGB) or 3D (if grayscale) numpy array where each slice corresponds to a patch centered on a point.
    - slices: A list of tuples, each containing slices for y and x dimensions, representing the slice in the original array.
    """
    
    # If box_size is a single integer, convert it to a tuple (height, width)
    if isinstance(box_size, int):
        box_size = (box_size, box_size)

    box_size = tuple([s + 1 - s % 2 for s in box_size])  # make odd if not

    half_height, half_width = box_size[0] // 2, box_size[1] // 2

    shape = (len(points), box_size[0], box_size[1])
    img_height, img_width = image.shape[:2]
    if image.ndim == 3:
        shape += (image.shape[2],)

    # Pre-fill the output array with the fill_value, adding channel dimension if needed
    patches = np.full(shape, fill_value, dtype=image.dtype)
    
    # Initialize a list to store the slices
    slices = []

    for i, point in enumerate(points):
        # Handle point order based on the argument 'point_order'
        if point_order == 'yx':
            y, x = point
        elif point_order == 'xy':
            x, y = point
        else:
            raise ValueError("point_order must be 'yx' or 'xy'")

        # Define the source slice with clamping to image bounds
        src_y_start = max(0, y - half_height)
        src_y_end = min(img_height, y + half_height + 1)
        src_x_start = max(0, x - half_width)
        src_x_end = min(img_width, x + half_width + 1)

        # Define the destination slice
        dst_y_start = half_height - (y - src_y_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = half_width - (x - src_x_start)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        # Fill the patch array
        patches[i, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[src_y_start:src_y_end, src_x_start:src_x_end]

        # Record the slices for the original image
        slices.append((
            slice(src_y_start, src_y_end),
            slice(src_x_start, src_x_end)
        ))

    return patches, slices