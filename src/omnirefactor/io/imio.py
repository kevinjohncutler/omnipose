from .imports import *
import imagecodecs

def imread(filename):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='.tiff':
        img = tifffile.imread(filename)
        return img
    elif ext=='.npy':
        return np.load(filename)
    elif ext=='.npz':
        return np.load(filename)['arr_0']
    elif ext=='.czi':
        img = AICSImage(filename).data
        return img
    else:
        try:
            # Read image including alpha channel if present (-1 flag)
            img = cv2.imread(filename, -1)
            if img is None:
                raise ValueError("Failed to read image")
            # Check dimensions
            if img.ndim == 2:
                # Grayscale image, no conversion needed
                return img
            elif img.shape[2] == 3:
                # Convert 3-channel BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                # Convert 4-channel BGRA to RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            return img
        except Exception as e:
            io_logger.critical('ERROR: could not read file, %s' % e)
            return None

def imwrite(filename, arr, **kwargs):
    """
    Save an image to file using imagecodecs for encoding.
    
    Supported extensions (besides .tif/.tiff and .npy):
      - .png: uses imagecodecs.png_encode (accepts 'level' for compression)
      - .jpg, .jpeg, .jp2: uses imagecodecs.jpeg_encode (accepts 'level', analogous to JPEG quality)
      - .webp: uses imagecodecs.webp_encode (accepts 'level'; 'quality' is kept as a backwards compatible alias)
      - .jxl: uses imagecodecs.jpegxl_encode (accepts 'quality', 'effort', 'distance', 'decoding_speed')
      - .bmp: uses imagecodecs.bmp_encode (no extra parameters; always lossless)
    For other extensions, PNG encoding is used as a fallback.
    
    Note: Unlike OpenCV, imagecodecs expects normal RGB/RGBA (not BGR/BGRA) channel ordering.
    """
    ext = os.path.splitext(filename)[-1].lower()
    
    if ext in ['.tif', '.tiff']:
        tifffile.imwrite(filename, arr, **kwargs)
        return
    elif ext == '.npy':
        np.save(filename, arr, **kwargs)
        return

    # Determine which encoder function to use based on the extension.
    encoded = None
    if ext == '.png':
        # For PNG, get 'compression'; other kwargs may be passed to the encoder.
        level = kwargs.pop('level', 9)
        encoded = imagecodecs.png_encode(arr, level=level, **kwargs)
    elif ext in ['.jpg', '.jpeg', '.jp2']:
        level = kwargs.pop('level', 95)
        encoded = imagecodecs.jpeg_encode(arr, level=level, **kwargs)
    elif ext == '.webp':
        # imagecodecs expects 'level'; accept legacy 'quality' and map it over.
        level = kwargs.pop('level', None)
        quality = kwargs.pop('quality', None)
        if quality is not None and level is None:
            level = quality
        if level is not None:
            encoded = imagecodecs.webp_encode(arr, level=level, **kwargs)
        else:
            encoded = imagecodecs.webp_encode(arr, **kwargs)
    elif ext == '.jxl':
        effort = kwargs.pop('effort', 1)
        distance = kwargs.pop('distance', 1.0)
        encoded = imagecodecs.jpegxl_encode(arr,
                                            effort=effort,
                                            distance=distance,
                                            **kwargs)
    elif ext == '.bmp':
        encoded = imagecodecs.bmp_encode(arr, **kwargs)
    else:
        # For unsupported extensions, default to PNG.
        encoded = imagecodecs.png_encode(arr, **kwargs)
    
    # Write the encoded byte buffer to the file.
    with open(filename, 'wb') as f:
        f.write(encoded)


def imsave(filename, arr):
    io_logger.warning('WARNING: imsave is deprecated, use io.imwrite instead')
    return imwrite(filename, arr)

# now allows for any extension(s) to be specified, allowing exclusion if necessary, non-image files, etc. 
def get_image_files(folder, mask_filter='_masks', img_filter='', look_one_level_down=False,
                    extensions = ['png','jpg','jpeg','tif','tiff'], pattern=None):
    """ find all images in a folder and if look_one_level_down all subfolders """
    mask_filters = ['_cp_masks', '_cp_output', '_flows', mask_filter]
    image_names = []
    
    folders = []
    if look_one_level_down:
        folders = natsorted(glob.glob(os.path.join(folder, "*",'')))  
    folders.append(folder)

    for folder in folders:
        for ext in extensions:
            image_names.extend(glob.glob(folder + ('/*%s.'+ext)%img_filter))
    
    image_names = natsorted(image_names)
    imn = []
    for im in image_names:
        imfile = os.path.splitext(im)[0]
        igood = all([(len(imfile) > len(mask_filter) and imfile[-len(mask_filter):] != mask_filter) or len(imfile) < len(mask_filter) 
                        for mask_filter in mask_filters])
        if len(img_filter)>0:
            igood &= imfile[-len(img_filter):]==img_filter
        if pattern is not None:
            # igood &= bool(re.search(pattern, imfile))
            igood &= bool(re.search(pattern + r'$', imfile))
        if igood:
            imn.append(im)
    image_names = imn

    if len(image_names)==0:
        raise ValueError('ERROR: no images in --dir folder')
    
    return image_names
