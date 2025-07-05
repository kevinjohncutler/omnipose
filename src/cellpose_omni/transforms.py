import numpy as np
import warnings
import cv2

import logging
transforms_logger = logging.getLogger(__name__)

from . import dynamics, utils
import itertools # ND tiling

# import omnipose, edt, fastremap
# OMNI_INSTALLED = True

try:
    import omnipose, edt, fastremap
    OMNI_INSTALLED = True
except:
    OMNI_INSTALLED = False
    print('OMNIPOSE NOT INSTALLED')

def _taper_mask(ly=224, lx=224, sig=7.5):
    bsize = max(224, max(ly, lx))
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    mask = 1/(1 + np.exp((xm - (bsize/2-20)) / sig)) 
    mask = mask * mask[:, np.newaxis]
    mask = mask[bsize//2-ly//2 : bsize//2+ly//2+ly%2, 
                bsize//2-lx//2 : bsize//2+lx//2+lx%2]
    return mask

def unaugment_tiles(y, unet=False):
    """ reverse test-time augmentations for averaging

    Parameters
    ----------

    y: float32
        array that's ntiles_y x ntiles_x x chan x Ly x Lx where chan = (dY, dX, cell prob)

    unet: bool (optional, False)
        whether or not unet output or cellpose output
    
    Returns
    -------

    y: float32

    """
    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if j%2==0 and i%2==1:
                y[j,i] = y[j,i, :,::-1, :]
                if not unet:
                    y[j,i,0] *= -1
            elif j%2==1 and i%2==0:
                y[j,i] = y[j,i, :,:, ::-1]
                if not unet:
                    y[j,i,1] *= -1
            elif j%2==1 and i%2==1:
                y[j,i] = y[j,i, :,::-1, ::-1]
                if not unet:
                    y[j,i,0] *= -1
                    y[j,i,1] *= -1
    return y


def average_tiles(y, ysub, xsub, Ly, Lx):
    """ average results of network over tiles

    Parameters
    -------------

    y: float, [ntiles x nclasses x bsize x bsize]
        output of cellpose network for each tile

    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles

    xsub : list
        list of arrays with start and end of tiles in X of length ntiles

    Ly : int
        size of pre-tiled image in Y (may be larger than original image if
        image size is less than bsize)

    Lx : int
        size of pre-tiled image in X (may be larger than original image if
        image size is less than bsize)

    Returns
    -------------

    yf: float32, [nclasses x Ly x Lx]
        network output averaged over tiles

    """
    Navg = np.zeros((Ly,Lx))
    yf = np.zeros((y.shape[1], Ly, Lx), np.float32)
    # taper edges of tiles
    mask = _taper_mask(ly=y.shape[-2], lx=y.shape[-1])
    for j in range(len(ysub)):
        yf[:, ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] += y[j] * mask
        Navg[ysub[j][0]:ysub[j][1],  xsub[j][0]:xsub[j][1]] += mask
    yf /= Navg
    return yf


def make_tiles(imgi, bsize=224, augment=False, tile_overlap=0.1):
    """ make tiles of image to run at test-time

    if augmented, tiles are flipped and tile_overlap=2.
        * original
        * flipped vertically
        * flipped horizontally
        * flipped vertically and horizontally

    Parameters
    ----------
    imgi : float32
        array that's nchan x Ly x Lx

    bsize : float (optional, default 224)
        size of tiles

    augment : bool (optional, default False)
        flip tiles and set tile_overlap=2.

    tile_overlap: float (optional, default 0.1)
        fraction of overlap of tiles

    Returns
    -------
    IMG : float32
        array that's ntiles x nchan x bsize x bsize

    ysub : list
        list of arrays with start and end of tiles in Y of length ntiles

    xsub : list
        list of arrays with start and end of tiles in X of length ntiles

    
    """

    nchan, Ly, Lx = imgi.shape
    if augment:
        bsize = np.int32(bsize)
        # pad if image smaller than bsize
        if Ly<bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, bsize-Ly, Lx))), axis=1)
            Ly = bsize
        if Lx<bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, Ly, bsize-Lx))), axis=2)
        Ly, Lx = imgi.shape[-2:]
        # tiles overlap by half of tile size
        ny = max(2, int(np.ceil(2. * Ly / bsize)))
        nx = max(2, int(np.ceil(2. * Lx / bsize)))
        ystart = np.linspace(0, Ly-bsize, ny).astype(int)
        xstart = np.linspace(0, Lx-bsize, nx).astype(int)

        ysub = []
        xsub = []

        # flip tiles so that overlapping segments are processed in rotation
        IMG = np.zeros((len(ystart), len(xstart), nchan,  bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j]+bsize])
                xsub.append([xstart[i], xstart[i]+bsize])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1],  xsub[-1][0]:xsub[-1][1]]
                # flip tiles to allow for augmentation of overlapping segments
                if j%2==0 and i%2==1:
                    IMG[j,i] = IMG[j,i, :,::-1, :]
                elif j%2==1 and i%2==0:
                    IMG[j,i] = IMG[j,i, :,:, ::-1]
                elif j%2==1 and i%2==1:
                    IMG[j,i] = IMG[j,i,:, ::-1, ::-1]
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
        bsizeY = np.int32(bsizeY)
        bsizeX = np.int32(bsizeX)
        # tiles overlap by 10% tile size
        ny = 1 if Ly<=bsize else int(np.ceil((1.+2*tile_overlap) * Ly / bsize))
        nx = 1 if Lx<=bsize else int(np.ceil((1.+2*tile_overlap) * Lx / bsize))
        ystart = np.linspace(0, Ly-bsizeY, ny).astype(int)
        xstart = np.linspace(0, Lx-bsizeX, nx).astype(int)

        ysub = []
        xsub = []
        IMG = np.zeros((len(ystart), len(xstart), nchan,  bsizeY, bsizeX), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j]+bsizeY])
                xsub.append([xstart[i], xstart[i]+bsizeX])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1],  xsub[-1][0]:xsub[-1][1]]
        
    return IMG, ysub, xsub, Ly, Lx


from omnipose.utils import get_flip, _taper_mask_ND, unaugment_tiles_ND, average_tiles_ND, make_tiles_ND
# def get_flip(idx):
#     """
#     ND slices for flipping arrays along particular axes 
#     based on the tile indices. Used in augment_tiles_ND()
#     and unaugment_tiles_ND(). 
#     """
#     return tuple([slice(None,None,None) if i%2 else 
#                   slice(None,None,-1) for i in idx])

# def _taper_mask_ND(shape=(224,224), sig=7.5):
#     dim = len(shape)
#     bsize = max(shape)
#     xm = np.arange(bsize)
#     xm = np.abs(xm - xm.mean())
#     # 1D distribution 
#     mask = 1/(1 + np.exp((xm - (bsize/2-20)) / sig)) 
#     # extend to ND
#     for j in range(dim-1):
#         mask = mask * mask[..., np.newaxis]
#     slc = tuple([slice(bsize//2-s//2,bsize//2+s//2+s%2) for s in shape])
#     mask = mask[slc]
#     return mask
    
# def unaugment_tiles_ND(y, inds, unet=False):
#     """ reverse test-time augmentations for averaging

#     Parameters
#     ----------

#     y: float32
#         array that's ntiles x chan x Ly x Lx where 
#         chan = (dY, dX, dist, boundary)

#     unet: bool (optional, False)
#         whether or not unet output or cellpose output
    
#     Returns
#     -------

#     y: float32

#     """
#     dim = len(inds[0])
    
#     for i,idx in enumerate(inds): 
        
#         # repeat the flip to undo it 
#         flip = get_flip(idx)
        
#         # flow field componenets need to be flipped 
#         factor = np.array([1 if i%2 else -1 for i in idx])
        
#         # apply the flip
#         y[i] = y[i][(Ellipsis,)+flip]
        
#         # apply the flow field flip
#         if not unet:
#             y[i][:dim] = [s*f for s,f in zip(y[i][:dim],factor)]
            
#     return y
    
# def average_tiles_ND(y,subs,shape):
#     """ average results of network over tiles

#     Parameters
#     -------------

#     y: float, [ntiles x nclasses x bsize x bsize]
#         output of cellpose network for each tile

#     subs : list
#         list of slices for each subtile 

#     shape : int, list or tuple
#         shape of pre-tiled image (may be larger than original image if
#         image size is less than bsize)

#     Returns
#     -------------

#     yf: float32, [nclasses x Ly x Lx]
#         network output averaged over tiles

#     """
#     Navg = np.zeros(shape)
#     yf = np.zeros((y.shape[1],)+shape, np.float32)
#     # taper edges of tiles
#     mask = _taper_mask_ND(y.shape[-len(shape):])
#     for j,slc in enumerate(subs):
#         yf[(Ellipsis,)+slc] += y[j] * mask
#         Navg[slc] += mask
#     yf /= Navg
#     return yf

# def make_tiles_ND(imgi, bsize=224, augment=False, tile_overlap=0.1):
#     """ make tiles of image to run at test-time

#     if augmented, tiles are flipped and tile_overlap=2.
#         * original
#         * flipped vertically
#         * flipped horizontally
#         * flipped vertically and horizontally

#     Parameters
#     ----------
#     imgi : float32
#         array that's nchan x Ly x Lx

#     bsize : float (optional, default 224)
#         size of tiles

#     augment : bool (optional, default False)
#         flip tiles and set tile_overlap=2.

#     tile_overlap: float (optional, default 0.1)
#         fraction of overlap of tiles

#     Returns
#     -------
#     IMG : float32
#         array that's ntiles x nchan x bsize x bsize

#     ysub : list
#         list of arrays with start and end of tiles in Y of length ntiles

#     xsub : list
#         list of arrays with start and end of tiles in X of length ntiles

    
#     """

#     nchan = imgi.shape[0]
#     shape = imgi.shape[1:]
#     dim = len(shape)
#     inds = []
#     if augment:
#         bsize = np.int32(bsize)
#         # pad if image smaller than bsize
#         pad_seq = [(0,0)]+[(0,max(0,bsize-s))for s in shape]
#         imgi = np.pad(imgi,pad_seq)
#         shape = imgi.shape[-dim:]
        
#         # tiles overlap by half of tile size
#         ntyx = [max(2, int(np.ceil(2. * s / bsize))) for s in shape]
#         start = [np.linspace(0, s-bsize, n).astype(int) for s,n in zip(shape,ntyx)]
        
#         intervals = [[slice(si,si+bsize) for si in s] for s in start]
#         subs = list(itertools.product(*intervals))
#         indexes = [np.arange(len(s)) for s in start]
#         inds = list(itertools.product(*indexes))
        
#         IMG = []
        
#         # here I flip if the index is odd 
#         for slc,idx in zip(subs,inds):        
#             flip = get_flip(idx) # avoid repetition with unaugment
#             IMG.append(imgi[(Ellipsis,)+slc][(Ellipsis,)+flip])
            
        
#         IMG = np.stack(IMG)
#     else:
#         tile_overlap = min(0.5, max(0.05, tile_overlap))
#         # bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
#         # B = [np.int32(min(b,s)) for s,b in zip(im.shape,bsize)] if bzise variable
#         bbox = tuple([np.int32(min(bsize,s)) for s in shape])
        
#         # tiles overlap by 10% tile size by default
#         ntyx = [1 if s<=bsize else int(np.ceil((1.+2*tile_overlap) * s / bsize)) 
#                 for s in shape]
#         start = [np.linspace(0, s-b, n).astype(int) for s,b,n in zip(shape,bbox,ntyx)]

#         intervals = [[slice(si,si+bsize) for si in s] for s in start]
#         subs = list(itertools.product(*intervals))
        
#         # IMG = np.zeros((len(ystart), len(xstart), nchan,  bsizeY, bsizeX), np.float32)
#         # IMG = np.zeros(tuple([len(s) for s in start])+(nchan,)+bbox, np.float32)
#         # IMG = np.stack([imgi[(Ellipsis,)+slc] for slc in subs])
#         print('normalizing each tile')
#         IMG = np.stack([normalize99(imgi[(Ellipsis,)+slc],omni=True) for slc in subs])
        
        
#     return IMG, subs, shape, inds

# needs to have a wider range to avoid weird effects with few cells in frame
# also turns out previous formulation can give negative numbers, messes up log operations etc. 
def normalize99(Y,lower=0.01,upper=99.99,omni=False):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile """
    if omni and OMNI_INSTALLED:
        X = omnipose.utils.normalize99(Y)
    else:
        X = Y.copy()
        x01 = np.percentile(X, 1)
        x99 = np.percentile(X, 99)
        X = (X - x01) / (x99 - x01)
    return X


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

def convert_image(x, channels, channel_axis=None, z_axis=None,
                  do_3D=False, normalize=True, invert=False,
                  nchan=2, dim=2, omni=False):
    """ return image with z first, channels last and normalized intensities """
    
    # squeeze image, and if channel_axis or z_axis given, transpose image
    if x.ndim > 3:
        to_squeeze = np.array([int(isq) for isq,s in enumerate(x.shape) if s==1])
        # remove channel axis if number of channels is 1
        if len(to_squeeze) > 0: 
            channel_axis = update_axis(channel_axis, to_squeeze, x.ndim) if channel_axis is not None else channel_axis
            z_axis = update_axis(z_axis, to_squeeze, x.ndim) if z_axis is not None else z_axis
        x = x.squeeze()
    # print('shape00',x.shape)
    # put z axis first
    if z_axis is not None and x.ndim > 2:
        x = move_axis(x, m_axis=z_axis, first=True)
        if channel_axis is not None:
            channel_axis += 1
        if x.ndim==3:
            x = x[...,np.newaxis]
    # print('shape01',x.shape,x.ndim,channel_axis,dim)
    # put channel axis last
    if channel_axis is not None and x.ndim > 2:
        x = move_axis(x, m_axis=channel_axis, first=False)
    elif x.ndim == dim:
        # x = x[...,np.newaxis]
        x = x[np.newaxis]
        
    
    # print('shape02',x.shape)

    if do_3D :
        if x.ndim < 3:
            transforms_logger.critical('ERROR: cannot process 2D images in 3D mode')
            raise ValueError('ERROR: cannot process 2D images in 3D mode') 
        elif x.ndim<4:
            x = x[...,np.newaxis]

    # print('shape03',x.shape)
    
    # this one must be the cuplrit... no, in fact it is not 
    if channel_axis is None:
        x = move_min_dim(x)
        channel_axis = -1 # moves to last 

    # print('shape04',x.shape)
        
    if x.ndim > 3:
        transforms_logger.info('multi-stack tiff read in as having %d planes %d channels'%
                (x.shape[0], x.shape[-1]))

    if channels is not None:
        channels = channels[0] if len(channels)==1 else channels
        if len(channels) < 2:
            transforms_logger.critical('ERROR: two channels not specified')
            raise ValueError('ERROR: two channels not specified') 
        x = reshape(x, channels=channels, channel_axis=channel_axis)
        # print('AAA',x.shape,channels)
    else:
        # print('BBB',do_3D,x.ndim,x.shape,nchan)
        # code above put channels last, so its making sure nchan matches below
        # not sure when this condition would be met, but it conflicts with 3D
        if x.shape[-1] > nchan and x.ndim>dim:
            transforms_logger.warning(('WARNING: more than %d channels given, use '
                                       '"channels" input for specifying channels -'
                                       'just using first %d channels to run processing')%(nchan,nchan))
            x = x[...,:nchan]
        
        if not do_3D and x.ndim>3 and dim==2: # error should only be thrown for 2D mode 
            transforms_logger.critical('ERROR: cannot process 4D images in 2D mode')
            raise ValueError('ERROR: cannot process 4D images in 2D mode')
            
        if x.shape[-1] < nchan:
            x = np.concatenate((x, 
                                np.tile(np.zeros_like(x), (1,1,nchan-1))), 
                                axis=-1)
            
    if normalize or invert:
        x = normalize_img(x, invert=invert, omni=omni)

    return x

def reshape(data, channels=[0,0], chan_first=False, channel_axis=0):
    """ reshape data using channels

    Parameters
    ----------

    data : numpy array that's (Z x ) Ly x Lx x nchan
        if data.ndim==8 and data.shape[0]<8, assumed to be nchan x Ly x Lx

    channels : list of int of length 2 (optional, default [0,0])
        First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        For instance, to train on grayscale images, input [0,0]. To train on images with cells
        in green and nuclei in blue, input [2,3].

    channel_axis : int, default 0
        the axis that corresponds to channels (usually 0 or -1)

    Returns
    -------
    data : numpy array that's (Z x ) Ly x Lx x nchan (if chan_first==False)

    """
    data = data.astype(np.float32)
    if data.ndim < 3: # plain 2D images get a new channel axis 
        data = data[...,np.newaxis]
    elif data.shape[0]<8 and data.ndim==3: # Assume stack is nchan x Ly x Lx, so reorder to Ly x Lx x nchan 
        data = np.transpose(data, (1,2,0))
        channel_axis = -1 
        # 8 is completely arbitrary and idk why we need to assume this, we should change to just using the channel axis 
    if data.shape[-1]==1:
        # use grayscale image
        # adds a second channel of zeros 
        data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    else:
        if channels[0]==0:
            # [0,0] does a mean of all channels, pads with 0 for second channel 
            data = data.mean(axis=channel_axis, keepdims=True) # also had a big bug: 3D volumes get squashed to 2D along x axis!!! Assumptions bad. 
            data = np.concatenate((data, np.zeros_like(data)), axis=-1) # forces images to always have 2 channels, possibly bad for multidimensional
        else:
            chanid = [channels[0]-1] # [0,0] would do a mean, [1,0] would actually take the first channel
            if channels[1] > 0:
                chanid.append(channels[1]-1)
            data = data[...,chanid]
            for i in range(data.shape[-1]):
                if np.ptp(data[...,i]) == 0.0:
                    if i==0:
                        warnings.warn("chan to seg' has value range of ZERO")
                    else:
                        warnings.warn("'chan2 (opt)' has value range of ZERO, can instead set chan2 to 0")
            if data.shape[-1]==1:
                data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    if chan_first:
        if data.ndim==4:
            data = np.transpose(data, (3,0,1,2))
        else:
            data = np.transpose(data, (2,0,1))
    return data

def normalize_img(img, axis=-1, invert=False, omni=False):
    """ normalize each channel of the image so that so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities

    optional inversion

    Parameters
    ------------

    img: ND-array (at least 3 dimensions)

    axis: channel axis to loop over for normalization

    Returns
    ---------------

    img: ND-array, float32
        normalized image of same size

    """
    if img.ndim<3:
        error_message = 'Image needs to have at least 3 dimensions'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)

    img = img.astype(np.float32)
    img = np.moveaxis(img, axis, 0)
    for k in range(img.shape[0]):
        # ptp can still give nan's with weird images
        if np.percentile(img[k],99) > np.percentile(img[k],1)+1e-3: #np.ptp(img[k]) > 1e-3:
            img[k] = normalize99(img[k],omni=omni)
            if invert:
                img[k] = -1*img[k] + 1   
    img = np.moveaxis(img, 0, axis)
    return img

def reshape_train_test(train_data, train_labels, test_data, test_labels, channels, channel_axis=0, normalize=True, dim=2, omni=False):
    """ check sizes and reshape train and test data for training """
    nimg = len(train_data)
    # check that arrays are correct size
    if nimg != len(train_labels):
        error_message = 'train data and labels not same length'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        return
    
    if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
        error_message = 'training data or labels are not at least two-dimensional'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        return

    if train_data[0].ndim > 3:
        error_message = 'training data is more than three-dimensional (should be 2D or 3D array)'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        return

    # check if test_data correct length
    if not (test_data is not None and test_labels is not None and
            len(test_data) > 0 and len(test_data)==len(test_labels)):
        test_data = None

    # print('reshape_train_test',train_data[0].shape,channels,channel_axis,normalize,omni)
    # make data correct shape and normalize it so that 0 and 1 are 1st and 99th percentile of data
    # reshape_and_normalize_data pads the train_data with an empty channel axis if it doesn't have one (single channel images/volumes). 
    train_data, test_data, run_test = reshape_and_normalize_data(train_data, 
                                                                 test_data=test_data, 
                                                                 channels=channels,
                                                                 channel_axis=channel_axis,
                                                                 normalize=normalize, 
                                                                 omni=omni, 
                                                                 dim=dim)

    if train_data is None:
        error_message = 'training data do not all have the same number of channels'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        return

    if not run_test:
        test_data, test_labels = None, None
        
    if not np.all([dta.shape[-dim:] == lbl.shape[-dim:] for dta, lbl in zip(train_data,train_labels)]):
        print([(dta.shape[-dim:],lbl.shape[-dim:]) for dta, lbl in zip(train_data,train_labels)])
        error_message = 'training data and labels are not the same shape, must be something wrong with preprocessing assumptions'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        return

    return train_data, train_labels, test_data, test_labels, run_test

def reshape_and_normalize_data(train_data, test_data=None, channels=None, channel_axis=0, normalize=True, omni=False, dim=2):
    """ inputs converted to correct shapes for *training* and rescaled so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities in each channel

    Parameters
    --------------

    train_data: list of ND-arrays, float
        list of training images of size [Ly x Lx], [nchan x Ly x Lx], or [Ly x Lx x nchan]

    test_data: list of ND-arrays, float (optional, default None)
        list of testing images of size [Ly x Lx], [nchan x Ly x Lx], or [Ly x Lx x nchan]

    channels: list of int of length 2 (optional, default None)
        First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        For instance, to train on grayscale images, input [0,0]. To train on images with cells
        in green and nuclei in blue, input [2,3].

    normalize: bool (optional, True)
        normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

    Returns
    -------------

    train_data: list of ND-arrays, float
        list of training images of size [2 x Ly x Lx]

    test_data: list of ND-arrays, float (optional, default None)
        list of testing images of size [2 x Ly x Lx]

    run_test: bool
        whether or not test_data was correct size and is useable during training

    """

    for test, data in enumerate([train_data, test_data]):
        if data is None:
            return train_data, test_data, False
        nimg = len(data)
        for i in range(nimg):
            if channels is None:
                if channel_axis is not None:
                    data[i] = move_axis_new(data[i], axis=channel_axis, pos=0) 
                else:
                    m = f'No channel axis specified. Image shape is {data[i].shape}. Supply channel_axis if incorrect.'
                    transforms_logger.warning(m)                
            
            if channels is not None:
                data[i] = reshape(data[i], channels=channels, chan_first=True, channel_axis=channel_axis) # the cuplrit with 3D

            # if data[i].ndim < 3:
            #     data[i] = data[i][np.newaxis,:,:]
            # we actually want this padding for single-channel volumes too
            
            # data with multiple channels will have channels defined and have an axis already; could also pass in nchan to avoid this assumption 
            # instead of this, we could just make the other parts of the code not rely on a channel axis and slice smarter 
            if channels is None and data[i].ndim==dim: 
                data[i] = data[i][np.newaxis]
            
            if normalize:
                data[i] = normalize_img(data[i], axis=0, omni=omni)

    return train_data, test_data, True

def resize_image(img0, Ly=None, Lx=None, rsz=None, interpolation=1, no_channels=False):
    """ resize image for computing flows / unresize for computing dynamics

    Parameters
    -------------

    img0: ND-array
        image of size [Y x X x nchan] or [Lz x Y x X x nchan] or [Lz x Y x X]

    Ly: int, optional

    Lx: int, optional

    rsz: float, optional
        resize coefficient(s) for image; if Ly is None then rsz is used

    interpolation: cv2 interp method (optional, default 1)

    Returns
    --------------

    imgs: ND-array 
        image of size [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

    """
    if Ly is None and rsz is None:
        error_message = 'must give size to resize to or factor to use for resizing'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)

    if Ly is None:
        # determine Ly and Lx using rsz
        if not isinstance(rsz, list) and not isinstance(rsz, np.ndarray):
            rsz = [rsz, rsz]
        if no_channels:
            Ly = int(img0.shape[-2] * rsz[-2])
            Lx = int(img0.shape[-1] * rsz[-1])
        else:
            Ly = int(img0.shape[-3] * rsz[-2])
            Lx = int(img0.shape[-2] * rsz[-1])
    
    # no_channels useful for z-stacks, so the third dimension is not treated as a channel
    # but if this is called for grayscale images, they first become [Ly,Lx,2] so ndim=3 but 
    if (img0.ndim>2 and no_channels) or (img0.ndim==4 and not no_channels):
        if no_channels:
            imgs = np.zeros((img0.shape[0], Ly, Lx), np.float32)
        else:
            imgs = np.zeros((img0.shape[0], Ly, Lx, img0.shape[-1]), np.float32)
        for i,img in enumerate(img0):
            imgs[i] = cv2.resize(img, (Lx, Ly), interpolation=interpolation)
            # imgs[i] = scipy.ndimage.zoom(img, resize/np.array(img.shape), order=order)
            
    else:
        imgs = cv2.resize(img0, (Lx, Ly), interpolation=interpolation)
    return imgs

def pad_image_ND(img0, div=16, extra=1, dim=2):
    """ pad image for test-time so that its dimensions are a multiple of 16 (2D or 3D)

    Parameters
    -------------

    img0: ND-array
        image of size [nchan (x Lz) x Ly x Lx]

    div: int (optional, default 16)

    Returns
    --------------

    I: ND-array
        padded image

    ysub: array, int
        yrange of pixels in I corresponding to img0

    xsub: array, int
        xrange of pixels in I corresponding to img0

    """
    inds = [k for k in range(-dim,0)]
    Lpad = [int(div * np.ceil(img0.shape[i]/div) - img0.shape[i]) for i in inds]
    pad1 = [extra*div//2 + Lpad[k]//2 for k in range(dim)]
    pad2 = [extra*div//2 + Lpad[k] - Lpad[k]//2 for k in range(dim)]
    
    emptypad = tuple([[0,0]]*(img0.ndim-dim))
    pads = emptypad+tuple(np.stack((pad1,pad2),axis=1))
    
    # changed from 'constant' - avoids a lot of edge artifacts!!!
    # any option that extends the data naturally will do... reflect seems to be the best 
    mode = 'reflect'
    I = np.pad(img0,pads,mode=mode)
    

    shape = img0.shape[-dim:] 
    subs = [np.arange(pad1[k],pad1[k]+shape[k]) for k in range(dim)]
    
    return I, subs

def random_rotate_and_resize(X, Y=None, scale_range=1., gamma_range=[.5,4], tyx=None, 
                             do_flip=True, rescale=None, unet=False,
                             inds=None, omni=False, dim=2, nchan=1, nclasses=3, device=None):
    """ augmentation by random rotation and resizing

        X and Y are lists or arrays of length nimg, with dims channels x Ly x Lx (channels optional)

        Parameters
        ----------
        X: LIST of ND-arrays, float
            list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]

        Y: LIST of ND-arrays, float (optional, default None)
            list of image labels of size [nlabels x Ly x Lx] or [Ly x Lx]. The 1st channel
            of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
            If Y.shape[0]==3 and not unet, then the labels are assumed to be [cell probability, Y flow, X flow]. 
            If unet, second channel is dist_to_bound.

        scale_range: float (optional, default 1.0)
            Range of resizing of images for augmentation. Images are resized by
            (1-scale_range/2) + scale_range * np.random.rand()
        
        gamma_range: float (optional, default 0.5)
           Images are gamma-adjusted im**gamma for gamma in (1-gamma_range,1+gamma_range) 

        xy: tuple, int (optional, default (224,224))
            size of transformed images to return

        do_flip: bool (optional, default True)
            whether or not to flip images horizontally

        rescale: array, float (optional, default None)
            how much to resize images by before performing augmentations

        unet: bool (optional, default False)

        Returns
        -------
        imgi: ND-array, float
            transformed images in array [nimg x nchan x xy[0] x xy[1]]

        lbl: ND-array, float
            transformed labels in array [nimg x nchan x xy[0] x xy[1]]

        scale: array, float
            amount each image was resized by

    """
    scale_range = max(0, min(2, float(scale_range))) # limit overall range to [0,2] i.e. 1+-1 

    if inds is None: # only relevant when debugging 
        nimg = len(X)
        inds = np.arange(nimg)
    
    return omnipose.core.random_rotate_and_resize(X, Y=Y, scale_range=scale_range, gamma_range=gamma_range,
                                                    tyx=tyx, do_flip=do_flip, rescale=rescale, inds=inds, 
                                                    nchan=nchan)

    # if omni and OMNI_INSTALLED:
    #     return omnipose.core.random_rotate_and_resize(X, Y=Y, scale_range=scale_range, gamma_range=gamma_range,
    #                                                   tyx=tyx, do_flip=do_flip, rescale=rescale, inds=inds, 
    #                                                   nchan=nchan)
    # else:
    #     # backwards compatibility; completely 'stock', no gamma augmentation or any other extra frills. 
    #     # [Y[i][1:] for i in inds] is necessary because the original transform function does not use masks (entry 0). 
    #     # This used to be done in the original function call. 
    #     if tyx is None:
    #         tyx = (224,)*dim
    #     print('yoyo',X[0].shape,Y[0].shape)
    #     return original_random_rotate_and_resize(X, Y=[y[1:] for y in Y] if Y is not None else None, 
    #                                              scale_range=scale_range, xy=tyx,
    #                                              do_flip=do_flip, rescale=rescale, unet=unet)


# I have the omni flag here just in case, but it actually does not affect the tests
def normalize_field(mu,omni=False):
    if omni and OMNI_INSTALLED:
        mu = omnipose.utils.normalize_field(mu) 
    else:
        mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)
    return mu


def _X2zoom(img, X2=1):
    """ zoom in image

    Parameters
    ----------
    img : numpy array that's Ly x Lx

    Returns
    -------
    img : numpy array that's Ly x Lx

    """
    ny,nx = img.shape[:2]
    img = cv2.resize(img, (int(nx * (2**X2)), int(ny * (2**X2))))
    return img

def _image_resizer(img, resize=512, to_uint8=False):
    """ resize image

    Parameters
    ----------
    img : numpy array that's Ly x Lx

    resize : int
        max size of image returned

    to_uint8 : bool
        convert image to uint8

    Returns
    -------
    img : numpy array that's Ly x Lx, Ly,Lx<resize

    """
    ny,nx = img.shape[:2]
    if to_uint8:
        if img.max()<=255 and img.min()>=0 and img.max()>1:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32)
            img -= img.min()
            img /= img.max()
            img *= 255
            img = img.astype(np.uint8)
    if np.array(img.shape).max() > resize:
        if ny>nx:
            nx = int(nx/ny * resize)
            ny = resize
        else:
            ny = int(ny/nx * resize)
            nx = resize
        shape = (nx,ny)
        img = cv2.resize(img, shape)
        img = img.astype(np.uint8)
    return img


def original_random_rotate_and_resize(X, Y=None, scale_range=1., xy = (224,224), 
                             do_flip=True, rescale=None, unet=False):
    """ augmentation by random rotation and resizing
        X and Y are lists or arrays of length nimg, with dims channels x Ly x Lx (channels optional)
        
        Parameters
        ----------
        X: LIST of ND-arrays, float
            list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]
        Y: LIST of ND-arrays, float (optional, default None)
            list of image labels of size [nlabels x Ly x Lx] or [Ly x Lx]. The 1st channel
            of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
            If Y.shape[0]==3 and not unet, then the labels are assumed to be [cell probability, Y flow, X flow]. 
            If unet, second channel is dist_to_bound.
        scale_range: float (optional, default 1.0)
            Range of resizing of images for augmentation. Images are resized by
            (1-scale_range/2) + scale_range * np.random.rand()
        xy: tuple, int (optional, default (224,224))
            size of transformed images to return
        do_flip: bool (optional, default True)
            whether or not to flip images horizontally
        rescale: array, float (optional, default None)
            how much to resize images by before performing augmentations
        unet: bool (optional, default False)
        
        Returns
        -------
        imgi: ND-array, float
            transformed images in array [nimg x nchan x xy[0] x xy[1]]
        lbl: ND-array, float
            transformed labels in array [nimg x nchan x xy[0] x xy[1]]
        scale: array, float
            amount by which each image was resized
    """

    print('this',X[0].shape)
    scale_range = max(0, min(2, float(scale_range)))
    nimg = len(X)
    if X[0].ndim>2:
        nchan = X[0].shape[0]
    else:
        nchan = 1
    imgi  = np.zeros((nimg, nchan, xy[0], xy[1]), np.float32)

    lbl = []
    if Y is not None:
        if Y[0].ndim>2:
            nt = Y[0].shape[0]
        else:
            nt = 1
        lbl = np.zeros((nimg, nt, xy[0], xy[1]), np.float32)

    scale = np.zeros(nimg, np.float32)
    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]

        # generate random augmentation parameters
        flip = np.random.rand()>.5
        theta = np.random.rand() * np.pi * 2
        scale[n] = (1-scale_range/2) + scale_range * np.random.rand()
        if rescale is not None:
            scale[n] *= 1. / rescale[n]
        dxy = np.maximum(0, np.array([Lx*scale[n]-xy[1],Ly*scale[n]-xy[0]]))
        dxy = (np.random.rand(2,) - .5) * dxy

        # create affine transform
        cc = np.array([Lx/2, Ly/2])
        cc1 = cc - np.array([Lx-xy[1], Ly-xy[0]])/2 + dxy
        pts1 = np.float32([cc,cc + np.array([1,0]), cc + np.array([0,1])])
        pts2 = np.float32([cc1,
                cc1 + scale[n]*np.array([np.cos(theta), np.sin(theta)]),
                cc1 + scale[n]*np.array([np.cos(np.pi/2+theta), np.sin(np.pi/2+theta)])])
        M = cv2.getAffineTransform(pts1,pts2)

        img = X[n].copy()
        
        if Y is not None:
            labels = Y[n].copy()
            if labels.ndim<3:
                labels = labels[np.newaxis,:,:]

        if flip and do_flip:
            img = img[..., ::-1]
            if Y is not None:
                labels = labels[..., ::-1]
                if nt > 1 and not unet:
                    labels[2] = -labels[2]

        for k in range(nchan):
            I = cv2.warpAffine(img[k], M, (xy[1],xy[0]), flags=1)
            imgi[n,k] = I

        if Y is not None:
            for k in range(nt):
                if k==0:
                    lbl[n,k] = cv2.warpAffine(labels[k], M, (xy[1],xy[0]), flags=0)
                else:
                    lbl[n,k] = cv2.warpAffine(labels[k], M, (xy[1],xy[0]), flags=1)

            if nt > 1 and not unet:
                v1 = lbl[n,2].copy()
                v2 = lbl[n,1].copy()
                lbl[n,1] = (-v1 * np.sin(-theta) + v2*np.cos(-theta))
                lbl[n,2] = (v1 * np.cos(-theta) + v2*np.sin(-theta))

    return imgi, lbl, scale