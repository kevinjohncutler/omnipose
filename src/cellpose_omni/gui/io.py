import os, datetime, gc, warnings, glob, shutil, copy
import numpy as np
import cv2
import fastremap 

from .. import utils, models
from ..io import imread, imsave, outlines_to_text, logger_setup

from PyQt6.QtWidgets import QFileDialog

logger, log_file = logger_setup()

import ncolor
from omnipose.utils import sinebow
from omnipose import core

GC = True # global toggle for garbage collection

import imageio.v3 as iio
def read_image_with_channel_axis(filename):
    """
    Reads an image using imageio.v3 and determines its channel axis using metadata.
    
    Priority is given to an 'axes' key in the metadata:
      - If present and if 'C' is found, that index is used.
    
    Otherwise, if a 'mode' key is present (e.g. "RGB", "RGBA", "CMYK"),
    the function checks whether the loaded image has an extra dimension compared
    to the spatial shape given in meta['shape']. For example, if meta['mode'] is "RGBA"
    and meta['shape'] is (height, width), we expect the image to have shape (height, width, 4)
    and the channel axis is assumed to be -1.
    
    If neither an 'axes' nor an applicable 'mode' is found, or if the mode indicates a single-channel image,
    the function returns None for the channel axis.
    
    Returns:
        img: The image as a NumPy array.
        channel_axis: The axis index corresponding to channels, or None if not applicable.
        meta: The metadata dictionary.
    """
    img = iio.imread(filename)
    # meta = iio.immeta(filename)
    try:
        meta = iio.immeta(filename)  
    except Exception as e:
        print(f"Warning: could not load metadata for {filename}: {e}")
        meta = {}
    
    print('image shape', img.shape)
    # Check if the image is grayscale saved as RGB

    if 'axes' in meta:
        axes = meta['axes']
        # If 'C' is present, use its index.
        if 'C' in axes:
            channel_axis = axes.index('C')
            # convert to negative index
            if channel_axis == img.ndim:
                channel_axis = -1
        else:
            channel_axis = None
    
    elif 'mode' in meta:
        mode = meta['mode']
        # Only consider modes that imply multiple channels.
        if mode in ('RGB', 'RGBA', 'CMYK'):
            channel_axis = -1
        else:
            # Modes like 'L' (grayscale) imply no separate channel axis.
            channel_axis = None
    else:
        # If neither 'axes' nor 'mode' is provided, we return None.
        channel_axis = None
    
    return img, channel_axis, meta

def _init_model_list(parent):
    models.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    parent.model_list_path = os.fspath(models.MODEL_DIR.joinpath('gui_models.txt'))
    parent.model_strings = []
    if not os.path.exists(parent.model_list_path):
        textfile = open(parent.model_list_path, 'w')
        textfile.close()
    else:
        with open(parent.model_list_path, 'r') as textfile:
            lines = [line.rstrip() for line in textfile]
            if len(lines) > 0:
                parent.model_strings.extend(lines)
    
def _add_model(parent, filename=None, load_model=True):
    if filename is None:
        name = QFileDialog.getOpenFileName(
            parent, "Add model to GUI"
            )
        filename = name[0]
    fname = os.path.split(filename)[-1]
    try:
        shutil.copyfile(filename, os.fspath(models.MODEL_DIR.joinpath(fname)))
    except shutil.SameFileError:
        pass
    logger.info(f'{filename} copied to models folder {os.fspath(models.MODEL_DIR)}')
    with open(parent.model_list_path, 'a') as textfile:
        textfile.write(fname + '\n')
    parent.ModelChoose.addItems([fname])
    parent.model_strings.append(fname)
    if len(parent.model_strings) > 0:
        # parent.ModelButton.setStyleSheet(parent.styleUnpressed)
        parent.ModelButton.setEnabled(True)
    
    for ind, model_string in enumerate(parent.model_strings[:-1]):
        if model_string == fname:
            _remove_model(parent, ind=ind+1, verbose=False)

    parent.ModelChoose.setCurrentIndex(len(parent.model_strings))
    if load_model:
        # parent.model_choose(len(parent.model_strings))
        parent.model_choose()
        

def _remove_model(parent, ind=None, verbose=True):
    if ind is None:
        ind = parent.ModelChoose.currentIndex()
    if ind > 0:
        ind -= 1
        if verbose:
            logger.info(f'deleting {parent.model_strings[ind]} from GUI')
        parent.ModelChoose.removeItem(ind+1)
        del parent.model_strings[ind]
        custom_strings = parent.model_strings
        if len(custom_strings) > 0:
            with open(parent.model_list_path, 'w') as textfile:
                for fname in custom_strings:
                    textfile.write(fname + '\n')
            parent.ModelChoose.setCurrentIndex(len(parent.model_strings))
        else:
            # write empty file
            textfile = open(parent.model_list_path, 'w')
            textfile.close()
            parent.ModelChoose.setCurrentIndex(0)
            parent.ModelButton.setEnabled(False)
    else:
        print('ERROR: no model selected to delete')

    

def _get_train_set(image_names):
    """ get training data and labels for images in current folder image_names"""
    train_data, train_labels, train_files = [], [], []
    for image_name_full in image_names:
        image_name = os.path.splitext(image_name_full)[0]
        label_name = None
        if os.path.exists(image_name + '_seg.npy'):
            dat = np.load(image_name + '_seg.npy', allow_pickle=True).item()
            masks = dat['masks'].squeeze()
            if masks.ndim==2:
                fastremap.renumber(masks, in_place=True)
                label_name = image_name + '_seg.npy'
            else:
                logger.info(f'_seg.npy found for {image_name} but masks.ndim!=2')
        if label_name is not None:
            train_files.append(image_name_full)
            train_data.append(imread(image_name_full))
            train_labels.append(masks)
    return train_data, train_labels, train_files

def _load_image(parent, filename=None, load_seg=True):
    """ load image with filename; if None, open QFileDialog """
    
    if filename is None:
        name = QFileDialog.getOpenFileName(
            parent, "Load image"
            )
        filename = name[0]
    
    if hasattr(parent, 'hist'):
        parent.hist.view_states = {}
        
    # from here, we now will just be loading an image and a mask image file format, not npy 
    try:
    
        logger.info(f'[_load image] loading image: {filename}')
        image, channel_axis, meta = read_image_with_channel_axis(filename)
        logger.info(f'[_load image] image shape: {image.shape}, channel_axis: {channel_axis}')

        # transform image to CYX
        # if image.ndim ==3 and image.shape[-1] == 3:
        #     print('Assuming RGB image, converting to CYX')
        #     image = np.transpose(image, (2,0,1))
        _initialize_images(parent, image, channel_axis)
        parent.reset()
        parent.recenter()
        
        parent.loaded = True
    except Exception as e:
        print('ERROR: images not compatible')
        print(f'ERROR: {e}')
        
    
        
    logger.info(f'called _load_image on {filename}')
    manual_file = os.path.splitext(filename)[0]+'_seg.npy'
    load_mask = False
    if load_seg:
        if os.path.isfile(manual_file) and not parent.autoloadMasks.isChecked():
            logger.info(f'segmentation npy file found: {manual_file}')
            _load_seg(parent, manual_file, image=imread(filename), image_file=filename)
            return # exit here, will not go on to load any mask files 
            
        elif os.path.isfile(os.path.splitext(filename)[0]+'_manual.npy'):
            logger.info(f'manual npy file found: {manual_file}')
            manual_file = os.path.splitext(filename)[0]+'_manual.npy'
            _load_seg(parent, manual_file, image=imread(filename), image_file=filename)
            return # likewise exit here, will not go on to load any mask files 
            # should merege this branch with the above? Not sure what use case manual npy is 
            
            
        elif parent.autoloadMasks.isChecked():
            logger.info('loading masks from _masks.tif file')
            mask_file = os.path.splitext(filename)[0]+'_masks'+os.path.splitext(filename)[-1]
            mask_file = os.path.splitext(filename)[0]+'_masks.tif' if not os.path.isfile(mask_file) else mask_file
            load_mask = True if os.path.isfile(mask_file) else False
    else:
        logger.info('not loading segmentation, just the image')
        

    if parent.loaded:
        logger.info(f'[_load_image] loaded image shape: {image.shape}')
        # parent.reset(image)
        parent.filename = filename
        # filename = os.path.split(parent.filename)[-1]
        # _initialize_images(parent, image)
        parent.clear_all()
        # parent.loaded = True
        parent.enable_buttons()
        if load_mask:
            print('loading masks')
            _load_masks(parent, filename=mask_file)
        # parent.threshslider.setEnabled(False)
        # parent.probslider.setEnabled(False)

            
def _initialize_images(parent, image, channel_axis):
    """
    Normalize an image array to shape (z, height, width, 3) using the provided channel_axis.

    Parameters
    ----------
    image : numpy.ndarray
        The loaded image, which may be 2D (H, W), 3D, or 4D.
    channel_axis : int or None
        If an int, it indicates the axis that holds the channel data.
        If None, the image is assumed to be grayscale.

    Returns
    -------
    image : numpy.ndarray
        The normalized image with shape (z, height, width, 3).
    """
    
    parent.onechan = channel_axis is None
    parent.shape = image.shape
    logger.info(f'initializing image, original shape: {image.shape}')
    
    # check if the image is grayscale saved as RGB
    # not usually a good idea, but there are valid reasons to do this 
    # for compatibility with image editing software

    if channel_axis in (-1,None) and image.shape[-1] in (3,4):
        tol = 1e-3
        gray_mask = np.sum(np.diff(image[...,:3],axis=-1),axis=-1) < tol
        if np.all(gray_mask):
            image = image[...,0]

        logger.info('Detected RGB image with grayscale data, converting to single-channel')    
        channel_axis = None
        parent.onechan = True
        
        
    # Move the specified channel axis to the last position (if any).
    if not parent.onechan:
        image = np.moveaxis(image, channel_axis, -1)
        # If we end up with a 3D array, check if the last dimension is color
        # (3 or 4 channels). If so, treat it as (H, W, C) for a single-plane image.
        if image.ndim == 3:
            if image.shape[-1] in (3, 4):
                # Leave it as (H, W, C).
                pass
            else:
                # It's likely (Z, H, W) => add a channel axis for single-channel data.
                image = image[np.newaxis, ...]
    else:
        # No explicit channel axis was given => assume grayscale.
        if image.ndim == 2:
            # (H, W) => add a "Z" axis of size 1 and a channel axis of size 1 => (1, H, W, 1)
            image = image[np.newaxis, ...]
            image = image[..., np.newaxis]
        elif image.ndim == 3:
            # If the image is (Z, H, W) grayscale, add a channel axis if not present
            if image.shape[-1] not in (3, 4):
                image = image[..., np.newaxis]
        else:
            raise ValueError(f'Unexpected shape for a grayscale image: {image.shape}')

    # If we're left with 3D, assume single-plane color => shape (H, W, C).
    # Convert to (1, H, W, C) for consistency with the rest of the code.
    if image.ndim == 3:
        if image.shape[-1] > 4:
            raise ValueError(f'Unexpected channel dimension: {image.shape}')
        image = image[np.newaxis, ...]

    # Ensure we have 4D data: (Z, H, W, C).
    if image.ndim != 4:
        raise ValueError(f'Unexpected image dimensions after processing: {image.shape}')

    # If the channel dimension is more than 3, slice off extras (e.g. drop alpha).
    if image.shape[-1] > 3:
        image = image[..., :3]

    logger.info(f'[_initialize_images] normalized image shape: {image.shape}')

    # Normalize to [0, 255].
    img_min = image.min()
    img_max = image.max()
    image = image.astype(np.float32)
    image -= img_min
    if img_max > img_min + 1e-3:
        image /= (img_max - img_min)
    # image *= 255

    # Update parent fields.
    parent.stack = image
    parent.NZ = image.shape[0]
    parent.scroll.setMaximum(parent.NZ - 1)
    parent.Ly, parent.Lx = image.shape[1:3]
    parent.layerz = np.zeros((parent.Ly, parent.Lx, 4), dtype=np.uint8)

    # Recompute saturations if needed.
    if parent.autobtn.isChecked() or len(parent.saturation) != parent.NZ:
        parent.compute_saturation()
    parent.compute_scale()
    parent.currentZ = int(np.floor(parent.NZ / 2))
    parent.scroll.setValue(parent.currentZ)
    parent.zpos.setText(str(parent.currentZ))
    parent.track_changes = []
    parent.recenter()
    
    return image
    

def _load_seg(parent, filename=None, image=None, image_file=None, channel_axis=None):
    """ load *_seg.npy with filename; if None, open QFileDialog """
    
    logger.info(f'loading segmentation: {filename}')
    
    if filename is None:
        name = QFileDialog.getOpenFileName(
            parent, "Load labelled data", filter="*.npy"
            )
        filename = name[0]
    try:
        dat = np.load(filename, allow_pickle=True).item()
        dat['masks'] # test if masks are present
        parent.loaded = True
    except:
        parent.loaded = False
        print('ERROR: not NPY')
        return
    
    if image is None:
        logger.info(f'loading image in _load_seg')
        found_image = False
        if 'filename' in dat:
            parent.filename = dat['filename']
            if os.path.isfile(parent.filename):
                parent.filename = dat['filename']
                found_image = True
            else:
                imgname = os.path.split(parent.filename)[1]
                root = os.path.split(filename)[0]
                parent.filename = root+'/'+imgname
                if os.path.isfile(parent.filename):
                    found_image = True
        if found_image:
            try:
                # image = imread(parent.filename)
                print('reading here')
                image, channel_axis, meta = read_image_with_channel_axis(filename)

            except:
                parent.loaded = False
                found_image = False
                print('ERROR: cannot find image file, loading from npy')
        # if not found_image:
        #     parent.filename = filename[:-11]
        #     if 'img' in dat:
        #         image = dat['img']
        #     else:
        #         print('ERROR: no image file found and no image in npy')
        #         return
    else:
        parent.filename = image_file

    logger.info(f'loaded image in _load_seg with shape {image.shape}')
    _initialize_images(parent,image, channel_axis)
    parent.reset()# this puts in some defaults if they are not present in the npy file
    
    
    if 'chan_choose' in dat:
        parent.ChannelChoose[0].setCurrentIndex(dat['chan_choose'][0])
        parent.ChannelChoose[1].setCurrentIndex(dat['chan_choose'][1])
    
    
    # Transfer fields from dat to parent directly
    exclude = ['runstring', 'img', 'zpos'] # these are not to be transferred because their formats are different when saved vs in the GUI
    for key, value in dat.items():
        if key not in exclude:
            setattr(parent, key, value)
            # print('setting', key)

            
    if 'runstring' in dat:
        parent.runstring.setPlainText(dat['runstring'])

    # fix formats using -1 as background
    if parent.masks.min()==-1:
        logger.warning('-1 found in masks, running formatting')
        parent.masks = ncolor.format_labels(parent.masks)
        
        
    # Update masks and outlines to ZYX format stored as parent.mask_stack and parent.outl_stack
    if parent.masks.ndim == 2:
        parent.mask_stack = parent.masks[np.newaxis, :, :]
        parent.outl_stack = parent.bounds[np.newaxis, :, :]

        
    if not hasattr(parent, 'links'):
        parent.links = None
    
    # we want to initialize the segmentation infrastructure like steps and coords, 
    # this also will create the affinity graph if not present 
    parent.initialize_seg()   
    parent.ncells = parent.masks.max()
    
    # handle colors - I feel like this needs improvement 
    if 'colors' in dat and len(dat['colors'])>=dat['masks'].max(): #== too sctrict, >= is fine 
        colors = dat['colors']
    else:
        colors = parent.colormap[:parent.ncells,:3]
    parent.cellcolors = np.append(parent.cellcolors, colors, axis=0)
    

    if 'est_diam' in dat:
        parent.Diameter.setText('%0.1f'%dat['est_diam'])
        parent.diameter = dat['est_diam']
        parent.compute_scale()
        
    if 'manual_changes' in dat: 
        parent.track_changes = dat['manual_changes']
        logger.info('loaded in previous changes')    
    if 'zdraw' in dat:
        parent.zdraw = dat['zdraw']
    else:
        parent.zdraw = [None for n in range(parent.ncells)]

    # print('dat contents',dat.keys())
    # ['outlines', 'colors', 'masks', 'chan_choose', 'img', 'filename', 'flows', 'ismanual', 'manual_changes', 'model_path', 'flow_threshold', 'cellprob_threshold', 'runstring'])
    
    parent.ismanual = np.zeros(parent.ncells, bool)
    if 'ismanual' in dat:
        if len(dat['ismanual']) == parent.ncells:
            parent.ismanual = dat['ismanual']

    if 'current_channel' in dat:
        logger.info(f'current channel: {dat["current_channel"]}')
        parent.color = (dat['current_channel']+2)%5
        parent.RGBDropDown.setCurrentIndex(parent.color)

    if 'flows' in dat:
        parent.flows = dat['flows']
        
        try:
            if parent.flows[0].shape[-3]!=dat['masks'].shape[-2]:
                Ly, Lx = dat['masks'].shape[-2:]
                
                for i in range[3]:
                    parent.flows[i] = cv2.resize(parent.flows[i].squeeze(), (Lx, Ly), interpolation=0)[np.newaxis,...]

            if parent.NZ==1:
                parent.recompute_masks = True
            else:
                parent.recompute_masks = False
                
        except:
            try:
                if len(parent.flows[0])>0:
                    parent.flows = parent.flows[0]
            except:
                parent.flows = [[],[],[],[],[[]]]
            parent.recompute_masks = False
            

    # Added functionality to jump right back into parameter tuning from saved flows 
    if 'model_path' in dat:
        parent.current_model = dat['model_path']
        # parent.initialize_model() 
    
    
    #  reinit overlay item 
    if hasattr(parent, 'pixelGridOverlay'):
        logger.info(f'resetting pixel grid')
        parent.pixelGridOverlay.reset()
    else:
        logger.info(f'no pixelGridOverlay to reset')
        
    
    if 'hist_states' in dat:
        parent.hist.view_states = dat['hist_states']
        logger.info(f'Loaded histogram states from seg file')
    else:
        parent.hist.view_states = {}
        logger.info('No histogram states found in seg file; resetting.')

    
    
    parent.enable_buttons()
    parent.update_layer()
    logger.info('loaded segmentation, enabling buttons')
    
    
    # important for enabling things  
    parent.loaded = True  
    
    del dat
    if GC: gc.collect()

def _load_masks(parent, filename=None):
    """ load zero-based masks (0=no cell, 1=cell one, ...) """
    if filename is None:
        name = QFileDialog.getOpenFileName(
            parent, "Load masks (PNG or TIFF)"
            )
        filename = name[0]
    logger.info(f'loading masks: {filename}')
    masks = imread(filename)
    parent.masks = masks
    
    # parent.initialize_seg() # redudnat if initialize_seg is called in _masks_to_gui

    if masks.shape[0]!=parent.NZ:
        print('ERROR: masks are not same depth (number of planes) as image stack')
        return
    
    _masks_to_gui(parent)
    
    # del masks 
    if GC: gc.collect()
    parent.update_layer()
    parent.update_plot()
    
    
def _masks_to_gui(parent, format_labels=False):
    """ masks loaded into GUI """
    masks = parent.masks
    shape = masks.shape 
    ndim = masks.ndim
    # if format_labels:
    #     masks = ncolor.format_labels(masks,clean=True)
    # else:
    #     fastremap.renumber(masks, in_place=True)
    logger.info(f'{parent.ncells} masks found')
    
    # print('calling masks to gui',masks.shape)
    parent.ncells = masks.max() #try to grab the cell count before ncolor

    np.random.seed(42) #stability for ncolor, should not be needed at this point 

    if parent.ncolor:
        masks, ncol = ncolor.label(masks,return_n=True) 
    else:
        masks = np.reshape(masks, shape)
        masks = masks.astype(np.uint16) if masks.max()<(2**16-1) else masks.astype(np.uint32)
        
        # the intrinsic values are masks and bounds, but I will use the old lingo
    # of mask_stack and outl_stack for the draw_layer function expecting ZYX stacks 
    
    if ndim==2:
        # print('reshaping masks to mask_stack stack')
        parent.mask_stack = masks[np.newaxis,:,:]
        parent.outl_stack = parent.bounds[np.newaxis,:,:]


    
    if parent.ncolor:
        # Approach 1: use a dictionary to color cells but keep their original label
        # Approach 2: actually change the masks to n-color
        # 2 is easier and more stable for editing. Only downside is that exporting will
        # require formatting and users may need to shuffle or add a color to avoid like
        # colors touching 
        # colors = parent.colormap[np.linspace(0,255,parent.ncells+1).astype(int), :3]
        c = sinebow(ncol+1)
        colors = (np.array(list(c.values()))[1:,:3] * (2**8-1) ).astype(np.uint8)

    else:
        colors = parent.colormap[:parent.ncells, :3]
        
    logger.info('creating cell colors and drawing masks')
    parent.cellcolors = np.concatenate((np.array([[255,255,255]]), colors), axis=0).astype(np.uint8)
    
    parent.draw_layer()
    if parent.ncells>0:
        parent.toggle_mask_ops()
    parent.ismanual = np.zeros(parent.ncells, bool)
    parent.zdraw = list(-1*np.ones(parent.ncells, np.int16))
    
    parent.update_layer()
    # parent.update_plot()
    parent.update_shape()
    parent.initialize_seg()
    
    
def _save_png(parent):
    """ save masks to png or tiff (if 3D) """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ==1:
        if parent.mask_stack[0].max() > 65534:
            logger.info('saving 2D masks to tif (too many masks for PNG)')
            imsave(base + '_cp_masks.tif', parent.mask_stack[0])
        else:
            logger.info('saving 2D masks to png')
            imsave(base + '_cp_masks.png', parent.mask_stack[0].astype(np.uint16))
    else:
        logger.info('saving 3D masks to tiff')
        imsave(base + '_cp_masks.tif', parent.mask_stack)

def _save_outlines(parent):
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ==1:
        logger.info('saving 2D outlines to text file, see docs for info to load into ImageJ')    
        outlines = utils.outlines_list(parent.mask_stack[0])
        outlines_to_text(base, outlines)
    else:
        print('ERROR: cannot save 3D outlines')
    

# def _save_sets(parent):
#     """ save masks to *_seg.npy """
#     filename = parent.filename
#     base = os.path.splitext(filename)[0]
#     flow_threshold, cellprob_threshold = parent.get_thresholds()
    
#     # print(parent.cellcolors,'color')
    
#     if parent.NZ > 1 and parent.is_stack:
#         np.save(base + '_seg.npy',
#                 {'outlines': parent.outl_stack,
#                  'colors': parent.cellcolors[1:],
#                  'masks': parent.mask_stack,
#                  'current_channel': (parent.color-2)%5,
#                  'filename': parent.filename,
#                  'flows': parent.flows,
#                  'zdraw': parent.zdraw,
#                  'model_path': parent.current_model_path if hasattr(parent, 'current_model_path') else 0,
#                  'flow_threshold': flow_threshold,
#                  'cellprob_threshold': cellprob_threshold,
#                  'runstring': parent.runstring.toPlainText()
#                  })
#     else:
#         image = parent.chanchoose(parent.stack[parent.currentZ].copy())
#         if image.ndim < 4:
#             image = image[np.newaxis,...]
#         np.save(base + '_seg.npy',
#                 {'outlines': parent.outl_stack.squeeze(),
#                  'colors': parent.cellcolors[1:],
#                  'masks': parent.mask_stack.squeeze(),
#                  'chan_choose': [parent.ChannelChoose[0].currentIndex(),
#                                  parent.ChannelChoose[1].currentIndex()],
#                  'img': image.squeeze(),
#                  'filename': parent.filename,
#                  'flows': parent.flows,
#                  'ismanual': parent.ismanual,
#                  'manual_changes': parent.track_changes,
#                  'model_path': parent.current_model_path if hasattr(parent, 'current_model_path') else 0,
#                  'flow_threshold': flow_threshold,
#                  'cellprob_threshold': cellprob_threshold,
#                  'runstring': parent.runstring.toPlainText()
#                 })
#     #print(parent.point_sets)
#     logger.info('%d RoIs saved to %s'%(parent.ncells, base + '_seg.npy'))

def _save_sets(parent):
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    flow_threshold, cellprob_threshold = parent.get_thresholds()

    seg_dict = {
        'filename': parent.filename,
        'masks': parent.mask_stack.squeeze(),
        'bounds': parent.outl_stack.squeeze(),
        'affinity_graph': parent.affinity_graph,
        'colors': parent.cellcolors[1:], # skip the dummy color index=0
        'flows': parent.flows,
        'saturation': parent.saturation,
        'ismanual': parent.ismanual,
        'manual_changes': parent.track_changes,
        'zdraw': parent.zdraw,
        'model_path': getattr(parent, 'current_model_path', None),
        'flow_threshold': flow_threshold,
        'cellprob_threshold': cellprob_threshold,
        'runstring': parent.runstring.toPlainText(),
        'chan_choose': [
            parent.ChannelChoose[0].currentIndex(),
            parent.ChannelChoose[1].currentIndex()
        ],
    }

    # 1) If your histogram has a dict of states, store it
    #    So each image's LUT config is saved uniquely
    if hasattr(parent.hist, 'view_states'):
        seg_dict['hist_states'] = parent.hist.view_states

    np.save(base + '_seg.npy', seg_dict, allow_pickle=True)
    logger.info(f'{parent.ncells} RoIs + data saved to {base}_seg.npy')