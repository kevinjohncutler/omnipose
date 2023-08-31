import os, datetime, gc, warnings, glob
from natsort import natsorted
import numpy as np
import cv2
import tifffile
import logging, pathlib, sys
from pathlib import Path

from csv import reader, writer

try:
    from omnipose.utils import format_labels
    import ncolor
    OMNI_INSTALLED = True
except:
    OMNI_INSTALLED = False

from . import utils, plot, transforms

try:
    from PyQt6 import QtGui, QtCore, Qt, QtWidgets
    GUI = True
except:
    GUI = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False
    
try:
    from google.cloud import storage
    SERVER_UPLOAD = True
except:
    SERVER_UPLOAD = False

io_logger = logging.getLogger(__name__)

def logger_setup(verbose=False):
    cp_dir = pathlib.Path.home().joinpath('.cellpose')
    cp_dir.mkdir(exist_ok=True)
    log_file = cp_dir.joinpath('run.log')
    try:
        log_file.unlink()
    except:
        print('creating new log file')
    logging.basicConfig(
                    level=logging.DEBUG if verbose else logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(sys.stdout)
                    ]
                )
    logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG) # does not fix CLI
    logger.info(f'WRITING LOG OUTPUT TO {log_file}')
    #logger.handlers[1].stream = sys.stdout

    return logger, log_file

# helper function to check for a path; if it doesn't exist, make it 
def check_dir(path):
    if not os.path.isdir(path):
        # os.mkdir(path)
        os.makedirs(path,exist_ok=True)
        
        
def load_links(filename):
    """
    Read a txt or csv file with label links. 
    These should look like:
        1,2 
        1,3
        4,7
        6,19
        .
        .
        .
    Returns links as a set of tuples. 
    """
    if filename is not None and os.path.exists(filename):
        links = set()
        file = open(filename,"r")
        lines = reader(file)
        for l in lines: 
            links.add(tuple([int(num) for num in l]))
        return links
    else:
        return []

def write_links(savedir,basename,links):
    """
    Write label link file. See load_links() for its output format. 
    
    Parameters
    ----------
    savedir: string
        directory in which to save
    basename: string
        file name base to which _links.txt is appended. 
    links: set
        set of label tuples {(x,y),(z,w),...}

    """
    with open(os.path.join(savedir,basename+'_links.txt'), "w",newline='') as out:
        csv_out = writer(out)
        for row in links:
            csv_out.writerow(row)

def outlines_to_text(base, outlines):
    with open(base + '_cp_outlines.txt', 'w') as f:
        for o in outlines:
            xy = list(o.flatten())
            xy_str = ','.join(map(str, xy))
            f.write(xy_str)
            f.write('\n')

def imread(filename):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='.tiff':
        img = tifffile.imread(filename)
        return img
    else:
        try:
            img = cv2.imread(filename, -1)#cv2.LOAD_IMAGE_ANYDEPTH)
            if img.ndim > 2:
                img = img[..., [2,1,0]]
            return img
        except Exception as e:
            io_logger.critical('ERROR: could not read file, %s'%e)
            return None

def imsave(filename, arr):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='tiff':
        tifffile.imsave(filename, arr)
    else:
        if len(arr.shape)>2:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, arr)
#         skimage.io.imsave(filename, arr.astype()) #cv2 doesn't handle transparency

# now allows for any extension(s) to be specified, allowing exlcusion if necessary, non-image files, etc. 
def get_image_files(folder, mask_filter='_masks', img_filter=None, look_one_level_down=False,
                    extensions = ['png','jpg','jpeg','tif','tiff']):
    """ find all images in a folder and if look_one_level_down all subfolders """
    mask_filters = ['_cp_masks', '_cp_output', '_flows', mask_filter]
    image_names = []
    if img_filter is None:
        img_filter = ''
    
    folders = []
    if look_one_level_down:
        # folders = natsorted(glob.glob(os.path.join(folder, "*/")))
        folders = natsorted(glob.glob(os.path.join(folder, "*",'')))  #forward slash is unix only, this should generalize to windows too  
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
        if igood:
            imn.append(im)
    image_names = imn

    if len(image_names)==0:
        raise ValueError('ERROR: no images in --dir folder')
    
    return image_names


def getname(path,suffix=''):
    return os.path.splitext(Path(path).name)[0].replace(suffix,'')

# I modified this work better with the save_masks function. Complexity added for subfolder and directory flexibility,
# and simplifications made because we can safely assume how output was saved.
# the one place it is needed internally 
def get_label_files(img_names, label_filter='_cp_masks', img_filter='', ext=None,
                    dir_above=False, subfolder='', parent=None, flows=False, links=False):
    """
    Get the corresponding labels and flows for the given file images. If no extension is given,
    looks for TIF, TIFF, and PNG. If multiple are found, the first in the list is returned. 
    If extension is given, no checks for file existence are made - useful for finding nonstandard output like txt or npy. 
    
    Parameters
    ----------
    img_names: list, str
        list of full image file paths
    label_filter: str
        the label filter sufix, defaults to _cp_masks
        can be _flows, _ncolor, etc. 
    ext: str
        the label extension
        can be .tif, .png, .txt, etc. 
    img_filter: str
        the image filter suffix, e.g. _img
    dir_above: bool
        whether or not masks are stored in the image parent folder    
    subfolder: str
        the name of the subfolder where the labels are stored
    parent: str
        parent folder or list of folders where masks are stored, if different from images 
    flows: Bool
        whether or not to search for and return stored flows
    links: bool
        whether or not to search for and return stored link files 
     
    Returns
    -------
    list of all absolute label paths (str)
    
    """

    nimg = len(img_names)
    label_base = [getname(i,suffix=img_filter) for i in img_names]
    
    # allow for the user to specify where the labels are stored, either as a single directory
    # or as a list of directories matching the length of the image list
    if parent is None:
        if dir_above: # for when masks are stored in the directory above (usually in subfolder)
            parent = [Path(i).parent.parent.absolute() for i in img_names]
        else: # for when masks are stored in the same containing forlder as images (usually not in subfolder)
            parent = [Path(i).parent.absolute() for i in img_names]
    
    elif not isinstance(label_folder, list):
        parent = [parent]*nimg
    
    if ext is None:
        label_paths = []
        extensions = ['.tif','.tiff','.png'] #order preference comes here 

        for p,b in zip(parent,label_base):            
            paths = [os.path.join(p,subfolder,b+label_filter+ext) for ext in extensions]
            found = [os.path.exists(path) for path in paths]
            nfound = np.sum(found)
            
            if nfound == 0:
                io_logger.warning('No TIF, TIFF, or PNG labels of type {} found for image {}.'.format(label_filter, b))
            else:
                idx = np.nonzero(found)[0][0]
                label_paths.append(paths[idx])
                if nfound > 1:
                    io_logger.warning("""Multiple labels of type {} also 
                    found for image {}. Deferring to {} label.""".format(label_filter, b, extensions[idx]))
            
        
    else:
        label_paths = [os.path.join(p,subfolder,b+label_filter+ext) for p,b in zip(parent,label_base)]
    
    ret = [label_paths]

    if flows:
        flow_paths = []
        imfilters = ['',img_filter] # this allows both flow name conventions to exist in one folder 

        for p,b in zip(parent,label_base):            
            paths = [os.path.join(p,subfolder,b+imf+'_flows.tif') for imf in imfilters]
            found = [os.path.exists(path) for path in paths]
            nfound = np.sum(found)

            if nfound == 0:
                io_logger.info('not all flows are present, will run flow generation for all images')
                flow_paths = None
                break
            else:
                idx = np.nonzero(found)[0][0]
                flow_paths.append(paths[idx])
        
        ret += [flow_paths]
        
    if links:
        link_paths = []
        imfilters = ['',img_filter] # this allows both flow name conventions to exist in one folder 
        
        for p,b in zip(parent,label_base):            
            paths = [os.path.join(p,subfolder,b+'_links.txt') for imf in imfilters]
            found = [os.path.exists(path) for path in paths]
            nfound = np.sum(found)

            if nfound == 0:
                link_paths.append(None)
            else:
                idx = np.nonzero(found)[0][0]
                link_paths.append(paths[idx])
            
        ret += [link_paths]
    return (*ret,) if len(ret)>1 else ret[0]

# edited to allow omni to not read in training flows if any exist; flows computed on-the-fly and code expects this 
# futher edited to automatically find link files for boundary or timelapse flow generation 
def load_train_test_data(train_dir, test_dir=None, image_filter='', mask_filter='_masks', 
                         unet=False, look_one_level_down=True, omni=False, do_links=True):
    """
    Loads the training and optional test data for training runs.
    """
    
    image_names = get_image_files(train_dir, mask_filter, image_filter, look_one_level_down)
    nimg_train = len(image_names)
    images = [imread(image_names[n]) for n in range(nimg_train)]

    label_names, flow_names, link_names = get_label_files(image_names, 
                                                          label_filter=mask_filter, 
                                                          img_filter=image_filter, 
                                                          flows=True, links=True)
    labels = [imread(l) for l in label_names]
    links = [load_links(l) for l in link_names]
    
    if flow_names is not None and not unet and not omni:
        for n in range(nimg_train):
            flows = imread(flow_names[n])
            if flows.shape[0]<4:
                labels[n] = np.concatenate((labels[n][np.newaxis,:,:], flows), axis=0) 
            else:
                labels[n] = flows
            
    # testing data
    nimg_test = 0
    test_images, test_labels, test_links, image_names_test = None,None,[None],None 
    if test_dir is not None:
        image_names_test = get_image_files(test_dir, mask_filter, image_filter, look_one_level_down)
        label_names_test, flow_names_test, link_names_test = get_label_files(image_names_test, 
                                                                            label_filter=mask_filter, 
                                                                            img_filter=image_filter, 
                                                                            flows=True, links=True)
        
        nimg_test = len(image_names_test)
        test_images = [imread(image_names_test[n]) for n in range(nimg_test)]
        test_labels = [imread(label_names_test[n]) for n in range(nimg_test)]
        test_links = [load_links(link_names_test[n]) for n in range(nimg_test)]
        if flow_names_test is not None and not unet:
            for n in range(nimg_test):
                flows = imread(flow_names_test[n])
                if flows.shape[0]<4:
                    test_labels[n] = np.concatenate((test_labels[n][np.newaxis,:,:], flows), axis=0) 
                else:
                    test_labels[n] = flows
    
    # Allow disabling the links even if link files were found 
    if not do_links:
        links = [None]*nimg_train
        test_links = [None]*nimg_test
    
    return images, labels, links, image_names, test_images, test_labels, test_links, image_names_test



def masks_flows_to_seg(images, masks, flows, diams, file_names, channels=None):
    """ save output of model eval to be loaded in GUI 

    can be list output (run on multiple images) or single output (run on single image)

    saved to file_names[k]+'_seg.npy'
    
    Parameters
    -------------

    images: (list of) 2D or 3D arrays
        images input into cellpose

    masks: (list of) 2D arrays, int
        masks output from cellpose_omni.eval, where 0=NO masks; 1,2,...=mask labels

    flows: (list of) list of ND arrays 
        flows output from cellpose_omni.eval

    diams: float array
        diameters used to run Cellpose

    file_names: (list of) str
        names of files of images

    channels: list of int (optional, default None)
        channels used to run Cellpose    
    
    """
    
    if channels is None:
        channels = [0,0]
    
    if isinstance(masks, list):
        if not isinstance(diams, (list, np.ndarray)):
            diams = diams * np.ones(len(masks), np.float32)
        for k, [image, mask, flow, diam, file_name] in enumerate(zip(images, masks, flows, diams, file_names)):
            channels_img = channels
            if channels_img is not None and len(channels) > 2:
                channels_img = channels[k]
            masks_flows_to_seg(image, mask, flow, diam, file_name, channels_img)
        return

    if len(channels)==1:
        channels = channels[0]
    
    flowi = []
    if flows[0].ndim==3:
        Ly, Lx = masks.shape[-2:]
        flowi.append(cv2.resize(flows[0], (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,...])
    else:
        flowi.append(flows[0])
    
    if flows[0].ndim==3:
        cellprob = (np.clip(transforms.normalize99(flows[2]),0,1) * 255).astype(np.uint8)
        cellprob = cv2.resize(cellprob, (Lx, Ly), interpolation=cv2.INTER_NEAREST)
        flowi.append(cellprob[np.newaxis,...])
        flowi.append(np.zeros(flows[0].shape, dtype=np.uint8))
        flowi[-1] = flowi[-1][np.newaxis,...]
    else:
        flowi.append((np.clip(transforms.normalize99(flows[2]),0,1) * 255).astype(np.uint8))
        flowi.append((flows[1][0]/10 * 127 + 127).astype(np.uint8))
    if len(flows)>2:
        flowi.append(flows[3])
        flowi.append(np.concatenate((flows[1], flows[2][np.newaxis,...]), axis=0))
    outlines = masks * utils.masks_to_outlines(masks)
    base = os.path.splitext(file_names)[0]
    if masks.ndim==3:
        np.save(base+ '_seg.npy',
                    {'outlines': outlines.astype(np.uint16) if outlines.max()<2**16-1 else outlines.astype(np.uint32),
                        'masks': masks.astype(np.uint16) if outlines.max()<2**16-1 else masks.astype(np.uint32),
                        'chan_choose': channels,
                        'img': images,
                        'ismanual': np.zeros(masks.max(), bool),
                        'filename': file_names,
                        'flows': flowi,
                        'est_diam': diams})
    else:
        if images.shape[0]<8:
            np.transpose(images, (1,2,0))
        np.save(base+ '_seg.npy',
                    {'img': images,
                        'outlines': outlines.astype(np.uint16) if outlines.max()<2**16-1 else outlines.astype(np.uint32),
                     'masks': masks.astype(np.uint16) if masks.max()<2**16-1 else masks.astype(np.uint32),
                     'chan_choose': channels,
                     'ismanual': np.zeros(masks.max(), bool),
                     'filename': file_names,
                     'flows': flowi,
                     'est_diam': diams})    

def save_to_png(images, masks, flows, file_names):
    """ deprecated (runs io.save_masks with png=True) 
    
        does not work for 3D images
    
    """
    save_masks(images, masks, flows, file_names, png=True)

# Now saves flows, masks, etc. to separate folders.
def save_masks(images, masks, flows, file_names, png=True, tif=False,
               suffix='',save_flows=False, save_outlines=False, outline_col=[1,0,0],
               save_ncolor=False, dir_above=False, in_folders=False, savedir=None, 
               save_txt=True, save_plot=True, omni=True, channel_axis=None):
    """ save masks + nicely plotted segmentation image to png and/or tiff

    if png, masks[k] for images[k] are saved to file_names[k]+'_cp_masks.png'

    if tif, masks[k] for images[k] are saved to file_names[k]+'_cp_masks.tif'

    if png and matplotlib installed, full segmentation figure is saved to file_names[k]+'_cp.png'

    only tif option works for 3D data
    
    Parameters
    -------------

    images: (list of) 2D, 3D or 4D arrays
        images input into cellpose

    masks: (list of) 2D arrays, int
        masks output from cellpose_omni.eval, where 0=NO masks; 1,2,...=mask labels

    flows: (list of) list of ND arrays 
        flows output from cellpose_omni.eval

    file_names: (list of) str
        names of files of images
        
    savedir: str
        absolute path where images will be saved. Default is none (saves to image directory)
    
    save_flows, save_outlines, save_ncolor, save_txt: bool
        Can choose which outputs/views to save.
        ncolor is a 4 (or 5, if 4 takes too long) index version of the labels that
        is way easier to visualize than having hundreds of unique colors that may
        be similar and touch. Any color map can be applied to it (0,1,2,3,4,...).
    
    """
    if isinstance(masks, list):
        for image, mask, flow, file_name in zip(images, masks, flows, file_names):
            save_masks(image, mask, flow, file_name, png=png, tif=tif, suffix=suffix, dir_above=dir_above,
                       save_flows=save_flows,save_outlines=save_outlines, outline_col=outline_col,
                       save_ncolor=save_ncolor, savedir=savedir, save_txt=save_txt, save_plot=save_plot,
                       in_folders=in_folders, omni=omni, channel_axis=channel_axis)
        return

    # make sure there is a leading underscore if any suffix was supplied
    if len(suffix):
        if suffix[0]!='_':
            suffix = '_'+suffix
        
    if masks.ndim > 2 and not tif:
        raise ValueError('cannot save 3D outputs as PNG, use tif option instead')
#     base = os.path.splitext(file_names)[0]
    
    if savedir is None: 
        if dir_above:
            savedir = Path(file_names).parent.parent.absolute() #go up a level to save in its own folder
        else:
            savedir = Path(file_names).parent.absolute()

    check_dir(savedir) 
            
    basename = os.path.splitext(os.path.basename(file_names))[0]
    if in_folders:
        maskdir = os.path.join(savedir,'masks')
        outlinedir = os.path.join(savedir,'outlines')
        txtdir = os.path.join(savedir,'txt_outlines')
        ncolordir = os.path.join(savedir,'ncolor_masks')
        flowdir = os.path.join(savedir,'flows')
        cpdir = os.path.join(savedir,'cp_output')
    else:
        maskdir = savedir
        outlinedir = savedir
        txtdir = savedir
        ncolordir = savedir
        flowdir = savedir
        cpdir = savedir
    check_dir(maskdir) 

    exts = []
    if masks.ndim > 2:
        png = False
        tif = True
    if png:    
        if masks.max() < 2**16:
            masks = masks.astype(np.uint16) 
            exts.append('.png')
        else:
            png = False 
            tif = True
            io_logger.warning('found more than 65535 masks in each image, cannot save PNG, saving as TIF')
    
    if tif:
        exts.append('.tif')

    # format_labels will also automatically use lowest bit depth possible
    if OMNI_INSTALLED:
        masks = format_labels(masks) 

    # save masks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ext in exts:
            
            imsave(os.path.join(maskdir,basename + '_cp_masks' + suffix + ext), masks)
    
    criterion3 = not (min(images.shape) > 3 and images.ndim >=3)
    
    if png and MATPLOTLIB and criterion3 and save_plot:
        img = images.copy()
        # if img.ndim<3:
        #     img = img[:,:,np.newaxis]
        # elif img.shape[0]<8:
        #     np.transpose(img, (1,2,0))
        
        fig = plt.figure(figsize=(12,3))
        plot.show_segmentation(fig, img, masks, flows[0], omni=omni, channel_axis=channel_axis)
        check_dir(cpdir) 
        fig.savefig(os.path.join(cpdir,basename + '_cp_output' + suffix + '.png'), dpi=300)
        plt.close(fig)

    # ImageJ txt outline files 
    if masks.ndim < 3 and save_txt:
        check_dir(txtdir)
        outlines = utils.outlines_list(masks)
        outlines_to_text(os.path.join(txtdir,basename), outlines)
    
    # RGB outline images
    if masks.ndim < 3 and save_outlines: 
        check_dir(outlinedir) 
        # outlines = utils.masks_to_outlines(masks)
        # outX, outY = np.nonzero(outlines)
        # img0 = transforms.normalize99(images,omni=omni)
        img0 = images.copy()        

        # if img0.shape[0] < 4:
        #     img0 = np.transpose(img0, (1,2,0))
        # if img0.shape[-1] < 3 or img0.ndim < 3:
        #     print(img0.shape,'sdfsfdssf')
        #     img0 = plot.image_to_rgb(img0, channels=channels, omni=omni) #channels=channels, 
        
        # img0 = (transforms.normalize99(img0,omni=omni)*(2**8-1)).astype(np.uint8)
        # imgout= img0.copy()
        # imgout[outX, outY] = np.array([255,0,0]) #pure red 
        imgout = plot.outline_view(img0,masks,color=outline_col)
        imsave(os.path.join(outlinedir, basename + '_outlines' + suffix + '.png'),  imgout)
    
    # ncolor labels (ready for color map application)
    if masks.ndim < 3 and OMNI_INSTALLED and save_ncolor:
        check_dir(ncolordir)
        #convert masks to minimal n-color reresentation 
        imsave(os.path.join(ncolordir, basename + '_cp_ncolor_masks' + suffix + '.png'),
               ncolor.label(masks))
    
    # save RGB flow picture
    if masks.ndim < 3 and save_flows:
        check_dir(flowdir)
        imsave(os.path.join(flowdir, basename + '_flows' + suffix + '.tif'), flows[0].astype(np.uint8))
        #save full flow data
        imsave(os.path.join(flowdir, basename + '_dP' + suffix + '.tif'), flows[1]) 
    
def save_server(parent=None, filename=None):
    """ Uploads a *_seg.npy file to the bucket.
    
    Parameters
    ----------------

    parent: PyQt.MainWindow (optional, default None)
        GUI window to grab file info from

    filename: str (optional, default None)
        if no GUI, send this file to server

    """
    if parent is not None:
        q = QtGui.QMessageBox.question(
                                    parent,
                                    "Send to server",
                                    "Are you sure? Only send complete and fully manually segmented data.\n (do not send partially automated segmentations)",
                                    QtGui.QMessageBox.Yes | QtGui.QMessageBox.No
                                  )
        if q != QtGui.QMessageBox.Yes:
            return
        else:
            filename = parent.filename

    if filename is not None:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        'key/cellpose-data-writer.json')
        bucket_name = 'cellpose_data'
        base = os.path.splitext(filename)[0]
        source_file_name = base + '_seg.npy'
        io_logger.info(f'sending {source_file_name} to server')
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f")
        filestring = time + '.npy'
        io_logger.info(f'name on server: {filestring}')
        destination_blob_name = filestring
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        io_logger.info(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )

