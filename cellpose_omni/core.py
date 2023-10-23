import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import logging
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import tempfile
import cv2
from scipy.stats import mode
from . import transforms, dynamics, utils, metrics, io

import omnipose

# from multiprocessing import Pool, cpu_count
# from functools import partial
# from focal_loss.focal_loss import FocalLoss

try:
    from mxnet import gluon, nd
    import mxnet as mx
    from . import resnet_style
    MXNET_ENABLED = True 
    mx_GPU = mx.gpu()
    mx_CPU = mx.cpu()
except:
    MXNET_ENABLED = False

try:
    import torch
    from torch.cuda.amp import autocast, GradScaler
    from torch import nn
    from torch.utils import mkldnn as mkldnn_utils
    TORCH_ENABLED = True
    from .resnet_torch import torch_GPU, torch_CPU, CPnet, ARM, empty_cache
except Exception as e:
    TORCH_ENABLED = False
    print('core.py torch import error',e)

core_logger = logging.getLogger(__name__)
tqdm_out = utils.TqdmToLogger(core_logger, level=logging.INFO)

# nclasses now specified by user or by model type in models.py
def parse_model_string(pretrained_model):
    if isinstance(pretrained_model, list):
        model_str = os.path.split(pretrained_model[0])[-1]
    else:
        model_str = os.path.split(pretrained_model)[-1]
    if len(model_str)>3 and model_str[:4]=='unet':
        nclasses = max(2, int(model_str[4]))
    elif len(model_str)>7 and model_str[:8]=='cellpose':
        nclasses = 3
    else:
        return True, True, False
    ostrs = model_str.split('_')[2::2]
    residual_on = ostrs[0]=='on'
    style_on = ostrs[1]=='on'
    concatenation = ostrs[2]=='on'
    return residual_on, style_on, concatenation

def use_gpu(gpu_number=0, use_torch=True):
    """ check if gpu works """
    if use_torch:
        return _use_gpu_torch(gpu_number)
    else:
        raise ValueError('cellpose only runs with pytorch now')

def _use_gpu_torch(gpu_number=0):
    try:
        # device = torch.device('cuda:' + str(gpu_number))
        device = torch.device(f'mps:{gpu_number}') if ARM else torch.device(f'cuda:{gpu_number}')
        _ = torch.zeros([1, 2, 3]).to(device)
        core_logger.info('** TORCH GPU version installed and working. **')
        return True
    except:# Exception as e:
        core_logger.info('TORCH GPU version not installed/working.')#, e)
        return False

def assign_device(use_torch=True, gpu=False, device=0):
    if gpu and use_gpu(use_torch=True):
        # device = torch.device(f'cuda:{device}')
        device = torch_GPU
        gpu = True
        core_logger.info('>>>> using GPU')
    else:
        device = torch_CPU
        core_logger.info('>>>> using CPU')
        gpu=False
    return device, gpu

def check_mkl(use_torch=True):
    #core_logger.info('Running test snippet to check if MKL-DNN working')
    if use_torch:
        mkl_enabled = torch.backends.mkldnn.is_available()
    else:
        process = subprocess.Popen(['python', 'test_mkl.py'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                    cwd=os.path.dirname(os.path.abspath(__file__)))
        stdout, stderr = process.communicate()
        if len(stdout)>0:
            mkl_enabled = True
        else:
            mkl_enabled = False
    if mkl_enabled:
        mkl_enabled = True
        #core_logger.info('MKL version working - CPU version is sped up.')
    elif not use_torch:
        core_logger.info('WARNING: MKL version on mxnet not working/installed - CPU version will be SLOW.')
        core_logger.info('see https://mxnet.apache.org/versions/1.6/api/python/docs/tutorials/performance/backend/mkldnn/mkldnn_readme.html#4)')
    else:
        core_logger.info('WARNING: MKL version on torch not working/installed - CPU version will be slightly slower.')
        core_logger.info('see https://pytorch.org/docs/stable/backends.html?highlight=mkl')
    return mkl_enabled

class UnetModel():
    def __init__(self, gpu=False, pretrained_model=False,
                 diam_mean=30., net_avg=True, device=None,
                 residual_on=False, style_on=False, concatenation=True,
                 nclasses=3, use_torch=True, nchan=2, dim=2, 
                 checkpoint=False, dropout=False, kernel_size=2):
        self.unet = True
        if use_torch:
            if not TORCH_ENABLED:
                use_torch = False
        self.torch = use_torch
        self.mkldnn = None
        if device is None:
            sdevice, gpu = assign_device(torch, gpu)
        self.device = device if device is not None else sdevice
        if device is not None:
            if use_torch:
                # device_gpu = self.device.type=='cuda'
                device_gpu = self.device.type=='mps' if ARM else self.device.type=='cuda'
                
            else:
                device_gpu = self.device.device_type=='gpu'
        self.gpu = gpu if device is None else device_gpu
        if use_torch and not self.gpu:
            self.mkldnn = check_mkl(self.torch)
        self.pretrained_model = pretrained_model
        self.diam_mean = diam_mean

        if pretrained_model:
            params = parse_model_string(pretrained_model)
            if params is not None:
                nclasses, residual_on, style_on, concatenation = params
        
        ostr = ['off', 'on']
        self.net_type = 'unet{}_residual_{}_style_{}_concatenation_{}'.format(nclasses,
                                                                                ostr[residual_on],
                                                                                ostr[style_on],
                                                                                ostr[concatenation])                                             
        if pretrained_model:
            core_logger.info(f'u-net net type: {self.net_type}')
        # create network
        self.nclasses = nclasses
        self.nbase = [32,64,128,256]
        self.nchan = nchan
        self.dim = dim
        self.checkpoint = checkpoint
        self.dropout = dropout
        self.kernel_size = kernel_size
        # print('torch is ffff', torch) # duplicated in unetmodel class
        
        if self.torch:
            self.nbase = [nchan, 32, 64, 128, 256]
            self.net = CPnet(self.nbase, 
                              self.nclasses, 
                              sz=3,
                              residual_on=residual_on, 
                              style_on=style_on,
                              concatenation=concatenation,
                              mkldnn=self.mkldnn, 
                              dim=self.dim, 
                              checkpoint=self.checkpoint,
                              dropout=self.dropout,
                              kernel_size=self.kernel_size).to(self.device)
        else:
            self.net = resnet_style.CPnet(self.nbase, nout=self.nclasses,
                                        residual_on=residual_on, 
                                        style_on=style_on,
                                        concatenation=concatenation)
            self.net.hybridize(static_alloc=True, static_shape=True)
            self.net.initialize(ctx = self.device)

        if pretrained_model is not None and isinstance(pretrained_model, str):
            self.net.load_model(pretrained_model, cpu=(not self.gpu))

    def eval(self, x, batch_size=8, channels=None, channels_last=False, invert=False, normalize=True,
             rescale=None, do_3D=False, anisotropy=None, net_avg=True, augment=False,
             tile=True, cell_threshold=None, boundary_threshold=None, min_size=15):
        """ segment list of images x

            Parameters
            ----------
            x: list or array of images
                can be list of 2D/3D images, or array of 2D/3D images, or 4D image array

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            channel_axis: int (optional, default None)
                if None, channels dimension is attempted to be automatically determined

            z_axis: int (optional, default None)
                if None, z dimension is attempted to be automatically determined

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            normalize: bool (optional, default True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0

            do_3D: bool (optional, default False)
                set to True to run 3D segmentation on 4D image input

            anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

            net_avg: bool (optional, default True)
                runs the 4 built-in networks and averages them if True, runs one network if False

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            cell_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            boundary_threshold: float (optional, default 0.0)
                cell probability threshold (all pixels with prob above threshold kept for masks)

            min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

            Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = flows at each pixel
                flows[k][2] = the cell distance field
                flows[k][3] = the cell boundary

            styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image

        """        
        x = [transforms.convert_image(xi, channels, channel_axis, z_axis, do_3D, 
                                    normalize, invert, nchan=self.nchan) for xi in x]
        nimg = len(x)
        self.batch_size = batch_size

        styles = []
        flows = []
        masks = []
        if rescale is None:
            rescale = np.ones(nimg)
        elif isinstance(rescale, float):
            rescale = rescale * np.ones(nimg)
        if nimg > 1:
            tqdm.tqdm.unit_divisor = nimg
            iterator = trange(nimg, file=tqdm_out)
        else:
            iterator = range(nimg)

        if isinstance(self.pretrained_model, list):
            model_path = self.pretrained_model[0]
            if not net_avg:
                self.net.load_model(self.pretrained_model[0])
                if not self.torch:
                    self.net.collect_params().grad_req = 'null'
        else:
            model_path = self.pretrained_model

        if cell_threshold is None or boundary_threshold is None:
            try:
                thresholds = np.load(model_path+'_cell_boundary_threshold.npy')
                cell_threshold, boundary_threshold = thresholds
                core_logger.info('>>>> found saved thresholds from validation set')
            except:
                core_logger.warning('WARNING: no thresholds found, using default / user input')

        cell_threshold = 2.0 if cell_threshold is None else cell_threshold
        boundary_threshold = 0.5 if boundary_threshold is None else boundary_threshold

        if not do_3D:
            for i in iterator:
                img = x[i].copy()
                shape = img.shape
                # rescale image for flow computation
                imgs = transforms.resize_image(img, rsz=rescale[i])
                print('This branch seems to have a bug',img.shape, imgs.shape)
                y, style = self._run_nets(img, net_avg=net_avg, augment=augment, 
                                          tile=tile)
                
                maski = utils.get_masks_unet(y, cell_threshold, boundary_threshold)
                maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                maski = transforms.resize_image(maski, shape[-3], shape[-2], 
                                                    interpolation=cv2.INTER_NEAREST)
                masks.append(maski)
                styles.append(style)
        else:
            for i in iterator:
                tic=time.time()
                yf, style = self._run_3D(x[i], rsz=rescale[i], anisotropy=anisotropy, 
                                         net_avg=net_avg, augment=augment, tile=tile)
                yf = yf.mean(axis=0)
                core_logger.info('probabilities computed %2.2fs'%(time.time()-tic))
                maski = utils.get_masks_unet(yf.transpose((1,2,3,0)), cell_threshold, boundary_threshold)
                maski = utils.fill_holes_and_remove_small_masks(maski, min_size=min_size)
                masks.append(maski)
                styles.append(style)
                core_logger.info('masks computed %2.2fs'%(time.time()-tic))
                flows.append(yf)

        if nolist:
            masks, flows, styles = masks[0], flows[0], styles[0]
        
        return masks, flows, styles

    def _to_device(self, x):
        if self.torch:
            X = torch.tensor(x,device=self.device).float()
        else:
            #if x.dtype != 'bool':
            X = nd.array(x.astype(np.float32), ctx=self.device)
        return X
    

    def _from_device(self, X):
        if self.torch:
            x = X.detach().cpu().numpy()
            # clear memory after evaluation (confirmed working)
            empty_cache() #could add overhead
        else:
            x = X.asnumpy()
        return x

    def network(self, x, return_conv=False):
        """ convert imgs to torch/mxnet and run network model and return numpy """
        X = self._to_device(x)
        if self.torch:
            self.net.eval()
            if self.mkldnn:
                self.net = mkldnn_utils.to_mkldnn(self.net)
            with torch.no_grad():
                y, style = self.net(X)
        else:
            y, style = self.net(X)
        del X 
        if self.mkldnn:
            self.net.to(torch_CPU)
        y = self._from_device(y)
        style = self._from_device(style)
        if return_conv: # conv is not even defined anywhere, why is this here?
            conv = self._from_device(conv)
            y = np.concatenate((y, conv), axis=1)
        
        return y, style
                
    def _run_nets(self, img, net_avg=True, augment=False, tile=True, tile_overlap=0.1, bsize=224, 
                  return_conv=False, progress=None):
        """ run network (if more than one, loop over networks and average results

        Parameters
        --------------

        img: float, [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

        net_avg: bool (optional, default True)
            runs the 4 built-in networks and averages them if True, runs one network if False

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU memory usage limited (recommended)

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI

        Returns
        ------------------

        y: array [3 x Ly x Lx] or [3 x Lz x Ly x Lx]
            y is output (averaged over networks);
            y[0] is Y flow; y[1] is X flow; y[2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles,
            but not averaged over networks.

        """
        if isinstance(self.pretrained_model, str) or not net_avg:  
            y, style = self._run_net(img, augment=augment, tile=tile, tile_overlap=tile_overlap,
                                     bsize=bsize, return_conv=return_conv)
        else:  
            for j in range(len(self.pretrained_model)):
                
                if self.torch and self.gpu:
                    net = self.net.module
                else:
                    net = self.net
                    
                net.load_model(self.pretrained_model[0], cpu=(not self.gpu))
                if not self.torch:
                    net.collect_params().grad_req = 'null'
                y0, style = self._run_net(img, augment=augment, tile=tile, 
                                          tile_overlap=tile_overlap, bsize=bsize,
                                          return_conv=return_conv)

                if j==0:
                    y = y0
                else:
                    y += y0
                if progress is not None:
                    progress.setValue(10 + 10*j)
            y = y / len(self.pretrained_model)
            
        return y, style

    def _run_net(self, imgs, augment=False, tile=True, tile_overlap=0.1, bsize=224,
                 return_conv=False):
        """ run network on image or stack of images

        (faster if augment is False)

        Parameters
        --------------

        imgs: array [Ly x Lx x nchan] or [Lz x Ly x Lx x nchan]

        rsz: float (optional, default 1.0)
            resize coefficient(s) for image

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended);
            cannot be turned off for 3D segmentation

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]

        Returns
        ------------------

        y: array [Ly x Lx x 3] or [Lz x Ly x Lx x 3]
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles

        """
        transpose = False
        if imgs.ndim==4 and self.dim==2: #doing cellpose 3D, model does 2D slices but image is 3D+chans  
            # make image Lz x nchan x Ly x Lx for net
            imgs = np.transpose(imgs, (0,3,1,2)) 
            detranspose = (0,2,3,1)
            return_conv = False
            transpose = True
        elif imgs.ndim>self.dim:
            # make image (nchan, *dims) for net
            order = (self.dim,)+tuple([k for k in range(self.dim)]) #(2,0,1) in 2D etc.
            imgs = np.transpose(imgs, order)
            transpose = True
            detranspose = tuple([k for k in range(1,self.dim+1)])+(0,)#(1,2,0)
        ## The do_3D option makes sense because that's the Cellpose3D slicing. For true 3D (set with dim=3),
        ## we assume nchan.Lz/t.Ly.Lx 

        # imgs is a misnomer since it should be just one image at this point 

        # pad image for net so volume dimensions are divisible by 4 
        imgs, subs = transforms.pad_image_ND(imgs,dim=self.dim)
        # slices from padding
        # slc = [slice(0, self.nclasses) for n in range(imgs.ndim)] # changed from imgs.shape[n]+1 for first slice size 
        slc = [slice(0, imgs.shape[n]+1) for n in range(imgs.ndim)]
        slc[-(self.dim+1)] = slice(0, self.nclasses + 32*return_conv + 1)
        for k in range(1,self.dim+1):
            slc[-k] = slice(subs[-k][0], subs[-k][-1]+1)
        slc = tuple(slc)

        # run network
        if tile or augment or (imgs.ndim==4 and self.dim==2): ## need to work out the tiling in ND... <<<<<
            y, style = self._run_tiled(imgs, augment=augment, bsize=bsize, 
                                      tile_overlap=tile_overlap, 
                                      return_conv=return_conv)
        else:

            imgs = np.expand_dims(imgs, axis=0)
            # at this point imgs is (B,C,*dims)
            y, style = self.network(imgs, return_conv=return_conv)
            y, style = y[0], style[0]
        style /= (style**2).sum()**0.5


        # slice out padding
        y = y[slc]

        # transpose so channels axis is last again
        # Seems really silly to keep doing this, maybe I should just heep channels first everywhere 
        if transpose:
            y = np.transpose(y, detranspose)

        return y, style
    
    def _run_tiled(self, imgi, augment=False, bsize=224, tile_overlap=0.1, return_conv=False):
        """ run network in tiles of size [bsize x bsize]

        Image is split into overlapping tiles of size [bsize x bsize].
        If augment, tiles have 50% overlap and are flipped at overlaps.
        The average of the network output over tiles is returned.

        Parameters
        --------------

        imgi: array [nchan x Ly x Lx] or [Lz x nchan x Ly x Lx]

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]
         
        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        Returns
        ------------------

        yf: array [3 x Ly x Lx] or [Lz x 3 x Ly x Lx]
            yf is averaged over tiles
            yf[0] is Y flow; yf[1] is X flow; yf[2] is cell probability

        styles: array [64]
            1D array summarizing the style of the image, averaged over tiles

        """

        if imgi.ndim==4 and self.dim==2: # in this case, must have a 3D image but using 2D models
            batch_size = self.batch_size 
            Lz, nchan = imgi.shape[:2]
            IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[0], bsize=bsize, 
                                                            augment=augment, tile_overlap=tile_overlap)
            ny, nx, nchan, ly, lx = IMG.shape
            batch_size *= max(4, (bsize**2 // (ly*lx))**0.5)
            yf = np.zeros((Lz, self.nclasses, imgi.shape[-2], imgi.shape[-1]), np.float32)
            styles = []
            if ny*nx > batch_size:
                ziterator = trange(Lz, file=tqdm_out)
                for i in ziterator:
                    yfi, stylei = self._run_tiled(imgi[i], augment=augment, 
                                                  bsize=bsize, tile_overlap=tile_overlap)
                    yf[i] = yfi
                    styles.append(stylei)
            else:
                # run multiple slices at the same time
                ntiles = ny*nx
                nimgs = max(2, int(np.round(batch_size / ntiles)))
                niter = int(np.ceil(Lz/nimgs))
                ziterator = trange(niter, file=tqdm_out)
                for k in ziterator:
                    IMGa = np.zeros((ntiles*nimgs, nchan, ly, lx), np.float32)
                    for i in range(min(Lz-k*nimgs, nimgs)):
                        IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi[k*nimgs+i], bsize=bsize, 
                                                                        augment=augment,
                                                                        tile_overlap=tile_overlap)
                        IMGa[i*ntiles:(i+1)*ntiles] = np.reshape(IMG, (ny*nx, nchan, ly, lx))
                    ya, stylea = self.network(IMGa)
                    for i in range(min(Lz-k*nimgs, nimgs)):
                        y = ya[i*ntiles:(i+1)*ntiles]
                        if augment:
                            y = np.reshape(y, (ny, nx, 3, ly, lx))
                            y = transforms.unaugment_tiles(y, self.unet)
                            y = np.reshape(y, (-1, 3, ly, lx))
                        yfi = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
                        yfi = yfi[:,:imgi.shape[2],:imgi.shape[3]]
                        yf[k*nimgs+i] = yfi
                        stylei = stylea[i*ntiles:(i+1)*ntiles].sum(axis=0)
                        stylei /= (stylei**2).sum()**0.5
                        styles.append(stylei)
            return yf, np.array(styles)
        else:
            IMG, subs, shape, inds = transforms.make_tiles_ND(imgi, bsize=bsize,
                                                              augment=augment,
                                                              tile_overlap=tile_overlap) #<<<
            # IMG now always returned in the form (ny*nx, nchan, ly, lx) 
            # for either tiling or augmenting
            
            batch_size = self.batch_size
            niter = int(np.ceil(IMG.shape[0] / batch_size))
            nout = self.nclasses + 32*return_conv
            y = np.zeros((IMG.shape[0], nout)+tuple(IMG.shape[-self.dim:]))
            for k in range(niter):
                irange = np.arange(batch_size*k, min(IMG.shape[0], batch_size*k+batch_size))
                y0, style = self.network(IMG[irange], return_conv=return_conv)
                arg = (len(irange),)+y0.shape[-(self.dim+1):]
                y[irange] = y0.reshape(arg)
                if k==0:
                    styles = style[0]
                styles += style.sum(axis=0)
            styles /= IMG.shape[0]
            if augment: 
                # now updated for ND 
                y = transforms.unaugment_tiles_ND(y, inds, self.unet)
            
            yf = transforms.average_tiles_ND(y, subs, shape) #<<<
            slc = tuple([slice(s) for s in shape])
            yf = yf[(Ellipsis,)+slc]
            styles /= (styles**2).sum()**0.5
            return yf, styles
            

    def _run_3D(self, imgs, rsz=1.0, anisotropy=None, net_avg=True, 
                augment=False, tile=True, tile_overlap=0.1, 
                bsize=224, progress=None):
        """ run network on stack of images

        (faster if augment is False)

        Parameters
        --------------

        imgs: array [Lz x Ly x Lx x nchan]

        rsz: float (optional, default 1.0)
            resize coefficient(s) for image

        anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

        net_avg: bool (optional, default True)
            runs the 4 built-in networks and averages them if True, runs one network if False

        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment

        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended);
            cannot be turned off for 3D segmentation

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        bsize: int (optional, default 224)
            size of tiles to use in pixels [bsize x bsize]

        progress: pyqt progress bar (optional, default None)
            to return progress bar status to GUI


        Returns
        ------------------

        yf: array [Lz x Ly x Lx x 3]
            y[...,0] is Y flow; y[...,1] is X flow; y[...,2] is cell probability

        style: array [64]
            1D array summarizing the style of the image,
            if tiled it is averaged over tiles

        """ 
        sstr = ['YX', 'ZY', 'ZX']
        if anisotropy is not None:
            rescaling = [[rsz, rsz],
                         [rsz*anisotropy, rsz],
                         [rsz*anisotropy, rsz]]
        else:
            rescaling = [rsz] * 3
        pm = [(0,1,2,3), (1,0,2,3), (2,0,1,3)]
        ipm = [(3,0,1,2), (3,1,0,2), (3,1,2,0)]
        yf = np.zeros((3, self.nclasses, imgs.shape[0], imgs.shape[1], imgs.shape[2]), np.float32)
        for p in range(3 - 2*self.unet):
            xsl = imgs.copy().transpose(pm[p])
            # rescale image for flow computation
            shape = xsl.shape
            xsl = transforms.resize_image(xsl, rsz=rescaling[p])
            # per image
            core_logger.info('running %s: %d planes of size (%d, %d)'%(sstr[p], shape[0], shape[1], shape[2]))
            y, style = self._run_nets(xsl, net_avg=net_avg, augment=augment, tile=tile, 
                                      bsize=bsize, tile_overlap=tile_overlap)
            y = transforms.resize_image(y, shape[1], shape[2])    
            yf[p] = y.transpose(ipm[p])
            if progress is not None:
                progress.setValue(25+15*p)
        return yf, style

    def loss_fn(self, lbl, y):
        """ loss function between true labels lbl and prediction y """
        # if available set boundary pixels to 2
        if lbl.shape[1]>1 and self.nclasses>2:
            boundary = lbl[:,1]<=4
            lbl = lbl[:,0]
            lbl[boundary] *= 2
        else:
            lbl = lbl[:,0]
        lbl = self._to_device(lbl)
        loss = 8 * 1./self.nclasses * self.criterion(y, lbl)
        del lbl
        return loss

    def train(self, train_data, train_labels, train_files=None, 
              test_data=None, test_labels=None, test_files=None,
              channels=None, normalize=True, save_path=None, save_every=50, save_each=False,
              learning_rate=0.2, n_epochs=500, momentum=0.9, weight_decay=0.00001, 
              batch_size=8, rescale=False):
        """ train function uses 0-1 mask label and boundary pixels for training """
        
        print('Probably not supposed to be in this branch.')

        nimg = len(train_data)

        train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data, train_labels,
                                                                                                   test_data, test_labels,
                                                                                                   channels, self.channel_axis, 
                                                                                                   normalize)
        # add dist_to_bound to labels
        if self.nclasses==3:
            core_logger.info('computing boundary pixels')
            train_classes = [np.stack((label, label>0, utils.distance_to_boundary(label)), axis=0).astype(np.float32)
                                for label in tqdm(train_labels, file=tqdm_out)]
        else:
            train_classes = [np.stack((label, label>0), axis=0).astype(np.float32)
                                for label in tqdm(train_labels, file=tqdm_out)]
        if run_test:
            if self.nclasses==3:
                test_classes = [np.stack((label, label>0, utils.distance_to_boundary(label)), axis=0).astype(np.float32)
                                    for label in tqdm(test_labels, file=tqdm_out)]
            else:
                test_classes = [np.stack((label, label>0), axis=0).astype(np.float32)
                                    for label in tqdm(test_labels, file=tqdm_out)]
        
        # split train data into train and val
        val_data = train_data[::8]
        val_classes = train_classes[::8]
        val_labels = train_labels[::8]
        del train_data[::8], train_classes[::8], train_labels[::8]


        model_path = self._train_net(train_data, train_classes, 
                                     test_data, test_classes, save_path, save_every, save_each,
                                     learning_rate, n_epochs, momentum, weight_decay, 
                                     batch_size, rescale)


        # find threshold using validation set
        core_logger.info('>>>> finding best thresholds using validation set')
        cell_threshold, boundary_threshold = self.threshold_validation(val_data, val_labels)
        np.save(model_path+'_cell_boundary_threshold.npy', np.array([cell_threshold, boundary_threshold]))

    def threshold_validation(self, val_data, val_labels):
        cell_thresholds = np.arange(-4.0, 4.25, 0.5)
        if self.nclasses==3:
            boundary_thresholds = np.arange(-2, 2.25, 1.0)
        else:
            boundary_thresholds = np.zeros(1)
        aps = np.zeros((cell_thresholds.size, boundary_thresholds.size, 3))
        for j,cell_threshold in enumerate(cell_thresholds):
            for k,boundary_threshold in enumerate(boundary_thresholds):
                masks = []
                for i in range(len(val_data)):
                    output,style = self._run_net(val_data[i].transpose(1,2,0), augment=False)
                    masks.append(utils.get_masks_unet(output, cell_threshold, boundary_threshold))
                ap = metrics.average_precision(val_labels, masks)[0]
                ap0 = ap.mean(axis=0)
                aps[j,k] = ap0
            if self.nclasses==3:
                kbest = aps[j].mean(axis=-1).argmax()
            else:
                kbest = 0
            if j%4==0:
                core_logger.info('best threshold at cell_threshold = {} => boundary_threshold = {}, ap @ 0.5 = {}'.format(cell_threshold, 
                                                                                                                          boundary_thresholds[kbest], 
                                                                        aps[j,kbest,0]))   
        if self.nclasses==3: 
            jbest, kbest = np.unravel_index(aps.mean(axis=-1).argmax(), aps.shape[:2])
        else:
            jbest = aps.squeeze().mean(axis=-1).argmax()
            kbest = 0
        cell_threshold, boundary_threshold = cell_thresholds[jbest], boundary_thresholds[kbest]
        core_logger.info('>>>> best overall thresholds: (cell_threshold = {}, boundary_threshold = {}); ap @ 0.5 = {}'.format(cell_threshold, 
                                                                                                                              boundary_threshold, 
                                                          aps[jbest,kbest,0]))
        return cell_threshold, boundary_threshold

    # import gc 
    def _train_step(self, x, lbl):
        # X = self._to_device(x)
        X = x.clone()
        if self.torch:
            self.optimizer.zero_grad()
            # https://towardsdatascience.com/optimize-pytorch-performance-for-speed-and-memory-efficiency-2022-84f453916ea6
            # self.optimizer.zero_grad(set_to_none=True) does nothing 
            self.net.train()
            
            if self.autocast:
                with autocast(): 
                    y = self.net(X)[0]
                    del X
                    loss = self.loss_fn(lbl,y)
                    del lbl, y 
                self.scaler.scale(loss).backward()
                train_loss = loss.detach()
                self.scaler.step(self.optimizer) 
                train_loss *= len(x)
                self.scaler.update()
            else:
                y = self.net(X)[0]
                del X
                loss = self.loss_fn(lbl,y)
                del lbl, y
                loss.backward()
                train_loss = loss.detach() # added detach, probably redundant but trying to fix memory leak 
                self.optimizer.step()
                train_loss *= len(x)
        else:
            with mx.autograd.record():
                y = self.net(X)[0]
                del X
                loss = self.loss_fn(lbl, y)
            loss.backward()
            train_loss = nd.sum(loss).asscalar()
            self.optimizer.step(x.shape[0])
        # gc.collect()
        return train_loss

    def _test_eval(self, x, lbl):
        X = self._to_device(x)
        if self.torch:
            self.net.eval()
            with torch.no_grad():
                y, style = self.net(X)
                del X
                loss = self.loss_fn(lbl,y)
                test_loss = loss.detach()
                test_loss *= len(x)
        else:
            y, style = self.net(X)
            del X
            loss = self.loss_fn(lbl, y)
            test_loss = nd.sum(loss).asnumpy()
        return test_loss

    def _set_optimizer(self, learning_rate, momentum, weight_decay, SGD=False):
        if self.torch:
            if SGD:
                self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate,
                                            momentum=momentum, weight_decay=weight_decay)
            else:
                import torch_optimizer as optim # for RADAM optimizer
                self.optimizer = optim.RAdam(self.net.parameters(), lr=learning_rate, betas=(0.95, 0.999), #changed to .95
                                            eps=1e-08, weight_decay=weight_decay)
                core_logger.info('>>> Using RAdam optimizer')
                self.optimizer.current_lr = learning_rate
        else:
            self.optimizer = gluon.Trainer(self.net.collect_params(), 'sgd',{'learning_rate': learning_rate,
                                'momentum': momentum, 'wd': weight_decay})

    def _set_learning_rate(self, lr):
        if self.torch:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.optimizer.set_learning_rate(lr)

    def _set_criterion(self):
        if self.unet:
            if self.torch:
                criterion = nn.SoftmaxCrossEntropyLoss(axis=1) # not working anymore?
            else:
                criterion = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
        else:
            if self.torch:
                self.criterion  = nn.MSELoss(reduction='mean')
                self.criterion2 = nn.BCEWithLogitsLoss(reduction='mean')
                self.criterion1 = omnipose.loss.SSL_Norm_MSE()
                self.criterion3 = omnipose.loss.SSL_Norm()
                self.criterion12 = omnipose.loss.WeightedMSELoss()
                self.criterion15 = omnipose.loss.NormLoss()
                self.criterion17 = omnipose.loss.SineSquaredLoss()
                # self.criterion0 = ivp_loss.IVPLoss(dx=0.2, # consider increasing t0 np.sqrt(2)/5 for diagonals 
                self.criterion0 = omnipose.loss.EulerLoss(self.device)
                # self.ivp_loss =  ivp_loss.IVPLoss(dx=0.25,n_steps=8,device=self)
                # self.criterion17 = nn.SoftmaxCrossEntropyLoss(axis=1)
                # self.criterion17 = FocalLoss()
                
            else:
                self.criterion  = gluon.loss.L2Loss()
                self.criterion2 = gluon.loss.SigmoidBinaryCrossEntropyLoss()

    # Restored defaults. Need to make sure rescale is properly turned off and omni turned on when using CLI. 
    # maybe replace individual 
    def _train_net(self, train_data, train_labels, train_links, test_data=None, test_labels=None,
                   test_links=None, save_path=None, save_every=100, save_each=False,
                   learning_rate=0.2, n_epochs=500, momentum=0.9, weight_decay=0.00001, 
                   SGD=True, batch_size=8, dataloader=False, num_workers=8, nimg_per_epoch=None, 
                   rescale=True, affinity_field=False,
                   netstr=None, do_autocast=False, tyx=None, timing=False): 
        """ train function uses loss function self.loss_fn in models.py"""
        
        d = datetime.datetime.now()
        self.autocast = do_autocast
        self.n_epochs = n_epochs
        if isinstance(learning_rate, (list, np.ndarray)):
            if isinstance(learning_rate, np.ndarray) and learning_rate.ndim > 1:
                raise ValueError('learning_rate.ndim must equal 1')
            elif len(learning_rate) != n_epochs:
                raise ValueError('if learning_rate given as list or np.ndarray it must have length n_epochs')
            self.learning_rate = learning_rate
            self.learning_rate_const = mode(learning_rate)[0][0]
        else:
            self.learning_rate_const = learning_rate
            # set learning rate schedule    
            if SGD:
                LR = np.linspace(0, self.learning_rate_const, 10)
                if self.n_epochs > 250:
                    LR = np.append(LR, self.learning_rate_const*np.ones(self.n_epochs-100))
                    for i in range(10):
                        LR = np.append(LR, LR[-1]/2 * np.ones(10))
                else:
                    LR = np.append(LR, self.learning_rate_const*np.ones(max(0,self.n_epochs-10)))
            else:
                LR = self.learning_rate_const * np.ones(self.n_epochs)
            self.learning_rate = LR

        self.batch_size = batch_size
        self._set_optimizer(self.learning_rate[0], momentum, weight_decay, SGD)
        self._set_criterion()
        
        nimg = len(train_data)
        
        # Set crop size; should probably generalize to fix sizes that don't fit requirements  
        n = 16
        base = 2
        L = max(round(224/(base**4)),1)*(base**4) # rounds 224 up to the right multiple to work for base 
        # not sure if 4 downsampling or 3, but the "multiple of 16" elsewhere makes me think it must be 4, 
        # but it appears that multiple of 8 actually works? maybe the n=16 above conflates my experiments in 3D
        if tyx is None:
            tyx = (L,)*self.dim if self.dim==2 else (8*n,)+(8*n,)*(self.dim-1) #must be divisible by 2**3 = 8
        
        # compute average cell diameter
        if rescale:
            if train_links is not None:
                core_logger.warning("""WARNING: rescaling not updated for multi-label objects. 
                                    Check rescaling manually for the right diameter.""")
                
            diam_train = np.array([utils.diameters(train_labels[k],omni=self.omni)[0] 
                                   for k in range(len(train_labels))])
            diam_train[diam_train<5] = 5.
            if test_data is not None:
                diam_test = np.array([utils.diameters(test_labels[k],omni=self.omni)[0] 
                                      for k in range(len(test_labels))])
                diam_test[diam_test<5] = 5.
            scale_range = 0.5
            core_logger.info('>>>> median diameter set to = %d'%self.diam_mean)
        else:
            scale_range = 1.0

        nchan = train_data[0].shape[0]
        core_logger.info('>>>> training network with %d channel input <<<<'%nchan)
        core_logger.info('>>>> LR: %0.5f, batch_size: %d, weight_decay: %0.5f'%(self.learning_rate_const, 
                                                                                self.batch_size, weight_decay))
        
        if test_data is not None:
            core_logger.info(f'>>>> ntrain = {nimg}, ntest = {len(test_data)}')
        else:
            core_logger.info(f'>>>> ntrain = {nimg}')
        
        t0 = time.time()
        toc = t0
        lsum, nsum = 0, 0

        if save_path is not None:
            _, file_label = os.path.split(save_path)
            file_path = os.path.join(save_path, 'models/')
            io.check_dir(file_path)
        else:
            core_logger.warning('WARNING: no save_path given, model not saving')

        ksave = 0
        rsc = 1.0

        # cannot train with mkldnn
        self.net.mkldnn = False

        # get indices for each epoch for training
        np.random.seed(0)
        inds_all = np.zeros((0,), 'int32')
        if nimg_per_epoch is None or nimg > nimg_per_epoch:
            nimg_per_epoch = nimg 
        core_logger.info(f'>>>> nimg_per_epoch = {nimg_per_epoch}')
        while len(inds_all) < n_epochs * nimg_per_epoch:
            rperm = np.random.permutation(nimg)
            inds_all = np.hstack((inds_all, rperm))
                
        if self.autocast:
            self.scaler = GradScaler()
        
        if dataloader: 
            # Generators
            # some parameters like gamma_range are not opened up here, just left to defaults
            # some are passed through the model parameters (self.omni etc)
            kwargs = {'rescale': rescale, 
                      'diam_train': diam_train if rescale else None,
                      'tyx': tyx,
                      'scale_range': scale_range,        
                      'omni': self.omni,
                      'dim': self.dim,
                      'nchan': self.nchan,
                      'nclasses': self.nclasses,
                      'device': self.device if not ARM else torch.device('cpu'), # MPS slower than cpu for flow
                      'affinity_field': affinity_field
                     }
            # torch.multiprocessing.set_start_method('spawn')
            
            training_set = omnipose.data.train_set(train_data, train_labels, train_links, **kwargs)       

            # reproducible batches 
            torch.manual_seed(42)
            
            sampler = torch.utils.data.sampler.BatchSampler(torch.utils.data.sampler.RandomSampler(training_set),
                                                            batch_size=batch_size,
                                                            drop_last=False) 
            
            # consider making drop_last true... it might still be a lot faster
            # or maybe find a way to make the last several bathces a bit smaller but still large to balance load 

            print('num_workers',num_workers,'batch_size', batch_size)
            
            # params = {'batch_size': batch_size,
            params = {'batch_size': 1, # this batch size means something like how many worker batches to aggregate 
                      'shuffle': False, # use sampler instead
                      'collate_fn': training_set.collate_fn,
                      'worker_init_fn': training_set.worker_init_fn,
                      'pin_memory': False, # only useful for CPU tensors
                      'num_workers': num_workers, 
                      'sampler': sampler,
                      'persistent_workers': True if num_workers>0 else False,
                      'multiprocessing_context': 'spawn',
                      'prefetch_factor': batch_size
                     }

            core_logger.info((">>>> Using torch dataloader. "
                              "Can take a couple min to initialize. "
                              "Using {} workers.").format(num_workers))
            
            train_loader = torch.utils.data.DataLoader(training_set, **params)
            
            if test_data is not None:
                print('will need to fix sampler')
                validation_set = omnipose.data.train_set(test_data, test_labels, test_links, **kwargs)
                validation_loader = torch.utils.data.DataLoader(validation_set, **params)

        # for debugging
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%H:%M:%S')
        core_logger.info('>>>> Start time: {}'.format(formatted_time))
        
        de = 2 # epoch print interval
        iepoch0 = 0
        epochtime = np.zeros(self.n_epochs)

        # for weighting function
        # blurrer = torchvision.transforms.GaussianBlur(kernel_size=11, sigma=2)

        # edge handling requires distance field for cutoffs
        # do this with batch?, turn off gradient return 
        # dist = train_labels
        
        for iepoch in range(self.n_epochs):    
            self.iepoch = iepoch
            if SGD:
                self._set_learning_rate(self.learning_rate[iepoch])
            
            np.random.seed(iepoch)
            datatime = []
            steptime = []
            if dataloader:
                # for batch_data, batch_labels, batch_idx in train_loader:
                for batch_idx, (batch_data, batch_labels, batch_inds) in enumerate(train_loader):
                # for batch_inds in train_loader.batch_sampler:
                #     batch_data, batch_labels = train_loader(batch_inds)
                    # shape = batch_data.shape

                    tic = time.time()

                    # print('\t Batch number',batch_idx)
                    # print('yoyoyo',batch_data.shape,batch_inds)

                    nbatch = len(batch_data)
                    dt = tic-toc
                    datatime += [dt]
                    if timing:
                        print('\t Dataloading time  (dataloader): {:.2f}'.format(dt))
                    train_loss = self._train_step(batch_data, batch_labels)

                    dt = time.time()-tic
                    steptime += [dt] 
                    if timing:
                        print('\t Step time: {:.2f}, batch size {}'.format(dt,nbatch))

                    lsum += train_loss
                    nsum += nbatch 

            else:
                rperm = inds_all[iepoch*nimg_per_epoch:(iepoch+1)*nimg_per_epoch]
                for ibatch in range(0,nimg_per_epoch,batch_size):
                    tic = time.time() 
                    inds = rperm[ibatch:ibatch+batch_size]
                    rsc = diam_train[inds] / self.diam_mean if rescale else np.ones(len(inds), np.float32)
                    # now passing in the full train array, need the labels for distance field

                    # with omni=False, lablels[i] is (4,*dims). With omni=True, labels is (*dims,)
                    imgi, lbl, scale = transforms.random_rotate_and_resize([train_data[i] for i in inds], 
                                                                           Y=[train_labels[i] for i in inds],
                                                                           rescale=rsc, 
                                                                           scale_range=scale_range, 
                                                                           unet=self.unet, 
                                                                           tyx=tyx, 
                                                                           inds=inds,
                                                                           omni=self.omni, 
                                                                           dim=self.dim, 
                                                                           nchan=self.nchan, 
                                                                           nclasses=self.nclasses, 
                                                                           device=self.device)
                    
                    
                    t2 = time.time()
                    # new parallized approach: random warp+ image augmentation, batch flows
                    links = [train_links[idx] for idx in inds]
                    # with omni=True, lbl is just (nbatch,*dims) sinc eit is labels only, but with omni=False then (nbatch,nclasses, *dims)

                    out = omnipose.core.masks_to_flows_batch(lbl, links, 
                                                             device=self.device, 
                                                             omni=self.omni, 
                                                             dim=self.dim,
                                                             affinity_field=affinity_field
                                                            )[:-2]
                    
                    X = out[:-1]
                    slices = out[-1]
                    masks,bd,T,mu = [torch.stack([x[(Ellipsis,)+slc] for slc in slices]) for x in X]
                    lbl = omnipose.core.batch_labels(masks,bd,T,mu,tyx,
                                                     dim=self.dim,
                                                     nclasses=self.nclasses,
                                                     device=self.device)
      

                    dt = time.time()-tic
                    datatime += [dt]
                    if timing:
                        print('\t Dataloading time (manual batching): {:.2f}'.format(dt))
                    nbatch = len(imgi) 

                    if self.unet and lbl.shape[1]>1 and rescale:
                        lbl[:,1] /= diam_batch[:,np.newaxis,np.newaxis]**2

                    tic = time.time()
                    # train step now expects tensors
                    # train_loss = self._train_step( self._to_device(imgi),  self._to_device(lbl))
                    # train_loss = self._train_step( torch.stack(imgi).to(self.device), torch.stack(lbl).to(self.device))
                    # train_loss = self._train_step( self._to_device(np.stack(imgi)),  self._to_device(np.stack(lbl)))
                    train_loss = self._train_step(self._to_device(np.stack(imgi)),lbl)


                    dt = time.time()-tic
                    steptime += [dt]
                    if timing:
                        print('\t Step time: {:.2f}, batch size {}'.format(dt,len(imgi)))

                    lsum += train_loss
                    nsum += nbatch

#                 tic = time.time()
#                 if iepoch%de==0 or iepoch==5:
#                     lavg = lavg / nsum
#                     if test_data is not None:
#                         lavgt, nsum = 0., 0
#                         np.random.seed(42)
#                         rperm = np.arange(0, len(test_data), 1, int)
#                         for ibatch in range(0,len(test_data),batch_size):
#                             inds = rperm[ibatch:ibatch+batch_size]
#                             rsc = diam_test[inds] / self.diam_mean if rescale else np.ones(len(inds), np.float32)
#                             imgi, lbl, scale = transforms.random_rotate_and_resize([test_data[i] for i in inds], 
#                                                                                    Y=[test_labels[i] for i in inds], 
#                                                                                    links=[test_links[i] for i in inds],
#                                                                                    rescale=rsc, 
#                                                                                    scale_range=0., 
#                                                                                    unet=self.unet, 
#                                                                                    tyx=tyx, 
#                                                                                    inds=inds,
#                                                                                    omni=self.omni, 
#                                                                                    dim=self.dim, 
#                                                                                    nchan=self.nchan, 
#                                                                                    nclasses=self.nclasses,
#                                                                                    device=self.device)
#                             if self.unet and lbl.shape[1]>1 and rescale:
#                                 lbl[:,1] *= scale[0]**2

#                             test_loss = self._test_eval(imgi, lbl)
#                             lavgt += test_loss
#                             nsum += len(imgi)

#                         core_logger.info('Epoch %d, Time %4.1fmin, Loss %2.4f, Loss Test %2.4f, LR %2.4f, Time per Epoch %3.1fs'%
#                                 (iepoch, (tic-t0) / 60 , lavg, lavgt/nsum, self.learning_rate[iepoch], (tic-toc) / de))
#                     else:
#                         core_logger.info('Epoch %d, Time %4.1fmin, Loss %2.4f, LR %2.4f, Time per Epoch %3.1fs, Avg batch loading time: %3.1fs'%
#                                 (iepoch, (tic-t0) / 60 , lavg, self.learning_rate[iepoch], (tic-toc) / de, np.mean(datatime)))

#                     toc = time.time() # end of batch 
#                     lavg, nsum = 0, 0

            tic = time.time()
            epochtime[iepoch] = tic
            if iepoch % de == 0:
                core_logger.info(
                    ("Train epoch: {} | "
                    "Time: {:.2f}min | "
                    "last epoch: {:.2f}s | "
                     # "batch size: {} | "
                    "<sec/epoch>: {:.2f}s | "
                    "<sec/batch>: {:.2f}s | "
                    "<Batch Loss>: {:.6f} | "
                    "<Epoch Loss>: {:.6f}").
                    strip().
                    format(iepoch, # epoch index 
                           (tic-t0) / 60, # absolute time spent in minutes 
                           0 if iepoch==iepoch0 else (tic-toc) / (iepoch-iepoch0), # time spent in this epoch 
                           # nbatch,
                           # (tic-t0) / (iepoch+1), # average time per epoch 
                           0 if iepoch==iepoch0 else np.mean(np.diff(epochtime[max(0,iepoch-3):max(1,iepoch)])),
                           np.mean(datatime[-3:]), # average time spent loading fata for last 3 epochs 
                           train_loss/nbatch, # average loss over this batch 
                           lsum/nsum) # average loss over this epoch 
                )
            iepoch0 = iepoch
            toc = time.time() # end of batch 
            lsum, nsum = 0, 0

            if save_path is not None:
                if iepoch==self.n_epochs-1 or iepoch%save_every==0 or (self.n_epochs-iepoch)<10:
                    # save model at the end
                    if save_each: #separate files as model progresses 
                        if netstr is None:
                            file_name = '{}_{}_{}_{}'.format(self.net_type, file_label, 
                                                             d.strftime("%Y_%m_%d_%H_%M_%S.%f"),
                                                             'epoch_'+str(iepoch)) 
                        else:
                            file_name = '{}_{}'.format(netstr, 'epoch_'+str(iepoch))
                    else:
                        if netstr is None:
                            file_name = '{}_{}_{}'.format(self.net_type, 
                                                          file_label, 
                                                          d.strftime("%Y_%m_%d_%H_%M_%S.%f"))
                        else:
                            file_name = netstr
                    file_name = os.path.join(file_path, file_name)
                    ksave += 1
                    core_logger.info(f'saving network parameters to {file_name}')

                    # self.net.save_model(file_name)
                    # whether or not we are using dataparallel 
                    # this logic appears elsewhere in models.py
                    if self.torch and self.gpu:
                        self.net.module.save_model(file_name)
                    else:
                        self.net.save_model(file_name)

            else:
                file_name = save_path

        # reset to mkldnn if available
        self.net.mkldnn = self.mkldnn

        return file_name

