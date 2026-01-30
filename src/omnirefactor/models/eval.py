from .imports import *
import inspect
import warnings
from ..kwargs import base_kwargs, split_kwargs_for
from ..logger import TqdmToLogger


def eval(self, x, batch_size=8, indices=None, channels=None, channel_axis=None,
         z_axis=None, normalize=True, invert=False,
         rescale_factor=None, diameter=None, do_3D=False, anisotropy=None, net_avg=True,
         augment=False, tile=False, tile_overlap=0.1, bsize=224, num_workers=8,
         loader_batch_size=1, # for torch dataloader (also used as max_batch_size for iter_batches)
         batch_mode='auto',  # for iter_batches: 'auto', 'group', 'pad', 'single'
         resample=True, progress=None, show_progress=True,
         omni=True, calc_trace=False, verbose=False, transparency=False,
         loop_run=False, model_loaded=False, hysteresis=True, **kwargs):
    """
        Evaluation for OmniModel. Segment list of images x, or 4D array - Z x nchan x Y x X

        Parameters
        ----------
        x: list or array of images
            can be list of 2D/3D/4D images, or array of 2D/3D/4D images

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

        normalize: bool (default, True)
            normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

        invert: bool (optional, default False)
            invert image pixel intensity before running network

        rescale_factor: float (optional, default None)
            resize factor for each image, if None, set to 1.0
            NOTE: the legacy kwarg `rescale` is deprecated; use `rescale_factor`.

        diameter: float (optional, default None)
            diameter for each image (only used if rescale_factor is None), 
            if diameter is None, set to diam_mean

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

        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows

        resample: bool (optional, default True)
            run dynamics at original image size (will be slower but create more accurate boundaries)

        interp: bool (optional, default True)
            interpolate during 2D dynamics (not available in 3D) 
            (in previous versions it was False)

        flow_threshold: float (optional, default 0.4)
            flow error threshold (all cells with errors below threshold are kept) (not used for 3D)

        mask_threshold: float (optional, default 0.0)
            all pixels with value above threshold kept for masks, decrease to find more and larger masks

        compute_masks: bool (optional, default True)
            Whether or not to compute dynamics and return masks.
            This is set to False when retrieving the styles for the size model.

        min_size: int (optional, default 15)
            minimum number of pixels per mask, can turn off with -1

        stitch_threshold: float (optional, default 0.0)
            if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation

        progress: pyqt progress bar (optional, default None)
            to return progress bar status to GUI
            
        omni: bool (optional, default False)
            use omnipose mask reconstruction features
        
        calc_trace: bool (optional, default False)
            calculate pixel traces and return as part of the flow
            
        verbose: bool (optional, default False)
            turn on additional output to logs for debugging 
        
        transparency: bool (optional, default False)
            modulate flow opacity by magnitude instead of brightness (can use flows on any color background) 
        
        loop_run: bool (optional, default False)
            internal variable for determining if model has been loaded, stops model loading in loop over images

        model_loaded: bool (optional, default False)
            internal variable for determining if model has been loaded, used in __main__.py

        Returns
        -------
        masks: list of 2D arrays, or single 3D array (if do_3D=True)
            labelled image, where 0=no masks; 1,2,...=mask labels

        flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
            flows[k][0] = 8-bit RGb phase plot of flow field
            flows[k][1] = flows at each pixel
            flows[k][2] = scalar cell probability (Cellpose) or distance transform (Omnipose)
            flows[k][3] = boundary output (nonempty for Omnipose)
            flows[k][4] = final pixel locations after Euler integration 
            flows[k][5] = pixel traces (nonempty for calc_trace=True)

        styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
            style vector summarizing each image, also used to estimate size of objects in image

    """
    
    base_args = base_kwargs(locals(), exclude={"self", "x", "kwargs"})
    mask_kwargs, step_kwargs, affinity_kwargs = split_kwargs(
        [core.compute_masks, core.steps_batch, core._get_affinity_torch],
        base_args,
        strict=False,
    )
    # lift commonly-used items back into locals for readability
    interp = step_kwargs.get("interp", True) # not used below? 
    niter = step_kwargs.get("niter")
    flow_threshold = mask_kwargs.get("flow_threshold", 0.4)
    mask_threshold = mask_kwargs.get("mask_threshold", 0.0)
    compute_masks = mask_kwargs.get("compute_masks", True) # not used below? 
    stitch_threshold = mask_kwargs.get("stitch_threshold", 0.0)
    affinity_seg = mask_kwargs.get("affinity_seg", False)
    suppress = mask_kwargs.get("suppress")
    
    # images are given has a list, especially when heterogeneous in shape
    is_grey = np.sum(channels)==0
    slice_ndim = self.dim+do_3D+(self.nchan>1 and not is_grey)+(channel_axis is not None)
    # the logic here needs to be updated to account for the fact that images may not already match the expected dims
    # and channels, namely mono channel might have a 2-channel model. I should just check for if the number of channels could
    # possibly match, and warn that internal conversion will happen or may break...
    is_list = isinstance(x, list)
    is_stack = is_image = False
    
    if verbose:
        models_logger.info(f'is_grey {is_grey}, slice_ndim {slice_ndim}, dim {self.dim}, nchan {self.nchan}, is_list {is_list}')
    
    if isinstance(x, np.ndarray):
        # [0,0] is a special instance where we want to run the model on a single channel
        dim_diff = x.ndim-slice_ndim
        opt = np.array([0,1])#-is_grey
        is_image, is_stack = [dim_diff==i for i in opt]
        correct_shape = dim_diff in opt     
                
    
    if verbose:
        models_logger.info(f'is_image {is_image}, is_stack {is_stack}, is_list {is_list}')
    # print('a1',interp,hysteresis,calc_trace)
    
    # allow for a dataset to be passed so that we can do batches 
    # will be defined in omnipose.data.train_set 
    is_dataset = isinstance(x,torch.utils.data.Dataset) # if using eval_set
    if is_dataset:
        correct_shape = True # assume the dataset has the right shape

    if not (is_list or is_stack or is_dataset or is_image or loop_run):
        models_logger.warning('input images must be a list of images, array of images, or dataloader')
    else:
        if is_list:
            correct_shape = np.all([x[i].squeeze().ndim == slice_ndim for i in range(len(x))])
   
        if not correct_shape:
            # print(slice_ndim,x.ndim,is_list,is_stack)
            models_logger.warning('input images do not match the expected number of dimensions ({}) \nand channels ({}) of model.'.format(self.dim,self.nchan))



    if verbose and (is_dataset or not (is_list or is_stack)):
        models_logger.info('Evaluating with flow_threshold %0.2f, mask_threshold %0.2f'%(flow_threshold, mask_threshold))
        if omni:
            models_logger.info(f'using omni model, cluster {mask_kwargs.get("cluster", False)}')

    
    # Note: dataset is finetuned for basic omnipose usage. No styles are returned, some options may not be supported.
    if is_dataset:

        if verbose:
            models_logger.warning('Using dataset evaluation branch. Some options not yet supported.')

        # set the tile parameter in dataset
        x.tile = tile

        # set the rescale parameter in dataset
        x.rescale_factor = 1.0 if rescale_factor is None else rescale_factor
        # avoid double normalization; handled in this eval branch
        x.normalize = False
        x.invert = False

        # Configure batch mode for iter_batches()
        x.batch_mode = x._resolve_batch_mode(batch_mode)
        x.max_batch_size = loader_batch_size
        x._build_batch_plan()

        # sample indices to evaluate
        indices = list(range(len(x))) if indices is None else indices

        dist, dP, bd, masks, bounds, p, tr, affinity, flow_RGB = [], [], [], [], [], [], [], [], []

        # Use new iter_batches() approach for shape-aware batching
        progress_bar = tqdm(total=len(indices), disable=not show_progress)
        for batch, inds, subs in x.iter_batches():
            batch = batch.float()
            if normalize or invert:
                batch_np = batch.numpy()
                batch_np = batch_np.transpose(0, 2, 3, 1)
                for i in range(batch_np.shape[0]):
                    batch_np[i] = transforms.normalize_img(batch_np[i], axis=-1, invert=invert, omni=omni)
                batch_np = batch_np.transpose(0, 3, 1, 2)
                batch = torch.from_numpy(batch_np)

            batch = batch.to(self.device)  # move to GPU

            shape = batch.shape
            nimg = batch.shape[0]
            nchan = batch.shape[1]

            # Pad channels if model expects more than we have (e.g., grayscale input, 2-channel model)
            if self.nchan > nchan:
                pad_shape = (nimg, self.nchan - nchan) + tuple(batch.shape[2:])
                batch = torch.cat([batch, torch.zeros(pad_shape, device=self.device, dtype=batch.dtype)], dim=1)
                nchan = self.nchan

            shape = batch.shape[-(self.dim+1):]  # nclasses, Y, X
            resize = shape[-self.dim:] if not resample else None

            # run the network on the batch
            with torch.no_grad():
                if tile:
                    yf = x._run_tiled(batch, self,
                                      batch_size=batch_size,
                                      bsize=bsize,
                                      augment=augment,
                                      tile_overlap=tile_overlap)
                else:
                    yf = self.run_network(batch, to_numpy=False)[0]

                del batch

            # Extract each image using its per-image subs (handles different sizes in 'pad' mode)
            # subs is a list of [slice_y, slice_x] or similar for each image in batch
            yf_list = []
            for i in range(nimg):
                img_subs = subs[i]  # List of slice objects for this image
                # Build the slice tuple: (all_classes, spatial_slices...)
                slc = [slice(None, self.nclasses)]  # Keep all output classes
                slc.extend(img_subs)  # Add spatial slices
                yf_i = yf[i][tuple(slc)]
                yf_list.append(yf_i)

            # Stack back if all same shape, otherwise process individually
            # For 'group' and 'single' modes, all should be same shape
            # For 'pad' mode, they may differ
            try:
                yf = torch.stack(yf_list, dim=0)
            except RuntimeError:
                # Different shapes - will process individually below
                pass

            # rescale and resample (only if stacked)
            if isinstance(yf, torch.Tensor) and resample and rescale_factor not in [None, 1.0, 0]:
                yf = torch_zoom(yf, 1 / rescale_factor)
                yf_list = [yf[i] for i in range(yf.shape[0])]
            elif resample and rescale_factor not in [None, 1.0, 0]:
                yf_list = [torch_zoom(yf_i.unsqueeze(0), 1 / rescale_factor).squeeze(0) for yf_i in yf_list]

            # Process each image in the batch
            for i in range(nimg):
                yf_i = yf_list[i] if isinstance(yf_list, list) else yf[i]

                # compared to the usual per-image pipeline, this one will not support cellpose or u-net
                flow_pred_i = yf_i[:self.dim]
                dist_pred_i = yf_i[self.dim]  # scalar field always after the vector field output

                if self.nclasses >= self.dim + 2:
                    bd_pred_i = yf_i[self.dim + 1]
                    bd_i = self._from_device(bd_pred_i.unsqueeze(0))[0]
                else:
                    bd_i = None

                # RGB flow visualization
                rgb_i = plot.rgb_flow(flow_pred_i.unsqueeze(0), transparency=transparency)

                # hysteresis thresholding
                if hysteresis:
                    foreground = hysteresis_threshold(dist_pred_i.unsqueeze(0).unsqueeze(0),
                                                      mask_threshold - 1, mask_threshold).squeeze()
                else:
                    foreground = dist_pred_i >= mask_threshold

                # add to output lists
                dP.append(self._from_device(flow_pred_i.unsqueeze(0))[0])
                dist.append(self._from_device(dist_pred_i.unsqueeze(0))[0])
                bd.append(bd_i)
                flow_RGB.append(self._from_device(rgb_i)[0])

                # run compute_masks
                mask_base = dict(mask_kwargs)
                for key in ("bd", "p", "coords", "iscell", "affinity_graph"):
                    mask_base.pop(key, None)
                if rescale_factor is None:
                    mask_base["rescale_factor"] = 1.0
                # Add defaults that run_batch provides but aren't in eval signature.
                # These must match the run_batch defaults for parity.
                # NOTE: setdefault() only sets a value when the key is MISSING.
                # It does NOT override existing keys with None values. For params
                # that may be explicitly None in mask_kwargs, use explicit checks.
                if mask_base.get("min_size") is None:
                    mask_base["min_size"] = 15
                if mask_base.get("flow_factor") is None:
                    mask_base["flow_factor"] = 5.0
                mask_base.setdefault("max_size", None)
                mask_base.setdefault("interp", True)
                mask_base.setdefault("cluster", False)
                mask_base.setdefault("suppress", None)
                mask_base.setdefault("affinity_seg", False)
                mask_base.setdefault("despur", False)
                mask_base.update({
                    "use_gpu": self.gpu,
                    "device": self.device,
                    "nclasses": self.nclasses,
                    "dim": self.dim,
                    "bd": bd_i,
                })
                outputs = core.compute_masks(
                    self._from_device(flow_pred_i.unsqueeze(0))[0],
                    self._from_device(dist_pred_i.unsqueeze(0))[0],
                    **mask_base
                )
                masks.append(outputs[0])
                p.append(outputs[1])
                tr.append(outputs[2])
                bounds.append(outputs[3])
                affinity.append(outputs[4])

                progress_bar.update()
                empty_cache()

            del yf_list

        masks = np.array(masks)
        bounds = np.array(bounds)
        p = np.array(p)
        tr = np.array(tr)
        ret = [masks, dP, dist, p, bd, tr, affinity, bounds, flow_RGB]
        
        progress_bar.close()


        for r in ret:
            r.squeeze() if isinstance(r,np.ndarray) else r 

        
        # the flow list stores: 
        # (1) RGB representation of flows
        # (2) flow components
        # (3) cellprob (cp) or distance field (op)
        # (4) pixel coordinates after Euler integration
        # (5) boundary output (nclasses=4)
        # (6) pixel trajectories during Euler integation (trace=True)
        # (7) nstep_by_npix affinity graph
        # (8) binary boundary map
        # 5-8 were added in Omnipose, hence the unusual placement in the list. 
        # flows = [[o for o in out] for out in zip(rgb, dP, cellprob, p, bd, tr, affinity, bounds)]
        flows = [list(item) for item in zip(flow_RGB, dP, dist, p, bd, tr, affinity, bounds)] # not sure which is faster of these yet
        return masks, flows, [] 
    
    # default non-dataset branch
    elif (is_list or is_stack) and correct_shape:
        masks, styles, flows = [], [], []

        tqdm_out = TqdmToLogger(models_logger, level=logging.INFO)
        nimg = len(x)
        iterator = trange(nimg, file=tqdm_out,disable=not show_progress) if nimg>1 else range(nimg)
        # note: ~ is bitwise flip, overloaded to act as elementwise not for numpy arrays
        # but for boolean variables, must use "not" operator isstead 
        if verbose:
            models_logger.info('Evaluating one image at a time')
        
        for i in iterator:
            dia = diameter[i] if isinstance(diameter, list) or isinstance(diameter, np.ndarray) else diameter
            rsc = rescale_factor[i] if isinstance(rescale_factor, list) or isinstance(rescale_factor, np.ndarray) else rescale_factor
            chn = channels if channels is None else channels[i] if (len(channels)==len(x) and 
                                                                    (isinstance(channels[i], list) 
                                                                     or isinstance(channels[i], np.ndarray)) and
                                                                    len(channels[i])==2) else channels
            
            base_eval = split_kwargs_for(eval, locals(), exclude={"self", "x", "kwargs"})
            eval_kwargs = dict(base_eval)
            eval_kwargs.update({
                "channels": chn,
                "channel_axis": channel_axis,
                "z_axis": z_axis,
                "rescale_factor": rsc,
                "diameter": dia,
                "loop_run": (i > 0),
                "model_loaded": model_loaded,
            })
            maski, stylei, flowi = self.eval(x[i], **eval_kwargs)
            masks.append(maski)
            flows.append(flowi)
            styles.append(stylei)
        return masks, styles, flows 
    
    else:
        if not model_loaded and (isinstance(self.pretrained_model, list) and not net_avg and not loop_run):

            # whether or not we are using dataparallel 
            if self.torch and self.gpu:
                models_logger.info(f'using dataparallel')
                net = self.net.module
                    
            else:
                net = self.net
                models_logger.info('not using dataparallel')
                
            if verbose: 
                models_logger.info(f'network initialized.')
            
            net.load_model(self.pretrained_model[0], cpu=(not self.gpu))

        if verbose: 
            models_logger.info('shape before transforms.convert_image(): {}'.format(x.shape))
            models_logger.info(f'model dim: {self.dim}')
            

        # This takes care of the special case of grayscale, padding with zeros if the model was trained like that
        x = transforms.convert_image(x, channels, channel_axis=channel_axis, z_axis=z_axis,
                                     do_3D=(do_3D or stitch_threshold>0), normalize=False, 
                                     invert=False, nchan=self.nchan, dim=self.dim, omni=omni)
        
        
        if verbose: 
            models_logger.info('shape after transforms.convert_image(): {}'.format(x.shape))
            
        if x.ndim < self.dim+2: # we need (nimg, *dims, nchan), so 2D has 4, 3D has 5, etc. 
            x = x[np.newaxis]

            if verbose: 
                models_logger.info('shape now {}'.format(x.shape))

        self.batch_size = batch_size
        rescale_factor = self.diam_mean / diameter if (rescale_factor is None and (diameter is not None and diameter>0)) else rescale_factor
        rescale_factor = 1.0 if rescale_factor is None else rescale_factor

        run_kwargs = split_kwargs_for(self.run_batch, locals(), exclude={"self", "x", "kwargs"})
        masks, styles, dP, cellprob, p, bd, tr, affinity, bounds = self.run_batch(x, **run_kwargs)
        
        # the flow list stores: 
        # (1) RGB representation of flows
        # (2) flow components
        # (3) cellprob (cp) or distance field (op)
        # (4) pixel coordinates after Euler integration
        # (5) boundary output (nclasses=4)
        # (6) pixel trajectories during Euler integation (trace=True)
        # (7) augmented affinity graph (coords+affinity) of shape (dim,nstep,npix)
        # (8) binary boundary map
        
        # 5-8 were added in Omnipose, hence the unusual placement in the list. 
        flows = [plot.dx_to_circ(dP,transparency=transparency) 
                 if self.nclasses>1 else np.zeros(cellprob.shape+(3+transparency,),np.uint8),
                 dP, cellprob, p, bd, tr, affinity, bounds]
    
        return masks, flows, styles


def run_batch(self, x, compute_masks=True, normalize=True, invert=False,
            rescale_factor=1.0, net_avg=True, resample=True,
            augment=False, tile=False, tile_overlap=0.1, bsize=224,
            mask_threshold=0.0, diam_threshold=12., flow_threshold=0.4, niter=None, flow_factor=5.0,
            min_size=15, max_size=None,
            interp=True, cluster=False, suppress=None, affinity_seg=False, despur=False,
            anisotropy=1.0, do_3D=False, stitch_threshold=0.0,
            omni=True, calc_trace=False,  show_progress=True, verbose=False, pad=0):
    
    
    # by this point, the image(s) will already have been formatted with channels, batch, etc 

    tic = time.time()
    shape = x.shape
    nimg = shape[0] 
    bd, tr, affinity = None, None, None
    # set up image padding for prediction - set to 0 not as it actually doesn't really help 
    # note that this is not the same padding as what you need for the network to run 
    pad_seq = [(pad,)*2]*self.dim + [(0,)*2] # do not pad channel axis 
    unpad = tuple([slice(pad,-pad) if pad else slice(None,None)]*self.dim) # works in case pad is zero
            
    if do_3D:
        img = np.asarray(x)
        if normalize or invert: # possibly make normalize a vector of upper-lower values  
            img = transforms.normalize_img(img, invert=invert, omni=omni)
        # have not tested padding in do_3d yet 
        # img = np.pad(img,pad_seq,'reflect')
        
        yf, styles = self._run_3D(img, rsz=rescale_factor, anisotropy=anisotropy, 
                                  net_avg=net_avg, augment=augment, tile=tile,
                                  tile_overlap=tile_overlap)
        # unpadding 
        # yf = yf[unpad+(Ellipsis,)]
        
        cellprob = np.sum([yf[k][2] for k in range(3)],axis=0)/3 if omni else np.sum([yf[k][2] for k in range(3)],axis=0)
        bd = np.sum([yf[k][3] for k in range(3)],axis=0)/3 if self.nclasses==(self.dim+2) else np.zeros_like(cellprob)
        dP = np.stack((yf[1][0] + yf[2][0], yf[0][0] + yf[2][1], yf[0][1] + yf[1][1]), axis=0) # (dZ, dY, dX)
        if omni:
            dP = np.stack([gaussian_filter(dP[a],sigma=1.5) for a in range(3)]) # remove some artifacts
            bd = gaussian_filter(bd,sigma=1.5)
            cellprob = gaussian_filter(cellprob,sigma=1.5)
            dP = dP/2 #should be averaging components 
        del yf
    else:
        tqdm_out = TqdmToLogger(models_logger, level=logging.INFO,)
        iterator = trange(nimg, file=tqdm_out, disable=not show_progress) if nimg>1 else range(nimg)
        styles = np.zeros((nimg, self.nbase[-1]), np.float32)
        
        #indexing a little weird here due to channels being last now 
        no_channel_axis = (x.ndim == self.dim + 1)
        if resample:
            s = tuple(shape[-self.dim:]) if no_channel_axis else tuple(shape[-(self.dim+1):-1])
        else:
            base_shape = shape[-self.dim:] if no_channel_axis else shape[-(self.dim+1):-1]
            s = tuple(np.round(np.array(base_shape) * rescale_factor).astype(int))
        
        dP = np.zeros((self.dim, nimg,)+s, np.float32)
        cellprob = np.zeros((nimg,)+s, np.float32)
        bounds = np.zeros((nimg,)+s, bool)
        
        for i in iterator:
            img = np.asarray(x[i])
            # at this point, img should be (*DIMS,C)

            if normalize or invert:
                img = transforms.normalize_img(img, invert=invert, omni=omni)
               # at this point, img should still be (*DIMS,C) 
               
            # pad the image to get cleaner output at the edges
            # padding with edge values seems to work the best
            # but actually, not really useful in the end...
            if pad>0:
                img = np.pad(img,pad_seq,'edge')

            if rescale_factor != 1.0:
                # if self.dim>2:
                #     print('WARNING, resample not updated for ND')
                # img = transforms.resize_image(img, rsz=rescale)
                
                if img.ndim>self.dim: # then there is a channel axis, assume it is last here 
                    img = np.stack([zoom(img[...,k], rescale_factor, order=3) for k in range(img.shape[-1])], axis=-1)
                else:
                    img = zoom(img, rescale_factor, order=1)
                    
            # inherited from Unet 
            # returns numpy arrays by default, not torch tensors
            yf, style = self._run_nets(img, net_avg=net_avg,
                                       augment=augment, tile=tile,
                                       normalize=normalize, 
                                       tile_overlap=tile_overlap, 
                                       bsize=bsize)
            # unpadding 
            yf = yf[unpad+(Ellipsis,)]
                            
            # resample interpolates the network output to native resolution prior to running Euler integration
            # this means the masks will have no scaling artifacts. We could *upsample* by some factor to make
            # the clustering etc. work even better, but that is not implemented yet 
            if resample and rescale_factor != 1.0:
                # for k in range(yf.shape[-1]):
                #     print('a',shape[1:1+self.dim]/np.array(yf.shape[:-1]))
                # ND version actually gives better results than CV2 in some places. 
                yf = np.stack([zoom(yf[...,k], shape[1:1+self.dim]/np.array(yf.shape[:-1]), order=1) 
                               for k in range(yf.shape[-1])],axis=-1)
                # scipy.ndimage.affine_transform(A, np.linalg.inv(M), output_shape=tyx,
            
            if self.nclasses>1:
                cellprob[i] = yf[...,self.dim] #scalar field always after the vector field output 
                order = (self.dim,)+tuple([k for k in range(self.dim)]) #(2,0,1)
                dP[:, i] = yf[...,:self.dim].transpose(order) 
            else:
                cellprob[i] = yf[...,0]
                # dP[i] =  np.zeros(cellprob)
                
            if self.nclasses>=self.dim+2:
                if i==0:
                    bd = np.zeros_like(cellprob)
                bd[i] = yf[...,self.dim+1]
                
            styles[i] = style
        del yf, style
    styles = styles.squeeze()
    
    net_time = time.time() - tic
    if nimg > 1:
        models_logger.info('network run in %2.2fs'%(net_time))





    if compute_masks:

        tic = time.time()

        # allow user to specify niter
        # Cellpose default is 200
        # Omnipose default is None (dynamically adjusts based on distance field)
        if niter is None and not omni:
            niter = 200 if (do_3D and not resample) else (1 / rescale_factor * 200)

        if do_3D:
            resize = None
            mask_base = split_kwargs_for(core.compute_masks, locals(), exclude={"self", "kwargs"})
            sig = inspect.signature(core.compute_masks)
            for key in list(sig.parameters)[:2]:
                mask_base.pop(key, None)
            for key in ("p", "coords", "iscell", "affinity_graph"):
                mask_base.pop(key, None)
            for key in ("p", "coords", "iscell", "affinity_graph"):
                mask_base.pop(key, None)
            sig = inspect.signature(core.compute_masks)
            for key in list(sig.parameters)[:2]:
                mask_base.pop(key, None)
            mask_base.update({
                "use_gpu": self.gpu,
                "device": self.device,
                "nclasses": self.nclasses,
                "dim": self.dim,
                "bd": bd,
            })
            masks, bounds, p, tr, affinity = core.compute_masks(
                dP,
                cellprob,
                **mask_base,
            )
        else:
            masks, bounds, p, tr, affinity = [], [], [], [], []
            resize = shape[-(self.dim+1):-1] if not resample else None
            mask_base = split_kwargs_for(core.compute_masks, locals(), exclude={"self", "kwargs"})
            sig = inspect.signature(core.compute_masks)
            for key in list(sig.parameters)[:2]:
                mask_base.pop(key, None)
            for key in ("p", "coords", "iscell", "affinity_graph"):
                mask_base.pop(key, None)
            for key in ("p", "coords", "iscell", "affinity_graph"):
                mask_base.pop(key, None)
            sig = inspect.signature(core.compute_masks)
            for key in list(sig.parameters)[:2]:
                mask_base.pop(key, None)
            mask_base.update({
                "use_gpu": self.gpu,
                "device": self.device,
                "nclasses": self.nclasses,
                "dim": self.dim,
            })
            for i in iterator:
                bdi = bd[i] if bd is not None else None
                mask_kwargs = dict(mask_base)
                mask_kwargs["bd"] = bdi
                outputs = core.compute_masks(
                    dP[:, i],
                    cellprob[i],
                    **mask_kwargs,
                )
                masks.append(outputs[0])
                p.append(outputs[1])
                tr.append(outputs[2])
                bounds.append(outputs[3])
                affinity.append(outputs[4])

            masks = np.array(masks)
            bounds = np.array(bounds)
            p = np.array(p)
            tr = np.array(tr)
            affinity = np.array(affinity)

            if stitch_threshold > 0 and nimg > 1:
                models_logger.info(f'stitching {nimg} planes using stitch_threshold={stitch_threshold:0.3f} to make 3D masks')
                masks = utils.stitch3D(masks, stitch_threshold=stitch_threshold)

        flow_time = time.time() - tic
        if nimg > 1:
            models_logger.info('masks created in %2.2fs'%(flow_time))

        ret = [masks, styles, dP, cellprob, p, bd, tr, affinity, bounds]
        ret = [r.squeeze() if isinstance(r,np.ndarray) else r for r in ret]

    else:
        #pass back zeros for masks and p if not compute_masks
        ret = [[], styles, dP, cellprob, [], bd, tr, affinity, bounds]
        ret = [r.squeeze() if isinstance(r,np.ndarray) else r for r in ret]

    empty_cache()
    return (*ret,)

    
