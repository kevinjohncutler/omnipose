from .imports import *
import warnings
from pathlib import Path
from ..kwargs import base_kwargs, split_kwargs_for
from ..logger import TqdmToLogger
from ..data.eval import eval_set as EvalSet
from ..io.imio import imread

_IMAGE_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.npy', '.npz', '.czi'}


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
    # --- Route kwargs to downstream functions ---
    base_args = base_kwargs(locals(), exclude={"self", "x", "kwargs"})
    mask_kwargs, *_ = split_kwargs(
        [core.compute_masks, core.steps_batch, core._get_affinity_torch],
        base_args, strict=False,
    )
    mask_threshold   = mask_kwargs.get("mask_threshold", 0.0)
    compute_masks    = mask_kwargs.get("compute_masks", True)
    stitch_threshold = mask_kwargs.get("stitch_threshold", 0.0)

    # --- Normalise x: string → list of paths ---
    if isinstance(x, str):
        p = Path(x)
        x = sorted(str(f) for f in p.iterdir() if f.suffix.lower() in _IMAGE_EXTS) if p.is_dir() else [x]

    # --- Detect input form ---
    is_dataset = isinstance(x, torch.utils.data.Dataset)
    is_image = is_stack = False
    correct_shape = True  # safe default; tightened below for numpy inputs

    if isinstance(x, np.ndarray) or (isinstance(x, list) and x and not isinstance(x[0], str)):
        is_grey   = channels is None or np.sum(channels) == 0
        slice_ndim = self.dim + do_3D + (self.nchan > 1 and not is_grey) + (channel_axis is not None)
        if isinstance(x, np.ndarray):
            dim_diff = x.ndim - slice_ndim
            is_image, is_stack = dim_diff == 0, dim_diff == 1
            correct_shape = dim_diff in (0, 1)
        else:
            correct_shape = all(x[i].squeeze().ndim == slice_ndim for i in range(len(x)))

    if not (is_image or is_stack or is_dataset or isinstance(x, list) or loop_run):
        models_logger.warning('input images must be a list of images, array of images, or dataloader')
    elif not correct_shape:
        models_logger.warning('input images do not match the expected number of dimensions ({}) '
                              'and channels ({}) of model.'.format(self.dim, self.nchan))

    if verbose:
        models_logger.info('Evaluating with flow_threshold %0.2f, mask_threshold %0.2f'
                           % (mask_kwargs.get("flow_threshold", 0.4), mask_threshold))
        if omni:
            models_logger.info(f'using omni model, cluster {mask_kwargs.get("cluster", False)}')

    # --- Model loading and rescale_factor ---
    if not model_loaded and isinstance(self.pretrained_model, list) and not net_avg and not loop_run:
        net = self.net.module if isinstance(self.net, nn.DataParallel) else self.net
        if verbose:
            models_logger.info('network initialized.')
        net.load_model(self.pretrained_model[0], cpu=(not self.gpu))

    self.batch_size = batch_size
    if rescale_factor is None:
        rescale_factor = self.diam_mean / diameter if (diameter is not None and diameter > 0) else 1.0

    # --- Precompute invariant compute_masks kwargs; bd and dim vary per call ---
    _mask_kw = dict(mask_kwargs)
    for _k in ("bd", "p", "coords", "iscell", "affinity_graph"):
        _mask_kw.pop(_k, None)
    if _mask_kw.get("rescale_factor") is None: _mask_kw["rescale_factor"] = 1.0
    if _mask_kw.get("min_size")       is None: _mask_kw["min_size"]       = 15
    if _mask_kw.get("flow_factor")    is None: _mask_kw["flow_factor"]    = 5.0
    _mask_kw.setdefault("max_size", None)
    _mask_kw.setdefault("interp", True)
    _mask_kw.setdefault("cluster", False)
    _mask_kw.setdefault("suppress", None)
    _mask_kw.setdefault("affinity_seg", False)
    _mask_kw.setdefault("despur", False)
    _mask_kw.update(use_gpu=self.gpu, device=self.device, nclasses=self.nclasses)

    def _mask_base(dim, bd):
        return {**_mask_kw, "dim": dim, "bd": bd}

    # --- Flatten all non-dataset inputs to a plain list ---
    if not is_dataset:
        if is_image:
            x_list = [x]
        elif is_stack:
            x_list = [x[i] for i in range(x.shape[0])]
        else:
            x_list = list(x)

        if not x_list:
            return np.array([]), [], []

    # ------------------------------------------------------------------ #
    # do_3D: run 2D model on three orthogonal planes and combine results.
    # ------------------------------------------------------------------ #
    if do_3D:
        dist3d, dP3d, bd3d, masks3d, p3d, tr3d, aff3d, rgb3d = [], [], [], [], [], [], [], []

        for raw in x_list:
            img = imread(raw) if isinstance(raw, str) else raw
            img = transforms.convert_image(img, channels,
                                           channel_axis=channel_axis, z_axis=z_axis,
                                           do_3D=True, normalize=False, invert=False,
                                           nchan=self.nchan, dim=self.dim, omni=omni)
            if normalize or invert:
                img = transforms.normalize_img(img, invert=invert, omni=omni)

            yf, _ = self._run_3D(img, rsz=rescale_factor, anisotropy=anisotropy,
                                  net_avg=net_avg, augment=augment, tile=tile,
                                  tile_overlap=tile_overlap)

            cellprob_i = np.sum([yf[k][self.dim] for k in range(3)], axis=0) / 3
            bd_i = (np.sum([yf[k][self.dim + 1] for k in range(3)], axis=0) / 3
                    if self.nclasses == self.dim + 2 else np.zeros_like(cellprob_i))
            dP_i = np.stack((yf[1][0] + yf[2][0],   # dZ
                             yf[0][0] + yf[2][1],    # dY
                             yf[0][1] + yf[1][1]),   # dX
                            axis=0)
            if omni:
                dP_i = np.stack([gaussian_filter(dP_i[a], sigma=1.5) for a in range(3)]) / 2
                bd_i      = gaussian_filter(bd_i,      sigma=1.5)
                cellprob_i = gaussian_filter(cellprob_i, sigma=1.5)

            rgb3d.append(plot.dx_to_circ(dP_i[:2], transparency=transparency))
            dP3d.append(dP_i); dist3d.append(cellprob_i); bd3d.append(bd_i)

            if compute_masks:
                out = core.compute_masks(dP_i, cellprob_i, **_mask_base(dim=3, bd=bd_i))
                masks3d.append(out[0]); p3d.append(out[1]); tr3d.append(out[2])
                aff3d.append(out[3])
            else:
                masks3d.append(np.zeros(img.shape[:3], int))
                p3d.append([]); tr3d.append([]); aff3d.append([])

        flows = [list(t) for t in zip(rgb3d, dP3d, dist3d, p3d, bd3d, tr3d, aff3d)]
        return np.array(masks3d), flows, []

    # --- Wrap raw inputs into EvalSet ---
    if not is_dataset:
        def _channels_for(i):
            if channels is None:
                return None
            if (not isinstance(channels[0], (int, np.integer))
                    and len(channels) == len(x_list)
                    and isinstance(channels[i], (list, np.ndarray))
                    and len(channels[i]) == 2):
                return channels[i]
            return channels

        converted = []
        for i, raw in enumerate(x_list):
            img = imread(raw) if isinstance(raw, str) else raw
            img = transforms.convert_image(img, _channels_for(i),
                                           channel_axis=channel_axis, z_axis=z_axis,
                                           do_3D=False, normalize=False, invert=False,
                                           nchan=self.nchan, dim=self.dim, omni=omni)
            # Ensure (C, *spatial) for EvalSet
            img = img[np.newaxis] if img.ndim == self.dim else np.moveaxis(img, -1, 0)
            converted.append(img)

        x = EvalSet(converted, dim=self.dim, normalize=False, invert=False,
                    rescale_factor=rescale_factor, channel_axis=1,
                    batch_mode=batch_mode, max_batch_size=loader_batch_size)

    # --- Configure EvalSet (applies to both wrapped and user-passed datasets) ---
    x.tile           = tile
    x.normalize      = False
    x.invert         = False
    x.rescale_factor = rescale_factor
    x.batch_mode     = x._resolve_batch_mode(batch_mode)
    x.max_batch_size = loader_batch_size
    x._build_batch_plan()

    # --- Unified inference loop ---
    def _safe_array(lst):
        try:
            return np.array(lst)
        except ValueError:            # inhomogeneous shapes — return object array
            arr = np.empty(len(lst), dtype=object)
            arr[:] = lst
            return arr

    dist, dP, bd, masks, bounds, p, tr, affinity, flow_RGB = [], [], [], [], [], [], [], [], []

    _fd = lambda t: self._from_device(t.unsqueeze(0))[0]

    progress_bar = tqdm(total=len(x), disable=not show_progress)
    for batch, inds, subs in x.iter_batches():
        batch = batch.float()
        if normalize or invert:
            b = batch.numpy().transpose(0, 2, 3, 1)
            for i in range(b.shape[0]):
                b[i] = transforms.normalize_img(b[i], axis=-1, invert=invert, omni=omni)
            batch = torch.from_numpy(b.transpose(0, 3, 1, 2))

        batch = batch.to(self.device)

        # Pad channels if model expects more (e.g. grayscale into 2-channel model)
        if self.nchan > batch.shape[1]:
            pad = torch.zeros((batch.shape[0], self.nchan - batch.shape[1]) + tuple(batch.shape[2:]),
                              device=self.device, dtype=batch.dtype)
            batch = torch.cat([batch, pad], dim=1)

        with torch.no_grad():
            if tile:
                yf_batch = x._run_tiled(batch, self, batch_size=batch_size,
                                        bsize=bsize, augment=augment, tile_overlap=tile_overlap)
            else:
                yf_batch = self.run_network(batch, to_numpy=False)[0]
        del batch

        # Unpad each image to its original spatial extent, then optionally resample
        nimg = yf_batch.shape[0]
        yf_list = [yf_batch[i][(slice(None, self.nclasses), *subs[i])] for i in range(nimg)]
        del yf_batch

        if resample and rescale_factor not in (None, 1.0, 0):
            yf_list = [torch_zoom(t.unsqueeze(0), 1 / rescale_factor).squeeze(0) for t in yf_list]

        # --- Batched GPU pre-processing: hysteresis threshold + Euler integration ---
        # Enabled when all images in the batch share the same spatial shape (pad/group mode)
        # and compute_masks is requested. Passes pre-computed iscell + p into compute_masks,
        # skipping the per-image CPU follow_flows call inside it.
        _shapes = [yf_i.shape[1:] for yf_i in yf_list]
        _gpu_batch = compute_masks and len(set(_shapes)) == 1
        _pre = {}  # i → (iscell_np, p_np)

        if _gpu_batch:
            _yf    = torch.stack(yf_list)                            # (B, nclasses, *spatial)
            _dP    = _yf[:, :self.dim]                               # (B, D, *spatial)
            _dt    = _yf[:, self.dim]                                # (B, *spatial)

            # Hysteresis threshold on full batch — GPU, no scikit-image
            _iscell = hysteresis_threshold(
                _dt[:, None], mask_threshold - 1, mask_threshold
            ).squeeze(1)  # (B, *spatial) bool, on device

            # niter estimate: 2*(dim+1)*mean(dist[foreground]) across whole batch
            _dt_fg  = _dt[_iscell]
            _niter  = (int(2 * (self.dim + 1) * _dt_fg.mean().item())
                       if _dt_fg.numel() > 0 else 200)

            # Dense batched Euler integration — GPU in, GPU out
            _p_batch = follow_flows_batch(
                _dP / 5., _niter, omni=omni, suppress=False
            )  # (B, D, *spatial)

            for _i in range(nimg):
                _pre[_i] = (_iscell[_i].cpu().numpy(),
                            _p_batch[_i].cpu().numpy())
            del _yf, _iscell, _p_batch

        for i, yf_i in enumerate(yf_list):
            # GPU tensors — do GPU work first, convert to numpy only when needed
            dP_gpu   = yf_i[:self.dim]
            dist_gpu = yf_i[self.dim]
            bd_gpu   = yf_i[self.dim + 1] if self.nclasses >= self.dim + 2 else None

            flow_RGB.append(self._from_device(plot.rgb_flow(dP_gpu.unsqueeze(0), transparency=transparency))[0])

            dP_i   = _fd(dP_gpu)
            dist_i = _fd(dist_gpu)
            bd_i   = _fd(bd_gpu) if bd_gpu is not None else None

            dP.append(dP_i)
            dist.append(dist_i)
            bd.append(bd_i)

            if compute_masks:
                extra = {}
                if i in _pre:
                    extra = dict(iscell=_pre[i][0], p=_pre[i][1])
                out = core.compute_masks(
                    dP_i,
                    dist_i,
                    **_mask_base(dim=self.dim, bd=bd_i),
                    **extra,
                )
                masks.append(out[0]); p.append(out[1]); tr.append(out[2])
                bounds.append(out[3]); affinity.append(out[4])
            else:
                masks.append(np.zeros_like(dist[-1], dtype=int))
                p.append([]); tr.append([]); bounds.append([]); affinity.append([])

            progress_bar.update()
            empty_cache()

    progress_bar.close()

    # Stitch 2D masks into 3D volume when requested
    if stitch_threshold > 0 and len(masks) > 1:
        models_logger.info(f'stitching {len(masks)} planes using stitch_threshold={stitch_threshold:0.3f}')
        masks = utils.stitch3D(np.array(masks), stitch_threshold=stitch_threshold)
    else:
        masks = _safe_array(masks)

    # flows: RGB | flow components | dist | Euler coords | boundary | traces | affinity | boundary map
    flows = [list(t) for t in zip(flow_RGB, dP, dist,
                                  _safe_array(p), bd, _safe_array(tr),
                                  affinity, _safe_array(bounds))]
    return masks, flows, []


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

        mask_base = split_kwargs_for(core.compute_masks, locals(), exclude={"self", "kwargs"})
        for key in ("dP", "dist", "p", "coords", "iscell", "affinity_graph", "bd"):
            mask_base.pop(key, None)
        mask_base.update({
            "use_gpu": self.gpu,
            "device": self.device,
            "nclasses": self.nclasses,
            "dim": self.dim,
        })

        if do_3D:
            resize = None
            masks, bounds, p, tr, affinity = core.compute_masks(
                dP,
                cellprob,
                bd=bd,
                **mask_base,
            )
        else:
            masks, bounds, p, tr, affinity = [], [], [], [], []
            resize = shape[-(self.dim+1):-1] if not resample else None
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
