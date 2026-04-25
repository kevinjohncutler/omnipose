import sys

from .imports import *
from . import run_metadata


def _calibrate_data_parallel(net, batch_size, tyx, n_gpu, device, n_calib=5):
    """
    Measure 1-GPU vs DataParallel inference speed at the actual batch/image size.

    Returns (use_dp, t_1gpu_ms, t_dp_ms).  Calibration uses forward-only passes;
    the fwd/bwd speedup ratio is virtually identical so the decision transfers to training.
    """
    nchan = net.nbase[0] if hasattr(net, 'nbase') else 1
    dummy = torch.randn((batch_size, nchan) + tuple(tyx), device=device)

    net.eval()
    with torch.no_grad():
        for _ in range(3):
            net(dummy)
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_calib):
            net(dummy)
    torch.cuda.synchronize(device)
    t_1gpu = (time.perf_counter() - t0) / n_calib * 1000

    net_dp = nn.DataParallel(net, device_ids=list(range(n_gpu)))
    with torch.no_grad():
        for _ in range(3):
            net_dp(dummy)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_calib):
            net_dp(dummy)
    torch.cuda.synchronize()
    t_dp = (time.perf_counter() - t0) / n_calib * 1000

    del net_dp, dummy
    empty_cache()

    return t_dp < t_1gpu, t_1gpu, t_dp



def train(self, train_data, train_labels, train_links=None,
          test_data=None, test_labels=None, test_links=None,
          channels=None, channel_axis=0, normalize=True,
          save_path=None, save_every=100, save_each=False,
          learning_rate=0.2, n_epochs=500, momentum=0.9, SGD=True,
          weight_decay=0.00001, batch_size=8, num_workers=-1, nimg_per_epoch=None,
          do_rescale=True, min_train_masks=5, netstr=None, tyx=None, timing=False, do_autocast=False,
          affinity_field=False, tensorboard=False, sym_kernels=False,
          symmetry_weight=1.0, compile=False, **kwargs):

    """ train network with images train_data 
    
        Parameters
        ------------------

        train_data: list of arrays (2D or 3D)
            images for training

        train_labels: list of arrays (2D or 3D)
            labels for train_data, where 0=no masks; 1,2,...=mask labels
            can include flows as additional images
            
        train_links: list of label links
            These lists of label pairs define which labels are "linked",
            i.e. should be treated as part of the same object. This is how
            Omnipose handles internal/self-contact boundaries during training. 

        test_data: list of arrays (2D or 3D)
            images for testing

        test_labels: list of arrays (2D or 3D)
            See train_labels.

        test_links: list of label links
            See train_links.

        channels: list of ints (default, None)
            channels to use for training

        normalize: bool (default, True)
            normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

        save_path: string (default, None)
            where to save trained model, if None it is not saved

        save_every: int (default, 100)
            save network every [save_every] epochs

        learning_rate: float or list/np.ndarray (default, 0.2)
            learning rate for training, if list, must be same length as n_epochs

        n_epochs: int (default, 500)
            how many times to go through whole training set during training

        weight_decay: float (default, 0.00001)

        SGD: bool (default, True) 
            use SGD as optimization instead of RAdam

        batch_size: int (optional, default 8)
            number of tyx-sized patches to run simultaneously on the GPU
            (can make smaller or bigger depending on GPU memory usage)

        nimg_per_epoch: int (optional, default None)
            minimum number of images to train on per epoch, 
            with a small training set (< 8 images) it may help to set to 8

        do_rescale: bool (default, True)
            whether or not to rescale images to diam_mean during training, 
            if True it assumes you will fit a size model after training or resize your images accordingly,
            if False it will try to train the model to be scale-invariant (works worse)

        min_train_masks: int (default, 5)
            minimum number of masks an image must have to use in training set

        netstr: str (default, None)
            name of network, otherwise saved with name as params + training start time
            
        tyx: int, tuple (default, 224x224 in 2D)
            size of image patches used for training
            

    """
    
    if "rescale_factor" in kwargs:
        do_rescale = bool(kwargs.pop("rescale_factor"))
    if "rescale" in kwargs:
        do_rescale = bool(kwargs.pop("rescale"))

    if do_rescale:
        models_logger.info(f'Training with rescale = {do_rescale:.2f}')

    # Shallow-copy caller's lists so normalization and format_labels never mutate them.
    # Elements (numpy arrays) are replaced, not modified in-place, so a list copy suffices.
    train_data = list(train_data)
    train_labels = list(train_labels)

    # --- Detect lazy mode: train_data is file paths rather than arrays ---
    lazy = len(train_data) > 0 and isinstance(train_data[0], (str, os.PathLike))
    norm_params = None

    if lazy:
        # Single fast pass: compute per-image per-channel normalization params
        # (reads each file once, discards array, keeps ~2 floats/channel/image).
        models_logger.info(f'>>>> Lazy data loading: computing norm params from {len(train_data)} files...')
        norm_params = data.norm.compute_norm_params(
            train_data,
            channel_axis=channel_axis,
            channels=channels,
            normalize=normalize,
            dim=self.dim,
            omni=self.omni,
        )
        # Count masks from label files without storing the arrays
        from ..io import imread as _imread
        nmasks = np.array([_imread(p).max() for p in train_labels])
        run_test = False
    else:
        # Standard eager path: normalize all arrays in the main process
        train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(
            train_data, train_labels,
            test_data, test_labels,
            channels, channel_axis,
            normalize,
            self.dim, self.omni)

        labels_to_flows = core.labels_to_flows
        models_logger.info('No precomputing flows with Omnipose. Computed during training.')

        # format_labels for any image without a link file (lazy mode does this in workers)
        for i, (labels, links) in enumerate(zip(train_labels, train_links)):
            if links is None:
                train_labels[i] = ncolor.format_labels(labels)

        nmasks = np.array([label.max() for label in train_labels])

    if not lazy and run_test:
        test_labels = labels_to_flows(test_labels, test_links,
                                      use_gpu=self.gpu, device=self.device, dim=self.dim)
    else:
        test_labels = None

    nremove = (nmasks < min_train_masks).sum()
    if nremove > 0:
        models_logger.warning(f'{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set')
        ikeep = np.nonzero(nmasks >= min_train_masks)[0]
        train_data = [train_data[i] for i in ikeep]
        train_labels = [train_labels[i] for i in ikeep]
        train_links = [train_links[i] for i in ikeep]
        if norm_params is not None:
            norm_params = [norm_params[i] for i in ikeep]
        
    if channels is None:
        models_logger.warning('channels is set to None, input must therefore have nchan channels (default is 2)')
    train_kwargs = split_kwargs_for(self._train_net, locals(), exclude={"self", "kwargs"})
    model_path = self._train_net(**train_kwargs)
    self.pretrained_model = model_path
    return model_path



def _train_step(self, x, lbl, symmetry_weight=1):
    def sample_op():
        # sequence of single-axis flips; avoids permutations to target pure reflections
        return [tuple(-1 if i == axis else 1 for i in range(self.dim))
                for axis in range(self.dim)]

    def inverse_perm(perm):
        inv = [0] * len(perm)
        for i, p in enumerate(perm):
            inv[p] = i
        return tuple(inv)

    def apply_spatial_transform(tensor, perm, flips):
        if self.dim == 0:  # pragma: no cover
            return tensor
        total_dims = tensor.ndim
        spatial = list(range(total_dims - self.dim, total_dims))
        permuted = [spatial[i] for i in perm]
        order = list(range(total_dims - self.dim)) + permuted
        tensor = tensor.permute(order)
        flip_axes = [total_dims - self.dim + i for i, s in enumerate(flips) if s == -1]
        return torch.flip(tensor, dims=flip_axes) if flip_axes else tensor

    def undo_spatial_transform(tensor, perm, flips):
        if self.dim == 0:  # pragma: no cover
            return tensor
        total_dims = tensor.ndim
        spatial = list(range(total_dims - self.dim, total_dims))
        flip_axes = [total_dims - self.dim + i for i, s in enumerate(flips) if s == -1]
        if flip_axes:
            tensor = torch.flip(tensor, dims=flip_axes)
        inv = inverse_perm(perm)
        permuted = [spatial[i] for i in inv]
        order = list(range(total_dims - self.dim)) + permuted
        return tensor.permute(order)

    def realign_flow_channels(flow, perm, flips):
        if self.dim == 0:  # pragma: no cover
            return flow
        inv = inverse_perm(perm)
        flow = flow[:, list(inv), ...]
        for ax, s in enumerate(flips):
            if s == -1:
                flow[:, ax] = -flow[:, ax]
        return flow

    def align_prediction(pred, perm, flips):
        pred = undo_spatial_transform(pred, perm, flips)
        if self.dim:
            pred[:, :self.dim] = realign_flow_channels(pred[:, :self.dim], perm, flips)
        return pred

    def forward_with_symmetry(batch):
        y_main = self.net(batch)[0]
        if symmetry_weight <= 0 or self.dim < 1:
            return y_main, torch.zeros([], device=self.device)
        sym_losses = []
        for flips in sample_op():
            perm = tuple(range(self.dim))
            batch_sym = apply_spatial_transform(batch, perm, flips)
            y_sym = self.net(batch_sym)[0]
            y_sym = align_prediction(y_sym, perm, flips)
            # keep both branches “honest” by letting gradients flow through each, no detach on y_main
            sym_losses.append(self.MSELoss(y_sym.detach(), y_main))
        sym_loss = torch.stack(sym_losses).mean()
        return y_main, sym_loss

    amp_ctx = autocast if self.autocast else nullcontext
    scaler = self.scaler if self.autocast else None

    def backward_and_step(total_loss):
        if scaler:
            scaled = scaler.scale(total_loss)
            scaled.backward()
            scaler.unscale_(self.optimizer)
        else:
            total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

        # Check for non-finite gradients after clipping
        for p in self.net.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError("Non-finite gradient detected during training step")

        if scaler:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()
        _symmetrize_kernels(self)

    X = x.clone()
    self.optimizer.zero_grad()
    self.net.train()

    with amp_ctx():
        y, sym_loss = forward_with_symmetry(X)
        del X
        loss, raw_loss, raw_losses = core_loss(self, lbl, y, ext_loss=sym_loss)

    if not torch.isfinite(loss):
        core_logger.error("Non-finite loss detected during training step")
        raise RuntimeError("Non-finite loss during training step")

    backward_and_step(loss)

    train_loss = raw_loss.detach()
    # Store raw_losses dict for TensorBoard logging (values already detached in core_loss)
    self._last_raw_losses = raw_losses
    return train_loss



def _set_optimizer(self, learning_rate, momentum, weight_decay, SGD=False):
    if SGD:
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate,
                                    momentum=momentum, weight_decay=weight_decay)
    else:
        import torch_optimizer as optim # for RADAM optimizer
        self.optimizer = optim.RAdam(self.net.parameters(), lr=learning_rate, betas=(0.99, 0.999), #changed to .95
                                    eps=1e-08, weight_decay=weight_decay)
        core_logger.info('>>> Using RAdam optimizer')
        self.optimizer.current_lr = learning_rate


def _set_learning_rate(self, lr):
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr


def _symmetrize_kernels(self):
    if not getattr(self, "sym_kernels", False):
        return
    with torch.no_grad():
        for module in self.net.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                w = module.weight
                if w is None:
                    continue
                dims = tuple(range(2, w.ndim))
                if not dims:
                    continue
                w_sym = 0.5 * (w + w.flip(dims=dims))
                module.weight.copy_(w_sym)


def _set_criterion(self):
    self.MSELoss  = metrics.loss.BatchMeanMSE()
    self.BCELoss = metrics.loss.BatchMeanBSE()
    self.SSNLoss = metrics.loss.SSL_Norm()
    self.WeightedMSE = metrics.loss.WeightedMSELoss()
    self.AffinityLoss = metrics.loss.AffinityLoss(self.device,self.dim)
    self.DerivativeLoss = metrics.loss.DerivativeLoss()


def _init_loss_history(self):
    """Initialize loss tracking for visualization.

    Loss history is stored as a regular dict attribute on self (the model).
    PyTorch's state_dict() only serializes Parameters and Buffers, so this
    is automatically ignored during model save/load. The history is saved
    separately to a JSON file.
    """
    self.loss_history = {
        'epoch': [],
        'batch': [],
        'train_loss': [],
        'epoch_loss': [],
        'learning_rate': [],
        'timestamp': [],
        # Individual raw loss components (before scale_to_tenths)
        'raw_losses': [],
    }
    self._tb_writer = None
    self._loss_history_path = None  # Set when save_path is known
    self._last_raw_losses = None  # Populated by _train_step
    self._raw_loss_min = {}  # Min for normalization
    self._raw_loss_max = {}  # Max for normalization
    self._raw_loss_count = {}  # Count for warmup


def _log_loss(self, epoch, batch, train_loss, epoch_loss=None):
    """Log loss values to history."""
    if not hasattr(self, 'loss_history'):
        self._init_loss_history()

    self.loss_history['epoch'].append(epoch)
    self.loss_history['batch'].append(batch)
    self.loss_history['train_loss'].append(float(train_loss))
    self.loss_history['epoch_loss'].append(float(epoch_loss) if epoch_loss is not None else None)
    self.loss_history['learning_rate'].append(float(self.learning_rate[epoch]) if hasattr(self, 'learning_rate') else None)
    self.loss_history['timestamp'].append(time.time())

    # Log individual raw loss components
    raw_losses = getattr(self, '_last_raw_losses', None)
    if raw_losses is not None:
        # Convert tensors to floats for JSON serialization
        raw_losses_float = {k: float(v) for k, v in raw_losses.items()}
        self.loss_history['raw_losses'].append(raw_losses_float)
    else:
        self.loss_history['raw_losses'].append(None)

    # TensorBoard logging if enabled
    if self._tb_writer is not None:
        global_step = epoch * getattr(self, '_steps_per_epoch', 1) + batch
        self._tb_writer.add_scalar('Loss/batch', train_loss, global_step)
        if epoch_loss is not None:
            self._tb_writer.add_scalar('Loss/epoch', epoch_loss, global_step)

        # Log min-max normalized losses (0-1 scale)
        # Track min/max continuously and normalize
        if raw_losses is not None:
            for name, value in raw_losses.items():
                val_f = float(value)
                self._raw_loss_count[name] = self._raw_loss_count.get(name, 0) + 1
                # Update min/max
                if name not in self._raw_loss_min:
                    self._raw_loss_min[name] = val_f
                    self._raw_loss_max[name] = val_f
                else:
                    self._raw_loss_min[name] = min(self._raw_loss_min[name], val_f)
                    self._raw_loss_max[name] = max(self._raw_loss_max[name], val_f)

            # After warmup, start logging normalized values
            first_loss = next(iter(raw_losses.keys()))
            if self._raw_loss_count.get(first_loss, 0) >= 5:
                norm_dict = {}
                for name, value in raw_losses.items():
                    val_f = float(value)
                    lo = self._raw_loss_min[name]
                    hi = self._raw_loss_max[name]
                    rng = hi - lo
                    norm_dict[name] = (val_f - lo) / rng if rng > 1e-12 else 0.5
                self._tb_writer.add_scalars('0_Loss', norm_dict, global_step)


def _enable_tensorboard(self, log_dir):
    """Enable TensorBoard logging."""
    from torch.utils.tensorboard import SummaryWriter
    self._tb_writer = SummaryWriter(log_dir)
    core_logger.info(f'>>>> TensorBoard logging enabled at {log_dir}')
    core_logger.info(f'>>>> To view: python -m tensorboard.main --logdir="{log_dir}"')
    core_logger.info(f'>>>> Open http://localhost:6006 - normalized losses overlaid at top (0_Loss)')


def _save_loss_history(self, path):
    """Save loss history to JSON file."""
    import json
    if hasattr(self, 'loss_history'):
        with open(path, 'w') as f:
            json.dump(self.loss_history, f)
        core_logger.info(f'>>>> Loss history saved to {path}')


# Restored defaults. Need to make sure rescale is properly turned off and omni turned on when using CLI. 
# maybe replace individual 

def _train_net(self, train_data, train_labels, train_links, test_data=None, test_labels=None,
               test_links=None, save_path=None, save_every=100, save_each=False,
               learning_rate=0.2, n_epochs=500, momentum=0.9, weight_decay=0.00001,
               SGD=True, batch_size=8, num_workers=-1, nimg_per_epoch=None,
               do_rescale=True, affinity_field=False,
               netstr=None, do_autocast=False, tyx=None, timing=False,
               tensorboard=False, sym_kernels=False,
               symmetry_weight=1.0, compile=False,
               norm_params=None, channel_axis=None):
    """ train function uses loss function core_loss in models.py

        Additional parameters:
        tensorboard: bool (default False)
            Enable TensorBoard logging. Logs will be saved to save_path/tensorboard/

    """

    # Resolve num_workers: -1 means "auto" (use workers when GPU is available)
    if num_workers < 0:
        if self.device.type != 'cpu':
            num_workers = min(4, max(1, (os.cpu_count() or 4) // 2))
        else:
            num_workers = 0
        core_logger.info(f'>>>> num_workers auto-detected: {num_workers}')

    # Snapshot all resolved _train_net arguments for run.json (written later,
    # once save_path + netstr are known).
    _train_net_locals = {k: v for k, v in locals().items()
                         if k not in {'self', 'train_data', 'train_labels',
                                      'train_links', 'test_data', 'test_labels',
                                      'test_links'}}
    self._run_json_path = None

    d = datetime.datetime.now()
    self.autocast = do_autocast
    self.n_epochs = n_epochs
    if isinstance(learning_rate, (list, np.ndarray)):
        if isinstance(learning_rate, np.ndarray) and learning_rate.ndim > 1:
            raise ValueError('learning_rate.ndim must equal 1')
        elif len(learning_rate) != n_epochs:
            raise ValueError('if learning_rate given as list or np.ndarray it must have length n_epochs')
        self.learning_rate = learning_rate
        mode_val = mode(learning_rate, keepdims=True).mode
        self.learning_rate_const = float(np.ravel(mode_val)[0])
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

    # Initialize loss tracking
    self._init_loss_history()
    self.sym_kernels = bool(sym_kernels)
    self.symmetry_weight = float(symmetry_weight)

    # Enable TensorBoard if requested
    if tensorboard and save_path is not None:
        tb_dir = os.path.join(save_path, 'tensorboard')
        check_dir(tb_dir)
        self._enable_tensorboard(tb_dir)

    # Loss history path will be set after netstr is determined (first save)
    # to include model name in the filename

    nimg = len(train_data)
    
    # Set crop size; should probably generalize to fix sizes that don't fit requirements
    n = 16
    base = 2
    L = max(round(224/(base**4)),1) * (base**4)

    # Validate provided tyx: ensure each element divisible by 8
    if tyx is not None:
        if not isinstance(tyx, tuple) or not all(isinstance(v, int) for v in tyx):
            raise ValueError(f'tyx must be a tuple of ints, got {tyx}')
        if any(v % 8 != 0 for v in tyx):
            old_tyx = tyx
            tyx = tuple(((v + 7) // 8) * 8 for v in tyx)
            core_logger.warning(f'Rounded up tyx from {old_tyx} to {tyx} to ensure divisibility by 8')
    else:
        # Default tyx assignment: divisible by 8
        if self.dim == 2:
            tyx = (L, L)
        else:
            tyx = (8 * n,) * self.dim
    
    # compute average cell diameter
    if do_rescale:
        if train_links is not None:
            core_logger.warning("""WARNING: rescaling not updated for multi-label objects. 
                                Check rescaling manually for the right diameter.""")
            
        _lazy_labels = len(train_labels) > 0 and isinstance(train_labels[0], (str, os.PathLike))
        if _lazy_labels:
            from ..io import imread as _imread
            _lbl_arrays = [_imread(p) for p in train_labels]
        else:
            _lbl_arrays = train_labels
        diam_train = np.array([core.diam.diameters(_lbl_arrays[k], omni=self.omni)
                               for k in range(len(_lbl_arrays))])
        diam_train[diam_train<5] = 5.
        if test_data is not None:
            diam_test = np.array([core.diam.diameters(test_labels[k],omni=self.omni)
                                  for k in range(len(test_labels))])
            diam_test[diam_test<5] = 5.
        scale_range = 1.5

        core_logger.info('>>>> median diameter set to = %d'%self.diam_mean)
    else:
        diam_train = np.ones(len(train_labels), np.float32)
        scale_range = 2.0

    _lazy_data = len(train_data) > 0 and isinstance(train_data[0], (str, os.PathLike))
    if _lazy_data:
        # Infer nchan from norm_params (one entry per channel per image)
        nchan = len(norm_params[0]) if norm_params else 1
    else:
        nchan = train_data[0].shape[0]
    core_logger.info('>>>> training network with %d channel input <<<<'%nchan)
    core_logger.info('>>>> LR: %0.5f, batch_size: %d, weight_decay: %0.5f'%(self.learning_rate_const, 
                                                                            self.batch_size, weight_decay))
    
    if test_data is not None:
        core_logger.info(f'>>>> ntrain = {nimg}, ntest = {len(test_data)}')
    else:
        core_logger.info(f'>>>> ntrain = {nimg}')
    
    # ------------------------------------------------------------------ #
    # Optional: compile model with torch.compile (persists after training)
    # ------------------------------------------------------------------ #
    if compile:
        # Point Inductor cache to a persistent directory so compiled Triton
        # kernels survive reboots. The cache is keyed on graph hash + tensor
        # shapes + PyTorch version + GPU model, so stale entries are safe.
        _cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'inductor')
        os.makedirs(_cache_dir, exist_ok=True)
        os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', _cache_dir)

        models_logger.info(
            '>>>> Compiling model with torch.compile. '
            'First run compiles Triton kernels (~60s); cached to %s for reuse. '
            'Expect 1.3-1.5x speedup once warmed up.', _cache_dir
        )
        self.net = torch.compile(self.net)

    # ------------------------------------------------------------------ #
    # Auto-calibrate DataParallel: only engage if it helps at this config
    # ------------------------------------------------------------------ #
    _unwrap_dp_after_training = False
    if getattr(self, 'n_gpu', 1) > 1 and not isinstance(self.net, nn.DataParallel):
        models_logger.info(
            f'>>>> Calibrating DataParallel ({self.n_gpu} GPUs) at batch={batch_size}, tyx={tyx}...'
        )
        _calib_net = self.net._orig_mod if hasattr(self.net, '_orig_mod') else self.net
        use_dp, t1, tdp = _calibrate_data_parallel(
            _calib_net, batch_size, tyx, self.n_gpu, self.device
        )
        if use_dp:
            models_logger.info(
                f'>>>> DataParallel engaged: 1-GPU {t1:.1f}ms → {self.n_gpu}-GPU {tdp:.1f}ms per batch'
                f' ({t1/tdp:.2f}x speedup)'
            )
            self.net = nn.DataParallel(self.net, device_ids=list(range(self.n_gpu)))
            _unwrap_dp_after_training = True
        else:
            px = tyx[0] if tyx else 224
            be_batch = int(round(57 / (t1 / batch_size) * self.n_gpu / (self.n_gpu - 1)))
            models_logger.info(
                f'>>>> DataParallel skipped: 1-GPU {t1:.1f}ms < {self.n_gpu}-GPU {tdp:.1f}ms at '
                f'batch={batch_size}, tyx={tyx}. Break-even ≈ batch {be_batch}.'
            )

    t0 = time.time()
    toc = t0
    lsum, nsum = 0, 0

    if save_path is not None:
        _, file_label = os.path.split(save_path)
        file_path = os.path.join(save_path, 'models/')
        check_dir(file_path)
    else:
        core_logger.warning('WARNING: no save_path given, model not saving')

    ksave = 0
    rsc = 1.0 # initialize, redefiled below

    # cannot train with mkldnn
    self.net.mkldnn = False

    # Log nimg_per_epoch (DataLoader handles shuffling via CyclingRandomBatchSampler)
    if nimg_per_epoch is None or nimg > nimg_per_epoch:
        nimg_per_epoch = nimg
    core_logger.info(f'>>>> nimg_per_epoch = {nimg_per_epoch}')
            
    if self.autocast:
        self.scaler = GradScaler()
    
    # DataLoader returns raw augmented arrays; flows are always computed on GPU in the training loop.
    kwargs = {'do_rescale': do_rescale,
              'diam_train': diam_train if do_rescale else None,
              'diam_mean': self.diam_mean,
              'tyx': tyx,
              'scale_range': scale_range,
              'omni': self.omni,
              'dim': self.dim,
              'nchan': self.nchan,
              'nclasses': self.nclasses,
              'device': self.device,
              'affinity_field': affinity_field,
              'allow_blank_masks': self.allow_blank_masks,
              'timing': timing,
             }

    if num_workers > 0:
        # spawn is CUDA-safe; fork can deadlock when CUDA is initialized
        torch.multiprocessing.set_start_method('spawn', force=True)

    # Lazy mode: train_data / train_labels are file path lists.
    # Workers load + normalize on demand — zero main-process RAM for image data.
    _lazy = norm_params is not None or (
        len(train_data) > 0 and isinstance(train_data[0], (str, os.PathLike))
    )

    _shm_pools = []
    if _lazy:
        # Pass paths + precomputed norm params; workers do the rest.
        kwargs['image_paths'] = list(train_data)
        kwargs['label_paths'] = list(train_labels)
        kwargs['norm_params'] = norm_params
        kwargs['_channel_axis'] = channel_axis
        # train_set still needs a length; pass empty lists as data placeholders.
        _data_placeholder = [None] * len(train_data)
        _labels_placeholder = [None] * len(train_labels)
        training_set = data.train.train_set(_data_placeholder, _labels_placeholder, train_links, **kwargs)
        core_logger.info(f'>>>> Lazy loading: {len(train_data)} image files, zero main-process RAM.')
    elif num_workers > 0:
        # Eager arrays, multi-worker: pack into shared memory pools.
        # Uses only 2 file descriptors total; spawn workers attach by name.
        from ..data.train import _ShmPool
        data_pool = _ShmPool(train_data)
        label_pool = _ShmPool(train_labels)
        _shm_pools = [data_pool, label_pool]
        kwargs['data_pool'] = data_pool
        kwargs['label_pool'] = label_pool
        total_mb = (data_pool._shm.size + label_pool._shm.size) / 1024**2
        core_logger.info(f'>>>> Shared memory: {len(train_data)} images packed into 2 segments ({total_mb:.1f} MB, zero-copy).')
        training_set = data.train.train_set(train_data, train_labels, train_links, **kwargs)
    else:
        training_set = data.train.train_set(train_data, train_labels, train_links, **kwargs)

    core_logger.info(">>>> Using torch dataloader with {} worker(s). Flows computed on GPU in main process.".format(num_workers))

    # CyclingRandomBatchSampler pre-generates all indices using np.random.seed(0)
    # to match the omnipose manual batching path when num_workers=0
    batch_sampler = data.train.CyclingRandomBatchSampler(
        training_set,
        batch_size=batch_size,
        n_epochs=n_epochs,
        nimg_per_epoch=nimg_per_epoch,
    )
    params = dict(
        batch_sampler=batch_sampler,
        collate_fn=training_set.collate_fn,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        persistent_workers=num_workers > 0,
        worker_init_fn=training_set.worker_init_fn,
    )
    if num_workers > 0:
        params["prefetch_factor"] = 8

    train_loader = torch.utils.data.DataLoader(training_set, **params)

    steps_per_epoch = len(batch_sampler)
    self._steps_per_epoch = steps_per_epoch  # Store for TensorBoard global step
    loader_iter = iter(train_loader)

    # for debugging
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%H:%M:%S')
    core_logger.info('>>>> Start time: {}'.format(formatted_time))
    
    de = 1 # epoch print interval
    epoch0 = 0
    epochtime = np.zeros(self.n_epochs)

    try:
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            if SGD:
                self._set_learning_rate(self.learning_rate[epoch])

            datatime = []
            steptime = []

            # reproducible batches - match omnipose manual path (core.py line 1312)
            np.random.seed(epoch)
            torch.manual_seed(epoch)   # seeds torch augmentation path

            inds = []
            for batch_idx in range(steps_per_epoch):
                # imgi: GPU tensor (num_workers=0) or numpy (num_workers>0 CPU aug path)
                # masks: always numpy (needed by concatenate_labels in masks_to_flows_batch)
                imgi, masks, links_batch, batch_inds = next(loader_iter)
                inds += batch_inds
                nbatch = len(imgi)
                tic = time.time()
                dt = tic - toc
                toc = tic
                datatime += [dt]
                if timing:
                    print('\t Dataloading time: {:.2f}'.format(dt))
                batch_data, batch_labels = training_set.compute_flows_gpu(
                    imgi, masks, links_batch, self.device)

                # update
                train_loss = self._train_step(batch_data, batch_labels, symmetry_weight=self.symmetry_weight)

                # log
                dt = time.time()-tic
                steptime += [dt]
                if timing:
                    print('\t Step time: {:.2f}, batch size {}'.format(dt,nbatch))
                    print('batch inds', batch_inds)

                lsum += train_loss
                nsum += 1

                # Log batch loss
                self._log_loss(epoch, nsum, train_loss)

            # print('inds',sorted(inds), np.unique(np.diff(np.array(sorted(inds)))))
            tic = time.time()
            epochtime[epoch] = tic
            if epoch % de == 0:
                core_logger.info(
                    ("Train epoch: {} | "
                    "Time: {:.2f}min | "
                    "last epoch: {:.2f}s | "
                     # "batch size: {} | "
                    "<sec/epoch>: {:.2f}s | "
                    "<sec/batch>: {:.2f}s | "
                    "<Last batch Loss>: {:.6f} | "
                    "<Epoch Loss>: {:.6f}").
                    strip().
                    format(epoch, # epoch index
                           (tic-t0) / 60, # absolute time spent in minutes
                           0 if epoch==epoch0 else (tic-toc) / (epoch-epoch0), # time spent in this epoch
                           # nbatch,
                           # (tic-t0) / (epoch+1), # average time per epoch
                           0 if epoch==epoch0 else (
                               np.mean(np.diff(epochtime[max(0,epoch-3):max(1,epoch)]))
                               if len(epochtime[max(0,epoch-3):max(1,epoch)]) > 1 else 0.0
                           ),
                           np.mean(datatime[-3:]) if datatime else 0.0, # average time spent loading fata for last 3 epochs
                           train_loss, #/batch_size, # average loss over this batch
                           lsum/nsum) # average loss over this epoch
                )
            epoch0 = epoch
            toc = time.time() # end of batch

            # Compute and log epoch-level loss
            epoch_avg_loss = lsum / nsum if nsum > 0 else 0
            if self._tb_writer is not None:
                self._tb_writer.add_scalar('Loss/epoch_avg', epoch_avg_loss, epoch)
                self._tb_writer.add_scalar('LearningRate', self.learning_rate[epoch], epoch)

            lsum, nsum = 0, 0

            if save_path is not None:
                in_final = (self.n_epochs-epoch)<10
                if netstr is None:
                    netstr =  '{}_{}_{}'.format(self.net_type, file_label,
                                                   d.strftime("%Y_%m_%d_%H_%M_%S.%f"))
                # Set loss history path once netstr is known (user-provided or generated)
                if self._loss_history_path is None:
                    self._loss_history_path = os.path.join(save_path, f'{netstr}_loss_history.json')
                    self._run_json_path = run_metadata.run_json_path_for(self._loss_history_path)
                    # Write initial run.json (status=running) now that all params are resolved
                    run_payload = run_metadata.capture_run_metadata(
                        self, self.net, _train_net_locals,
                        train_data, train_labels,
                        test_data, test_labels,
                        save_path=save_path, netstr=netstr,
                        name=getattr(self, '_run_name', None),
                        sweep=getattr(self, '_run_sweep', None),
                        tags=getattr(self, '_run_tags', None),
                        notes=getattr(self, '_run_notes', None),
                    )
                    run_metadata.write_run_json(self._run_json_path, run_payload)

                base = netstr+'{}'
                if epoch==self.n_epochs-1 or epoch%save_every==0 or in_final:

                    suffixes = ['']
                    if save_each or in_final:
                        # I will want to add a toggle for this
                        suffixes+=['_epoch_'+str(epoch)]

                    for s in suffixes:
                        file_name = base.format(s)

                        file_name = os.path.join(file_path, file_name)
                        ksave += 1
                        core_logger.info(f'saving network parameters to file://{file_name}')

                        if isinstance(self.net, nn.DataParallel):
                            self.net.module.save_model(file_name)
                        else:
                            self.net.save_model(file_name)

                    # Save loss history alongside model checkpoint (survives crashes)
                    if self._loss_history_path is not None:
                        self._save_loss_history(self._loss_history_path)

            else:
                file_name = save_path

    finally:
        # Runs on both normal exit and exception — ensures workers and shm are always released.
        self.net.mkldnn = False

        if self._loss_history_path is not None:
            self._save_loss_history(self._loss_history_path)

        # Finalize run.json: stamp status + duration + summary
        if getattr(self, '_run_json_path', None) is not None:
            exc_info = sys.exc_info()
            status = 'failed' if exc_info[0] is not None else 'completed'
            try:
                run_metadata.mark_run_finished(
                    self._run_json_path,
                    self.loss_history if hasattr(self, 'loss_history') else {},
                    status=status,
                )
            except Exception as _exc:  # noqa: BLE001 — don't mask training error
                core_logger.warning(f'Failed to finalize run.json: {_exc}')

        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None

        # Delete DataLoader and iterator first so workers stop touching shm before we unlink.
        try:
            del train_loader
        except Exception:
            pass
        try:
            del loader_iter
        except Exception:
            pass
        import gc
        gc.collect()

        for pool in _shm_pools:
            pool.close()
        for pool in _shm_pools:
            pool.unlink()

    # Unwrap DataParallel so eval/save use the plain net
    if _unwrap_dp_after_training:
        self.net = self.net.module
        models_logger.info('>>>> DataParallel unwrapped after training.')

    # Unwrap torch.compile after training. Eval sees variable image sizes
    # which would trigger ~60s recompilations per unique shape — not worth it
    # for typical usage. Users who want compiled eval can re-wrap manually:
    #   model.net = torch.compile(model.net)
    if compile and hasattr(self.net, '_orig_mod'):
        self.net = self.net._orig_mod
        models_logger.info('>>>> torch.compile unwrapped after training (eval uses eager).')

    return file_name
