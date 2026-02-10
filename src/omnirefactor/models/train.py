from .imports import *
from ..kwargs import base_kwargs, split_kwargs_for
from ..core.loss import loss as core_loss



def train(self, train_data, train_labels, train_links=None, train_files=None,
          test_data=None, test_labels=None, test_links=None, test_files=None,
          channels=None, channel_axis=0, normalize=True,
          save_path=None, save_every=100, save_each=False,
          learning_rate=0.2, n_epochs=500, momentum=0.9, SGD=True,
          weight_decay=0.00001, batch_size=8, num_workers=0, nimg_per_epoch=None,
          do_rescale=True, min_train_masks=5, netstr=None, tyx=None, timing=False, do_autocast=False,
          affinity_field=False, tensorboard=False, check_grad_every=0, **kwargs):

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

        train_files: list of strings
            file names for images in train_data (to save flows for future runs)

        test_data: list of arrays (2D or 3D)
            images for testing

        test_labels: list of arrays (2D or 3D)
            See train_labels. 
    
        test_links: list of label links
            See train_links. 
        
        test_files: list of strings
            file names for images in test_data (to save flows for future runs)

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
    
    # torch.backends.cudnn.benchmark = True
    
    # rank = args.nr * args.gpus + gpu        
    # distributed.init_process_group(backend='nccl',                                         
    #                                 init_method='env://',                                   
    #                                 world_size=args.world_size,                              
    #                                 rank=rank                                               
    #                         )    
    
    if "rescale_factor" in kwargs:
        do_rescale = bool(kwargs.pop("rescale_factor"))
    if "rescale" in kwargs:
        do_rescale = bool(kwargs.pop("rescale"))

    if do_rescale:
        models_logger.info(f'Training with rescale = {do_rescale:.2f}')
    # images may need some dimension shuffling to conform to standard, this is link-independent 
    
    train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data, train_labels,  
                                                                                               test_data, test_labels,
                                                                                               channels, channel_axis,
                                                                                               normalize, 
                                                                                               self.dim, self.omni)
    
    # print('shape', train_data[0].shape, channels, train_labels[0].shape)

    # check if train_labels have flows
    # if not, flows computed, returned with labels as train_flows[i][0]
    labels_to_flows = core.labels_to_flows

    # Omnipose needs to recompute labels on-the-fly after image warping
    models_logger.info('No precomuting flows with Omnipose. Computed during training.')
    
    # We assume that if links are given, labels are properly formatted as 0,1,2,...,N
    # might be worth implementing a remapping for the links just in case...
    # for now, just skip this for any labels that come with a link file 
    for i,(labels,links) in enumerate(zip(train_labels,train_links)):
        if links is None:
            train_labels[i] = ncolor.format_labels(labels)
    
    # nmasks is inflated when using multi-label objects, so keep that in mind if you care about min_train_masks 
    nmasks = np.array([label.max() for label in train_labels])



    # if self.omni and OMNI_INSTALLED:
    #     models_logger.info('No precomuting flows with Omnipose. Computed during training.')
        
    #     # We assume that if links are given, labels are properly formatted as 0,1,2,...,N
    #     # might be worth implementing a remapping for the links just in case...
    #     # for now, just skip this for any labels that come with a link file 
    #     for i,(labels,links) in enumerate(zip(train_labels,train_links)):
    #         if links is None:
    #             train_labels[i] = utils.format_labels(labels)
        
    #     # nmasks is inflated when using multi-label objects, so keep that in mind if you care about min_train_masks 
    #     nmasks = np.array([label.max() for label in train_labels])
    # else:
    #     train_labels = labels_to_flows(labels=train_labels, links=train_links, files=train_files, 
    #                                    use_gpu=self.gpu, device=self.device, dim=self.dim)
    #     nmasks = np.array([label[0].max() for label in train_labels])

    if run_test:
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
            sym_losses.append(self.MSELoss(y_sym, y_main))
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

        # Conditional gradient check (expensive - can be disabled or made periodic)
        check_every = getattr(self, '_check_grad_every', 0)
        batch_num = getattr(self, '_current_batch', 0)
        if check_every == 0 or (batch_num % check_every == 0):
            for name, param in self.net.named_parameters():
                if param.grad is None:
                    continue
                if not torch.isfinite(param.grad).all():
                    core_logger.error("Non-finite grad detected in %s", name)
                    raise RuntimeError(f"Non-finite gradient in {name}")

        if scaler:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

    X = x.clone()
    self.optimizer.zero_grad()
    self.net.train()

    with amp_ctx():
        y, sym_loss = forward_with_symmetry(X)
        del X
        loss, raw_loss, raw_losses = core_loss(self, lbl, y, ext_loss=sym_loss)
        # print(loss,sym_loss)
        # loss = loss + symmetry_weight * sym_loss

    if not torch.isfinite(loss):
        core_logger.error("Non-finite loss detected during training step")
        raise RuntimeError("Non-finite loss during training step")

    backward_and_step(loss)

    train_loss = raw_loss.detach()
    # Store raw_losses dict for TensorBoard logging (values already detached in core_loss)
    self._last_raw_losses = raw_losses
    return train_loss


def _test_eval(self, x, lbl):
    X = self._to_device(x)
    self.net.eval()
    with torch.no_grad():
        y, style = self.net(X)
        del X
        loss, raw_loss, raw_losses = core_loss(self, lbl, y)
        test_loss = raw_loss.detach()
        test_loss *= len(x)
    return test_loss


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



def _set_criterion(self):
    # removed self.torch, self.unet if/else; all torch, need to refactor unet
    # self.MSELoss  = nn.MSELoss(reduction='mean')
    # self.BCELoss = nn.BCEWithLogitsLoss(reduction='mean')

    self.MSELoss  = metrics.loss.BatchMeanMSE()
    self.BCELoss = metrics.loss.BatchMeanBSE()
    self.SSNLoss = metrics.loss.SSL_Norm()
    self.WeightedMSE = metrics.loss.WeightedMSELoss()
    self.AffinityLoss = metrics.loss.AffinityLoss(self.device,self.dim)
    self.DerivativeLoss = metrics.loss.DerivativeLoss()
    # self.MeanAdjustedMSELoss = loss.MeanAdjustedMSELoss()
    # self.TruncatedMSELoss = loss.TruncatedMSELoss(t=10)


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
    try:
        from torch.utils.tensorboard import SummaryWriter
        self._tb_writer = SummaryWriter(log_dir)
        core_logger.info(f'>>>> TensorBoard logging enabled at {log_dir}')
        core_logger.info(f'>>>> To view: python -m tensorboard.main --logdir="{log_dir}"')
        core_logger.info(f'>>>> Open http://localhost:6006 - normalized losses overlaid at top (0_Loss)')
    except ImportError:
        core_logger.warning('TensorBoard not available. Install with: pip install tensorboard')


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
               SGD=True, batch_size=8, num_workers=0, nimg_per_epoch=None,
               do_rescale=True, affinity_field=False,
               netstr=None, do_autocast=False, tyx=None, timing=False,
               tensorboard=False, check_grad_every=0):
    """ train function uses loss function core_loss in models.py

        Additional parameters:
        tensorboard: bool (default False)
            Enable TensorBoard logging. Logs will be saved to save_path/tensorboard/
        check_grad_every: int (default 0)
            Check gradients for NaN/Inf every N batches. 0 = check every batch (default).
            Set higher (e.g., 100) to reduce overhead.
    """
    
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
    self._check_grad_every = check_grad_every

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
            
        diam_train = np.array([core.diam.diameters(train_labels[k],omni=self.omni)
                               for k in range(len(train_labels))])
        diam_train[diam_train<5] = 5.
        if test_data is not None:
            diam_test = np.array([core.diam.diameters(test_labels[k],omni=self.omni)
                                  for k in range(len(test_labels))])
            diam_test[diam_test<5] = 5.
        # scale_range = 0.5
        scale_range = 1.5 # I now want to use this as a multiplicative factor
        
        core_logger.info('>>>> median diameter set to = %d'%self.diam_mean)
    else:
        diam_train = np.ones(len(train_labels), np.float32)
        # scale_range = 1.0
        scale_range = 2.0 # I now want to use this as a multiplicative factor
        # this means that the scale will be 1/2 to 2 instead of 1/2 to 1.5

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
    
    # DataLoader-based training (single path - manual batching removed)
    # Parameters like gamma_range are not opened up here, just left to defaults
    # Some are passed through the model parameters (self.omni etc)
    kwargs = {'do_rescale': do_rescale,
              'diam_train': diam_train if do_rescale else None,
              'diam_mean': self.diam_mean,
              'tyx': tyx,
              'scale_range': scale_range,
              'omni': self.omni,
              'dim': self.dim,
              'nchan': self.nchan,
              'nclasses': self.nclasses,
              # device: use CPU if workers > 0 (cannot use CUDA on fork)
              'device': torch.device('cpu') if num_workers > 0 else self.device,
              'affinity_field': affinity_field,
              'allow_blank_masks': self.allow_blank_masks,
              'timing': timing,
             }

    torch.multiprocessing.set_start_method('fork', force=True)
    training_set = data.train.train_set(train_data, train_labels, train_links, **kwargs)

    if num_workers > 0:
        core_logger.info('>>>> Warming up dataloader transforms (compiling JIT kernels once on parent).')
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            _ = training_set[0]
        finally:
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)

    core_logger.info((">>>> Using torch dataloader. "
                      "Can take a couple min to initialize. "
                      "Using {} workers.").format(num_workers))

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

    if test_data is not None:
        print('will need to fix sampler')
        validation_set = data.train.train_set(test_data, test_labels, test_links, **kwargs)
        validation_loader = torch.utils.data.DataLoader(validation_set, **params)

    # for debugging
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%H:%M:%S')
    core_logger.info('>>>> Start time: {}'.format(formatted_time))
    
    de = 1 # epoch print interval
    epoch0 = 0
    epochtime = np.zeros(self.n_epochs)

    # for weighting function
    # blurrer = torchvision.transforms.GaussianBlur(kernel_size=11, sigma=2)

    # edge handling requires distance field for cutoffs
    # do this with batch?, turn off gradient return 
    # dist = train_labels
 
    
    for epoch in range(self.n_epochs):
        self.epoch = epoch
        if SGD:
            self._set_learning_rate(self.learning_rate[epoch])

        datatime = []
        steptime = []

        # reproducible batches - match omnipose manual path (core.py line 1312)
        np.random.seed(epoch)

        inds = []
        for batch_idx in range(steps_per_epoch):
            self._current_batch = epoch * steps_per_epoch + batch_idx
            batch_data, batch_labels, batch_inds = next(loader_iter)
            inds += batch_inds

            # log
            nbatch = len(batch_data)
            tic = time.time()
            dt = tic-toc
            toc = tic
            datatime += [dt]
            if timing:
                print('\t Dataloading time: {:.2f}'.format(dt))

            # to device - has to be done after the batch is created
            batch_data = batch_data.to(self.device, non_blocking=True)
            batch_labels = batch_labels.to(self.device, non_blocking=True)

            # update
            train_loss = self._train_step(batch_data, batch_labels)

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

                    # self.net.save_model(file_name)
                    # whether or not we are using dataparallel
                    # this logic appears elsewhere in models.py
                    if self.torch and self.gpu:
                        self.net.module.save_model(file_name)
                    else:
                        self.net.save_model(file_name)

                # Save loss history alongside model checkpoint (survives crashes)
                if self._loss_history_path is not None:
                    self._save_loss_history(self._loss_history_path)

        else:
            file_name = save_path

    # mkldnn disabled; keep torch path consistent
    self.net.mkldnn = False

    # Save final loss history (uses netstr-based path set during training)
    if self._loss_history_path is not None:
        self._save_loss_history(self._loss_history_path)

    # Close TensorBoard writer
    if self._tb_writer is not None:
        self._tb_writer.close()
        self._tb_writer = None

    # Explicitly delete DataLoader and iterator, then collect garbage to clean up workers
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
    return file_name
