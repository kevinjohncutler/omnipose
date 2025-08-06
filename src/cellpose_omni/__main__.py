#!/usr/bin/env python3
import sys, os, time, subprocess, logging

# We import only *lightweight* stuff at the top
from .models import MODEL_NAMES, C2_MODEL_NAMES, BD_MODEL_NAMES, CP_MODELS
try:
    from cellpose_omni.gui import gui
    # import cellpose_omni.gui.gui_test as gui
    
    GUI_ENABLED = True
    GUI_ERROR = None
    GUI_IMPORT = False
except ImportError as err:
    GUI_ENABLED = False
    GUI_ERROR = err
    GUI_IMPORT = True
except Exception as err:
    GUI_ENABLED = False
    GUI_ERROR = err
    GUI_IMPORT = False
    raise

from omnipose.dependencies import gui_deps

logger = logging.getLogger(__name__)

def confirm_prompt(question):
    """Simple yes/no prompt."""
    reply = None
    while reply not in ("", "y", "n"):
        reply = input(f"{question} (y/n): ").lower()
    return (reply in ("", "y"))

def install(package):
    """Pip-install the given package."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_omni(logger, omni=False):
    """If user wants Omnipose but it's not installed, offer to install it."""
    if omni and 'omnipose' not in sys.modules:
        logger.info('Omnipose features requested but not installed.')
        if confirm_prompt('Install Omnipose now?'):
            install('omnipose')
            return True
        logger.info('Omnipose not installed, turning off omni features.')
        return False
    return omni

def main(args):
    """
    Entry point for the app.  
    If len(args.dir)==0, we attempt the GUI.  
    Otherwise, we do the CLI logic (training, evaluation, etc.).
    """
    # 1) Convert tyx argument if present
    if args.tyx is not None:
        args.tyx = tuple(int(s) for s in args.tyx.split(','))
        # could make this turn a single int into a tuple depending on dim 

    # 2) If no directory given => run GUI
    if len(args.dir) == 0:
        # If the GUI is missing, offer to install
        if not GUI_ENABLED:
            print(f"GUI not available: {GUI_ERROR}")
            if GUI_IMPORT:
                print("GUI dependencies may not be installed. Prompting...")
                confirm = confirm_prompt('Install GUI dependencies? (PyQt6, etc.)')
                if confirm:
                    for dep in gui_deps:
                        install(dep)
                    subprocess.check_call([sys.executable, "-m", "omnipose"])
        else:
            # We have the GUI available—just run it
            gui.run()
            
            # gui_path = os.path.join(os.path.dirname(__file__), "gui", "gui.py")
            
            # print("🚀 Launching GUI...")
            # subprocess.Popen([sys.executable, gui_path])
            # subprocess.Popen([sys.executable, "-m", "cellpose_omni.gui.gui_test"])

        return

    # 3) Otherwise => run CLI logic
    #
    #    We do *heavy* imports here so that launching the GUI alone
    #    doesn't pay the overhead of cellpose_omni, NumPy, tqdm, etc.
    #
    import time
    import numpy as np
    from tqdm import tqdm
    from cellpose_omni import utils, models, io
    import logging
    logger = logging.getLogger(__name__)
    tqdm_out = utils.TqdmToLogger(logger, level=logging.INFO)

    # handle check_mkl
    if args.check_mkl:
        mkl_enabled = models.check_mkl((not args.mxnet))
    else:
        mkl_enabled = True

    if not args.train and (mkl_enabled and args.mkldnn):
        os.environ["MXNET_SUBGRAPH_BACKEND"] = "MKLDNN"
    else:
        os.environ["MXNET_SUBGRAPH_BACKEND"] = ""

    # If verbose => set up logging
    if args.verbose:
        from .io import logger_setup
        logger, log_file = logger_setup(args.verbose)
        print('log file', log_file)
    else:
        logger = logging.getLogger(__name__)

    # If user didn’t provide images => confirm they are sure
    if (not args.train) and (not args.train_size):
        # Are we saving anything at all? Warn if not
        saving_something = (
            args.save_png or args.save_tif or args.save_flows
            or args.save_ncolor or args.save_txt
        )
        if not (saving_something or args.testing):
            print('Running without saving any output.')
            if not confirm_prompt('Proceed anyway?'):
                sys.exit(0)

    device, gpu_available = models.assign_device(args.use_gpu, args.gpu_number)

    # Check builtin vs custom models
    builtin_model = np.any([args.pretrained_model == s for s in MODEL_NAMES])
    cellpose_model = np.any([args.pretrained_model == s for s in CP_MODELS])
    
    # cytoplasmic = ('cyto' in args.pretrained_model)
    # nuclear = ('nuclei' in args.pretrained_model)
    # bacterial = ('bact' in args.pretrained_model) or ('worm' in args.pretrained_model)

    # If model is in C2_MODEL_NAMES => force nchan=2
    if args.pretrained_model in C2_MODEL_NAMES:
        logger.info('Model uses 2 channels => setting nchan=2')
        args.nchan = 2

    # If in BD_MODEL_NAMES => nclasses=3
    if args.pretrained_model in BD_MODEL_NAMES:
        logger.info('Boundary field => setting nclasses=3')
        args.nclasses = 3

    # if nchan=2 => channels = [chan, chan2], else None
    if args.nchan == 2 and not args.all_channels:
        channels = [args.chan, args.chan2]
    else:
        channels = None

    # If 'omni' in the model name => force args.omni=True
    if args.pretrained_model is not None and 'omni' in args.pretrained_model:
        args.omni = True

    # Possibly install scikit-learn if user wants DBSCAN
    if args.cluster and 'sklearn' not in sys.modules:
        print('DBSCAN requires scikit-learn')
        if confirm_prompt('Install scikit-learn?'):
            install('scikit-learn')
        else:
            print('sklearn not installed, disabling DBSCAN.')
            args.cluster = False

    # Confirm omnipose usage or install
    omni = check_omni(logger, args.omni)
    args.omni = omni

    # If user is on mxnet + omni => bail or turn off
    if args.omni and args.mxnet:
        logger.info('Omni only in PyTorch.')
        if not confirm_prompt('Continue with --omni=False?'):
            sys.exit(0)
        logger.info('Turning off omni...')
        args.omni = False

    # Evaluate or train
    if not args.train and not args.train_size:
        # EVALUATION
        tic = time.time()

        if not builtin_model:
            # custom model path => check if it exists
            if not os.path.exists(args.pretrained_model):
                logger.warning('Provided model path not found, defaulting to cyto model')
                args.pretrained_model = 'cyto'
            else:
                logger.info(f'Using custom model {args.pretrained_model}')

        image_names = io.get_image_files(
            args.dir, 
            args.mask_filter,
            img_filter=args.img_filter,
            look_one_level_down=args.look_one_level_down
        )
        nimg = len(image_names)
        logger.info(f"Running on {nimg} image(s) using {args.nchan} channel(s).")

        # Builtin model => standard cellpose
        if builtin_model:
            # if args.mxnet:
            #     # e.g. no cyto2 in mxnet
            #     if args.pretrained_model == 'cyto2':
            #         logger.warning("cyto2 not in mxnet => using cyto")
            #         args.pretrained_model = 'cyto'
            # # create the model
            # from cellpose_omni import OMNI_MODELS
            # if args.pretrained_model in OMNI_MODELS and args.mxnet:
            #     logger.warning("Omnipose models not in mxnet => using pytorch")
            #     args.mxnet = False

            if cellpose_model:
                model = models.Cellpose(
                    gpu=args.use_gpu,
                    device=device,
                    model_type=args.pretrained_model,
                    use_torch=(not args.mxnet),
                    omni=args.omni,
                    net_avg=(not args.fast_mode and not args.no_net_avg)
                )
            else:
                cpmodel_path = models.model_path(args.pretrained_model, 0, True)
                model = models.CellposeModel(
                    gpu=args.use_gpu,
                    device=device,
                    pretrained_model=cpmodel_path,
                    use_torch=True,
                    nclasses=args.nclasses,
                    logits=args.logits,
                    nsample=args.nsample,
                    nchan=args.nchan,
                    dim=args.dim,
                    omni=args.omni,
                    net_avg=False
                )
        else:
            # custom pretrained
            if args.all_channels:
                channels = None
                
            model = models.CellposeModel(
                gpu=args.use_gpu,
                device=device,
                pretrained_model=args.pretrained_model,
                use_torch=True,
                nclasses=args.nclasses,
                logits=args.logits,
                nsample=args.nsample,
                nchan=args.nchan,
                dim=args.dim,
                omni=args.omni,
                net_avg=False,
            )
            

        # handle diameter
        if args.diameter == 0:
            # auto diameter only works for standard Cellpose models
            if cellpose_model:
                diameter = None
                logger.info('Estimating diameter for each image')
            else:
                diameter = model.diam_mean
                logger.info('Custom model => using stored diam_mean')
        else:
            diameter = args.diameter
            logger.info(f'Using diameter {diameter} for all images')

        # run
        from tqdm import tqdm
        for image_name in tqdm(image_names, file=tqdm_out):
            image = io.imread(image_name)
            out = model.eval(
                image,
                channels=channels,
                diameter=diameter,
                rescale=args.rescale,
                do_3D=args.do_3D,
                net_avg=(not args.fast_mode and not args.no_net_avg),
                augment=False,
                resample=(not args.no_resample and not args.fast_mode),
                flow_threshold=args.flow_threshold,
                mask_threshold=args.mask_threshold,
                niter=args.niter,
                diam_threshold=args.diam_threshold,
                invert=args.invert,
                batch_size=args.batch_size,
                interp=(not args.no_interp),
                cluster=args.cluster,
                suppress=(not args.no_suppress),
                channel_axis=args.channel_axis,
                z_axis=args.z_axis,
                omni=args.omni,
                affinity_seg=args.affinity_seg,
                anisotropy=args.anisotropy,
                verbose=args.verbose,
                min_size=args.min_size,
                max_size=args.max_size,
                transparency=args.transparency,
                model_loaded=True
            )
            masks, flows = out[:2]
            if len(out) > 3:
                diams = out[-1]
            else:
                diams = diameter

            if args.exclude_on_edges:
                masks = utils.remove_edge_masks(masks)

            if not args.no_npy:
                io.masks_flows_to_seg(image, masks, flows, diams, image_name, channels)

            if (
                args.save_png or args.save_tif or args.save_flows 
                or args.save_ncolor or args.save_txt
            ):
                io.save_masks(
                    image,
                    masks,
                    flows,
                    image_name,
                    png=args.save_png,
                    tif=args.save_tif,
                    save_flows=args.save_flows,
                    save_outlines=args.save_outlines,
                    save_ncolor=args.save_ncolor,
                    dir_above=args.dir_above,
                    savedir=args.savedir,
                    save_txt=args.save_txt,
                    in_folders=args.in_folders,
                    omni=args.omni,
                    channel_axis=args.channel_axis
                )
        logger.info(f'Completed in {(time.time() - tic):0.3f} sec.')

    else:
        # TRAINING or TRAIN_SIZE logic
        import numpy as np
        from cellpose_omni import utils, models, io

        builtin_model = np.any([args.pretrained_model == s for s in MODEL_NAMES])
        if builtin_model and args.mxnet and args.pretrained_model=='cyto2':
            logger.warning('cyto2 not in mxnet => using cyto')
            args.pretrained_model = 'cyto'
        if builtin_model:
            cpmodel_path = models.model_path(args.pretrained_model, 0, (not args.mxnet))
            if 'cyto' in args.pretrained_model:
                szmean = 30.
            elif 'nuclei' in args.pretrained_model:
                szmean = 17.
            elif 'bact' in args.pretrained_model or 'worm' in args.pretrained_model:
                szmean = 0.
            else:
                szmean = 30.
        else:
            # print('defining cpmodel_path')
            cpmodel_path = os.fspath(args.pretrained_model) if args.pretrained_model is not None else args.pretrained_model
            szmean = args.diameter if args.diameter else 30.

        test_dir = args.test_dir if len(args.test_dir) > 0 else None
        from .io import load_train_test_data
        output = load_train_test_data(
            args.dir, 
            test_dir=test_dir,
            image_filter=args.img_filter,
            mask_filter=args.mask_filter,
            unet=args.unet,
            look_one_level_down=args.look_one_level_down,
            omni=args.omni,
            do_links=args.links
        )
        # assign output to variables
        (images, labels, links, image_names,
         test_images, test_labels, test_links, image_names_test) = output

        # maybe figure out channel_axis vs nchan
        # ...
        # Create the model
        if args.unet:
            # lazy import for unet
            from cellpose_omni import core
            model = core.UnetModel(
                device=device,
                pretrained_model=cpmodel_path,
                diam_mean=szmean,
                residual_on=args.residual_on,
                style_on=args.style_on,
                concatenation=args.concatenation,
                nclasses=args.nclasses,
                nchan=args.nchan
            )
        else:
            # print('def model   ',cpmodel_path, isinstance(cpmodel_path, str))
            model = models.CellposeModel(
                device=device,
                gpu=args.use_gpu,
                pretrained_model=cpmodel_path,
                diam_mean=szmean,
                residual_on=args.residual_on,
                style_on=args.style_on,
                concatenation=args.concatenation,
                nchan=args.nchan,
                nclasses=args.nclasses,
                logits=args.logits,
                nsample=args.nsample,
                dim=args.dim,
                omni=args.omni,
                checkpoint=args.checkpoint,
                dropout=args.dropout,
                kernel_size=args.kernel_size,
                dilation=args.dilation,
                scale_factor=args.scale_factor,
                allow_blank_masks=args.allow_blank_masks
            )
            

        # TRAIN segmentation model
        if args.RAdam:
            logger.warning('RAdam not preferred now, SGD recommended.')
            
        if args.train:
            cpmodel_path = model.train(
                images, labels, links, train_files=image_names,
                test_data=test_images, test_labels=test_labels, test_links=test_links,
                test_files=image_names_test,
                learning_rate=args.learning_rate,
                channels=None,  # or channels if you prefer
                channel_axis=args.channel_axis,
                save_path=os.path.realpath(args.dir),
                save_every=args.save_every,
                save_each=args.save_each,
                rescale=(args.diameter != 0),
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                dataloader=args.dataloader,
                num_workers=args.num_workers,
                min_train_masks=args.min_train_masks if args.logits else 0,
                SGD=(not args.RAdam),
                tyx=args.tyx,
                timing=args.timing,
                do_autocast=args.amp,
                affinity_field=args.affinity_field
            )
            model.pretrained_model = cpmodel_path
            logger.info(f'Model trained and saved to {cpmodel_path}')

        # TRAIN size model
        if args.train_size:
            from cellpose_omni import models
            sz_model = models.SizeModel(cp_model=model, device=device)
            sz_model.train(images, labels, test_images, test_labels,
                           channels=None, batch_size=args.batch_size)
            if test_images is not None:
                predicted_diams, diams_style = sz_model.eval(test_images, channels=None)
                # Evaluate correlation
                if test_labels[0].ndim > 2:
                    tlabels = [lbl[0] for lbl in test_labels]
                else:
                    tlabels = test_labels
                import numpy as np
                real_diams = np.array([utils.diameters(lbl)[0] for lbl in tlabels])
                ccs = np.corrcoef(diams_style, real_diams)[0,1]
                cc = np.corrcoef(predicted_diams, real_diams)[0,1]
                logger.info(f'style correlation: {ccs:.4f}; final correlation: {cc:.4f}')

                # optionally save predicted diameters
                out_npy = os.path.join(args.test_dir, f'{os.path.split(cpmodel_path)[1]}_predicted_diams.npy')
                np.save(out_npy, {
                    'predicted_diams': predicted_diams,
                    'diams_style': diams_style
                })
                logger.info(f'Size model predictions saved to {out_npy}')

if __name__ == '__main__':
    main()