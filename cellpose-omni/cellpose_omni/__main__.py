import sys, os, glob, pathlib, time
import subprocess
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from cellpose_omni import utils, models, io

from .models import MODEL_NAMES, C2_MODEL_NAMES, BD_MODEL_NAMES, CP_MODELS

import torch

try:
    from cellpose_omni.gui import gui 
    GUI_ENABLED = True 
except ImportError as err:
    GUI_ERROR = err
    GUI_ENABLED = False
    GUI_IMPORT = True
except Exception as err:
    GUI_ENABLED = False
    GUI_ERROR = err
    GUI_IMPORT = False
    raise
    
import logging
logger = logging.getLogger(__name__)

def confirm_prompt(question):
    reply = None
    while reply not in ("", "y", "n"):
        reply = input(f"{question} (y/n): ").lower()
    return (reply in ("", "y"))

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
def check_omni(logger,omni=False):
    if omni and not 'omnipose' not in sys.modules:
        logger.info('Omnipose features requested but not installed.')
        confirm = confirm_prompt('Install Omnipose?')
        if confirm:
            install('omnipose')
        else:
            logger.info('Omnipose not installed. Running with omni=False')
        return confirm
    


# settings re-grouped a bit
# added omni as a parameter
def main(args):

    # convert the tyx string to a vector
    if args.tyx is not None:
        args.tyx = tuple([int(s) for s in (args.tyx.split(','))])
    
    # handle mxnet option 
    if args.check_mkl:
        mkl_enabled = models.check_mkl((not args.mxnet))
    else:
        mkl_enabled = True

    if not args.train and (mkl_enabled and args.mkldnn):
        os.environ["MXNET_SUBGRAPH_BACKEND"]="MKLDNN"
    else:
        os.environ["MXNET_SUBGRAPH_BACKEND"]=""
    
    if len(args.dir)==0:
        # previously, we would just ask users to run the install command for the GUI
        # In omnipose, the GUI is so useful for debugging that I enforced it.
        # However, PyQt6 can cause enough issues within existing environments that I have reconsidered...
        # Compromise: if users request the GUI with 'python -m omnipose', this prompt will install it for them 
        if not GUI_ENABLED:
            print('GUI ERROR: %s'%GUI_ERROR)
            if GUI_IMPORT:
                print('GUI dependencies may not be installed (normal for first run).')
                confirm = confirm_prompt('Install GUI dependencies? (Note: uses PyQt6.)')
                if confirm:
                    current_dir = os.path.abspath(os.path.dirname(__file__))
                    cellpose_omni_path = os.path.dirname(current_dir)
                    print('cellpose_omni_path',cellpose_omni_path)
                    next_command = ['&&', 'omnipose'] # run omnipose again
                    install('cellpose-omni[gui] @ file://{}#egg=cellpose-omni'.format(cellpose_omni_path)) # local version)       
                    subprocess.check_call([sys.executable, "-m", "omnipose"])
                                                             
        else:
            print('heyyyo\n')
            gui.run()

    else:
        if args.verbose:
            from .io import logger_setup
            logger, log_file = logger_setup(args.verbose)
            print('log file',log_file)
        else:
            print('!NEW LOGGING SETUP! To see cellpose progress, set --verbose')
            print('No --verbose => no progress or info printed')
            logger = logging.getLogger(__name__)

        use_gpu = False

        # find images
        if len(args.img_filter)>0:
            img_filter = args.img_filter
        else:
            img_filter = ''


        # Check with user if they REALLY mean to run without saving anything 
        if not (args.train or args.train_size):
            saving_something = args.save_png or args.save_tif or args.save_flows or args.save_ncolor or args.save_txt
            if not (saving_something or args.testing): 
                print('Running without saving any output.')
                confirm = confirm_prompt('Proceed Anyway?')
                if not confirm:
                    exit()
                    
        device, gpu = models.assign_device((not args.mxnet), args.use_gpu)
        
        #define available model names, right now we have three broad categories 
        builtin_model = np.any([args.pretrained_model==s for s in MODEL_NAMES])
        cellpose_model = np.any([args.pretrained_model==s for s in CP_MODELS])
        cytoplasmic = 'cyto' in args.pretrained_model
        nuclear = 'nuclei' in args.pretrained_model
        #inelegant but necessary workaround for models that I provide without multiple models
        # long term should just check to see if they exist locally or on the server, disable model averaging if not found 
        bacterial = ('bact' in args.pretrained_model) or ('worm' in args.pretrained_model) 

        # set nchan for builtin models
        # print('hey',args.pretrained_model,args.pretrained_model in C2_MODEL_NAMES, args.chan, args.chan2,args.nchan)
        
        if args.pretrained_model in C2_MODEL_NAMES:
            if args.nchan is not None:
                logger.info('This pretrained model uses 2 channels, setting nchan=2')
            args.nchan = 2
        
        # Pretrained omni models originally had 3 prediction classes 
        if args.pretrained_model in BD_MODEL_NAMES:
            logger.info('This model uses boundary field, setting nclasses=3.')
            args.nclasses = 3

        # Handle channel assignemnt for 2 vs 1 channels
        # For >2 channels, use None. 
        if args.nchan is not None and args.nchan>1:
            channels = [args.chan, args.chan2]
        else:
            channels = None

        # print('ddd', args.nchan, channels)

        # force omni on for those models, but don't toggle it off if manually specified via --omni or by invoking python -m omnipose
        if 'omni' in args.pretrained_model:
            args.omni = True
        
        # should revisit this for affiity graph segmentation 
        if args.cluster and 'sklearn' not in sys.modules:
            print('DBSCAN clustering requires scikit-learn.')
            confirm = confirm_prompt('Install scikit-learn?')
            if confirm:
                install('scikit-learn')
            else:
                print('scikit-learn not installed. DBSCAN clustering will be automatically disabled.')
                          
        omni = check_omni(args.omni) # repeat the above check but factor it for use elsewhere

        # omni changes not implemented for mxnet. Full parity for cpu/gpu in pytorch. 
        if args.omni and args.mxnet:
            logger.info('omni only implemented in pytorch.')
            confirm = confirm_prompt('Continue with omni set to false?')
            if not confirm:
                exit()
            else:
                logger.info('omni set to false.')
                args.omni = False

        if args.omni and args.train:
            # assume instance segmentation unless otherwise specified 
            if args.nclasses is None:
                args.nclasses = 2 # now do *not* use boundary prediction by default 
                
            # args.dropout = True
            # args.RAdam = True
            # args.RAdam = False
            
            logger.info('Training omni model. Setting nclasses={}, RAdam={}'.format(args.nclasses,args.RAdam))


        # EVALUATION BRANCH
        if not args.train and not args.train_size:
            tic = time.time()
            if not builtin_model:
                cpmodel_path = args.pretrained_model
                if not os.path.exists(cpmodel_path):
                    logger.warning('model path does not exist, using cyto model')
                    args.pretrained_model = 'cyto'
                else:
                    logger.info(f'running model {cpmodel_path}')

            image_names = io.get_image_files(args.dir, 
                                             args.mask_filter, 
                                             img_filter=img_filter,
                                             look_one_level_down=args.look_one_level_down)
            nimg = len(image_names)
                
            logger.info('running omnipose on {} image(s) using {} channel(s).'.format(nimg, args.nchan))
            if channels is not None:
                cstr0 = ['MONO', 'RED', 'GREEN', 'BLUE']
                cstr1 = ['NONE', 'RED', 'GREEN', 'BLUE']
                logger.info('channel(s) to seg: {} and  {}'.format(cstr0[channels[0]], cstr1[channels[1]]))

            if args.omni:
                logger.info(f'omni is ON, cluster is {args.cluster}')
             
            # handle built-in model exceptions
            if builtin_model:
                if args.mxnet:
                    if args.pretrained_model=='cyto2':
                        logger.warning('cyto2 model not available in mxnet, using cyto model')
                        args.pretrained_model = 'cyto'
                    if args.pretrained_model in OMNI_MODELS:
                        logger.warning('omnipose models not available in mxnet, using pytorch')
                        args.mxnet = False
                if cellpose_model: # ones with a size model, also never true 3d etc.                 
                    model = models.Cellpose(gpu=gpu, device=device, model_type=args.pretrained_model, 
                                            use_torch=(not args.mxnet), omni=args.omni, 
                                            net_avg=(not args.fast_mode and not args.no_net_avg))
                else:
                    cpmodel_path = models.model_path(args.pretrained_model, 0, True)
                    model = models.CellposeModel(gpu=gpu, device=device, 
                                                 pretrained_model=cpmodel_path,
                                                 use_torch=True,
                                                 nclasses=args.nclasses, 
                                                 nchan=args.nchan,
                                                 dim=args.dim, 
                                                 omni=args.omni,
                                                 net_avg=False)
            else:
                if args.all_channels:
                    channels = None  
                model = models.CellposeModel(gpu=gpu, device=device, 
                                             pretrained_model=cpmodel_path,
                                             use_torch=True,
                                             nclasses=args.nclasses, 
                                             nchan=args.nchan, 
                                             dim=args.dim, 
                                             omni=args.omni,
                                             net_avg=False)
            

            # handle diameters
            if args.diameter==0:
                # if builtin_model:
                if cellpose_model:
                    diameter = None
                    logger.info('estimating diameter for each image')
                else:
                    logger.info('using user-specified model, no auto-diameter estimation available')
                    diameter = model.diam_mean
            else:
                diameter = args.diameter
                logger.info('using diameter %0.2f for all images'%diameter)
            
            tqdm_out = utils.TqdmToLogger(logger,level=logging.INFO)
            
            # currently, CLI eval forces the network to run on just one image at a time
            # We can change this if images are all the same size by running several as a batch
            # (this assumes that several full images will already fit on the GPU)
            # This means that this loop should probably be a dataloader. Before doing that,
            # the eval function itself needs to be updated to accept several images 
            # it then needs to run those as a batch, stitch the output for postprocessing, and parse it back out
            for image_name in tqdm(image_names, file=tqdm_out):
                image = io.imread(image_name)
                out = model.eval(image, channels=channels, diameter=diameter, rescale = args.rescale,
                                do_3D=args.do_3D, net_avg=(not args.fast_mode and not args.no_net_avg),
                                augment=False,
                                resample=(not args.no_resample and not args.fast_mode),
                                flow_threshold=args.flow_threshold,
                                mask_threshold=args.mask_threshold,
                                niter = args.niter,
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
                                transparency=args.transparency, # RGB flows made in the eval step
                                model_loaded=True)
                masks, flows = out[:2]
                if len(out) > 3:
                    diams = out[-1]
                else:
                    diams = diameter
                if args.exclude_on_edges:
                    masks = utils.remove_edge_masks(masks)
                if not args.no_npy:
                    io.masks_flows_to_seg(image, masks, flows, diams, image_name, channels)
                if saving_something:
                    io.save_masks(image, masks, flows, image_name, png=args.save_png, tif=args.save_tif,
                                  save_flows=args.save_flows,save_outlines=args.save_outlines,
                                  save_ncolor=args.save_ncolor,dir_above=args.dir_above,savedir=args.savedir,
                                  save_txt=args.save_txt,in_folders=args.in_folders, omni=args.omni, channel_axis=args.channel_axis)
            logger.info('completed in %0.3f sec'%(time.time()-tic))
            
        # TRAINING BRANCH    
        else:
            if builtin_model:
                if args.mxnet and args.pretrained_model=='cyto2':
                    logger.warning('cyto2 model not available in mxnet, using cyto model')
                    args.pretrained_model = 'cyto'
                cpmodel_path = models.model_path(args.pretrained_model, 0, not args.mxnet)
                if cytoplasmic:
                    szmean = 30.
                elif nuclear:
                    szmean = 17.
                elif bacterial:
                    szmean = 0. #bacterial models are not rescaled 
            else:
                cpmodel_path = os.fspath(args.pretrained_model)
                szmean = args.diameter # respect user defined, defaults to 30
                
            test_dir = None if len(args.test_dir)==0 else args.test_dir
            output = io.load_train_test_data(args.dir, test_dir, img_filter, args.mask_filter, 
                                             args.unet, args.look_one_level_down, args.omni, args.links)
            images, labels, links, image_names, test_images, test_labels, test_links, image_names_test = output

            
            # I see no reason to keep the 2-channel default of cellpose
            # omnipose will just support training on homogeneous image shapes, which is handled by setting channels=None
            # channels = None        
            
            img = images[0]  
            dim = img.ndim 
            shape = img.shape
            # training with all channels
            # if args.all_channels:
            if args.channel_axis is None:
                if args.dim != dim: # user dim allows us to discern ND from (N-1)D+C
                    nchan = min(shape) # This assumes that the channel axis is the smallest 
                    args.channel_axis = np.argwhere([s==nchan for s in shape])[0][0]
                    logger.info('channel axis detected at position %s, manually specify if incorrect'%args.channel_axis)
                    nchan = shape[args.channel_axis]
                elif args.nchan is None:
                    nchan = 1
                    args.channel_axis = 0 
                else:
                    nchan = args.nchan
            
            rstr = 'Be sure to use --nchan {} when running the model.'.format(nchan)
            if args.nchan is None:
                logger.info('setting nchan to {}. '.format(nchan) + rstr)
            elif args.nchan != nchan:
                logger.warning('provided nchan {} does not match {} data channels. Using the latter. '.format(args.nchan,nchan) + rstr)
            
            args.nchan = nchan
            
            # model path
            if not os.path.exists(cpmodel_path):
                if not args.train:
                    error_message = 'ERROR: model path missing or incorrect - cannot train size model'
                    logger.critical(error_message)
                    raise ValueError(error_message)
                cpmodel_path = False
                logger.info('training from scratch')
            else:
                # all of this assumes Cellpose
                args.diameter = szmean 
                logger.info('pretrained model %s is being used'%cpmodel_path)
                args.residual_on = 1
                args.style_on = 1
                args.concatenation = 0
                
            # previously, only training from scratch allowed you to change diameter/szmean
            # it also enforced 
            if args.diameter==0:
                rescale = False 
                logger.info('median diameter set to 0 => no rescaling during training')
            else:
                rescale = True         

            if rescale and args.train:
                logger.info('during training rescaling images to fixed diameter of %0.1f pixels'%args.diameter)
                
            # initialize model
            import torch
            # torch.use_deterministic_algorithms(True) need to set envoronment variable for this 
            torch.manual_seed(42)
            # torch.backends.cudnn.benchmark = False # slower somehow with True
            
            if args.unet:
                model = core.UnetModel(device=device,
                                        pretrained_model=cpmodel_path, 
                                        diam_mean=szmean,
                                        residual_on=args.residual_on,
                                        style_on=args.style_on,
                                        concatenation=args.concatenation,
                                        nclasses=args.nclasses,
                                        nchan=args.nchan)
            else:
                model = models.CellposeModel(device=device,
                                             gpu=gpu, # why was this not being passed in before?
                                             use_torch=(not args.mxnet),
                                             pretrained_model=cpmodel_path,
                                             diam_mean=szmean,
                                             residual_on=args.residual_on,
                                             style_on=args.style_on,
                                             concatenation=args.concatenation,
                                             nchan=args.nchan,
                                             nclasses=args.nclasses,
                                             dim=args.dim, # init to 2D pr 3D
                                             omni=args.omni,
                                             checkpoint=args.checkpoint,
                                             dropout=args.dropout,
                                             kernel_size=args.kernel_size) 
            
            # allow multiple GPUs, maybe wrap in test to see if there are multiple GPUs
            # model = nn.DataParallel(model)
            
            # train segmentation model
            if args.train:
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                # print('imagesize',images[0].shape)
                cpmodel_path = model.train(images, labels, links, train_files=image_names,
                                           test_data=test_images, 
                                           test_labels=test_labels, 
                                           test_links=test_links, 
                                           test_files=image_names_test,
                                           learning_rate=args.learning_rate, 
                                           channels=channels,
                                           channel_axis=args.channel_axis,
                                           save_path=os.path.realpath(args.dir), 
                                           save_every=args.save_every,
                                           save_each=args.save_each,
                                           rescale=rescale,
                                           n_epochs=args.n_epochs,
                                           batch_size=args.batch_size, 
                                           dataloader=args.dataloader,
                                           num_workers=args.num_workers,
                                           min_train_masks=args.min_train_masks,
                                           SGD=(not args.RAdam),
                                           tyx=args.tyx,
                                           timing=args.timing,
                                           do_autocast=args.amp,
                                           affinity_field=args.affinity_field)
                # print(prof)
                model.pretrained_model = cpmodel_path
                logger.info('model trained and saved to %s'%cpmodel_path)

            # train size model
            if args.train_size:
                sz_model = models.SizeModel(cp_model=model, device=device)
                sz_model.train(images, labels, test_images, test_labels, channels=channels, batch_size=args.batch_size)
                if test_images is not None:
                    predicted_diams, diams_style = sz_model.eval(test_images, channels=channels)
                    if test_labels[0].ndim>2:
                        tlabels = [lbl[0] for lbl in test_labels]
                    else:
                        tlabels = test_labels 
                    ccs = np.corrcoef(diams_style, np.array([utils.diameters(lbl)[0] for lbl in tlabels]))[0,1]
                    cc = np.corrcoef(predicted_diams, np.array([utils.diameters(lbl)[0] for lbl in tlabels]))[0,1]
                    logger.info('style test correlation: %0.4f; final test correlation: %0.4f'%(ccs,cc))
                    np.save(os.path.join(args.test_dir, '%s_predicted_diams.npy'%os.path.split(cpmodel_path)[1]), 
                            {'predicted_diams': predicted_diams, 'diams_style': diams_style})

if __name__ == '__main__':
    main()
    
