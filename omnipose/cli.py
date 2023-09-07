import argparse

class CustomArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        formatter = self._get_formatter()
        # add help for all arguments
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()
        # print the help message
        self._print_message(formatter.format_help(), file)

class CustomHelpFormatter(argparse.HelpFormatter):
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = 'usage: '
        # get the program name
        prog = self._prog
        # get the group names
        group_names = [f'[{group.title}]' for group in groups]
        # format the usage string
        usage_str = f'{prefix}{prog} {" ".join(group_names)}\n'
        return usage_str

    
def get_arg_parser():
    """ Parses command line arguments for cellpose_omni main function
    Note: this function has to be in a separate file to allow autodoc to work for CLI.
    The autodoc_mock_imports in conf.py does not work for sphinx-argparse sometimes,
    see https://github.com/ashb/sphinx-argparse/issues/9#issue-1097057823
    """
    
    # parser = CustomArgumentParser()
    # parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
    parser = argparse.ArgumentParser(usage='%(prog)s [image args] [model args] [...]')
    # parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)

        
    # settings for locating and formatting images
    input_img_args = parser.add_argument_group("input image arguments")
    input_img_args.add_argument('--dir',
                                default=[], 
                                type=str, help='folder containing data on which to run or train')
    input_img_args.add_argument('--look_one_level_down', action='store_true', help='run processing on all subdirectories of current folder')
    input_img_args.add_argument('--mxnet', action='store_true', help='use mxnet')
    input_img_args.add_argument('--img_filter',default=[], type=str, help='filter images by this suffix')
    input_img_args.add_argument('--channel_axis', default=None, type=int, 
                                help='axis of image which corresponds to image channels')
    input_img_args.add_argument('--z_axis', default=None, type=int, 
                                help='axis of image which corresponds to Z dimension')
    input_img_args.add_argument('--chan', default=0, type=int, 
                                help='channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s')
    input_img_args.add_argument('--chan2', default=0, type=int, 
                                help='nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s')
    input_img_args.add_argument('--invert', action='store_true', help='invert grayscale channel')
    input_img_args.add_argument('--all_channels', action='store_true', help='use all channels in image if using own model and images with special channels')
    input_img_args.add_argument('--dim', default=2, type=int, 
                                help='number of spatiotemporal dimensions of images (not counting channels). Default: %(default)s') ##
    
    # model settings 
    model_args = parser.add_argument_group("model arguments")
    model_args.add_argument('--pretrained_model', required=False, default='cyto', type=str, help='model to use')
    model_args.add_argument('--unet', required=False, default=0, type=int, help='run standard unet instead of cellpose flow output')
    model_args.add_argument('--nclasses', default=None, type=int, help='number of prediction classes for model (3 for Cellpose, 4 for Omnipose boundary field)')
    model_args.add_argument('--nchan', default=None, type=int, help='number of channels on which model is trained')
    model_args.add_argument('--kernel_size', default=2, type=int, help='kernel size for maskpool. Starts at 2, higher means more aggressive downsampling.')


    # algorithm settings
    algorithm_args = parser.add_argument_group("algorithm arguments")
    algorithm_args.add_argument('--omni', action='store_true', help='Omnipose algorithm (disabled by default)')
    algorithm_args.add_argument('--affinity_seg',action='store_true', help='use new affinity segmentation algorithm (disabled by default)')
    algorithm_args.add_argument('--cluster', action='store_true', help='DBSCAN clustering. Reduces oversegmentation of thin features (disabled by default)')
    algorithm_args.add_argument('--no_suppress', action='store_true', help='Euler integration 1/t suppression reduces oversegmentation but can give undersegmentation in 3D; this flag disables it.')
    algorithm_args.add_argument('--fast_mode', action='store_true', help='make code run faster by turning off 4 network averaging and resampling')
    algorithm_args.add_argument('--no_resample', action='store_true', 
                                help="disable dynamics on full image (makes algorithm faster for images with large diameters)")
    algorithm_args.add_argument('--no_net_avg', action='store_true', help='make code run faster by only running 1 network')
    algorithm_args.add_argument('--no_interp', action='store_true', help='do not interpolate when running dynamics (was default)')
    algorithm_args.add_argument('--do_3D', action='store_true', help='process images as 3D stacks of images (nplanes x nchan x Ly x Lx')
    algorithm_args.add_argument('--diameter', required=False, default=0., type=float, 
                                help='cell diameter, 0 disables unless sizemodel is present. Default: %(default)s')
    algorithm_args.add_argument('--rescale', required=False, default=None, type=float, 
                                help='image rescaling factor (r = diameter / model diameter)')
    algorithm_args.add_argument('--stitch_threshold', required=False, default=0.0, type=float, help='compute masks in 2D then stitch together masks with IoU>0.9 across planes')
    algorithm_args.add_argument('--flow_threshold', default=0.4, type=float, help='flow error threshold, 0 turns off this optional QC step. Default: %(default)s')
    algorithm_args.add_argument('--mask_threshold', default=0, type=float, help='mask threshold, default is 0, decrease to find more and larger masks')
    algorithm_args.add_argument('--niter', default=None, type=float, help='Number of Euler iterations, enter value to override Omnipose diameter estimation (under/over-segment)')
    algorithm_args.add_argument('--anisotropy', required=False, default=1.0, type=float,
                                help='anisotropy of volume in 3D')
    algorithm_args.add_argument('--diam_threshold', required=False, default=12.0, type=float, 
                                help='cell diameter threshold for upscaling before mask rescontruction, default 12')
    algorithm_args.add_argument('--exclude_on_edges', action='store_true', help='discard masks which touch edges of image')
    algorithm_args.add_argument('--min_size', default=15, type=float, help='minimum size for masks, helps if small debris is labeled')
    algorithm_args.add_argument('--max_size', default=None, type=float, help='maximum size for masks, helps if background patches are labeled')


    # output settings
    output_args = parser.add_argument_group("output arguments")
    output_args.add_argument('--save_png', action='store_true', help='save masks as png')
    output_args.add_argument('--save_tif', action='store_true', help='save masks as tif')
    output_args.add_argument('--no_npy', action='store_true', help='suppress saving of npy')
    output_args.add_argument('--savedir', default=None, type=str, help='folder to which segmentation results will be saved (defaults to input image directory)')
    output_args.add_argument('--dir_above', action='store_true', help='save output folders adjacent to image folder instead of inside it (off by default)')
    output_args.add_argument('--in_folders', action='store_true', help='flag to save output in folders (off by default)')
    output_args.add_argument('--save_flows', action='store_true', help='whether or not to save RGB images of flows when masks are saved (disabled by default)')
    output_args.add_argument('--save_outlines', action='store_true', help='whether or not to save RGB outline images when masks are saved (disabled by default)')
    output_args.add_argument('--save_ncolor', action='store_true', help='whether or not to save minimal "n-color" masks (disabled by default')
    output_args.add_argument('--save_txt', action='store_true', help='flag to enable txt outlines for ImageJ (disabled by default)')
    output_args.add_argument('--transparency', action='store_true', help='store flows with background transparent (alpha=flow magnitude) (disabled by default)')
    

    # training settings
    training_args = parser.add_argument_group("training arguments")
    training_args.add_argument('--train', action='store_true', help='train network using images in dir')
    training_args.add_argument('--train_size', action='store_true', help='train size network at end of training')
    training_args.add_argument('--mask_filter',
                        default='_masks', type=str, help='end string for masks to run on. Default: %(default)s')
    training_args.add_argument('--test_dir',
                        default=[], type=str, help='folder containing test data (optional)')
    training_args.add_argument('--learning_rate',
                        default=0.2, type=float, help='learning rate. Default: %(default)s')
    training_args.add_argument('--n_epochs',
                        default=500, type=int, help='number of epochs. Default: %(default)s')
    training_args.add_argument('--batch_size',
                        default=8, type=int, help='batch size. Default: %(default)s')
    training_args.add_argument('--num_workers',
                    default=0, type=int, help='number of dataloader workers. Default: %(default)s')
    training_args.add_argument('--dataloader',action='store_true', help='Use pytorch dataloader instead of older manual loading code.')
    training_args.add_argument('--min_train_masks',
                        default=1, type=int, help='minimum number of masks a training image must have to be used. Default: %(default)s')
    training_args.add_argument('--residual_on',
                        default=1, type=int, help='use residual connections')
    training_args.add_argument('--style_on',
                        default=1, type=int, help='use style vector')
    training_args.add_argument('--concatenation',
                        default=0, type=int, help='concatenate downsampled layers with upsampled layers (off by default which means they are added)')
    training_args.add_argument('--save_every',
                        default=100, type=int, help='number of epochs to skip between saves. Default: %(default)s')
    training_args.add_argument('--save_each', action='store_true', help='save the model under a different filename per --save_every epoch for later comparsion')
    training_args.add_argument('--RAdam', action='store_true', help='use RAdam instead of SGD')
    training_args.add_argument('--checkpoint', action='store_true', help='turn on checkpoints to reduce memory usage')
    training_args.add_argument('--dropout',action='store_true', help='Use dropout in training')
    training_args.add_argument('--tyx',
                        default=None, type=str, help='list of yx, zyx, or tyx dimensions for training')
    training_args.add_argument('--links',action='store_true', help='Search and use link files for multi-label objects.')
    training_args.add_argument('--amp',action='store_true', help='Use Automatic Mixed Precision.')
    training_args.add_argument('--affinity_field',action='store_true', help='Use summed affinity instead of distance field.')
    
    # settings for CPU vs GPU
    hardware_args = parser.add_argument_group("hardware arguments")
    hardware_args.add_argument('--use_gpu', action='store_true', help='use gpu if torch or mxnet with cuda installed')
    hardware_args.add_argument('--check_mkl', action='store_true', help='check if mkl working')
    hardware_args.add_argument('--mkldnn', action='store_true', help='for mxnet, force MXNET_SUBGRAPH_BACKEND = "MKLDNN"')
    
    # misc settings
    development_args = parser.add_argument_group("development arguments")
    development_args.add_argument('--verbose', action='store_true', help='flag to output extra information (e.g. diameter metrics) for debugging and fine-tuning parameters')
    development_args.add_argument('--testing', action='store_true', help='flag to suppress CLI user confirmation for saving output; for test scripts')
    development_args.add_argument('--timing', action='store_true', help='flag to output timing information for select modules')

    return parser