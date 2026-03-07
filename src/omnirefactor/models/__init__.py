from .imports import *
from .helpers import resolve_pretrained_model



def _model_props(
    omni=True,
    nclasses=None,
    logits=False,
    nsample=4,
    diam_mean=30.0,
    nchan=1,
    allow_blank_masks=False,
    channel_axis=None,
):
    return locals()

class OmniModel:
    """

    Parameters
    -------------------

    gpu: bool (optional, default False)
        whether or not to save model to GPU, will check if GPU available
        
    pretrained_model: str or list of strings (optional, default False)
        path to pretrained cellpose model(s), if None or False, no model loaded
        
    model_type: str (optional, default None)
        'cyto'=cytoplasm model; 'nuclei'=nucleus model; if None, pretrained_model used
        
    net_avg: bool (optional, default True)
        loads the 4 built-in networks and averages them if True, loads one network if False
        
    torch: bool (optional, default True)
        use torch nn rather than mxnet
        
    diam_mean: float (optional, default 27.)
        mean 'diameter', 27. is built in value for 'cyto' model
        
    device: mxnet device (optional, default None)
        where model is saved (mx.gpu() or mx.cpu()), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. mx.gpu(4))
        
    model_dir: str (optional, default None)
        overwrite the built in model directory where cellpose looks for models
    

    """
    
    def __init__(self, gpu=False, pretrained_model=False,
                 model_type=None, net_avg=True, use_torch=True,
                 device=None, **kwargs):
    
        if not torch:
            if not MXNET_ENABLED:
                use_torch = True
        self.torch = use_torch

        # print('torch is', torch) # duplicated in unetmodel class
        if isinstance(pretrained_model, np.ndarray):
            pretrained_model = list(pretrained_model)
        elif isinstance(pretrained_model, str):
            pretrained_model = [pretrained_model]
                
        # initialize according to arguments
        # these are overwritten if a model requires it (bact_omni the most restrictive)
        # split kwargs by signature: OmniModel props vs UnetND args
        model_kwargs, net_kwargs = split_kwargs([_model_props, UnetND.__init__], kwargs)
        self.__dict__.update(model_kwargs)
        # expose UnetND kwargs on the model (excluding init-only params)
        net_props = {k: v for k, v in net_kwargs.items() if k not in {"self", "nbase", "nout"}}
        self.__dict__.update(net_props)
        
        # channel axis might be useful here 
        pretrained_model, pretrained_model_string, net_avg, updates, residual_on, style_on, concatenation = (
            resolve_pretrained_model(
                pretrained_model=pretrained_model,
                model_type=model_type,
                net_avg=net_avg,
                use_torch=torch,
                model_names=MODEL_NAMES,
                bd_model_names=BD_MODEL_NAMES,
                c2_model_names=C2_MODEL_NAMES,
                omni=self.omni,
            )
        )
        if updates:
            self.__dict__.update(updates)
        else:
            if pretrained_model:
                pretrained_model_string = pretrained_model[0]
                params = parse_model_string(pretrained_model_string)
                if params is not None:
                    residual_on, style_on, concatenation = params 
        
        # set omni flag to true if the name contains it
        if pretrained_model_string is not None:
            self.omni = 'omni' in os.path.splitext(Path(pretrained_model_string).name)[0] if self.omni is None else self.omni 

        
        if not self.logits:
            # convert abstract prediction classes number to actual count
            # flow field components increase this by dim-1
            self.nclasses = self.nclasses + (self.dim-1)

        net_kwargs.pop("self", None)
        net_kwargs.pop("nbase", None)
        net_kwargs.pop("nout", None)
        net_kwargs.setdefault("sz", 3)

        if residual_on is not None:
            net_kwargs["residual_on"] = residual_on
            net_kwargs["style_on"] = style_on
            net_kwargs["concatenation"] = concatenation

        residual_on = net_kwargs.get("residual_on", True)
        style_on = net_kwargs.get("style_on", True)
        concatenation = net_kwargs.get("concatenation", False)
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation

        # initialize network (inlined from former OmniModelCore)
        self.unet = True
        self.gpu = gpu
        if device is None:
            self.device, gpu_available = assign_device(self.gpu)
        else:
            self.device = device
            self.gpu = self.device.type != 'cpu'

        if not self.torch:
            raise RuntimeError("Torch backend required for OmniModel.")

        self.nbase = [self.nchan] + [32 * (2**i) for i in range(self.nsample)]
        net_kwargs["residual_on"] = residual_on
        net_kwargs["style_on"] = style_on
        net_kwargs["concatenation"] = concatenation
        self.net = UnetND(self.nbase, self.nclasses, **net_kwargs).to(self.device)
        models_logger.info(f'u-net config: {self.nbase, self.nclasses, self.dim}')


        self.unet = False
        self.pretrained_model = pretrained_model
        
        if self.pretrained_model and len(self.pretrained_model)==1:
            
            # dataparallel A1
            # if self.torch and self.gpu:
            #     net = self.net.module
            # else:
            #     net = self.net
            # if self.torch and gpu:
            #     self.net = nn.DataParallel(self.net)

            self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu))

                
        ostr = ['off', 'on']
        omnistr = ['','_omni'] #toggle by containing omni phrase 
        self.net_type = 'cellpose_residual_{}_style_{}_concatenation_{}{}_abstract_nclasses_{}_nchan_{}_dim_{}'.format(ostr[residual_on],
                                                                                   ostr[style_on],
                                                                                   ostr[concatenation],
                                                                                   omnistr[self.omni],
                                                                                   self.nclasses-(self.dim-1), # "abstract"
                                                                                   self.nchan, 
                                                                                   self.dim) 
        
        if self.torch and gpu:
            self.net = nn.DataParallel(self.net)
            
            #A1
            # distributed.init_process_group()
            # distributed.launch
            # distributed.init_process_group(backend='nccl',rank=0,world_size=2)
            # self.net = nn.parallel.DistributedDataParallel(self.net)
#             rank = 0 # one computer
#             world_size = 1
#             setup(rank, world_size)
#             # distributed.init_process_group('nccl', 
#             #                        init_method='env://')
#             self.net = DDP(self.net,
#                            device_ids=[rank],
#                            output_device=rank)
            
            
    # eval contains most of the tricky code handling all the cases for nclasses 
    # to get eval to efficiently run on an entire image set, we could pass a torch dataset
    # this dataset could either parse out images loaded from memory or from storage 

from ..load.object import load_submodules

package_dir = os.path.dirname(__file__)
package_name = __package__
load_submodules(OmniModel, package_dir, package_name, exclude_modules={"imports", "logging", "helpers"})

__all__ = ["OmniModel"]
