
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import datetime

from torch.amp import autocast 
import torch.utils.checkpoint as cp

# from . import transforms, io, dynamics, utils
from omnipose.gpu import ARM, torch_GPU, torch_CPU, empty_cache

def dilation_list(x,N):
    return np.round(np.linspace(1, x, N)).astype(int).tolist()
    
    
def batchconv(in_channels, out_channels, kernel_size, dim, dilation, relu=True):
    BatchNorm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
    ConvND = nn.Conv2d if dim == 2 else nn.Conv3d
    
    # Adjust padding for dilated convolutions
    padding = ((kernel_size - 1) * dilation) // 2
    # padding = (kernel_size * dilation) // 2 +1
    
    # padding = kernel_size // 2
    
    layers = [BatchNorm(in_channels, eps=1e-5, momentum=0.05)] # cp uses momentum 0.05 now? default is .1
    if relu:
        layers.append(nn.ReLU(inplace=True))
    layers.append(ConvND(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, 
                         padding_mode='reflect')) # this padding mode is not supported by mkldnn 


    return nn.Sequential(*layers)

class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz, dim, dilation):
        super().__init__()
        
        self.conv = nn.Sequential()
        # self.proj = batchconv(in_channels, out_channels, 1, dim, dilation, relu=False)
        self.proj = batchconv(in_channels, out_channels, 1, dim, 1, relu=False)
        
        
        for t in range(4):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz, dim, dilation))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz, dim, dilation))
                
    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x
        
class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz, dim, dilation):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz, dim, dilation))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz, dim, dilation))
                
    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x
        
class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, parent, dilation=None):
        super().__init__()
        sz = parent.sz
        concatenation = parent.concatenation
        dim = parent.dim
        dilation = parent.dilation if dilation is None else dilation
        
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz, dim, dilation))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation, concatenation=concatenation))
        self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation))
        self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation))
        self.proj = batchconv(in_channels, out_channels, 1, dim, 1, relu=False) # kernel_size, dim, dilation, relu
        

    def forward(self, x, y, style, mkldnn=False):
        # print('resup')
        # print(x.shape)
        # print(y.shape)
        # print(style.shape)
        # print('\n')
        x = self.proj(x) + self.conv[1](style, self.conv[0](x) + y, mkldnn=False) # use conv_0 and conv_1, first fonvolution + skip connection
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=False), mkldnn=False) # use conv_2 and conv_3, additional residual connections
        return x

        
class convup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, parent, dilation=None):
        super().__init__()
        sz = parent.sz
        concatenation = parent.concatenation
        dim = parent.dim
        dilation = parent.dilation if dilation is None else dilation
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz, dim, dilation)) # kernel_size, dim, dilation, relu
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, dim, dilation, concatenation=concatenation))
        
    def forward(self, x, y, style):
        # print('convup')
        # print(x.shape)
        # print(y.shape)
        # print(style.shape)
        # print('\n')
        x = self.conv[1](style, self.conv[0](x) + y)
        return x
        

class batchconvstyle(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, dim, dilation, concatenation=False):
        super().__init__()
        self.conv = batchconv(in_channels, out_channels, sz, dim, dilation)
        if concatenation:
            self.full = nn.Linear(style_channels, out_channels*2) 
        else:
            self.full = nn.Linear(style_channels, out_channels)
        self.dim = dim
        
    def forward(self, style, x, mkldnn=False, y=None):
        if y is not None: # not sure why this was added 
            x = x + y
            
        feat = self.full(style)
        
        # number of unsqueezing steps depends on dimension!
        for k in range(self.dim):
            feat = feat.unsqueeze(-1)
            
        y = x + feat
        y = self.conv(y)
        return y
    

class downsample(nn.Module):
    # def __init__(self, nbase, sz, residual_on=True, kernel_size=2, dim=2, checkpoint=False):
    def __init__(self, parent):
    
        super().__init__()
        nbase = parent.nbase
        sz = parent.sz
        residual_on = parent.residual_on
        dim = parent.dim
        dilation = parent.dilation
        kernel_size = parent.kernel_size
        scale_factor = parent.scale_factor
        
        
        self.checkpoint = parent.checkpoint
        self.down = nn.Sequential()
        
        # Maxpool only has support for 1-3 dimensions
        # will have to write our own implementation for 4d 
        maxpool = nn.MaxPool2d if dim == 2 else nn.MaxPool3d
        
        # maybe consider AdaptiveMaxPool
        
        self.maxpool = maxpool(kernel_size=kernel_size, stride=scale_factor) # (2,2), stride by default is the same as kernel window
         
        N = len(nbase)-1
        dilations = dilation_list(dilation, N)
        # for n in range(len(nbase)-1):
        for n, dilation in enumerate(dilations):
            # downscaling takes in_channels, out_channels, sz, dim
            # the number of output channels increases as we go deeper into the network to capture more abstract, high-level features
            # The feature maps get smaller spatially (via max pooling), so the network compensates by increasing the number of channels to preserve representational capacity
            if residual_on:
                self.down.add_module('res_down_%d'%n, 
                                     resdown(nbase[n], nbase[n+1], sz, dim, dilation))
            else:
                self.down.add_module('conv_down_%d'%n, 
                                     convdown(nbase[n], nbase[n+1], sz, dim, dilation))
      
    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n>0:
                y = cp.checkpoint(self.maxpool,xd[n-1]) if self.checkpoint else self.maxpool(xd[n-1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd
        

class upsample(nn.Module):
    def __init__(self, parent):
        super().__init__()
        nbase = parent.nbaseup
        kernel_size = parent.kernel_size
        scale_factor = parent.scale_factor
        
        self.upsampling = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        # self.upsampling = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        
        # upscaling is defined here in the same order as the downscaling, i.e. the deepest map is last
        # the forward method matches the last layer of the downscaling with the first layer of the upscaling
        self.up = nn.Sequential()
        self.checkpoint = parent.checkpoint
        
        
        N = len(nbase)-1
        dilations = dilation_list(parent.dilation, N)#[::-1]
        for k, dilation in enumerate(dilations):
        
            n = k+1
            # upscaling takes in_channels, out_channels, style_channels, sz, concatenation, dim
            if parent.residual_on:
                self.up.add_module('res_up_%d'%(n-1), 
                                resup(nbase[n], nbase[n-1], nbase[-1], parent, dilation=dilation))
            else:
                self.up.add_module('conv_up_%d'%(n-1),
                                convup(nbase[n], nbase[n-1], nbase[-1], parent, dilation=dilation))
                

                
    # def forward(self, style, xd, mkldnn=False): # input style, T0 is xd (the downsampled data, feature map)
    #     x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)
    #     for n in range(len(self.up)-2,-1,-1):
    #         if mkldnn:
    #             x = self.upsampling(x.to_dense()).to_mkldnn()
    #         else:
    #             # x = self.upsampling(x)
    #             x = cp.checkpoint(self.upsampling,x) if self.checkpoint else self.upsampling(x) # checkpoint doesn't do much here
                
    #         x =  cp.checkpoint(self.up[n], x, xd[n], style, mkldnn) if self.checkpoint else self.up[n](x, xd[n], style, mkldnn=mkldnn) # ok this one saves a ton of memory,2GB 
            
    #     return x
    
    def forward(self, style, xd, mkldnn=False):  # input style, T0 is xd (the downsampled data, feature map)
        x = xd[-1]  # Start with the deepest feature map
        for n in range(len(self.up)):  # Iterate through all layers
            idx = -(n + 1)  # Convert to negative index
            if n > 0:  # Skip upsampling for the first layer
                x = cp.checkpoint(self.upsampling, x) if self.checkpoint else self.upsampling(x)
            
            x = cp.checkpoint(self.up[idx], x, xd[idx], style, False) if self.checkpoint else self.up[idx](x, xd[idx], style, mkldnn=False)
            
        return x
        
        
class make_style(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.dim = parent.dim
        self.flatten = nn.Flatten()
        self.avg_pool = F.avg_pool2d if self.dim==2 else F.avg_pool3d


    def forward(self, x0):
        style = self.avg_pool(x0, kernel_size=tuple(x0.shape[-self.dim:]))
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5

        return style

    
class CPnet(nn.Module):
    def __init__(self, nbase, nout, sz, residual_on=True, 
                 style_on=True, concatenation=False, mkldnn=False, dim=2, 
                 checkpoint=False, dropout=False, kernel_size=2, scale_factor=2, dilation=1):
        super(CPnet, self).__init__()
        
        self.checkpoint = checkpoint # master switch 
        self.kernel_size = kernel_size # for maxpool
        self.scale_factor = scale_factor
        self.dilation = dilation
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        # self.sz = kernel_size
        self.dim = dim 
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = False
        # self.downsample = downsample(nbase, sz, residual_on=residual_on, kernel_size=self.kernel_size, dim=self.dim)
        self.downsample = downsample(self)
        
        # The output channels at each level of upsampling correspond to the input channels from the next deeper level of the downsampling phase
        # Deepest feature map (bottleneck) has the same number of channels for its input and output during the upsampling.
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.nbaseup = nbaseup #nbase[1:].append(nbase[-1])
    
        self.upsample = upsample(self) # replace with learned upsampling? 
        self.make_style = make_style(self)
        # self.output = batchconv(nbaseup[0], nout, 1, self.dim, self.dilation) # in, out, kernel, dim, dilation
        self.output = batchconv(nbaseup[0], nout, 1, self.dim, 1)
        
        self.style_on = style_on
        
        self.do_dropout = dropout
        if self.do_dropout:
            self.dropout = nn.Dropout(0.1) # make this toggle on with omni?
    
    # for layer in self.children():
    #     if hasattr(layer, 'reset_parameters'):
    #         layer.reset_parameters()
    
    
    # @autocast() #significant decrease in GPU memory usage (e.g. 19.8GB vs 11.8GB for a particular test run)
    def forward(self, data):
        T0 = self.downsample(data)
        # T0 = cp.checkpoint(self.downsample,data) #casues a warning but appears to work, 11 to 8 GB! 
        
        style = self.make_style(T0[-1])
        style = cp.checkpoint(self.make_style,T0[-1]) if self.checkpoint else self.make_style(T0[-1])
            
        style0 = style
        if not self.style_on:
            style = style * 0
        
        T0 = self.upsample(style, T0, False)
        # T0 = cp.checkpoint(self.upsample, style, T0, self.mkldnn) #not working
        
        if self.do_dropout:
            T0 = self.dropout(T0)

        # T0 = self.output(T0)
        T0 = cp.checkpoint(self.output,T0) if self.checkpoint else self.output(T0) #only  small reduction, 300MB
        
        # cellpose now uses a T1 as well, not sure why to return what is before the upscaling 
        
        return T0, style0

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

        
    def load_model(self, filename, cpu=False):
        if not cpu:
            self.load_state_dict(torch.load(filename,
                                            map_location=torch_GPU,
                                            weights_only=True))
            

            # checkpoint = torch.load(filename, map_location=torch_GPU,  weights_only=False)

            # # Extract the state dictionary
            # if 'state_dict' in checkpoint:
            #     state_dict = checkpoint['state_dict']
            # else:
            #     state_dict = checkpoint
                
            # # Load the state dictionary into the model
            # try:
            #     self.load_state_dict(state_dict, strict=False)
            # except Exception as e:
            #     print('Failed to load model:', e)
            
        else:
            self.__init__(self.nbase,
                          self.nout,
                          self.sz,
                          self.residual_on,
                          self.style_on,
                          self.concatenation,
                          self.mkldnn,
                          self.dim,
                          self.checkpoint,
                          self.do_dropout,
                          self.kernel_size, self.scale_factor, self.dilation)
            state_dict = torch.load(filename, map_location=torch_CPU, weights_only=True)
            # print('ggg',state_dict)
            try:
#                 from collections import OrderedDict
#                 new_state_dict = OrderedDict()

#                 for k, v in state_dict.items():
#                     if 'module' not in k:
#                         k = 'module.'+k
#                     else:
#                         k = k.replace('features.module.', 'module.features.')
#                     new_state_dict[k]=v

#                 self.load_state_dict(new_state_dict, strict=False)
                self.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print('failed to load model', e)

