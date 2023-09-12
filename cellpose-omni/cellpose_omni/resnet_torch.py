
import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import datetime

from torch.cuda.amp import autocast 
import torch.utils.checkpoint as cp

from . import transforms, io, dynamics, utils

# I wanted to try out an ND implementation, so this is just for testing 
CONVND = False
# CONVND = True

if CONVND:
    from .convNd import convNd

from omnipose.gpu import ARM, torch_GPU, torch_CPU, empty_cache
    
sz = 3 #kernel size, works as xy or xyz/xyt equally well 
WEIGHT = 1e-4
BIAS = 1e-4
def batchconv(in_channels, out_channels, sz, dim):
    if dim==2:
        return nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-5),
            nn.ReLU(inplace=True),
            convNd(in_channels, out_channels, num_dims=dim, kernel_size=sz, stride=1, padding=sz//2,
                  kernel_initializer=lambda x: torch.nn.init.constant_(x, WEIGHT), 
                  bias_initializer=lambda x: torch.nn.init.constant_(x, BIAS)) if CONVND else 
            nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
        )
    elif dim==3:
        return nn.Sequential(
            nn.BatchNorm3d(in_channels, eps=1e-5),
            nn.ReLU(inplace=True),
            convNd(in_channels, out_channels, dim, sz, stride=1, padding=sz//2) if CONVND else 
            nn.Conv3d(in_channels, out_channels, sz, padding=sz//2),

        )  

def batchconv0(in_channels, out_channels, sz, dim):
    if dim==2:
        return nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-5),
            convNd(in_channels, out_channels, num_dims=dim, kernel_size=sz, stride=1, padding=sz//2,
                  kernel_initializer=lambda x: torch.nn.init.constant_(x, WEIGHT), 
                  bias_initializer=lambda x: torch.nn.init.constant_(x, BIAS)) if CONVND else 
            nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
        )
    elif dim==3:
        return nn.Sequential(
            nn.BatchNorm3d(in_channels, eps=1e-5),
            convNd(in_channels, out_channels, dim, sz, stride=1, padding=sz//2) if CONVND 
            else nn.Conv3d(in_channels, out_channels, sz, padding=sz//2),
        )  

class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz, dim):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv0(in_channels, out_channels, 1, dim)
        for t in range(4):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz, dim))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz, dim))
                
    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x

class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz, dim):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t==0:
                self.conv.add_module('conv_%d'%t, batchconv(in_channels, out_channels, sz, dim))
            else:
                self.conv.add_module('conv_%d'%t, batchconv(out_channels, out_channels, sz, dim))
                
    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x

class downsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True, kernel_size=2, dim=2, checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.down = nn.Sequential()
        
        # Maxpool only has support for 1-3 dimensions
        # will have to write our own implementation for 4d 
        if dim==2:
            maxpool = nn.MaxPool2d
        elif dim==3:
            maxpool = nn.MaxPool3d
        self.maxpool = maxpool(kernel_size) # (2,2), stride by defualt is the same as kernel window
        
        for n in range(len(nbase)-1):
            if residual_on:
                self.down.add_module('res_down_%d'%n, resdown(nbase[n], nbase[n+1], sz, dim))
            else:
                self.down.add_module('conv_down_%d'%n, convdown(nbase[n], nbase[n+1], sz, dim))
      
    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n>0:
                # y = self.maxpool(xd[n-1])
                y = cp.checkpoint(self.maxpool,xd[n-1]) if self.checkpoint else self.maxpool(xd[n-1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd
    
class batchconvstyle(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False, dim=2):
        super().__init__()
        self.concatenation = concatenation
        self.conv = batchconv(in_channels, out_channels, sz, dim)
        if concatenation:
            self.full = nn.Linear(style_channels, out_channels*2) 
        else:
            self.full = nn.Linear(style_channels, out_channels)
        self.dim = dim
        
    def forward(self, style, x, mkldnn=False):
        feat = self.full(style)
        
        # numer of unsqueezing steps depends on dimension!
        for k in range(self.dim):
            feat = feat.unsqueeze(-1)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat).to_mkldnn()
        else:
            y = x + feat
        y = self.conv(y)
        return y
    
class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False, dim=2):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz, dim))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, concatenation=concatenation, dim=dim))
        self.conv.add_module('conv_2', batchconvstyle(out_channels, out_channels, style_channels, sz, dim=dim))
        self.conv.add_module('conv_3', batchconvstyle(out_channels, out_channels, style_channels, sz, dim=dim))
        self.proj  = batchconv0(in_channels, out_channels, 1, dim=dim)

    def forward(self, x, y, style, mkldnn=False):
        # print('shape',self.conv[0](x).shape,y.shape)
        x = self.proj(x) + self.conv[1](style, self.conv[0](x) + y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn)
        return x
    
class convup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False, dim=2):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', batchconv(in_channels, out_channels, sz, dim))
        self.conv.add_module('conv_1', batchconvstyle(out_channels, out_channels, style_channels, sz, concatenation=concatenation, dim=dim))
        
    def forward(self, x, y, style):
        x = self.conv[1](style, self.conv[0](x) + y)
        return x
    
class make_style(nn.Module):
    def __init__(self,dim=2):
        super().__init__()
        #self.pool_all = nn.AvgPool2d(28)
        self.flatten = nn.Flatten()
        self.dim = dim

    def forward(self, x0):
        #style = self.pool_all(x0)
        if self.dim==2:
            avg_pool = F.avg_pool2d
        elif self.dim==3:
            avg_pool = F.avg_pool3d
            
        # style = avg_pool(x0, kernel_size=(x0.shape[-2],x0.shape[-1]))
        style = avg_pool(x0, kernel_size=tuple(x0.shape[-self.dim:]))
        
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5

        return style
    
class upsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True, concatenation=False, kernel_size=2, dim=2, checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.upsampling = nn.Upsample(scale_factor=kernel_size, mode='nearest')
        self.up = nn.Sequential()
        for n in range(1,len(nbase)):
            if residual_on:
                self.up.add_module('res_up_%d'%(n-1), 
                    resup(nbase[n], nbase[n-1], nbase[-1], sz, concatenation, dim))
            else:
                self.up.add_module('conv_up_%d'%(n-1), 
                    convup(nbase[n], nbase[n-1], nbase[-1], sz, concatenation, dim))

    def forward(self, style, xd, mkldnn=False):
        x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)
        for n in range(len(self.up)-2,-1,-1):
            if mkldnn:
                x = self.upsampling(x.to_dense()).to_mkldnn()
            else:
                # x = self.upsampling(x)
                x = cp.checkpoint(self.upsampling,x) if self.checkpoint else self.upsampling(x) # doesn't do much 
                
            # x = self.up[n](x, xd[n], style, mkldnn=mkldnn)
            x =  cp.checkpoint(self.up[n], x, xd[n], style, mkldnn) if self.checkpoint else self.up[n](x, xd[n], style, mkldnn=mkldnn)# ok this one saves a ton of memory,2GB 
            
        return x
    
class CPnet(nn.Module):
    def __init__(self, nbase, nout, sz, residual_on=True, 
                 style_on=True, concatenation=False, mkldnn=False, dim=2, 
                 checkpoint=False, dropout=False, kernel_size=2):
        super(CPnet, self).__init__()
        
        self.checkpoint = checkpoint # master switch 
        self.kernel_size = kernel_size # for maxpool
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.dim = dim 
        self.residual_on = residual_on
        self.style_on = style_on
        self.concatenation = concatenation
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(nbase, sz, residual_on=residual_on, kernel_size=self.kernel_size, dim=self.dim)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, residual_on=residual_on, concatenation=concatenation, 
                                 kernel_size=self.kernel_size, dim=self.dim, checkpoint=self.checkpoint)
        self.make_style = make_style(dim=self.dim)
        self.output = batchconv(nbaseup[0], nout, 1, self.dim)
        self.style_on = style_on
        
        self.do_dropout = dropout
        if self.do_dropout:
            self.dropout = nn.Dropout(0.1) # make this toggle on with omni?
    
    # for layer in self.children():
    #     if hasattr(layer, 'reset_parameters'):
    #         layer.reset_parameters()
    # @autocast() #significant decrease in GPU memory usage (e.g. 19.8GB vs 11.8GB for a particular test run)
    def forward(self, data):
        if self.mkldnn:
            data = data.to_mkldnn()
        T0 = self.downsample(data)
        # T0 = cp.checkpoint(self.downsample,data) #casues a warning but appears to work, 11 to 8 GB! 
        
        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense()) 
        else:
            style = self.make_style(T0[-1])
            style = cp.checkpoint(self.make_style,T0[-1]) if self.checkpoint else self.make_style(T0[-1])
            
        style0 = style
        if not self.style_on:
            style = style * 0
        
        T0 = self.upsample(style, T0, self.mkldnn)
        # T0 = cp.checkpoint(self.upsample, style, T0, self.mkldnn) #not working
        
        if self.do_dropout:
            T0 = self.dropout(T0)

        # T0 = self.output(T0)
        T0 = cp.checkpoint(self.output,T0) if self.checkpoint else self.output(T0) #only  small reduction, 300MB
        
        if self.mkldnn:
            T0 = T0.to_dense()    
            #T1 = T1.to_dense()
        return T0, style0

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

        
    def load_model(self, filename, cpu=False):
        if not cpu:
            self.load_state_dict(torch.load(filename,map_location=torch_GPU))
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
                          self.kernel_size)
            state_dict = torch.load(filename, map_location=torch_CPU)
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
