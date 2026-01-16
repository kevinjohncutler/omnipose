from .imports import *

from skimage import color
from ..utils.color import sinebow
from ..utils.transforms import rescale

def channel_overlay(channels, color_indexes, colors=None, a=1, cmaps=None):
    """Overlay selected channels as colors onto the remaining channels as grayscale."""
    N = len(channels)
    n = len(color_indexes)
    
    # Identify the grayscale channels
    grayscale_indexes = [i for i in range(N) if i not in color_indexes]
    
    # Calculate the grayscale image
    grayscale = np.mean(np.take(channels, grayscale_indexes, axis=0), axis=0) if len(grayscale_indexes) else np.zeros_like(channels[0])

    # If colors are not provided, generate them
    if colors is None:
        angle = np.arange(0, 1, 1/n) * 2 * np.pi
        angles = np.stack((angle, angle + 2*np.pi/3, angle + 4*np.pi/3), axis=-1)
        colors = (np.cos(angles) + 1) / 2
        
    else:
        colors = np.stack(colors)
        
        if colors.ndim==1:
            colors = np.expand_dims(colors, axis=0)
    
    # if there is an alpha channel to colors, mostly for color map
    nchan = colors.shape[1] if cmaps is None else 4
    
    # Create an array to hold the RGB image
    rgb = np.zeros(channels[0].shape+(nchan,))
    
    # Apply the overlays to each color channel
    for i,idx in enumerate(color_indexes):
        mapped_chan = None if cmaps is None else cmaps[i](channels[idx])
        for j in range(nchan):
            if cmaps is None:
                cc =  a * channels[idx] * colors[i,j] # color contribution 
            else:
                cc = a * mapped_chan[...,j]
            rgb[..., j] += (1 - cc) * grayscale + cc
        
    rgb /= n
    
    return rgb


import ncolor
def mask_outline_overlay(img,masks,outlines,mono=None):
    """
    Apply a color overlay to a grayscale image based on a label matrix.
    mono is a single color to use. Otherwise, N sinebow colors are used. 
    """
    if mono is None:
        m,n = ncolor.label(masks,max_depth=20,return_n=True)
        c = sinebow(n)
        colors = np.array(list(c.values()))[1:]
    else:
        colors = mono
        m = masks>0
    if img.ndim == 3:
        im = rescale(color.rgb2gray(img))
    else:
        im = img
    overlay = color.label2rgb(m,im,colors,
                              bg_label=0,
                              alpha=np.stack([((m>0)*1.+outlines*0.75)/3]*3,axis=-1))
    return overlay

def mono_mask_bd(masks,outlines,color=[1,0,0],a=0.25):
    m = masks>0
    alpha = (m>0)*a+outlines*(1-a)
    return np.stack([m*c for c in color]+[alpha],axis=-1)