from . import core, utils

from numba import njit

import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.collections as mcoll
import types

import numpy as np
from matplotlib.backend_bases import GraphicsContextBase, RendererBase
from matplotlib.collections import LineCollection

class GC(GraphicsContextBase):
    def __init__(self):
        super().__init__()
        self._capstyle = 'round'

def custom_new_gc(self):
    return GC()


def plot_edges(shape,affinity_graph,neighbors,coords,
               figsize=1,fig=None,ax=None, extent=None, slc=None, pic=None, 
               edgecol=[.75]*3+[.5],linewidth=0.15,step_inds=None,
               cmap='inferno',origin='lower'):
    

    nstep,npix = affinity_graph.shape 
    coords = tuple(coords)
    indexes, neigh_inds, ind_matrix = utils.get_neigh_inds(tuple(neighbors),coords,shape)
    
    if step_inds is None:
        step_inds = np.arange(nstep) 
    px_inds = np.arange(npix)
    
    edge_list = core.affinity_to_edges(affinity_graph.astype(bool),
                                       neigh_inds,
                                       step_inds,
                                       px_inds)
    
    aff_coords = np.array(coords).T
    segments = np.stack([[aff_coords[:,::-1][e]+0.5  for e in edge] for edge in edge_list])
    # segments = np.stack([[aff_coords[e]+0.5  for e in edge] for edge in edge_list])
    
    RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)
    newfig = fig is None and ax is None
    if newfig:
        if type(figsize) is not (list or tuple):
            figsize = (figsize,figsize)
        fig, ax = plt.subplots(figsize=figsize)
    
    # ax.invert_yaxis()
    if extent is None:
        extent = np.array([0,shape[1],0,shape[0]])
    
    nopic = pic is None
    if nopic:
        summed_affinity = np.zeros(shape,dtype=int)
        summed_affinity[coords] = np.sum(affinity_graph,axis=0)
        # print(np.unique(summed_affinity))
        # c = sinebow(8)
        # colors = np.array(list(c.values()))
        # affinity_cmap = mpl.colors.ListedColormap(colors)
        # colors = mpl.colormaps.get_cmap(cmap).reversed()(np.linspace(-1,1,8))
        colors = mpl.colormaps.get_cmap(cmap).reversed()(np.linspace(0,1,9))    
        # colors = mpl.colormaps.get_cmap(cmap)(np.linspace(0,1,8))        
        
        colors = np.vstack((np.array([0]*4),colors))

        affinity_cmap = mpl.colors.ListedColormap(colors)
        pic = affinity_cmap(summed_affinity)
        
    ax.imshow(pic[slc] if slc is not None else pic, extent=extent,origin=origin)
    
#         # Generate random values between 0.5 and 1
#     random_values = np.random.uniform(.75, 1, size=(len(segments),))

#     # Multiply base_color by random values
#     colors = edgecol * random_values[:, np.newaxis]
    colors = edgecol
    
    line_segments = mcoll.LineCollection(segments, color=colors,linewidths=linewidth)
    ax.add_collection(line_segments)
    
    if newfig:
        plt.axis('off')
        ax.invert_yaxis()
        plt.show()
    
    if nopic:
        return summed_affinity, affinity_cmap
    
    
def sinebow(N,bg_color=[0,0,0,0], offset=0):
    """ Generate a color dictionary for use in visualizing N-colored labels. Background color 
    defaults to transparent black. 
    
    Parameters
    ----------
    N: int
        number of distinct colors to generate (excluding background)
        
    bg_color: ndarray, list, or tuple of length 4
        RGBA values specifying the background color at the front of the  dictionary.
    
    Returns
    --------------
    Dictionary with entries {int:RGBA array} to map integer labels to RGBA colors. 
    
    """
    colordict = {0:bg_color}
    for j in range(N): 
        k = j+offset
        angle = k*2*np.pi / (N) 
        r = ((np.cos(angle)+1)/2)
        g = ((np.cos(angle+2*np.pi/3)+1)/2)
        b = ((np.cos(angle+4*np.pi/3)+1)/2)
        colordict.update({j+1:[r,g,b,1]})
    return colordict

@njit
def colorize(im,colors=None,offset=0):
    N = len(im)
    if colors is None:
        angle = np.arange(0,1,1/N)*2*np.pi+offset
        angles = np.stack((angle,angle+2*np.pi/3,angle+4*np.pi/3),axis=-1)
        colors = (np.cos(angles)+1)/2
    rgb = np.zeros((im.shape[1], im.shape[2], 3))
    for i in range(N):
        for j in range(3):
            rgb[..., j] += im[i] * colors[i, j]
    rgb /= N
    return rgb

import ncolor
def apply_ncolor(masks,offset=0,cmap=None,max_depth=20,expand=True):
    m,n = ncolor.label(masks,
                       max_depth=max_depth,
                       return_n=True,
                       conn=2, 
                       expand=expand)
    if cmap is None:
        c = sinebow(n,offset=offset)
        colors = np.array(list(c.values()))
        cmap = mpl.colors.ListedColormap(colors)
        return cmap(m)
    else:
        return cmap(utils.rescale(m))

def imshow(img,figsize=2,ax=None,hold=False,**kwargs):
    if type(figsize) is not (list or tuple):
        figsize = (figsize,figsize)
    if ax is None:
        fig,ax = plt.subplots(frameon=False,figsize=figsize)
    else:
        hold=True
    ax.imshow(img,**kwargs)
    ax.axis("off")
    if not hold:
        plt.show()

# def get_cmap(masks):
#     lut = ncolor.get_lut(masks)
#     c = sinebow(lut.max())
#     colors = [c[l] for l in lut]
#     cmap = mpl.colors.ListedColormap(colors)
#     return cmap