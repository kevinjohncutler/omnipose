from .utils import rescale, torch_norm
from .color import sinebow

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

import types

import numpy as np
from matplotlib.backend_bases import GraphicsContextBase, RendererBase

from mpl_toolkits.axes_grid1 import ImageGrid

from skimage import img_as_ubyte


def setup():
    # Import necessary libraries
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, HTML

    # Custom CSS to center plots
    display(HTML("""
    <style>
        .jp-OutputArea-output img {
            display: block;
            margin: 0 auto;
        }
    </style>
    """))
    
    # Inject into the global namespace of the notebook
    ipython = get_ipython()  # Get the IPython instance
    ipython.user_global_ns['mpl'] = mpl
    ipython.user_global_ns['plt'] = plt
    ipython.user_global_ns['widgets'] = widgets
    ipython.user_global_ns['display'] = display

    # Set matplotlib inline for Jupyter notebooks
    ipython.run_line_magic('matplotlib', 'inline')

    # Define rc_params
    rc_params = {
        'figure.dpi': 300,
        'image.cmap': 'gray',
        'image.interpolation': 'nearest',
        'figure.frameon': False,
        'axes.grid': False,
        'axes.facecolor': 'none',      # Transparent axes
        'figure.facecolor': 'none',    # Transparent figure background
        'savefig.facecolor': 'none',   # Transparent save background
        'text.color': 'gray',          # Gray text for flexibility
        'axes.labelcolor': 'gray',
        'xtick.color': 'gray',
        'ytick.color': 'gray',
        'axes.edgecolor': 'gray'
    }

    # Update rcParams
    mpl.rcParams.update(rc_params)
    

def figure(nrow=None, ncol=None, **kwargs):
    fig = Figure(**kwargs)
    # fig = plt.figure(**kwargs)
    if nrow is not None and ncol is not None:
        axs = []
        for i in range(nrow * ncol):
            ax = fig.add_subplot(nrow, ncol, i + 1)
            axs.append(ax)
        return fig, axs
    else:
        ax = fig.add_subplot(111)
        return fig, ax

class GC(GraphicsContextBase):
    def __init__(self):
        super().__init__()
        self._capstyle = 'round'

def custom_new_gc(self):
    return GC()

def plot_edges(shape,affinity_graph,neighbors,coords,
               figsize=1,fig=None,ax=None, extent=None, slc=None, pic=None, 
               edgecol=[.75]*3+[.5],linewidth=0.15,step_inds=None,
               cmap='inferno',origin='lower',bounds=None):
    
    # import core here because that can take a while to load
    from .core import affinity_to_edges
    from .utils import get_neigh_inds 
    from matplotlib.collections import LineCollection
    
    print('adjust this to make edges appear even on edges or when target is 0')
    nstep,npix = affinity_graph.shape 
    coords = tuple(coords)
    indexes, neigh_inds, ind_matrix = get_neigh_inds(tuple(neighbors),coords,shape)
    
    if step_inds is None:
        step_inds = np.arange(nstep) 
    px_inds = np.arange(npix)
    
    edge_list = affinity_to_edges(affinity_graph.astype(bool),
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
        # fig, ax = plt.subplots(figsize=figsize)
        fig, ax = figure(figsize=figsize)
        
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
        
    
#         # Generate random values between 0.5 and 1
#     random_values = np.random.uniform(.75, 1, size=(len(segments),))

#     # Multiply base_color by random values
#     colors = edgecol * random_values[:, np.newaxis]
    colors = edgecol
    
    ax.imshow(pic[slc] if slc is not None else pic, extent=extent,origin=origin)


    line_segments = LineCollection(segments, color=colors,linewidths=linewidth)

    # if bounds is None:
    #     line_segments = LineCollection(segments, color=colors,linewidths=linewidth)

    # # if bounds is not None:
    # #     clip_rect = Rectangle((bounds[0], bounds[1]), bounds[2], bounds[3])
    # #     clip_rect.set_transform(ax.transData)
    # #     line_segments.set_clip_path(clip_rect)
        
    # else:
    #     # Create a bounding box that defines the extent
    #     bbox = Bbox.from_extents(bounds[0], bounds[1], bounds[0]+bounds[2], bounds[1]+bounds[3])

    #     # Create a path for each line segment and clip it to the bounding box
    #     clipped_segments = [Path(seg).clip_to_bbox(bbox).to_polygons() for seg in segments]

    #     # Create a line collection with the clipped segments
    #     line_segments = LineCollection(clipped_segments)


    ax.add_collection(line_segments)
    
    if newfig:
        # plt.axis('off')
        ax.set_axis_off()
        ax.invert_yaxis()
        # plt.show()
        canvas = FigureCanvas(fig)
        canvas.draw()
    
    if nopic:
        return summed_affinity, affinity_cmap

# @njit
# def colorize(im,colors=None,color_weights=None,offset=0):
#     N = len(im)
#     if colors is None:
#         angle = np.arange(0,1,1/N)*2*np.pi+offset
#         angles = np.stack((angle,angle+2*np.pi/3,angle+4*np.pi/3),axis=-1)
#         colors = (np.cos(angles)+1)/2
        
#     if color_weights is not None:
#         colors *= color_weights
        
#     rgb = np.zeros((im.shape[1], im.shape[2], 3))
#     for i in range(N):
#         for j in range(3):
#             rgb[..., j] += im[i] * colors[i, j] 
#     rgb /= N
#     return rgb

# @njit
def colorize(im, colors=None, color_weights=None, offset=0, channel_axis=-1):
    N = len(im)
    if colors is None:
        angle = np.arange(0, 1, 1/N) * 2 * np.pi + offset
        angles = np.stack((angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3), axis=-1)
        colors = (np.cos(angles) + 1) / 2

    if color_weights is not None:
        colors *= np.expand_dims(color_weights,-1)

    rgb_shape = im.shape[1:] + (colors.shape[1],)
    if channel_axis == 0:
        rgb_shape = rgb_shape[::-1]
    rgb = np.zeros(rgb_shape)

    # Use broadcasting to multiply im and colors and sum along the 0th dimension
    rgb = (np.expand_dims(im, axis=-1) * colors.reshape(colors.shape[0], 1, 1, colors.shape[1])).mean(axis=0)

    return rgb


def colorize_GPU(im, colors=None, color_weights=None, offset=0, channel_axis=-1):

    import torch 
    
    N = im.shape[0]
    device = im.device

    if colors is None:
        angle = torch.linspace(0, 1, N, device=device) * 2 * np.pi + offset
        angles = torch.stack((angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3), dim=-1)
        colors = (torch.cos(angles) + 1) / 2

    if color_weights is not None:
        colors *= color_weights.unsqueeze(-1)
        # colors /= color_weights.sum()
    # rgb_shape = im.shape[1:]+(colors.shape[1],)
    # if channel_axis == 0:
    #     rgb_shape = tuple(rgb_shape[::-1])
    # rgb = torch.ones(rgb_shape, device=device)

    # Use broadcasting to multiply im and colors and sum along the 0th dimension
    # rgb = (im.unsqueeze(-1) * colors.view(colors.shape[0], 1, 1, colors.shape[1])).mean(dim=0)
    
    im = im.unsqueeze(-1)  # Add an extra dimension to `im`
    # colors = colors.view(colors.shape[0], 1, 1, colors.shape[1])  # Reshape `colors`

    # print(im.shape,colors.shape)
    # Compute the mean and assign the result to `rgb`
    # rgb = (im*colors).mean(dim=0)
    
    # Perform the multiplication and mean computation using `einsum` - way faster 
    rgb = torch.einsum('ijkl,il->jkl', im.float(), colors.float()) / N

    return rgb
    
def apply_ncolor(masks,offset=0,cmap=None,max_depth=20,expand=True, maxv=1):

    import ncolor
    from cmap import Colormap
    cmap = Colormap(cmap) if isinstance(cmap, str) else cmap

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
        return cmap(rescale(m)/maxv)


def imshow(imgs, figsize=2, ax=None, hold=False, titles=None, title_size=8, spacing=0.05, 
           textcolor=[0.5]*3, dpi=300, text_scale = 1, **kwargs):

    if isinstance(imgs, list):
        if titles is None:
            titles = [None] * len(imgs)
        if title_size is None:
            title_size = figsize / len(imgs) * text_scale
        # fig = plt.figure(figsize=(figsize * len(imgs), figsize),frameon=False, facecolor = [0]*4)
        # fig = Figure(figsize=(figsize * len(imgs), figsize),frameon=False, facecolor = [0]*4)
        
        # grid = ImageGrid(fig, 111, nrows_ncols=(1, len(imgs)), axes_pad=spacing, share_all=False)
        fig, axes = figure(nrow=1, ncol=len(imgs), figsize=(figsize * len(imgs), figsize), frameon=False, facecolor = [0]*4)
        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow(img, **kwargs)
            ax.axis("off")
            ax.set_frame_on(False)
            ax.set_facecolor([0]*4)
            if title is not None:
                ax.set_title(title, fontsize=title_size,color=textcolor)
    else:
        if type(figsize) is not (list or tuple):
            figsize = (figsize, figsize)
        if title_size is None:
            title_size = figsize[0] * text_scale
        if ax is None:
            # fig, ax = plt.subplots(frameon=False, figsize=figsize, facecolor =[0]*4,dpi=dpi)
            subplot_args = {'frameon': False,
                            'figsize': figsize,
                            'facecolor': [0, 0, 0, 0],
                            'dpi': dpi
                        }
            fig, ax = figure(**subplot_args)
            # canvas = FigureCanvas(fig)
            
        
        else:
            hold = True
            
        ax.imshow(imgs, **kwargs)
        ax.axis("off")
        ax.set_frame_on(False)
        ax.set_facecolor([0]*4)
        if titles is not None:
            ax.set_title(titles, fontsize=title_size, color=textcolor)
        
        fig = ax.get_figure()
    # if not hold:
        # plt.show()
        # canvas.draw()
        # print('fff')
    return fig 

# def get_cmap(masks):
#     lut = ncolor.get_lut(masks)
#     c = sinebow(lut.max())
#     colors = [c[l] for l in lut]
#     cmap = mpl.colors.ListedColormap(colors)
#     return cmap

# @njit()
# def rgb_flow(dP,transparency=False,mask=None,norm=False):
#     """ dP is 2 x Y x X => 'optic' flow representation 
    
#     Parameters
#     -------------
    
#     dP: NDarray, float
#         Flow field component stack [B,dy,dx]
        
#     transparency: bool, default False
#         magnitude of flow controls opacity, not lightness (clear background)
        
#     mask: 2D array 
#         Multiplies each RGB component to suppress noise
    
#     """

#     mag = np.sqrt(np.sum(dP**2,axis=1))
#     if norm:
#         mag = np.clip(utils.normalize99(mag), 0, 1.).astype(np.float32)
    
#     angles = np.arctan2(dP[:,1], dP[:,0])+np.pi
#     a = 2
#     r = ((np.cos(angles)+1)/a)
#     g = ((np.cos(angles+2*np.pi/3)+1)/a)
#     b = ((np.cos(angles+4*np.pi/3)+1)/a)

#     if transparency:
#         im = np.stack((r,g,b,mag),axis=-1)
#     else:
#         im = np.stack((r*mag,g*mag,b*mag),axis=-1)
        
#     if mask is not None and transparency and dP.shape[0]<3:
#         im[...,-1] *= mask
        
#     im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
#     return im


# from numba import jit

# @jit(nopython=True)
# @njit()
# def rgb_flow(dP, transparency=True, mask=None, norm=True):
#     mag = np.sqrt(np.sum(dP**2,axis=1)).reshape(1, -1)
#     vecs = dP[:,0] + dP[:,1]*1j
#     roots = np.exp(1j * np.pi * (2  * np.arange(3) / 3 +1))
#     rgb = (np.real(roots * vecs.reshape(-1, 1) / np.max(mag)).T + 1 ) / 2
#     if norm:
#         # mag = np.clip(utils.normalize99(mag), 0, 1.).astype(np.float32)
#         mag -= np.min(mag)
#         mag /= np.max(mag)

#     shape = dP.shape
#     newshape = (shape[0], shape[3], shape[2], 3+transparency)
#     # newshape = (shape[0], shape[2], shape[3], 3+transparency)

#     if transparency:
#         im = np.concatenate((rgb, mag), axis=0)
#     else:
#         im = rgb * mag

#     im = (np.clip(im.T.reshape(newshape), 0, 1) * 255).astype(np.uint8)
#     # im = np.swapaxes(im,1,2)
#     return im


# @njit()
# def rgb_flow(dP, transparency=True, mask=None, norm=True):
#     mag = np.sqrt(np.sum(dP**2,axis=1))
#     vecs = dP[:,0] + dP[:,1]*1j
#     roots = np.exp(1j * np.pi * (2  * np.arange(3) / 3 +1)).reshape((1, 1, 1, -1))
#     rgb = (np.real(vecs[...,None]*roots / np.max(mag)) + 1 ) / 2

#     if norm:
#         mag -= np.min(mag)
#         mag /= np.max(mag)

#     shape = dP.shape
#     newshape = (shape[0], shape[2], shape[3], 3+transparency)

#     print(rgb.shape,newshape, mag.shape, vecs.shape)
#     if transparency:
#         im = np.empty(newshape)
#         im[..., :3] = rgb
#         im[..., 3] = mag
#     else:
#         im = rgb * mag

#     im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
#     return im


def rgb_flow(dP, transparency=True, mask=None, norm=True, device=None):
    """Meant for stacks of dP, unsqueeze if using on a single plane."""
    
    import torch
    
    if device is None:
        device = torch.device('cpu')
    
    if isinstance(dP,torch.Tensor):
        device = dP.device
    else:
        dP = torch.from_numpy(dP).to(device)
        
    mag = torch_norm(dP,dim=1)
    vecs = dP[:,0] + dP[:,1]*1j
    roots = torch.exp(1j * np.pi * (2  * torch.arange(3, device=device) / 3 +1)) 
    rgb = (torch.real(vecs.unsqueeze(-1)*roots.view(1, 1, 1, -1) / torch.max(mag)) + 1 ) / 2 

    # f = 1.5
    # rgb /= f
    # rgb += (1-1/f)/2
    
    
    if norm:
        mag -= torch.min(mag)
        mag /= torch.max(mag)
    if transparency:
        im = torch.cat((rgb, mag[..., None]), dim=-1)
    else:
        im = rgb * mag[..., None]

    im = (torch.clamp(im, 0, 1) * 255).type(torch.uint8)
    return im 



def create_colormap(image, labels):
    """
    Create a colormap based on the average color of each label in the image.

    Parameters
    ----------
    image: ndarray
        An RGB image.
    labels: ndarray
        A 2D array of labels corresponding to the image.

    Returns
    -------
    colormap: ndarray
        A colormap where each row is the RGB color for the corresponding label.
    """
    # Ensure the image is in the range 0-255
    image = img_as_ubyte(image)

    # Initialize an array to hold the RGB color for each label
    colormap = np.zeros((labels.max() + 1, 3), dtype=np.uint8)

    # Calculate the average color for each label
    for label in np.unique(labels):
        mask = labels == label
        colormap[label] = image[mask].mean(axis=0)

    return colormap
    

def color_from_RGB(im,rgb,m,bd=None, mode='inner',connectivity=2):
    from skimage import color
    
    if bd is None:
        from skimage.segmentation import find_boundaries
        bd = find_boundaries(m,mode=mode,connectivity=connectivity)
    
    alpha = (m>0)*.5
    alpha[bd] = 1
    alpha = np.stack([alpha]*3,axis=-1)
    m = ncolor.format_labels(m)
    cmap = create_colormap(rgb,m)
    clrs = rescale(cmap[1:])
    overlay = color.label2rgb(m,im,clrs,
                              bg_label=0,
                              alpha=alpha
                            # saturation=1,
                            # kind='overlay',
                            # alpha=1
                              )
    return overlay
    
    
def split_list(lst, N):
    return [lst[i:i + N] for i in range(0, len(lst), N)]
        

# def image_grid(images, column_titles=None, row_titles=None,
               
#                plot_labels=None, 
#                 xticks=[], yticks=[], 
#                 outline=False, outline_color=[0.5]*3, outline_width=.5,
#                 padding=0.05, 
#                 fontsize=8, fontcolor=[0.5]*3,
#                 facecolor=None,
#                 fig_scale=6, dpi=300,
#                 order='ij',
#                 lpad = 0.05,
#                 lpos='top_middle',
#                 return_axes=False,
#                 append_fig=None,
#                 append_axes=None,
#                 **kwargs):
#     """Display a grid of images with uniform spacing.
#     Accepts a neested list of images, with each sublist having cosnsitent YXC diemnsions. 
    
#     """
    
#     label_positions = {'top_middle': {'coords': (0.5, 1-lpad), 'va': 'top', 'ha': 'center'},
#                         'bottom_left': {'coords': (lpad, lpad), 'va': 'bottom', 'ha': 'left'},
#                         'bottom_middle': {'coords': (0.5, lpad), 'va': 'bottom', 'ha': 'center'},
#                         'top_left': {'coords': (lpad, 1-lpad), 'va': 'top', 'ha': 'left'},
#                         # Add more positions as needed
#                     }

#     # get the dimensions of the grid
#     nrow = len(images)
#     ncol = [len(i) for i in images]
#     grid_dims = [nrow,max(ncol)]
#     ij = order=='ij'
#     ji = order=='ji'

#     n,m = grid_dims 
#     # Get the shapes of the images in each row 
#     # (we assume each row has consistent xy dims, hence use the first [0] entry sets the shape, and that the images are YXC)
#     # image_shapes = np.stack([i[0].shape[:2] for i in images])
#     image_shapes = np.stack([i[0].shape[:2] for i in images if i is not None and i[0] is not None])
#     # Padding between images
#     p = padding
    
#     # normalize dimension along row or column 
#     a = list(image_shapes[:,0] / image_shapes[:,1]) if ij else list(image_shapes[:,1] / image_shapes[:,0])
#     b = np.ones_like(a) 
    
#     # Cumulative dimension
#     ca = np.cumsum(a)
    
#     start_a = np.array([[0]*m]+[[(ca[i]+(i+1)*p)]*m for i in range(n-1)]).flatten().astype(float)
#     start_b = np.array([[(bi+p)*i for i in range(m)] for bi in b]).flatten().astype(float)
    
#     # Calculate the positions and sizes of the images in the grid
#     da = np.array([[ai]*m for ai in a]).flatten().astype(float)
#     db = np.array([[bi]*m for bi in b]).flatten().astype(float)
        
#     # Map the variables to their values
#     variables = {'ji': (start_a, start_b, da, db), 'ij': (start_b, start_a, db, da)}

#     # Assign the values to the variables
#     left, bottom, width, height = variables[order]

#     # Normalize the positions and sizes
#     max_w = left[-1]+width[-1]
#     max_h = bottom[-1]+height[-1]
#     left /= max_w
#     bottom /= max_h
#     width /= max_w
#     height /= max_h
    
#     # Create the figure
#     fig = Figure(figsize=(fig_scale,fig_scale*max_h/max_w),                    
#                  frameon=False if facecolor is None else True, 
#                  facecolor=[0]*4 if facecolor is None else facecolor,
#                  dpi=dpi)
    
#     # here m and n need to represent the actual grid layout rather than indexing 
#     if ij:
#         n,m = grid_dims 
#     elif ji:
#         m,n = grid_dims
#     else:
#         raise ValueError('order must be "ij" or "ji"')
    
#     # Add the subplots
#     axes = []
#     for i in range(n*m):
#         # ax = fig.add_axes([left[i], bottom[i], width[i], height[i]])
#         ax = fig.add_axes([left[i], 1-bottom[i]-height[i], width[i], height[i]])

#         axes.append(ax)
    
#     # add outline around each image, remove ticks
#     for i,ax in enumerate(axes):

#         ax.set_xticks(xticks)
#         ax.set_yticks(yticks)
#         ax.patch.set_alpha(0)
        
#         # Display the image
#         j,k = np.unravel_index(i,grid_dims)        
#         if k < ncol[j]: # not all nows may have the same number of images 
#             if images[j][k] is not None:
#                 ax.imshow(images[j][k],**kwargs)
#                 # for s in ax.spines.values():
#                 #     if outline:
#                 #         s.set_color(outline_color)
#                 #         s.set_linewidth(outline_width)
#                 #     # else:
#                 #     #     s.set_visible(False)
                    
#             if outline:
#                 for s in ax.spines.values():
#                     s.set_color(outline_color)
#                     s.set_linewidth(outline_width)
#             else:
#                 for s in ax.spines.values():
#                     s.set_visible(False)

#             if plot_labels is not None and plot_labels[j][k] is not None:
#                 coords = label_positions[lpos]['coords']
#                 va = label_positions[lpos]['va']
#                 ha = label_positions[lpos]['ha']
#                 ax.text(coords[0],coords[1], plot_labels[j][k], fontsize=fontsize, color=fontcolor, 
#                         va=va, ha=ha, transform=ax.transAxes)
#         else:
#             ax.set_visible(False)
            
#         # Set the column titles
#         if column_titles is not None:
#             if ij and i < m:
#                 idx = i
#             elif ji and i % n == 0:
#                 idx = i // n
#             else:
#                 idx = None
#             if idx is not None:
#                 # ax.set_title(column_titles[idx], fontsize=fontsize, c=fontcolor)
#                 ax.text(0.5, 1+p, column_titles[idx], rotation=0, fontsize=fontsize, color=fontcolor, 
#                         va='bottom', ha='center', transform=ax.transAxes)
    
#         # Set the row titles
#         if row_titles is not None:
#             if ij and i % m == 0:
#                 idx = i // m
#             elif ji and i < n:
#                 idx = i
#             else:
#                 idx = None
#             if idx is not None:
#                 ax.text(-p, 0.5, row_titles[idx], rotation=0, fontsize=fontsize, color=fontcolor, 
#                         va='center', ha='right', transform=ax.transAxes)
                
    
#     if return_axes:
#         return fig, axes
#     else:   
#         return fig

def image_grid(images, column_titles=None, row_titles=None, 
               plot_labels=None, 
               xticks=[], yticks=[], 
               outline=False, outline_color=[0.5]*3, outline_width=.5,
               padding=0.05, interset_padding=0.1,
               fontsize=8, fontcolor=[0.5]*3,
               facecolor=None,
               figsize=6, dpi=300,
               order='ij',
               stack_direction='horizontal',  # New parameter for stack direction
               lpad = 0.05,
               lpos='top_middle',
               return_axes=False,
               **kwargs):

    """Display a grid of images with uniform spacing.
    Accepts a list or nested list of images, with each sublist having consistent YXC dimensions. 
    If multiple sets of images are provided, extra padding will be added between sets.
    stack_direction: 'horizontal' or 'vertical' controls how multiple sets are arranged.
    """
    
    label_positions = {'top_middle': {'coords': (0.5, 1-lpad), 'va': 'top', 'ha': 'center'},
                        'bottom_left': {'coords': (lpad, lpad), 'va': 'bottom', 'ha': 'left'},
                        'bottom_middle': {'coords': (0.5, lpad), 'va': 'bottom', 'ha': 'center'},
                        'top_left': {'coords': (lpad, 1-lpad), 'va': 'top', 'ha': 'left'},
                    }

    # Check if 'images' is a list of lists, meaning multiple image sets
    if isinstance(images[0][0], list):
        multiple_sets = True
    else:
        multiple_sets = False
        images = [images]  # Treat single set as a list of one
        plot_labels = [plot_labels] if plot_labels is not None else None
        
    # Compute grid dimensions
    n_sets = len(images)
    nrow = len(images[0])
    ncol = [len(i) for i in images[0]]
    grid_dims = [nrow, max(ncol)]
    ij = order == 'ij'
    
    # Get image shapes from the first set (allowing for variations in shape)
    image_shapes = []
    for i in images[0]:
        if i and i[0] is not None:
            image_shapes.append(i[0].shape[:2])
        else:
            # Assign a default shape (e.g., (1,1)) for missing images
            image_shapes.append((1, 1))

    # Padding between images
    p = padding
    
    # Calculate aspect ratios and cumulative dimensions for the grid
    a = [img_shape[0] / img_shape[1] for img_shape in image_shapes] if ij else [img_shape[1] / img_shape[0] for img_shape in image_shapes]
    b = np.ones_like(a)
    
    # Cumulative dimensions for positioning
    ca = np.cumsum(a)
    start_a = np.array([[0]*grid_dims[1]] + [[ca[i]+(i+1)*p]*grid_dims[1] for i in range(grid_dims[0]-1)]).flatten().astype(float)
    start_b = np.array([[(bi + p)*i for i in range(grid_dims[1])] for bi in b]).flatten().astype(float)
    
    # Positions and sizes for the first set
    da = np.array([[ai]*grid_dims[1] for ai in a]).flatten().astype(float)
    db = np.array([[bi]*grid_dims[1] for bi in b]).flatten().astype(float)
    
    left = np.copy(start_b)
    bottom = np.copy(start_a)
    width = np.copy(db)
    height = np.copy(da)
        
    def adjust_for_multiple_sets(left, bottom, width, height, stack_direction):
        """ Adjust the positions for multiple image sets. """
        if multiple_sets:
            offsets = []
            for i in range(1, n_sets):
                if stack_direction == 'horizontal':
                    set_offset = max(left + width) + interset_padding  # Add space between sets horizontally
                    next_left = np.copy(left) + set_offset * i
                    offsets.append((next_left, bottom, width, height))
                elif stack_direction == 'vertical':
                    set_offset = max(bottom + height) + interset_padding  # Add space between sets vertically
                    next_bottom = np.copy(bottom) + set_offset * i
                    offsets.append((left, next_bottom, width, height))
            
            # Combine positions across sets
            left = np.concatenate([left] + [offset[0] for offset in offsets])
            bottom = np.concatenate([bottom] + [offset[1] for offset in offsets])
            width = np.concatenate([width] + [offset[2] for offset in offsets])
            height = np.concatenate([height] + [offset[3] for offset in offsets])
        
        return left, bottom, width, height
        

    # Adjust positions for multiple sets based on stack_direction
    left, bottom, width, height = adjust_for_multiple_sets(left, bottom, width, height, stack_direction)

    # Normalize the positions and sizes
    max_w = max(left + width)
    max_h = max(bottom + height)
    left /= max_w
    bottom /= max_h
    width /= max_w
    height /= max_h
    
    pos = [left, bottom, width, height]
    
    # Create the figure
    fig = Figure(figsize=(figsize,figsize*max_h/max_w),                    
                 frameon=False if facecolor is None else True, 
                 facecolor=[0]*4 if facecolor is None else facecolor,
                 dpi=dpi)
    
    # Add the subplots
    axes = []
    for i in range(len(left)):
        ax = fig.add_axes([left[i], 1-bottom[i]-height[i], width[i], height[i]])
        axes.append(ax)
    
    # Add outline and remove ticks for each subplot
    for i, ax in enumerate(axes):
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.patch.set_alpha(0)
        
        # Find the correct image based on the grid dimensions and number of sets
        set_idx = i // (grid_dims[0] * grid_dims[1])
        row_idx = (i % (grid_dims[0] * grid_dims[1])) // grid_dims[1]
        col_idx = i % grid_dims[1]
        
        if col_idx < len(images[set_idx][row_idx]):
            img = images[set_idx][row_idx][col_idx]
            if img is not None:
                ax.imshow(img, **kwargs)
            
            # Add plot labels
            if plot_labels is not None and plot_labels[set_idx][row_idx][col_idx] is not None:
                coords = label_positions[lpos]['coords']
                va = label_positions[lpos]['va']
                ha = label_positions[lpos]['ha']
                ax.text(coords[0], coords[1], plot_labels[set_idx][row_idx][col_idx], 
                        fontsize=fontsize, color=fontcolor, va=va, ha=ha, transform=ax.transAxes)
    
        # Set the column titles (only for the top row)
        if column_titles is not None and row_idx == 0:
            if ij and col_idx < grid_dims[1]:
                if stack_direction != 'vertical' or set_idx == 0:
                    ax.text(0.5, 1 + p, column_titles[col_idx], rotation=0, fontsize=fontsize, color=fontcolor, 
                            va='bottom', ha='center', transform=ax.transAxes)

        # Set the row titles (only for the first column and only if stacking is not horizontal or it's the first set)
        if row_titles is not None and col_idx == 0:
            if ij and row_idx < grid_dims[0]:
                if stack_direction != 'horizontal' or set_idx == 0:
                    ax.text(-p, 0.5, row_titles[row_idx], rotation=0, fontsize=fontsize, color=fontcolor, 
                            va='center', ha='right', transform=ax.transAxes)
    
    
        if outline:
            for s in ax.spines.values():
                s.set_color(outline_color)
                s.set_linewidth(outline_width)
        else:
            for s in ax.spines.values():
                s.set_visible(False)
                
    if return_axes:
        return fig, axes, pos
    else:
        return fig
        

def color_grid(colors, **kwargs):
    # Convert colors to a numpy array
    colors = np.array(colors)
    
    # If colors is a 1D array (single color), reshape it to a 2D array
    if colors.ndim == 1:
        colors = colors.reshape(1, -1)
    
    # Ensure colors have 3 components (RGB)
    if colors.shape[-1] == 4:
        # If colors have 4 components (RGBA), remove the alpha component
        colors = colors[:, :3]
    
    # Create a list of 1x1 images
    images = [[np.full((1, 1, 3), color, dtype=np.float32)] for color in colors]
    
    # Display the image grid
    return image_grid(images, **kwargs)





    
# from https://stackoverflow.com/a/63530703/13326811



def colored_line_segments(xs,ys,zs=None,color='k',mid_colors=False):

    from scipy.interpolate import interp1d
    from matplotlib.colors import colorConverter

    if isinstance(color,str):
        color = colorConverter.to_rgba(color)[:-1]
        color = np.array([color for i in range(len(xs))])   
    segs = []
    seg_colors = []    
    lastColor = [color[0][0],color[0][1],color[0][2]]        
    start = [xs[0],ys[0]]
    end = [xs[0],ys[0]]        
    if not zs is None:
        start.append(zs[0])
        end.append(zs[0])     
    else:
        zs = [zs]*len(xs)            
    for x,y,z,c in zip(xs,ys,zs,color):
        if mid_colors:
            seg_colors.append([(chan+lastChan)*.5 for chan,lastChan in zip(c,lastColor)])        
        else:   
            seg_colors.append(c)        
        lastColor = c[:-1]           
        if not z is None:
            start = [end[0],end[1],end[2]]
            end = [x,y,z]
        else:
            start = [end[0],end[1]]
            end = [x,y]                 
        segs.append([start,end])               
    colors = [(*color,1) for color in seg_colors]    
    return segs, colors

def segmented_resample(xs,ys,zs=None,color='k',n_resample=100,mid_colors=False):   
    from scipy.interpolate import interp1d
    from matplotlib.colors import colorConverter 
    n_points = len(xs)
    if isinstance(color,str):
        color = colorConverter.to_rgba(color)[:-1]
        color = np.array([color for i in range(n_points)])   
    n_segs = (n_points-1)*(n_resample-1)        
    xsInterp = np.linspace(0,1,n_resample)    
    segs = []
    seg_colors = []
    hiResXs = [xs[0]]
    hiResYs = [ys[0]]    
    if not zs is None:
        hiResZs = [zs[0]]        
    RGB = color.swapaxes(0,1)
    for i in range(n_points-1):        
        fit_xHiRes = interp1d([0,1],xs[i:i+2])
        fit_yHiRes = interp1d([0,1],ys[i:i+2])        
        xHiRes = fit_xHiRes(xsInterp)
        yHiRes = fit_yHiRes(xsInterp)    
        hiResXs = hiResXs+list(xHiRes[1:])
        hiResYs = hiResYs+list(yHiRes[1:])   
        R_HiRes = interp1d([0,1],RGB[0][i:i+2])(xsInterp)        
        G_HiRes = interp1d([0,1],RGB[1][i:i+2])(xsInterp)      
        B_HiRes = interp1d([0,1],RGB[2][i:i+2])(xsInterp)                               
        lastColor = [R_HiRes[0],G_HiRes[0],B_HiRes[0]]                
        start = [xHiRes[0],yHiRes[0]]
        end = [xHiRes[0],yHiRes[0]]           
        if not zs is None:
            fit_zHiRes = interp1d([0,1],zs[i:i+2])             
            zHiRes = fit_zHiRes(xsInterp)             
            hiResZs = hiResZs+list(zHiRes[1:]) 
            start.append(zHiRes[0])
            end.append(zHiRes[0])                
        else:
            zHiRes = [zs]*len(xHiRes) 
            
        if mid_colors: seg_colors.append([R_HiRes[0],G_HiRes[0],B_HiRes[0]])        
        for x,y,z,r,g,b in zip(xHiRes[1:],yHiRes[1:],zHiRes[1:],R_HiRes[1:],G_HiRes[1:],B_HiRes[1:]):
            if mid_colors:
                seg_colors.append([(chan+lastChan)*.5 for chan,lastChan in zip((r,g,b),lastColor)])
            else:            
                seg_colors.append([r,g,b])            
            lastColor = [r,g,b]            
            if not z is None:
                start = [end[0],end[1],end[2]]
                end = [x,y,z]  
            else:
                start = [end[0],end[1]]
                end = [x,y]                
            segs.append([start,end])

    colors = [(*color,1) for color in seg_colors]    
    data = [hiResXs,hiResYs] 
    if not zs is None:
        data = [hiResXs,hiResYs,hiResZs] 
    return segs, colors, data      

def faded_segment_resample(xs,ys,zs=None,color='k',fade_len=20,n_resample=100,direction='Head'):      
    segs, colors, hiResData = segmented_resample(xs,ys,zs,color,n_resample)    
    n_segs = len(segs)   
    if fade_len>len(segs):
        fade_len=n_segs    
    if direction=='Head':
        #Head fade
        alphas = np.concatenate((np.zeros(n_segs-fade_len),np.linspace(0,1,fade_len)))
    else:        
        #Tail fade
        alphas = np.concatenate((np.linspace(1,0,fade_len),np.zeros(n_segs-fade_len)))
    colors = [(*color[:-1],alpha) for color,alpha in zip(colors,alphas)]
    return segs, colors, hiResData 
    

# https://stackoverflow.com/a/27537018/13326811 
def _get_perp_line(current_seg, out_of_page, linewidth):
    perp = np.cross(current_seg, out_of_page)[0:2]
    perp_unit = _get_unit_vector(perp)
    current_seg_perp_line = perp_unit*linewidth
    return current_seg_perp_line

def _get_unit_vector(vector):
    vector_size = (vector[0]**2 + vector[1]**2)**0.5
    vector_unit = vector / vector_size
    return vector_unit[0:2]


def colored_line(x, y, ax, z=None, line_width=1, MAP='jet'):
    # use pcolormesh to make interpolated rectangles
    num_pts = len(x)
    [xs, ys, zs] = [
        np.zeros((num_pts,2)),
        np.zeros((num_pts,2)),
        np.zeros((num_pts,2))
    ]

    dist = 0
    out_of_page = [0, 0, 1]
    for i in range(num_pts):
        # set the colors and the x,y locations of the source line
        xs[i][0] = x[i]
        ys[i][0] = y[i]
        if i > 0:
            x_delta =  x[i] - x[i-1]
            y_delta =  y[i] - y[i-1]
            seg_length = (x_delta**2 + y_delta**2)**0.5
            dist += seg_length
            zs[i] = [dist, dist]

        # define the offset perpendicular points
        if i == num_pts - 1:
            current_seg = [x[i]-x[i-1], y[i]-y[i-1], 0]
        else:
            current_seg = [x[i+1]-x[i], y[i+1]-y[i], 0]
        current_seg_perp = _get_perp_line(
            current_seg, out_of_page, line_width)
        if i == 0 or i == num_pts - 1:
            xs[i][1] = xs[i][0] + current_seg_perp[0]
            ys[i][1] = ys[i][0] + current_seg_perp[1]
            continue
        current_pt = [x[i], y[i]]
        current_seg_unit = _get_unit_vector(current_seg)
        previous_seg = [x[i]-x[i-1], y[i]-y[i-1], 0]
        previous_seg_perp = _get_perp_line(
            previous_seg, out_of_page, line_width)
        previous_seg_unit = _get_unit_vector(previous_seg)
        # current_pt + previous_seg_perp + scalar * previous_seg_unit =
        # current_pt + current_seg_perp - scalar * current_seg_unit =
        scalar = (
            (current_seg_perp - previous_seg_perp) /
            (previous_seg_unit + current_seg_unit)
        )
        new_pt = current_pt + previous_seg_perp + scalar[0] * previous_seg_unit
        xs[i][1] = new_pt[0]
        ys[i][1] = new_pt[1]

    # fig, ax = plt.subplots()
    # cm = cm.get_cmap(MAP)
    cm = mpl.colormaps[MAP]
    
    ax.pcolormesh(xs, ys, zs, shading='gouraud', cmap=cm)

def color_swatches(colors, figsize=0.5, dpi=150, fontsize=5, fontcolor='w', padding=0.05, 
                   titles=None, ncol=None):
    if ncol is None:
        ncol = len(colors)
    # Convert colors to a numpy array
    colors = np.array(colors)
    
    # If colors is a 1D array (single color), reshape it to a 2D array
    if colors.ndim == 1:
        colors = colors.reshape(1, -1)
    
    # Create a list of swatches
    swatches = [np.full((1, 1, 3), color, dtype=np.float32) for color in colors]
    
    # Display the swatches
    # return imshow(swatches, figsize=figsize, dpi=dpi, titles=titles)
    return image_grid(split_list(swatches,ncol),
                        plot_labels=split_list(titles,ncol) if titles is not None else None,
                      padding=0.05, fontsize=fontsize, 
                      fontcolor=fontcolor,
                      facecolor=[0]*4, figsize=figsize*ncol, dpi=dpi)
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):

    from matplotlib.colors import LinearSegmentedColormap

    cmap=mpl.colormaps[cmap] if isinstance(cmap, str) else cmap

    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


from operator import sub
def get_aspect(ax):
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio
    
from .utils import kernel_setup, get_neighbors
from .core import boundary_to_masks, masks_to_affinity, get_contour
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from skimage.segmentation import find_boundaries
from scipy.interpolate import splprep, splev
from matplotlib.collections import PatchCollection

def vector_contours(fig,ax,mask, smooth_factor=5, color = 'r', linewidth=1):
    pad = 1
    msk = np.pad(mask,pad,mode='constant')


    # set up dimensions
    dim = msk.ndim
    shape = msk.shape
    steps,inds,idx,fact,sign = kernel_setup(dim)

    # remove spur points - this method is way easier than running core._despur() on the priginal affinity graph
    bd = find_boundaries(msk,mode='inner',connectivity=2)
    msk, bounds, _ = boundary_to_masks(bd,binary_mask=msk>0,connectivity=1,min_size=0) 

    # generate affinity graph
    coords = np.nonzero(msk)
    affinity_graph =  masks_to_affinity(msk, coords, steps, inds, idx, fact, sign, dim)
    neighbors = get_neighbors(tuple(coords),steps,dim,shape) # shape (d,3**d,npix)

    # find contours 
    contour_map, contour_list, unique_L = get_contour(msk,
                                                    affinity_graph,
                                                    coords,
                                                    neighbors,
                                                    cardinal_only=1)

    # List to hold patches
    patches = []
    for contour in contour_list:
        if len(contour) > 1:
            pts = np.stack([c[contour] for c in coords]).T
            tck, u = splprep(pts.T, u=None, s=len(pts)/smooth_factor, per=1) 
            u_new = np.linspace(u.min(), u.max(), len(pts))
            x_new, y_new = splev(u_new, tck, der=0) 

            # Define the points of the polygon
            points = np.column_stack([y_new-pad, x_new-pad])
            
            # Create a Path from the points
            path = mpath.Path(points, closed=True)
            
            # Create a PathPatch from the Path
            patch = mpatches.PathPatch(path, fill=None, edgecolor=color, 
                                    #    linewidth= fig.dpi/72, 
                                        linewidth=linewidth,
                                       capstyle='round')
            
            # ax.add_patch(patch)
            
            # Add patch to list
            patches.append(patch)

    # Create a PatchCollection from the list of patches
    # Add the PatchCollection to the axis/axes
    if isinstance(ax,list):
        for a in ax:
            patch_collection = PatchCollection(patches, match_original=True)
            a.add_collection(patch_collection)
    else:
        patch_collection = PatchCollection(patches, match_original=True)
        ax.add_collection(patch_collection)