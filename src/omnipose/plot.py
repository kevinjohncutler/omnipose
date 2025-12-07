from .utils import rescale, torch_norm
from .color import sinebow

import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'  # keep text as real text in the SVG
mpl.rcParams['text.usetex'] = False      # Avoid LaTeX (which converts text to paths)

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.transforms import Bbox
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.path import Path
# import matplotlib.pyplot as plt

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
    from tqdm.notebook import tqdm as notebook_tqdm  # progress bars
    
    # Custom CSS to center plots
    # and make widget backgrounds transparent for VS Code
    display(HTML("""
    <style>
        .jp-OutputArea-output img {
            display: block;
            margin: 0 auto;
        }
        .cell-output-ipywidget-background {
            background-color: transparent !important;
        }
        .jp-OutputArea,
        .jp-OutputArea-child,
        .jp-OutputArea-output,
        .jp-Cell-outputWrapper,
        .jp-Cell-outputArea {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        :root {
            --jp-widgets-color: var(--vscode-editor-foreground, currentColor);
            --jp-widgets-font-size: var(--vscode-editor-font-size, inherit);
        }

        .widget-hprogress {
            background-color: transparent !important;
            border: none !important;
            display: inline-flex !important;
            justify-content: center;
            align-items: center;
        }
        .widget-hprogress .p-ProgressBar,
        .widget-hprogress .p-ProgressBar-track,
        .widget-hprogress .widget-progress .progress,
        .widget-hprogress .progress {
            background-color: rgba(128, 128, 128, 0.5) !important;
            border-radius: 6px !important;
            border: none !important;
            box-sizing: border-box;
            overflow: hidden;
        }
        .widget-hprogress .progress-bar,
        .widget-hprogress .p-ProgressBar-fill,
        .widget-hprogress [role="progressbar"]::part(value) {
            background-color: #8a8a8a !important;
            border-radius: 6px !important;
        }
    </style>
    """))

    # Ensure notebook tqdm bars pick up the neutral grey fill + subtle track
    def _patch_tqdm_progress():
        if getattr(notebook_tqdm, "_omnipose_bar_styled", False):
            return

        default_fill = "#8a8a8a"
        original_status_printer = notebook_tqdm.status_printer

        def _status_printer(*args, **kwargs):
            container = original_status_printer(*args, **kwargs)
            try:
                _, pbar, _ = container.children
            except Exception:
                return container

            style = getattr(pbar, "style", None)
            if style is not None:
                style.bar_color = default_fill

            return container

        notebook_tqdm.status_printer = staticmethod(_status_printer)
        notebook_tqdm._omnipose_bar_styled = True

    _patch_tqdm_progress()
    
    # Inject into the global namespace of the notebook
    ipython = get_ipython()  # Get the IPython instance
    ipython.user_global_ns['mpl'] = mpl
    ipython.user_global_ns['plt'] = plt
    ipython.user_global_ns['widgets'] = widgets
    ipython.user_global_ns['display'] = display
    ipython.user_global_ns['tqdm'] = notebook_tqdm

    # Set matplotlib inline for Jupyter notebooks
    ipython.run_line_magic('matplotlib', 'inline')

    # Define rc_params
    rc_params = {
        'figure.dpi': 300,
        'figure.figsize': (2, 2),      
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
        'axes.edgecolor': 'gray',
        # Legend defaults - place legend outside axes on the right, no frame
        'legend.loc': 'center left',
        'legend.frameon': False,
        'legend.framealpha': 0,
        'legend.borderaxespad': 0.0,
        'lines.scale_dashes': False 
    }

    # Update rcParams
    mpl.rcParams.update(rc_params)

    # Monkey-patch Axes.legend to default to outside-right placement with no frame
    from matplotlib.axes import Axes as _Axes
    _orig_legend = _Axes.legend
    def _legend(self, *args, **kwargs):
        kwargs.setdefault('loc', 'center left')
        kwargs.setdefault('bbox_to_anchor', (1.02, 0.5))
        kwargs.setdefault('frameon', False)
        kwargs.setdefault('framealpha', 0)
        kwargs.setdefault('borderaxespad', 0.0)
        return _orig_legend(self, *args, **kwargs)
    _Axes.legend = _legend


def _get_sinebow():
    """Get sinebow function lazily."""
    if 'sinebow' not in globals():
        from .color import sinebow
        globals()['sinebow'] = sinebow
    return globals()['sinebow']

def _get_rescale():
    """Get rescale function lazily."""
    if 'rescale' not in globals():
        from .utils import rescale
        globals()['rescale'] = rescale
    return globals()['rescale']

def _get_torch_norm():
    """Get torch_norm function lazily."""
    if 'torch_norm' not in globals():
        from .utils import torch_norm
        globals()['torch_norm'] = torch_norm
    return globals()['torch_norm']

# Make functions available at module level for backward compatibility
def __getattr__(name):
    """Handle lazy loading of commonly used functions."""
    if name == 'sinebow':
        return _get_sinebow()
    elif name == 'rescale':
        return _get_rescale()
    elif name == 'torch_norm':
        return _get_torch_norm()
    raise AttributeError(f"module 'omnipose.plot' has no attribute '{name}'")


def figure(nrow=None, ncol=None, aspect=1, **kwargs):
    figsize = kwargs.get('figsize', 2)
    if not isinstance(figsize, (list, tuple, np.ndarray)) and figsize is not None:
        figsize = (figsize*aspect, figsize)
        
    kwargs['figsize'] = figsize
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
        # if type(figsize) is not (list or tuple):
        if not isinstance(figsize, (list, tuple)):
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
    else:
        return None,None
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
    
import numpy as np
import matplotlib as mpl
import types
from matplotlib.collections import LineCollection
from matplotlib.backend_bases import RendererBase
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def plot_edges(
        shape,
        affinity_graph,
        neighbors,
        coords,
        figsize=1,
        fig=None,
        ax=None,
        extent=None,
        slc=None,
        pic=None,
        edgecol=[.75]*3 + [.5],
        linewidth=0.15,
        step_inds=None,
        cmap='inferno',
        origin='lower',
        bounds=None,
):
    """
    Render an affinity graph as line segments laid over an optional image.

    Boundary pixels (including linear index 0) are handled explicitly, so every
    valid edge appears—even when its target lies on the image border.
    """
    # ——————————————————————————————————————————— imports that take time kept local
    from .utils import get_neigh_inds
    from .core import affinity_to_edges  # retained in case callers expect it

    nstep, npix = affinity_graph.shape
    coords = tuple(coords)

    # build lookup tables for neighbours
    indexes, neigh_inds, ind_matrix = get_neigh_inds(tuple(neighbors), coords, shape)

    # default to all steps if none supplied
    if step_inds is None:
        step_inds = np.arange(nstep)

    px_inds = np.arange(npix)

    # -------------------------------------------------------------------------
    # Build edge list manually so edges touching the border are never lost
    # -------------------------------------------------------------------------
    aff_coords = np.array(coords).T                # (2, N) -> (y, x)
    segments = []

    for s in step_inds:
        mask = affinity_graph[s].astype(bool)      # where an edge exists
        if not mask.any():
            continue

        src_idx = px_inds[mask]
        dst_idx = neigh_inds[s, mask]

        valid = dst_idx >= 0                       # drop out-of-bounds neighbours
        src_idx = src_idx[valid]
        dst_idx = dst_idx[valid]

        for a, b in zip(src_idx, dst_idx):
            # flip Y/X order for imshow coords and shift to pixel-centres (+0.5)
            segments.append(aff_coords[:, ::-1][[a, b]] + 0.5)

    if not segments:
        raise ValueError("No edges found to plot; check affinity_graph and neighbours.")

    segments = np.stack(segments)

    # -------------------------------------------------------------------------
    # Figure / axes handling
    # -------------------------------------------------------------------------
    RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)

    newfig = fig is None and ax is None
    if newfig:
        if not isinstance(figsize, (list, tuple)):
            figsize = (figsize, figsize)
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)

    if extent is None:
        extent = np.array([0, shape[1], 0, shape[0]])

    # -------------------------------------------------------------------------
    # Background image (affinity heat-map) - create if not supplied
    # -------------------------------------------------------------------------
    nopic = pic is None
    if nopic:
        summed_affinity = np.zeros(shape, dtype=int)
        summed_affinity[coords] = np.sum(affinity_graph, axis=0)

        # build a visually pleasing reversed colormap
        colors = mpl.colormaps.get_cmap(cmap).reversed()(np.linspace(0, 1, 9))
        colors = np.vstack((np.array([0]*4), colors))  # prepend transparent/black
        affinity_cmap = mpl.colors.ListedColormap(colors)
        pic = affinity_cmap(summed_affinity)

    ax.imshow(pic[slc] if slc is not None else pic,
              extent=extent,
              origin=origin)

    # -------------------------------------------------------------------------
    # Draw edges
    # -------------------------------------------------------------------------
    line_segments = LineCollection(segments, color=edgecol, linewidths=linewidth)
    ax.add_collection(line_segments)

    if newfig:
        ax.set_axis_off()
        ax.invert_yaxis()
        canvas = FigureCanvas(fig)
        canvas.draw()

    # -------------------------------------------------------------------------
    # Return values mirror original signature
    # -------------------------------------------------------------------------
    if nopic:
        return summed_affinity, affinity_cmap
    return None, None


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


# def colorize_GPU(im, colors=None, color_weights=None, offset=0, channel_axis=-1):

#     import torch 
    
#     N = im.shape[0]
#     device = im.device

#     if colors is None:
#         angle = torch.linspace(0, 1, N, device=device) * 2 * np.pi + offset
#         angles = torch.stack((angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3), dim=-1)
#         colors = (torch.cos(angles) + 1) / 2

#     if color_weights is not None:
#         colors *= color_weights.unsqueeze(-1)
#         # colors /= color_weights.sum()
    
#     im = im.unsqueeze(-1)  # Add an extra dimension to `im`
    
#     # Perform the multiplication and mean computation using `einsum` - way faster than using view
#     rgb = torch.einsum('ijkl,il->jkl', im.float(), colors.float()) / N

#     return rgb

# def colorize_GPU(im, colors=None, color_weights=None, offset=0, intervals=None):
#     import torch
#     import string
    
#     N = im.shape[0]  # Number of channels
#     device = im.device

#     if colors is None:
#         angle = torch.linspace(0, 1, N, device=device) * 2 * np.pi + offset
#         angles = torch.stack((angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3), dim=-1)
#         colors = (torch.cos(angles) + 1) / 2  # Generate RGB colors

#     if color_weights is not None:
#         colors *= color_weights.unsqueeze(-1)  # Apply color weights to colors

#     # Determine the number of spatial dimensions
#     num_spatial_dims = im.ndim - 1  # Exclude the channel dimension

#     # Generate index letters for einsum (excluding 'c' and 'l')
#     idx_letters = ''.join(letter for letter in string.ascii_lowercase if letter not in {'c', 'l'})

#     spatial_indices = idx_letters[:num_spatial_dims]

#     # Build the einsum equation dynamically
#     im_indices = 'c' + spatial_indices
#     colors_indices = 'c l'  # 'l' corresponds to the RGB channels
#     output_indices = spatial_indices + 'l'
#     einsum_eq = f'{im_indices},{colors_indices}->{output_indices}'


#     # print('einsum_eq:',einsum_eq)
#     # Perform the weighted sum across channels to produce the RGB image
#     rgb = torch.einsum(einsum_eq, im.float(), colors.float()) / N

#     return rgb



def colorize_GPU(im, colors=None, color_weights=None, intervals=None, offset=0):
    import torch
    import numpy as np
    import string
    from opt_einsum import contract
    

    C = im.shape[0]  # Number of input channels
    device = im.device

    # Determine the number of spatial dimensions
    num_spatial_dims = im.ndim - 1  # Exclude the channel dimension

    # Generate index letters for einsum (excluding 'c' and 'l')
    idx_letters = ''.join(letter for letter in string.ascii_lowercase if letter not in {'c', 'l'})
    spatial_indices = idx_letters[:num_spatial_dims]

    # Build einsum indices dynamically
    im_indices = 'c' + spatial_indices
    aggregator_indices = 'cN'  # 'N' corresponds to the interval/bin dimension
    colors_indices = 'cl'  # Colors indexed by channel and RGB
    output_indices = 'N' + spatial_indices + 'l'
    einsum_eq = f'{im_indices},{aggregator_indices},{colors_indices}->{output_indices}'
    # print('einsum_eq:', einsum_eq)


    if intervals is None:
        intervals = [C]

    N = len(intervals)  # Number of intervals
    
    aggregator = torch.zeros(C, N, device=device)
    start = 0
    for i, length in enumerate(intervals):
        aggregator[start:start + length, i] = 1 / length
        start += length

    # Generate colors if not provided
    if colors is None:
        angle = torch.linspace(0, 1, C, device=device) * 2 * np.pi + offset
        angles = torch.stack((angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3), dim=-1)
        colors = (torch.cos(angles) + 1) / 2  # Generate RGB colors for intervals or channels


    # Apply color weights if provided
    if color_weights is not None:
        colors *= color_weights.unsqueeze(-1)

    # Perform einsum operation
    # rgb = torch.einsum(einsum_eq, im.float(), aggregator.float(), colors.float())
    rgb = contract(einsum_eq, im.float(), aggregator.float(), colors.float()) # big difference on CPU
    
    

    # Squeeze the interval dimension if no intervals are used
    if N==1:
        rgb = rgb.squeeze(0)  # Shape: [spatial_dims..., 3]

    return rgb


def colorize_dask(im_dask, colors=None, color_weights=None, intervals=None, offset=0):
    import dask.array as da
    import numpy as np

    # Get the channel count and spatial shape.
    C = im_dask.shape[0]
    spatial_shape = im_dask.shape[1:]
    spatial_size = np.prod(spatial_shape)
    
    # If intervals is not provided, treat the entire channel set as one interval.
    if intervals is None:
        intervals = [C]
    N = len(intervals)
    
    # Build aggregator matrix of shape (C, N)
    aggregator = np.zeros((C, N), dtype=np.float32)
    start = 0
    for i, size in enumerate(intervals):
        aggregator[start:start + size, i] = 1.0 / size
        start += size

    # Create default colors if not provided; shape will be (C, 3)
    if colors is None:
        angle = np.linspace(0, 1, C, endpoint=False) * 2 * np.pi + offset
        angles = np.stack([angle, angle + 2*np.pi/3, angle + 4*np.pi/3], axis=-1)
        colors = (np.cos(angles) + 1.0) / 2.0
    if color_weights is not None:
        colors *= color_weights[:, None]

    # Combine aggregator and colors: shape (C, N, 3)
    aggregator_colors = aggregator[..., None] * colors[:, None, :]
    
    # Reshape aggregator_colors to (C, N*3)
    agg_col_reshaped = aggregator_colors.reshape(C, N * 3)
    
    # Reshape the dask array to (C, Z*Y*X)
    # im_flat = im_dask.reshape(C, -1,limit='')
    im_flat = im_dask.reshape(C, -1)
    
    
    # Contract over the channel dimension using dot:
    # Compute a dot product: (N*3, Z*Y*X) = (C, N*3).T dot (C, Z*Y*X)
    out_flat = da.dot(agg_col_reshaped.T, im_flat)
    
    # Reshape the output to (N, 3, Z, Y, X)
    out_reshaped = out_flat.reshape(N, 3, *spatial_shape)
    # Move the channel axis to the end: (N, Z, Y, X, 3)
    out_final = da.moveaxis(out_reshaped, 1, -1)
    
    # For a single interval, squeeze out the interval dimension
    if N == 1:
        out_final = out_final.squeeze(axis=0)
        
    return out_final
    
def colorize_dask_fast(im_dask, colors=None, color_weights=None, intervals=None, offset=0):
    import dask.array as da
    import numpy as np

    C = im_dask.shape[0]
    spatial_shape = im_dask.shape[1:]
    spatial_size = np.prod(spatial_shape)

    if intervals is None:
        intervals = [C]
    N = len(intervals)

    # Precompute aggregator matrix (C, N)
    aggregator = np.zeros((C, N), dtype=np.float32)
    start = 0
    for i, size in enumerate(intervals):
        aggregator[start:start + size, i] = 1.0 / size
        start += size

    # Compute colors (C, 3)
    if colors is None:
        angle = np.linspace(0, 1, C, endpoint=False) * 2 * np.pi + offset
        angles = np.stack([angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3], axis=-1)
        colors = (np.cos(angles) + 1.0) / 2.0
    if color_weights is not None:
        colors *= color_weights[:, None]

    # Precompute final weighting matrix: (C, N, 3)
    weights = aggregator[..., None] * colors[:, None, :]  # shape (C, N, 3)

    # Collapse color dimensions early: (C, N*3)
    weights_flat = weights.reshape(C, N * 3)

    # Flatten input to shape (C, ZYX)
    im_flat = im_dask.reshape(C, -1)

    # Matrix multiplication: (N*3, ZYX)
    out_flat = da.dot(weights_flat.T, im_flat)

    # Reshape to (N, 3, Z, Y, X)
    out = out_flat.reshape(N, 3, *spatial_shape)

    # Move color channel to last axis: (N, Z, Y, X, 3)
    out = da.moveaxis(out, 1, -1)

    # If single interval, squeeze it out
    if N == 1:
        out = out.squeeze(axis=0)

    return out
    

    
# def colorize_dask_2(im_dask, colors=None, color_weights=None, intervals=None, offset=0): slow
#     import dask.array as da
#     import numpy as np

#     # Determine the number of channels
#     C = im_dask.shape[0]
#     if intervals is None:
#         intervals = [C]
#     N = len(intervals)

#     # Build the aggregator matrix (C, N)
#     aggregator = np.zeros((C, N), dtype=np.float32)
#     start = 0
#     for i, size in enumerate(intervals):
#         aggregator[start:start+size, i] = 1.0 / size
#         start += size

#     # Generate default colors if not provided (shape: (C, 3))
#     if colors is None:
#         angle = np.linspace(0, 1, C, endpoint=False) * 2 * np.pi + offset
#         angles = np.stack([angle, angle + 2*np.pi/3, angle + 4*np.pi/3], axis=-1)
#         colors = (np.cos(angles) + 1.0) / 2.0
#     if color_weights is not None:
#         colors *= color_weights[:, None]

#     # Compute aggregator_colors: shape (C, N, 3)
#     aggregator_colors = aggregator[..., None] * colors[:, None, :]

#     # Use tensordot to contract the channel axis
#     # im_dask has shape (C, ...); aggregator_colors has shape (C, N, 3)
#     # tensordot over axis 0 will yield an output of shape (..., N, 3)
#     out = da.tensordot(im_dask, aggregator_colors, axes=([0], [0]))
    
#     # Rearrange axes: move the N-axis (currently second-to-last) to the front,
#     # so the output shape becomes (N, ..., 3)
#     out = da.moveaxis(out, -2, 0)
    
#     # If only one interval is used, squeeze out the extra dimension
#     if N == 1:
#         out = out.squeeze(axis=0)
        
#     return out

def colorize_dask(im_dask, colors=None, color_weights=None, intervals=None, offset=0): # slower
    import dask.array as da
    import numpy as np
    from opt_einsum import contract

    # Determine the number of channels
    C = im_dask.shape[0]
    if intervals is None:
        intervals = [C]
    N = len(intervals)
    
    # Build the aggregator matrix of shape (C, N)
    aggregator = np.zeros((C, N), dtype=np.float32)
    start = 0
    for i, size in enumerate(intervals):
        aggregator[start:start+size, i] = 1.0 / size
        start += size

    # Generate default colors if none provided; result shape is (C, 3)
    if colors is None:
        angle = np.linspace(0, 1, C, endpoint=False) * 2 * np.pi + offset
        angles = np.stack([angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3], axis=-1)
        colors = (np.cos(angles) + 1.0) / 2.0

    # Apply any provided color weights
    if color_weights is not None:
        colors *= color_weights[:, None]

    # Combine aggregator and colors to form an array of shape (C, N, 3)
    aggregator_colors = aggregator[..., None] * colors[:, None, :]

    # Use dask.array.einsum to perform the colorization
    # out = da.einsum('c..., cnr -> n...r', im_dask, aggregator_colors)
    out = contract('c..., cnr -> n...r', im_dask, aggregator_colors)
    
    
    # If only one interval is used, squeeze out the extra axis
    if N == 1:
        out = out.squeeze(axis=0)
        
    return out
    
def colorize_dask(im_dask, colors=None, color_weights=None, intervals=None, offset=0):
    import numpy as np
    from opt_einsum import contract
    import dask

    # Number of channels
    C = im_dask.shape[0]
    spatial_shape = im_dask.shape[1:]
    spatial_size = np.prod(spatial_shape)

    # Interval setup
    if intervals is None:
        intervals = [C]
    N = len(intervals)

    # Build aggregator: shape (C, N)
    aggregator = np.zeros((C, N), dtype=np.float32)
    start = 0
    for i, size in enumerate(intervals):
        aggregator[start : start + size, i] = 1.0 / size
        start += size

    # Default color generation: shape (C, 3)
    if colors is None:
        angle = np.linspace(0, 1, C, endpoint=False) * 2 * np.pi + offset
        angles = np.stack([angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3], axis=-1)
        colors = (np.cos(angles) + 1.0) / 2.0

    # Apply any color weights
    if color_weights is not None:
        colors *= color_weights[:, None]

    # Combine aggregator and colors: shape (C, N, 3)
    # Keep this as a NumPy array to avoid a big Dask overhead
    agg_colors = aggregator[..., None] * colors[:, None, :]

    # Flatten input from (C, Z, Y, X) -> (C, Z*Y*X)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    
        im_flat = im_dask.reshape(C, spatial_size)

    # Perform a single einsum contraction:
    # cX * cNr -> NXr
    # c -> channel axis, X -> flattened spatial axis, N-> interval groups, r -> RGB
    out_flat = contract('cX,cNr->NXr', im_flat, agg_colors)

    # Reshape from (N, X, 3) -> (N, Z, Y, X, 3)
    out = out_flat.reshape(N, *spatial_shape, 3)

    # For a single interval, remove the interval dimension
    if N == 1:
        out = out[0]  # shape (Z, Y, X, 3)

    return out
    

def colorize_dask_matmul(im_dask, colors=None, color_weights=None, intervals=None, offset=0):
    """
    A faster version of colorize_dask that uses a single matrix multiply instead
    of explicit loops or opt_einsum for the core contraction step.
    """
    import numpy as np
    import dask
    
    
    # Number of channels
    C = im_dask.shape[0]
    spatial_shape = im_dask.shape[1:]
    spatial_size = np.prod(spatial_shape)

    # Interval setup
    if intervals is None:
        intervals = [C]
    N = len(intervals)

    # Build aggregator: shape (C, N)
    aggregator = np.zeros((C, N), dtype=np.float32)
    start = 0
    for i, size in enumerate(intervals):
        aggregator[start : start + size, i] = 1.0 / size
        start += size

    # Default color generation: shape (C, 3)
    if colors is None:
        angle = np.linspace(0, 1, C, endpoint=False) * 2 * np.pi + offset
        angles = np.stack([angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3], axis=-1)
        colors = (np.cos(angles) + 1.0) / 2.0  # shape (C, 3)

    # Apply any color weights
    if color_weights is not None:
        colors = colors * color_weights[:, None]

    # aggregator (C, N), colors (C, 3)
    # aggregator[..., None] * colors => shape (C, N, 3)
    # then collapse to shape (C, N*3) so that a single matrix multiply can be used
    combined = (aggregator[..., None] * colors[:, None, :]).reshape(C, N * 3).astype(np.float32)

    # Flatten the input image: (C, Z, Y, X) -> (C, Z*Y*X)
    # Casting to float32 can help ensure everything matches for the matrix multiply
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        im_flat = im_dask.reshape(C, spatial_size, merge_chunks=True).astype(np.float32)
        # im_flat = da.reshape(im_dask, (C, spatial_size), merge_chunks=True).astype(np.float32)
        

    # Matrix multiplication:
    #   im_flat^T is shape (X, C)
    #   combined is shape (C, N*3)
    # => result is shape (X, N*3)
    out_mat = im_flat.T @ combined

    # Reshape:
    #   out_mat:  (X, N*3)
    #   => (X, N, 3) => transpose to (N, X, 3)
    out_flat = out_mat.reshape(spatial_size, N, 3).transpose(1, 0, 2)

    # Finally shape it to (N, Z, Y, X, 3)
    out = out_flat.reshape(N, *spatial_shape, 3)

    # If only one interval, remove that dimension to keep the same behavior
    if N == 1:
        out = out[0]  # shape (Z, Y, X, 3)

    return out

def apply_ncolor(masks,offset=0,cmap=None,max_depth=20,expand=True, maxv=1, greedy=False):

    import ncolor
    from cmap import Colormap
    cmap = Colormap(cmap) if isinstance(cmap, str) else cmap

    m,n = ncolor.label(masks,
                       max_depth=max_depth,
                       return_n=True,
                       conn=2, 
                       expand=expand, 
                       greedy=greedy)
    if cmap is None:
        c = sinebow(n,offset=offset)
        colors = np.array(list(c.values()))
        cmap = mpl.colors.ListedColormap(colors)
        return cmap(m)
    else:
        return cmap(rescale(m)/maxv)


def set_outline(ax, outline_color=None, outline_width=0):
    """
    - Always hide axis ticks (ax.axis("off")).
    - If outline_color is not None and outline_width > 0,
        show spines with that color/width.
    - Otherwise, hide spines (no border).
    """
    # Always turn off ticks:
    # ax.axis("off")
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0)

    # Decide whether to draw spines:
    if outline_color is not None and outline_width > 0:
        for spine in ax.spines.values():
            spine.set_edgecolor(outline_color)
            spine.set_linewidth(outline_width)
    else:
        # Hide spines entirely
        for s in ax.spines.values():
            s.set_visible(False)

def imshow(imgs, figsize=2, ax=None, hold=False, titles=None, title_size=8, spacing=0.05,
           textcolor=[0.5]*3, dpi=300, text_scale=1,
           outline_color=None,     # e.g. [0.5]*3
           outline_width=0.5,     # e.g. 0.5
           show=False,
           **kwargs):
    """
    Display one or more images. Optionally add an outline (colored border)
    around each image if outline_color is not None and outline_width > 0.
    Otherwise, axes ticks etc. remain off, as before.
    """

    # -------------------------------------------------------------
    # If imgs is a list, we display multiple images side by side
    # -------------------------------------------------------------
    if isinstance(imgs, list):
        if titles is None:
            titles = [None] * len(imgs)
        if title_size is None:
            title_size = figsize / len(imgs) * text_scale

        # normalize figsize so figure() gets a 2-tuple of numbers
        if isinstance(figsize, (list, tuple, np.ndarray)):
            if len(figsize) >= 2:
                fig_w, fig_h = float(figsize[0]), float(figsize[1])
            elif len(figsize) == 1:
                fig_w = fig_h = float(figsize[0])
            else:
                fig_w = fig_h = 2.0
        else:
            # scalar: scale width by number of panels like original behavior
            fig_w = float(figsize) * len(imgs)
            fig_h = float(figsize)
        figsize_list = (fig_w, fig_h)

        # Create figure + subplots for multiple images
        fig, axes = figure(
            nrow=1, ncol=len(imgs),
            figsize=figsize_list,
            dpi=dpi,
            frameon=False,
            facecolor=[0, 0, 0, 0]
        )

        for this_ax, img, ttl in zip(axes, imgs, titles):
            this_ax.imshow(img, **kwargs)
            set_outline(this_ax, outline_color, outline_width)
            this_ax.set_facecolor([0, 0, 0, 0])

            if ttl is not None:
                this_ax.set_title(ttl, fontsize=title_size, color=textcolor)

    # -------------------------------------------------------------
    # Otherwise, just one image
    # -------------------------------------------------------------
    else:
        if not isinstance(figsize, (list, tuple, np.ndarray)):
            figsize = (figsize, figsize)
        elif len(figsize) == 2:
            figsize = (figsize[0], figsize[1])
        else:
            figsize = (figsize[0], figsize[0])
        if title_size is None:
            title_size = figsize[0] * text_scale

        if ax is None:
            subplot_args = {
                'frameon': False,
                'figsize': figsize,
                'facecolor': [0, 0, 0, 0],
                'dpi': dpi
            }
            fig, ax = figure(**subplot_args)
        else:
            hold = True
            fig = ax.get_figure()

        ax.imshow(imgs, **kwargs)
        set_outline(ax, outline_color, outline_width)
        ax.set_facecolor([0, 0, 0, 0])

        if titles is not None:
            ax.set_title(titles, fontsize=title_size, color=textcolor)

    if not hold:
        display(fig)
    else:
        return fig

# def get_cmap(masks):
#     lut = ncolor.get_lut(masks) # make sure int64
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
    from .utils import normalize99, qnorm, safe_divide
    if device is None:
        device = torch.device('cpu')
    
    if isinstance(dP,torch.Tensor):
        device = dP.device
    else:
        dP = torch.from_numpy(dP).to(device)
        
    mag = torch_norm(dP,dim=1)
    dP = safe_divide(dP,mag.unsqueeze(1))  # Normalize the flow vectors
    
    if norm:
        # mag -= torch.min(mag)
        # mag /= torch.max(mag)
        mag = normalize99(mag)
    
    vecs = dP[:,0] + dP[:,1]*1j
    roots = torch.exp(1j * np.pi * (2  * torch.arange(3, device=device) / 3 + 1 ))
    rgb = (torch.real(vecs.unsqueeze(-1)*roots.view(1, 1, 1, -1)) + 1 ) / 2 
    

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
        

LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _coerce_pad(pad):
    if pad is None:
        return 0.0, 0.0
    if isinstance(pad, (list, tuple, np.ndarray)):
        if len(pad) == 0:
            return 0.0, 0.0
        if len(pad) == 1:
            val = float(pad[0])
            return val, val
        return float(pad[0]), float(pad[1])
    val = float(pad)
    return val, val


def _get_renderer(fig):
    if fig is None:
        return None
    canvas = getattr(fig, 'canvas', None)
    if canvas is None:
        try:
            canvas = FigureCanvas(fig)
            fig.set_canvas(canvas)
        except Exception:
            return None
    try:
        renderer = canvas.get_renderer()
    except Exception:
        renderer = None
    if renderer is not None:
        return renderer
    try:
        canvas.draw()
        return canvas.get_renderer()
    except Exception:
        return getattr(canvas, '_renderer', None)


def _ensure_figure_rendered(fig):
    if fig is None:
        return
    canvas = getattr(fig, 'canvas', None)
    if canvas is None:
        try:
            canvas = FigureCanvas(fig)
            fig.set_canvas(canvas)
        except Exception:
            return
    try:
        canvas.draw()
    except Exception:
        pass


def _get_display_rgb(image_artist, cache=None):
    if image_artist is None:
        return None

    cache_key = id(image_artist)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    raw = image_artist.get_array()
    if raw is None:
        if cache is not None:
            cache[cache_key] = None
        return None

    arr = np.ma.asarray(raw)
    if arr.size == 0:
        if cache is not None:
            cache[cache_key] = None
        return None

    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 1):
        data = np.squeeze(arr)
        norm = image_artist.norm
        if norm is None:
            norm = mpl.colors.Normalize()
            norm.autoscale(data)
        mapped = np.ma.asarray(image_artist.cmap(norm(data)))[..., :3]
        if isinstance(mapped, np.ma.MaskedArray):
            rgb = mapped.astype(np.float32).filled(np.nan)
        else:
            rgb = np.asarray(mapped, dtype=np.float32)
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        data = arr.astype(np.float32, copy=False)
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            rng = info.max - info.min
            if rng > 0:
                data = (data - info.min) / float(rng)
        clipped = np.clip(data, 0.0, 1.0)[..., :3]
        if isinstance(clipped, np.ma.MaskedArray):
            rgb = clipped.astype(np.float32).filled(np.nan)
        else:
            rgb = np.asarray(clipped, dtype=np.float32)
    else:
        rgb = None

    if cache is not None:
        cache[cache_key] = rgb
    return rgb


def _get_text_bbox_display(text, pad=(1.05, 1.25)):
    if text is None:
        return None

    fig = text.figure
    _ensure_figure_rendered(fig)
    canvas = getattr(fig, 'canvas', None)
    if canvas is None:
        canvas = FigureCanvas(fig)
        fig.set_canvas(canvas)
    try:
        canvas.draw()
        renderer = canvas.get_renderer()
    except Exception:
        renderer = None

    bbox = None
    if renderer is not None:
        try:
            bbox = text.get_window_extent(renderer=renderer)
        except Exception:
            bbox = None

    if bbox is not None and fig is not None:
        text_str = text.get_text() or ""
        if text_str:
            try:
                prop = text.get_fontproperties()
                if prop is None:
                    prop = FontProperties()
                else:
                    prop = prop.copy()
                prop.set_size(text.get_fontsize())
                path = TextPath((0, 0), text_str, prop=prop, usetex=False)
                path_bbox = path.get_extents()
            except Exception:
                path_bbox = None
            if path_bbox is not None:
                scale = (fig.dpi or 72.0) / 72.0
                target_width = path_bbox.width * scale
                target_height = path_bbox.height * scale
                x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
                width = x1 - x0
                height = y1 - y0
                ha = text.get_ha()
                va = text.get_va()
                eps = 0.5
                if target_width - width > eps:
                    if ha == 'left':
                        x1 = x0 + target_width
                    elif ha == 'right':
                        x0 = x1 - target_width
                    else:
                        cx = 0.5 * (x0 + x1)
                        x0 = cx - target_width * 0.5
                        x1 = cx + target_width * 0.5
                if target_height - height > eps:
                    if va in ('bottom', 'baseline'):
                        y1 = y0 + target_height
                    elif va == 'top':
                        y0 = y1 - target_height
                    else:
                        cy = 0.5 * (y0 + y1)
                        y0 = cy - target_height * 0.5
                        y1 = cy + target_height * 0.5
                bbox = Bbox.from_extents(x0, y0, x1, y1)
                fudge = max(1.0, (fig.dpi or 72.0) * 0.002)
                bbox = Bbox.from_extents(bbox.x0, bbox.y0,
                                         bbox.x1 + fudge,
                                         bbox.y1 + fudge)

    if bbox is None:
        pos = text.get_position()
        transform = text.get_transform()
        if fig is None or transform is None or pos is None:
            return None
        try:
            anchor = transform.transform(pos)
        except Exception:
            return None
        text_str = text.get_text() or ""
        fontsize_pt = text.get_fontsize() or 0
        dpi = getattr(fig, 'dpi', 72) or 72
        base_font_px = max(fontsize_pt, 1e-3) * (dpi / 72.0)
        approx_char_px = 0.6 * base_font_px
        lines = text_str.splitlines() or [text_str]
        n_lines = max(1, len(lines))
        max_line_len = max((len(line) for line in lines), default=1)
        text_width_px = max(max_line_len * approx_char_px, base_font_px)
        text_height_px = max(n_lines * base_font_px, base_font_px)
        half_w = max(2.0, 0.5 * text_width_px)
        half_h = max(2.0, 0.5 * text_height_px)
        ha = text.get_ha()
        va = text.get_va()
        center_x, center_y = anchor
        if ha == 'left':
            center_x += half_w
        elif ha == 'right':
            center_x -= half_w
        if va in ('bottom', 'baseline'):
            center_y += half_h
        elif va == 'top':
            center_y -= half_h
        bbox = Bbox.from_extents(center_x - half_w, center_y - half_h,
                                 center_x + half_w, center_y + half_h)

    if bbox is None:
        return None

    if pad is not None and len(pad) == 2:
        bbox = bbox.expanded(pad[0], pad[1])
    return bbox


def _sample_label_lightness(image_artist, ax, text, cache=None):
    rgb = _get_display_rgb(image_artist, cache=cache)
    if rgb is None or ax is None or text is None:
        return None

    bbox = _get_text_bbox_display(text)
    if bbox is None:
        return None

    axis_bbox = getattr(ax, 'bbox', None)
    if axis_bbox is None:
        return None

    disp_min = np.array([max(bbox.x0, axis_bbox.x0), max(bbox.y0, axis_bbox.y0)])
    disp_max = np.array([min(bbox.x1, axis_bbox.x1), min(bbox.y1, axis_bbox.y1)])

    if disp_min[0] >= disp_max[0] or disp_min[1] >= disp_max[1]:
        return None

    try:
        inv = ax.transData.inverted()
        data_corners = inv.transform(np.vstack((disp_min, disp_max)))
    except Exception:
        return None

    data_x = np.sort(data_corners[:, 0])
    data_y = np.sort(data_corners[:, 1])

    try:
        x0, x1, y0, y1 = image_artist.get_extent()
    except Exception:
        return None

    nx = rgb.shape[1]
    ny = rgb.shape[0]
    if nx == 0 or ny == 0:
        return None

    denom_x = x1 - x0
    denom_y = y1 - y0
    if denom_x == 0 or denom_y == 0:
        return None

    col_fracs = np.sort((np.array(data_x) - x0) / denom_x)
    row_fracs = np.sort((np.array(data_y) - y0) / denom_y)

    def _index_range(fracs, length):
        if length <= 1:
            return 0, 0
        scale = length - 1
        start = int(np.floor(np.clip(fracs[0], 0.0, 1.0) * scale))
        end = int(np.ceil(np.clip(fracs[1], 0.0, 1.0) * scale))
        start = min(max(start, 0), length - 1)
        end = min(max(end, 0), length - 1)
        return start, end

    col_start, col_end = _index_range(col_fracs, nx)
    row_start, row_end = _index_range(row_fracs, ny)

    if col_end < col_start or row_end < row_start:
        return None

    patch = rgb[row_start:row_end + 1, col_start:col_end + 1]
    if patch.size == 0:
        return None

    luminance = np.tensordot(patch, LUMA_WEIGHTS, axes=([-1], [0]))
    value = float(np.nanmean(luminance))
    if not np.isfinite(value):
        return None
    return value


def recolor_label(ax, text, image_artist=None,
                  threshold=0.6,
                  light_color=(0.8, 0.8, 0.8),
                  dark_color=(0.2, 0.2, 0.2),
                  cache=None):
    """Update a label's color based on the lightness of the underlying image."""
    
    if text is None:
        return None

    if ax is None:
        ax = text.axes

    if image_artist is None and ax is not None and ax.images:
        image_artist = ax.images[-1]

    if ax is None or image_artist is None:
        return None

    lightness = _sample_label_lightness(image_artist, ax, text, cache=cache)
    if lightness is None:
        return None

    if lightness >= threshold:
        text.set_color(dark_color)
    else:
        text.set_color(light_color)

    return lightness


def _create_pill_patch(ax, rect_x, rect_y, rect_w, rect_h,
                       orientation, facecolor, zorder):
    if rect_w <= 0 or rect_h <= 0:
        return None

    x0, y0 = rect_x, rect_y
    x1, y1 = rect_x + rect_w, rect_y + rect_h
    verts = []
    codes = []
    N = 20

    def add(pt, code):
        verts.append(pt)
        codes.append(code)

    if orientation in ("right", "left"):
        r = min(rect_h / 2.0, rect_w)
        if r <= 0:
            return None
        if orientation == "right":
            cx = x1 - r
            cy = y0 + rect_h / 2.0
            theta = np.linspace(-np.pi / 2, np.pi / 2, N)
            add((x0, y0), Path.MOVETO)
            add((cx, y0), Path.LINETO)
            for t in theta[1:-1]:
                add((cx + r * np.cos(t), cy + r * np.sin(t)), Path.LINETO)
            add((cx, y1), Path.LINETO)
            add((x0, y1), Path.LINETO)
        else:  # left rounded
            cx = x0 + r
            cy = y0 + rect_h / 2.0
            theta = np.linspace(np.pi / 2, 3 * np.pi / 2, N)
            add((x1, y0), Path.MOVETO)
            add((cx, y0), Path.LINETO)
            for t in theta[1:-1]:
                add((cx + r * np.cos(t), cy + r * np.sin(t)), Path.LINETO)
            add((cx, y1), Path.LINETO)
            add((x1, y1), Path.LINETO)
    elif orientation in ("top", "bottom"):
        r = min(rect_w / 2.0, rect_h)
        if r <= 0:
            return None
        cx = x0 + rect_w / 2.0
        if orientation == "top":
            cy = y1 - r
            theta = np.linspace(0, np.pi, N)
            add((x0, y0), Path.MOVETO)
            add((x1, y0), Path.LINETO)
            add((x1, cy), Path.LINETO)
            for t in theta[1:-1]:
                add((cx + r * np.cos(t), cy + r * np.sin(t)), Path.LINETO)
            add((x0, cy), Path.LINETO)
            add((x0, y0), Path.LINETO)
        else:
            cy = y0 + r
            theta = np.linspace(np.pi, 2 * np.pi, N)
            add((x0, y1), Path.MOVETO)
            add((x1, y1), Path.LINETO)
            add((x1, cy), Path.LINETO)
            for t in theta[1:-1]:
                add((cx + r * np.cos(t), cy + r * np.sin(t)), Path.LINETO)
            add((x0, cy), Path.LINETO)
            add((x0, y1), Path.LINETO)
    else:
        return None

    add(verts[0], Path.CLOSEPOLY)
    path = Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=facecolor,
                               edgecolor='none', linewidth=0,
                               transform=ax.transAxes)
    patch.set_zorder(zorder)
    return patch


def add_label_background(ax, text, width_px=None, pad_px=2.0,
                         color=None, alpha=1.0,
                         align_to_axes=True, cache_bbox=None,
                         zorder_offset=-0.3,
                         use_axis_width=False,
                         debug_bbox=False,
                         style='full'):
    """Add a rectangular background patch behind ``text`` within ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes for the label.
    text : matplotlib.text.Text
        The text artist to backfill.
    width_px : float, optional
        Desired width of the panel in display pixels.  If ``None`` the
        text width (plus padding) is used.
    pad_px : float or (float, float)
        Padding in display pixels along (x, y).
    color : color-like or ``None``
        Fill color for the panel. ``None``/``'auto'`` uses white.
    alpha : float
        Opacity of the panel fill.
    align_to_axes : bool
        If ``True`` the panel edge nearest the axes (based on the text
        alignment) is locked to the axes boundary.
    cache_bbox : matplotlib.transforms.Bbox, optional
        Precomputed display-space bbox for the text.
    zorder_offset : float
        Offset applied so the background sits behind the text.
    use_axis_width : bool
        If ``True`` the rectangle spans the full axis width.
    """

    if text is None:
        return None

    if ax is None:
        ax = text.axes

    if ax is None:
        return None

    fig = ax.figure

    bbox = cache_bbox
    if bbox is None:
        bbox = _get_text_bbox_display(text, pad=None)

    if bbox is None:
        return None

    axis_bbox = getattr(ax, 'bbox', None)
    if axis_bbox is None:
        return None

    pad_x, pad_y = _coerce_pad(pad_px)
    symmetric_x = pad_x < 0
    symmetric_y = pad_y < 0
    if symmetric_x:
        pad_x = 0.0
    if symmetric_y:
        pad_y = 0.0
    axis_width = axis_bbox.width
    axis_height = axis_bbox.height
    if axis_width <= 0 or axis_height <= 0:
        return None

    left = bbox.x0 - pad_x
    right = bbox.x1 + pad_x
    bottom = bbox.y0 - pad_y
    top = bbox.y1 + pad_y

    ha = text.get_ha()
    va = text.get_va()
    anchor_left = align_to_axes and ha == 'left'
    anchor_right = align_to_axes and ha == 'right'
    anchor_bottom = align_to_axes and va in ('bottom', 'baseline')
    anchor_top = align_to_axes and va == 'top'

    pill_mode = (style == 'pill')
    epsilon = 1.0
    if pill_mode:
        if anchor_left and not anchor_right:
            right += epsilon
        elif anchor_right and not anchor_left:
            left -= epsilon
        elif anchor_top and not anchor_bottom:
            bottom -= epsilon
        elif anchor_bottom and not anchor_top:
            top += epsilon
    symmetric_x = pad_x < 0
    symmetric_y = pad_y < 0
    if symmetric_x:
        pad_x = 0.0
    if symmetric_y:
        pad_y = 0.0
    axis_width = axis_bbox.width
    axis_height = axis_bbox.height
    if axis_width <= 0 or axis_height <= 0:
        return None

    left = bbox.x0 - pad_x
    right = bbox.x1 + pad_x
    bottom = bbox.y0 - pad_y
    top = bbox.y1 + pad_y

    ha = text.get_ha()
    va = text.get_va()
    anchor_left = align_to_axes and ha == 'left'
    anchor_right = align_to_axes and ha == 'right'
    anchor_bottom = align_to_axes and va in ('bottom', 'baseline')
    anchor_top = align_to_axes and va == 'top'

    pill_mode = (style == 'pill')

    if pill_mode and align_to_axes:
        if ha == 'left':
            anchor_left, anchor_right = True, False
            anchor_top = anchor_bottom = False
        elif ha == 'right':
            anchor_right, anchor_left = True, False
            anchor_top = anchor_bottom = False
        elif va == 'top':
            anchor_top = True
            anchor_bottom = anchor_left = anchor_right = False
        elif va in ('bottom', 'baseline'):
            anchor_bottom = True
            anchor_top = anchor_left = anchor_right = False
        else:
            anchor_left = anchor_right = anchor_top = anchor_bottom = False

    base_left, base_right = left, right
    base_bottom, base_top = bottom, top

    left_adjust = 0.0
    right_adjust = 0.0
    bottom_adjust = 0.0
    top_adjust = 0.0

    if anchor_left:
        left_adjust = base_left - axis_bbox.x0
        left = axis_bbox.x0
    if anchor_right:
        right_adjust = axis_bbox.x1 - base_right
        right = axis_bbox.x1
    if anchor_bottom:
        bottom_adjust = base_bottom - axis_bbox.y0
        bottom = axis_bbox.y0
    if anchor_top:
        top_adjust = axis_bbox.y1 - base_top
        top = axis_bbox.y1

    desired_width = right - left
    if desired_width <= 0:
        desired_width = (bbox.width + 2 * pad_x)

    if use_axis_width:
        target_width = axis_width
    elif width_px is not None:
        target_width = min(max(width_px, desired_width), axis_width)
    else:
        target_width = desired_width

    if target_width > desired_width:
        extra = target_width - desired_width
        if anchor_left and not anchor_right:
            right = min(axis_bbox.x1, right + extra)
        elif anchor_right and not anchor_left:
            left = max(axis_bbox.x0, left - extra)
        else:
            left -= 0.5 * extra
            right += 0.5 * extra

    if symmetric_x:
        if anchor_left and not anchor_right:
            right = min(axis_bbox.x1, right + left_adjust)
        elif anchor_right and not anchor_left:
            left = max(axis_bbox.x0, left - right_adjust)
        else:
            left -= 0.5 * (left_adjust + right_adjust)
            right += 0.5 * (left_adjust + right_adjust)

    if symmetric_y:
        if anchor_bottom and not anchor_top:
            top = min(axis_bbox.y1, top + bottom_adjust)
        elif anchor_top and not anchor_bottom:
            bottom = max(axis_bbox.y0, bottom - top_adjust)
        else:
            bottom -= 0.5 * (bottom_adjust + top_adjust)
            top += 0.5 * (bottom_adjust + top_adjust)
    # Extend capsules so the rounded end does not intrude into the text region
    if style == 'pill' and top > bottom and right > left:
        s = .25
        height = top - bottom
        width = right - left
        if anchor_left and not anchor_right:
            right += s * height
        elif anchor_right and not anchor_left:
            left -= s * height
        elif anchor_top and not anchor_bottom:
            bottom -= s * width
        elif anchor_bottom and not anchor_top:
            top += s * width

    # Ensure the rectangle fully covers the text even when anchored to axes
    right = max(right, bbox.x1)
    left = min(left, bbox.x0)
    top = max(top, bbox.y1)
    bottom = min(bottom, bbox.y0)

    left = max(axis_bbox.x0, min(axis_bbox.x1, left))
    right = max(axis_bbox.x0, min(axis_bbox.x1, right))
    bottom = max(axis_bbox.y0, min(axis_bbox.y1, bottom))
    top = max(axis_bbox.y0, min(axis_bbox.y1, top))

    if right <= left or top <= bottom:
        return None

    if debug_bbox and fig is not None:
        try:
            prop = text.get_fontproperties()
            if prop is None:
                prop = FontProperties()
            else:
                prop = prop.copy()
            prop.set_size(text.get_fontsize())
            path = TextPath((0, 0), text.get_text() or "", prop=prop, usetex=False)
            path_bbox = path.get_extents()
            path_width = path_bbox.width * (fig.dpi or 72.0) / 72.0
        except Exception:
            path_width = None
        win_width = (bbox.x1 - bbox.x0)
        if path_width is not None:
            print(f"[bbox debug] label={text.get_text()} window_width={win_width:.2f} path_width={path_width:.2f}")
        else:
            print(f"[bbox debug] label={text.get_text()} window_width={win_width:.2f}")

    rect_x = (left - axis_bbox.x0) / axis_width
    rect_y = (bottom - axis_bbox.y0) / axis_height
    rect_w = (right - left) / axis_width
    rect_h = (top - bottom) / axis_height
    if rect_w <= 0 or rect_h <= 0:
        return None

    if color is None or color == 'auto':
        base_color = 'white'
    else:
        base_color = color

    facecolor = mpl.colors.to_rgba(base_color, alpha)

    orientation = None
    if style == 'pill':
        if anchor_left and not anchor_right:
            orientation = 'right'
        elif anchor_right and not anchor_left:
            orientation = 'left'
        elif anchor_top and not anchor_bottom:
            orientation = 'bottom'
        elif anchor_bottom and not anchor_top:
            orientation = 'top'

    if style == 'pill' and orientation is not None:
        patch = _create_pill_patch(ax, rect_x, rect_y, rect_w, rect_h,
                                   orientation, facecolor, text.get_zorder() + zorder_offset)
        if patch is None:
            patch = mpatches.Rectangle((rect_x, rect_y), rect_w, rect_h,
                                       facecolor=facecolor, edgecolor='none',
                                       linewidth=0, transform=ax.transAxes)
    else:
        patch = mpatches.Rectangle((rect_x, rect_y), rect_w, rect_h,
                                   facecolor=facecolor, edgecolor='none',
                                   linewidth=0, transform=ax.transAxes)
    patch.set_zorder(text.get_zorder() + zorder_offset)
    ax.add_patch(patch)
    text.set_zorder(max(text.get_zorder(), patch.get_zorder() + 0.1))

    if debug_bbox:
        inv_axes = ax.transAxes.inverted()
        raw_x0, raw_y0 = inv_axes.transform((bbox.x0, bbox.y0))
        raw_x1, raw_y1 = inv_axes.transform((bbox.x1, bbox.y1))
        dbg_x = min(raw_x0, raw_x1)
        dbg_y = min(raw_y0, raw_y1)
        dbg_w = abs(raw_x1 - raw_x0)
        dbg_h = abs(raw_y1 - raw_y0)
        dbg_patch = mpatches.Rectangle((dbg_x, dbg_y), dbg_w, dbg_h,
                                       facecolor='none', edgecolor='red',
                                       linewidth=0.5, linestyle='--',
                                       transform=ax.transAxes)
        dbg_patch.set_zorder(patch.get_zorder() + 0.1)
        ax.add_patch(dbg_patch)

    return patch


def apply_label_backgrounds(texts, pad_px=2.0, color='outline', alpha=1.0,
                            align_to_axes=True, default_color=None,
                            full_width=False, debug_bbox=False,
                            style='full'):
    """Apply uniform label backgrounds to a list of text artists."""

    if not texts:
        return

    processed = []
    widths = []
    pad_x, _ = _coerce_pad(pad_px)

    for text in texts:
        if text is None:
            continue
        bbox = _get_text_bbox_display(text, pad=None)
        if bbox is None:
            continue
        processed.append((text, bbox))
        widths.append(bbox.width)

    if not processed:
        return

    target_width = None
    if not full_width:
        target_width = max(widths) + 2 * pad_x

    for text, bbox in processed:
        ax = text.axes
        if ax is None:
            continue
        facecolor = color
        if facecolor in (None, 'outline', 'auto'):
            facecolor = default_color
        add_label_background(
            ax,
            text,
            width_px=target_width,
            pad_px=pad_px,
            color=facecolor,
            alpha=alpha,
            align_to_axes=align_to_axes,
            cache_bbox=bbox,
            use_axis_width=full_width,
            debug_bbox=debug_bbox,
            style=style,
        )


def image_grid(images, column_titles=None, row_titles=None, 
               plot_labels=None, 
               xticks=[], yticks=[], 
               outline=False, outline_color=[0.5]*3, outline_width=.5,
               padding=0.05, interset_padding=0.1,
               fontsize=8, fontcolor=[0.5]*3,
               adaptive_label_color=False,
               label_lightness_threshold=0.6,
               light_label_color=[0.8]*3,
               dark_label_color=[0.2]*3,
               label_background=False,
               label_background_color='outline',
               label_background_alpha=1.0,
               label_background_pad=2.0,
               label_background_align_axes=True,
               label_background_full_width=False,
               label_background_debug=False,
               label_background_style='full',
               facecolor=None,
               figsize=6, 
               dpi=300,
               order='ij',
               reverse_row=False,
               stack_direction='horizontal',
               lpad=0.05,
               lpos='top_middle',
               return_axes=False,
               fig=None,
               offset=[0, 0],
               supcolor=None,
               right_justify_rows=False,  # New flag for right justification
               **kwargs):
    
    if supcolor is None:
        supcolor = fontcolor

    label_positions = {
        'top_middle': {'coords': (0.5, 1 - lpad), 'va': 'top', 'ha': 'center'},
        'bottom_left': {'coords': (lpad, lpad), 'va': 'bottom', 'ha': 'left'},
        'bottom_middle': {'coords': (0.5, lpad), 'va': 'bottom', 'ha': 'center'},
        'top_left': {'coords': (lpad, 1 - lpad), 'va': 'top', 'ha': 'left'},
        'above_middle': {'coords': (.5, 1 +lpad), 'va': 'bottom', 'ha': 'center'},
        
    }

    rgba_cache = {} if adaptive_label_color else None
    label_texts = [] if label_background else None

    # Check if 'images' is a list of lists of lists, meaning multiple image sets
    if isinstance(images[0][0], list):
        multiple_sets = True
    else:
        multiple_sets = False
        images = [images]  # Treat single set as a list of one
        plot_labels = [plot_labels] if plot_labels is not None else None

    n_sets = len(images)
    ij = order == 'ij'
    
    # if (not ij and column_titles is not None) or (ij and row_titles is not None):
    #     row_titles, column_titles = column_titles, row_titles

    # ── swap the title lists when using column-major order ─────────────────
    if not ij:
        column_titles, row_titles = row_titles, column_titles

    # Initialize lists to hold positions and sizes
    all_left = []
    all_bottom = []
    all_width = []
    all_height = []

    # Initialize offset for stacking
    total_offset_x = 0
    total_offset_y = 0

    for set_idx, image_set in enumerate(images):
        # ───────────────────── grid dimensions ───────────────────────────
        if ij:
            nrows = len(image_set)
            ncols = max(len(row) for row in image_set)
        else:
            ncols = len(image_set)
            nrows = max(len(col) for col in image_set)

        # ───────────────────── constant-size axis setup ──────────────────
        p     = padding          # gap between axes
        base  = 1.0              # fixed width (ij) or height (!ij)
        positions = []

        if ij:  # constant widths → variable heights
            cur_bottom = total_offset_y
            for r, row in enumerate(image_set):
                rep   = next((im for im in row if im is not None), None)
                ratio = (rep.shape[0] / rep.shape[1]) if rep is not None else 1.0
                h     = ratio * base

                row_offset = ((ncols - len(row)) * (base + p)) if right_justify_rows else 0
                for c, _ in enumerate(row):
                    left   = total_offset_x + row_offset + c * (base + p)
                    bottom = cur_bottom
                    positions.append((left, bottom, base, h))

                cur_bottom += h + p

            set_span_x = (base + p) * ncols - p
            set_span_y = cur_bottom - total_offset_y - p

        else:   # constant heights → variable widths
            cur_left = total_offset_x
            for c, col in enumerate(image_set):
                rep    = next((im for im in col if im is not None), None)
                aspect = (rep.shape[1] / rep.shape[0]) if rep is not None else 1.0
                w      = aspect * base

                for r, _ in enumerate(col):
                    left   = cur_left
                    bottom = total_offset_y + r * (base + p)
                    positions.append((left, bottom, w, base))

                cur_left += w + p

            set_span_x = cur_left - total_offset_x - p
            set_span_y = (base + p) * nrows - p

        # ───────────────────── collect positions ─────────────────────────
        lefts, bottoms, widths, heights = zip(*positions)
        all_left.extend(lefts);   all_bottom.extend(bottoms)
        all_width.extend(widths); all_height.extend(heights)

        # ───────────────────── inter-set stacking ────────────────────────
        if multiple_sets and set_idx < n_sets - 1:
            if stack_direction == 'horizontal':
                total_offset_x += set_span_x + interset_padding
            elif stack_direction == 'vertical':
                total_offset_y += set_span_y + interset_padding


    # Normalize positions
    lefts = np.array(all_left)
    bottoms = np.array(all_bottom)
    widths = np.array(all_width)
    heights = np.array(all_height)

    max_w = max(lefts + widths)
    max_h = max(bottoms + heights)
    lefts /= max_w
    widths /= max_w

    # Adjust bottoms for top-down layout
    bottoms = (max_h - bottoms - heights) / max_h
    heights /= max_h

    # Use the existing figure if provided; otherwise, create a new one
    if fig is None:
        # if not isinstance(figsize, (list, tuple)):
        figsize=(figsize, figsize * max_h / max_w) if ij else (figsize * max_w / max_h, figsize)
    
        fig = Figure(figsize=figsize,
                     frameon=False if facecolor is None else True,
                     facecolor=[0] * 4 if facecolor is None else facecolor,
                     dpi=dpi)
        FigureCanvas(fig)

    # Apply offsets to the left and bottom positions
    lefts += offset[0]
    bottoms += offset[1]

    # Add the subplots
    axes = []
    for idx, (left, bottom, width, height) in enumerate(zip(lefts, bottoms, widths, heights)):
        ax = fig.add_axes([left, bottom, width, height])
        axes.append(ax)

    # Add images to the subplots
    idx = 0
    for set_idx, image_set in enumerate(images):
        for row_idx, row in enumerate(image_set):
            for col_idx, img in enumerate(row):
                ax = axes[idx]
                idx += 1

                ax.set_xticks(xticks)
                ax.set_yticks(yticks)
                ax.patch.set_alpha(0)

                image_artist = None
                if img is not None:
                    image_artist = ax.imshow(img, **kwargs)

                # Add plot labels
                if plot_labels is not None:
                    try:
                        label = plot_labels[set_idx][row_idx][col_idx]
                    except IndexError:
                        label = None

                    if label is not None:
                        coords = label_positions[lpos]['coords']
                        va = label_positions[lpos]['va']
                        ha = label_positions[lpos]['ha']

                        text = ax.text(coords[0], coords[1], label,
                                       fontsize=fontsize, color=fontcolor, va=va, ha=ha, transform=ax.transAxes)

                        if adaptive_label_color and image_artist is not None:
                            recolor_label(
                                ax,
                                text,
                                image_artist=image_artist,
                                threshold=label_lightness_threshold,
                                light_color=light_label_color,
                                dark_color=dark_label_color,
                                cache=rgba_cache,
                            )

                        if label_background:
                            label_texts.append(text)

                        if img is None:
                            text.set_color([.5] * 4)

                # ── column titles ──────────────────────────────────────────
                if column_titles is not None:
                    want_title = (
                        (ij  and row_idx == 0 and col_idx < len(column_titles)) or
                        (not ij and col_idx == 0 and row_idx < len(column_titles))
                    )
                    if want_title and (stack_direction != 'vertical' or set_idx == 0):
                        title_idx = col_idx if ij else row_idx
                        ax.text(0.5, 1 + p,
                                column_titles[title_idx],
                                rotation=0, fontsize=fontsize, color=supcolor,
                                va='bottom', ha='center', transform=ax.transAxes)

                # ── row titles ─────────────────────────────────────────────
                if row_titles is not None:
                    want_title = (
                        (ij  and col_idx == 0 and row_idx < len(row_titles)) or
                        (not ij and row_idx == 0 and col_idx < len(row_titles))
                    )
                    if want_title and (stack_direction != 'horizontal' or set_idx == 0):
                        title_idx = row_idx if ij else col_idx
                        ax.text(-p, 0.5,
                                row_titles[title_idx],
                                rotation=0, fontsize=fontsize, color=supcolor,
                                va='center', ha='right', transform=ax.transAxes)

                # Add outline if needed
                if outline:
                    for s in ax.spines.values():
                        s.set_color(outline_color)
                        s.set_linewidth(outline_width)
                else:
                    for s in ax.spines.values():
                        s.set_visible(False)

    if label_background and label_texts:
        apply_label_backgrounds(
            label_texts,
            pad_px=label_background_pad,
            color=label_background_color,
            alpha=label_background_alpha,
            align_to_axes=label_background_align_axes,
            default_color=outline_color,
            full_width=label_background_full_width,
            debug_bbox=label_background_debug,
            style=label_background_style,
        )

    if return_axes:
        pos = [lefts, bottoms, widths, heights]
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
import matplotlib.path as mpath
from skimage.segmentation import find_boundaries
from scipy.interpolate import splprep, splev
from matplotlib.collections import PatchCollection

def vector_contours(fig,ax,mask, crop=None, smooth_factor=5, color = 'r', linewidth=1,
                    y_offset=0, x_offset=0,
                    pad=2,
                    mode='constant',
                    zorder=1,
                    ):

    msk = np.pad(mask,pad,mode='edge')

    # msk = np.pad(mask,pad,mode=mode)
    
    if crop is not None:
        # Crop the mask to the specified region
        msk = msk[crop]
        
    msk = np.pad(msk,1,mode='constant', constant_values=0)

    # set up dimensions
    dim = msk.ndim
    shape = msk.shape
    steps,inds,idx,fact,sign = kernel_setup(dim)

    # remove spur points - this method is way easier than running core._despur() on the priginal affinity graph
    bd = find_boundaries(msk,mode='inner',connectivity=2)
    msk, bounds, _ = boundary_to_masks(bd,binary_mask=msk>0,connectivity=1,min_size=0) 

    # generate affinity graph
    coords = np.nonzero(msk)
    neighbors = get_neighbors(tuple(coords),steps,dim,shape) # shape (d,3**d,npix)
    affinity_graph =  masks_to_affinity(msk, coords, steps, inds, idx, fact, sign, dim, neighbors)

    # find contours 
    contour_map, contour_list, unique_L = get_contour(msk,
                                                    affinity_graph,
                                                    coords,
                                                    neighbors,
                                                    cardinal_only=True)

    # List to hold patches
    patches = []
    for contour in contour_list:
        if len(contour) > 1:
            pts = np.stack([c[contour] for c in coords]).T[:, ::-1]  # YX to XY
            pts+= np.array([x_offset,y_offset])  # Apply offsets
            tck, u = splprep(pts.T, u=None, s=len(pts)/smooth_factor, per=1) 
            u_new = np.linspace(u.min(), u.max(), len(pts))
            x_new, y_new = splev(u_new, tck, der=0) 
            

            # Define the points of the polygon
            # points = np.column_stack([y_new-pad+y_offset, x_new-pad+x_offset])
            # points = np.column_stack([ x_new-pad+x_offset,y_new-pad+y_offset])
            # points = np.column_stack([ x_new-2*pad+x_offset,y_new-2*pad+y_offset])
            # points = np.column_stack([x_new-pad,y_new-pad])
            if isinstance(pad,tuple):
                # If pad is a tuple, apply it to x and y separately
                points = np.column_stack([x_new-(pad[0][0]+1), y_new-(pad[1][0]+1)])
            else:
                points = np.column_stack([x_new-(pad+1),y_new-(pad+1)])
            
            
            
            
            # Create a Path from the points
            path = mpath.Path(points, closed=True)
            
            # Create a PathPatch from the Path
            patch = mpatches.PathPatch(path, fill=None, edgecolor=color, 
                                    #    linewidth= fig.dpi/72, 
                                        linewidth=linewidth,
                                        zorder=zorder,
                                       capstyle='round')
            
            # ax.add_patch(patch)
            
            # Add patch to list
            patches.append(patch)

    # Create a PatchCollection from the list of patches
    # Add the PatchCollection to the axis/axes
    if isinstance(ax,list):
        for a in ax:
            patch_collection = PatchCollection(patches, match_original=True, snap=False)
            a.add_collection(patch_collection)
    else:
        patch_collection = PatchCollection(patches, match_original=True, snap=False)
        ax.add_collection(patch_collection)
