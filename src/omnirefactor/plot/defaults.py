import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure

from ..utils.color import sinebow
from ..transforms.normalize import rescale
from ..transforms.vector import torch_norm


def apply_mpl_defaults():
    mpl.rcParams['svg.fonttype'] = 'none'  # keep text as real text in the SVG
    mpl.rcParams['text.usetex'] = False    # avoid LaTeX (which converts text to paths)


def setup():
    # Import necessary libraries
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, HTML
    from tqdm.notebook import tqdm as notebook_tqdm  # progress bars

    apply_mpl_defaults()

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
        'lines.scale_dashes': False,
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


def figure(nrow=None, ncol=None, aspect=1, **kwargs):
    figsize = kwargs.get('figsize', 2)
    if not isinstance(figsize, (list, tuple, np.ndarray)) and figsize is not None:
        figsize = (figsize * aspect, figsize)

    kwargs['figsize'] = figsize
    fig = Figure(**kwargs)
    if nrow is not None and ncol is not None:
        axs = []
        for i in range(nrow * ncol):
            ax = fig.add_subplot(nrow, ncol, i + 1)
            axs.append(ax)
        return fig, axs
    else:
        ax = fig.add_subplot(111)
        return fig, ax


apply_mpl_defaults()
