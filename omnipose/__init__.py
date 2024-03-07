__all__ = ['core', 'utils', 'loss', 'plot', 'cli', 'data', 'gpu', 'misc']
from . import core, utils, loss, plot, misc, cli, data, gpu
import pkg_resources
__version__ = pkg_resources.get_distribution("omnipose").version
