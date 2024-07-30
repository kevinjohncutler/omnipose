# Set NUMEXPR_MAX_THREADS to the number of available CPU cores
import os
import multiprocessing
os.environ['NUMEXPR_MAX_THREADS'] = str(multiprocessing.cpu_count())


# controlled import to prevent MIP print statement 
# import mip
# from aicsimageio import AICSImage 

# Use of sets...
import warnings
from numba.core.errors import NumbaPendingDeprecationWarning
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


import pkg_resources
__all__ = ['core', 'utils', 'loss', 'plot', 'misc', 'cli', 'data', 'gpu', 'stacks', 'measure']
__version__ = pkg_resources.get_distribution("omnipose").version
def __getattr__(name):
    if name in __all__:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module 'omnipose' has no attribute '{name}'")
