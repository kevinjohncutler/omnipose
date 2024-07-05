# from . import core, utils
# from .__main__ import main
# import pkg_resources
# __version__ = pkg_resources.get_distribution("omnipose").version

import pkg_resources
__all__ = ['core', 'models','io','metrics','plot']
__version__ = pkg_resources.get_distribution("omnipose").version
def __getattr__(name):
    if name in __all__:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module 'omnipose' has no attribute '{name}'")
