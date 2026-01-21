"""
Install lazy sub-module discovery / attribute forwarding on a package/submodule.
"""
from types import ModuleType
import importlib
import pkgutil
import sys


def enable_submodules(pkg_name: str) -> None:
    """Attach __all__, __getattr__, and __dir__ to *pkg_name*."""
    pkg: ModuleType = sys.modules[pkg_name]
    submods = {info.name for info in pkgutil.iter_modules(pkg.__path__)}
    submods.discard("__main__")
    pkg.__all__ = list(submods)

    def _getattr(name: str):
        if name == "__main__":
            raise AttributeError(f"module {pkg_name!r} has no attribute {name!r}")
        if name in submods:
            mod = importlib.import_module(f"{pkg_name}.{name}")
            setattr(pkg, name, mod)
            return mod
        for sub in submods:
            mod = importlib.import_module(f"{pkg_name}.{sub}")
            if hasattr(mod, name):
                attr = getattr(mod, name)
                setattr(pkg, name, attr)
                return attr
        raise AttributeError(f"module {pkg_name!r} has no attribute {name!r}")

    def _dir():
        items = set(pkg.__dict__) | submods
        for sub in submods:
            try:
                mod = importlib.import_module(f"{pkg_name}.{sub}")
            except ImportError:
                continue
            items.update(n for n in dir(mod) if not n.startswith("_"))
        return sorted(items)

    pkg.__getattr__ = _getattr  # type: ignore[attr-defined]
    pkg.__dir__ = _dir          # type: ignore[attr-defined]
