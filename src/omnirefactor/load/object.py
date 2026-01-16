import importlib
import inspect
import pkgutil
from pathlib import Path


def attach_helpers(cls, modules_or_packages, *, skip_subpackages=True, exclude_modules=None):
    """
    Import helper modules/packages and copy their public helpers onto *cls*.
    """
    if exclude_modules is None:
        exclude_modules = set()
    elif not isinstance(exclude_modules, set):
        exclude_modules = set(exclude_modules)

    if isinstance(modules_or_packages, str):
        modules_or_packages = [modules_or_packages]

    for name in modules_or_packages:
        mod = importlib.import_module(name)
        if getattr(mod, "__file__", None) is not None:
            pkg_dir = Path(mod.__file__).parent
            pkg_name = mod.__package__ or name.split(".")[0]
        elif hasattr(mod, "__path__"):
            pkg_dir = Path(next(iter(mod.__path__)))
            pkg_name = name
        else:
            continue

        load_submodules(
            cls,
            package_dir=str(pkg_dir),
            package_name=pkg_name,
            exclude_modules=exclude_modules,
            skip_subpackages=skip_subpackages,
        )


def attach_function_to_object(cls, name, func):
    """
    Attach 'func' to cls, honoring decorator tags for properties/classmethods.
    """
    if isinstance(cls, dict):
        cls[name] = func
    elif getattr(func, "__is_property__", False):
        setattr(cls, name, property(func))
    elif getattr(func, "__is_classmethod__", False):
        setattr(cls, name, classmethod(func))
    else:
        setattr(cls, name, func)


def load_submodules(
    cls,
    package_dir,
    package_name,
    exclude_modules=None,
    skip_subpackages=False,
    attach_vars: bool = False,
):
    """
    Load all Python submodules (files only, not packages) in the given directory,
    excluding basenames in exclude_modules, then attach top-level functions to cls.
    """
    if exclude_modules is None:
        exclude_modules = set()
    else:
        exclude_modules = set(exclude_modules)

    for module_info in pkgutil.iter_modules([package_dir]):
        if module_info.ispkg and skip_subpackages:
            continue

        mod_name = module_info.name
        if mod_name in exclude_modules or mod_name == "__init__":
            continue

        full_name = f"{package_name}.{mod_name}"
        mod = importlib.import_module(full_name)

        for attr_name, obj in inspect.getmembers(mod, inspect.isfunction):
            orig = inspect.unwrap(obj)
            if getattr(orig, "__module__", None) == mod.__name__:
                attach_function_to_object(cls, attr_name, obj)

        if attach_vars:
            for name, val in vars(mod).items():
                if not name.startswith("_") and not callable(val):
                    if isinstance(cls, dict):
                        cls[name] = val
                    else:
                        setattr(cls, name, val)
