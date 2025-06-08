import pkgutil
import importlib
import inspect
import pytest
import omnipose

# Gather all omnipose submodules for import smoke‑testing
modules = [
    module_name
    for _, module_name, _ in pkgutil.walk_packages(omnipose.__path__, omnipose.__name__ + ".")
]

@pytest.mark.parametrize("module_name", modules)
def test_module_import(module_name):
    # Smoke‑test: import the submodule
    importlib.import_module(module_name)

# Collect public functions and classes from submodules that import successfully
members = []
for module_name in modules:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        # If import fails, rely on the separate import test to report the error.
        continue
    for name, obj in inspect.getmembers(module):
        if (inspect.isfunction(obj) or inspect.isclass(obj)) and not name.startswith("_"):
            members.append((module_name, name))

@pytest.mark.parametrize("module_name, member_name", members)
def test_member_access(module_name, member_name):
    # Smoke‑test: access each public member
    module = importlib.import_module(module_name)
    getattr(module, member_name)