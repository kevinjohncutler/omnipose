import os
import importlib
import importlib.machinery
import pkgutil
import inspect
from pathlib import Path
import pytest
import omnirefactor
import sys
from pathlib import Path
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from _git_helpers import REPO_ROOT as _REPO_ROOT, is_git_tracked as _is_git_tracked

_SKIP_GUI = os.environ.get("OMNIREF_SKIP_GUI_TESTS", "1") != "0"

modules = [
    module_name
    for _, module_name, _ in pkgutil.walk_packages(
        omnirefactor.__path__, omnirefactor.__name__ + "."
    )
]
if _SKIP_GUI:
    modules = [name for name in modules if not name.startswith("omnirefactor.gui")]


@pytest.mark.parametrize("module_name", modules)
def test_module_import(module_name):
    importlib.import_module(module_name)


members = []
for module_name in modules:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        continue
    for name, obj in inspect.getmembers(module):
        if (inspect.isfunction(obj) or inspect.isclass(obj)) and not name.startswith("_"):
            members.append((module_name, name))


@pytest.mark.parametrize("module_name, member_name", members)
def test_member_access(module_name, member_name):
    module = importlib.import_module(module_name)
    getattr(module, member_name)


@pytest.mark.parametrize("module_name", modules)
def test_module_source_tracked(module_name):
    if _REPO_ROOT is None:
        pytest.skip("Not inside a Git repository")

    module = importlib.import_module(module_name)
    src = getattr(module, "__file__", None)
    if src is None:
        pytest.skip(f"{module_name} has no __file__ attribute")

    path = Path(src)
    if path.suffix == ".pyc":
        path = path.with_suffix(".py")
    else:
        ext_suffix = next(
            (s for s in importlib.machinery.EXTENSION_SUFFIXES if path.name.endswith(s)),
            None,
        )
        if ext_suffix:
            stem = path.name[: -len(ext_suffix)]
            for replacement in (".pyx", ".py"):
                candidate = path.with_name(stem + replacement)
                if candidate.exists():
                    path = candidate
                    break

    try:
        path.relative_to(_REPO_ROOT)
    except ValueError:
        pytest.skip(f"{module_name} is outside the repository")

    assert _is_git_tracked(path), f"{path} is not tracked by Git"
