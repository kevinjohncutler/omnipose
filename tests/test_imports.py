import pkgutil
import importlib
import inspect
import pytest
import omnipose

# --- Git repository root detection and tracking helper ---
from pathlib import Path
import subprocess

# Repository root is determined once; tests that require Git will be skipped
try:
    _REPO_ROOT = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
except Exception:
    _REPO_ROOT = None  # Git not available or not inside a repo

def _is_git_tracked(path: Path) -> bool:
    """
    Return True if *path* is tracked by Git.
    """
    if _REPO_ROOT is None:
        return False
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(_REPO_ROOT))],
            cwd=_REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, ValueError):
        return False

# Gather all omnipose submodules for import smoke-testing
modules = [
    module_name
    for _, module_name, _ in pkgutil.walk_packages(omnipose.__path__, omnipose.__name__ + ".")
]

@pytest.mark.parametrize("module_name", modules)
def test_module_import(module_name):
    # Smoke-test: import the submodule
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
    # Smoke-test: access each public member
    module = importlib.import_module(module_name)
    getattr(module, member_name)


# --- New test: Ensure all imported modules' sources are tracked by Git ---
@pytest.mark.parametrize("module_name", modules)
def test_module_source_tracked(module_name):
    """
    Ensure that every imported omnipose module's source file is tracked in Git.
    """
    if _REPO_ROOT is None:
        pytest.skip("Not inside a Git repository")

    module = importlib.import_module(module_name)
    src = getattr(module, "__file__", None)
    if src is None:
        pytest.skip(f"{module_name} has no __file__ attribute")

    path = Path(src)
    # Map cached byte-code back to its source file
    if path.suffix == ".pyc":
        path = path.with_suffix(".py")

    try:
        path.relative_to(_REPO_ROOT)
    except ValueError:
        pytest.skip(f"{module_name} is outside the repository")

    assert _is_git_tracked(path), f"{path} is not tracked by Git"