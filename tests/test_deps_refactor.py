from __future__ import annotations
import importlib
import inspect
import pkgutil
import sys
import sysconfig
import pathlib
from importlib.metadata import packages_distributions

import pytest
import omnipose

root_pkgs = [omnipose]

_STDLIB_BASE = pathlib.Path(sysconfig.get_paths()["stdlib"]).resolve()
_DISTMAP_RAW = packages_distributions()

PACKAGE_ALIASES = {
    "colour": "colour_science",
    "skimage": "scikit_image",
}


def is_stdlib_module(name: str) -> bool:
    root = name.split(".", 1)[0]
    if root in sys.builtin_module_names:
        return True
    try:
        mod = importlib.import_module(root)
    except Exception:
        return False
    path = getattr(mod, "__file__", None)
    if path:
        p = pathlib.Path(path).resolve()
        if "site-packages" not in p.parts and "dist-packages" not in p.parts:
            try:
                if p.is_relative_to(_STDLIB_BASE):
                    return True
            except AttributeError:
                if str(p).startswith(str(_STDLIB_BASE)):
                    return True
    return False


def normalize(name: str) -> str:
    return name.replace("-", "_").lower()


IGNORED_ROOTS: set[str] = {"__main__", "__mp_main__", "test_deps_refactor"}


def is_ignored_root(name: str) -> bool:
    return name.startswith("_") or name in IGNORED_ROOTS


third_party: set[str] = set()
root_origins: dict[str, set[str]] = {}

optional_roots: set[str] = set()
for root_pkg in root_pkgs:
    for _, modname, _ in pkgutil.walk_packages(root_pkg.__path__, root_pkg.__name__ + "."):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for attr in mod.__dict__.values():
            if inspect.ismodule(attr):
                root = attr.__name__.split(".", 1)[0]
                if (
                    not is_stdlib_module(root)
                    and root not in ("omnipose",)
                    and not is_ignored_root(root)
                ):
                    root_norm = normalize(root)
                    third_party.add(root_norm)
                    root_origins.setdefault(root_norm, set()).add(modname)

third_party = {PACKAGE_ALIASES.get(pkg, pkg) for pkg in third_party}

root2dist = {k.lower(): [d.lower() for d in v] for k, v in _DISTMAP_RAW.items()}

resolved: set[str] = set()

for root in third_party:
    root_lc = root.lower()
    mapped = root2dist.get(root_lc, [])

    if mapped:
        if len(mapped) > 1:
            preferred = [root_lc] if root_lc in mapped else [mapped[0]]
        else:
            preferred = mapped
        for dist in preferred:
            resolved.add(normalize(dist))
    else:
        resolved.add(normalize(root_lc))

import importlib.metadata as _md
for root in list(third_party):
    root_norm = normalize(root)
    if root_norm not in {normalize(d) for d in resolved}:
        for dist in _md.distributions():
            files = getattr(dist, "files", None) or []
            for file in files:
                if file.parts and (file.parts[0] == root or file.parts[0] == root + ".py"):
                    pkg_name = normalize(dist.metadata["Name"])
                    resolved.add(pkg_name)
                    break
            if normalize(dist.metadata["Name"]) in resolved:
                break

from omnipose.dependencies import install_deps, gui_deps
from packaging.requirements import Requirement

DEPENDENCIES = install_deps + gui_deps

declared = {normalize(Requirement(dep).name) for dep in DEPENDENCIES}
missing = sorted((resolved - declared) - optional_roots)
unused = sorted(declared - resolved)
used = sorted(resolved & declared)

if missing:
    @pytest.mark.parametrize("pkg", missing)
    def test_missing_dependency(pkg: str) -> None:
        pytest.fail(f"`{pkg}` is imported but missing from dependencies.py")
else:
    def test_missing_dependency():
        pass


@pytest.mark.parametrize("pkg", used)
def test_used_requirement(pkg: str) -> None:
    pass


@pytest.mark.parametrize("pkg", unused)
@pytest.mark.xfail(reason="unused requirements are not fatal")
def test_unused_requirement(pkg: str) -> None:
    pass
