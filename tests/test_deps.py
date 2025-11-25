from __future__ import annotations
import importlib, inspect, pkgutil, sys, sysconfig, pathlib
from importlib.metadata import packages_distributions

import pytest
import omnipose
import cellpose_omni

# ------------------------------------------------------------------ #
# 0.  Packages whose imports we want to analyse                      #
# ------------------------------------------------------------------ #
root_pkgs = [omnipose]
if cellpose_omni is not None:
    root_pkgs.append(cellpose_omni)

# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
_STDLIB_BASE = pathlib.Path(sysconfig.get_paths()["stdlib"]).resolve()
_DISTMAP_RAW  = packages_distributions()          # {top-level-import: [dists…]}

# print(_DISTMAP_RAW.get('qdarktheme'))

def is_stdlib_module(name: str) -> bool:
    """
    Return *True* iff ``name`` lives in the Python *standard library*.

    A module qualifies as std-lib when **either**

    1. it is a built-in C extension (appears in ``sys.builtin_module_names``); or
    2. its file path resolves inside the interpreter’s own *stdlib* directory
       **and** it is *not* installed inside ``site-packages`` / ``dist-packages``.

    Everything else - including anything that import-metadata can map to a wheel
    - is treated as *third-party* and therefore yields *False*.
    """
    root = name.split(".", 1)[0]

    # 1) built-in modules
    if root in sys.builtin_module_names:
        return True

    # Attempt to import the root; if that fails, assume third-party
    try:
        mod = importlib.import_module(root)
    except Exception:
        return False

    # 2) inspect the on-disk location
    path = getattr(mod, "__file__", None)
    if path:
        p = pathlib.Path(path).resolve()

        # ignore anything living in site-/dist-packages
        if "site-packages" not in p.parts and "dist-packages" not in p.parts:
            try:  # Python 3.9+
                if p.is_relative_to(_STDLIB_BASE):
                    return True
            except AttributeError:  # <3.9 fallback
                if str(p).startswith(str(_STDLIB_BASE)):
                    return True

    # If importlib.metadata can resolve it to a distribution it is definitely
    # *not* std-lib, so fall back to *False* (third-party).
    return False


def normalize(name: str) -> str:
    """
    Return a lowercase identifier with *both* hyphens **and dots**
    normalised to underscores.

    This lets us compare things like

        colour-science  ↔  colour_science
        PyQt6.sip       ↔  pyqt6_sip
    """
    # return name.replace("-", "_").replace(".", "_").lower()
    return name.lower()
    
    

IGNORED_ROOTS: set[str] = {"__main__", "__mp_main__", "test_deps"}

def is_ignored_root(name: str) -> bool:
    return name.startswith("_") or name in IGNORED_ROOTS


# ------------------------------------------------------------------ #
# 1.  Discover external top-level imports                            #
# ------------------------------------------------------------------ #
third_party: set[str]           = set()
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
                    and root not in ("omnipose", "cellpose_omni")
                    and not is_ignored_root(root)
                ):
                    root_norm = normalize(root)
                    third_party.add(root_norm)
                    root_origins.setdefault(root_norm, set()).add(modname)

print("Import roots mapping:")
for root, origins in sorted(root_origins.items()):
    print(f"  {root}: {', '.join(sorted(origins))}")
print("Optional import roots (ignored):", sorted(optional_roots))

# ------------------------------------------------------------------ #
# 2.  Map import roots ➜ distribution names (case-insensitive)       #
# ------------------------------------------------------------------ #
root2dist = {k.lower(): [d.lower() for d in v] for k, v in _DISTMAP_RAW.items()}

# print(root2dist)
# for k,v in root2dist.items():
#     if len(v) > 1:
#         print(f"Multiple distributions for {k}: {', '.join(v)}")
# root2dist = {k.lower(): [d[0].lower()] for k, v in _DISTMAP_RAW.items()}


resolved: set[str] = set()

for root in third_party:
    root_lc = root.lower()
    mapped = root2dist.get(root_lc, [])

    if mapped:
        # Handle multiple distributions: prefer one matching the import root if present
        if len(mapped) > 1:
            preferred = [root_lc] if root_lc in mapped else [mapped[0]]
        else:
            preferred = mapped
        for dist in preferred:
            resolved.add(normalize(dist))
    else:
        # Fallback: keep the bare import root itself
        resolved.add(normalize(root_lc))

# Fallback: for any root not in _DISTMAP_RAW, scan distributions for matching top-level modules
import importlib.metadata as _md
for root in list(third_party):
    root_norm = normalize(root)
    if root_norm not in {normalize(d) for d in resolved}:
        for dist in _md.distributions():
            files = getattr(dist, 'files', None) or []
            for file in files:
                # file.parts like ('qdarktheme', '__init__.py') or ('qdarktheme.py',)
                if file.parts and (file.parts[0] == root or file.parts[0] == root + '.py'):
                    pkg_name = normalize(dist.metadata['Name'])
                    resolved.add(pkg_name)
                    break
            if normalize(dist.metadata['Name']) in resolved:
                break

# ------------------------------------------------------------------ #
# 3.  Compare against omnipose.dependencies                          #
# ------------------------------------------------------------------ #
from omnipose.dependencies import install_deps, gui_deps
from packaging.requirements import Requirement

DEPENDENCIES = install_deps + gui_deps
declared = {normalize(Requirement(dep).name) for dep in DEPENDENCIES}
missing = sorted((resolved - declared) - optional_roots)   # imports that have no matching requirement
unused  = sorted(declared - resolved)   # requirements that map to no imported dist
used = sorted(resolved & declared)



# ------------------------------------------------------------------ #
# 4.  Tests                                                          #
# ------------------------------------------------------------------ #
if missing:
    @pytest.mark.parametrize("pkg", missing)
    def test_missing_dependency(pkg: str) -> None:
        pytest.fail(f"`{pkg}` is imported but missing from dependencies.py")
else:
    def test_missing_dependency():
        """No missing dependencies."""
        pass

@pytest.mark.parametrize("pkg", used)
def test_used_requirement(pkg: str) -> None:
    """
    Positive check — this requirement is declared **and** actually imported
    somewhere in the code-base.  The empty body is enough for pytest to
    report a distinct ‘PASSED’ for each such package.
    """
    pass


# @pytest.mark.parametrize("pkg", unused)
# @pytest.mark.xfail(reason="unused requirements are not fatal")
# def test_unused_requirement(pkg: str) -> None:
#     pass
    