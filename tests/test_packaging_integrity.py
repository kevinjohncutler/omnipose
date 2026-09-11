from __future__ import annotations

from pathlib import Path
import importlib
import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

import sys
from pathlib import Path
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
from _git_helpers import REPO_ROOT, is_git_tracked, path_in_repo

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _module_from_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    module, sep, _ = value.partition(":")
    module = module.strip()
    return module if sep and module else None


def _iter_setuptools_scm_modules() -> set[str]:
    try:
        with PYPROJECT.open("rb") as fh:
            config = tomllib.load(fh)
    except FileNotFoundError:
        return set()
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"Unable to parse {PYPROJECT}: {exc}")
    scm_cfg = config.get("tool", {}).get("setuptools_scm", {})
    if not isinstance(scm_cfg, dict):
        return set()
    modules: set[str] = set()
    for key in ("version_scheme", "local_scheme", "fallback_version"):
        module = _module_from_string(scm_cfg.get(key))
        if module:
            modules.add(module)
    return modules


@pytest.mark.skipif(REPO_ROOT is None, reason="Not inside a Git repository")
def test_setuptools_scm_hooks_tracked():
    modules = _iter_setuptools_scm_modules()
    if not modules:
        pytest.skip("No custom setuptools_scm hooks configured")

    missing: list[str] = []
    untracked: list[Path] = []

    for module_name in sorted(modules):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing.append(f"{module_name}: {exc}")
            continue

        src = getattr(module, "__file__", None)
        if not src:
            continue
        path = Path(src).resolve()

        if not path_in_repo(path):
            continue

        if not is_git_tracked(path):
            untracked.append(path)

    if missing:
        pytest.fail(
            "pyproject.toml references setuptools_scm hooks that cannot be imported:\n"
            + "\n".join(missing)
        )

    assert not untracked, (
        "setuptools_scm hook modules exist locally but are not tracked by Git:\n"
        + "\n".join(str(p) for p in untracked)
    )
