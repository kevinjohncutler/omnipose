"""
Fast-ish smoke test: make sure ``docs/conf.py`` executes without error.

Running a full Sphinx build inside CI is too slow (>20 s).  Importing the
configuration file alone catches 99 % of the problems that break building
(e.g. bad ``sys.path`` manipulation, missing modules, syntax errors).
"""

from __future__ import annotations

import runpy
from pathlib import Path
import os

from contextlib import contextmanager


@contextmanager
def _temp_env(**updates):
    """
    Temporarily set environment variables inside a `with` block.
    Restores the previous values afterward.
    """
    old = {k: os.environ.get(k) for k in updates}
    os.environ.update({k: v for k, v in updates.items() if v is not None})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_conf_py_executes():
    """Conf must run without raising any exception (isolated env)."""
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir  = repo_root / "docs"
    conf_path = docs_dir / "conf.py"

    cwd = os.getcwd()
    try:
        os.chdir(docs_dir)
        # Run conf.py but ensure NUMBA_DISABLE_JIT does not leak to other tests
        with _temp_env(NUMBA_DISABLE_JIT=None):
            runpy.run_path(conf_path, run_name="__docs_conf__")
    finally:
        os.chdir(cwd)