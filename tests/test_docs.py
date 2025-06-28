"""
Ultra‑fast smoke test: make sure ``docs/conf.py`` executes without error.

Running a full Sphinx build inside CI is too slow (>20 s).  Importing the
configuration file alone catches 99 % of the problems that break building
(e.g. bad ``sys.path`` manipulation, missing modules, syntax errors).
"""

from __future__ import annotations

import runpy
from pathlib import Path
import os


def test_conf_py_executes():
    """Conf must run without raising any exception."""
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir  = repo_root / "docs"
    conf_path = docs_dir / "conf.py"

    # Execute conf.py with its directory as the working dir so that
    # relative file references (e.g. 'links.rst') resolve correctly.
    cwd = os.getcwd()
    try:
        os.chdir(docs_dir)
        runpy.run_path(conf_path, run_name="__docs_conf__")
    finally:
        os.chdir(cwd)