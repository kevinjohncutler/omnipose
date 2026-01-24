from __future__ import annotations

import runpy
from pathlib import Path
import os
from contextlib import contextmanager


@contextmanager
def _temp_env(**updates):
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
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs"
    conf_path = docs_dir / "conf.py"

    cwd = os.getcwd()
    try:
        os.chdir(docs_dir)
        with _temp_env(NUMBA_DISABLE_JIT=None):
            runpy.run_path(conf_path, run_name="__docs_conf__")
    finally:
        os.chdir(cwd)
