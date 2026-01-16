from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Optional

try:
    REPO_ROOT: Optional[Path] = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )
except Exception:
    REPO_ROOT = None


def _repo_relative(path: Path) -> Path:
    if REPO_ROOT is None:
        raise ValueError("repository root is unknown")
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"{resolved} is outside the repository") from exc


def is_git_tracked(path: Path) -> bool:
    if REPO_ROOT is None:
        return False
    try:
        rel = _repo_relative(path)
    except ValueError:
        return False
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(rel)],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def path_in_repo(path: Path) -> bool:
    if REPO_ROOT is None:
        return False
    try:
        _repo_relative(path)
        return True
    except ValueError:
        return False


__all__ = ["REPO_ROOT", "is_git_tracked", "path_in_repo"]
