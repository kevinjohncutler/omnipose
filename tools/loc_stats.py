#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
}


@dataclass
class Stats:
    files: int = 0
    lines: int = 0
    functions: int = 0

    def __iadd__(self, other: "Stats") -> "Stats":
        self.files += other.files
        self.lines += other.lines
        self.functions += other.functions
        return self


def iter_files(root: Path, extensions: set[str]) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_dir():
            if path.name in EXCLUDED_DIRS:
                # Skip excluded directories entirely.
                for _ in path.rglob("*"):
                    pass
                continue
            continue
        if path.suffix in extensions:
            yield path


def count_loc(lines: Iterable[str]) -> int:
    total = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        total += 1
    return total


def count_functions(tree: ast.AST, top_level_only: bool) -> int:
    if top_level_only:
        return sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in tree.body
            if isinstance(node, ast.AST)
        )
    return sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in ast.walk(tree)
    )


def stats_for_file(path: Path, top_level_only: bool) -> Stats:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    loc = count_loc(lines)
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        functions = 0
    else:
        functions = count_functions(tree, top_level_only=top_level_only)
    return Stats(files=1, lines=loc, functions=functions)


def compute_stats(root: Path, extensions: set[str], top_level_only: bool) -> Stats:
    stats = Stats()
    for path in iter_files(root, extensions):
        stats += stats_for_file(path, top_level_only=top_level_only)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count lines of code (non-empty, non-comment) and functions in a repo."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan (default: current directory).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".py"],
        help="File extensions to include (default: .py).",
    )
    parser.add_argument(
        "--top-level",
        action="store_true",
        help="Count only top-level functions (exclude methods and nested defs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.extensions}
    stats = compute_stats(root, extensions, top_level_only=args.top_level)

    print(f"Root: {root}")
    print(f"Extensions: {', '.join(sorted(extensions))}")
    print(f"Files: {stats.files}")
    print(f"Lines of code: {stats.lines}")
    print(f"Functions: {stats.functions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
