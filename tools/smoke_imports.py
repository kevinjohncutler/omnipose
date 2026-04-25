#!/usr/bin/env python3
"""Smoke-test imports by importing every module in a package."""
from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from typing import Iterable, List


def iter_modules(package, skip: Iterable[str]) -> List[str]:
    skip_list = list(skip)
    names: List[str] = []
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        name = module.name
        if any(token in name for token in skip_list):
            continue
        names.append(name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default="omnipose")
    parser.add_argument("--skip", action="append", default=[], help="substring to skip (repeatable)")
    args = parser.parse_args()

    package = importlib.import_module(args.package)
    failures = []
    for name in iter_modules(package, args.skip):
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append((name, exc))

    if failures:
        print("Import failures:")
        for name, exc in failures:
            print(f"- {name}: {exc.__class__.__name__}: {exc}")
        return 1
    print("All modules imported successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
