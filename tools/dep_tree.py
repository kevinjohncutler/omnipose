from __future__ import annotations

import argparse
import importlib.metadata as metadata
from typing import Iterable, Set


try:
    from packaging.requirements import Requirement
except Exception:  # pragma: no cover
    Requirement = None


def _iter_requirements(dist: metadata.Distribution) -> Iterable[str]:
    reqs = dist.requires or []
    for req in reqs:
        if Requirement is None:
            yield req.split(";")[0].strip().split()[0]
            continue
        try:
            parsed = Requirement(req)
        except Exception:
            continue
        if parsed.marker and not parsed.marker.evaluate():
            continue
        yield parsed.name


def _print_tree(name: str, indent: str, seen: Set[str], max_depth: int) -> None:
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        print(f"{indent}{name}==missing")
        return

    version = dist.version or "unknown"
    print(f"{indent}{name}=={version}")

    if max_depth == 0:
        return

    if name in seen:
        print(f"{indent}  - (cycle)")
        return

    seen.add(name)
    for child in sorted(set(_iter_requirements(dist))):
        _print_tree(child, indent + "  ", seen, max_depth - 1)
    seen.remove(name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print dependency tree with installed versions.")
    parser.add_argument("package", nargs="?", default="omnirefactor")
    parser.add_argument("--max-depth", type=int, default=20)
    args = parser.parse_args(argv)

    _print_tree(args.package, "", set(), args.max_depth)


if __name__ == "__main__":
    main()
