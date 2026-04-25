#!/usr/bin/env python3
"""Check for missing globals in functions and circular imports."""
from __future__ import annotations

import argparse
import ast
import builtins
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class ModuleInfo:
    name: str
    path: Path


def iter_py_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if path.name == "__pycache__":
            continue
        yield path


def module_name_for(path: Path, pkg_root: Path, pkg_name: str) -> str:
    rel = path.relative_to(pkg_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join([pkg_name] + parts)


def resolve_relative(module: str, level: int, name: Optional[str]) -> str:
    if level == 0:
        return name or ""
    parts = module.split(".")
    if level > len(parts):
        base = []
    else:
        base = parts[: -level]
    if name:
        base.append(name)
    return ".".join(base)


def parse_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def collect_module_scope(tree: ast.Module) -> Set[str]:
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(collect_targets(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(collect_targets(node.target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name)
    return names


def collect_targets(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            names.update(collect_targets(elt))
    return names


def collect_all_names(tree: ast.Module) -> Optional[Set[str]]:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return extract_str_list(node.value)
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                return extract_str_list(node.value)
    return None


def extract_str_list(node: ast.AST) -> Optional[Set[str]]:
    if isinstance(node, (ast.List, ast.Tuple)):
        items = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                items.append(elt.value)
            else:
                return None
        return set(items)
    return None


@dataclass
class Imports:
    explicit: Set[str]
    star: Set[str]


def collect_imports(tree: ast.Module, module: str, pkg_modules: Set[str]) -> Imports:
    explicit: Set[str] = set()
    star: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                explicit.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            target = resolve_relative(module, node.level, node.module)
            for alias in node.names:
                if alias.name == "*":
                    if target in pkg_modules:
                        star.add(target)
                else:
                    explicit.add(alias.asname or alias.name)
    return Imports(explicit=explicit, star=star)


class FunctionUsage(ast.NodeVisitor):
    def __init__(self) -> None:
        self.used: Set[str] = set()
        self.defined: Set[str] = set()
        self.globals: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.used.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.defined.add(node.id)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self.defined.add(node.arg)

    def visit_Global(self, node: ast.Global) -> None:
        self.globals.update(node.names)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Skip annotations
        if node.value:
            self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.defined.add(node.target.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Don't recurse into nested functions
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass


def collect_missing_names(tree: ast.Module, module_name: str, module_scope: Set[str],
                          star_exports: Set[str]) -> Dict[str, Set[str]]:
    missing: Dict[str, Set[str]] = {}
    builtins_set = set(dir(builtins))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visitor = FunctionUsage()
            visitor.visit(node)
            # Args
            for arg in node.args.args + node.args.kwonlyargs:
                visitor.defined.add(arg.arg)
            if node.args.vararg:
                visitor.defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                visitor.defined.add(node.args.kwarg.arg)

            available = set(module_scope) | star_exports | visitor.globals | builtins_set
            unresolved = {n for n in visitor.used if n not in visitor.defined and n not in available}
            if unresolved:
                missing[node.name] = unresolved
    return missing


def build_exports(module_name: str, module_map: Dict[str, ModuleInfo], memo: Dict[str, Set[str]]) -> Set[str]:
    if module_name in memo:
        return memo[module_name]
    info = module_map.get(module_name)
    if not info:
        memo[module_name] = set()
        return memo[module_name]
    tree = parse_ast(info.path)
    module_scope = collect_module_scope(tree)
    exports = collect_all_names(tree)
    imports = collect_imports(tree, module_name, set(module_map))
    star_exports: Set[str] = set()
    for target in imports.star:
        star_exports |= build_exports(target, module_map, memo)
    if exports is None:
        exports = {n for n in module_scope | star_exports if not n.startswith("_")}
    memo[module_name] = exports
    return exports


def build_import_graph(module_map: Dict[str, ModuleInfo]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {name: set() for name in module_map}
    pkg_modules = set(module_map)
    for name, info in module_map.items():
        tree = parse_ast(info.path)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                target = resolve_relative(name, node.level, node.module)
                if target in pkg_modules:
                    graph[name].add(target)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if mod in pkg_modules:
                        graph[name].add(mod)
    return graph


def find_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    cycles: List[List[str]] = []
    temp: Set[str] = set()
    perm: Set[str] = set()

    def visit(node: str, stack: List[str]) -> None:
        if node in perm:
            return
        if node in temp:
            if node in stack:
                idx = stack.index(node)
                cycles.append(stack[idx:] + [node])
            return
        temp.add(node)
        for nxt in graph.get(node, []):
            visit(nxt, stack + [nxt])
        temp.remove(node)
        perm.add(node)

    for node in graph:
        visit(node, [node])
    return cycles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path("refactor/src/omnipose"))
    parser.add_argument("--package-name", type=str, default="omnipose")
    args = parser.parse_args()

    pkg_root = args.package_root.resolve()
    pkg_name = args.package_name

    module_map: Dict[str, ModuleInfo] = {}
    for path in iter_py_files(pkg_root):
        name = module_name_for(path, pkg_root, pkg_name)
        module_map[name] = ModuleInfo(name=name, path=path)

    export_memo: Dict[str, Set[str]] = {}
    pkg_modules = set(module_map)

    missing_total: Dict[str, Dict[str, Set[str]]] = {}
    for module_name, info in module_map.items():
        tree = parse_ast(info.path)
        module_scope = collect_module_scope(tree)
        imports = collect_imports(tree, module_name, pkg_modules)
        star_exports: Set[str] = set()
        for target in imports.star:
            star_exports |= build_exports(target, module_map, export_memo)
        module_scope |= imports.explicit
        missing = collect_missing_names(tree, module_name, module_scope, star_exports)
        if missing:
            missing_total[module_name] = missing

    graph = build_import_graph(module_map)
    cycles = find_cycles(graph)

    if cycles:
        print("Import cycles detected:")
        for cycle in cycles:
            print("  " + " -> ".join(cycle))

    if missing_total:
        print("\nMissing names by function:")
        for module_name, funcs in sorted(missing_total.items()):
            print(f"\n{module_name}:")
            for func, names in sorted(funcs.items()):
                joined = ", ".join(sorted(names))
                print(f"  {func}: {joined}")

    if cycles or missing_total:
        return 1
    print("No missing names or import cycles detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
