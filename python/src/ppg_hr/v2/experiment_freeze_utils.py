"""Shared source-identity and atomic-write helpers for experiment freezes."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import uuid
from collections import deque
from collections.abc import Sequence
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

from .recovery_contracts import canonical_sha256

V2_RUNTIME_ROOT_MODULES = (
    "ppg_hr.v2.recovery_candidates",
    "ppg_hr.v2.runtime_policy",
    "ppg_hr.v2.solver",
    "ppg_hr.v2.spectrum_tracking",
    "ppg_hr.v2.types",
)


def filesystem_path(path: Path) -> str:
    resolved = os.path.abspath(os.fspath(path))
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(filesystem_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{uuid.uuid4().hex}.tmp")
    with open(filesystem_path(temp), "w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(filesystem_path(temp), filesystem_path(path))


def _module_source_path(source_root: Path, module: str) -> Path | None:
    relative = Path(*module.split("."))
    module_path = source_root / relative.with_suffix(".py")
    if module_path.is_file():
        return module_path
    package_path = source_root / relative / "__init__.py"
    return package_path if package_path.is_file() else None


def _path_module(source_root: Path, path: Path) -> str:
    relative = path.relative_to(source_root)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _import_targets(
    *,
    tree: ast.AST,
    current_module: str,
    current_path: Path,
) -> set[str]:
    targets: set[str] = set()
    package_parts = current_module.split(".")
    if current_path.name != "__init__.py":
        package_parts.pop()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            keep = len(package_parts) - (node.level - 1)
            base_parts = package_parts[: max(0, keep)]
            if node.module:
                base_parts.extend(node.module.split("."))
            base = ".".join(base_parts)
        else:
            base = node.module or ""
        if base:
            targets.add(base)
        for alias in node.names:
            if alias.name != "*" and base:
                targets.add(f"{base}.{alias.name}")
    return {target for target in targets if target.startswith("ppg_hr")}


def runtime_dependency_closure(
    source_root: Path,
    *,
    root_modules: Sequence[str] = V2_RUNTIME_ROOT_MODULES,
) -> tuple[Path, ...]:
    """Resolve a local Python import closure from explicit runtime roots."""

    source_root = Path(source_root).resolve()
    roots = tuple(root_modules)
    queue = deque(roots)
    seen_modules: set[str] = set()
    paths: set[Path] = set()
    while queue:
        module = queue.popleft()
        if module in seen_modules:
            continue
        seen_modules.add(module)
        path = _module_source_path(source_root, module)
        if path is None:
            continue
        path = path.resolve()
        if path in paths:
            continue
        paths.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        current_module = _path_module(source_root, path)
        module_parts = current_module.split(".")
        package_parts = (
            module_parts
            if path.name == "__init__.py"
            else module_parts[:-1]
        )
        queue.extend(
            ".".join(package_parts[:index])
            for index in range(1, len(package_parts) + 1)
        )
        queue.extend(
            sorted(
                _import_targets(
                    tree=tree,
                    current_module=current_module,
                    current_path=path,
                )
            )
        )
    missing_roots = [
        module
        for module in roots
        if _module_source_path(source_root, module) is None
    ]
    if missing_roots:
        raise ValueError("runtime_root_missing:" + ",".join(missing_roots))
    return tuple(
        sorted(paths, key=lambda path: path.relative_to(source_root).as_posix())
    )


def runtime_source_identity(
    source_root: Path,
    *,
    root_modules: Sequence[str] = V2_RUNTIME_ROOT_MODULES,
) -> dict[str, Any]:
    """Hash a complete runtime closure, including package initializers."""

    source_root = Path(source_root).resolve()
    source_files = {
        path.relative_to(source_root).as_posix(): file_sha256(path)
        for path in runtime_dependency_closure(
            source_root,
            root_modules=root_modules,
        )
    }
    return {
        "source_files": source_files,
        "source_bundle_sha256": canonical_sha256(source_files),
    }


def runtime_config_schema_identity() -> dict[str, Any]:
    """Hash solver and run configuration schemas used by frozen candidates."""

    from ppg_hr.params import SolverParams

    from .types import V2RunConfig

    config_schemas: dict[str, dict[str, Any]] = {}
    for schema in (SolverParams, V2RunConfig):
        schema_fields: dict[str, Any] = {}
        for field in fields(schema):
            if field.default is not MISSING:
                default: Any = field.default
            elif field.default_factory is not MISSING:
                default = field.default_factory()
            else:
                default = "<required>"
            schema_fields[field.name] = {
                "annotation": str(field.type),
                "default": default,
            }
        config_schemas[schema.__name__] = schema_fields
    return {
        "config_schemas": config_schemas,
        "config_schema_sha256": canonical_sha256(config_schemas),
    }
