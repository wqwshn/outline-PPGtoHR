"""Fail-closed artifact freezing for Stage R recovery contracts."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import uuid
from collections import deque
from collections.abc import Sequence
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

from .recovery_candidates import (
    RecoveryCandidate,
    RecoveryCandidateError,
    recovery_candidates_v1,
)
from .recovery_contracts import (
    RECOVERY_PREFLIGHT_HASH_FIELDS,
    canonical_sha256,
    require_sha256,
)
from .recovery_selection import recovery_selection_contract_v1

_RUNTIME_ROOT_MODULES = (
    "ppg_hr.v2.recovery_candidates",
    "ppg_hr.v2.runtime_policy",
    "ppg_hr.v2.solver",
    "ppg_hr.v2.spectrum_tracking",
    "ppg_hr.v2.types",
)
_EXPECTED_IDS = (
    "current_fixed_floor_control_v1",
    "relative_gap_timeout_v1",
    "relative_gap_rise_guard_v1",
)


def _filesystem_path(path: Path) -> str:
    resolved = os.path.abspath(os.fspath(path))
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_filesystem_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{uuid.uuid4().hex}.tmp")
    with open(_filesystem_path(temp), "w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(_filesystem_path(temp), _filesystem_path(path))


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


def runtime_dependency_closure(source_root: Path) -> tuple[Path, ...]:
    """Resolve the local Python import closure of the recovery runtime."""

    source_root = Path(source_root).resolve()
    queue = deque(_RUNTIME_ROOT_MODULES)
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
        for module in _RUNTIME_ROOT_MODULES
        if _module_source_path(source_root, module) is None
    ]
    if missing_roots:
        raise RecoveryCandidateError(
            "recovery_runtime_root_missing:" + ",".join(missing_roots)
        )
    return tuple(
        sorted(paths, key=lambda path: path.relative_to(source_root).as_posix())
    )


def runtime_source_identity(source_root: Path) -> dict[str, Any]:
    """Hash the complete runtime closure, including package initializers."""

    source_root = Path(source_root).resolve()
    source_files = {
        path.relative_to(source_root).as_posix(): _file_sha256(path)
        for path in runtime_dependency_closure(source_root)
    }
    return {
        "source_files": source_files,
        "source_bundle_sha256": canonical_sha256(source_files),
    }


def freeze_recovery_candidate_registry(
    candidates: Sequence[RecoveryCandidate],
    *,
    solver_hash: str,
    config_schema_hash: str,
) -> dict[str, Any]:
    """Freeze exactly one control and two new candidates without solver runs."""

    try:
        require_sha256("solver_hash", solver_hash)
        require_sha256("config_schema_hash", config_schema_hash)
    except ValueError as exc:
        raise RecoveryCandidateError("invalid_recovery_registry_hash") from exc
    frozen = tuple(candidates)
    if len(frozen) != 3:
        raise RecoveryCandidateError("candidate_count_must_be_three")
    if tuple(candidate.candidate_id for candidate in frozen) != _EXPECTED_IDS:
        raise RecoveryCandidateError("candidate_identity_or_order_mismatch")
    if sum(candidate.design_role == "control" for candidate in frozen) != 1:
        raise RecoveryCandidateError("exactly_one_control_required")
    if len({candidate.sha256 for candidate in frozen}) != len(frozen):
        raise RecoveryCandidateError("duplicate_candidate_identity")
    payload = {
        "registry_version": "lyx_recovery_candidate_registry_v1",
        "status": "frozen_zero_formal_runs",
        "solver_hash": solver_hash,
        "config_schema_hash": config_schema_hash,
        "candidate_count": 3,
        "control_candidate_id": _EXPECTED_IDS[0],
        "new_candidate_count": 2,
        "formal_solver_run_count": 0,
        "uses_reference_hr_online": False,
        "candidates": [
            {
                **candidate.to_dict(),
                "candidate_sha256": candidate.sha256,
            }
            for candidate in frozen
        ],
    }
    payload["registry_sha256"] = canonical_sha256(payload)
    return payload


def freeze_recovery_candidate_artifacts(
    *,
    output_dir: Path,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Atomically freeze Stage R contracts without evaluating any record."""

    if output_dir.exists():
        raise RecoveryCandidateError(
            "recovery_candidate_freeze_output_already_exists"
        )
    if source_root is None:
        source_root = Path(__file__).resolve().parents[2]
    source_root = Path(source_root).resolve()
    source_identity = runtime_source_identity(source_root)
    source_files = source_identity["source_files"]
    source_bundle_sha256 = source_identity["source_bundle_sha256"]

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
    config_schema_sha256 = canonical_sha256(config_schemas)
    registry = freeze_recovery_candidate_registry(
        recovery_candidates_v1(),
        solver_hash=source_bundle_sha256,
        config_schema_hash=config_schema_sha256,
    )
    selection = recovery_selection_contract_v1()

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(
        f".{output_dir.name}.{uuid.uuid4().hex}.staging"
    )
    try:
        os.makedirs(_filesystem_path(staging))
        registry_path = staging / "recovery_candidate_registry.json"
        selection_path = staging / "recovery_selection_contract.json"
        _atomic_write_json(registry_path, registry)
        _atomic_write_json(selection_path, selection)
        receipt = {
            "receipt_version": "lyx_recovery_candidate_freeze_receipt_v1",
            "status": "frozen_zero_formal_runs",
            "source_files": source_files,
            "source_bundle_sha256": source_bundle_sha256,
            "config_schemas": config_schemas,
            "config_schema_sha256": config_schema_sha256,
            "recovery_candidate_registry_sha256": (
                registry["registry_sha256"]
            ),
            "recovery_selection_contract_sha256": (
                selection["contract_sha256"]
            ),
            "formal_solver_run_count": 0,
            "diagnostic_solver_run_count": 0,
            "independent_bo_run_count": 0,
            "preflight_status": "awaiting_full_preflight_contracts",
            "required_preflight_hashes": list(
                RECOVERY_PREFLIGHT_HASH_FIELDS
            ),
            "artifacts": {
                registry_path.name: _file_sha256(registry_path),
                selection_path.name: _file_sha256(selection_path),
            },
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _atomic_write_json(
            staging / "recovery_candidate_freeze_receipt.json",
            receipt,
        )
        os.replace(_filesystem_path(staging), _filesystem_path(output_dir))
    except BaseException:
        if staging.exists():
            shutil.rmtree(_filesystem_path(staging))
        raise
    return receipt
