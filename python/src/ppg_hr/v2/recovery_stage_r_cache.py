"""Content-addressed cache and resumable single-identity Stage R execution."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from .bo_space_generalization import _cache_json_ready
from .phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    read_json,
)
from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import (
    AttemptIdentity,
    AttemptRegistry,
    CacheEvidence,
)
from .recovery_stage_r_common import (
    StageRNumericalResult,
    StageRNumericalRunner,
    StageRPlanError,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageRPlanError(f"{name}_must_be_object")
    return value


def _attempt_identity_from_item(
    item: Mapping[str, Any],
) -> AttemptIdentity:
    names = {field.name for field in fields(AttemptIdentity)}
    return AttemptIdentity(**{name: item[name] for name in names})


def _write_stage_r_cache_result(
    *,
    result_dir: Path,
    identity: AttemptIdentity,
    item: Mapping[str, Any],
    numerical: StageRNumericalResult,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = result_dir / "trajectory.npz"
    trajectory_temp = result_dir / (
        f".trajectory.{uuid.uuid4().hex}.tmp"
    )
    with trajectory_temp.open("wb") as handle:
        np.savez_compressed(
            handle,
            HR=np.asarray(numerical.solver_result.HR, dtype=float),
        )
    os.replace(trajectory_temp, trajectory_path)
    details_path = result_dir / "solver_details.json"
    atomic_write_json(
        details_path,
        _cache_json_ready(
            {
                "err_stats": numerical.solver_result.err_stats,
                "metadata": numerical.solver_result.metadata,
                "window_table": numerical.solver_result.window_table,
            }
        ),
    )
    result_path = result_dir / "result.json"
    payload = {
        "producer": "content_addressed_solver_cache_v1",
        "result_version": "lyx_stage_r_solver_result_v1",
        "status": "complete",
        "valid": True,
        "identity": identity.to_dict(),
        "config": _json_ready(item["config"]),
        "data_identity": {
            "raw_data_sha256": item["raw_data_sha256"],
            "reference_sha256": item["reference_sha256"],
            "combined_data_sha256": item["data_sha256"],
        },
        "coordinate": {
            key: item[key]
            for key in (
                "stage",
                "scene",
                "record_id",
                "filter_profile_id",
                "recovery_candidate_id",
                "candidate_min_bpm",
                "penalty_candidate_id",
            )
        },
        "metrics": _json_ready(dict(numerical.metrics)),
        "spectral_audit": (
            None
            if numerical.spectral_audit is None
            else _json_ready(dict(numerical.spectral_audit))
        ),
        "trajectory": {
            "path": trajectory_path.name,
            "sha256": file_sha256(trajectory_path),
            "solver_details_path": details_path.name,
            "solver_details_sha256": file_sha256(details_path),
        },
    }
    atomic_write_json(result_path, payload)
    receipt_path = result_dir / "cache_receipt.json"
    atomic_write_json(
        receipt_path,
        {
            "identity_sha256": identity.sha256,
            "result_path": result_path.name,
            "result_sha256": file_sha256(result_path),
        },
    )
    return payload


def _load_stage_r_cache_result(
    *,
    evidence: CacheEvidence,
) -> dict[str, Any]:
    payload = read_json(evidence.result_path)
    identity = _require_mapping(
        "stage_r_cached_identity",
        payload.get("identity"),
    )
    config = _require_mapping(
        "stage_r_cached_config",
        payload.get("config"),
    )
    data_identity = _require_mapping(
        "stage_r_cached_data_identity",
        payload.get("data_identity"),
    )
    if (
        canonical_sha256(config) != identity.get("config_hash")
        or data_identity.get("combined_data_sha256")
        != identity.get("data_sha256")
    ):
        raise StageRPlanError(
            "stage_r_cached_result_identity_mismatch"
        )
    trajectory = _require_mapping(
        "stage_r_trajectory",
        payload.get("trajectory"),
    )
    for path_field, hash_field in (
        ("path", "sha256"),
        ("solver_details_path", "solver_details_sha256"),
    ):
        path = (
            evidence.result_path.parent
            / str(trajectory[path_field])
        ).resolve()
        if (
            not path.is_relative_to(evidence.result_path.parent)
            or not path.is_file()
            or file_sha256(path) != trajectory.get(hash_field)
        ):
            raise StageRPlanError(
                f"stage_r_cached_trajectory_hash_mismatch:{path_field}"
            )
    _require_mapping("stage_r_cached_metrics", payload.get("metrics"))
    return payload


def execute_stage_r_identity(
    *,
    registry: AttemptRegistry,
    item: dict[str, Any],
    numerical_runner: StageRNumericalRunner,
    spectral_audit_dir: Path,
) -> dict[str, Any]:
    """Resolve one identity from immutable cache or one charged solver run."""

    identity = _attempt_identity_from_item(item)
    if canonical_sha256(_json_ready(item["config"])) != identity.config_hash:
        raise StageRPlanError(
            f"stage_r_identity_config_hash_mismatch:{identity.sha256}"
        )
    result_dir = registry.trusted_cache_root / identity.sha256
    receipt_path = result_dir / "cache_receipt.json"
    cache_hit = receipt_path.is_file()
    if cache_hit:
        evidence = CacheEvidence.from_path(
            receipt_path,
            expected_identity=identity,
            trusted_cache_root=registry.trusted_cache_root,
        )
        payload = _load_stage_r_cache_result(evidence=evidence)
        recovery = registry.reconcile_interrupted_attempt(
            identity,
            evidence=evidence,
        )
        if recovery == "no_running_attempt":
            registry.record_cache_hit(identity, evidence=evidence)
    else:
        registry.reconcile_interrupted_attempt(
            identity,
            evidence=None,
        )

        def operation() -> dict[str, Any]:
            numerical = numerical_runner(item, spectral_audit_dir)
            return _write_stage_r_cache_result(
                result_dir=result_dir,
                identity=identity,
                item=item,
                numerical=numerical,
            )

        payload = registry.execute_registered(identity, operation)
        evidence = CacheEvidence.from_path(
            receipt_path,
            expected_identity=identity,
            trusted_cache_root=registry.trusted_cache_root,
        )
        registry.bind_cache_evidence(identity, evidence=evidence)
    provenance = registry.matrix_execution_summary((identity,))
    cache_hit = (
        provenance["identity_with_solver_attempt_count"] == 0
    )
    return {
        "identity_sha256": identity.sha256,
        "stage": item["stage"],
        "record_id": item["record_id"],
        "scene": item["scene"],
        "filter_profile_id": item["filter_profile_id"],
        "recovery_candidate_id": item["recovery_candidate_id"],
        "candidate_min_bpm": item["candidate_min_bpm"],
        "cache_hit": cache_hit,
        "cache_receipt_sha256": evidence.receipt_sha256,
        "result_sha256": evidence.result_sha256,
        "metrics": dict(payload["metrics"]),
        "spectral_audit": payload.get("spectral_audit"),
    }
