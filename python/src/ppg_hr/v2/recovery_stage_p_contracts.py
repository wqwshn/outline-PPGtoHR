"""Frozen contracts shared by the LYX Stage P interaction modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields
from typing import Any

from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import AttemptIdentity

PENALTY_INTERACTION_STAGE = "penalty_interaction"
ROLLBACK_BACKUP_STAGE = "rollback_backup_matrix"
HISTORICAL_RECOVERY_AB_STAGE = "historical_recovery_ab"

EXPECTED_LOGICAL_RESULT_COUNT = 3 * 8 * 12
EXPECTED_REUSED_RESULT_COUNT = 8 * 12
EXPECTED_NEW_IDENTITY_COUNT = 2 * 8 * 12

EXPECTED_PENALTY_IDS = {
    "current_soft_penalty_control_v1",
    "resolution_adaptive_width_v1",
    "trusted_history_corridor_v1",
}

EXPECTED_SELECTION_RANKING_KEY = [
    "hard_gate_failure_count",
    "right_censored_recovery_count",
    "worst_l10",
    "worst_mae",
    "mean_mae",
    "mechanism_complexity",
    "penalty_id",
]

StagePProgressCallback = Callable[[Mapping[str, Any]], None]


class StagePPlanError(RuntimeError):
    """A Stage P artifact violates the frozen experiment contract."""


def require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StagePPlanError(f"{name}_must_be_object")
    return value


def require_list(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise StagePPlanError(f"{name}_must_be_array")
    return value


def require_hash(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StagePPlanError(f"{name}_must_be_lowercase_sha256")
    return value


def verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    artifact_name: str,
) -> str:
    declared = require_hash(hash_field, payload.get(hash_field))
    body = {key: value for key, value in payload.items() if key != hash_field}
    if canonical_sha256(body) != declared:
        raise StagePPlanError(f"{artifact_name}_hash_mismatch")
    return declared


def validate_recovery_candidate_registry(
    registry: Mapping[str, Any],
    *,
    artifact_name: str,
) -> dict[str, dict[str, Any]]:
    """Validate the full frozen recovery registry and every candidate body."""

    verify_embedded_hash(
        registry,
        hash_field="registry_sha256",
        artifact_name=artifact_name,
    )
    candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in (
            dict(require_mapping(f"{artifact_name}_candidate", raw))
            for raw in require_list(
                f"{artifact_name}_candidates",
                registry.get("candidates"),
            )
        )
    }
    if (
        len(candidates) != 3
        or registry.get("control_candidate_id") not in candidates
        or any(
            require_hash(
                f"{artifact_name}_candidate_sha256:{candidate_id}",
                candidate.get("candidate_sha256"),
            )
            != canonical_sha256(
                {key: value for key, value in candidate.items() if key != "candidate_sha256"}
            )
            for candidate_id, candidate in candidates.items()
        )
    ):
        raise StagePPlanError(f"{artifact_name}_mismatch")
    return candidates


def attempt_identity_from_item(
    item: Mapping[str, Any],
) -> AttemptIdentity:
    names = {field.name for field in fields(AttemptIdentity)}
    return AttemptIdentity(**{name: item[name] for name in names})


def merge_identity_result_metadata(
    *,
    item: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore frozen experimental coordinates omitted by the cache view."""

    fields_to_restore = (
        "solver_hash",
        "config_hash",
        "metric_contract_hash",
        "evaluation_hash",
        "data_sha256",
        "stage",
        "attempt_kind",
        "parent_experiment_id",
        "config",
        "data_path",
        "reference_path",
        "raw_data_sha256",
        "reference_sha256",
        "method_names",
        "matrix_role",
        "filter_profile_id",
        "filter_profile_sha256",
        "filter_profile_design_role",
        "physical_memory_ms",
        "actual_taps",
        "nominal_mu",
        "sentinel_role",
        "record_id",
        "scene",
        "true_rise_applicable",
        "recovery_candidate_id",
        "recovery_candidate_sha256",
        "candidate_min_bpm",
        "penalty_candidate_id",
        "penalty_candidate_sha256",
        "historical_parameter_source_sha256",
    )
    return {
        **{name: item[name] for name in fields_to_restore if name in item},
        **dict(row),
    }
