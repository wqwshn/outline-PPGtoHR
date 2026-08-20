"""Shared frozen contracts for the LYX Stage F modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields
from typing import Any

from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import (
    AttemptIdentity,
    BudgetContract,
    ExplorationRegistry,
)

_PROVISIONAL_STAGE = "penalty_interaction"


_CURRENT_ROLE_STAGE = "current_role_matrix"


_EXPECTED_SCENE_COUNTS = {
    "jianpan": 3,
    "kaihe": 3,
    "run": 3,
    "xiezi": 3,
}


_EXPECTED_FS_QUOTA = {25: 3, 50: 3, 100: 2}


_EXPECTED_ROLE_COUNTS = {"core": 6, "coverage_boundary": 2}


_RATE_NORMALIZED_PROFILE_IDS = {
    "p100-short-rate-normalized-low-40",
    "p100-short-rate-normalized-midlow-40",
}


_REUSED_RATE_NORMALIZED_P50_PROFILE_IDS = {
    "p50-short-low-40",
    "p50-short-midlow-40",
}


class StageFPlanError(RuntimeError):
    """The Stage F proposal violates a frozen experiment contract."""


StageFProgressCallback = Callable[[Mapping[str, Any]], None]


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageFPlanError(f"{name}_must_be_object")
    return value


def _require_list(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise StageFPlanError(f"{name}_must_be_array")
    return value


def _require_hash(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StageFPlanError(f"{name}_must_be_lowercase_sha256")
    return value


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    artifact_name: str,
) -> str:
    declared = _require_hash(hash_field, payload.get(hash_field))
    body = {
        key: value
        for key, value in payload.items()
        if key != hash_field
    }
    if canonical_sha256(body) != declared:
        raise StageFPlanError(f"{artifact_name}_hash_mismatch")
    return declared


def _budget_contract_from_payload(
    payload: Mapping[str, Any],
) -> BudgetContract:
    return BudgetContract(
        contract_version=str(payload["contract_version"]),
        stage_unique_limits=dict(
            _require_mapping(
                "stage_f_stage_unique_limits",
                payload.get("stage_unique_limits"),
            )
        ),
        normal_unique_identity_limit=payload.get(
            "normal_unique_identity_limit"
        ),
        supplemental_stage=(
            None
            if payload.get("supplemental_stage") is None
            else str(payload["supplemental_stage"])
        ),
        stage_attempt_kinds=dict(
            _require_mapping(
                "stage_f_stage_attempt_kinds",
                payload.get("stage_attempt_kinds"),
            )
        ),
        max_unique_identities=int(payload["max_unique_identities"]),
        max_attempts=int(payload["max_attempts"]),
        retry_limit=int(payload["retry_limit"]),
    )


def _exploration_registry_from_payload(
    payload: Mapping[str, Any],
) -> ExplorationRegistry:
    return ExplorationRegistry(
        registry_version=str(payload["registry_version"]),
        unique_budget=int(payload["unique_budget"]),
        allowed_identity_sha256=tuple(
            str(value)
            for value in _require_list(
                "stage_f_allowed_exploration_identities",
                payload.get("allowed_identity_sha256"),
            )
        ),
    )


def _attempt_identity_from_item(
    item: Mapping[str, Any],
) -> AttemptIdentity:
    names = {field.name for field in fields(AttemptIdentity)}
    return AttemptIdentity(**{name: item[name] for name in names})
