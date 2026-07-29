"""Pre-register the bounded LYX Stage R diagnostic and sentinel matrices.

This module deliberately stops before numerical execution.  It freezes the
exact 60 diagnostic and 108 formal identities, then requires an exact human
authorization receipt before a later executor may register or solve them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .experiment_freeze_utils import runtime_source_identity
from .phase2_experiment_io import atomic_write_json, file_sha256, read_json
from .recovery_contracts import canonical_sha256, require_sha256
from .recovery_experiment_governance import AttemptIdentity
from .recovery_filter_stability import StabilityAuditContract
from .recovery_profile_metrics import (
    RECOVERY_PROFILE_METRIC_VERSION,
    RECOVERY_PROFILE_SMOOTH_WIN_LEN,
    RECOVERY_PROFILE_TIME_BIAS_S,
)

_EXPECTED_SCENE_COUNTS = {
    "jianpan": 3,
    "kaihe": 3,
    "run": 3,
    "xiezi": 3,
}
_SENTINEL_ROLES = ("conservative", "intermediate", "aggressive")
_FIXED_FLOOR_BPM = (85.0, 80.0, 70.0, 60.0, 50.0)
_DIAGNOSTIC_STAGE = "fixed_lower_bound_diagnostic"
_FORMAL_STAGE = "recovery_sentinel"
_AUTHORIZATION_STATE = "awaiting_human_stage_r_execution_decision"


class StageRPlanError(RuntimeError):
    """A frozen source or Stage R identity is incomplete or inconsistent."""


class StageRAuthorizationError(StageRPlanError):
    """The exact 168-identity execution proposal has not been approved."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageRPlanError(f"{name}_must_be_object")
    return value


def _require_list(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise StageRPlanError(f"{name}_must_be_array")
    return value


def _require_hash(name: str, value: object) -> str:
    text = str(value)
    try:
        require_sha256(name, text)
    except ValueError as exc:
        raise StageRPlanError(str(exc)) from exc
    return text


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    artifact_name: str,
) -> str:
    declared = _require_hash(hash_field, payload.get(hash_field))
    unhashed = dict(payload)
    unhashed.pop(hash_field, None)
    if canonical_sha256(unhashed) != declared:
        raise StageRPlanError(f"{artifact_name}_hash_mismatch")
    return declared


def _verify_filter_profile_library(
    profile_library: Mapping[str, Any],
) -> None:
    raw_profiles = _require_list(
        "filter_profiles",
        profile_library.get("profiles"),
    )
    for raw in raw_profiles:
        profile = _require_mapping("filter_profile", raw)
        declared = _require_hash(
            "profile_sha256",
            profile.get("profile_sha256"),
        )
        profile_identity = {
            "profile_id": profile.get("profile_id"),
            "design_role": profile.get("design_role"),
            "fs_target": profile.get("fs_target"),
            "memory_ms": profile.get("physical_memory_ms"),
            "nominal_mu": profile.get("nominal_mu"),
            "recovery_sentinel_role": profile.get(
                "recovery_sentinel_role"
            ),
            "actual_taps": profile.get("actual_taps"),
        }
        if canonical_sha256(profile_identity) != declared:
            raise StageRPlanError(
                "filter_profile_hash_mismatch:"
                f"{profile.get('profile_id', '')}"
            )
    _verify_embedded_hash(
        profile_library,
        hash_field="library_sha256",
        artifact_name="filter_profile_library",
    )


def _verify_registry(
    registry: Mapping[str, Any],
    *,
    candidates_field: str,
    candidate_name: str,
    candidate_id_field: str,
    candidate_hash_field: str,
    artifact_name: str,
) -> None:
    raw_candidates = _require_list(
        candidates_field,
        registry.get(candidates_field),
    )
    for raw in raw_candidates:
        candidate = _require_mapping(candidate_name, raw)
        _verify_embedded_hash(
            candidate,
            hash_field=candidate_hash_field,
            artifact_name=(
                f"{candidate_name}:"
                f"{candidate.get(candidate_id_field, '')}"
            ),
        )
    _verify_embedded_hash(
        registry,
        hash_field="registry_sha256",
        artifact_name=artifact_name,
    )


def _records_from_sources(
    baseline_manifest: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_manifest_records = _require_list(
        "baseline_manifest_records",
        baseline_manifest.get("records"),
    )
    raw_metric_records = _require_list(
        "baseline_metric_records",
        baseline_metrics.get("records"),
    )
    if len(raw_manifest_records) != 12 or len(raw_metric_records) != 12:
        raise StageRPlanError("stage_r_requires_exactly_12_records")
    metric_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_metric_records:
        item = _require_mapping("baseline_metric_record", raw)
        record_id = str(item.get("sample_id", ""))
        if not record_id or record_id in metric_by_id:
            raise StageRPlanError("invalid_or_duplicate_baseline_metric_record")
        metric_by_id[record_id] = item

    records: list[dict[str, Any]] = []
    for raw in raw_manifest_records:
        item = _require_mapping("baseline_manifest_record", raw)
        record_id = str(item.get("sample_id", ""))
        metric_record = metric_by_id.get(record_id)
        if metric_record is None:
            raise StageRPlanError(f"baseline_metric_record_missing:{record_id}")
        if str(item.get("scene", "")) != str(metric_record.get("scene", "")):
            raise StageRPlanError(f"baseline_record_scene_mismatch:{record_id}")
        actual_params = _require_mapping(
            "baseline_actual_params",
            metric_record.get("actual_params"),
        )
        metrics = _require_mapping(
            "baseline_metrics",
            metric_record.get("metrics"),
        )
        required_params = {
            "analysis_scope",
            "smooth_win_len",
            "spec_penalty_width",
            "time_bias",
        }
        if not required_params <= set(actual_params):
            missing = ",".join(sorted(required_params - set(actual_params)))
            raise StageRPlanError(
                f"baseline_actual_params_missing:{record_id}:{missing}"
            )
        if (
            actual_params["analysis_scope"] != "full"
            or actual_params["smooth_win_len"] != RECOVERY_PROFILE_SMOOTH_WIN_LEN
            or float(actual_params["time_bias"]) != RECOVERY_PROFILE_TIME_BIAS_S
        ):
            raise StageRPlanError(
                f"baseline_metric_contract_not_frozen:{record_id}"
            )
        data_sha256 = _require_hash("data_sha256", item.get("data_sha256"))
        reference_sha256 = _require_hash(
            "reference_sha256",
            item.get("reference_sha256"),
        )
        rise_count = int(metrics.get("physiological_rise_episode_count", 0))
        records.append(
            {
                "record_id": record_id,
                "scene": str(item["scene"]),
                "data_path": str(item.get("sensor_path", "")),
                "reference_path": str(item.get("reference_path", "")),
                "data_sha256": data_sha256,
                "reference_sha256": reference_sha256,
                "combined_data_sha256": canonical_sha256(
                    {
                        "data_sha256": data_sha256,
                        "reference_sha256": reference_sha256,
                    }
                ),
                "historical_actual_params": dict(actual_params),
                "independent_metrics": dict(metrics),
                "true_rise_applicable": rise_count > 0,
            }
        )
    if set(metric_by_id) != {record["record_id"] for record in records}:
        raise StageRPlanError("baseline_record_identity_mismatch")
    if dict(sorted(Counter(record["scene"] for record in records).items())) != (
        _EXPECTED_SCENE_COUNTS
    ):
        raise StageRPlanError("stage_r_scene_panel_mismatch")
    if any(
        record["true_rise_applicable"]
        and record["scene"] not in {"run", "kaihe"}
        for record in records
    ):
        raise StageRPlanError("true_rise_applicability_scene_mismatch")
    return sorted(records, key=lambda record: (record["scene"], record["record_id"]))


def _sentinels_from_library(
    profile_library: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_profiles = _require_list(
        "filter_profiles",
        profile_library.get("profiles"),
    )
    profiles: dict[str, Mapping[str, Any]] = {}
    for raw in raw_profiles:
        profile = _require_mapping("filter_profile", raw)
        profile_id = str(profile.get("profile_id", ""))
        if not profile_id or profile_id in profiles:
            raise StageRPlanError("invalid_or_duplicate_filter_profile")
        profiles[profile_id] = profile
    mapping = _require_mapping(
        "recovery_sentinels",
        profile_library.get("recovery_sentinels"),
    )
    if set(mapping) != set(_SENTINEL_ROLES):
        raise StageRPlanError("recovery_sentinel_role_mismatch")
    sentinels: dict[str, dict[str, Any]] = {}
    for role in _SENTINEL_ROLES:
        profile_id = str(mapping[role])
        source = profiles.get(profile_id)
        if source is None:
            raise StageRPlanError(f"recovery_sentinel_missing:{role}")
        if source.get("recovery_sentinel_role") != role:
            raise StageRPlanError(f"recovery_sentinel_profile_role_mismatch:{role}")
        sentinels[role] = {
            "role": role,
            "profile_id": profile_id,
            "profile_sha256": _require_hash(
                "profile_sha256",
                source.get("profile_sha256"),
            ),
            "fs_target": int(source["fs_target"]),
            "physical_memory_ms": int(source["physical_memory_ms"]),
            "actual_taps": int(source["actual_taps"]),
            "nominal_mu": float(source["nominal_mu"]),
        }
    return sentinels


def _candidates_from_registry(
    recovery_registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_candidates = _require_list(
        "recovery_candidates",
        recovery_registry.get("candidates"),
    )
    if len(raw_candidates) != 3 or recovery_registry.get("candidate_count") != 3:
        raise StageRPlanError("stage_r_requires_exactly_3_recovery_candidates")
    candidates: list[dict[str, Any]] = []
    for raw in raw_candidates:
        candidate = _require_mapping("recovery_candidate", raw)
        candidates.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "candidate_sha256": _require_hash(
                    "candidate_sha256",
                    candidate.get("candidate_sha256"),
                ),
                "mechanism_complexity": int(
                    candidate["mechanism_complexity"]
                ),
                "constants": dict(
                    _require_mapping(
                        "recovery_candidate_constants",
                        candidate.get("constants"),
                    )
                ),
            }
        )
    candidate_ids = [item["candidate_id"] for item in candidates]
    if len(set(candidate_ids)) != 3:
        raise StageRPlanError("duplicate_recovery_candidate")
    if recovery_registry.get("control_candidate_id") not in candidate_ids:
        raise StageRPlanError("control_recovery_candidate_missing")
    return sorted(
        candidates,
        key=lambda item: (item["mechanism_complexity"], item["candidate_id"]),
    )


def _base_config(
    *,
    record: Mapping[str, Any],
    profile: Mapping[str, Any],
    control_penalty_id: str,
) -> dict[str, Any]:
    historical = dict(
        _require_mapping(
            "historical_actual_params",
            record["historical_actual_params"],
        )
    )
    historical.update(
        {
            "analysis_scope": "full",
            "adaptive_filter": "lms",
            "algorithm_preset": "lite",
            "reference_groups_order": ["HF"],
            "fs_target": int(profile["fs_target"]),
            "lms_mu_base": float(profile["nominal_mu"]),
            "lms_mu_min": 1e-6,
            "max_order": int(profile["actual_taps"]),
            "smooth_win_len": RECOVERY_PROFILE_SMOOTH_WIN_LEN,
            "time_bias": RECOVERY_PROFILE_TIME_BIAS_S,
            "penalty_candidate_id": control_penalty_id,
        }
    )
    return {
        "data_path": record["data_path"],
        "reference_path": record["reference_path"],
        "parameters": _json_ready(historical),
    }


def _identity_item(
    *,
    parent_experiment_id: str,
    stage: str,
    attempt_kind: str,
    solver_hash: str,
    metric_contract_hash: str,
    evaluation_hash: str,
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    coordinate: Mapping[str, Any],
) -> dict[str, Any]:
    config_hash = canonical_sha256(_json_ready(config))
    identity = AttemptIdentity(
        solver_hash=solver_hash,
        config_hash=config_hash,
        metric_contract_hash=metric_contract_hash,
        evaluation_hash=evaluation_hash,
        data_sha256=str(record["combined_data_sha256"]),
        record_id=str(record["record_id"]),
        stage=stage,
        attempt_kind=attempt_kind,  # type: ignore[arg-type]
        parent_experiment_id=parent_experiment_id,
    )
    return {
        **identity.to_dict(),
        "scene": record["scene"],
        "data_path": record["data_path"],
        "reference_path": record["reference_path"],
        "raw_data_sha256": record["data_sha256"],
        "reference_sha256": record["reference_sha256"],
        "true_rise_applicable": record["true_rise_applicable"],
        "config": _json_ready(config),
        **_json_ready(coordinate),
    }


def build_stage_r_proposal(
    *,
    baseline_manifest_path: Path,
    baseline_metrics_path: Path,
    profile_library_path: Path,
    recovery_registry_path: Path,
    recovery_selection_path: Path,
    penalty_registry_path: Path,
    budget_contract_path: Path,
    parent_experiment_id: str,
    solver_hash: str,
    metric_contract_hash: str,
    spectral_gate_contract_hash: str,
    evaluation_hash: str,
    threshold_anchor_role: str = "conservative",
) -> dict[str, Any]:
    """Build the exact proposal without registering or solving any identity."""

    if not parent_experiment_id:
        raise StageRPlanError("parent_experiment_id_must_not_be_empty")
    for name, value in (
        ("solver_hash", solver_hash),
        ("metric_contract_hash", metric_contract_hash),
        ("spectral_gate_contract_hash", spectral_gate_contract_hash),
        ("evaluation_hash", evaluation_hash),
    ):
        _require_hash(name, value)
    if threshold_anchor_role not in _SENTINEL_ROLES:
        raise StageRPlanError("unknown_threshold_anchor_role")

    source_paths = {
        "baseline_manifest": Path(baseline_manifest_path).resolve(),
        "baseline_metrics": Path(baseline_metrics_path).resolve(),
        "profile_library": Path(profile_library_path).resolve(),
        "recovery_registry": Path(recovery_registry_path).resolve(),
        "recovery_selection": Path(recovery_selection_path).resolve(),
        "penalty_registry": Path(penalty_registry_path).resolve(),
        "budget_contract": Path(budget_contract_path).resolve(),
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise StageRPlanError(f"stage_r_source_missing:{name}:{path}")
    baseline_manifest = read_json(source_paths["baseline_manifest"])
    baseline_metrics = read_json(source_paths["baseline_metrics"])
    profile_library = read_json(source_paths["profile_library"])
    recovery_registry = read_json(source_paths["recovery_registry"])
    recovery_selection = read_json(source_paths["recovery_selection"])
    penalty_registry = read_json(source_paths["penalty_registry"])
    budget_contract = read_json(source_paths["budget_contract"])
    _verify_filter_profile_library(profile_library)
    _verify_registry(
        recovery_registry,
        candidates_field="candidates",
        candidate_name="recovery_candidate",
        candidate_id_field="candidate_id",
        candidate_hash_field="candidate_sha256",
        artifact_name="recovery_candidate_registry",
    )
    _verify_embedded_hash(
        recovery_selection,
        hash_field="contract_sha256",
        artifact_name="recovery_selection_contract",
    )
    _verify_registry(
        penalty_registry,
        candidates_field="candidates",
        candidate_name="penalty_candidate",
        candidate_id_field="penalty_id",
        candidate_hash_field="candidate_sha256",
        artifact_name="penalty_candidate_registry",
    )
    records = _records_from_sources(baseline_manifest, baseline_metrics)
    sentinels = _sentinels_from_library(profile_library)
    candidates = _candidates_from_registry(recovery_registry)
    anchor = sentinels[threshold_anchor_role]
    control_penalty_id = str(
        penalty_registry.get("control_penalty_id", "")
    )
    if not control_penalty_id:
        raise StageRPlanError("control_penalty_id_missing")
    stage_limits = _require_mapping(
        "stage_unique_limits",
        budget_contract.get("stage_unique_limits"),
    )
    if (
        stage_limits.get(_DIAGNOSTIC_STAGE) != 60
        or stage_limits.get(_FORMAL_STAGE) != 108
    ):
        raise StageRPlanError("stage_r_budget_contract_mismatch")

    identities: list[dict[str, Any]] = []
    for floor_bpm in _FIXED_FLOOR_BPM:
        for record in records:
            config = _base_config(
                record=record,
                profile=anchor,
                control_penalty_id=control_penalty_id,
            )
            config["parameters"].update(
                {
                    "recovery_candidate_id": None,
                    "high_lock_escape_candidate_min_bpm": floor_bpm,
                }
            )
            identities.append(
                _identity_item(
                    parent_experiment_id=parent_experiment_id,
                    stage=_DIAGNOSTIC_STAGE,
                    attempt_kind="diagnostic",
                    solver_hash=solver_hash,
                    metric_contract_hash=metric_contract_hash,
                    evaluation_hash=evaluation_hash,
                    record=record,
                    config=config,
                    coordinate={
                        "threshold_anchor_role": threshold_anchor_role,
                        "filter_profile_id": anchor["profile_id"],
                        "filter_profile_sha256": anchor["profile_sha256"],
                        "candidate_min_bpm": floor_bpm,
                        "recovery_candidate_id": (
                            "fixed_floor_diagnostic_control"
                        ),
                        "penalty_candidate_id": control_penalty_id,
                    },
                )
            )
    for candidate in candidates:
        for role in _SENTINEL_ROLES:
            profile = sentinels[role]
            for record in records:
                config = _base_config(
                    record=record,
                    profile=profile,
                    control_penalty_id=control_penalty_id,
                )
                config["parameters"]["recovery_candidate_id"] = candidate[
                    "candidate_id"
                ]
                identities.append(
                    _identity_item(
                        parent_experiment_id=parent_experiment_id,
                        stage=_FORMAL_STAGE,
                        attempt_kind="formal",
                        solver_hash=solver_hash,
                        metric_contract_hash=metric_contract_hash,
                        evaluation_hash=evaluation_hash,
                        record=record,
                        config=config,
                        coordinate={
                            "sentinel_role": role,
                            "filter_profile_id": profile["profile_id"],
                            "filter_profile_sha256": profile["profile_sha256"],
                            "candidate_min_bpm": candidate["constants"].get(
                                "candidate_min_bpm"
                            ),
                            "recovery_candidate_id": candidate["candidate_id"],
                            "recovery_candidate_sha256": candidate[
                                "candidate_sha256"
                            ],
                            "mechanism_complexity": candidate[
                                "mechanism_complexity"
                            ],
                            "penalty_candidate_id": control_penalty_id,
                        },
                    )
                )
    if len(identities) != 168:
        raise StageRPlanError("stage_r_identity_count_mismatch")
    identity_hashes = [item["identity_sha256"] for item in identities]
    if len(set(identity_hashes)) != len(identity_hashes):
        raise StageRPlanError("duplicate_stage_r_identity")

    frozen_contracts = {
        "metric_contract_hash": metric_contract_hash,
        "spectral_gate_contract_hash": spectral_gate_contract_hash,
        "recovery_candidate_registry_hash": _require_hash(
            "recovery_candidate_registry_hash",
            recovery_registry.get("registry_sha256"),
        ),
        "recovery_selection_contract_hash": _require_hash(
            "recovery_selection_contract_hash",
            recovery_selection.get("contract_sha256"),
        ),
        "penalty_registry_hash": _require_hash(
            "penalty_registry_hash",
            penalty_registry.get("registry_sha256"),
        ),
        "filter_profile_design_rule_hash": _require_hash(
            "filter_profile_design_rule_hash",
            profile_library.get("design_rule_sha256"),
        ),
        "budget_contract_hash": canonical_sha256(budget_contract),
    }
    proposal = {
        "proposal_version": "lyx_stage_r_execution_proposal_v1",
        "status": "awaiting_human_execution_authorization",
        "authorization_state": _AUTHORIZATION_STATE,
        "parent_experiment_id": parent_experiment_id,
        "threshold_scan_values_bpm": list(_FIXED_FLOOR_BPM),
        "threshold_anchor_role": threshold_anchor_role,
        "threshold_anchor_profile_id": anchor["profile_id"],
        "threshold_anchor_profile_sha256": anchor["profile_sha256"],
        "diagnostic_unique_budget": 60,
        "formal_unique_budget": 108,
        "unique_budget": 168,
        "retry_limit": 1,
        "worst_case_attempt_budget": 336,
        "independent_bo_authorized": False,
        "algorithm_level_holdout": False,
        "evidence_class": "development_reuse_pilot",
        "diagnostic_result_may_modify_formal_candidates": False,
        "fifty_bpm_endpoint_nominatable": False,
        "frozen_contracts": frozen_contracts,
        "source_artifacts": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in source_paths.items()
        },
        "record_panel": [
            {
                key: record[key]
                for key in (
                    "record_id",
                    "scene",
                    "data_sha256",
                    "reference_sha256",
                    "combined_data_sha256",
                    "true_rise_applicable",
                )
            }
            for record in records
        ],
        "sentinels": sentinels,
        "recovery_candidates": candidates,
        "identities": identities,
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def validate_stage_r_execution_authorization(
    proposal: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed unless the user approved this exact 168-identity proposal."""

    declared_proposal_hash = _require_hash(
        "proposal_sha256",
        proposal.get("proposal_sha256"),
    )
    canonical_proposal = dict(proposal)
    canonical_proposal.pop("proposal_sha256", None)
    if canonical_sha256(canonical_proposal) != declared_proposal_hash:
        raise StageRAuthorizationError("stage_r_proposal_hash_mismatch")
    if receipt is None or receipt.get("approved") is not True:
        raise StageRAuthorizationError(
            "stage_r_execution_authorization_required"
        )
    required = {
        "approved",
        "decision_state",
        "proposal_sha256",
        "diagnostic_unique_budget",
        "formal_unique_budget",
        "unique_budget",
        "threshold_anchor_profile_id",
        "independent_bo_authorized",
        "approved_at",
        "approved_by",
    }
    missing = sorted(required - set(receipt))
    if missing:
        raise StageRAuthorizationError(
            "stage_r_authorization_missing_fields:" + ",".join(missing)
        )
    expected = {
        "decision_state": _AUTHORIZATION_STATE,
        "proposal_sha256": proposal.get("proposal_sha256"),
        "diagnostic_unique_budget": proposal.get("diagnostic_unique_budget"),
        "formal_unique_budget": proposal.get("formal_unique_budget"),
        "unique_budget": proposal.get("unique_budget"),
        "threshold_anchor_profile_id": proposal.get(
            "threshold_anchor_profile_id"
        ),
    }
    mismatched = sorted(
        name
        for name, expected_value in expected.items()
        if receipt.get(name) != expected_value
    )
    if mismatched:
        raise StageRAuthorizationError(
            "stage_r_authorization_identity_mismatch:" + ",".join(mismatched)
        )
    if receipt.get("independent_bo_authorized") is not False:
        raise StageRAuthorizationError(
            "stage_r_independent_bo_must_remain_unauthorized"
        )
    if not isinstance(receipt.get("approved_at"), str) or not receipt.get(
        "approved_at"
    ):
        raise StageRAuthorizationError(
            "stage_r_authorization_approved_at_invalid"
        )
    if not isinstance(receipt.get("approved_by"), str) or not receipt.get(
        "approved_by"
    ):
        raise StageRAuthorizationError(
            "stage_r_authorization_approved_by_invalid"
        )
    return dict(receipt)


def stage_r_metric_contract_v1() -> dict[str, Any]:
    payload = {
        "contract_version": RECOVERY_PROFILE_METRIC_VERSION,
        "smooth_win_len": RECOVERY_PROFILE_SMOOTH_WIN_LEN,
        "time_bias_s": RECOVERY_PROFILE_TIME_BIAS_S,
        "e10_definition": "absolute_error_bpm >= 10",
        "e20_definition": "absolute_error_bpm >= 20",
        "recovery_definition": (
            "first later start of 3 consecutive windows with absolute_error_bpm < 10"
        ),
        "right_censored_recovery_is_failure": True,
        "true_rise_min_windows": 10,
        "true_rise_min_gain_bpm": 15.0,
        "uses_offline_future_dependency": True,
    }
    payload["contract_sha256"] = canonical_sha256(payload)
    return payload


def stage_r_spectral_gate_contract_v1() -> dict[str, Any]:
    stability = StabilityAuditContract.corrected_v2()
    payload = {
        "contract_version": "lyx_stage_r_spectral_gate_v1",
        "filter_stability_contract": stability.to_dict(),
        "filter_stability_contract_sha256": stability.sha256,
        "evaluation_grain": "sentinel_profile_x_record",
        "candidate_invariant": True,
        "reuse_within_same_sentinel_record": True,
        "failure_rule": "any_failed_or_missing_metric_fails_closed",
    }
    payload["contract_sha256"] = canonical_sha256(payload)
    return payload


def _evaluation_source_identity() -> dict[str, Any]:
    source_files = {}
    for path in (
        Path(__file__),
        Path(__file__).with_name("recovery_profile_metrics.py"),
        Path(__file__).with_name("recovery_selection.py"),
        Path(__file__).with_name("recovery_filter_stability.py"),
    ):
        source_files[path.name] = file_sha256(path)
    return {
        "source_files": source_files,
        "evaluation_hash": canonical_sha256(source_files),
    }


def propose_stage_r_execution(
    *,
    baseline_manifest_path: Path,
    baseline_metrics_path: Path,
    profile_library_path: Path,
    recovery_registry_path: Path,
    recovery_selection_path: Path,
    penalty_registry_path: Path,
    budget_contract_path: Path,
    output_dir: Path,
    source_root: Path,
    parent_experiment_id: str,
    threshold_anchor_role: str = "conservative",
) -> dict[str, Any]:
    """Atomically publish contracts and a zero-run Stage R proposal."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise StageRPlanError(f"stage_r_output_already_exists:{destination}")
    metric_contract = stage_r_metric_contract_v1()
    spectral_contract = stage_r_spectral_gate_contract_v1()
    evaluation = _evaluation_source_identity()
    solver = runtime_source_identity(Path(source_root).resolve())
    proposal = build_stage_r_proposal(
        baseline_manifest_path=baseline_manifest_path,
        baseline_metrics_path=baseline_metrics_path,
        profile_library_path=profile_library_path,
        recovery_registry_path=recovery_registry_path,
        recovery_selection_path=recovery_selection_path,
        penalty_registry_path=penalty_registry_path,
        budget_contract_path=budget_contract_path,
        parent_experiment_id=parent_experiment_id,
        solver_hash=solver["source_bundle_sha256"],
        metric_contract_hash=metric_contract["contract_sha256"],
        spectral_gate_contract_hash=spectral_contract["contract_sha256"],
        evaluation_hash=evaluation["evaluation_hash"],
        threshold_anchor_role=threshold_anchor_role,
    )
    staging = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.staging"
    )
    try:
        staging.mkdir(parents=True)
        atomic_write_json(staging / "metric_contract.json", metric_contract)
        atomic_write_json(
            staging / "spectral_gate_contract.json",
            spectral_contract,
        )
        atomic_write_json(staging / "solver_source_identity.json", solver)
        atomic_write_json(staging / "evaluation_source_identity.json", evaluation)
        atomic_write_json(staging / "stage_r_execution_proposal.json", proposal)
        receipt = {
            "receipt_version": "lyx_stage_r_proposal_receipt_v1",
            "status": "awaiting_human_execution_authorization",
            "formal_solver_run_count": 0,
            "diagnostic_solver_run_count": 0,
            "independent_bo_run_count": 0,
            "proposal_sha256": proposal["proposal_sha256"],
            "artifacts": {
                name: file_sha256(staging / name)
                for name in (
                    "metric_contract.json",
                    "spectral_gate_contract.json",
                    "solver_source_identity.json",
                    "evaluation_source_identity.json",
                    "stage_r_execution_proposal.json",
                )
            },
        }
        atomic_write_json(staging / "proposal_receipt.json", receipt)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
        return receipt
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="冻结 LYX Stage R 的 60+108 个求解身份 proposal",
    )
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--baseline-metrics", required=True, type=Path)
    parser.add_argument("--profile-library", required=True, type=Path)
    parser.add_argument("--recovery-registry", required=True, type=Path)
    parser.add_argument("--recovery-selection", required=True, type=Path)
    parser.add_argument("--penalty-registry", required=True, type=Path)
    parser.add_argument("--budget-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--parent-experiment-id", required=True)
    parser.add_argument(
        "--threshold-anchor-role",
        choices=_SENTINEL_ROLES,
        default="conservative",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = propose_stage_r_execution(
        baseline_manifest_path=args.baseline_manifest,
        baseline_metrics_path=args.baseline_metrics,
        profile_library_path=args.profile_library,
        recovery_registry_path=args.recovery_registry,
        recovery_selection_path=args.recovery_selection,
        penalty_registry_path=args.penalty_registry,
        budget_contract_path=args.budget_contract,
        output_dir=args.output_dir,
        source_root=args.source_root,
        parent_experiment_id=args.parent_experiment_id,
        threshold_anchor_role=args.threshold_anchor_role,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
