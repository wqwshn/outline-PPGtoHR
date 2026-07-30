"""Pre-register and execute the bounded LYX Stage R experiment.

The proposal freezes exactly 60 diagnostic and 108 formal identities.  The
executor remains unavailable until an exact human authorization receipt is
validated, then atomically registers the whole matrix before any solve.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from . import recovery_stage_r_cache as stage_r_cache
from . import recovery_stage_r_reporting as stage_r_reporting
from .experiment_freeze_utils import runtime_source_identity
from .phase2_experiment_io import atomic_write_json, file_sha256, read_json
from .preprocess import load_v2_reference
from .recovery_contracts import canonical_sha256, require_sha256
from .recovery_experiment_governance import (
    AttemptIdentity,
    AttemptRegistry,
    BudgetContract,
    ExplorationRegistry,
    FrozenExperimentContractHashes,
    validate_recovery_experiment_preflight,
)
from .recovery_filter_profiles import FilterProfile
from .recovery_filter_stability import (
    FilterAuditRecord,
)
from .recovery_profile_metrics import (
    RECOVERY_PROFILE_METRIC_VERSION,
    RECOVERY_PROFILE_SMOOTH_WIN_LEN,
    RECOVERY_PROFILE_TIME_BIAS_S,
    evaluate_recovery_profile_metrics,
)
from .recovery_selection import (
    RecoveryCandidateEvaluation,
    RecoveryPanelRecord,
    RecoveryRecordEvaluation,
    select_recovery_candidate_evaluations,
)
from .recovery_spectral_gate import (
    StageRSpectralGateContract,
    audit_stage_r_profile_record,
)
from .recovery_stage_r_common import (
    StageRAuthorizationError,
    StageRNumericalResult,
    StageRNumericalRunner,
    StageRPlanError,
    StageRProgressCallback,
)
from .solver import V2SolverResult, solve_v2
from .types import V2RunConfig

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


_independent_bo_review_package = stage_r_reporting.build_independent_bo_review_package
_validate_completed_stage_r_execution = stage_r_reporting.validate_completed_stage_r_execution


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
            "recovery_sentinel_role": profile.get("recovery_sentinel_role"),
            "actual_taps": profile.get("actual_taps"),
        }
        if canonical_sha256(profile_identity) != declared:
            raise StageRPlanError(f"filter_profile_hash_mismatch:{profile.get('profile_id', '')}")
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
            artifact_name=(f"{candidate_name}:{candidate.get(candidate_id_field, '')}"),
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
            raise StageRPlanError(f"baseline_actual_params_missing:{record_id}:{missing}")
        if (
            actual_params["analysis_scope"] != "full"
            or actual_params["smooth_win_len"] != RECOVERY_PROFILE_SMOOTH_WIN_LEN
            or float(actual_params["time_bias"]) != RECOVERY_PROFILE_TIME_BIAS_S
        ):
            raise StageRPlanError(f"baseline_metric_contract_not_frozen:{record_id}")
        data_sha256 = _require_hash("data_sha256", item.get("data_sha256"))
        reference_sha256 = _require_hash(
            "reference_sha256",
            item.get("reference_sha256"),
        )
        raw_method_names = item.get("method_names")
        if (
            not isinstance(raw_method_names, list)
            or not raw_method_names
            or not all(isinstance(method, str) and method for method in raw_method_names)
        ):
            raise StageRPlanError(f"baseline_method_names_invalid:{record_id}")
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
                "method_names": list(raw_method_names),
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
        record["true_rise_applicable"] and record["scene"] not in {"run", "kaihe"}
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
                "mechanism_complexity": int(candidate["mechanism_complexity"]),
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
        "method_names": list(record["method_names"]),
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
        "method_names": list(record["method_names"]),
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
    control_penalty_id = str(penalty_registry.get("control_penalty_id", ""))
    if not control_penalty_id:
        raise StageRPlanError("control_penalty_id_missing")
    stage_limits = _require_mapping(
        "stage_unique_limits",
        budget_contract.get("stage_unique_limits"),
    )
    if stage_limits.get(_DIAGNOSTIC_STAGE) != 60 or stage_limits.get(_FORMAL_STAGE) != 108:
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
                        "physical_memory_ms": anchor["physical_memory_ms"],
                        "actual_taps": anchor["actual_taps"],
                        "nominal_mu": anchor["nominal_mu"],
                        "candidate_min_bpm": floor_bpm,
                        "recovery_candidate_id": ("fixed_floor_diagnostic_control"),
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
                config["parameters"]["recovery_candidate_id"] = candidate["candidate_id"]
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
                            "physical_memory_ms": profile["physical_memory_ms"],
                            "actual_taps": profile["actual_taps"],
                            "nominal_mu": profile["nominal_mu"],
                            "candidate_min_bpm": candidate["constants"].get("candidate_min_bpm"),
                            "recovery_candidate_id": candidate["candidate_id"],
                            "recovery_candidate_sha256": candidate["candidate_sha256"],
                            "mechanism_complexity": candidate["mechanism_complexity"],
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
                    "method_names",
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
        raise StageRAuthorizationError("stage_r_execution_authorization_required")
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
        raise StageRAuthorizationError("stage_r_authorization_missing_fields:" + ",".join(missing))
    expected = {
        "decision_state": _AUTHORIZATION_STATE,
        "proposal_sha256": proposal.get("proposal_sha256"),
        "diagnostic_unique_budget": proposal.get("diagnostic_unique_budget"),
        "formal_unique_budget": proposal.get("formal_unique_budget"),
        "unique_budget": proposal.get("unique_budget"),
        "threshold_anchor_profile_id": proposal.get("threshold_anchor_profile_id"),
    }
    mismatched = sorted(
        name for name, expected_value in expected.items() if receipt.get(name) != expected_value
    )
    if mismatched:
        raise StageRAuthorizationError(
            "stage_r_authorization_identity_mismatch:" + ",".join(mismatched)
        )
    if receipt.get("independent_bo_authorized") is not False:
        raise StageRAuthorizationError("stage_r_independent_bo_must_remain_unauthorized")
    if not isinstance(receipt.get("approved_at"), str) or not receipt.get("approved_at"):
        raise StageRAuthorizationError("stage_r_authorization_approved_at_invalid")
    if not isinstance(receipt.get("approved_by"), str) or not receipt.get("approved_by"):
        raise StageRAuthorizationError("stage_r_authorization_approved_by_invalid")
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
        "selection_recovery_delay_s": (
            "max_recovered_delay_s when present; 0 when no recovery episode; "
            "total_window_count * 1 s when every observed episode is right-censored"
        ),
        "true_rise_min_windows": 10,
        "true_rise_min_gain_bpm": 15.0,
        "uses_offline_future_dependency": True,
    }
    payload["contract_sha256"] = canonical_sha256(payload)
    return payload


def stage_r_spectral_gate_contract_v2() -> dict[str, Any]:
    contract = StageRSpectralGateContract()
    payload = contract.to_dict()
    payload["contract_sha256"] = canonical_sha256(payload)
    return payload


def _evaluation_source_identity(source_root: Path) -> dict[str, Any]:
    root_modules = (
        "ppg_hr.v2.recovery_stage_r_experiment",
        "ppg_hr.v2.recovery_stage_r_runner",
    )
    identity = runtime_source_identity(
        Path(source_root).resolve(),
        root_modules=root_modules,
    )
    return {
        "root_modules": list(root_modules),
        **identity,
        "evaluation_hash": identity["source_bundle_sha256"],
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
    spectral_contract = stage_r_spectral_gate_contract_v2()
    evaluation = _evaluation_source_identity(source_root)
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
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.staging")
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


def _budget_contract_from_payload(
    payload: Mapping[str, Any],
) -> BudgetContract:
    return BudgetContract(
        contract_version=str(payload["contract_version"]),
        stage_unique_limits=dict(
            _require_mapping(
                "stage_unique_limits",
                payload.get("stage_unique_limits"),
            )
        ),
        normal_unique_identity_limit=payload.get("normal_unique_identity_limit"),
        supplemental_stage=payload.get("supplemental_stage"),
        stage_attempt_kinds=dict(
            _require_mapping(
                "stage_attempt_kinds",
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
    raw_allowlist = _require_list(
        "allowed_identity_sha256",
        payload.get("allowed_identity_sha256"),
    )
    return ExplorationRegistry(
        registry_version=str(payload["registry_version"]),
        unique_budget=int(payload["unique_budget"]),
        allowed_identity_sha256=tuple(str(item) for item in raw_allowlist),
    )


def _attempt_identity_from_item(
    item: Mapping[str, Any],
) -> AttemptIdentity:
    names = {field.name for field in fields(AttemptIdentity)}
    return AttemptIdentity(**{name: item[name] for name in names})


def _verify_proposal_source_artifacts(
    proposal: Mapping[str, Any],
) -> dict[str, Path]:
    raw_sources = _require_mapping(
        "source_artifacts",
        proposal.get("source_artifacts"),
    )
    paths: dict[str, Path] = {}
    for name, raw in raw_sources.items():
        source = _require_mapping(f"source_artifact:{name}", raw)
        path = Path(str(source.get("path", ""))).resolve()
        if not path.is_file():
            raise StageRPlanError(f"stage_r_execution_source_missing:{name}:{path}")
        expected = _require_hash(
            f"source_artifact_sha256:{name}",
            source.get("sha256"),
        )
        if file_sha256(path) != expected:
            raise StageRPlanError(f"stage_r_execution_source_hash_mismatch:{name}")
        paths[str(name)] = path
    expected_names = {
        "baseline_manifest",
        "baseline_metrics",
        "profile_library",
        "recovery_registry",
        "recovery_selection",
        "penalty_registry",
        "budget_contract",
    }
    if set(paths) != expected_names:
        raise StageRPlanError("stage_r_execution_source_set_mismatch")
    return paths


def _validate_stage_r_execution_preflight(
    *,
    proposal_dir: Path,
    proposal: Mapping[str, Any],
    source_root: Path,
) -> tuple[dict[str, Path], BudgetContract]:
    proposal_root = Path(proposal_dir).resolve()
    metric_contract = read_json(proposal_root / "metric_contract.json")
    spectral_contract = read_json(proposal_root / "spectral_gate_contract.json")
    proposal_receipt = read_json(proposal_root / "proposal_receipt.json")
    if proposal_receipt.get(
        "status"
    ) != "awaiting_human_execution_authorization" or proposal_receipt.get(
        "proposal_sha256"
    ) != proposal.get("proposal_sha256"):
        raise StageRPlanError("stage_r_proposal_receipt_mismatch")
    receipt_artifacts = _require_mapping(
        "proposal_receipt_artifacts",
        proposal_receipt.get("artifacts"),
    )
    for name, expected_hash in receipt_artifacts.items():
        artifact_path = proposal_root / str(name)
        if not artifact_path.is_file() or file_sha256(artifact_path) != expected_hash:
            raise StageRPlanError(f"stage_r_proposal_artifact_hash_mismatch:{name}")
    _verify_embedded_hash(
        metric_contract,
        hash_field="contract_sha256",
        artifact_name="stage_r_metric_contract",
    )
    _verify_embedded_hash(
        spectral_contract,
        hash_field="contract_sha256",
        artifact_name="stage_r_spectral_contract",
    )
    frozen_solver = read_json(proposal_root / "solver_source_identity.json")
    current_solver = runtime_source_identity(Path(source_root).resolve())
    if frozen_solver != current_solver:
        raise StageRPlanError("stage_r_solver_source_changed_after_proposal")
    frozen_evaluation = read_json(proposal_root / "evaluation_source_identity.json")
    if frozen_evaluation != _evaluation_source_identity(source_root):
        raise StageRPlanError("stage_r_evaluation_source_changed_after_proposal")

    source_paths = _verify_proposal_source_artifacts(proposal)
    profile_library = read_json(source_paths["profile_library"])
    recovery_registry = read_json(source_paths["recovery_registry"])
    recovery_selection = read_json(source_paths["recovery_selection"])
    penalty_registry = read_json(source_paths["penalty_registry"])
    budget_payload = read_json(source_paths["budget_contract"])
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
    actual_contracts = {
        "metric_contract_hash": metric_contract["contract_sha256"],
        "spectral_gate_contract_hash": spectral_contract["contract_sha256"],
        "recovery_candidate_registry_hash": recovery_registry["registry_sha256"],
        "recovery_selection_contract_hash": recovery_selection["contract_sha256"],
        "penalty_registry_hash": penalty_registry["registry_sha256"],
        "filter_profile_design_rule_hash": profile_library["design_rule_sha256"],
        "budget_contract_hash": canonical_sha256(budget_payload),
    }
    frozen_contracts = dict(
        _require_mapping(
            "frozen_contracts",
            proposal.get("frozen_contracts"),
        )
    )
    validate_recovery_experiment_preflight(
        expected=FrozenExperimentContractHashes(**frozen_contracts),
        actual=actual_contracts,
    )
    budget = _budget_contract_from_payload(budget_payload)
    if budget.sha256 != actual_contracts["budget_contract_hash"]:
        raise StageRPlanError("stage_r_budget_contract_payload_mismatch")
    return source_paths, budget


def _stage_r_run_config(
    item: Mapping[str, Any],
) -> V2RunConfig:
    config = _require_mapping("identity_config", item.get("config"))
    parameters = dict(
        _require_mapping(
            "identity_config_parameters",
            config.get("parameters"),
        )
    )
    field_names = {field.name for field in fields(V2RunConfig)}
    values = {name: value for name, value in parameters.items() if name in field_names}
    values["data_path"] = Path(str(config["data_path"])).resolve()
    values["ref_path"] = Path(str(config["reference_path"])).resolve()
    for name in (
        "reference_groups_order",
        "motion_gate_filter_allowlist",
    ):
        if name in values and isinstance(values[name], list):
            values[name] = tuple(values[name])
    return V2RunConfig(**values)


def _load_or_run_spectral_audit(
    item: Mapping[str, Any],
    *,
    spectral_audit_dir: Path,
) -> dict[str, Any]:
    profile_id = str(item["filter_profile_id"])
    record_id = str(item["record_id"])
    parameters = _require_mapping(
        "identity_config_parameters",
        _require_mapping("identity_config", item.get("config")).get(
            "parameters"
        ),
    )
    reference_stage_limit = parameters.get(
        "adaptive_reference_stage_limit"
    )
    audit_path = spectral_audit_dir / profile_id / f"{record_id}.json"
    if audit_path.is_file():
        payload = read_json(audit_path)
        _verify_embedded_hash(
            payload,
            hash_field="audit_sha256",
            artifact_name=f"stage_r_spectral_audit:{profile_id}:{record_id}",
        )
        expected = {
            "profile_id": profile_id,
            "profile_sha256": item["filter_profile_sha256"],
            "record_id": record_id,
            "data_sha256": item["raw_data_sha256"],
            "reference_sha256": item["reference_sha256"],
            "audit_contract_sha256": (StageRSpectralGateContract().sha256),
        }
        if any(payload.get(name) != value for name, value in expected.items()):
            raise StageRPlanError(
                f"stage_r_spectral_audit_identity_mismatch:{profile_id}:{record_id}"
            )
        audit_payload = _require_mapping(
            "stage_r_spectral_audit",
            payload.get("audit"),
        )
        if (
            payload.get("reference_stage_limit")
            != reference_stage_limit
            or audit_payload.get("reference_stage_limit")
            != reference_stage_limit
        ):
            raise StageRPlanError(
                "stage_r_spectral_audit_reference_stage_limit_mismatch:"
                f"{profile_id}:{record_id}"
            )
        return {
            **dict(audit_payload),
            "audit_sha256": payload["audit_sha256"],
        }

    raw_sentinel_role = item.get("sentinel_role")
    profile = FilterProfile(
        profile_id=profile_id,
        design_role="core",
        fs_target=int(item["config"]["parameters"]["fs_target"]),
        memory_ms=int(item["physical_memory_ms"]),
        nominal_mu=float(item["config"]["parameters"]["lms_mu_base"]),
        recovery_sentinel_role=(
            None
            if raw_sentinel_role is None
            else str(raw_sentinel_role)
        ),
    )
    record = FilterAuditRecord(
        record_id=record_id,
        scene=str(item["scene"]),
        data_path=str(item["data_path"]),
        reference_path=str(item["reference_path"]),
        data_sha256=str(item["raw_data_sha256"]),
        reference_sha256=str(item["reference_sha256"]),
    )
    contract = StageRSpectralGateContract()
    audit = audit_stage_r_profile_record(
        profile,
        record,
        contract=contract,
        reference_stage_limit=reference_stage_limit,
    )
    payload = {
        "audit_version": "lyx_stage_r_spectral_record_audit_v1",
        "profile_id": profile_id,
        "profile_sha256": item["filter_profile_sha256"],
        "record_id": record_id,
        "data_sha256": item["raw_data_sha256"],
        "reference_sha256": item["reference_sha256"],
        "audit_contract_sha256": contract.sha256,
        "candidate_invariant": True,
        "audit": audit,
    }
    if reference_stage_limit is not None:
        payload["reference_stage_limit"] = reference_stage_limit
    payload["audit_sha256"] = canonical_sha256(payload)
    atomic_write_json(audit_path, payload)
    return {**audit, "audit_sha256": payload["audit_sha256"]}


def run_stage_r_numerical_identity(
    item: dict[str, Any],
    spectral_audit_dir: Path,
) -> StageRNumericalResult:
    data_path = Path(str(item["data_path"])).resolve()
    reference_path = Path(str(item["reference_path"])).resolve()
    if file_sha256(data_path) != item["raw_data_sha256"]:
        raise StageRPlanError(f"stage_r_data_hash_mismatch:{item['record_id']}")
    if file_sha256(reference_path) != item["reference_sha256"]:
        raise StageRPlanError(f"stage_r_reference_hash_mismatch:{item['record_id']}")
    config = _stage_r_run_config(item)
    result = solve_v2(config)
    metadata = dict(result.metadata)
    metadata["smooth_win_len"] = config.smooth_win_len
    result = V2SolverResult(
        HR=result.HR,
        err_stats=result.err_stats,
        metadata=metadata,
        window_table=result.window_table,
    )
    metrics = evaluate_recovery_profile_metrics(
        result,
        ref_data=load_v2_reference(reference_path),
        method_names=tuple(str(name) for name in item["method_names"]),
    )
    spectral_audit = (
        _load_or_run_spectral_audit(
            item,
            spectral_audit_dir=spectral_audit_dir,
        )
        if (
            item.get("spectral_audit_required") is True
            or item["stage"] == _FORMAL_STAGE
        )
        else None
    )
    return StageRNumericalResult(
        solver_result=result,
        metrics=asdict(metrics),
        spectral_audit=spectral_audit,
    )


_run_stage_r_numerical_identity = run_stage_r_numerical_identity


def _selection_recovery_delay(metrics: Mapping[str, Any]) -> float:
    raw = metrics.get("max_recovered_delay_s")
    if raw is not None:
        value = float(raw)
    elif int(metrics.get("recovery_episode_count", 0)) == 0:
        value = 0.0
    else:
        value = float(metrics["total_window_count"])
    if not math.isfinite(value) or value < 0.0:
        raise StageRPlanError("stage_r_recovery_delay_invalid")
    return value


def _threshold_diagnostic_summary(
    result_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for floor_bpm in sorted(_FIXED_FLOOR_BPM):
        metrics = [
            _require_mapping("diagnostic_metrics", row["metrics"])
            for row in result_rows
            if row["stage"] == _DIAGNOSTIC_STAGE and float(row["candidate_min_bpm"]) == floor_bpm
        ]
        if len(metrics) != 12:
            raise StageRPlanError(f"stage_r_diagnostic_result_count_mismatch:{floor_bpm}")
        rows.append(
            {
                "candidate_min_bpm": floor_bpm,
                "record_count": 12,
                "worst_l10": max(int(item["longest_e10_run_windows"]) for item in metrics),
                "worst_l20": max(int(item["longest_e20_run_windows"]) for item in metrics),
                "worst_mae": max(float(item["final_motion_mae_bpm"]) for item in metrics),
                "mean_mae": sum(float(item["final_motion_mae_bpm"]) for item in metrics) / 12.0,
                "right_censored_recovery_count": sum(
                    int(item["right_censored_recovery_count"]) for item in metrics
                ),
                "nominatable": False,
            }
        )
    payload = {
        "summary_version": "lyx_stage_r_threshold_diagnostic_v1",
        "diagnostic_only": True,
        "may_modify_formal_candidates": False,
        "fifty_bpm_endpoint_nominatable": False,
        "thresholds": rows,
    }
    payload["summary_sha256"] = canonical_sha256(payload)
    return payload


def _build_stage_r_selection(
    *,
    proposal: Mapping[str, Any],
    result_rows: Sequence[Mapping[str, Any]],
    baseline_metrics_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_payload = read_json(baseline_metrics_path)
    baseline_records = _require_list(
        "baseline_metric_records",
        baseline_payload.get("records"),
    )
    independent = {
        str(item["sample_id"]): _require_mapping(
            "independent_metrics",
            _require_mapping("baseline_metric_record", item).get("metrics"),
        )
        for item in baseline_records
    }
    formal = [row for row in result_rows if row["stage"] == _FORMAL_STAGE]
    if len(formal) != 108:
        raise StageRPlanError("stage_r_formal_result_count_mismatch")
    by_coordinate = {
        (
            str(row["recovery_candidate_id"]),
            str(row["filter_profile_id"]),
            str(row["record_id"]),
        ): row
        for row in formal
    }
    if len(by_coordinate) != 108:
        raise StageRPlanError("stage_r_formal_result_coordinate_mismatch")
    spectral_hashes: dict[tuple[str, str], set[str]] = {}
    for row in formal:
        spectral = _require_mapping(
            "formal_spectral_audit",
            row.get("spectral_audit"),
        )
        spectral_hashes.setdefault(
            (
                str(row["filter_profile_id"]),
                str(row["record_id"]),
            ),
            set(),
        ).add(
            _require_hash(
                "spectral_audit_sha256",
                spectral.get("audit_sha256"),
            )
        )
    if len(spectral_hashes) != 36 or any(len(hashes) != 1 for hashes in spectral_hashes.values()):
        raise StageRPlanError("stage_r_spectral_audit_candidate_invariance_mismatch")
    control_id = "current_fixed_floor_control_v1"
    panel = [
        RecoveryPanelRecord(
            record_id=str(item["record_id"]),
            scene=str(item["scene"]),
            true_rise_applicable=bool(item["true_rise_applicable"]),
        )
        for item in _require_list(
            "record_panel",
            proposal.get("record_panel"),
        )
    ]
    sentinel_ids = [str(proposal["sentinels"][role]["profile_id"]) for role in _SENTINEL_ROLES]
    evaluations: list[RecoveryCandidateEvaluation] = []
    serialized: list[dict[str, Any]] = []
    for candidate in _require_list(
        "recovery_candidates",
        proposal.get("recovery_candidates"),
    ):
        candidate_id = str(candidate["candidate_id"])
        records: list[RecoveryRecordEvaluation] = []
        for panel_record in panel:
            for sentinel_id in sentinel_ids:
                row = by_coordinate[(candidate_id, sentinel_id, panel_record.record_id)]
                current = by_coordinate[(control_id, sentinel_id, panel_record.record_id)]
                metrics = _require_mapping(
                    "formal_metrics",
                    row["metrics"],
                )
                current_metrics = _require_mapping(
                    "current_formal_metrics",
                    current["metrics"],
                )
                independent_metrics = independent[panel_record.record_id]
                spectral = _require_mapping(
                    "formal_spectral_audit",
                    row["spectral_audit"],
                )
                records.append(
                    RecoveryRecordEvaluation(
                        record_id=panel_record.record_id,
                        sentinel_id=sentinel_id,
                        scene=panel_record.scene,
                        spectral_gate_passed=bool(
                            spectral.get("stability_pass") and spectral.get("spectral_gate_pass")
                        ),
                        l10=float(metrics["longest_e10_run_windows"]),
                        l20=float(metrics["longest_e20_run_windows"]),
                        mae=float(metrics["final_motion_mae_bpm"]),
                        independent_l10=float(independent_metrics["longest_e10_run_windows"]),
                        independent_l20=float(independent_metrics["longest_e20_run_windows"]),
                        independent_mae=float(independent_metrics["final_motion_mae_bpm"]),
                        current_l10=float(current_metrics["longest_e10_run_windows"]),
                        current_mae=float(current_metrics["final_motion_mae_bpm"]),
                        recovery_delay=_selection_recovery_delay(metrics),
                        right_censored_recovery_count=int(metrics["right_censored_recovery_count"]),
                        current_right_censored_recovery_count=int(
                            current_metrics["right_censored_recovery_count"]
                        ),
                        true_rise_underestimate=(
                            float(metrics["max_rise_underestimate_bpm"])
                            if panel_record.true_rise_applicable
                            and metrics.get("max_rise_underestimate_bpm") is not None
                            else None
                        ),
                        current_true_rise_underestimate=(
                            float(current_metrics["max_rise_underestimate_bpm"])
                            if panel_record.true_rise_applicable
                            and current_metrics.get("max_rise_underestimate_bpm") is not None
                            else None
                        ),
                    )
                )
        evaluation = RecoveryCandidateEvaluation(
            candidate_id=candidate_id,
            mechanism_complexity=int(candidate["mechanism_complexity"]),
            records=tuple(records),
        )
        evaluations.append(evaluation)
        serialized.append(_json_ready(asdict(evaluation)))
    selection = select_recovery_candidate_evaluations(
        evaluations,
        expected_records=panel,
        expected_sentinel_ids=sentinel_ids,
    )
    return selection, serialized


def execute_stage_r_proposal(
    *,
    proposal_dir: Path,
    authorization_receipt_path: Path | None,
    governance_dir: Path,
    output_dir: Path,
    source_root: Path,
    _numerical_runner: StageRNumericalRunner | None = None,
    progress_callback: StageRProgressCallback | None = None,
) -> dict[str, Any]:
    """Execute only the exact authorized Stage R proposal, resumably."""

    proposal_root = Path(proposal_dir).resolve()
    proposal = read_json(proposal_root / "stage_r_execution_proposal.json")
    receipt = (
        None
        if authorization_receipt_path is None
        else read_json(Path(authorization_receipt_path).resolve())
    )
    authorization = validate_stage_r_execution_authorization(
        proposal,
        receipt=receipt,
    )
    source_paths, source_budget = _validate_stage_r_execution_preflight(
        proposal_dir=proposal_root,
        proposal=proposal,
        source_root=source_root,
    )
    governance_root = Path(governance_dir).resolve()
    governance_budget_payload = read_json(governance_root / "budget_contract.json")
    governance_budget = _budget_contract_from_payload(governance_budget_payload)
    if (
        governance_budget.sha256 != source_budget.sha256
        or governance_budget.to_dict() != source_budget.to_dict()
    ):
        raise StageRPlanError("stage_r_governance_budget_mismatch")
    exploration = _exploration_registry_from_payload(
        read_json(governance_root / "exploration_registry.json")
    )
    registry = AttemptRegistry.open(
        governance_root / "attempt_registry.json",
        budget_contract=governance_budget,
        exploration_registry=exploration,
    )
    raw_identities = _require_list(
        "stage_r_identities",
        proposal.get("identities"),
    )
    identities = tuple(
        _attempt_identity_from_item(_require_mapping("stage_r_identity", item))
        for item in raw_identities
    )
    if len(identities) != 168 or [identity.sha256 for identity in identities] != [
        str(item["identity_sha256"]) for item in raw_identities
    ]:
        raise StageRPlanError("stage_r_execution_identity_matrix_mismatch")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    binding_path = destination / "execution_binding.json"
    authorization_sha256 = canonical_sha256(authorization)
    binding = {
        "binding_version": "lyx_stage_r_execution_binding_v1",
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": authorization_sha256,
        "solver_source_bundle_sha256": identities[0].solver_hash,
        "evaluation_hash": identities[0].evaluation_hash,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    if binding_path.is_file():
        if read_json(binding_path) != binding:
            raise StageRPlanError("stage_r_execution_binding_mismatch")
    else:
        atomic_write_json(binding_path, binding)

    completion_path = destination / "stage_r_completion.json"
    if completion_path.is_file():
        return _validate_completed_stage_r_execution(
            completion_path=completion_path,
            proposal=proposal,
            authorization_sha256=authorization_sha256,
            governance_root=governance_root,
            destination=destination,
            registry=registry,
            identities=identities,
        )

    registered = registry.register_identities(identities)
    if registered != tuple(identity.sha256 for identity in identities):
        raise StageRPlanError("stage_r_bulk_registration_mismatch")

    runner = _numerical_runner or run_stage_r_numerical_identity
    spectral_audit_dir = destination / "spectral_audits"
    results: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_identities, start=1):
        result = stage_r_cache.execute_stage_r_identity(
            registry=registry,
            item=dict(_require_mapping("stage_r_identity", raw_item)),
            numerical_runner=runner,
            spectral_audit_dir=spectral_audit_dir,
        )
        results.append(result)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "stage_r_identity_complete",
                    "completed": index,
                    "total": len(raw_identities),
                    "identity_sha256": result["identity_sha256"],
                    "stage": result["stage"],
                    "record_id": result["record_id"],
                    "cache_hit": result["cache_hit"],
                }
            )
    registry.assert_complete_matrix(identities)
    result_index = {
        "index_version": "lyx_stage_r_result_index_v1",
        "proposal_sha256": proposal["proposal_sha256"],
        "result_count": len(results),
        "results": results,
    }
    result_index["index_sha256"] = canonical_sha256(result_index)
    atomic_write_json(destination / "identity_result_index.json", result_index)

    diagnostic = _threshold_diagnostic_summary(results)
    atomic_write_json(
        destination / "threshold_diagnostic_summary.json",
        diagnostic,
    )
    selection, evaluations = _build_stage_r_selection(
        proposal=proposal,
        result_rows=results,
        baseline_metrics_path=source_paths["baseline_metrics"],
    )
    evaluation_payload = {
        "evaluation_version": "lyx_stage_r_formal_evaluations_v1",
        "proposal_sha256": proposal["proposal_sha256"],
        "candidate_evaluations": evaluations,
    }
    evaluation_payload["evaluation_sha256"] = canonical_sha256(evaluation_payload)
    atomic_write_json(
        destination / "formal_candidate_evaluations.json",
        evaluation_payload,
    )
    atomic_write_json(
        destination / "recovery_selection.json",
        selection,
    )
    if selection["status"] == "no_safe_recovery_candidate":
        atomic_write_json(
            destination / "independent_bo_review_package.json",
            _independent_bo_review_package(
                proposal_sha256=proposal["proposal_sha256"],
                authorization_sha256=authorization_sha256,
                selection=selection,
                candidate_evaluations=evaluations,
            ),
        )
    registry_summary = registry.summary()
    diagnostic_identities = tuple(
        identity for identity in identities if identity.stage == _DIAGNOSTIC_STAGE
    )
    formal_identities = tuple(
        identity for identity in identities if identity.stage == _FORMAL_STAGE
    )
    diagnostic_matrix = registry.matrix_execution_summary(diagnostic_identities)
    formal_matrix = registry.matrix_execution_summary(formal_identities)
    matrix_summary = registry.matrix_execution_summary(identities)
    matrix_snapshot = registry.matrix_snapshot(identities)
    atomic_write_json(
        destination / "attempt_registry_stage_r_snapshot.json",
        matrix_snapshot,
    )
    artifact_names = [
        "execution_binding.json",
        "identity_result_index.json",
        "threshold_diagnostic_summary.json",
        "formal_candidate_evaluations.json",
        "recovery_selection.json",
        "attempt_registry_stage_r_snapshot.json",
    ]
    if selection["status"] == "no_safe_recovery_candidate":
        artifact_names.append("independent_bo_review_package.json")
    artifacts = {name: file_sha256(destination / name) for name in artifact_names}
    governance_receipt = {
        "receipt_version": "lyx_stage_r_governance_receipt_v2",
        "status": selection["status"],
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": authorization_sha256,
        "identity_matrix_sha256": canonical_sha256([identity.sha256 for identity in identities]),
        "attempt_registry_file_sha256_at_completion": file_sha256(
            governance_root / "attempt_registry.json"
        ),
        "attempt_registry_summary_at_completion": registry_summary,
        "attempt_registry_matrix_snapshot_sha256": matrix_snapshot["snapshot_sha256"],
        "matrix_execution_summary": matrix_summary,
        "diagnostic_unique_identities": 60,
        "formal_unique_identities": 108,
        "independent_bo_run_count": 0,
        "artifacts": artifacts,
    }
    governance_receipt["receipt_sha256"] = canonical_sha256(governance_receipt)
    governance_receipt_path = governance_root / "stage_r_governance_receipt.json"
    atomic_write_json(governance_receipt_path, governance_receipt)
    completion = {
        "completion_version": "lyx_stage_r_completion_v2",
        "status": selection["status"],
        "evidence_class": "development_reuse_pilot",
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": authorization_sha256,
        "diagnostic_result_count": 60,
        "formal_result_count": 108,
        "diagnostic_solver_run_count": diagnostic_matrix["identity_with_solver_attempt_count"],
        "formal_solver_run_count": formal_matrix["identity_with_solver_attempt_count"],
        "independent_bo_run_count": 0,
        "provisional_recovery_id": selection["provisional_recovery_id"],
        "rollback_backup_id": selection["rollback_backup_id"],
        "next_state": (
            "awaiting_human_independent_bo_decision"
            if selection["status"] == "no_safe_recovery_candidate"
            else "ready_for_stage_f_filter_matrix"
        ),
        "attempt_registry_summary_at_completion": registry_summary,
        "matrix_execution_summary": matrix_summary,
        "artifacts": artifacts,
        "governance_receipt_sha256": governance_receipt["receipt_sha256"],
        "governance_receipt_file_sha256": file_sha256(governance_receipt_path),
    }
    completion["completion_sha256"] = canonical_sha256(completion)
    # This is the transaction commit marker and must be written last.
    atomic_write_json(completion_path, completion)
    return _validate_completed_stage_r_execution(
        completion_path=completion_path,
        proposal=proposal,
        authorization_sha256=authorization_sha256,
        governance_root=governance_root,
        destination=destination,
        registry=registry,
        identities=identities,
    )


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
