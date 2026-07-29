"""LYX Stage F filter-profile enumeration and fair-control experiment.

Stage R v3 freezes its evaluation source closure before this module is
introduced.  Stage F therefore reuses the frozen Stage R cache/config
interfaces without refactoring their implementation; shared extraction must
wait until the authorized Stage R matrix has executed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from .experiment_freeze_utils import runtime_source_identity
from .phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    read_json,
)
from .preprocess import load_v2_reference
from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import (
    AttemptIdentity,
    AttemptRegistry,
    BudgetContract,
    ExplorationRegistry,
)
from .recovery_filter_profiles import FilterProfile
from .recovery_filter_stability import (
    FilterAuditRecord,
)
from .recovery_profile_metrics import (
    evaluate_recovery_profile_metrics,
)
from .recovery_spectral_gate import (
    StageRSpectralGateContract,
    audit_stage_r_profile_record,
)
from .recovery_stage_r_cache import execute_stage_r_identity
from .recovery_stage_r_common import (
    StageRNumericalResult,
    StageRNumericalRunner,
)
from .recovery_stage_r_experiment import (
    _stage_r_run_config,
    stage_r_metric_contract_v1,
    stage_r_spectral_gate_contract_v1,
)
from .solver import V2SolverResult, solve_v2

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


def _candidate_registry(
    registry: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    registry_hash = _verify_embedded_hash(
        registry,
        hash_field="registry_sha256",
        artifact_name="stage_f_recovery_registry",
    )
    candidates: dict[str, dict[str, Any]] = {}
    for raw in _require_list(
        "stage_f_recovery_candidates",
        registry.get("candidates"),
    ):
        candidate = dict(_require_mapping("stage_f_recovery_candidate", raw))
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id or candidate_id in candidates:
            raise StageFPlanError(
                "invalid_or_duplicate_stage_f_recovery_candidate"
            )
        _verify_embedded_hash(
            candidate,
            hash_field="candidate_sha256",
            artifact_name=f"stage_f_recovery_candidate:{candidate_id}",
        )
        candidates[candidate_id] = candidate
    if (
        len(candidates) != 3
        or registry.get("candidate_count") != 3
        or registry.get("control_candidate_id") not in candidates
    ):
        raise StageFPlanError("stage_f_recovery_registry_mismatch")
    _require_hash("recovery_registry_sha256", registry_hash)
    return candidates


def _control_penalty(
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    _verify_embedded_hash(
        registry,
        hash_field="registry_sha256",
        artifact_name="stage_f_penalty_registry",
    )
    candidates: dict[str, dict[str, Any]] = {}
    for raw in _require_list(
        "stage_f_penalty_candidates",
        registry.get("candidates"),
    ):
        candidate = dict(_require_mapping("stage_f_penalty_candidate", raw))
        penalty_id = str(candidate.get("penalty_id", ""))
        if not penalty_id or penalty_id in candidates:
            raise StageFPlanError(
                "invalid_or_duplicate_stage_f_penalty_candidate"
            )
        _verify_embedded_hash(
            candidate,
            hash_field="candidate_sha256",
            artifact_name=f"stage_f_penalty_candidate:{penalty_id}",
        )
        candidates[penalty_id] = candidate
    control_id = str(registry.get("control_penalty_id", ""))
    if (
        registry.get("candidate_count") != 3
        or len(candidates) != 3
        or control_id not in candidates
    ):
        raise StageFPlanError("stage_f_penalty_registry_mismatch")
    return candidates[control_id]


def _profiles(
    library: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _verify_embedded_hash(
        library,
        hash_field="library_sha256",
        artifact_name="stage_f_filter_profile_library",
    )
    raw_profiles = _require_list(
        "stage_f_filter_profiles",
        library.get("profiles"),
    )
    profiles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_profiles:
        profile = dict(_require_mapping("stage_f_filter_profile", raw))
        profile_id = str(profile.get("profile_id", ""))
        if not profile_id or profile_id in seen:
            raise StageFPlanError(
                "invalid_or_duplicate_stage_f_filter_profile"
            )
        seen.add(profile_id)
        identity = {
            "profile_id": profile_id,
            "design_role": profile.get("design_role"),
            "fs_target": profile.get("fs_target"),
            "memory_ms": profile.get("physical_memory_ms"),
            "nominal_mu": profile.get("nominal_mu"),
            "recovery_sentinel_role": profile.get(
                "recovery_sentinel_role"
            ),
            "actual_taps": profile.get("actual_taps"),
        }
        declared = _require_hash(
            "profile_sha256",
            profile.get("profile_sha256"),
        )
        if canonical_sha256(identity) != declared:
            raise StageFPlanError(
                f"stage_f_filter_profile_hash_mismatch:{profile_id}"
            )
        profiles.append(profile)
    if library.get("profile_count") != 8 or len(profiles) != 8:
        raise StageFPlanError("stage_f_requires_exactly_eight_profiles")
    if dict(
        sorted(Counter(int(item["fs_target"]) for item in profiles).items())
    ) != _EXPECTED_FS_QUOTA:
        raise StageFPlanError("stage_f_profile_fs_quota_mismatch")
    if dict(
        sorted(Counter(str(item["design_role"]) for item in profiles).items())
    ) != _EXPECTED_ROLE_COUNTS:
        raise StageFPlanError("stage_f_profile_role_count_mismatch")
    return profiles


def _validate_profile_library_completion(
    completion: Mapping[str, Any],
    *,
    library_sha256: str,
    profile_ids: set[str],
) -> None:
    _verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="stage_f_profile_library_completion",
    )
    selection = _require_mapping(
        "stage_f_profile_library_selection",
        completion.get("selection"),
    )
    candidate_receipt_hashes = _require_mapping(
        "stage_f_candidate_profile_receipt_sha256",
        completion.get("candidate_profile_receipt_sha256"),
    )
    final_receipt_hashes = _require_mapping(
        "stage_f_final_profile_receipt_sha256",
        completion.get("final_profile_receipt_sha256"),
    )
    selected_p100_ids = {
        str(value)
        for value in _require_list(
            "stage_f_selected_p100_profile_ids",
            selection.get("selected_p100_profile_ids"),
        )
    }
    selected_p50_ids = {
        str(value)
        for value in _require_list(
            "stage_f_selected_p50_profile_ids",
            selection.get("selected_p50_profile_ids"),
        )
    }
    selected_profile_ids = {
        str(value)
        for value in _require_list(
            "stage_f_selected_profile_ids",
            selection.get("selected_profile_ids"),
        )
    }
    if (
        completion.get("receipt_version")
        != "lyx_filter_rate_normalized_supplement_completion_v2"
        or completion.get("status") != "complete"
        or completion.get("evidence_class")
        != "development_reuse_pilot"
        or completion.get("algorithm_level_holdout") is not False
        or completion.get("final_profile_count") != 8
        or completion.get("new_rate_normalized_run_count") != 8
        or completion.get("exploration_run_count") != 8
        or completion.get("reused_p50_numeric_result_count") != 8
        or completion.get("candidate_profile_count") != 2
        or {
            str(value)
            for value in _require_list(
                "stage_f_candidate_eligible_profile_ids",
                completion.get("candidate_eligible_profile_ids"),
            )
        }
        != _RATE_NORMALIZED_PROFILE_IDS
        or set(candidate_receipt_hashes)
        != _RATE_NORMALIZED_PROFILE_IDS
        or any(
            _require_hash(
                f"stage_f_candidate_profile_receipt_sha256:{profile_id}",
                receipt_hash,
            )
            != receipt_hash
            for profile_id, receipt_hash in candidate_receipt_hashes.items()
        )
        or set(final_receipt_hashes) != profile_ids
        or any(
            _require_hash(
                f"stage_f_final_profile_receipt_sha256:{profile_id}",
                receipt_hash,
            )
            != receipt_hash
            for profile_id, receipt_hash in final_receipt_hashes.items()
        )
        or set(
            str(value)
            for value in _require_list(
                "stage_f_completed_profile_ids",
                completion.get("final_profile_ids"),
            )
        )
        != profile_ids
        or completion.get("final_library_sha256") != library_sha256
        or completion.get("actual_hr_tracking_trajectory_count") != 0
        or completion.get("independent_bo_run_count") != 0
        or selection.get("status") != "complete"
        or selected_p100_ids != _RATE_NORMALIZED_PROFILE_IDS
        or selected_p50_ids
        != _REUSED_RATE_NORMALIZED_P50_PROFILE_IDS
        or selected_profile_ids
        != (
            _RATE_NORMALIZED_PROFILE_IDS
            | _REUSED_RATE_NORMALIZED_P50_PROFILE_IDS
        )
    ):
        raise StageFPlanError(
            "stage_f_profile_library_completion_mismatch"
        )


def _validate_baseline_contract_receipt(
    receipt: Mapping[str, Any],
) -> None:
    if (
        receipt.get("receipt_version")
        != "lyx_recovery_profile_baseline_receipt_v1"
        or receipt.get("status") != "complete"
        or receipt.get("metric_contract_version")
        != "lyx_recovery_profile_metric_v1"
        or receipt.get("record_count") != 12
        or dict(
            _require_mapping(
                "stage_f_baseline_scene_counts",
                receipt.get("scene_counts"),
            )
        )
        != _EXPECTED_SCENE_COUNTS
        or "record_metrics.json"
        not in _require_mapping(
            "stage_f_baseline_artifact_sha256",
            receipt.get("artifact_sha256"),
        )
    ):
        raise StageFPlanError("stage_f_baseline_receipt_mismatch")


def _stage_r_templates(
    proposal: Mapping[str, Any],
    *,
    control_recovery_id: str,
    parent_experiment_id: str,
    solver_hash: str,
) -> list[dict[str, Any]]:
    proposal_hash = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="stage_r_proposal",
    )
    if (
        proposal.get("parent_experiment_id") != parent_experiment_id
        or proposal.get("independent_bo_authorized") is not False
        or proposal.get("diagnostic_unique_budget") != 60
        or proposal.get("formal_unique_budget") != 108
        or proposal.get("unique_budget") != 168
    ):
        raise StageFPlanError("stage_r_proposal_contract_mismatch")
    _require_hash("stage_r_proposal_sha256", proposal_hash)
    panel = _require_list("stage_r_record_panel", proposal.get("record_panel"))
    if (
        len(panel) != 12
        or dict(
            sorted(
                Counter(
                    str(
                        _require_mapping(
                            "stage_r_record",
                            record,
                        )["scene"]
                    )
                    for record in panel
                ).items()
            )
        )
        != _EXPECTED_SCENE_COUNTS
    ):
        raise StageFPlanError("stage_f_record_panel_mismatch")
    templates: dict[str, dict[str, Any]] = {}
    for raw in _require_list(
        "stage_r_identities",
        proposal.get("identities"),
    ):
        item = dict(_require_mapping("stage_r_identity", raw))
        if (
            item.get("stage") != "recovery_sentinel"
            or item.get("recovery_candidate_id")
            != control_recovery_id
        ):
            continue
        record_id = str(item.get("record_id", ""))
        if not record_id or record_id in templates:
            continue
        if (
            item.get("solver_hash") != solver_hash
            or item.get("parent_experiment_id") != parent_experiment_id
        ):
            raise StageFPlanError(
                f"stage_f_template_identity_mismatch:{record_id}"
            )
        config = _require_mapping(
            "stage_f_template_config",
            item.get("config"),
        )
        if canonical_sha256(config) != item.get("config_hash"):
            raise StageFPlanError(
                f"stage_f_template_config_hash_mismatch:{record_id}"
            )
        templates[record_id] = item
    panel_ids = {
        str(_require_mapping("stage_r_record", record)["record_id"])
        for record in panel
    }
    if set(templates) != panel_ids:
        raise StageFPlanError("stage_f_template_record_set_mismatch")
    return sorted(
        templates.values(),
        key=lambda item: (str(item["scene"]), str(item["record_id"])),
    )


def _baseline_metrics_by_record(
    payload: Mapping[str, Any],
    *,
    record_panel_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    required_metrics = {
        "longest_e10_run_windows",
        "longest_e20_run_windows",
        "final_motion_mae_bpm",
        "right_censored_recovery_count",
        "max_recovered_delay_s",
        "max_rise_underestimate_bpm",
    }
    metrics_by_record: dict[str, dict[str, Any]] = {}
    for raw in _require_list(
        "stage_f_baseline_metric_records",
        payload.get("records"),
    ):
        item = _require_mapping("stage_f_baseline_metric_record", raw)
        record_id = str(item.get("sample_id", ""))
        metrics = dict(
            _require_mapping(
                "stage_f_independent_metrics",
                item.get("metrics"),
            )
        )
        if (
            not record_id
            or record_id in metrics_by_record
            or not required_metrics <= set(metrics)
            or metrics.get("metric_contract_version")
            != "lyx_recovery_profile_metric_v1"
            or record_id not in record_panel_by_id
            or item.get("scene")
            != record_panel_by_id[record_id].get("scene")
            or item.get("data_sha256")
            != record_panel_by_id[record_id].get("data_sha256")
            or item.get("reference_sha256")
            != record_panel_by_id[record_id].get(
                "reference_sha256"
            )
        ):
            raise StageFPlanError(
                "invalid_or_incomplete_stage_f_baseline_metrics"
            )
        metrics_by_record[record_id] = metrics
    if set(metrics_by_record) != set(record_panel_by_id):
        raise StageFPlanError("stage_f_baseline_record_set_mismatch")
    return metrics_by_record


def _validate_stage_r_completion(
    completion: Mapping[str, Any],
    *,
    stage_r_proposal_sha256: str,
    candidate_ids: set[str],
) -> tuple[str, str | None]:
    _verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="stage_r_completion",
    )
    provisional = str(completion.get("provisional_recovery_id", ""))
    backup_raw = completion.get("rollback_backup_id")
    backup = None if backup_raw is None else str(backup_raw)
    if (
        completion.get("status") != "selected"
        or completion.get("proposal_sha256")
        != stage_r_proposal_sha256
        or completion.get("diagnostic_result_count") != 60
        or completion.get("formal_result_count") != 108
        or completion.get("independent_bo_run_count") != 0
        or completion.get("next_state")
        != "ready_for_stage_f_filter_matrix"
        or provisional not in candidate_ids
        or (backup is not None and backup not in candidate_ids)
    ):
        raise StageFPlanError("stage_r_completion_not_ready_for_stage_f")
    return provisional, backup


def _identity_item(
    *,
    template: Mapping[str, Any],
    profile: Mapping[str, Any],
    candidate: Mapping[str, Any],
    penalty: Mapping[str, Any],
    matrix_role: str,
    stage: str,
    parent_experiment_id: str,
    solver_hash: str,
    metric_contract_hash: str,
    evaluation_hash: str,
) -> dict[str, Any]:
    config = deepcopy(
        dict(
            _require_mapping(
                "stage_f_template_config",
                template["config"],
            )
        )
    )
    parameters = dict(
        _require_mapping(
            "stage_f_template_parameters",
            config.get("parameters"),
        )
    )
    parameters.update(
        {
            "fs_target": int(profile["fs_target"]),
            "max_order": int(profile["actual_taps"]),
            "lms_mu_base": float(profile["nominal_mu"]),
            "recovery_candidate_id": candidate["candidate_id"],
            "penalty_candidate_id": penalty["penalty_id"],
        }
    )
    config["parameters"] = parameters
    config_hash = canonical_sha256(config)
    attempt = AttemptIdentity(
        solver_hash=solver_hash,
        config_hash=config_hash,
        metric_contract_hash=metric_contract_hash,
        evaluation_hash=evaluation_hash,
        data_sha256=str(template["data_sha256"]),
        record_id=str(template["record_id"]),
        stage=stage,
        attempt_kind="formal",
        parent_experiment_id=parent_experiment_id,
    )
    constants = _require_mapping(
        "stage_f_recovery_constants",
        candidate.get("constants"),
    )
    return {
        **attempt.to_dict(),
        "matrix_role": matrix_role,
        "scene": template["scene"],
        "data_path": template["data_path"],
        "reference_path": template["reference_path"],
        "raw_data_sha256": template["raw_data_sha256"],
        "reference_sha256": template["reference_sha256"],
        "method_names": list(template["method_names"]),
        "true_rise_applicable": template["true_rise_applicable"],
        "config": config,
        "filter_profile_id": profile["profile_id"],
        "filter_profile_sha256": profile["profile_sha256"],
        "filter_profile_design_role": profile["design_role"],
        "physical_memory_ms": profile["physical_memory_ms"],
        "actual_taps": profile["actual_taps"],
        "nominal_mu": profile["nominal_mu"],
        "sentinel_role": profile.get("recovery_sentinel_role"),
        "recovery_candidate_id": candidate["candidate_id"],
        "recovery_candidate_sha256": candidate["candidate_sha256"],
        "candidate_min_bpm": constants.get("candidate_min_bpm"),
        "penalty_candidate_id": penalty["penalty_id"],
        "penalty_candidate_sha256": penalty["candidate_sha256"],
    }


def build_stage_f_proposal(
    *,
    stage_r_proposal: Mapping[str, Any],
    stage_r_completion: Mapping[str, Any],
    profile_library: Mapping[str, Any],
    profile_library_completion: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    baseline_contract_receipt: Mapping[str, Any],
    recovery_registry: Mapping[str, Any],
    penalty_registry: Mapping[str, Any],
    budget_contract: Mapping[str, Any],
    parent_experiment_id: str,
    solver_hash: str,
    metric_contract_hash: str,
    spectral_gate_contract_hash: str,
    evaluation_hash: str,
) -> dict[str, Any]:
    """Freeze the two Stage F 8×12 matrices without running a solver."""

    if not parent_experiment_id:
        raise StageFPlanError("parent_experiment_id_must_not_be_empty")
    for name, value in (
        ("solver_hash", solver_hash),
        ("metric_contract_hash", metric_contract_hash),
        ("spectral_gate_contract_hash", spectral_gate_contract_hash),
        ("evaluation_hash", evaluation_hash),
    ):
        _require_hash(name, value)
    candidates = _candidate_registry(recovery_registry)
    control_recovery_id = str(recovery_registry["control_candidate_id"])
    penalty = _control_penalty(penalty_registry)
    profiles = _profiles(profile_library)
    _validate_profile_library_completion(
        profile_library_completion,
        library_sha256=str(profile_library["library_sha256"]),
        profile_ids={
            str(profile["profile_id"])
            for profile in profiles
        },
    )
    _validate_baseline_contract_receipt(
        baseline_contract_receipt
    )
    stage_r_proposal_sha256 = _require_hash(
        "stage_r_proposal_sha256",
        stage_r_proposal.get("proposal_sha256"),
    )
    templates = _stage_r_templates(
        stage_r_proposal,
        control_recovery_id=control_recovery_id,
        parent_experiment_id=parent_experiment_id,
        solver_hash=solver_hash,
    )
    record_panel_by_id = {
        str(record["record_id"]): record
        for record in (
            _require_mapping("stage_r_record", raw)
            for raw in _require_list(
                "stage_r_record_panel",
                stage_r_proposal.get("record_panel"),
            )
        )
    }
    baseline_by_record = _baseline_metrics_by_record(
        baseline_metrics,
        record_panel_by_id=record_panel_by_id,
    )
    provisional_id, backup_id = _validate_stage_r_completion(
        stage_r_completion,
        stage_r_proposal_sha256=stage_r_proposal_sha256,
        candidate_ids=set(candidates),
    )
    frozen = _require_mapping(
        "stage_r_frozen_contracts",
        stage_r_proposal.get("frozen_contracts"),
    )
    expected_contracts = {
        "metric_contract_hash": metric_contract_hash,
        "spectral_gate_contract_hash": spectral_gate_contract_hash,
        "recovery_candidate_registry_hash": recovery_registry[
            "registry_sha256"
        ],
        "penalty_registry_hash": penalty_registry["registry_sha256"],
        "filter_profile_design_rule_hash": profile_library[
            "design_rule_sha256"
        ],
        "budget_contract_hash": canonical_sha256(budget_contract),
    }
    if any(frozen.get(name) != value for name, value in expected_contracts.items()):
        raise StageFPlanError("stage_f_frozen_contract_mismatch")
    stage_limits = _require_mapping(
        "stage_f_stage_unique_limits",
        budget_contract.get("stage_unique_limits"),
    )
    if (
        stage_limits.get(_PROVISIONAL_STAGE) != 288
        or stage_limits.get(_CURRENT_ROLE_STAGE) != 96
    ):
        raise StageFPlanError("stage_f_budget_contract_mismatch")

    identities: list[dict[str, Any]] = []
    matrix_lanes = [
        (
            "provisional_recovery",
            _PROVISIONAL_STAGE,
            provisional_id,
        )
    ]
    control_reuses_provisional = provisional_id == control_recovery_id
    if not control_reuses_provisional:
        matrix_lanes.append(
            (
                "same_role_current_control",
                _CURRENT_ROLE_STAGE,
                control_recovery_id,
            )
        )
    for matrix_role, stage, candidate_id in matrix_lanes:
        candidate = candidates[candidate_id]
        for profile in profiles:
            for template in templates:
                identities.append(
                    _identity_item(
                        template=template,
                        profile=profile,
                        candidate=candidate,
                        penalty=penalty,
                        matrix_role=matrix_role,
                        stage=stage,
                        parent_experiment_id=parent_experiment_id,
                        solver_hash=solver_hash,
                        metric_contract_hash=metric_contract_hash,
                        evaluation_hash=evaluation_hash,
                    )
                )
    identity_hashes = [
        str(identity["identity_sha256"])
        for identity in identities
    ]
    expected_unique = 96 if control_reuses_provisional else 192
    if (
        len(identities) != expected_unique
        or len(set(identity_hashes)) != expected_unique
    ):
        raise StageFPlanError("stage_f_identity_matrix_mismatch")
    logical_tasks = [
        {
            "matrix_role": identity["matrix_role"],
            "logical_stage": identity["stage"],
            "numerical_identity_stage": identity["stage"],
            "record_id": identity["record_id"],
            "scene": identity["scene"],
            "filter_profile_id": identity["filter_profile_id"],
            "recovery_candidate_id": identity[
                "recovery_candidate_id"
            ],
            "penalty_candidate_id": identity["penalty_candidate_id"],
            "identity_sha256": identity["identity_sha256"],
            "numeric_source_role": identity["matrix_role"],
        }
        for identity in identities
    ]
    if control_reuses_provisional:
        logical_tasks.extend(
            {
                "matrix_role": "same_role_current_control",
                "logical_stage": _CURRENT_ROLE_STAGE,
                "numerical_identity_stage": identity["stage"],
                "record_id": identity["record_id"],
                "scene": identity["scene"],
                "filter_profile_id": identity["filter_profile_id"],
                "recovery_candidate_id": identity[
                    "recovery_candidate_id"
                ],
                "penalty_candidate_id": identity[
                    "penalty_candidate_id"
                ],
                "identity_sha256": identity["identity_sha256"],
                "numeric_source_role": "provisional_recovery",
            }
            for identity in identities
        )
    if len(logical_tasks) != 192:
        raise StageFPlanError("stage_f_logical_matrix_mismatch")
    proposal = {
        "proposal_version": "lyx_stage_f_execution_proposal_v1",
        "status": "ready_for_execution",
        "parent_experiment_id": parent_experiment_id,
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "independent_bo_authorized": False,
        "upstream_completion_bindings": {
            "profile_library_completion_sha256": (
                profile_library_completion["completion_sha256"]
            ),
        },
        "stage_r_proposal_sha256": stage_r_proposal_sha256,
        "stage_r_completion_sha256": stage_r_completion[
            "completion_sha256"
        ],
        "provisional_recovery_id": provisional_id,
        "rollback_backup_id": backup_id,
        "current_control_recovery_id": control_recovery_id,
        "control_penalty_id": penalty["penalty_id"],
        "profile_count": 8,
        "record_count": 12,
        "logical_task_count": 192,
        "planned_unique_identity_count": expected_unique,
        "reused_logical_task_count": (
            96 if control_reuses_provisional else 0
        ),
        "provisional_matrix_unique_budget": 96,
        "current_role_matrix_unique_budget": (
            0 if control_reuses_provisional else 96
        ),
        "frozen_contracts": {
            **dict(frozen),
            "stage_f_evaluation_hash": evaluation_hash,
        },
        "profiles": profiles,
        "record_panel": [
            {
                **dict(_require_mapping("stage_r_record", record)),
                "independent_metrics": baseline_by_record[
                    str(
                        _require_mapping(
                            "stage_r_record",
                            record,
                        )["record_id"]
                    )
                ],
            }
            for record in stage_r_proposal["record_panel"]
        ],
        "identities": identities,
        "logical_tasks": logical_tasks,
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def _validate_stage_r_source_bindings(
    stage_r_proposal: Mapping[str, Any],
    *,
    source_paths: Mapping[str, Path],
    baseline_contract_receipt: Mapping[str, Any],
) -> None:
    _verify_embedded_hash(
        stage_r_proposal,
        hash_field="proposal_sha256",
        artifact_name="stage_r_proposal",
    )
    stage_r_sources = _require_mapping(
        "stage_r_source_artifacts",
        stage_r_proposal.get("source_artifacts"),
    )
    for name in (
        "baseline_metrics",
        "profile_library",
        "recovery_registry",
        "penalty_registry",
        "budget_contract",
    ):
        source = _require_mapping(
            f"stage_r_source_artifact:{name}",
            stage_r_sources.get(name),
        )
        if source.get("sha256") != file_sha256(source_paths[name]):
            raise StageFPlanError(
                f"stage_f_stage_r_source_mismatch:{name}"
            )
    baseline_artifacts = _require_mapping(
        "stage_f_baseline_artifact_sha256",
        baseline_contract_receipt.get("artifact_sha256"),
    )
    if (
        baseline_artifacts.get("record_metrics.json")
        != file_sha256(source_paths["baseline_metrics"])
    ):
        raise StageFPlanError(
            "stage_f_baseline_metric_artifact_mismatch"
        )


def propose_stage_f_execution(
    *,
    stage_r_proposal_path: Path,
    stage_r_completion_path: Path,
    profile_library_path: Path,
    profile_library_completion_path: Path,
    baseline_metrics_path: Path,
    baseline_contract_receipt_path: Path,
    recovery_registry_path: Path,
    penalty_registry_path: Path,
    budget_contract_path: Path,
    output_dir: Path,
    source_root: Path,
    parent_experiment_id: str,
) -> dict[str, Any]:
    """Atomically publish a zero-run Stage F execution proposal."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise StageFPlanError(
            f"stage_f_output_already_exists:{destination}"
        )
    source_paths = {
        "stage_r_proposal": Path(stage_r_proposal_path).resolve(),
        "stage_r_completion": Path(stage_r_completion_path).resolve(),
        "profile_library": Path(profile_library_path).resolve(),
        "profile_library_completion": Path(
            profile_library_completion_path
        ).resolve(),
        "baseline_metrics": Path(baseline_metrics_path).resolve(),
        "baseline_contract_receipt": Path(
            baseline_contract_receipt_path
        ).resolve(),
        "recovery_registry": Path(recovery_registry_path).resolve(),
        "penalty_registry": Path(penalty_registry_path).resolve(),
        "budget_contract": Path(budget_contract_path).resolve(),
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise StageFPlanError(f"stage_f_source_missing:{name}:{path}")
    stage_r_proposal = read_json(source_paths["stage_r_proposal"])
    stage_r_completion = read_json(source_paths["stage_r_completion"])
    profile_library = read_json(source_paths["profile_library"])
    profile_library_completion = read_json(
        source_paths["profile_library_completion"]
    )
    baseline_metrics = read_json(source_paths["baseline_metrics"])
    baseline_contract_receipt = read_json(
        source_paths["baseline_contract_receipt"]
    )
    recovery_registry = read_json(source_paths["recovery_registry"])
    penalty_registry = read_json(source_paths["penalty_registry"])
    budget_contract = read_json(source_paths["budget_contract"])
    _validate_baseline_contract_receipt(
        baseline_contract_receipt
    )
    _validate_stage_r_source_bindings(
        stage_r_proposal,
        source_paths=source_paths,
        baseline_contract_receipt=baseline_contract_receipt,
    )
    metric_contract = stage_r_metric_contract_v1()
    spectral_contract = stage_r_spectral_gate_contract_v1()
    solver = runtime_source_identity(Path(source_root).resolve())
    evaluation_roots = (
        "ppg_hr.v2.recovery_stage_f_experiment",
        "ppg_hr.v2.recovery_stage_f_runner",
    )
    evaluation = runtime_source_identity(
        Path(source_root).resolve(),
        root_modules=evaluation_roots,
    )
    evaluation = {
        "root_modules": list(evaluation_roots),
        **evaluation,
        "evaluation_hash": evaluation["source_bundle_sha256"],
    }
    proposal = build_stage_f_proposal(
        stage_r_proposal=stage_r_proposal,
        stage_r_completion=stage_r_completion,
        profile_library=profile_library,
        profile_library_completion=profile_library_completion,
        baseline_metrics=baseline_metrics,
        baseline_contract_receipt=baseline_contract_receipt,
        recovery_registry=recovery_registry,
        penalty_registry=penalty_registry,
        budget_contract=budget_contract,
        parent_experiment_id=parent_experiment_id,
        solver_hash=solver["source_bundle_sha256"],
        metric_contract_hash=metric_contract["contract_sha256"],
        spectral_gate_contract_hash=spectral_contract[
            "contract_sha256"
        ],
        evaluation_hash=evaluation["evaluation_hash"],
    )
    proposal.pop("proposal_sha256")
    proposal["source_artifacts"] = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
        }
        for name, path in source_paths.items()
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)

    staging = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.staging"
    )
    if staging.parent != destination.parent:
        raise StageFPlanError("stage_f_staging_parent_mismatch")
    try:
        staging.mkdir(parents=True)
        atomic_write_json(staging / "metric_contract.json", metric_contract)
        atomic_write_json(
            staging / "spectral_gate_contract.json",
            spectral_contract,
        )
        atomic_write_json(staging / "solver_source_identity.json", solver)
        atomic_write_json(
            staging / "evaluation_source_identity.json",
            evaluation,
        )
        atomic_write_json(
            staging / "stage_f_execution_proposal.json",
            proposal,
        )
        artifact_names = (
            "metric_contract.json",
            "spectral_gate_contract.json",
            "solver_source_identity.json",
            "evaluation_source_identity.json",
            "stage_f_execution_proposal.json",
        )
        receipt = {
            "receipt_version": "lyx_stage_f_proposal_receipt_v1",
            "status": "ready_for_execution",
            "formal_solver_run_count": 0,
            "diagnostic_solver_run_count": 0,
            "independent_bo_run_count": 0,
            "logical_task_count": proposal["logical_task_count"],
            "planned_unique_identity_count": proposal[
                "planned_unique_identity_count"
            ],
            "reused_logical_task_count": proposal[
                "reused_logical_task_count"
            ],
            "proposal_sha256": proposal["proposal_sha256"],
            "artifacts": {
                name: file_sha256(staging / name)
                for name in artifact_names
            },
        }
        atomic_write_json(staging / "proposal_receipt.json", receipt)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
        return receipt
    except Exception:
        if staging.exists() and staging.parent == destination.parent:
            shutil.rmtree(staging)
        raise


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


def _verify_stage_f_preflight(
    *,
    proposal_root: Path,
    source_root: Path,
) -> tuple[dict[str, Any], BudgetContract]:
    proposal = read_json(
        proposal_root / "stage_f_execution_proposal.json"
    )
    _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="stage_f_proposal",
    )
    if (
        proposal.get("status") != "ready_for_execution"
        or proposal.get("independent_bo_authorized") is not False
        or proposal.get("algorithm_level_holdout") is not False
        or proposal.get("logical_task_count") != 192
        or proposal.get("planned_unique_identity_count")
        not in {96, 192}
    ):
        raise StageFPlanError("stage_f_proposal_contract_mismatch")
    receipt_path = proposal_root / "proposal_receipt.json"
    if not receipt_path.is_file():
        raise StageFPlanError("stage_f_proposal_receipt_missing")
    receipt = read_json(receipt_path)
    receipt_artifacts = _require_mapping(
        "stage_f_proposal_receipt_artifacts",
        receipt.get("artifacts"),
    )
    expected_artifact_names = {
        "metric_contract.json",
        "spectral_gate_contract.json",
        "solver_source_identity.json",
        "evaluation_source_identity.json",
        "stage_f_execution_proposal.json",
    }
    if (
        receipt.get("receipt_version")
        != "lyx_stage_f_proposal_receipt_v1"
        or receipt.get("status") != "ready_for_execution"
        or receipt.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or receipt.get("formal_solver_run_count") != 0
        or receipt.get("diagnostic_solver_run_count") != 0
        or receipt.get("independent_bo_run_count") != 0
        or receipt.get("logical_task_count")
        != proposal.get("logical_task_count")
        or receipt.get("planned_unique_identity_count")
        != proposal.get("planned_unique_identity_count")
        or receipt.get("reused_logical_task_count")
        != proposal.get("reused_logical_task_count")
        or set(receipt_artifacts) != expected_artifact_names
        or any(
            not (proposal_root / name).is_file()
            or file_sha256(proposal_root / name) != expected_hash
            for name, expected_hash in receipt_artifacts.items()
        )
    ):
        raise StageFPlanError("stage_f_proposal_receipt_mismatch")
    source_artifacts = _require_mapping(
        "stage_f_source_artifacts",
        proposal.get("source_artifacts"),
    )
    for name, raw in source_artifacts.items():
        artifact = _require_mapping(
            f"stage_f_source_artifact:{name}",
            raw,
        )
        path = Path(str(artifact.get("path", ""))).resolve()
        if (
            not path.is_file()
            or file_sha256(path) != artifact.get("sha256")
        ):
            raise StageFPlanError(
                f"stage_f_source_artifact_mismatch:{name}"
            )
    if "profile_library_completion" not in source_artifacts:
        raise StageFPlanError(
            "stage_f_profile_library_completion_source_missing"
        )
    completion_source = _require_mapping(
        "stage_f_profile_library_completion_source",
        source_artifacts["profile_library_completion"],
    )
    completion_payload = read_json(
        Path(str(completion_source["path"])).resolve()
    )
    completion_content_hash = _verify_embedded_hash(
        completion_payload,
        hash_field="completion_sha256",
        artifact_name="stage_f_profile_library_completion",
    )
    completion_bindings = _require_mapping(
        "stage_f_upstream_completion_bindings",
        proposal.get("upstream_completion_bindings"),
    )
    if (
        set(completion_bindings)
        != {"profile_library_completion_sha256"}
        or completion_bindings.get(
            "profile_library_completion_sha256"
        )
        != completion_content_hash
    ):
        raise StageFPlanError(
            "stage_f_profile_library_completion_binding_mismatch"
        )
    metric_contract = read_json(proposal_root / "metric_contract.json")
    spectral_contract = read_json(
        proposal_root / "spectral_gate_contract.json"
    )
    _verify_embedded_hash(
        metric_contract,
        hash_field="contract_sha256",
        artifact_name="stage_f_metric_contract",
    )
    _verify_embedded_hash(
        spectral_contract,
        hash_field="contract_sha256",
        artifact_name="stage_f_spectral_contract",
    )
    frozen = _require_mapping(
        "stage_f_frozen_contracts",
        proposal.get("frozen_contracts"),
    )
    if (
        metric_contract["contract_sha256"]
        != frozen.get("metric_contract_hash")
        or spectral_contract["contract_sha256"]
        != frozen.get("spectral_gate_contract_hash")
    ):
        raise StageFPlanError("stage_f_runtime_contract_mismatch")
    solver_identity = read_json(
        proposal_root / "solver_source_identity.json"
    )
    current_solver = runtime_source_identity(Path(source_root).resolve())
    if solver_identity != current_solver:
        raise StageFPlanError("stage_f_solver_source_changed")
    evaluation_identity = read_json(
        proposal_root / "evaluation_source_identity.json"
    )
    roots = tuple(
        str(value)
        for value in _require_list(
            "stage_f_evaluation_roots",
            evaluation_identity.get("root_modules"),
        )
    )
    current_evaluation = runtime_source_identity(
        Path(source_root).resolve(),
        root_modules=roots,
    )
    if (
        evaluation_identity.get("source_files")
        != current_evaluation.get("source_files")
        or evaluation_identity.get("source_bundle_sha256")
        != current_evaluation.get("source_bundle_sha256")
        or evaluation_identity.get("evaluation_hash")
        != current_evaluation.get("source_bundle_sha256")
        or evaluation_identity.get("evaluation_hash")
        != frozen.get("stage_f_evaluation_hash")
    ):
        raise StageFPlanError("stage_f_evaluation_source_changed")
    identities = _require_list(
        "stage_f_identities",
        proposal.get("identities"),
    )
    expected_unique = int(proposal["planned_unique_identity_count"])
    if len(identities) != expected_unique:
        raise StageFPlanError("stage_f_identity_count_mismatch")
    parsed = [_attempt_identity_from_item(item) for item in identities]
    if (
        len({identity.sha256 for identity in parsed}) != expected_unique
        or [identity.sha256 for identity in parsed]
        != [str(item["identity_sha256"]) for item in identities]
        or any(
            identity.solver_hash
            != solver_identity["source_bundle_sha256"]
            for identity in parsed
        )
    ):
        raise StageFPlanError("stage_f_identity_matrix_mismatch")
    budget_payload = read_json(
        Path(
            str(
                _require_mapping(
                    "stage_f_budget_source",
                    source_artifacts["budget_contract"],
                )["path"]
            )
        )
    )
    budget = _budget_contract_from_payload(budget_payload)
    if budget.sha256 != frozen.get("budget_contract_hash"):
        raise StageFPlanError("stage_f_budget_hash_mismatch")
    return proposal, budget


def _metric_float(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if value is None:
        return float("inf")
    number = float(value)
    if not number == number:
        return float("inf")
    return number


_SPECTRAL_AGGREGATE_FIELDS = (
    "prominence_db_delta_median",
    "visible_top3_rate_delta",
    "hr_band_share_delta_median",
    "pulse_power_retention_median",
    "residual_artifact_corr_delta_median",
)


def _normalize_stage_f_spectral_evidence(
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(audit)
    summary = dict(
        _require_mapping(
            "stage_f_spectral_gate_summary",
            audit.get("stage_r_spectral_gate"),
        )
    )
    for name in _SPECTRAL_AGGREGATE_FIELDS:
        summary.setdefault(name, None)
    normalized["stage_r_spectral_gate"] = summary
    return normalized


def _validate_spectral_evidence(
    spectral: Mapping[str, Any],
) -> None:
    summary = _require_mapping(
        "stage_f_spectral_gate_summary",
        spectral.get("stage_r_spectral_gate"),
    )
    if not set(_SPECTRAL_AGGREGATE_FIELDS) <= set(summary):
        raise StageFPlanError(
            "stage_f_spectral_evidence_incomplete"
        )
    nested_passed = summary.get("spectral_gate_pass") is True
    try:
        aggregates_finite = all(
            math.isfinite(float(summary[name]))
            for name in _SPECTRAL_AGGREGATE_FIELDS
        )
    except (KeyError, TypeError, ValueError) as error:
        if nested_passed:
            raise StageFPlanError(
                "stage_f_passed_spectral_evidence_incomplete"
            ) from error
        aggregates_finite = False
    if (
        nested_passed
        and (
            int(summary.get("valid_window_count", 0)) <= 0
            or not aggregates_finite
        )
    ):
        raise StageFPlanError(
            "stage_f_passed_spectral_evidence_incomplete"
        )
    if not nested_passed:
        try:
            failed_values_valid = all(
                summary[name] is None
                or math.isfinite(float(summary[name]))
                for name in _SPECTRAL_AGGREGATE_FIELDS
            )
        except (TypeError, ValueError):
            failed_values_valid = False
        if not failed_values_valid:
            raise StageFPlanError(
                "stage_f_failed_spectral_evidence_invalid"
            )
    windows = _require_list(
        "stage_f_spectral_window_metrics",
        summary.get("window_metrics"),
    )
    window_fields = {
        "visible_top3_before",
        "visible_top3_after",
        "prominence_db_delta",
        "hr_band_share_delta",
        "pulse_power_retention",
        "residual_artifact_corr_before",
        "residual_artifact_corr_after",
        "residual_artifact_corr_delta",
    }
    complete_windows = True
    for raw_window in windows:
        window = _require_mapping(
            "stage_f_spectral_window_metric",
            raw_window,
        )
        if (
            not window_fields <= set(window)
            or not isinstance(window["visible_top3_before"], bool)
            or not isinstance(window["visible_top3_after"], bool)
        ):
            complete_windows = False
            break
        try:
            if any(
                not math.isfinite(float(window[name]))
                for name in window_fields
                if not name.startswith("visible_top3_")
            ):
                complete_windows = False
                break
        except (TypeError, ValueError):
            complete_windows = False
            break
    if (
        len(windows) != int(summary["valid_window_count"])
        or not complete_windows
    ):
        raise StageFPlanError(
            "stage_f_passed_spectral_window_evidence_incomplete"
        )


def _qualification(
    *,
    candidate: Mapping[str, Any],
    current: Mapping[str, Any],
    independent: Mapping[str, Any],
    true_rise_applicable: bool,
) -> dict[str, Any]:
    candidate_metrics = _require_mapping(
        "stage_f_candidate_metrics",
        candidate.get("metrics"),
    )
    current_metrics = _require_mapping(
        "stage_f_current_metrics",
        current.get("metrics"),
    )
    spectral = _require_mapping(
        "stage_f_candidate_spectral_audit",
        candidate.get("spectral_audit"),
    )
    _validate_spectral_evidence(spectral)
    reasons: list[str] = []
    if (
        spectral.get("stability_pass") is not True
        or spectral.get("spectral_gate_pass") is not True
    ):
        reasons.append("spectral_gate_contract_v1")
    candidate_l10 = int(candidate_metrics["longest_e10_run_windows"])
    candidate_l20 = int(candidate_metrics["longest_e20_run_windows"])
    independent_l10 = int(independent["longest_e10_run_windows"])
    independent_l20 = int(independent["longest_e20_run_windows"])
    if candidate_l10 > max(10, independent_l10 + 2):
        reasons.append("independent_l10_gate")
    if candidate_l20 > max(2, independent_l20):
        reasons.append("independent_l20_gate")
    candidate_mae = _metric_float(
        candidate_metrics,
        "final_motion_mae_bpm",
    )
    independent_mae = _metric_float(
        independent,
        "final_motion_mae_bpm",
    )
    current_mae = _metric_float(
        current_metrics,
        "final_motion_mae_bpm",
    )
    if candidate_mae - independent_mae > 2.0:
        reasons.append("independent_mae_gate")
    if (
        int(candidate_metrics["right_censored_recovery_count"])
        > int(current_metrics["right_censored_recovery_count"])
    ):
        reasons.append("new_right_censored_recovery")
    if (
        int(current_metrics["longest_e10_run_windows"]) <= 10
        and candidate_l10 >= 20
    ):
        reasons.append("current_l10_catastrophic_regression")
    if candidate_mae - current_mae > 2.0:
        reasons.append("current_mae_gate")
    if true_rise_applicable:
        candidate_rise = _metric_float(
            candidate_metrics,
            "max_rise_underestimate_bpm",
        )
        current_rise = _metric_float(
            current_metrics,
            "max_rise_underestimate_bpm",
        )
        if candidate_rise - current_rise > 2.0:
            reasons.append("true_rise_underestimate_gate")
    return {
        "qualified": not reasons,
        "elimination_reasons": reasons,
        "independent_delta_mae_bpm": candidate_mae - independent_mae,
        "current_delta_mae_bpm": candidate_mae - current_mae,
    }


def _with_self_hash(
    payload: dict[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    payload[field] = canonical_sha256(payload)
    return payload


def _validate_completed_stage_f(
    *,
    completion_path: Path,
    proposal: Mapping[str, Any],
    governance_root: Path,
    destination: Path,
    registry: AttemptRegistry,
    identities: tuple[AttemptIdentity, ...],
) -> dict[str, Any]:
    completion = read_json(completion_path)
    _verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="stage_f_completion",
    )
    if completion.get("proposal_sha256") != proposal.get("proposal_sha256"):
        raise StageFPlanError("stage_f_completion_proposal_mismatch")
    expected_unique = int(proposal["planned_unique_identity_count"])
    expected_numerical_spectral_reuse = expected_unique - 96
    expected_logical_reuse = 192 - expected_unique
    if (
        completion.get("completion_version")
        != "lyx_stage_f_completion_v1"
        or completion.get("status") != "complete"
        or completion.get("evidence_class")
        != "development_reuse_pilot"
        or completion.get("algorithm_level_holdout") is not False
        or completion.get("logical_task_count") != 192
        or completion.get("logical_result_count") != 192
        or completion.get("formal_result_count") != expected_unique
        or completion.get("profile_enumeration_result_count") != 96
        or completion.get(
            "same_role_current_control_result_count"
        )
        != 96
        or completion.get("planned_unique_identity_count")
        != expected_unique
        or completion.get("unique_spectral_audit_count") != 96
        or completion.get("spectral_audit_result_binding_count")
        != expected_unique
        or completion.get(
            "spectral_audit_numerical_reuse_count"
        )
        != expected_numerical_spectral_reuse
        or completion.get("spectral_audit_logical_reuse_count")
        != expected_logical_reuse
        or completion.get("reused_logical_task_count")
        != expected_logical_reuse
        or completion.get("independent_bo_run_count") != 0
        or completion.get("next_state")
        != "ready_for_penalty_interaction_completion"
    ):
        raise StageFPlanError("stage_f_completion_contract_mismatch")
    artifacts = _require_mapping(
        "stage_f_completion_artifacts",
        completion.get("artifacts"),
    )
    if set(artifacts) != {
        "profile_enumeration_matrix.json",
        "same_role_current_control_matrix.json",
        "profile_sample_in_upper_bound.json",
        "attempt_registry_stage_f_snapshot.json",
    }:
        raise StageFPlanError(
            "stage_f_completion_artifact_set_mismatch"
        )
    for name, expected_hash in artifacts.items():
        path = (destination / str(name)).resolve()
        if (
            not path.is_relative_to(destination)
            or not path.is_file()
            or file_sha256(path) != expected_hash
        ):
            raise StageFPlanError(
                f"stage_f_completion_artifact_mismatch:{name}"
            )
    governance_path = governance_root / "stage_f_governance_receipt.json"
    if (
        not governance_path.is_file()
        or file_sha256(governance_path)
        != completion.get("governance_receipt_file_sha256")
    ):
        raise StageFPlanError(
            "stage_f_completion_governance_receipt_mismatch"
        )
    governance = read_json(governance_path)
    _verify_embedded_hash(
        governance,
        hash_field="receipt_sha256",
        artifact_name="stage_f_governance_receipt",
    )
    if (
        governance.get("receipt_version")
        != "lyx_stage_f_governance_receipt_v1"
        or governance.get("status") != "complete"
        or governance.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or governance.get("logical_task_count") != 192
        or governance.get("logical_result_count") != 192
        or governance.get("formal_result_count") != expected_unique
        or governance.get("profile_enumeration_result_count") != 96
        or governance.get(
            "same_role_current_control_result_count"
        )
        != 96
        or governance.get("planned_unique_identity_count")
        != expected_unique
        or governance.get("unique_spectral_audit_count") != 96
        or governance.get("spectral_audit_result_binding_count")
        != expected_unique
        or governance.get(
            "spectral_audit_numerical_reuse_count"
        )
        != expected_numerical_spectral_reuse
        or governance.get("spectral_audit_logical_reuse_count")
        != expected_logical_reuse
        or governance.get("reused_logical_task_count")
        != expected_logical_reuse
        or governance.get("independent_bo_run_count") != 0
        or governance.get("artifacts") != artifacts
        or governance.get("receipt_sha256")
        != completion.get("governance_receipt_sha256")
    ):
        raise StageFPlanError("stage_f_governance_binding_mismatch")
    snapshot = read_json(
        destination / "attempt_registry_stage_f_snapshot.json"
    )
    registry.assert_matrix_matches_snapshot(identities, snapshot)
    matrix_summary = registry.matrix_execution_summary(identities)
    if (
        snapshot.get("snapshot_sha256")
        != governance.get("attempt_registry_matrix_snapshot_sha256")
        or matrix_summary != completion.get("matrix_execution_summary")
        or matrix_summary != governance.get("matrix_execution_summary")
        or completion.get("formal_solver_run_count")
        != matrix_summary["identity_with_solver_attempt_count"]
        or completion.get("cache_hit_count")
        != matrix_summary["cache_only_identity_count"]
        or completion.get("failed_attempt_count")
        != matrix_summary["failed_attempt_count"]
    ):
        raise StageFPlanError("stage_f_completion_registry_mismatch")
    return completion


def _load_or_run_stage_f_spectral_audit(
    item: Mapping[str, Any],
    *,
    spectral_audit_dir: Path,
) -> dict[str, Any]:
    profile_id = str(item["filter_profile_id"])
    record_id = str(item["record_id"])
    audit_path = (
        spectral_audit_dir / profile_id / f"{record_id}.json"
    )
    contract = StageRSpectralGateContract()
    expected = {
        "profile_id": profile_id,
        "profile_sha256": item["filter_profile_sha256"],
        "record_id": record_id,
        "data_sha256": item["raw_data_sha256"],
        "reference_sha256": item["reference_sha256"],
        "audit_contract_sha256": contract.sha256,
    }
    if audit_path.is_file():
        payload = read_json(audit_path)
        _verify_embedded_hash(
            payload,
            hash_field="audit_sha256",
            artifact_name=(
                f"stage_f_spectral_audit:{profile_id}:{record_id}"
            ),
        )
        if any(
            payload.get(name) != value
            for name, value in expected.items()
        ):
            raise StageFPlanError(
                "stage_f_spectral_audit_identity_mismatch:"
                f"{profile_id}:{record_id}"
            )
        audit = _normalize_stage_f_spectral_evidence(
            _require_mapping(
                "stage_f_spectral_audit",
                payload.get("audit"),
            )
        )
        _validate_spectral_evidence(audit)
        return {
            **audit,
            "audit_sha256": payload["audit_sha256"],
        }
    profile = FilterProfile(
        profile_id=profile_id,
        design_role=str(item["filter_profile_design_role"]),  # type: ignore[arg-type]
        fs_target=int(item["config"]["parameters"]["fs_target"]),
        memory_ms=int(item["physical_memory_ms"]),
        nominal_mu=float(item["config"]["parameters"]["lms_mu_base"]),
        recovery_sentinel_role=item.get("sentinel_role"),  # type: ignore[arg-type]
    )
    record = FilterAuditRecord(
        record_id=record_id,
        scene=str(item["scene"]),
        data_path=str(item["data_path"]),
        reference_path=str(item["reference_path"]),
        data_sha256=str(item["raw_data_sha256"]),
        reference_sha256=str(item["reference_sha256"]),
    )
    audit = _normalize_stage_f_spectral_evidence(
        audit_stage_r_profile_record(
            profile,
            record,
            contract=contract,
        )
    )
    _validate_spectral_evidence(audit)
    payload = {
        "audit_version": "lyx_stage_f_spectral_record_audit_v1",
        **expected,
        "candidate_invariant": True,
        "audit": audit,
    }
    payload["audit_sha256"] = canonical_sha256(payload)
    atomic_write_json(audit_path, payload)
    return {**audit, "audit_sha256": payload["audit_sha256"]}


def _run_stage_f_numerical_identity(
    item: dict[str, Any],
    spectral_audit_dir: Path,
) -> StageRNumericalResult:
    data_path = Path(str(item["data_path"])).resolve()
    reference_path = Path(str(item["reference_path"])).resolve()
    if file_sha256(data_path) != item["raw_data_sha256"]:
        raise StageFPlanError(
            f"stage_f_data_hash_mismatch:{item['record_id']}"
        )
    if file_sha256(reference_path) != item["reference_sha256"]:
        raise StageFPlanError(
            f"stage_f_reference_hash_mismatch:{item['record_id']}"
        )
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
    spectral_audit = _load_or_run_stage_f_spectral_audit(
        item,
        spectral_audit_dir=spectral_audit_dir,
    )
    return StageRNumericalResult(
        solver_result=result,
        metrics=asdict(metrics),
        spectral_audit=spectral_audit,
    )


def _execute_stage_f_identity_with_retry(
    *,
    registry: AttemptRegistry,
    item: dict[str, Any],
    numerical_runner: StageRNumericalRunner,
    spectral_audit_dir: Path,
    retry_limit: int,
    progress_callback: StageFProgressCallback | None,
) -> dict[str, Any]:
    for attempt_index in range(retry_limit + 1):
        try:
            return execute_stage_r_identity(
                registry=registry,
                item=item,
                numerical_runner=numerical_runner,
                spectral_audit_dir=spectral_audit_dir,
            )
        except Exception as error:
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "stage_f_filter_matrix_retry",
                        "identity_sha256": item[
                            "identity_sha256"
                        ],
                        "failed_attempt_index": attempt_index,
                        "will_retry": attempt_index < retry_limit,
                        "failure_type": type(error).__name__,
                    }
                )
            if attempt_index >= retry_limit:
                raise
    raise AssertionError("unreachable_stage_f_retry_loop")


def execute_stage_f_proposal(
    *,
    proposal_dir: Path,
    governance_dir: Path,
    output_dir: Path,
    source_root: Path,
    _numerical_runner: StageRNumericalRunner | None = None,
    progress_callback: StageFProgressCallback | None = None,
) -> dict[str, Any]:
    """Execute or resume the exact frozen Stage F identity matrix."""

    proposal_root = Path(proposal_dir).resolve()
    proposal, source_budget = _verify_stage_f_preflight(
        proposal_root=proposal_root,
        source_root=Path(source_root).resolve(),
    )
    numerical_runner = (
        _run_stage_f_numerical_identity
        if _numerical_runner is None
        else _numerical_runner
    )
    governance_root = Path(governance_dir).resolve()
    governance_budget = _budget_contract_from_payload(
        read_json(governance_root / "budget_contract.json")
    )
    if (
        governance_budget.sha256 != source_budget.sha256
        or governance_budget.to_dict() != source_budget.to_dict()
    ):
        raise StageFPlanError("stage_f_governance_budget_mismatch")
    exploration = _exploration_registry_from_payload(
        read_json(governance_root / "exploration_registry.json")
    )
    registry = AttemptRegistry.open(
        governance_root / "attempt_registry.json",
        budget_contract=governance_budget,
        exploration_registry=exploration,
    )
    raw_identities = _require_list(
        "stage_f_identities",
        proposal.get("identities"),
    )
    identities = tuple(
        _attempt_identity_from_item(item)
        for item in raw_identities
    )
    destination = Path(output_dir).resolve()
    completion_path = destination / "stage_f_completion.json"
    if completion_path.is_file():
        return _validate_completed_stage_f(
            completion_path=completion_path,
            proposal=proposal,
            governance_root=governance_root,
            destination=destination,
            registry=registry,
            identities=identities,
        )
    destination.mkdir(parents=True, exist_ok=True)
    for identity in identities:
        registry.register_identity(identity)
    spectral_dir = destination / "spectral_audits"
    result_rows: list[dict[str, Any]] = []
    total = len(raw_identities)
    for index, raw in enumerate(raw_identities, start=1):
        item = dict(_require_mapping("stage_f_identity", raw))
        row = _execute_stage_f_identity_with_retry(
            registry=registry,
            item=item,
            numerical_runner=numerical_runner,
            spectral_audit_dir=spectral_dir,
            retry_limit=governance_budget.retry_limit,
            progress_callback=progress_callback,
        )
        result_rows.append(row)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "stage_f_filter_matrix",
                    "completed": index,
                    "total": total,
                    "identity_sha256": row["identity_sha256"],
                    "cache_hit": row["cache_hit"],
                }
            )
    spectral_hashes_by_coordinate: dict[
        tuple[str, str],
        set[str],
    ] = {}
    for row in result_rows:
        spectral = _require_mapping(
            "stage_f_result_spectral_audit",
            row.get("spectral_audit"),
        )
        coordinate = (
            str(row["filter_profile_id"]),
            str(row["record_id"]),
        )
        spectral_hashes_by_coordinate.setdefault(
            coordinate,
            set(),
        ).add(
            _require_hash(
                "stage_f_spectral_audit_sha256",
                spectral.get("audit_sha256"),
            )
        )
    if (
        len(spectral_hashes_by_coordinate) != 96
        or any(
            len(hashes) != 1
            for hashes in spectral_hashes_by_coordinate.values()
        )
    ):
        raise StageFPlanError(
            "stage_f_spectral_audit_candidate_invariance_mismatch"
        )
    registry.assert_complete_matrix(identities)
    by_identity = {
        str(row["identity_sha256"]): row
        for row in result_rows
    }
    logical_rows: list[dict[str, Any]] = []
    for raw_task in _require_list(
        "stage_f_logical_tasks",
        proposal.get("logical_tasks"),
    ):
        task = dict(
            _require_mapping("stage_f_logical_task", raw_task)
        )
        identity_hash = str(task["identity_sha256"])
        numerical = by_identity.get(identity_hash)
        if (
            numerical is None
            or numerical.get("stage")
            != task.get("numerical_identity_stage")
            or task.get("logical_stage")
            not in {_PROVISIONAL_STAGE, _CURRENT_ROLE_STAGE}
            or (
                task.get("matrix_role") == "provisional_recovery"
                and task.get("logical_stage") != _PROVISIONAL_STAGE
            )
            or (
                task.get("matrix_role")
                == "same_role_current_control"
                and task.get("logical_stage") != _CURRENT_ROLE_STAGE
            )
        ):
            raise StageFPlanError(
                "stage_f_logical_numerical_identity_mismatch"
            )
        logical_rows.append({**dict(numerical), **task})
    primary = [
        row
        for row in logical_rows
        if row["matrix_role"] == "provisional_recovery"
    ]
    current = [
        row
        for row in logical_rows
        if row["matrix_role"] == "same_role_current_control"
    ]
    if len(primary) != 96 or len(current) != 96:
        raise StageFPlanError("stage_f_logical_result_matrix_mismatch")
    current_by_coordinate = {
        (str(row["record_id"]), str(row["filter_profile_id"])): row
        for row in current
    }
    independent_by_record = {
        str(record["record_id"]): _require_mapping(
            "stage_f_independent_metrics",
            _require_mapping(
                "stage_f_record_panel_item",
                record,
            )["independent_metrics"],
        )
        for record in _require_list(
            "stage_f_record_panel",
            proposal.get("record_panel"),
        )
    }
    true_rise_by_record = {
        str(record["record_id"]): bool(record["true_rise_applicable"])
        for record in proposal["record_panel"]
    }
    profiles_by_id = {
        str(profile["profile_id"]): profile
        for profile in proposal["profiles"]
    }
    qualified_primary: list[dict[str, Any]] = []
    for row in primary:
        coordinate = (
            str(row["record_id"]),
            str(row["filter_profile_id"]),
        )
        gate = _qualification(
            candidate=row,
            current=current_by_coordinate[coordinate],
            independent=independent_by_record[coordinate[0]],
            true_rise_applicable=true_rise_by_record[coordinate[0]],
        )
        qualified_primary.append({**row, "qualification": gate})
    control_rows = [
        {
            **row,
            "qualification": _qualification(
                candidate=row,
                current=row,
                independent=independent_by_record[str(row["record_id"])],
                true_rise_applicable=true_rise_by_record[
                    str(row["record_id"])
                ],
            ),
        }
        for row in current
    ]
    upper_records: list[dict[str, Any]] = []
    for record_id in sorted(independent_by_record):
        eligible = [
            row
            for row in qualified_primary
            if row["record_id"] == record_id
            and row["qualification"]["qualified"]
        ]
        eligible.sort(
            key=lambda row: (
                int(row["metrics"]["longest_e10_run_windows"]),
                _metric_float(row["metrics"], "max_recovered_delay_s"),
                _metric_float(row["metrics"], "final_motion_mae_bpm"),
                int(
                    profiles_by_id[str(row["filter_profile_id"])][
                        "actual_taps"
                    ]
                ),
                str(row["filter_profile_id"]),
            )
        )
        upper_records.append(
            {
                "record_id": record_id,
                "scene": next(
                    str(row["scene"])
                    for row in qualified_primary
                    if row["record_id"] == record_id
                ),
                "status": (
                    "selected" if eligible else "no_safe_profile_for_record"
                ),
                "selected_profile_id": (
                    eligible[0]["filter_profile_id"]
                    if eligible
                    else None
                ),
                "selected_identity_sha256": (
                    eligible[0]["identity_sha256"]
                    if eligible
                    else None
                ),
                "qualified_profile_count": len(eligible),
            }
        )
    artifacts_payload = {
        "profile_enumeration_matrix.json": _with_self_hash(
            {
                "matrix_version": "lyx_stage_f_profile_matrix_v1",
                "matrix_role": "provisional_recovery",
                "algorithm_level_holdout": False,
                "row_count": 96,
                "unique_spectral_audit_count": 96,
                "rows": qualified_primary,
            },
            field="matrix_sha256",
        ),
        "same_role_current_control_matrix.json": _with_self_hash(
            {
                "matrix_version": (
                    "lyx_stage_f_current_role_matrix_v1"
                ),
                "matrix_role": "same_role_current_control",
                "algorithm_level_holdout": False,
                "row_count": 96,
                "unique_spectral_audit_count": 96,
                "rows": control_rows,
            },
            field="matrix_sha256",
        ),
        "profile_sample_in_upper_bound.json": _with_self_hash(
            {
                "upper_bound_version": (
                    "lyx_stage_f_sample_in_upper_bound_v1"
                ),
                "evidence_class": "diagnostic_sample_in_upper_bound",
                "algorithm_level_holdout": False,
                "record_count": 12,
                "records": upper_records,
            },
            field="upper_bound_sha256",
        ),
    }
    for name, payload in artifacts_payload.items():
        atomic_write_json(destination / name, payload)
    matrix_snapshot = registry.matrix_snapshot(identities)
    snapshot_name = "attempt_registry_stage_f_snapshot.json"
    atomic_write_json(destination / snapshot_name, matrix_snapshot)
    artifact_names = (*artifacts_payload, snapshot_name)
    artifacts = {
        name: file_sha256(destination / name)
        for name in artifact_names
    }
    matrix_summary = registry.matrix_execution_summary(identities)
    governance_receipt = {
        "receipt_version": "lyx_stage_f_governance_receipt_v1",
        "status": "complete",
        "proposal_sha256": proposal["proposal_sha256"],
        "identity_matrix_sha256": canonical_sha256(
            [identity.sha256 for identity in identities]
        ),
        "attempt_registry_matrix_snapshot_sha256": matrix_snapshot[
            "snapshot_sha256"
        ],
        "matrix_execution_summary": matrix_summary,
        "logical_task_count": 192,
        "logical_result_count": 192,
        "formal_result_count": len(identities),
        "profile_enumeration_result_count": 96,
        "same_role_current_control_result_count": 96,
        "planned_unique_identity_count": len(identities),
        "unique_spectral_audit_count": 96,
        "spectral_audit_result_binding_count": len(result_rows),
        "spectral_audit_numerical_reuse_count": (
            len(result_rows) - 96
        ),
        "spectral_audit_logical_reuse_count": (
            192 - len(result_rows)
        ),
        "reused_logical_task_count": proposal[
            "reused_logical_task_count"
        ],
        "independent_bo_run_count": 0,
        "artifacts": artifacts,
    }
    governance_receipt["receipt_sha256"] = canonical_sha256(
        governance_receipt
    )
    governance_path = governance_root / "stage_f_governance_receipt.json"
    atomic_write_json(governance_path, governance_receipt)
    completion = {
        "completion_version": "lyx_stage_f_completion_v1",
        "status": "complete",
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "proposal_sha256": proposal["proposal_sha256"],
        "logical_task_count": 192,
        "logical_result_count": 192,
        "formal_result_count": len(identities),
        "profile_enumeration_result_count": 96,
        "same_role_current_control_result_count": 96,
        "planned_unique_identity_count": len(identities),
        "unique_spectral_audit_count": 96,
        "spectral_audit_result_binding_count": len(result_rows),
        "spectral_audit_numerical_reuse_count": (
            len(result_rows) - 96
        ),
        "spectral_audit_logical_reuse_count": (
            192 - len(result_rows)
        ),
        "reused_logical_task_count": proposal[
            "reused_logical_task_count"
        ],
        "formal_solver_run_count": matrix_summary[
            "identity_with_solver_attempt_count"
        ],
        "cache_hit_count": matrix_summary["cache_only_identity_count"],
        "failed_attempt_count": matrix_summary["failed_attempt_count"],
        "independent_bo_run_count": 0,
        "matrix_execution_summary": matrix_summary,
        "artifacts": artifacts,
        "governance_receipt_sha256": governance_receipt[
            "receipt_sha256"
        ],
        "governance_receipt_file_sha256": file_sha256(
            governance_path
        ),
        "next_state": "ready_for_penalty_interaction_completion",
    }
    completion["completion_sha256"] = canonical_sha256(completion)
    atomic_write_json(completion_path, completion)
    return _validate_completed_stage_f(
        completion_path=completion_path,
        proposal=proposal,
        governance_root=governance_root,
        destination=destination,
        registry=registry,
        identities=identities,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="冻结 LYX Stage F 8×12 双矩阵零运行 proposal",
    )
    parser.add_argument("--stage-r-proposal", required=True, type=Path)
    parser.add_argument("--stage-r-completion", required=True, type=Path)
    parser.add_argument("--profile-library", required=True, type=Path)
    parser.add_argument(
        "--profile-library-completion",
        required=True,
        type=Path,
    )
    parser.add_argument("--baseline-metrics", required=True, type=Path)
    parser.add_argument(
        "--baseline-contract-receipt",
        required=True,
        type=Path,
    )
    parser.add_argument("--recovery-registry", required=True, type=Path)
    parser.add_argument("--penalty-registry", required=True, type=Path)
    parser.add_argument("--budget-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--parent-experiment-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = propose_stage_f_execution(
        stage_r_proposal_path=args.stage_r_proposal,
        stage_r_completion_path=args.stage_r_completion,
        profile_library_path=args.profile_library,
        profile_library_completion_path=(
            args.profile_library_completion
        ),
        baseline_metrics_path=args.baseline_metrics,
        baseline_contract_receipt_path=(
            args.baseline_contract_receipt
        ),
        recovery_registry_path=args.recovery_registry,
        penalty_registry_path=args.penalty_registry,
        budget_contract_path=args.budget_contract,
        output_dir=args.output_dir,
        source_root=args.source_root,
        parent_experiment_id=args.parent_experiment_id,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
