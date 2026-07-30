"""Freeze the exact zero-run LYX Stage F experiment plan."""

from __future__ import annotations

import os
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .experiment_freeze_utils import runtime_source_identity
from .phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    read_json,
)
from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import AttemptIdentity
from .recovery_stage_f_contracts import (
    _CURRENT_ROLE_STAGE,
    _EXPECTED_FS_QUOTA,
    _EXPECTED_ROLE_COUNTS,
    _EXPECTED_SCENE_COUNTS,
    _PROVISIONAL_STAGE,
    _RATE_NORMALIZED_PROFILE_IDS,
    _REUSED_RATE_NORMALIZED_P50_PROFILE_IDS,
    StageFPlanError,
    _require_hash,
    _require_list,
    _require_mapping,
    _verify_embedded_hash,
)
from .recovery_stage_r_experiment import (
    stage_r_metric_contract_v1,
    stage_r_spectral_gate_contract_v2,
)


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
    spectral_contract = stage_r_spectral_gate_contract_v2()
    solver = runtime_source_identity(Path(source_root).resolve())
    evaluation_roots = (
        "ppg_hr.v2.recovery_stage_f_contracts",
        "ppg_hr.v2.recovery_stage_f_execution",
        "ppg_hr.v2.recovery_stage_f_experiment",
        "ppg_hr.v2.recovery_stage_f_plan",
        "ppg_hr.v2.recovery_stage_f_reporting",
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
