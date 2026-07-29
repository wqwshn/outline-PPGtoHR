"""Aggregate Stage F matrices and publish hash-closed evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    read_json,
)
from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import (
    AttemptIdentity,
    AttemptRegistry,
)
from .recovery_stage_f_contracts import (
    _CURRENT_ROLE_STAGE,
    _PROVISIONAL_STAGE,
    StageFPlanError,
    _require_hash,
    _require_list,
    _require_mapping,
    _verify_embedded_hash,
)

_SPECTRAL_AGGREGATE_FIELDS = (
    "prominence_db_delta_median",
    "visible_top3_rate_delta",
    "hr_band_share_delta_median",
    "pulse_power_retention_median",
    "residual_artifact_corr_delta_median",
)


def normalize_stage_f_spectral_evidence(
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Make every aggregate explicit before evidence is persisted."""

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


def validate_spectral_evidence(
    spectral: Mapping[str, Any],
) -> None:
    """Fail closed on incomplete passed or failed spectral evidence."""

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


def _metric_float(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if value is None:
        return float("inf")
    number = float(value)
    if not number == number:
        return float("inf")
    return number


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
    validate_spectral_evidence(spectral)
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


def validate_completed_stage_f(
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


def finalize_stage_f_report(
    *,
    proposal: Mapping[str, Any],
    governance_root: Path,
    destination: Path,
    registry: AttemptRegistry,
    identities: tuple[AttemptIdentity, ...],
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build and validate the complete Stage F evidence package."""

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
    completion_path = destination / "stage_f_completion.json"
    atomic_write_json(completion_path, completion)
    return validate_completed_stage_f(
        completion_path=completion_path,
        proposal=proposal,
        governance_root=governance_root,
        destination=destination,
        registry=registry,
        identities=identities,
    )
