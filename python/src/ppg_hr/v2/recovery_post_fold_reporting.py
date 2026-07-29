"""Post-fold human gate, development report, and challenge-scene handoff.

This module is deliberately reporting-only.  It consumes frozen Stage P and
fold-replay artifacts, evaluates the two preregistered post-fold booleans, and
publishes evidence-bound reports.  It never invokes a solver or independent BO.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .phase2_experiment_io import atomic_write_json, file_sha256, read_json
from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import (
    AttemptRegistry,
    ExplorationRegistry,
    IndependentBORequest,
    validate_independent_bo_authorization,
)
from .recovery_fold_replay_contracts import (
    EXPECTED_LOGICAL_SLOT_COUNT,
    EXPECTED_PROFILE_COUNT,
    EXPECTED_RECORD_COUNT,
    EXPECTED_SCENE_COUNTS,
    FoldReplayError,
    budget_contract_from_payload,
    finite_float,
    nonnegative_int,
    require_hash,
    require_list,
    require_mapping,
    verify_embedded_hash,
)
from .recovery_profile_upper_bound import build_sample_in_upper_bound_payloads

_POST_FOLD_GAP_BPM = 2.0
_POST_FOLD_MIN_RECORDS = 3
_POST_FOLD_MIN_SCENES = 2
_POST_FOLD_MIN_FAILED_SLOTS = 3
_ORIGINAL_BODY_UNIQUE_LIMIT = 684
_ORIGINAL_BODY_ATTEMPT_LIMIT = 1368
_KNOWN_APPROVED_BUDGETS = {
    "lyx_recovery_filter_budget_v3": (724, 1448),
    "lyx_recovery_filter_budget_v4": (748, 1496),
    "lyx_recovery_filter_budget_v5": (756, 1512),
}
_REVIEW_CONTEXT_FIELDS = {
    "solver_hash",
    "search_space_hash",
    "metric_contract_hash",
    "seed_manifest_hash",
    "unique_budget",
    "estimated_runtime",
    "estimated_cache_size",
    "plausible_mechanism_causes",
    "recommendation",
    "run_answers",
    "no_run_answers",
}


def _with_hash(payload: dict[str, Any], field: str) -> dict[str, Any]:
    payload[field] = canonical_sha256(payload)
    return payload


def _metric_summary(raw: object, *, name: str) -> dict[str, Any]:
    metrics = require_mapping(name, raw)
    rise = metrics.get("max_rise_underestimate_bpm")
    return {
        "final_motion_mae_bpm": finite_float(
            f"{name}_final_motion_mae_bpm",
            metrics.get("final_motion_mae_bpm"),
        ),
        "longest_e10_run_windows": nonnegative_int(
            f"{name}_longest_e10_run_windows",
            metrics.get("longest_e10_run_windows"),
        ),
        "longest_e20_run_windows": nonnegative_int(
            f"{name}_longest_e20_run_windows",
            metrics.get("longest_e20_run_windows"),
        ),
        "right_censored_recovery_count": nonnegative_int(
            f"{name}_right_censored_recovery_count",
            metrics.get("right_censored_recovery_count"),
        ),
        "max_rise_underestimate_bpm": (
            None
            if rise is None
            else finite_float(
                f"{name}_max_rise_underestimate_bpm",
                rise,
            )
        ),
    }


def _sample_metric_summary(raw: object, *, name: str) -> dict[str, Any]:
    """Preserve the intentionally smaller sample-in upper-bound schema."""

    metrics = require_mapping(name, raw)
    delay = metrics.get("selection_recovery_delay_s")
    return {
        "final_motion_mae_bpm": finite_float(
            f"{name}_final_motion_mae_bpm",
            metrics.get("final_motion_mae_bpm"),
        ),
        "longest_e10_run_windows": nonnegative_int(
            f"{name}_longest_e10_run_windows",
            metrics.get("longest_e10_run_windows"),
        ),
        "selection_recovery_delay_s": finite_float(
            f"{name}_selection_recovery_delay_s",
            delay,
        ),
        "actual_taps": nonnegative_int(
            f"{name}_actual_taps",
            metrics.get("actual_taps"),
        ),
    }


def _optional_delta(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> float | None:
    if left is None or right is None:
        return None
    return float(left["final_motion_mae_bpm"]) - float(right["final_motion_mae_bpm"])


def _validate_review_context(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], IndependentBORequest]:
    context = dict(raw)
    if set(context) != _REVIEW_CONTEXT_FIELDS:
        raise FoldReplayError("post_fold_review_context_field_set_mismatch")
    for field in (
        "solver_hash",
        "search_space_hash",
        "metric_contract_hash",
        "seed_manifest_hash",
    ):
        require_hash(f"post_fold_{field}", context.get(field))
    unique_budget = nonnegative_int(
        "post_fold_unique_budget",
        context.get("unique_budget"),
    )
    if unique_budget <= 0:
        raise FoldReplayError("post_fold_unique_budget_must_be_positive")
    for field in (
        "estimated_runtime",
        "estimated_cache_size",
        "recommendation",
        "run_answers",
        "no_run_answers",
    ):
        if not isinstance(context.get(field), str) or not context[field].strip():
            raise FoldReplayError(f"post_fold_{field}_must_be_nonempty_string")
    causes = require_list(
        "post_fold_plausible_mechanism_causes",
        context.get("plausible_mechanism_causes"),
    )
    if not causes or any(not isinstance(value, str) or not value.strip() for value in causes):
        raise FoldReplayError("post_fold_plausible_mechanism_causes_invalid")
    request = IndependentBORequest(
        solver_hash=str(context["solver_hash"]),
        search_space_hash=str(context["search_space_hash"]),
        metric_contract_hash=str(context["metric_contract_hash"]),
        seed_manifest_hash=str(context["seed_manifest_hash"]),
        unique_budget=unique_budget,
    )
    return context, request


def default_challenge_scene_manifest() -> dict[str, Any]:
    """Freeze the next-cycle scene policy without reading challenge results."""

    return _with_hash(
        {
            "manifest_version": "lyx_challenge_scene_manifest_v1",
            "status": "frozen_unseen_scene_plan",
            "development_scene_ids": sorted(EXPECTED_SCENE_COUNTS),
            "reserved_challenge_scene_ids": ["bobi"],
            "challenge_scene_roles": {
                "bobi": "high_dynamic_challenge",
            },
            "additional_scene_policy": (
                "include_remaining_LYX_scenes_not_used_by_the_development_cycle"
            ),
            "challenge_result_read_count": 0,
            "rule_revision_after_challenge_count": 0,
            "cross_person_study_order": "after_unseen_scene_validation",
        },
        "manifest_sha256",
    )


def _validate_fold_sources(
    *,
    fold_replay_report: Mapping[str, Any],
    fold_selection_receipt: Mapping[str, Any],
    target_audits_by_fold: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    verify_embedded_hash(
        fold_replay_report,
        hash_field="report_sha256",
        artifact_name="post_fold_replay_report",
    )
    verify_embedded_hash(
        fold_selection_receipt,
        hash_field="receipt_sha256",
        artifact_name="post_fold_selection_receipt",
    )
    summaries = [
        dict(require_mapping("post_fold_fold_summary", raw))
        for raw in require_list(
            "post_fold_fold_summaries",
            fold_replay_report.get("folds"),
        )
    ]
    if (
        fold_replay_report.get("report_version") != "lyx_fold_replay_report_v1"
        or fold_replay_report.get("status") != "complete"
        or fold_replay_report.get("evidence_class") != "development_replay_audit"
        or fold_replay_report.get("algorithm_level_holdout") is not False
        or fold_replay_report.get("logical_slot_count") != EXPECTED_LOGICAL_SLOT_COUNT
        or fold_replay_report.get("denominator_slot_count") != EXPECTED_LOGICAL_SLOT_COUNT
        or len(summaries) != EXPECTED_LOGICAL_SLOT_COUNT
        or fold_replay_report.get("independent_bo_run_count") != 0
        or fold_replay_report.get("next_state") != "ready_for_post_fold_independent_bo_gate"
        or fold_replay_report.get("fold_selection_receipt_sha256")
        != fold_selection_receipt.get("receipt_sha256")
        or fold_selection_receipt.get("folds") != summaries
        or fold_selection_receipt.get("evidence_class") != "development_replay_audit"
        or fold_selection_receipt.get("algorithm_level_holdout") is not False
    ):
        raise FoldReplayError("post_fold_replay_contract_mismatch")
    fold_ids = [str(summary.get("fold_id", "")) for summary in summaries]
    if (
        any(not fold_id for fold_id in fold_ids)
        or len(set(fold_ids)) != EXPECTED_LOGICAL_SLOT_COUNT
        or set(target_audits_by_fold) != set(fold_ids)
    ):
        raise FoldReplayError("post_fold_target_audit_panel_mismatch")
    validated: dict[str, Mapping[str, Any]] = {}
    for summary in summaries:
        fold_id = str(summary["fold_id"])
        audit = target_audits_by_fold[fold_id]
        verify_embedded_hash(
            audit,
            hash_field="receipt_sha256",
            artifact_name=f"post_fold_target_audit:{fold_id}",
        )
        if (
            audit.get("receipt_version") != "lyx_fold_target_audit_receipt_v1"
            or audit.get("fold_id") != fold_id
            or audit.get("selection_sha256") != summary.get("selection_sha256")
            or audit.get("audit_target_record_id")
            not in {None, summary.get("audit_target_record_id")}
            or audit.get("selected_filter_profile_id")
            not in {None, summary.get("selected_filter_profile_id")}
            or audit.get("status") != summary.get("target_audit_status")
            or audit.get("audit_pass") is not summary.get("audit_pass")
            or audit.get("failure_reasons") != summary.get("failure_reasons")
            or audit.get("target_performance_read_count")
            != summary.get("target_performance_read_count")
        ):
            raise FoldReplayError("post_fold_target_audit_binding_mismatch")
        validated[fold_id] = audit
    passed = sum(summary.get("audit_pass") is True for summary in summaries)
    no_safe = sum(
        summary.get("selection_status") == "no_safe_shared_candidate" for summary in summaries
    )
    if (
        passed != fold_replay_report.get("passed_slot_count")
        or len(summaries) - passed != fold_replay_report.get("failed_slot_count")
        or no_safe != fold_replay_report.get("no_safe_shared_candidate_count")
    ):
        raise FoldReplayError("post_fold_replay_count_mismatch")
    return summaries, validated


def _validate_final_interaction(
    final_interaction_audit: Mapping[str, Any],
) -> tuple[
    list[Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
]:
    verify_embedded_hash(
        final_interaction_audit,
        hash_field="audit_sha256",
        artifact_name="post_fold_final_interaction_audit",
    )
    rows = [
        require_mapping("post_fold_final_profile_row", raw)
        for raw in require_list(
            "post_fold_final_profile_rows",
            final_interaction_audit.get("rows"),
        )
    ]
    independent = {
        str(record_id): require_mapping(
            f"post_fold_independent_metrics:{record_id}",
            metrics,
        )
        for record_id, metrics in require_mapping(
            "post_fold_independent_metrics_by_record",
            final_interaction_audit.get("independent_metrics_by_record"),
        ).items()
    }
    scene_by_record: dict[str, str] = {}
    coordinates: set[tuple[str, str]] = set()
    for row in rows:
        record_id = str(row.get("record_id", ""))
        scene = str(row.get("scene", ""))
        profile_id = str(row.get("filter_profile_id", ""))
        if not record_id or not scene or not profile_id:
            raise FoldReplayError("post_fold_final_profile_coordinate_missing")
        if scene_by_record.setdefault(record_id, scene) != scene:
            raise FoldReplayError("post_fold_final_profile_scene_mismatch")
        coordinates.add((record_id, profile_id))
    if (
        final_interaction_audit.get("audit_version") != "lyx_final_interaction_audit_v1"
        or final_interaction_audit.get("status") != "complete"
        or final_interaction_audit.get("algorithm_level_holdout") is not False
        or final_interaction_audit.get("row_count") != 96
        or len(rows) != 96
        or len(coordinates) != 96
        or len(scene_by_record) != EXPECTED_RECORD_COUNT
        or set(independent) != set(scene_by_record)
        or any(
            sum(record_scene == scene for record_scene in scene_by_record.values()) != expected
            for scene, expected in EXPECTED_SCENE_COUNTS.items()
        )
        or final_interaction_audit.get("independent_bo_run_count") != 0
    ):
        raise FoldReplayError("post_fold_final_interaction_contract_mismatch")
    expected_upper = build_sample_in_upper_bound_payloads(
        final_profile_rows=rows,
        scene_by_record=scene_by_record,
    )
    if any(final_interaction_audit.get(name) != value for name, value in expected_upper.items()):
        raise FoldReplayError("post_fold_sample_in_upper_bound_mismatch")
    sample_by_record = {
        str(record["record_id"]): require_mapping(
            "post_fold_sample_in_record",
            record,
        )
        for record in expected_upper["sample_in_upper_bound"]["records"]
    }
    return rows, independent, sample_by_record


def _validate_historical_report(
    historical_ab_report: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    verify_embedded_hash(
        historical_ab_report,
        hash_field="report_sha256",
        artifact_name="post_fold_historical_ab_report",
    )
    records = {
        str(record["record_id"]): record
        for record in (
            require_mapping("post_fold_historical_ab_record", raw)
            for raw in require_list(
                "post_fold_historical_ab_records",
                historical_ab_report.get("records"),
            )
        )
    }
    if (
        historical_ab_report.get("report_version") != "lyx_historical_recovery_ab_report_v1"
        or historical_ab_report.get("status") != "complete"
        or len(records) != EXPECTED_RECORD_COUNT
        or historical_ab_report.get("independent_bo_run_count") != 0
    ):
        raise FoldReplayError("post_fold_historical_ab_contract_mismatch")
    return records


def _validate_current_role_matrix(
    current_role_matrix: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    verify_embedded_hash(
        current_role_matrix,
        hash_field="matrix_sha256",
        artifact_name="post_fold_current_role_matrix",
    )
    rows = [
        require_mapping("post_fold_current_role_row", raw)
        for raw in require_list(
            "post_fold_current_role_rows",
            current_role_matrix.get("rows"),
        )
    ]
    indexed = {
        (str(row.get("record_id", "")), str(row.get("filter_profile_id", ""))): row for row in rows
    }
    if (
        current_role_matrix.get("matrix_version") != "lyx_stage_f_current_role_matrix_v1"
        or current_role_matrix.get("matrix_role") != "same_role_current_control"
        or current_role_matrix.get("algorithm_level_holdout") is not False
        or current_role_matrix.get("row_count") != 96
        or len(rows) != 96
        or len(indexed) != 96
        or len({record_id for record_id, _ in indexed}) != EXPECTED_RECORD_COUNT
        or len({profile_id for _, profile_id in indexed}) != EXPECTED_PROFILE_COUNT
    ):
        raise FoldReplayError("post_fold_current_role_matrix_contract_mismatch")
    return indexed


def _comparison_rows(
    *,
    summaries: Sequence[Mapping[str, Any]],
    target_audits_by_fold: Mapping[str, Mapping[str, Any]],
    independent_by_record: Mapping[str, Mapping[str, Any]],
    sample_by_record: Mapping[str, Mapping[str, Any]],
    historical_by_record: Mapping[str, Mapping[str, Any]],
    current_role_by_coordinate: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sorted(
        summaries,
        key=lambda value: (
            str(value["scene"]),
            str(value["audit_target_record_id"]),
        ),
    ):
        fold_id = str(summary["fold_id"])
        record_id = str(summary["audit_target_record_id"])
        scene = str(summary["scene"])
        audit = target_audits_by_fold[fold_id]
        historical = historical_by_record.get(record_id)
        sample = sample_by_record.get(record_id)
        if historical is None or sample is None or record_id not in independent_by_record:
            raise FoldReplayError("post_fold_comparison_record_mismatch")
        if historical.get("scene") != scene or sample.get("scene") != scene:
            raise FoldReplayError("post_fold_comparison_scene_mismatch")
        independent_metrics = _metric_summary(
            independent_by_record[record_id],
            name=f"post_fold_independent:{record_id}",
        )
        historical_current = _metric_summary(
            historical.get("current_metrics"),
            name=f"post_fold_historical_current:{record_id}",
        )
        historical_final = _metric_summary(
            historical.get("final_metrics"),
            name=f"post_fold_historical_final:{record_id}",
        )
        sample_metrics = (
            None
            if sample.get("selected_metrics") is None
            else _sample_metric_summary(
                sample["selected_metrics"],
                name=f"post_fold_sample_in:{record_id}",
            )
        )
        selected_profile_id = summary.get("selected_filter_profile_id")
        shared_metrics = (
            None
            if audit.get("metrics") is None
            else _metric_summary(
                audit["metrics"],
                name=f"post_fold_shared:{record_id}",
            )
        )
        current_role_row = (
            None
            if selected_profile_id is None
            else current_role_by_coordinate.get((record_id, str(selected_profile_id)))
        )
        if selected_profile_id is not None and current_role_row is None:
            raise FoldReplayError("post_fold_current_role_coordinate_missing")
        current_role_metrics = (
            None
            if current_role_row is None
            else _metric_summary(
                current_role_row.get("metrics"),
                name=f"post_fold_current_role:{record_id}",
            )
        )
        rows.append(
            {
                "fold_id": fold_id,
                "record_id": record_id,
                "scene": scene,
                "selected_filter_profile_id": selected_profile_id,
                "audit_pass": audit.get("audit_pass") is True,
                "selection_status": summary.get("selection_status"),
                "failure_reasons": list(audit.get("failure_reasons", [])),
                "historical_independent_bo_lite": independent_metrics,
                "same_identity_recovery_ab": {
                    "current_mechanism": historical_current,
                    "final_recovery_mechanism": historical_final,
                },
                "same_role_current_mechanism": current_role_metrics,
                "combination_library_sample_in_upper_bound": {
                    "selected_profile_id": sample.get("selected_profile_id"),
                    "engineering_gate_pass": sample.get("selected_qualified") is True,
                    "metrics": sample_metrics,
                },
                "scene_shared_profile_replay": shared_metrics,
                "mae_gaps_bpm": {
                    "shared_minus_historical_independent_bo_lite": _optional_delta(
                        shared_metrics,
                        independent_metrics,
                    ),
                    "shared_minus_historical_parameter_new_recovery": _optional_delta(
                        shared_metrics,
                        historical_final,
                    ),
                    "shared_minus_same_role_current_mechanism": _optional_delta(
                        shared_metrics,
                        current_role_metrics,
                    ),
                    "shared_minus_sample_in_upper_bound": _optional_delta(
                        shared_metrics,
                        sample_metrics,
                    ),
                },
            }
        )
    if (
        len(rows) != EXPECTED_RECORD_COUNT
        or len({row["record_id"] for row in rows}) != EXPECTED_RECORD_COUNT
    ):
        raise FoldReplayError("post_fold_comparison_panel_incomplete")
    return rows


def evaluate_post_fold_independent_bo_gate(
    *,
    fold_replay_report: Mapping[str, Any],
    fold_selection_receipt: Mapping[str, Any],
    target_audits_by_fold: Mapping[str, Mapping[str, Any]],
    final_interaction_audit: Mapping[str, Any],
    historical_ab_report: Mapping[str, Any],
    current_role_matrix: Mapping[str, Any],
    review_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate exactly the two preregistered post-fold Boolean conditions."""

    summaries, audits = _validate_fold_sources(
        fold_replay_report=fold_replay_report,
        fold_selection_receipt=fold_selection_receipt,
        target_audits_by_fold=target_audits_by_fold,
    )
    _, independent, sample = _validate_final_interaction(final_interaction_audit)
    historical = _validate_historical_report(historical_ab_report)
    current_role = _validate_current_role_matrix(current_role_matrix)
    comparison_rows = _comparison_rows(
        summaries=summaries,
        target_audits_by_fold=audits,
        independent_by_record=independent,
        sample_by_record=sample,
        historical_by_record=historical,
        current_role_by_coordinate=current_role,
    )
    gap_rows = [
        row
        for row in comparison_rows
        if row["mae_gaps_bpm"]["shared_minus_sample_in_upper_bound"] is not None
        and row["mae_gaps_bpm"]["shared_minus_sample_in_upper_bound"] > _POST_FOLD_GAP_BPM
        and row["combination_library_sample_in_upper_bound"]["engineering_gate_pass"] is True
    ]
    condition_1 = (
        len(gap_rows) >= _POST_FOLD_MIN_RECORDS
        and len({str(row["scene"]) for row in gap_rows}) >= _POST_FOLD_MIN_SCENES
    )
    failed_rows = [row for row in comparison_rows if row["audit_pass"] is False]
    condition_2 = len(failed_rows) >= _POST_FOLD_MIN_FAILED_SLOTS and all(
        row["combination_library_sample_in_upper_bound"]["engineering_gate_pass"] is True
        for row in failed_rows
    )
    triggered = condition_1 or condition_2
    context: dict[str, Any] | None = None
    request: IndependentBORequest | None = None
    if triggered:
        context, request = _validate_review_context(review_context)
    elif review_context:
        context, request = _validate_review_context(review_context)
    trigger_record_ids = {
        str(row["record_id"])
        for row in (([*gap_rows] if condition_1 else []) + ([*failed_rows] if condition_2 else []))
    }
    trigger_records = [
        {
            **row,
            "trigger_reasons": sorted(
                {
                    *(
                        ["scene_shared_minus_sample_in_upper_bound_gt_2_bpm"]
                        if condition_1 and row in gap_rows
                        else []
                    ),
                    *(
                        ["failed_development_replay_slot"]
                        if condition_2 and row in failed_rows
                        else []
                    ),
                }
            ),
        }
        for row in comparison_rows
        if row["record_id"] in trigger_record_ids
    ]
    status = (
        "awaiting_human_independent_bo_decision"
        if triggered
        else "ready_for_final_development_report"
    )
    receipt = {
        "receipt_version": "lyx_post_fold_independent_bo_gate_v1",
        "status": status,
        "evidence_class": "development_replay_audit",
        "algorithm_level_holdout": False,
        "triggered": triggered,
        "conditions": {
            "shared_replay_vs_sample_in_upper_bound_gap": {
                "triggered": condition_1,
                "threshold_bpm": _POST_FOLD_GAP_BPM,
                "minimum_record_count": _POST_FOLD_MIN_RECORDS,
                "minimum_scene_count": _POST_FOLD_MIN_SCENES,
                "record_ids": sorted(str(row["record_id"]) for row in gap_rows),
                "scene_ids": sorted({str(row["scene"]) for row in gap_rows}),
            },
            "failed_slots_with_safe_sample_in_upper_bound": {
                "triggered": condition_2,
                "minimum_failed_slot_count": _POST_FOLD_MIN_FAILED_SLOTS,
                "failed_slot_count": len(failed_rows),
                "record_ids": sorted(str(row["record_id"]) for row in failed_rows),
                "all_corresponding_sample_in_upper_bounds_pass": all(
                    row["combination_library_sample_in_upper_bound"]["engineering_gate_pass"]
                    is True
                    for row in failed_rows
                ),
            },
        },
        "trigger_records": trigger_records,
        "review_packet": (
            None
            if not triggered
            else {
                **dict(context or {}),
                "independent_bo_request": {
                    "solver_hash": request.solver_hash,
                    "search_space_hash": request.search_space_hash,
                    "metric_contract_hash": request.metric_contract_hash,
                    "seed_manifest_hash": request.seed_manifest_hash,
                    "unique_budget": request.unique_budget,
                },
            }
        ),
        "fold_replay_report_sha256": fold_replay_report["report_sha256"],
        "final_interaction_audit_sha256": final_interaction_audit["audit_sha256"],
        "historical_ab_report_sha256": historical_ab_report["report_sha256"],
        "current_role_matrix_sha256": current_role_matrix["matrix_sha256"],
        "independent_bo_authorized": False,
        "independent_bo_run_count": 0,
        "next_state": status,
    }
    return _with_hash(receipt, "receipt_sha256")


def validate_post_fold_independent_bo_authorization(
    *,
    gate_receipt: Mapping[str, Any],
    authorization_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind any future BO authorization to the exact post-fold request."""

    verify_embedded_hash(
        gate_receipt,
        hash_field="receipt_sha256",
        artifact_name="post_fold_authorization_gate_receipt",
    )
    review_packet = require_mapping(
        "post_fold_authorization_review_packet",
        gate_receipt.get("review_packet"),
    )
    request_payload = require_mapping(
        "post_fold_authorization_request",
        review_packet.get("independent_bo_request"),
    )
    if (
        gate_receipt.get("receipt_version") != "lyx_post_fold_independent_bo_gate_v1"
        or gate_receipt.get("triggered") is not True
        or gate_receipt.get("status") != "awaiting_human_independent_bo_decision"
        or gate_receipt.get("independent_bo_authorized") is not False
        or gate_receipt.get("independent_bo_run_count") != 0
    ):
        raise FoldReplayError("post_fold_authorization_gate_not_waiting")
    request = IndependentBORequest(
        solver_hash=str(request_payload["solver_hash"]),
        search_space_hash=str(request_payload["search_space_hash"]),
        metric_contract_hash=str(request_payload["metric_contract_hash"]),
        seed_manifest_hash=str(request_payload["seed_manifest_hash"]),
        unique_budget=int(request_payload["unique_budget"]),
    )
    return validate_independent_bo_authorization(
        request,
        receipt=authorization_receipt,
    )


def _budget_audit(
    *,
    budget_contract: Mapping[str, Any],
    attempt_registry_summary: Mapping[str, Any],
) -> dict[str, Any]:
    contract = budget_contract_from_payload(budget_contract)
    version = str(budget_contract.get("contract_version", ""))
    expected_limits = _KNOWN_APPROVED_BUDGETS.get(version)
    if expected_limits is None or expected_limits != (
        contract.max_unique_identities,
        contract.max_attempts,
    ):
        raise FoldReplayError("post_fold_budget_contract_not_approved")
    planned = nonnegative_int(
        "post_fold_planned_unique_identity_count",
        attempt_registry_summary.get("planned_unique_identity_count"),
    )
    actual = nonnegative_int(
        "post_fold_actual_unique_run_count",
        attempt_registry_summary.get("actual_unique_run_count"),
    )
    attempts = nonnegative_int(
        "post_fold_logical_task_count",
        attempt_registry_summary.get("logical_task_count"),
    ) - nonnegative_int(
        "post_fold_cache_hit_count",
        attempt_registry_summary.get("cache_hit_count"),
    )
    if (
        planned > contract.max_unique_identities
        or actual > planned
        or attempts > contract.max_attempts
    ):
        raise FoldReplayError("post_fold_budget_limit_exceeded")
    return {
        "original_mechanism_body_contract": {
            "max_unique_identities": _ORIGINAL_BODY_UNIQUE_LIMIT,
            "max_attempts": _ORIGINAL_BODY_ATTEMPT_LIMIT,
        },
        "active_approved_amended_contract": {
            "contract_version": version,
            "contract_sha256": contract.sha256,
            "max_unique_identities": contract.max_unique_identities,
            "max_attempts": contract.max_attempts,
            "retry_limit": contract.retry_limit,
        },
        "execution_summary": dict(attempt_registry_summary),
        "within_active_approved_contract": True,
        "independent_bo_included": False,
    }


def build_final_development_report(
    *,
    gate_receipt: Mapping[str, Any],
    fold_replay_report: Mapping[str, Any],
    fold_selection_receipt: Mapping[str, Any],
    target_audits_by_fold: Mapping[str, Mapping[str, Any]],
    final_interaction_audit: Mapping[str, Any],
    historical_ab_report: Mapping[str, Any],
    current_role_matrix: Mapping[str, Any],
    budget_contract: Mapping[str, Any],
    attempt_registry_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the five-layer, claim-bounded development replay report."""

    verify_embedded_hash(
        gate_receipt,
        hash_field="receipt_sha256",
        artifact_name="post_fold_gate_receipt",
    )
    summaries, audits = _validate_fold_sources(
        fold_replay_report=fold_replay_report,
        fold_selection_receipt=fold_selection_receipt,
        target_audits_by_fold=target_audits_by_fold,
    )
    _, independent, sample = _validate_final_interaction(final_interaction_audit)
    historical = _validate_historical_report(historical_ab_report)
    current_role = _validate_current_role_matrix(current_role_matrix)
    rows = _comparison_rows(
        summaries=summaries,
        target_audits_by_fold=audits,
        independent_by_record=independent,
        sample_by_record=sample,
        historical_by_record=historical,
        current_role_by_coordinate=current_role,
    )
    if (
        gate_receipt.get("receipt_version") != "lyx_post_fold_independent_bo_gate_v1"
        or gate_receipt.get("fold_replay_report_sha256") != fold_replay_report.get("report_sha256")
        or gate_receipt.get("final_interaction_audit_sha256")
        != final_interaction_audit.get("audit_sha256")
        or gate_receipt.get("historical_ab_report_sha256")
        != historical_ab_report.get("report_sha256")
        or gate_receipt.get("current_role_matrix_sha256")
        != current_role_matrix.get("matrix_sha256")
        or gate_receipt.get("independent_bo_run_count") != 0
    ):
        raise FoldReplayError("post_fold_gate_source_binding_mismatch")
    failed_rows = [row for row in rows if not row["audit_pass"]]
    all_slots_pass = not failed_rows
    freeze_allowed = all_slots_pass and gate_receipt.get("triggered") is False
    status = (
        "awaiting_human_independent_bo_decision"
        if gate_receipt.get("triggered") is True
        else ("complete" if freeze_allowed else "development_cycle_incomplete")
    )
    report = {
        "report_version": "lyx_recovery_filter_final_development_report_v1",
        "status": status,
        "evidence_class": "development_replay_audit",
        "algorithm_level_holdout": False,
        "central_argument": (
            "本轮证据只回答冻结恢复、惩罚、档位库与场景内选择器在四个开发场景"
            "上的可重复工程表现；它不构成未见场景或跨个体泛化证据。"
        ),
        "terminology_ledger": {
            "historical_independent_bo_lite": "历史独立 BO Lite 工程精度锚点",
            "same_identity_recovery_ab": "同一历史参数身份下的恢复机制 A/B",
            "same_role_current_mechanism": "相同档位角色下的当前机制对照",
            "combination_library_sample_in_upper_bound": "冻结组合库样本内上限",
            "scene_shared_profile_replay": "场景共享档位开发内重放",
            "trace_rescue": "历史探索背景方法，不是主要基线",
        },
        "comparison_layer_order": [
            "historical_independent_bo_lite",
            "same_identity_recovery_ab",
            "same_role_current_mechanism",
            "combination_library_sample_in_upper_bound",
            "scene_shared_profile_replay",
        ],
        "record_comparisons": rows,
        "failure_audit": {
            "denominator_slot_count": EXPECTED_LOGICAL_SLOT_COUNT,
            "passed_slot_count": EXPECTED_LOGICAL_SLOT_COUNT - len(failed_rows),
            "failed_slot_count": len(failed_rows),
            "no_safe_shared_candidate_count": sum(
                row["selection_status"] == "no_safe_shared_candidate" for row in rows
            ),
            "failed_slots": [
                {
                    "fold_id": row["fold_id"],
                    "record_id": row["record_id"],
                    "scene": row["scene"],
                    "failure_reasons": row["failure_reasons"],
                    "scene_shared_profile_replay": row["scene_shared_profile_replay"],
                }
                for row in failed_rows
            ],
            "long_tail_and_recovery_fields_retained": [
                "longest_e10_run_windows",
                "longest_e20_run_windows",
                "right_censored_recovery_count",
                "max_rise_underestimate_bpm",
            ],
        },
        "post_fold_independent_bo_gate": {
            "receipt_sha256": gate_receipt["receipt_sha256"],
            "triggered": gate_receipt["triggered"],
            "conditions": gate_receipt["conditions"],
            "independent_bo_authorized": False,
            "independent_bo_run_count": 0,
        },
        "budget_audit": _budget_audit(
            budget_contract=budget_contract,
            attempt_registry_summary=attempt_registry_summary,
        ),
        "trace_rescue_treatment": {
            "role": "historical_exploration_background_only",
            "included_in_primary_baseline_tables": False,
            "included_in_win_counts": False,
            "included_in_conclusive_ranking": False,
        },
        "conclusion": {
            "claim_status": "development_replay_audit_only",
            "candidate_freeze_allowed": freeze_allowed,
            "unseen_record_generalization_passed": False,
            "unseen_scene_generalization_passed": False,
            "cross_person_generalization_passed": False,
            "next_state": (
                "awaiting_human_independent_bo_decision"
                if gate_receipt.get("triggered") is True
                else (
                    "ready_for_challenge_scene_handoff"
                    if freeze_allowed
                    else "development_cycle_incomplete"
                )
            ),
        },
    }
    return _with_hash(report, "report_sha256")


def _format_metric(metrics: Mapping[str, Any] | None, field: str) -> str:
    if metrics is None:
        return "NA"
    value = metrics.get(field)
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_final_development_report_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a concise human-readable companion to the hashed JSON report."""

    verify_embedded_hash(
        report,
        hash_field="report_sha256",
        artifact_name="post_fold_final_report",
    )
    lines = [
        "# LYX 恢复—滤波档位开发内重放报告",
        "",
        f"状态：`{report['status']}`。证据等级：`development_replay_audit`。",
        "",
        str(report["central_argument"]),
        "",
        "## 五层公平比较",
        "",
        "| 记录 | 场景 | 历史独立 BO MAE | 历史参数新恢复 MAE | 同角色当前机制 MAE | 样本内上限 MAE | 场景共享重放 MAE | 重放通过 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for raw in require_list(
        "post_fold_markdown_record_comparisons",
        report.get("record_comparisons"),
    ):
        row = require_mapping("post_fold_markdown_record_row", raw)
        lines.append(
            "| {record} | {scene} | {independent} | {historical_final} | "
            "{current_role} | {sample} | {shared} | {passed} |".format(
                record=row["record_id"],
                scene=row["scene"],
                independent=_format_metric(
                    row["historical_independent_bo_lite"],
                    "final_motion_mae_bpm",
                ),
                historical_final=_format_metric(
                    row["same_identity_recovery_ab"]["final_recovery_mechanism"],
                    "final_motion_mae_bpm",
                ),
                current_role=_format_metric(
                    row["same_role_current_mechanism"],
                    "final_motion_mae_bpm",
                ),
                sample=_format_metric(
                    row["combination_library_sample_in_upper_bound"]["metrics"],
                    "final_motion_mae_bpm",
                ),
                shared=_format_metric(
                    row["scene_shared_profile_replay"],
                    "final_motion_mae_bpm",
                ),
                passed="是" if row["audit_pass"] else "否",
            )
        )
    failure = require_mapping(
        "post_fold_markdown_failure_audit",
        report.get("failure_audit"),
    )
    lines.extend(
        [
            "",
            "## 失败与长尾审计",
            "",
            (
                f"12 个槽位全部保留在分母中：通过 {failure['passed_slot_count']}，"
                f"失败 {failure['failed_slot_count']}，其中 "
                f"`no_safe_shared_candidate` {failure['no_safe_shared_candidate_count']}。"
            ),
        ]
    )
    failed_slots = require_list(
        "post_fold_markdown_failed_slots",
        failure.get("failed_slots"),
    )
    if failed_slots:
        lines.extend(
            [
                "",
                "| 折 | 记录 | 场景 | 失败原因 | L10 | L20 | 右删失恢复 | 上升压制 |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for raw in failed_slots:
            slot = require_mapping("post_fold_markdown_failed_slot", raw)
            metrics = slot.get("scene_shared_profile_replay")
            lines.append(
                "| {fold} | {record} | {scene} | {reasons} | {l10} | {l20} | "
                "{rc} | {rise} |".format(
                    fold=slot["fold_id"],
                    record=slot["record_id"],
                    scene=slot["scene"],
                    reasons=", ".join(slot["failure_reasons"]),
                    l10=_format_metric(metrics, "longest_e10_run_windows"),
                    l20=_format_metric(metrics, "longest_e20_run_windows"),
                    rc=_format_metric(
                        metrics,
                        "right_censored_recovery_count",
                    ),
                    rise=_format_metric(
                        metrics,
                        "max_rise_underestimate_bpm",
                    ),
                )
            )
    gate = require_mapping(
        "post_fold_markdown_gate",
        report.get("post_fold_independent_bo_gate"),
    )
    budget = require_mapping(
        "post_fold_markdown_budget",
        report.get("budget_audit"),
    )
    active = require_mapping(
        "post_fold_markdown_active_budget",
        budget.get("active_approved_amended_contract"),
    )
    lines.extend(
        [
            "",
            "## 人工门与预算",
            "",
            (
                f"折后独立 BO 门触发：{'是' if gate['triggered'] else '否'}。"
                "本报告未授权、未运行任何独立 BO。"
            ),
            (
                f"机制主体原始上限为 {_ORIGINAL_BODY_UNIQUE_LIMIT} 个唯一身份/"
                f"{_ORIGINAL_BODY_ATTEMPT_LIMIT} 次尝试；当前经人工修订的 "
                f"`{active['contract_version']}` 上限为 "
                f"{active['max_unique_identities']}/{active['max_attempts']}。"
            ),
            "",
            "## 解释边界",
            "",
            (
                "TraceRescue 只保留为历史探索背景，不进入主要基线表、胜负计数或"
                "结论性排序。当前数据没有检验未见场景、未见记录或跨个体泛化。"
            ),
            "",
            "## 结论",
            "",
            (
                f"`candidate_freeze_allowed={str(report['conclusion']['candidate_freeze_allowed']).lower()}`；"
                f"下一状态为 `{report['conclusion']['next_state']}`。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_challenge_scene_handoff(
    *,
    final_report: Mapping[str, Any],
    final_interaction_audit: Mapping[str, Any],
    fold_selection_receipt: Mapping[str, Any],
    challenge_scene_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the first unseen-scene validation interface."""

    verify_embedded_hash(
        final_report,
        hash_field="report_sha256",
        artifact_name="challenge_handoff_final_report",
    )
    verify_embedded_hash(
        final_interaction_audit,
        hash_field="audit_sha256",
        artifact_name="challenge_handoff_final_interaction_audit",
    )
    verify_embedded_hash(
        fold_selection_receipt,
        hash_field="receipt_sha256",
        artifact_name="challenge_handoff_selection_receipt",
    )
    verify_embedded_hash(
        challenge_scene_manifest,
        hash_field="manifest_sha256",
        artifact_name="challenge_scene_manifest",
    )
    development = {
        str(value)
        for value in require_list(
            "challenge_development_scene_ids",
            challenge_scene_manifest.get("development_scene_ids"),
        )
    }
    reserved = {
        str(value)
        for value in require_list(
            "challenge_reserved_scene_ids",
            challenge_scene_manifest.get("reserved_challenge_scene_ids"),
        )
    }
    conclusion = require_mapping(
        "challenge_final_report_conclusion",
        final_report.get("conclusion"),
    )
    if (
        final_report.get("status") != "complete"
        or final_report.get("evidence_class") != "development_replay_audit"
        or conclusion.get("candidate_freeze_allowed") is not True
        or conclusion.get("next_state") != "ready_for_challenge_scene_handoff"
        or challenge_scene_manifest.get("manifest_version") != "lyx_challenge_scene_manifest_v1"
        or challenge_scene_manifest.get("status") != "frozen_unseen_scene_plan"
        or development != set(EXPECTED_SCENE_COUNTS)
        or not reserved
        or "bobi" not in reserved
        or development & reserved
        or challenge_scene_manifest.get("challenge_result_read_count") != 0
        or challenge_scene_manifest.get("rule_revision_after_challenge_count") != 0
    ):
        raise FoldReplayError("challenge_scene_handoff_precondition_failed")
    handoff = {
        "handoff_version": "lyx_unseen_scene_challenge_handoff_v1",
        "status": "ready_for_unseen_scene_validation",
        "source_evidence_class": "development_replay_audit",
        "algorithm_level_holdout": False,
        "final_report_sha256": final_report["report_sha256"],
        "final_interaction_audit_sha256": final_interaction_audit["audit_sha256"],
        "fold_selection_receipt_sha256": fold_selection_receipt["receipt_sha256"],
        "challenge_scene_manifest_sha256": challenge_scene_manifest["manifest_sha256"],
        "frozen_algorithm": {
            "final_recovery_id": final_interaction_audit["final_recovery_id"],
            "selected_penalty_id": final_interaction_audit["selected_penalty_id"],
            "profile_receipt_sha256_by_id": {
                str(profile_id): require_mapping(
                    f"challenge_profile_receipt:{profile_id}",
                    receipt,
                )["receipt_sha256"]
                for profile_id, receipt in require_mapping(
                    "challenge_profile_receipts",
                    final_interaction_audit.get("profile_receipts"),
                ).items()
            },
            "selector_receipt_sha256": fold_selection_receipt["receipt_sha256"],
            "candidate_or_threshold_revision_count": 0,
        },
        "challenge_protocol": {
            "reserved_scene_ids": sorted(reserved),
            "bobi_role": "high_dynamic_challenge",
            "additional_scene_policy": challenge_scene_manifest["additional_scene_policy"],
            "challenge_result_read_count_at_freeze": 0,
            "first_use": "whole_pipeline_transfer_acceptance",
            "if_results_change_any_rule": (
                "reclassify_viewed_challenge_scenes_as_development_data_and_"
                "restart_validation_on_new_unseen_scenes"
            ),
            "cross_person_study_order": "after_unseen_scene_validation",
        },
        "claims_not_yet_allowed": [
            "unseen_record_generalization_passed",
            "unseen_scene_generalization_passed",
            "cross_person_generalization_passed",
        ],
        "independent_bo_run_count": 0,
        "next_state": "awaiting_unseen_scene_execution_plan",
    }
    return _with_hash(handoff, "handoff_sha256")


def _load_target_audits(
    *,
    fold_replay_report: Mapping[str, Any],
    fold_output_root: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, str]]]:
    root = fold_output_root.resolve()
    audits: dict[str, Mapping[str, Any]] = {}
    sources: dict[str, dict[str, str]] = {}
    for raw in require_list(
        "post_fold_publish_fold_summaries",
        fold_replay_report.get("folds"),
    ):
        summary = require_mapping("post_fold_publish_fold_summary", raw)
        fold_id = str(summary["fold_id"])
        path = (root / str(summary["target_audit_receipt"])).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FoldReplayError(f"post_fold_target_audit_source_missing:{fold_id}:{path}")
        audits[fold_id] = read_json(path)
        sources[f"target_audit:{fold_id}"] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    return audits, sources


def publish_post_fold_package(
    *,
    fold_replay_report_path: Path,
    fold_selection_receipt_path: Path,
    final_interaction_audit_path: Path,
    historical_ab_report_path: Path,
    current_role_matrix_path: Path,
    review_context_path: Path,
    budget_contract_path: Path,
    exploration_registry_path: Path,
    attempt_registry_path: Path,
    challenge_scene_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Atomically publish the post-fold gate and bounded report package."""

    source_paths = {
        "fold_replay_report": Path(fold_replay_report_path).resolve(),
        "fold_selection_receipt": Path(fold_selection_receipt_path).resolve(),
        "final_interaction_audit": Path(final_interaction_audit_path).resolve(),
        "historical_ab_report": Path(historical_ab_report_path).resolve(),
        "current_role_matrix": Path(current_role_matrix_path).resolve(),
        "review_context": Path(review_context_path).resolve(),
        "budget_contract": Path(budget_contract_path).resolve(),
        "exploration_registry": Path(exploration_registry_path).resolve(),
        "attempt_registry": Path(attempt_registry_path).resolve(),
        "challenge_scene_manifest": Path(challenge_scene_manifest_path).resolve(),
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FoldReplayError(f"post_fold_source_missing:{name}:{path}")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FoldReplayError(f"post_fold_output_already_exists:{destination}")
    fold_report = read_json(source_paths["fold_replay_report"])
    fold_selection = read_json(source_paths["fold_selection_receipt"])
    final_audit = read_json(source_paths["final_interaction_audit"])
    historical = read_json(source_paths["historical_ab_report"])
    current_role = read_json(source_paths["current_role_matrix"])
    review_context = read_json(source_paths["review_context"])
    budget_payload = read_json(source_paths["budget_contract"])
    exploration_payload = read_json(source_paths["exploration_registry"])
    challenge_manifest = read_json(source_paths["challenge_scene_manifest"])
    target_audits, target_sources = _load_target_audits(
        fold_replay_report=fold_report,
        fold_output_root=source_paths["fold_replay_report"].parent,
    )
    budget = budget_contract_from_payload(budget_payload)
    exploration = ExplorationRegistry(
        registry_version=str(exploration_payload["registry_version"]),
        unique_budget=int(exploration_payload["unique_budget"]),
        allowed_identity_sha256=tuple(
            str(value)
            for value in require_list(
                "post_fold_exploration_allowlist",
                exploration_payload.get("allowed_identity_sha256"),
            )
        ),
    )
    registry = AttemptRegistry.open(
        source_paths["attempt_registry"],
        budget_contract=budget,
        exploration_registry=exploration,
    )
    registry_summary = registry.summary()
    gate = evaluate_post_fold_independent_bo_gate(
        fold_replay_report=fold_report,
        fold_selection_receipt=fold_selection,
        target_audits_by_fold=target_audits,
        final_interaction_audit=final_audit,
        historical_ab_report=historical,
        current_role_matrix=current_role,
        review_context=require_mapping(
            "post_fold_review_context",
            review_context,
        ),
    )
    report = build_final_development_report(
        gate_receipt=gate,
        fold_replay_report=fold_report,
        fold_selection_receipt=fold_selection,
        target_audits_by_fold=target_audits,
        final_interaction_audit=final_audit,
        historical_ab_report=historical,
        current_role_matrix=current_role,
        budget_contract=budget_payload,
        attempt_registry_summary=registry_summary,
    )
    handoff = None
    if report["conclusion"]["candidate_freeze_allowed"]:
        handoff = build_challenge_scene_handoff(
            final_report=report,
            final_interaction_audit=final_audit,
            fold_selection_receipt=fold_selection,
            challenge_scene_manifest=challenge_manifest,
        )
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    try:
        atomic_write_json(
            staging / "post_fold_independent_bo_gate_receipt.json",
            gate,
        )
        if gate["triggered"]:
            packet = {
                "packet_version": "lyx_post_fold_independent_bo_review_packet_v1",
                "gate_receipt_sha256": gate["receipt_sha256"],
                "trigger_conditions": gate["conditions"],
                "trigger_records": gate["trigger_records"],
                **dict(gate["review_packet"]),
                "independent_bo_authorized": False,
                "independent_bo_run_count": 0,
            }
            packet["packet_sha256"] = canonical_sha256(packet)
            atomic_write_json(
                staging / "independent_bo_review_packet.json",
                packet,
            )
        atomic_write_json(staging / "final_development_report.json", report)
        (staging / "final_development_report.md").write_text(
            render_final_development_report_markdown(report),
            encoding="utf-8",
        )
        if handoff is not None:
            atomic_write_json(
                staging / "challenge_scene_handoff.json",
                handoff,
            )
        source_artifacts = {
            **{
                name: {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for name, path in source_paths.items()
            },
            **target_sources,
        }
        artifact_files = sorted(path for path in staging.iterdir() if path.is_file())
        artifact_index = {
            "index_version": "lyx_post_fold_artifact_index_v1",
            "source_artifacts": source_artifacts,
            "artifacts": {path.name: file_sha256(path) for path in artifact_files},
        }
        artifact_index["index_sha256"] = canonical_sha256(artifact_index)
        atomic_write_json(staging / "artifact_index.json", artifact_index)
        completion = {
            "completion_version": "lyx_post_fold_completion_v1",
            "status": report["status"],
            "evidence_class": "development_replay_audit",
            "gate_receipt_sha256": gate["receipt_sha256"],
            "final_report_sha256": report["report_sha256"],
            "challenge_handoff_sha256": (None if handoff is None else handoff["handoff_sha256"]),
            "logical_slot_count": EXPECTED_LOGICAL_SLOT_COUNT,
            "denominator_slot_count": EXPECTED_LOGICAL_SLOT_COUNT,
            "planned_unique_identity_count": 0,
            "actual_unique_run_count": 0,
            "independent_bo_authorized": False,
            "independent_bo_run_count": 0,
            "artifact_index_sha256": artifact_index["index_sha256"],
            "next_state": report["conclusion"]["next_state"],
        }
        completion["completion_sha256"] = canonical_sha256(completion)
        atomic_write_json(staging / "post_fold_completion.json", completion)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
        return completion
    except Exception:
        if staging.exists() and staging.parent == destination.parent:
            shutil.rmtree(staging)
        raise
