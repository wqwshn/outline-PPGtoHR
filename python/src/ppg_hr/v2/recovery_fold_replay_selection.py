"""Leakage-safe selection and target audit for LYX fold replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .recovery_contracts import canonical_sha256
from .recovery_fold_replay_contracts import (
    EXPECTED_PROFILE_COUNT,
    FoldReplayError,
    finite_float,
    nonnegative_int,
    require_list,
    require_mapping,
)
from .recovery_profile_upper_bound import (
    ProfileUpperBoundError,
    selection_recovery_delay,
)


def _metric_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = require_mapping(
        "fold_replay_candidate_metrics",
        row.get("metrics"),
    )
    l10 = nonnegative_int(
        "fold_replay_longest_e10_run_windows",
        metrics.get("longest_e10_run_windows"),
    )
    l20 = nonnegative_int(
        "fold_replay_longest_e20_run_windows",
        metrics.get("longest_e20_run_windows"),
    )
    mae = finite_float(
        "fold_replay_final_motion_mae_bpm",
        metrics.get("final_motion_mae_bpm"),
    )
    delay = selection_recovery_delay(metrics)
    right_censored = nonnegative_int(
        "fold_replay_right_censored_recovery_count",
        metrics.get("right_censored_recovery_count"),
    )
    rise = metrics.get("max_rise_underestimate_bpm")
    if rise is not None:
        rise = finite_float("fold_replay_max_rise_underestimate_bpm", rise)
    return {
        "longest_e10_run_windows": l10,
        "longest_e20_run_windows": l20,
        "final_motion_mae_bpm": mae,
        "selection_recovery_delay_s": delay,
        "right_censored_recovery_count": right_censored,
        "max_rise_underestimate_bpm": rise,
    }


def _spectral_traceability(row: Mapping[str, Any]) -> dict[str, Any]:
    audit = require_mapping(
        "fold_replay_spectral_audit",
        row.get("spectral_audit"),
    )
    summary = require_mapping(
        "fold_replay_stage_r_spectral_gate",
        audit.get("stage_r_spectral_gate"),
    )
    valid = nonnegative_int(
        "fold_replay_spectral_valid_window_count",
        summary.get("valid_window_count"),
    )
    invalid = nonnegative_int(
        "fold_replay_spectral_invalid_window_count",
        summary.get("invalid_window_count"),
    )
    return {
        "stability_pass": audit.get("stability_pass") is True,
        "spectral_gate_pass": (
            audit.get("spectral_gate_pass") is True and summary.get("spectral_gate_pass") is True
        ),
        "valid_window_count": valid,
        "invalid_window_count": invalid,
    }


def _row_evidence(
    row: Mapping[str, Any],
    *,
    expected_record_id: str,
    expected_profile_id: str,
) -> dict[str, Any]:
    if (
        row.get("record_id") != expected_record_id
        or row.get("filter_profile_id") != expected_profile_id
    ):
        raise FoldReplayError("fold_replay_training_coordinate_mismatch")
    qualification = require_mapping(
        "fold_replay_training_qualification",
        row.get("qualification"),
    )
    reasons = [
        str(value)
        for value in require_list(
            "fold_replay_training_elimination_reasons",
            qualification.get("elimination_reasons"),
        )
    ]
    independent_delta = finite_float(
        "fold_replay_training_independent_delta_mae_bpm",
        qualification.get("independent_delta_mae_bpm"),
    )
    metrics = _metric_summary(row)
    spectral = _spectral_traceability(row)
    if (
        not spectral["stability_pass"] or not spectral["spectral_gate_pass"]
    ) and "spectral_gate_contract_v1" not in reasons:
        reasons.append("spectral_gate_contract_v1")
    actual_taps = nonnegative_int(
        "fold_replay_actual_taps",
        row.get("actual_taps"),
    )
    if actual_taps <= 0:
        raise FoldReplayError("fold_replay_actual_taps_must_be_positive")
    return {
        "record_id": expected_record_id,
        "identity_sha256": row["identity_sha256"],
        "qualified": qualification.get("qualified") is True and not reasons,
        "elimination_reasons": reasons,
        "independent_delta_mae_bpm": independent_delta,
        "metrics": metrics,
        "spectral_traceability": spectral,
        "actual_taps": actual_taps,
    }


def _candidate_evidence(
    *,
    profile_id: str,
    training_record_ids: Sequence[str],
    rows_by_record: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    record_evidence: list[dict[str, Any]] = []
    metric_errors: list[str] = []
    for record_id in training_record_ids:
        try:
            row = rows_by_record[record_id][profile_id]
            record_evidence.append(
                _row_evidence(
                    row,
                    expected_record_id=record_id,
                    expected_profile_id=profile_id,
                )
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            FoldReplayError,
            ProfileUpperBoundError,
        ) as error:
            metric_errors.append(f"{record_id}:{error}")
    elimination_reasons: list[str] = []
    if metric_errors:
        elimination_reasons.append("metric_or_mask_contract_failure")
    for record in record_evidence:
        elimination_reasons.extend(
            f"{record['record_id']}:{reason}" for reason in record["elimination_reasons"]
        )
        if not record["qualified"] and not record["elimination_reasons"]:
            elimination_reasons.append(f"{record['record_id']}:qualification_contract_inconsistent")
    mean_delta = (
        sum(float(record["independent_delta_mae_bpm"]) for record in record_evidence)
        / len(record_evidence)
        if len(record_evidence) == len(training_record_ids)
        else None
    )
    if mean_delta is not None and mean_delta > 1.0:
        elimination_reasons.append("training_pair_mean_independent_mae_gate")
    qualified = not elimination_reasons and len(record_evidence) == len(training_record_ids)
    ranking_key: list[Any] | None = None
    if qualified:
        taps = {int(record["actual_taps"]) for record in record_evidence}
        if len(taps) != 1:
            raise FoldReplayError("fold_replay_profile_taps_inconsistent")
        ranking_key = [
            max(int(record["metrics"]["longest_e10_run_windows"]) for record in record_evidence),
            max(
                float(record["metrics"]["selection_recovery_delay_s"]) for record in record_evidence
            ),
            max(float(record["metrics"]["final_motion_mae_bpm"]) for record in record_evidence),
            sum(float(record["metrics"]["final_motion_mae_bpm"]) for record in record_evidence)
            / len(record_evidence),
            sum(
                int(record["spectral_traceability"]["invalid_window_count"])
                for record in record_evidence
            ),
            -sum(
                int(record["spectral_traceability"]["valid_window_count"])
                for record in record_evidence
            ),
            taps.pop(),
            profile_id,
        ]
    return {
        "filter_profile_id": profile_id,
        "qualified": qualified,
        "elimination_reasons": sorted(set(elimination_reasons)),
        "metric_contract_errors": metric_errors,
        "training_pair_mean_independent_delta_mae_bpm": mean_delta,
        "record_evidence": record_evidence,
        "ranking_key": ranking_key,
    }


def select_fold_profile(
    *,
    fold_id: str,
    scene: str,
    training_record_payloads: Sequence[Mapping[str, Any]],
    audit_target_record_id: str,
    profile_ids: Sequence[str],
) -> dict[str, Any]:
    """Select from sanitized training payloads without target performance."""

    if len(training_record_payloads) != 2:
        raise FoldReplayError("fold_replay_requires_two_training_records")
    if len(profile_ids) != EXPECTED_PROFILE_COUNT or len(set(profile_ids)) != len(profile_ids):
        raise FoldReplayError("fold_replay_profile_set_mismatch")
    rows_by_record: dict[str, dict[str, Mapping[str, Any]]] = {}
    for raw_payload in training_record_payloads:
        payload = require_mapping(
            "fold_replay_training_record_payload",
            raw_payload,
        )
        record_id = str(payload.get("record_id", ""))
        if (
            not record_id
            or record_id == audit_target_record_id
            or payload.get("scene") != scene
            or record_id in rows_by_record
        ):
            raise FoldReplayError("fold_replay_training_role_mismatch")
        profile_rows = [
            require_mapping("fold_replay_training_profile_row", raw)
            for raw in require_list(
                "fold_replay_training_profile_rows",
                payload.get("profile_rows"),
            )
        ]
        indexed = {str(row.get("filter_profile_id", "")): row for row in profile_rows}
        if len(profile_rows) != EXPECTED_PROFILE_COUNT or set(indexed) != set(profile_ids):
            raise FoldReplayError("fold_replay_training_profile_matrix_mismatch")
        rows_by_record[record_id] = indexed
    training_record_ids = tuple(sorted(rows_by_record))
    candidates = [
        _candidate_evidence(
            profile_id=profile_id,
            training_record_ids=training_record_ids,
            rows_by_record=rows_by_record,
        )
        for profile_id in sorted(profile_ids)
    ]
    qualified = [candidate for candidate in candidates if candidate["qualified"] is True]
    qualified.sort(key=lambda candidate: tuple(candidate["ranking_key"]))
    selected = qualified[0] if qualified else None
    receipt = {
        "receipt_version": "lyx_fold_selection_receipt_v1",
        "fold_id": fold_id,
        "scene": scene,
        "training_record_ids": list(training_record_ids),
        "audit_target_record_id": audit_target_record_id,
        "evidence_class": "development_replay_audit",
        "algorithm_level_holdout": False,
        "status": ("selected" if selected is not None else "no_safe_shared_candidate"),
        "selected_filter_profile_id": (
            selected["filter_profile_id"] if selected is not None else None
        ),
        "selected_ranking_key": (selected["ranking_key"] if selected is not None else None),
        "candidate_count": len(candidates),
        "qualified_candidate_count": len(qualified),
        "candidate_elimination_chain": candidates,
        "target_performance_read_count_before_freeze": 0,
        "candidate_or_threshold_revision_count": 0,
    }
    receipt["selection_sha256"] = canonical_sha256(receipt)
    return receipt


def audit_selected_target(
    *,
    selection_receipt: Mapping[str, Any],
    target_result_payload: Mapping[str, Any],
    expected_identity_sha256: str,
) -> dict[str, Any]:
    """Audit exactly one selected target result after selection is frozen."""

    if selection_receipt.get("status") != "selected":
        raise FoldReplayError("fold_replay_target_audit_requires_selection")
    selected_profile_id = str(selection_receipt["selected_filter_profile_id"])
    record_id = str(selection_receipt["audit_target_record_id"])
    payload = require_mapping(
        "fold_replay_target_result_payload",
        target_result_payload,
    )
    if payload.get("record_id") != record_id:
        raise FoldReplayError("fold_replay_target_result_record_mismatch")
    row = require_mapping(
        "fold_replay_selected_target_row",
        payload.get("selected_row"),
    )
    if (
        row.get("record_id") != record_id
        or row.get("filter_profile_id") != selected_profile_id
        or row.get("identity_sha256") != expected_identity_sha256
    ):
        return {
            "status": "identity_mismatch_requires_supplement",
            "audit_pass": False,
            "failure_reasons": ["identity_mismatch_requires_supplement"],
            "selected_filter_profile_id": selected_profile_id,
            "audit_target_record_id": record_id,
            "expected_identity_sha256": expected_identity_sha256,
            "observed_identity_sha256": row.get("identity_sha256"),
            "metrics": None,
            "qualification": None,
        }
    try:
        qualification = require_mapping(
            "fold_replay_target_qualification",
            row.get("qualification"),
        )
        reasons = [
            str(value)
            for value in require_list(
                "fold_replay_target_elimination_reasons",
                qualification.get("elimination_reasons"),
            )
        ]
        metrics = _metric_summary(row)
        spectral = _spectral_traceability(row)
        if (
            not spectral["stability_pass"] or not spectral["spectral_gate_pass"]
        ) and "spectral_gate_contract_v1" not in reasons:
            reasons.append("spectral_gate_contract_v1")
    except (
        KeyError,
        TypeError,
        ValueError,
        FoldReplayError,
        ProfileUpperBoundError,
    ):
        return {
            "status": "failed",
            "audit_pass": False,
            "failure_reasons": ["metric_or_mask_contract_failure"],
            "selected_filter_profile_id": selected_profile_id,
            "audit_target_record_id": record_id,
            "expected_identity_sha256": expected_identity_sha256,
            "observed_identity_sha256": row.get("identity_sha256"),
            "metrics": None,
            "qualification": None,
        }
    if qualification.get("qualified") is not True and not reasons:
        reasons.append("metric_or_mask_contract_failure")
    reasons = sorted(set(reasons))
    return {
        "status": "passed" if not reasons else "failed",
        "audit_pass": not reasons,
        "failure_reasons": reasons,
        "selected_filter_profile_id": selected_profile_id,
        "audit_target_record_id": record_id,
        "expected_identity_sha256": expected_identity_sha256,
        "observed_identity_sha256": row["identity_sha256"],
        "metrics": metrics,
        "qualification": dict(qualification),
    }
