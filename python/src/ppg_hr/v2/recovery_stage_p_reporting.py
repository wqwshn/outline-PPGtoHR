"""Mechanical Stage P penalty selection and interaction reporting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .recovery_contracts import canonical_sha256
from .recovery_stage_f_reporting import _qualification
from .recovery_stage_p_contracts import (
    EXPECTED_LOGICAL_RESULT_COUNT,
    EXPECTED_PENALTY_IDS,
    EXPECTED_SELECTION_RANKING_KEY,
    StagePPlanError,
    require_list,
    require_mapping,
)


def _finite_metric(
    metrics: Mapping[str, Any],
    name: str,
) -> float:
    value = metrics.get(name)
    if value is None:
        return math.inf
    number = float(value)
    return number if math.isfinite(number) else math.inf


def build_penalty_interaction_report(
    *,
    proposal: Mapping[str, Any],
    current_rows: Sequence[Mapping[str, Any]],
    current_role_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine all three 8×12 matrices and freeze the penalty winner."""

    penalties = {
        str(candidate["penalty_id"]): candidate
        for candidate in (
            require_mapping("stage_p_penalty", raw)
            for raw in require_list(
                "stage_p_penalties",
                proposal.get("penalties"),
            )
        )
    }
    if (
        set(penalties) != EXPECTED_PENALTY_IDS
        or proposal.get("selection_ranking_key") != EXPECTED_SELECTION_RANKING_KEY
    ):
        raise StagePPlanError("stage_p_reporting_contract_mismatch")
    current_penalty_by_coordinate = {
        (
            str(row["filter_profile_id"]),
            str(row["record_id"]),
        ): dict(row)
        for row in current_rows
    }
    current_role_by_coordinate = {
        (
            str(row["filter_profile_id"]),
            str(row["record_id"]),
        ): dict(row)
        for row in current_role_rows
    }
    if len(current_penalty_by_coordinate) != 96 or set(current_role_by_coordinate) != set(
        current_penalty_by_coordinate
    ):
        raise StagePPlanError("stage_p_current_matrix_incomplete")
    independent_by_record = {
        str(record["record_id"]): require_mapping(
            "stage_p_independent_metrics",
            require_mapping("stage_p_record", record).get("independent_metrics"),
        )
        for record in require_list(
            "stage_p_record_panel",
            proposal.get("record_panel"),
        )
    }
    true_rise_by_record = {
        str(record["record_id"]): bool(record["true_rise_applicable"])
        for record in proposal["record_panel"]
    }
    combined = [dict(row) for row in current_rows]
    for raw in new_rows:
        row = dict(raw)
        coordinate = (
            str(row["filter_profile_id"]),
            str(row["record_id"]),
        )
        current = current_role_by_coordinate.get(coordinate)
        if current is None:
            raise StagePPlanError("stage_p_new_matrix_coordinate_unknown")
        row["qualification"] = _qualification(
            candidate=row,
            current=current,
            independent=independent_by_record[coordinate[1]],
            true_rise_applicable=true_rise_by_record[coordinate[1]],
        )
        combined.append(row)
    seen = {
        (
            str(row["penalty_candidate_id"]),
            str(row["filter_profile_id"]),
            str(row["record_id"]),
        )
        for row in combined
    }
    if (
        len(combined) != EXPECTED_LOGICAL_RESULT_COUNT
        or len(seen) != EXPECTED_LOGICAL_RESULT_COUNT
        or {coordinate[0] for coordinate in seen} != EXPECTED_PENALTY_IDS
        or any(
            require_mapping(
                "stage_p_row_qualification",
                row.get("qualification"),
            ).get("qualified")
            not in {True, False}
            for row in combined
        )
    ):
        raise StagePPlanError("stage_p_combined_matrix_mismatch")
    score_rows: list[dict[str, Any]] = []
    for penalty_id in sorted(penalties):
        rows = [row for row in combined if row["penalty_candidate_id"] == penalty_id]
        metrics = [require_mapping("stage_p_row_metrics", row.get("metrics")) for row in rows]
        hard_gate_failure_count = sum(not bool(row["qualification"]["qualified"]) for row in rows)
        hard_gate_violation_count = sum(
            len(row["qualification"]["elimination_reasons"]) for row in rows
        )
        maes = [_finite_metric(metric, "final_motion_mae_bpm") for metric in metrics]
        ranking_values: list[object] = [
            hard_gate_failure_count,
            sum(int(metric["right_censored_recovery_count"]) for metric in metrics),
            max(int(metric["longest_e10_run_windows"]) for metric in metrics),
            max(maes),
            sum(maes) / len(maes),
            int(penalties[penalty_id]["mechanism_complexity"]),
            penalty_id,
        ]
        score_rows.append(
            {
                "penalty_id": penalty_id,
                "hard_gate_failure_count": hard_gate_failure_count,
                "hard_gate_violation_count": hard_gate_violation_count,
                "right_censored_recovery_count": ranking_values[1],
                "worst_l10": ranking_values[2],
                "worst_mae": ranking_values[3],
                "mean_mae": ranking_values[4],
                "mechanism_complexity": ranking_values[5],
                "ranking_key": ranking_values,
            }
        )
    score_rows.sort(key=lambda row: tuple(row["ranking_key"]))
    report = {
        "report_version": "lyx_stage_p_interaction_report_v1",
        "status": "selected",
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "proposal_sha256": proposal["proposal_sha256"],
        "logical_result_count": len(combined),
        "reused_stage_f_result_count": 96,
        "new_formal_result_count": 192,
        "penalty_selection_rule": EXPECTED_SELECTION_RANKING_KEY,
        "selected_penalty_id": score_rows[0]["penalty_id"],
        "penalty_scores": score_rows,
        "rows": combined,
        "rollback_backup_id": proposal["rollback_backup_id"],
        "independent_bo_run_count": 0,
        "next_state": "ready_for_rollback_backup_proposal",
    }
    report["report_sha256"] = canonical_sha256(report)
    return report
