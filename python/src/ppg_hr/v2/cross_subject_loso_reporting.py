"""Verification and aggregate tables for the final grouped-LOSO report."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean, median, pstdev
from typing import Any


class CompactVerificationError(RuntimeError):
    """A targeted full solve does not reproduce its compact P2 cell."""


def verify_materialized_facts(
    compact: Mapping[str, Any],
    candidate: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    absolute_tolerance: float = 1e-9,
) -> None:
    """Verify every compact performance fact against a targeted full solve."""

    float_pairs = {
        "candidate_mae_bpm": (compact["candidate_mae_bpm"], candidate["mae_bpm"]),
        "g2_margin_s": (compact["g2_margin_s"], gate["g2_margin_s"]),
        "g3_margin_s": (compact["g3_margin_s"], gate["g3_margin_s"]),
        "g4_margin_bpm": (compact["g4_margin_bpm"], gate["g4_margin_bpm"]),
        "g7_margin_s": (compact["g7_margin_s"], gate["g7_margin_s"]),
    }
    for name, (expected, actual) in float_pairs.items():
        if not math.isclose(
            float(expected), float(actual), rel_tol=0.0, abs_tol=absolute_tolerance
        ):
            raise CompactVerificationError(f"metric_mismatch:{name}:{expected}:{actual}")

    candidate_pairs = {
        "candidate_l10": "l10",
        "candidate_l20": "l20",
        "candidate_e10": "e10",
        "candidate_e20": "e20",
        "candidate_right_censored_recovery_count": "right_censored_recovery_count",
        "candidate_full_window_count": "full_window_count",
        "candidate_reliable_window_count": "reliable_window_count",
        "candidate_motion_window_count": "motion_window_count",
        "candidate_evaluation_window_sha256": "evaluation_window_sha256",
    }
    for compact_name, materialized_name in candidate_pairs.items():
        if compact[compact_name] != candidate[materialized_name]:
            raise CompactVerificationError(f"metric_mismatch:{compact_name}")

    gate_names = (
        "g1i_pass",
        "g2_pass",
        "g3_pass",
        "g4_pass",
        "g5_pass",
        "g7_pass",
        "qualified",
        "g5_right_censored_count",
    )
    for name in gate_names:
        if compact[name] != gate[name]:
            raise CompactVerificationError(f"gate_mismatch:{name}")
    expected_failed = tuple(json.loads(str(compact["failed_gates_json"])))
    if expected_failed != tuple(gate["failed_gates"]):
        raise CompactVerificationError("gate_mismatch:failed_gates")


def build_reporting_tables(
    record_rows: Sequence[Mapping[str, Any]],
    fold_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build fold-primary, record-secondary summaries without dropping any row."""

    if not record_rows or not fold_rows:
        raise ValueError("reporting_rows_empty")
    scene_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    scene_folds: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in record_rows:
        scene_records[str(row["scene"])].append(row)
    for row in fold_rows:
        scene_folds[str(row["scene"])].append(row)
    if set(scene_records) != set(scene_folds):
        raise ValueError("record_fold_scene_mismatch")

    scene_rows = [
        _aggregate_scope(scene, scene_records[scene], scene_folds[scene])
        for scene in sorted(scene_folds)
    ]
    overall = _aggregate_scope("overall", list(record_rows), list(fold_rows))
    overall.pop("scope", None)

    gate_summary = []
    for gate_name in (
        "g1i_pass",
        "g2_pass",
        "g3_pass",
        "g4_pass",
        "g5_pass",
        "g7_pass",
        "qualified",
    ):
        passed = sum(_as_bool(row[gate_name]) for row in record_rows)
        gate_summary.append(
            {
                "gate": gate_name,
                "passed_records": passed,
                "total_records": len(record_rows),
                "pass_fraction": passed / len(record_rows),
            }
        )

    frequencies = Counter(str(row["selected_coordinate_id"]) for row in fold_rows)
    coordinate_frequency = [
        {"coordinate_id": coordinate_id, "selected_fold_count": count}
        for coordinate_id, count in sorted(
            frequencies.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "scene_summary": scene_rows,
        "overall": overall,
        "gate_summary": gate_summary,
        "coordinate_frequency": coordinate_frequency,
    }


def _aggregate_scope(
    scope: str,
    record_rows: Sequence[Mapping[str, Any]],
    fold_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_fold = [float(row["candidate_mean_mae_bpm"]) for row in fold_rows]
    baseline_fold = [float(row["baseline_mean_mae_bpm"]) for row in fold_rows]
    candidate_record = [float(row["candidate_mae_bpm"]) for row in record_rows]
    baseline_record = [float(row["baseline_mae_bpm"]) for row in record_rows]
    candidate_primary = mean(candidate_fold)
    baseline_primary = mean(baseline_fold)
    return {
        "scope": scope,
        "fold_count": len(fold_rows),
        "record_count": len(record_rows),
        "primary_mean_of_fold_mean_mae_bpm": candidate_primary,
        "primary_median_fold_mean_mae_bpm": median(candidate_fold),
        "primary_fold_mean_mae_sd_bpm": pstdev(candidate_fold),
        "primary_min_fold_mean_mae_bpm": min(candidate_fold),
        "primary_max_fold_mean_mae_bpm": max(candidate_fold),
        "baseline_mean_of_fold_mean_mae_bpm": baseline_primary,
        "primary_delta_vs_baseline_bpm": candidate_primary - baseline_primary,
        "raw_record_mean_mae_bpm": mean(candidate_record),
        "raw_record_median_mae_bpm": median(candidate_record),
        "raw_record_max_mae_bpm": max(candidate_record),
        "baseline_raw_record_mean_mae_bpm": mean(baseline_record),
        "raw_record_delta_vs_baseline_bpm": mean(candidate_record) - mean(baseline_record),
        "passed_records": sum(int(row["passed_records"]) for row in fold_rows),
        "heldout_records": sum(int(row["heldout_records"]) for row in fold_rows),
        "strict_all_pass_folds": sum(_as_bool(row["strict_all_pass"]) for row in fold_rows),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true"}
