"""Shared raw-coverage and safe-qualified profile upper-bound model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class ProfileUpperBoundError(RuntimeError):
    """A profile matrix cannot support the frozen upper-bound contract."""


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileUpperBoundError(f"{name}_must_be_object")
    return value


def selection_recovery_delay(metrics: Mapping[str, Any]) -> float:
    """Normalize delay exactly as the frozen Stage R metric contract."""

    raw = metrics.get("max_recovered_delay_s")
    if raw is not None:
        value = float(raw)
    elif int(metrics.get("recovery_episode_count", 0)) == 0:
        value = 0.0
    else:
        value = float(metrics["total_window_count"])
    if not math.isfinite(value) or value < 0.0:
        raise ProfileUpperBoundError("sample_in_upper_bound_recovery_delay_invalid")
    return value


def build_sample_in_upper_bound_payloads(
    *,
    final_profile_rows: Sequence[Mapping[str, Any]],
    scene_by_record: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Build distinct raw-coverage and safe-qualified sample-in bounds."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in final_profile_rows:
        record_id = str(row["record_id"])
        grouped.setdefault(record_id, []).append(row)
    if (
        len(final_profile_rows) != 96
        or set(grouped) != set(scene_by_record)
        or any(len(rows) != 8 for rows in grouped.values())
    ):
        raise ProfileUpperBoundError("sample_in_upper_bound_matrix_incomplete")

    def ranking_key(
        row: Mapping[str, Any],
    ) -> tuple[int, float, float, int, str]:
        metrics = _mapping(
            "sample_in_upper_bound_metrics",
            row.get("metrics"),
        )
        return (
            int(metrics["longest_e10_run_windows"]),
            selection_recovery_delay(metrics),
            float(metrics["final_motion_mae_bpm"]),
            int(row["actual_taps"]),
            str(row["filter_profile_id"]),
        )

    def selected_summary(
        *,
        record_id: str,
        selected: Mapping[str, Any] | None,
        qualified_profile_count: int,
        no_selection_status: str,
    ) -> dict[str, Any]:
        if selected is None:
            return {
                "record_id": record_id,
                "scene": scene_by_record[record_id],
                "status": no_selection_status,
                "selected_profile_id": None,
                "selected_identity_sha256": None,
                "selected_qualified": None,
                "selected_metrics": None,
                "qualified_profile_count": qualified_profile_count,
            }
        qualification = _mapping(
            "sample_in_upper_bound_qualification",
            selected.get("qualification"),
        )
        metrics = _mapping(
            "sample_in_upper_bound_selected_metrics",
            selected.get("metrics"),
        )
        return {
            "record_id": record_id,
            "scene": scene_by_record[record_id],
            "status": "selected",
            "selected_profile_id": selected["filter_profile_id"],
            "selected_identity_sha256": selected["identity_sha256"],
            "selected_qualified": qualification.get("qualified") is True,
            "selected_metrics": {
                "longest_e10_run_windows": int(metrics["longest_e10_run_windows"]),
                "max_recovered_delay_s": (
                    None
                    if metrics.get("max_recovered_delay_s") is None
                    else float(metrics["max_recovered_delay_s"])
                ),
                "selection_recovery_delay_s": selection_recovery_delay(metrics),
                "final_motion_mae_bpm": float(metrics["final_motion_mae_bpm"]),
                "actual_taps": int(selected["actual_taps"]),
            },
            "qualified_profile_count": qualified_profile_count,
        }

    raw_records: list[dict[str, Any]] = []
    safe_records: list[dict[str, Any]] = []
    for record_id in sorted(grouped):
        rows = grouped[record_id]
        qualified = [
            row
            for row in rows
            if _mapping(
                "sample_in_upper_bound_qualification",
                row.get("qualification"),
            ).get("qualified")
            is True
        ]
        raw_records.append(
            selected_summary(
                record_id=record_id,
                selected=min(rows, key=ranking_key),
                qualified_profile_count=len(qualified),
                no_selection_status="unreachable",
            )
        )
        safe_records.append(
            selected_summary(
                record_id=record_id,
                selected=(min(qualified, key=ranking_key) if qualified else None),
                qualified_profile_count=len(qualified),
                no_selection_status="no_safe_profile_for_record",
            )
        )
    return {
        "sample_in_upper_bound": {
            "evidence_class": "diagnostic_sample_in_upper_bound",
            "definition": "raw_coverage_best_across_all_frozen_profiles",
            "algorithm_level_holdout": False,
            "record_count": 12,
            "records": raw_records,
        },
        "safe_qualified_upper_bound": {
            "evidence_class": "diagnostic_safe_qualified_upper_bound",
            "definition": "best_across_engineering_qualified_profiles_only",
            "algorithm_level_holdout": False,
            "record_count": 12,
            "records": safe_records,
        },
    }
