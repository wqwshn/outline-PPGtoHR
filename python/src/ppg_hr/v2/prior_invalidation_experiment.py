"""Evaluate causal directional invalidation of a stale handoff prior."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from .handoff_only_switch_experiment import (
    FAILURE_SAMPLES,
    _independent_reset_invariance,
    build_replay_configs,
    count_down_up_bounces,
)
from .post_motion_reset_fft_reacquire import load_lite_report_config
from .solver import solve_v2
from .types import V2RunConfig

TYPICAL_SAMPLES = (
    "run2",
    "xiezi2",
    "run1",
    "kaihe2",
    "kaihe3",
    "tiaosheng3",
    "bobi2",
)


def build_prior_invalidation_configs(
    base: V2RunConfig,
) -> tuple[V2RunConfig, V2RunConfig]:
    """Return the current safe handoff and one-variable invalidation candidate."""

    _, current = build_replay_configs(base)
    candidate = replace(
        current,
        post_motion_dual_reset_prior_invalidation_enable=True,
        post_motion_dual_reset_prior_invalidation_hits=3,
        post_motion_dual_reset_prior_invalidation_gap_bpm=40.0,
        post_motion_dual_reset_prior_invalidation_decline_bpm=0.5,
    )
    return current, candidate


def aligned_reference_bpm(hr: np.ndarray, time_bias: float) -> np.ndarray:
    """Return the reference paired with each algorithm window by MAE timing."""

    values = np.asarray(hr, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float)
    time = values[:, 0]
    aligned_time = time + float(time_bias)
    interpolate = interp1d(
        time,
        values[:, 1],
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=False,
    )
    reference = np.asarray(interpolate(aligned_time), dtype=float)
    reference[(aligned_time < np.min(time)) | (aligned_time > np.max(time))] = np.nan
    return reference


def evaluate_report(report_path: str | Path) -> dict[str, Any]:
    """Replay archived, current, and directional-invalidation mechanisms."""

    path = Path(report_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = load_lite_report_config(payload)
    archived_config, _ = build_replay_configs(base)
    current_config, candidate_config = build_prior_invalidation_configs(base)
    archived = solve_v2(archived_config)
    current = solve_v2(current_config)
    candidate = solve_v2(candidate_config)
    return _evaluate_results(payload, archived, current, candidate)


def _first_handoff(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    return next((row for row in rows if bool(row.get("handoff_consumed"))), None)


def _bounce_count(result) -> int:
    motion_end = float(result.metadata["motion_segment"]["end_s"])
    values = [
        float(row["final_hr_bpm"])
        for row in result.window_table
        if motion_end < float(row["center_s"]) <= motion_end + 60.0
    ]
    return count_down_up_bounces(values)


def _evaluate_results(payload, archived, current, candidate) -> dict[str, Any]:
    sample = Path(str(payload["data_path"])).stem.split("_")[0]
    current_first = _first_handoff(current.window_table)
    new_first = _first_handoff(candidate.window_table)
    events = [
        row
        for row in candidate.window_table
        if bool((row.get("handoff_trace") or {}).get("prior_invalidation_event"))
    ]
    independent = _independent_reset_invariance(
        current.window_table,
        candidate.window_table,
    )
    independent_diff = (
        float(np.max(np.abs(current.HR[:, 2] - candidate.HR[:, 2])))
        if current.HR.shape == candidate.HR.shape
        else float("nan")
    )
    old_mae = float(archived.err_stats["post_motion_60s_mae_bpm"])
    current_mae = float(current.err_stats["post_motion_60s_mae_bpm"])
    new_mae = float(candidate.err_stats["post_motion_60s_mae_bpm"])
    old_e20 = int(archived.err_stats["post_motion_60s_e20_count"])
    new_e20 = int(candidate.err_stats["post_motion_60s_e20_count"])
    row = {
        "sample": sample,
        "cohort": "failure" if sample in FAILURE_SAMPLES else "normal",
        "old_post60_mae_bpm": old_mae,
        "current_post60_mae_bpm": current_mae,
        "new_post60_mae_bpm": new_mae,
        "delta_vs_old_post60_mae_bpm": new_mae - old_mae,
        "delta_vs_current_post60_mae_bpm": new_mae - current_mae,
        "old_post60_e10_count": int(archived.err_stats["post_motion_60s_e10_count"]),
        "current_post60_e10_count": int(current.err_stats["post_motion_60s_e10_count"]),
        "new_post60_e10_count": int(candidate.err_stats["post_motion_60s_e10_count"]),
        "old_post60_e20_count": old_e20,
        "current_post60_e20_count": int(current.err_stats["post_motion_60s_e20_count"]),
        "new_post60_e20_count": new_e20,
        "current_first_handoff_center_s": (
            None if current_first is None else float(current_first["center_s"])
        ),
        "new_first_handoff_center_s": (
            None if new_first is None else float(new_first["center_s"])
        ),
        "current_first_handoff_state": (
            "" if current_first is None else str(current_first.get("switch_state", ""))
        ),
        "new_first_handoff_state": (
            "" if new_first is None else str(new_first.get("switch_state", ""))
        ),
        "prior_invalidation_event_count": len(events),
        "first_prior_invalidation_center_s": (
            None if not events else float(events[0]["center_s"])
        ),
        "old_down_up_bounce_count": _bounce_count(archived),
        "new_down_up_bounce_count": _bounce_count(candidate),
        "independent_reset_max_abs_diff_bpm": independent_diff,
        "independent_reset_value_mismatch_count": independent["value_mismatch_count"],
        "independent_reset_raw_top5_mismatch_count": independent[
            "raw_top5_mismatch_count"
        ],
        "independent_reset_trace_mismatch_count": independent["trace_mismatch_count"],
    }
    invariance = bool(
        independent_diff == 0.0
        and independent["value_mismatch_count"] == 0
        and independent["raw_top5_mismatch_count"] == 0
        and independent["trace_mismatch_count"] == 0
    )
    row["independent_reset_invariance_pass"] = invariance
    row["non_regression_pass"] = bool(
        new_mae - old_mae <= 1.0
        and new_e20 <= old_e20
        and row["new_down_up_bounce_count"] <= row["old_down_up_bounce_count"]
        and invariance
    )
    if sample == "kaihe3":
        row["acceptance_pass"] = bool(
            new_first is None
            and new_mae - old_mae <= 1.0
            and new_e20 <= old_e20
            and invariance
        )
    elif sample in FAILURE_SAMPLES:
        row["acceptance_pass"] = bool(
            new_mae <= 3.0
            and new_e20 == 0
            and row["new_down_up_bounce_count"] <= row["old_down_up_bounce_count"]
            and invariance
        )
    else:
        row["acceptance_pass"] = row["non_regression_pass"]
    return row


def run_experiment(
    report_dir: str | Path,
    output_dir: str | Path,
    samples: Sequence[str],
) -> list[dict[str, Any]]:
    source = Path(report_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        matches = sorted(source.glob(f"{sample}_*-v2.json"))
        if len(matches) != 1:
            raise ValueError(f"{sample}: expected one report, found {len(matches)}")
        rows.append(evaluate_report(matches[0]))
    fieldnames = list(rows[0]) if rows else []
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output / "metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    summary = {
        "sample_count": len(rows),
        "prior_invalidation_samples": [
            row["sample"] for row in rows if row["prior_invalidation_event_count"]
        ],
        "failed_acceptance": [
            row["sample"] for row in rows if not row["acceptance_pass"]
        ],
    }
    summary["decision"] = "GO" if not summary["failed_acceptance"] else "NO-GO"
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--samples", nargs="*", default=list(TYPICAL_SAMPLES))
    parser.add_argument("--all-reports", action="store_true")
    args = parser.parse_args()
    samples = (
        [path.name.split("_", 1)[0] for path in sorted(args.report_dir.glob("*-v2.json"))]
        if args.all_reports
        else args.samples
    )
    rows = run_experiment(args.report_dir, args.output_dir, samples)
    print(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
