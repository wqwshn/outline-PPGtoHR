"""Fixed-parameter evaluation for the handoff-only dual-reset switch boundary."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .post_motion_reset_fft_reacquire import load_lite_report_config
from .solver import solve_v2
from .types import V2RunConfig

DEFAULT_SAMPLES = (
    "bobi2",
    "kaihe2",
    "kaihe3",
    "tiaosheng3",
    "run1",
    "run2",
    "woli1",
    "woli2",
    "xiezi2",
    "jianpan3",
)
FAILURE_SAMPLES = frozenset({"bobi2", "kaihe2", "kaihe3", "tiaosheng3"})


def count_down_up_bounces(
    values: Sequence[float],
    *,
    jump_bpm: float = 20.0,
    recovery_windows: int = 5,
) -> int:
    """Count severe downward jumps followed by a near-term upward recovery."""

    data = np.asarray(values, dtype=float)
    count = 0
    for idx in range(1, data.size):
        if data[idx] - data[idx - 1] > -float(jump_bpm):
            continue
        stop = min(data.size, idx + int(recovery_windows) + 1)
        if np.any(data[idx + 1 : stop] - data[idx] >= float(jump_bpm)):
            count += 1
    return count


def _tail_metrics(rows: Sequence[dict[str, Any]], motion_end_s: float) -> dict[str, Any]:
    tail = [row for row in rows if motion_end_s < float(row["center_s"]) <= motion_end_s + 60.0]
    errors = np.asarray(
        [float(row["final_hr_bpm"]) - float(row["ref_hr_bpm"]) for row in tail],
        dtype=float,
    )
    final = np.asarray([float(row["final_hr_bpm"]) for row in tail], dtype=float)
    return {
        "mae_bpm": float(np.mean(np.abs(errors))) if errors.size else float("nan"),
        "e10_count": int(np.count_nonzero(np.abs(errors) > 10.0)),
        "e20_count": int(np.count_nonzero(np.abs(errors) > 20.0)),
        "window_count": int(errors.size),
        "max_jump_bpm": (float(np.max(np.abs(np.diff(final)))) if final.size > 1 else 0.0),
        "down_up_bounce_count": count_down_up_bounces(final),
    }


def _independent_reset_invariance(
    baseline_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> dict[str, int]:
    """Compare the independent reset's value and complete causal trace."""

    baseline = {
        int(row["window_idx"]): row
        for row in baseline_rows
        if "independent_reset_bpm" in row
    }
    candidate = {
        int(row["window_idx"]): row
        for row in candidate_rows
        if "independent_reset_bpm" in row
    }
    shared = sorted(set(baseline) & set(candidate))
    return {
        "window_count": len(shared),
        "value_mismatch_count": sum(
            float(baseline[idx]["independent_reset_bpm"])
            != float(candidate[idx]["independent_reset_bpm"])
            for idx in shared
        ),
        "raw_top5_mismatch_count": sum(
            baseline[idx].get("raw_top5") != candidate[idx].get("raw_top5")
            for idx in shared
        ),
        "trace_mismatch_count": sum(
            baseline[idx].get("independent_reset_trace")
            != candidate[idx].get("independent_reset_trace")
            for idx in shared
        ),
    }


def evaluate_report(report_path: str | Path) -> dict[str, Any]:
    """Replay one frozen N5 best point with the experimental switch boundary."""

    path = Path(report_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = load_lite_report_config(payload)
    baseline_config, candidate_config = build_replay_configs(base)
    baseline_result = solve_v2(baseline_config)
    result = solve_v2(candidate_config)
    return _evaluate_results(payload, baseline_result, result)


def build_replay_configs(
    base: V2RunConfig,
) -> tuple[V2RunConfig, V2RunConfig]:
    """Build the shared baseline/candidate pair used by metrics and figures."""

    baseline_config = replace(
        base,
        post_motion_dual_reset_enable=True,
        post_motion_dual_reset_experiment_mode="a0",
        post_motion_dual_reset_handoff_only_switch=False,
    )
    candidate_config = replace(
        base,
        post_motion_dual_reset_enable=True,
        post_motion_dual_reset_experiment_mode="a2",
        post_motion_dual_reset_handoff_only_switch=True,
        post_motion_dual_reset_post_switch_hold_actual_final=True,
        post_motion_dual_reset_gap_rescue_gap_bpm=18.0,
        post_motion_dual_reset_observability_periodicity_min=0.4,
        post_motion_dual_reset_observability_peak_competition_min=1.1,
        post_motion_dual_reset_observability_recovery_hits=2,
    )
    return baseline_config, candidate_config


def _evaluate_results(payload, baseline_result, result) -> dict[str, Any]:
    """Calculate acceptance metrics from one shared replay pair."""

    motion_end_s = float(result.metadata["motion_segment"]["end_s"])
    old = _tail_metrics(baseline_result.window_table, motion_end_s)
    new = _tail_metrics(result.window_table, motion_end_s)
    independent_diff = (
        float(np.max(np.abs(baseline_result.HR[:, 2] - result.HR[:, 2])))
        if baseline_result.HR.shape == result.HR.shape
        else float("nan")
    )
    dual = result.metadata["post_motion_dual_reset"]
    independent = _independent_reset_invariance(
        baseline_result.window_table,
        result.window_table,
    )
    first_consumed = next(
        (row for row in result.window_table if bool(row.get("handoff_consumed"))),
        None,
    )
    sample = Path(str(payload["data_path"])).stem.replace("_HB_0711", "")
    row = {
        "sample": sample,
        "cohort": "failure" if sample in FAILURE_SAMPLES else "normal",
        "old_full_aae_bpm": float(baseline_result.err_stats["final_aae_bpm"]),
        "new_full_aae_bpm": float(result.err_stats["final_aae_bpm"]),
        "old_post60_mae_bpm": float(baseline_result.err_stats["post_motion_60s_mae_bpm"]),
        "new_post60_mae_bpm": float(result.err_stats["post_motion_60s_mae_bpm"]),
        "delta_post60_mae_bpm": float(
            result.err_stats["post_motion_60s_mae_bpm"]
            - baseline_result.err_stats["post_motion_60s_mae_bpm"]
        ),
        "old_post60_e20_count": int(baseline_result.err_stats["post_motion_60s_e20_count"]),
        "new_post60_e20_count": int(result.err_stats["post_motion_60s_e20_count"]),
        "old_post60_e10_count": int(baseline_result.err_stats["post_motion_60s_e10_count"]),
        "new_post60_e10_count": int(result.err_stats["post_motion_60s_e10_count"]),
        "old_post60_max_jump_bpm": old["max_jump_bpm"],
        "new_post60_max_jump_bpm": new["max_jump_bpm"],
        "old_down_up_bounce_count": old["down_up_bounce_count"],
        "new_down_up_bounce_count": new["down_up_bounce_count"],
        "independent_reset_max_abs_diff_bpm": independent_diff,
        "independent_reset_window_count": independent["window_count"],
        "independent_reset_value_mismatch_count": independent[
            "value_mismatch_count"
        ],
        "independent_reset_raw_top5_mismatch_count": independent[
            "raw_top5_mismatch_count"
        ],
        "independent_reset_trace_mismatch_count": independent[
            "trace_mismatch_count"
        ],
        "suppressed_legacy_switch_count": len(dual.get("suppressed_legacy_switch_events", [])),
        "first_handoff_center_s": (
            None if first_consumed is None else float(first_consumed["center_s"])
        ),
        "first_handoff_state": (
            "" if first_consumed is None else str(first_consumed.get("switch_state", ""))
        ),
    }
    invariance_pass = bool(
        row["independent_reset_max_abs_diff_bpm"] == 0.0
        and row["independent_reset_value_mismatch_count"] == 0
        and row["independent_reset_raw_top5_mismatch_count"] == 0
        and row["independent_reset_trace_mismatch_count"] == 0
    )
    row["independent_reset_invariance_pass"] = invariance_pass
    row["non_regression_pass"] = bool(
        row["new_post60_e20_count"] <= row["old_post60_e20_count"]
        and row["delta_post60_mae_bpm"] <= 1.0
        and row["new_down_up_bounce_count"] <= row["old_down_up_bounce_count"]
        and invariance_pass
    )
    if sample == "kaihe3":
        row["acceptance_pass"] = bool(
            first_consumed is None
            and row["delta_post60_mae_bpm"] <= 1.0
            and row["new_post60_e10_count"] <= row["old_post60_e10_count"]
            and row["new_post60_e20_count"] <= row["old_post60_e20_count"]
            and row["new_down_up_bounce_count"] <= row["old_down_up_bounce_count"]
            and invariance_pass
        )
    elif sample in FAILURE_SAMPLES:
        row["acceptance_pass"] = bool(
            row["new_post60_mae_bpm"] <= 3.0
            and row["new_post60_e20_count"] == 0
            and row["new_down_up_bounce_count"] <= row["old_down_up_bounce_count"]
            and invariance_pass
        )
    else:
        row["acceptance_pass"] = row["non_regression_pass"]
    return row


def run_experiment(
    report_dir: str | Path,
    output_dir: str | Path,
    samples: Sequence[str] = DEFAULT_SAMPLES,
) -> list[dict[str, Any]]:
    """Evaluate representative failures and normal sentinels and write artefacts."""

    source = Path(report_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in samples:
        matches = sorted(source.glob(f"{sample}_*-v2.json"))
        if len(matches) != 1:
            raise ValueError(f"{sample}: expected one report, found {len(matches)}")
        rows.append(evaluate_report(matches[0]))

    fieldnames = list(rows[0]) if rows else []
    with (output / "representative_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output / "representative_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    summary = {
        "sample_count": len(rows),
        "failure_count": sum(row["cohort"] == "failure" for row in rows),
        "normal_count": sum(row["cohort"] == "normal" for row in rows),
        "normal_regressions": [
            row["sample"]
            for row in rows
            if row["cohort"] == "normal" and not row["non_regression_pass"]
        ],
        "failed_acceptance": [
            row["sample"] for row in rows if not row["acceptance_pass"]
        ],
    }
    summary["decision"] = (
        "GO" if not summary["failed_acceptance"] else "NO-GO"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--samples", nargs="*", default=list(DEFAULT_SAMPLES))
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
