"""Run kaihe2 old-vs-new peak-tracking optimisation comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ppg_hr.v2.optimizer import V2BayesConfig, optimise_v2
from ppg_hr.v2.report import load_v2_report
from ppg_hr.v2.types import V2RunConfig


ROOT = Path(__file__).resolve().parents[1]
BUG_DIR = ROOT / "bug" / "心率算法优化尝试"
BASELINE_REPORT = BUG_DIR / "multi_kaihe2-green-raw_bandpass-lms-full-HF-v2.json"
DATA_PATH = BUG_DIR / "multi_kaihe2.csv"
REF_PATH = BUG_DIR / "multi_kaihe2_HR_ref.csv"
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "kaihe2_peak_tracking_optimization_20260614"


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_v2_report(BASELINE_REPORT)
    new_report_path = output_dir / "multi_kaihe2-green-raw_bandpass-lms-full-HF-v2-optimized.json"
    comparison_path = output_dir / "kaihe2_peak_tracking_comparison.json"

    print("Baseline report:", BASELINE_REPORT)
    print("New report:", new_report_path)
    base_cfg = _base_config_from_report(baseline)

    progress_rows: list[dict[str, Any]] = []

    def on_trial(info: dict[str, Any]) -> None:
        progress_rows.append(info)
        trial_idx = int(info["trial_idx"])
        trial_total = int(info["trial_total"])
        if trial_idx == 1 or trial_idx % 10 == 0 or trial_idx == trial_total:
            print(
                "repeat "
                f"{info['repeat_idx']}/{info['repeat_total']} "
                f"trial {trial_idx}/{trial_total} "
                f"value={float(info['value']):.3f} "
                f"best={float(info['best_overall']):.3f}"
            )

    result = optimise_v2(
        base_cfg,
        V2BayesConfig(),
        out_path=new_report_path,
        on_trial_step=on_trial,
    )
    new_report = load_v2_report(result.report_path)
    comparison = {
        "baseline": _summarise_report(baseline),
        "new_optimized": _summarise_report(new_report),
        "new_report_path": str(result.report_path),
        "trial_count": len(progress_rows),
    }
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print("Comparison saved:", comparison_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the optimized report and comparison JSON.",
    )
    return parser.parse_args()


def _base_config_from_report(payload: dict[str, Any]) -> V2RunConfig:
    transform_params = payload.get("ppg_input_transform_params") or {}
    return V2RunConfig(
        data_path=DATA_PATH,
        ref_path=REF_PATH,
        ppg_mode=str(payload.get("ppg_mode", "green")),
        ppg_input_transform=str(payload.get("ppg_input_transform", "raw_bandpass")),
        ppg_input_baseline_seconds=float(transform_params.get("baseline_seconds", 5.0)),
        analysis_scope=str(payload.get("analysis_scope", "full")),
        adaptive_filter=str(payload.get("adaptive_filter", "lms")),
        reference_groups_order=tuple(payload.get("reference_groups_order", ("HF",))),
    )


def _summarise_report(payload: dict[str, Any]) -> dict[str, Any]:
    window_table = payload.get("window_table") or []
    time_bias = float(payload.get("time_bias", 0.0))
    all_rows = _window_errors(window_table, time_bias, None, None)
    motion_rows = _window_errors(window_table, time_bias, "motion", None)
    focus_rows = _window_errors(window_table, time_bias, None, (80.0, 140.0))
    return {
        "err_stats": payload.get("err_stats", {}),
        "best_params": payload.get("best_params", {}),
        "window_mae_bpm": _mae(all_rows),
        "motion_window_mae_bpm": _mae(motion_rows),
        "focus_80_140s_mae_bpm": _mae(focus_rows),
        "focus_80_140s_max_abs_error_bpm": _max_abs_error(focus_rows),
        "top_focus_error_windows": _top_windows(focus_rows, 10),
    }


def _window_errors(
    window_table: list[dict[str, Any]],
    time_bias: float,
    window_kind: str | None,
    aligned_range: tuple[float, float] | None,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in window_table:
        if window_kind is not None and row.get("window_kind") != window_kind:
            continue
        center = float(row.get("center_s", float("nan")))
        aligned = center + time_bias
        if aligned_range is not None:
            start, end = aligned_range
            if aligned < start or aligned > end:
                continue
        ref = float(row.get("ref_hr_bpm", float("nan")))
        pred = float(row.get("final_hr_bpm", float("nan")))
        if not np.isfinite(ref) or not np.isfinite(pred):
            continue
        rows.append(
            {
                "window_idx": float(row.get("window_idx", -1)),
                "aligned_time_s": aligned,
                "ref_hr_bpm": ref,
                "final_hr_bpm": pred,
                "error_bpm": pred - ref,
                "abs_error_bpm": abs(pred - ref),
            }
        )
    return rows


def _mae(rows: list[dict[str, float]]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([row["abs_error_bpm"] for row in rows]))


def _max_abs_error(rows: list[dict[str, float]]) -> float:
    if not rows:
        return float("nan")
    return float(max(row["abs_error_bpm"] for row in rows))


def _top_windows(rows: list[dict[str, float]], count: int) -> list[dict[str, float]]:
    return [
        {
            "window_idx": int(row["window_idx"]),
            "aligned_time_s": round(row["aligned_time_s"], 3),
            "ref_hr_bpm": round(row["ref_hr_bpm"], 3),
            "final_hr_bpm": round(row["final_hr_bpm"], 3),
            "error_bpm": round(row["error_bpm"], 3),
            "abs_error_bpm": round(row["abs_error_bpm"], 3),
        }
        for row in sorted(rows, key=lambda item: item["abs_error_bpm"], reverse=True)[
            :count
        ]
    ]


if __name__ == "__main__":
    main()
