"""Window-level spectral analysis for LMS/KLMS gate factorial results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from .lms_klms_gate_factorial import scenario_for_sample
from .output_paths import prepare_output_dir, safe_output_path

TRUE_PEAK_TOLERANCE_BPM = 5.0


@dataclass(frozen=True)
class SpectralAnalysisResult:
    output_dir: Path
    window_csv: Path
    sample_summary_csv: Path
    scenario_summary_csv: Path
    report_md: Path


def analyze_result_root(
    result_root: Path | str,
    *,
    output_dir: Path | str | None = None,
    sample_ids: Sequence[str] = (),
    scenarios: Sequence[str] = (),
) -> SpectralAnalysisResult:
    root = Path(result_root)
    out = Path(output_dir) if output_dir is not None else root / "analysis"
    prepare_output_dir(out)
    windows = list(_iter_window_metrics(root, sample_ids=sample_ids, scenarios=scenarios))
    window_csv = _write_rows(out, "motion_window_metrics.csv", windows)
    sample_rows = _sample_summary_rows(windows)
    scenario_rows = _scenario_summary_rows(windows)
    sample_csv = _write_rows(out, "sample_summary.csv", sample_rows)
    scenario_csv = _write_rows(out, "scenario_summary.csv", scenario_rows)
    report_md = _write_markdown_report(out, sample_rows, scenario_rows, windows)
    return SpectralAnalysisResult(
        output_dir=out,
        window_csv=window_csv,
        sample_summary_csv=sample_csv,
        scenario_summary_csv=scenario_csv,
        report_md=report_md,
    )


def window_metrics_from_row(
    *,
    row: dict[str, Any],
    sample_id: str,
    scenario: str,
    condition: str,
    adaptive_filter: str,
) -> dict[str, Any]:
    trace = row.get("spectrum_tracking") or {}
    ref_bpm = _first_float(row.get("ref_hr_bpm"), trace.get("ref_hr_bpm"))
    final_bpm = _first_float(row.get("final_hr_bpm"), trace.get("final_hr_bpm"))
    previous_bpm = _first_float(trace.get("previous_hr_bpm"))
    search_min_bpm = _first_float(trace.get("search_min_bpm"))
    search_max_bpm = _first_float(trace.get("search_max_bpm"))
    tracked_bpm = _first_float(trace.get("tracked_hr_bpm"))
    selected_rank = _as_int(trace.get("selected_peak_rank"))
    candidates = _float_list(
        trace.get("unpenalized_candidate_peaks_bpm") or trace.get("candidate_peaks_bpm")
    )
    amplitudes = _float_list(
        trace.get("unpenalized_candidate_peak_amplitudes")
        or trace.get("candidate_peak_amplitudes")
    )
    true_peak = _nearest_true_peak(candidates, amplitudes, ref_bpm)
    visible = true_peak is not None
    true_peak_bpm = true_peak["bpm"] if true_peak else None
    true_peak_distance = true_peak["distance_bpm"] if true_peak else None
    true_peak_rank = true_peak["rank"] if true_peak else None
    true_peak_amp_ratio = true_peak["amp_ratio"] if true_peak else None
    range_reachable = bool(
        visible
        and true_peak_bpm is not None
        and search_min_bpm is not None
        and search_max_bpm is not None
        and search_min_bpm <= true_peak_bpm <= search_max_bpm
    )
    output_reached = bool(
        ref_bpm is not None
        and final_bpm is not None
        and abs(float(final_bpm) - float(ref_bpm)) <= TRUE_PEAK_TOLERANCE_BPM
    )
    selected_true_peak = bool(
        visible
        and selected_rank is not None
        and true_peak_rank is not None
        and int(selected_rank) == int(true_peak_rank)
    )
    tracked_near_ref = bool(
        ref_bpm is not None
        and tracked_bpm is not None
        and abs(float(tracked_bpm) - float(ref_bpm)) <= TRUE_PEAK_TOLERANCE_BPM
    )
    primary = _primary_failure_reason(
        output_reached=output_reached,
        visible=visible,
        range_reachable=range_reachable,
        selected_true_peak=selected_true_peak,
        tracked_near_ref=tracked_near_ref,
        reacquire_triggered=bool(trace.get("reacquire_triggered")),
        high_lock_triggered=bool(trace.get("high_lock_triggered")),
    )
    penalty_centers = _float_list(trace.get("penalty_centers_bpm"))
    hf_stage = _best_hf_stage(row.get("adaptive_stages") or [])
    search_center_bpm = (
        (search_min_bpm + search_max_bpm) / 2.0
        if search_min_bpm is not None and search_max_bpm is not None
        else None
    )
    return {
        "condition": condition,
        "sample": sample_id,
        "scenario": scenario,
        "adaptive_filter": adaptive_filter,
        "window_idx": _as_int(row.get("window_idx")),
        "center_s": _first_float(row.get("center_s")),
        "window_kind": row.get("window_kind"),
        "window_stage": row.get("window_stage"),
        "ref_bpm": ref_bpm,
        "final_bpm": final_bpm,
        "abs_error_bpm": _abs_diff(final_bpm, ref_bpm),
        "output_reached": output_reached,
        "true_peak_visible": visible,
        "true_peak_bpm": true_peak_bpm,
        "true_peak_distance_bpm": true_peak_distance,
        "true_peak_rank": true_peak_rank,
        "true_peak_amp_ratio": true_peak_amp_ratio,
        "main_peak_error_bpm": _abs_diff(candidates[0] if candidates else None, ref_bpm),
        "candidate_count": len(candidates),
        "range_reachable": range_reachable,
        "ref_inside_search_range": bool(
            ref_bpm is not None
            and search_min_bpm is not None
            and search_max_bpm is not None
            and search_min_bpm <= ref_bpm <= search_max_bpm
        ),
        "previous_hr_bpm": previous_bpm,
        "previous_error_bpm": _signed_diff(previous_bpm, ref_bpm),
        "search_min_bpm": search_min_bpm,
        "search_max_bpm": search_max_bpm,
        "search_center_error_bpm": _signed_diff(search_center_bpm, ref_bpm),
        "tracked_hr_bpm": tracked_bpm,
        "selected_peak_rank": selected_rank,
        "selected_true_peak": selected_true_peak,
        "reacquire_mode": trace.get("reacquire_mode"),
        "reacquire_triggered": bool(trace.get("reacquire_triggered")),
        "reacquire_candidate_bpm": trace.get("reacquire_candidate_bpm"),
        "reacquire_reason": trace.get("reacquire_reason"),
        "reacquire_candidate_rejected_reason": trace.get(
            "reacquire_candidate_rejected_reason"
        ),
        "reacquire_action": trace.get("reacquire_action"),
        "high_lock_mode": trace.get("high_lock_mode"),
        "high_lock_triggered": bool(trace.get("high_lock_triggered")),
        "high_lock_reason": trace.get("high_lock_reason"),
        "penalty_centers_bpm": ";".join(f"{value:.3f}" for value in penalty_centers),
        "nearest_penalty_to_ref_bpm": _nearest_distance(penalty_centers, ref_bpm),
        "penalty_confidence": _first_float(trace.get("penalty_confidence")),
        "hf_best_corr": hf_stage.get("corr"),
        "hf_best_delay_samples": hf_stage.get("delay_samples"),
        "hf_best_channel": hf_stage.get("channel"),
        "hf_best_M": hf_stage.get("M"),
        "hf_best_K": hf_stage.get("K"),
        "primary_failure_reason": primary,
    }


def _iter_window_metrics(
    root: Path,
    *,
    sample_ids: Sequence[str],
    scenarios: Sequence[str],
) -> Iterable[dict[str, Any]]:
    wanted_samples = {str(value) for value in sample_ids}
    wanted_scenarios = {str(value) for value in scenarios}
    for report_path in sorted(root.glob("*/json/*-v2.json")):
        condition = report_path.parent.parent.name
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        sample_id = Path(str(payload.get("data_path", report_path.stem))).stem
        scenario = scenario_for_sample(sample_id) or "unknown"
        if wanted_samples and sample_id not in wanted_samples:
            continue
        if wanted_scenarios and scenario not in wanted_scenarios:
            continue
        adaptive_filter = str(payload.get("adaptive_filter", ""))
        for row in payload.get("window_table", []):
            if not (bool(row.get("is_motion")) and bool(row.get("used_adaptive"))):
                continue
            metrics = window_metrics_from_row(
                row=row,
                sample_id=sample_id,
                scenario=scenario,
                condition=condition,
                adaptive_filter=adaptive_filter,
            )
            metrics["motion_gate_filter_allowlist"] = ";".join(
                str(v) for v in payload.get("motion_gate_filter_allowlist", [])
            )
            metrics["motion_low_reacquire_effective"] = bool(
                payload.get("motion_low_reacquire_effective", payload.get("reacquire_enable"))
            )
            metrics["motion_high_lock_escape_effective"] = bool(
                payload.get("motion_high_lock_escape_effective")
                if "motion_high_lock_escape_effective" in payload
                else (payload.get("high_lock_escape") or {}).get("enabled")
            )
            yield metrics


def _sample_summary_rows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in windows:
        groups[(row["condition"], row["sample"], row["scenario"])].append(row)
    output: list[dict[str, Any]] = []
    for (condition, sample, scenario), rows in sorted(groups.items()):
        output.append(_summary_row(rows, condition=condition, sample=sample, scenario=scenario))
    return output


def _scenario_summary_rows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in windows:
        groups[(row["condition"], row["scenario"])].append(row)
    output: list[dict[str, Any]] = []
    for (condition, scenario), rows in sorted(groups.items()):
        output.append(_summary_row(rows, condition=condition, sample="", scenario=scenario))
    return output


def _summary_row(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    sample: str,
    scenario: str,
) -> dict[str, Any]:
    failure_counts = Counter(str(row["primary_failure_reason"]) for row in rows)
    return {
        "condition": condition,
        "sample": sample,
        "scenario": scenario,
        "window_count": len(rows),
        "mae_bpm": _mean_present(row.get("abs_error_bpm") for row in rows),
        "hit_rate": _rate(row.get("output_reached") for row in rows),
        "visible_rate": _rate(row.get("true_peak_visible") for row in rows),
        "range_reachable_rate": _rate(row.get("range_reachable") for row in rows),
        "output_reached_rate": _rate(row.get("output_reached") for row in rows),
        "mean_previous_error_bpm": _mean_present(row.get("previous_error_bpm") for row in rows),
        "mean_search_center_error_bpm": _mean_present(
            row.get("search_center_error_bpm") for row in rows
        ),
        "top_failure_reason": failure_counts.most_common(1)[0][0] if failure_counts else "",
        "failure_reason_counts": ";".join(
            f"{reason}:{count}" for reason, count in sorted(failure_counts.items())
        ),
    }


def _write_rows(output_dir: Path, filename: str, rows: list[dict[str, Any]]) -> Path:
    path = safe_output_path(output_dir, filename)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_markdown_report(
    output_dir: Path,
    sample_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    windows: list[dict[str, Any]],
) -> Path:
    path = safe_output_path(output_dir, "lms_klms_spectral_analysis_report.md")
    best_lines = sorted(
        sample_rows,
        key=lambda row: (
            str(row.get("sample")),
            str(row.get("condition")),
        ),
    )[:12]
    body = [
        "# LMS/KLMS 运动段频谱可见性分析",
        "",
        "本报告由受控 8 条件结果生成，只统计 `is_motion=True && used_adaptive=True` 的运动段自适应窗口。",
        "",
        "## 数据规模",
        "",
        f"- 运动段窗口数：{len(windows)}",
        f"- 样本汇总行数：{len(sample_rows)}",
        f"- 场景汇总行数：{len(scenario_rows)}",
        "",
        "## 初步样本摘要",
        "",
        "| condition | sample | windows | MAE | visible_rate | range_reachable_rate | top_failure |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in best_lines:
        body.append(
            "| {condition} | {sample} | {window_count} | {mae_bpm:.3f} | "
            "{visible_rate:.3f} | {range_reachable_rate:.3f} | {top_failure_reason} |".format(
                **row
            )
        )
    body.extend(
        [
            "",
            "## 解释边界",
            "",
            "输出误差不能反推频谱干净度。若 KLMS 输出更好但真实峰可见率不高，解释应转向真实峰可达性、previous HR、搜索范围和机制门控状态。",
        ]
    )
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def _primary_failure_reason(
    *,
    output_reached: bool,
    visible: bool,
    range_reachable: bool,
    selected_true_peak: bool,
    tracked_near_ref: bool,
    reacquire_triggered: bool,
    high_lock_triggered: bool,
) -> str:
    if output_reached:
        if reacquire_triggered:
            return "mechanism_low_reacquire_helped"
        if high_lock_triggered:
            return "mechanism_high_escape_helped"
        return "already_correct"
    if not visible:
        return "no_visible_ref_peak"
    if not range_reachable:
        return "visible_not_in_range"
    if high_lock_triggered:
        return "mechanism_high_escape_hurt"
    if selected_true_peak or tracked_near_ref:
        return "selected_but_limited_away"
    return "in_range_not_selected"


def _nearest_true_peak(
    candidates: list[float],
    amplitudes: list[float],
    ref_bpm: float | None,
) -> dict[str, Any] | None:
    if ref_bpm is None:
        return None
    if not candidates:
        return None
    max_amp = max([abs(value) for value in amplitudes] or [0.0])
    matches = []
    for idx, bpm in enumerate(candidates):
        distance = abs(float(bpm) - float(ref_bpm))
        if distance <= TRUE_PEAK_TOLERANCE_BPM:
            amp = amplitudes[idx] if idx < len(amplitudes) else math.nan
            matches.append((distance, idx, float(bpm), float(amp)))
    if not matches:
        return None
    distance, idx, bpm, amp = sorted(matches, key=lambda item: (item[0], item[1]))[0]
    return {
        "bpm": bpm,
        "distance_bpm": distance,
        "rank": idx + 1,
        "amp_ratio": amp / max_amp if max_amp > 0 and math.isfinite(amp) else math.nan,
    }


def _best_hf_stage(stages: list[dict[str, Any]]) -> dict[str, Any]:
    hf_stages = [
        stage
        for stage in stages
        if str(stage.get("sensor_type", "")).strip().upper() == "HF"
    ]
    if not hf_stages:
        return {}
    return max(hf_stages, key=lambda stage: abs(_first_float(stage.get("corr")) or 0.0))


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list | tuple):
        return []
    out = []
    for item in value:
        parsed = _first_float(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _abs_diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(float(left) - float(right))


def _signed_diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _nearest_distance(values: list[float], target: float | None) -> float | None:
    if target is None or not values:
        return None
    return min(abs(float(value) - float(target)) for value in values)


def _mean_present(values: Iterable[Any]) -> float:
    present = [_first_float(value) for value in values]
    filtered = [value for value in present if value is not None]
    return float(mean(filtered)) if filtered else math.nan


def _rate(values: Iterable[Any]) -> float:
    items = [bool(value) for value in values]
    return float(sum(items) / len(items)) if items else math.nan


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--historical-baseline-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.historical_baseline_root is not None:
        print(
            "historical_baseline_root is recorded for study context but not mixed into controlled metrics: "
            f"{args.historical_baseline_root}"
        )
    result = analyze_result_root(
        args.result_root,
        output_dir=args.output,
        sample_ids=tuple(args.sample),
        scenarios=tuple(args.scenario),
    )
    print(f"window_csv={result.window_csv}")
    print(f"sample_summary_csv={result.sample_summary_csv}")
    print(f"scenario_summary_csv={result.scenario_summary_csv}")
    print(f"report_md={result.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
