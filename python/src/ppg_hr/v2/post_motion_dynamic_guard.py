"""LYX post-motion dynamic guard experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .algorithm_presets import DirectionalTrackingParams
from .motion_aware_fft_baseline import FFT_CHAIN_POST_GUARD_RESET, run_baseline_sample
from .plotting import render_v2_report
from .post_motion_dynamic_guard_policy import (
    DynamicGuardConfig,
    dynamic_guard_overrides_from_config,
    event_dicts,
    rank_dynamic_guard_candidates,
    switch_mask_and_events,
)
from .post_motion_reset_fft_reacquire import (
    compute_lite_baseline_metrics,
    enumerate_lyx_samples,
    select_representative_lyx_samples,
)
from .report import save_v2_report
from .solver import V2SolverResult, solve_v2
from .types import V2RunConfig

_STAGE1_EVIDENCE_PLOT_CURVES = ("reference", "fft", "adaptive")
_STAGE1_EVIDENCE_COMPARISON_GROUPS = (("ACC",),)
_STAGE1_PLOT_CURVE_SEMANTICS = {
    "adaptive": "Final-HF",
    "fft": "post-motion reset FFT",
    "comparison_groups": ["ACC"],
}


def build_dynamic_guard_candidate_configs() -> list[DynamicGuardConfig]:
    return [
        DynamicGuardConfig(name="lite_recovery_transition_gap3_stable3"),
        DynamicGuardConfig(
            name="lite_recovery_transition_gap2_stable3",
            crossover_gap_bpm=2.0,
        ),
        DynamicGuardConfig(
            name="lite_recovery_transition_gap3_stable2",
            stable_windows=2,
        ),
        DynamicGuardConfig(
            name="middle_transition_gap3_stable3",
            recovery_step_down_bpm=4.0,
            recovery_step_up_bpm=1.5,
        ),
        DynamicGuardConfig(
            name="old_reset_transition_gap3_stable3",
            recovery_step_down_bpm=6.0,
            recovery_step_up_bpm=1.5,
        ),
        DynamicGuardConfig(
            name="gap20_c3",
            crossover_gap_bpm=2.0,
            rescue_gap_bpm=20.0,
            gap_rescue_windows=4,
            gap_rescue_min_hits=3,
            gap_rescue_fft_stable_windows=3,
            gap_rescue_fft_stable_bpm=6.0,
        ),
        DynamicGuardConfig(
            name="gap25_c3",
            crossover_gap_bpm=2.0,
            rescue_gap_bpm=25.0,
            gap_rescue_windows=4,
            gap_rescue_min_hits=3,
            gap_rescue_fft_stable_windows=3,
            gap_rescue_fft_stable_bpm=6.0,
        ),
        DynamicGuardConfig(
            name="gap20_c4",
            crossover_gap_bpm=2.0,
            rescue_gap_bpm=20.0,
            gap_rescue_windows=5,
            gap_rescue_min_hits=4,
            gap_rescue_fft_stable_windows=3,
            gap_rescue_fft_stable_bpm=6.0,
        ),
        DynamicGuardConfig(
            name="gap20_c3_strict_fft",
            crossover_gap_bpm=2.0,
            rescue_gap_bpm=20.0,
            gap_rescue_windows=4,
            gap_rescue_min_hits=3,
            gap_rescue_fft_stable_windows=4,
            gap_rescue_fft_stable_bpm=4.0,
        ),
        DynamicGuardConfig(
            name="rescue_gap20_rise3_lite_recovery",
            rescue_gap_bpm=20.0,
            rising_windows=3,
            rising_slope_bpm_per_window=1.5,
        ),
    ]


def full_lyx_overrides_for_candidate(config: DynamicGuardConfig) -> dict[str, object]:
    return dynamic_guard_overrides_from_config(config)


def run_post_motion_dynamic_guard_stage1(
    *,
    data_root: Path | str,
    lite_batch_dir: Path | str,
    output_dir: Path | str,
    configs: list[DynamicGuardConfig] | None = None,
    representative_only: bool = True,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_samples = enumerate_lyx_samples(data_root)
    samples = (
        select_representative_lyx_samples(all_samples)
        if representative_only
        else all_samples
    )
    active_configs = configs or build_dynamic_guard_candidate_configs()
    lite_rows = compute_lite_baseline_metrics(lite_batch_dir)
    lite_by_sample = {str(row["sample_id"]): row for row in lite_rows}
    sample_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    report_paths: list[Path] = []

    for sample in samples:
        base_cfg = V2RunConfig(
            data_path=sample.data_path,
            ref_path=sample.ref_path,
            algorithm_preset="lite",
            adaptive_filter="lms",
            reference_groups_order=("HF",),
            post_motion_reacquire_enable=False,
        )
        source_result = solve_v2(base_cfg)
        motion = source_result.metadata.get("motion_segment") or {}
        if "end_s" not in motion:
            continue
        for config in active_configs:
            reset_fft_by_time = _reset_fft_chain_by_time(sample, base_cfg, config)
            combined_hr, events = _combine_dynamic_guard(
                source_result.HR,
                motion,
                config,
                reset_fft_by_time=reset_fft_by_time,
            )
            row = _sample_metrics(
                sample_id=sample.sample_id,
                config=config,
                motion_end_s=float(motion["end_s"]),
                hr=combined_hr,
                events=events,
                lite_baseline=lite_by_sample.get(sample.sample_id, {}),
            )
            sample_rows.append(row)
            for event in events:
                switch_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "candidate_name": config.name,
                        **event,
                    }
                )
            report_path = _write_candidate_report(
                output_dir=out / "json",
                source_result=source_result,
                combined_hr=combined_hr,
                config=config,
                switch_events=events,
            )
            report_paths.append(report_path)

    ranking_rows = rank_dynamic_guard_candidates(sample_rows)
    _write_dict_csv(out / "representative_sample_metrics.csv", sample_rows)
    _write_dict_csv(out / "candidate_ranking.csv", ranking_rows)
    _write_dict_csv(out / "switch_event_table.csv", switch_rows)
    gated_names = _stage1_gated_candidate_names(ranking_rows)
    plot_names, plot_tier = _stage1_plot_candidate_names_and_tier(ranking_rows)
    gated_reports = [
        path
        for path in report_paths
        if _candidate_name_from_report_path(path) in gated_names
    ]
    plot_reports = [
        path
        for path in report_paths
        if _candidate_name_from_report_path(path) in plot_names
    ]
    _render_reports(
        plot_reports,
        out / "png" / plot_tier,
        out / "csv" / plot_tier,
    )
    gated_png_dir = (
        (out / "png" / "stage1_gated_candidates").as_posix()
        if gated_reports
        else ""
    )
    best_effort_png_dir = (
        (out / "png" / "stage1_best_effort").as_posix()
        if plot_tier == "stage1_best_effort" and plot_reports
        else ""
    )
    (out / "post_motion_dynamic_guard_metadata.json").write_text(
        json.dumps(
            {
                "data_root": str(Path(data_root)),
                "lite_batch_dir": str(Path(lite_batch_dir)),
                "output_dir": str(out),
                "sample_count": len(samples),
                "candidate_count": len(active_configs),
                "selected_candidate": (
                    ranking_rows[0]["candidate_name"] if ranking_rows else ""
                ),
                "selection_tier": (
                    ranking_rows[0]["selection_tier"] if ranking_rows else ""
                ),
                "gated_candidates": sorted(gated_names),
                "stage1_gated_candidate_png_dir": gated_png_dir,
                "stage1_best_effort_png_dir": best_effort_png_dir,
                "plot_candidate_tier": plot_tier,
                "plot_candidate_count": len(plot_reports),
                "plot_curve_semantics": dict(_STAGE1_PLOT_CURVE_SEMANTICS),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_text = render_dynamic_guard_markdown_report(
        ranking_rows=ranking_rows,
        sample_rows=sample_rows,
        switch_rows=switch_rows,
        output_dir=out,
    )
    (out / "post_motion_dynamic_guard_report.md").write_text(
        report_text,
        encoding="utf-8",
    )
    return {
        "sample_rows": sample_rows,
        "switch_rows": switch_rows,
        "ranking_rows": ranking_rows,
        "candidate_reports": report_paths,
        "gated_candidate_reports": gated_reports,
        "plot_candidate_reports": plot_reports,
        "selected_candidate_reports": plot_reports,
        "stage1_gated_candidate_png_dir": gated_png_dir,
        "stage1_best_effort_png_dir": best_effort_png_dir,
        "plot_curve_semantics": dict(_STAGE1_PLOT_CURVE_SEMANTICS),
    }


def render_dynamic_guard_markdown_report(
    *,
    ranking_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    switch_rows: list[dict[str, Any]],
    output_dir: Path | str,
) -> str:
    selected = ranking_rows[0] if ranking_rows else {}
    candidate_name = str(selected.get("candidate_name", ""))
    selection_tier = str(selected.get("selection_tier", ""))
    _plot_names, plot_tier = _stage1_plot_candidate_names_and_tier(ranking_rows)
    best_effort_note = (
        "该候选只是 best-effort，也就是没有候选完整通过阶段 1 门控时的相对最优组合；"
        "它可以进入后续全量复核，但不能被写成正式通过。"
        if selection_tier == "best_effort_candidate"
        else "该候选属于 promoted_candidate，表示它通过了阶段 1 的代表样本门控。"
    )
    out = Path(output_dir)
    lines = [
        "# 运动后动态保护窗实验报告",
        "",
        "## 一句话结论",
        "",
        (
            f"阶段 1 当前排序第一的候选是 `{candidate_name}`，门控层级为 "
            f"`{selection_tier}`。{best_effort_note}"
        ),
        "",
        "## 方法概述",
        "",
        (
            "本轮实验让 adaptive 链路和运动后 reset FFT 链路并行运行，"
            "Final-HF 在运动段后先保留 adaptive，直到 reset FFT 与 adaptive "
            "出现稳定、且符合恢复段动态追踪参数约束的可达交汇。"
        ),
        (
            "如果 adaptive 在运动结束后仍持续上升，并且明显高于 reset FFT，"
            "实验会触发 `adaptive_rising_rescue`，提前切回 reset FFT 以补救运动段漂移。"
        ),
        "",
        "## 阶段 1 可视化证据",
        "",
        (
            "阶段 1 会为通过门控的 `promoted_candidate` 输出候选心率 PNG；如果没有候选完全通过，"
            "则为进入后续全量复核的 `best_effort_candidate` 输出同样格式的 PNG。"
        ),
        (
            "这些图统一包含四类判读信息：黑线为参考心率，红色主曲线为 Final-HF，"
            "灰色虚线为运动后 reset FFT 链路，蓝色对比曲线为 ACC 参考顺序下的链路。"
        ),
        f"当前阶段 1 可视化目录为 `{(out / 'png' / plot_tier).as_posix()}`。",
        "",
        "## 指标怎么读",
        "",
        (
            "`fixed 60 s post-motion MAE` 指运动结束后固定 60 s 内，Final-HF "
            "与参考心率之间的平均绝对误差；它避免长静息尾段掩盖恢复初段失败。"
        ),
        (
            "`delta vs Lite` 是新候选相对旧 Lite 输出的误差变化，负值表示误差降低。"
        ),
        (
            "`switch_reason` 记录切换原因，目前只有 `stable_crossover` 和 "
            "`adaptive_rising_rescue` 两类。"
        ),
        (
            "`best_effort_candidate` 表示没有候选完全通过门控时仍进入后续全量复核的"
            "相对最优组合。"
        ),
        "",
        "## 候选排名",
        "",
        (
            "| Candidate | Tier | key regression | fixed 60 s delta | "
            "high-drift gain | low-lock windows |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in ranking_rows:
        lines.append(
            "| {candidate} | {tier} | {key} | {delta} | {gain} | {low} |".format(
                candidate=row.get("candidate_name", ""),
                tier=row.get("selection_tier", ""),
                key=_format_float(row.get("max_key_sample_regression_bpm")),
                delta=_format_float(row.get("mean_delta_vs_lite_60s_mae_bpm")),
                gain=_format_float(row.get("high_drift_gain_bpm")),
                low=_format_count(row.get("low_lock_window_count")),
            )
        )

    lines.extend(
        [
            "",
            "## 代表样本结果",
            "",
            (
                "| Sample | Candidate | old Lite post MAE | new post MAE | "
                "fixed 60 s MAE | delta | switch reason |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sample_rows:
        lines.append(
            "| {sample} | {candidate} | {old} | {new} | {post60} | {delta} | {reason} |".format(
                sample=row.get("sample_id", ""),
                candidate=row.get("candidate_name", ""),
                old=_format_float(row.get("old_lite_post_motion_mae_bpm")),
                new=_format_float(row.get("post_motion_full_final_mae_bpm")),
                post60=_format_float(row.get("post_motion_60s_final_mae_bpm")),
                delta=_format_float(row.get("delta_vs_lite_post_mae_bpm")),
                reason=row.get("selected_switch_reason", ""),
            )
        )

    lines.extend(
        [
            "",
            "## 切换事件",
            "",
            "| Sample | Candidate | switch time (s) | reason |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in switch_rows:
        lines.append(
            "| {sample} | {candidate} | {time} | {reason} |".format(
                sample=row.get("sample_id", ""),
                candidate=row.get("candidate_name", ""),
                time=_format_float(row.get("center_s")),
                reason=row.get("switch_reason", ""),
            )
        )

    lines.extend(
        [
            "",
            "## 证据来源",
            "",
            f"- `{out / 'representative_sample_metrics.csv'}`",
            f"- `{out / 'candidate_ranking.csv'}`",
            f"- `{out / 'switch_event_table.csv'}`",
            f"- `{out / 'png' / plot_tier}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _reset_fft_chain_by_time(
    sample: Any,
    base_cfg: V2RunConfig,
    config: DynamicGuardConfig,
) -> dict[float, float]:
    reset_tracking = DirectionalTrackingParams(
        range_up_bpm=20.0,
        range_down_bpm=25.0,
        limit_up_bpm=float(config.recovery_step_up_bpm),
        step_up_bpm=float(config.recovery_step_up_bpm),
        limit_down_bpm=float(config.recovery_step_down_bpm),
        step_down_bpm=float(config.recovery_step_down_bpm),
    )
    run = run_baseline_sample(
        sample,
        fft_chain=FFT_CHAIN_POST_GUARD_RESET,
        guard_seconds=float(config.min_elapsed_s),
        base_config=base_cfg,
        post_reset_tracking=reset_tracking,
    )
    return {
        round(float(row["time_s"]), 6): float(row["fft_baseline_bpm"])
        for row in run.window_rows
        if "time_s" in row and "fft_baseline_bpm" in row
    }


def _combine_dynamic_guard(
    source_hr: np.ndarray,
    motion_segment: dict[str, float],
    config: DynamicGuardConfig,
    *,
    reset_fft_by_time: dict[float, float] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    hr = np.asarray(source_hr, dtype=float).copy()
    source = np.zeros((hr.shape[0], 9), dtype=float)
    source[:, 0] = hr[:, 0]
    source[:, 1] = hr[:, 1] / 60.0
    source[:, 2] = hr[:, 3] / 60.0
    source[:, 4] = hr[:, 2] / 60.0
    for idx, t in enumerate(hr[:, 0]):
        reset_bpm = (reset_fft_by_time or {}).get(round(float(t), 6))
        if reset_bpm is not None and np.isfinite(reset_bpm):
            source[idx, 4] = float(reset_bpm) / 60.0
            hr[idx, 2] = float(reset_bpm)
    mask, events = switch_mask_and_events(
        source,
        motion_segment=motion_segment,
        config=config,
    )
    if hr.shape[1] > 5:
        hr[:, 5] = mask.astype(float)
    hr[:, 3] = np.where(mask, source[:, 2] * 60.0, source[:, 4] * 60.0)
    return hr, event_dicts(events)


def _sample_metrics(
    *,
    sample_id: str,
    config: DynamicGuardConfig,
    motion_end_s: float,
    hr: np.ndarray,
    events: list[dict[str, Any]],
    lite_baseline: dict[str, Any],
) -> dict[str, Any]:
    post = hr[hr[:, 0] > motion_end_s + 1e-9]
    post60 = hr[
        (hr[:, 0] > motion_end_s + 1e-9)
        & (hr[:, 0] <= motion_end_s + 60.0 + 1e-9)
    ]
    post_mae = _hr_mae(post)
    post60_mae = _hr_mae(post60)
    old_post = _as_float(lite_baseline.get("lite_post_motion_mae_bpm"))
    old_60s = _as_float(lite_baseline.get("lite_post_motion_60s_mae_bpm"))
    return {
        "sample_id": sample_id,
        "candidate_name": config.name,
        "post_motion_full_final_mae_bpm": post_mae,
        "post_motion_60s_final_mae_bpm": post60_mae,
        "old_lite_post_motion_mae_bpm": old_post,
        "old_lite_post_motion_60s_mae_bpm": old_60s,
        "delta_vs_lite_post_mae_bpm": post_mae - old_post,
        "delta_vs_lite_60s_mae_bpm": post60_mae - old_60s,
        "switch_count": len(events),
        "missing_switch_reason_count": sum(
            1 for event in events if not event.get("switch_reason")
        ),
        "dynamic_reachable_failure_count": sum(
            1 for event in events if not bool(event.get("reachable", False))
        ),
        "low_lock_window_count": _low_lock_count(post),
        "selected_switch_reason": (
            str(events[0].get("switch_reason", "")) if events else ""
        ),
        "selected_switch_s": (
            _as_float(events[0].get("center_s")) if events else float("nan")
        ),
    }


def _write_candidate_report(
    *,
    output_dir: Path,
    source_result: Any,
    combined_hr: np.ndarray,
    config: DynamicGuardConfig,
    switch_events: list[dict[str, Any]],
) -> Path:
    metadata = dict(source_result.metadata)
    metadata["post_motion_dynamic_guard"] = {
        "enabled": True,
        "candidate_name": config.name,
        "config": config.to_dict(),
        "reset_fft_enabled": True,
        "evidence_plot": dict(_STAGE1_PLOT_CURVE_SEMANTICS),
        "switch_events": switch_events,
    }
    result = V2SolverResult(
        HR=combined_hr,
        err_stats=getattr(source_result, "err_stats", {}),
        metadata=metadata,
        window_table=getattr(source_result, "window_table", []),
    )
    data_stem = Path(str(metadata.get("data_path", "sample.csv"))).stem
    return save_v2_report(
        output_dir / f"{data_stem}-{config.name}-v2.json",
        result,
        best_params={},
    )


def _render_reports(report_paths: list[Path], png_dir: Path, csv_dir: Path) -> None:
    png_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    for report_path in report_paths:
        render_v2_report(
            report_path,
            out_dir=png_dir,
            csv_dir=csv_dir,
            output_prefix=report_path.stem,
            plot_curves=_STAGE1_EVIDENCE_PLOT_CURVES,
            comparison_groups=_STAGE1_EVIDENCE_COMPARISON_GROUPS,
        )


def _stage1_gated_candidate_names(ranking_rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("candidate_name", ""))
        for row in ranking_rows
        if str(row.get("selection_tier", "")) == "promoted_candidate"
    }


def _stage1_plot_candidate_names_and_tier(
    ranking_rows: list[dict[str, Any]],
) -> tuple[set[str], str]:
    promoted = _stage1_gated_candidate_names(ranking_rows)
    if promoted:
        return promoted, "stage1_gated_candidates"
    if ranking_rows:
        return {str(ranking_rows[0].get("candidate_name", ""))}, "stage1_best_effort"
    return set(), "stage1_gated_candidates"


def _stage1_selected_candidate_names(ranking_rows: list[dict[str, Any]]) -> set[str]:
    names, _tier = _stage1_plot_candidate_names_and_tier(ranking_rows)
    return names


def _candidate_name_from_report_path(path: Path) -> str:
    stem = Path(path).stem
    marker = "-v2"
    if stem.endswith(marker):
        stem = stem[: -len(marker)]
    parts = stem.split("-", 1)
    return parts[1] if len(parts) == 2 else stem


def _hr_mae(hr: np.ndarray) -> float:
    arr = np.asarray(hr, dtype=float)
    if arr.size == 0:
        return float("nan")
    err = np.abs(arr[:, 3] - arr[:, 1])
    err = err[np.isfinite(err)]
    return float(np.mean(err)) if err.size else float("nan")


def _low_lock_count(hr: np.ndarray) -> int:
    arr = np.asarray(hr, dtype=float)
    if arr.size == 0:
        return 0
    return int(np.sum((arr[:, 1] - arr[:, 3]) > 10.0))


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_float(value: Any) -> str:
    numeric = _as_float(value)
    return f"{numeric:.3f}" if np.isfinite(numeric) else "NA"


def _format_count(value: Any) -> str:
    numeric = _as_float(value)
    return str(int(numeric)) if np.isfinite(numeric) else "NA"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--lite-batch-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--full-lyx", action="store_true")
    args = parser.parse_args(argv)
    run_post_motion_dynamic_guard_stage1(
        data_root=args.data_root,
        lite_batch_dir=args.lite_batch_dir,
        output_dir=args.output_dir,
        representative_only=not args.full_lyx,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
