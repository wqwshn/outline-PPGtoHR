"""Motion-aware pure FFT baseline study utilities."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal.windows import hamming

from ppg_hr.preprocess.utils import smoothdata_movmedian

from .algorithm_presets import DirectionalTrackingParams
from .signal_preparation import motion_flag_at_center, prepare_v2_signals
from .solver import (
    _classify_window_stage,
    _process_spectrum_with_trace,
    _ref_at,
    _symmetric_tracking_params,
)
from .spectrum_tracking import SpectrumTrackingTrace
from .types import V2RunConfig

FFT_CHAIN_CONTINUOUS = "continuous_fft"
FFT_CHAIN_POST_GUARD_RESET = "post_guard_reset_fft"
FFT_CHAIN_POST_GUARD_WEAK_INHERIT = "post_guard_weak_inherit_fft"
FFT_CHAINS = (
    FFT_CHAIN_CONTINUOUS,
    FFT_CHAIN_POST_GUARD_RESET,
    FFT_CHAIN_POST_GUARD_WEAK_INHERIT,
)
DEFAULT_GUARD_SECONDS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
DEFAULT_WEAK_INHERIT_RANGE_BPM = 40.0
POST_GUARD_MAE_TARGET_BPM = 3.0
DEFAULT_TOPK_CONSENSUS_TOLERANCE_BPM = 6.0


@dataclass(frozen=True)
class BaselineSample:
    cohort: str
    sample_id: str
    data_path: Path
    ref_path: Path


@dataclass(frozen=True)
class BaselineRun:
    sample: BaselineSample
    fft_chain: str
    guard_seconds: float
    motion_segment: dict[str, float] | None
    window_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def enumerate_motion_aware_fft_samples(
    data_root: Path | str,
    *,
    cohorts: Iterable[str] = ("LYX", "TS"),
) -> list[BaselineSample]:
    root = Path(data_root)
    samples: list[BaselineSample] = []
    for cohort in cohorts:
        cohort_dir = root / cohort
        if not cohort_dir.is_dir():
            continue
        for data_path in sorted(cohort_dir.glob("*.csv")):
            if data_path.name.endswith("_HR_ref.csv") or data_path.name.endswith("_ref.csv"):
                continue
            ref_path = data_path.with_name(f"{data_path.stem}_HR_ref{data_path.suffix}")
            if not ref_path.is_file():
                continue
            samples.append(
                BaselineSample(
                    cohort=str(cohort),
                    sample_id=data_path.stem,
                    data_path=data_path,
                    ref_path=ref_path,
                )
            )
    return samples


def run_baseline_sample(
    sample: BaselineSample,
    *,
    fft_chain: str,
    guard_seconds: float,
    base_config: V2RunConfig | None = None,
    prepared_signals: Any | None = None,
    weak_inherit_range_bpm: float = DEFAULT_WEAK_INHERIT_RANGE_BPM,
    post_reset_tracking: DirectionalTrackingParams | None = None,
    post_reset_min_bpm_floor: float | None = None,
    first_window_no_hold_count: int = 0,
    post_reset_consensus_k: int = 0,
    post_reset_consensus_windows: int = 0,
    post_reset_consensus_tolerance_bpm: float = DEFAULT_TOPK_CONSENSUS_TOLERANCE_BPM,
) -> BaselineRun:
    cfg = _config_for_sample(sample, base_config)
    if fft_chain not in FFT_CHAINS:
        allowed = ", ".join(FFT_CHAINS)
        raise ValueError(f"Unsupported fft_chain={fft_chain!r}; expected one of {allowed}")
    prepared = prepared_signals if prepared_signals is not None else prepare_v2_signals(cfg)
    params = prepared.params
    fs = prepared.fs
    ppg = prepared.ppg
    motion_detection = prepared.motion_detection
    motion_segment = motion_detection.motion_segment
    rest_tracking = _symmetric_tracking_params(
        params.hr_range_rest,
        params.slew_limit_rest,
        params.slew_step_rest,
    )
    reset_tracking = post_reset_tracking or rest_tracking
    weak_tracking = DirectionalTrackingParams(
        range_up_bpm=float(weak_inherit_range_bpm),
        range_down_bpm=float(weak_inherit_range_bpm),
        limit_up_bpm=float(weak_inherit_range_bpm),
        step_up_bpm=float(weak_inherit_range_bpm),
        limit_down_bpm=float(weak_inherit_range_bpm),
        step_down_bpm=float(weak_inherit_range_bpm),
    )

    rows: list[dict[str, Any]] = []
    history: list[float] = []
    local_history: list[float] = []
    consensus_observations: list[tuple[float, ...]] = []
    reacquire_started = False
    consensus_enabled = int(post_reset_consensus_k) > 0 and int(post_reset_consensus_windows) > 0
    motion_end_s = (
        float(motion_segment["end_s"]) if motion_segment is not None else float("nan")
    )
    guard_end_s = motion_end_s + float(guard_seconds) if math.isfinite(motion_end_s) else float("nan")

    time_1 = float(params.time_start)
    time_end = len(prepared.ppg_ori) / fs - params.time_buffer
    while True:
        time_2 = time_1 + float(cfg.window_seconds)
        idx_s = int(round(time_1 * fs))
        idx_e = int(round(time_2 * fs))
        if idx_e > len(ppg):
            break
        center = time_1 + float(cfg.window_seconds) / 2.0
        sig_p = ppg[idx_s:idx_e]
        sig_fft = (sig_p - sig_p.mean()) * hamming(len(sig_p))
        should_reacquire = bool(
            motion_segment is not None
            and center > guard_end_s + 1e-9
            and fft_chain in {FFT_CHAIN_POST_GUARD_RESET, FFT_CHAIN_POST_GUARD_WEAK_INHERIT}
            and not reacquire_started
        )

        if should_reacquire and fft_chain == FFT_CHAIN_POST_GUARD_RESET and consensus_enabled:
            local_history = []
            hr_hz, trace = _track_fft_window(
                sig_fft,
                fs,
                params,
                local_history,
                reset_tracking,
            )
            candidates = _topk_consensus_candidates(
                trace,
                k=int(post_reset_consensus_k),
                min_bpm_floor=post_reset_min_bpm_floor,
            )
            consensus_observations.append(candidates)
            selected_bpm = float("nan")
            consensus_status = "pending"
            consensus_reason = "waiting_for_consensus"
            if len(consensus_observations) >= int(post_reset_consensus_windows):
                selected = _find_topk_consensus_peak(
                    consensus_observations[-int(post_reset_consensus_windows) :],
                    tolerance_bpm=float(post_reset_consensus_tolerance_bpm),
                )
                if math.isfinite(selected):
                    selected_bpm = selected
                    hr_hz = selected_bpm / 60.0
                    trace.tracked_hr_bpm = selected_bpm
                    trace.slew_limited_hr_bpm = selected_bpm
                    trace.selected_peak_rank = _candidate_rank(candidates, selected_bpm)
                    trace.candidate_source = "topk_consensus_reset"
                    consensus_status = "selected"
                    consensus_reason = ""
                    reacquire_started = True
                else:
                    selected_bpm = _post_reset_forced_candidate_bpm(
                        trace,
                        post_reset_min_bpm_floor,
                    )
                    if math.isfinite(selected_bpm):
                        hr_hz = selected_bpm / 60.0
                        trace.tracked_hr_bpm = selected_bpm
                        trace.slew_limited_hr_bpm = selected_bpm
                        trace.selected_peak_rank = _candidate_rank(candidates, selected_bpm)
                    trace.candidate_source = "topk_consensus_fallback"
                    consensus_status = "fallback"
                    consensus_reason = "no_stable_peak"
                    reacquire_started = True
            else:
                trace.candidate_source = "topk_consensus_pending"
            _attach_consensus_trace_fields(
                trace,
                status=consensus_status,
                selected_bpm=selected_bpm,
                failure_reason=consensus_reason,
                window_count=len(consensus_observations),
                k=int(post_reset_consensus_k),
            )
        elif should_reacquire and fft_chain == FFT_CHAIN_POST_GUARD_RESET:
            local_history = []
            hr_hz, trace = _track_fft_window(
                sig_fft,
                fs,
                params,
                local_history,
                reset_tracking,
            )
            trace.candidate_source = "post_guard_reset"
            reacquire_started = True
        elif should_reacquire and fft_chain == FFT_CHAIN_POST_GUARD_WEAK_INHERIT:
            terminal_previous = history[-1] if history else float("nan")
            weak_history = [terminal_previous] if np.isfinite(terminal_previous) else []
            hr_hz, trace = _track_fft_window(
                sig_fft,
                fs,
                params,
                weak_history,
                weak_tracking,
            )
            if trace.candidate_source == "held_previous":
                hr_hz = float(trace.raw_candidate_hr_bpm) / 60.0
                trace.tracked_hr_bpm = float(trace.raw_candidate_hr_bpm)
                trace.slew_limited_hr_bpm = float(trace.raw_candidate_hr_bpm)
                trace.selected_peak_rank = 1 if trace.raw_candidate_hr_bpm > 0.0 else 0
                trace.candidate_source = "weak_inherit_raw_fallback"
            else:
                trace.candidate_source = f"weak_inherit_{trace.candidate_source}"
            local_history = []
            reacquire_started = True
        else:
            active_history = history if not reacquire_started else local_history
            active_tracking = rest_tracking if not reacquire_started else reset_tracking
            hr_hz, trace = _track_fft_window(
                sig_fft,
                fs,
                params,
                active_history,
                active_tracking,
            )

        if (
            fft_chain == FFT_CHAIN_POST_GUARD_RESET
            and reacquire_started
            and not consensus_enabled
            and len(local_history) < int(first_window_no_hold_count)
        ):
            forced_bpm = _post_reset_forced_candidate_bpm(trace, post_reset_min_bpm_floor)
            if math.isfinite(forced_bpm) and (
                trace.candidate_source == "held_previous"
                or (
                    post_reset_min_bpm_floor is not None
                    and float(trace.raw_candidate_hr_bpm) < float(post_reset_min_bpm_floor)
                )
            ):
                hr_hz = forced_bpm / 60.0
                trace.tracked_hr_bpm = forced_bpm
                trace.slew_limited_hr_bpm = forced_bpm
                trace.candidate_source = "post_guard_raw_fallback"

        history.append(float(hr_hz))
        if reacquire_started:
            local_history.append(float(hr_hz))

        ref_bpm = _ref_at(center + float(cfg.time_bias), prepared.ref_data) if prepared.ref_data.size else float("nan")
        is_motion = motion_flag_at_center(center, motion_detection)
        window_stage = _baseline_window_stage(center, motion_segment, guard_end_s)
        error_bpm = float(hr_hz) * 60.0 - float(ref_bpm) if np.isfinite(ref_bpm) else float("nan")
        row = {
            "cohort": sample.cohort,
            "sample_id": sample.sample_id,
            "fft_chain": fft_chain,
            "guard_seconds": float(guard_seconds),
            "time_s": float(center),
            "ref_bpm": float(ref_bpm),
            "fft_baseline_bpm": float(hr_hz) * 60.0,
            "window_stage": window_stage,
            "is_motion": bool(is_motion),
            "candidate_source": trace.candidate_source,
            "candidate_peaks_bpm": tuple(
                float(value) for value in getattr(trace, "candidate_peaks_bpm", ())
            ),
            "raw_candidate_bpm": float(trace.raw_candidate_hr_bpm),
            "previous_hr_bpm": trace.previous_hr_bpm,
            "selected_peak_rank": int(trace.selected_peak_rank),
            "consensus_status": str(getattr(trace, "consensus_status", "")),
            "consensus_selected_bpm": float(
                getattr(trace, "consensus_selected_bpm", float("nan"))
            ),
            "consensus_failure_reason": str(
                getattr(trace, "consensus_failure_reason", "")
            ),
            "consensus_window_count": int(getattr(trace, "consensus_window_count", 0) or 0),
            "consensus_k": int(getattr(trace, "consensus_k", 0) or 0),
            "error_bpm": error_bpm,
        }
        row["failure_reason"] = classify_window_failure(row)
        rows.append(row)
        time_1 += float(cfg.window_step_seconds)
        if time_1 > time_end:
            break

    _smooth_rows_in_place(rows, cfg)
    for row in rows:
        row["error_bpm"] = (
            float(row["fft_baseline_bpm"]) - float(row["ref_bpm"])
            if np.isfinite(float(row["ref_bpm"]))
            else float("nan")
        )
        row["failure_reason"] = classify_window_failure(row)
    summary = summarise_sample_metrics(
        rows,
        sample_id=sample.sample_id,
        cohort=sample.cohort,
        fft_chain=fft_chain,
        guard_seconds=float(guard_seconds),
        motion_end_s=motion_end_s,
    )
    return BaselineRun(
        sample=sample,
        fft_chain=fft_chain,
        guard_seconds=float(guard_seconds),
        motion_segment=motion_segment,
        window_rows=rows,
        summary=summary,
    )


def run_motion_aware_fft_baseline_study(
    data_root: Path | str,
    output_dir: Path | str,
    *,
    cohorts: Iterable[str] = ("LYX", "TS"),
    fft_chains: Iterable[str] = FFT_CHAINS,
    guard_seconds_values: Iterable[float] = DEFAULT_GUARD_SECONDS,
    base_config: V2RunConfig | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = enumerate_motion_aware_fft_samples(data_root, cohorts=cohorts)
    sample_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for sample in samples:
        cfg = _config_for_sample(sample, base_config)
        prepared = prepare_v2_signals(cfg)
        for fft_chain in fft_chains:
            for guard_seconds in guard_seconds_values:
                run = run_baseline_sample(
                    sample,
                    fft_chain=fft_chain,
                    guard_seconds=float(guard_seconds),
                    base_config=cfg,
                    prepared_signals=prepared,
                )
                sample_rows.append(run.summary)
                window_rows.extend(run.window_rows)

    aggregate_rows = aggregate_combo_rows(sample_rows)
    _write_csv(out / "motion_aware_fft_sample_metrics.csv", sample_rows)
    _write_csv(out / "motion_aware_fft_window_metrics.csv", window_rows)
    _write_csv(out / "motion_aware_fft_aggregate_metrics.csv", aggregate_rows)
    write_key_plots(out, sample_rows, window_rows, aggregate_rows)
    report_path = out / "motion_aware_fft_baseline_report.md"
    report_path.write_text(
        render_markdown_report(
            data_root=Path(data_root),
            output_dir=out,
            sample_rows=sample_rows,
            aggregate_rows=aggregate_rows,
        ),
        encoding="utf-8",
    )
    metadata = {
        "data_root": str(Path(data_root)),
        "output_dir": str(out),
        "sample_count": len(samples),
        "fft_chains": list(fft_chains),
        "guard_seconds": [float(v) for v in guard_seconds_values],
        "report_path": str(report_path),
    }
    (out / "motion_aware_fft_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "metadata": metadata,
        "sample_rows": sample_rows,
        "aggregate_rows": aggregate_rows,
    }


def classify_window_failure(row: dict[str, Any]) -> str:
    if str(row.get("candidate_source", "")) == "held_previous":
        return "held_previous"
    error = _as_float(row.get("error_bpm"))
    if not math.isfinite(error):
        return "no_valid_peak"
    if abs(error) < 3.0:
        return "accurate"
    if abs(error) < 5.0:
        return "borderline"
    if error <= -5.0:
        return "low_lock"
    return "high_lock"


def summarise_sample_metrics(
    rows: list[dict[str, Any]],
    *,
    sample_id: str,
    cohort: str,
    fft_chain: str,
    guard_seconds: float,
    motion_end_s: float,
) -> dict[str, Any]:
    post_guard_start = float(motion_end_s) + float(guard_seconds)
    post_guard_rows = [
        row for row in rows if _as_float(row.get("time_s")) > post_guard_start + 1e-9
    ]
    post_60_rows = [
        row
        for row in rows
        if _as_float(row.get("time_s")) > float(motion_end_s) + 1e-9
        and _as_float(row.get("time_s")) <= float(motion_end_s) + 60.0 + 1e-9
    ]
    post_full_rows = [
        row for row in rows if _as_float(row.get("time_s")) > float(motion_end_s) + 1e-9
    ]
    post_guard_mae = _mae(post_guard_rows)
    failure_counts = _failure_counts(post_guard_rows)
    return {
        "cohort": cohort,
        "sample_id": sample_id,
        "fft_chain": fft_chain,
        "guard_seconds": float(guard_seconds),
        "motion_end_s": float(motion_end_s),
        "post_guard_window_count": len(post_guard_rows),
        "post_motion_60s_window_count": len(post_60_rows),
        "post_motion_full_window_count": len(post_full_rows),
        "post_guard_mae_bpm": post_guard_mae,
        "post_motion_60s_mae_bpm": _mae(post_60_rows),
        "post_motion_full_mae_bpm": _mae(post_full_rows),
        "post_guard_max_abs_error_bpm": _max_abs_error(post_guard_rows),
        "passes_post_guard_3bpm": bool(
            math.isfinite(post_guard_mae) and post_guard_mae < POST_GUARD_MAE_TARGET_BPM
        ),
        "primary_failure_reason": _primary_failure_reason(failure_counts),
        **{f"failure_{key}_windows": value for key, value in failure_counts.items()},
    }


def aggregate_combo_rows(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in sample_rows:
        grouped.setdefault((str(row["fft_chain"]), float(row["guard_seconds"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (fft_chain, guard_seconds), rows in sorted(grouped.items()):
        maes = np.asarray([_as_float(row.get("post_guard_mae_bpm")) for row in rows], dtype=float)
        finite = maes[np.isfinite(maes)]
        passing = [bool(row.get("passes_post_guard_3bpm")) for row in rows]
        worst = max(rows, key=lambda row: _as_float(row.get("post_guard_mae_bpm")))
        out.append(
            {
                "fft_chain": fft_chain,
                "guard_seconds": guard_seconds,
                "sample_count": len(rows),
                "passing_sample_count": int(sum(passing)),
                "all_samples_pass_3bpm": bool(rows and all(passing)),
                "mean_post_guard_mae_bpm": float(np.mean(finite)) if finite.size else float("nan"),
                "std_post_guard_mae_bpm": float(np.std(finite)) if finite.size else float("nan"),
                "mean_post_motion_60s_mae_bpm": _finite_mean(
                    [row.get("post_motion_60s_mae_bpm") for row in rows]
                ),
                "mean_post_motion_full_mae_bpm": _finite_mean(
                    [row.get("post_motion_full_mae_bpm") for row in rows]
                ),
                "max_sample_mae_bpm": _as_float(worst.get("post_guard_mae_bpm")),
                "max_sample_id": str(worst.get("sample_id", "")),
                "max_sample_cohort": str(worst.get("cohort", "")),
                "dominant_failure_reason": _dominant_sample_failure_reason(rows),
            }
        )
    return out


def write_key_plots(
    output_dir: Path,
    sample_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plot_dir = output_dir / "png"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    ordered = sorted(
        aggregate_rows,
        key=lambda row: (
            -int(row.get("passing_sample_count", 0)),
            _as_float(row.get("mean_post_guard_mae_bpm")),
            _as_float(row.get("std_post_guard_mae_bpm")),
        ),
    )
    key_combos = [(row["fft_chain"], float(row["guard_seconds"])) for row in ordered[:3]]
    lookup: dict[tuple[str, float, str, str], list[dict[str, Any]]] = {}
    for row in window_rows:
        lookup.setdefault(
            (
                str(row.get("fft_chain")),
                float(row.get("guard_seconds", 0.0)),
                str(row.get("cohort")),
                str(row.get("sample_id")),
            ),
            [],
        ).append(row)

    for fft_chain, guard_seconds in key_combos:
        combo_samples = [
            row
            for row in sample_rows
            if str(row.get("fft_chain")) == fft_chain
            and float(row.get("guard_seconds", 0.0)) == guard_seconds
        ]
        worst = sorted(
            combo_samples,
            key=lambda row: _as_float(row.get("post_guard_mae_bpm")),
            reverse=True,
        )[:3]
        for sample in worst:
            key = (
                fft_chain,
                guard_seconds,
                str(sample.get("cohort")),
                str(sample.get("sample_id")),
            )
            rows = sorted(lookup.get(key, []), key=lambda row: _as_float(row.get("time_s")))
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(10, 4), dpi=140)
            times = [_as_float(row.get("time_s")) for row in rows]
            ref = [_as_float(row.get("ref_bpm")) for row in rows]
            fft = [_as_float(row.get("fft_baseline_bpm")) for row in rows]
            ax.plot(times, ref, color="black", linewidth=1.5, label="Reference")
            ax.plot(times, fft, color="#0072B2", linewidth=1.2, label=fft_chain)
            motion_end = _as_float(sample.get("motion_end_s"))
            if math.isfinite(motion_end):
                ax.axvline(motion_end, color="#D55E00", linewidth=1.0, linestyle="--", label="motion end")
                ax.axvline(
                    motion_end + guard_seconds,
                    color="#009E73",
                    linewidth=1.0,
                    linestyle=":",
                    label="guard end",
                )
            ax.set_title(
                f"{sample.get('cohort')}/{sample.get('sample_id')} | "
                f"{fft_chain} guard={guard_seconds:.0f}s MAE={_as_float(sample.get('post_guard_mae_bpm')):.2f}"
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Heart rate (BPM)")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            out = plot_dir / (
                f"{sample.get('cohort')}-{sample.get('sample_id')}-"
                f"{fft_chain}-guard{guard_seconds:.0f}.png"
            )
            fig.savefig(out)
            plt.close(fig)
            written.append(out)
    return written


def render_markdown_report(
    *,
    data_root: Path,
    output_dir: Path,
    sample_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> str:
    winners = [row for row in aggregate_rows if bool(row.get("all_samples_pass_3bpm"))]
    if winners:
        winners = sorted(
            winners,
            key=lambda row: (
                _as_float(row.get("mean_post_guard_mae_bpm")),
                _as_float(row.get("std_post_guard_mae_bpm")),
            ),
        )
        recommendation = (
            f"推荐候选：`{winners[0]['fft_chain']}`，"
            f"guard={winners[0]['guard_seconds']}s。"
        )
    else:
        recommendation = "本轮没有组合满足所有样本 post-guard MAE < 3 BPM。"

    lines = [
        "# 运动感知纯 FFT 基线研究报告",
        "",
        f"输入数据：`{data_root}`",
        f"输出目录：`{output_dir}`",
        "",
        "## 结论",
        "",
        recommendation,
        "",
        "## 组合汇总",
        "",
        "| FFT chain | guard(s) | pass | post-guard MAE | 60s MAE | full post MAE | std | worst sample | worst MAE | failure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {fft_chain} | {guard_seconds:.0f} | {passing_sample_count}/{sample_count} | "
            "{mean_post_guard_mae_bpm:.3f} | {mean_post_motion_60s_mae_bpm:.3f} | "
            "{mean_post_motion_full_mae_bpm:.3f} | {std_post_guard_mae_bpm:.3f} | "
            "{max_sample_cohort}/{max_sample_id} | {max_sample_mae_bpm:.3f} | "
            "{dominant_failure_reason} |".format(**row)
        )
    failing = [
        row for row in sample_rows if not bool(row.get("passes_post_guard_3bpm"))
    ]
    lines.extend(
        [
            "",
            "## 未达标样本",
            "",
            "| Cohort | Sample | FFT chain | guard(s) | MAE | 主失败原因 |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in sorted(
        failing,
        key=lambda r: (
            str(r.get("fft_chain")),
            float(r.get("guard_seconds", 0.0)),
            str(r.get("cohort")),
            str(r.get("sample_id")),
        ),
    )[:200]:
        lines.append(
            "| {cohort} | {sample_id} | {fft_chain} | {guard_seconds:.0f} | "
            "{post_guard_mae_bpm:.3f} | {primary_failure_reason} |".format(**row)
        )
    if len(failing) > 200:
        lines.append(f"| ... | ... | ... | ... | ... | 其余 {len(failing) - 200} 行见 CSV |")
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `motion_aware_fft_sample_metrics.csv`",
            "- `motion_aware_fft_window_metrics.csv`",
            "- `motion_aware_fft_aggregate_metrics.csv`",
            "- `png/`",
        ]
    )
    return "\n".join(lines) + "\n"


def _config_for_sample(sample: BaselineSample, base_config: V2RunConfig | None) -> V2RunConfig:
    if base_config is None:
        return V2RunConfig(
            data_path=sample.data_path,
            ref_path=sample.ref_path,
            reference_groups_order=(),
            spec_penalty_enable=False,
            postprocess_dynamics_enable=False,
        )
    values = dict(base_config.__dict__)
    values.update(
        {
            "data_path": sample.data_path,
            "ref_path": sample.ref_path,
            "reference_groups_order": (),
            "spec_penalty_enable": False,
            "postprocess_dynamics_enable": False,
        }
    )
    return V2RunConfig(**values)


def _track_fft_window(
    sig_fft: np.ndarray,
    fs: int,
    params: Any,
    history: list[float],
    tracking: DirectionalTrackingParams,
) -> tuple[float, SpectrumTrackingTrace]:
    history_arr = np.asarray(history + [0.0], dtype=float)
    return _process_spectrum_with_trace(
        sig_fft,
        np.zeros_like(sig_fft),
        fs,
        params,
        len(history),
        history_arr,
        False,
        tracking,
        path="motion_aware_fft_baseline",
        window_kind="rest",
    )


def _post_reset_forced_candidate_bpm(
    trace: SpectrumTrackingTrace,
    min_bpm_floor: float | None,
) -> float:
    floor = float(min_bpm_floor) if min_bpm_floor is not None else float("-inf")
    for candidate in getattr(trace, "candidate_peaks_bpm", ()):
        value = float(candidate)
        if math.isfinite(value) and value >= floor:
            return value
    raw = float(getattr(trace, "raw_candidate_hr_bpm", float("nan")))
    return raw if math.isfinite(raw) else float("nan")


def _topk_consensus_candidates(
    trace: SpectrumTrackingTrace,
    *,
    k: int,
    min_bpm_floor: float | None,
) -> tuple[float, ...]:
    floor = float(min_bpm_floor) if min_bpm_floor is not None else float("-inf")
    out: list[float] = []
    for candidate in getattr(trace, "candidate_peaks_bpm", ()):
        value = float(candidate)
        if math.isfinite(value) and value >= floor:
            out.append(value)
        if len(out) >= int(k):
            break
    if not out:
        raw = float(getattr(trace, "raw_candidate_hr_bpm", float("nan")))
        if math.isfinite(raw) and raw >= floor:
            out.append(raw)
    return tuple(out)


def _find_topk_consensus_peak(
    observations: list[tuple[float, ...]],
    *,
    tolerance_bpm: float,
) -> float:
    if not observations:
        return float("nan")
    current = observations[-1]
    previous = observations[:-1]
    if not current or not previous:
        return float("nan")
    tolerance = abs(float(tolerance_bpm))
    for candidate in current:
        if all(
            any(abs(float(candidate) - float(prior)) <= tolerance for prior in window)
            for window in previous
        ):
            return float(candidate)
    return float("nan")


def _candidate_rank(candidates: tuple[float, ...], selected_bpm: float) -> int:
    selected = float(selected_bpm)
    for idx, candidate in enumerate(candidates, start=1):
        if math.isclose(float(candidate), selected, abs_tol=1e-6):
            return idx
    return 0


def _attach_consensus_trace_fields(
    trace: SpectrumTrackingTrace,
    *,
    status: str,
    selected_bpm: float,
    failure_reason: str,
    window_count: int,
    k: int,
) -> None:
    trace.consensus_status = str(status)
    trace.consensus_selected_bpm = float(selected_bpm)
    trace.consensus_failure_reason = str(failure_reason)
    trace.consensus_window_count = int(window_count)
    trace.consensus_k = int(k)


def _baseline_window_stage(
    center_s: float,
    motion_segment: dict[str, float] | None,
    guard_end_s: float,
) -> str:
    if motion_segment is None:
        return "rest"
    if center_s <= float(motion_segment["end_s"]) + 1e-9:
        return _classify_window_stage(center_s, motion_segment, True)
    if center_s <= guard_end_s + 1e-9:
        return "post_motion_guard"
    return "post_motion_reacquire"


def _smooth_rows_in_place(rows: list[dict[str, Any]], cfg: V2RunConfig) -> None:
    if not rows:
        return
    values = np.asarray([row["fft_baseline_bpm"] for row in rows], dtype=float)
    smoothed = smoothdata_movmedian(values, int(cfg.smooth_win_len))
    for row, value in zip(rows, smoothed, strict=False):
        row["fft_baseline_bpm"] = float(value)


def _mae(rows: list[dict[str, Any]]) -> float:
    values = [
        abs(_as_float(row.get("fft_baseline_bpm")) - _as_float(row.get("ref_bpm")))
        for row in rows
    ]
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _max_abs_error(rows: list[dict[str, Any]]) -> float:
    values = [abs(_as_float(row.get("error_bpm"))) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return float(max(finite)) if finite else float("nan")


def _failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = ("held_previous", "low_lock", "high_lock", "borderline", "accurate", "no_valid_peak")
    counts = {key: 0 for key in keys}
    for row in rows:
        reason = str(row.get("failure_reason") or classify_window_failure(row))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _primary_failure_reason(counts: dict[str, int]) -> str:
    for key in ("held_previous", "low_lock", "high_lock", "no_valid_peak", "borderline"):
        if counts.get(key, 0) > 0:
            return key
    return "accurate"


def _dominant_sample_failure_reason(rows: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("primary_failure_reason", ""))
        if reason and reason != "accurate":
            counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return "accurate"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _finite_mean(values: Iterable[Any]) -> float:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort", action="append", dest="cohorts")
    parser.add_argument("--fft-chain", action="append", dest="fft_chains")
    parser.add_argument("--guard-seconds", type=float, action="append", dest="guards")
    args = parser.parse_args(argv)
    run_motion_aware_fft_baseline_study(
        args.data_root,
        args.output_dir,
        cohorts=tuple(args.cohorts or ("LYX", "TS")),
        fft_chains=tuple(args.fft_chains or FFT_CHAINS),
        guard_seconds_values=tuple(args.guards or DEFAULT_GUARD_SECONDS),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
