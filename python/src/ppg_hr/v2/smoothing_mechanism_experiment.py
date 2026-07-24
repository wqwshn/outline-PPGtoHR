"""LYX 平滑机制前置实验。

该模块只比较固定参数锚点下的 ``smooth_win_len``，并强制停在人工审核门。
它不会启动独立 BO 或 K 折 BO。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from .post_motion_reset_fft_reacquire import load_lite_report_config
from .reference_groups import method_label
from .reference_overlap import aligned_reference_bpm
from .solver import V2SolverResult, solve_v2

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


DEFAULT_SMOOTH_DURATIONS_S = (1, 3, 5, 7, 9, 11)
ANCHOR_TYPES = ("independent_bo", "shared_holdout")
ANCHOR_TITLES = {
    "independent_bo": "Archived independent-BO anchor",
    "shared_holdout": "Shared K-fold holdout anchor",
}
REFERENCE_COLOR = "#333333"
FINAL_COLOR = "#D97732"
CONTROL_COLOR = "#2C7FB8"
BASELINE_COLOR = "#8A8A8A"
MOTION_COLOR = "#DCE6EC"


@dataclass(frozen=True)
class SmoothingAnchor:
    """一条记录的一套不可搜索参数锚点。"""

    anchor_type: str
    sample: str
    scene: str
    report_path: Path
    error_csv: Path
    data_path: Path
    ref_path: Path
    source_smooth_win_len: int
    source_time_bias_s: float
    fold_id: str = ""
    summary_fft_aae_bpm: float = math.nan
    summary_final_aae_bpm: float = math.nan


def validate_smoothing_durations(durations: Iterable[int]) -> tuple[int, ...]:
    """验证平滑候选；1 s 必须作为无跨窗口平滑对照。"""

    values = tuple(int(value) for value in durations)
    if not values or values[0] != 1 or 1 not in values:
        raise ValueError("平滑候选必须包含并以 1 s 对照开始")
    if len(set(values)) != len(values):
        raise ValueError("平滑候选不能重复")
    if any(value <= 0 or value % 2 == 0 for value in values):
        raise ValueError("平滑候选必须是正奇数")
    if tuple(sorted(values)) != values:
        raise ValueError("平滑候选必须严格递增")
    return values


def discover_smoothing_anchors(
    independent_batch_dir: Path | str,
    generalization_dir: Path | str,
    *,
    expected_record_count: int = 24,
) -> list[SmoothingAnchor]:
    """发现并严格配对独立 BO 与三折留出参数锚点。"""

    independent_root = Path(independent_batch_dir).resolve()
    generalization_root = Path(generalization_dir).resolve()
    independent: dict[str, SmoothingAnchor] = {}
    for report_path in sorted((independent_root / "json").glob("*.json")):
        payload = _load_json(report_path)
        anchor = _anchor_from_report(
            anchor_type="independent_bo",
            report_path=report_path,
            error_csv=independent_root / "csv" / f"{report_path.stem}-error.csv",
            payload=payload,
            scene="",
            fold_id="",
            summary_fft=_nested_float(payload, "err_stats", "fft_aae_bpm"),
            summary_final=_nested_float(payload, "err_stats", "final_aae_bpm"),
        )
        _insert_unique(independent, anchor)

    summary_path = generalization_root / "v2_generalization_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    shared: dict[str, SmoothingAnchor] = {}
    with summary_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if (
                str(row.get("evaluation_mode", "")).strip() != "k_fold_holdout"
                or str(row.get("split", "")).strip() != "test"
                or str(row.get("status", "")).strip() != "ok"
            ):
                continue
            report_path = Path(str(row.get("report_path", ""))).resolve()
            payload = _load_json(report_path)
            anchor = _anchor_from_report(
                anchor_type="shared_holdout",
                report_path=report_path,
                error_csv=Path(str(row.get("error_csv", ""))).resolve(),
                payload=payload,
                scene=str(row.get("motion_type", "")).strip(),
                fold_id=str(row.get("fold_id", "")).strip(),
                summary_fft=_as_float(row.get("fft_aae_bpm")),
                summary_final=_as_float(row.get("final_aae_bpm")),
            )
            summary_sample = Path(str(row.get("sample", ""))).stem
            if summary_sample and summary_sample != anchor.sample:
                raise ValueError(
                    f"summary/report 样本身份不一致：{summary_sample} != {anchor.sample}"
                )
            _insert_unique(shared, anchor)

    if len(independent) != expected_record_count:
        raise ValueError(
            f"独立 BO 锚点数量应为 {expected_record_count}，实际为 {len(independent)}"
        )
    if len(shared) != expected_record_count:
        raise ValueError(
            f"三折留出锚点数量应为 {expected_record_count}，实际为 {len(shared)}"
        )
    if set(independent) != set(shared):
        independent_only = sorted(set(independent) - set(shared))
        shared_only = sorted(set(shared) - set(independent))
        raise ValueError(
            "两套锚点样本集合不一致；"
            f"仅独立 BO={independent_only}；仅三折留出={shared_only}"
        )
    rows = [*independent.values(), *shared.values()]
    return sorted(rows, key=lambda row: (ANCHOR_TYPES.index(row.anchor_type), row.sample))


def compute_smoothing_metrics(
    hr: np.ndarray,
    *,
    time_bias: float,
    reference_bounds: tuple[float, float] | None = None,
    reliable_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """计算全记录、严格运动段和变化形态指标。"""

    arr = np.asarray(hr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError("HR 必须是至少五列的二维数组")
    if arr.shape[0] == 0:
        return _empty_metrics()
    ref = aligned_reference_bpm(
        arr,
        float(time_bias),
        reference_bounds=reference_bounds,
        mask_outside_bounds=True,
    )
    aligned_time = arr[:, 0] + float(time_bias)
    final = arr[:, 3]
    reset_fft = arr[:, 2]
    motion = arr[:, 4] >= 0.5
    finite_common = np.isfinite(aligned_time) & np.isfinite(ref)
    valid_final = finite_common & np.isfinite(final)
    valid_fft = finite_common & np.isfinite(reset_fft)
    valid_motion_final = valid_final & motion
    valid_motion_fft = valid_fft & motion
    if reliable_mask is None:
        reliable = np.ones(arr.shape[0], dtype=bool)
    else:
        reliable = np.asarray(reliable_mask, dtype=bool)
        if reliable.shape != (arr.shape[0],):
            raise ValueError("reliable_mask 长度必须与 HR 行数一致")
    valid_reliable_final = valid_final & reliable
    valid_reliable_fft = valid_fft & reliable
    valid_reliable_motion_final = valid_motion_final & reliable
    valid_reliable_motion_fft = valid_motion_fft & reliable
    ref_span = _percentile_span(ref[finite_common])
    final_span = _percentile_span(final[valid_final])
    transitions = _transition_metrics(aligned_time, ref, final, valid_final)
    return {
        "full_final_mae_bpm": _mae(final, ref, valid_final),
        "motion_final_mae_bpm": _mae(final, ref, valid_motion_final),
        "full_reset_fft_mae_bpm": _mae(reset_fft, ref, valid_fft),
        "motion_reset_fft_mae_bpm": _mae(reset_fft, ref, valid_motion_fft),
        "full_final_mae_reliable_bpm": _mae(
            final, ref, valid_reliable_final
        ),
        "motion_final_mae_reliable_bpm": _mae(
            final, ref, valid_reliable_motion_final
        ),
        "full_reset_fft_mae_reliable_bpm": _mae(
            reset_fft, ref, valid_reliable_fft
        ),
        "motion_reset_fft_mae_reliable_bpm": _mae(
            reset_fft, ref, valid_reliable_motion_fft
        ),
        "valid_full_windows": int(np.count_nonzero(valid_final)),
        "valid_motion_windows": int(np.count_nonzero(valid_motion_final)),
        "valid_reliable_windows": int(np.count_nonzero(valid_reliable_final)),
        "valid_reliable_motion_windows": int(
            np.count_nonzero(valid_reliable_motion_final)
        ),
        "nonfinite_final_windows": int(np.count_nonzero(~np.isfinite(final))),
        "nonfinite_reset_fft_windows": int(np.count_nonzero(~np.isfinite(reset_fft))),
        "reference_span_bpm": ref_span,
        "final_span_bpm": final_span,
        "span_compression_bpm": final_span - ref_span,
        **transitions,
    }


def write_human_decision_template(
    output_dir: Path | str,
    *,
    evidence_shortlist: Sequence[int] = (),
) -> Path:
    """写入不可自动放行的人工决定模板。"""

    path = Path(output_dir) / "human_smoothing_decision.json"
    payload = {
        "status": "pending_human_review",
        "selected_smooth_win_len_s": None,
        "formal_experiment_authorized": False,
        "evidence_shortlist_s": [int(value) for value in evidence_shortlist],
        "allowed_decisions": [
            "approve_one_duration",
            "request_supplement_or_rerun",
            "reject_unified_duration",
        ],
        "reviewer": None,
        "reviewed_at": None,
        "rationale": "",
    }
    _write_json(path, payload)
    return path


def run_smoothing_mechanism_experiment(
    *,
    independent_batch_dir: Path | str,
    generalization_dir: Path | str,
    output_dir: Path | str,
    durations_s: Iterable[int] = DEFAULT_SMOOTH_DURATIONS_S,
    expected_record_count: int = 24,
    sample_names: Iterable[str] | None = None,
    workers: int = 1,
) -> dict[str, Path]:
    """运行双锚点平滑实验并生成待人工审核的完整证据包。"""

    durations = validate_smoothing_durations(durations_s)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures_dir = output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    anchors = discover_smoothing_anchors(
        independent_batch_dir,
        generalization_dir,
        expected_record_count=expected_record_count,
    )
    if sample_names is not None:
        requested = {Path(str(value)).stem for value in sample_names}
        available = {anchor.sample for anchor in anchors}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"请求的样本不存在：{missing}")
        anchors = [anchor for anchor in anchors if anchor.sample in requested]
        if not anchors:
            raise ValueError("样本过滤后没有可运行的参数锚点")
    if workers <= 0:
        raise ValueError("workers 必须是正整数")
    _write_anchor_manifest(output / "anchor_manifest.csv", anchors)
    identity_rows = audit_method_identity(anchors)
    _write_rows(output / "method_identity_audit.csv", identity_rows)
    if not identity_rows or not all(bool(row["audit_ok"]) for row in identity_rows):
        raise RuntimeError("方法身份对账未全部通过，拒绝启动平滑求解")

    run_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    total = len(anchors) * len(durations)
    tasks = [(anchor, duration) for anchor in anchors for duration in durations]
    if workers == 1:
        completed = (
            (anchor, duration, _solve_anchor_duration(anchor, duration))
            for anchor, duration in tasks
        )
        for ordinal, (anchor, duration, result_pair) in enumerate(completed, start=1):
            print(
                f"[{ordinal:03d}/{total}] {anchor.anchor_type} "
                f"{anchor.sample} smooth={duration}s",
                flush=True,
            )
            run_row, trajectories = result_pair
            run_rows.append(run_row)
            trajectory_rows.extend(trajectories)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_solve_anchor_duration, anchor, duration): (
                    anchor,
                    duration,
                )
                for anchor, duration in tasks
            }
            for ordinal, future in enumerate(as_completed(futures), start=1):
                anchor, duration = futures[future]
                run_row, trajectories = future.result()
                print(
                    f"[{ordinal:03d}/{total}] completed {anchor.anchor_type} "
                    f"{anchor.sample} smooth={duration}s",
                    flush=True,
                )
                run_rows.append(run_row)
                trajectory_rows.extend(trajectories)

    run_frame = _add_paired_deltas(pd.DataFrame(run_rows))
    trajectory_frame = pd.DataFrame(trajectory_rows)
    results_path = output / "smoothing_results.csv"
    run_frame.to_csv(results_path, index=False, encoding="utf-8-sig")
    trajectories_path = output / "smoothing_trajectories.csv.gz"
    _write_gzip_frame(trajectories_path, trajectory_frame)

    duration_summary = _duration_summary(run_frame)
    duration_summary_path = output / "smoothing_duration_summary.csv"
    duration_summary.to_csv(duration_summary_path, index=False, encoding="utf-8-sig")
    scene_summary = _scene_summary(run_frame)
    scene_summary_path = output / "smoothing_scene_summary.csv"
    scene_summary.to_csv(scene_summary_path, index=False, encoding="utf-8-sig")
    agreement = _anchor_direction_agreement(run_frame)
    agreement_path = output / "smoothing_anchor_agreement.csv"
    agreement.to_csv(agreement_path, index=False, encoding="utf-8-sig")

    shortlist = _evidence_shortlist(duration_summary)
    write_human_decision_template(output, evidence_shortlist=shortlist)
    figure_paths = render_smoothing_figures(
        run_frame,
        trajectory_frame,
        figures_dir,
        durations=durations,
    )
    report_path = output / "smoothing_review_report.md"
    _write_review_report(
        report_path,
        run_frame=run_frame,
        duration_summary=duration_summary,
        agreement=agreement,
        identity_rows=identity_rows,
        shortlist=shortlist,
        figure_paths=figure_paths,
    )
    manifest_path = output / "run_manifest.json"
    _write_run_manifest(
        manifest_path,
        anchors=anchors,
        durations=durations,
        output_dir=output,
        independent_batch_dir=Path(independent_batch_dir).resolve(),
        generalization_dir=Path(generalization_dir).resolve(),
        workers=workers,
    )
    return {
        "output_dir": output,
        "results": results_path,
        "duration_summary": duration_summary_path,
        "scene_summary": scene_summary_path,
        "anchor_agreement": agreement_path,
        "trajectories": trajectories_path,
        "identity_audit": output / "method_identity_audit.csv",
        "report": report_path,
        "decision": output / "human_smoothing_decision.json",
        "manifest": manifest_path,
    }


def _solve_anchor_duration(
    anchor: SmoothingAnchor,
    duration: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(anchor.report_path)
    cfg = load_lite_report_config(payload)
    if not math.isclose(float(cfg.window_step_seconds), 1.0, abs_tol=1e-9):
        raise ValueError(
            f"{anchor.sample} 的窗口步长不是 1 s，不能把 smooth_win_len 解释为秒"
        )
    cfg = replace(
        cfg,
        smooth_win_len=int(duration),
        time_bias=5.0,
        lms_mu_min=1e-6,
    )
    started = time.perf_counter()
    result = solve_v2(cfg)
    runtime_s = time.perf_counter() - started
    metrics = compute_smoothing_metrics(
        result.HR,
        time_bias=cfg.time_bias,
        reference_bounds=_reference_bounds_from_result(result),
        reliable_mask=_reliable_mask_from_result(result),
    )
    run_row = {
        "anchor_type": anchor.anchor_type,
        "sample": anchor.sample,
        "scene": anchor.scene,
        "fold_id": anchor.fold_id,
        "smooth_duration_s": duration,
        "future_lookahead_s": (duration - 1) / 2.0,
        "forced_time_bias_s": cfg.time_bias,
        "forced_lms_mu_min": cfg.lms_mu_min,
        "source_smooth_win_len": anchor.source_smooth_win_len,
        "source_time_bias_s": anchor.source_time_bias_s,
        "runtime_s": runtime_s,
        "status": "ok",
        **metrics,
    }
    return (
        run_row,
        _trajectory_rows(anchor, duration, result, time_bias=cfg.time_bias),
    )


def audit_method_identity(anchors: Sequence[SmoothingAnchor]) -> list[dict[str, Any]]:
    """逐条对账归档 error CSV 中的 Final 与 reset FFT 身份及数值。"""

    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        payload = _load_json(anchor.report_path)
        adaptive_filter = str(payload.get("adaptive_filter", "")).strip()
        groups = tuple(str(value) for value in payload.get("reference_groups_order", []))
        expected_final = method_label(adaptive_filter, groups)
        methods = _read_method_rows(anchor.error_csv)
        fft_source = next(
            (name for name in ("reset FFT", "FFT") if name in methods),
            "",
        )
        final_source = expected_final if expected_final in methods else ""
        fft_total = _row_float(methods.get(fft_source), "total_aae")
        final_total = _row_float(methods.get(final_source), "total_aae")
        fft_delta = fft_total - anchor.summary_fft_aae_bpm
        final_delta = final_total - anchor.summary_final_aae_bpm
        identity_ok = bool(
            fft_source
            and final_source == expected_final
        )
        numeric_reconciled = bool(
            np.isfinite(fft_delta)
            and np.isfinite(final_delta)
            and abs(fft_delta) <= 5e-4
            and abs(final_delta) <= 5e-4
        )
        unreliable_windows = int(payload.get("unreliable_windows", 0) or 0)
        if numeric_reconciled:
            difference_reason = ""
        elif unreliable_windows > 0:
            difference_reason = (
                "solver_summary_excludes_unreliable_windows_"
                "while_error_csv_uses_interpolated_visible_curve"
            )
        else:
            difference_reason = "unexplained_numeric_difference"
        audit_ok = bool(
            identity_ok
            and (
                numeric_reconciled
                or difference_reason.startswith("solver_summary_excludes_")
            )
        )
        rows.append(
            {
                "anchor_type": anchor.anchor_type,
                "sample": anchor.sample,
                "scene": anchor.scene,
                "fold_id": anchor.fold_id,
                "error_csv": str(anchor.error_csv),
                "expected_final_method": expected_final,
                "resolved_final_method": final_source,
                "resolved_fft_method": fft_source,
                "final_total_aae_csv": final_total,
                "final_total_aae_summary": anchor.summary_final_aae_bpm,
                "final_delta_bpm": final_delta,
                "fft_total_aae_csv": fft_total,
                "fft_total_aae_summary": anchor.summary_fft_aae_bpm,
                "fft_delta_bpm": fft_delta,
                "unreliable_windows": unreliable_windows,
                "identity_ok": identity_ok,
                "numeric_reconciled": numeric_reconciled,
                "numeric_difference_reason": difference_reason,
                "audit_ok": audit_ok,
            }
        )
    return rows


def render_smoothing_figures(
    run_frame: pd.DataFrame,
    trajectory_frame: pd.DataFrame,
    output_dir: Path | str,
    *,
    durations: Sequence[int],
) -> list[Path]:
    """渲染配对变化、场景分层和代表性同轴曲线。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    paths = [
        *_render_paired_delta_figure(run_frame, output, durations),
        *_render_scene_heatmap(run_frame, output, durations),
    ]
    selections = _representative_selections(run_frame)
    for label, anchor_type, sample in selections:
        paths.extend(
            _render_timeseries_grid(
                run_frame,
                trajectory_frame,
                output,
                durations=durations,
                label=label,
                anchor_type=anchor_type,
                sample=sample,
            )
        )
    return paths


def _anchor_from_report(
    *,
    anchor_type: str,
    report_path: Path,
    error_csv: Path,
    payload: dict[str, Any],
    scene: str,
    fold_id: str,
    summary_fft: float,
    summary_final: float,
) -> SmoothingAnchor:
    data_path = Path(str(payload.get("data_path", ""))).resolve()
    ref_path = Path(str(payload.get("ref_path", ""))).resolve()
    sample = data_path.stem
    if not sample:
        raise ValueError(f"报告缺少 data_path：{report_path}")
    inferred_scene = _scene_from_sample(sample)
    if scene and scene != inferred_scene:
        raise ValueError(
            f"场景身份不一致：summary={scene}, sample={sample}, inferred={inferred_scene}"
        )
    for required in (report_path, error_csv, data_path, ref_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    best_params = payload.get("best_params")
    best = best_params if isinstance(best_params, dict) else {}
    source_smooth = int(best.get("smooth_win_len", payload.get("smooth_win_len", 7)))
    source_time_bias = float(best.get("time_bias", payload.get("time_bias", 5.0)))
    return SmoothingAnchor(
        anchor_type=anchor_type,
        sample=sample,
        scene=scene or inferred_scene,
        report_path=report_path.resolve(),
        error_csv=error_csv.resolve(),
        data_path=data_path,
        ref_path=ref_path,
        source_smooth_win_len=source_smooth,
        source_time_bias_s=source_time_bias,
        fold_id=fold_id,
        summary_fft_aae_bpm=summary_fft,
        summary_final_aae_bpm=summary_final,
    )


def _insert_unique(target: dict[str, SmoothingAnchor], anchor: SmoothingAnchor) -> None:
    if anchor.sample in target:
        raise ValueError(
            f"{anchor.anchor_type} 存在重复样本锚点：{anchor.sample}"
        )
    target[anchor.sample] = anchor


def _scene_from_sample(sample: str) -> str:
    match = re.match(r"([A-Za-z_]+?)(?=\d)", sample)
    if not match:
        raise ValueError(f"无法从样本名解析场景：{sample}")
    return match.group(1).rstrip("_").lower()


def _empty_metrics() -> dict[str, float | int]:
    names = (
        "full_final_mae_bpm",
        "motion_final_mae_bpm",
        "full_reset_fft_mae_bpm",
        "motion_reset_fft_mae_bpm",
        "full_final_mae_reliable_bpm",
        "motion_final_mae_reliable_bpm",
        "full_reset_fft_mae_reliable_bpm",
        "motion_reset_fft_mae_reliable_bpm",
        "reference_span_bpm",
        "final_span_bpm",
        "span_compression_bpm",
        "up_reference_strength_bpm_per_s",
        "up_final_strength_bpm_per_s",
        "up_strength_retention",
        "up_timing_offset_s",
        "down_reference_strength_bpm_per_s",
        "down_final_strength_bpm_per_s",
        "down_strength_retention",
        "down_timing_offset_s",
    )
    result: dict[str, float | int] = {name: math.nan for name in names}
    result.update(
        {
            "valid_full_windows": 0,
            "valid_motion_windows": 0,
            "valid_reliable_windows": 0,
            "valid_reliable_motion_windows": 0,
            "nonfinite_final_windows": 0,
            "nonfinite_reset_fft_windows": 0,
        }
    )
    return result


def _transition_metrics(
    times: np.ndarray,
    reference: np.ndarray,
    final: np.ndarray,
    valid: np.ndarray,
    *,
    half_window_s: float = 2.5,
    search_radius_s: float = 12.0,
) -> dict[str, float]:
    mask = valid & np.isfinite(times) & np.isfinite(reference) & np.isfinite(final)
    if np.count_nonzero(mask) < 8:
        return {
            "up_reference_strength_bpm_per_s": math.nan,
            "up_final_strength_bpm_per_s": math.nan,
            "up_strength_retention": math.nan,
            "up_timing_offset_s": math.nan,
            "down_reference_strength_bpm_per_s": math.nan,
            "down_final_strength_bpm_per_s": math.nan,
            "down_strength_retention": math.nan,
            "down_timing_offset_s": math.nan,
        }
    t = np.asarray(times[mask], dtype=float)
    ref = np.asarray(reference[mask], dtype=float)
    pred = np.asarray(final[mask], dtype=float)
    order = np.argsort(t)
    t, ref, pred = t[order], ref[order], pred[order]
    unique = np.concatenate(([True], np.diff(t) > 1e-9))
    t, ref, pred = t[unique], ref[unique], pred[unique]
    if t.size < 8 or t[-1] - t[0] < 2 * half_window_s:
        return _transition_metrics(
            np.asarray([]),
            np.asarray([]),
            np.asarray([]),
            np.asarray([], dtype=bool),
        )
    centers = t[
        (t >= t[0] + half_window_s) & (t <= t[-1] - half_window_s)
    ]
    ref_slope = (
        np.interp(centers + half_window_s, t, ref)
        - np.interp(centers - half_window_s, t, ref)
    ) / (2 * half_window_s)
    pred_slope = (
        np.interp(centers + half_window_s, t, pred)
        - np.interp(centers - half_window_s, t, pred)
    ) / (2 * half_window_s)
    return {
        **_direction_transition(
            centers,
            ref_slope,
            pred_slope,
            direction="up",
            search_radius_s=search_radius_s,
        ),
        **_direction_transition(
            centers,
            ref_slope,
            pred_slope,
            direction="down",
            search_radius_s=search_radius_s,
        ),
    }


def _direction_transition(
    centers: np.ndarray,
    ref_slope: np.ndarray,
    pred_slope: np.ndarray,
    *,
    direction: str,
    search_radius_s: float,
) -> dict[str, float]:
    prefix = "up" if direction == "up" else "down"
    ref_idx = int(np.nanargmax(ref_slope) if direction == "up" else np.nanargmin(ref_slope))
    ref_strength = float(ref_slope[ref_idx])
    near = np.abs(centers - centers[ref_idx]) <= search_radius_s
    candidate_indices = np.flatnonzero(near)
    if candidate_indices.size == 0:
        pred_idx = ref_idx
    else:
        local = pred_slope[candidate_indices]
        local_idx = int(np.nanargmax(local) if direction == "up" else np.nanargmin(local))
        pred_idx = int(candidate_indices[local_idx])
    pred_strength = float(pred_slope[pred_idx])
    denominator = abs(ref_strength)
    retention = abs(pred_strength) / denominator if denominator > 1e-9 else math.nan
    return {
        f"{prefix}_reference_strength_bpm_per_s": abs(ref_strength),
        f"{prefix}_final_strength_bpm_per_s": abs(pred_strength),
        f"{prefix}_strength_retention": retention,
        f"{prefix}_timing_offset_s": float(centers[pred_idx] - centers[ref_idx]),
    }


def _mae(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    values = np.abs(np.asarray(pred)[mask] - np.asarray(ref)[mask])
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else math.nan


def _percentile_span(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan
    return float(np.percentile(finite, 95) - np.percentile(finite, 5))


def _reference_bounds_from_result(
    result: V2SolverResult,
) -> tuple[float, float] | None:
    overlap = result.metadata.get("reference_overlap")
    if not isinstance(overlap, dict):
        return None
    start = _as_float(overlap.get("ref_start_s"))
    end = _as_float(overlap.get("ref_end_s"))
    if not np.isfinite(start) or not np.isfinite(end):
        return None
    return start, end


def _reliable_mask_from_result(result: V2SolverResult) -> np.ndarray:
    reliable_by_time = {
        round(float(row.get("center_s", math.nan)), 9): bool(
            row.get("reliable", True)
        )
        for row in result.window_table
        if np.isfinite(_as_float(row.get("center_s")))
    }
    return np.asarray(
        [
            reliable_by_time.get(round(float(center_s), 9), True)
            for center_s in np.asarray(result.HR)[:, 0]
        ],
        dtype=bool,
    )


def _trajectory_rows(
    anchor: SmoothingAnchor,
    duration: int,
    result: V2SolverResult,
    *,
    time_bias: float,
) -> list[dict[str, Any]]:
    hr = np.asarray(result.HR, dtype=float)
    if hr.size == 0:
        return []
    ref = aligned_reference_bpm(
        hr,
        time_bias,
        reference_bounds=_reference_bounds_from_result(result),
        mask_outside_bounds=True,
    )
    rows: list[dict[str, Any]] = []
    for index, values in enumerate(hr):
        rows.append(
            {
                "anchor_type": anchor.anchor_type,
                "sample": anchor.sample,
                "scene": anchor.scene,
                "smooth_duration_s": duration,
                "center_time_s": values[0],
                "aligned_time_s": values[0] + time_bias,
                "reference_bpm": ref[index],
                "reset_fft_bpm": values[2],
                "final_bpm": values[3],
                "is_motion": int(values[4] >= 0.5),
            }
        )
    return rows


def _add_paired_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["anchor_type", "sample"]
    baseline = (
        frame.loc[frame["smooth_duration_s"] == 1, keys + [
            "full_final_mae_bpm",
            "motion_final_mae_bpm",
        ]]
        .rename(
            columns={
                "full_final_mae_bpm": "control_1s_full_final_mae_bpm",
                "motion_final_mae_bpm": "control_1s_motion_final_mae_bpm",
            }
        )
    )
    if baseline.duplicated(keys).any() or baseline.shape[0] * len(
        frame["smooth_duration_s"].unique()
    ) != frame.shape[0]:
        raise ValueError("每个锚点/样本必须且只能有一条 1 s 对照")
    out = frame.merge(baseline, on=keys, how="left", validate="many_to_one")
    out["delta_full_final_mae_vs_1s_bpm"] = (
        out["full_final_mae_bpm"] - out["control_1s_full_final_mae_bpm"]
    )
    out["delta_motion_final_mae_vs_1s_bpm"] = (
        out["motion_final_mae_bpm"] - out["control_1s_motion_final_mae_bpm"]
    )
    best = (
        out.groupby(keys, as_index=False)["motion_final_mae_bpm"]
        .min()
        .rename(columns={"motion_final_mae_bpm": "best_motion_final_mae_bpm"})
    )
    out = out.merge(best, on=keys, how="left", validate="many_to_one")
    out["delta_motion_final_mae_vs_record_best_bpm"] = (
        out["motion_final_mae_bpm"] - out["best_motion_final_mae_bpm"]
    )
    return out.sort_values(keys + ["smooth_duration_s"]).reset_index(drop=True)


def _duration_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for anchor_type in [*ANCHOR_TYPES, "all"]:
        subset = frame if anchor_type == "all" else frame[frame["anchor_type"] == anchor_type]
        for duration, group in subset.groupby("smooth_duration_s", sort=True):
            delta = group["delta_motion_final_mae_vs_1s_bpm"].to_numpy(dtype=float)
            rows.append(
                {
                    "anchor_type": anchor_type,
                    "smooth_duration_s": int(duration),
                    "future_lookahead_s": (int(duration) - 1) / 2.0,
                    "record_count": int(group.shape[0]),
                    "mean_motion_final_mae_bpm": _finite_stat(
                        group["motion_final_mae_bpm"], np.mean
                    ),
                    "median_motion_final_mae_bpm": _finite_stat(
                        group["motion_final_mae_bpm"], np.median
                    ),
                    "max_motion_final_mae_bpm": _finite_stat(
                        group["motion_final_mae_bpm"], np.max
                    ),
                    "mean_delta_motion_vs_1s_bpm": _finite_stat(delta, np.mean),
                    "median_delta_motion_vs_1s_bpm": _finite_stat(delta, np.median),
                    "max_delta_motion_vs_1s_bpm": _finite_stat(delta, np.max),
                    "p90_delta_motion_vs_1s_bpm": _finite_percentile(delta, 90),
                    "improved_fraction": _finite_fraction(delta < -1e-9, delta),
                    "within_0p5bpm_fraction": _finite_fraction(
                        np.abs(delta) <= 0.5, delta
                    ),
                    "mean_full_final_mae_bpm": _finite_stat(
                        group["full_final_mae_bpm"], np.mean
                    ),
                    "mean_span_compression_bpm": _finite_stat(
                        group["span_compression_bpm"], np.mean
                    ),
                    "mean_up_strength_retention": _finite_stat(
                        group["up_strength_retention"], np.mean
                    ),
                    "mean_down_strength_retention": _finite_stat(
                        group["down_strength_retention"], np.mean
                    ),
                    "median_abs_up_timing_offset_s": _finite_stat(
                        np.abs(group["up_timing_offset_s"]), np.median
                    ),
                    "median_abs_down_timing_offset_s": _finite_stat(
                        np.abs(group["down_timing_offset_s"]), np.median
                    ),
                    "mean_runtime_s": _finite_stat(group["runtime_s"], np.mean),
                    "nonfinite_output_windows": int(
                        group["nonfinite_final_windows"].sum()
                        + group["nonfinite_reset_fft_windows"].sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _scene_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (anchor_type, scene, duration), group in frame.groupby(
        ["anchor_type", "scene", "smooth_duration_s"],
        sort=True,
    ):
        delta = group["delta_motion_final_mae_vs_1s_bpm"]
        rows.append(
            {
                "anchor_type": anchor_type,
                "scene": scene,
                "smooth_duration_s": int(duration),
                "record_count": int(group.shape[0]),
                "mean_motion_final_mae_bpm": _finite_stat(
                    group["motion_final_mae_bpm"], np.mean
                ),
                "median_motion_final_mae_bpm": _finite_stat(
                    group["motion_final_mae_bpm"], np.median
                ),
                "max_motion_final_mae_bpm": _finite_stat(
                    group["motion_final_mae_bpm"], np.max
                ),
                "mean_delta_motion_vs_1s_bpm": _finite_stat(delta, np.mean),
                "median_delta_motion_vs_1s_bpm": _finite_stat(delta, np.median),
                "max_delta_motion_vs_1s_bpm": _finite_stat(delta, np.max),
            }
        )
    return pd.DataFrame(rows)


def _anchor_direction_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot(
        index=["sample", "smooth_duration_s"],
        columns="anchor_type",
        values="delta_motion_final_mae_vs_1s_bpm",
    ).reset_index()
    rows: list[dict[str, Any]] = []
    for duration, group in pivot.groupby("smooth_duration_s", sort=True):
        left = group["independent_bo"].to_numpy(dtype=float)
        right = group["shared_holdout"].to_numpy(dtype=float)
        valid = np.isfinite(left) & np.isfinite(right)
        equivalent = (np.abs(left) <= 0.1) & (np.abs(right) <= 0.1)
        same_sign = np.sign(left) == np.sign(right)
        agree = valid & (equivalent | same_sign)
        rows.append(
            {
                "smooth_duration_s": int(duration),
                "paired_record_count": int(np.count_nonzero(valid)),
                "direction_agreement_rate_0p1bpm": (
                    float(np.mean(agree[valid])) if np.any(valid) else math.nan
                ),
                "median_abs_anchor_delta_difference_bpm": (
                    float(np.median(np.abs(left[valid] - right[valid])))
                    if np.any(valid)
                    else math.nan
                ),
                "max_abs_anchor_delta_difference_bpm": (
                    float(np.max(np.abs(left[valid] - right[valid])))
                    if np.any(valid)
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _evidence_shortlist(duration_summary: pd.DataFrame) -> tuple[int, ...]:
    rows = duration_summary[
        (duration_summary["anchor_type"] == "all")
        & (duration_summary["smooth_duration_s"] > 1)
    ].copy()
    if rows.empty:
        return ()
    rows = rows.sort_values(
        [
            "max_delta_motion_vs_1s_bpm",
            "mean_delta_motion_vs_1s_bpm",
            "smooth_duration_s",
        ]
    )
    return tuple(int(value) for value in rows.head(3)["smooth_duration_s"])


def _render_paired_delta_figure(
    frame: pd.DataFrame,
    output: Path,
    durations: Sequence[int],
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharex=True, sharey=True)
    for ax, anchor_type in zip(axes, ANCHOR_TYPES, strict=True):
        subset = frame[frame["anchor_type"] == anchor_type]
        for _, group in subset.groupby("sample"):
            group = group.sort_values("smooth_duration_s")
            ax.plot(
                group["smooth_duration_s"],
                group["delta_motion_final_mae_vs_1s_bpm"],
                color="#B5B5B5",
                linewidth=0.55,
                alpha=0.55,
                marker="o",
                markersize=1.8,
            )
        median = subset.groupby("smooth_duration_s")[
            "delta_motion_final_mae_vs_1s_bpm"
        ].median()
        ax.plot(
            median.index,
            median.values,
            color=FINAL_COLOR,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="Record median",
            zorder=5,
        )
        ax.axhline(0, color=REFERENCE_COLOR, linewidth=0.8, linestyle="--")
        ax.set_title(ANCHOR_TITLES[anchor_type])
        ax.set_xticks(durations)
        ax.set_xlabel("Smoothing duration (s)")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)
    axes[0].set_ylabel("Strict-motion Final MAE delta vs 1 s (BPM)")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Paired smoothing-duration changes across LYX records", y=1.01)
    return _save_figure(fig, output / "figure_01_paired_motion_mae_delta")


def _render_scene_heatmap(
    frame: pd.DataFrame,
    output: Path,
    durations: Sequence[int],
) -> list[Path]:
    scenes = sorted(frame["scene"].unique())
    row_keys = [(anchor, scene) for anchor in ANCHOR_TYPES for scene in scenes]
    values = np.full((len(row_keys), len(durations)), np.nan)
    for row_idx, (anchor, scene) in enumerate(row_keys):
        subset = frame[
            (frame["anchor_type"] == anchor) & (frame["scene"] == scene)
        ]
        medians = subset.groupby("smooth_duration_s")[
            "delta_motion_final_mae_vs_1s_bpm"
        ].median()
        for col_idx, duration in enumerate(durations):
            if duration in medians.index:
                values[row_idx, col_idx] = medians.loc[duration]
    finite = np.abs(values[np.isfinite(values)])
    limit = max(0.5, float(np.max(finite)) if finite.size else 0.5)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue_white_orange",
        ["#2C7FB8", "#F7F7F7", "#D97732"],
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(durations)), labels=[str(value) for value in durations])
    ax.set_yticks(
        np.arange(len(row_keys)),
        labels=[
            f"{'Independent' if anchor == 'independent_bo' else 'Shared'} · {scene}"
            for anchor, scene in row_keys
        ],
    )
    ax.set_xlabel("Smoothing duration (s)")
    ax.set_title("Scene-level median strict-motion MAE change vs 1 s")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#222222",
                )
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("MAE delta (BPM); blue improves, orange degrades")
    return _save_figure(fig, output / "figure_02_scene_median_heatmap")


def _representative_selections(
    frame: pd.DataFrame,
) -> list[tuple[str, str, str]]:
    candidates = frame[frame["smooth_duration_s"] > 1].copy()
    if candidates.empty:
        return []
    rows: list[tuple[str, str, str]] = []

    def add(label: str, row: pd.Series) -> None:
        key = (str(row["anchor_type"]), str(row["sample"]))
        if key not in {(anchor, sample) for _, anchor, sample in rows}:
            rows.append((label, key[0], key[1]))

    add(
        "highest_delta",
        candidates.loc[candidates["delta_motion_final_mae_vs_1s_bpm"].idxmax()],
    )
    add(
        "lowest_delta",
        candidates.loc[candidates["delta_motion_final_mae_vs_1s_bpm"].idxmin()],
    )
    pivot = candidates.pivot_table(
        index=["sample", "smooth_duration_s"],
        columns="anchor_type",
        values="delta_motion_final_mae_vs_1s_bpm",
        aggfunc="first",
    ).dropna()
    if not pivot.empty:
        difference = np.abs(pivot["independent_bo"] - pivot["shared_holdout"])
        sample, duration = difference.idxmax()
        row = candidates[
            (candidates["sample"] == sample)
            & (candidates["smooth_duration_s"] == duration)
        ].iloc[0]
        add("anchor_disagreement", row)
    absolute = np.abs(candidates["delta_motion_final_mae_vs_1s_bpm"])
    add("near_neutral", candidates.loc[absolute.idxmin()])
    return rows[:4]


def _render_timeseries_grid(
    run_frame: pd.DataFrame,
    trajectory_frame: pd.DataFrame,
    output: Path,
    *,
    durations: Sequence[int],
    label: str,
    anchor_type: str,
    sample: str,
) -> list[Path]:
    subset = trajectory_frame[
        (trajectory_frame["anchor_type"] == anchor_type)
        & (trajectory_frame["sample"] == sample)
    ]
    if subset.empty:
        return []
    ymin = np.nanpercentile(
        subset[["reference_bpm", "reset_fft_bpm", "final_bpm"]].to_numpy(), 1
    )
    ymax = np.nanpercentile(
        subset[["reference_bpm", "reset_fft_bpm", "final_bpm"]].to_numpy(), 99
    )
    margin = max(4.0, 0.08 * (ymax - ymin))
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.4), sharex=True, sharey=True)
    for ax, duration in zip(axes.flat, durations, strict=True):
        group = subset[subset["smooth_duration_s"] == duration].sort_values(
            "aligned_time_s"
        )
        ax.plot(
            group["aligned_time_s"],
            group["reference_bpm"],
            color=REFERENCE_COLOR,
            linewidth=1.4,
            label="Reference HR",
        )
        ax.plot(
            group["aligned_time_s"],
            group["reset_fft_bpm"],
            color=BASELINE_COLOR,
            linewidth=0.9,
            linestyle="--",
            label="reset FFT",
        )
        ax.plot(
            group["aligned_time_s"],
            group["final_bpm"],
            color=CONTROL_COLOR if duration == 1 else FINAL_COLOR,
            linewidth=1.1,
            label="LMS+H Final",
        )
        _shade_motion(ax, group)
        metric = run_frame[
            (run_frame["anchor_type"] == anchor_type)
            & (run_frame["sample"] == sample)
            & (run_frame["smooth_duration_s"] == duration)
        ].iloc[0]
        ax.set_title(
            f"{duration} s · motion MAE {metric['motion_final_mae_bpm']:.2f} BPM"
        )
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.grid(color="#ECECEC", linewidth=0.4)
    for ax in axes[-1, :]:
        ax.set_xlabel("Aligned time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Heart rate (BPM)")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
    )
    fig.suptitle(
        f"{sample} · {ANCHOR_TITLES[anchor_type]} · {label.replace('_', ' ')}",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    safe_sample = re.sub(r"[^A-Za-z0-9_-]+", "_", sample)
    return _save_figure(
        fig,
        output / f"timeseries_{label}_{anchor_type}_{safe_sample}",
    )


def _shade_motion(ax: Any, frame: pd.DataFrame) -> None:
    motion = frame["is_motion"].to_numpy(dtype=bool)
    times = frame["aligned_time_s"].to_numpy(dtype=float)
    if not np.any(motion):
        return
    indices = np.flatnonzero(motion)
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [indices.size - 1]))
    step = float(np.median(np.diff(times))) if times.size > 1 else 1.0
    for start_idx, end_idx in zip(starts, ends, strict=True):
        left = times[indices[start_idx]] - step / 2
        right = times[indices[end_idx]] + step / 2
        ax.axvspan(left, right, color=MOTION_COLOR, alpha=0.45, linewidth=0)


def _save_figure(fig: Any, stem: Path) -> list[Path]:
    svg = stem.with_suffix(".svg")
    png = stem.with_suffix(".png")
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(fig)
    return [svg, png]


def _configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "axes.linewidth": 0.65,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 120,
            "savefig.facecolor": "white",
        }
    )


def _write_review_report(
    path: Path,
    *,
    run_frame: pd.DataFrame,
    duration_summary: pd.DataFrame,
    agreement: pd.DataFrame,
    identity_rows: Sequence[dict[str, Any]],
    shortlist: Sequence[int],
    figure_paths: Sequence[Path],
) -> None:
    overall = duration_summary[duration_summary["anchor_type"] == "all"]
    lines = [
        "# LYX 平滑机制人工审核报告",
        "",
        "## 当前状态",
        "",
        "**等待人工审核；正式独立 BO 与 K 折实验未获授权，也未启动。**",
        "",
        "## 数据与执行完整性",
        "",
        f"- 参数锚点：{run_frame[['anchor_type', 'sample']].drop_duplicates().shape[0]} 套"
        "（24 条记录 × 2 套锚点）。",
        f"- 求解：{run_frame.shape[0]} 次；状态均为 ok。",
        f"- 方法身份对账：{sum(bool(row['identity_ok']) for row in identity_rows)}/"
        f"{len(identity_rows)} 通过。",
        f"- 数值同口径直接一致："
        f"{sum(bool(row['numeric_reconciled']) for row in identity_rows)}/"
        f"{len(identity_rows)}；其余差异均需在审计表给出原因。",
        f"- 非有限输出窗口：{int(run_frame['nonfinite_final_windows'].sum() + run_frame['nonfinite_reset_fft_windows'].sum())}。",
        "- 两套锚点均用当前代码重放；只覆盖 smooth_win_len、time_bias=5.0 s、"
        "lms_mu_min=1e-6。归档报告不是逐点轨迹真值。",
        "- 主表同时保存经典可视曲线的全窗口 MAE 与求解器目标使用的可靠窗口 MAE；"
        "二者禁止混用。",
        "- 归档配置加载器会关闭旧 post_motion_reacquire 开关；该行为已写入复现清单，"
        "所有时长和两套锚点使用相同规则。",
        "",
        "## 汇总证据",
        "",
        "| 时长 (s) | 未来依赖 (s) | 运动 MAE 均值 | 相对 1 s 均值 | 中位变化 | 最大退化 |"
        " 0.5 BPM 内比例 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"| {int(row['smooth_duration_s'])} | {row['future_lookahead_s']:.1f} | "
            f"{row['mean_motion_final_mae_bpm']:.3f} | "
            f"{row['mean_delta_motion_vs_1s_bpm']:+.3f} | "
            f"{row['median_delta_motion_vs_1s_bpm']:+.3f} | "
            f"{row['max_delta_motion_vs_1s_bpm']:+.3f} | "
            f"{row['within_0p5bpm_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 两套锚点方向一致性",
            "",
            "| 时长 (s) | 方向一致率（±0.1 BPM 视为等效） | 锚点差异中位数 | 最大锚点差异 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for _, row in agreement.iterrows():
        lines.append(
            f"| {int(row['smooth_duration_s'])} | "
            f"{row['direction_agreement_rate_0p1bpm']:.1%} | "
            f"{row['median_abs_anchor_delta_difference_bpm']:.3f} | "
            f"{row['max_abs_anchor_delta_difference_bpm']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 供人工优先查看的时长",
            "",
            (
                "- 按“最坏退化 → 平均退化 → 更短时长”排序得到的证据短名单："
                + (", ".join(f"{value} s" for value in shortlist) if shortlist else "无")
                + "。这不是自动选择结果。"
            ),
            "- 最终判断还需同时查看场景热图、最差记录和代表性同轴心率曲线。",
            "- 居中中值平滑存在未来数据依赖；时长 3/5/7/9/11 s 分别约需"
            " 1/2/3/4/5 s 的未来窗口。",
            "",
            "## 可视化",
            "",
        ]
    )
    for figure in figure_paths:
        if figure.suffix.lower() == ".png":
            lines.append(f"- `{figure.relative_to(path.parent).as_posix()}`")
    lines.extend(
        [
            "",
            "## 人工门",
            "",
            "请在 `human_smoothing_decision.json` 中记录以下三类决定之一：",
            "",
            "1. 选择一个统一时长，并明确授权进入正式实验；",
            "2. 要求补充或重跑；",
            "3. 拒绝统一固定时长，返回机制设计。",
            "",
            "在 `formal_experiment_authorized=true` 前，后续正式实验不得运行。",
            "",
            "## 限制",
            "",
            "- 24 条记录均来自 LYX，同场景记录相关，不将其当作独立人群重复。",
            "- 本实验使用全部 LYX 数据做机制开发，不提供未见个体泛化证据。",
            "- 过渡强度和位置指标是辅助可解释性证据，不能替代逐条心率曲线审阅。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_anchor_manifest(path: Path, anchors: Sequence[SmoothingAnchor]) -> None:
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        payload = _load_json(anchor.report_path)
        cfg = load_lite_report_config(payload)
        best_params = payload.get("best_params")
        rows.append(
            {
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in asdict(anchor).items()
                },
                "source_best_params_json": json.dumps(
                    best_params if isinstance(best_params, dict) else {},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "loaded_post_motion_reacquire_enable": (
                    cfg.post_motion_reacquire_enable
                ),
                "loaded_post_motion_dynamic_guard_enable": (
                    cfg.post_motion_dynamic_guard_enable
                ),
                "forced_time_bias_s": 5.0,
                "forced_lms_mu_min": 1e-6,
            }
        )
    _write_rows(path, rows)


def _write_run_manifest(
    path: Path,
    *,
    anchors: Sequence[SmoothingAnchor],
    durations: Sequence[int],
    output_dir: Path,
    independent_batch_dir: Path,
    generalization_dir: Path,
    workers: int,
) -> None:
    input_paths: set[Path] = set()
    for anchor in anchors:
        input_paths.update(
            {
                anchor.report_path,
                anchor.error_csv,
                anchor.data_path,
                anchor.ref_path,
            }
        )
    output_paths = sorted(
        candidate
        for candidate in output_dir.rglob("*")
        if candidate.is_file() and candidate != path
    )
    manifest = {
        "schema_version": "lyx_smoothing_mechanism_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git": _git_state(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "experiment": {
            "record_count": len(anchors) // 2,
            "anchor_count": len(anchors),
            "durations_s": list(durations),
            "planned_solve_count": len(anchors) * len(durations),
            "workers": int(workers),
            "time_bias_s": 5.0,
            "lms_mu_min": 1e-6,
            "window_step_requirement_s": 1.0,
            "future_lookahead_s": {
                str(value): (value - 1) / 2.0 for value in durations
            },
            "human_gate_required": True,
            "formal_experiment_authorized": False,
        },
        "anchor_replay_semantics": {
            "loader": "load_lite_report_config",
            "overridden_fields": [
                "smooth_win_len",
                "time_bias",
                "lms_mu_min",
            ],
            "loader_disables_post_motion_reacquire_enable": True,
            "archive_is_accuracy_anchor_not_bit_exact_trajectory_oracle": True,
        },
        "source_roots": {
            "independent_batch_dir": str(independent_batch_dir),
            "generalization_dir": str(generalization_dir),
        },
        "inputs": [_file_descriptor(value) for value in sorted(input_paths)],
        "outputs": [_file_descriptor(value) for value in output_paths],
    }
    _write_json(path, manifest)


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {
            "commit": commit,
            "dirty": bool(status.strip()),
            "status_porcelain": status.splitlines(),
            "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc)}


def _file_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_gzip_frame(path: Path, frame: pd.DataFrame) -> None:
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as fh:
        frame.to_csv(fh, index=False)


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"不可序列化：{type(value).__name__}")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return payload


def _read_method_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {
            str(row.get("method", "")).strip(): row
            for row in csv.DictReader(fh)
            if str(row.get("method", "")).strip()
        }


def _row_float(row: dict[str, str] | None, key: str) -> float:
    return _as_float(row.get(key)) if row else math.nan


def _nested_float(payload: dict[str, Any], parent: str, key: str) -> float:
    value = payload.get(parent)
    return _as_float(value.get(key)) if isinstance(value, dict) else math.nan


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite_stat(values: Any, function: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(function(arr)) if arr.size else math.nan


def _finite_percentile(values: Any, percentile: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, percentile)) if arr.size else math.nan


def _finite_fraction(condition: Any, reference: Any) -> float:
    condition_arr = np.asarray(condition, dtype=bool)
    reference_arr = np.asarray(reference, dtype=float)
    valid = np.isfinite(reference_arr)
    return float(np.mean(condition_arr[valid])) if np.any(valid) else math.nan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 LYX 双锚点平滑机制前置实验，并停在人工审核门。",
    )
    parser.add_argument("--independent-batch-dir", type=Path, required=True)
    parser.add_argument("--generalization-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=list(DEFAULT_SMOOTH_DURATIONS_S),
    )
    parser.add_argument("--expected-record-count", type=int, default=24)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="可选：只运行指定样本名，用于冒烟验证；正式实验不传。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = run_smoothing_mechanism_experiment(
        independent_batch_dir=args.independent_batch_dir,
        generalization_dir=args.generalization_dir,
        output_dir=args.output_dir,
        durations_s=args.durations,
        expected_record_count=args.expected_record_count,
        sample_names=args.samples,
        workers=args.workers,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
