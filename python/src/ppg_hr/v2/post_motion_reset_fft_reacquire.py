"""LYX-only post-motion reset FFT reacquire experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .algorithm_presets import DirectionalTrackingParams
from .motion_aware_fft_baseline import (
    BaselineSample,
    FFT_CHAIN_POST_GUARD_RESET,
    run_baseline_sample,
)
from .plotting import render_v2_report
from .report import save_v2_report
from .solver import V2SolverResult, solve_v2
from .types import V2RunConfig

SOURCE_MODE_REUSED_BO_SOURCE = "reused_bo_source"
SOURCE_MODE_OLD_HR_PREFIX_SPLICE = "old_hr_prefix_splice"
SOURCE_MODE_FIXED_LITE_SOURCE = "fixed_lite_source"
BOUNDARY_STRATEGY_NONE = "none"
BOUNDARY_STRATEGY_SMOOTH_BRIDGE = "smooth_bridge"
BOUNDARY_STRATEGY_ADAPTIVE_FALLBACK = "adaptive_fallback"
SOURCE_MODES = (
    SOURCE_MODE_REUSED_BO_SOURCE,
    SOURCE_MODE_OLD_HR_PREFIX_SPLICE,
    SOURCE_MODE_FIXED_LITE_SOURCE,
)
POST_GUARD_PASS_THRESHOLD_BPM = 3.0
SOURCE_REPLAY_P95_DRIFT_THRESHOLD_BPM = 5.0
HIGH_DRIFT_OLD_LITE_MAE_BPM = 20.0
MAX_SINGLE_SAMPLE_REGRESSION_BPM = 2.0
FIXED_60S_MEAN_REGRESSION_LIMIT_BPM = 1.0
BOUNDARY_JUMP_RISK_BPM = 20.0

REPRESENTATIVE_LYX_SAMPLE_IDS = frozenset(
    {
        "multi_fuwo1_0613",
        "multi_fuwo2_0613",
        "multi_fuwo1_0519",
        "multi_bobi1_0617",
        "multi_bobi1_0613",
        "multi_bobi3_0617",
        "multi_tiaosheng1_0613",
        "multi_tiaosheng2_0613",
        "multi_tiaosheng1_0617",
        "multi_kaihe1_0613",
        "multi_kaihe2_0613",
        "multi_kaihe2_0617",
        "multi_wanju2_0613",
        "multi_wanju2_0617",
        "multi_wanju1_0617",
    }
)


def enumerate_lyx_samples(data_root: Path | str) -> list[BaselineSample]:
    root = Path(data_root)
    samples: list[BaselineSample] = []
    for data_path in sorted(root.glob("*.csv")):
        if data_path.name.endswith("_HR_ref.csv") or data_path.name.endswith("_ref.csv"):
            continue
        ref_path = data_path.with_name(f"{data_path.stem}_HR_ref{data_path.suffix}")
        if not ref_path.is_file():
            continue
        samples.append(
            BaselineSample(
                cohort="LYX",
                sample_id=data_path.stem,
                data_path=data_path,
                ref_path=ref_path,
            )
        )
    return samples


def select_representative_lyx_samples(samples: Iterable[BaselineSample]) -> list[BaselineSample]:
    by_id = {sample.sample_id: sample for sample in samples if sample.cohort == "LYX"}
    return [
        by_id[sample_id]
        for sample_id in sorted(REPRESENTATIVE_LYX_SAMPLE_IDS)
        if sample_id in by_id
    ]


def compute_lite_baseline_metrics(batch_dir: Path | str) -> list[dict[str, Any]]:
    root = Path(batch_dir)
    rows: list[dict[str, Any]] = []
    for report_path in sorted((root / "json").glob("*.json")):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        motion = payload.get("motion_segment") or {}
        if "end_s" not in motion:
            continue
        sample_id = _sample_id_from_report(report_path, payload)
        hr_path = root / "csv" / f"{report_path.stem}-hr.csv"
        if not hr_path.is_file():
            continue
        hr_rows = _read_hr_csv(hr_path)
        motion_end_s = float(motion["end_s"])
        post = [row for row in hr_rows if row["time_s"] > motion_end_s]
        post60 = [
            row
            for row in hr_rows
            if row["time_s"] > motion_end_s and row["time_s"] <= motion_end_s + 60.0
        ]
        rows.append(
            {
                "sample_id": sample_id,
                "motion_end_s": motion_end_s,
                "lite_post_motion_mae_bpm": _mae(post, "final_bpm"),
                "lite_post_motion_60s_mae_bpm": _mae(post60, "final_bpm"),
                "lite_post_motion_fft_mae_bpm": _mae(post, "fft_bpm"),
                "lite_window_count": len(post),
                "lite_60s_window_count": len(post60),
                "lite_report_path": str(report_path),
                "lite_hr_csv": str(hr_path),
            }
        )
    return rows


def load_lite_report_config(report: Path | str | dict[str, Any]) -> V2RunConfig:
    """Restore a replay config from an old Lite BO v2 report."""

    payload = _load_report_payload(report)
    field_names = {field.name for field in fields(V2RunConfig)}
    values = {
        key: value
        for key, value in payload.items()
        if key in field_names and key not in {"extras"}
    }
    best_params = payload.get("best_params") or {}
    if isinstance(best_params, dict):
        values.update({key: value for key, value in best_params.items() if key in field_names})
    _apply_report_dynamic_guard_state(values, payload, field_names)
    if "data_path" not in values or "ref_path" not in values:
        raise ValueError("Lite report must include data_path and ref_path")
    values["data_path"] = Path(values["data_path"])
    values["ref_path"] = Path(values["ref_path"])
    if "reference_groups_order" in values:
        values["reference_groups_order"] = tuple(values["reference_groups_order"])
    values["post_motion_reacquire_enable"] = False
    return V2RunConfig(**values)


def _apply_report_dynamic_guard_state(
    values: dict[str, Any],
    payload: dict[str, Any],
    field_names: set[str],
) -> None:
    guard = payload.get("post_motion_dynamic_guard")
    if isinstance(guard, dict):
        mapping = {
            "enabled": "post_motion_dynamic_guard_enable",
            "min_elapsed_s": "post_motion_dynamic_guard_min_elapsed_s",
            "stable_windows": "post_motion_dynamic_guard_stable_windows",
            "crossover_gap_bpm": "post_motion_dynamic_guard_crossover_gap_bpm",
            "upward_gap_bpm": "post_motion_dynamic_guard_upward_gap_bpm",
            "fft_floor_bpm": "post_motion_dynamic_guard_fft_floor_bpm",
            "recovery_step_up_bpm": "post_motion_dynamic_guard_recovery_step_up_bpm",
            "recovery_step_down_bpm": "post_motion_dynamic_guard_recovery_step_down_bpm",
            "rising_windows": "post_motion_dynamic_guard_rising_windows",
            "rising_slope_bpm_per_window": (
                "post_motion_dynamic_guard_rising_slope_bpm_per_window"
            ),
            "rescue_gap_bpm": "post_motion_dynamic_guard_rescue_gap_bpm",
            "gap_rescue_enable": "post_motion_dynamic_guard_gap_rescue_enable",
            "gap_rescue_windows": "post_motion_dynamic_guard_gap_rescue_windows",
            "gap_rescue_min_hits": "post_motion_dynamic_guard_gap_rescue_min_hits",
            "gap_rescue_fft_stable_windows": (
                "post_motion_dynamic_guard_gap_rescue_fft_stable_windows"
            ),
            "gap_rescue_fft_stable_bpm": (
                "post_motion_dynamic_guard_gap_rescue_fft_stable_bpm"
            ),
        }
        for report_key, cfg_key in mapping.items():
            if cfg_key in field_names and report_key in guard:
                values[cfg_key] = guard[report_key]
        return

    if "post_motion_dynamic_guard_enable" not in values:
        values["post_motion_dynamic_guard_enable"] = False


def run_lite_source_replay_audit(
    *,
    lite_batch_dir: Path | str,
    output_dir: Path | str,
    sample_ids: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    root = Path(lite_batch_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for report_path in sorted((root / "json").glob("*.json")):
        payload = _load_report_payload(report_path)
        sample_id = _sample_id_from_report(report_path, payload)
        if sample_ids is not None and sample_id not in sample_ids:
            continue
        hr_path = root / "csv" / f"{report_path.stem}-hr.csv"
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "lite_report_path": str(report_path),
            "lite_hr_csv": str(hr_path),
            "replay_status": "ok",
            "data_path": str(payload.get("data_path", "")),
            "ref_path": str(payload.get("ref_path", "")),
            "algorithm_preset": str(payload.get("algorithm_preset", "")),
            "adaptive_filter": str(payload.get("adaptive_filter", "")),
            "reference_groups_order": json.dumps(
                payload.get("reference_groups_order", []),
                ensure_ascii=False,
            ),
        }
        try:
            if not hr_path.is_file():
                raise FileNotFoundError(hr_path)
            cfg = load_lite_report_config(payload)
            replay = solve_v2(cfg)
            old_rows = _read_hr_csv(hr_path)
            row.update(_source_replay_diff_metrics(replay.HR, old_rows))
            if int(row.get("matched_window_count", 0)) == 0:
                row["replay_status"] = "no_match"
        except Exception as exc:
            row.update(
                {
                    "replay_status": f"error:{type(exc).__name__}",
                    "matched_window_count": 0,
                    "old_window_count": 0,
                    "replay_window_count": 0,
                    "mean_abs_diff_bpm": float("nan"),
                    "p95_abs_diff_bpm": float("nan"),
                    "max_abs_diff_bpm": float("nan"),
                    "error": str(exc),
                }
            )
        rows.append(row)
    _write_dict_csv(out / "lite_source_replay_metrics.csv", rows)
    metadata = {
        "lite_batch_dir": str(root),
        "output_dir": str(out),
        "sample_count": len(rows),
        "report_path": str(out / "lite_source_replay_metrics.csv"),
    }
    (out / "lite_source_replay_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"metadata": metadata, "rows": rows}


def _load_report_payload(report: Path | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(report, dict):
        return dict(report)
    return json.loads(Path(report).read_text(encoding="utf-8"))


def _source_replay_diff_metrics(
    replay_hr: np.ndarray,
    old_hr_rows: list[dict[str, float]],
) -> dict[str, Any]:
    replay = np.asarray(replay_hr, dtype=float)
    old_by_time = {round(float(row["time_s"]), 6): row for row in old_hr_rows}
    diffs: list[float] = []
    for row in replay:
        old = old_by_time.get(round(float(row[0]), 6))
        if old is None:
            continue
        replay_final = float(row[3])
        old_final = float(old["final_bpm"])
        if math.isfinite(replay_final) and math.isfinite(old_final):
            diffs.append(abs(replay_final - old_final))
    arr = np.asarray(diffs, dtype=float)
    return {
        "matched_window_count": int(arr.size),
        "old_window_count": len(old_hr_rows),
        "replay_window_count": int(replay.shape[0]) if replay.ndim == 2 else 0,
        "mean_abs_diff_bpm": float(np.mean(arr)) if arr.size else float("nan"),
        "p95_abs_diff_bpm": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "max_abs_diff_bpm": float(np.max(arr)) if arr.size else float("nan"),
    }


def _sample_id_from_report(report_path: Path, payload: dict[str, Any]) -> str:
    data_path = str(payload.get("data_path") or "")
    if data_path:
        return Path(data_path).stem
    marker = "-green-"
    return report_path.stem.split(marker, 1)[0] if marker in report_path.stem else report_path.stem


def _read_hr_csv(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [
            {
                "time_s": float(row["time_s"]),
                "ref_bpm": float(row["ref_bpm"]),
                "fft_bpm": float(row["fft_bpm"]),
                "final_bpm": float(row["final_bpm"]),
                "is_motion": float(row.get("is_motion", 0.0)),
                "used_adaptive": float(row.get("used_adaptive", 0.0)),
            }
            for row in csv.DictReader(f)
        ]


def _mae(rows: list[dict[str, float]], column: str) -> float:
    errors = [
        abs(float(row[column]) - float(row["ref_bpm"]))
        for row in rows
        if math.isfinite(float(row[column])) and math.isfinite(float(row["ref_bpm"]))
    ]
    return float(np.mean(errors)) if errors else float("nan")


@dataclass(frozen=True)
class PostMotionResetConfig:
    name: str
    guard_seconds: float
    peak_strategy: str = "raw_main_peak"
    min_bpm_floor: float | None = None
    range_up_bpm: float = 20.0
    range_down_bpm: float = 35.0
    step_up_bpm: float = 1.5
    step_down_bpm: float = 6.0
    first_window_no_hold_count: int = 3
    topk_k: int = 0
    consensus_windows: int = 0
    consensus_tolerance_bpm: float = 6.0
    boundary_strategy: str = BOUNDARY_STRATEGY_NONE
    bridge_windows: int = 0
    adaptive_fallback_windows: int = 0
    run_scope: str = "representative"

    def tracking_params(self) -> DirectionalTrackingParams:
        return DirectionalTrackingParams(
            range_up_bpm=float(self.range_up_bpm),
            range_down_bpm=float(self.range_down_bpm),
            limit_up_bpm=float(self.step_up_bpm),
            step_up_bpm=float(self.step_up_bpm),
            limit_down_bpm=float(self.step_down_bpm),
            step_down_bpm=float(self.step_down_bpm),
        )


def build_stage1_guard_configs() -> list[PostMotionResetConfig]:
    return [
        PostMotionResetConfig(name=f"guard{int(guard)}_raw_reset", guard_seconds=float(guard))
        for guard in (0.0, 5.0, 10.0, 15.0, 20.0)
    ]


def build_representative_candidate_configs() -> list[PostMotionResetConfig]:
    configs: list[PostMotionResetConfig] = []
    seen: set[str] = set()

    def add(config: PostMotionResetConfig) -> None:
        if config.name not in seen:
            seen.add(config.name)
            configs.append(config)

    for config in build_stage1_guard_configs():
        add(config)

    for guard in (0.0, 5.0):
        for floor in (55.0, 60.0):
            add(
                PostMotionResetConfig(
                    name=f"guard{int(guard)}_min{int(floor)}_reset",
                    guard_seconds=guard,
                    peak_strategy="min_bpm_floor",
                    min_bpm_floor=floor,
                )
            )

    for guard in (0.0, 5.0):
        for k, windows in ((2, 2), (3, 3)):
            add(
                PostMotionResetConfig(
                    name=f"guard{int(guard)}_topk{k}_consensus{windows}_reset",
                    guard_seconds=guard,
                    peak_strategy="topk_consensus_reset",
                    topk_k=k,
                    consensus_windows=windows,
                    first_window_no_hold_count=0,
                )
            )

    add(
        PostMotionResetConfig(
            name="guard5_raw_smooth2_reset",
            guard_seconds=5.0,
            boundary_strategy=BOUNDARY_STRATEGY_SMOOTH_BRIDGE,
            bridge_windows=2,
        )
    )
    add(
        PostMotionResetConfig(
            name="guard5_topk2_consensus2_fallback2_reset",
            guard_seconds=5.0,
            peak_strategy="topk_consensus_reset",
            topk_k=2,
            consensus_windows=2,
            first_window_no_hold_count=0,
            boundary_strategy=BOUNDARY_STRATEGY_ADAPTIVE_FALLBACK,
            adaptive_fallback_windows=2,
        )
    )

    for guard in (0.0, 5.0):
        for range_down, step_down, no_hold in (
            (25.0, 4.0, 3),
            (35.0, 6.0, 3),
            (45.0, 8.0, 5),
        ):
            add(
                PostMotionResetConfig(
                    name=(
                        f"guard{int(guard)}_down{int(range_down)}_"
                        f"step{int(step_down)}_nohold{int(no_hold)}"
                    ),
                    guard_seconds=guard,
                    range_down_bpm=range_down,
                    step_down_bpm=step_down,
                    first_window_no_hold_count=no_hold,
                )
            )

    return configs


def combine_source_and_reset_rows(
    source_hr: np.ndarray,
    reset_rows: list[dict[str, Any]],
    *,
    motion_end_s: float,
    config: PostMotionResetConfig,
) -> np.ndarray:
    combined = np.asarray(source_hr, dtype=float).copy()
    by_time = {round(float(row["time_s"]), 6): row for row in reset_rows}
    reacquire_start = float(motion_end_s) + float(config.guard_seconds)
    bridge_count = max(0, int(config.bridge_windows))
    bridge_applied = 0
    fallback_remaining = 0
    for idx in range(combined.shape[0]):
        t = float(combined[idx, 0])
        if t <= reacquire_start + 1e-9:
            combined[idx, 5] = 1.0 if t >= float(motion_end_s) - 1e-9 else combined[idx, 5]
            continue
        reset = by_time.get(round(t, 6))
        if reset is None:
            combined[idx, 5] = 0.0
            combined[idx, 3] = combined[idx, 2]
            continue
        reset["boundary_strategy"] = config.boundary_strategy
        if str(reset.get("candidate_source", "")) == "topk_consensus_pending":
            combined[idx, 5] = 1.0
            continue
        if (
            config.boundary_strategy == BOUNDARY_STRATEGY_ADAPTIVE_FALLBACK
            and fallback_remaining <= 0
            and (
                str(reset.get("consensus_status", "")) in {"fallback", "failed"}
                or str(reset.get("candidate_source", "")) == "topk_consensus_fallback"
            )
        ):
            fallback_remaining = max(0, int(config.adaptive_fallback_windows))
        if fallback_remaining > 0:
            reset["fallback_applied"] = True
            reset["fallback_reason"] = "adaptive_fallback"
            combined[idx, 5] = 1.0
            fallback_remaining -= 1
            continue
        candidate = float(reset["fft_baseline_bpm"])
        combined[idx, 2] = candidate
        if (
            config.boundary_strategy == BOUNDARY_STRATEGY_SMOOTH_BRIDGE
            and bridge_applied < bridge_count
        ):
            weight = float(bridge_applied + 1) / float(bridge_count + 1)
            source_value = float(combined[idx, 3])
            combined[idx, 3] = source_value * (1.0 - weight) + candidate * weight
            reset["bridge_weight"] = weight
            reset["bridge_source_bpm"] = source_value
            reset["bridge_output_bpm"] = float(combined[idx, 3])
            bridge_applied += 1
        else:
            combined[idx, 3] = candidate
        combined[idx, 5] = 0.0
    return combined


def summarise_candidate_metrics(
    *,
    sample_id: str,
    config: PostMotionResetConfig,
    motion_end_s: float,
    combined_hr: np.ndarray,
    reset_rows: list[dict[str, Any]],
    lite_baseline: dict[str, Any] | None,
    source_mode: str = SOURCE_MODE_FIXED_LITE_SOURCE,
    source_replay: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hr = np.asarray(combined_hr, dtype=float)
    post_guard_start = float(motion_end_s) + float(config.guard_seconds)
    post_guard = hr[hr[:, 0] > post_guard_start + 1e-9]
    post60 = hr[
        (hr[:, 0] > float(motion_end_s) + 1e-9)
        & (hr[:, 0] <= float(motion_end_s) + 60.0 + 1e-9)
    ]
    post_full = hr[hr[:, 0] > float(motion_end_s) + 1e-9]
    jumps = np.abs(np.diff(post_full[:, 3])) if post_full.shape[0] > 1 else np.asarray([], dtype=float)
    boundary_jump = _boundary_jump(hr, post_guard_start)
    failures = _failure_counts(reset_rows)
    lite = lite_baseline or {}
    post_mae = _hr_mae(post_guard)
    post60_mae = _hr_mae(post60)
    post_guard_signed_error = _hr_mean_signed_error(post_guard)
    delta_vs_lite_post = post_mae - float(
        lite.get("lite_post_motion_mae_bpm", float("nan"))
    )
    delta_vs_lite_60s = post60_mae - float(
        lite.get("lite_post_motion_60s_mae_bpm", float("nan"))
    )
    source_replay_row = source_replay or {}
    source_replay_p95 = _as_float(source_replay_row.get("p95_abs_diff_bpm"))
    source_replay_status = str(source_replay_row.get("replay_status", ""))
    consensus = _consensus_summary(reset_rows)
    primary_failure_bucket = _primary_failure_bucket(
        source_mode=source_mode,
        source_replay_status=source_replay_status,
        source_replay_p95_diff_bpm=source_replay_p95,
        post_guard_mae_bpm=post_mae,
        post_guard_signed_error_bpm=post_guard_signed_error,
        post_motion_60s_mae_bpm=post60_mae,
        delta_vs_lite_60s_mae_bpm=delta_vs_lite_60s,
        boundary_jump_bpm=boundary_jump,
        failures=failures,
        consensus_status=str(consensus["status"]),
    )
    return {
        "sample_id": sample_id,
        "source_mode": source_mode,
        "candidate_name": config.name,
        "motion_end_s": float(motion_end_s),
        "guard_end_s": post_guard_start,
        "reset_takeover_s": _reset_takeover_s_from_rows(reset_rows, post_guard_start),
        "fallback_window_count": _fallback_window_count(reset_rows),
        "guard_seconds": float(config.guard_seconds),
        "peak_strategy": config.peak_strategy,
        "topk_k": int(config.topk_k),
        "consensus_windows": int(config.consensus_windows),
        "consensus_status": consensus["status"],
        "consensus_selected_bpm": consensus["selected_bpm"],
        "consensus_failure_reason": consensus["failure_reason"],
        "consensus_takeover_s": consensus["takeover_s"],
        "boundary_strategy": config.boundary_strategy,
        "bridge_windows": int(config.bridge_windows),
        "adaptive_fallback_windows": int(config.adaptive_fallback_windows),
        "post_guard_final_mae_bpm": post_mae,
        "post_guard_mean_signed_error_bpm": post_guard_signed_error,
        "post_motion_60s_final_mae_bpm": post60_mae,
        "post_motion_full_final_mae_bpm": _hr_mae(post_full),
        "old_lite_post_motion_mae_bpm": float(
            lite.get("lite_post_motion_mae_bpm", float("nan"))
        ),
        "new_post_guard_mae_bpm": post_mae,
        "new_post_motion_60s_mae_bpm": post60_mae,
        "source_replay_p95_diff_bpm": source_replay_p95,
        "source_replay_status": source_replay_status,
        "boundary_jump_bpm": boundary_jump,
        "boundary_p95_abs_jump_bpm": float(np.percentile(jumps, 95)) if jumps.size else 0.0,
        "passes_post_guard_3bpm": bool(
            math.isfinite(post_mae) and post_mae < POST_GUARD_PASS_THRESHOLD_BPM
        ),
        "delta_vs_lite_post_mae_bpm": delta_vs_lite_post,
        "delta_vs_lite_60s_mae_bpm": delta_vs_lite_60s,
        "primary_failure_bucket": primary_failure_bucket,
        **{f"failure_{key}_windows": value for key, value in failures.items()},
    }


def aggregate_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row.get("source_mode", "")), str(row["candidate_name"])),
            [],
        ).append(row)
    out: list[dict[str, Any]] = []
    for (source_mode, name), items in sorted(grouped.items()):
        worst = max(items, key=lambda row: _as_float(row.get("delta_vs_lite_post_mae_bpm")))
        bucket_counts = _failure_bucket_counts(items)
        high_drift_items = [
            row
            for row in items
            if _as_float(row.get("old_lite_post_motion_mae_bpm"))
            >= HIGH_DRIFT_OLD_LITE_MAE_BPM
        ]
        high_drift_improved = [
            row
            for row in high_drift_items
            if _as_float(row.get("delta_vs_lite_post_mae_bpm")) < 0.0
        ]
        mean_delta_60s = _finite_mean(row.get("delta_vs_lite_60s_mae_bpm") for row in items)
        max_boundary_jump = max(_as_float(row.get("boundary_jump_bpm")) for row in items)
        max_source_replay_p95 = max(
            _as_float(row.get("source_replay_p95_diff_bpm")) for row in items
        )
        passing_count = sum(1 for row in items if bool(row.get("passes_post_guard_3bpm")))
        gate_reasons = _candidate_gate_failure_reasons(
            source_mode=source_mode,
            sample_count=len(items),
            passing_sample_count=passing_count,
            mean_delta_vs_lite_post_mae_bpm=_finite_mean(
                row.get("delta_vs_lite_post_mae_bpm") for row in items
            ),
            max_regression_delta_bpm=_as_float(worst.get("delta_vs_lite_post_mae_bpm")),
            mean_delta_vs_lite_60s_mae_bpm=mean_delta_60s,
            max_boundary_jump_bpm=max_boundary_jump,
            high_drift_sample_count=len(high_drift_items),
            high_drift_improved_count=len(high_drift_improved),
            max_source_replay_p95_diff_bpm=max_source_replay_p95,
            source_replay_statuses=[str(row.get("source_replay_status", "")) for row in items],
        )
        out.append(
            {
                "source_mode": source_mode,
                "candidate_name": name,
                "sample_count": len(items),
                "passing_sample_count": passing_count,
                "mean_post_guard_final_mae_bpm": _finite_mean(
                    row.get("post_guard_final_mae_bpm") for row in items
                ),
                "mean_post_motion_60s_final_mae_bpm": _finite_mean(
                    row.get("post_motion_60s_final_mae_bpm") for row in items
                ),
                "mean_delta_vs_lite_post_mae_bpm": _finite_mean(
                    row.get("delta_vs_lite_post_mae_bpm") for row in items
                ),
                "mean_delta_vs_lite_60s_mae_bpm": mean_delta_60s,
                "max_regression_sample_id": worst.get("sample_id", ""),
                "max_regression_delta_bpm": worst.get(
                    "delta_vs_lite_post_mae_bpm", float("nan")
                ),
                "max_boundary_jump_bpm": max_boundary_jump,
                "high_drift_sample_count": len(high_drift_items),
                "high_drift_improved_count": len(high_drift_improved),
                "max_source_replay_p95_diff_bpm": max_source_replay_p95,
                "dominant_failure_bucket": _dominant_bucket(bucket_counts),
                "dominant_failure_count": max(bucket_counts.values()) if bucket_counts else 0,
                "gate_decision": "go" if not gate_reasons else "no_go",
                "gate_failure_reasons": ";".join(gate_reasons),
                **{f"bucket_{key}_count": value for key, value in bucket_counts.items()},
            }
        )
    return out


def _hr_mae(hr: np.ndarray) -> float:
    arr = np.asarray(hr, dtype=float)
    if arr.size == 0:
        return float("nan")
    errors = np.abs(arr[:, 3] - arr[:, 1])
    errors = errors[np.isfinite(errors)]
    return float(np.mean(errors)) if errors.size else float("nan")


def _hr_mean_signed_error(hr: np.ndarray) -> float:
    arr = np.asarray(hr, dtype=float)
    if arr.size == 0:
        return float("nan")
    errors = arr[:, 3] - arr[:, 1]
    errors = errors[np.isfinite(errors)]
    return float(np.mean(errors)) if errors.size else float("nan")


def _boundary_jump(hr: np.ndarray, post_guard_start: float) -> float:
    after = np.flatnonzero(hr[:, 0] > float(post_guard_start) + 1e-9)
    if after.size == 0 or int(after[0]) == 0:
        return 0.0
    idx = int(after[0])
    return abs(float(hr[idx, 3]) - float(hr[idx - 1, 3]))


def _reset_takeover_s(hr: np.ndarray, post_guard_start: float) -> float:
    after = np.flatnonzero(hr[:, 0] > float(post_guard_start) + 1e-9)
    if after.size == 0:
        return float("nan")
    return float(hr[int(after[0]), 0])


def _reset_takeover_s_from_rows(
    reset_rows: list[dict[str, Any]],
    post_guard_start: float,
) -> float:
    for row in sorted(reset_rows, key=lambda item: _as_float(item.get("time_s"))):
        if _as_float(row.get("time_s")) <= float(post_guard_start) + 1e-9:
            continue
        if str(row.get("candidate_source", "")) == "topk_consensus_pending":
            continue
        if bool(row.get("fallback_applied", False)):
            continue
        return _as_float(row.get("time_s"))
    return float("nan")


def _fallback_window_count(reset_rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in reset_rows if bool(row.get("fallback_applied", False)))


def _consensus_summary(reset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    consensus_rows = [
        row
        for row in reset_rows
        if str(row.get("consensus_status", ""))
    ]
    if not consensus_rows:
        return {
            "status": "",
            "selected_bpm": float("nan"),
            "failure_reason": "",
            "takeover_s": float("nan"),
        }
    terminal = next(
        (
            row
            for row in consensus_rows
            if str(row.get("consensus_status", "")) in {"selected", "fallback", "failed"}
        ),
        consensus_rows[-1],
    )
    return {
        "status": str(terminal.get("consensus_status", "")),
        "selected_bpm": _as_float(terminal.get("consensus_selected_bpm")),
        "failure_reason": str(terminal.get("consensus_failure_reason", "")),
        "takeover_s": _as_float(terminal.get("time_s"))
        if str(terminal.get("consensus_status", "")) in {"selected", "fallback"}
        else float("nan"),
    }


def _failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "accurate": 0,
        "borderline": 0,
        "low_lock": 0,
        "high_lock": 0,
        "held_previous": 0,
        "no_valid_peak": 0,
    }
    for row in rows:
        reason = str(row.get("failure_reason", "no_valid_peak"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _primary_failure_bucket(
    *,
    source_mode: str,
    source_replay_status: str,
    source_replay_p95_diff_bpm: float,
    post_guard_mae_bpm: float,
    post_guard_signed_error_bpm: float,
    post_motion_60s_mae_bpm: float,
    delta_vs_lite_60s_mae_bpm: float,
    boundary_jump_bpm: float,
    failures: dict[str, int],
    consensus_status: str = "",
) -> str:
    if _has_source_replay_drift(
        source_mode=source_mode,
        status=source_replay_status,
        p95_diff_bpm=source_replay_p95_diff_bpm,
    ):
        return "source_replay_drift"
    if _as_float(boundary_jump_bpm) > BOUNDARY_JUMP_RISK_BPM:
        return "boundary_jump"
    if (
        _as_float(post_guard_mae_bpm) < POST_GUARD_PASS_THRESHOLD_BPM
        and (
            _as_float(post_motion_60s_mae_bpm) >= POST_GUARD_PASS_THRESHOLD_BPM
            or _as_float(delta_vs_lite_60s_mae_bpm)
            > FIXED_60S_MEAN_REGRESSION_LIMIT_BPM
        )
    ):
        return "late_scoring"
    if (
        str(consensus_status) in {"fallback", "failed"}
        and _as_float(post_guard_mae_bpm) >= POST_GUARD_PASS_THRESHOLD_BPM
    ):
        return "consensus_failed"
    if int(failures.get("low_lock", 0)) > 0 and int(
        failures.get("low_lock", 0)
    ) >= int(failures.get("high_lock", 0)):
        return "reset_low_lock"
    if int(failures.get("high_lock", 0)) > 0:
        return "reset_high_lock"
    if _as_float(post_guard_mae_bpm) >= POST_GUARD_PASS_THRESHOLD_BPM:
        return "reset_low_lock" if _as_float(post_guard_signed_error_bpm) < 0.0 else "reset_high_lock"
    return "pass"


def _has_source_replay_drift(
    *,
    source_mode: str,
    status: str,
    p95_diff_bpm: float,
) -> bool:
    if source_mode != SOURCE_MODE_REUSED_BO_SOURCE:
        return False
    status_norm = str(status or "").strip().lower()
    if status_norm and status_norm != "ok":
        return True
    return _as_float(p95_diff_bpm) > SOURCE_REPLAY_P95_DRIFT_THRESHOLD_BPM


def _failure_bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        bucket = str(row.get("primary_failure_bucket", ""))
        if bucket and bucket != "pass":
            counts[bucket] += 1
    return dict(counts)


def _dominant_bucket(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _candidate_gate_failure_reasons(
    *,
    source_mode: str,
    sample_count: int,
    passing_sample_count: int,
    mean_delta_vs_lite_post_mae_bpm: float,
    max_regression_delta_bpm: float,
    mean_delta_vs_lite_60s_mae_bpm: float,
    max_boundary_jump_bpm: float,
    high_drift_sample_count: int,
    high_drift_improved_count: int,
    max_source_replay_p95_diff_bpm: float,
    source_replay_statuses: list[str],
) -> list[str]:
    reasons: list[str] = []
    if source_mode == SOURCE_MODE_FIXED_LITE_SOURCE:
        reasons.append("diagnostic_source_mode")
    if passing_sample_count < sample_count:
        reasons.append("post_guard_threshold")
    if _as_float(mean_delta_vs_lite_post_mae_bpm) > 0.0:
        reasons.append("mean_not_better_than_lite")
    if high_drift_improved_count < high_drift_sample_count:
        reasons.append("high_drift_not_improved")
    if _as_float(max_regression_delta_bpm) > MAX_SINGLE_SAMPLE_REGRESSION_BPM:
        reasons.append("non_regression")
    if _as_float(mean_delta_vs_lite_60s_mae_bpm) > FIXED_60S_MEAN_REGRESSION_LIMIT_BPM:
        reasons.append("fixed_60s_regression")
    if _as_float(max_boundary_jump_bpm) > BOUNDARY_JUMP_RISK_BPM:
        reasons.append("boundary_jump")
    if any(
        _has_source_replay_drift(
            source_mode=source_mode,
            status=status,
            p95_diff_bpm=max_source_replay_p95_diff_bpm,
        )
        for status in source_replay_statuses
    ):
        reasons.append("source_replay_drift")
    return reasons


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite_mean(values: Iterable[Any]) -> float:
    arr = np.asarray([_as_float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def run_post_motion_reset_fft_study(
    *,
    data_root: Path | str,
    lite_batch_dir: Path | str,
    output_dir: Path | str,
    configs: list[PostMotionResetConfig] | None = None,
    representative_only: bool = True,
    source_modes: Iterable[str] | None = None,
    source_replay_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_samples = enumerate_lyx_samples(data_root)
    samples = select_representative_lyx_samples(all_samples) if representative_only else all_samples
    active_configs = configs or build_representative_candidate_configs()
    active_source_modes = tuple(source_modes or (SOURCE_MODE_FIXED_LITE_SOURCE,))
    lite_rows = compute_lite_baseline_metrics(lite_batch_dir)
    lite_by_sample = {str(row["sample_id"]): row for row in lite_rows}
    source_replay_by_sample = {
        str(row.get("sample_id")): row for row in (source_replay_rows or [])
    }
    sample_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    candidate_reports: list[Path] = []

    for sample in samples:
        lite_baseline = lite_by_sample.get(sample.sample_id)
        for source_mode in active_source_modes:
            source_cfg, source_result = _source_for_mode(sample, source_mode, lite_baseline)
            motion_segment = source_result.metadata.get("motion_segment") or {}
            if "end_s" not in motion_segment:
                continue
            motion_end_s = float(motion_segment["end_s"])
            for config in active_configs:
                baseline_run = run_baseline_sample(
                    sample,
                    fft_chain=FFT_CHAIN_POST_GUARD_RESET,
                    guard_seconds=float(config.guard_seconds),
                    base_config=source_cfg,
                    post_reset_tracking=config.tracking_params(),
                    post_reset_min_bpm_floor=config.min_bpm_floor,
                    first_window_no_hold_count=int(config.first_window_no_hold_count),
                    post_reset_consensus_k=int(config.topk_k),
                    post_reset_consensus_windows=int(config.consensus_windows),
                    post_reset_consensus_tolerance_bpm=float(config.consensus_tolerance_bpm),
                )
                combined_hr = combine_source_and_reset_rows(
                    source_result.HR,
                    baseline_run.window_rows,
                    motion_end_s=motion_end_s,
                    config=config,
                )
                sample_row = summarise_candidate_metrics(
                    sample_id=sample.sample_id,
                    config=config,
                    motion_end_s=motion_end_s,
                    combined_hr=combined_hr,
                    reset_rows=baseline_run.window_rows,
                    lite_baseline=lite_baseline,
                    source_mode=source_mode,
                    source_replay=source_replay_by_sample.get(sample.sample_id),
                )
                sample_rows.append(sample_row)
                for row in baseline_run.window_rows:
                    window_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "source_mode": source_mode,
                            "candidate_name": config.name,
                            **row,
                        }
                    )
                report_path = write_candidate_v2_report(
                    output_dir=out / "json",
                    sample=sample,
                    config=config,
                    source_result=source_result,
                    combined_hr=combined_hr,
                    source_mode=source_mode,
                )
                candidate_reports.append(report_path)

    aggregate_rows = aggregate_candidate_rows(sample_rows)
    _write_dict_csv(out / "representative_sample_metrics.csv", sample_rows)
    _write_dict_csv(out / "representative_window_metrics.csv", window_rows)
    _write_dict_csv(out / "candidate_aggregate_metrics.csv", aggregate_rows)
    _write_dict_csv(out / "lite_baseline_post_motion_metrics.csv", lite_rows)
    render_candidate_plots(candidate_reports, out / "png", out / "csv")
    report_path = out / "post_motion_reset_fft_reacquire_report.md"
    report_path.write_text(
        render_reacquire_markdown_report(
            sample_rows=sample_rows,
            aggregate_rows=aggregate_rows,
            output_dir=out,
            representative_only=representative_only,
        ),
        encoding="utf-8",
    )
    metadata = {
        "data_root": str(Path(data_root)),
        "lite_batch_dir": str(Path(lite_batch_dir)),
        "output_dir": str(out),
        "sample_count": len(samples),
        "representative_only": bool(representative_only),
        "candidate_count": len(active_configs),
        "source_modes": list(active_source_modes),
        "report_path": str(report_path),
    }
    (out / "post_motion_reset_fft_reacquire_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "metadata": metadata,
        "sample_rows": sample_rows,
        "aggregate_rows": aggregate_rows,
        "candidate_reports": candidate_reports,
    }


def _source_for_mode(
    sample: BaselineSample,
    source_mode: str,
    lite_baseline: dict[str, Any] | None,
) -> tuple[V2RunConfig, Any]:
    if source_mode not in SOURCE_MODES:
        allowed = ", ".join(SOURCE_MODES)
        raise ValueError(f"Unsupported source_mode={source_mode!r}; expected one of {allowed}")
    if source_mode == SOURCE_MODE_FIXED_LITE_SOURCE:
        cfg = V2RunConfig(
            data_path=sample.data_path,
            ref_path=sample.ref_path,
            algorithm_preset="lite",
            reference_groups_order=("HF",),
            adaptive_filter="lms",
            analysis_scope="full",
            post_motion_reacquire_enable=False,
        )
        return cfg, solve_v2(cfg)
    if not lite_baseline:
        raise ValueError(f"Missing Lite baseline row for sample_id={sample.sample_id}")
    report_path = Path(str(lite_baseline["lite_report_path"]))
    cfg = load_lite_report_config(report_path)
    if source_mode == SOURCE_MODE_REUSED_BO_SOURCE:
        return cfg, solve_v2(cfg)
    payload = _load_report_payload(report_path)
    hr_rows = _read_hr_csv(Path(str(lite_baseline["lite_hr_csv"])))
    return cfg, SimpleNamespaceLike(
        HR=_hr_rows_to_array(hr_rows),
        metadata=payload,
        err_stats={},
        window_table=list(payload.get("window_table", [])),
    )


@dataclass
class SimpleNamespaceLike:
    HR: np.ndarray
    metadata: dict[str, Any]
    err_stats: dict[str, Any]
    window_table: list[Any]


def _hr_rows_to_array(rows: list[dict[str, float]]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(row["time_s"]),
                float(row["ref_bpm"]),
                float(row["fft_bpm"]),
                float(row["final_bpm"]),
                float(row.get("is_motion", 0.0)),
                float(row.get("used_adaptive", 0.0)),
            ]
            for row in rows
        ],
        dtype=float,
    )


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_candidate_v2_report(
    *,
    output_dir: Path | str,
    sample: BaselineSample,
    config: PostMotionResetConfig,
    source_result: Any,
    combined_hr: np.ndarray,
    source_mode: str | None = None,
) -> Path:
    metadata = dict(source_result.metadata)
    reset_metadata = {
        "candidate_name": config.name,
        "guard_seconds": float(config.guard_seconds),
        "peak_strategy": config.peak_strategy,
        "min_bpm_floor": config.min_bpm_floor,
        "range_up_bpm": float(config.range_up_bpm),
        "range_down_bpm": float(config.range_down_bpm),
        "step_up_bpm": float(config.step_up_bpm),
        "step_down_bpm": float(config.step_down_bpm),
        "first_window_no_hold_count": int(config.first_window_no_hold_count),
        "topk_k": int(config.topk_k),
        "consensus_windows": int(config.consensus_windows),
        "consensus_tolerance_bpm": float(config.consensus_tolerance_bpm),
        "boundary_strategy": config.boundary_strategy,
        "bridge_windows": int(config.bridge_windows),
        "adaptive_fallback_windows": int(config.adaptive_fallback_windows),
    }
    if source_mode is not None:
        reset_metadata["source_mode"] = source_mode
    metadata.update(
        {
            "data_path": str(sample.data_path),
            "ref_path": str(sample.ref_path),
            "algorithm_preset": metadata.get("algorithm_preset", "lite"),
            "reference_groups_order": list(metadata.get("reference_groups_order", ["HF"])),
            "post_motion_reset_fft": reset_metadata,
        }
    )
    result = V2SolverResult(
        HR=np.asarray(combined_hr, dtype=float),
        err_stats=_candidate_err_stats(combined_hr),
        metadata=metadata,
        window_table=list(getattr(source_result, "window_table", [])),
    )
    out = Path(output_dir)
    prefix = (
        f"{sample.sample_id}-{source_mode}-{config.name}"
        if source_mode
        else f"{sample.sample_id}-{config.name}"
    )
    return save_v2_report(out / f"{prefix}-v2.json", result, best_params={}, history=[])


def render_candidate_plots(
    report_paths: list[Path],
    png_dir: Path | str,
    csv_dir: Path | str,
) -> list[Path]:
    figures: list[Path] = []
    for report_path in report_paths:
        prefix = (
            report_path.name[: -len("-v2.json")]
            if report_path.name.endswith("-v2.json")
            else report_path.stem
        )
        arte = render_v2_report(
            report_path,
            out_dir=Path(png_dir),
            csv_dir=Path(csv_dir),
            output_prefix=prefix,
            comparison_groups=(("ACC",),),
        )
        figures.append(arte.figure_png)
    return figures


def _candidate_err_stats(hr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(hr, dtype=float)
    if arr.size == 0:
        return {"fft_aae_bpm": float("nan"), "final_aae_bpm": float("nan")}
    fft_errors = np.abs(arr[:, 2] - arr[:, 1])
    final_errors = np.abs(arr[:, 3] - arr[:, 1])
    finite_fft = fft_errors[np.isfinite(fft_errors)]
    finite_final = final_errors[np.isfinite(final_errors)]
    return {
        "fft_aae_bpm": float(np.mean(finite_fft)) if finite_fft.size else float("nan"),
        "final_aae_bpm": float(np.mean(finite_final)) if finite_final.size else float("nan"),
    }


def render_reacquire_markdown_report(
    *,
    sample_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    output_dir: Path,
    representative_only: bool,
) -> str:
    best = _best_candidate(aggregate_rows)
    scope = "代表样本" if representative_only else "LYX 全量样本"
    conclusion = _conclusion_text(best)
    lines = [
        "# 运动后静息 Reset FFT 重捕获实验报告",
        "",
        "## 结论",
        "",
        conclusion,
        "",
        f"- 评估范围：{scope}",
        f"- 输出目录：`{output_dir}`",
        "",
        "## Go/No-Go",
        "",
        "| Source mode | 候选 | 结论 | 未通过门槛 | 主导失败桶 | 高漂移改善 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(
        aggregate_rows,
        key=lambda item: (
            0 if str(item.get("gate_decision", "")) == "go" else 1,
            _as_float(item.get("mean_post_guard_final_mae_bpm")),
        ),
    ):
        display = _display_row(row)
        lines.append(
            "| {source_mode} | {candidate_name} | {gate_label} | {gate_failure_reasons} | "
            "{dominant_failure_bucket} | {high_drift_improved_count}/{high_drift_sample_count} |".format(
                **display
            )
        )
    lines.extend(
        [
        "",
        "## 候选总览",
        "",
        "| Source mode | 候选 | N | 通过数 | post-guard MAE | 60s MAE | vs Lite | 60s delta | 最大退化样本 | 最大退化 | 最大边界跳变 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in sorted(
        aggregate_rows,
        key=lambda item: (
            _as_float(item.get("mean_post_guard_final_mae_bpm")),
            -int(item.get("passing_sample_count", 0)),
        ),
    ):
        display = _display_row(row)
        lines.append(
            "| {source_mode} | {candidate_name} | {sample_count} | {passing_sample_count} | "
            "{mean_post_guard_final_mae_bpm:.3f} | "
            "{mean_post_motion_60s_final_mae_bpm:.3f} | "
            "{mean_delta_vs_lite_post_mae_bpm:.3f} | "
            "{mean_delta_vs_lite_60s_mae_bpm:.3f} | {max_regression_sample_id} | "
            "{max_regression_delta_bpm:.3f} | {max_boundary_jump_bpm:.3f} |".format(
                **display
            )
        )
    if any(str(row.get("source_mode")) == SOURCE_MODE_FIXED_LITE_SOURCE for row in aggregate_rows):
        lines.extend(
            [
                "",
                f"`{SOURCE_MODE_FIXED_LITE_SOURCE}` 仅作为诊断 source 对照，不作为最终推荐候选。",
            ]
        )

    bucket_rows = _failure_bucket_summary_rows(sample_rows)
    lines.extend(
        [
            "",
            "## 失败桶",
            "",
            "| 失败桶 | 样本数 | 代表案例 | 下一步 |",
            "| --- | ---: | --- | --- |",
        ]
    )
    if bucket_rows:
        for row in bucket_rows:
            lines.append(
                "| {bucket} | {count} | {cases} | {next_step} |".format(**row)
            )
    else:
        lines.append("| 无 | 0 | 无 | 可进入候选复核 |")

    failing = [
        row
        for row in sample_rows
        if (not bool(row.get("passes_post_guard_3bpm")))
        or _as_float(row.get("delta_vs_lite_post_mae_bpm")) > 1.0
        or str(row.get("primary_failure_bucket", "")) not in ("", "pass")
    ]
    lines.extend(
        [
            "",
            "## 失败样本",
            "",
            "| 样本 | Source mode | 候选 | 失败桶 | 新 post-guard MAE | vs Lite | 60s MAE | source replay P95 | 边界跳变 | 主判断 |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if failing:
        for row in sorted(
            failing,
            key=lambda item: _as_float(item.get("post_guard_final_mae_bpm")),
            reverse=True,
        ):
            reason = (
                "退化复核"
                if _as_float(row.get("delta_vs_lite_post_mae_bpm")) > 1.0
                else "未达 <3 BPM"
            )
            display = _display_row(row)
            lines.append(
                "| {sample_id} | {source_mode} | {candidate_name} | {primary_failure_bucket} | "
                "{post_guard_final_mae_bpm:.3f} | "
                "{delta_vs_lite_post_mae_bpm:.3f} | "
                "{post_motion_60s_final_mae_bpm:.3f} | "
                "{source_replay_p95_diff_bpm:.3f} | {boundary_jump_bpm:.3f} | "
                f"{reason} |".format(**display)
            )
    else:
        lines.append("| 无 | 无 | 无 | 无 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 全部通过 |")

    lines.extend(
        [
            "",
            "## 边界平滑",
            "",
            "| 候选 | 最大边界跳变 | P95 跳变最大样本值 | 超 20 BPM 样本数 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for source_mode, name in sorted(
        {
            (str(row.get("source_mode", "")), str(row["candidate_name"]))
            for row in sample_rows
        }
    ):
        items = [
            row
            for row in sample_rows
            if str(row.get("source_mode", "")) == source_mode
            and str(row["candidate_name"]) == name
        ]
        display_name = f"{source_mode} / {name}" if source_mode else name
        lines.append(
            f"| {display_name} | "
            f"{max(_as_float(row.get('boundary_jump_bpm')) for row in items):.3f} | "
            f"{max(_as_float(row.get('boundary_p95_abs_jump_bpm')) for row in items):.3f} | "
            f"{sum(1 for row in items if _as_float(row.get('boundary_jump_bpm')) > 20.0)} |"
        )

    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `representative_sample_metrics.csv`",
            "- `representative_window_metrics.csv`",
            "- `candidate_aggregate_metrics.csv`",
            "- `lite_baseline_post_motion_metrics.csv`",
            "- `png/`：v2 批量全流程样式心率图，包含 ACC 对比参考信号曲线。",
            "",
            "## 验收说明",
            "",
            "最终验收需由独立子代理复核：motion_end 口径、旧 Lite 对照路径、PNG 的 ACC 对比曲线、报告结论与 CSV 数据一致性。",
        ]
    )
    return "\n".join(lines) + "\n"


def _display_row(row: dict[str, Any]) -> dict[str, Any]:
    display = dict(row)
    display.setdefault("source_mode", "")
    display.setdefault("gate_decision", "")
    display.setdefault("gate_failure_reasons", "")
    display.setdefault("dominant_failure_bucket", "")
    display.setdefault("high_drift_improved_count", 0)
    display.setdefault("high_drift_sample_count", 0)
    display.setdefault("mean_delta_vs_lite_60s_mae_bpm", float("nan"))
    display.setdefault("primary_failure_bucket", "")
    display.setdefault("source_replay_p95_diff_bpm", float("nan"))
    display.setdefault("boundary_jump_bpm", float("nan"))
    display["gate_label"] = "GO" if display.get("gate_decision") == "go" else "NO-GO"
    return display


def _failure_bucket_summary_rows(sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket_counts = _failure_bucket_counts(sample_rows)
    rows: list[dict[str, Any]] = []
    for bucket, count in sorted(bucket_counts.items(), key=lambda item: (-item[1], item[0])):
        bucket_items = [
            row
            for row in sample_rows
            if str(row.get("primary_failure_bucket", "")) == bucket
        ]
        cases = [
            _case_label(row)
            for row in sorted(
                bucket_items,
                key=lambda item: _as_float(item.get("post_guard_final_mae_bpm")),
                reverse=True,
            )[:2]
        ]
        rows.append(
            {
                "bucket": bucket,
                "count": count,
                "cases": "; ".join(cases),
                "next_step": _bucket_next_step(bucket),
            }
        )
    return rows


def _case_label(row: dict[str, Any]) -> str:
    source_mode = str(row.get("source_mode", ""))
    candidate = str(row.get("candidate_name", ""))
    sample = str(row.get("sample_id", ""))
    source_and_candidate = f"{source_mode}/{candidate}" if source_mode else candidate
    return f"{sample} ({source_and_candidate})"


def _bucket_next_step(bucket: str) -> str:
    return {
        "source_replay_drift": "优先复核旧 Lite replay 或使用 old_hr_prefix_splice",
        "reset_low_lock": "进入首窗峰共识或 floor reset 诊断",
        "consensus_failed": "放宽 top-k 共识窗口或增加 adaptive fallback",
        "reset_high_lock": "复核运动残余峰和上升限幅",
        "boundary_jump": "进入边界平滑或 adaptive fallback 诊断",
        "late_scoring": "缩短保护窗或保留固定 60s 约束复核",
    }.get(bucket, "人工复核窗口级证据")


def _best_candidate(aggregate_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not aggregate_rows:
        return None
    formal_rows = [
        row
        for row in aggregate_rows
        if str(row.get("source_mode", "")) != SOURCE_MODE_FIXED_LITE_SOURCE
    ]
    go_rows = [row for row in formal_rows if str(row.get("gate_decision", "")) == "go"]
    rows = go_rows or formal_rows or aggregate_rows
    return sorted(
        rows,
        key=lambda row: (
            0 if str(row.get("gate_decision", "")) == "go" else 1,
            _as_float(row.get("mean_post_guard_final_mae_bpm")),
            _as_float(row.get("mean_post_motion_60s_final_mae_bpm")),
        ),
    )[0]


def _conclusion_text(best: dict[str, Any] | None) -> str:
    if best is None:
        return "本轮没有产生可评估候选，暂不采用任何 reset FFT 重捕获参数。"
    delta = _as_float(best.get("mean_delta_vs_lite_post_mae_bpm"))
    regression = _as_float(best.get("max_regression_delta_bpm"))
    boundary_jump = _as_float(best.get("max_boundary_jump_bpm"))
    passing = int(best.get("passing_sample_count", 0) or 0)
    name = str(best.get("candidate_name", ""))
    source_mode = str(best.get("source_mode", ""))
    display_name = f"{source_mode} / {name}" if source_mode else name
    gate_decision = str(best.get("gate_decision", ""))
    if gate_decision == "go":
        return (
            f"GO：推荐候选 `{display_name}` 进入 LYX 全量复核；代表样本门槛已通过，"
            "下一步只保留少量候选做全量非回归验证。"
        )
    if gate_decision == "no_go":
        reasons = str(best.get("gate_failure_reasons", ""))
        bucket = str(best.get("dominant_failure_bucket", "")) or "未归类"
        return (
            f"NO-GO：`{display_name}` 未通过代表样本门槛（{reasons}），"
            f"主导失败桶为 `{bucket}`；本轮暂不进入 LYX 全量，应先按失败桶继续机制诊断。"
        )
    if passing <= 0:
        return f"`{display_name}` 的代表样本均未达到 <3 BPM，本轮暂不采用，需要继续诊断 reset 低锁和保护窗边界。"
    if regression > 2.0:
        return f"`{display_name}` 出现超过 2 BPM 的单样本退化，本轮暂不采用，需要先复核失败样本。"
    if boundary_jump > 20.0:
        return f"`{display_name}` 的边界跳变超过 20 BPM，本轮暂不采用，需要重新约束切换边界。"
    if delta < 0.0 and regression <= 2.0:
        return (
            f"推荐候选 `{display_name}` 进入 LYX 全量复核：平均运动后误差优于旧 Lite 对照，"
            "且未出现超过 2 BPM 的不可接受退化。"
        )
    if delta < 0.0:
        return f"`{display_name}` 平均优于旧 Lite 对照，但存在单样本退化，需要先复核失败样本再继续推进。"
    return f"`{display_name}` 未超过旧 Lite 对照，本轮暂不采用，需要继续诊断 reset 后低锁或保护窗边界。"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--lite-batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--all-lyx", action="store_true", help="run selected candidates on all LYX samples")
    args = parser.parse_args(argv)
    result = run_post_motion_reset_fft_study(
        data_root=args.data_root,
        lite_batch_dir=args.lite_batch_dir,
        output_dir=args.output_dir,
        representative_only=not bool(args.all_lyx),
    )
    print(f"report={result['metadata']['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
