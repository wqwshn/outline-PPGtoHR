"""v2 SpO2 computation with independent Ut1/Ut2 PPG recovery paths."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from ..core.adaptive_filter import apply_adaptive_cascade
from .output_paths import prepare_output_dir, safe_output_path
from .preprocess import RAW_COLUMNS, safe_cf_ratio
from .reference_groups import channel_names_for_group, normalise_reference_order

MAX30101_FULL_SCALE_NA = 16384.0
MAX30101_ADC_LEVELS = float(2**18)
MAX30101_UA_PER_COUNT = (MAX30101_FULL_SCALE_NA / MAX30101_ADC_LEVELS) / 1000.0


@dataclass(frozen=True)
class V2SpO2Coefficients:
    a: float = 1.5958422
    b: float = -34.6596622
    c: float = 112.6898759


@dataclass(frozen=True)
class V2SpO2Config:
    data_path: Path
    output_dir: Path | None = None
    reference_groups_order: tuple[str, ...] = ("HF",)
    fs_origin: int = 100
    window_seconds: float = 4.0
    window_step_seconds: float = 1.0
    delay_search_samples: int = 20
    max_order: int = 20
    min_order: int = 1
    adaptive_filter: str = "lms"
    lms_mu_base: float = 0.12
    lms_mu_min: float = 1e-6
    M_base: int = 1
    C_scale: float = 1.0
    K_max: int = 16
    klms_step_size: float = 0.1
    klms_sigma: float = 1.0
    klms_epsilon: float = 0.1
    as_lms_rho: float = 1e-4
    as_lms_mu_max: float = 0.05
    volterra_max_order_vol: int = 3
    rff_D: int = 100
    rff_sigma: float = 1.0
    rff_seed: int = 42
    adaptive_enabled: bool = True
    deglitch_enabled: bool = True
    deglitch_window_seconds: float = 0.25
    deglitch_n_sigmas: float = 6.0
    ppg_lowpass_enabled: bool = True
    ppg_lowpass_cutoff_hz: float = 8.0
    ppg_lowpass_order: int = 3
    reference_lowpass_enabled: bool = True
    reference_lowpass_cutoff_hz: float = 5.0
    reference_lowpass_order: int = 3
    bp_low_hz: float = 0.5
    bp_high_hz: float = 5.0
    lp_cutoff_hz: float = 8.0
    filter_order: int = 3
    min_beat_interval_seconds: float = 0.40
    valley_search_seconds: float = 0.12
    peak_search_seconds: float = 0.16
    smooth_seconds: float = 0.06
    spo2_smooth_seconds: float = 7.0
    rest_motion_score_threshold: float = 0.02
    motion_threshold_mode: str = "adaptive"
    motion_threshold_quantile: float = 0.35
    motion_threshold_mad_scale: float = 6.0
    motion_threshold_min_delta: float = 0.005
    motion_context_seconds: float = 2.0
    r_min: float = 0.05
    r_max: float = 3.0
    coefficients: V2SpO2Coefficients = field(default_factory=V2SpO2Coefficients)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class V2SpO2Result:
    spo2_table: list[dict[str, Any]]
    beat_table: list[dict[str, Any]]
    metadata: dict[str, Any]
    waveforms: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class SpO2RawSignals:
    fs: int
    time_s: np.ndarray
    red: np.ndarray
    ir: np.ndarray
    references: dict[str, np.ndarray]
    valid_mask: np.ndarray
    red_original: np.ndarray
    ir_original: np.ndarray
    artifact_rejection: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class CleanedSpO2Signals:
    red_clean: np.ndarray
    ir_clean: np.ndarray
    stages: list[dict[str, Any]]


def spo2_from_r(
    r: np.ndarray | float,
    coefficients: V2SpO2Coefficients | None = None,
) -> np.ndarray:
    coeffs = coefficients or V2SpO2Coefficients()
    values = np.asarray(r, dtype=float)
    raw = coeffs.a * values**2 + coeffs.b * values + coeffs.c
    return np.clip(raw, 0.0, 100.0)


def _ppg_adc_to_ua(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) * MAX30101_UA_PER_COUNT


def solve_spo2_v2(config: V2SpO2Config) -> V2SpO2Result:
    cfg = V2SpO2Config(
        **{
            **config.__dict__,
            "reference_groups_order": normalise_reference_order(
                config.reference_groups_order
            ),
        }
    )
    signals = _load_spo2_raw_signals(cfg)
    fs = int(signals.fs)
    window_len = int(round(float(cfg.window_seconds) * fs))
    step_len = int(round(float(cfg.window_step_seconds) * fs))
    if window_len <= 0 or step_len <= 0:
        raise ValueError("window_seconds and window_step_seconds must be positive")
    if signals.red.size < window_len:
        raise ValueError(
            f"Need at least one {cfg.window_seconds:g}s SpO2 window, got "
            f"{signals.red.size / fs:.2f}s"
        )

    spo2_table: list[dict[str, Any]] = []
    beat_table: list[dict[str, Any]] = []
    stage_rows_ut1: list[list[dict[str, Any]]] = []
    stage_rows_ut2: list[list[dict[str, Any]]] = []
    acc_mag = np.sqrt(
        signals.references["accx"] ** 2
        + signals.references["accy"] ** 2
        + signals.references["accz"] ** 2
    )
    window_rows: list[dict[str, Any]] = []
    for window_idx, start in enumerate(range(0, signals.red.size - window_len + 1, step_len)):
        end = start + window_len
        motion_score = float(np.std(acc_mag[start:end], ddof=1))
        window_rows.append(
            {
                "window_idx": int(window_idx),
                "start": int(start),
                "end": int(end),
                "start_s": float(signals.time_s[start]),
                "end_s": float(signals.time_s[end - 1]),
                "center_s": float(signals.time_s[start] + cfg.window_seconds / 2.0),
                "motion_score": motion_score,
            }
        )
    motion_segments, recovery_segments, motion_threshold = _detect_motion_segments(
        window_rows,
        total_samples=int(signals.red.size),
        fs=fs,
        cfg=cfg,
    )
    red_ut1, ir_ut1, recovery_stages_ut1 = _recover_motion_segments_single_reference(
        signals.red,
        signals.ir,
        signals.references["hf1"],
        recovery_segments,
        channel="hf1",
        fs=fs,
        cfg=cfg,
    )
    red_ut2, ir_ut2, recovery_stages_ut2 = _recover_motion_segments_single_reference(
        signals.red,
        signals.ir,
        signals.references["hf2"],
        recovery_segments,
        channel="hf2",
        fs=fs,
        cfg=cfg,
    )
    last_raw_spo2 = float("nan")
    last_spo2_ut1 = float("nan")
    last_spo2_ut2 = float("nan")

    for spec in window_rows:
        window_idx = int(spec["window_idx"])
        start = int(spec["start"])
        end = int(spec["end"])
        motion_score = float(spec["motion_score"])
        recovery_applied = _window_overlaps_segments(start, end, motion_segments)
        if recovery_applied:
            window_red_ut1 = red_ut1[start:end]
            window_ir_ut1 = ir_ut1[start:end]
            window_red_ut2 = red_ut2[start:end]
            window_ir_ut2 = ir_ut2[start:end]
            window_stages_ut1 = [
                stage
                for stage in recovery_stages_ut1
                if start < int(stage["end"]) and end > int(stage["start"])
            ]
            window_stages_ut2 = [
                stage
                for stage in recovery_stages_ut2
                if start < int(stage["end"]) and end > int(stage["start"])
            ]
        else:
            window_red_ut1 = signals.red[start:end]
            window_ir_ut1 = signals.ir[start:end]
            window_red_ut2 = signals.red[start:end]
            window_ir_ut2 = signals.ir[start:end]
            window_stages_ut1 = []
            window_stages_ut2 = []
        stage_rows_ut1.append(window_stages_ut1)
        stage_rows_ut2.append(window_stages_ut2)

        raw_out = _compute_spo2_window(
            red=signals.red[start:end],
            ir=signals.ir[start:end],
            fs=fs,
            cfg=cfg,
            scheme="raw",
        )
        ut1_out = _compute_spo2_window(
            red=window_red_ut1,
            ir=window_ir_ut1,
            fs=fs,
            cfg=cfg,
            scheme="ut1",
        )
        ut2_out = _compute_spo2_window(
            red=window_red_ut2,
            ir=window_ir_ut2,
            fs=fs,
            cfg=cfg,
            scheme="ut2",
        )

        raw_spo2 = float(raw_out["spo2"])
        spo2_ut1 = float(ut1_out["spo2"])
        spo2_ut2 = float(ut2_out["spo2"])
        raw_carried = False
        carried_ut1 = False
        carried_ut2 = False
        if np.isfinite(raw_spo2):
            last_raw_spo2 = raw_spo2
        elif np.isfinite(last_raw_spo2):
            raw_spo2 = last_raw_spo2
            raw_carried = True
        if np.isfinite(spo2_ut1):
            last_spo2_ut1 = spo2_ut1
        elif np.isfinite(last_spo2_ut1):
            spo2_ut1 = last_spo2_ut1
            carried_ut1 = True
        if np.isfinite(spo2_ut2):
            last_spo2_ut2 = spo2_ut2
        elif np.isfinite(last_spo2_ut2):
            spo2_ut2 = last_spo2_ut2
            carried_ut2 = True

        missing_ratio = 1.0 - float(np.mean(signals.valid_mask[start:end]))
        center_s = float(spec["center_s"])
        row = {
            "window_idx": int(window_idx),
            "start_s": float(spec["start_s"]),
            "end_s": float(spec["end_s"]),
            "center_s": center_s,
            "motion_score": motion_score,
            "recovery_applied": recovery_applied,
            "raw_spo2": raw_spo2,
            "spo2_ut1": spo2_ut1,
            "spo2_ut2": spo2_ut2,
            "raw_r_median": float(raw_out["r_median"]),
            "r_median_ut1": float(ut1_out["r_median"]),
            "r_median_ut2": float(ut2_out["r_median"]),
            "raw_valid_beat_count": int(raw_out["valid_beat_count"]),
            "valid_beat_count_ut1": int(ut1_out["valid_beat_count"]),
            "valid_beat_count_ut2": int(ut2_out["valid_beat_count"]),
            "raw_carried_forward": raw_carried,
            "carried_forward_ut1": carried_ut1,
            "carried_forward_ut2": carried_ut2,
            "missing_ratio": missing_ratio,
            "reliable_raw": bool(
                missing_ratio <= 0.20
                and int(raw_out["valid_beat_count"]) > 0
            ),
            "reliable_ut1": bool(
                missing_ratio <= 0.20
                and int(ut1_out["valid_beat_count"]) > 0
            ),
            "reliable_ut2": bool(
                missing_ratio <= 0.20
                and int(ut2_out["valid_beat_count"]) > 0
            ),
        }
        spo2_table.append(row)
        for beat in raw_out["beat_rows"] + ut1_out["beat_rows"] + ut2_out["beat_rows"]:
            beat_table.append(
                {
                    "window_idx": int(window_idx),
                    "window_center_s": center_s,
                    **beat,
                }
            )

    _smooth_spo2_table(spo2_table, cfg)
    _apply_rest_adaptive_policy(spo2_table, cfg)
    stability_summary = _spo2_stability_summary(spo2_table, motion_segments)

    metadata = {
        "schema_version": "v2_spo2",
        "data_path": str(cfg.data_path),
        "fs": fs,
        "window_seconds": float(cfg.window_seconds),
        "window_step_seconds": float(cfg.window_step_seconds),
        "spo2_smooth_seconds": float(cfg.spo2_smooth_seconds),
        "rest_motion_score_threshold": float(cfg.rest_motion_score_threshold),
        "delay_search_samples": int(cfg.delay_search_samples),
        "max_order": int(cfg.max_order),
        "reference_groups_order": list(cfg.reference_groups_order),
        "adaptive_filter": str(cfg.adaptive_filter),
        "adaptive_enabled": bool(cfg.adaptive_enabled),
        "ppg_lowpass_enabled": bool(cfg.ppg_lowpass_enabled),
        "ppg_lowpass_cutoff_hz": float(cfg.ppg_lowpass_cutoff_hz),
        "ppg_lowpass_order": int(cfg.ppg_lowpass_order),
        "reference_lowpass_enabled": bool(cfg.reference_lowpass_enabled),
        "reference_lowpass_cutoff_hz": float(cfg.reference_lowpass_cutoff_hz),
        "reference_lowpass_order": int(cfg.reference_lowpass_order),
        "artifact_rejection": signals.artifact_rejection,
        "motion_threshold": float(motion_threshold),
        "motion_segments": motion_segments,
        "continuous_recovery_segments": recovery_segments,
        "recovery_stage_rows_ut1": recovery_stages_ut1,
        "recovery_stage_rows_ut2": recovery_stages_ut2,
        "spo2_stability_summary": stability_summary,
        "stage_rows_ut1": stage_rows_ut1,
        "stage_rows_ut2": stage_rows_ut2,
    }
    waveforms = {
        "time_s": signals.time_s,
        "red_preprocessed": signals.red,
        "ir_preprocessed": signals.ir,
        "red_ut1": red_ut1,
        "ir_ut1": ir_ut1,
        "red_ut2": red_ut2,
        "ir_ut2": ir_ut2,
        "acc_mag": acc_mag,
        "motion_window_center_s": np.asarray(
            [row["center_s"] for row in window_rows],
            dtype=float,
        ),
        "motion_score": np.asarray(
            [row["motion_score"] for row in window_rows],
            dtype=float,
        ),
        "ut1": signals.references.get("hf1", np.asarray([], dtype=float)),
        "ut2": signals.references.get("hf2", np.asarray([], dtype=float)),
    }
    return V2SpO2Result(
        spo2_table=spo2_table,
        beat_table=beat_table,
        metadata=metadata,
        waveforms=waveforms,
    )


def save_spo2_report(
    result: V2SpO2Result,
    *,
    out_dir: str | Path,
    output_prefix: str,
) -> dict[str, Path]:
    out = prepare_output_dir(out_dir)
    prefix = str(output_prefix).strip() or "spo2"
    json_path = safe_output_path(out, f"{prefix}-spo2.json")
    csv_path = safe_output_path(out, f"{prefix}-spo2.csv")
    waveform_csv_path = safe_output_path(out, f"{prefix}-spo2-waveforms.csv")
    payload = {
        "schema_version": "v2_spo2",
        "metadata": _jsonify(result.metadata),
        "spo2_table": _jsonify(result.spo2_table),
        "beat_table": _jsonify(result.beat_table),
        "waveforms": _jsonify(result.waveforms),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = result.spo2_table
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    time_s = np.asarray(result.waveforms["time_s"], dtype=float)
    sample_count = int(time_s.size)

    def waveform(name: str) -> np.ndarray:
        values = np.asarray(result.waveforms[name], dtype=float)
        if values.size != sample_count:
            raise ValueError(
                f"Waveform {name!r} has {values.size} samples; expected {sample_count}"
            )
        return values

    motion_center_s = np.asarray(
        result.waveforms.get("motion_window_center_s", []),
        dtype=float,
    )
    motion_values = np.asarray(result.waveforms.get("motion_score", []), dtype=float)
    if motion_center_s.size and motion_center_s.size == motion_values.size:
        sample_motion_score = np.interp(
            time_s,
            motion_center_s,
            motion_values,
            left=float(motion_values[0]),
            right=float(motion_values[-1]),
        )
    else:
        sample_motion_score = np.full(sample_count, float("nan"), dtype=float)

    waveform_columns = [
        "time_s",
        "red_preprocessed_ua",
        "ir_preprocessed_ua",
        "red_ut1_ua",
        "ir_ut1_ua",
        "red_ut2_ua",
        "ir_ut2_ua",
        "ut1_mv",
        "ut2_mv",
        "motion_score",
    ]
    waveform_matrix = np.column_stack(
        [
            time_s,
            _ppg_adc_to_ua(waveform("red_preprocessed")),
            _ppg_adc_to_ua(waveform("ir_preprocessed")),
            _ppg_adc_to_ua(waveform("red_ut1")),
            _ppg_adc_to_ua(waveform("ir_ut1")),
            _ppg_adc_to_ua(waveform("red_ut2")),
            _ppg_adc_to_ua(waveform("ir_ut2")),
            waveform("ut1"),
            waveform("ut2"),
            sample_motion_score,
        ]
    )
    with waveform_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(waveform_columns)
        writer.writerows(waveform_matrix)

    return {
        "json": json_path,
        "csv": csv_path,
        "waveform_csv": waveform_csv_path,
    }


def load_spo2_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "v2_spo2":
        raise ValueError(f"{path} is not a v2 SpO2 report")
    return payload


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer | np.floating | np.bool_):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonify(v) for v in obj]
    return obj


def _centered_finite_moving_average(values: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = arr.copy()
    if arr.size == 0 or width <= 1:
        return out
    half = int(width) // 2
    for idx in range(arr.size):
        start = max(0, idx - half)
        end = min(arr.size, idx + half + 1)
        segment = arr[start:end]
        finite = segment[np.isfinite(segment)]
        out[idx] = float(np.mean(finite)) if finite.size else float("nan")
    return out


def _smooth_spo2_table(rows: list[dict[str, Any]], cfg: V2SpO2Config) -> None:
    if not rows:
        return
    step = max(float(cfg.window_step_seconds), 1e-9)
    width = max(1, int(round(float(cfg.spo2_smooth_seconds) / step)))
    if width % 2 == 0:
        width += 1

    for key in ("raw_spo2", "spo2_ut1", "spo2_ut2"):
        values = np.asarray([float(row.get(key, float("nan"))) for row in rows])
        smoothed = _centered_finite_moving_average(values, width)
        for row, value, original in zip(rows, smoothed, values, strict=True):
            row.setdefault(f"{key}_unsmoothed", float(original))
            row[key] = float(value)


def _apply_rest_adaptive_policy(
    rows: list[dict[str, Any]],
    cfg: V2SpO2Config,
) -> None:
    for row in rows:
        recovery_applied = bool(row.get("recovery_applied", False))
        if recovery_applied:
            continue
        for suffix in ("ut1", "ut2"):
            row[f"spo2_{suffix}"] = float(row.get("raw_spo2", float("nan")))
            row[f"r_median_{suffix}"] = float(
                row.get("raw_r_median", float("nan"))
            )
            row[f"valid_beat_count_{suffix}"] = int(
                row.get("raw_valid_beat_count", 0)
            )
            row[f"carried_forward_{suffix}"] = bool(
                row.get("raw_carried_forward", False)
            )
            row[f"reliable_{suffix}"] = bool(row.get("reliable_raw", False))


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _spo2_stability_summary(
    rows: list[dict[str, Any]],
    motion_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows or not motion_segments:
        empty = {
            "motion_mean": float("nan"),
            "rest_reference_mean": float("nan"),
            "motion_delta_vs_rest": float("nan"),
        }
        return {"has_motion": False, "raw": empty, "ut1": empty, "ut2": empty}
    first_start = float(motion_segments[0]["start_s"])
    last_end = float(motion_segments[-1]["end_s"])
    pre_rest = [row for row in rows if float(row.get("center_s", 0.0)) < first_start]
    post_rest = [row for row in rows if float(row.get("center_s", 0.0)) > last_end]
    motion_rows = [
        row
        for row in rows
        if first_start <= float(row.get("center_s", 0.0)) <= last_end
    ]
    rest_reference = _finite_mean(
        [float(row.get("raw_spo2", float("nan"))) for row in pre_rest + post_rest]
    )
    result: dict[str, Any] = {
        "has_motion": True,
        "motion_start_s": first_start,
        "motion_end_s": last_end,
    }
    for label, key in (
        ("raw", "raw_spo2"),
        ("ut1", "spo2_ut1"),
        ("ut2", "spo2_ut2"),
    ):
        motion_mean = _finite_mean(
            [float(row.get(key, float("nan"))) for row in motion_rows]
        )
        result[label] = {
            "motion_mean": motion_mean,
            "rest_reference_mean": rest_reference,
            "motion_delta_vs_rest": (
                abs(motion_mean - rest_reference)
                if np.isfinite(motion_mean) and np.isfinite(rest_reference)
                else float("nan")
            ),
        }
    return result


def _clean_numeric_array(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
        dtype=float,
        copy=True,
    )
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if finite.any():
        idx = np.flatnonzero(finite)
        bad = np.flatnonzero(~finite)
        arr[bad] = np.interp(bad, idx, arr[idx])
    else:
        arr[:] = 0.0
    return arr


def _valid_mask_from_raw(raw: pd.DataFrame) -> np.ndarray:
    finite = np.ones(len(raw), dtype=bool)
    for column in RAW_COLUMNS.values():
        values = pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float)
        finite &= np.isfinite(values)
    if "ValidFlag" in raw.columns:
        flag = pd.to_numeric(raw["ValidFlag"], errors="coerce").to_numpy(dtype=float)
        finite &= flag > 0
    return finite


def _hampel_deglitch(
    values: np.ndarray,
    *,
    window: int,
    n_sigmas: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(values, dtype=float).copy()
    out = arr.copy()
    n = arr.size
    width = max(3, int(window))
    if width % 2 == 0:
        width += 1
    half = width // 2
    replaced: list[int] = []
    sigma_scale = max(float(n_sigmas), 0.0)
    for idx in range(n):
        lo = max(0, idx - half)
        hi = min(n, idx + half + 1)
        segment = arr[lo:hi]
        finite = segment[np.isfinite(segment)]
        if finite.size < 3 or not np.isfinite(arr[idx]):
            continue
        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med)))
        diff = abs(float(arr[idx]) - med)
        robust_sigma = 1.4826 * mad
        if robust_sigma <= 1e-12:
            is_spike = diff > sigma_scale
        else:
            is_spike = diff > sigma_scale * robust_sigma
        if is_spike:
            out[idx] = med
            replaced.append(idx)
    return out, {
        "enabled": True,
        "window": int(width),
        "n_sigmas": float(n_sigmas),
        "replaced_count": int(len(replaced)),
        "replaced_ratio": float(len(replaced) / n) if n else 0.0,
    }


def _maybe_deglitch_channel(
    values: np.ndarray,
    *,
    label: str,
    cfg: V2SpO2Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not bool(cfg.deglitch_enabled):
        return np.asarray(values, dtype=float).copy(), {
            "enabled": False,
            "window": 0,
            "n_sigmas": float(cfg.deglitch_n_sigmas),
            "replaced_count": 0,
            "replaced_ratio": 0.0,
        }
    window = max(3, int(round(float(cfg.deglitch_window_seconds) * int(cfg.fs_origin))))
    cleaned, info = _hampel_deglitch(
        np.asarray(values, dtype=float),
        window=window,
        n_sigmas=float(cfg.deglitch_n_sigmas),
    )
    return cleaned, {"channel": label, **info}


def _lowpass_reference_signal(
    values: np.ndarray,
    *,
    fs: int,
    cutoff_hz: float,
    order: int,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(values, dtype=float).copy()
    info = {
        "enabled": bool(enabled),
        "cutoff_hz": float(cutoff_hz),
        "order": int(order),
    }
    if not bool(enabled):
        return arr, info
    if arr.size < 16:
        return arr, {**info, "applied": False, "reason": "too_short"}
    nyq = fs / 2.0
    cutoff = float(cutoff_hz) / nyq
    if not (0.0 < cutoff < 1.0):
        return arr, {**info, "applied": False, "reason": "invalid_cutoff"}
    try:
        b, a = butter(max(1, int(order)), cutoff, btype="lowpass")
        filtered = filtfilt(b, a, arr)
    except ValueError:
        return arr, {**info, "applied": False, "reason": "filter_error"}
    return filtered, {**info, "applied": True}


def _lowpass_ppg_signal(
    values: np.ndarray,
    *,
    fs: int,
    cutoff_hz: float,
    order: int,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _lowpass_reference_signal(
        values,
        fs=fs,
        cutoff_hz=cutoff_hz,
        order=order,
        enabled=enabled,
    )


def _load_spo2_raw_signals(cfg: V2SpO2Config) -> SpO2RawSignals:
    raw = pd.read_csv(cfg.data_path)
    missing = [name for name in RAW_COLUMNS.values() if name not in raw.columns]
    if missing:
        raise KeyError(f"Missing required v2 sensor columns: {', '.join(missing)}")
    if raw.empty:
        raise ValueError(f"Sensor CSV is empty: {cfg.data_path}")

    fs = int(cfg.fs_origin)
    if "Time(s)" in raw.columns:
        time_s = _clean_numeric_array(raw["Time(s)"])
    else:
        time_s = np.arange(len(raw), dtype=float) / float(fs)

    uc1_raw = _clean_numeric_array(raw[RAW_COLUMNS["uc1"]])
    uc2_raw = _clean_numeric_array(raw[RAW_COLUMNS["uc2"]])
    ut1_raw = _clean_numeric_array(raw[RAW_COLUMNS["ut1"]])
    ut2_raw = _clean_numeric_array(raw[RAW_COLUMNS["ut2"]])
    red_raw = _clean_numeric_array(raw[RAW_COLUMNS["ppg_red"]])
    ir_raw = _clean_numeric_array(raw[RAW_COLUMNS["ppg_ir"]])
    accx_raw = _clean_numeric_array(raw[RAW_COLUMNS["accx"]])
    accy_raw = _clean_numeric_array(raw[RAW_COLUMNS["accy"]])
    accz_raw = _clean_numeric_array(raw[RAW_COLUMNS["accz"]])

    artifact_rejection: dict[str, dict[str, Any]] = {}
    red, artifact_rejection[RAW_COLUMNS["ppg_red"]] = _maybe_deglitch_channel(
        red_raw,
        label=RAW_COLUMNS["ppg_red"],
        cfg=cfg,
    )
    ir, artifact_rejection[RAW_COLUMNS["ppg_ir"]] = _maybe_deglitch_channel(
        ir_raw,
        label=RAW_COLUMNS["ppg_ir"],
        cfg=cfg,
    )
    red, artifact_rejection["PPG_Red_lowpass"] = _lowpass_ppg_signal(
        red,
        fs=fs,
        cutoff_hz=float(cfg.ppg_lowpass_cutoff_hz),
        order=int(cfg.ppg_lowpass_order),
        enabled=bool(cfg.ppg_lowpass_enabled),
    )
    ir, artifact_rejection["PPG_IR_lowpass"] = _lowpass_ppg_signal(
        ir,
        fs=fs,
        cutoff_hz=float(cfg.ppg_lowpass_cutoff_hz),
        order=int(cfg.ppg_lowpass_order),
        enabled=bool(cfg.ppg_lowpass_enabled),
    )
    uc1, artifact_rejection[RAW_COLUMNS["uc1"]] = _maybe_deglitch_channel(
        uc1_raw,
        label=RAW_COLUMNS["uc1"],
        cfg=cfg,
    )
    uc2, artifact_rejection[RAW_COLUMNS["uc2"]] = _maybe_deglitch_channel(
        uc2_raw,
        label=RAW_COLUMNS["uc2"],
        cfg=cfg,
    )
    ut1, artifact_rejection[RAW_COLUMNS["ut1"]] = _maybe_deglitch_channel(
        ut1_raw,
        label=RAW_COLUMNS["ut1"],
        cfg=cfg,
    )
    ut2, artifact_rejection[RAW_COLUMNS["ut2"]] = _maybe_deglitch_channel(
        ut2_raw,
        label=RAW_COLUMNS["ut2"],
        cfg=cfg,
    )
    accx, artifact_rejection[RAW_COLUMNS["accx"]] = _maybe_deglitch_channel(
        accx_raw,
        label=RAW_COLUMNS["accx"],
        cfg=cfg,
    )
    accy, artifact_rejection[RAW_COLUMNS["accy"]] = _maybe_deglitch_channel(
        accy_raw,
        label=RAW_COLUMNS["accy"],
        cfg=cfg,
    )
    accz, artifact_rejection[RAW_COLUMNS["accz"]] = _maybe_deglitch_channel(
        accz_raw,
        label=RAW_COLUMNS["accz"],
        cfg=cfg,
    )
    references = {
        "hf1": ut1,
        "hf2": ut2,
        "cf1": safe_cf_ratio(uc1, ut1),
        "cf2": safe_cf_ratio(uc2, ut2),
        "accx": accx,
        "accy": accy,
        "accz": accz,
    }
    for key in ("cf1", "cf2"):
        references[key], artifact_rejection[key] = _maybe_deglitch_channel(
            references[key],
            label=key,
            cfg=cfg,
        )
    for key in ("hf1", "hf2"):
        references[key], artifact_rejection[f"{key}_lowpass"] = _lowpass_reference_signal(
            references[key],
            fs=fs,
            cutoff_hz=float(cfg.reference_lowpass_cutoff_hz),
            order=int(cfg.reference_lowpass_order),
            enabled=bool(cfg.reference_lowpass_enabled),
        )
    return SpO2RawSignals(
        fs=fs,
        time_s=time_s,
        red=red,
        ir=ir,
        references=references,
        valid_mask=_valid_mask_from_raw(raw),
        red_original=red_raw,
        ir_original=ir_raw,
        artifact_rejection=artifact_rejection,
    )


def _ordered_references(
    references: dict[str, np.ndarray],
    groups: tuple[str, ...],
) -> dict[str, np.ndarray]:
    ordered: dict[str, np.ndarray] = {}
    for group in normalise_reference_order(groups):
        for channel in channel_names_for_group(group):
            if channel in references:
                ordered[channel] = references[channel]
    return ordered


def _delay_to_order(delay_samples: int, cfg: V2SpO2Config) -> int:
    return int(np.clip(abs(int(delay_samples)), int(cfg.min_order), int(cfg.max_order)))


def _estimate_motion_threshold(
    scores: np.ndarray,
    cfg: V2SpO2Config,
) -> float:
    arr = np.asarray(scores, dtype=float)
    finite = arr[np.isfinite(arr)]
    fixed = float(cfg.rest_motion_score_threshold)
    if finite.size == 0 or str(cfg.motion_threshold_mode).lower() != "adaptive":
        return fixed
    q = float(np.clip(cfg.motion_threshold_quantile, 0.05, 0.80))
    cutoff = float(np.quantile(finite, q))
    baseline_values = finite[finite <= cutoff]
    if baseline_values.size == 0:
        baseline_values = finite
    baseline = float(np.median(baseline_values))
    mad = float(np.median(np.abs(baseline_values - baseline)))
    adaptive = baseline + max(
        float(cfg.motion_threshold_min_delta),
        float(cfg.motion_threshold_mad_scale) * 1.4826 * mad,
    )
    if not np.isfinite(adaptive):
        return fixed
    return float(min(fixed, adaptive))


def _detect_motion_segments(
    window_rows: list[dict[str, Any]],
    *,
    total_samples: int,
    fs: int,
    cfg: V2SpO2Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    if not window_rows:
        return [], [], float(cfg.rest_motion_score_threshold)
    scores = np.asarray(
        [float(row.get("motion_score", float("nan"))) for row in window_rows],
        dtype=float,
    )
    threshold = _estimate_motion_threshold(scores, cfg)
    motion_mask = np.isfinite(scores) & (scores > threshold)
    motion_segments: list[dict[str, Any]] = []
    idx = 0
    while idx < len(window_rows):
        if not motion_mask[idx]:
            idx += 1
            continue
        start_idx = idx
        while idx + 1 < len(window_rows) and motion_mask[idx + 1]:
            idx += 1
        end_idx = idx
        start = int(window_rows[start_idx]["start"])
        end = int(window_rows[end_idx]["end"])
        motion_segments.append(
            {
                "start": start,
                "end": end,
                "start_s": start / float(fs),
                "end_s": end / float(fs),
                "start_window_idx": int(window_rows[start_idx]["window_idx"]),
                "end_window_idx": int(window_rows[end_idx]["window_idx"]),
                "max_motion_score": float(np.nanmax(scores[start_idx : end_idx + 1])),
            }
        )
        idx += 1

    context = max(0, int(round(float(cfg.motion_context_seconds) * fs)))
    recovery_segments = [
        {
            **segment,
            "start": max(0, int(segment["start"]) - context),
            "end": min(int(total_samples), int(segment["end"]) + context),
            "start_s": max(0, int(segment["start"]) - context) / float(fs),
            "end_s": min(int(total_samples), int(segment["end"]) + context) / float(fs),
            "motion_start": int(segment["start"]),
            "motion_end": int(segment["end"]),
        }
        for segment in motion_segments
    ]
    return motion_segments, recovery_segments, float(threshold)


def _window_overlaps_segments(
    start: int,
    end: int,
    segments: list[dict[str, Any]],
) -> bool:
    return any(start < int(segment["end"]) and end > int(segment["start"]) for segment in segments)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    n = min(x.size, y.size)
    if n < 4:
        return 0.0
    x = x[:n] - float(np.mean(x[:n]))
    y = y[:n] - float(np.mean(y[:n]))
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _rank_references_for_window(
    *,
    target: np.ndarray,
    references: dict[str, np.ndarray],
    start: int,
    end: int,
    cfg: V2SpO2Config,
) -> list[dict[str, Any]]:
    target_seg = np.asarray(target[start:end], dtype=float)
    ranked: list[dict[str, Any]] = []
    max_lag = int(cfg.delay_search_samples)
    for channel, signal in references.items():
        ref_signal = np.asarray(signal, dtype=float)
        best_corr = 0.0
        best_delay = 0
        for delay in range(-max_lag, max_lag + 1):
            rel = np.arange(target_seg.size)
            ref_idx = start + rel + delay
            valid = (ref_idx >= 0) & (ref_idx < ref_signal.size)
            if int(np.count_nonzero(valid)) < 4:
                continue
            corr = _safe_corr(target_seg[valid], ref_signal[ref_idx[valid]])
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_delay = delay
        ranked.append(
            {
                "channel": channel,
                "corr": abs(float(best_corr)),
                "signed_corr": float(best_corr),
                "delay_samples": int(best_delay),
                "order": _delay_to_order(best_delay, cfg),
            }
        )
    return sorted(ranked, key=lambda row: row["corr"], reverse=True)


def _best_delay_corr(
    target: np.ndarray,
    reference: np.ndarray,
    *,
    max_lag: int,
) -> tuple[float, int]:
    target_arr = np.asarray(target, dtype=float)
    ref_arr = np.asarray(reference, dtype=float)
    n = min(target_arr.size, ref_arr.size)
    if n < 4:
        return 0.0, 0
    target_arr = target_arr[:n]
    ref_arr = ref_arr[:n]
    best_corr = 0.0
    best_delay = 0
    rel = np.arange(n)
    for delay in range(-int(max_lag), int(max_lag) + 1):
        ref_idx = rel + delay
        valid = (ref_idx >= 0) & (ref_idx < n)
        if int(np.count_nonzero(valid)) < 4:
            continue
        corr = _safe_corr(target_arr[valid], ref_arr[ref_idx[valid]])
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_delay = delay
    return float(best_corr), int(best_delay)


def _rank_joint_references_for_segment(
    *,
    red: np.ndarray,
    ir: np.ndarray,
    references: dict[str, np.ndarray],
    channels: tuple[str, ...],
    start: int,
    end: int,
    cfg: V2SpO2Config,
) -> list[dict[str, Any]]:
    red_seg = np.asarray(red[start:end], dtype=float)
    ir_seg = np.asarray(ir[start:end], dtype=float)
    ranked: list[dict[str, Any]] = []
    for channel in channels:
        if channel not in references:
            continue
        ref_seg = np.asarray(references[channel][start:end], dtype=float)
        red_corr, red_delay = _best_delay_corr(
            red_seg,
            ref_seg,
            max_lag=int(cfg.delay_search_samples),
        )
        ir_corr, ir_delay = _best_delay_corr(
            ir_seg,
            ref_seg,
            max_lag=int(cfg.delay_search_samples),
        )
        score = float(np.median([abs(red_corr), abs(ir_corr)]))
        if abs(ir_corr) >= abs(red_corr):
            signed_corr = ir_corr
            delay = ir_delay
        else:
            signed_corr = red_corr
            delay = red_delay
        ranked.append(
            {
                "channel": channel,
                "corr": score,
                "signed_corr": float(signed_corr),
                "delay_samples": int(delay),
                "order": _delay_to_order(delay, cfg),
            }
        )
    return sorted(ranked, key=lambda row: row["corr"], reverse=True)


def _cascade_forward_taps(group: str, cfg: V2SpO2Config) -> int:
    if str(group).upper() in {"HF", "CF"}:
        return 0
    if str(group).upper() == "ACC":
        return max(0, min(int(cfg.K_max), 1))
    return 0


def _normalised_reference(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    centered = arr - float(np.mean(arr)) if arr.size else arr
    sd = float(np.std(centered, ddof=1)) if centered.size > 1 else 0.0
    if sd <= 1e-12 or not np.isfinite(sd):
        return centered
    return centered / sd


def _reference_quiet_mask(
    references: dict[str, np.ndarray],
    *,
    start: int,
    end: int,
) -> np.ndarray:
    n = max(0, int(end) - int(start))
    if n == 0:
        return np.asarray([], dtype=bool)
    envelopes: list[np.ndarray] = []
    for values in references.values():
        segment = np.asarray(values[start:end], dtype=float)
        if segment.size != n:
            continue
        finite = segment[np.isfinite(segment)]
        if finite.size == 0:
            continue
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        if mad > 1e-12 and np.isfinite(mad):
            envelopes.append(np.abs(segment - median) / (1.4826 * mad))
            continue
        sd = float(np.std(segment, ddof=1)) if segment.size > 1 else 0.0
        if sd > 1e-12 and np.isfinite(sd):
            envelopes.append(np.abs(segment - float(np.mean(segment))) / sd)
        else:
            envelopes.append(np.zeros(n, dtype=float))
    if not envelopes:
        return np.ones(n, dtype=bool)
    envelope = np.nanmax(np.vstack(envelopes), axis=0)
    finite = envelope[np.isfinite(envelope)]
    if finite.size == 0:
        return np.ones(n, dtype=bool)
    cutoff = float(np.quantile(finite, 0.25))
    return np.isfinite(envelope) & (envelope <= cutoff)


def _baseline_from_anchor_masks(
    values: np.ndarray,
    *,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    fs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    local = np.asarray(values, dtype=float)
    n = local.size
    if n == 0:
        return local.copy(), {"mode": "empty", "left_anchor_count": 0, "right_anchor_count": 0}
    width = max(3, int(round(1.0 * fs)))
    left_idx = np.flatnonzero(np.asarray(left_mask, dtype=bool))
    right_idx = np.flatnonzero(np.asarray(right_mask, dtype=bool))
    left_anchor_count = int(left_idx.size)
    right_anchor_count = int(right_idx.size)

    if left_idx.size:
        use = left_idx[-width:]
        left_x = float(np.median(use))
        left_y = float(np.median(local[use]))
    else:
        left_x = float("nan")
        left_y = float("nan")
    if right_idx.size:
        use = right_idx[:width]
        right_x = float(np.median(use))
        right_y = float(np.median(local[use]))
    else:
        right_x = float("nan")
        right_y = float("nan")

    rel = np.arange(n, dtype=float)
    if np.isfinite(left_y) and np.isfinite(right_y) and right_x > left_x:
        baseline = np.interp(rel, [left_x, right_x], [left_y, right_y])
        mode = "linear_bridge"
    elif np.isfinite(left_y):
        baseline = np.full(n, left_y, dtype=float)
        mode = "left_constant"
    elif np.isfinite(right_y):
        baseline = np.full(n, right_y, dtype=float)
        mode = "right_constant"
    else:
        finite = local[np.isfinite(local)]
        fallback = float(np.median(finite)) if finite.size else 0.0
        baseline = np.full(n, fallback, dtype=float)
        mode = "segment_median"
    return baseline, {
        "mode": mode,
        "left_anchor_count": left_anchor_count,
        "right_anchor_count": right_anchor_count,
        "start_value": float(baseline[0]),
        "end_value": float(baseline[-1]),
    }


def _estimate_recovery_baseline(
    values: np.ndarray,
    references: dict[str, np.ndarray],
    segment: dict[str, Any],
    *,
    start: int,
    end: int,
    fs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    local = np.asarray(values[start:end], dtype=float)
    n = local.size
    if n == 0:
        return local.copy(), {"mode": "empty", "left_anchor_count": 0, "right_anchor_count": 0}

    rel = np.arange(n)
    motion_start = int(segment.get("motion_start", start))
    motion_end = int(segment.get("motion_end", end))
    left_mask = rel < max(0, motion_start - start)
    right_mask = rel >= min(n, max(0, motion_end - start))

    min_anchor = max(3, int(round(0.5 * fs)))
    if int(np.count_nonzero(left_mask)) < min_anchor and int(np.count_nonzero(right_mask)) < min_anchor:
        quiet = _reference_quiet_mask(references, start=start, end=end)
        left_mask = quiet & (rel < n // 2)
        right_mask = quiet & (rel >= n // 2)

    return _baseline_from_anchor_masks(
        local,
        left_mask=left_mask,
        right_mask=right_mask,
        fs=fs,
    )


def _run_adc_scale_lms_stage(
    *,
    desired: np.ndarray,
    reference: np.ndarray,
    order: int,
    K: int,
    mu: float,
) -> np.ndarray:
    d = np.asarray(desired, dtype=float).copy()
    u = _normalised_reference(reference)
    n = min(d.size, u.size)
    if n == 0:
        return d
    d = d[:n]
    u = u[:n]
    m = max(1, int(order))
    k = max(0, int(K))
    span = m + k
    if n - k < m:
        return d.copy()
    out = d.copy()
    weights = np.zeros(span, dtype=float)
    step = max(float(mu), 1e-12)
    eps = 1e-9
    for idx in range(m - 1, n - k):
        x_vec = u[idx - m + 1 : idx + k + 1][::-1]
        estimate = float(weights @ x_vec)
        err = float(d[idx] - estimate)
        out[idx] = err
        weights += (2.0 * step * err / (float(x_vec @ x_vec) + eps)) * x_vec
    out[~np.isfinite(out)] = d[~np.isfinite(out)]
    return out


def _run_adc_scale_as_lms_stage(
    *,
    desired: np.ndarray,
    reference: np.ndarray,
    order: int,
    K: int,
    cfg: V2SpO2Config,
) -> np.ndarray:
    d = np.asarray(desired, dtype=float).copy()
    u = _normalised_reference(reference)
    n = min(d.size, u.size)
    if n == 0:
        return d
    d = d[:n]
    u = u[:n]
    m = max(1, int(order))
    k = max(0, int(K))
    span = m + k
    if n - k < m:
        return d.copy()
    out = d.copy()
    weights = np.zeros(span, dtype=float)
    gamma = np.zeros(span, dtype=float)
    mu = float(np.clip(cfg.lms_mu_base, cfg.lms_mu_min, cfg.as_lms_mu_max))
    rho = max(0.0, float(cfg.as_lms_rho))
    for idx in range(m - 1, n - k):
        x_vec = u[idx - m + 1 : idx + k + 1][::-1]
        estimate = float(weights @ x_vec)
        err = float(d[idx] - estimate)
        out[idx] = err
        gamma_dot_u = float(gamma @ x_vec)
        step = 2.0 * mu
        weights += step * x_vec * err
        gamma += 2.0 * err * x_vec - step * x_vec * gamma_dot_u
        mu = float(np.clip(mu + rho * err * gamma_dot_u, cfg.lms_mu_min, cfg.as_lms_mu_max))
    out[~np.isfinite(out)] = d[~np.isfinite(out)]
    return out


def _restore_standardised_output(
    candidate: np.ndarray,
    desired: np.ndarray,
) -> np.ndarray:
    cand = np.asarray(candidate, dtype=float)
    d = np.asarray(desired, dtype=float)
    if cand.size == 0:
        return d.copy()
    finite = d[np.isfinite(d)]
    mean = float(np.mean(finite)) if finite.size else 0.0
    sd = float(np.std(finite, ddof=1)) if finite.size > 1 else 1.0
    if not np.isfinite(sd) or sd <= 1e-12:
        sd = 1.0
    out = cand * sd + mean
    out[~np.isfinite(out)] = mean
    return out


def _align_stage_output(previous: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    prev = np.asarray(previous, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if cand.size == prev.size:
        return cand.copy()
    out = prev.copy()
    n = min(out.size, cand.size)
    if n > 0:
        out[:n] = cand[:n]
    return out


def _run_spo2_adaptive_stage(
    *,
    desired: np.ndarray,
    reference: np.ndarray,
    group: str,
    order: int,
    K: int,
    corr: float,
    cfg: V2SpO2Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    strategy = str(cfg.adaptive_filter).strip().lower()
    mu = _adaptive_mu(corr, cfg)
    scale_restored = False
    if strategy == "lms":
        candidate = _run_adc_scale_lms_stage(
            desired=desired,
            reference=reference,
            order=order,
            K=K,
            mu=mu,
        )
    elif strategy == "as_lms":
        candidate = _run_adc_scale_as_lms_stage(
            desired=desired,
            reference=reference,
            order=order,
            K=K,
            cfg=cfg,
        )
    else:
        cascade_mu_base = min(float(cfg.lms_mu_base), 0.01)
        candidate = apply_adaptive_cascade(
            strategy=strategy,
            mu_base=cascade_mu_base,
            corr=float(corr),
            order=int(order),
            K=int(K),
            u=np.asarray(reference, dtype=float),
            d=np.asarray(desired, dtype=float),
            params=cfg,  # type: ignore[arg-type]
        )
        candidate = _restore_standardised_output(candidate, desired)
        mu = max(float(cfg.lms_mu_min), cascade_mu_base - abs(float(corr)) / 100.0)
        scale_restored = True
    aligned = _align_stage_output(desired, candidate)
    return aligned, {
        "filter_type": strategy,
        "sensor_type": str(group),
        "M": int(order),
        "K": int(K),
        "mu": float(mu),
        "scale_restored": bool(scale_restored),
        "input_len": int(np.asarray(desired).size),
        "output_len": int(np.asarray(candidate).size),
    }


def _recover_motion_segments_continuous(
    red: np.ndarray,
    ir: np.ndarray,
    references: dict[str, np.ndarray],
    recovery_segments: list[dict[str, Any]],
    *,
    fs: int,
    cfg: V2SpO2Config,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    red_out = np.asarray(red, dtype=float).copy()
    ir_out = np.asarray(ir, dtype=float).copy()
    stages: list[dict[str, Any]] = []
    if not bool(cfg.adaptive_enabled):
        return red_out, ir_out, stages
    for segment_idx, segment in enumerate(recovery_segments):
        start = max(0, int(segment["start"]))
        end = min(red_out.size, ir_out.size, int(segment["end"]))
        if end <= start + 3:
            continue
        red_baseline, red_baseline_info = _estimate_recovery_baseline(
            red_out,
            references,
            segment,
            start=start,
            end=end,
            fs=fs,
        )
        ir_baseline, ir_baseline_info = _estimate_recovery_baseline(
            ir_out,
            references,
            segment,
            start=start,
            end=end,
            fs=fs,
        )
        red_seg = red_out[start:end].copy() - red_baseline
        ir_seg = ir_out[start:end].copy() - ir_baseline
        for group in normalise_reference_order(cfg.reference_groups_order):
            channels = channel_names_for_group(group)
            ranked = _rank_joint_references_for_segment(
                red=red_out,
                ir=ir_out,
                references=references,
                channels=channels,
                start=start,
                end=end,
                cfg=cfg,
            )
            for row in ranked:
                corr = float(row["corr"])
                if corr <= 1e-12:
                    continue
                channel = str(row["channel"])
                ref_segment = np.asarray(references[channel][start:end], dtype=float)
                order = int(row["order"])
                K = _cascade_forward_taps(group, cfg)
                red_seg, red_diag = _run_spo2_adaptive_stage(
                    desired=red_seg,
                    reference=ref_segment,
                    group=group,
                    order=order,
                    K=K,
                    corr=corr,
                    cfg=cfg,
                )
                ir_seg, ir_diag = _run_spo2_adaptive_stage(
                    desired=ir_seg,
                    reference=ref_segment,
                    group=group,
                    order=order,
                    K=K,
                    corr=corr,
                    cfg=cfg,
                )
                stage = {
                    **red_diag,
                    "segment_idx": int(segment_idx),
                    "channel": channel,
                    "corr": corr,
                    "signed_corr": float(row["signed_corr"]),
                    "delay_samples": int(row["delay_samples"]),
                    "start": int(start),
                    "end": int(end),
                    "start_s": float(start / float(fs)),
                    "end_s": float(end / float(fs)),
                    "red_output_len": int(red_diag["output_len"]),
                    "ir_output_len": int(ir_diag["output_len"]),
                    "red_baseline_mode": str(red_baseline_info["mode"]),
                    "ir_baseline_mode": str(ir_baseline_info["mode"]),
                    "red_baseline_start": float(red_baseline_info["start_value"]),
                    "red_baseline_end": float(red_baseline_info["end_value"]),
                    "ir_baseline_start": float(ir_baseline_info["start_value"]),
                    "ir_baseline_end": float(ir_baseline_info["end_value"]),
                }
                stages.append(stage)
        red_out[start:end] = red_baseline + _align_stage_output(
            red_out[start:end] - red_baseline,
            red_seg,
        )
        ir_out[start:end] = ir_baseline + _align_stage_output(
            ir_out[start:end] - ir_baseline,
            ir_seg,
        )
    return red_out, ir_out, stages


def _recover_motion_segments_single_reference(
    red: np.ndarray,
    ir: np.ndarray,
    reference: np.ndarray,
    recovery_segments: list[dict[str, Any]],
    *,
    channel: str,
    fs: int,
    cfg: V2SpO2Config,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    red_out = np.asarray(red, dtype=float).copy()
    ir_out = np.asarray(ir, dtype=float).copy()
    ref = np.asarray(reference, dtype=float)
    stages: list[dict[str, Any]] = []
    if not bool(cfg.adaptive_enabled):
        return red_out, ir_out, stages

    references = {str(channel): ref}
    for segment_idx, segment in enumerate(recovery_segments):
        start = max(0, int(segment["start"]))
        end = min(red_out.size, ir_out.size, ref.size, int(segment["end"]))
        if end <= start + 3:
            continue
        red_baseline, red_baseline_info = _estimate_recovery_baseline(
            red_out,
            references,
            segment,
            start=start,
            end=end,
            fs=fs,
        )
        ir_baseline, ir_baseline_info = _estimate_recovery_baseline(
            ir_out,
            references,
            segment,
            start=start,
            end=end,
            fs=fs,
        )
        ranked = _rank_joint_references_for_segment(
            red=red_out,
            ir=ir_out,
            references=references,
            channels=(str(channel),),
            start=start,
            end=end,
            cfg=cfg,
        )
        if not ranked or float(ranked[0]["corr"]) <= 1e-12:
            continue
        row = ranked[0]
        corr = float(row["corr"])
        order = int(row["order"])
        ref_segment = ref[start:end]
        red_seg, red_diag = _run_spo2_adaptive_stage(
            desired=red_out[start:end] - red_baseline,
            reference=ref_segment,
            group="HF",
            order=order,
            K=0,
            corr=corr,
            cfg=cfg,
        )
        ir_seg, ir_diag = _run_spo2_adaptive_stage(
            desired=ir_out[start:end] - ir_baseline,
            reference=ref_segment,
            group="HF",
            order=order,
            K=0,
            corr=corr,
            cfg=cfg,
        )
        red_out[start:end] = red_baseline + _align_stage_output(
            red_out[start:end] - red_baseline,
            red_seg,
        )
        ir_out[start:end] = ir_baseline + _align_stage_output(
            ir_out[start:end] - ir_baseline,
            ir_seg,
        )
        stages.append(
            {
                **red_diag,
                "segment_idx": int(segment_idx),
                "channel": str(channel),
                "corr": corr,
                "signed_corr": float(row["signed_corr"]),
                "delay_samples": int(row["delay_samples"]),
                "start": int(start),
                "end": int(end),
                "start_s": float(start / float(fs)),
                "end_s": float(end / float(fs)),
                "red_output_len": int(red_diag["output_len"]),
                "ir_output_len": int(ir_diag["output_len"]),
                "red_baseline_mode": str(red_baseline_info["mode"]),
                "ir_baseline_mode": str(ir_baseline_info["mode"]),
                "red_baseline_start": float(red_baseline_info["start_value"]),
                "red_baseline_end": float(red_baseline_info["end_value"]),
                "ir_baseline_start": float(ir_baseline_info["start_value"]),
                "ir_baseline_end": float(ir_baseline_info["end_value"]),
            }
        )
    return red_out, ir_out, stages


def _adaptive_mu(corr: float, cfg: V2SpO2Config) -> float:
    del corr
    return max(float(cfg.lms_mu_min), float(cfg.lms_mu_base))


def _amplitude_preserving_lms(
    *,
    desired: np.ndarray,
    reference: np.ndarray,
    order: int,
    corr: float,
    cfg: V2SpO2Config,
) -> np.ndarray:
    d = np.asarray(desired, dtype=float)
    u = np.asarray(reference, dtype=float)
    n = min(d.size, u.size)
    if n == 0:
        return d.copy()
    d = d[:n]
    u = u[:n]
    m = int(np.clip(order, int(cfg.min_order), int(cfg.max_order)))
    if n < m + 1:
        return d.copy()

    baseline = float(np.median(d))
    d_center = d - baseline
    u_center = u - float(np.mean(u))
    u_std = float(np.std(u_center, ddof=1))
    if u_std <= 1e-12 or not np.isfinite(u_std):
        return d.copy()
    u_norm = u_center / u_std

    mu = _adaptive_mu(corr, cfg)
    weights = np.zeros(m, dtype=float)
    cleaned_center = d_center.copy()
    for idx in range(m - 1, n):
        x_vec = u_norm[idx - m + 1 : idx + 1][::-1]
        estimate = float(weights @ x_vec)
        err = float(d_center[idx] - estimate)
        cleaned_center[idx] = err
        denom = 1e-9 + float(x_vec @ x_vec)
        weights += (2.0 * mu * err / denom) * x_vec

    cleaned = cleaned_center + baseline
    if cleaned.size < desired.size:
        tail = np.asarray(desired[cleaned.size :], dtype=float)
        cleaned = np.concatenate([cleaned, tail])
    median_shift = float(np.median(desired) - np.median(cleaned))
    return cleaned + median_shift


def _clean_red_ir_adaptive(
    red: np.ndarray,
    ir: np.ndarray,
    references: dict[str, np.ndarray],
    *,
    start: int,
    end: int,
    cfg: V2SpO2Config,
) -> CleanedSpO2Signals:
    red_arr = np.asarray(red, dtype=float)
    ir_arr = np.asarray(ir, dtype=float)
    has_full_target = red_arr.size >= end and ir_arr.size >= end
    seg_start = start if has_full_target else 0
    seg_end = end if has_full_target else min(red_arr.size, ir_arr.size)
    red_clean = red_arr[seg_start:seg_end].copy()
    ir_clean = ir_arr[seg_start:seg_end].copy()
    if not cfg.adaptive_enabled:
        return CleanedSpO2Signals(red_clean=red_clean, ir_clean=ir_clean, stages=[])

    ordered_refs = _ordered_references(references, cfg.reference_groups_order)
    ranked = _rank_references_for_window(
        target=ir_arr if has_full_target else ir_clean,
        references=ordered_refs,
        start=seg_start,
        end=seg_end,
        cfg=cfg,
    )
    stages: list[dict[str, Any]] = []
    for row in ranked:
        corr = float(row["corr"])
        if corr <= 1e-12:
            continue
        channel = str(row["channel"])
        ref_source = np.asarray(references[channel], dtype=float)
        ref_segment = ref_source[seg_start:seg_end]
        order = int(row["order"])
        red_clean = _amplitude_preserving_lms(
            desired=red_clean,
            reference=ref_segment,
            order=order,
            corr=corr,
            cfg=cfg,
        )
        ir_clean = _amplitude_preserving_lms(
            desired=ir_clean,
            reference=ref_segment,
            order=order,
            corr=corr,
            cfg=cfg,
        )
        stages.append(
            {
                "channel": channel,
                "corr": corr,
                "signed_corr": float(row["signed_corr"]),
                "delay_samples": int(row["delay_samples"]),
                "order": order,
                "mu": _adaptive_mu(corr, cfg),
                "filter_type": "causal_lms",
            }
        )
    return CleanedSpO2Signals(
        red_clean=red_clean,
        ir_clean=ir_clean,
        stages=stages,
    )


def _safe_zero_phase_filter(
    values: np.ndarray,
    *,
    fs: int,
    kind: str,
    cfg: V2SpO2Config,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < 16:
        return arr - float(np.mean(arr)) if kind == "bandpass" else arr.copy()
    nyq = fs / 2.0
    order = int(cfg.filter_order)
    try:
        if kind == "bandpass":
            low = max(float(cfg.bp_low_hz), 1e-3) / nyq
            high = min(float(cfg.bp_high_hz), 0.45 * fs) / nyq
            if not (0.0 < low < high < 1.0):
                return arr - float(np.mean(arr))
            b, a = butter(order, [low, high], btype="bandpass")
            return filtfilt(b, a, arr)
        if kind == "lowpass":
            cutoff = min(float(cfg.lp_cutoff_hz), 0.45 * fs) / nyq
            if not (0.0 < cutoff < 1.0):
                return arr.copy()
            b, a = butter(order, cutoff, btype="lowpass")
            return filtfilt(b, a, arr)
    except ValueError:
        return arr - float(np.mean(arr)) if kind == "bandpass" else arr.copy()
    raise ValueError(f"Unsupported filter kind: {kind!r}")


def _moving_average(values: np.ndarray, samples: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    width = max(1, int(samples))
    if width <= 1 or arr.size < width:
        return arr.copy()
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(arr, kernel, mode="same")


def _local_extreme_index(
    values: np.ndarray,
    start: int,
    end: int,
    mode: str,
) -> int | None:
    arr = np.asarray(values, dtype=float)
    lo = max(0, int(start))
    hi = min(arr.size - 1, int(end))
    if hi < lo:
        return None
    segment = arr[lo : hi + 1]
    if segment.size == 0:
        return None
    if mode == "min":
        return int(lo + np.argmin(segment))
    if mode == "max":
        return int(lo + np.argmax(segment))
    raise ValueError(f"Unsupported extreme mode: {mode!r}")


def _calc_ac_dc_by_valley_line(
    adc: np.ndarray,
    v1_idx: int,
    p_idx: int,
    v2_idx: int,
) -> tuple[float, float]:
    x = np.asarray(adc, dtype=float)
    v1 = int(v1_idx)
    p = int(p_idx)
    v2 = int(v2_idx)
    if not (0 <= v1 < p < v2 < x.size):
        return float("nan"), float("nan")
    y1 = float(x[v1])
    y2 = float(x[p])
    y3 = float(x[v2])
    dc = y1 + (y3 - y1) * ((p - v1) / float(v2 - v1))
    ac = abs(y2 - dc)
    return float(ac), float(dc)


def _compute_spo2_window(
    *,
    red: np.ndarray,
    ir: np.ndarray,
    fs: int,
    cfg: V2SpO2Config,
    scheme: str,
) -> dict[str, Any]:
    red_arr = np.asarray(red, dtype=float)
    ir_arr = np.asarray(ir, dtype=float)
    n = min(red_arr.size, ir_arr.size)
    red_arr = red_arr[:n]
    ir_arr = ir_arr[:n]
    if n < max(8, int(round(float(cfg.min_beat_interval_seconds) * fs)) * 2):
        return {
            "spo2": float("nan"),
            "r_median": float("nan"),
            "valid_beat_count": 0,
            "beat_rows": [],
        }

    ir_det = _safe_zero_phase_filter(ir_arr, fs=fs, kind="bandpass", cfg=cfg)
    red_det = _safe_zero_phase_filter(red_arr, fs=fs, kind="bandpass", cfg=cfg)
    ir_adc = _safe_zero_phase_filter(ir_arr, fs=fs, kind="lowpass", cfg=cfg)
    red_adc = _safe_zero_phase_filter(red_arr, fs=fs, kind="lowpass", cfg=cfg)
    smooth_len = max(1, int(round(float(cfg.smooth_seconds) * fs)))
    smooth_ir = _moving_average(ir_det, smooth_len)
    smooth_red = _moving_average(red_det, smooth_len)

    min_distance = max(1, int(round(float(cfg.min_beat_interval_seconds) * fs)))
    valleys, _ = find_peaks(-smooth_ir, distance=min_distance)
    valley_half = max(1, int(round(float(cfg.valley_search_seconds) * fs)))
    peak_half = max(1, int(round(float(cfg.peak_search_seconds) * fs)))
    r_values: list[float] = []
    beat_rows: list[dict[str, Any]] = []

    for beat_idx in range(max(0, valleys.size - 1)):
        v1_ir = int(valleys[beat_idx])
        v2_ir = int(valleys[beat_idx + 1])
        if v2_ir <= v1_ir + 2:
            continue
        p_ir = _local_extreme_index(smooth_ir, v1_ir, v2_ir, "max")
        if p_ir is None or p_ir <= v1_ir or p_ir >= v2_ir:
            continue

        ac_ir, dc_ir = _calc_ac_dc_by_valley_line(ir_adc, v1_ir, p_ir, v2_ir)
        v1_red = _local_extreme_index(
            smooth_red,
            max(0, v1_ir - valley_half),
            min(p_ir - 1, v1_ir + valley_half),
            "min",
        )
        v2_red = _local_extreme_index(
            smooth_red,
            max(p_ir + 1, v2_ir - valley_half),
            min(n - 1, v2_ir + valley_half),
            "min",
        )
        if v1_red is None or v2_red is None or v2_red <= v1_red + 2:
            continue
        peak_start = max(v1_red + 1, p_ir - peak_half)
        peak_end = min(v2_red - 1, p_ir + peak_half)
        if peak_end <= peak_start:
            peak_start = v1_red + 1
            peak_end = v2_red - 1
        p_red = _local_extreme_index(smooth_red, peak_start, peak_end, "max")
        if p_red is None or p_red <= v1_red or p_red >= v2_red:
            continue

        ac_red, dc_red = _calc_ac_dc_by_valley_line(red_adc, v1_red, p_red, v2_red)
        if not (ac_ir > 1e-4 and ac_red > 1e-4 and dc_ir > 1e-4 and dc_red > 1e-4):
            continue
        r_beat = (ac_red / dc_red) / (ac_ir / dc_ir)
        if not (np.isfinite(r_beat) and float(cfg.r_min) < r_beat < float(cfg.r_max)):
            continue
        r_value = float(r_beat)
        r_values.append(r_value)
        beat_rows.append(
            {
                "scheme": str(scheme),
                "beat_idx": int(beat_idx),
                "v1_ir": int(v1_ir),
                "p_ir": int(p_ir),
                "v2_ir": int(v2_ir),
                "v1_red": int(v1_red),
                "p_red": int(p_red),
                "v2_red": int(v2_red),
                "ac_ir": float(ac_ir),
                "dc_ir": float(dc_ir),
                "ac_red": float(ac_red),
                "dc_red": float(dc_red),
                "r": r_value,
            }
        )

    if not r_values:
        return {
            "spo2": float("nan"),
            "r_median": float("nan"),
            "valid_beat_count": 0,
            "beat_rows": beat_rows,
        }
    r_median = float(np.median(np.asarray(r_values, dtype=float)))
    spo2 = float(spo2_from_r(r_median, cfg.coefficients))
    return {
        "spo2": spo2,
        "r_median": r_median,
        "valid_beat_count": len(r_values),
        "beat_rows": beat_rows,
    }
