"""v2 SpO2 computation from Red/IR PPG with amplitude-preserving LMS cleanup."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from .preprocess import RAW_COLUMNS, safe_cf_ratio
from .reference_groups import channel_names_for_group, normalise_reference_order


@dataclass(frozen=True)
class V2SpO2Coefficients:
    a: float = 1.5958422
    b: float = -34.6596622
    c: float = 112.6898759


@dataclass(frozen=True)
class V2SpO2Config:
    data_path: Path
    output_dir: Path | None = None
    reference_groups_order: tuple[str, ...] = ("HF", "CF", "ACC")
    fs_origin: int = 100
    window_seconds: float = 4.0
    window_step_seconds: float = 1.0
    delay_search_samples: int = 20
    max_order: int = 20
    min_order: int = 1
    lms_mu_base: float = 0.01
    lms_mu_min: float = 1e-6
    adaptive_enabled: bool = True
    bp_low_hz: float = 0.5
    bp_high_hz: float = 5.0
    lp_cutoff_hz: float = 8.0
    filter_order: int = 3
    min_beat_interval_seconds: float = 0.40
    valley_search_seconds: float = 0.12
    peak_search_seconds: float = 0.16
    smooth_seconds: float = 0.06
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

    red_accum = np.zeros_like(signals.red, dtype=float)
    ir_accum = np.zeros_like(signals.ir, dtype=float)
    overlap = np.zeros_like(signals.red, dtype=float)
    spo2_table: list[dict[str, Any]] = []
    beat_table: list[dict[str, Any]] = []
    adaptive_stage_rows: list[list[dict[str, Any]]] = []
    acc_mag = np.sqrt(
        signals.references["accx"] ** 2
        + signals.references["accy"] ** 2
        + signals.references["accz"] ** 2
    )
    last_raw_spo2 = float("nan")
    last_adaptive_spo2 = float("nan")

    for window_idx, start in enumerate(range(0, signals.red.size - window_len + 1, step_len)):
        end = start + window_len
        cleaned = _clean_red_ir_adaptive(
            signals.red,
            signals.ir,
            signals.references,
            start=start,
            end=end,
            cfg=cfg,
        )
        red_accum[start:end] += cleaned.red_clean
        ir_accum[start:end] += cleaned.ir_clean
        overlap[start:end] += 1.0
        adaptive_stage_rows.append(cleaned.stages)

        raw_out = _compute_spo2_window(
            red=signals.red[start:end],
            ir=signals.ir[start:end],
            fs=fs,
            cfg=cfg,
            scheme="raw",
        )
        adaptive_out = _compute_spo2_window(
            red=cleaned.red_clean,
            ir=cleaned.ir_clean,
            fs=fs,
            cfg=cfg,
            scheme="adaptive",
        )

        raw_spo2 = float(raw_out["spo2"])
        adaptive_spo2 = float(adaptive_out["spo2"])
        raw_carried = False
        adaptive_carried = False
        if np.isfinite(raw_spo2):
            last_raw_spo2 = raw_spo2
        elif np.isfinite(last_raw_spo2):
            raw_spo2 = last_raw_spo2
            raw_carried = True
        if np.isfinite(adaptive_spo2):
            last_adaptive_spo2 = adaptive_spo2
        elif np.isfinite(last_adaptive_spo2):
            adaptive_spo2 = last_adaptive_spo2
            adaptive_carried = True

        missing_ratio = 1.0 - float(np.mean(signals.valid_mask[start:end]))
        center_s = float(signals.time_s[start] + cfg.window_seconds / 2.0)
        row = {
            "window_idx": int(window_idx),
            "start_s": float(signals.time_s[start]),
            "end_s": float(signals.time_s[end - 1]),
            "center_s": center_s,
            "motion_score": float(np.std(acc_mag[start:end], ddof=1)),
            "spo2": adaptive_spo2,
            "raw_spo2": raw_spo2,
            "adaptive_spo2": adaptive_spo2,
            "raw_r_median": float(raw_out["r_median"]),
            "adaptive_r_median": float(adaptive_out["r_median"]),
            "raw_valid_beat_count": int(raw_out["valid_beat_count"]),
            "adaptive_valid_beat_count": int(adaptive_out["valid_beat_count"]),
            "raw_carried_forward": raw_carried,
            "adaptive_carried_forward": adaptive_carried,
            "missing_ratio": missing_ratio,
            "reliable": bool(
                missing_ratio <= 0.20
                and int(adaptive_out["valid_beat_count"]) > 0
            ),
        }
        spo2_table.append(row)
        for beat in raw_out["beat_rows"] + adaptive_out["beat_rows"]:
            beat_table.append(
                {
                    "window_idx": int(window_idx),
                    "window_center_s": center_s,
                    **beat,
                }
            )

    red_clean = np.divide(
        red_accum,
        overlap,
        out=signals.red.copy(),
        where=overlap > 0,
    )
    ir_clean = np.divide(
        ir_accum,
        overlap,
        out=signals.ir.copy(),
        where=overlap > 0,
    )
    metadata = {
        "schema_version": "v2_spo2",
        "data_path": str(cfg.data_path),
        "fs": fs,
        "window_seconds": float(cfg.window_seconds),
        "window_step_seconds": float(cfg.window_step_seconds),
        "delay_search_samples": int(cfg.delay_search_samples),
        "max_order": int(cfg.max_order),
        "reference_groups_order": list(cfg.reference_groups_order),
        "adaptive_enabled": bool(cfg.adaptive_enabled),
        "adaptive_stage_rows": adaptive_stage_rows,
    }
    waveforms = {
        "time_s": signals.time_s,
        "red_raw": signals.red,
        "ir_raw": signals.ir,
        "red_clean": red_clean,
        "ir_clean": ir_clean,
        "acc_mag": acc_mag,
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
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = str(output_prefix).strip() or "spo2"
    json_path = out / f"{prefix}-spo2.json"
    csv_path = out / f"{prefix}-spo2.csv"
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
    return {"json": json_path, "csv": csv_path}


def load_spo2_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "v2_spo2":
        raise ValueError(f"{path} is not a v2 SpO2 report")
    return payload


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer | np.floating):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonify(v) for v in obj]
    return obj


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

    uc1 = _clean_numeric_array(raw[RAW_COLUMNS["uc1"]])
    uc2 = _clean_numeric_array(raw[RAW_COLUMNS["uc2"]])
    ut1 = _clean_numeric_array(raw[RAW_COLUMNS["ut1"]])
    ut2 = _clean_numeric_array(raw[RAW_COLUMNS["ut2"]])
    references = {
        "hf1": ut1,
        "hf2": ut2,
        "cf1": safe_cf_ratio(uc1, ut1),
        "cf2": safe_cf_ratio(uc2, ut2),
        "accx": _clean_numeric_array(raw[RAW_COLUMNS["accx"]]),
        "accy": _clean_numeric_array(raw[RAW_COLUMNS["accy"]]),
        "accz": _clean_numeric_array(raw[RAW_COLUMNS["accz"]]),
    }
    return SpO2RawSignals(
        fs=fs,
        time_s=time_s,
        red=_clean_numeric_array(raw[RAW_COLUMNS["ppg_red"]]),
        ir=_clean_numeric_array(raw[RAW_COLUMNS["ppg_ir"]]),
        references=references,
        valid_mask=_valid_mask_from_raw(raw),
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


def _adaptive_mu(corr: float, cfg: V2SpO2Config) -> float:
    corr_abs = abs(float(corr))
    corr_for_formula = corr_abs / 100.0 if corr_abs > 1.0 else corr_abs
    return max(float(cfg.lms_mu_min), float(cfg.lms_mu_base) - corr_for_formula / 100.0)


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
