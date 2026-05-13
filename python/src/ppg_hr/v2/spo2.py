"""v2 SpO2 computation from Red/IR PPG with amplitude-preserving LMS cleanup."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    raise NotImplementedError("solve_spo2_v2 is implemented in later plan tasks")


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
    red_clean = np.asarray(red, dtype=float).copy()
    ir_clean = np.asarray(ir, dtype=float).copy()
    if not cfg.adaptive_enabled:
        return CleanedSpO2Signals(red_clean=red_clean, ir_clean=ir_clean, stages=[])

    ordered_refs = _ordered_references(references, cfg.reference_groups_order)
    ranked = _rank_references_for_window(
        target=ir_clean,
        references=ordered_refs,
        start=start,
        end=end,
        cfg=cfg,
    )
    stages: list[dict[str, Any]] = []
    for row in ranked:
        corr = float(row["corr"])
        if corr <= 1e-12:
            continue
        channel = str(row["channel"])
        ref_segment = np.asarray(references[channel][start:end], dtype=float)
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
