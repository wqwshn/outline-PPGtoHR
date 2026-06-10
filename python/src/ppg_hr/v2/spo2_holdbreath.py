"""Red/IR-only SpO2 evaluation for hold-breath experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time as Time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .spo2 import V2SpO2Config, _compute_spo2_window, _load_spo2_raw_signals


@dataclass(frozen=True)
class PulseOximeterModel:
    smooth_seconds: float = 1.0
    lag_seconds: float = 0.0
    bias: float = 0.0


@dataclass(frozen=True)
class HoldBreathSpO2Config:
    data_path: Path
    truth_path: Path | None = None
    output_dir: Path | None = None
    trim_seconds: float = 30.0
    fs_origin: int = 100
    window_seconds: float = 4.0
    window_step_seconds: float = 1.0
    fit_device_model: bool = True
    device_model: PulseOximeterModel | None = None
    smooth_grid_seconds: tuple[float, ...] = (1.0, 3.0, 5.0, 7.0, 9.0)
    lag_grid_seconds: tuple[float, ...] = tuple(float(v) for v in range(-12, 13))
    fit_bias: bool = True


@dataclass(frozen=True)
class HoldBreathTruth:
    time_s: np.ndarray
    spo2: np.ndarray
    path: Path


@dataclass
class HoldBreathSpO2Result:
    spo2_table: list[dict[str, Any]]
    aligned_table: list[dict[str, Any]]
    metrics: dict[str, float | int | bool]
    metadata: dict[str, Any]


def find_holdbreath_truth_path(data_path: str | Path) -> Path:
    data = Path(data_path)
    for suffix in (".csv", ".xlsx", ".xls"):
        candidate = data.with_name(f"{data.stem}_ref{suffix}")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Hold-breath truth file not found for {data}")


def load_holdbreath_truth(path: str | Path) -> HoldBreathTruth:
    truth_path = Path(path)
    raw = truth_path.read_bytes()
    if raw[:4] == b"PK\x03\x04":
        frame = pd.read_excel(truth_path, header=None)
    else:
        frame = pd.read_csv(truth_path, header=None)
    frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if frame.shape[1] < 1:
        raise ValueError(f"No SpO2 truth columns found in {truth_path}")

    spo2_col = _select_spo2_column(frame)
    spo2 = pd.to_numeric(frame.iloc[:, spo2_col], errors="coerce").to_numpy(dtype=float)
    if frame.shape[1] >= 2:
        time_col = 0 if spo2_col != 0 else 1
        time_s = _parse_truth_time_column(frame.iloc[:, time_col])
    else:
        time_s = np.arange(spo2.size, dtype=float)

    mask = np.isfinite(time_s) & np.isfinite(spo2)
    time_s = time_s[mask]
    spo2 = spo2[mask]
    if time_s.size == 0:
        raise ValueError(f"No finite SpO2 truth samples found in {truth_path}")
    order = np.argsort(time_s)
    return HoldBreathTruth(time_s=time_s[order], spo2=spo2[order], path=truth_path)


def compute_holdbreath_metrics(
    time_s: np.ndarray,
    calculated: np.ndarray,
    truth: np.ndarray,
    *,
    analysis_start_s: float | None = None,
    analysis_end_s: float | None = None,
) -> dict[str, float | int]:
    t = np.asarray(time_s, dtype=float)
    calc = np.asarray(calculated, dtype=float)
    ref = np.asarray(truth, dtype=float)
    mask = np.isfinite(t) & np.isfinite(calc) & np.isfinite(ref)
    if analysis_start_s is not None:
        mask &= t >= float(analysis_start_s)
    if analysis_end_s is not None:
        mask &= t <= float(analysis_end_s)
    if not np.any(mask):
        return {
            "sample_count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "mean_bias": float("nan"),
            "max_abs_error": float("nan"),
            "nadir_spo2_error": float("nan"),
            "nadir_time_error_s": float("nan"),
            "pearson_r": float("nan"),
        }

    mt = t[mask]
    mc = calc[mask]
    mr = ref[mask]
    err = mc - mr
    if mc.size >= 2 and np.std(mc) > 0.0 and np.std(mr) > 0.0:
        pearson = float(np.corrcoef(mc, mr)[0, 1])
    else:
        pearson = float("nan")
    calc_nadir_idx = int(np.argmin(mc))
    truth_nadir_idx = int(np.argmin(mr))
    return {
        "sample_count": int(err.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean_bias": float(np.mean(err)),
        "max_abs_error": float(np.max(np.abs(err))),
        "nadir_spo2_error": float(mc[calc_nadir_idx] - mr[truth_nadir_idx]),
        "nadir_time_error_s": float(mt[calc_nadir_idx] - mt[truth_nadir_idx]),
        "pearson_r": pearson,
    }


def solve_spo2_holdbreath(config: HoldBreathSpO2Config) -> HoldBreathSpO2Result:
    cfg = HoldBreathSpO2Config(**{**config.__dict__, "data_path": Path(config.data_path)})
    truth_path = Path(cfg.truth_path) if cfg.truth_path is not None else find_holdbreath_truth_path(cfg.data_path)
    truth = load_holdbreath_truth(truth_path)
    raw_cfg = V2SpO2Config(
        data_path=cfg.data_path,
        fs_origin=int(cfg.fs_origin),
        window_seconds=float(cfg.window_seconds),
        window_step_seconds=float(cfg.window_step_seconds),
        adaptive_enabled=False,
        spo2_smooth_seconds=1.0,
    )
    signals = _load_spo2_raw_signals(raw_cfg)
    fs = int(signals.fs)
    window_len = int(round(float(cfg.window_seconds) * fs))
    step_len = int(round(float(cfg.window_step_seconds) * fs))
    spo2_table: list[dict[str, Any]] = []
    for window_idx, start in enumerate(range(0, signals.red.size - window_len + 1, step_len)):
        end = start + window_len
        out = _compute_spo2_window(
            red=signals.red[start:end],
            ir=signals.ir[start:end],
            fs=fs,
            cfg=raw_cfg,
            scheme="raw",
        )
        center_s = float(signals.time_s[start] + cfg.window_seconds / 2.0)
        spo2_table.append(
            {
                "window_idx": int(window_idx),
                "start_s": float(signals.time_s[start]),
                "end_s": float(signals.time_s[end - 1]),
                "center_s": center_s,
                "spo2_raw": float(out["spo2"]),
                "r_median": float(out["r_median"]),
                "valid_beat_count": int(out["valid_beat_count"]),
                "reliable": bool(int(out["valid_beat_count"]) > 0),
            }
        )

    table_time = np.asarray([row["center_s"] for row in spo2_table], dtype=float)
    raw_spo2 = np.asarray([row["spo2_raw"] for row in spo2_table], dtype=float)
    model = cfg.device_model or PulseOximeterModel()
    calculated = _apply_device_model(table_time, raw_spo2, model)
    truth_at_time = np.interp(
        table_time,
        truth.time_s,
        truth.spo2,
        left=float(truth.spo2[0]),
        right=float(truth.spo2[-1]),
    )
    analysis_start_s = float(cfg.trim_seconds)
    analysis_end_s = float(signals.time_s[-1] - cfg.trim_seconds)
    metrics = compute_holdbreath_metrics(
        table_time,
        calculated,
        truth_at_time,
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
    )
    raw_metrics = compute_holdbreath_metrics(
        table_time,
        raw_spo2,
        truth_at_time,
        analysis_start_s=analysis_start_s,
        analysis_end_s=analysis_end_s,
    )
    metrics["raw_mae"] = float(raw_metrics["mae"])
    metrics["modeled_mae"] = float(metrics["mae"])
    metrics["device_model_fit"] = False

    aligned_table = [
        {
            "time_s": float(t),
            "spo2_calculated": float(calc),
            "spo2_truth": float(ref),
            "error": float(calc - ref),
            "spo2_raw": float(raw),
            "device_model_lag_s": float(model.lag_seconds),
            "device_model_smooth_s": float(model.smooth_seconds),
        }
        for t, calc, ref, raw in zip(table_time, calculated, truth_at_time, raw_spo2, strict=True)
    ]
    metadata = {
        "schema_version": "v2_spo2_holdbreath",
        "data_path": str(cfg.data_path),
        "truth_path": str(truth.path),
        "fs": fs,
        "analysis_start_s": analysis_start_s,
        "analysis_end_s": analysis_end_s,
        "fit_device_model": bool(cfg.fit_device_model),
        "device_model": {
            "smooth_seconds": float(model.smooth_seconds),
            "lag_seconds": float(model.lag_seconds),
            "bias": float(model.bias),
        },
    }
    return HoldBreathSpO2Result(
        spo2_table=spo2_table,
        aligned_table=aligned_table,
        metrics=metrics,
        metadata=metadata,
    )


def _select_spo2_column(frame: pd.DataFrame) -> int:
    best_idx = 0
    best_count = -1
    for idx in range(frame.shape[1]):
        values = pd.to_numeric(frame.iloc[:, idx], errors="coerce").to_numpy(dtype=float)
        count = int(np.count_nonzero(np.isfinite(values) & (values >= 50.0) & (values <= 105.0)))
        if count >= best_count:
            best_idx = idx
            best_count = count
    return best_idx


def _parse_truth_time_column(values: pd.Series) -> np.ndarray:
    parsed = [_parse_time_value(value) for value in values]
    arr = np.asarray(parsed, dtype=float)
    if np.isfinite(arr).sum() >= max(1, arr.size // 2):
        first = float(arr[np.isfinite(arr)][0])
        out = arr - first
        if np.nanmin(out) < 0.0:
            out = np.arange(arr.size, dtype=float)
        return out
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(numeric).sum() >= max(1, numeric.size // 2):
        first = float(numeric[np.isfinite(numeric)][0])
        return numeric - first
    return np.arange(values.size, dtype=float)


def _parse_time_value(value: Any) -> float:
    if isinstance(value, Time):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6)
    if isinstance(value, pd.Timestamp):
        return float(value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6)
    if isinstance(value, str):
        stripped = value.strip()
        if ":" in stripped:
            stamp = pd.to_datetime(stripped, errors="coerce")
            if pd.notna(stamp):
                return _parse_time_value(stamp)
    return float("nan")


def _apply_device_model(
    time_s: np.ndarray,
    spo2: np.ndarray,
    model: PulseOximeterModel,
) -> np.ndarray:
    smoothed = _centered_finite_moving_average(
        np.asarray(spo2, dtype=float),
        max(1, int(round(float(model.smooth_seconds)))),
    )
    shifted = np.interp(
        np.asarray(time_s, dtype=float) - float(model.lag_seconds),
        np.asarray(time_s, dtype=float),
        smoothed,
        left=float(smoothed[0]) if smoothed.size else float("nan"),
        right=float(smoothed[-1]) if smoothed.size else float("nan"),
    )
    return shifted + float(model.bias)


def _centered_finite_moving_average(values: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or width <= 1:
        return arr.copy()
    if width % 2 == 0:
        width += 1
    half = width // 2
    out = np.full(arr.shape, float("nan"), dtype=float)
    for idx in range(arr.size):
        lo = max(0, idx - half)
        hi = min(arr.size, idx + half + 1)
        finite = arr[lo:hi][np.isfinite(arr[lo:hi])]
        if finite.size:
            out[idx] = float(np.mean(finite))
    return out
