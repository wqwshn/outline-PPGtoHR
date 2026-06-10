from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .types import PreprocessConfig, PressureRecord


REQUIRED_COLUMNS = ("Time(s)", "Ut1(mV)", "Ut2(mV)", "PPG_Red", "PPG_IR")


def interpolate_nonfinite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if not np.any(finite):
        return np.zeros_like(arr)
    indices = np.arange(arr.size, dtype=float)
    arr[~finite] = np.interp(indices[~finite], indices[finite], arr[finite])
    return arr


def hampel_deglitch(
    values: np.ndarray,
    *,
    fs_hz: float,
    window_s: float,
    n_sigmas: float,
) -> tuple[np.ndarray, int]:
    arr = interpolate_nonfinite(values)
    if arr.size < 3:
        return arr, 0
    half_window = max(1, int(round(float(fs_hz) * float(window_s) / 2.0)))
    series = pd.Series(arr)
    width = 2 * half_window + 1
    median = series.rolling(width, center=True, min_periods=1).median().to_numpy()
    deviation = np.abs(arr - median)
    mad = (
        pd.Series(deviation)
        .rolling(width, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    robust_sigma = 1.4826 * mad
    fallback = float(np.median(robust_sigma[robust_sigma > 0.0])) if np.any(robust_sigma > 0.0) else 0.0
    threshold = float(n_sigmas) * np.where(robust_sigma > 0.0, robust_sigma, fallback)
    mask = (threshold > 0.0) & (deviation > threshold)
    out = arr.copy()
    out[mask] = median[mask]
    return out, int(np.count_nonzero(mask))


def zero_phase_lowpass(
    values: np.ndarray,
    *,
    fs_hz: float,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    arr = interpolate_nonfinite(values)
    nyquist = 0.5 * float(fs_hz)
    if arr.size < 4 or not 0.0 < float(cutoff_hz) < nyquist:
        return arr
    sos = butter(
        max(1, int(order)),
        float(cutoff_hz) / nyquist,
        btype="lowpass",
        output="sos",
    )
    padlen = 3 * (2 * len(sos) + 1)
    if arr.size <= padlen:
        return arr
    return np.asarray(sosfiltfilt(sos, arr), dtype=float)


def _read_channel(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def load_record(path: str | Path, config: PreprocessConfig) -> PressureRecord:
    data_path = Path(path)
    frame = pd.read_csv(data_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    time_s = interpolate_nonfinite(_read_channel(frame, "Time(s)"))
    red, red_count = hampel_deglitch(
        _read_channel(frame, "PPG_Red"),
        fs_hz=config.fs_hz,
        window_s=config.hampel_window_s,
        n_sigmas=config.hampel_n_sigmas,
    )
    ir, ir_count = hampel_deglitch(
        _read_channel(frame, "PPG_IR"),
        fs_hz=config.fs_hz,
        window_s=config.hampel_window_s,
        n_sigmas=config.hampel_n_sigmas,
    )
    ut1, ut1_count = hampel_deglitch(
        _read_channel(frame, "Ut1(mV)"),
        fs_hz=config.fs_hz,
        window_s=config.hampel_window_s,
        n_sigmas=config.hampel_n_sigmas,
    )
    ut2, ut2_count = hampel_deglitch(
        _read_channel(frame, "Ut2(mV)"),
        fs_hz=config.fs_hz,
        window_s=config.hampel_window_s,
        n_sigmas=config.hampel_n_sigmas,
    )
    red = zero_phase_lowpass(
        red,
        fs_hz=config.fs_hz,
        cutoff_hz=config.ppg_lowpass_hz,
        order=config.filter_order,
    )
    ir = zero_phase_lowpass(
        ir,
        fs_hz=config.fs_hz,
        cutoff_hz=config.ppg_lowpass_hz,
        order=config.filter_order,
    )
    ut1 = zero_phase_lowpass(
        ut1,
        fs_hz=config.fs_hz,
        cutoff_hz=config.ut_lowpass_hz,
        order=config.filter_order,
    )
    ut2 = zero_phase_lowpass(
        ut2,
        fs_hz=config.fs_hz,
        cutoff_hz=config.ut_lowpass_hz,
        order=config.filter_order,
    )
    return PressureRecord(
        time_s=time_s,
        red_adc=red,
        ir_adc=ir,
        ut1_mv=ut1,
        ut2_mv=ut2,
        ut_common_mv=0.5 * (ut1 + ut2),
        ut_difference_mv=0.5 * (ut1 - ut2),
        fs_hz=float(config.fs_hz),
        metadata={
            "data_path": str(data_path),
            "sample_count": int(len(frame)),
            "deglitch_counts": {
                "red": red_count,
                "ir": ir_count,
                "ut1": ut1_count,
                "ut2": ut2_count,
            },
        },
    )
