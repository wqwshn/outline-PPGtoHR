"""v2 protocol loading and preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from ppg_hr.preprocess.utils import fillmissing_linear, fillmissing_nearest

from .types import V2Dataset

RAW_COLUMNS: dict[str, str] = {
    "uc1": "Uc1(mV)",
    "uc2": "Uc2(mV)",
    "ut1": "Ut1(mV)",
    "ut2": "Ut2(mV)",
    "ppg_green": "PPG_Green",
    "ppg_red": "PPG_Red",
    "ppg_ir": "PPG_IR",
    "accx": "AccX(g)",
    "accy": "AccY(g)",
    "accz": "AccZ(g)",
    "gyrox": "GyroX(dps)",
    "gyroy": "GyroY(dps)",
    "gyroz": "GyroZ(dps)",
}

V2_CHANNELS: tuple[str, ...] = (
    "ppg_green",
    "ppg_red",
    "ppg_ir",
    "hf1",
    "hf2",
    "cf1",
    "cf2",
    "accx",
    "accy",
    "accz",
    "gyrox",
    "gyroy",
    "gyroz",
)


def load_v2_dataset(
    sensor_csv: str | Path,
    ref_csv: str | Path,
    fs_origin: int = 100,
) -> V2Dataset:
    sensor_path = Path(sensor_csv)
    ref_path = Path(ref_csv)
    if not sensor_path.is_file():
        raise FileNotFoundError(f"Sensor CSV not found: {sensor_path}")
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference CSV not found: {ref_path}")

    raw = pd.read_csv(sensor_path)
    if raw.empty:
        raise ValueError(f"Sensor CSV is empty: {sensor_path}")
    _validate_columns(raw)

    fs = int(fs_origin)
    clean = _clean_frame(raw, fs)
    ref = _parse_reference_csv(ref_path)
    valid_mask = _extract_valid_mask(raw)
    return V2Dataset(
        sample_stem=sensor_path.stem,
        fs=fs,
        data=clean,
        ref_data=ref,
        valid_mask=valid_mask,
    )


def safe_cf_ratio(uc: np.ndarray, ut: np.ndarray) -> np.ndarray:
    numerator = _clean_numeric(uc)
    denominator = _clean_numeric(ut) - numerator
    denominator[np.abs(denominator) < 1e-9] = np.nan
    out = numerator / denominator
    out[~np.isfinite(out)] = np.nan
    out = fillmissing_linear(out)
    out = fillmissing_nearest(out)
    out[~np.isfinite(out)] = 0.0
    return out.astype(float, copy=False)


def filtered_channels(frame: pd.DataFrame, fs: int) -> pd.DataFrame:
    out = frame.copy()
    for name in V2_CHANNELS:
        if name.startswith("ppg"):
            low, high = 0.5, 5.0
        elif name.startswith(("acc", "gyro")):
            low, high = 0.5, 10.0
        else:
            low, high = 0.1, 5.0
        out[name] = _safe_bandpass(out[name].to_numpy(dtype=float), int(fs), low, high)
    return out


def _clean_frame(raw: pd.DataFrame, fs: int) -> pd.DataFrame:
    n = len(raw)
    uc1 = _clean_numeric(raw[RAW_COLUMNS["uc1"]])
    uc2 = _clean_numeric(raw[RAW_COLUMNS["uc2"]])
    ut1 = _clean_numeric(raw[RAW_COLUMNS["ut1"]])
    ut2 = _clean_numeric(raw[RAW_COLUMNS["ut2"]])
    data = {
        "time_s": np.arange(n, dtype=float) / float(fs),
        "ppg_green": _clean_numeric(raw[RAW_COLUMNS["ppg_green"]]),
        "ppg_red": _clean_numeric(raw[RAW_COLUMNS["ppg_red"]]),
        "ppg_ir": _clean_numeric(raw[RAW_COLUMNS["ppg_ir"]]),
        "hf1": ut1,
        "hf2": ut2,
        "cf1": safe_cf_ratio(uc1, ut1),
        "cf2": safe_cf_ratio(uc2, ut2),
        "accx": _clean_numeric(raw[RAW_COLUMNS["accx"]]),
        "accy": _clean_numeric(raw[RAW_COLUMNS["accy"]]),
        "accz": _clean_numeric(raw[RAW_COLUMNS["accz"]]),
        "gyrox": _clean_numeric(raw[RAW_COLUMNS["gyrox"]]),
        "gyroy": _clean_numeric(raw[RAW_COLUMNS["gyroy"]]),
        "gyroz": _clean_numeric(raw[RAW_COLUMNS["gyroz"]]),
    }
    return pd.DataFrame(data, columns=("time_s", *V2_CHANNELS))


def _validate_columns(raw: pd.DataFrame) -> None:
    missing = [col for col in RAW_COLUMNS.values() if col not in raw.columns]
    if missing:
        raise KeyError(f"Missing required v2 sensor columns: {', '.join(missing)}")


def _clean_numeric(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
        dtype=float,
        copy=True,
    )
    arr[~np.isfinite(arr)] = np.nan
    arr = fillmissing_linear(arr)
    arr = fillmissing_nearest(arr)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _extract_valid_mask(raw: pd.DataFrame) -> np.ndarray:
    finite = np.ones(len(raw), dtype=bool)
    for column in RAW_COLUMNS.values():
        values = pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float)
        finite &= np.isfinite(values)
    if "ValidFlag" not in raw.columns:
        return finite
    flag = pd.to_numeric(raw["ValidFlag"], errors="coerce").to_numpy(dtype=float)
    return finite & (flag > 0)


def _safe_bandpass(
    values: np.ndarray,
    fs: int,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    baseline = arr - float(np.nanmean(arr))
    if arr.size < 16:
        return baseline
    nyq = fs / 2.0
    low = max(float(low_hz), 1e-3)
    high = min(float(high_hz), 0.45 * fs)
    if not (0 < low < high < nyq):
        return baseline
    b, a = butter(4, [low / nyq, high / nyq], btype="bandpass")
    try:
        return filtfilt(b, a, arr)
    except ValueError:
        return baseline


def _parse_reference_csv(ref_csv: Path) -> np.ndarray:
    if ref_csv.stem.endswith("_HR_ref"):
        return _parse_hr_ref_csv_v2(ref_csv)
    ref = pd.read_csv(ref_csv, skiprows=3, header=None)
    if ref.shape[1] < 3:
        raise ValueError(f"Reference CSV {ref_csv} has fewer than 3 columns")
    times = ref.iloc[:, 1].astype(str).str.strip()
    bpm = pd.to_numeric(ref.iloc[:, 2], errors="coerce").to_numpy(dtype=float)

    def to_seconds(value: str) -> float:
        try:
            return pd.to_timedelta(value).total_seconds()
        except (TypeError, ValueError):
            try:
                return float(value)
            except ValueError:
                return float("nan")

    seconds = np.asarray([to_seconds(x) for x in times], dtype=float)
    mask = np.isfinite(seconds) & np.isfinite(bpm)
    return np.column_stack([seconds[mask], bpm[mask]])


def _parse_hr_ref_csv_v2(ref_csv: Path) -> np.ndarray:
    """解析 _HR_ref.csv 格式：header 行 + elapsed_seconds / hr_bpm 列。"""
    ref = pd.read_csv(ref_csv)
    if "elapsed_seconds" not in ref.columns or "hr_bpm" not in ref.columns:
        raise ValueError(
            f"HR ref CSV {ref_csv} 缺少 elapsed_seconds / hr_bpm 列，"
            f"实际列: {list(ref.columns)}"
        )
    time_s = pd.to_numeric(ref["elapsed_seconds"], errors="coerce").to_numpy(dtype=float)
    bpm = pd.to_numeric(ref["hr_bpm"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(time_s) & np.isfinite(bpm)
    return np.column_stack([time_s[mask], bpm[mask]])
