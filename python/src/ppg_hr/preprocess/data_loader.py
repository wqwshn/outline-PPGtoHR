"""Sensor + ground-truth CSV ingestion (Python port of ``process_and_merge_sensor_data_new.m``)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .utils import (
    fillmissing_linear,
    fillmissing_nearest,
    filloutliers_movmedian_linear,
)

__all__ = ["ProcessedDataset", "load_dataset", "SENSOR_COLUMNS"]

SAMPLE_RATE_HZ: int = 100

# Mapping of internal short name -> raw CSV column header
SENSOR_COLUMNS: dict[str, str] = {
    "Uc1": "Uc1(mV)",
    "Uc2": "Uc2(mV)",
    "Ut1": "Ut1(mV)",
    "Ut2": "Ut2(mV)",
    "PPG_Green": "PPG_Green",
    "PPG_Red": "PPG_Red",
    "PPG_IR": "PPG_IR",
    "AccX": "AccX(g)",
    "AccY": "AccY(g)",
    "AccZ": "AccZ(g)",
    "GyroX": "GyroX(dps)",
    "GyroY": "GyroY(dps)",
    "GyroZ": "GyroZ(dps)",
}


@dataclass
class ProcessedDataset:
    """Result of :func:`load_dataset`.

    Attributes
    ----------
    data:
        Pandas DataFrame with one row per 10 ms sample. Columns:
        ``Time_s`` plus, for each entry of :data:`SENSOR_COLUMNS`, the
        cleaned raw value and the band-pass-filtered ``<name>_Filt`` value.
    ref_data:
        Two-column ``(N, 2)`` numpy array. Column 0 is the reference time
        in seconds, column 1 is the reference heart-rate in BPM.
    """

    data: pd.DataFrame
    ref_data: np.ndarray
    valid_mask: np.ndarray | None = None
    source_kind: str = "legacy_csv"
    quality_metadata: dict[str, object] = field(default_factory=dict)


def _bandpass_coeffs(fs: int = SAMPLE_RATE_HZ) -> tuple[np.ndarray, np.ndarray]:
    nyquist = fs / 2.0
    return butter(4, [0.5 / nyquist, 5.0 / nyquist], btype="bandpass")


def _clean_signal(values: np.ndarray, name: str, fs: int) -> np.ndarray:
    cleaned = fillmissing_nearest(values)
    if "PPG" in name:
        neg = cleaned < 0
        if neg.any():
            cleaned = cleaned.astype(float).copy()
            cleaned[neg] = np.nan
            cleaned = fillmissing_linear(cleaned)
            cleaned = fillmissing_nearest(cleaned)
    return filloutliers_movmedian_linear(cleaned, window=fs)


def _parse_reference_csv(gt_csv: Path) -> np.ndarray:
    gt = pd.read_csv(gt_csv, skiprows=3, header=None)
    if gt.shape[1] < 3:
        raise ValueError(f"Reference CSV {gt_csv} has fewer than 3 columns")
    raw_time = gt.iloc[:, 1].astype(str).str.strip()
    raw_bpm = gt.iloc[:, 2]

    def _to_seconds(t: str) -> float:
        try:
            return pd.to_timedelta(t).total_seconds()
        except (ValueError, TypeError):
            try:
                return float(t)
            except ValueError:
                return float("nan")

    time_s = np.array([_to_seconds(t) for t in raw_time], dtype=float)
    bpm = pd.to_numeric(raw_bpm, errors="coerce").to_numpy(dtype=float)
    valid = ~(np.isnan(time_s) | np.isnan(bpm))
    return np.column_stack([time_s[valid], bpm[valid]])


def _detect_source_kind(raw: pd.DataFrame) -> str:
    timeline_cols = {"ValidFlag", "InterpFlag", "GapLen"}
    if timeline_cols.issubset(raw.columns):
        return "timeline_csv"
    if {"Time(s)", "MissingBefore"}.issubset(raw.columns):
        return "compact_raw_csv"
    return "legacy_csv"


def _extract_valid_mask(raw: pd.DataFrame, selected: list[str]) -> np.ndarray:
    finite = np.ones(len(raw), dtype=bool)
    for short in selected:
        original = SENSOR_COLUMNS[short]
        values = pd.to_numeric(raw[original], errors="coerce").to_numpy(dtype=float)
        finite &= np.isfinite(values)
    if "ValidFlag" not in raw.columns:
        return finite
    flag = pd.to_numeric(raw["ValidFlag"], errors="coerce").to_numpy(dtype=float)
    return finite & (flag > 0)


def _time_axis(raw: pd.DataFrame, fs: int) -> np.ndarray:
    if "Time(s)" not in raw.columns:
        return np.arange(len(raw), dtype=float) / float(fs)
    values = pd.to_numeric(raw["Time(s)"], errors="coerce").to_numpy(dtype=float)
    if values.size != len(raw) or not np.isfinite(values).all():
        return np.arange(len(raw), dtype=float) / float(fs)
    return values


def load_dataset(
    sensor_csv: str | Path,
    gt_csv: str | Path,
    *,
    fs: int = SAMPLE_RATE_HZ,
    columns: Iterable[str] | None = None,
) -> ProcessedDataset:
    """Load and preprocess a sensor + reference CSV pair.

    Parameters
    ----------
    sensor_csv:
        Path to the raw sensor CSV (14-column layout, see :data:`SENSOR_COLUMNS`).
    gt_csv:
        Path to the Polar-style reference CSV (header on rows 1-3, data from row 4).
    fs:
        Target sampling rate (defaults to 100 Hz, matching the original MATLAB pipeline).
    columns:
        Optional iterable of channel short-names to process. ``None`` means all
        13 channels in :data:`SENSOR_COLUMNS`.
    """
    sensor_path = Path(sensor_csv)
    gt_path = Path(gt_csv)
    if not sensor_path.is_file():
        raise FileNotFoundError(f"Sensor CSV not found: {sensor_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"Reference CSV not found: {gt_path}")

    raw = pd.read_csv(sensor_path)
    n = len(raw)
    if n == 0:
        raise ValueError(f"Sensor CSV is empty: {sensor_path}")

    selected = list(columns) if columns is not None else list(SENSOR_COLUMNS)
    for short in selected:
        if short not in SENSOR_COLUMNS:
            raise KeyError(f"Unknown sensor column '{short}'")
        original = SENSOR_COLUMNS[short]
        if original not in raw.columns:
            raise KeyError(f"Column '{original}' missing in {sensor_path}")

    source_kind = _detect_source_kind(raw)
    valid_mask = _extract_valid_mask(raw, selected)
    df = pd.DataFrame()
    df["Time_s"] = _time_axis(raw, fs)

    for short in selected:
        original = SENSOR_COLUMNS[short]
        df[short] = raw[original].astype(float).to_numpy()

    b, a = _bandpass_coeffs(fs)
    for short in selected:
        cleaned = _clean_signal(df[short].to_numpy(dtype=float), short, fs)
        df[short] = cleaned
        df[f"{short}_Filt"] = filtfilt(b, a, cleaned)

    ref_data = _parse_reference_csv(gt_path)
    quality_metadata = {
        "source_kind": source_kind,
        "valid_count": int(valid_mask.sum()),
        "missing_count": int((~valid_mask).sum()),
    }
    if "GapLen" in raw.columns:
        gap_len = pd.to_numeric(raw["GapLen"], errors="coerce").fillna(0)
        quality_metadata["max_gap_samples"] = int(gap_len.max())
    return ProcessedDataset(
        data=df,
        ref_data=ref_data,
        valid_mask=valid_mask,
        source_kind=source_kind,
        quality_metadata=quality_metadata,
    )
