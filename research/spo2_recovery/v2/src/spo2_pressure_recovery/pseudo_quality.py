from __future__ import annotations

import numpy as np

from .pseudo_truth import EventPseudoTruth
from .types import PressureEvent, PressureRecord


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    n = min(left.size, right.size)
    left = left[:n]
    right = right[:n]
    mask = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(mask) < 3:
        return 0.0
    left = left[mask] - float(np.mean(left[mask]))
    right = right[mask] - float(np.mean(right[mask]))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denom) if denom > 0.0 else 0.0


def _range(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return float(max(np.percentile(finite, 95) - np.percentile(finite, 5), 1e-9))


def _boundary_jump_fraction(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 4:
        return 0.0
    jumps = [abs(arr[1] - arr[0]), abs(arr[-1] - arr[-2])]
    return float(max(jumps) / _range(arr))


def pseudo_truth_quality(
    record: PressureRecord,
    event: PressureEvent,
    truth: EventPseudoTruth,
) -> dict[str, float | bool | int]:
    fs = float(record.fs_hz)
    start = int(np.clip(round(event.loading_start_s * fs), 0, record.time_s.size - 1))
    n = min(truth.time_s.size, record.time_s.size - start)
    pressure = record.ut_common_mv[start : start + n]
    red_corr = abs(_safe_corr(truth.red_dc[:n], pressure))
    ir_corr = abs(_safe_corr(truth.ir_dc[:n], pressure))
    red_jump = _boundary_jump_fraction(truth.red[:n])
    ir_jump = _boundary_jump_fraction(truth.ir[:n])
    usable = bool(
        truth.quality.get("usable", 0.0) > 0.0
        and red_jump <= 0.35
        and ir_jump <= 0.35
        and red_corr <= 0.50
        and ir_corr <= 0.50
    )
    return {
        "event_id": int(event.event_id),
        "red_boundary_jump_fraction": red_jump,
        "ir_boundary_jump_fraction": ir_jump,
        "red_pressure_corr": red_corr,
        "ir_pressure_corr": ir_corr,
        "usable": usable,
    }
