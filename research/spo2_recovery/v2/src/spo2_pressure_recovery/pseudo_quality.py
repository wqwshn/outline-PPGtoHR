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


def _index(time_s: np.ndarray, seconds: float, fs_hz: float) -> int:
    if time_s.size == 0:
        return 0
    return int(np.clip(round(float(seconds) * float(fs_hz)), 0, time_s.size - 1))


def _local_ac_range(record: PressureRecord, event: PressureEvent, channel: str) -> float:
    values = record.red_adc if channel == "red" else record.ir_adc
    fs = float(record.fs_hz)
    guard = int(round(0.50 * fs))
    pre_start = _index(record.time_s, event.pre_rest_start_s, fs)
    core_start = _index(record.time_s, event.loading_start_s, fs)
    core_end = _index(record.time_s, event.post_rest_start_s, fs)
    post_end = _index(record.time_s, event.post_rest_end_s, fs) + 1
    pre = values[pre_start : max(pre_start, core_start - guard)]
    post = values[min(values.size, core_end + guard + 1) : min(values.size, post_end)]
    stable = np.r_[pre, post]
    if stable.size < 4:
        stable = values[max(0, core_start - guard) : min(values.size, core_end + guard + 1)]
    return _range(stable)


def _external_boundary_jump(
    record: PressureRecord,
    event: PressureEvent,
    truth: EventPseudoTruth,
    values: np.ndarray,
    channel: str,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or truth.time_s.size == 0:
        return 0.0, 0.0
    observed = record.red_adc if channel == "red" else record.ir_adc
    fs = float(record.fs_hz)
    start = _index(record.time_s, float(truth.time_s[0]), fs)
    end = _index(record.time_s, float(truth.time_s[min(truth.time_s.size - 1, arr.size - 1)]), fs)
    jumps: list[float] = []
    if start > 0:
        jumps.append(abs(float(arr[0]) - float(observed[start - 1])))
    if end + 1 < observed.size:
        jumps.append(abs(float(arr[min(arr.size - 1, end - start)]) - float(observed[end + 1])))
    jump_adc = float(max(jumps)) if jumps else 0.0
    return jump_adc, float(jump_adc / _local_ac_range(record, event, channel))


def pseudo_truth_quality(
    record: PressureRecord,
    event: PressureEvent,
    truth: EventPseudoTruth,
) -> dict[str, float | bool | int]:
    fs = float(record.fs_hz)
    truth_start = _index(record.time_s, float(truth.time_s[0]) if truth.time_s.size else event.loading_start_s, fs)
    core_start = _index(record.time_s, event.loading_start_s, fs)
    core_end = _index(record.time_s, event.post_rest_start_s, fs)
    offset = max(0, core_start - truth_start)
    core_n = min(truth.time_s.size - offset, record.time_s.size - core_start, core_end - core_start + 1)
    pressure = record.ut_common_mv[core_start : core_start + core_n]
    red_corr = abs(_safe_corr(truth.red_dc[offset : offset + core_n], pressure))
    ir_corr = abs(_safe_corr(truth.ir_dc[offset : offset + core_n], pressure))
    n = min(truth.time_s.size, record.time_s.size - truth_start)
    red_jump = _boundary_jump_fraction(truth.red[:n])
    ir_jump = _boundary_jump_fraction(truth.ir[:n])
    red_external_adc, red_external = _external_boundary_jump(
        record,
        event,
        truth,
        truth.red[:n],
        "red",
    )
    ir_external_adc, ir_external = _external_boundary_jump(
        record,
        event,
        truth,
        truth.ir[:n],
        "ir",
    )
    usable = bool(
        truth.quality.get("usable", 0.0) > 0.0
        and red_jump <= 0.35
        and ir_jump <= 0.35
        and red_external <= 0.30
        and ir_external <= 0.30
        and red_corr <= 0.50
        and ir_corr <= 0.50
    )
    return {
        "event_id": int(event.event_id),
        "red_boundary_jump_fraction": red_jump,
        "ir_boundary_jump_fraction": ir_jump,
        "red_external_boundary_jump_adc": red_external_adc,
        "ir_external_boundary_jump_adc": ir_external_adc,
        "red_external_boundary_jump_ac_fraction": red_external,
        "ir_external_boundary_jump_ac_fraction": ir_external,
        "red_pressure_corr": red_corr,
        "ir_pressure_corr": ir_corr,
        "usable": usable,
    }
