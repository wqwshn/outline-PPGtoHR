from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .data import interpolate_nonfinite, zero_phase_lowpass
from .types import EventConfig, PressureEvent


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    binary = np.asarray(mask, dtype=bool)
    changes = np.diff(np.r_[False, binary, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _merge_short_gaps(mask: np.ndarray, maximum_gap: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if maximum_gap <= 0:
        return out
    for start, end in _true_runs(~out):
        if start > 0 and end < out.size and end - start <= maximum_gap:
            out[start:end] = True
    return out


def _remove_short_runs(mask: np.ndarray, minimum_length: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for start, end in _true_runs(out):
        if end - start < minimum_length:
            out[start:end] = False
    return out


def _event_delta(values: np.ndarray, *, start: int, peak: int, fs_hz: float) -> float:
    baseline_width = max(10, int(round(fs_hz)))
    pre_start = max(0, start - baseline_width)
    baseline_slice = values[pre_start:start]
    baseline = float(np.median(baseline_slice)) if baseline_slice.size else float(values[start])
    peak_radius = max(1, int(round(0.10 * fs_hz)))
    peak_slice = values[max(start, peak - peak_radius) : min(values.size, peak + peak_radius + 1)]
    peak_value = float(np.median(peak_slice)) if peak_slice.size else float(values[peak])
    return peak_value - baseline


def detect_pressure_events(
    time_s: np.ndarray,
    ut1_mv: np.ndarray,
    ut2_mv: np.ndarray,
    config: EventConfig,
) -> list[PressureEvent]:
    time = interpolate_nonfinite(time_s)
    ut1 = interpolate_nonfinite(ut1_mv)
    ut2 = interpolate_nonfinite(ut2_mv)
    n = min(time.size, ut1.size, ut2.size)
    if n < 3:
        return []
    time = time[:n]
    ut1 = ut1[:n]
    ut2 = ut2[:n]
    fs = float(config.fs_hz)
    smooth1 = zero_phase_lowpass(
        ut1,
        fs_hz=fs,
        cutoff_hz=config.response_cutoff_hz,
        order=3,
    )
    trend1 = zero_phase_lowpass(
        smooth1,
        fs_hz=fs,
        cutoff_hz=config.trend_cutoff_hz,
        order=3,
    )
    response1 = smooth1 - trend1

    baseline_count = min(n, max(int(round(5.0 * fs)), int(round(0.20 * n))))
    baseline_values = response1[:baseline_count]
    baseline = float(np.median(baseline_values))
    mad = float(np.median(np.abs(baseline_values - baseline)))
    noise_threshold = float(config.onset_threshold_mad) * 1.4826 * mad
    threshold = baseline + max(float(config.minimum_response_mv), noise_threshold)

    mask = response1 > threshold
    mask = _merge_short_gaps(mask, int(round(config.merge_gap_s * fs)))
    mask = _remove_short_runs(mask, int(round(config.minimum_duration_s * fs)))
    runs = _true_runs(mask)

    smooth2 = zero_phase_lowpass(
        ut2,
        fs_hz=fs,
        cutoff_hz=config.response_cutoff_hz,
        order=3,
    )
    events: list[PressureEvent] = []
    context = int(round(config.context_s * fs))
    guard = max(1, int(round(0.20 * fs)))
    for event_id, (start, end) in enumerate(runs, start=1):
        local = response1[start:end]
        if local.size == 0:
            continue
        peak = start + int(np.argmax(local))
        ut1_delta = _event_delta(smooth1, start=start, peak=peak, fs_hz=fs)
        ut2_delta = _event_delta(smooth2, start=start, peak=peak, fs_hz=fs)
        common_delta = 0.5 * (ut1_delta + ut2_delta)
        difference_peak = 0.5 * abs(ut1_delta - ut2_delta)
        asymmetry = difference_peak / max(abs(common_delta), 1e-12)
        same_direction = ut1_delta * ut2_delta > 0.0
        bilateral = bool(same_direction and asymmetry <= config.off_center_ratio)
        previous_end = runs[event_id - 2][1] if event_id > 1 else 0
        next_start = runs[event_id][0] if event_id < len(runs) else n - 1
        pre_start = max(0, start - context, previous_end + guard)
        post_end = min(n - 1, end + context, next_start - guard)
        events.append(
            PressureEvent(
                event_id=event_id,
                pre_rest_start_s=float(time[pre_start]),
                loading_start_s=float(time[start]),
                peak_s=float(time[peak]),
                release_start_s=float(time[peak]),
                post_rest_start_s=float(time[min(n - 1, end)]),
                post_rest_end_s=float(time[post_end]),
                ut1_delta_mv=float(ut1_delta),
                ut2_delta_mv=float(ut2_delta),
                common_delta_mv=float(common_delta),
                difference_peak_mv=float(difference_peak),
                bilateral_consistent=bilateral,
                off_center=not bilateral,
            )
        )
    return events


def events_to_frame(events: list[PressureEvent]) -> pd.DataFrame:
    return pd.DataFrame([asdict(event) for event in events])


def event_sample_bounds(
    event: PressureEvent,
    *,
    fs_hz: float,
    length: int,
) -> dict[str, slice]:
    def index(seconds: float) -> int:
        return int(np.clip(round(float(seconds) * float(fs_hz)), 0, length))

    loading = index(event.loading_start_s)
    peak = index(event.peak_s)
    release = index(event.release_start_s)
    post = index(event.post_rest_start_s)
    return {
        "pre_rest": slice(index(event.pre_rest_start_s), loading),
        "loading": slice(loading, peak),
        "hold": slice(peak, release),
        "release": slice(release, post),
        "post_rest": slice(post, index(event.post_rest_end_s)),
    }
