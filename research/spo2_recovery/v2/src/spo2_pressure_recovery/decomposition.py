from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

from .data import interpolate_nonfinite, zero_phase_lowpass
from .types import DecompositionConfig


@dataclass
class PPGDecomposition:
    dc: np.ndarray
    ac: np.ndarray
    envelope: np.ndarray


def _zero_phase_bandpass(
    values: np.ndarray,
    *,
    fs_hz: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    arr = interpolate_nonfinite(values)
    nyquist = 0.5 * float(fs_hz)
    if arr.size < 16 or not 0.0 < low_hz < high_hz < nyquist:
        return arr - np.mean(arr)
    sos = butter(
        max(1, int(order)),
        [float(low_hz) / nyquist, float(high_hz) / nyquist],
        btype="bandpass",
        output="sos",
    )
    return np.asarray(sosfiltfilt(sos, arr), dtype=float)


def decompose_ppg(
    values: np.ndarray,
    config: DecompositionConfig,
) -> PPGDecomposition:
    arr = interpolate_nonfinite(values)
    dc = zero_phase_lowpass(
        arr,
        fs_hz=config.fs_hz,
        cutoff_hz=config.dc_lowpass_hz,
        order=config.filter_order,
    )
    ac = _zero_phase_bandpass(
        arr,
        fs_hz=config.fs_hz,
        low_hz=config.pulse_low_hz,
        high_hz=config.pulse_high_hz,
        order=config.filter_order,
    )
    raw_envelope = np.abs(hilbert(ac)) if ac.size else ac.copy()
    envelope = zero_phase_lowpass(
        raw_envelope,
        fs_hz=config.fs_hz,
        cutoff_hz=config.envelope_lowpass_hz,
        order=config.filter_order,
    )
    envelope = np.maximum(envelope, np.finfo(float).eps)
    return PPGDecomposition(dc=dc, ac=ac, envelope=envelope)


def detect_beats(
    ac: np.ndarray,
    *,
    fs_hz: float,
    min_bpm: float = 30.0,
    max_bpm: float = 180.0,
) -> np.ndarray:
    values = interpolate_nonfinite(ac)
    if values.size < 3:
        return np.asarray([], dtype=int)
    minimum_distance = max(1, int(round(60.0 * fs_hz / max_bpm)))
    maximum_period = max(minimum_distance + 1, int(round(60.0 * fs_hz / min_bpm)))
    prominence = max(np.std(values) * 0.15, np.finfo(float).eps)
    valleys, _ = find_peaks(
        -values,
        distance=minimum_distance,
        prominence=prominence,
    )
    if valleys.size < 2:
        return valleys.astype(int)
    intervals = np.diff(valleys)
    valid_pair = intervals <= maximum_period
    keep = np.r_[valid_pair, False] | np.r_[False, valid_pair]
    return valleys[keep].astype(int)


def extract_beats(
    values: np.ndarray,
    valleys: np.ndarray,
    *,
    phase_samples: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    signal = interpolate_nonfinite(values)
    boundaries = np.asarray(valleys, dtype=int)
    beats: list[np.ndarray] = []
    periods: list[int] = []
    target_phase = np.linspace(0.0, 1.0, int(phase_samples), endpoint=False)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end - start < 3:
            continue
        beat = signal[start:end]
        source_phase = np.linspace(0.0, 1.0, beat.size, endpoint=False)
        beats.append(np.interp(target_phase, source_phase, beat))
        periods.append(int(end - start))
    if not beats:
        return (
            np.empty((0, int(phase_samples)), dtype=float),
            np.asarray([], dtype=int),
        )
    return np.vstack(beats), np.asarray(periods, dtype=int)
