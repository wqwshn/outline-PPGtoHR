"""Causal raw-PPG evidence for post-motion observability recovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate

from .raw_fft_candidates import RawFftCandidateFrame


@dataclass(frozen=True)
class PpgObservabilityEvidence:
    periodicity: float
    peak_competition: float


def measure_ppg_observability(
    signal: np.ndarray,
    fs: float,
    candidates: RawFftCandidateFrame,
) -> PpgObservabilityEvidence:
    """Measure answer-free pulse periodicity and spectral peak separation."""

    values = np.asarray(signal, dtype=float).ravel()
    values = values[np.isfinite(values)]
    periodicity = _physiological_periodicity(values, float(fs))
    peaks = candidates.top()
    if not peaks or peaks[0][1] <= 0.0:
        competition = 0.0
    elif len(peaks) == 1 or peaks[1][1] <= 0.0:
        competition = float("inf")
    else:
        competition = float(peaks[0][1] / peaks[1][1])
    return PpgObservabilityEvidence(
        periodicity=float(periodicity),
        peak_competition=competition,
    )


def _physiological_periodicity(values: np.ndarray, fs: float) -> float:
    if values.size < 3 or not np.isfinite(fs) or fs <= 0.0:
        return 0.0
    centered = values - float(np.mean(values))
    energy = float(np.dot(centered, centered))
    if energy <= np.finfo(float).eps:
        return 0.0
    autocorrelation = correlate(centered, centered, mode="full", method="fft")
    positive = autocorrelation[values.size - 1 :]
    min_lag = max(1, int(np.floor(fs * 60.0 / 240.0)))
    max_lag = min(positive.size - 1, int(np.ceil(fs * 60.0 / 42.0)))
    if max_lag < min_lag:
        return 0.0
    score = float(np.max(positive[min_lag : max_lag + 1]) / energy)
    return float(np.clip(score, 0.0, 1.0))
