"""Shared raw-PPG FFT candidate evidence for reset trackers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import hamming

_FFT_LENGTH = 1 << 13
_MIN_FREQUENCY_HZ = 0.7
_MAX_FREQUENCY_HZ = 4.0
_FULL_CANDIDATE_PEAK_THRESHOLD_RATIO = 0.15


@dataclass(frozen=True)
class RawFftCandidateFrame:
    frequencies_hz: np.ndarray
    amplitudes: np.ndarray
    peak_indices: np.ndarray
    ordered_peak_indices: np.ndarray

    def top(self, count: int = 5) -> tuple[tuple[float, float], ...]:
        idx = self.ordered_peak_indices[:count]
        return tuple(
            (float(self.frequencies_hz[i] * 60.0), float(self.amplitudes[i])) for i in idx
        )


def extract_raw_fft_candidates(signal: np.ndarray, fs: float) -> RawFftCandidateFrame:
    """Extract the solver's complete raw-PPG FFT candidate evidence."""

    frequencies_hz, amplitudes = _candidate_peak_spectrum(signal, fs)
    return _frame_from_spectrum(frequencies_hz, amplitudes)


def _frame_from_spectrum(
    frequencies_hz: np.ndarray,
    amplitudes: np.ndarray,
) -> RawFftCandidateFrame:
    peak_indices = _candidate_peak_indices(
        frequencies_hz,
        amplitudes,
        threshold_ratio=_FULL_CANDIDATE_PEAK_THRESHOLD_RATIO,
    )
    ordered_peak_indices = peak_indices[
        np.argsort(-amplitudes[peak_indices], kind="stable")
    ]
    return RawFftCandidateFrame(
        frequencies_hz=frequencies_hz,
        amplitudes=amplitudes,
        peak_indices=peak_indices,
        ordered_peak_indices=ordered_peak_indices,
    )


def _candidate_peak_spectrum(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    sig = np.asarray(signal, dtype=float).ravel()
    sig = sig[np.isfinite(sig)]
    if sig.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    work = (sig - float(np.nanmean(sig))) * hamming(sig.size)
    spectrum = np.fft.fft(work, _FFT_LENGTH)
    amp = np.abs(spectrum[: _FFT_LENGTH // 2]) / max(1, work.size)
    amp[1:] *= 2.0
    freq = float(fs) * np.arange(_FFT_LENGTH // 2, dtype=float) / float(_FFT_LENGTH)
    band = (freq > _MIN_FREQUENCY_HZ) & (freq < _MAX_FREQUENCY_HZ)
    return freq[band], amp[band]


def _candidate_peak_indices(
    freqs: np.ndarray,
    amps: np.ndarray,
    *,
    threshold_ratio: float,
) -> np.ndarray:
    if freqs.size == 0 or amps.size == 0:
        return np.asarray([], dtype=int)
    peaks, _ = find_peaks(amps)
    if peaks.size == 0:
        return np.asarray([], dtype=int)
    peak_amps = amps[peaks]
    finite = np.isfinite(peak_amps)
    if not finite.any():
        return np.asarray([], dtype=int)
    peaks = peaks[finite]
    peak_amps = peak_amps[finite]
    threshold = float(np.nanmax(peak_amps)) * float(threshold_ratio)
    return peaks[peak_amps > threshold]
