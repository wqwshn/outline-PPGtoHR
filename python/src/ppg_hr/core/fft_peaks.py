"""Detect spectral peaks within the cardiac band (port of ``FFT_Peaks.m``).

Pads ``signal`` to ``2**13`` samples, computes the one-sided amplitude
spectrum, then returns peaks whose amplitude exceeds ``percent`` of the local
maximum AND whose 1-based bin index lies strictly inside the open interval
``(1*Len/Fs + 1, 4*Len/Fs)`` (i.e. the cardiac 1–4 Hz window in bin indices).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

__all__ = ["fft_peaks"]

_FFT_LEN: int = 1 << 13  # 8192


def fft_peaks(
    signal: np.ndarray, fs: float, percent: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frequencies_hz, amplitudes)`` of cardiac-band spectral peaks."""
    sig = np.atleast_1d(np.asarray(signal, dtype=float)).ravel()
    a = sig.size
    if a == 0:
        return np.array([]), np.array([])

    spectrum = np.fft.fft(sig, _FFT_LEN)
    amp_full = np.abs(spectrum) / a
    half = _FFT_LEN // 2
    amp1 = amp_full[:half].copy()
    amp1[1:] *= 2.0
    freq_axis = fs * np.arange(half) / _FFT_LEN

    # ``locs`` from MATLAB ``findpeaks`` is 1-based; scipy.signal.find_peaks
    # returns 0-based indices, so add 1 when comparing with the original
    # band edges to preserve MATLAB semantics exactly.
    peaks_idx, _ = find_peaks(amp1)
    if peaks_idx.size == 0:
        return np.array([]), np.array([])

    locs_1based = peaks_idx + 1
    free_low = 1.0 * _FFT_LEN / fs + 1.0
    free_high = 4.0 * _FFT_LEN / fs

    valid_mask = (locs_1based < free_high) & (locs_1based > free_low)
    pks2 = amp1[peaks_idx][valid_mask]
    if pks2.size == 0:
        return np.array([]), np.array([])

    threshold = pks2.max() * percent
    keep_mask = (amp1[peaks_idx] > threshold) & valid_mask
    keep_idx = peaks_idx[keep_mask]
    return freq_axis[keep_idx], amp1[keep_idx]
