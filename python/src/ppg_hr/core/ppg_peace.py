"""Spectral energy ratio between 0–1 Hz and 1–3 Hz bands (port of ``PpgPeace.m``).

Used as a "is this PPG segment too still / too noisy?" indicator. Higher
values mean more sub-cardiac energy relative to the cardiac band.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ppg_peace"]

_FFT_LEN: int = 1 << 10  # 1024
_BAND_LOW_HZ: int = 1
_BAND_HIGH_HZ: int = 3


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(ddof=1)
    if sd == 0.0 or not np.isfinite(sd):
        return x - x.mean()
    return (x - x.mean()) / sd


def ppg_peace(signal: np.ndarray, fs: float) -> float:
    """Return the energy ratio Sq(0–1Hz) / Sq(1–3Hz)."""
    sig = np.atleast_1d(np.asarray(signal, dtype=float)).ravel()
    a = sig.size
    if a == 0:
        return float("nan")

    sig = _zscore(sig)
    spectrum = np.fft.fft(sig, _FFT_LEN)
    amp_full = np.abs(spectrum) / a
    half = _FFT_LEN // 2
    amp1 = amp_full[:half].copy()
    amp1[1:] *= 2.0

    cut_low = int(np.floor(_BAND_LOW_HZ * _FFT_LEN / fs))
    cut_high = int(np.floor(_BAND_HIGH_HZ * _FFT_LEN / fs))
    sq01 = float(np.sum(amp1[:cut_low] ** 2))
    sq12 = float(np.sum(amp1[cut_low:cut_high] ** 2))
    if sq12 == 0.0:
        return float("nan") if sq01 == 0.0 else float("inf")
    return sq01 / sq12
