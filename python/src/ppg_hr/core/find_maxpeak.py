"""Sort candidate peak frequencies by descending amplitude (port of ``Find_maxpeak.m``)."""

from __future__ import annotations

import numpy as np

__all__ = ["find_maxpeak"]


def find_maxpeak(
    freqs: np.ndarray, _placeholder, amps: np.ndarray
) -> np.ndarray:
    """Return ``freqs`` reordered by descending ``amps``.

    The middle argument is kept for signature parity with the MATLAB caller
    (which passes the same frequency vector twice).
    """
    f = np.atleast_1d(np.asarray(freqs)).ravel()
    a = np.atleast_1d(np.asarray(amps)).ravel()
    if f.size == 0 or a.size == 0:
        return np.array([])
    # MATLAB ``sort(..., 'descend')`` is stable; numpy.argsort default is not,
    # but stability matters only on amplitude ties (rare in float64 spectra).
    idx = np.argsort(-a, kind="stable")
    return f[idx]
