"""Normalised LMS adaptive filter (port of ``lmsFunc_h.m``).

Both ``u`` (reference) and ``d`` (desired) are z-score normalised with sample
standard deviation (ddof=1) before adaptation, matching MATLAB ``zscore``.

The output ``e`` has length ``N - K`` and stores the prediction error from
sample index ``M-1`` onward (zero before that). ``ee`` is allocated to length
``N`` and never written, kept for signature parity with the MATLAB function.
"""

from __future__ import annotations

import numpy as np

__all__ = ["lms_filter"]


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(ddof=1)
    if sd == 0.0 or not np.isfinite(sd):
        return x - x.mean()
    return (x - x.mean()) / sd


def lms_filter(
    mu: float,
    M: int,
    K: int,
    u: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run normalised LMS; return ``(e, w, ee)``."""
    u_arr = _zscore(np.atleast_1d(np.asarray(u, dtype=float)).ravel())
    d_arr = _zscore(np.atleast_1d(np.asarray(d, dtype=float)).ravel())

    n_samples = u_arr.size
    if d_arr.size != n_samples:
        raise ValueError("u and d must have the same length")

    w = np.zeros(M + K, dtype=float)
    e = np.zeros(max(n_samples - K, 0), dtype=float)
    ee = np.zeros(n_samples, dtype=float)

    if M < 1 or n_samples - K < M:
        return e, w, ee

    # MATLAB loop: n = M : N-K (1-based, inclusive)
    # Python (0-based): n_py from M-1 to N-K-1 inclusive
    for n_py in range(M - 1, n_samples - K):
        # MATLAB ``u(n+K : -1 : n-M+1)`` = u_py[n_py+K], u_py[n_py+K-1], ..., u_py[n_py-M+1]
        idx = np.arange(n_py + K, n_py - M, -1)
        uvec = u_arr[idx]
        err = d_arr[n_py] - w @ uvec
        e[n_py] = err
        w = w + (2.0 * mu) * uvec * err

    return e, w, ee
