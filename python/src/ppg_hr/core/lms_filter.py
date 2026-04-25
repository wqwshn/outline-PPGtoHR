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

try:  # Optional acceleration; pure Python fallback is kept below.
    from numba import njit
except Exception:  # pragma: no cover - depends on local environment
    njit = None


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(ddof=1)
    if sd == 0.0 or not np.isfinite(sd):
        return x - x.mean()
    return (x - x.mean()) / sd


def _lms_filter_core_python(
    mu: float,
    M: int,
    K: int,
    u_arr: np.ndarray,
    d_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = u_arr.size
    w = np.zeros(M + K, dtype=float)
    e = np.zeros(max(n_samples - K, 0), dtype=float)
    ee = np.zeros(n_samples, dtype=float)

    if M < 1 or n_samples - K < M:
        return e, w, ee

    # MATLAB loop: n = M : N-K (1-based, inclusive)
    # Python (0-based): n_py from M-1 to N-K-1 inclusive
    for n_py in range(M - 1, n_samples - K):
        # MATLAB ``u(n+K : -1 : n-M+1)`` spans ``M + K`` samples.
        uvec = u_arr[n_py - M + 1 : n_py + K + 1][::-1]
        err = d_arr[n_py] - w @ uvec
        e[n_py] = err
        w += (2.0 * mu) * uvec * err

    return e, w, ee


if njit is not None:

    @njit(cache=True)
    def _lms_filter_core_numba(
        mu: float,
        M: int,
        K: int,
        u_arr: np.ndarray,
        d_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples = u_arr.size
        w = np.zeros(M + K, dtype=np.float64)
        e = np.zeros(max(n_samples - K, 0), dtype=np.float64)
        ee = np.zeros(n_samples, dtype=np.float64)

        if M < 1 or n_samples - K < M:
            return e, w, ee

        step = 2.0 * mu
        span = M + K
        for n_py in range(M - 1, n_samples - K):
            pred = 0.0
            for j in range(span):
                pred += w[j] * u_arr[n_py + K - j]
            err = d_arr[n_py] - pred
            e[n_py] = err
            for j in range(span):
                w[j] += step * u_arr[n_py + K - j] * err

        return e, w, ee

else:
    _lms_filter_core_numba = None


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

    # MATLAB lmsFunc_h uses N = length(u) and only needs d up to index N-K.
    # In the cascade case the previous-stage output e shrinks by K each round,
    # so d may legitimately be shorter than u.
    n_samples = u_arr.size
    if d_arr.size < n_samples - K:
        raise ValueError(
            f"d must have at least N-K={n_samples - K} samples, got {d_arr.size}"
        )

    if _lms_filter_core_numba is not None:
        return _lms_filter_core_numba(mu, int(M), int(K), u_arr, d_arr)
    return _lms_filter_core_python(mu, int(M), int(K), u_arr, d_arr)
