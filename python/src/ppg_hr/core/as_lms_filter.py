"""Adaptive step-size LMS filter based on Ram et al. 2012 AS-LMS."""

from __future__ import annotations

import numpy as np

from .lms_filter import _zscore

__all__ = ["as_lms_filter"]


def _clip_mu(mu: float, mu_min: float, mu_max: float) -> float:
    lo = float(mu_min) if np.isfinite(mu_min) else 0.0
    lo = max(0.0, lo)
    hi = float(mu_max) if np.isfinite(mu_max) else lo
    hi = max(lo, hi)
    value = float(mu) if np.isfinite(mu) else lo
    return min(hi, max(lo, value))


def _as_lms_filter_core_python(
    mu0: float,
    rho: float,
    mu_min: float,
    mu_max: float,
    M: int,
    K: int,
    u_arr: np.ndarray,
    d_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = u_arr.size
    span = M + K
    w = np.zeros(span, dtype=float)
    gamma = np.zeros(span, dtype=float)
    e = np.zeros(max(n_samples - K, 0), dtype=float)
    mu = _clip_mu(mu0, mu_min, mu_max)
    mu_trace = np.full(n_samples, mu, dtype=float)

    if M < 1 or n_samples - K < M:
        return e, w, mu_trace

    rho_value = max(0.0, float(rho) if np.isfinite(rho) else 0.0)
    for n_py in range(M - 1, n_samples - K):
        pred = 0.0
        for j in range(span):
            pred += w[j] * u_arr[n_py + K - j]
        err = d_arr[n_py] - pred
        e[n_py] = err

        gamma_dot_u = 0.0
        for j in range(span):
            gamma_dot_u += gamma[j] * u_arr[n_py + K - j]

        step = 2.0 * mu
        for j in range(span):
            u_j = u_arr[n_py + K - j]
            w[j] += step * u_j * err
            gamma[j] += 2.0 * err * u_j - step * u_j * gamma_dot_u

        mu = _clip_mu(mu + rho_value * err * gamma_dot_u, mu_min, mu_max)
        mu_trace[n_py] = mu

    return e, w, mu_trace


def as_lms_filter(
    mu0: float,
    M: int,
    K: int,
    u: np.ndarray,
    d: np.ndarray,
    *,
    rho: float = 1e-4,
    mu_min: float = 1e-6,
    mu_max: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run AS-LMS; return ``(e, w, mu_trace)``.

    ``rho=0`` intentionally degrades to the existing LMS recurrence, preserving
    the project's current ``2 * mu`` weight-update convention.
    """
    u_arr = _zscore(np.atleast_1d(np.asarray(u, dtype=float)).ravel())
    d_arr = _zscore(np.atleast_1d(np.asarray(d, dtype=float)).ravel())

    n_samples = u_arr.size
    if d_arr.size < n_samples - K:
        raise ValueError(
            f"d must have at least N-K={n_samples - K} samples, got {d_arr.size}"
        )

    return _as_lms_filter_core_python(
        float(mu0),
        float(rho),
        float(mu_min),
        float(mu_max),
        int(M),
        int(K),
        u_arr,
        d_arr,
    )
