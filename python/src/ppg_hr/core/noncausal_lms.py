"""Noncausal normalized LMS for v2 single-path filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .noncausal_tap import build_noncausal_tap_matrix


@dataclass(frozen=True)
class LmsDesign:
    M: int
    K: int
    mu: float
    curr_corr: float
    mode: str
    sensor_type: str


def map_delay_to_lms_design(
    delay_samples: int,
    sensor_type: str,
    params: Any,
    *,
    abs_corr: float = 0.0,
) -> LmsDesign:
    delay = int(delay_samples)
    max_order = int(getattr(params, "max_order", 16))
    m_base = int(getattr(params, "M_base", 1))
    c_scale = float(getattr(params, "C_scale", 1.0))
    k_max = int(getattr(params, "K_max", max_order))

    if delay > 0:
        M = min(max(1, int(np.floor(abs(delay) * c_scale))), max_order)
        K = 0
        mode = "causal"
    elif delay < 0:
        M = max(1, m_base)
        K = min(k_max, int(np.floor(abs(delay) * c_scale)))
        mode = "noncausal"
    else:
        M = max(1, m_base)
        K = 0
        mode = "zero_delay"

    mu_base = float(getattr(params, "lms_mu_base", 0.01))
    mu_min = float(getattr(params, "lms_mu_min", 1e-6))
    mu = max(mu_min, mu_base - abs(float(abs_corr)) / 100.0)
    return LmsDesign(
        M=int(M),
        K=int(K),
        mu=float(mu),
        curr_corr=float(abs_corr),
        mode=mode,
        sensor_type=str(sensor_type),
    )


def noncausal_lms_filter(
    u: np.ndarray,
    d: np.ndarray,
    M: int,
    K: int,
    mu: float,
) -> np.ndarray:
    u_arr = _zscore(np.asarray(u, dtype=float).ravel())
    d_arr = _zscore(np.asarray(d, dtype=float).ravel())
    n = min(u_arr.size, d_arr.size)
    if n == 0:
        return np.asarray([], dtype=float)

    u_arr = u_arr[:n]
    d_arr = d_arr[:n]
    X, valid_indices = build_noncausal_tap_matrix(u_arr, M, K)
    out = d_arr.copy()
    if valid_indices.size == 0:
        return out

    weights = np.zeros(X.shape[1], dtype=float)
    step = max(float(mu), 1e-12)
    eps = 1e-9
    for idx, uvec in zip(valid_indices, X, strict=True):
        y = float(weights @ uvec)
        err = float(d_arr[idx] - y)
        out[idx] = err
        weights += (step / (float(uvec @ uvec) + eps)) * uvec * err
    out[~np.isfinite(out)] = 0.0
    return out


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    if arr.size == 0:
        return arr
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    mean = float(np.mean(arr))
    if not np.isfinite(sd) or sd <= 1e-12:
        return arr - mean
    return (arr - mean) / sd
