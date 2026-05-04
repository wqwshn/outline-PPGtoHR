"""Random Fourier Feature LMS for v2 noncausal filtering."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .noncausal_lms import _zscore
from .noncausal_tap import build_noncausal_tap_matrix


@lru_cache(maxsize=64)
def get_rff_weights(
    D: int,
    span: int,
    sigma: float,
    rff_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    D = max(1, int(D))
    span = max(1, int(span))
    sigma = max(float(sigma), 1e-6)
    rng = np.random.default_rng(int(rff_seed) % (2**32))
    W = rng.normal(loc=0.0, scale=1.0 / sigma, size=(D, span))
    b = rng.uniform(0.0, 2.0 * np.pi, size=D)
    W.setflags(write=False)
    b.setflags(write=False)
    return W, b


def noncausal_rff_lms_filter(
    u: np.ndarray,
    d: np.ndarray,
    M: int,
    K: int,
    mu: float,
    D: int,
    sigma: float,
    rff_seed: int,
    mu_min: float = 1e-6,
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

    D_i = max(1, int(D))
    W, b = get_rff_weights(D_i, X.shape[1], float(sigma), int(rff_seed))
    Z = float(np.sqrt(2.0 / D_i)) * np.cos(X @ W.T + b)
    theta = np.zeros(D_i, dtype=float)
    step = max(float(mu_min), float(mu) if np.isfinite(mu) else float(mu_min))
    for idx, z in zip(valid_indices, Z, strict=True):
        y = float(theta @ z)
        err = float(d_arr[idx] - y)
        out[idx] = err
        theta += step * err * z
    out[~np.isfinite(out)] = 0.0
    return out
