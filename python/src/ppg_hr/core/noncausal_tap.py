"""Noncausal tap-matrix helpers shared by v2 adaptive filters."""

from __future__ import annotations

import numpy as np


def build_noncausal_tap_matrix(
    u: np.ndarray,
    M: int,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(u, dtype=float).ravel()
    M = max(1, int(M))
    K = max(0, int(K))
    span = M + K
    n = arr.size
    if n == 0 or n < span:
        return np.zeros((0, span), dtype=float), np.zeros(0, dtype=int)

    start = K
    stop = n - M + 1
    indices = np.arange(start, stop, dtype=int)
    X = np.zeros((indices.size, span), dtype=float)
    for row, idx in enumerate(indices):
        tap_idx = np.arange(idx + K, idx - M, -1, dtype=int)
        X[row] = arr[tap_idx]
    return X, indices
