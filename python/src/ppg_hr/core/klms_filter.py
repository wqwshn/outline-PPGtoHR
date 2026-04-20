"""Quantized Kernel LMS (QKLMS) — port of ``ref/.../KLMS/lmsFunc_h.m``.

Gaussian-kernel adaptive filter with a quantized dictionary of centers.
Both ``u`` (reference) and ``d`` (desired) are z-score normalised with sample
standard deviation (``ddof=1``) before adaptation, matching MATLAB ``zscore``.

Output ``e`` is allocated to length ``N`` (not ``N - K``) and never written
outside ``[M - 1, N - K)`` — this mirrors MATLAB's ``e = zeros(N, 1)``.

Notes
-----
MATLAB compares ``dists`` (which already holds *squared* distances, from
``sum(diffs.^2, 1)``) directly against ``epsilon``. We mirror that literal
behaviour — ``epsilon`` is effectively a squared-distance threshold despite
being labelled "距离阈值" in the reference comments.
"""

from __future__ import annotations

import numpy as np

__all__ = ["klms_filter"]


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(ddof=1)
    if sd == 0.0 or not np.isfinite(sd):
        return x - x.mean()
    return (x - x.mean()) / sd


def klms_filter(
    mu: float,
    M: int,
    K: int,
    u: np.ndarray,
    d: np.ndarray,
    sigma: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run QKLMS; return ``(e, A, C)``.

    Parameters
    ----------
    mu : float
        Learning rate.
    M : int
        Embedding size (FIR-equivalent tap count).
    K : int
        Delay parameter (samples added to the end of the input window).
    u, d : np.ndarray
        Reference and desired signals; each is z-scored internally.
    sigma : float
        Gaussian kernel bandwidth.
    epsilon : float
        Squared-distance quantization threshold. ``0`` grows the dictionary
        every step; very large values collapse the dictionary to one center.

    Returns
    -------
    e : np.ndarray, shape ``(N,)``
        Prediction error; zeros outside ``[M - 1, N - K)``.
    A : np.ndarray, shape ``(L,)``
        Dictionary weights.
    C : np.ndarray, shape ``(M + K, L)``
        Dictionary centers (one column per entry).
    """
    u_arr = _zscore(np.atleast_1d(np.asarray(u, dtype=float)).ravel())
    d_arr = _zscore(np.atleast_1d(np.asarray(d, dtype=float)).ravel())

    n_samples = u_arr.size
    if d_arr.size < n_samples - K:
        raise ValueError(
            f"d must have at least N-K={n_samples - K} samples, got {d_arr.size}"
        )

    e = np.zeros(n_samples, dtype=float)
    C = np.zeros((M + K, 0), dtype=float)
    A = np.zeros(0, dtype=float)

    if M < 1 or n_samples - K < M:
        return e, A, C

    two_sigma2 = 2.0 * float(sigma) ** 2
    eps_threshold = float(epsilon)

    # MATLAB n = M : N-K (1-based inclusive) → Python 0-based [M-1, N-K).
    for n_py in range(M - 1, n_samples - K):
        idx = np.arange(n_py + K, n_py - M, -1)
        uvec = u_arr[idx]

        if C.shape[1] == 0:
            err = float(d_arr[n_py])
            e[n_py] = err
            C = uvec.reshape(-1, 1)
            A = np.array([mu * err], dtype=float)
            continue

        diffs = C - uvec[:, None]
        dists = np.sum(diffs * diffs, axis=0)
        if two_sigma2 > 0:
            kappa = np.exp(-dists / two_sigma2)
        else:
            kappa = (dists == 0).astype(float)
        y = float(A @ kappa)
        err = float(d_arr[n_py]) - y
        e[n_py] = err

        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        if min_dist <= eps_threshold:
            A[min_idx] += mu * err
        else:
            C = np.concatenate([C, uvec.reshape(-1, 1)], axis=1)
            A = np.concatenate([A, np.array([mu * err])])

    return e, A, C
