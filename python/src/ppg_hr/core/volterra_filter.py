"""Second-order Volterra LMS — port of ``ref/.../Volterra/lmsFunc_h.m``.

Adds all unique quadratic cross-products of a delay window to the linear
FIR basis before the LMS update. ``M2 == 0`` degrades exactly to the linear
LMS (guarded by a dedicated regression test).

Both ``u`` and ``d`` are z-score normalised with sample stddev (ddof=1).

Output ``e`` is allocated to length ``N`` (mirrors MATLAB ``zeros(N,1)``).
"""

from __future__ import annotations

import numpy as np

from .lms_filter import lms_filter

__all__ = ["volterra_filter"]


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std(ddof=1)
    if sd == 0.0 or not np.isfinite(sd):
        return x - x.mean()
    return (x - x.mean()) / sd


def volterra_filter(
    mu: float,
    M1: int,
    M2: int,
    K: int,
    u: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run second-order Volterra LMS; return ``(e, w, ee)``.

    Parameters
    ----------
    mu : float
        Step size; the update uses ``2 * mu`` as in plain NLMS.
    M1 : int
        Linear FIR order.
    M2 : int
        Quadratic order; ``0`` degrades to linear LMS.
    K : int
        Delay.
    """
    if M2 == 0:
        e_lms, w, ee = lms_filter(mu, M1, K, u, d)
        e = np.zeros(np.atleast_1d(np.asarray(u)).size, dtype=float)
        e[: e_lms.size] = e_lms
        return e, w, ee

    u_arr = _zscore(np.atleast_1d(np.asarray(u, dtype=float)).ravel())
    d_arr = _zscore(np.atleast_1d(np.asarray(d, dtype=float)).ravel())

    n_samples = u_arr.size
    if d_arr.size < n_samples - K:
        raise ValueError(
            f"d must have at least N-K={n_samples - K} samples, got {d_arr.size}"
        )

    L1 = M1 + K
    if M2 > 0:
        L2 = M2 + K
        w_len = L1 + L2 * (L2 + 1) // 2
        tril_mask: np.ndarray | None = np.tril(np.ones((L2, L2), dtype=bool))
    else:
        L2 = 0
        w_len = L1
        tril_mask = None

    w = np.zeros(w_len, dtype=float)
    e = np.zeros(n_samples, dtype=float)
    ee = np.zeros(n_samples, dtype=float)

    m_start = max(M1, M2)
    if m_start < 1 or n_samples - K < m_start:
        return e, w, ee

    two_mu = 2.0 * float(mu)

    for n_py in range(m_start - 1, n_samples - K):
        idx1 = np.arange(n_py + K, n_py - M1, -1)
        u1 = u_arr[idx1]

        if M2 > 0:
            idx2 = np.arange(n_py + K, n_py - M2, -1)
            u2_base = u_arr[idx2]
            u2_mat = np.outer(u2_base, u2_base)
            assert tril_mask is not None
            u2 = u2_mat[tril_mask]
            U_vol = np.concatenate([u1, u2])
        else:
            U_vol = u1

        err = float(d_arr[n_py] - w @ U_vol)
        e[n_py] = err
        w = w + two_mu * U_vol * err

    return e, w, ee
