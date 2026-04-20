"""Tests for ``volterra_filter`` (second-order Volterra LMS)."""

from __future__ import annotations

import numpy as np

from ppg_hr.core.lms_filter import lms_filter
from ppg_hr.core.volterra_filter import volterra_filter


def test_output_shapes() -> None:
    rng = np.random.default_rng(0)
    n, M1, M2, K = 200, 4, 3, 1
    u = rng.normal(size=n)
    d = rng.normal(size=n)
    e, w, ee = volterra_filter(0.005, M1, M2, K, u, d)
    # MATLAB: e = zeros(N,1) — length N.
    assert e.shape == (n,)
    L1 = M1 + K
    L2 = M2 + K
    assert w.shape == (L1 + L2 * (L2 + 1) // 2,)
    assert ee.shape == (n,)


def test_initial_zeros_before_Mstart_and_trailing_K() -> None:
    rng = np.random.default_rng(1)
    n, M1, M2, K = 100, 5, 3, 2
    e, _, _ = volterra_filter(0.005, M1, M2, K, rng.normal(size=n), rng.normal(size=n))
    m_start = max(M1, M2)
    assert np.all(e[: m_start - 1] == 0)
    assert np.all(e[n - K :] == 0)
    assert np.any(e[m_start - 1 : n - K] != 0)


def test_M2_zero_equals_lms_bit_for_bit() -> None:
    """With M2=0 the Volterra filter must degrade to the linear LMS exactly.

    This is the most important regression guard — the MATLAB source comments
    explicitly call out this property.
    """
    rng = np.random.default_rng(2)
    n, M1, K = 300, 6, 1
    u = rng.normal(size=n)
    d = rng.normal(size=n)

    e_lms, w_lms, _ = lms_filter(0.01, M1, K, u, d)
    e_vol, w_vol, _ = volterra_filter(0.01, M1, 0, K, u, d)

    # lms_filter returns e of length N-K; volterra returns length N with the
    # same values in [M1-1, N-K) and zeros in [N-K, N).
    np.testing.assert_array_equal(e_vol[: n - K], e_lms)
    np.testing.assert_array_equal(e_vol[n - K :], 0)
    np.testing.assert_array_equal(w_vol, w_lms)


def test_zscore_invariance() -> None:
    rng = np.random.default_rng(3)
    u = rng.normal(size=300)
    d = rng.normal(size=300)
    e1, _, _ = volterra_filter(0.005, 4, 2, 1, u, d)
    e2, _, _ = volterra_filter(0.005, 4, 2, 1, 7.5 * u + 3.0, 2.0 * d - 1.0)
    np.testing.assert_allclose(e1, e2, atol=1e-9, rtol=1e-9)


def test_empty_when_M_exceeds_length() -> None:
    e, w, _ = volterra_filter(0.01, 100, 3, 0, np.zeros(10), np.zeros(10))
    assert e.shape == (10,)
    assert np.all(e == 0)
    L1, L2 = 100, 3
    assert w.shape == (L1 + L2 * (L2 + 1) // 2,)
    assert np.all(w == 0)
