"""Tests for AS-LMS adaptive step-size filter."""

from __future__ import annotations

import numpy as np

from ppg_hr.core.as_lms_filter import as_lms_filter
from ppg_hr.core.lms_filter import lms_filter


def test_as_lms_output_shapes_and_mu_bounds() -> None:
    rng = np.random.default_rng(0)
    n, M, K = 240, 5, 1
    u = rng.normal(size=n)
    d = 0.35 * u + rng.normal(scale=0.2, size=n)

    e, w, mu_trace = as_lms_filter(
        0.01,
        M,
        K,
        u,
        d,
        rho=1e-4,
        mu_min=0.002,
        mu_max=0.02,
    )

    assert e.shape == (n - K,)
    assert w.shape == (M + K,)
    assert mu_trace.shape == (n,)
    assert np.all(np.isfinite(e))
    assert np.all(np.isfinite(w))
    assert np.all(np.isfinite(mu_trace))
    assert float(mu_trace.min()) >= 0.002
    assert float(mu_trace.max()) <= 0.02
    assert np.any(np.diff(mu_trace[M:]) != 0.0)


def test_zero_rho_matches_existing_lms_exactly() -> None:
    rng = np.random.default_rng(1)
    u = rng.normal(size=300)
    d = rng.normal(size=300)
    M, K, mu = 4, 1, 0.005

    e_as, w_as, _ = as_lms_filter(
        mu,
        M,
        K,
        u,
        d,
        rho=0.0,
        mu_min=0.0,
        mu_max=1.0,
    )
    e_lms, w_lms, _ = lms_filter(mu, M, K, u, d)

    np.testing.assert_array_equal(e_as, e_lms)
    np.testing.assert_array_equal(w_as, w_lms)


def test_degenerate_inputs_remain_finite() -> None:
    u = np.zeros(20)
    d = np.zeros(20)

    e, w, mu_trace = as_lms_filter(
        0.01,
        100,
        0,
        u,
        d,
        rho=1e-4,
        mu_min=1e-6,
        mu_max=0.05,
    )

    assert e.shape == (20,)
    assert w.shape == (100,)
    assert np.all(e == 0.0)
    assert np.all(w == 0.0)
    assert np.all(mu_trace == 0.01)
