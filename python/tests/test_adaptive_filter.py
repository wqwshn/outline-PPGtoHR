"""Tests for ``apply_adaptive_cascade`` dispatch layer."""

from __future__ import annotations

import numpy as np
import pytest

from ppg_hr.core.adaptive_filter import apply_adaptive_cascade
from ppg_hr.core.klms_filter import klms_filter
from ppg_hr.core.lms_filter import lms_filter
from ppg_hr.core.volterra_filter import volterra_filter
from ppg_hr.params import SolverParams


def _signals(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), rng.normal(size=n)


def test_lms_dispatch_bit_for_bit() -> None:
    """apply_adaptive_cascade('lms', ...) must match lms_filter(mu_base - corr/100, ...)."""
    u, d = _signals()
    params = SolverParams(adaptive_filter="lms", lms_mu_base=0.01)
    corr = 0.3
    out = apply_adaptive_cascade(
        strategy="lms", mu_base=0.01, corr=corr,
        order=5, K=1, u=u, d=d, params=params,
    )
    expected, _, _ = lms_filter(0.01 - corr / 100.0, 5, 1, u, d)
    np.testing.assert_array_equal(out, expected)


def test_klms_dispatch_uses_fixed_step_size() -> None:
    """KLMS must use params.klms_step_size, NOT mu_base - corr/100."""
    u, d = _signals()
    params = SolverParams(
        adaptive_filter="klms",
        klms_step_size=0.2, klms_sigma=1.0, klms_epsilon=0.1,
    )
    corr = 0.3
    out = apply_adaptive_cascade(
        strategy="klms", mu_base=0.01, corr=corr,
        order=5, K=1, u=u, d=d, params=params,
    )
    expected, _, _ = klms_filter(0.2, 5, 1, u, d, sigma=1.0, epsilon=0.1)
    np.testing.assert_array_equal(out, expected)


def test_volterra_dispatch_uses_corr_adaptive_step() -> None:
    """Volterra keeps mu_base - corr/100 (matching MATLAB reference)."""
    u, d = _signals()
    params = SolverParams(
        adaptive_filter="volterra",
        lms_mu_base=0.01, volterra_max_order_vol=3,
    )
    corr = 0.3
    out = apply_adaptive_cascade(
        strategy="volterra", mu_base=0.01, corr=corr,
        order=5, K=1, u=u, d=d, params=params,
    )
    expected, _, _ = volterra_filter(0.01 - corr / 100.0, 5, 3, 1, u, d)
    np.testing.assert_array_equal(out, expected)


def test_unknown_strategy_raises() -> None:
    u, d = _signals(n=50)
    with pytest.raises(ValueError, match="unknown.*strategy"):
        apply_adaptive_cascade(
            strategy="bogus", mu_base=0.01, corr=0.0,
            order=3, K=0, u=u, d=d, params=SolverParams(),
        )
