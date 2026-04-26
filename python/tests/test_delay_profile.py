"""Tests for adaptive delay-search prefit profiles."""

from __future__ import annotations

import numpy as np

from ppg_hr.core.delay_profile import (
    DelayBounds,
    DelaySearchProfile,
    estimate_delay_search_profile,
)
from ppg_hr.params import SolverParams


def _shifted_signal(base: np.ndarray, lag_samples: int) -> np.ndarray:
    out = np.zeros_like(base)
    if lag_samples >= 0:
        out[lag_samples:] = base[: len(base) - lag_samples]
    else:
        out[:lag_samples] = base[-lag_samples:]
    return out


def test_profile_finds_narrow_hf_and_acc_bounds() -> None:
    fs = 50
    n = 80 * fs
    rng = np.random.default_rng(44)
    ppg = rng.normal(size=n)
    hf = _shifted_signal(ppg, -4)
    acc = _shifted_signal(ppg, 7)
    acc_mag = np.abs(acc) + 0.2
    params = SolverParams(
        fs_target=fs,
        delay_prefit_windows=6,
        delay_prefit_min_corr=0.2,
        delay_prefit_margin_samples=1,
        delay_prefit_min_span_samples=2,
    )

    profile = estimate_delay_search_profile(
        fs=fs,
        ppg=ppg,
        acc_signals=[acc],
        hf_signals=[hf],
        acc_mag=acc_mag,
        motion_threshold=0.01,
        params=params,
    )

    assert isinstance(profile, DelaySearchProfile)
    assert not profile.hf.fallback
    assert not profile.acc.fallback
    assert profile.hf.bounds.min_lag <= -4 <= profile.hf.bounds.max_lag
    assert profile.acc.bounds.min_lag <= 7 <= profile.acc.bounds.max_lag
    assert profile.hf.bounds.width < profile.default_bounds.width
    assert profile.acc.bounds.width < profile.default_bounds.width


def test_profile_falls_back_for_low_correlation() -> None:
    fs = 50
    n = 30 * fs
    ppg = np.ones(n)
    zeros = np.zeros(n)
    params = SolverParams(fs_target=fs)

    profile = estimate_delay_search_profile(
        fs=fs,
        ppg=ppg,
        acc_signals=[zeros],
        hf_signals=[zeros],
        acc_mag=zeros,
        motion_threshold=1.0,
        params=params,
    )

    assert profile.hf.fallback
    assert profile.acc.fallback
    assert profile.hf.bounds == profile.default_bounds
    assert profile.acc.bounds == profile.default_bounds
    assert "insufficient" in profile.hf.reason


def test_delay_bounds_as_tuple_and_width() -> None:
    bounds = DelayBounds(-3, 5)
    assert bounds.as_tuple() == (-3, 5)
    assert bounds.width == 8


# ---------------------------------------------------------------------------
# Staged (tiered) delay prefit tests
# ---------------------------------------------------------------------------


def test_profile_small_lag_stops_early() -> None:
    """Small lag should be found within early tiers and produce narrow bounds."""
    fs = 25
    n = 80 * fs
    rng = np.random.default_rng(44)
    ppg = rng.normal(size=n)
    hf = _shifted_signal(ppg, -3)
    acc = _shifted_signal(ppg, 4)
    acc_mag = np.abs(acc) + 0.2
    params = SolverParams(
        fs_target=fs,
        delay_prefit_max_seconds=0.8,
        delay_prefit_windows=20,
        delay_prefit_min_corr=0.1,
        delay_prefit_margin_samples=1,
        delay_prefit_min_span_samples=2,
    )
    profile = estimate_delay_search_profile(
        fs=fs,
        ppg=ppg,
        acc_signals=[acc],
        hf_signals=[hf],
        acc_mag=acc_mag,
        motion_threshold=0.01,
        params=params,
    )
    assert profile.scanned_windows < 20
    assert profile.hf.bounds.min_lag <= -3 <= profile.hf.bounds.max_lag
    assert profile.acc.bounds.min_lag <= 4 <= profile.acc.bounds.max_lag


def test_profile_large_lag_expands_to_wider_level() -> None:
    """Large lag requires expanding beyond the first tier.

    Uses low-pass filtered noise (broad autocorrelation) so the true lag
    peak is detectable even at ~0.3-0.4 s offset.
    """
    fs = 50
    n = 80 * fs
    rng = np.random.default_rng(42)
    raw = rng.normal(size=n)
    # Moving-average (40 samples) gives significant correlation up to ~0.8 s
    ppg = np.convolve(raw, np.ones(40) / 40, mode="same")
    hf = _shifted_signal(ppg, -15)  # 0.3 s — beyond L1 (±10), within L2 (±20)
    acc = _shifted_signal(ppg, 18)  # 0.36 s — beyond L1, within L2
    acc_mag = np.abs(acc) + 0.2
    params = SolverParams(
        fs_target=fs,
        delay_prefit_max_seconds=0.8,
        delay_prefit_windows=20,
        delay_prefit_min_corr=0.1,
        delay_prefit_margin_samples=2,
        delay_prefit_min_span_samples=4,
    )
    profile = estimate_delay_search_profile(
        fs=fs,
        ppg=ppg,
        acc_signals=[acc],
        hf_signals=[hf],
        acc_mag=acc_mag,
        motion_threshold=0.01,
        params=params,
    )
    assert profile.hf.bounds.min_lag <= -15 <= profile.hf.bounds.max_lag
    assert profile.acc.bounds.min_lag <= 18 <= profile.acc.bounds.max_lag
    assert profile.scanned_windows >= 10
