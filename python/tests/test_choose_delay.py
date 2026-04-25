"""Tests for ``choose_delay``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.core import choose_delay
from ppg_hr.core.choose_delay import _lagged_segment_correlations
from ppg_hr.io.golden import assert_array_close, load_golden


def _shifted_signal(base: np.ndarray, lag_samples: int) -> np.ndarray:
    out = np.zeros_like(base)
    if lag_samples >= 0:
        out[lag_samples:] = base[: len(base) - lag_samples]
    else:
        out[:lag_samples] = base[-lag_samples:]
    return out


def _slow_lagged_correlations(
    ppg_seg: np.ndarray,
    signal: np.ndarray,
    starts: np.ndarray,
    win_len: int,
) -> np.ndarray:
    out = np.zeros(starts.size, dtype=float)
    for i, start in enumerate(starts):
        if start < 0 or start + win_len > signal.size:
            continue
        seg = signal[start : start + win_len]
        if ppg_seg.size != seg.size or ppg_seg.size < 2:
            continue
        sa = ppg_seg.std(ddof=1)
        sb = seg.std(ddof=1)
        if sa == 0.0 or sb == 0.0 or not np.isfinite(sa) or not np.isfinite(sb):
            continue
        corr = np.corrcoef(ppg_seg, seg)[0, 1]
        out[i] = 0.0 if not np.isfinite(corr) else float(corr)
    return out


def test_perfect_correlation_returns_zero_delay() -> None:
    fs = 100
    n = 30 * fs
    rng = np.random.default_rng(0)
    ppg = rng.normal(size=n)
    acc = ppg.copy()
    mh, ma, td_h, td_a = choose_delay(fs, 5.0, ppg, [acc, acc, acc], [acc, acc])
    assert td_a == 0 and td_h == 0
    assert ma.max() > 0.99 and mh.max() > 0.99


def test_lag_recovered() -> None:
    fs = 100
    n = 30 * fs
    rng = np.random.default_rng(1)
    ppg = rng.normal(size=n)
    # Reference shifted by +3 samples (acc_signal[t] = ppg[t-3])
    acc = _shifted_signal(ppg, 3)
    _, _, _td_h, td_a = choose_delay(fs, 5.0, ppg, [acc], [acc])
    # When ii = +3, p1 increases -> seg = acc[p1+3 : ...] = ppg[p1 : ...] (perfect match)
    assert td_a == 3


def test_lag_bounds_constrain_acc_and_hf_independently() -> None:
    fs = 100
    n = 30 * fs
    rng = np.random.default_rng(33)
    ppg = rng.normal(size=n)
    acc = _shifted_signal(ppg, 8)
    hf = _shifted_signal(ppg, -5)

    _mh, _ma, td_h, td_a = choose_delay(
        fs,
        5.0,
        ppg,
        [acc],
        [hf],
        lag_bounds_acc=(6, 10),
        lag_bounds_hf=(-7, -3),
    )

    assert 6 <= td_a <= 10
    assert -7 <= td_h <= -3


def test_invalid_lag_bounds_fall_back_to_default_range() -> None:
    fs = 100
    n = 30 * fs
    rng = np.random.default_rng(34)
    ppg = rng.normal(size=n)
    acc = _shifted_signal(ppg, 3)

    _mh, _ma, _td_h, td_a = choose_delay(
        fs,
        5.0,
        ppg,
        [acc],
        [acc],
        lag_bounds_acc=(10, 2),
    )

    assert td_a == 3


def test_lagged_segment_correlations_match_slow_reference() -> None:
    rng = np.random.default_rng(12)
    ppg_seg = rng.normal(size=80)
    signal = rng.normal(size=260)
    starts = np.array([-3, 0, 11, 47, 180, 181])

    expected = _slow_lagged_correlations(ppg_seg, signal, starts, ppg_seg.size)
    actual = _lagged_segment_correlations(ppg_seg, signal, starts, ppg_seg.size)

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_zero_signals_return_zero_corr() -> None:
    fs = 100
    n = 20 * fs
    ppg = np.random.default_rng(2).normal(size=n)
    zeros = np.zeros(n)
    mh, ma, td_h, td_a = choose_delay(fs, 5.0, ppg, [zeros], [zeros])
    assert mh.max() == 0 and ma.max() == 0
    # delay_range = round(0.2 * fs) = 20; argmax of all zeros -> first lag = -20
    expected_lag = -round(0.2 * fs)
    assert td_h == expected_lag and td_a == expected_lag


def test_matches_golden(golden_dir: Path) -> None:
    mat_path = golden_dir / "choose_delay.mat"
    if not mat_path.is_file():
        pytest.skip("Run MATLAB/gen_golden_all.m to produce choose_delay.mat")
    snap = load_golden(mat_path)
    mh, ma, td_h, td_a = choose_delay(
        int(snap["cd_fs"]),
        float(snap["cd_time1"]),
        np.asarray(snap["full_ppg"]).ravel(),
        [
            np.asarray(snap["full_accx"]).ravel(),
            np.asarray(snap["full_accy"]).ravel(),
            np.asarray(snap["full_accz"]).ravel(),
        ],
        [
            np.asarray(snap["full_hf1"]).ravel(),
            np.asarray(snap["full_hf2"]).ravel(),
        ],
    )
    assert_array_close(mh, np.asarray(snap["exp_mh"]).ravel(), atol=1e-9, rtol=1e-9)
    assert_array_close(ma, np.asarray(snap["exp_ma"]).ravel(), atol=1e-9, rtol=1e-9)
    assert td_h == int(snap["exp_td_h"])
    assert td_a == int(snap["exp_td_a"])
