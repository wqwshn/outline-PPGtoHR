"""Tests for ``choose_delay``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.core import choose_delay
from ppg_hr.io.golden import assert_array_close, load_golden


def _shifted_signal(base: np.ndarray, lag_samples: int) -> np.ndarray:
    out = np.zeros_like(base)
    if lag_samples >= 0:
        out[lag_samples:] = base[: len(base) - lag_samples]
    else:
        out[:lag_samples] = base[-lag_samples:]
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


def test_zero_signals_return_zero_corr() -> None:
    fs = 100
    n = 20 * fs
    ppg = np.random.default_rng(2).normal(size=n)
    zeros = np.zeros(n)
    mh, ma, td_h, td_a = choose_delay(fs, 5.0, ppg, [zeros], [zeros])
    assert mh.max() == 0 and ma.max() == 0
    assert td_h == -5 and td_a == -5  # argmax of all zeros -> first index


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
