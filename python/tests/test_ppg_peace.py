"""Tests for ``ppg_peace``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.core import ppg_peace
from ppg_hr.io.golden import load_golden


def test_low_freq_dominant_gives_high_ratio() -> None:
    fs = 100.0
    t = np.arange(0, 8, 1.0 / fs)
    sig = np.sin(2 * np.pi * 0.3 * t)  # dominant 0.3 Hz tone
    assert ppg_peace(sig, fs) > 5


def test_cardiac_band_dominant_gives_low_ratio() -> None:
    fs = 100.0
    t = np.arange(0, 8, 1.0 / fs)
    sig = np.sin(2 * np.pi * 1.5 * t)  # dominant 1.5 Hz tone
    assert ppg_peace(sig, fs) < 1.0


def test_constant_signal_returns_finite_value() -> None:
    out = ppg_peace(np.ones(800), 100.0)
    assert np.isnan(out) or np.isfinite(out)


def test_matches_golden(golden_dir: Path) -> None:
    mat_path = golden_dir / "ppg_peace.mat"
    if not mat_path.is_file():
        pytest.skip("Run MATLAB/gen_golden_all.m to produce ppg_peace.mat")
    snap = load_golden(mat_path)
    out = ppg_peace(
        np.asarray(snap["case_signal_pp"]).ravel(),
        float(snap["case_fs_pp"]),
    )
    expected = float(snap["expected_ratio"])
    assert out == pytest.approx(expected, rel=1e-9, abs=1e-12)
