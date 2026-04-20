"""Tests for ``find_real_hr``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.core import find_real_hr
from ppg_hr.io.golden import assert_array_close, load_golden


def _ref():
    times = np.arange(0.0, 200.0, 1.0)
    bpm = 60.0 + 0.5 * times  # increasing reference
    return np.column_stack([times, bpm])


def test_returns_hz_at_window_center() -> None:
    ref = _ref()
    # window starting at t=10 s, center is at 14 s, BPM = 60 + 0.5*14 = 67
    out = find_real_hr("dummy", 10.0, ref)
    assert out == pytest.approx(67.0 / 60.0, rel=1e-12)


def test_extrapolation_beyond_range() -> None:
    ref = _ref()
    out = find_real_hr("dummy", 1000.0, ref)
    assert out > 0  # extrapolated, not zero


def test_invalid_ref_returns_zero() -> None:
    out = find_real_hr("dummy", 5.0, np.array([[1.0, 60.0]]))
    assert out == 0.0


def test_matches_golden(golden_dir: Path) -> None:
    mat_path = golden_dir / "find_real_hr.mat"
    if not mat_path.is_file():
        pytest.skip("Run MATLAB/gen_golden_all.m to produce find_real_hr.mat")
    snap = load_golden(mat_path)
    ref = snap["ref_data"]
    times = np.atleast_1d(snap["case_time_currents"]).ravel()
    expected = np.atleast_1d(snap["expected_hr_real"]).ravel()
    actual = np.array([find_real_hr("dummy", t, ref) for t in times])
    assert_array_close(actual, expected, atol=1e-9, rtol=1e-9)
