"""Tests for ``find_near_biggest``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.core import find_near_biggest
from ppg_hr.io.golden import load_golden


def test_first_in_range_wins() -> None:
    fre = np.array([1.5, 1.6, 2.0, 2.5, 3.0])
    hr, idx = find_near_biggest(fre, hr_prev=1.55, range_plus=0.4, range_minus=-0.3)
    # Δ=−0.05 in (−0.3, 0.4) → 1.5 wins (1-based whichPeak=1)
    assert hr == 1.5 and idx == 1


def test_returns_hr_prev_when_no_match() -> None:
    fre = np.array([5.0, 6.0, 7.0])
    hr, idx = find_near_biggest(fre, hr_prev=1.5, range_plus=0.2, range_minus=-0.2)
    assert hr == 1.5 and idx == 0


def test_only_first_five_considered() -> None:
    fre = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 1.5])  # 6th is in range, ignored
    hr, idx = find_near_biggest(fre, hr_prev=1.5, range_plus=0.5, range_minus=-0.5)
    assert hr == 1.5 and idx == 0


def test_strict_open_interval() -> None:
    fre = np.array([2.0])
    # Δ = 0.5, range_plus=0.5 → strict <, should NOT match
    hr, idx = find_near_biggest(fre, hr_prev=1.5, range_plus=0.5, range_minus=-0.5)
    assert idx == 0 and hr == 1.5


def test_matches_golden(golden_dir: Path) -> None:
    mat_path = golden_dir / "find_near_biggest.mat"
    if not mat_path.is_file():
        pytest.skip("Run MATLAB/gen_golden_all.m to produce find_near_biggest.mat")
    snap = load_golden(mat_path)
    hr, idx = find_near_biggest(
        snap["case_fre"],
        float(snap["case_hr_prev"]),
        float(snap["case_range_plus"]),
        float(snap["case_range_minus"]),
    )
    assert hr == pytest.approx(float(snap["expected_hr"]))
    assert idx == int(snap["expected_which"])
