"""Tests for ``find_maxpeak``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.core import find_maxpeak
from ppg_hr.io.golden import assert_array_close, load_golden


def test_empty_input_returns_empty() -> None:
    out = find_maxpeak(np.array([]), None, np.array([]))
    assert out.size == 0


def test_descending_amplitude_order() -> None:
    freqs = np.array([1.0, 2.0, 3.0, 4.0])
    amps = np.array([0.1, 0.5, 0.3, 0.8])
    out = find_maxpeak(freqs, freqs, amps)
    np.testing.assert_array_equal(out, [4.0, 2.0, 3.0, 1.0])


def test_single_peak() -> None:
    out = find_maxpeak(np.array([2.5]), None, np.array([0.7]))
    np.testing.assert_array_equal(out, [2.5])


def test_handles_column_vector() -> None:
    freqs = np.array([[1.0], [2.0], [3.0]])
    amps = np.array([[0.3], [0.1], [0.9]])
    out = find_maxpeak(freqs, freqs, amps)
    np.testing.assert_array_equal(out, [3.0, 1.0, 2.0])


def test_matches_golden(golden_dir: Path) -> None:
    mat_path = golden_dir / "find_maxpeak.mat"
    if not mat_path.is_file():
        pytest.skip("Run MATLAB/gen_golden_all.m to produce find_maxpeak.mat")
    snap = load_golden(mat_path)
    out = find_maxpeak(snap["case_freqs"], None, snap["case_amps"])
    assert_array_close(out, snap["expected_sorted"], atol=1e-12, rtol=1e-12)
