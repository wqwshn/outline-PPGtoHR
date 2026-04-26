"""Tests for ``fft_peaks``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.core import fft_peaks
from ppg_hr.io.golden import assert_array_close, load_golden


def test_empty_signal() -> None:
    fre, amp = fft_peaks(np.array([]), 100.0, 0.3)
    assert fre.size == 0 and amp.size == 0


def test_pure_tone_inside_band() -> None:
    fs = 100.0
    t = np.arange(0, 8, 1.0 / fs)
    sig = np.sin(2 * np.pi * 2.0 * t)  # 2 Hz tone, inside 1–4 Hz band
    fre, amp = fft_peaks(sig, fs, 0.3)
    assert fre.size >= 1
    assert any(abs(f - 2.0) < 0.05 for f in fre)


def test_sub_1hz_cardiac_tone_inside_band() -> None:
    fs = 100.0
    t = np.arange(0, 12, 1.0 / fs)
    sig = np.sin(2 * np.pi * 0.8 * t)  # 48 BPM, below the old 1 Hz cutoff.
    fre, amp = fft_peaks(sig, fs, 0.3)
    assert fre.size >= 1
    assert amp.size == fre.size
    assert any(abs(f - 0.8) < 0.05 for f in fre)


def test_high_freq_outside_band_filtered() -> None:
    fs = 100.0
    t = np.arange(0, 8, 1.0 / fs)
    sig = np.sin(2 * np.pi * 8.0 * t)  # 8 Hz, above 4 Hz upper bound
    fre, _ = fft_peaks(sig, fs, 0.3)
    # All returned peaks must lie strictly inside the 0.7-4 Hz cardiac band.
    assert fre.size == 0 or np.all((fre > 0.7) & (fre < 4.0))


def test_threshold_filters_small_peaks() -> None:
    fs = 100.0
    t = np.arange(0, 8, 1.0 / fs)
    big = np.sin(2 * np.pi * 2.0 * t)
    small = 0.05 * np.sin(2 * np.pi * 3.0 * t)
    fre, _ = fft_peaks(big + small, fs, percent=0.5)
    # only the big peak should pass the 0.5 * max threshold
    assert all(abs(f - 2.0) < 0.05 for f in fre)


def test_matches_golden(golden_dir: Path) -> None:
    mat_path = golden_dir / "fft_peaks.mat"
    if not mat_path.is_file():
        pytest.skip("Run MATLAB/gen_golden_all.m to produce fft_peaks.mat")
    snap = load_golden(mat_path)
    fre, amp = fft_peaks(
        np.asarray(snap["case_signal"]).ravel(),
        float(snap["case_fs"]),
        float(snap["case_percent"]),
    )
    assert_array_close(fre, snap["expected_fre"], atol=1e-9, rtol=1e-9)
    assert_array_close(amp, snap["expected_famp"], atol=1e-9, rtol=1e-9)
