from __future__ import annotations

import numpy as np
import pytest

from research.rest_algri_optim.scripts.rest_tracking_core import (
    SegmentMetrics,
    assign_rest_segments,
    compute_segment_metrics,
    objective_from_metrics,
    select_tracked_frequency,
)


def test_assign_rest_segments_uses_longest_motion_run() -> None:
    hr = np.zeros((12, 9), dtype=float)
    hr[:, 0] = np.arange(12, dtype=float)
    hr[2:4, 7] = 1.0
    hr[6:9, 7] = 1.0
    hr[:, 8] = hr[:, 7]

    labels = assign_rest_segments(hr)

    assert labels.tolist() == [
        "pre_rest",
        "pre_rest",
        "other_motion",
        "other_motion",
        "pre_rest",
        "pre_rest",
        "motion",
        "motion",
        "motion",
        "post_rest",
        "post_rest",
        "post_rest",
    ]


def test_compute_segment_metrics_uses_reliable_rest_windows() -> None:
    hr = np.zeros((8, 9), dtype=float)
    hr[:, 0] = np.arange(8, dtype=float)
    hr[:, 1] = 1.0
    hr[:, 4] = np.array([1.00, 1.02, 1.00, 1.50, 1.60, 1.00, 0.98, 1.02])
    hr[:, 5] = np.array([1.00, 1.01, 1.00, 1.50, 1.60, 1.01, 0.99, 1.01])
    hr[3:5, 7] = 1.0
    hr[:, 8] = hr[:, 7]
    ref_bpm = np.full(8, 60.0)
    reliable = np.array([True, True, False, True, True, True, True, True])

    metrics = compute_segment_metrics(
        hr=hr,
        ref_bpm=ref_bpm,
        reliable_mask=reliable,
        final_col=5,
        pure_fft_col=4,
    )

    assert metrics.rest_all_mae == pytest.approx(0.48)
    assert metrics.pre_rest_mae == pytest.approx(0.3)
    assert metrics.post_rest_mae == pytest.approx(0.6)
    assert metrics.passed(threshold=1.5)


def test_objective_is_worst_rest_subsegment() -> None:
    metrics = SegmentMetrics(
        rest_all_mae=1.0,
        pre_rest_mae=1.4,
        post_rest_mae=2.2,
        pure_fft_rest_all_mae=3.0,
        pure_fft_pre_rest_mae=2.0,
        pure_fft_post_rest_mae=4.0,
        n_rest_all=10,
        n_pre_rest=4,
        n_post_rest=6,
    )

    assert objective_from_metrics(metrics) == pytest.approx(2.2)
    assert not metrics.passed(threshold=1.5)


def test_current_mode_keeps_previous_when_top_five_have_no_near_peak() -> None:
    freqs = np.array([1.8, 1.9, 2.0, 2.1, 2.2, 1.18])
    amps = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])

    out = select_tracked_frequency(
        freqs=freqs,
        amps=amps,
        prev_hr=1.2,
        mode="current",
        range_hz=0.05,
        limit_bpm=6.0,
        step_bpm=4.0,
    )

    assert out == pytest.approx(1.2)


def test_all_peaks_near_prev_can_use_sixth_peak() -> None:
    freqs = np.array([1.8, 1.9, 2.0, 2.1, 2.2, 1.18])
    amps = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])

    out = select_tracked_frequency(
        freqs=freqs,
        amps=amps,
        prev_hr=1.2,
        mode="all_peaks_near_prev",
        range_hz=0.05,
        limit_bpm=6.0,
        step_bpm=4.0,
    )

    assert out == pytest.approx(1.18)


def test_raw_peak_fallback_moves_by_existing_slew_step() -> None:
    freqs = np.array([1.55, 1.9, 2.0])
    amps = np.array([10.0, 4.0, 3.0])

    out = select_tracked_frequency(
        freqs=freqs,
        amps=amps,
        prev_hr=1.2,
        mode="fallback_slew_to_raw_peak",
        range_hz=0.05,
        limit_bpm=6.0,
        step_bpm=4.0,
    )

    assert out == pytest.approx(1.2 + 4.0 / 60.0)


def test_combined_mode_uses_all_peaks_before_raw_fallback() -> None:
    freqs = np.array([1.55, 1.9, 1.22])
    amps = np.array([10.0, 4.0, 3.0])

    out = select_tracked_frequency(
        freqs=freqs,
        amps=amps,
        prev_hr=1.2,
        mode="all_peaks_with_raw_fallback",
        range_hz=0.05,
        limit_bpm=6.0,
        step_bpm=4.0,
    )

    assert out == pytest.approx(1.22)
