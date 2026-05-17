from __future__ import annotations

import numpy as np
import pytest

from research.rest_algri_optim.scripts.rest_tracking_core import (
    SegmentMetrics,
    assign_rest_segments,
    compute_segment_metrics,
    objective_from_metrics,
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
