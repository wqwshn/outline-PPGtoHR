from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.post_motion_reset_fft_reacquire import load_lite_report_config
from ppg_hr.v2.prior_invalidation_experiment import (
    build_prior_invalidation_configs,
    evaluate_report,
)
from ppg_hr.v2.reference_overlap import aligned_reference_bpm


def _hb_report(sample: str) -> Path | None:
    relative = Path(
        "data/202607-multiperson/0711-HB/v2_batch_outputs/"
        "20260715_dual_reset_n5_hb24_lite_1x40/json/"
        f"{sample}_HB_0711-green-raw_bandpass-lms-full-HF-v2.json"
    )
    cwd = Path.cwd()
    for root in (cwd, cwd.parent.parent):
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def test_prior_invalidation_candidate_changes_only_experimental_handoff_knobs() -> None:
    report = _hb_report("run2")
    if report is None:
        pytest.skip("HB run2 N5 report is not available")
    base = load_lite_report_config(json.loads(report.read_text(encoding="utf-8")))

    current, candidate = build_prior_invalidation_configs(base)

    assert current.post_motion_dual_reset_prior_invalidation_enable is False
    assert candidate.post_motion_dual_reset_prior_invalidation_enable is True
    assert candidate.post_motion_dual_reset_prior_invalidation_hits == 3
    assert candidate.post_motion_dual_reset_prior_invalidation_gap_bpm == 40.0
    assert candidate.post_motion_dual_reset_prior_invalidation_raw_decline_bpm == 0.5
    assert (
        candidate.post_motion_dual_reset_prior_invalidation_prior_decline_bpm_per_window
        == 0.5
    )
    assert current.data_path == candidate.data_path
    assert current.time_bias == candidate.time_bias


def test_aligned_reference_uses_the_same_positive_time_bias_as_mae() -> None:
    hr = np.asarray(
        [
            [0.0, 100.0, 0.0, 0.0],
            [1.0, 90.0, 0.0, 0.0],
            [2.0, 80.0, 0.0, 0.0],
        ]
    )

    aligned = aligned_reference_bpm(hr, time_bias=1.0)

    assert aligned[:2].tolist() == [90.0, 80.0]
    assert np.isnan(aligned[2])

    reference_extends_past_algorithm_windows = aligned_reference_bpm(
        hr,
        time_bias=1.0,
        reference_bounds=(0.0, 3.0),
    )
    assert reference_extends_past_algorithm_windows.tolist() == [90.0, 80.0, 70.0]


def test_run2_real_replay_invalidates_prior_earlier_without_touching_independent_fft() -> None:
    report = _hb_report("run2")
    if report is None:
        pytest.skip("HB run2 N5 report is not available")

    row = evaluate_report(report)

    assert row["prior_invalidation_event_count"] == 1
    assert row["new_first_handoff_center_s"] < row["current_first_handoff_center_s"]
    assert row["new_post60_mae_bpm"] < row["current_post60_mae_bpm"]
    assert row["independent_reset_invariance_pass"] is True


def test_xiezi2_rising_low_peak_does_not_invalidate_declining_prior() -> None:
    report = _hb_report("xiezi2")
    if report is None:
        pytest.skip("HB xiezi2 N5 report is not available")

    row = evaluate_report(report)

    assert row["prior_invalidation_event_count"] == 0
    assert row["new_post60_mae_bpm"] == pytest.approx(
        row["current_post60_mae_bpm"]
    )
    assert row["independent_reset_invariance_pass"] is True
