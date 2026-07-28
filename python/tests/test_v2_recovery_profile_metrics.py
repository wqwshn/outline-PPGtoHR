from __future__ import annotations

import numpy as np
import pytest

from ppg_hr.v2.recovery_profile_metrics import evaluate_recovery_profile_metrics
from ppg_hr.v2.solver import V2SolverResult


def _solver_result(
    *,
    final_bpm: list[float],
    reset_bpm: list[float] | None = None,
    centers_s: list[float] | None = None,
    motion: list[bool] | None = None,
) -> V2SolverResult:
    row_count = len(final_bpm)
    centers = np.asarray(
        centers_s if centers_s is not None else list(range(row_count)),
        dtype=float,
    )
    reset = np.asarray(reset_bpm if reset_bpm is not None else final_bpm, dtype=float)
    hr = np.column_stack(
        [
            centers,
            np.full(row_count, 999.0),
            reset,
            np.asarray(final_bpm, dtype=float),
            np.asarray(
                motion if motion is not None else [True] * row_count,
                dtype=float,
            ),
            np.ones(row_count),
        ]
    )
    return V2SolverResult(
        HR=hr,
        err_stats={},
        metadata={
            "analysis_scope": "full",
            "adaptive_filter": "lms",
            "reference_groups_order": ["HF"],
            "time_bias": 5.0,
        },
        window_table=[
            {
                "window_idx": idx,
                "center_s": float(center_s),
                "reliable": True,
            }
            for idx, center_s in enumerate(centers)
        ],
    )


def test_recovery_profile_metrics_report_long_error_runs_and_censoring() -> None:
    result = _solver_result(
        final_bpm=[
            100.0,
            110.0,
            111.0,
            109.0,
            120.0,
            121.0,
            100.0,
            100.0,
            100.0,
            115.0,
            100.0,
            100.0,
        ],
        reset_bpm=[102.0] * 12,
    )
    reference = np.column_stack(
        [
            np.arange(0.0, 30.0),
            np.full(30, 100.0),
        ]
    )

    metrics = evaluate_recovery_profile_metrics(
        result,
        ref_data=reference,
        method_names=("reset FFT", "LMS+H"),
    )

    assert metrics.metric_contract_version == "lyx_recovery_profile_metric_v1"
    assert metrics.time_bias_s == 5.0
    assert metrics.base_motion_window_count == 12
    assert metrics.final_motion_mae_bpm == pytest.approx(43.0 / 6.0)
    assert metrics.reset_motion_mae_bpm == 2.0
    assert metrics.e10_window_count == 5
    assert metrics.e20_window_count == 2
    assert metrics.longest_e10_run_windows == 2
    assert metrics.longest_e20_run_windows == 2
    assert metrics.recovery_episode_count == 3
    assert metrics.right_censored_recovery_count == 1
    assert metrics.max_recovered_delay_s == 5.0
    assert metrics.recovered_delay_s == (5.0, 2.0)
    assert metrics.right_censored_recovery == (False, False, True)
    assert metrics.physiological_rise_episode_count == 0
    assert metrics.rise_underestimate_bpm == ()
    assert len(metrics.base_motion_window_sha256) == 64


def test_recovery_profile_metrics_break_error_runs_at_time_gaps() -> None:
    result = _solver_result(
        final_bpm=[
            100.0,
            100.0,
            100.0,
            100.0,
            120.0,
            120.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ],
        centers_s=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    reference = np.column_stack(
        [
            np.arange(0.0, 30.0),
            np.full(30, 100.0),
        ]
    )

    metrics = evaluate_recovery_profile_metrics(
        result,
        ref_data=reference,
        method_names=("reset FFT", "LMS+H"),
    )

    assert metrics.longest_e10_run_windows == 1
    assert metrics.longest_e20_run_windows == 1
    assert metrics.recovery_episode_count == 2


def test_recovery_profile_metrics_do_not_recover_across_time_gaps() -> None:
    result = _solver_result(
        final_bpm=[
            100.0,
            111.0,
            112.0,
            111.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
            100.0,
        ],
        centers_s=[0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
    )
    ref_data = np.column_stack(
        [np.arange(0.0, 30.0), np.full(30, 100.0)]
    )

    metrics = evaluate_recovery_profile_metrics(
        result,
        ref_data=ref_data,
        method_names=("reset FFT", "LMS+H"),
    )

    assert metrics.recovery_episode_count == 1
    assert metrics.right_censored_recovery_count == 1
    assert metrics.max_recovered_delay_s is None


def test_recovery_profile_metrics_break_runs_at_missing_motion_windows() -> None:
    result = _solver_result(
        final_bpm=[111.0] * 20,
        motion=[index % 2 == 0 for index in range(20)],
    )
    ref_data = np.column_stack(
        [np.arange(0.0, 40.0), np.full(40, 100.0)]
    )

    metrics = evaluate_recovery_profile_metrics(
        result,
        ref_data=ref_data,
        method_names=("reset FFT", "LMS+H"),
    )

    assert metrics.base_motion_window_count == 10
    assert metrics.longest_e10_run_windows == 1


def test_recovery_profile_metrics_report_physiological_rise_underestimate() -> None:
    row_count = 16
    centers = np.arange(row_count, dtype=float)
    aligned_reference = 100.0 + centers
    result = _solver_result(
        final_bpm=list(aligned_reference - 4.0),
        centers_s=list(centers),
    )
    reference_times = np.arange(0.0, 30.0)
    reference = np.column_stack(
        [
            reference_times,
            95.0 + reference_times,
        ]
    )

    metrics = evaluate_recovery_profile_metrics(
        result,
        ref_data=reference,
        method_names=("reset FFT", "LMS+H"),
    )

    assert metrics.physiological_rise_episode_count == 1
    assert metrics.max_rise_underestimate_bpm == pytest.approx(4.0)
    assert metrics.rise_underestimate_bpm == pytest.approx((4.0,))
