from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.params import SolverParams
from research.rest_algri_optim.scripts.rest_tracking_core import (
    EvaluationResult,
    SearchConfig,
    SegmentMetrics,
    assign_rest_segments,
    compute_segment_metrics,
    discover_cases,
    evaluate_arrays,
    objective_from_metrics,
    run_case_search,
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


def _make_tiny_raw(
    n_sec: int = 70,
    fs: int = 100,
    hr_hz: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    n = n_sec * fs
    t = np.arange(n) / fs
    ppg = np.sin(2 * np.pi * hr_hz * t)
    motion = np.zeros(n)
    motion_mask = (t >= 25.0) & (t <= 40.0)
    motion[motion_mask] = 1.5 * np.sin(2 * np.pi * 2.0 * t[motion_mask])
    raw = np.zeros((n, 11), dtype=float)
    raw[:, 5] = ppg
    raw[:, 6] = ppg
    raw[:, 7] = ppg
    raw[:, 3] = 0.1 * motion
    raw[:, 4] = 0.1 * motion
    raw[:, 8] = motion
    raw[:, 9] = motion
    raw[:, 10] = motion
    ref = np.column_stack([np.arange(n_sec, dtype=float), np.full(n_sec, hr_hz * 60.0)])
    return raw, ref


def test_evaluate_arrays_returns_metrics_and_curve() -> None:
    raw, ref = _make_tiny_raw()
    params = SolverParams(
        fs_target=100,
        calib_time=10.0,
        time_buffer=2.0,
        smooth_win_len=3,
        time_bias=0.0,
    )

    result = evaluate_arrays(
        raw_data=raw,
        ref_data=ref,
        base_params=params,
        mode="current",
        hr_range_rest_bpm=30.0,
        slew_limit_rest_bpm=6.0,
        slew_step_rest_bpm=4.0,
        smooth_win_len=3,
        time_bias_s=0.0,
    )

    assert isinstance(result, EvaluationResult)
    assert result.curve.shape[0] == result.solver_result.HR.shape[0]
    assert {
        "t_pred_s",
        "ref_bpm",
        "final_bpm",
        "pure_fft_bpm",
        "motion_flag",
        "segment",
    } <= set(result.curve.dtype.names)
    assert np.isfinite(result.objective)


def test_time_bias_changes_prediction_time() -> None:
    raw, ref = _make_tiny_raw()
    params = SolverParams(fs_target=100, calib_time=10.0, time_buffer=2.0)

    a = evaluate_arrays(
        raw_data=raw,
        ref_data=ref,
        base_params=params,
        mode="current",
        hr_range_rest_bpm=30.0,
        slew_limit_rest_bpm=6.0,
        slew_step_rest_bpm=4.0,
        smooth_win_len=3,
        time_bias_s=0.0,
    )
    b = evaluate_arrays(
        raw_data=raw,
        ref_data=ref,
        base_params=params,
        mode="current",
        hr_range_rest_bpm=30.0,
        slew_limit_rest_bpm=6.0,
        slew_step_rest_bpm=4.0,
        smooth_win_len=3,
        time_bias_s=2.0,
    )

    np.testing.assert_allclose(b.solver_result.T_Pred - a.solver_result.T_Pred, 2.0)


def test_discover_cases_pairs_ref_suffixes(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("sensor\n", encoding="utf-8")
    (tmp_path / "a_ref.csv").write_text("ref\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("sensor\n", encoding="utf-8")
    (tmp_path / "b_HR_ref.csv").write_text("ref\n", encoding="utf-8")
    (tmp_path / "b_ref.csv").write_text("older ref\n", encoding="utf-8")

    cases = discover_cases(tmp_path)

    assert [case.name for case in cases] == ["a", "b"]
    assert cases[0].ref_path.name == "a_ref.csv"
    assert cases[1].ref_path.name == "b_HR_ref.csv"


def test_search_config_default_modes_are_unified_mechanisms() -> None:
    cfg = SearchConfig(max_trials=4, random_state=7)

    assert cfg.modes == (
        "current",
        "fallback_slew_to_raw_peak",
        "all_peaks_near_prev",
        "all_peaks_with_raw_fallback",
    )
    assert cfg.max_trials == 4
    assert cfg.random_state == 7


def test_run_case_search_with_preloaded_arrays() -> None:
    raw, ref = _make_tiny_raw()
    params = SolverParams(fs_target=100, calib_time=10.0, time_buffer=2.0)
    cfg = SearchConfig(
        max_trials=3,
        random_state=1,
        modes=("current",),
        hr_range_rest_bpm=(20.0, 30.0),
        slew_limit_rest_bpm=(4.0, 6.0),
        slew_step_rest_bpm=(2.0, 4.0),
        smooth_win_len=(3,),
        time_bias_s=(0.0, 1.0),
    )

    result = run_case_search(
        case_name="synthetic",
        raw_data=raw,
        ref_data=ref,
        base_params=params,
        config=cfg,
    )

    assert result.case_name == "synthetic"
    assert result.best is not None
    assert len(result.trials) == 3
    assert result.best.objective == min(trial.objective for trial in result.trials)
