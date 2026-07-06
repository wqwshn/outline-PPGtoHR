from __future__ import annotations

import numpy as np
import pytest

from ppg_hr.v2.post_motion_dynamic_guard_policy import (
    DynamicGuardConfig,
    rank_dynamic_guard_candidates,
    switch_mask_and_events,
    transition_is_reachable,
)


def _source(
    *,
    times: list[float],
    adaptive: list[float],
    fft: list[float],
    ref: list[float] | None = None,
) -> np.ndarray:
    arr = np.zeros((len(times), 9), dtype=float)
    arr[:, 0] = np.asarray(times, dtype=float)
    arr[:, 1] = np.asarray(ref if ref is not None else adaptive, dtype=float) / 60.0
    arr[:, 2] = np.asarray(adaptive, dtype=float) / 60.0
    arr[:, 4] = np.asarray(fft, dtype=float) / 60.0
    return arr


def test_transition_is_reachable_uses_recovery_step_limits() -> None:
    cfg = DynamicGuardConfig(
        name="lite_recovery",
        recovery_step_up_bpm=1.5,
        recovery_step_down_bpm=3.0,
    )

    assert transition_is_reachable(100.0, 97.0, cfg)
    assert transition_is_reachable(100.0, 101.5, cfg)
    assert not transition_is_reachable(100.0, 96.5, cfg)
    assert not transition_is_reachable(100.0, 102.0, cfg)


def test_stable_crossover_switches_after_reachable_windows() -> None:
    source = _source(
        times=[98, 99, 100, 101, 102, 103, 104, 105],
        adaptive=[120, 115, 110, 104, 101, 98, 95, 92],
        fft=[118, 112, 108, 103, 100, 97, 94, 91],
    )
    cfg = DynamicGuardConfig(
        name="stable",
        min_elapsed_s=1.0,
        stable_windows=3,
        crossover_gap_bpm=3.0,
        recovery_step_up_bpm=1.5,
        recovery_step_down_bpm=3.0,
    )

    mask, events = switch_mask_and_events(
        source,
        motion_segment={"start_s": 80.0, "end_s": 100.0},
        config=cfg,
    )

    assert mask.tolist() == [True, True, True, True, True, True, False, False]
    assert len(events) == 1
    assert events[0].switch_reason == "stable_crossover"
    assert events[0].center_s == pytest.approx(104.0)
    assert events[0].reachable is True


def test_adaptive_rising_rescue_switches_before_crossover() -> None:
    source = _source(
        times=[98, 99, 100, 101, 102, 103, 104],
        adaptive=[118, 120, 122, 125, 128, 131, 134],
        fft=[116, 112, 108, 103, 102, 101, 100],
    )
    cfg = DynamicGuardConfig(
        name="rescue",
        min_elapsed_s=1.0,
        rising_windows=3,
        rising_slope_bpm_per_window=1.5,
        rescue_gap_bpm=20.0,
        fft_floor_bpm=55.0,
    )

    mask, events = switch_mask_and_events(
        source,
        motion_segment={"start_s": 80.0, "end_s": 100.0},
        config=cfg,
    )

    assert mask.tolist() == [True, True, True, True, True, False, False]
    assert events[0].switch_reason == "adaptive_rising_rescue"
    assert events[0].center_s == pytest.approx(103.0)


def test_gap_rescue_switches_when_adaptive_falls_but_stays_high() -> None:
    source = _source(
        times=[98, 99, 100, 101, 102, 103, 104, 105, 106],
        adaptive=[130, 128, 126, 124, 121, 118, 115, 112, 109],
        fft=[118, 112, 108, 90, 88, 86, 84, 82, 80],
    )
    cfg = DynamicGuardConfig(
        name="gap_rescue",
        min_elapsed_s=1.0,
        crossover_gap_bpm=2.0,
        recovery_step_down_bpm=3.0,
        rescue_gap_bpm=20.0,
        gap_rescue_windows=4,
        gap_rescue_min_hits=3,
        gap_rescue_fft_stable_windows=3,
        gap_rescue_fft_stable_bpm=6.0,
        fft_floor_bpm=55.0,
    )

    mask, events = switch_mask_and_events(
        source,
        motion_segment={"start_s": 80.0, "end_s": 100.0},
        config=cfg,
    )

    assert mask.tolist() == [True, True, True, True, True, True, False, False, False]
    assert len(events) == 1
    assert events[0].switch_reason == "gap_rescue"
    assert events[0].center_s == pytest.approx(104.0)
    assert events[0].reachable is False
    assert events[0].hard_switch is True
    assert events[0].gap_rescue_count >= 3
    assert events[0].fft_stable_count >= 3


def test_gap_rescue_rejects_unstable_reset_fft() -> None:
    source = _source(
        times=[98, 99, 100, 101, 102, 103, 104, 105, 106],
        adaptive=[130, 128, 126, 124, 121, 118, 115, 112, 109],
        fft=[118, 112, 108, 90, 70, 92, 68, 94, 66],
    )
    cfg = DynamicGuardConfig(
        name="gap_rescue",
        min_elapsed_s=1.0,
        rescue_gap_bpm=20.0,
        gap_rescue_windows=4,
        gap_rescue_min_hits=3,
        gap_rescue_fft_stable_windows=3,
        gap_rescue_fft_stable_bpm=6.0,
        fft_floor_bpm=55.0,
    )

    mask, events = switch_mask_and_events(
        source,
        motion_segment={"start_s": 80.0, "end_s": 100.0},
        config=cfg,
    )

    assert mask.tolist() == [True, True, True, True, True, True, True, True, True]
    assert events == []


def test_low_fft_blocks_switch_even_when_gap_matches() -> None:
    source = _source(
        times=[100, 101, 102, 103, 104, 105],
        adaptive=[70, 68, 66, 64, 62, 60],
        fft=[50, 50, 50, 50, 50, 50],
    )
    cfg = DynamicGuardConfig(
        name="lowlock",
        min_elapsed_s=0.0,
        stable_windows=2,
        crossover_gap_bpm=20.0,
        fft_floor_bpm=55.0,
    )

    mask, events = switch_mask_and_events(
        source,
        motion_segment={"start_s": 80.0, "end_s": 100.0},
        config=cfg,
    )

    assert mask.tolist() == [True, True, True, True, True, True]
    assert events == []


def test_rank_dynamic_guard_candidates_prefers_non_regression_before_rescue_gain() -> None:
    rows = [
        {
            "candidate_name": "rescues_but_regresses",
            "sample_id": "multi_bobi1_0613",
            "delta_vs_lite_60s_mae_bpm": 2.5,
            "delta_vs_lite_post_mae_bpm": 2.5,
            "old_lite_post_motion_mae_bpm": 2.0,
            "dynamic_reachable_failure_count": 0,
            "low_lock_window_count": 0,
            "missing_switch_reason_count": 0,
        },
        {
            "candidate_name": "balanced",
            "sample_id": "multi_bobi1_0613",
            "delta_vs_lite_60s_mae_bpm": 0.2,
            "delta_vs_lite_post_mae_bpm": 0.2,
            "old_lite_post_motion_mae_bpm": 2.0,
            "dynamic_reachable_failure_count": 0,
            "low_lock_window_count": 0,
            "missing_switch_reason_count": 0,
        },
        {
            "candidate_name": "balanced",
            "sample_id": "multi_fuwo1_0613",
            "delta_vs_lite_60s_mae_bpm": -15.0,
            "delta_vs_lite_post_mae_bpm": -15.0,
            "old_lite_post_motion_mae_bpm": 42.0,
            "dynamic_reachable_failure_count": 0,
            "low_lock_window_count": 0,
            "missing_switch_reason_count": 0,
        },
    ]

    ranked = rank_dynamic_guard_candidates(rows)

    assert ranked[0]["candidate_name"] == "balanced"
    assert ranked[0]["selection_tier"] in {
        "promoted_candidate",
        "best_effort_candidate",
    }
