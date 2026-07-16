from __future__ import annotations

import numpy as np

from ppg_hr.v2.post_motion_dual_reset_experiment import (
    OfflineScoreWindow,
    ReplayEvidenceWindow,
    SampleReplay,
)
from ppg_hr.v2.post_motion_dual_reset_runtime import (
    DualResetRuntimeWindow,
    FrozenDualResetConfig,
    apply_frozen_dual_reset,
)
from ppg_hr.v2.post_motion_observability_experiment import (
    ObservabilityCandidate,
    build_predeclared_candidates,
    evaluate_candidate_matrix,
    evaluate_observability_candidate,
)
from ppg_hr.v2.ppg_observability import measure_ppg_observability
from ppg_hr.v2.raw_fft_candidates import RawFftCandidateFrame, extract_raw_fft_candidates


def _frame(*peaks: tuple[float, float]) -> RawFftCandidateFrame:
    frequencies = np.asarray([bpm / 60.0 for bpm, _ in peaks], dtype=float)
    amplitudes = np.asarray([amplitude for _, amplitude in peaks], dtype=float)
    indices = np.arange(len(peaks), dtype=int)
    return RawFftCandidateFrame(frequencies, amplitudes, indices, indices)


def _windows() -> tuple[DualResetRuntimeWindow, ...]:
    return tuple(
        DualResetRuntimeWindow(
            window_idx=index,
            center_s=11.0 + index,
            reliable=True,
            archived_final_bpm=140.0 - index,
            archived_final_history=(146.0, 144.0, 142.0),
            candidates=_frame((136.0 - index, 1.0), (58.0, 0.45)),
        )
        for index in range(5)
    )


def _observability_windows() -> tuple[DualResetRuntimeWindow, ...]:
    evidence = (
        (7.0, 0.75, 1.8, 136.0),
        (10.0, 0.20, 1.1, 80.0),
        (11.0, 0.72, 1.7, 134.0),
        (12.0, 0.74, 1.8, 133.0),
    )
    return tuple(
        DualResetRuntimeWindow(
            window_idx=index,
            start_s=start_s,
            center_s=start_s + 4.0,
            reliable=True,
            archived_final_bpm=140.0 - index,
            archived_final_history=(146.0, 144.0, 142.0),
            candidates=_frame((top_bpm, 1.0), (58.0, 0.4)),
            periodicity=periodicity,
            peak_competition=competition,
        )
        for index, (start_s, periodicity, competition, top_bpm) in enumerate(evidence)
    )


def test_explicit_a0_replay_is_identical_to_the_frozen_legacy_runtime() -> None:
    windows = _windows()
    baseline = np.asarray([140.0, 139.0, 138.0, 137.0, 136.0])

    legacy = apply_frozen_dual_reset(
        windows,
        motion_end_s=10.0,
        baseline_final_bpm=baseline,
    )
    explicit_a0 = apply_frozen_dual_reset(
        windows,
        motion_end_s=10.0,
        baseline_final_bpm=baseline,
        config=FrozenDualResetConfig(experiment_mode="a0"),
    )

    np.testing.assert_array_equal(explicit_a0.final_bpm, legacy.final_bpm)
    assert explicit_a0.window_rows == legacy.window_rows
    assert explicit_a0.metadata["experiment_mode"] == "a0"


def test_a1_requires_a_complete_post_motion_window_and_continuous_ppg_evidence() -> None:
    windows = _observability_windows()
    baseline = np.asarray([140.0, 139.0, 138.0, 137.0])

    a0 = apply_frozen_dual_reset(
        windows,
        motion_end_s=10.0,
        baseline_final_bpm=baseline,
        config=FrozenDualResetConfig(experiment_mode="a0"),
    )
    a1 = apply_frozen_dual_reset(
        windows,
        motion_end_s=10.0,
        baseline_final_bpm=baseline,
        config=FrozenDualResetConfig(
            experiment_mode="a1",
            observability_periodicity_min=0.5,
            observability_peak_competition_min=1.3,
            observability_recovery_hits=2,
        ),
    )

    assert [row["observability_state"] for row in a1.window_rows] == [
        "unobservable",
        "unobservable",
        "recovering",
        "recovered",
    ]
    assert [row["window_fully_post_motion"] for row in a1.window_rows] == [
        False,
        True,
        True,
        True,
    ]
    assert all(
        not row["switch_target_ready"] for row in a1.window_rows[:3]
    )
    assert [row["independent_reset_bpm"] for row in a1.window_rows] == [
        row["independent_reset_bpm"] for row in a0.window_rows
    ]


def test_single_dominant_peak_is_valid_observability_evidence() -> None:
    windows = tuple(
        DualResetRuntimeWindow(
            window_idx=index,
            start_s=10.0 + index,
            center_s=14.0 + index,
            reliable=True,
            archived_final_bpm=140.0,
            archived_final_history=(146.0, 144.0, 142.0),
            candidates=_frame((136.0 - index, 1.0)),
            periodicity=0.75,
            peak_competition=float("inf"),
        )
        for index in range(2)
    )

    result = apply_frozen_dual_reset(
        windows,
        motion_end_s=10.0,
        baseline_final_bpm=np.asarray([140.0, 140.0]),
        config=FrozenDualResetConfig(
            experiment_mode="a1",
            observability_recovery_hits=2,
        ),
    )

    assert [row["observability_state"] for row in result.window_rows] == [
        "recovering",
        "recovered",
    ]


def test_raw_ppg_observability_separates_periodic_pulse_from_broadband_noise() -> None:
    fs = 100.0
    time_s = np.arange(800, dtype=float) / fs
    pulse = np.sin(2.0 * np.pi * 2.0 * time_s)
    noise = np.random.default_rng(42).normal(size=time_s.size)

    pulse_evidence = measure_ppg_observability(
        pulse,
        fs,
        extract_raw_fft_candidates(pulse, fs),
    )
    noise_evidence = measure_ppg_observability(
        noise,
        fs,
        extract_raw_fft_candidates(noise, fs),
    )

    assert pulse_evidence.periodicity > 0.8
    assert noise_evidence.periodicity < 0.35
    assert pulse_evidence.periodicity > noise_evidence.periodicity
    assert set(pulse_evidence.__dict__) == {"periodicity", "peak_competition"}


def test_a2_reinitialises_handoff_once_and_revokes_ready_after_quality_loss() -> None:
    quality = (False, True, True, True, True, False, True, True)
    windows = tuple(
        DualResetRuntimeWindow(
            window_idx=index,
            start_s=10.0 + index,
            center_s=14.0 + index,
            reliable=True,
            archived_final_bpm=145.0 - index,
            archived_final_history=(150.0, 148.0, 146.0),
            candidates=_frame((136.0 - min(index, 4), 1.0), (58.0, 0.4)),
            periodicity=0.75 if good else 0.2,
            peak_competition=1.8 if good else 1.1,
        )
        for index, good in enumerate(quality)
    )
    result = apply_frozen_dual_reset(
        windows,
        motion_end_s=10.0,
        baseline_final_bpm=np.asarray([145.0 - index for index in range(len(windows))]),
        config=FrozenDualResetConfig(
            experiment_mode="a2",
            hits_required=1,
            qualification_windows=1,
            observability_recovery_hits=2,
        ),
    )

    assert result.window_rows[2]["observability_state"] == "recovered"
    assert result.window_rows[2]["handoff_reinitialization_count"] == 1
    assert result.window_rows[2]["switch_target_ready"] is False
    assert result.window_rows[2]["switch_final_bpm"] == 143.0
    assert result.window_rows[4]["switch_target_ready"] is True
    assert result.window_rows[5]["observability_state"] == "lost_after_recovery"
    assert result.window_rows[5]["switch_target_ready"] is False
    assert result.window_rows[7]["handoff_reinitialization_count"] == 1
    assert result.window_rows[7]["switch_target_ready"] is False


def test_a1_hard_switches_only_after_observability_and_target_are_ready() -> None:
    windows = tuple(
        DualResetRuntimeWindow(
            window_idx=index,
            start_s=10.0 + index,
            center_s=14.0 + index,
            reliable=True,
            archived_final_bpm=170.0,
            archived_final_history=(176.0, 174.0, 172.0),
            candidates=_frame((136.0 - index, 1.0), (58.0, 0.4)),
            periodicity=0.75,
            peak_competition=1.8,
        )
        for index in range(6)
    )
    result = apply_frozen_dual_reset(
        windows,
        motion_end_s=10.0,
        baseline_final_bpm=np.full(len(windows), 170.0),
        config=FrozenDualResetConfig(
            experiment_mode="a1",
            hits_required=1,
            qualification_windows=1,
            observability_recovery_hits=2,
            gap_rescue_gap_bpm=20.0,
        ),
    )

    first_ready = next(
        index
        for index, row in enumerate(result.window_rows)
        if row["switch_target_ready"]
    )
    assert all(
        row["switch_final_bpm"] == 170.0
        for row in result.window_rows[:first_ready]
    )
    assert result.window_rows[first_ready]["switch_state"] == "gap_rescue"
    assert result.window_rows[first_ready]["switch_final_bpm"] == result.window_rows[
        first_ready
    ]["handoff_reset_bpm"]


def test_stable_crossover_requires_consecutive_reachable_ready_windows() -> None:
    rows = [
        {
            "center_s": 15.0,
            "archived_final_bpm": 130.0,
            "handoff_bpm": 127.0,
            "observability_state": "recovered",
            "switch_target_ready": True,
        },
        {
            "center_s": 16.0,
            "archived_final_bpm": 129.0,
            "handoff_bpm": 127.5,
            "observability_state": "recovered",
            "switch_target_ready": True,
        },
    ]

    from ppg_hr.v2.post_motion_dual_reset_runtime import (
        ready_gated_handoff_timeline,
    )

    timeline = ready_gated_handoff_timeline(
        rows,
        motion_end_s=10.0,
        config=FrozenDualResetConfig(
            experiment_mode="a1",
            stable_crossover_gap_bpm=6.0,
            stable_crossover_windows=2,
        ),
    )

    assert timeline["switch_states"] == (
        "ready_waiting_crossover",
        "stable_crossover",
    )
    assert timeline["final_bpm"] == (130.0, 127.5)


def test_gap_rescue_holds_actual_final_when_observability_drops_after_switch() -> None:
    rows = [
        {
            "center_s": 11.0 + index,
            "archived_final_bpm": 140.0 - 2.0 * index,
            "handoff_bpm": handoff,
            "observability_state": observability,
            "observability_reason": reason,
            "switch_target_ready": ready,
            "switch_target_readiness_reason": ready_reason,
        }
        for index, (handoff, observability, reason, ready, ready_reason) in enumerate(
            [
                (80.0, "recovered", "observable", True, "ready"),
                (138.0, "lost_after_recovery", "low_periodicity", False, "observability_lost"),
                (136.0, "recovering", "awaiting_continuity", False, "observability_recovering"),
                (134.0, "recovered", "observable", False, "insufficient_ready_history"),
                (79.0, "recovered", "observable", True, "ready"),
            ]
        )
    ]

    from ppg_hr.v2.post_motion_dual_reset_runtime import (
        ready_gated_handoff_timeline,
    )

    timeline = ready_gated_handoff_timeline(
        rows,
        motion_end_s=10.0,
        config=FrozenDualResetConfig(experiment_mode="a2"),
    )

    assert timeline["final_bpm"] == (80.0, 80.0, 80.0, 80.0, 79.0)
    assert timeline["switch_states"][1:4] == (
        "handoff_frozen",
        "handoff_frozen",
        "handoff_frozen",
    )


def test_confirmation_deadline_permanently_safe_abstains() -> None:
    rows = [
        {
            "center_s": center,
            "archived_final_bpm": 140.0,
            "handoff_bpm": 130.0,
            "observability_state": "recovering",
            "observability_reason": "awaiting_continuity",
            "switch_target_ready": False,
        }
        for center in (29.0, 31.0, 33.0)
    ]

    from ppg_hr.v2.post_motion_dual_reset_runtime import (
        ready_gated_handoff_timeline,
    )

    timeline = ready_gated_handoff_timeline(
        rows,
        motion_end_s=10.0,
        config=FrozenDualResetConfig(experiment_mode="a1"),
    )

    assert timeline["switch_states"] == (
        "observability_frozen",
        "safe_abstain",
        "safe_abstain",
    )
    assert timeline["final_bpm"] == (140.0, 140.0, 140.0)


def test_frozen_replay_reports_post60_risk_and_transition_timing() -> None:
    evidence = tuple(
        ReplayEvidenceWindow(
            center_s=14.0 + index,
            start_s=10.0 + index,
            candidates=_frame((136.0 - index, 1.0), (58.0, 0.4)),
            reliable=True,
            archived_final_history=(176.0, 174.0, 172.0),
            periodicity=0.75,
            peak_competition=1.8,
        )
        for index in range(6)
    )
    offline = tuple(
        OfflineScoreWindow(
            center_s=14.0 + index,
            aligned_time_s=19.0 + index,
            archived_time_s=19.0 + index,
            ref_bpm=136.0 - index,
            archived_final_bpm=170.0,
        )
        for index in range(6)
    )
    replay = SampleReplay("synthetic", 10.0, evidence, offline)

    result = evaluate_observability_candidate(
        replay,
        ObservabilityCandidate(
            name="a1_test",
            mode="a1",
            periodicity_min=0.5,
            peak_competition_min=1.3,
            recovery_hits=2,
            hits_required=1,
            qualification_windows=1,
        ),
    )

    assert result.sample_metrics["first_recovery_delay_s"] == 5.0
    assert result.sample_metrics["first_ready_delay_s"] == 7.0
    assert result.sample_metrics["post60_e20_count"] == 3
    assert result.sample_metrics["post60_final_mae_bpm"] < 20.0
    assert result.window_rows[-1]["switch_state"] == "handoff_active"
    assert result.window_rows[0]["handoff_consumed"] is False
    assert result.window_rows[-1]["handoff_consumed"] is True


def test_a0_a1_a2_matrix_keeps_independent_reset_exactly_identical() -> None:
    evidence = tuple(
        ReplayEvidenceWindow(
            center_s=14.0 + index,
            start_s=10.0 + index,
            candidates=_frame((136.0 - index, 1.0), (58.0, 0.4)),
            reliable=True,
            archived_final_history=(176.0, 174.0, 172.0),
            periodicity=0.75,
            peak_competition=1.8,
        )
        for index in range(6)
    )
    offline = tuple(
        OfflineScoreWindow(
            center_s=14.0 + index,
            aligned_time_s=19.0 + index,
            archived_time_s=19.0 + index,
            ref_bpm=136.0 - index,
            archived_final_bpm=170.0,
        )
        for index in range(6)
    )
    replay = SampleReplay("synthetic", 10.0, evidence, offline)

    matrix = evaluate_candidate_matrix(replay, build_predeclared_candidates())

    assert set(matrix) == {
        "a0",
        "a1_loose",
        "a1_central",
        "a1_strict",
        "a2_loose",
        "a2_central",
        "a2_strict",
    }
    assert len({result.independent_reset_bpm for result in matrix.values()}) == 1


def test_a2_reuses_only_confirmed_recovery_windows_to_ready_before_a1() -> None:
    evidence = tuple(
        ReplayEvidenceWindow(
            center_s=14.0 + index,
            start_s=10.0 + index,
            candidates=_frame((136.0 - index, 1.0), (58.0, 0.4)),
            reliable=True,
            archived_final_history=(176.0, 174.0, 172.0),
            periodicity=0.75,
            peak_competition=1.8,
        )
        for index in range(8)
    )
    offline = tuple(
        OfflineScoreWindow(
            center_s=14.0 + index,
            aligned_time_s=19.0 + index,
            archived_time_s=19.0 + index,
            ref_bpm=136.0 - index,
            archived_final_bpm=170.0,
        )
        for index in range(8)
    )
    replay = SampleReplay("synthetic", 10.0, evidence, offline)
    a1 = evaluate_observability_candidate(
        replay,
        ObservabilityCandidate("a1", "a1", recovery_hits=3),
    )
    a2 = evaluate_observability_candidate(
        replay,
        ObservabilityCandidate("a2", "a2", recovery_hits=3),
    )

    assert a2.sample_metrics["first_ready_delay_s"] < a1.sample_metrics[
        "first_ready_delay_s"
    ]
    assert a2.sample_metrics["first_ready_delay_s"] == 7.0
    assert a2.sample_metrics["reinitialization_count"] == 1
    assert a2.window_rows[2]["switch_target_ready"] is False
