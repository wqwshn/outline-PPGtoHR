from __future__ import annotations

import numpy as np
import pytest

from ppg_hr.v2.algorithm_presets import DirectionalTrackingParams
from ppg_hr.v2.post_motion_dual_reset import DualResetInput, DualResetTracker
from ppg_hr.v2.raw_fft_candidates import RawFftCandidateFrame


def _frame(*candidates: tuple[float, float]) -> RawFftCandidateFrame:
    frequencies_hz = np.asarray([bpm / 60.0 for bpm, _ in candidates], dtype=float)
    amplitudes = np.asarray([amplitude for _, amplitude in candidates], dtype=float)
    peak_indices = np.arange(len(candidates), dtype=int)
    ordered_peak_indices = np.argsort(-amplitudes, kind="stable")
    return RawFftCandidateFrame(
        frequencies_hz=frequencies_hz,
        amplitudes=amplitudes,
        peak_indices=peak_indices,
        ordered_peak_indices=ordered_peak_indices,
    )


def _tracking() -> DirectionalTrackingParams:
    return DirectionalTrackingParams(
        range_up_bpm=20.0,
        range_down_bpm=25.0,
        limit_up_bpm=10.0,
        step_up_bpm=5.0,
        limit_down_bpm=10.0,
        step_down_bpm=5.0,
    )


def test_handoff_uses_final_prior_without_contaminating_independent_reset() -> None:
    tracker = DualResetTracker(tracking=_tracking())

    result = tracker.step(
        DualResetInput(
            center_s=30.0,
            candidates=_frame((55.0, 1.0), (135.0, 0.5)),
            reliable=True,
            previous_final_bpm=(138.0, 136.0, 134.0),
        )
    )

    assert result.independent_bpm == 55.0
    assert result.handoff_bpm == 135.0
    assert result.qualification.qualified is False
    assert result.qualification.reason == "insufficient_history"
    assert result.independent_trace["selection"] == "raw_evidence/no_prior"
    assert result.handoff_trace["selection"] == "trend_persistence_decay"


def test_both_paths_track_directionally_from_their_distinct_initial_lock() -> None:
    tracker = DualResetTracker(tracking=_tracking())
    first = tracker.step(
        DualResetInput(
            0.0,
            _frame((55.0, 1.0), (135.0, 0.5)),
            True,
            (138.0, 136.0, 134.0),
        )
    )
    second = tracker.step(
        DualResetInput(
            2.0,
            _frame((135.0, 1.0), (55.0, 0.5)),
            True,
            (138.0, 136.0, 134.0),
        )
    )

    assert (first.independent_bpm, first.handoff_bpm) == (55.0, 135.0)
    assert (second.independent_bpm, second.handoff_bpm) == (55.0, 135.0)
    assert second.independent_trace == {
        "selection": "raw_evidence/no_prior",
        "previous_bpm": 55.0,
        "search_min_bpm": 30.0,
        "search_max_bpm": 75.0,
        "selected_rank": 2,
        "source": "raw_local_peaks",
        "selected_candidate_bpm": 55.0,
        "tracked_bpm": 55.0,
        "limited_bpm": 55.0,
    }
    assert second.handoff_trace["previous_bpm"] == 135.0
    assert second.handoff_trace["search_min_bpm"] == 110.0
    assert second.handoff_trace["search_max_bpm"] == 155.0
    assert second.handoff_trace["selected_rank"] == 1
    assert second.handoff_trace["source"] == "raw_local_peaks"
    assert second.handoff_trace["tracked_bpm"] == 135.0
    assert second.handoff_trace["limited_bpm"] == 135.0


def test_handoff_qualifies_after_three_stable_hits_in_four_windows() -> None:
    tracker = DualResetTracker()
    results = []

    for index, bpm in enumerate((132.0, 131.0, 130.0, 129.0)):
        results.append(
            tracker.step(
                DualResetInput(
                    center_s=30.0 + index * 2.0,
                    candidates=_frame((bpm, 1.0)),
                    reliable=True,
                    previous_final_bpm=(138.0, 136.0, 134.0),
                )
            )
        )

    assert [result.handoff_bpm for result in results] == [132.0, 131.0, 130.0, 129.0]
    assert [result.qualification.qualified for result in results] == [False, False, False, True]
    assert results[-1].qualification.stable_hits == 3
    assert results[-1].qualification.observed_windows == 4


def test_limiter_smoothing_cannot_create_raw_trajectory_hits() -> None:
    tracking = DirectionalTrackingParams(
        range_up_bpm=60.0,
        range_down_bpm=60.0,
        limit_up_bpm=1.0,
        step_up_bpm=1.0,
        limit_down_bpm=1.0,
        step_down_bpm=1.0,
    )
    tracker = DualResetTracker(
        tracking=tracking,
        mechanism="cold_reset",
        trajectory_tolerance_bpm=5.0,
    )
    results = [
        tracker.step(DualResetInput(float(index), _frame((bpm, 1.0)), True, ()))
        for index, bpm in enumerate((100.0, 120.0, 140.0, 160.0))
    ]

    assert [result.handoff_bpm for result in results] == [100.0, 101.0, 102.0, 103.0]
    assert [result.handoff_trace["selected_candidate_bpm"] for result in results] == [
        100.0,
        120.0,
        140.0,
        160.0,
    ]
    assert results[-1].qualification.stable_hits == 0
    assert results[-1].qualification.qualified is False
    assert results[-1].qualification.reason == "trajectory_unstable"


def test_candidate_can_qualify_while_switch_target_is_not_ready() -> None:
    tracker = DualResetTracker(
        tracking=DirectionalTrackingParams(
            range_up_bpm=20.0,
            range_down_bpm=25.0,
            limit_up_bpm=1.5,
            step_up_bpm=1.5,
            limit_down_bpm=3.5,
            step_down_bpm=3.0,
        ),
        mechanism="trend_persistence",
        trajectory_tolerance_bpm=6.0,
    )
    final_history = (168.0, 166.0, 164.0)
    tracker.step(
        DualResetInput(0.0, _frame((50.0, 1.0)), True, final_history)
    )

    results = [
        tracker.step(
            DualResetInput(
                float(index),
                _frame((145.0 - index, 1.0), (55.0, 0.5)),
                True,
                final_history,
            )
        )
        for index in range(1, 7)
    ]
    result = results[-1]

    assert result.handoff_trace["selected_candidate_bpm"] == 139.0
    assert result.handoff_bpm < 70.0
    assert result.candidate_qualification.qualified is True
    assert result.switch_target_readiness.ready is False
    assert result.switch_target_readiness.reason == "candidate_handoff_gap"
    assert result.switch_target_readiness.candidate_handoff_gap_bpm > 60.0


def test_unreliable_window_revokes_ready_and_requires_fresh_evidence() -> None:
    tracker = DualResetTracker(mechanism="cold_reset")
    ready = None
    for index in range(6):
        ready = tracker.step(
            DualResetInput(float(index), _frame((100.0, 1.0)), True, ())
        )

    assert ready is not None
    assert ready.switch_target_readiness.ready is True

    revoked = tracker.step(
        DualResetInput(6.0, _frame((100.0, 1.0)), False, ())
    )
    first_recovery = tracker.step(
        DualResetInput(7.0, _frame((100.0, 1.0)), True, ())
    )

    assert revoked.switch_target_readiness.ready is False
    assert revoked.switch_target_readiness.revoked_reason == "unreliable"
    assert revoked.switch_target_readiness.state_age_windows == 1
    assert first_recovery.switch_target_readiness.ready is False


def test_remote_candidate_identity_change_revokes_old_readiness() -> None:
    tracker = DualResetTracker(mechanism="cold_reset")
    for index in range(6):
        result = tracker.step(
            DualResetInput(float(index), _frame((100.0, 1.0)), True, ())
        )
    assert result.switch_target_readiness.ready is True

    changed = tracker.step(
        DualResetInput(6.0, _frame((110.0, 1.0)), True, ())
    )

    assert changed.switch_target_readiness.ready is False
    assert changed.switch_target_readiness.revoked_reason == (
        "candidate_identity_changed"
    )
    assert changed.switch_target_readiness.stable_hits == 0
    assert changed.candidate_qualification.revoked_reason == (
        "candidate_identity_changed"
    )
    assert changed.candidate_qualification.state_age_windows == 1


def test_controlled_reanchor_moves_only_handoff_after_causal_evidence() -> None:
    control = DualResetTracker(mechanism="trend_persistence")
    reanchored = DualResetTracker(
        mechanism="trend_persistence",
        controlled_reanchor=True,
    )
    final_history = (168.0, 166.0, 164.0)
    initial = DualResetInput(0.0, _frame((50.0, 1.0)), True, final_history)
    control.step(initial)
    reanchored.step(initial)

    control_results = []
    reanchored_results = []
    for index, bpm in enumerate((145.0, 144.0, 143.0, 142.0, 141.0), start=1):
        window = DualResetInput(
            float(index),
            _frame((bpm, 1.0), (55.0, 0.5)),
            True,
            final_history,
        )
        control_results.append(control.step(window))
        reanchored_results.append(reanchored.step(window))

    events = [
        result
        for result in reanchored_results
        if result.handoff_trace["reanchor_event"]
    ]
    assert len(events) == 1
    assert events[0].handoff_bpm == events[0].handoff_trace[
        "selected_candidate_bpm"
    ]
    assert events[0].independent_bpm == control_results[2].independent_bpm
    assert events[0].switch_target_readiness.ready is False
    assert reanchored_results[-1].switch_target_readiness.ready is True
    assert control_results[-1].handoff_bpm < 70.0


def test_controlled_reanchor_rejects_candidate_conflicting_with_causal_prior() -> None:
    tracker = DualResetTracker(
        mechanism="trend_persistence",
        controlled_reanchor=True,
        reanchor_prior_guard_bpm=45.0,
    )
    final_history = (168.0, 166.0, 164.0)
    tracker.step(DualResetInput(0.0, _frame((50.0, 1.0)), True, final_history))

    results = [
        tracker.step(
            DualResetInput(
                float(index),
                _frame((95.0, 1.0), (55.0, 0.5)),
                True,
                final_history,
            )
        )
        for index in range(1, 6)
    ]

    assert not any(result.handoff_trace["reanchor_event"] for result in results)
    assert results[-1].candidate_qualification.reason == "causal_prior_conflict"


def test_controlled_reanchor_does_not_jump_across_reachable_gap() -> None:
    tracker = DualResetTracker(
        mechanism="trend_persistence",
        controlled_reanchor=True,
        reanchor_min_gap_bpm=25.0,
    )
    final_history = (125.0, 123.0, 121.0)
    tracker.step(
        DualResetInput(0.0, _frame((110.0, 1.0)), True, final_history)
    )
    results = [
        tracker.step(
            DualResetInput(
                float(index),
                _frame((90.0, 1.0), (110.0, 0.5)),
                True,
                final_history,
            )
        )
        for index in range(1, 6)
    ]

    assert not any(result.handoff_trace["reanchor_event"] for result in results)
    assert results[-1].handoff_bpm > 90.0


def test_handoff_abandons_wrong_prior_for_persistent_remote_raw_top_peak() -> None:
    tracker = DualResetTracker()
    results = []

    for index, top_bpm in enumerate((55.0, 56.0, 57.0)):
        results.append(
            tracker.step(
                DualResetInput(
                    center_s=30.0 + index * 2.0,
                    candidates=_frame((top_bpm, 1.0), (135.0, 0.5)),
                    reliable=True,
                    previous_final_bpm=(138.0, 136.0, 134.0),
                )
            )
        )

    assert [result.handoff_bpm for result in results] == [135.0, 135.0, 132.0]
    assert results[-1].handoff_trace["selection"] == "trend_persistence_decay"
    assert results[-1].handoff_trace["source"] == "persistent_raw_top_1"
    assert results[-1].handoff_trace["tracked_bpm"] == 57.0
    assert results[-1].handoff_trace["limited_bpm"] == 132.0


def test_final_prior_uses_causal_medians_clipped_trend_and_configured_decay() -> None:
    final_history = (120.0, 110.0, 100.0, 90.0, 80.0, 70.0, 60.0)
    frame = _frame((55.0, 1.0), (67.0, 0.5))
    fast_decay = DualResetTracker(prior_half_life_s=5.0)
    slow_decay = DualResetTracker(prior_half_life_s=15.0)

    fast_initial = fast_decay.step(DualResetInput(30.0, frame, True, final_history))
    slow_decay.step(DualResetInput(30.0, frame, True, final_history))
    fast_later = fast_decay.step(DualResetInput(40.0, frame, True, final_history))
    slow_later = slow_decay.step(DualResetInput(40.0, frame, True, final_history))

    assert fast_initial.handoff_bpm == 67.0
    assert fast_initial.handoff_trace["final_anchor_bpm"] == 70.0
    assert fast_initial.handoff_trace["final_trend_bpm_per_window"] == -3.0
    assert fast_initial.handoff_trace["predicted_prior_bpm"] == 67.0
    assert fast_later.handoff_bpm == 64.0
    assert slow_later.handoff_bpm == 64.0
    assert fast_later.handoff_trace["tracked_bpm"] == 55.0
    assert slow_later.handoff_trace["tracked_bpm"] == 55.0
    assert fast_later.handoff_trace["prior_weight"] == 0.25
    assert slow_later.handoff_trace["prior_weight"] == pytest.approx(2.0 ** (-2.0 / 3.0))


def test_final_prior_is_frozen_against_later_final_feedback() -> None:
    original_history = (138.0, 136.0, 134.0)
    changed_history = (50.0, 60.0, 70.0)
    frame = _frame((132.0, 0.5), (135.0, 1.0))
    changed = DualResetTracker(tracking=_tracking())
    control = DualResetTracker(tracking=_tracking())

    changed.step(DualResetInput(10.0, frame, True, original_history))
    control.step(DualResetInput(10.0, frame, True, original_history))
    changed_result = changed.step(DualResetInput(15.0, frame, True, changed_history))
    control_result = control.step(DualResetInput(15.0, frame, True, original_history))

    assert changed_result.handoff_bpm == control_result.handoff_bpm
    assert changed_result.handoff_trace["final_anchor_bpm"] == 136.0
    assert changed_result.handoff_trace["final_trend_bpm_per_window"] == -2.0
    assert changed_result.handoff_trace["predicted_prior_bpm"] == 132.0
    assert changed_result.handoff_trace["prior_weight"] == pytest.approx(2.0**-0.5)


def test_missing_raw_candidate_explicitly_holds_and_disqualifies_after_limit() -> None:
    tracker = DualResetTracker(max_held_previous=1)
    for index, bpm in enumerate((132.0, 131.0, 130.0, 129.0)):
        tracker.step(DualResetInput(float(index), _frame((bpm, 1.0)), True, ()))

    first_hold = tracker.step(DualResetInput(4.0, _frame(), True, ()))
    second_hold = tracker.step(DualResetInput(5.0, _frame(), True, ()))

    assert first_hold.handoff_bpm == 129.0
    assert first_hold.qualification.held_previous_count == 1
    assert first_hold.handoff_trace["selection"] == "held_previous"
    assert first_hold.handoff_trace["selected_candidate_bpm"] is None
    assert second_hold.handoff_bpm == 129.0
    assert second_hold.qualification.qualified is False
    assert second_hold.qualification.reason == "held_previous"
    assert second_hold.qualification.held_previous_count == 2


def test_held_count_is_number_of_held_windows_in_recent_three() -> None:
    tracker = DualResetTracker(tracking=_tracking(), max_held_previous=1)
    tracker.step(DualResetInput(0.0, _frame((129.0, 1.0)), True, ()))

    first_hold = tracker.step(DualResetInput(1.0, _frame(), True, ()))
    raw = tracker.step(DualResetInput(2.0, _frame((128.0, 1.0)), True, ()))
    second_hold = tracker.step(DualResetInput(3.0, _frame(), True, ()))

    assert first_hold.qualification.held_previous_count == 1
    assert raw.qualification.held_previous_count == 1
    assert second_hold.qualification.held_previous_count == 2
    assert second_hold.qualification.reason == "held_previous"


def test_missing_candidate_breaks_remote_raw_top_persistence() -> None:
    tracker = DualResetTracker(max_held_previous=1)
    final_history = (138.0, 136.0, 134.0)

    first = tracker.step(
        DualResetInput(0.0, _frame((55.0, 1.0), (135.0, 0.5)), True, final_history)
    )
    tracker.step(DualResetInput(1.0, _frame(), True, final_history))
    second = tracker.step(
        DualResetInput(2.0, _frame((56.0, 1.0), (135.0, 0.5)), True, final_history)
    )
    third = tracker.step(
        DualResetInput(3.0, _frame((57.0, 1.0), (135.0, 0.5)), True, final_history)
    )
    fourth = tracker.step(
        DualResetInput(4.0, _frame((58.0, 1.0), (135.0, 0.5)), True, final_history)
    )

    assert [first.handoff_bpm, second.handoff_bpm, third.handoff_bpm] == [135.0] * 3
    assert fourth.handoff_bpm == 132.0
    assert fourth.handoff_trace["tracked_bpm"] == 58.0


@pytest.mark.parametrize(
    "kwargs",
    (
        {"hits_required": 0},
        {"qualification_windows": 0},
        {"hits_required": 5, "qualification_windows": 4},
        {"trajectory_tolerance_bpm": -0.1},
        {"trajectory_tolerance_bpm": 0.0},
        {"min_amp_ratio": -0.1},
        {"min_amp_ratio": 1.1},
        {"max_held_previous": -1},
    ),
)
def test_qualification_configuration_rejects_invalid_values(
    kwargs: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError):
        DualResetTracker(**kwargs)


def test_no_final_prior_trace_is_explicitly_raw_evidence() -> None:
    result = DualResetTracker(tracking=_tracking()).step(
        DualResetInput(0.0, _frame((80.0, 1.0)), True, ())
    )

    assert result.handoff_trace["selection"] == "raw_evidence/no_prior"
    assert result.handoff_trace["predicted_prior_bpm"] is None


def test_mechanism_configs_are_public_cumulative_ablations() -> None:
    history = (140.0, 136.0, 132.0)
    first_frame = _frame((55.0, 1.0), (133.0, 0.5), (136.0, 0.4))
    names = (
        "cold_reset",
        "final_anchor",
        "final_trend",
        "trend_persistence",
        "trend_persistence_decay",
    )
    trackers = {
        name: DualResetTracker(
            tracking=_tracking(), mechanism=name, prior_half_life_s=10.0
        )
        for name in names
    }

    initial = {
        name: tracker.step(DualResetInput(0.0, first_frame, True, history))
        for name, tracker in trackers.items()
    }

    assert [initial[name].handoff_bpm for name in names] == [
        55.0,
        136.0,
        133.0,
        133.0,
        133.0,
    ]
    assert [initial[name].handoff_trace["mechanism"] for name in names] == list(names)
    assert [initial[name].handoff_trace["selection"] for name in names] == [
        "raw_evidence/no_prior",
        "final_anchor",
        "final_trend",
        "trend_persistence",
        "trend_persistence_decay",
    ]

    second_frame = _frame((139.0, 1.0), (130.0, 0.4))
    no_decay = trackers["final_trend"].step(
        DualResetInput(10.0, second_frame, True, (50.0, 60.0, 70.0))
    )
    decayed = trackers["trend_persistence_decay"].step(
        DualResetInput(10.0, second_frame, True, (50.0, 60.0, 70.0))
    )

    assert no_decay.handoff_bpm == 130.0
    assert decayed.handoff_bpm == 139.0
    assert no_decay.handoff_trace["prior_weight"] == 1.0
    assert decayed.handoff_trace["prior_weight"] == 0.5


def test_unknown_mechanism_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="mechanism"):
        DualResetTracker(mechanism="unknown")


@pytest.mark.parametrize(
    "mechanism",
    (
        "cold_reset",
        "final_anchor",
        "final_trend",
        "trend_persistence",
        "trend_persistence_decay",
    ),
)
def test_non_held_selected_candidates_belong_to_current_raw_frame(
    mechanism: str,
) -> None:
    tracker = DualResetTracker(tracking=_tracking(), mechanism=mechanism)
    frames = (
        _frame((55.0, 1.0), (135.0, 0.5)),
        _frame((135.0, 1.0), (55.0, 0.5)),
    )

    for index, frame in enumerate(frames):
        result = tracker.step(
            DualResetInput(float(index), frame, True, (138.0, 136.0, 134.0))
        )
        raw_bpm = {bpm for bpm, _ in frame.top()}
        for trace in (result.independent_trace, result.handoff_trace):
            if trace["source"] != "held_previous":
                assert trace["selected_candidate_bpm"] in raw_bpm
                assert trace["tracked_bpm"] == trace["selected_candidate_bpm"]
