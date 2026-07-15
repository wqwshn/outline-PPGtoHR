from __future__ import annotations

import numpy as np

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


def test_handoff_uses_final_prior_without_contaminating_independent_reset() -> None:
    tracker = DualResetTracker()

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
    assert result.independent_trace["selection"] == "raw_top_1"
    assert result.handoff_trace["selection"] == "decayed_final_prior"


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

    assert [result.handoff_bpm for result in results] == [135.0, 135.0, 57.0]
    assert results[-1].handoff_trace["selection"] == "persistent_raw_top_1"


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
    assert fast_later.handoff_bpm == 55.0
    assert slow_later.handoff_bpm == 67.0
    assert fast_later.handoff_bpm in {bpm for bpm, _ in frame.top()}
    assert slow_later.handoff_bpm in {bpm for bpm, _ in frame.top()}


def test_missing_raw_candidate_explicitly_holds_and_disqualifies_after_limit() -> None:
    tracker = DualResetTracker(max_held_previous=1)
    for index, bpm in enumerate((132.0, 131.0, 130.0, 129.0)):
        tracker.step(DualResetInput(float(index), _frame((bpm, 1.0)), True, ()))

    first_hold = tracker.step(DualResetInput(4.0, _frame(), True, ()))
    second_hold = tracker.step(DualResetInput(5.0, _frame(), True, ()))

    assert first_hold.handoff_bpm == 129.0
    assert first_hold.qualification.held_previous_count == 1
    assert first_hold.handoff_trace["selection"] == "held_previous"
    assert second_hold.handoff_bpm == 129.0
    assert second_hold.qualification.qualified is False
    assert second_hold.qualification.reason == "held_previous"
    assert second_hold.qualification.held_previous_count == 2


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
    assert fourth.handoff_bpm == 58.0
