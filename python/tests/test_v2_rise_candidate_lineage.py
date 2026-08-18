from __future__ import annotations

import numpy as np

from ppg_hr.v2.algorithm_presets import DirectionalTrackingParams
from ppg_hr.v2.rise_candidate_lineage import (
    LEGACY_RISE_CONFIRMATION_POLICY_ID,
    MATURE_PRESELECTOR_ADOPTION_POLICY_ID,
    RECALL_PRIORITY_OR_POLICY_ID,
    RISE_LINEAGE_NET_RISE_POLICY_ID,
    RiseCandidateLineageState,
    RiseConfirmationFrame,
    RiseConfirmationObservation,
    advance_rise_candidate_lineage,
    decide_rise_confirmation,
    finalize_rise_candidate_lineage,
)

TRACKING = DirectionalTrackingParams(
    range_up_bpm=25.0,
    range_down_bpm=25.0,
    limit_up_bpm=10.0,
    step_up_bpm=3.5,
    limit_down_bpm=10.0,
    step_down_bpm=1.5,
)


def _step(
    state: RiseCandidateLineageState,
    *,
    previous_bpm: float,
    candidates_bpm: list[float],
    selected_rank: int,
    motion_fundamental_bpm: float | None = None,
    harmonic_guard_half_width_bpm: float = 0.0,
):
    frequencies = np.asarray(candidates_bpm, dtype=float) / 60.0
    order = np.arange(len(candidates_bpm), dtype=int)
    return advance_rise_candidate_lineage(
        state=state,
        freqs=frequencies,
        candidate_order=order,
        selected_peak_idx=int(order[selected_rank - 1]),
        previous_hz=previous_bpm / 60.0,
        tracking=TRACKING,
        window_kind="motion",
        motion_fundamental_hz=(
            None
            if motion_fundamental_bpm is None
            else motion_fundamental_bpm / 60.0
        ),
        harmonic_guard_half_width_hz=harmonic_guard_half_width_bpm / 60.0,
    )


def test_confirmed_rise_lineage_survives_two_stronger_distractors() -> None:
    state = RiseCandidateLineageState()

    first = _step(
        state,
        previous_bpm=82.8,
        candidates_bpm=[100.3, 180.9, 80.6],
        selected_rank=1,
    )
    second = _step(
        state,
        previous_bpm=86.3,
        candidates_bpm=[102.5, 177.2, 43.2, 82.0],
        selected_rank=1,
    )
    third = _step(
        state,
        previous_bpm=89.8,
        candidates_bpm=[172.1, 230.7, 120.8, 43.2, 80.6, 103.3],
        selected_rank=3,
    )
    fourth = _step(
        state,
        previous_bpm=93.3,
        candidates_bpm=[68.8, 169.9, 227.8, 43.2, 79.8, 104.7],
        selected_rank=5,
    )

    assert first.reanchored is False
    assert second.reanchored is False
    assert third.reanchored is True
    assert third.selected_peak_idx == 5
    assert third.candidate_hz == frequencies_bpm(103.3)
    assert fourth.reanchored is True
    assert fourth.selected_peak_idx == 5
    assert fourth.candidate_hz == frequencies_bpm(104.7)


def test_lineage_does_not_start_for_an_ordinary_unslewed_rise() -> None:
    state = RiseCandidateLineageState()

    decision = _step(
        state,
        previous_bpm=90.0,
        candidates_bpm=[96.0, 130.0, 75.0],
        selected_rank=1,
    )

    assert decision.reanchored is False
    assert decision.reason == "seed_jump_not_slew_limited"
    assert state.count == 0


def test_lineage_resets_outside_motion() -> None:
    state = RiseCandidateLineageState(candidate_hz=100.0 / 60.0, count=2, age=2)

    decision = advance_rise_candidate_lineage(
        state=state,
        freqs=np.asarray([101.0, 130.0]) / 60.0,
        candidate_order=np.asarray([0, 1]),
        selected_peak_idx=0,
        previous_hz=95.0 / 60.0,
        tracking=TRACKING,
        window_kind="recovery",
    )

    assert decision.reanchored is False
    assert decision.reason == "outside_motion"
    assert state.candidate_hz is None
    assert state.count == 0


def test_confirmed_lineage_holds_one_window_when_current_peak_disappears() -> None:
    state = RiseCandidateLineageState(
        candidate_hz=104.7 / 60.0,
        count=6,
        age=6,
        reanchored_last_window=True,
    )

    held = _step(
        state,
        previous_bpm=96.8,
        candidates_bpm=[71.0, 168.5, 227.1, 85.0],
        selected_rank=4,
    )
    expired = _step(
        state,
        previous_bpm=104.7,
        candidates_bpm=[95.0, 170.0, 230.0],
        selected_rank=1,
    )

    assert held.reanchored is True
    assert held.selected_peak_idx is None
    assert held.reason == "confirmed_hold_reanchor"
    assert expired.reanchored is False
    assert state.candidate_hz is None


def test_confirmed_lineage_cannot_hold_if_it_never_reanchored() -> None:
    state = RiseCandidateLineageState(
        candidate_hz=130.4 / 60.0,
        count=4,
        age=4,
    )

    decision = _step(
        state,
        previous_bpm=128.9,
        candidates_bpm=[117.2, 145.0],
        selected_rank=1,
    )

    assert decision.reanchored is False
    assert decision.reason == "lineage_expired"
    assert state.candidate_hz is None


def test_confirmed_lineage_does_not_replace_a_live_candidate_after_rise_gap_closes() -> None:
    state = RiseCandidateLineageState(
        candidate_hz=82.9 / 60.0,
        count=2,
        age=2,
    )

    decision = _step(
        state,
        previous_bpm=81.3,
        candidates_bpm=[93.2, 82.9, 177.2],
        selected_rank=1,
    )

    assert decision.reanchored is False
    assert decision.reason == "confirmed_rise_gap_closed"


def test_confirmed_lineage_cannot_reanchor_to_motion_third_harmonic() -> None:
    state = RiseCandidateLineageState(
        candidate_hz=171.0 / 60.0,
        count=4,
        age=4,
    )

    decision = _step(
        state,
        previous_bpm=160.0,
        candidates_bpm=[147.0, 172.0, 230.0],
        selected_rank=1,
        motion_fundamental_bpm=57.0,
        harmonic_guard_half_width_bpm=3.0,
    )

    assert decision.reanchored is False
    assert decision.reason == "motion_third_harmonic_guard"


def frequencies_bpm(value: float) -> float:
    return value / 60.0


def _observation(
    *,
    policy_id: str,
    seed_bpm: float = 100.0,
    lineage_bpm: float = 104.0,
    adopted_after_maturity: bool = False,
    age: int = 3,
) -> RiseConfirmationObservation:
    frames = (
        RiseConfirmationFrame(
            window_id=10,
            previous_track_bpm=90.0,
            preselector_candidate_bpm=seed_bpm,
            lineage_candidate_bpm=seed_bpm,
            raw_support_ratio=0.7,
            candidate_source="post_penalty_local_peak",
            preselector_adopted_after_maturity=False,
        ),
        RiseConfirmationFrame(
            window_id=11,
            previous_track_bpm=93.5,
            preselector_candidate_bpm=lineage_bpm + 20.0,
            lineage_candidate_bpm=lineage_bpm,
            raw_support_ratio=0.6,
            candidate_source="post_penalty_local_peak",
            preselector_adopted_after_maturity=adopted_after_maturity,
        ),
    )
    return RiseConfirmationObservation(
        policy_id=policy_id,
        step_up_bpm=TRACKING.step_up_bpm,
        age=age,
        seed_candidate_bpm=seed_bpm,
        frames=frames,
        authorized_before=False,
    )


def test_net_rise_policy_uses_the_complete_seed_to_request_delta() -> None:
    accepted = decide_rise_confirmation(
        _observation(
            policy_id=RISE_LINEAGE_NET_RISE_POLICY_ID,
            seed_bpm=100.0,
            lineage_bpm=103.5,
        )
    )
    rejected = decide_rise_confirmation(
        _observation(
            policy_id=RISE_LINEAGE_NET_RISE_POLICY_ID,
            seed_bpm=100.0,
            lineage_bpm=103.49,
        )
    )

    assert accepted.action == "authorize"
    assert accepted.net_rise_bpm == 3.5
    assert rejected.action == "reject"
    assert rejected.reason == "net_rise_below_step_up"


def test_mature_adoption_policy_waits_then_authorizes_after_normal_adoption() -> None:
    waiting = decide_rise_confirmation(
        _observation(
            policy_id=MATURE_PRESELECTOR_ADOPTION_POLICY_ID,
            adopted_after_maturity=False,
            age=3,
        )
    )
    accepted = decide_rise_confirmation(
        _observation(
            policy_id=MATURE_PRESELECTOR_ADOPTION_POLICY_ID,
            adopted_after_maturity=True,
            age=4,
        )
    )
    expired = decide_rise_confirmation(
        _observation(
            policy_id=MATURE_PRESELECTOR_ADOPTION_POLICY_ID,
            adopted_after_maturity=False,
            age=6,
        )
    )

    assert waiting.action == "hold"
    assert accepted.action == "authorize"
    assert expired.action == "reject"
    assert expired.reason == "adoption_wait_expired"


def test_recall_priority_or_policy_accepts_either_preregistered_evidence() -> None:
    by_rise = decide_rise_confirmation(
        _observation(
            policy_id=RECALL_PRIORITY_OR_POLICY_ID,
            seed_bpm=100.0,
            lineage_bpm=103.5,
        )
    )
    by_adoption = decide_rise_confirmation(
        _observation(
            policy_id=RECALL_PRIORITY_OR_POLICY_ID,
            seed_bpm=100.0,
            lineage_bpm=101.0,
            adopted_after_maturity=True,
        )
    )

    assert by_rise.action == "authorize"
    assert by_rise.reason == "net_rise_authorized"
    assert by_adoption.action == "authorize"
    assert by_adoption.reason == "mature_preselector_adoption_authorized"


def test_confirmation_observation_has_no_reference_or_sample_identity_fields() -> None:
    fields = set(RiseConfirmationObservation.__dataclass_fields__)
    frame_fields = set(RiseConfirmationFrame.__dataclass_fields__)

    assert not any("ref" in name.lower() for name in fields | frame_fields)
    assert not any("sample" in name.lower() for name in fields | frame_fields)
    assert not any("record" in name.lower() for name in fields | frame_fields)


def test_legacy_policy_remains_the_explicit_default() -> None:
    decision = decide_rise_confirmation(
        _observation(policy_id=LEGACY_RISE_CONFIRMATION_POLICY_ID)
    )

    assert decision.policy_id == LEGACY_RISE_CONFIRMATION_POLICY_ID
    assert decision.action == "authorize"


def test_waiting_lineage_releases_at_six_motion_windows() -> None:
    state = RiseCandidateLineageState()
    decisions = []
    for window_id, candidate in enumerate(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        start=1,
    ):
        decisions.append(
            advance_rise_candidate_lineage(
                state=state,
                freqs=np.asarray([90.0, candidate]) / 60.0,
                raw_amplitudes=np.asarray([1.0, 0.8]),
                candidate_order=np.asarray([1, 0]),
                selected_peak_idx=1 if window_id == 1 else 0,
                previous_hz=(78.0 + window_id * 2.0) / 60.0,
                tracking=TRACKING,
                window_kind="motion",
                window_id=window_id,
                confirmation_policy_id=MATURE_PRESELECTOR_ADOPTION_POLICY_ID,
            )
        )

    assert decisions[-1].confirmation.action == "reject"
    assert decisions[-1].reason == "confirmation_adoption_wait_expired"
    assert decisions[-1].ownership_event == "release"
    assert state.candidate_hz is None


def test_authorized_lineage_holds_one_missing_peak_without_new_confirmation() -> None:
    state = RiseCandidateLineageState(
        candidate_hz=104.0 / 60.0,
        seed_candidate_hz=100.0 / 60.0,
        count=4,
        age=4,
        authorized=True,
        reanchored_last_window=True,
    )

    held = advance_rise_candidate_lineage(
        state=state,
        freqs=np.asarray([90.0, 130.0]) / 60.0,
        raw_amplitudes=np.asarray([1.0, 0.5]),
        candidate_order=np.asarray([0, 1]),
        selected_peak_idx=0,
        previous_hz=96.0 / 60.0,
        tracking=TRACKING,
        window_kind="motion",
        window_id=20,
        confirmation_policy_id=RISE_LINEAGE_NET_RISE_POLICY_ID,
    )

    assert held.reanchored is True
    assert held.reason == "confirmed_hold_reanchor"
    assert held.confirmation.action == "already_authorized"
    assert held.observation.frames[-1].raw_support_ratio is None
    assert held.ownership_event == "hold"


def test_motion_scope_exit_terminates_shadow_owner() -> None:
    state = RiseCandidateLineageState(
        candidate_hz=104.0 / 60.0,
        seed_candidate_hz=100.0 / 60.0,
        count=3,
        age=3,
    )

    decision = advance_rise_candidate_lineage(
        state=state,
        freqs=np.asarray([104.0]) / 60.0,
        raw_amplitudes=np.asarray([1.0]),
        candidate_order=np.asarray([0]),
        selected_peak_idx=0,
        previous_hz=96.0 / 60.0,
        tracking=TRACKING,
        window_kind="recovery",
        window_id=21,
        confirmation_policy_id=RISE_LINEAGE_NET_RISE_POLICY_ID,
    )

    assert decision.ownership_event == "terminate"
    assert decision.ownership_trace["owner_before_exists"] is True
    assert decision.ownership_trace["owner_after_exists"] is False
    assert state.candidate_hz is None


def test_eof_finalization_preserves_candidate_policy_identity() -> None:
    state = RiseCandidateLineageState(
        candidate_hz=104.0 / 60.0,
        seed_candidate_hz=100.0 / 60.0,
        count=3,
        age=3,
    )

    trace = finalize_rise_candidate_lineage(
        state,
        window_id=22,
        confirmation_policy_id=RISE_LINEAGE_NET_RISE_POLICY_ID,
    )

    assert trace["confirmation_policy_id"] == RISE_LINEAGE_NET_RISE_POLICY_ID
    assert trace["ownership_event"] == "terminate"


def test_observation_preserves_the_runtime_candidate_source() -> None:
    state = RiseCandidateLineageState()

    decision = advance_rise_candidate_lineage(
        state=state,
        freqs=np.asarray([90.0, 110.0]) / 60.0,
        raw_amplitudes=np.asarray([1.0, 0.8]),
        candidate_order=np.asarray([1, 0]),
        selected_peak_idx=1,
        selected_candidate_source="protection_suppressed",
        previous_hz=90.0 / 60.0,
        tracking=TRACKING,
        window_kind="motion",
        window_id=23,
        confirmation_policy_id=RISE_LINEAGE_NET_RISE_POLICY_ID,
    )

    assert decision.observation is not None
    assert decision.observation.frames[0].candidate_source == "protection_suppressed"
    assert decision.ownership_trace["owner_after_source"] == "protection_suppressed"


def test_legacy_decision_view_can_freeze_without_freezing_formal_owner() -> None:
    state = RiseCandidateLineageState()
    frequencies = np.asarray([90.0, 110.0]) / 60.0

    acquired = advance_rise_candidate_lineage(
        state=state,
        freqs=frequencies,
        candidate_order=np.asarray([1, 0]),
        selected_peak_idx=1,
        previous_hz=90.0 / 60.0,
        tracking=TRACKING,
        window_kind="motion",
        window_id=30,
        confirmation_policy_id=LEGACY_RISE_CONFIRMATION_POLICY_ID,
    )
    missing = advance_rise_candidate_lineage(
        state=state,
        freqs=frequencies,
        candidate_order=np.asarray([], dtype=int),
        selected_peak_idx=None,
        previous_hz=93.5 / 60.0,
        tracking=TRACKING,
        window_kind="motion",
        window_id=31,
        confirmation_policy_id=LEGACY_RISE_CONFIRMATION_POLICY_ID,
    )

    assert acquired.ownership_event == "acquire"
    assert acquired.ownership_trace["owner_after_age"] == 1
    assert missing.reason == "lineage_missing"
    assert missing.age == 1
    assert state.candidate_hz == frequencies_bpm(110.0)
    assert missing.ownership_event == "release"
    assert missing.ownership_trace["owner_before_age"] == 1
    assert missing.ownership_trace["owner_age_advanced_to"] == 2
    assert missing.ownership_trace["owner_after_exists"] is False


def test_legacy_reanchor_target_must_match_reacquired_formal_owner() -> None:
    state = RiseCandidateLineageState()
    candidate_a_bpm = 100.0
    candidate_b_bpm = 110.0

    acquired_a = advance_rise_candidate_lineage(
        state=state,
        freqs=np.asarray([candidate_a_bpm]) / 60.0,
        candidate_order=np.asarray([0]),
        selected_peak_idx=0,
        previous_hz=85.0 / 60.0,
        tracking=TRACKING,
        window_kind="motion",
        window_id=40,
        confirmation_policy_id=LEGACY_RISE_CONFIRMATION_POLICY_ID,
    )
    released_a = advance_rise_candidate_lineage(
        state=state,
        freqs=np.asarray([120.0]) / 60.0,
        candidate_order=np.asarray([0]),
        selected_peak_idx=0,
        previous_hz=85.0 / 60.0,
        tracking=TRACKING,
        window_kind="motion",
        window_id=41,
        confirmation_policy_id=LEGACY_RISE_CONFIRMATION_POLICY_ID,
    )
    acquired_b = advance_rise_candidate_lineage(
        state=state,
        freqs=np.asarray([candidate_a_bpm, candidate_b_bpm]) / 60.0,
        candidate_order=np.asarray([0, 1]),
        selected_peak_idx=1,
        previous_hz=85.0 / 60.0,
        tracking=TRACKING,
        window_kind="motion",
        window_id=42,
        confirmation_policy_id=LEGACY_RISE_CONFIRMATION_POLICY_ID,
    )
    reanchored_a = advance_rise_candidate_lineage(
        state=state,
        freqs=np.asarray([candidate_a_bpm, candidate_b_bpm]) / 60.0,
        candidate_order=np.asarray([0, 1]),
        selected_peak_idx=1,
        previous_hz=85.0 / 60.0,
        tracking=TRACKING,
        window_kind="motion",
        window_id=43,
        confirmation_policy_id=LEGACY_RISE_CONFIRMATION_POLICY_ID,
    )
    terminated = advance_rise_candidate_lineage(
        state=state,
        freqs=np.asarray([candidate_a_bpm]) / 60.0,
        candidate_order=np.asarray([0]),
        selected_peak_idx=0,
        previous_hz=85.0 / 60.0,
        tracking=TRACKING,
        window_kind="recovery",
        window_id=44,
        confirmation_policy_id=LEGACY_RISE_CONFIRMATION_POLICY_ID,
    )

    assert acquired_a.ownership_event == "acquire"
    assert acquired_a.ownership_trace["owner_after_candidate_bpm"] == candidate_a_bpm
    assert released_a.ownership_event == "release"
    assert released_a.ownership_trace["owner_after_exists"] is False
    assert acquired_b.ownership_event == "acquire"
    assert acquired_b.ownership_trace["owner_after_candidate_bpm"] == candidate_b_bpm
    assert acquired_b.candidate_hz == frequencies_bpm(candidate_a_bpm)
    assert candidate_b_bpm != candidate_a_bpm
    assert reanchored_a.reanchored is True
    assert reanchored_a.confirmation.action == "authorize"
    assert reanchored_a.selected_peak_idx == 0
    assert reanchored_a.candidate_hz == frequencies_bpm(candidate_a_bpm)
    assert reanchored_a.ownership_trace["ownership_events"] == [
        "release",
        "acquire",
    ]
    assert reanchored_a.ownership_trace["owner_before_candidate_bpm"] == candidate_b_bpm
    assert reanchored_a.ownership_trace["owner_after_candidate_bpm"] == candidate_a_bpm
    assert reanchored_a.ownership_trace["owner_after_age"] == 1
    assert reanchored_a.ownership_trace["lineage_action"] == "authorize"
    assert terminated.ownership_event == "terminate"
    assert terminated.ownership_trace["owner_before_candidate_bpm"] == candidate_a_bpm
    assert terminated.ownership_trace["owner_after_exists"] is False
