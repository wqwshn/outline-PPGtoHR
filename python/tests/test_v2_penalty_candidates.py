from __future__ import annotations

from pathlib import Path

import pytest

from ppg_hr.v2.penalty_candidate_freeze import (
    freeze_penalty_candidate_artifacts,
)
from ppg_hr.v2.penalty_candidates import (
    CandidateVisibilityMode,
    PenaltyPolicyObservation,
    SuppressedProtectedShadowState,
    advance_suppressed_protected_shadow,
    apply_candidate_visibility,
    decide_penalty_policy,
    finalize_suppressed_protected_shadow,
    nondestructive_motion_penalty_candidate_v1,
    penalty_candidate_by_id,
    penalty_candidates_v1,
    suppressed_protected_continuous_visibility_candidate_v1,
    suppressed_protected_same_window_visibility_candidate_v1,
    suppressed_protected_shadow_candidate_v1,
)
from ppg_hr.v2.runtime_policy import runtime_policy_from_config
from ppg_hr.v2.types import V2RunConfig


def test_penalty_registry_declares_control_and_two_new_strategies() -> None:
    candidates = penalty_candidates_v1()

    assert [candidate.penalty_id for candidate in candidates] == [
        "current_soft_penalty_control_v1",
        "resolution_adaptive_width_v1",
        "trusted_history_corridor_v1",
    ]
    assert [candidate.design_role for candidate in candidates] == [
        "control",
        "new_candidate",
        "new_candidate",
    ]
    assert [candidate.mechanism_complexity for candidate in candidates] == [
        0,
        1,
        2,
    ]
    assert len({candidate.sha256 for candidate in candidates}) == 3
    for candidate in candidates:
        payload = candidate.to_dict()
        assert payload["formula"]
        assert payload["constants"]
        assert payload["boundaries"]
        assert payload["fallback_rules"]
        assert payload["runtime_evidence_fields"]
        assert payload["trace_fields"]
        assert payload["uses_reference_hr_runtime"] is False
        assert payload["causal_online_ready"] is False
        assert "offline v2 motion" in payload["runtime_information_boundary"]


def test_nondestructive_candidate_is_explicit_opt_in_outside_historical_registry() -> None:
    historical_ids = {
        candidate.penalty_id for candidate in penalty_candidates_v1()
    }
    candidate = nondestructive_motion_penalty_candidate_v1()

    assert candidate.penalty_id == "nondestructive_weighted_visible_v1"
    assert candidate.penalty_id not in historical_ids
    assert penalty_candidate_by_id(candidate.penalty_id) == candidate
    assert (
        candidate.candidate_visibility_mode
        is CandidateVisibilityMode.WEIGHTED_VISIBLE
    )
    assert all(
        item.candidate_visibility_mode
        is CandidateVisibilityMode.HARD_EXCLUSION
        for item in penalty_candidates_v1()
    )
    assert candidate.constants["width_mode"] == "configured"
    assert candidate.constants["corridor_mode"] == "single_previous_track"
    assert candidate.uses_reference_hr_runtime is False
    assert "record" not in candidate.runtime_evidence_fields
    assert "scene" not in candidate.runtime_evidence_fields


def test_suppressed_protected_shadow_is_explicit_opt_in_and_default_stays_hard() -> None:
    default = runtime_policy_from_config(
        V2RunConfig(data_path=Path("data.csv"), ref_path=Path("ref.csv"))
    )
    candidate = suppressed_protected_shadow_candidate_v1()
    opted_in = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            penalty_candidate_id=candidate.penalty_id,
        )
    )

    assert candidate.penalty_id == "suppressed_protected_shadow_v1"
    assert penalty_candidate_by_id(candidate.penalty_id) == candidate
    assert default.motion_penalty.candidate_visibility_mode == "hard_exclusion"
    assert opted_in.motion_penalty.candidate_visibility_mode == "shadow_release"
    assert candidate.uses_reference_hr_runtime is False
    assert not {"record", "scene", "coordinate", "hr_ref"} & set(
        candidate.runtime_evidence_fields
    )


def test_same_window_visibility_is_explicit_opt_in_without_owner_or_labels() -> None:
    default = runtime_policy_from_config(
        V2RunConfig(data_path=Path("data.csv"), ref_path=Path("ref.csv"))
    )
    candidate = suppressed_protected_same_window_visibility_candidate_v1()
    opted_in = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            penalty_candidate_id=candidate.penalty_id,
        )
    )

    assert candidate.penalty_id == "suppressed_protected_same_window_visibility_v1"
    assert penalty_candidate_by_id(candidate.penalty_id) == candidate
    assert default.motion_penalty.candidate_visibility_mode == "hard_exclusion"
    assert opted_in.motion_penalty.candidate_visibility_mode == "same_window_release"
    assert candidate.uses_reference_hr_runtime is False
    assert not {"record", "scene", "coordinate", "hr_ref", "error", "gate"} & set(
        candidate.runtime_evidence_fields
    )
    assert not any("owner" in field for field in candidate.trace_fields)


def test_continuous_visibility_is_explicit_opt_in_without_labels() -> None:
    default = runtime_policy_from_config(
        V2RunConfig(data_path=Path("data.csv"), ref_path=Path("ref.csv"))
    )
    candidate = suppressed_protected_continuous_visibility_candidate_v1()
    opted_in = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            penalty_candidate_id=candidate.penalty_id,
        )
    )

    assert candidate.penalty_id == "suppressed_protected_continuous_visibility_v1"
    assert penalty_candidate_by_id(candidate.penalty_id) == candidate
    assert default.motion_penalty.candidate_visibility_mode == "hard_exclusion"
    assert opted_in.motion_penalty.candidate_visibility_mode == "shadow_release"
    assert candidate.uses_reference_hr_runtime is False
    assert not {
        "record",
        "scene",
        "coordinate",
        "hr_ref",
        "error",
        "gate",
    } & set(candidate.runtime_evidence_fields)
    assert "shadow_owner_event" in candidate.trace_fields


def test_shadow_owner_acquire_window_is_inert_and_empty_windows_age_to_expiry() -> None:
    state = SuppressedProtectedShadowState()

    acquired = advance_suppressed_protected_shadow(
        state,
        window_id=10,
        window_kind="motion",
        candidate_bins=(21, 42),
        candidate_bpms=(90.0, 138.0),
        would_remove_bins=(21,),
        protection_suppressed=True,
        protected_target_bin=21,
        protected_target_bpm=90.0,
    )

    assert acquired.owner_event == "acquire"
    assert acquired.released_candidate_bin is None
    assert acquired.owner_after_exists is True
    assert state.age == 0
    assert (state.origin_window, state.origin_bin, state.origin_bpm) == (10, 21, 90.0)

    first = advance_suppressed_protected_shadow(
        state,
        window_id=11,
        window_kind="motion",
        candidate_bins=(),
        candidate_bpms=(),
        would_remove_bins=(),
    )
    second = advance_suppressed_protected_shadow(
        state,
        window_id=12,
        window_kind="motion",
        candidate_bins=(),
        candidate_bpms=(),
        would_remove_bins=(),
    )
    expired = advance_suppressed_protected_shadow(
        state,
        window_id=13,
        window_kind="motion",
        candidate_bins=(),
        candidate_bpms=(),
        would_remove_bins=(),
    )

    assert [(first.owner_event, first.owner_age), (second.owner_event, second.owner_age)] == [
        ("carry", 1),
        ("carry", 2),
    ]
    assert expired.owner_event == "terminal"
    assert expired.owner_reason == "expired"
    assert expired.owner_match_result == "empty_candidates"
    assert expired.owner_age == 3
    assert expired.owner_after_exists is False
    assert state.active is False


def test_shadow_owner_can_release_on_the_third_motion_window() -> None:
    state = SuppressedProtectedShadowState(
        active=True,
        origin_window=10,
        origin_bin=21,
        origin_bpm=90.0,
        age=2,
        current_window=12,
        current_bin=21,
        current_bpm=90.0,
    )

    released = advance_suppressed_protected_shadow(
        state,
        window_id=13,
        window_kind="motion",
        candidate_bins=(21,),
        candidate_bpms=(90.0,),
        would_remove_bins=(21,),
    )

    assert released.owner_event == "release"
    assert released.owner_reason == "released_to_selector"
    assert released.owner_age == 3
    assert released.owner_after_exists is False
    assert state.active is False


def test_continuous_owner_keeps_target_visible_until_third_motion_window() -> None:
    state = SuppressedProtectedShadowState()

    acquired = advance_suppressed_protected_shadow(
        state,
        window_id=10,
        window_kind="motion",
        candidate_bins=(21, 42),
        candidate_bpms=(90.0, 138.0),
        would_remove_bins=(21,),
        protection_suppressed=True,
        protected_target_bin=21,
        protected_target_bpm=90.0,
        continuous_visibility=True,
    )
    first = advance_suppressed_protected_shadow(
        state,
        window_id=11,
        window_kind="motion",
        candidate_bins=(22, 42),
        candidate_bpms=(91.0, 138.0),
        would_remove_bins=(22,),
        continuous_visibility=True,
    )
    second = advance_suppressed_protected_shadow(
        state,
        window_id=12,
        window_kind="motion",
        candidate_bins=(23, 42),
        candidate_bpms=(92.0, 138.0),
        would_remove_bins=(23,),
        continuous_visibility=True,
    )
    third = advance_suppressed_protected_shadow(
        state,
        window_id=13,
        window_kind="motion",
        candidate_bins=(24, 42),
        candidate_bpms=(93.0, 138.0),
        would_remove_bins=(24,),
        continuous_visibility=True,
    )

    assert acquired.owner_event == "acquire"
    assert acquired.released_candidate_bin is None
    assert [(first.owner_event, first.owner_age), (second.owner_event, second.owner_age)] == [
        ("visible", 1),
        ("visible", 2),
    ]
    assert first.released_candidate_bin == 22
    assert second.released_candidate_bin == 23
    assert first.owner_after_exists is True
    assert second.owner_after_exists is True
    assert third.owner_event == "release"
    assert third.owner_reason == "visibility_expired"
    assert third.owner_age == 3
    assert third.released_candidate_bin == 24
    assert third.owner_after_exists is False
    assert state.active is False


def test_continuous_owner_missing_target_terminates_without_visibility() -> None:
    state = SuppressedProtectedShadowState()
    advance_suppressed_protected_shadow(
        state,
        window_id=20,
        window_kind="motion",
        candidate_bins=(40,),
        candidate_bpms=(90.0,),
        would_remove_bins=(40,),
        protection_suppressed=True,
        protected_target_bin=40,
        protected_target_bpm=90.0,
        continuous_visibility=True,
    )

    missing = advance_suppressed_protected_shadow(
        state,
        window_id=21,
        window_kind="motion",
        candidate_bins=(),
        candidate_bpms=(),
        would_remove_bins=(),
        continuous_visibility=True,
    )

    assert missing.owner_event == "terminal"
    assert missing.owner_reason == "target_missing"
    assert missing.owner_match_result == "empty_candidates"
    assert missing.released_candidate_bin is None
    assert missing.owner_after_exists is False
    assert state.active is False


@pytest.mark.parametrize(
    ("window_kind", "candidate_bins", "candidate_bpms", "would_remove_bins", "reason"),
    [
        ("motion", (50,), (100.0,), (50,), "target_drifted"),
        ("motion", (41,), (92.0,), (), "left_penalty_band"),
        ("recovery", (40,), (90.0,), (40,), "motion_ended"),
        ("rest", (40,), (90.0,), (40,), "motion_ended"),
    ],
)
def test_continuous_owner_terminal_conditions_close_without_visibility(
    window_kind: str,
    candidate_bins: tuple[int, ...],
    candidate_bpms: tuple[float, ...],
    would_remove_bins: tuple[int, ...],
    reason: str,
) -> None:
    state = SuppressedProtectedShadowState()
    advance_suppressed_protected_shadow(
        state,
        window_id=30,
        window_kind="motion",
        candidate_bins=(40,),
        candidate_bpms=(90.0,),
        would_remove_bins=(40,),
        protection_suppressed=True,
        protected_target_bin=40,
        protected_target_bpm=90.0,
        continuous_visibility=True,
    )

    terminal = advance_suppressed_protected_shadow(
        state,
        window_id=31,
        window_kind=window_kind,
        candidate_bins=candidate_bins,
        candidate_bpms=candidate_bpms,
        would_remove_bins=would_remove_bins,
        continuous_visibility=True,
    )

    assert terminal.owner_event == "terminal"
    assert terminal.owner_reason == reason
    assert terminal.released_candidate_bin is None
    assert terminal.owner_after_exists is False
    assert state.active is False


def test_continuous_owner_reacquires_only_on_a_later_suppression_event() -> None:
    state = SuppressedProtectedShadowState()
    advance_suppressed_protected_shadow(
        state,
        window_id=40,
        window_kind="motion",
        candidate_bins=(40,),
        candidate_bpms=(90.0,),
        would_remove_bins=(40,),
        protection_suppressed=True,
        protected_target_bin=40,
        protected_target_bpm=90.0,
        continuous_visibility=True,
    )
    for window_id in (41, 42):
        visible = advance_suppressed_protected_shadow(
            state,
            window_id=window_id,
            window_kind="motion",
            candidate_bins=(40,),
            candidate_bpms=(90.0,),
            would_remove_bins=(40,),
            continuous_visibility=True,
        )
        assert visible.owner_event == "visible"

    released = advance_suppressed_protected_shadow(
        state,
        window_id=43,
        window_kind="motion",
        candidate_bins=(40, 60),
        candidate_bpms=(90.0, 110.0),
        would_remove_bins=(40, 60),
        protection_suppressed=True,
        protected_target_bin=60,
        protected_target_bpm=110.0,
        continuous_visibility=True,
    )
    reacquired = advance_suppressed_protected_shadow(
        state,
        window_id=44,
        window_kind="motion",
        candidate_bins=(60,),
        candidate_bpms=(110.0,),
        would_remove_bins=(60,),
        protection_suppressed=True,
        protected_target_bin=60,
        protected_target_bpm=110.0,
        continuous_visibility=True,
    )

    assert released.owner_event == "release"
    assert released.owner_origin_window == 40
    assert released.owner_after_exists is False
    assert reacquired.owner_event == "acquire"
    assert reacquired.owner_origin_window == 44
    assert reacquired.owner_after_exists is True


def test_continuous_owner_eof_finalization_is_explicit() -> None:
    state = SuppressedProtectedShadowState()
    advance_suppressed_protected_shadow(
        state,
        window_id=50,
        window_kind="motion",
        candidate_bins=(12,),
        candidate_bpms=(88.0,),
        would_remove_bins=(12,),
        protection_suppressed=True,
        protected_target_bin=12,
        protected_target_bpm=88.0,
        continuous_visibility=True,
    )

    terminal = finalize_suppressed_protected_shadow(
        state,
        window_id=51,
        reason="eof",
    )

    assert terminal.owner_event == "terminal"
    assert terminal.owner_reason == "eof"
    assert terminal.owner_origin_window == 50
    assert terminal.owner_after_exists is False
    assert state.active is False


def test_shadow_owner_missing_finite_target_terminates_immediately() -> None:
    state = SuppressedProtectedShadowState(
        active=True,
        origin_window=10,
        origin_bin=21,
        origin_bpm=90.0,
        current_window=10,
        current_bin=21,
        current_bpm=90.0,
    )

    missing = advance_suppressed_protected_shadow(
        state,
        window_id=11,
        window_kind="motion",
        candidate_bins=(21,),
        candidate_bpms=(float("nan"),),
        would_remove_bins=(21,),
    )

    assert missing.owner_event == "terminal"
    assert missing.owner_reason == "target_missing"
    assert missing.owner_match_result == "no_finite_candidates"
    assert missing.owner_age == 1
    assert missing.owner_after_exists is False


def test_shadow_owner_releases_only_the_matched_penalty_bin_then_closes() -> None:
    state = SuppressedProtectedShadowState()
    advance_suppressed_protected_shadow(
        state,
        window_id=20,
        window_kind="motion",
        candidate_bins=(40, 80),
        candidate_bpms=(90.0, 140.0),
        would_remove_bins=(40,),
        protection_suppressed=True,
        protected_target_bin=40,
        protected_target_bpm=90.0,
    )

    released = advance_suppressed_protected_shadow(
        state,
        window_id=21,
        window_kind="motion",
        candidate_bins=(39, 41, 80),
        candidate_bpms=(89.0, 92.0, 140.0),
        would_remove_bins=(39, 41),
    )
    visibility = apply_candidate_visibility(
        CandidateVisibilityMode.SHADOW_RELEASE,
        all_peak_indices=(39, 41, 80),
        hard_selectable_indices=(80,),
        hard_removed_indices=(39, 41),
        released_peak_index=released.released_candidate_bin,
    )

    assert released.owner_event == "release"
    assert released.owner_reason == "released_to_selector"
    assert released.owner_current_bin == 39
    assert released.owner_current_bpm == pytest.approx(89.0)
    assert released.owner_age == 1
    assert released.owner_after_exists is False
    assert visibility.selectable_indices == (39, 80)
    assert visibility.removed_indices == (41,)
    assert visibility.would_remove_indices == (39, 41)
    assert visibility.hard_removal_applied is True
    assert state.active is False


def test_shadow_owner_uses_bin_identity_to_break_equal_bpm_matches() -> None:
    state = SuppressedProtectedShadowState()
    advance_suppressed_protected_shadow(
        state,
        window_id=20,
        window_kind="motion",
        candidate_bins=(40,),
        candidate_bpms=(90.0,),
        would_remove_bins=(40,),
        protection_suppressed=True,
        protected_target_bin=40,
        protected_target_bpm=90.0,
    )

    released = advance_suppressed_protected_shadow(
        state,
        window_id=21,
        window_kind="motion",
        candidate_bins=(38, 41),
        candidate_bpms=(89.0, 91.0),
        would_remove_bins=(38, 41),
    )

    assert released.released_candidate_bin == 41
    assert released.released_candidate_bpm == pytest.approx(91.0)


def test_shadow_owner_requires_a_later_independent_event_before_reacquiring() -> None:
    state = SuppressedProtectedShadowState()
    advance_suppressed_protected_shadow(
        state,
        window_id=20,
        window_kind="motion",
        candidate_bins=(40, 80),
        candidate_bpms=(90.0, 140.0),
        would_remove_bins=(40,),
        protection_suppressed=True,
        protected_target_bin=40,
        protected_target_bpm=90.0,
    )

    released = advance_suppressed_protected_shadow(
        state,
        window_id=21,
        window_kind="motion",
        candidate_bins=(40, 60),
        candidate_bpms=(90.0, 110.0),
        would_remove_bins=(40, 60),
        protection_suppressed=True,
        protected_target_bin=60,
        protected_target_bpm=110.0,
    )
    reacquired = advance_suppressed_protected_shadow(
        state,
        window_id=22,
        window_kind="motion",
        candidate_bins=(60,),
        candidate_bpms=(110.0,),
        would_remove_bins=(60,),
        protection_suppressed=True,
        protected_target_bin=60,
        protected_target_bpm=110.0,
    )

    assert released.owner_event == "release"
    assert released.owner_after_exists is False
    assert reacquired.owner_event == "acquire"
    assert reacquired.owner_origin_window == 22
    assert reacquired.owner_after_exists is True


@pytest.mark.parametrize(
    ("window_kind", "candidate_bins", "candidate_bpms", "would_remove_bins", "reason"),
    [
        ("motion", (50,), (100.0,), (50,), "target_drifted"),
        ("motion", (41,), (92.0,), (), "left_penalty_band"),
        ("recovery", (40,), (90.0,), (40,), "motion_ended"),
    ],
)
def test_shadow_owner_terminal_conditions_close_without_release(
    window_kind: str,
    candidate_bins: tuple[int, ...],
    candidate_bpms: tuple[float, ...],
    would_remove_bins: tuple[int, ...],
    reason: str,
) -> None:
    state = SuppressedProtectedShadowState()
    advance_suppressed_protected_shadow(
        state,
        window_id=30,
        window_kind="motion",
        candidate_bins=(40,),
        candidate_bpms=(90.0,),
        would_remove_bins=(40,),
        protection_suppressed=True,
        protected_target_bin=40,
        protected_target_bpm=90.0,
    )

    terminal = advance_suppressed_protected_shadow(
        state,
        window_id=31,
        window_kind=window_kind,
        candidate_bins=candidate_bins,
        candidate_bpms=candidate_bpms,
        would_remove_bins=would_remove_bins,
    )

    assert terminal.owner_event == "terminal"
    assert terminal.owner_reason == reason
    assert terminal.released_candidate_bin is None
    assert terminal.owner_after_exists is False
    assert state.active is False


def test_shadow_owner_eof_finalization_is_explicit_and_auditable() -> None:
    state = SuppressedProtectedShadowState()
    advance_suppressed_protected_shadow(
        state,
        window_id=40,
        window_kind="motion",
        candidate_bins=(12,),
        candidate_bpms=(88.0,),
        would_remove_bins=(12,),
        protection_suppressed=True,
        protected_target_bin=12,
        protected_target_bpm=88.0,
    )

    terminal = finalize_suppressed_protected_shadow(
        state,
        window_id=41,
        reason="eof",
    )

    assert terminal.owner_event == "terminal"
    assert terminal.owner_reason == "eof"
    assert terminal.owner_origin_window == 40
    assert terminal.owner_origin_bin == 12
    assert terminal.owner_origin_bpm == pytest.approx(88.0)
    assert terminal.owner_after_exists is False
    assert state.active is False


def test_candidate_visibility_preserves_would_remove_provenance() -> None:
    hard = apply_candidate_visibility(
        CandidateVisibilityMode.HARD_EXCLUSION,
        all_peak_indices=(2, 5, 8),
        hard_selectable_indices=(5, 8),
        hard_removed_indices=(2,),
    )
    weighted = apply_candidate_visibility(
        CandidateVisibilityMode.WEIGHTED_VISIBLE,
        all_peak_indices=(2, 5, 8),
        hard_selectable_indices=(5, 8),
        hard_removed_indices=(2,),
    )

    assert hard.selectable_indices == (5, 8)
    assert hard.removed_indices == (2,)
    assert hard.would_remove_indices == (2,)
    assert hard.hard_removal_applied is True
    assert weighted.selectable_indices == (2, 5, 8)
    assert weighted.removed_indices == ()
    assert weighted.would_remove_indices == (2,)
    assert weighted.hard_removal_applied is False


def test_same_window_visibility_releases_only_the_exact_protected_bin() -> None:
    inactive = apply_candidate_visibility(
        CandidateVisibilityMode.SAME_WINDOW_RELEASE,
        all_peak_indices=(2, 5, 8),
        hard_selectable_indices=(8,),
        hard_removed_indices=(2, 5),
    )
    active = apply_candidate_visibility(
        CandidateVisibilityMode.SAME_WINDOW_RELEASE,
        all_peak_indices=(2, 5, 8),
        hard_selectable_indices=(8,),
        hard_removed_indices=(2, 5),
        released_peak_index=5,
    )

    assert inactive.selectable_indices == (8,)
    assert inactive.removed_indices == (2, 5)
    assert active.selectable_indices == (5, 8)
    assert active.removed_indices == (2,)
    assert active.would_remove_indices == (2, 5)
    assert active.hard_removal_applied is True


def test_resolution_adaptive_width_uses_causal_window_resolution_and_bounds() -> None:
    candidate = penalty_candidates_v1()[1]

    ordinary = decide_penalty_policy(
        candidate,
        PenaltyPolicyObservation(
            window_kind="motion",
            configured_half_width_hz=0.2,
            fs_hz=25.0,
            window_samples=200,
        ),
    )
    coarse = decide_penalty_policy(
        candidate,
        PenaltyPolicyObservation(
            window_kind="motion",
            configured_half_width_hz=0.2,
            fs_hz=100.0,
            window_samples=400,
        ),
    )
    fine = decide_penalty_policy(
        candidate,
        PenaltyPolicyObservation(
            window_kind="motion",
            configured_half_width_hz=0.2,
            fs_hz=25.0,
            window_samples=500,
        ),
    )
    same_duration_higher_fs = decide_penalty_policy(
        candidate,
        PenaltyPolicyObservation(
            window_kind="motion",
            configured_half_width_hz=0.2,
            fs_hz=100.0,
            window_samples=800,
        ),
    )

    assert ordinary.resolution_hz == pytest.approx(0.125)
    assert ordinary.effective_half_width_hz == pytest.approx(0.1875)
    assert ordinary.candidate_exclusion_half_width_hz == pytest.approx(
        0.1875 + 1.0 / 60.0
    )
    assert coarse.effective_half_width_hz == pytest.approx(0.30)
    assert fine.effective_half_width_hz == pytest.approx(0.10)
    assert same_duration_higher_fs.effective_half_width_hz == pytest.approx(
        ordinary.effective_half_width_hz
    )
    assert ordinary.width_source == "causal_window_resolution"


def test_trusted_history_corridor_protects_supported_rise_but_not_wrong_track() -> None:
    candidate = penalty_candidates_v1()[2]

    supported_rise = decide_penalty_policy(
        candidate,
        PenaltyPolicyObservation(
            window_kind="motion",
            configured_half_width_hz=0.2,
            fs_hz=50.0,
            window_samples=400,
            previous_track_bpm=126.0,
            recent_track_bpm=(120.0, 123.0, 126.0),
            unpenalized_candidate_bpm=(128.0, 92.0),
            unpenalized_candidate_amp_ratio=(0.70, 1.0),
            motion_reference_confidence=0.8,
            base_corridor_half_width_bpm=7.0,
        ),
    )
    unsupported_wrong_track = decide_penalty_policy(
        candidate,
        PenaltyPolicyObservation(
            window_kind="motion",
            configured_half_width_hz=0.2,
            fs_hz=50.0,
            window_samples=400,
            previous_track_bpm=150.0,
            recent_track_bpm=(150.0, 150.0, 150.0),
            unpenalized_candidate_bpm=(120.0, 90.0),
            unpenalized_candidate_amp_ratio=(1.0, 0.8),
            motion_reference_confidence=0.8,
            base_corridor_half_width_bpm=7.0,
        ),
    )

    assert supported_rise.history_confidence == pytest.approx(0.70)
    assert supported_rise.protection_half_width_bpm == pytest.approx(7.0)
    assert supported_rise.protection_status == "applied_trusted_history"
    assert supported_rise.unpenalized_previous_support_visible is True
    assert unsupported_wrong_track.protection_half_width_bpm is None
    assert unsupported_wrong_track.protection_status == "unsupported_history_track"
    assert (
        unsupported_wrong_track.unpenalized_previous_support_visible is False
    )


def test_penalty_candidates_fail_closed_to_declared_fallbacks() -> None:
    adaptive = decide_penalty_policy(
        penalty_candidates_v1()[1],
        PenaltyPolicyObservation(
            window_kind="motion",
            configured_half_width_hz=0.2,
            fs_hz=0.0,
            window_samples=0,
        ),
    )
    insufficient_history = decide_penalty_policy(
        penalty_candidates_v1()[2],
        PenaltyPolicyObservation(
            window_kind="motion",
            configured_half_width_hz=0.2,
            fs_hz=50.0,
            window_samples=400,
            previous_track_bpm=120.0,
            recent_track_bpm=(120.0, 121.0),
            base_corridor_half_width_bpm=7.0,
        ),
    )
    blocked_reacquire = decide_penalty_policy(
        penalty_candidates_v1()[2],
        PenaltyPolicyObservation(
            window_kind="motion",
            configured_half_width_hz=0.2,
            fs_hz=50.0,
            window_samples=400,
            previous_track_bpm=120.0,
            recent_track_bpm=(118.0, 119.0, 120.0),
            base_corridor_half_width_bpm=7.0,
            recovery_reacquire_active=True,
        ),
    )

    assert adaptive.effective_half_width_hz == pytest.approx(0.2)
    assert adaptive.width_source == "configured_fallback"
    assert insufficient_history.protection_half_width_bpm is None
    assert insufficient_history.protection_status == "insufficient_history"
    assert blocked_reacquire.protection_half_width_bpm is None
    assert (
        blocked_reacquire.protection_status
        == "blocked_by_recovery_reacquire"
    )


def test_runtime_policy_resolves_frozen_penalty_identity() -> None:
    policy = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            penalty_candidate_id="resolution_adaptive_width_v1",
        )
    )

    assert policy.motion_penalty.penalty_id == "resolution_adaptive_width_v1"
    assert policy.motion_penalty.width_mode == "resolution_adaptive"
    assert policy.motion_penalty.corridor_mode == "single_previous_track"


def test_runtime_policy_serializes_opt_in_visibility_without_changing_default() -> None:
    default = runtime_policy_from_config(
        V2RunConfig(data_path=Path("data.csv"), ref_path=Path("ref.csv"))
    )
    candidate = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            penalty_candidate_id="nondestructive_weighted_visible_v1",
        )
    )

    assert "candidate_visibility_mode" not in default.motion_penalty.metadata()
    assert candidate.motion_penalty.metadata()["candidate_visibility_mode"] == (
        "weighted_visible"
    )


def test_penalty_artifact_freeze_records_zero_solver_runs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "penalty_candidates_v1"

    receipt = freeze_penalty_candidate_artifacts(output_dir=output)

    assert receipt["formal_solver_run_count"] == 0
    assert receipt["diagnostic_solver_run_count"] == 0
    assert receipt["independent_bo_run_count"] == 0
    assert receipt["penalty_registry_sha256"]
    assert receipt["source_bundle_sha256"]
    assert (output / "penalty_registry.json").is_file()
    assert (output / "penalty_freeze_receipt.json").is_file()
