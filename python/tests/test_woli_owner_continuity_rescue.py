from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2 import solver
from ppg_hr.v2.low_reacquire_candidates import (
    BOUNDED_LOW_OWNER_HARMONIC_SUPPORT_ID,
)
from ppg_hr.v2.runtime_policy import runtime_policy_from_config
from ppg_hr.v2.types import V2RunConfig

LOW_PARAMS = {
    "candidate_id": BOUNDED_LOW_OWNER_HARMONIC_SUPPORT_ID,
    "allow_owned_harmonic_penalty_support": True,
    "require_bounded_catchup": True,
}


def _low_step(
    state: solver.SpectrumReacquireState,
    *,
    previous_bpm: float,
    candidate_bpm: float,
    penalty_centers_bpm: tuple[float, ...],
    params: dict[str, str | bool] | None = LOW_PARAMS,
    window_id: int,
) -> solver.SpectrumReacquireDecision:
    return solver._apply_motion_reacquire(
        freqs=np.asarray([candidate_bpm / 60.0]),
        raw_amps=np.asarray([1.0]),
        raw_order=np.asarray([0]),
        previous_hz=previous_bpm / 60.0,
        legacy_hz=previous_bpm / 60.0,
        state=state,
        enabled=True,
        window_kind="motion",
        params=params,
        penalty_centers_hz=tuple(value / 60.0 for value in penalty_centers_bpm),
        window_id=window_id,
        min_jump_hz=20.0 / 60.0,
        max_jump_hz=40.0 / 60.0,
        min_low_track_drift_hz=2.625 / 60.0,
        max_owner_retreat_hz=3.5 / 60.0,
    )


def test_bounded_low_owner_support_bridges_one_harmonic_penalty_window() -> None:
    state = solver.SpectrumReacquireState(low_lock_count=8)

    first = _low_step(
        state,
        previous_bpm=68.0,
        candidate_bpm=95.2148,
        penalty_centers_bpm=(50.0, 120.0),
        window_id=100,
    )
    supported = _low_step(
        state,
        previous_bpm=69.75,
        candidate_bpm=94.4824,
        penalty_centers_bpm=(50.0, 94.4824),
        window_id=101,
    )
    confirmed = _low_step(
        state,
        previous_bpm=71.5,
        candidate_bpm=93.3838,
        penalty_centers_bpm=(50.0, 120.0),
        window_id=102,
    )

    assert first.mode == "challenge"
    assert supported.mode == "challenge"
    assert supported.observed_candidate_role == "support"
    assert supported.owned_penalty_support_used is True
    assert confirmed.triggered is True
    assert confirmed.owned_penalty_support_used is True
    assert confirmed.evidence_route == "owned_penalty_support_bounded_catchup"
    assert confirmed.candidate_drift_hz * 60.0 == pytest.approx(-1.831, abs=0.002)


def test_low_owner_support_is_opt_in_and_cannot_acquire_an_owner() -> None:
    legacy_state = solver.SpectrumReacquireState(low_lock_count=8)
    _low_step(
        legacy_state,
        previous_bpm=68.0,
        candidate_bpm=95.2148,
        penalty_centers_bpm=(50.0, 120.0),
        params=None,
        window_id=100,
    )
    rejected = _low_step(
        legacy_state,
        previous_bpm=69.75,
        candidate_bpm=94.4824,
        penalty_centers_bpm=(50.0, 94.4824),
        params=None,
        window_id=101,
    )

    no_owner_state = solver.SpectrumReacquireState(low_lock_count=8)
    no_owner = _low_step(
        no_owner_state,
        previous_bpm=69.75,
        candidate_bpm=94.4824,
        penalty_centers_bpm=(50.0, 94.4824),
        window_id=101,
    )

    assert rejected.triggered is False
    assert rejected.reason == "no_qualified_upward_candidate"
    assert no_owner.triggered is False
    assert no_owner.reason == "no_qualified_upward_candidate"


def test_bounded_low_owner_support_rejects_excessive_owner_retreat() -> None:
    state = solver.SpectrumReacquireState(low_lock_count=8)
    _low_step(
        state,
        previous_bpm=68.0,
        candidate_bpm=95.2,
        penalty_centers_bpm=(50.0, 120.0),
        window_id=100,
    )
    _low_step(
        state,
        previous_bpm=69.75,
        candidate_bpm=94.5,
        penalty_centers_bpm=(50.0, 94.5),
        window_id=101,
    )
    rejected = _low_step(
        state,
        previous_bpm=71.5,
        candidate_bpm=91.6,
        penalty_centers_bpm=(50.0, 120.0),
        window_id=102,
    )

    assert rejected.triggered is False
    assert rejected.reason == "insufficient_owned_penalty_catchup_evidence"
    assert rejected.owned_penalty_support_used is True


def _high_step(
    state: solver.SpectrumHighLockEscapeState,
    *,
    candidate_bpm: float,
    penalty_centers_bpm: tuple[float, ...],
    window_id: int,
) -> solver.SpectrumHighLockEscapeDecision:
    policy = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            recovery_candidate_id="bounded_relative_rise_guard_v1",
        )
    )
    return solver._apply_motion_high_lock_escape(
        freqs=np.asarray([candidate_bpm / 60.0]),
        raw_amps=np.asarray([1.0]),
        raw_order=np.asarray([0]),
        previous_hz=130.0 / 60.0,
        legacy_hz=130.0 / 60.0,
        state=state,
        enabled=True,
        params=policy.high_lock_escape.as_solver_params(),
        window_kind="motion",
        selected_peak_rank=0,
        candidate_source="held_previous",
        penalty_centers_hz=tuple(value / 60.0 for value in penalty_centers_bpm),
        protection_applied=False,
        protected_penalty_overlap=False,
        window_id=window_id,
    )


def test_bounded_high_owner_support_confirms_but_never_acquires() -> None:
    state = solver.SpectrumHighLockEscapeState()
    first = _high_step(
        state,
        candidate_bpm=80.0,
        penalty_centers_bpm=(100.0,),
        window_id=126,
    )
    second = _high_step(
        state,
        candidate_bpm=79.0,
        penalty_centers_bpm=(100.0,),
        window_id=127,
    )
    supported = _high_step(
        state,
        candidate_bpm=78.0,
        penalty_centers_bpm=(78.0,),
        window_id=128,
    )

    no_owner = _high_step(
        solver.SpectrumHighLockEscapeState(),
        candidate_bpm=78.0,
        penalty_centers_bpm=(78.0,),
        window_id=128,
    )

    assert first.count == 1
    assert second.count == 2
    assert supported.triggered is True
    assert supported.owned_penalty_support is True
    assert supported.observed_candidate_role == "support"
    assert supported.hr_hz * 60.0 == pytest.approx(110.0)
    assert no_owner.triggered is False
    assert no_owner.mode == "locked"
    assert no_owner.suppressed_reason == "challenger_near_penalty"


def test_runtime_policy_resolves_both_explicit_woli_rescue_identities() -> None:
    policy = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            low_reacquire_candidate_id=(
                BOUNDED_LOW_OWNER_HARMONIC_SUPPORT_ID
            ),
            recovery_candidate_id="bounded_relative_rise_guard_v1",
        )
    )

    assert policy.low_reacquire.candidate_id == (
        BOUNDED_LOW_OWNER_HARMONIC_SUPPORT_ID
    )
    assert policy.high_lock_escape.candidate_min_bpm == pytest.approx(70.0)
    assert policy.high_lock_escape.allow_owned_penalty_support is True


def test_owned_high_lock_action_reaches_final_writer_with_bounded_step() -> None:
    cfg = V2RunConfig(
        data_path=Path("data.csv"),
        ref_path=Path("ref.csv"),
        recovery_candidate_id="bounded_relative_rise_guard_v1",
    )
    source = np.zeros((2, 9), dtype=float)
    source[:, 0] = [127.0, 128.0]
    source[:, 5] = np.asarray([101.07421875, 81.07421875]) / 60.0

    rescued, applied = solver._postprocess_dynamic_final_hr_bpm(
        source,
        np.asarray([True, True]),
        {"start_s": 60.0, "end_s": 130.0},
        cfg,
        runtime_policy=runtime_policy_from_config(cfg),
        high_lock_final_action_mask=np.asarray([False, True]),
        window_stages=["motion", "motion"],
    )
    legacy, _ = solver._postprocess_dynamic_final_hr_bpm(
        source,
        np.asarray([True, True]),
        {"start_s": 60.0, "end_s": 130.0},
        cfg,
        runtime_policy=runtime_policy_from_config(cfg),
        window_stages=["motion", "motion"],
    )

    assert rescued.tolist() == pytest.approx([101.07421875, 81.07421875])
    assert legacy.tolist() == pytest.approx([101.07421875, 99.57421875])
    assert applied == 1


def test_owned_high_lock_final_writer_never_exceeds_candidate_down_step() -> None:
    cfg = V2RunConfig(
        data_path=Path("data.csv"),
        ref_path=Path("ref.csv"),
        recovery_candidate_id="bounded_relative_rise_guard_v1",
    )
    source = np.zeros((2, 9), dtype=float)
    source[:, 5] = np.asarray([130.0, 70.0]) / 60.0

    rescued, _ = solver._postprocess_dynamic_final_hr_bpm(
        source,
        np.asarray([True, True]),
        {"start_s": 0.0, "end_s": 10.0},
        cfg,
        runtime_policy=runtime_policy_from_config(cfg),
        high_lock_final_action_mask=np.asarray([False, True]),
        window_stages=["motion", "motion"],
    )

    assert rescued.tolist() == pytest.approx([130.0, 110.0])
