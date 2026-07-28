from __future__ import annotations

from pathlib import Path

import pytest

from ppg_hr.v2.penalty_candidate_freeze import (
    freeze_penalty_candidate_artifacts,
)
from ppg_hr.v2.penalty_candidates import (
    PenaltyPolicyObservation,
    decide_penalty_policy,
    penalty_candidates_v1,
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
