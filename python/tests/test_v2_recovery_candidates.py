from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ppg_hr.v2.recovery_candidate_freeze import (
    freeze_recovery_candidate_artifacts,
    freeze_recovery_candidate_registry,
    runtime_source_identity,
)
from ppg_hr.v2.recovery_candidates import (
    RecoveryCandidateError,
    RecoveryObservation,
    RecoveryStateMachine,
    recovery_candidates_v1,
)
from ppg_hr.v2.recovery_selection import (
    RecoveryCandidateEvaluation,
    RecoveryPanelRecord,
    RecoveryRecordEvaluation,
    recovery_selection_contract_v1,
    select_recovery_candidate_evaluations,
)
from ppg_hr.v2.runtime_policy import runtime_policy_from_config
from ppg_hr.v2.types import V2RunConfig


def test_recovery_registry_freezes_control_and_two_predeclared_candidates() -> None:
    registry = freeze_recovery_candidate_registry(
        recovery_candidates_v1(),
        solver_hash="a" * 64,
        config_schema_hash="b" * 64,
    )

    assert [item["candidate_id"] for item in registry["candidates"]] == [
        "current_fixed_floor_control_v1",
        "relative_gap_timeout_v1",
        "relative_gap_rise_guard_v1",
    ]
    assert registry["candidate_count"] == 3
    assert registry["control_candidate_id"] == "current_fixed_floor_control_v1"
    assert registry["new_candidate_count"] == 2
    assert registry["formal_solver_run_count"] == 0
    assert registry["uses_reference_hr_online"] is False
    assert len(registry["registry_sha256"]) == 64
    assert all(len(item["candidate_sha256"]) == 64 for item in registry["candidates"])


def test_recovery_registry_records_complete_state_machine_contracts() -> None:
    candidates = recovery_candidates_v1()

    for candidate in candidates:
        payload = candidate.to_dict()
        assert payload["formula"]
        assert payload["constants"]
        assert payload["states"] == [
            "locked",
            "challenge",
            "reacquiring",
            "cooldown",
        ]
        assert payload["confirmation_rule"]
        assert payload["challenge_timeout_rule"]
        assert payload["reacquire_timeout_rule"]
        assert payload["failure_exit_rules"]
        assert payload["cooldown_rule"]
        assert payload["true_rise_protection_rule"]
        assert payload["online_evidence_fields"]
        assert not {
            "reference_hr",
            "offline_error",
            "other_algorithm_output",
        }.intersection(payload["online_evidence_fields"])


def test_recovery_registry_rejects_fourth_candidate() -> None:
    candidates = recovery_candidates_v1()

    with pytest.raises(RecoveryCandidateError, match="candidate_count_must_be_three"):
        freeze_recovery_candidate_registry(
            (*candidates, candidates[-1]),
            solver_hash="a" * 64,
            config_schema_hash="b" * 64,
        )


def test_recovery_artifact_freeze_binds_source_and_records_zero_runs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "recovery_candidates_v1"

    receipt = freeze_recovery_candidate_artifacts(output_dir=output_dir)

    registry = (
        output_dir / "recovery_candidate_registry.json"
    ).read_text(encoding="utf-8")
    selection = (
        output_dir / "recovery_selection_contract.json"
    ).read_text(encoding="utf-8")
    assert receipt["status"] == "frozen_zero_formal_runs"
    assert receipt["formal_solver_run_count"] == 0
    assert receipt["diagnostic_solver_run_count"] == 0
    assert receipt["independent_bo_run_count"] == 0
    assert receipt["source_files"]
    assert "ppg_hr/core/fft_peaks.py" in receipt["source_files"]
    assert "ppg_hr/params.py" in receipt["source_files"]
    assert "ppg_hr/v2/algorithm_presets.py" in receipt["source_files"]
    assert "ppg_hr/v2/preprocess.py" in receipt["source_files"]
    assert "ppg_hr/v2/post_motion_dual_reset.py" in receipt["source_files"]
    assert (
        "ppg_hr/v2/post_motion_minimal_handoff.py"
        in receipt["source_files"]
    )
    assert len(receipt["source_bundle_sha256"]) == 64
    assert len(receipt["config_schema_sha256"]) == 64
    assert set(receipt["config_schemas"]) == {"SolverParams", "V2RunConfig"}
    assert receipt["preflight_status"] == "awaiting_full_preflight_contracts"
    assert len(receipt["required_preflight_hashes"]) == 7
    assert receipt["artifacts"]["recovery_candidate_registry.json"]
    assert receipt["artifacts"]["recovery_selection_contract.json"]
    assert '"candidate_count": 3' in registry
    assert '"no_candidate_state": "no_safe_recovery_candidate"' in selection

    with pytest.raises(
        RecoveryCandidateError,
        match="recovery_candidate_freeze_output_already_exists",
    ):
        freeze_recovery_candidate_artifacts(output_dir=output_dir)


def test_recovery_selection_contract_is_fail_closed_and_lexicographic() -> None:
    contract = recovery_selection_contract_v1()

    assert contract["per_record_hard_gates"] == [
        "spectral_gate_contract_v1",
        "l10_engineering_gate",
        "l20_engineering_gate",
        "mae_independent_delta_le_2_bpm",
        "no_new_right_censored_recovery",
        "true_rise_underestimate_delta_le_2_bpm",
        "current_l10_catastrophic_regression_gate",
        "mae_current_delta_le_2_bpm",
        "loo_training_pair_mean_independent_mae_delta_le_1_bpm",
    ]
    assert contract["ranking_key"] == [
        "worst_l10",
        "right_censored_recovery_count",
        "worst_recovery_delay",
        "worst_mae",
        "mean_mae",
        "mechanism_complexity",
        "candidate_id",
    ]
    assert contract["tie_rule"] == "candidate_id_ascending"
    assert contract["no_candidate_state"] == "no_safe_recovery_candidate"
    assert contract["single_candidate_rollback_backup_id"] is None
    assert contract["evaluation_panel_contract"]["record_count"] == 12
    assert contract["evaluation_panel_contract"]["sentinel_count"] == 3
    assert (
        contract["evaluation_panel_contract"]["result_count_per_candidate"]
        == 36
    )
    assert contract["formal_solver_run_count"] == 0
    assert len(contract["contract_sha256"]) == 64


def _record_evaluation(
    *,
    record_id: str,
    sentinel_id: str,
    scene: str,
    l10: float = 8.0,
    l20: float = 1.0,
    mae: float = 4.0,
    recovery_delay: float = 2.0,
    right_censored: int = 0,
    true_rise_underestimate: float | None = 1.0,
) -> RecoveryRecordEvaluation:
    return RecoveryRecordEvaluation(
        record_id=record_id,
        sentinel_id=sentinel_id,
        scene=scene,
        spectral_gate_passed=True,
        l10=l10,
        l20=l20,
        mae=mae,
        independent_l10=8.0,
        independent_l20=1.0,
        independent_mae=3.5,
        current_l10=8.0,
        current_mae=3.5,
        recovery_delay=recovery_delay,
        right_censored_recovery_count=right_censored,
        current_right_censored_recovery_count=0,
        true_rise_underestimate=true_rise_underestimate,
        current_true_rise_underestimate=(
            None if true_rise_underestimate is None else 1.0
        ),
    )


_SELECTION_RECORDS = (
    ("run1", "run"),
    ("run2", "run"),
    ("run3", "run"),
    ("kaihe1", "kaihe"),
    ("kaihe2", "kaihe"),
    ("kaihe3", "kaihe"),
    ("xiezi1", "xiezi"),
    ("xiezi2", "xiezi"),
    ("xiezi3", "xiezi"),
    ("jianpan1", "jianpan"),
    ("jianpan2", "jianpan"),
    ("jianpan3", "jianpan"),
)
_SELECTION_PANEL = tuple(
    RecoveryPanelRecord(
        record_id=record_id,
        scene=scene,
        true_rise_applicable=(
            scene in {"run", "kaihe"} and record_id != "kaihe3"
        ),
    )
    for record_id, scene in _SELECTION_RECORDS
)
_SELECTION_SENTINEL_IDS = ("conservative", "middle", "aggressive")


def _candidate_evaluation(
    *,
    candidate_id: str,
    mechanism_complexity: int,
    l10: float,
    bad_coordinate: tuple[str, str] | None = None,
    right_censored: int = 0,
) -> RecoveryCandidateEvaluation:
    return RecoveryCandidateEvaluation(
        candidate_id=candidate_id,
        mechanism_complexity=mechanism_complexity,
        records=tuple(
            _record_evaluation(
                record_id=record_id,
                sentinel_id=sentinel_id,
                scene=scene,
                l10=(
                    21.0
                    if bad_coordinate == (sentinel_id, record_id)
                    else l10
                ),
                right_censored=right_censored,
                true_rise_underestimate=(
                    1.0
                    if scene in {"run", "kaihe"} and record_id != "kaihe3"
                    else None
                ),
            )
            for sentinel_id in _SELECTION_SENTINEL_IDS
            for record_id, scene in _SELECTION_RECORDS
        ),
    )


def test_recovery_selector_applies_hard_gates_before_lexicographic_rank() -> None:
    evaluations = (
        _candidate_evaluation(
            candidate_id="current_fixed_floor_control_v1",
            mechanism_complexity=0,
            l10=9.0,
        ),
        _candidate_evaluation(
            candidate_id="relative_gap_timeout_v1",
            mechanism_complexity=1,
            l10=7.0,
        ),
        _candidate_evaluation(
            candidate_id="relative_gap_rise_guard_v1",
            mechanism_complexity=2,
            l10=6.0,
            bad_coordinate=("middle", "run2"),
        ),
    )

    decision = select_recovery_candidate_evaluations(
        evaluations,
        expected_records=_SELECTION_PANEL,
        expected_sentinel_ids=_SELECTION_SENTINEL_IDS,
    )

    assert decision["status"] == "selected"
    assert decision["provisional_recovery_id"] == "relative_gap_timeout_v1"
    assert decision["rollback_backup_id"] == "current_fixed_floor_control_v1"
    assert decision["eligible_candidate_ids"] == [
        "relative_gap_timeout_v1",
        "current_fixed_floor_control_v1",
    ]
    assert decision["eliminated_candidates"] == {
        "relative_gap_rise_guard_v1": [
            "middle/run2:l10_engineering_gate",
            "middle/run2:current_l10_catastrophic_regression_gate",
        ]
    }


def test_recovery_selector_fails_closed_when_no_candidate_is_safe() -> None:
    unsafe = tuple(
        _candidate_evaluation(
            candidate_id=candidate.candidate_id,
            mechanism_complexity=candidate.mechanism_complexity,
            l10=8.0,
            right_censored=1,
        )
        for candidate in recovery_candidates_v1()
    )

    decision = select_recovery_candidate_evaluations(
        unsafe,
        expected_records=_SELECTION_PANEL,
        expected_sentinel_ids=_SELECTION_SENTINEL_IDS,
    )

    assert decision["status"] == "no_safe_recovery_candidate"
    assert decision["provisional_recovery_id"] is None
    assert decision["rollback_backup_id"] is None


def test_recovery_selector_rejects_incomplete_sentinel_record_panel() -> None:
    complete = tuple(
        _candidate_evaluation(
            candidate_id=candidate.candidate_id,
            mechanism_complexity=candidate.mechanism_complexity,
            l10=8.0,
        )
        for candidate in recovery_candidates_v1()
    )
    incomplete = (
        RecoveryCandidateEvaluation(
            candidate_id=complete[0].candidate_id,
            mechanism_complexity=complete[0].mechanism_complexity,
            records=complete[0].records[:-1],
        ),
        *complete[1:],
    )

    with pytest.raises(
        RecoveryCandidateError,
        match="recovery_selection_panel_coordinate_mismatch",
    ):
        select_recovery_candidate_evaluations(
            incomplete,
            expected_records=_SELECTION_PANEL,
            expected_sentinel_ids=_SELECTION_SENTINEL_IDS,
        )


def test_recovery_selector_rejects_scene_identity_or_na_reclassification() -> None:
    complete = tuple(
        _candidate_evaluation(
            candidate_id=candidate.candidate_id,
            mechanism_complexity=candidate.mechanism_complexity,
            l10=8.0,
        )
        for candidate in recovery_candidates_v1()
    )
    changed_scene_records = (
        replace(complete[0].records[0], scene="xiezi"),
        *complete[0].records[1:],
    )
    changed_scene = (
        replace(complete[0], records=changed_scene_records),
        *complete[1:],
    )
    with pytest.raises(
        RecoveryCandidateError,
        match="recovery_selection_record_scene_mismatch",
    ):
        select_recovery_candidate_evaluations(
            changed_scene,
            expected_records=_SELECTION_PANEL,
            expected_sentinel_ids=_SELECTION_SENTINEL_IDS,
        )

    hidden_rise_records = (
        replace(
            complete[0].records[0],
            true_rise_underestimate=None,
            current_true_rise_underestimate=None,
        ),
        *complete[0].records[1:],
    )
    hidden_rise = (
        replace(complete[0], records=hidden_rise_records),
        *complete[1:],
    )
    with pytest.raises(
        RecoveryCandidateError,
        match="recovery_selection_true_rise_applicability_mismatch",
    ):
        select_recovery_candidate_evaluations(
            hidden_rise,
            expected_records=_SELECTION_PANEL,
            expected_sentinel_ids=_SELECTION_SENTINEL_IDS,
        )


def test_runtime_source_identity_tracks_ancestor_package_initializers(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src"
    package = source_root / "ppg_hr"
    v2_package = package / "v2"
    v2_package.mkdir(parents=True)
    (package / "__init__.py").write_text("ROOT = 1\n", encoding="utf-8")
    (v2_package / "__init__.py").write_text("V2 = 1\n", encoding="utf-8")
    for module_name in (
        "recovery_candidates",
        "runtime_policy",
        "solver",
        "spectrum_tracking",
        "types",
    ):
        (v2_package / f"{module_name}.py").write_text(
            "from ppg_hr import ROOT\n",
            encoding="utf-8",
        )

    before = runtime_source_identity(source_root)
    (v2_package / "__init__.py").write_text("V2 = 2\n", encoding="utf-8")
    after = runtime_source_identity(source_root)

    assert "ppg_hr/__init__.py" in before["source_files"]
    assert "ppg_hr/v2/__init__.py" in before["source_files"]
    assert before["source_bundle_sha256"] != after["source_bundle_sha256"]


def _observation(
    *,
    current_track_bpm: float = 180.0,
    current_track_delta_bpm: float = 0.0,
    challenger_bpm: float | None = 120.0,
    challenger_stability_bpm: float = 0.0,
) -> RecoveryObservation:
    return RecoveryObservation(
        window_kind="motion",
        current_track_bpm=current_track_bpm,
        current_track_delta_bpm=current_track_delta_bpm,
        challenger_bpm=challenger_bpm,
        challenger_amp_ratio=0.8,
        challenger_stability_bpm=challenger_stability_bpm,
        high_lock_risk_labels=("held_previous",),
        challenger_near_penalty=False,
    )


def test_relative_gap_candidate_confirms_then_reacquires_without_reference_hr() -> None:
    candidate = recovery_candidates_v1()[1]
    machine = RecoveryStateMachine(candidate)

    decisions = [machine.step(_observation()) for _ in range(3)]

    assert decisions[0].mode == "challenge"
    assert decisions[0].output_bpm == pytest.approx(180.0)
    assert decisions[2].triggered is True
    assert decisions[2].mode == "reacquiring"
    assert decisions[2].output_bpm == pytest.approx(160.0)
    assert decisions[2].candidate_id == "relative_gap_timeout_v1"
    assert decisions[2].trace["uses_reference_hr_online"] is False


def test_rate_guard_candidate_protects_causal_true_rise() -> None:
    candidate = recovery_candidates_v1()[2]
    machine = RecoveryStateMachine(candidate)

    decisions = [
        machine.step(_observation(current_track_delta_bpm=2.0))
        for _ in range(3)
    ]

    assert all(decision.triggered is False for decision in decisions)
    assert decisions[-1].output_bpm == pytest.approx(180.0)
    assert decisions[-1].suppressed_reason == "physiological_rise_guard"
    assert decisions[-1].trace["true_rise_guard"] is True


def test_relative_candidate_times_out_and_enters_cooldown() -> None:
    candidate = recovery_candidates_v1()[1]
    machine = RecoveryStateMachine(candidate)
    alternating = [120.0, 135.0, 120.0, 135.0, 120.0]

    decisions = [
        machine.step(
            _observation(
                challenger_bpm=value,
            )
        )
        for value in alternating
    ]

    assert decisions[-1].mode == "cooldown"
    assert decisions[-1].suppressed_reason == "challenge_timeout"
    follow_up = machine.step(_observation())
    assert follow_up.mode == "cooldown"
    assert follow_up.suppressed_reason == "cooldown"


def test_reacquire_candidate_loss_exits_safely() -> None:
    machine = RecoveryStateMachine(recovery_candidates_v1()[1])
    for _ in range(3):
        triggered = machine.step(_observation())
    assert triggered.triggered is True

    exited = machine.step(_observation(challenger_bpm=None))

    assert exited.mode == "cooldown"
    assert exited.output_bpm == pytest.approx(180.0)
    assert exited.suppressed_reason == "candidate_lost"
    assert exited.exit_from_mode == "reacquiring"
    assert exited.exit_age == 1
    assert exited.timeout_windows == 8


def test_recovery_target_exit_preserves_pre_exit_trace() -> None:
    machine = RecoveryStateMachine(recovery_candidates_v1()[1])
    for _ in range(3):
        machine.step(_observation())

    reached = machine.step(
        _observation(
            current_track_bpm=140.0,
            challenger_bpm=120.0,
        )
    )

    assert reached.output_bpm == pytest.approx(120.0)
    assert reached.mode == "cooldown"
    assert reached.suppressed_reason == "target_reached"
    assert reached.exit_from_mode == "reacquiring"
    assert reached.exit_age == 2
    assert reached.timeout_windows == 8


def test_reacquire_support_cannot_ratchet_the_confirmed_target() -> None:
    machine = RecoveryStateMachine(recovery_candidates_v1()[1])
    for _ in range(3):
        triggered = machine.step(_observation(challenger_bpm=120.0))
    assert triggered.challenger_bpm == pytest.approx(120.0)

    supported = machine.step(
        _observation(
            current_track_bpm=160.0,
            challenger_bpm=128.0,
            challenger_stability_bpm=8.0,
        )
    )

    assert supported.mode == "reacquiring"
    assert supported.output_bpm == pytest.approx(140.0)
    assert supported.challenger_bpm == pytest.approx(120.0)


def test_no_challenger_never_starts_recovery() -> None:
    machine = RecoveryStateMachine(recovery_candidates_v1()[1])

    decision = machine.step(_observation(challenger_bpm=None))

    assert decision.mode == "locked"
    assert decision.triggered is False
    assert decision.suppressed_reason == "candidate_lost"


def test_runtime_policy_resolves_frozen_recovery_candidate_identity() -> None:
    policy = runtime_policy_from_config(
        V2RunConfig(
            data_path=Path("data.csv"),
            ref_path=Path("ref.csv"),
            recovery_candidate_id="relative_gap_rise_guard_v1",
        )
    )

    assert policy.high_lock_escape.candidate_id == "relative_gap_rise_guard_v1"
    assert policy.high_lock_escape.gate_mode == "relative_gap_rise_guard"
    assert policy.high_lock_escape.relative_gap_ratio == pytest.approx(0.15)
    assert policy.high_lock_escape.rise_guard_bpm_per_window == pytest.approx(1.5)
    assert policy.high_lock_escape.challenge_timeout_windows == 6
    assert policy.high_lock_escape.reacquire_timeout_windows == 8
    assert policy.high_lock_escape.candidate_min_bpm is None


def test_runtime_policy_rejects_unknown_recovery_candidate() -> None:
    with pytest.raises(
        RecoveryCandidateError,
        match="unknown_recovery_candidate",
    ):
        runtime_policy_from_config(
            V2RunConfig(
                data_path=Path("data.csv"),
                ref_path=Path("ref.csv"),
                recovery_candidate_id="unknown",
            )
        )
