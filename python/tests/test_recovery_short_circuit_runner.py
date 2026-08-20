from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPOSITORY_ROOT / "python" / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import recovery_short_circuit_runner as short_circuit  # noqa: E402
from recovery_short_circuit_runner import (  # noqa: E402
    AUTHORIZATION_SCOPE_END,
    GATE_A_IDENTITY_UPPER_BOUND,
    GATE_B_IDENTITY_UPPER_BOUND,
    HARD_RECORD_ORDER,
    PROPOSAL_VERSION,
    REMAINING_RECOVERY_IDS,
    TOTAL_IDENTITY_UPPER_BOUND,
    USER_AUTHORIZATION_TEXT,
    V13_IDENTITY_LIMIT,
    RecoveryShortCircuitError,
    _assert_amendment_identity_capacity,
    _candidate_summary,
    _rank_training_candidates,
    _scene_selector_replay_decision,
    _shared_candidates,
    _validate_freeze_binding,
    _validate_reveal_binding,
    build_authorization,
    validate_authorization,
    validate_proposal,
)

from ppg_hr.v2.recovery_contracts import (  # noqa: E402
    canonical_sha256,
)

V2_RUNNER_PATH = (
    TOOLS_ROOT
    / "recovery_short_circuit_v2"
    / "recovery_short_circuit_runner.py"
)
V2_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "recovery_short_circuit_runner_v2",
    V2_RUNNER_PATH,
)
assert V2_RUNNER_SPEC is not None
assert V2_RUNNER_SPEC.loader is not None
short_circuit_v2 = importlib.util.module_from_spec(V2_RUNNER_SPEC)
V2_RUNNER_SPEC.loader.exec_module(short_circuit_v2)


def _proposal(tmp_path: Path) -> dict[str, object]:
    spec = tmp_path / "spec.md"
    scheduler = tmp_path / "scheduler.py"
    spec.write_text("spec", encoding="utf-8")
    scheduler.write_text("scheduler", encoding="utf-8")
    from ppg_hr.v2.experiment_freeze_utils import file_sha256

    proposal: dict[str, object] = {
        "proposal_version": PROPOSAL_VERSION,
        "remaining_recovery_ids": list(REMAINING_RECOVERY_IDS),
        "hard_record_order": list(HARD_RECORD_ORDER),
        "gate_a_stop_rule": "first_zero_eligible_candidate_count",
        "gate_b_admit_count": 1,
        "gate_b_fs_target": 25,
        "gate_b_grid_size_per_record": 100,
        "authorization_scope_end": AUTHORIZATION_SCOPE_END,
        "direct_repair_proposal_sha256": "d" * 64,
        "direct_repair_authorization_sha256": "e" * 64,
        "spec_path": "spec.md",
        "spec_file_sha256": file_sha256(spec),
        "scheduler_path": "scheduler.py",
        "scheduler_file_sha256": file_sha256(scheduler),
        "identity_budget": {
            "combined_upper_bound": TOTAL_IDENTITY_UPPER_BOUND,
            "v13_stage_limit": V13_IDENTITY_LIMIT,
            "budget_expansion": 0,
            "automatic_retry": False,
        },
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def _completion(
    *,
    record_id: str,
    eligible_count: int,
    mae: float,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "cell_sha256": "a" * 64,
        "completion_sha256": "b" * 64,
        "eligible_candidate_count": eligible_count,
        "selected": {
            "eligible": eligible_count > 0,
            "metrics": {"final_motion_mae_bpm": mae},
        },
    }


def test_budget_is_strictly_below_existing_v13_limit() -> None:
    assert GATE_A_IDENTITY_UPPER_BOUND == 1800
    assert GATE_B_IDENTITY_UPPER_BOUND == 1200
    assert TOTAL_IDENTITY_UPPER_BOUND == 4350
    assert TOTAL_IDENTITY_UPPER_BOUND < V13_IDENTITY_LIMIT


def test_authorization_is_bound_until_experiment_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    proposal = _proposal(tmp_path)
    validate_proposal(
        proposal=proposal,
        repository_root=tmp_path,
    )
    receipt = build_authorization(
        proposal=proposal,
        granted_at="2026-07-31T16:30:00+08:00",
        repository_root=tmp_path,
    )
    assert receipt["user_authorization_text"] == USER_AUTHORIZATION_TEXT
    assert (
        receipt["authorization_scope_end"]
        == AUTHORIZATION_SCOPE_END
    )
    assert receipt["budget_expansion"] == 0
    assert validate_authorization(
        proposal=proposal,
        receipt=receipt,
    ) == receipt


def test_authorization_requires_timezone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    proposal = _proposal(tmp_path)
    with pytest.raises(
        RecoveryShortCircuitError,
        match="authorization_time_invalid",
    ):
        build_authorization(
            proposal=proposal,
            granted_at="2026-07-31T21:50:00",
            repository_root=tmp_path,
        )


def test_authorized_receipt_remains_valid_after_old_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    proposal = _proposal(tmp_path)
    receipt = build_authorization(
        proposal=proposal,
        granted_at="2026-07-31T16:30:00+08:00",
        repository_root=tmp_path,
    )
    assert validate_authorization(
        proposal=proposal,
        receipt=receipt,
    ) == receipt


def test_source_drift_is_rejected(tmp_path: Path) -> None:
    proposal = _proposal(tmp_path)
    (tmp_path / "scheduler.py").write_text(
        "changed",
        encoding="utf-8",
    )
    with pytest.raises(
        RecoveryShortCircuitError,
        match="scheduler_path_drift",
    ):
        validate_proposal(
            proposal=proposal,
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("reporting_proposal_sha256", "expected_validator"),
    [
        ("d" * 64, "direct"),
        ("c" * 64, "historical"),
    ],
)
def test_bound_completion_routes_to_matching_repair_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reporting_proposal_sha256: str,
    expected_validator: str,
) -> None:
    completion_path = tmp_path / "cell_completion.json"
    completion_path.write_text("{}", encoding="utf-8")
    completion = {
        "reporting_repair": {
            "repair_proposal_sha256": reporting_proposal_sha256,
        }
    }
    calls: list[str] = []
    monkeypatch.setattr(
        short_circuit_v2,
        "read_json",
        lambda _path: completion,
    )
    monkeypatch.setattr(
        short_circuit_v2,
        "validate_existing_direct_completion",
        lambda **_kwargs: calls.append("direct"),
    )
    monkeypatch.setattr(
        short_circuit_v2,
        "_validate_existing_completion",
        lambda **_kwargs: calls.append("historical"),
    )

    result = short_circuit_v2._validate_bound_completion(
        completion_path=completion_path,
        cell={"cell_sha256": "a" * 64},
        original={"proposal_sha256": "b" * 64},
        repair_proposal={"proposal_sha256": "c" * 64},
        repair_authorization={},
        direct_repair_proposal={"proposal_sha256": "d" * 64},
        direct_repair_authorization={},
        repository_root=tmp_path,
    )

    assert result == completion
    assert calls == [expected_validator]


def test_candidate_stops_at_first_zero_eligible_cell() -> None:
    completions = [
        _completion(
            record_id=HARD_RECORD_ORDER[0],
            eligible_count=3,
            mae=4.0,
        ),
        _completion(
            record_id=HARD_RECORD_ORDER[1],
            eligible_count=0,
            mae=12.0,
        ),
    ]
    summary = _candidate_summary(
        recovery_id=REMAINING_RECOVERY_IDS[0],
        completions=completions,
        fs25_eligible_counts=(1, 0),
        mechanism_complexity=1,
    )
    assert summary["status"] == "eliminated"
    assert summary["completed_hard_cell_count"] == 2
    assert (
        summary["eliminated_at_record_id"]
        == HARD_RECORD_ORDER[1]
    )


def test_candidate_survives_only_after_all_six_cells() -> None:
    completions = [
        _completion(
            record_id=record_id,
            eligible_count=index + 1,
            mae=2.0 + index,
        )
        for index, record_id in enumerate(HARD_RECORD_ORDER)
    ]
    summary = _candidate_summary(
        recovery_id=REMAINING_RECOVERY_IDS[1],
        completions=completions,
        fs25_eligible_counts=(1, 2, 3, 4, 5, 6),
        mechanism_complexity=2,
    )
    assert summary["status"] == "survivor"
    assert summary["eliminated_at_record_id"] is None
    assert summary["max_selected_eligible_mae"] == 7.0
    assert summary["eligible_candidate_count_sum"] == 21
    assert summary["fs25_ready_for_selector_replay"] is True


def test_shared_grid_is_exact_fs25_hundred() -> None:
    candidates = _shared_candidates()
    assert len(candidates) == 100
    assert {
        item.requested_params["fs_target"]
        for item in candidates
    } == {25}
    assert len(
        {
            (
                item.requested_params["memory_ms"],
                item.requested_params["mu_base"],
                item.requested_params[
                    "exclusion_half_width_bpm"
                ],
            )
            for item in candidates
        }
    ) == 100


def test_training_ranking_uses_worst_record_before_neighbors() -> None:
    candidates = _shared_candidates()[:2]
    records = ("train_a", "train_b")
    rows: dict[tuple[str, str], dict[str, object]] = {}
    maes = ((2.0, 4.0), (3.0, 3.5))
    for candidate, pair in zip(candidates, maes, strict=True):
        for record_id, mae in zip(records, pair, strict=True):
            rows[(record_id, candidate.candidate_id)] = {
                "eligible": True,
                "identity_sha256": "a" * 64,
                "metrics": {
                    "final_motion_mae_bpm": mae,
                    "longest_e10_run_windows": 0,
                    "longest_e20_run_windows": 0,
                },
            }
    ranked = _rank_training_candidates(
        candidates=candidates,
        train_record_ids=records,
        rows=rows,
    )
    assert ranked[0]["candidate_id"] == candidates[1].candidate_id
    assert ranked[0]["worst_train_mae"] == 3.5


def test_scene_decision_requires_all_three_eligible_folds() -> None:
    folds = [
        {
            "status": "revealed",
            "selection": {"worst_train_mae": 2.0},
            "held_out": {
                "eligible": index != 2,
                "mae": 2.5,
                "independent_bo_lite_mae": 2.0,
            },
        }
        for index in range(3)
    ]
    decision = _scene_selector_replay_decision(
        scene="jianpan",
        folds=folds,
    )
    assert decision["met_reference_lines"] is False
    assert decision["all_held_out_eligible"] is False


def test_existing_registry_above_amendment_cap_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        short_circuit,
        "_recovery_stage_identity_count",
        lambda _registry: TOTAL_IDENTITY_UPPER_BOUND + 1,
    )
    with pytest.raises(
        RecoveryShortCircuitError,
        match="amendment_identity_budget_exceeded",
    ):
        _assert_amendment_identity_capacity(
            object(),
            additional_upper_bound=0,
        )


def test_freeze_and_reveal_reject_semantic_rebinding() -> None:
    amendment = {"proposal_sha256": "p" * 64}
    gate_a = {"completion_sha256": "g" * 64}
    selection = {
        "candidate_id": "candidate-1",
        "requested_params": {"fs_target": 25},
    }
    freeze: dict[str, object] = {
        "status": "parameter_frozen",
        "proposal_sha256": amendment["proposal_sha256"],
        "gate_a_completion_sha256": gate_a["completion_sha256"],
        "training_candidate_audit_sha256": "a" * 64,
        "recovery_candidate_id": REMAINING_RECOVERY_IDS[0],
        "fold_id": "run__held_out__run3",
        "scene": "run",
        "train_record_ids": ["run1", "run2"],
        "held_out_record_id": "run3",
        "candidate_count": 100,
        "selected_candidate_id": "candidate-1",
        "selection": selection,
    }
    freeze["receipt_sha256"] = canonical_sha256(freeze)
    _validate_freeze_binding(
        freeze,
        amendment=amendment,
        gate_a=gate_a,
        recovery_id=REMAINING_RECOVERY_IDS[0],
        fold_id="run__held_out__run3",
        scene="run",
        train_ids=["run1", "run2"],
        held_out_id="run3",
        training_audit_sha256="a" * 64,
        expected_status="parameter_frozen",
        expected_selection=selection,
    )
    reveal: dict[str, object] = {
        "status": "revealed",
        "proposal_sha256": amendment["proposal_sha256"],
        "freeze_receipt_sha256": freeze["receipt_sha256"],
        "fold_id": "run__held_out__run3",
        "scene": "run",
        "train_record_ids": ["run1", "run2"],
        "held_out_record_id": "run3",
        "selection": selection,
        "held_out": {"identity_sha256": "i" * 64},
        "revealed_at": "2026-07-31T17:00:00+08:00",
    }
    reveal["receipt_sha256"] = canonical_sha256(reveal)
    _validate_reveal_binding(
        reveal,
        amendment=amendment,
        freeze=freeze,
        fold_id="run__held_out__run3",
        scene="run",
        train_ids=["run1", "run2"],
        held_out_id="run3",
        expected_held_identity_sha256="i" * 64,
    )
    rebound = dict(reveal)
    rebound["selection"] = {
        "candidate_id": "candidate-2",
        "requested_params": {"fs_target": 25},
    }
    rebound.pop("receipt_sha256")
    rebound["receipt_sha256"] = canonical_sha256(rebound)
    with pytest.raises(
        RecoveryShortCircuitError,
        match="reveal_binding_drift",
    ):
        _validate_reveal_binding(
            rebound,
            amendment=amendment,
            freeze=freeze,
            fold_id="run__held_out__run3",
            scene="run",
            train_ids=["run1", "run2"],
            held_out_id="run3",
            expected_held_identity_sha256="i" * 64,
        )
