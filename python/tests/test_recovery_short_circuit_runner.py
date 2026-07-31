from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPOSITORY_ROOT / "python" / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from recovery_short_circuit_runner import (  # noqa: E402
    AUTHORIZATION_CUTOFF,
    GATE_A_IDENTITY_UPPER_BOUND,
    GATE_B_IDENTITY_UPPER_BOUND,
    HARD_RECORD_ORDER,
    PROPOSAL_VERSION,
    REMAINING_RECOVERY_IDS,
    TOTAL_IDENTITY_UPPER_BOUND,
    USER_AUTHORIZATION_TEXT,
    V13_IDENTITY_LIMIT,
    RecoveryShortCircuitError,
    _candidate_summary,
    _rank_training_candidates,
    _scene_shared_decision,
    _shared_candidates,
    build_authorization,
    validate_authorization,
    validate_proposal,
)

from ppg_hr.v2.recovery_contracts import (  # noqa: E402
    canonical_sha256,
)


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
        "authorization_cutoff": AUTHORIZATION_CUTOFF,
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


def test_authorization_is_bound_to_user_window(
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
    )
    assert receipt["user_authorization_text"] == USER_AUTHORIZATION_TEXT
    assert receipt["expires_at"] == AUTHORIZATION_CUTOFF
    assert receipt["budget_expansion"] == 0
    assert validate_authorization(
        proposal=proposal,
        receipt=receipt,
    ) == receipt


def test_authorization_after_cutoff_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    proposal = _proposal(tmp_path)
    with pytest.raises(
        RecoveryShortCircuitError,
        match="authorization_outside_window",
    ):
        build_authorization(
            proposal=proposal,
            granted_at="2026-07-31T20:00:01+08:00",
        )


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
        mechanism_complexity=2,
    )
    assert summary["status"] == "survivor"
    assert summary["eliminated_at_record_id"] is None
    assert summary["max_selected_eligible_mae"] == 7.0
    assert summary["eligible_candidate_count_sum"] == 21


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
    decision = _scene_shared_decision(
        scene="jianpan",
        folds=folds,
    )
    assert decision["passed"] is False
    assert decision["all_held_out_eligible"] is False
