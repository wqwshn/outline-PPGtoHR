from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppg_hr.v2.recovery_independent_bo_experiment import (
    RecoveryIndependentBOAuthorizationError,
    build_recovery_independent_bo_identity,
    build_recovery_independent_bo_proposal,
    validate_recovery_independent_bo_preflight,
    validate_recovery_independent_bo_execution_authorization,
)
from ppg_hr.v2.bo_space_generalization import build_bo_search_space
from ppg_hr.v2.recovery_contracts import canonical_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "experiments"
    / "lyx_recovery_filter_profile"
)


def _read(relative: str) -> dict[str, object]:
    return json.loads(
        (EXPERIMENT_ROOT / relative).read_text(encoding="utf-8")
    )


def _proposal() -> dict[str, object]:
    return build_recovery_independent_bo_proposal(
        stage_r_proposal=_read(
            "stage_r_rank1_replan_v2/stage_r_rank1_replan_proposal.json"
        ),
        stage_r_completion=_read(
            "stage_r_rank1_replan_execution_v2/completion.json"
        ),
        stage_r_selection=_read(
            "stage_r_rank1_replan_execution_v2/recovery_selection.json"
        ),
        stage_r_result_index=_read(
            "stage_r_rank1_replan_execution_v2/identity_result_index.json"
        ),
        repository_root=REPOSITORY_ROOT,
    )


def test_zero_run_proposal_freezes_exact_36_cell_5400_identity_search() -> None:
    proposal = _proposal()

    assert proposal["status"] == "frozen_zero_solver_runs"
    assert proposal["search_space"]["candidate_count"] == 300
    assert len(proposal["search_cells"]) == 36
    assert proposal["unique_budget"] == 5400
    assert proposal["worst_case_attempt_budget"] == 10800
    assert proposal["formal_solver_run_count"] == 0
    assert proposal["independent_bo_run_count"] == 0
    assert proposal["automatic_stage_f_execution"] is False
    assert proposal["budget_contract"]["contract_version"] == (
        "lyx_recovery_filter_budget_v13"
    )
    assert proposal["independent_bo_request"]["unique_budget"] == 5400


def test_json_round_tripped_proposal_passes_source_preflight() -> None:
    proposal = json.loads(json.dumps(_proposal()))

    validate_recovery_independent_bo_preflight(
        proposal=proposal,
        repository_root=REPOSITORY_ROOT,
    )


def test_execution_authorization_must_bind_exact_proposal_and_bo_request() -> None:
    proposal = _proposal()
    request = proposal["independent_bo_request"]
    receipt = {
        "approved": True,
        "decision_state": "awaiting_human_independent_bo_decision",
        **request,
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_contract_hash": proposal["budget_contract_hash"],
        "approved_at": "2026-07-31T00:30:00+08:00",
        "approved_by": "user",
        "authorization_basis": "blanket_proposal_authorization_until_deadline",
        "blanket_authorization_expires_at": "2026-07-31T10:00:00+08:00",
    }

    assert validate_recovery_independent_bo_execution_authorization(
        proposal,
        receipt=receipt,
    ) == receipt

    drifted = dict(receipt)
    drifted["unique_budget"] = 5399
    with pytest.raises(
        RecoveryIndependentBOAuthorizationError,
        match="independent_bo_authorization_invalid",
    ):
        validate_recovery_independent_bo_execution_authorization(
            proposal,
            receipt=drifted,
        )


def test_bo_candidate_becomes_a_registered_rank1_recovery_identity() -> None:
    proposal = _proposal()
    candidate = build_bo_search_space("physical_v1").candidates[-1]

    identity = build_recovery_independent_bo_identity(
        proposal=proposal,
        cell=proposal["search_cells"][0],
        candidate=candidate,
    )

    assert identity["stage"] == "recovery_independent_bo"
    assert identity["attempt_kind"] == "formal"
    assert identity["sentinel_role"] == "fixed_rank1"
    assert identity["adaptive_reference_stage_limit"] == 1
    assert identity["physical_memory_ms"] == 200
    assert identity["actual_taps"] == 20
    assert identity["config"]["parameters"]["fs_target"] == 100
    assert identity["config"]["parameters"]["lms_mu_base"] == 0.016
    assert identity["config"]["parameters"]["spec_penalty_width"] == 0.3
    assert identity["config_hash"] == canonical_sha256(identity["config"])
    assert identity["identity_sha256"] == identity["cache_identity_sha256"]
