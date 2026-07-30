from __future__ import annotations

from pathlib import Path

import pytest

from ppg_hr.v2.phase2_experiment_io import read_json
from ppg_hr.v2.recovery_rank1_filter_revision import (
    Rank1FilterRevisionAuthorizationError,
    Rank1FilterRevisionContract,
    build_rank1_filter_revision_proposal,
    evaluate_rank1_filter_revision_decision,
    validate_rank1_filter_revision_authorization,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
    ROOT / "data" / "experiments" / "lyx_recovery_filter_profile"
)
MECHANISM_PROPOSAL_DIR = (
    EXPERIMENT_ROOT / "filter_mechanism_decomposition_v1"
)
MECHANISM_EXECUTION_DIR = (
    EXPERIMENT_ROOT / "filter_mechanism_decomposition_execution_v1"
)


def _source_results() -> dict[str, dict[str, object]]:
    manifest = read_json(MECHANISM_EXECUTION_DIR / "result_manifest.json")
    return {
        str(entry["record_id"]): read_json(
            MECHANISM_EXECUTION_DIR / str(entry["path"])
        )
        for entry in manifest["results"]
    }


def _proposal() -> dict[str, object]:
    return build_rank1_filter_revision_proposal(
        mechanism_proposal=read_json(
            MECHANISM_PROPOSAL_DIR
            / "filter_mechanism_decomposition_proposal.json"
        ),
        mechanism_completion=read_json(
            MECHANISM_EXECUTION_DIR / "completion.json"
        ),
        mechanism_decision=read_json(
            MECHANISM_EXECUTION_DIR / "decision_receipt.json"
        ),
        mechanism_manifest=read_json(
            MECHANISM_EXECUTION_DIR / "result_manifest.json"
        ),
        mechanism_results=_source_results(),
        parent_experiment_id="lyx-recovery-filter-profile-v1",
        solver_hash="a" * 64,
        evaluation_hash="b" * 64,
    )


def test_contract_is_one_frozen_rank1_stage_without_search() -> None:
    contract = Rank1FilterRevisionContract()

    assert contract.revision_id == "p25-short-low-rank1-v1"
    assert contract.base_profile_id == "p25-short-low"
    assert contract.adaptive_reference_stage_limit == 1
    assert contract.reference_groups_order == ("HF",)
    assert contract.fs_target == 25
    assert contract.memory_ms == 40
    assert contract.actual_taps == 1
    assert contract.nominal_mu == 0.008
    assert contract.to_dict()["parameter_search"] is False


def test_proposal_is_zero_run_and_binds_12_expected_rank1_lanes() -> None:
    proposal = _proposal()

    assert proposal["status"] == "authorized_scope_frozen_zero_runs"
    assert proposal["unique_budget"] == 12
    assert proposal["worst_case_attempt_budget"] == 24
    assert proposal["diagnostic_run_count"] == 0
    assert proposal["parameter_search_authorized"] is False
    assert proposal["independent_bo_authorized"] is False
    assert proposal["automatic_stage_r_execution"] is False
    assert len(proposal["identities"]) == 12
    assert len(set(proposal["identity_sha256"])) == 12
    assert all(
        item["expected_rank1_lane_sha256"]
        for item in proposal["identities"]
    )


def test_authorization_is_exact_and_excludes_stage_r() -> None:
    proposal = _proposal()
    frozen = proposal["frozen_contracts"]
    receipt = {
        "approved": True,
        "decision_state": proposal["authorization_state"],
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_contract_hash": frozen["budget_contract_hash"],
        "unique_budget": 12,
        "stage": proposal["stage"],
        "identity_panel_sha256": proposal["identity_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "source_result_panel_sha256": proposal[
            "source_result_panel_sha256"
        ],
        "solver_hash": frozen["solver_hash"],
        "evaluation_hash": frozen["evaluation_hash"],
        "revision_contract_hash": frozen["revision_contract_hash"],
        "spectral_gate_contract_hash": frozen[
            "spectral_gate_contract_hash"
        ],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": False,
        "approved_at": "2026-07-30T23:00:00+08:00",
        "approved_by": "user",
    }

    with pytest.raises(
        Rank1FilterRevisionAuthorizationError,
        match="rank1_filter_revision_execution_authorization_required",
    ):
        validate_rank1_filter_revision_authorization(
            proposal,
            receipt=None,
        )
    assert validate_rank1_filter_revision_authorization(
        proposal,
        receipt=receipt,
    ) == receipt
    with pytest.raises(
        Rank1FilterRevisionAuthorizationError,
        match="rank1_filter_revision_authorization_mismatch",
    ):
        validate_rank1_filter_revision_authorization(
            proposal,
            receipt={**receipt, "automatic_stage_r_execution": True},
        )


def _result(
    index: int,
    *,
    exact: bool = True,
    complete: bool = True,
    stage_count_ok: bool = True,
) -> dict[str, object]:
    return {
        "record_id": f"record-{index}",
        "exact_rank1_reproduction_pass": exact,
        "spectral_gate_pass": complete,
        "all_gate_pass": complete,
        "single_reference_stage_per_valid_window": stage_count_ok,
        "source_ranked_cascade_spectral_gate_pass": index not in {0, 1},
    }


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            [_result(i, exact=i != 0) for i in range(12)],
            "rank1_revision_reproduction_invalid",
        ),
        (
            [_result(i, stage_count_ok=i != 0) for i in range(12)],
            "rank1_revision_reproduction_invalid",
        ),
        (
            [_result(i, complete=i != 0) for i in range(12)],
            "rank1_revision_reproduction_invalid",
        ),
        (
            [_result(i) for i in range(12)],
            "rank1_filter_revision_validated",
        ),
    ],
)
def test_decision_is_fail_closed(
    rows: list[dict[str, object]],
    expected: str,
) -> None:
    decision = evaluate_rank1_filter_revision_decision(rows)

    assert decision["decision"] == expected
    assert decision["automatic_stage_r_execution"] is False
    assert decision["may_nominate_recovery_candidate"] is False
    if expected == "rank1_filter_revision_validated":
        assert (
            decision["next_state"]
            == "awaiting_stage_r_replan_human_review"
        )
