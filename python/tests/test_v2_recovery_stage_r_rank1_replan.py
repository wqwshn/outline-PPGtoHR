from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2 import recovery_stage_r_rank1_replan as replan_module
from ppg_hr.v2.phase2_experiment_io import atomic_write_json, read_json
from ppg_hr.v2.recovery_contracts import canonical_sha256
from ppg_hr.v2.recovery_experiment_governance import (
    AttemptRegistry,
    BudgetContract,
    ExplorationRegistry,
)
from ppg_hr.v2.recovery_selection import (
    recovery_selection_contract_rank1_replan_v1,
)
from ppg_hr.v2.recovery_stage_r_common import StageRNumericalResult
from ppg_hr.v2.recovery_stage_r_experiment import (
    stage_r_metric_contract_v1,
    stage_r_spectral_gate_contract_v2,
)
from ppg_hr.v2.recovery_stage_r_rank1_replan import (
    StageRRank1AuthorizationError,
    build_stage_r_rank1_replan_proposal,
    execute_stage_r_rank1_replan,
    prepare_stage_r_rank1_replan_governance,
    propose_stage_r_rank1_replan,
    validate_stage_r_rank1_execution_authorization,
)
from ppg_hr.v2.solver import V2SolverResult

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = (
    ROOT / "data" / "experiments" / "lyx_recovery_filter_profile"
)
RANK1_PROPOSAL_DIR = EXPERIMENT / "rank1_filter_revision_v1"
RANK1_EXECUTION_DIR = (
    EXPERIMENT / "rank1_filter_revision_execution_v1"
)


def _rank1_results() -> dict[str, dict[str, object]]:
    manifest = read_json(RANK1_EXECUTION_DIR / "result_manifest.json")
    return {
        str(entry["record_id"]): read_json(
            RANK1_EXECUTION_DIR / str(entry["path"])
        )
        for entry in manifest["results"]
    }


def _proposal(
    *,
    baseline_manifest: dict[str, object] | None = None,
    baseline_metrics: dict[str, object] | None = None,
    penalty_registry: dict[str, object] | None = None,
) -> dict[str, object]:
    metric = stage_r_metric_contract_v1()
    spectral = stage_r_spectral_gate_contract_v2()
    selection = recovery_selection_contract_rank1_replan_v1()
    budget = BudgetContract.proposed_v11_stage_r_rank1_replan()
    return build_stage_r_rank1_replan_proposal(
        rank1_proposal=read_json(
            RANK1_PROPOSAL_DIR / "rank1_filter_revision_proposal.json"
        ),
        rank1_completion=read_json(
            RANK1_EXECUTION_DIR / "completion.json"
        ),
        rank1_decision=read_json(
            RANK1_EXECUTION_DIR / "decision_receipt.json"
        ),
        rank1_manifest=read_json(
            RANK1_EXECUTION_DIR / "result_manifest.json"
        ),
        rank1_results=_rank1_results(),
        prior_stage_r_proposal=read_json(
            EXPERIMENT / "stage_r_v3" / "stage_r_execution_proposal.json"
        ),
        prior_stage_r_governance_receipt=read_json(
            EXPERIMENT
            / "governance_v5"
            / "stage_r_governance_receipt.json"
        ),
        recovery_registry=read_json(
            EXPERIMENT
            / "recovery_candidates_v1"
            / "recovery_candidate_registry.json"
        ),
        penalty_registry=(
            penalty_registry
            if penalty_registry is not None
            else read_json(
                EXPERIMENT
                / "penalty_candidates_v1"
                / "penalty_registry.json"
            )
        ),
        baseline_manifest=(
            baseline_manifest
            if baseline_manifest is not None
            else read_json(
                EXPERIMENT
                / "lyx_independent_bo_baseline_manifest_v1.json"
            )
        ),
        baseline_metrics=(
            baseline_metrics
            if baseline_metrics is not None
            else read_json(
                EXPERIMENT
                / "independent_bo_baseline_v1"
                / "record_metrics.json"
            )
        ),
        parent_experiment_id="lyx-recovery-filter-profile-v1",
        solver_hash="a" * 64,
        evaluation_hash="b" * 64,
        metric_contract_hash=metric["contract_sha256"],
        spectral_gate_contract_hash=spectral["contract_sha256"],
        selection_contract_hash=selection["contract_sha256"],
        budget_contract_hash=budget.sha256,
    )


def _authorization(proposal: dict[str, object]) -> dict[str, object]:
    frozen = proposal["frozen_contracts"]
    return {
        "approved": True,
        "decision_state": "awaiting_human_budget_decision",
        "proposal_sha256": proposal["proposal_sha256"],
        "stage": proposal["stage"],
        "profile_design_rule_hash": frozen["selection_contract_hash"],
        "record_manifest_hash": proposal["record_panel_sha256"],
        "added_unique_identities": 36,
        "normal_unique_identity_limit": 888,
        "max_unique_identities": 900,
        "max_attempts": 1800,
        "budget_contract_hash": frozen["budget_contract_hash"],
        "identity_panel_sha256": proposal["identity_panel_sha256"],
        "baseline_identity_panel_sha256": proposal[
            "baseline_identity_panel_sha256"
        ],
        "solver_hash": frozen["solver_hash"],
        "evaluation_hash": frozen["evaluation_hash"],
        "metric_contract_hash": frozen["metric_contract_hash"],
        "spectral_gate_contract_hash": frozen[
            "spectral_gate_contract_hash"
        ],
        "selection_contract_hash": frozen["selection_contract_hash"],
        "filter_revision_contract_hash": frozen[
            "filter_revision_contract_hash"
        ],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": True,
        "approved_at": "2026-07-30T23:59:00+08:00",
        "approved_by": "user",
    }


def _publish_proposal(destination: Path) -> dict[str, object]:
    return propose_stage_r_rank1_replan(
        rank1_proposal_path=(
            RANK1_PROPOSAL_DIR
            / "rank1_filter_revision_proposal.json"
        ),
        rank1_completion_path=(
            RANK1_EXECUTION_DIR / "completion.json"
        ),
        rank1_decision_path=(
            RANK1_EXECUTION_DIR / "decision_receipt.json"
        ),
        rank1_manifest_path=(
            RANK1_EXECUTION_DIR / "result_manifest.json"
        ),
        prior_stage_r_proposal_path=(
            EXPERIMENT / "stage_r_v3" / "stage_r_execution_proposal.json"
        ),
        prior_stage_r_governance_receipt_path=(
            EXPERIMENT
            / "governance_v5"
            / "stage_r_governance_receipt.json"
        ),
        recovery_registry_path=(
            EXPERIMENT
            / "recovery_candidates_v1"
            / "recovery_candidate_registry.json"
        ),
        penalty_registry_path=(
            EXPERIMENT
            / "penalty_candidates_v1"
            / "penalty_registry.json"
        ),
        baseline_manifest_path=(
            EXPERIMENT
            / "lyx_independent_bo_baseline_manifest_v1.json"
        ),
        baseline_metrics_path=(
            EXPERIMENT
            / "independent_bo_baseline_v1"
            / "record_metrics.json"
        ),
        source_budget_contract_path=(
            EXPERIMENT / "governance_v10" / "budget_contract.json"
        ),
        spec_path=(
            ROOT
            / "docs"
            / "experiments"
            / "2026-07-30-lyx-stage-r-rank1-replan-spec.md"
        ),
        output_dir=destination,
        source_root=ROOT / "python" / "src",
        parent_experiment_id="lyx-recovery-filter-profile-v1",
    )


def test_replan_proposal_freezes_only_36_new_formal_identities() -> None:
    proposal = _proposal()

    assert proposal["status"] == (
        "awaiting_human_execution_authorization"
    )
    assert proposal["stage"] == "recovery_sentinel_rank1_replan"
    assert proposal["new_threshold_diagnostic_unique_budget"] == 0
    assert proposal["historical_threshold_diagnostic_count"] == 60
    assert proposal["historical_stage_r_formal_identity_count"] == 108
    assert proposal["formal_unique_budget"] == 36
    assert proposal["unique_budget"] == 36
    assert proposal["worst_case_attempt_budget"] == 72
    assert proposal["diagnostic_run_count"] == 0
    assert proposal["formal_run_count"] == 0
    assert proposal["parameter_search_authorized"] is False
    assert proposal["independent_bo_authorized"] is False
    assert proposal["automatic_stage_f_execution"] is False
    assert proposal["may_nominate_recovery_candidate"] is True

    identities = proposal["identities"]
    assert len(identities) == 36
    assert len(set(proposal["identity_sha256"])) == 36
    assert {
        item["recovery_candidate_id"] for item in identities
    } == {
        "current_fixed_floor_control_v1",
        "relative_gap_timeout_v1",
        "relative_gap_rise_guard_v1",
    }
    assert {
        (item["record_id"], item["scene"]) for item in identities
    } == {
        (item["record_id"], item["scene"])
        for item in proposal["record_panel"]
    }
    assert all(
        item["filter_profile_id"] == "p25-short-low-rank1-v1"
        and item["actual_taps"] == 1
        and item["physical_memory_ms"] == 40
        and item["nominal_mu"] == 0.008
        and item["config"]["parameters"][
            "adaptive_reference_stage_limit"
        ]
        == 1
        and item["config"]["parameters"]["fs_target"] == 25
        and item["config"]["parameters"]["max_order"] == 1
        for item in identities
    )


def test_replan_authorization_is_exact_and_keeps_bo_disabled() -> None:
    proposal = _proposal()
    receipt = _authorization(proposal)

    with pytest.raises(
        StageRRank1AuthorizationError,
        match="stage_r_rank1_execution_authorization_required",
    ):
        validate_stage_r_rank1_execution_authorization(
            proposal,
            receipt=None,
        )
    assert validate_stage_r_rank1_execution_authorization(
        proposal,
        receipt=receipt,
    ) == receipt
    with pytest.raises(
        StageRRank1AuthorizationError,
        match="stage_r_rank1_authorization_mismatch",
    ):
        validate_stage_r_rank1_execution_authorization(
            proposal,
            receipt={**receipt, "independent_bo_authorized": True},
        )


def test_replan_publication_is_zero_run_and_has_no_authorization(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "proposal"

    receipt = _publish_proposal(destination)

    assert receipt["status"] == (
        "awaiting_human_execution_authorization"
    )
    assert receipt["identity_count"] == 36
    assert receipt["formal_run_count"] == 0
    assert receipt["independent_bo_run_count"] == 0
    assert receipt["may_execute_without_new_authorization"] is False
    assert not (destination / "execution_authorization.json").exists()
    request = read_json(
        destination / "budget_amendment_request.json"
    )
    assert request["approved"] is False
    assert request["added_unique_identities"] == 36
    assert request["max_unique_identities"] == 900


def test_replan_selection_consumes_exact_candidate_invariant_panel() -> None:
    proposal = _proposal()
    baseline_path = (
        EXPERIMENT
        / "independent_bo_baseline_v1"
        / "record_metrics.json"
    )
    baseline = read_json(baseline_path)
    metrics_by_record = {
        str(item["sample_id"]): dict(item["metrics"])
        for item in baseline["records"]
    }
    rows = [
        {
            "recovery_candidate_id": item[
                "recovery_candidate_id"
            ],
            "record_id": item["record_id"],
            "metrics": metrics_by_record[item["record_id"]],
            "spectral_audit": {
                "stability_pass": True,
                "spectral_gate_pass": True,
                "audit_sha256": (
                    str(item["record_id"]).encode("utf-8").hex()[:64]
                ).ljust(64, "0"),
            },
        }
        for item in proposal["identities"]
    ]

    selection, evaluations = replan_module._build_rank1_selection(
        proposal=proposal,
        result_rows=rows,
        baseline_metrics_path=baseline_path,
    )

    assert selection["status"] == "selected"
    assert selection["provisional_recovery_id"] == (
        "current_fixed_floor_control_v1"
    )
    assert selection["eligible_candidate_ids"] == [
        "current_fixed_floor_control_v1",
        "relative_gap_timeout_v1",
        "relative_gap_rise_guard_v1",
    ]
    assert len(evaluations) == 3
    assert all(len(item["records"]) == 12 for item in evaluations)


def test_replan_rejects_independent_bo_baseline_identity_drift() -> None:
    baseline_metrics = deepcopy(
        read_json(
            EXPERIMENT
            / "independent_bo_baseline_v1"
            / "record_metrics.json"
        )
    )
    baseline_metrics["records"][0]["scene"] = "wrong-scene"

    with pytest.raises(
        replan_module.StageRRank1Error,
        match="stage_r_rank1_baseline_identity_drift",
    ):
        _proposal(baseline_metrics=baseline_metrics)


def test_replan_requires_the_current_soft_penalty_control() -> None:
    registry = deepcopy(
        read_json(
            EXPERIMENT
            / "penalty_candidates_v1"
            / "penalty_registry.json"
        )
    )
    registry["control_penalty_id"] = "candidate_soft_clip_v1"
    unsigned = {
        key: value
        for key, value in registry.items()
        if key != "registry_sha256"
    }
    registry["registry_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(
        replan_module.StageRRank1Error,
        match="stage_r_rank1_control_penalty_id_mismatch",
    ):
        _proposal(penalty_registry=registry)


def test_replan_right_censored_delay_uses_finite_window_fallback() -> None:
    assert replan_module._selection_recovery_delay(
        {
            "max_recovered_delay_s": None,
            "recovery_episode_count": 2,
            "right_censored_recovery_count": 2,
            "total_window_count": 37,
        }
    ) == 37.0


def test_replan_fake_execution_is_recomputable_and_resumable(
    tmp_path: Path,
) -> None:
    proposal_dir = tmp_path / "proposal"
    _publish_proposal(proposal_dir)
    proposal = read_json(
        proposal_dir / "stage_r_rank1_replan_proposal.json"
    )
    authorization_path = tmp_path / "authorization.json"
    atomic_write_json(
        authorization_path,
        _authorization(proposal),
    )
    source_governance = tmp_path / "governance_v10"
    source_budget = BudgetContract.proposed_v10_rank1_filter_revision()
    exploration = ExplorationRegistry.zero_budget_v1()
    atomic_write_json(
        source_governance / "budget_contract.json",
        source_budget.to_dict(),
    )
    atomic_write_json(
        source_governance / "exploration_registry.json",
        exploration.to_dict(),
    )
    AttemptRegistry.create(
        source_governance / "attempt_registry.json",
        budget_contract=source_budget,
        exploration_registry=exploration,
    )
    governance = tmp_path / "governance_v11"
    prepare_stage_r_rank1_replan_governance(
        proposal_dir=proposal_dir,
        authorization_receipt_path=authorization_path,
        source_governance_dir=source_governance,
        governance_dir=governance,
        source_root=ROOT / "python" / "src",
    )
    baseline = read_json(
        EXPERIMENT
        / "independent_bo_baseline_v1"
        / "record_metrics.json"
    )
    metrics_by_record = {
        str(item["sample_id"]): dict(item["metrics"])
        for item in baseline["records"]
    }

    def fake_runner(
        item: dict[str, object],
        _spectral_audit_dir: Path,
    ) -> StageRNumericalResult:
        record_id = str(item["record_id"])
        return StageRNumericalResult(
            solver_result=V2SolverResult(
                HR=np.asarray([60.0, 61.0]),
                err_stats={},
                metadata={},
                window_table=[],
            ),
            metrics=metrics_by_record[record_id],
            spectral_audit={
                "stability_pass": True,
                "spectral_gate_pass": True,
                "audit_sha256": canonical_sha256(
                    {"record_id": record_id}
                ),
            },
        )

    output_dir = tmp_path / "execution"
    completion = execute_stage_r_rank1_replan(
        proposal_dir=proposal_dir,
        authorization_receipt_path=authorization_path,
        governance_dir=governance,
        output_dir=output_dir,
        source_root=ROOT / "python" / "src",
        numerical_runner=fake_runner,
    )
    resumed = execute_stage_r_rank1_replan(
        proposal_dir=proposal_dir,
        authorization_receipt_path=authorization_path,
        governance_dir=governance,
        output_dir=output_dir,
        source_root=ROOT / "python" / "src",
        numerical_runner=lambda *_args: pytest.fail(
            "completed execution must not rerun the solver"
        ),
    )

    assert completion == resumed
    assert completion["formal_result_count"] == 36
    assert completion["formal_solver_run_count"] == 36
    assert completion["independent_bo_run_count"] == 0
    assert completion["status"] == "selected"
    assert completion["next_state"] == (
        "awaiting_stage_f_rank1_replan_human_review"
    )
