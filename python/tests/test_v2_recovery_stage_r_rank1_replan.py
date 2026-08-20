from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2 import recovery_stage_r_experiment as stage_r_module
from ppg_hr.v2 import recovery_stage_r_rank1_replan as replan_module
from ppg_hr.v2.phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    read_json,
)
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
    publish_stage_r_rank1_failure_receipt,
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
    proposal_generation: int = 1,
    runtime_failure_context: dict[str, object] | None = None,
) -> dict[str, object]:
    metric = stage_r_metric_contract_v1()
    spectral = stage_r_spectral_gate_contract_v2()
    selection = recovery_selection_contract_rank1_replan_v1()
    budget = (
        BudgetContract.proposed_v11_stage_r_rank1_replan()
        if proposal_generation == 1
        else BudgetContract.proposed_v12_stage_r_rank1_runtime_fix()
    )
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
        solver_hash=("a" if proposal_generation == 1 else "c") * 64,
        evaluation_hash=(
            "b" if proposal_generation == 1 else "d"
        )
        * 64,
        metric_contract_hash=metric["contract_sha256"],
        spectral_gate_contract_hash=spectral["contract_sha256"],
        selection_contract_hash=selection["contract_sha256"],
        budget_contract_hash=budget.sha256,
        proposal_generation=proposal_generation,
        runtime_failure_context=runtime_failure_context,
    )


def _authorization(proposal: dict[str, object]) -> dict[str, object]:
    frozen = proposal["frozen_contracts"]
    generation = (
        1
        if proposal["proposal_version"]
        == "lyx_stage_r_rank1_replan_proposal_v1"
        else 2
    )
    target_budget = (
        BudgetContract.proposed_v11_stage_r_rank1_replan()
        if generation == 1
        else BudgetContract.proposed_v12_stage_r_rank1_runtime_fix()
    )
    receipt = {
        "approved": True,
        "decision_state": "awaiting_human_budget_decision",
        "proposal_sha256": proposal["proposal_sha256"],
        "stage": proposal["stage"],
        "profile_design_rule_hash": frozen["selection_contract_hash"],
        "record_manifest_hash": proposal["record_panel_sha256"],
        "added_unique_identities": 36,
        "normal_unique_identity_limit": (
            target_budget.normal_unique_identity_limit
        ),
        "max_unique_identities": target_budget.max_unique_identities,
        "max_attempts": target_budget.max_attempts,
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
    if generation == 2:
        receipt["superseded_proposal_sha256"] = proposal[
            "runtime_failure_replacement"
        ]["superseded_proposal_sha256"]
    return receipt


def _failure_context(
    superseded_proposal: dict[str, object],
) -> dict[str, object]:
    context = {
        "context_version": (
            "lyx_stage_r_rank1_runtime_failure_context_v1"
        ),
        "superseded_proposal_sha256": superseded_proposal[
            "proposal_sha256"
        ],
        "superseded_authorization_sha256": "1" * 64,
        "source_governance_receipt_sha256": "2" * 64,
        "source_attempt_registry_file_sha256": "3" * 64,
        "source_exploration_registry_file_sha256": "5" * 64,
        "failed_execution_binding_sha256": "4" * 64,
        "failed_identity_sha256": superseded_proposal[
            "identity_sha256"
        ][0],
        "original_identity_panel_sha256": superseded_proposal[
            "identity_panel_sha256"
        ],
        "failed_record_id": superseded_proposal["identities"][0][
            "record_id"
        ],
        "failed_recovery_candidate_id": superseded_proposal[
            "identities"
        ][0]["recovery_candidate_id"],
        "failure_reason": (
            "ValueError:invalid_recovery_sentinel_role"
        ),
        "failure_class": "post_solver_spectral_audit",
        "failed_attempt_count": 1,
        "succeeded_identity_count": 0,
        "unattempted_identity_count": 35,
        "solver_invocation_count": 1,
        "persisted_numerical_result_count": 0,
        "solver_output_reusable": False,
        "replacement_identity_count": 36,
        "original_identity_reuse_authorized": False,
        "original_retry_authorized": False,
    }
    context["context_sha256"] = canonical_sha256(context)
    return context


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


def test_runtime_fix_proposal_replaces_all_source_bound_identities() -> None:
    superseded = _proposal()
    replacement = _proposal(
        proposal_generation=2,
        runtime_failure_context=_failure_context(superseded),
    )

    assert replacement["proposal_version"] == (
        "lyx_stage_r_rank1_replan_proposal_v2"
    )
    assert replacement["runtime_failure_replacement"][
        "superseded_proposal_sha256"
    ] == superseded["proposal_sha256"]
    assert replacement["runtime_failure_replacement"][
        "original_retry_authorized"
    ] is False
    assert len(replacement["identity_sha256"]) == 36
    assert set(replacement["identity_sha256"]).isdisjoint(
        superseded["identity_sha256"]
    )
    replan_module._validate_replacement_identity_panel(
        replacement,
        superseded_proposal=superseded,
    )
    assert replacement["frozen_contracts"][
        "budget_contract_hash"
    ] == BudgetContract.proposed_v12_stage_r_rank1_runtime_fix().sha256
    assert validate_stage_r_rank1_execution_authorization(
        replacement,
        receipt=_authorization(replacement),
    )["max_unique_identities"] == 936


def test_runtime_fix_rejects_scientific_coordinate_drift() -> None:
    superseded = _proposal()
    replacement = _proposal(
        proposal_generation=2,
        runtime_failure_context=_failure_context(superseded),
    )
    replacement["identities"][0]["nominal_mu"] = 0.009

    with pytest.raises(
        replan_module.StageRRank1Error,
        match="stage_r_rank1_replacement_coordinate_drift",
    ):
        replan_module._validate_replacement_identity_panel(
            replacement,
            superseded_proposal=superseded,
        )


def test_runtime_failure_receipt_freezes_exact_failed_attempt(
    tmp_path: Path,
) -> None:
    proposal = _proposal()
    authorization = _authorization(proposal)
    proposal_path = tmp_path / "proposal.json"
    authorization_path = tmp_path / "authorization.json"
    atomic_write_json(proposal_path, proposal)
    atomic_write_json(authorization_path, authorization)
    budget = BudgetContract.proposed_v11_stage_r_rank1_replan()
    exploration = ExplorationRegistry.zero_budget_v1()
    governance = tmp_path / "governance"
    atomic_write_json(
        governance / "exploration_registry.json",
        exploration.to_dict(),
    )
    registry = AttemptRegistry.create(
        governance / "attempt_registry.json",
        budget_contract=budget,
        exploration_registry=exploration,
    )
    identities = tuple(
        replan_module._identity_from_item(item)
        for item in proposal["identities"]
    )
    registry.register_identities(identities)

    def fail_after_solver() -> None:
        raise ValueError("invalid_recovery_sentinel_role")

    with pytest.raises(
        ValueError,
        match="invalid_recovery_sentinel_role",
    ):
        registry.execute_registered(
            identities[0],
            fail_after_solver,
        )
    governance_receipt = {
        "receipt_version": (
            "lyx_stage_r_rank1_replan_governance_v1"
        ),
        "status": "prepared_zero_runs",
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": canonical_sha256(authorization),
        "target_budget_contract_hash": budget.sha256,
    }
    governance_receipt["receipt_sha256"] = canonical_sha256(
        governance_receipt
    )
    governance_receipt_path = (
        governance / "governance_receipt.json"
    )
    atomic_write_json(
        governance_receipt_path,
        governance_receipt,
    )
    atomic_write_json(
        governance / "execution_authorization.json",
        authorization,
    )
    binding = {
        "binding_version": (
            "lyx_stage_r_rank1_execution_binding_v1"
        ),
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": canonical_sha256(authorization),
        "solver_source_bundle_sha256": proposal[
            "frozen_contracts"
        ]["solver_hash"],
        "evaluation_hash": proposal["frozen_contracts"][
            "evaluation_hash"
        ],
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    binding_path = tmp_path / "execution" / "execution_binding.json"
    atomic_write_json(binding_path, binding)

    receipt = publish_stage_r_rank1_failure_receipt(
        superseded_proposal_path=proposal_path,
        superseded_authorization_path=authorization_path,
        source_governance_receipt_path=governance_receipt_path,
        source_attempt_registry_path=(
            governance / "attempt_registry.json"
        ),
        source_exploration_registry_path=(
            governance / "exploration_registry.json"
        ),
        failed_execution_binding_path=binding_path,
        output_path=tmp_path / "failure_receipt.json",
    )

    assert receipt["status"] == (
        "failed_after_solver_before_result_persistence"
    )
    assert receipt["failed_identity_sha256"] == identities[0].sha256
    assert receipt["failed_attempt_count"] == 1
    assert receipt["unattempted_identity_count"] == 35
    assert receipt["solver_invocation_count"] == 1
    assert receipt["persisted_numerical_result_count"] == 0
    assert receipt["solver_output_reusable"] is False
    assert receipt["original_retry_authorized"] is False

    replacement = _proposal(
        proposal_generation=2,
        runtime_failure_context=(
            replan_module._runtime_failure_context_from_receipt(
                receipt=receipt,
                superseded_proposal=proposal,
                superseded_authorization=authorization,
                source_governance_receipt=governance_receipt,
                failed_execution_binding=binding,
                source_exploration_registry_file_sha256=(
                    file_sha256(
                        governance / "exploration_registry.json"
                    )
                ),
            )
        ),
    )
    resolved = {
        "superseded_proposal": proposal_path,
        "superseded_authorization": authorization_path,
        "source_exploration_registry": (
            governance / "exploration_registry.json"
        ),
        "failed_execution_binding": binding_path,
        "runtime_failure_receipt": tmp_path / "failure_receipt.json",
    }
    replan_module._validate_v2_source_governance_state(
        replacement,
        source_dir=governance,
        resolved=resolved,
    )
    drifted_exploration = exploration.to_dict()
    drifted_exploration["unexpected_change"] = True
    atomic_write_json(
        governance / "exploration_registry.json",
        drifted_exploration,
    )
    with pytest.raises(
        replan_module.StageRRank1Error,
        match=(
            "stage_r_rank1_v11_registry_changed_after_failure_receipt"
        ),
    ):
        replan_module._validate_v2_source_governance_state(
            replacement,
            source_dir=governance,
            resolved=resolved,
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


def test_rank1_spectral_audit_does_not_require_legacy_sentinel_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = dict(_proposal()["identities"][0])
    observed: dict[str, object] = {}

    def fake_audit(
        profile: object,
        _record: object,
        *,
        contract: object,
        reference_stage_limit: int | None,
    ) -> dict[str, object]:
        del contract
        observed["profile"] = profile
        observed["reference_stage_limit"] = reference_stage_limit
        return {
            "stability_pass": True,
            "spectral_gate_pass": True,
            "reference_stage_limit": reference_stage_limit,
        }

    monkeypatch.setattr(
        stage_r_module,
        "audit_stage_r_profile_record",
        fake_audit,
    )

    audit = stage_r_module._load_or_run_spectral_audit(
        identity,
        spectral_audit_dir=tmp_path / "spectral",
    )

    profile = observed["profile"]
    assert profile.profile_id == "p25-short-low-rank1-v1"
    assert profile.recovery_sentinel_role is None
    assert observed["reference_stage_limit"] == 1
    assert audit["spectral_gate_pass"] is True


def test_recovery_independent_bo_reuses_rank1_spectral_role_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = dict(_proposal()["identities"][0])
    identity["stage"] = "recovery_independent_bo"
    identity["filter_profile_id"] = "bo-physical-v1-test"
    identity["filter_profile_sha256"] = "f" * 64
    observed: dict[str, object] = {}

    def fake_audit(
        profile: object,
        _record: object,
        *,
        contract: object,
        reference_stage_limit: int | None,
    ) -> dict[str, object]:
        del contract
        observed["profile"] = profile
        return {
            "stability_pass": True,
            "spectral_gate_pass": True,
            "reference_stage_limit": reference_stage_limit,
        }

    monkeypatch.setattr(
        stage_r_module,
        "audit_stage_r_profile_record",
        fake_audit,
    )

    stage_r_module._load_or_run_spectral_audit(
        identity,
        spectral_audit_dir=tmp_path / "spectral",
    )

    assert observed["profile"].recovery_sentinel_role is None


def test_legacy_stage_r_spectral_audit_still_rejects_invalid_role(
    tmp_path: Path,
) -> None:
    identity = dict(_proposal()["identities"][0])
    identity["stage"] = "recovery_sentinel"
    identity["sentinel_role"] = "fixed_rank1"

    with pytest.raises(
        ValueError,
        match="invalid_recovery_sentinel_role",
    ):
        stage_r_module._load_or_run_spectral_audit(
            identity,
            spectral_audit_dir=tmp_path / "spectral",
        )


def test_stage_r_spectral_audit_runs_before_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = dict(_proposal()["identities"][0])

    def fail_audit(*_args: object, **_kwargs: object) -> object:
        raise ValueError("invalid_recovery_sentinel_role")

    monkeypatch.setattr(
        stage_r_module,
        "_load_or_run_spectral_audit",
        fail_audit,
    )
    monkeypatch.setattr(
        stage_r_module,
        "solve_v2",
        lambda *_args, **_kwargs: pytest.fail(
            "spectral audit failure must stop before solver"
        ),
    )

    with pytest.raises(
        ValueError,
        match="invalid_recovery_sentinel_role",
    ):
        stage_r_module.run_stage_r_numerical_identity(
            identity,
            tmp_path / "spectral",
        )


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
    governance_receipt_path = governance / "governance_receipt.json"
    valid_governance_receipt = read_json(governance_receipt_path)
    tampered_governance_receipt = deepcopy(
        valid_governance_receipt
    )
    tampered_governance_receipt["independent_bo_authorized"] = True
    tampered_governance_receipt.pop("receipt_sha256")
    tampered_governance_receipt["receipt_sha256"] = canonical_sha256(
        tampered_governance_receipt
    )
    atomic_write_json(
        governance_receipt_path,
        tampered_governance_receipt,
    )
    with pytest.raises(
        replan_module.StageRRank1Error,
        match="stage_r_rank1_governance_receipt_mismatch",
    ):
        execute_stage_r_rank1_replan(
            proposal_dir=proposal_dir,
            authorization_receipt_path=authorization_path,
            governance_dir=governance,
            output_dir=tmp_path / "tampered_execution",
            source_root=ROOT / "python" / "src",
            numerical_runner=lambda *_args: pytest.fail(
                "tampered governance must stop before solver"
            ),
        )
    assert not (tmp_path / "tampered_execution").exists()
    atomic_write_json(
        governance_receipt_path,
        valid_governance_receipt,
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
