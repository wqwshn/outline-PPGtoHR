from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2 import recovery_spectral_metric_control as control_module
from ppg_hr.v2.phase2_experiment_io import (
    atomic_write_json,
    read_json,
)
from ppg_hr.v2.recovery_experiment_governance import (
    AttemptRegistry,
    BudgetContract,
    ExplorationRegistry,
)
from ppg_hr.v2.recovery_spectral_gate import (
    StageRPreparedWindow,
    StageRSpectralGateContract,
)
from ppg_hr.v2.recovery_spectral_metric_control import (
    SpectralMetricControlAuthorizationError,
    SpectralMetricScaleControlContract,
    build_spectral_metric_control_proposal,
    evaluate_spectral_metric_control_decision,
    evaluate_spectral_metric_scale_controls,
    execute_spectral_metric_control,
    prepare_spectral_metric_control_governance,
    propose_spectral_metric_control,
    validate_spectral_metric_control_authorization,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
    ROOT / "data" / "experiments" / "lyx_recovery_filter_profile"
)


def _prepared_window(center_s: float) -> StageRPreparedWindow:
    fs = 25
    time_s = np.arange(0.0, 8.0, 1.0 / fs)
    pulse = 20.0 * np.sin(2.0 * np.pi * 1.5 * time_s)
    artifact = 4.0 * np.sin(2.0 * np.pi * 2.5 * time_s)
    original = 1000.0 + pulse + artifact
    return StageRPreparedWindow(
        original=original,
        ranked_references=(
            ("acc_x", artifact, 0.8),
            ("acc_y", np.cos(2.0 * np.pi * 2.5 * time_s), 0.4),
        ),
        primary_reference=artifact,
        delay_samples=0,
        order=1,
        fs=fs,
        reference_hr_bpm=90.0,
        window_center_s=center_s,
    )


def test_scale_controls_isolate_legacy_raw_vs_standardized_mismatch() -> None:
    result = evaluate_spectral_metric_scale_controls(
        [_prepared_window(center) for center in (4.0, 6.0, 8.0)],
        spectral_contract=StageRSpectralGateContract(),
        control_contract=SpectralMetricScaleControlContract(),
    )

    retention = result["pulse_power_retention_median"]
    assert result["complete_window_evidence"] is True
    assert result["zero_update_weight_max_abs"] == 0.0
    assert result["direct_bypass_pass"] is True
    assert result["same_scale_zero_update_pass"] is True
    assert result["legacy_scale_mismatch_reproduced"] is True
    assert retention["direct_raw_bypass"] == pytest.approx(1.0)
    assert retention["same_scale_zero_update_lms"] == pytest.approx(1.0)
    assert retention["legacy_raw_vs_zero_update_lms"] < 0.01
    assert result["legacy_to_same_scale_retention_ratio"] < 0.01


def _decision_row(
    record_id: str,
    *,
    direct: bool = True,
    same_scale: bool = True,
    mismatch: bool = True,
    complete: bool = True,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "direct_bypass_pass": direct,
        "same_scale_zero_update_pass": same_scale,
        "legacy_scale_mismatch_reproduced": mismatch,
        "complete_window_evidence": complete,
    }


def test_scale_control_decision_is_fail_closed_and_has_partial_branch() -> None:
    contract = SpectralMetricScaleControlContract()
    confirmed = [_decision_row(f"record-{index}") for index in range(12)]
    direct_failure = [
        *confirmed[:-1],
        _decision_row("record-11", direct=False),
    ]
    same_failure = [
        *confirmed[:-1],
        _decision_row("record-11", same_scale=False),
    ]
    partial = [
        *confirmed[:-1],
        _decision_row("record-11", mismatch=False),
    ]
    none = [
        _decision_row(f"record-{index}", mismatch=False)
        for index in range(12)
    ]

    assert evaluate_spectral_metric_control_decision(
        confirmed,
        control_contract=contract,
    )["decision"] == "legacy_scale_mismatch_confirmed"
    assert evaluate_spectral_metric_control_decision(
        direct_failure,
        control_contract=contract,
    )["decision"] == "spectral_evaluator_invalid"
    assert evaluate_spectral_metric_control_decision(
        same_failure,
        control_contract=contract,
    )["decision"] == "zero_update_path_invalid"
    assert evaluate_spectral_metric_control_decision(
        partial,
        control_contract=contract,
    )["decision"] == "legacy_scale_mismatch_partial"
    assert evaluate_spectral_metric_control_decision(
        none,
        control_contract=contract,
    )["decision"] == "legacy_scale_mismatch_not_reproduced"


def _proposal() -> dict[str, object]:
    return build_spectral_metric_control_proposal(
        p25_proposal=read_json(
            EXPERIMENT_ROOT
            / "p25_spectral_diagnostic_v1"
            / "p25_spectral_diagnostic_proposal.json"
        ),
        p25_completion=read_json(
            EXPERIMENT_ROOT
            / "p25_spectral_diagnostic_execution_v1"
            / "completion.json"
        ),
        p25_decision=read_json(
            EXPERIMENT_ROOT
            / "p25_spectral_diagnostic_execution_v1"
            / "decision_receipt.json"
        ),
        parent_experiment_id="lyx-recovery-filter-profile-v1",
        solver_hash="a" * 64,
        evaluation_hash="b" * 64,
    )


def test_proposal_freezes_12_records_three_lanes_and_v7_budget() -> None:
    proposal = _proposal()
    budget = BudgetContract.proposed_v7_spectral_metric_control()

    assert proposal["unique_budget"] == 12
    assert proposal["deterministic_lane_count_per_identity"] == 3
    assert len(proposal["identities"]) == 12
    assert len(set(proposal["identity_sha256"])) == 12
    assert proposal["parameter_search_authorized"] is False
    assert proposal["independent_bo_authorized"] is False
    assert proposal["frozen_contracts"]["budget_contract_hash"] == budget.sha256
    assert budget.normal_unique_identity_limit == 792
    assert budget.max_unique_identities == 804
    assert budget.max_attempts == 1608
    assert budget.stage_unique_limits[
        "spectral_metric_scale_control_diagnostic"
    ] == 12


def test_authorization_binds_exact_control_proposal() -> None:
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
        "solver_hash": frozen["solver_hash"],
        "evaluation_hash": frozen["evaluation_hash"],
        "control_contract_hash": frozen["control_contract_hash"],
        "spectral_gate_contract_hash": frozen[
            "spectral_gate_contract_hash"
        ],
        "control_profile_hash": frozen["control_profile_hash"],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "approved_at": "2026-07-30T12:00:00+08:00",
        "approved_by": "user",
    }

    with pytest.raises(
        SpectralMetricControlAuthorizationError,
        match="spectral_metric_control_execution_authorization_required",
    ):
        validate_spectral_metric_control_authorization(
            proposal,
            receipt=None,
        )
    assert validate_spectral_metric_control_authorization(
        proposal,
        receipt=receipt,
    ) == receipt
    with pytest.raises(
        SpectralMetricControlAuthorizationError,
        match="spectral_metric_control_authorization_mismatch",
    ):
        validate_spectral_metric_control_authorization(
            proposal,
            receipt={**receipt, "identity_panel_sha256": "f" * 64},
        )


def _authorization(proposal: dict[str, object]) -> dict[str, object]:
    frozen = proposal["frozen_contracts"]
    return {
        "approved": True,
        "decision_state": proposal["authorization_state"],
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_contract_hash": frozen["budget_contract_hash"],
        "unique_budget": 12,
        "stage": proposal["stage"],
        "identity_panel_sha256": proposal["identity_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "solver_hash": frozen["solver_hash"],
        "evaluation_hash": frozen["evaluation_hash"],
        "control_contract_hash": frozen["control_contract_hash"],
        "spectral_gate_contract_hash": frozen[
            "spectral_gate_contract_hash"
        ],
        "control_profile_hash": frozen["control_profile_hash"],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "approved_at": "2026-07-30T12:00:00+08:00",
        "approved_by": "user",
    }


def test_prepare_and_execute_control_panel_obey_exact_human_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal_dir = tmp_path / "proposal"
    propose_spectral_metric_control(
        p25_proposal_path=(
            EXPERIMENT_ROOT
            / "p25_spectral_diagnostic_v1"
            / "p25_spectral_diagnostic_proposal.json"
        ),
        p25_completion_path=(
            EXPERIMENT_ROOT
            / "p25_spectral_diagnostic_execution_v1"
            / "completion.json"
        ),
        p25_decision_path=(
            EXPERIMENT_ROOT
            / "p25_spectral_diagnostic_execution_v1"
            / "decision_receipt.json"
        ),
        source_budget_contract_path=(
            EXPERIMENT_ROOT / "governance_v6" / "budget_contract.json"
        ),
        spectral_gate_contract_path=(
            EXPERIMENT_ROOT
            / "p25_spectral_diagnostic_v1"
            / "spectral_gate_contract.json"
        ),
        output_dir=proposal_dir,
        source_root=ROOT / "python" / "src",
        parent_experiment_id="lyx-recovery-filter-profile-v1",
    )
    proposal = read_json(
        proposal_dir / "spectral_metric_control_proposal.json"
    )
    authorization_path = proposal_dir / "execution_authorization.json"
    atomic_write_json(authorization_path, _authorization(proposal))
    unapproved_path = proposal_dir / "unapproved_authorization.json"
    atomic_write_json(unapproved_path, {"approved": False})

    source_governance = tmp_path / "governance-v6"
    source_budget = BudgetContract.approved_v6_p25_diagnostic()
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
    target_governance = tmp_path / "governance-v7"

    with pytest.raises(
        SpectralMetricControlAuthorizationError,
        match="spectral_metric_control_execution_authorization_required",
    ):
        prepare_spectral_metric_control_governance(
            proposal_dir=proposal_dir,
            authorization_receipt_path=unapproved_path,
            source_governance_dir=source_governance,
            governance_dir=target_governance,
            source_root=ROOT / "python" / "src",
        )

    governance = prepare_spectral_metric_control_governance(
        proposal_dir=proposal_dir,
        authorization_receipt_path=authorization_path,
        source_governance_dir=source_governance,
        governance_dir=target_governance,
        source_root=ROOT / "python" / "src",
    )
    assert governance["new_unique_identity_count"] == 12
    assert governance["attempt_registry_summary"][
        "planned_unique_identity_count"
    ] == 12

    def fake_audit(
        record: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        control = SpectralMetricScaleControlContract()
        spectral = StageRSpectralGateContract()
        return {
            "record_id": record.record_id,
            "scene": record.scene,
            "profile_id": "p25-short-low",
            "profile_sha256": proposal["frozen_contracts"][
                "control_profile_hash"
            ],
            "control_version": control.control_version,
            "control_contract_sha256": control.sha256,
            "spectral_gate_contract_sha256": spectral.sha256,
            "prepared_window_count": 3,
            "complete_window_evidence": True,
            "zero_update_weight_max_abs": 0.0,
            "direct_bypass_pass": True,
            "same_scale_zero_update_pass": True,
            "legacy_scale_mismatch_reproduced": True,
            "legacy_to_same_scale_retention_ratio": 0.001,
            "pulse_power_retention_median": {
                "direct_raw_bypass": 1.0,
                "legacy_raw_vs_zero_update_lms": 0.001,
                "same_scale_zero_update_lms": 1.0,
            },
            "lanes": {},
        }

    monkeypatch.setattr(
        control_module,
        "audit_spectral_metric_scale_record",
        fake_audit,
    )
    output_dir = tmp_path / "execution"
    completion = execute_spectral_metric_control(
        proposal_dir=proposal_dir,
        governance_dir=target_governance,
        output_dir=output_dir,
        source_root=ROOT / "python" / "src",
    )
    assert completion["status"] == "legacy_scale_mismatch_confirmed"
    assert completion["diagnostic_result_count"] == 12
    assert completion["diagnostic_run_count"] == 12
    assert completion["parameter_search_run_count"] == 0
    assert completion["independent_bo_run_count"] == 0
    assert execute_spectral_metric_control(
        proposal_dir=proposal_dir,
        governance_dir=target_governance,
        output_dir=output_dir,
        source_root=ROOT / "python" / "src",
    ) == completion
