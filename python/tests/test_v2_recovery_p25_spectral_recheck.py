from __future__ import annotations

from pathlib import Path

import pytest

from ppg_hr.v2 import recovery_p25_spectral_recheck as recheck_module
from ppg_hr.v2.phase2_experiment_io import atomic_write_json, read_json
from ppg_hr.v2.recovery_experiment_governance import (
    AttemptRegistry,
    BudgetContract,
    ExplorationRegistry,
)
from ppg_hr.v2.recovery_p25_spectral_recheck import (
    P25SpectralRecheckAuthorizationError,
    P25SpectralRecheckError,
    build_p25_spectral_recheck_proposal,
    evaluate_p25_spectral_recheck_decision,
    execute_p25_spectral_recheck,
    prepare_p25_spectral_recheck_governance,
    propose_p25_spectral_recheck,
    validate_p25_spectral_recheck_authorization,
)
from ppg_hr.v2.recovery_spectral_gate import StageRSpectralGateContract

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
    REPO_ROOT / "data" / "experiments" / "lyx_recovery_filter_profile"
)
P25_PROFILE_IDS = {
    "p25-short-low",
    "p25-short-mid",
    "p25-long-mid",
}
RECORD_IDS = {
    "jianpan1_LYX_0708",
    "jianpan2_LYX_0708",
    "jianpan3_LYX_0708",
    "kaihe1_LYX_0613",
    "kaihe1_LYX_0617",
    "kaihe3_LYX_0613",
    "run1_LYX_0708",
    "run2_LYX_0708",
    "run3_LYX_0708",
    "xiezi2_LYX_0708",
    "xiezi3_LYX_0708",
    "xiezi4_LYX_0708",
}


def test_build_recheck_freezes_new_v2_identity_product() -> None:
    proposal = build_p25_spectral_recheck_proposal(
        prior_p25_proposal=read_json(
            EXPERIMENT_ROOT
            / "p25_spectral_diagnostic_v1"
            / "p25_spectral_diagnostic_proposal.json"
        ),
        prior_p25_completion=read_json(
            EXPERIMENT_ROOT
            / "p25_spectral_diagnostic_execution_v1"
            / "completion.json"
        ),
        prior_p25_decision=read_json(
            EXPERIMENT_ROOT
            / "p25_spectral_diagnostic_execution_v1"
            / "decision_receipt.json"
        ),
        scale_control_proposal=read_json(
            EXPERIMENT_ROOT
            / "spectral_metric_scale_control_v1"
            / "spectral_metric_control_proposal.json"
        ),
        scale_control_completion=read_json(
            EXPERIMENT_ROOT
            / "spectral_metric_scale_control_execution_v1"
            / "completion.json"
        ),
        scale_control_decision=read_json(
            EXPERIMENT_ROOT
            / "spectral_metric_scale_control_execution_v1"
            / "decision_receipt.json"
        ),
        source_budget_contract=read_json(
            EXPERIMENT_ROOT / "governance_v7" / "budget_contract.json"
        ),
        parent_experiment_id="lyx_recovery_filter_profile_v2",
        solver_hash="a" * 64,
        evaluation_hash="b" * 64,
    )

    identities = proposal["identities"]
    assert len(identities) == 36
    assert {
        (item["filter_profile_id"], item["record_id"])
        for item in identities
    } == {
        (profile_id, record_id)
        for profile_id in P25_PROFILE_IDS
        for record_id in RECORD_IDS
    }
    assert len({item["identity_sha256"] for item in identities}) == 36
    assert {
        (item["stage"], item["attempt_kind"])
        for item in identities
    } == {("filter_profile_p25_spectral_recheck_v2", "diagnostic")}
    assert not {
        item["identity_sha256"]
        for item in identities
    }.intersection(
        {
            item["identity_sha256"]
            for item in read_json(
                EXPERIMENT_ROOT
                / "p25_spectral_diagnostic_v1"
                / "p25_spectral_diagnostic_proposal.json"
            )["identities"]
        }
    )
    assert (
        proposal["frozen_contracts"]["spectral_gate_contract_hash"]
        == StageRSpectralGateContract().sha256
    )
    assert {
        (item["filter_profile_id"], item["fs_target"])
        for item in proposal["profile_panel"]
    } == {(profile_id, 25) for profile_id in P25_PROFILE_IDS}
    assert proposal["unique_budget"] == 36
    assert proposal["worst_case_attempt_budget"] == 72
    assert proposal["independent_bo_authorized"] is False
    assert proposal["automatic_stage_r_execution"] is False
    assert proposal["automatic_stage_f_execution"] is False


SPECTRAL_GATE_NAMES = {
    "prominence_db_delta_pass",
    "visible_top3_rate_delta_pass",
    "hr_band_share_delta_pass",
    "pulse_power_retention_pass",
    "residual_artifact_corr_delta_pass",
    "complete_window_evidence_pass",
}


def _decision_rows(
    *,
    full_pass_profile: str | None = None,
    universal_pulse_failure: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile_id in sorted(P25_PROFILE_IDS):
        for record_id in sorted(RECORD_IDS):
            full_pass = profile_id == full_pass_profile
            gates = {name: full_pass for name in SPECTRAL_GATE_NAMES}
            if universal_pulse_failure:
                gates["pulse_power_retention_pass"] = False
            rows.append(
                {
                    "filter_profile_id": profile_id,
                    "record_id": record_id,
                    "metrics": {"final_motion_mae_bpm": 999.0},
                    "spectral_audit": {
                        "stability_pass": full_pass,
                        "stage_r_spectral_gate": {
                            "spectral_gate_pass": all(gates.values()),
                            "gates": gates,
                        },
                    },
                }
            )
    return rows


def test_corrected_decision_has_only_mechanism_or_stage_r_branches() -> None:
    sentinel = evaluate_p25_spectral_recheck_decision(
        _decision_rows(full_pass_profile="p25-short-mid")
    )
    mechanism = evaluate_p25_spectral_recheck_decision(
        _decision_rows(universal_pulse_failure=True)
    )

    assert sentinel["decision"] == "stage_r_sentinel_revision_candidate"
    assert sentinel["complete_pass_profile_ids"] == ["p25-short-mid"]
    assert sentinel["next_state"] == "awaiting_human_stage_r_revision_decision"
    assert mechanism["decision"] == "p25_failure_review_required"
    assert mechanism["next_state"] == (
        "awaiting_human_filter_mechanism_or_independent_bo_review"
    )
    assert "spectral_metric_control_audit_required" not in {
        sentinel["decision"],
        mechanism["decision"],
    }
    assert set(mechanism["global_gate_failure_counts"]) == SPECTRAL_GATE_NAMES
    assert mechanism["global_gate_failure_counts"] == {
        name: 36 for name in SPECTRAL_GATE_NAMES
    }
    assert mechanism["independent_bo_review_package_generated"] is False
    assert mechanism["independent_bo_authorized"] is False
    assert sentinel["automatic_stage_r_execution"] is False
    assert sentinel["automatic_stage_f_execution"] is False


def test_corrected_decision_fails_closed_on_incomplete_gate_payload() -> None:
    rows = _decision_rows()
    gates = rows[0]["spectral_audit"]["stage_r_spectral_gate"]["gates"]
    del gates["prominence_db_delta_pass"]

    with pytest.raises(
        P25SpectralRecheckError,
        match="p25_spectral_recheck_gate_set_mismatch",
    ):
        evaluate_p25_spectral_recheck_decision(rows)


def _authorization(proposal: dict[str, object]) -> dict[str, object]:
    frozen = proposal["frozen_contracts"]
    return {
        "approved": True,
        "decision_state": proposal["authorization_state"],
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_contract_hash": frozen["budget_contract_hash"],
        "unique_budget": 36,
        "stage": proposal["stage"],
        "identity_panel_sha256": proposal["identity_panel_sha256"],
        "profile_panel_sha256": proposal["profile_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "solver_hash": frozen["solver_hash"],
        "evaluation_hash": frozen["evaluation_hash"],
        "metric_contract_hash": frozen["metric_contract_hash"],
        "spectral_gate_contract_hash": frozen[
            "spectral_gate_contract_hash"
        ],
        "decision_contract_hash": frozen["decision_contract_hash"],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "approved_at": "2026-07-30T20:00:00+08:00",
        "approved_by": "user",
    }


def test_propose_prepare_execute_obey_exact_v8_human_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal_dir = tmp_path / "proposal"
    receipt = propose_p25_spectral_recheck(
        prior_p25_proposal_path=(
            EXPERIMENT_ROOT
            / "p25_spectral_diagnostic_v1"
            / "p25_spectral_diagnostic_proposal.json"
        ),
        prior_p25_completion_path=(
            EXPERIMENT_ROOT
            / "p25_spectral_diagnostic_execution_v1"
            / "completion.json"
        ),
        prior_p25_decision_path=(
            EXPERIMENT_ROOT
            / "p25_spectral_diagnostic_execution_v1"
            / "decision_receipt.json"
        ),
        scale_control_proposal_path=(
            EXPERIMENT_ROOT
            / "spectral_metric_scale_control_v1"
            / "spectral_metric_control_proposal.json"
        ),
        scale_control_completion_path=(
            EXPERIMENT_ROOT
            / "spectral_metric_scale_control_execution_v1"
            / "completion.json"
        ),
        scale_control_decision_path=(
            EXPERIMENT_ROOT
            / "spectral_metric_scale_control_execution_v1"
            / "decision_receipt.json"
        ),
        source_budget_contract_path=(
            EXPERIMENT_ROOT / "governance_v7" / "budget_contract.json"
        ),
        output_dir=proposal_dir,
        source_root=REPO_ROOT / "python" / "src",
        parent_experiment_id="lyx-recovery-filter-profile-v1",
    )
    proposal = read_json(
        proposal_dir / "p25_spectral_recheck_proposal.json"
    )
    assert receipt["proposal_sha256"] == proposal["proposal_sha256"]
    assert receipt["diagnostic_run_count"] == 0
    assert all(
        artifact["path_base"] == "repository_root"
        and not Path(artifact["path"]).is_absolute()
        for artifact in proposal["source_artifacts"].values()
    )

    with pytest.raises(
        P25SpectralRecheckAuthorizationError,
        match="p25_spectral_recheck_execution_authorization_required",
    ):
        validate_p25_spectral_recheck_authorization(
            proposal,
            receipt=None,
        )
    authorization = _authorization(proposal)
    authorization_path = proposal_dir / "execution_authorization.json"
    atomic_write_json(authorization_path, authorization)
    unapproved_path = proposal_dir / "unapproved_authorization.json"
    atomic_write_json(unapproved_path, {"approved": False})

    source_governance = tmp_path / "governance-v7"
    source_budget = BudgetContract.proposed_v7_spectral_metric_control()
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
    target_governance = tmp_path / "governance-v8"
    with pytest.raises(
        P25SpectralRecheckAuthorizationError,
        match="p25_spectral_recheck_execution_authorization_required",
    ):
        prepare_p25_spectral_recheck_governance(
            proposal_dir=proposal_dir,
            authorization_receipt_path=unapproved_path,
            source_governance_dir=source_governance,
            governance_dir=target_governance,
            source_root=REPO_ROOT / "python" / "src",
        )
    assert not target_governance.exists()

    governance = prepare_p25_spectral_recheck_governance(
        proposal_dir=proposal_dir,
        authorization_receipt_path=authorization_path,
        source_governance_dir=source_governance,
        governance_dir=target_governance,
        source_root=REPO_ROOT / "python" / "src",
    )
    assert governance["status"] == "prepared_zero_runs"
    assert governance["new_unique_identity_count"] == 36
    assert governance["attempt_registry_summary"][
        "actual_unique_run_count"
    ] == 0

    def fake_audit(
        profile: object,
        record: object,
        *,
        contract: StageRSpectralGateContract,
    ) -> dict[str, object]:
        passed = profile.profile_id == "p25-short-mid"
        gates = {name: passed for name in SPECTRAL_GATE_NAMES}
        return {
            "record_id": record.record_id,
            "scene": record.scene,
            "stability_pass": passed,
            "stage_r_spectral_gate": {
                "spectral_gate_pass": passed,
                "gates": gates,
            },
            "stage_r_spectral_gate_contract_sha256": contract.sha256,
        }

    monkeypatch.setattr(
        recheck_module,
        "audit_stage_r_profile_record",
        fake_audit,
    )
    output_dir = tmp_path / "execution"
    completion = execute_p25_spectral_recheck(
        proposal_dir=proposal_dir,
        governance_dir=target_governance,
        output_dir=output_dir,
        source_root=REPO_ROOT / "python" / "src",
    )
    assert completion["status"] == "stage_r_sentinel_revision_candidate"
    assert completion["diagnostic_result_count"] == 36
    assert completion["diagnostic_run_count"] == 36
    assert completion["parameter_search_run_count"] == 0
    assert completion["independent_bo_run_count"] == 0
    assert execute_p25_spectral_recheck(
        proposal_dir=proposal_dir,
        governance_dir=target_governance,
        output_dir=output_dir,
        source_root=REPO_ROOT / "python" / "src",
    ) == completion
