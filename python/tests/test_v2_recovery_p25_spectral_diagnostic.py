from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2 import recovery_p25_spectral_diagnostic as p25_module
from ppg_hr.v2 import recovery_stage_r_experiment as stage_r_module
from ppg_hr.v2.phase2_experiment_io import atomic_write_json, read_json
from ppg_hr.v2.recovery_contracts import canonical_sha256
from ppg_hr.v2.recovery_experiment_governance import (
    AttemptRegistry,
    BudgetContract,
    ExplorationRegistry,
)
from ppg_hr.v2.recovery_p25_spectral_diagnostic import (
    P25SpectralDiagnosticAuthorizationError,
    P25SpectralDiagnosticError,
    build_p25_spectral_diagnostic_proposal,
    evaluate_p25_spectral_diagnostic_decision,
    execute_p25_spectral_diagnostic,
    prepare_p25_spectral_diagnostic_governance,
    propose_p25_spectral_diagnostic,
    validate_p25_spectral_diagnostic_authorization,
)
from ppg_hr.v2.recovery_stage_r_common import StageRNumericalResult
from ppg_hr.v2.solver import V2SolverResult

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "data" / "experiments" / "lyx_recovery_filter_profile"
STAGE_R_PROPOSAL_PATH = EXPERIMENT_ROOT / "stage_r_v3" / "stage_r_execution_proposal.json"
STAGE_R_COMPLETION_PATH = EXPERIMENT_ROOT / "stage_r_execution_v1" / "stage_r_completion.json"
PROFILE_LIBRARY_PATH = EXPERIMENT_ROOT / "filter_profiles_v4" / "filter_profile_library_freeze.json"
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


def _proposed_budget_v6() -> dict[str, object]:
    return {
        "contract_version": "lyx_recovery_filter_budget_v6",
        "stage_unique_limits": {
            "fixed_lower_bound_diagnostic": 60,
            "historical_recovery_ab": 24,
            "recovery_sentinel": 108,
            "filter_profile_stability_audit": 64,
            "filter_profile_rate_normalization_exploration": 8,
            "filter_profile_p25_spectral_diagnostic": 36,
            "penalty_interaction": 288,
            "current_role_matrix": 96,
            "rollback_backup_matrix": 96,
            "fold_replay": 12,
        },
        "normal_unique_identity_limit": 780,
        "supplemental_stage": "fold_replay",
        "stage_attempt_kinds": {
            "fixed_lower_bound_diagnostic": "diagnostic",
            "historical_recovery_ab": "formal",
            "recovery_sentinel": "formal",
            "filter_profile_stability_audit": "diagnostic",
            "filter_profile_rate_normalization_exploration": "exploration",
            "filter_profile_p25_spectral_diagnostic": "diagnostic",
            "penalty_interaction": "formal",
            "current_role_matrix": "formal",
            "rollback_backup_matrix": "formal",
            "fold_replay": "formal",
        },
        "max_unique_identities": 792,
        "max_attempts": 1584,
        "retry_limit": 1,
    }


def _publish_proposal(tmp_path: Path) -> Path:
    budget_path = tmp_path / "budget_contract.json"
    atomic_write_json(budget_path, _proposed_budget_v6())
    output_dir = tmp_path / "p25-proposal"
    propose_p25_spectral_diagnostic(
        stage_r_proposal_path=STAGE_R_PROPOSAL_PATH,
        stage_r_completion_path=STAGE_R_COMPLETION_PATH,
        profile_library_path=PROFILE_LIBRARY_PATH,
        budget_contract_path=budget_path,
        metric_contract_path=(EXPERIMENT_ROOT / "stage_r_v3" / "metric_contract.json"),
        spectral_gate_contract_path=(
            EXPERIMENT_ROOT / "stage_r_v3" / "spectral_gate_contract.json"
        ),
        output_dir=output_dir,
        source_root=REPO_ROOT / "python" / "src",
        parent_experiment_id="lyx_recovery_filter_profile_v1",
    )
    return output_dir


def _execution_authorization(
    proposal: dict[str, object],
) -> dict[str, object]:
    frozen = proposal["frozen_contracts"]
    assert isinstance(frozen, dict)
    return {
        "approved": True,
        "decision_state": ("awaiting_human_p25_spectral_diagnostic_decision"),
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_contract_hash": frozen["budget_contract_hash"],
        "unique_budget": 36,
        "stage": "filter_profile_p25_spectral_diagnostic",
        "profile_panel_sha256": proposal["profile_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "solver_hash": frozen["solver_hash"],
        "evaluation_hash": frozen["evaluation_hash"],
        "metric_contract_hash": frozen["metric_contract_hash"],
        "spectral_gate_contract_hash": frozen["spectral_gate_contract_hash"],
        "decision_contract_hash": frozen["decision_contract_hash"],
        "independent_bo_authorized": False,
        "approved_at": "2026-07-30T12:00:00+08:00",
        "approved_by": "test-user",
    }


def test_build_proposal_freezes_exact_p25_profile_record_product() -> None:
    stage_r_proposal = read_json(STAGE_R_PROPOSAL_PATH)
    stage_r_completion = read_json(STAGE_R_COMPLETION_PATH)
    profile_library = read_json(PROFILE_LIBRARY_PATH)

    proposal = build_p25_spectral_diagnostic_proposal(
        stage_r_proposal=stage_r_proposal,
        stage_r_completion=stage_r_completion,
        profile_library=profile_library,
        budget_contract=_proposed_budget_v6(),
        parent_experiment_id="lyx_recovery_filter_profile_v1",
        solver_hash="a" * 64,
        evaluation_hash="b" * 64,
    )

    identities = proposal["identities"]
    assert len(identities) == 36
    assert {(item["filter_profile_id"], item["record_id"]) for item in identities} == {
        (profile_id, record_id) for profile_id in P25_PROFILE_IDS for record_id in RECORD_IDS
    }
    assert len({item["identity_sha256"] for item in identities}) == 36
    assert {(item["stage"], item["attempt_kind"]) for item in identities} == {
        ("filter_profile_p25_spectral_diagnostic", "diagnostic")
    }
    assert all(item["spectral_audit_required"] is True for item in identities)
    assert proposal["unique_budget"] == 36
    assert proposal["worst_case_attempt_budget"] == 72
    assert proposal["independent_bo_authorized"] is False
    assert proposal["may_nominate_recovery_candidate"] is False


def test_proposal_rejects_self_consistent_but_unfrozen_p25_profile() -> None:
    library = deepcopy(read_json(PROFILE_LIBRARY_PATH))
    profile = next(item for item in library["profiles"] if item["profile_id"] == "p25-short-low")
    profile["actual_taps"] = 2
    profile["profile_sha256"] = canonical_sha256(
        {
            "profile_id": profile["profile_id"],
            "design_role": profile["design_role"],
            "fs_target": profile["fs_target"],
            "memory_ms": profile["physical_memory_ms"],
            "nominal_mu": profile["nominal_mu"],
            "recovery_sentinel_role": profile["recovery_sentinel_role"],
            "actual_taps": profile["actual_taps"],
        }
    )
    library.pop("library_sha256")
    library["library_sha256"] = canonical_sha256(library)

    with pytest.raises(
        P25SpectralDiagnosticError,
        match="p25_spectral_profile_contract_mismatch:p25-short-low",
    ):
        build_p25_spectral_diagnostic_proposal(
            stage_r_proposal=read_json(STAGE_R_PROPOSAL_PATH),
            stage_r_completion=read_json(STAGE_R_COMPLETION_PATH),
            profile_library=library,
            budget_contract=_proposed_budget_v6(),
            parent_experiment_id="lyx_recovery_filter_profile_v1",
            solver_hash="a" * 64,
            evaluation_hash="b" * 64,
        )


def test_authorization_must_bind_every_frozen_diagnostic_identity() -> None:
    proposal = build_p25_spectral_diagnostic_proposal(
        stage_r_proposal=read_json(STAGE_R_PROPOSAL_PATH),
        stage_r_completion=read_json(STAGE_R_COMPLETION_PATH),
        profile_library=read_json(PROFILE_LIBRARY_PATH),
        budget_contract=_proposed_budget_v6(),
        parent_experiment_id="lyx_recovery_filter_profile_v1",
        solver_hash="a" * 64,
        evaluation_hash="b" * 64,
    )
    frozen = proposal["frozen_contracts"]
    receipt = {
        "approved": True,
        "decision_state": ("awaiting_human_p25_spectral_diagnostic_decision"),
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_contract_hash": frozen["budget_contract_hash"],
        "unique_budget": 36,
        "stage": "filter_profile_p25_spectral_diagnostic",
        "profile_panel_sha256": proposal["profile_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "solver_hash": frozen["solver_hash"],
        "evaluation_hash": frozen["evaluation_hash"],
        "metric_contract_hash": frozen["metric_contract_hash"],
        "spectral_gate_contract_hash": frozen["spectral_gate_contract_hash"],
        "decision_contract_hash": frozen["decision_contract_hash"],
        "independent_bo_authorized": False,
        "approved_at": "2026-07-30T12:00:00+08:00",
        "approved_by": "test-user",
    }

    with pytest.raises(
        P25SpectralDiagnosticAuthorizationError,
        match="p25_spectral_execution_authorization_required",
    ):
        validate_p25_spectral_diagnostic_authorization(
            proposal,
            receipt=None,
        )
    forged = {**receipt, "profile_panel_sha256": "f" * 64}
    with pytest.raises(
        P25SpectralDiagnosticAuthorizationError,
        match="p25_spectral_authorization_identity_mismatch:profile_panel_sha256",
    ):
        validate_p25_spectral_diagnostic_authorization(
            proposal,
            receipt=forged,
        )
    assert (
        validate_p25_spectral_diagnostic_authorization(
            proposal,
            receipt=receipt,
        )
        == receipt
    )


def test_propose_publishes_zero_run_review_package(
    tmp_path: Path,
) -> None:
    budget_path = tmp_path / "budget_contract.json"
    atomic_write_json(budget_path, _proposed_budget_v6())
    output_dir = tmp_path / "p25-proposal"

    receipt = propose_p25_spectral_diagnostic(
        stage_r_proposal_path=STAGE_R_PROPOSAL_PATH,
        stage_r_completion_path=STAGE_R_COMPLETION_PATH,
        profile_library_path=PROFILE_LIBRARY_PATH,
        budget_contract_path=budget_path,
        metric_contract_path=(EXPERIMENT_ROOT / "stage_r_v3" / "metric_contract.json"),
        spectral_gate_contract_path=(
            EXPERIMENT_ROOT / "stage_r_v3" / "spectral_gate_contract.json"
        ),
        output_dir=output_dir,
        source_root=REPO_ROOT / "python" / "src",
        parent_experiment_id="lyx_recovery_filter_profile_v1",
    )

    proposal = read_json(output_dir / "p25_spectral_diagnostic_proposal.json")
    budget_request = read_json(output_dir / "budget_amendment_request.json")
    assert receipt["status"] == "awaiting_human_execution_authorization"
    assert receipt["diagnostic_solver_run_count"] == 0
    assert receipt["independent_bo_run_count"] == 0
    assert proposal["proposal_sha256"] == receipt["proposal_sha256"]
    assert len(proposal["identities"]) == 36
    assert set(proposal["source_artifacts"]) == {
        "stage_r_proposal",
        "stage_r_completion",
        "profile_library",
        "budget_contract",
        "metric_contract",
        "spectral_gate_contract",
    }
    assert budget_request["proposal_sha256"] == proposal["proposal_sha256"]
    assert budget_request["added_unique_identities"] == 36
    assert budget_request["max_unique_identities"] == 792
    assert budget_request["max_attempts"] == 1584
    assert not (output_dir / "execution_authorization.json").exists()


def test_prepare_governance_requires_exact_authorization_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal_dir = _publish_proposal(tmp_path)
    proposal = read_json(proposal_dir / "p25_spectral_diagnostic_proposal.json")
    source_governance = tmp_path / "governance-v5"
    source_budget = BudgetContract.approved_v5()
    exploration = ExplorationRegistry.zero_budget_v1()
    AttemptRegistry.create(
        source_governance / "attempt_registry.json",
        budget_contract=source_budget,
        exploration_registry=exploration,
    )
    atomic_write_json(
        source_governance / "budget_contract.json",
        source_budget.to_dict(),
    )
    atomic_write_json(
        source_governance / "exploration_registry.json",
        exploration.to_dict(),
    )
    target_governance = tmp_path / "governance-v6"

    with pytest.raises(
        P25SpectralDiagnosticAuthorizationError,
        match="p25_spectral_execution_authorization_required",
    ):
        prepare_p25_spectral_diagnostic_governance(
            proposal_dir=proposal_dir,
            authorization_receipt_path=None,
            source_governance_dir=source_governance,
            governance_dir=target_governance,
            source_root=REPO_ROOT / "python" / "src",
        )
    assert not target_governance.exists()

    authorization_path = tmp_path / "authorization.json"
    atomic_write_json(
        authorization_path,
        _execution_authorization(proposal),
    )
    monkeypatch.setattr(
        p25_module,
        "runtime_source_identity",
        lambda *_args, **_kwargs: {
            "source_bundle_sha256": "f" * 64,
        },
    )
    with pytest.raises(
        P25SpectralDiagnosticError,
        match="p25_spectral_runtime_source_identity_mismatch",
    ):
        prepare_p25_spectral_diagnostic_governance(
            proposal_dir=proposal_dir,
            authorization_receipt_path=authorization_path,
            source_governance_dir=source_governance,
            governance_dir=target_governance,
            source_root=REPO_ROOT / "python" / "src",
        )
    assert not target_governance.exists()
    monkeypatch.undo()

    receipt = prepare_p25_spectral_diagnostic_governance(
        proposal_dir=proposal_dir,
        authorization_receipt_path=authorization_path,
        source_governance_dir=source_governance,
        governance_dir=target_governance,
        source_root=REPO_ROOT / "python" / "src",
    )

    target_budget = BudgetContract.approved_v6_p25_diagnostic()
    migrated = AttemptRegistry.open(
        target_governance / "attempt_registry.json",
        budget_contract=target_budget,
        exploration_registry=exploration,
    )
    assert receipt["status"] == "prepared_zero_runs"
    assert receipt["new_unique_identity_count"] == 36
    assert migrated.summary()["planned_unique_identity_count"] == 36
    assert migrated.summary()["actual_unique_run_count"] == 0


def _decision_rows(
    *,
    full_pass_profile: str | None = None,
    universal_pulse_failure: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile_id in sorted(P25_PROFILE_IDS):
        for record_id in sorted(RECORD_IDS):
            full_pass = profile_id == full_pass_profile
            pulse_pass = (
                False
                if universal_pulse_failure
                else (
                    full_pass
                    or (profile_id == "p25-long-mid" and record_id == sorted(RECORD_IDS)[0])
                )
            )
            rows.append(
                {
                    "filter_profile_id": profile_id,
                    "record_id": record_id,
                    "metrics": {
                        "final_motion_mae_bpm": 2.0,
                        "longest_e10_run_windows": 1,
                    },
                    "spectral_audit": {
                        "stability_pass": full_pass,
                        "stage_r_spectral_gate": {
                            "spectral_gate_pass": full_pass,
                            "gates": {
                                "prominence_db_delta_pass": full_pass,
                                "visible_top3_rate_delta_pass": (full_pass),
                                "hr_band_share_delta_pass": full_pass,
                                "pulse_power_retention_pass": pulse_pass,
                                "residual_artifact_corr_delta_pass": (full_pass),
                                "complete_window_evidence_pass": (full_pass),
                            },
                        },
                    },
                }
            )
    return rows


def test_decision_is_spectral_only_and_has_three_exclusive_branches() -> None:
    sentinel_rows = _decision_rows(full_pass_profile="p25-short-mid")
    sentinel = evaluate_p25_spectral_diagnostic_decision(sentinel_rows)
    tampered_hr = [
        {
            **row,
            "metrics": {
                "final_motion_mae_bpm": 999.0,
                "longest_e10_run_windows": 999,
            },
        }
        for row in sentinel_rows
    ]

    assert sentinel == evaluate_p25_spectral_diagnostic_decision(tampered_hr)
    assert sentinel["decision"] == "stage_r_sentinel_revision_candidate"
    assert sentinel["complete_pass_profile_ids"] == ["p25-short-mid"]
    assert (
        evaluate_p25_spectral_diagnostic_decision(_decision_rows(universal_pulse_failure=True))[
            "decision"
        ]
        == "spectral_metric_control_audit_required"
    )
    assert (
        evaluate_p25_spectral_diagnostic_decision(_decision_rows())["decision"]
        == "filter_mechanism_revision_required"
    )
    incomplete = _decision_rows()
    gates = incomplete[0]["spectral_audit"]["stage_r_spectral_gate"]["gates"]
    del gates["prominence_db_delta_pass"]
    with pytest.raises(
        P25SpectralDiagnosticError,
        match="p25_spectral_decision_gate_set_mismatch",
    ):
        evaluate_p25_spectral_diagnostic_decision(incomplete)


def _fake_p25_numerical_result(
    _item: dict[str, object],
    _spectral_audit_dir: Path,
) -> StageRNumericalResult:
    return StageRNumericalResult(
        solver_result=V2SolverResult(
            HR=np.asarray([60.0, 61.0]),
            err_stats={"mean": 0.0},
            metadata={"smooth_win_len": 5},
            window_table=(),
        ),
        metrics={"final_motion_mae_bpm": 999.0},
        spectral_audit={
            "stability_pass": False,
            "stage_r_spectral_gate": {
                "spectral_gate_pass": False,
                "gates": {
                    "prominence_db_delta_pass": False,
                    "visible_top3_rate_delta_pass": False,
                    "hr_band_share_delta_pass": False,
                    "pulse_power_retention_pass": False,
                    "residual_artifact_corr_delta_pass": False,
                    "complete_window_evidence_pass": False,
                },
            },
        },
    )


def test_execute_is_resumable_and_commits_spectral_decision_last(
    tmp_path: Path,
) -> None:
    proposal_dir = _publish_proposal(tmp_path)
    proposal = read_json(proposal_dir / "p25_spectral_diagnostic_proposal.json")
    source_governance = tmp_path / "source-governance"
    source_budget = BudgetContract.approved_v5()
    exploration = ExplorationRegistry.zero_budget_v1()
    AttemptRegistry.create(
        source_governance / "attempt_registry.json",
        budget_contract=source_budget,
        exploration_registry=exploration,
    )
    atomic_write_json(
        source_governance / "budget_contract.json",
        source_budget.to_dict(),
    )
    atomic_write_json(
        source_governance / "exploration_registry.json",
        exploration.to_dict(),
    )
    authorization_path = tmp_path / "execution-authorization.json"
    atomic_write_json(
        authorization_path,
        _execution_authorization(proposal),
    )
    governance = tmp_path / "governance-v6"
    prepare_p25_spectral_diagnostic_governance(
        proposal_dir=proposal_dir,
        authorization_receipt_path=authorization_path,
        source_governance_dir=source_governance,
        governance_dir=governance,
        source_root=REPO_ROOT / "python" / "src",
    )
    output = tmp_path / "execution"
    progress: list[dict[str, object]] = []

    completion = execute_p25_spectral_diagnostic(
        proposal_dir=proposal_dir,
        governance_dir=governance,
        output_dir=output,
        source_root=REPO_ROOT / "python" / "src",
        _numerical_runner=_fake_p25_numerical_result,
        progress_callback=progress.append,
    )

    assert completion["status"] == "spectral_metric_control_audit_required"
    assert completion["diagnostic_result_count"] == 36
    assert completion["diagnostic_solver_run_count"] == 36
    assert completion["independent_bo_run_count"] == 0
    assert completion["algorithm_level_holdout"] is False
    assert completion["evidence_class"] == "development_reuse_pilot"
    assert len(progress) == 36
    decision = read_json(output / "decision_receipt.json")
    assert decision["decision"] == completion["status"]
    profile_summary = read_json(output / "profile_gate_summary.json")
    assert set(profile_summary["profiles"]) == P25_PROFILE_IDS
    assert (output / "completion.json").is_file()
    assert len(list((output / "spectral_audits").rglob("*.json"))) == 36
    rerun = execute_p25_spectral_diagnostic(
        proposal_dir=proposal_dir,
        governance_dir=governance,
        output_dir=output,
        source_root=REPO_ROOT / "python" / "src",
        _numerical_runner=lambda _item, _audit_dir: pytest.fail(
            "valid completion must not rerun diagnostics"
        ),
    )
    assert rerun == completion
    tampered_completion = {
        **completion,
        "status": "stage_r_sentinel_revision_candidate",
        "next_state": "awaiting_human_stage_r_revision_decision",
    }
    tampered_completion.pop("completion_sha256")
    tampered_completion["completion_sha256"] = canonical_sha256(tampered_completion)
    atomic_write_json(output / "completion.json", tampered_completion)
    with pytest.raises(
        P25SpectralDiagnosticError,
        match="p25_spectral_completion_governance_receipt_mismatch",
    ):
        execute_p25_spectral_diagnostic(
            proposal_dir=proposal_dir,
            governance_dir=governance,
            output_dir=output,
            source_root=REPO_ROOT / "python" / "src",
            _numerical_runner=_fake_p25_numerical_result,
        )
    atomic_write_json(output / "completion.json", completion)

    execution_receipt_path = governance / "p25_spectral_execution_receipt.json"
    tampered_receipt = read_json(execution_receipt_path)
    tampered_receipt["status"] = "tampered"
    atomic_write_json(execution_receipt_path, tampered_receipt)
    with pytest.raises(
        P25SpectralDiagnosticError,
        match="p25_spectral_completion_governance_receipt_mismatch",
    ):
        execute_p25_spectral_diagnostic(
            proposal_dir=proposal_dir,
            governance_dir=governance,
            output_dir=output,
            source_root=REPO_ROOT / "python" / "src",
            _numerical_runner=_fake_p25_numerical_result,
        )


def test_shared_runner_audits_non_sentinel_p25_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    proposal = build_p25_spectral_diagnostic_proposal(
        stage_r_proposal=read_json(STAGE_R_PROPOSAL_PATH),
        stage_r_completion=read_json(STAGE_R_COMPLETION_PATH),
        profile_library=read_json(PROFILE_LIBRARY_PATH),
        budget_contract=_proposed_budget_v6(),
        parent_experiment_id="lyx_recovery_filter_profile_v1",
        solver_hash="a" * 64,
        evaluation_hash="b" * 64,
    )
    identity = dict(proposal["identities"][0])
    observed: dict[str, object] = {}

    @dataclass(frozen=True)
    class DummyMetrics:
        final_motion_mae_bpm: float = 1.0

    monkeypatch.setattr(
        stage_r_module,
        "solve_v2",
        lambda _config: V2SolverResult(
            HR=np.asarray([60.0, 61.0]),
            err_stats={},
            metadata={},
            window_table=(),
        ),
    )
    monkeypatch.setattr(
        stage_r_module,
        "load_v2_reference",
        lambda _path: np.asarray([[0.0, 60.0], [1.0, 61.0]]),
    )
    monkeypatch.setattr(
        stage_r_module,
        "evaluate_recovery_profile_metrics",
        lambda *_args, **_kwargs: DummyMetrics(),
    )

    def fake_audit(profile, _record, *, contract):
        observed["sentinel_role"] = profile.recovery_sentinel_role
        observed["contract_sha256"] = contract.sha256
        return {
            "stability_pass": True,
            "stage_r_spectral_gate": {
                "spectral_gate_pass": True,
                "gates": {
                    "prominence_db_delta_pass": True,
                    "visible_top3_rate_delta_pass": True,
                    "hr_band_share_delta_pass": True,
                    "pulse_power_retention_pass": True,
                    "residual_artifact_corr_delta_pass": True,
                    "complete_window_evidence_pass": True,
                },
            },
        }

    monkeypatch.setattr(
        stage_r_module,
        "audit_stage_r_profile_record",
        fake_audit,
    )

    result = stage_r_module.run_stage_r_numerical_identity(
        identity,
        tmp_path / "spectral-audits",
    )

    assert observed["sentinel_role"] is None
    assert result.spectral_audit["stability_pass"] is True
