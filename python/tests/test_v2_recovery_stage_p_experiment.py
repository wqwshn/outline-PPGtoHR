from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ppg_hr.v2 import (
    recovery_interaction_execution as interaction_execution,
)
from ppg_hr.v2 import (
    recovery_pre_fold_execution as pre_fold_execution,
)
from ppg_hr.v2 import recovery_stage_p_execution as stage_p_execution
from ppg_hr.v2 import recovery_stage_p_plan as stage_p_plan
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
from ppg_hr.v2.recovery_interaction_execution import (
    execute_rollback_backup_proposal,
    propose_rollback_backup_execution,
)
from ppg_hr.v2.recovery_interaction_resolution import (
    build_final_interaction_audit,
    build_rollback_backup_proposal,
    resolve_recovery_interaction,
)
from ppg_hr.v2.recovery_pre_fold_execution import (
    propose_historical_recovery_ab_execution,
)
from ppg_hr.v2.recovery_pre_fold_gate import (
    build_historical_recovery_ab_proposal,
    build_historical_recovery_ab_report,
    evaluate_pre_fold_independent_bo_gate,
    publish_pre_fold_independent_bo_gate,
    publish_stage_r_no_safe_independent_bo_gate,
)
from ppg_hr.v2.recovery_profile_upper_bound import (
    ProfileUpperBoundError,
    build_sample_in_upper_bound_payloads,
    selection_recovery_delay,
)
from ppg_hr.v2.recovery_stage_f_experiment import build_stage_f_proposal
from ppg_hr.v2.recovery_stage_p_contracts import (
    merge_identity_result_metadata,
)
from ppg_hr.v2.recovery_stage_p_experiment import (
    StagePPlanError,
    build_penalty_interaction_report,
    build_stage_p_proposal,
    execute_stage_p_proposal,
    propose_stage_p_execution,
)
from ppg_hr.v2.recovery_stage_r_common import StageRNumericalResult
from tests.test_v2_recovery_stage_f_experiment import (
    _stage_f_inputs,
    _synthetic_recovery_metrics,
    _synthetic_solver_result,
    _synthetic_spectral_evidence,
)


def _with_hash(
    payload: dict[str, object],
    field: str,
) -> dict[str, object]:
    result = dict(payload)
    result[field] = canonical_sha256(payload)
    return result


def test_profile_upper_bound_normalizes_none_and_right_censored_delay() -> None:
    assert (
        selection_recovery_delay(
            {
                "max_recovered_delay_s": None,
                "recovery_episode_count": 0,
                "total_window_count": 120,
            }
        )
        == 0.0
    )
    assert (
        selection_recovery_delay(
            {
                "max_recovered_delay_s": None,
                "recovery_episode_count": 2,
                "right_censored_recovery_count": 2,
                "total_window_count": 120,
            }
        )
        == 120.0
    )
    assert (
        selection_recovery_delay(
            {
                "max_recovered_delay_s": 3.5,
                "recovery_episode_count": 1,
                "total_window_count": 120,
            }
        )
        == 3.5
    )
    with pytest.raises(
        ProfileUpperBoundError,
        match="sample_in_upper_bound_recovery_delay_invalid",
    ):
        selection_recovery_delay(
            {
                "max_recovered_delay_s": -1.0,
                "recovery_episode_count": 1,
                "total_window_count": 120,
            }
        )


def _penalty_registry() -> dict[str, object]:
    candidates = [
        _with_hash(
            {
                "penalty_id": "current_soft_penalty_control_v1",
                "mechanism_complexity": 0,
            },
            "candidate_sha256",
        ),
        _with_hash(
            {
                "penalty_id": "resolution_adaptive_width_v1",
                "mechanism_complexity": 1,
            },
            "candidate_sha256",
        ),
        _with_hash(
            {
                "penalty_id": "trusted_history_corridor_v1",
                "mechanism_complexity": 2,
            },
            "candidate_sha256",
        ),
    ]
    return _with_hash(
        {
            "candidate_count": 3,
            "penalty_count": 3,
            "new_penalty_count": 2,
            "control_penalty_id": "current_soft_penalty_control_v1",
            "no_fourth_strategy_after_freeze": True,
            "selection_ranking_key": [
                "hard_gate_failure_count",
                "right_censored_recovery_count",
                "worst_l10",
                "worst_mae",
                "mean_mae",
                "mechanism_complexity",
                "penalty_id",
            ],
            "candidates": candidates,
        },
        "registry_sha256",
    )


def _stage_p_inputs(tmp_path: Path) -> dict[str, object]:
    inputs = _stage_f_inputs(tmp_path)
    penalty_registry = _penalty_registry()
    inputs["penalty_registry"] = penalty_registry
    stage_r_proposal = deepcopy(inputs["stage_r_proposal"])
    stage_r_proposal.pop("proposal_sha256")
    stage_r_proposal["frozen_contracts"]["penalty_registry_hash"] = penalty_registry[
        "registry_sha256"
    ]
    stage_r_proposal["proposal_sha256"] = canonical_sha256(stage_r_proposal)
    inputs["stage_r_proposal"] = stage_r_proposal
    stage_r_completion = deepcopy(inputs["stage_r_completion"])
    stage_r_completion.pop("completion_sha256")
    stage_r_completion["proposal_sha256"] = stage_r_proposal["proposal_sha256"]
    stage_r_completion["completion_sha256"] = canonical_sha256(stage_r_completion)
    inputs["stage_r_completion"] = stage_r_completion
    stage_f = build_stage_f_proposal(**inputs)
    rows = []
    current_role_rows = []
    for task in stage_f["logical_tasks"]:
        if task["matrix_role"] not in {
            "provisional_recovery",
            "same_role_current_control",
        }:
            continue
        identity = next(
            item
            for item in stage_f["identities"]
            if item["identity_sha256"] == task["identity_sha256"]
        )
        result_row = {
            **identity,
            **task,
            "metrics": {
                "longest_e10_run_windows": 4,
                "longest_e20_run_windows": 1,
                "final_motion_mae_bpm": 3.0,
                "right_censored_recovery_count": 0,
                "max_recovered_delay_s": 2.0,
                "max_rise_underestimate_bpm": (1.0 if identity["true_rise_applicable"] else None),
            },
            "spectral_audit": {
                "stability_pass": True,
                "spectral_gate_pass": True,
                "stage_r_spectral_gate": {
                    "spectral_gate_pass": True,
                    "valid_window_count": 1,
                    "invalid_window_count": 0,
                    "prominence_db_delta_median": 1.0,
                    "visible_top3_rate_delta": 0.0,
                    "hr_band_share_delta_median": 0.1,
                    "pulse_power_retention_median": 0.9,
                    "residual_artifact_corr_delta_median": -0.1,
                    "window_metrics": [
                        {
                            "visible_top3_before": True,
                            "visible_top3_after": True,
                            "prominence_db_delta": 1.0,
                            "hr_band_share_delta": 0.1,
                            "pulse_power_retention": 0.9,
                            "residual_artifact_corr_before": 0.4,
                            "residual_artifact_corr_after": 0.3,
                            "residual_artifact_corr_delta": -0.1,
                        }
                    ],
                },
                "audit_sha256": canonical_sha256(
                    {
                        "profile_id": identity["filter_profile_id"],
                        "record_id": identity["record_id"],
                    }
                ),
            },
            "qualification": {
                "qualified": True,
                "elimination_reasons": [],
                "independent_delta_mae_bpm": 0.0,
                "current_delta_mae_bpm": 0.0,
            },
        }
        if task["matrix_role"] == "provisional_recovery":
            rows.append(result_row)
        else:
            current_role_rows.append(result_row)
    matrix = _with_hash(
        {
            "matrix_version": "lyx_stage_f_profile_matrix_v1",
            "matrix_role": "provisional_recovery",
            "algorithm_level_holdout": False,
            "row_count": 96,
            "unique_spectral_audit_count": 96,
            "rows": rows,
        },
        "matrix_sha256",
    )
    current_role_matrix = _with_hash(
        {
            "matrix_version": "lyx_stage_f_current_role_matrix_v1",
            "matrix_role": "same_role_current_control",
            "algorithm_level_holdout": False,
            "row_count": 96,
            "unique_spectral_audit_count": 96,
            "rows": current_role_rows,
        },
        "matrix_sha256",
    )
    completion = _with_hash(
        {
            "completion_version": "lyx_stage_f_completion_v1",
            "status": "complete",
            "proposal_sha256": stage_f["proposal_sha256"],
            "independent_bo_run_count": 0,
            "next_state": "ready_for_penalty_interaction_completion",
        },
        "completion_sha256",
    )
    return {
        "stage_f_proposal": stage_f,
        "stage_f_completion": completion,
        "stage_f_profile_matrix": matrix,
        "stage_f_current_role_matrix": current_role_matrix,
        "penalty_registry": penalty_registry,
        "budget_contract": inputs["budget_contract"],
        "parent_experiment_id": inputs["parent_experiment_id"],
        "solver_hash": inputs["solver_hash"],
        "metric_contract_hash": inputs["metric_contract_hash"],
        "evaluation_hash": "8" * 64,
    }


def test_stage_p_proposal_reuses_96_and_freezes_exactly_192_new_identities(
    tmp_path: Path,
) -> None:
    proposal = build_stage_p_proposal(**_stage_p_inputs(tmp_path))

    assert proposal["logical_task_count"] == 288
    assert proposal["reused_stage_f_result_count"] == 96
    assert proposal["planned_new_unique_identity_count"] == 192
    assert len(proposal["identities"]) == 192
    assert len({identity["identity_sha256"] for identity in proposal["identities"]}) == 192
    assert {identity["penalty_candidate_id"] for identity in proposal["identities"]} == {
        "resolution_adaptive_width_v1",
        "trusted_history_corridor_v1",
    }
    assert {task["penalty_candidate_id"] for task in proposal["logical_tasks"]} == {
        "current_soft_penalty_control_v1",
        "resolution_adaptive_width_v1",
        "trusted_history_corridor_v1",
    }
    assert all(identity["stage"] == "penalty_interaction" for identity in proposal["identities"])
    assert proposal["independent_bo_authorized"] is False


def test_stage_p_proposal_rejects_a_fourth_penalty(
    tmp_path: Path,
) -> None:
    inputs = _stage_p_inputs(tmp_path)
    registry = deepcopy(inputs["penalty_registry"])
    registry.pop("registry_sha256")
    registry["penalty_count"] = 4
    registry["candidates"].append(
        _with_hash(
            {
                "penalty_id": "unregistered_fourth",
                "mechanism_complexity": 3,
            },
            "candidate_sha256",
        )
    )
    registry["registry_sha256"] = canonical_sha256(registry)
    inputs["penalty_registry"] = registry

    with pytest.raises(
        StagePPlanError,
        match="stage_p_penalty_registry_mismatch",
    ):
        build_stage_p_proposal(**inputs)


def test_stage_p_proposal_rejects_tampered_penalty_registry_outer_hash(
    tmp_path: Path,
) -> None:
    inputs = _stage_p_inputs(tmp_path)
    registry = deepcopy(inputs["penalty_registry"])
    candidate = registry["candidates"][1]
    candidate.pop("candidate_sha256")
    candidate["mechanism_complexity"] = 99
    candidate["candidate_sha256"] = canonical_sha256(candidate)
    inputs["penalty_registry"] = registry

    with pytest.raises(
        StagePPlanError,
        match="stage_p_penalty_registry_hash_mismatch",
    ):
        build_stage_p_proposal(**inputs)


def test_stage_p_proposal_rejects_rehashed_registry_after_stage_f_freeze(
    tmp_path: Path,
) -> None:
    inputs = _stage_p_inputs(tmp_path)
    registry = deepcopy(inputs["penalty_registry"])
    registry.pop("registry_sha256")
    candidate = registry["candidates"][1]
    candidate.pop("candidate_sha256")
    candidate["mechanism_complexity"] = 99
    candidate["candidate_sha256"] = canonical_sha256(candidate)
    registry["registry_sha256"] = canonical_sha256(registry)
    inputs["penalty_registry"] = registry

    with pytest.raises(
        StagePPlanError,
        match="stage_p_frozen_contract_mismatch",
    ):
        build_stage_p_proposal(**inputs)


def test_stage_p_penalty_selection_is_lexicographic(
    tmp_path: Path,
) -> None:
    proposal = build_stage_p_proposal(**_stage_p_inputs(tmp_path))
    current_rows = deepcopy(_stage_p_inputs(tmp_path)["stage_f_profile_matrix"]["rows"])
    current_role_rows = deepcopy(_stage_p_inputs(tmp_path)["stage_f_current_role_matrix"]["rows"])
    new_rows = []
    for identity in proposal["identities"]:
        row = {
            **identity,
            "metrics": {
                "longest_e10_run_windows": 4,
                "longest_e20_run_windows": 1,
                "final_motion_mae_bpm": (
                    2.5
                    if identity["penalty_candidate_id"] == "trusted_history_corridor_v1"
                    else 3.0
                ),
                "right_censored_recovery_count": 0,
                "max_recovered_delay_s": 2.0,
                "max_rise_underestimate_bpm": (1.0 if identity["true_rise_applicable"] else None),
            },
            "spectral_audit": next(
                row["spectral_audit"]
                for row in current_rows
                if row["filter_profile_id"] == identity["filter_profile_id"]
                and row["record_id"] == identity["record_id"]
            ),
        }
        new_rows.append(row)

    report = build_penalty_interaction_report(
        proposal=proposal,
        current_rows=current_rows,
        current_role_rows=current_role_rows,
        new_rows=new_rows,
    )

    assert report["logical_result_count"] == 288
    assert report["selected_penalty_id"] == "trusted_history_corridor_v1"
    assert report["next_state"] == "ready_for_rollback_backup_proposal"
    no_backup_proposal = deepcopy(proposal)
    no_backup_proposal.pop("proposal_sha256")
    no_backup_proposal["rollback_backup_id"] = None
    no_backup_proposal["proposal_sha256"] = canonical_sha256(no_backup_proposal)
    no_backup_report = build_penalty_interaction_report(
        proposal=no_backup_proposal,
        current_rows=current_rows,
        current_role_rows=current_role_rows,
        new_rows=new_rows,
    )
    assert no_backup_report["next_state"] == "ready_for_rollback_backup_proposal"


def test_stage_p_new_penalties_use_the_current_recovery_role_as_control(
    tmp_path: Path,
) -> None:
    inputs = _stage_p_inputs(tmp_path)
    proposal = build_stage_p_proposal(**inputs)
    current_penalty_rows = deepcopy(inputs["stage_f_profile_matrix"]["rows"])
    current_role_rows = deepcopy(inputs["stage_f_current_role_matrix"]["rows"])
    for row in current_role_rows:
        row["metrics"]["final_motion_mae_bpm"] = 0.0
    new_rows = []
    for identity in proposal["identities"]:
        new_rows.append(
            {
                **identity,
                "metrics": {
                    "longest_e10_run_windows": 4,
                    "longest_e20_run_windows": 1,
                    "final_motion_mae_bpm": 3.0,
                    "right_censored_recovery_count": 0,
                    "max_recovered_delay_s": 2.0,
                    "max_rise_underestimate_bpm": (
                        1.0 if identity["true_rise_applicable"] else None
                    ),
                },
                "spectral_audit": next(
                    row["spectral_audit"]
                    for row in current_penalty_rows
                    if row["filter_profile_id"] == identity["filter_profile_id"]
                    and row["record_id"] == identity["record_id"]
                ),
            }
        )

    report = build_penalty_interaction_report(
        proposal=proposal,
        current_rows=current_penalty_rows,
        current_role_rows=current_role_rows,
        new_rows=new_rows,
    )

    new_penalty_rows = [
        row
        for row in report["rows"]
        if row["penalty_candidate_id"] != "current_soft_penalty_control_v1"
    ]
    assert all(
        "current_mae_gate" in row["qualification"]["elimination_reasons"]
        for row in new_penalty_rows
    )


def test_cached_result_restores_frozen_penalty_and_profile_metadata(
    tmp_path: Path,
) -> None:
    proposal = build_stage_p_proposal(**_stage_p_inputs(tmp_path))
    item = proposal["identities"][0]
    cache_view = {
        "identity_sha256": item["identity_sha256"],
        "record_id": item["record_id"],
        "filter_profile_id": item["filter_profile_id"],
        "metrics": {"final_motion_mae_bpm": 3.0},
    }

    merged = merge_identity_result_metadata(
        item=item,
        row=cache_view,
    )

    assert merged["penalty_candidate_id"] == item["penalty_candidate_id"]
    assert merged["actual_taps"] == item["actual_taps"]
    assert merged["recovery_candidate_id"] == item["recovery_candidate_id"]
    assert merged["solver_hash"] == item["solver_hash"]
    assert merged["config"] == item["config"]
    assert merged["data_path"] == item["data_path"]
    assert merged["reference_path"] == item["reference_path"]


def _write_stage_p_sources(
    tmp_path: Path,
    inputs: dict[str, object],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in (
        "stage_f_proposal",
        "stage_f_completion",
        "stage_f_profile_matrix",
        "stage_f_current_role_matrix",
        "penalty_registry",
        "budget_contract",
    ):
        path = tmp_path / f"{name}.json"
        atomic_write_json(path, inputs[name])
        paths[f"{name}_path"] = path
    return paths


def _patch_stage_p_runtime(monkeypatch) -> None:
    def fake_runtime_identity(
        _source_root: Path,
        *,
        root_modules=None,
    ) -> dict[str, object]:
        bundle = "8" * 64 if root_modules else "1" * 64
        return {
            "source_files": {
                "ppg_hr/v2/recovery_stage_p_experiment.py": bundle,
            },
            "source_bundle_sha256": bundle,
        }

    for module in (stage_p_plan, stage_p_execution):
        monkeypatch.setattr(
            module,
            "runtime_source_identity",
            fake_runtime_identity,
        )
    monkeypatch.setattr(
        stage_p_plan,
        "stage_r_metric_contract_v1",
        lambda: {
            "contract_version": "test_metric",
            "contract_sha256": canonical_sha256({"contract_version": "test_metric"}),
        },
    )


def _publish_stage_p_proposal(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Path, dict[str, object]]:
    inputs = _stage_p_inputs(tmp_path)
    paths = _write_stage_p_sources(tmp_path, inputs)
    _patch_stage_p_runtime(monkeypatch)
    proposal_dir = tmp_path / "stage_p_proposal"
    receipt = propose_stage_p_execution(
        **paths,
        output_dir=proposal_dir,
        source_root=Path(__file__).parents[1] / "src",
        parent_experiment_id=str(inputs["parent_experiment_id"]),
    )
    return proposal_dir, receipt


def test_stage_p_publication_is_atomic_and_zero_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proposal_dir, receipt = _publish_stage_p_proposal(
        tmp_path,
        monkeypatch,
    )

    assert receipt["formal_solver_run_count"] == 0
    assert receipt["planned_new_unique_identity_count"] == 192
    assert receipt["reused_stage_f_result_count"] == 96
    assert set(path.name for path in proposal_dir.iterdir()) == {
        "metric_contract.json",
        "solver_source_identity.json",
        "evaluation_source_identity.json",
        "stage_p_execution_proposal.json",
        "proposal_receipt.json",
    }


def test_stage_p_execution_runs_only_new_identities_and_is_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proposal_dir, _ = _publish_stage_p_proposal(
        tmp_path,
        monkeypatch,
    )
    governance_dir = tmp_path / "governance"
    budget = BudgetContract.approved_v5()
    exploration = ExplorationRegistry.zero_budget_v1()
    atomic_write_json(
        governance_dir / "budget_contract.json",
        budget.to_dict(),
    )
    atomic_write_json(
        governance_dir / "exploration_registry.json",
        exploration.to_dict(),
    )
    AttemptRegistry.create(
        governance_dir / "attempt_registry.json",
        budget_contract=budget,
        exploration_registry=exploration,
    )
    calls: list[str] = []

    def numerical_runner(
        item: dict[str, object],
        _spectral_dir: Path,
    ) -> StageRNumericalResult:
        calls.append(str(item["identity_sha256"]))
        true_rise = bool(item["true_rise_applicable"])
        metrics = _synthetic_recovery_metrics(true_rise=true_rise)
        return StageRNumericalResult(
            solver_result=_synthetic_solver_result(),
            metrics={
                **metrics.__dict__,
                "final_motion_mae_bpm": (
                    2.5 if item["penalty_candidate_id"] == "trusted_history_corridor_v1" else 3.0
                ),
            },
            spectral_audit=_synthetic_spectral_evidence(
                profile_id=str(item["filter_profile_id"]),
                record_id=str(item["record_id"]),
                include_audit_hash=True,
            ),
        )

    output_dir = tmp_path / "stage_p_output"
    completion = execute_stage_p_proposal(
        proposal_dir=proposal_dir,
        governance_dir=governance_dir,
        output_dir=output_dir,
        source_root=Path(__file__).parents[1] / "src",
        _numerical_runner=numerical_runner,
    )

    assert len(calls) == 192
    assert len(set(calls)) == 192
    assert completion["status"] == "selected"
    assert completion["logical_result_count"] == 288
    assert completion["new_formal_result_count"] == 192
    assert completion["reused_stage_f_result_count"] == 96
    assert completion["selected_penalty_id"] == "trusted_history_corridor_v1"
    assert completion["independent_bo_run_count"] == 0
    assert (
        read_json(output_dir / "penalty_selection_receipt.json")["no_fourth_strategy_after_freeze"]
        is True
    )

    rerun = execute_stage_p_proposal(
        proposal_dir=proposal_dir,
        governance_dir=governance_dir,
        output_dir=output_dir,
        source_root=Path(__file__).parents[1] / "src",
        _numerical_runner=lambda *_args: pytest.fail("completed Stage P must not rerun a solver"),
    )
    assert rerun == completion
    assert len(calls) == 192


def _stage_p_selected_artifacts(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    inputs = _stage_p_inputs(tmp_path)
    proposal = build_stage_p_proposal(**inputs)
    current_rows = deepcopy(inputs["stage_f_profile_matrix"]["rows"])
    current_role_rows = deepcopy(inputs["stage_f_current_role_matrix"]["rows"])
    new_rows = []
    for identity in proposal["identities"]:
        new_rows.append(
            {
                **identity,
                "metrics": {
                    "longest_e10_run_windows": 4,
                    "longest_e20_run_windows": 1,
                    "final_motion_mae_bpm": (
                        2.5
                        if identity["penalty_candidate_id"] == "trusted_history_corridor_v1"
                        else 3.0
                    ),
                    "right_censored_recovery_count": 0,
                    "max_recovered_delay_s": 2.0,
                    "max_rise_underestimate_bpm": (
                        1.0 if identity["true_rise_applicable"] else None
                    ),
                },
                "spectral_audit": next(
                    row["spectral_audit"]
                    for row in current_rows
                    if row["filter_profile_id"] == identity["filter_profile_id"]
                    and row["record_id"] == identity["record_id"]
                ),
            }
        )
    report = build_penalty_interaction_report(
        proposal=proposal,
        current_rows=current_rows,
        current_role_rows=current_role_rows,
        new_rows=new_rows,
    )
    completion = _with_hash(
        {
            "completion_version": "lyx_stage_p_completion_v1",
            "status": "selected",
            "proposal_sha256": proposal["proposal_sha256"],
            "selected_penalty_id": report["selected_penalty_id"],
        },
        "completion_sha256",
    )
    return inputs, proposal, report, completion


def _rollback_proposal_inputs(
    tmp_path: Path,
    *,
    report: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    inputs, stage_p_proposal, default_report, completion = _stage_p_selected_artifacts(tmp_path)
    interaction_report = default_report if report is None else report
    if report is not None:
        completion = _with_hash(
            {
                "completion_version": "lyx_stage_p_completion_v1",
                "status": "selected",
                "proposal_sha256": stage_p_proposal["proposal_sha256"],
                "selected_penalty_id": interaction_report["selected_penalty_id"],
            },
            "completion_sha256",
        )
    proposal_inputs = {
        "stage_f_proposal": inputs["stage_f_proposal"],
        "stage_p_proposal": stage_p_proposal,
        "stage_p_completion": completion,
        "penalty_interaction_report": interaction_report,
        "recovery_registry": _stage_f_inputs(tmp_path)["recovery_registry"],
        "budget_contract": inputs["budget_contract"],
        "parent_experiment_id": inputs["parent_experiment_id"],
        "solver_hash": inputs["solver_hash"],
        "metric_contract_hash": inputs["metric_contract_hash"],
        "evaluation_hash": "9" * 64,
    }
    return proposal_inputs, inputs


def test_rollback_proposal_freezes_only_the_fixed_backup_under_final_penalty(
    tmp_path: Path,
) -> None:
    proposal_inputs, _ = _rollback_proposal_inputs(tmp_path)

    proposal = build_rollback_backup_proposal(**proposal_inputs)

    assert proposal["planned_unique_identity_count"] == 96
    assert proposal["rollback_limit"] == 1
    assert proposal["candidate_reselection_allowed"] is False
    assert proposal["penalty_reselection_allowed"] is False
    assert {identity["recovery_candidate_id"] for identity in proposal["identities"]} == {
        "current_fixed_floor_control_v1"
    }
    assert {identity["penalty_candidate_id"] for identity in proposal["identities"]} == {
        "trusted_history_corridor_v1"
    }
    assert all(identity["stage"] == "rollback_backup_matrix" for identity in proposal["identities"])


def test_rollback_proposal_rejects_tampered_recovery_registry_outer_hash(
    tmp_path: Path,
) -> None:
    proposal_inputs, _ = _rollback_proposal_inputs(tmp_path)
    registry = deepcopy(proposal_inputs["recovery_registry"])
    candidate = registry["candidates"][0]
    candidate.pop("candidate_sha256")
    candidate["constants"]["candidate_min_bpm"] = 99.0
    candidate["candidate_sha256"] = canonical_sha256(candidate)
    proposal_inputs["recovery_registry"] = registry

    with pytest.raises(
        StagePPlanError,
        match="rollback_recovery_registry_hash_mismatch",
    ):
        build_rollback_backup_proposal(**proposal_inputs)


def test_rollback_proposal_rejects_rehashed_registry_after_stage_f_freeze(
    tmp_path: Path,
) -> None:
    proposal_inputs, _ = _rollback_proposal_inputs(tmp_path)
    registry = deepcopy(proposal_inputs["recovery_registry"])
    registry.pop("registry_sha256")
    candidate = registry["candidates"][0]
    candidate.pop("candidate_sha256")
    candidate["constants"]["candidate_min_bpm"] = 99.0
    candidate["candidate_sha256"] = canonical_sha256(candidate)
    registry["registry_sha256"] = canonical_sha256(registry)
    proposal_inputs["recovery_registry"] = registry

    with pytest.raises(
        StagePPlanError,
        match="rollback_upstream_contract_mismatch",
    ):
        build_rollback_backup_proposal(**proposal_inputs)


def test_rollback_proposal_rejects_rehashed_budget_after_stage_f_freeze(
    tmp_path: Path,
) -> None:
    proposal_inputs, _ = _rollback_proposal_inputs(tmp_path)
    budget = deepcopy(proposal_inputs["budget_contract"])
    budget["max_attempts"] += 1
    proposal_inputs["budget_contract"] = budget

    with pytest.raises(
        StagePPlanError,
        match="rollback_upstream_contract_mismatch",
    ):
        build_rollback_backup_proposal(**proposal_inputs)


def test_rollback_is_mechanical_and_happens_at_most_once(
    tmp_path: Path,
) -> None:
    proposal_inputs, inputs = _rollback_proposal_inputs(tmp_path)
    report = deepcopy(proposal_inputs["penalty_interaction_report"])
    provisional_rows = [
        row
        for row in report["rows"]
        if row["penalty_candidate_id"] == report["selected_penalty_id"]
    ]
    target_records = {"jianpan1", "run1"}
    touched: set[str] = set()
    for row in provisional_rows:
        if row["record_id"] in target_records and row["record_id"] not in touched:
            row["qualification"] = {
                **row["qualification"],
                "qualified": False,
                "elimination_reasons": ["independent_l10_gate"],
            }
            row["metrics"] = {
                **row["metrics"],
                "longest_e10_run_windows": 12,
            }
            touched.add(row["record_id"])
    report.pop("report_sha256")
    report["report_sha256"] = canonical_sha256(report)
    proposal_inputs["penalty_interaction_report"] = report
    proposal = build_rollback_backup_proposal(**proposal_inputs)
    backup_rows = []
    for identity in proposal["identities"]:
        backup_rows.append(
            {
                **identity,
                "metrics": {
                    "longest_e10_run_windows": 4,
                    "longest_e20_run_windows": 1,
                    "final_motion_mae_bpm": 3.0,
                    "right_censored_recovery_count": 0,
                    "max_recovered_delay_s": 2.0,
                    "max_rise_underestimate_bpm": (
                        1.0 if identity["true_rise_applicable"] else None
                    ),
                },
                "spectral_audit": next(
                    row["spectral_audit"]
                    for row in provisional_rows
                    if row["filter_profile_id"] == identity["filter_profile_id"]
                    and row["record_id"] == identity["record_id"]
                ),
            }
        )

    receipt = resolve_recovery_interaction(
        proposal=proposal,
        provisional_rows=provisional_rows,
        backup_rows=backup_rows,
        current_role_rows=inputs["stage_f_current_role_matrix"]["rows"],
    )

    assert receipt["status"] == "rolled_back"
    assert receipt["rollback_triggered"] is True
    assert receipt["rollback_count"] == 1
    assert receipt["final_recovery_id"] == "current_fixed_floor_control_v1"
    assert receipt["candidate_reselection_count"] == 0
    assert receipt["penalty_reselection_count"] == 0
    assert receipt["selected_penalty_id"] == "trusted_history_corridor_v1"
    final_audit = build_final_interaction_audit(
        proposal=proposal,
        rollback_receipt=receipt,
        penalty_interaction_report=report,
        backup_rows=backup_rows,
        current_role_rows=inputs["stage_f_current_role_matrix"]["rows"],
    )
    assert final_audit["row_count"] == 96
    assert final_audit["rollback_count"] == 1
    assert len(final_audit["profile_receipts"]) == 8
    assert final_audit["sample_in_upper_bound"]["record_count"] == 12
    assert (
        final_audit["sample_in_upper_bound"]["definition"]
        == "raw_coverage_best_across_all_frozen_profiles"
    )
    assert (
        final_audit["safe_qualified_upper_bound"]["definition"]
        == "best_across_engineering_qualified_profiles_only"
    )
    assert all(
        row["recovery_candidate_id"] == "current_fixed_floor_control_v1"
        for row in final_audit["rows"]
    )


def test_rollback_missing_or_unavailable_backup_waits_for_human(
    tmp_path: Path,
) -> None:
    proposal_inputs, inputs = _rollback_proposal_inputs(tmp_path)
    report = proposal_inputs["penalty_interaction_report"]
    proposal = build_rollback_backup_proposal(**proposal_inputs)
    provisional_rows = [
        row
        for row in report["rows"]
        if row["penalty_candidate_id"] == report["selected_penalty_id"]
    ]
    backup_rows = [
        {
            **identity,
            "metrics": {
                "longest_e10_run_windows": 4,
                "longest_e20_run_windows": 1,
                "final_motion_mae_bpm": 3.0,
                "right_censored_recovery_count": 0,
                "max_recovered_delay_s": 2.0,
                "max_rise_underestimate_bpm": (1.0 if identity["true_rise_applicable"] else None),
            },
            "spectral_audit": next(
                row["spectral_audit"]
                for row in provisional_rows
                if row["filter_profile_id"] == identity["filter_profile_id"]
                and row["record_id"] == identity["record_id"]
            ),
        }
        for identity in proposal["identities"]
    ]

    missing_receipt = resolve_recovery_interaction(
        proposal=proposal,
        provisional_rows=provisional_rows,
        backup_rows=backup_rows[:-1],
        current_role_rows=inputs["stage_f_current_role_matrix"]["rows"],
    )

    assert missing_receipt["status"] == "awaiting_human_interaction_decision"
    assert missing_receipt["final_recovery_id"] is None
    assert missing_receipt["rollback_count"] == 0
    assert (
        missing_receipt["trigger_evidence"]["evidence_failure"]["reason"]
        == "rollback_backup_pairing_identity_mismatch"
    )

    no_backup_proposal = deepcopy(proposal)
    no_backup_proposal.pop("proposal_sha256")
    no_backup_proposal["rollback_backup_id"] = None
    no_backup_proposal["proposal_sha256"] = canonical_sha256(no_backup_proposal)
    unavailable_receipt = resolve_recovery_interaction(
        proposal=no_backup_proposal,
        provisional_rows=provisional_rows,
        backup_rows=[],
        current_role_rows=inputs["stage_f_current_role_matrix"]["rows"],
    )

    assert unavailable_receipt["status"] == "awaiting_human_interaction_decision"
    assert (
        unavailable_receipt["trigger_evidence"]["evidence_failure"]["reason"]
        == "rollback_backup_unavailable"
    )


def test_rollback_execution_publishes_waiting_receipt_without_final_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proposal_inputs, inputs = _rollback_proposal_inputs(tmp_path)
    proposal = build_rollback_backup_proposal(**proposal_inputs)
    proposal.pop("proposal_sha256")
    proposal["status"] = "no_backup_execution_required"
    proposal["rollback_backup_id"] = None
    proposal["planned_unique_identity_count"] = 0
    proposal["identities"] = []
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    budget = BudgetContract.approved_v5()
    monkeypatch.setattr(
        interaction_execution,
        "_verify_preflight",
        lambda **_kwargs: (
            proposal,
            budget,
            proposal_inputs["penalty_interaction_report"],
            inputs["stage_f_current_role_matrix"],
        ),
    )
    monkeypatch.setattr(
        interaction_execution,
        "build_final_interaction_audit",
        lambda **_kwargs: pytest.fail("human-wait path must not build a final interaction audit"),
    )
    governance_dir = tmp_path / "rollback-wait-governance"
    exploration = ExplorationRegistry.zero_budget_v1()
    atomic_write_json(
        governance_dir / "budget_contract.json",
        budget.to_dict(),
    )
    atomic_write_json(
        governance_dir / "exploration_registry.json",
        exploration.to_dict(),
    )
    AttemptRegistry.create(
        governance_dir / "attempt_registry.json",
        budget_contract=budget,
        exploration_registry=exploration,
    )
    output_dir = tmp_path / "rollback-wait-output"

    completion = execute_rollback_backup_proposal(
        proposal_dir=tmp_path / "unused-rollback-proposal",
        governance_dir=governance_dir,
        output_dir=output_dir,
        source_root=Path(__file__).parents[1] / "src",
    )

    assert completion["status"] == "awaiting_human_interaction_decision"
    assert completion["next_state"] == "awaiting_human_interaction_decision"
    assert completion["final_recovery_id"] is None
    assert completion["formal_result_count"] == 0
    assert not (output_dir / "final_interaction_audit.json").exists()
    assert (
        read_json(output_dir / "recovery_rollback_receipt.json")["status"]
        == "awaiting_human_interaction_decision"
    )


def test_rollback_publication_is_zero_run_and_source_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proposal_inputs, inputs = _rollback_proposal_inputs(tmp_path)
    source_payloads = {
        "stage_f_proposal": proposal_inputs["stage_f_proposal"],
        "stage_p_proposal": proposal_inputs["stage_p_proposal"],
        "stage_p_completion": proposal_inputs["stage_p_completion"],
        "penalty_interaction_report": proposal_inputs["penalty_interaction_report"],
        "stage_f_current_role_matrix": inputs["stage_f_current_role_matrix"],
        "recovery_registry": proposal_inputs["recovery_registry"],
        "budget_contract": proposal_inputs["budget_contract"],
    }
    source_paths = {}
    for name, payload in source_payloads.items():
        path = tmp_path / f"rollback-{name}.json"
        atomic_write_json(path, payload)
        source_paths[f"{name}_path"] = path

    def fake_runtime_identity(
        _source_root: Path,
        *,
        root_modules=None,
    ) -> dict[str, object]:
        bundle = "9" * 64 if root_modules else "1" * 64
        return {
            "source_files": {
                "ppg_hr/v2/recovery_interaction_resolution.py": bundle,
            },
            "source_bundle_sha256": bundle,
        }

    monkeypatch.setattr(
        interaction_execution,
        "runtime_source_identity",
        fake_runtime_identity,
    )
    monkeypatch.setattr(
        interaction_execution,
        "stage_r_metric_contract_v1",
        lambda: {
            "contract_version": "test_metric",
            "contract_sha256": proposal_inputs["metric_contract_hash"],
        },
    )
    proposal_dir = tmp_path / "rollback-proposal"

    receipt = propose_rollback_backup_execution(
        **source_paths,
        output_dir=proposal_dir,
        source_root=Path(__file__).parents[1] / "src",
        parent_experiment_id=str(proposal_inputs["parent_experiment_id"]),
    )

    assert receipt["formal_solver_run_count"] == 0
    assert receipt["planned_unique_identity_count"] == 96
    assert receipt["independent_bo_run_count"] == 0
    assert (
        read_json(proposal_dir / "rollback_backup_proposal.json")["penalty_reselection_allowed"]
        is False
    )


def _historical_parameter_manifest(
    stage_f_proposal: dict[str, object],
) -> dict[str, object]:
    records = []
    for record in stage_f_proposal["record_panel"]:
        records.append(
            {
                "record_id": record["record_id"],
                "scene": record["scene"],
                "parameter_source_sha256": canonical_sha256({"record_id": record["record_id"]}),
                "parameters": {
                    "analysis_scope": "full",
                    "fs_target": 50,
                    "lms_mu_base": 0.006,
                    "lms_mu_min": 1e-6,
                    "max_order": 4,
                    "smooth_win_len": 5,
                    "spec_penalty_width": 0.05,
                    "time_bias": 5.0,
                },
            }
        )
    return _with_hash(
        {
            "manifest_version": ("lyx_historical_parameter_replay_manifest_v1"),
            "record_count": 12,
            "records": records,
        },
        "manifest_sha256",
    )


def _retained_rollback_receipt(
    stage_p_proposal: dict[str, object],
) -> dict[str, object]:
    return _with_hash(
        {
            "receipt_version": "lyx_recovery_rollback_receipt_v1",
            "status": "retained",
            "proposal_sha256": "a" * 64,
            "provisional_recovery_id": ("relative_gap_timeout_v1"),
            "rollback_backup_id": ("current_fixed_floor_control_v1"),
            "selected_penalty_id": ("trusted_history_corridor_v1"),
            "rollback_triggered": False,
            "rollback_count": 0,
            "final_recovery_id": "relative_gap_timeout_v1",
            "candidate_reselection_count": 0,
            "penalty_reselection_count": 0,
            "trigger_evidence": {
                "rule_1_coordinates": [],
                "rule_2_coordinates": [],
                "rule_3_coordinates": [],
            },
            "independent_bo_run_count": 0,
            "next_state": ("ready_for_historical_recovery_ab_proposal"),
            "stage_p_proposal_sha256": stage_p_proposal["proposal_sha256"],
        },
        "receipt_sha256",
    )


def test_historical_ab_freezes_same_parameters_and_only_recovery_differs(
    tmp_path: Path,
) -> None:
    inputs, stage_p_proposal, _, _ = _stage_p_selected_artifacts(tmp_path)
    recovery_registry = _stage_f_inputs(tmp_path)["recovery_registry"]
    rollback_receipt = _retained_rollback_receipt(stage_p_proposal)

    proposal = build_historical_recovery_ab_proposal(
        stage_f_proposal=inputs["stage_f_proposal"],
        stage_p_proposal=stage_p_proposal,
        rollback_receipt=rollback_receipt,
        historical_parameter_manifest=_historical_parameter_manifest(inputs["stage_f_proposal"]),
        recovery_registry=recovery_registry,
        budget_contract=inputs["budget_contract"],
        parent_experiment_id=str(inputs["parent_experiment_id"]),
        solver_hash=str(inputs["solver_hash"]),
        metric_contract_hash=str(inputs["metric_contract_hash"]),
        evaluation_hash="b" * 64,
    )

    assert proposal["logical_task_count"] == 24
    assert proposal["planned_unique_identity_count"] == 24
    assert proposal["reused_logical_task_count"] == 0
    assert {task["arm"] for task in proposal["logical_tasks"]} == {
        "current_recovery",
        "final_recovery",
    }
    for record_id in {task["record_id"] for task in proposal["logical_tasks"]}:
        pair = [
            identity for identity in proposal["identities"] if identity["record_id"] == record_id
        ]
        assert len(pair) == 2
        left = deepcopy(pair[0]["config"]["parameters"])
        right = deepcopy(pair[1]["config"]["parameters"])
        left.pop("recovery_candidate_id")
        right.pop("recovery_candidate_id")
        assert left == right

    tampered_budget = deepcopy(inputs["budget_contract"])
    tampered_budget["max_attempts"] += 1
    with pytest.raises(
        StagePPlanError,
        match="historical_ab_upstream_contract_mismatch",
    ):
        build_historical_recovery_ab_proposal(
            stage_f_proposal=inputs["stage_f_proposal"],
            stage_p_proposal=stage_p_proposal,
            rollback_receipt=rollback_receipt,
            historical_parameter_manifest=_historical_parameter_manifest(
                inputs["stage_f_proposal"]
            ),
            recovery_registry=recovery_registry,
            budget_contract=tampered_budget,
            parent_experiment_id=str(inputs["parent_experiment_id"]),
            solver_hash=str(inputs["solver_hash"]),
            metric_contract_hash=str(inputs["metric_contract_hash"]),
            evaluation_hash="b" * 64,
        )


def test_historical_ab_rejects_tampered_recovery_registry(
    tmp_path: Path,
) -> None:
    inputs, stage_p_proposal, _, _ = _stage_p_selected_artifacts(tmp_path)
    recovery_registry = deepcopy(_stage_f_inputs(tmp_path)["recovery_registry"])
    candidate = recovery_registry["candidates"][0]
    candidate.pop("candidate_sha256")
    candidate["constants"]["candidate_min_bpm"] = 99.0
    candidate["candidate_sha256"] = canonical_sha256(candidate)

    with pytest.raises(
        StagePPlanError,
        match="historical_recovery_registry_hash_mismatch",
    ):
        build_historical_recovery_ab_proposal(
            stage_f_proposal=inputs["stage_f_proposal"],
            stage_p_proposal=stage_p_proposal,
            rollback_receipt=_retained_rollback_receipt(stage_p_proposal),
            historical_parameter_manifest=_historical_parameter_manifest(
                inputs["stage_f_proposal"]
            ),
            recovery_registry=recovery_registry,
            budget_contract=inputs["budget_contract"],
            parent_experiment_id=str(inputs["parent_experiment_id"]),
            solver_hash=str(inputs["solver_hash"]),
            metric_contract_hash=str(inputs["metric_contract_hash"]),
            evaluation_hash="b" * 64,
        )


def test_historical_ab_publication_is_zero_run_and_source_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs, stage_p_proposal, _, _ = _stage_p_selected_artifacts(tmp_path)
    source_payloads = {
        "stage_f_proposal": inputs["stage_f_proposal"],
        "stage_p_proposal": stage_p_proposal,
        "rollback_receipt": _retained_rollback_receipt(stage_p_proposal),
        "historical_parameter_manifest": (
            _historical_parameter_manifest(inputs["stage_f_proposal"])
        ),
        "recovery_registry": _stage_f_inputs(tmp_path)["recovery_registry"],
        "budget_contract": inputs["budget_contract"],
    }
    source_paths = {}
    for name, payload in source_payloads.items():
        path = tmp_path / f"historical-{name}.json"
        atomic_write_json(path, payload)
        source_paths[f"{name}_path"] = path

    def fake_runtime_identity(
        _source_root: Path,
        *,
        root_modules=None,
    ) -> dict[str, object]:
        bundle = "b" * 64 if root_modules else "1" * 64
        return {
            "source_files": {
                "ppg_hr/v2/recovery_pre_fold_gate.py": bundle,
            },
            "source_bundle_sha256": bundle,
        }

    monkeypatch.setattr(
        pre_fold_execution,
        "runtime_source_identity",
        fake_runtime_identity,
    )
    monkeypatch.setattr(
        pre_fold_execution,
        "stage_r_metric_contract_v1",
        lambda: {
            "contract_version": "test_metric",
            "contract_sha256": inputs["metric_contract_hash"],
        },
    )
    proposal_dir = tmp_path / "historical-proposal"

    receipt = propose_historical_recovery_ab_execution(
        **source_paths,
        output_dir=proposal_dir,
        source_root=Path(__file__).parents[1] / "src",
        parent_experiment_id=str(inputs["parent_experiment_id"]),
    )

    assert receipt["formal_solver_run_count"] == 0
    assert receipt["planned_unique_identity_count"] == 24
    assert receipt["logical_task_count"] == 24
    assert receipt["independent_bo_run_count"] == 0


def test_pre_fold_gate_never_auto_runs_independent_bo(
    tmp_path: Path,
) -> None:
    inputs, stage_p_proposal, report, _ = _stage_p_selected_artifacts(tmp_path)
    rollback_receipt = _retained_rollback_receipt(stage_p_proposal)
    historical_proposal = build_historical_recovery_ab_proposal(
        stage_f_proposal=inputs["stage_f_proposal"],
        stage_p_proposal=stage_p_proposal,
        rollback_receipt=rollback_receipt,
        historical_parameter_manifest=_historical_parameter_manifest(inputs["stage_f_proposal"]),
        recovery_registry=_stage_f_inputs(tmp_path)["recovery_registry"],
        budget_contract=inputs["budget_contract"],
        parent_experiment_id=str(inputs["parent_experiment_id"]),
        solver_hash=str(inputs["solver_hash"]),
        metric_contract_hash=str(inputs["metric_contract_hash"]),
        evaluation_hash="b" * 64,
    )
    numerical_rows = []
    for identity in historical_proposal["identities"]:
        is_final = identity["recovery_candidate_id"] == historical_proposal["final_recovery_id"]
        numerical_rows.append(
            {
                **identity,
                "metrics": {
                    "longest_e10_run_windows": 4,
                    "longest_e20_run_windows": 1,
                    "final_motion_mae_bpm": (
                        6.1 if is_final and identity["record_id"] == "run1" else 3.0
                    ),
                    "right_censored_recovery_count": 0,
                    "max_recovered_delay_s": 2.0,
                    "max_rise_underestimate_bpm": (
                        1.0 if identity["true_rise_applicable"] else None
                    ),
                },
                "spectral_audit": {
                    "stability_pass": True,
                    "spectral_gate_pass": True,
                },
            }
        )
    ab_report = build_historical_recovery_ab_report(
        proposal=historical_proposal,
        numerical_rows=numerical_rows,
    )
    final_rows = [
        row
        for row in report["rows"]
        if row["penalty_candidate_id"] == report["selected_penalty_id"]
    ]
    independent_by_record = {
        str(record["record_id"]): record["independent_metrics"]
        for record in historical_proposal["record_panel"]
    }
    review_context = {
        "planned_search_space_hash": "c" * 64,
        "planned_seed_manifest_hash": "d" * 64,
        "planned_unique_budget": 120,
        "estimated_runtime": "12-record wall-clock estimate",
        "estimated_cache_size": "bounded cache estimate",
        "recommendation": "run only after human approval",
        "run_answers": "solver-specific recovery of unsafe records",
        "no_run_answers": "retains current development conclusion",
    }

    gate = evaluate_pre_fold_independent_bo_gate(
        historical_ab_report=ab_report,
        final_profile_rows=final_rows,
        independent_metrics_by_record=independent_by_record,
        review_context=review_context,
    )

    assert gate["triggered"] is True
    assert gate["status"] == "awaiting_human_independent_bo_decision"
    assert gate["independent_bo_run_count"] == 0
    assert gate["independent_bo_authorized"] is False
    assert gate["review_packet"] == review_context

    gap_records = {"jianpan1", "run1", "xiezi1"}
    gap_rows = deepcopy(final_rows)
    for row in gap_rows:
        if row["record_id"] not in gap_records:
            continue
        row["qualification"] = {
            **row["qualification"],
            "qualified": False,
            "elimination_reasons": ["independent_mae_gate"],
        }
        row["metrics"] = {
            **row["metrics"],
            "final_motion_mae_bpm": (
                float(independent_by_record[row["record_id"]]["final_motion_mae_bpm"]) + 2.5
            ),
        }
    gap_gate = evaluate_pre_fold_independent_bo_gate(
        historical_ab_report=ab_report,
        final_profile_rows=gap_rows,
        independent_metrics_by_record=independent_by_record,
        review_context=review_context,
    )

    assert gap_gate["conditions"]["sample_in_upper_bound_gap"]["triggered"] is True
    assert set(gap_gate["conditions"]["sample_in_upper_bound_gap"]["record_ids"]) == gap_records
    assert all(
        record["selected_qualified"] is False
        for record in gap_gate["conditions"]["sample_in_upper_bound_gap"]["records"]
    )
    assert all(record["reason_categories"] for record in gap_gate["trigger_records"])

    historical_path = tmp_path / "historical-ab-report.json"
    final_audit_path = tmp_path / "final-interaction-audit.json"
    review_context_path = tmp_path / "review-context.json"
    atomic_write_json(historical_path, ab_report)
    atomic_write_json(
        final_audit_path,
        _with_hash(
            {
                "audit_version": "lyx_final_interaction_audit_v1",
                "status": "complete",
                "row_count": 96,
                "rows": final_rows,
                "independent_metrics_by_record": (independent_by_record),
                **build_sample_in_upper_bound_payloads(
                    final_profile_rows=final_rows,
                    scene_by_record={
                        str(record["record_id"]): str(record["scene"])
                        for record in ab_report["records"]
                    },
                ),
            },
            "audit_sha256",
        ),
    )
    atomic_write_json(review_context_path, review_context)

    published = publish_pre_fold_independent_bo_gate(
        historical_ab_report_path=historical_path,
        final_interaction_audit_path=final_audit_path,
        review_context_path=review_context_path,
        output_dir=tmp_path / "pre-fold-gate",
    )

    assert published["status"] == gate["status"]
    assert published["independent_bo_run_count"] == 0
    assert (
        read_json(tmp_path / "pre-fold-gate" / "independent_bo_review_packet.json")[
            "independent_bo_authorized"
        ]
        is False
    )
    review_packet = read_json(tmp_path / "pre-fold-gate" / "independent_bo_review_packet.json")
    assert review_packet["trigger_records"]
    assert all(
        record["historical_parameter_recovery_ab"] for record in review_packet["trigger_records"]
    )
    assert set(review_packet["source_artifacts"]) == {
        "historical_ab_report",
        "final_interaction_audit",
        "review_context",
    }


def test_stage_r_no_safe_publishes_immediate_human_gate(
    tmp_path: Path,
) -> None:
    authorization_sha256 = "e" * 64
    proposal = _with_hash(
        {
            "proposal_version": "lyx_stage_r_execution_proposal_v1",
            "status": "awaiting_human_execution_authorization",
            "independent_bo_authorized": False,
        },
        "proposal_sha256",
    )
    review_package = _with_hash(
        {
            "package_version": "lyx_stage_r_independent_bo_review_v1",
            "status": "awaiting_human_independent_bo_decision",
            "proposal_sha256": proposal["proposal_sha256"],
            "authorization_sha256": authorization_sha256,
            "trigger": "no_safe_recovery_candidate",
            "requested_human_decision": (
                "whether_to_prepare_a_separate_exact_independent_bo_proposal"
            ),
            "independent_bo_authorized": False,
            "independent_bo_run_count": 0,
            "execution_identity_count": 150,
        },
        "package_sha256",
    )
    proposal_path = tmp_path / "stage-r-proposal.json"
    completion_path = tmp_path / "stage-r-completion.json"
    review_package_path = tmp_path / "stage-r-independent-bo-review.json"
    atomic_write_json(proposal_path, proposal)
    atomic_write_json(review_package_path, review_package)
    completion = _with_hash(
        {
            "completion_version": "lyx_stage_r_completion_v2",
            "status": "no_safe_recovery_candidate",
            "proposal_sha256": proposal["proposal_sha256"],
            "authorization_sha256": authorization_sha256,
            "independent_bo_run_count": 0,
            "next_state": "awaiting_human_independent_bo_decision",
            "artifacts": {
                "independent_bo_review_package.json": file_sha256(review_package_path),
            },
        },
        "completion_sha256",
    )
    atomic_write_json(completion_path, completion)

    receipt = publish_stage_r_no_safe_independent_bo_gate(
        stage_r_proposal_path=proposal_path,
        stage_r_completion_path=completion_path,
        stage_r_review_package_path=review_package_path,
        output_dir=tmp_path / "stage-r-no-safe-gate",
    )

    assert receipt["status"] == "awaiting_human_independent_bo_decision"
    assert receipt["trigger_stage"] == "stage_r"
    assert receipt["stage_f_allowed"] is False
    assert receipt["independent_bo_run_count"] == 0
    assert receipt["independent_bo_authorized"] is False
    assert receipt["conditions"]["no_safe_recovery_candidate"]["triggered"] is True
