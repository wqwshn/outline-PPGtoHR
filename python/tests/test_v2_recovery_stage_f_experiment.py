from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2 import recovery_stage_f_execution as stage_f_execution
from ppg_hr.v2 import recovery_stage_f_experiment as stage_f_module
from ppg_hr.v2 import recovery_stage_f_plan as stage_f_plan
from ppg_hr.v2 import recovery_stage_f_reporting as stage_f_reporting
from ppg_hr.v2 import recovery_stage_f_runner as stage_f_runner
from ppg_hr.v2.phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    read_json,
)
from ppg_hr.v2.recovery_contracts import canonical_sha256
from ppg_hr.v2.recovery_experiment_governance import (
    AttemptIdentity,
    AttemptRegistry,
    BudgetContract,
    ExplorationRegistry,
)
from ppg_hr.v2.recovery_profile_metrics import (
    RecoveryProfileMetricResult,
)
from ppg_hr.v2.recovery_stage_f_experiment import (
    StageFPlanError,
    build_stage_f_proposal,
    execute_stage_f_proposal,
    propose_stage_f_execution,
)
from ppg_hr.v2.recovery_stage_r_common import StageRNumericalResult
from ppg_hr.v2.solver import V2SolverResult


def _with_hash(
    payload: dict[str, object],
    hash_field: str,
) -> dict[str, object]:
    result = dict(payload)
    result[hash_field] = canonical_sha256(payload)
    return result


def _stage_f_inputs(tmp_path: Path) -> dict[str, object]:
    solver_hash = "1" * 64
    metric_hash = canonical_sha256(
        {"contract_version": "test_metric"}
    )
    spectral_hash = canonical_sha256(
        {"contract_version": "test_spectral"}
    )
    evaluation_hash = "4" * 64
    parent_experiment_id = "lyx_recovery_filter_profile_v1"
    recovery_candidates = [
        _with_hash(
            {
                "candidate_id": "current_fixed_floor_control_v1",
                "mechanism_complexity": 0,
                "constants": {"candidate_min_bpm": 85.0},
            },
            "candidate_sha256",
        ),
        _with_hash(
            {
                "candidate_id": "relative_gap_timeout_v1",
                "mechanism_complexity": 1,
                "constants": {"candidate_min_bpm": None},
            },
            "candidate_sha256",
        ),
        _with_hash(
            {
                "candidate_id": "relative_gap_rise_guard_v1",
                "mechanism_complexity": 2,
                "constants": {"candidate_min_bpm": None},
            },
            "candidate_sha256",
        ),
    ]
    recovery_registry = _with_hash(
        {
            "candidate_count": 3,
            "control_candidate_id": "current_fixed_floor_control_v1",
            "candidates": recovery_candidates,
        },
        "registry_sha256",
    )
    penalty_candidates = [
        _with_hash(
            {
                "penalty_id": "current_soft_penalty_control_v1",
                "mechanism_complexity": 0,
            },
            "candidate_sha256",
        ),
        _with_hash(
            {
                "penalty_id": "bounded_soft_penalty_v1",
                "mechanism_complexity": 1,
            },
            "candidate_sha256",
        ),
        _with_hash(
            {
                "penalty_id": "persistence_aware_penalty_v1",
                "mechanism_complexity": 2,
            },
            "candidate_sha256",
        ),
    ]
    penalty_registry = _with_hash(
        {
            "candidate_count": 3,
            "control_penalty_id": "current_soft_penalty_control_v1",
            "candidates": penalty_candidates,
        },
        "registry_sha256",
    )
    profile_specs = (
        ("p25-short-low", "core", 25, 40, 0.008, None),
        ("p25-short-mid", "core", 25, 40, 0.012, None),
        ("p25-long-mid", "core", 25, 200, 0.010, None),
        (
            "p50-short-low-40",
            "core",
            50,
            40,
            0.006,
            "intermediate",
        ),
        ("p50-short-midlow-40", "core", 50, 40, 0.008, None),
        (
            "p50-short-low",
            "core",
            50,
            80,
            0.006,
            "conservative",
        ),
        (
            "p100-short-rate-normalized-low-40",
            "coverage_boundary",
            100,
            40,
            0.003,
            "aggressive",
        ),
        (
            "p100-short-rate-normalized-midlow-40",
            "coverage_boundary",
            100,
            40,
            0.004,
            None,
        ),
    )
    profiles: list[dict[str, object]] = []
    for (
        profile_id,
        role,
        fs_target,
        memory_ms,
        nominal_mu,
        sentinel_role,
    ) in profile_specs:
        identity = {
            "profile_id": profile_id,
            "design_role": role,
            "fs_target": fs_target,
            "memory_ms": memory_ms,
            "nominal_mu": nominal_mu,
            "recovery_sentinel_role": sentinel_role,
            "actual_taps": max(
                1,
                int(round(fs_target * memory_ms / 1000.0)),
            ),
        }
        profiles.append(
            {
                "profile_id": profile_id,
                "design_role": role,
                "fs_target": fs_target,
                "physical_memory_ms": memory_ms,
                "actual_taps": identity["actual_taps"],
                "nominal_mu": nominal_mu,
                "recovery_sentinel_role": sentinel_role,
                "profile_sha256": canonical_sha256(identity),
            }
        )
    profile_library = _with_hash(
        {
            "status": "complete",
            "profile_count": 8,
            "design_rule_sha256": "5" * 64,
            "fs_target_quota": {"25": 3, "50": 3, "100": 2},
            "role_counts": {"core": 6, "coverage_boundary": 2},
            "recovery_sentinels": {
                "conservative": "p50-short-low",
                "intermediate": "p50-short-low-40",
                "aggressive": (
                    "p100-short-rate-normalized-low-40"
                ),
            },
            "profiles": profiles,
        },
        "library_sha256",
    )
    templates: list[dict[str, object]] = []
    record_panel: list[dict[str, object]] = []
    baseline_records: list[dict[str, object]] = []
    control = recovery_candidates[0]
    anchor = next(
        profile
        for profile in profiles
        if profile["profile_id"] == "p50-short-low"
    )
    for scene in ("jianpan", "kaihe", "run", "xiezi"):
        for index in range(1, 4):
            record_id = f"{scene}{index}"
            data_path = tmp_path / f"{record_id}.csv"
            reference_path = tmp_path / f"{record_id}_ref.csv"
            data_path.write_text("time,ppg\n0,0\n", encoding="utf-8")
            reference_path.write_text(
                "time,hr\n0,60\n",
                encoding="utf-8",
            )
            raw_hash = file_sha256(data_path)
            reference_hash = file_sha256(reference_path)
            combined_hash = canonical_sha256(
                {
                    "data_sha256": raw_hash,
                    "reference_sha256": reference_hash,
                }
            )
            config = {
                "data_path": str(data_path),
                "reference_path": str(reference_path),
                "method_names": ["reset FFT", "LMS+H"],
                "parameters": {
                    "analysis_scope": "full",
                    "adaptive_filter": "lms",
                    "algorithm_preset": "lite",
                    "reference_groups_order": ["HF"],
                    "fs_target": 50,
                    "lms_mu_base": 0.006,
                    "lms_mu_min": 1e-6,
                    "max_order": 4,
                    "smooth_win_len": 5,
                    "spec_penalty_width": 0.2,
                    "time_bias": 5.0,
                    "penalty_candidate_id": (
                        "current_soft_penalty_control_v1"
                    ),
                    "recovery_candidate_id": (
                        "current_fixed_floor_control_v1"
                    ),
                },
            }
            attempt = AttemptIdentity(
                solver_hash=solver_hash,
                config_hash=canonical_sha256(config),
                metric_contract_hash=metric_hash,
                evaluation_hash="6" * 64,
                data_sha256=combined_hash,
                record_id=record_id,
                stage="recovery_sentinel",
                attempt_kind="formal",
                parent_experiment_id=parent_experiment_id,
            )
            templates.append(
                {
                    **attempt.to_dict(),
                    "scene": scene,
                    "data_path": str(data_path),
                    "reference_path": str(reference_path),
                    "raw_data_sha256": raw_hash,
                    "reference_sha256": reference_hash,
                    "method_names": ["reset FFT", "LMS+H"],
                    "true_rise_applicable": scene in {"kaihe", "run"},
                    "config": config,
                    "sentinel_role": "conservative",
                    "filter_profile_id": anchor["profile_id"],
                    "filter_profile_sha256": anchor["profile_sha256"],
                    "physical_memory_ms": anchor[
                        "physical_memory_ms"
                    ],
                    "recovery_candidate_id": control["candidate_id"],
                    "recovery_candidate_sha256": control[
                        "candidate_sha256"
                    ],
                    "candidate_min_bpm": 85.0,
                    "penalty_candidate_id": (
                        "current_soft_penalty_control_v1"
                    ),
                }
            )
            independent_metrics = {
                "metric_contract_version": (
                    "lyx_recovery_profile_metric_v1"
                ),
                "longest_e10_run_windows": 4,
                "longest_e20_run_windows": 1,
                "final_motion_mae_bpm": 3.0,
                "right_censored_recovery_count": 0,
                "max_recovered_delay_s": 2.0,
                "max_rise_underestimate_bpm": (
                    1.0 if scene in {"kaihe", "run"} else None
                ),
            }
            record_panel.append(
                {
                    "record_id": record_id,
                    "scene": scene,
                    "data_sha256": raw_hash,
                    "reference_sha256": reference_hash,
                    "combined_data_sha256": combined_hash,
                    "method_names": ["reset FFT", "LMS+H"],
                    "true_rise_applicable": scene in {"kaihe", "run"},
                    "independent_metrics": independent_metrics,
                }
            )
            baseline_records.append(
                {
                    "sample_id": record_id,
                    "scene": scene,
                    "data_sha256": raw_hash,
                    "reference_sha256": reference_hash,
                    "metrics": independent_metrics,
                }
            )
    budget = BudgetContract.approved_v5().to_dict()
    baseline_metrics = {"records": baseline_records}
    stage_r_source_payloads = {
        "baseline_metrics": baseline_metrics,
        "profile_library": profile_library,
        "recovery_registry": recovery_registry,
        "penalty_registry": penalty_registry,
        "budget_contract": budget,
    }
    stage_r_source_artifacts: dict[str, dict[str, str]] = {}
    for name, payload in stage_r_source_payloads.items():
        source_path = tmp_path / f"stage_r_source_{name}.json"
        atomic_write_json(source_path, payload)
        stage_r_source_artifacts[name] = {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
        }
    baseline_contract_receipt = {
        "receipt_version": (
            "lyx_recovery_profile_baseline_receipt_v1"
        ),
        "status": "complete",
        "metric_contract_version": (
            "lyx_recovery_profile_metric_v1"
        ),
        "record_count": 12,
        "scene_counts": {
            "jianpan": 3,
            "kaihe": 3,
            "run": 3,
            "xiezi": 3,
        },
        "artifact_sha256": {
            "record_metrics.json": stage_r_source_artifacts[
                "baseline_metrics"
            ]["sha256"],
        },
    }
    profile_library_completion = _with_hash(
        {
            "receipt_version": (
                "lyx_filter_rate_normalized_supplement_completion_v2"
            ),
            "status": "complete",
            "evidence_class": "development_reuse_pilot",
            "algorithm_level_holdout": False,
            "final_profile_count": 8,
            "final_profile_ids": [
                str(profile["profile_id"])
                for profile in profiles
            ],
            "final_library_sha256": profile_library[
                "library_sha256"
            ],
            "new_rate_normalized_run_count": 8,
            "exploration_run_count": 8,
            "reused_p50_numeric_result_count": 8,
            "candidate_profile_count": 2,
            "candidate_eligible_profile_ids": [
                "p100-short-rate-normalized-low-40",
                "p100-short-rate-normalized-midlow-40",
            ],
            "candidate_profile_receipt_sha256": {
                "p100-short-rate-normalized-low-40": "a" * 64,
                "p100-short-rate-normalized-midlow-40": "b" * 64,
            },
            "final_profile_receipt_sha256": {
                str(profile["profile_id"]): (
                    f"{index + 1:x}" * 64
                )[:64]
                for index, profile in enumerate(profiles)
            },
            "actual_hr_tracking_trajectory_count": 0,
            "independent_bo_run_count": 0,
            "selection": {
                "status": "complete",
                "selected_p100_profile_ids": [
                    "p100-short-rate-normalized-low-40",
                    "p100-short-rate-normalized-midlow-40",
                ],
                "selected_p50_profile_ids": [
                    "p50-short-low-40",
                    "p50-short-midlow-40",
                ],
                "selected_profile_ids": [
                    "p50-short-low-40",
                    "p50-short-midlow-40",
                    "p100-short-rate-normalized-low-40",
                    "p100-short-rate-normalized-midlow-40",
                ],
            },
        },
        "completion_sha256",
    )
    stage_r_proposal = {
        "proposal_version": "lyx_stage_r_execution_proposal_v1",
        "status": "awaiting_human_execution_authorization",
        "parent_experiment_id": parent_experiment_id,
        "record_panel": record_panel,
        "identities": templates,
        "recovery_candidates": recovery_candidates,
        "frozen_contracts": {
            "metric_contract_hash": metric_hash,
            "spectral_gate_contract_hash": spectral_hash,
            "recovery_candidate_registry_hash": recovery_registry[
                "registry_sha256"
            ],
            "recovery_selection_contract_hash": "7" * 64,
            "penalty_registry_hash": penalty_registry[
                "registry_sha256"
            ],
            "filter_profile_design_rule_hash": "5" * 64,
            "budget_contract_hash": canonical_sha256(budget),
        },
        "diagnostic_unique_budget": 60,
        "formal_unique_budget": 108,
        "unique_budget": 168,
        "independent_bo_authorized": False,
        "source_artifacts": stage_r_source_artifacts,
    }
    stage_r_proposal["proposal_sha256"] = canonical_sha256(
        stage_r_proposal
    )
    stage_r_completion = {
        "completion_version": "lyx_stage_r_completion_v2",
        "status": "selected",
        "proposal_sha256": stage_r_proposal["proposal_sha256"],
        "diagnostic_result_count": 60,
        "formal_result_count": 108,
        "independent_bo_run_count": 0,
        "provisional_recovery_id": "relative_gap_timeout_v1",
        "rollback_backup_id": "current_fixed_floor_control_v1",
        "next_state": "ready_for_stage_f_filter_matrix",
    }
    stage_r_completion["completion_sha256"] = canonical_sha256(
        stage_r_completion
    )
    return {
        "stage_r_proposal": stage_r_proposal,
        "stage_r_completion": stage_r_completion,
        "profile_library": profile_library,
        "baseline_metrics": baseline_metrics,
        "baseline_contract_receipt": baseline_contract_receipt,
        "profile_library_completion": profile_library_completion,
        "recovery_registry": recovery_registry,
        "penalty_registry": penalty_registry,
        "budget_contract": budget,
        "parent_experiment_id": parent_experiment_id,
        "solver_hash": solver_hash,
        "metric_contract_hash": metric_hash,
        "spectral_gate_contract_hash": spectral_hash,
        "evaluation_hash": evaluation_hash,
    }


_STAGE_F_SOURCE_NAMES = (
    "stage_r_proposal",
    "stage_r_completion",
    "profile_library",
    "profile_library_completion",
    "baseline_metrics",
    "baseline_contract_receipt",
    "recovery_registry",
    "penalty_registry",
    "budget_contract",
)


def _write_stage_f_source_files(
    tmp_path: Path,
    inputs: dict[str, object],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in _STAGE_F_SOURCE_NAMES:
        path = tmp_path / f"{name}.json"
        atomic_write_json(path, inputs[name])
        paths[f"{name}_path"] = path
    return paths


def _patch_stage_f_runtime(monkeypatch) -> None:
    def fake_runtime_identity(
        _source_root: Path,
        *,
        root_modules=None,
    ) -> dict[str, object]:
        bundle = "4" * 64 if root_modules else "1" * 64
        return {
            "source_files": {
                "ppg_hr/v2/recovery_stage_f_experiment.py": bundle
            },
            "source_bundle_sha256": bundle,
        }

    for module in (stage_f_plan, stage_f_execution):
        monkeypatch.setattr(
            module,
            "runtime_source_identity",
            fake_runtime_identity,
        )
    monkeypatch.setattr(
        stage_f_plan,
        "stage_r_metric_contract_v1",
        lambda: {
            "contract_version": "test_metric",
            "contract_sha256": canonical_sha256(
                {"contract_version": "test_metric"}
            ),
        },
    )
    monkeypatch.setattr(
        stage_f_plan,
        "stage_r_spectral_gate_contract_v1",
        lambda: {
            "contract_version": "test_spectral",
            "contract_sha256": canonical_sha256(
                {"contract_version": "test_spectral"}
            ),
        },
    )


def _publish_stage_f_proposal(
    tmp_path: Path,
    monkeypatch,
    *,
    inputs: dict[str, object] | None = None,
) -> Path:
    frozen_inputs = (
        _stage_f_inputs(tmp_path)
        if inputs is None
        else inputs
    )
    paths = _write_stage_f_source_files(tmp_path, frozen_inputs)
    _patch_stage_f_runtime(monkeypatch)
    proposal_dir = tmp_path / "stage_f_proposal"
    propose_stage_f_execution(
        **paths,
        output_dir=proposal_dir,
        source_root=Path(__file__).parents[1] / "src",
        parent_experiment_id=str(
            frozen_inputs["parent_experiment_id"]
        ),
    )
    return proposal_dir


def _synthetic_solver_result() -> V2SolverResult:
    return V2SolverResult(
        HR=np.asarray(
            [[0.0, 60.0, 60.0, 60.0, 1.0, 0.0]]
        ),
        err_stats={},
        metadata={"time_bias": 5.0, "smooth_win_len": 5},
        window_table=[],
    )


def _synthetic_recovery_metrics(
    *,
    true_rise: bool,
) -> RecoveryProfileMetricResult:
    return RecoveryProfileMetricResult(
        metric_contract_version="lyx_recovery_profile_metric_v1",
        base_metric_contract_version="lyx_bo_formal_metric_v1",
        time_bias_s=5.0,
        smooth_win_len=5,
        uses_offline_future_dependency=True,
        final_method="LMS+H",
        reset_fft_method="reset FFT",
        total_window_count=10,
        base_motion_window_count=10,
        base_motion_window_sha256="a" * 64,
        excluded_reference_window_count=0,
        excluded_unreliable_window_count=0,
        excluded_non_motion_window_count=0,
        final_motion_mae_bpm=3.0,
        reset_motion_mae_bpm=3.0,
        e10_window_count=4,
        e20_window_count=1,
        longest_e10_run_windows=4,
        longest_e20_run_windows=1,
        recovery_episode_count=1,
        right_censored_recovery_count=0,
        max_recovered_delay_s=2.0,
        recovery_episodes=(),
        physiological_rise_episode_count=1 if true_rise else 0,
        max_rise_underestimate_bpm=1.0 if true_rise else None,
        physiological_rise_episodes=(),
    )


def _synthetic_spectral_evidence(
    *,
    profile_id: str,
    record_id: str,
    include_audit_hash: bool,
) -> dict[str, object]:
    evidence: dict[str, object] = {
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
    }
    if include_audit_hash:
        evidence["audit_sha256"] = canonical_sha256(
            {
                "profile_id": profile_id,
                "record_id": record_id,
            }
        )
    return evidence


def test_stage_f_proposal_freezes_two_fair_eight_by_twelve_matrices(
    tmp_path: Path,
) -> None:
    proposal = build_stage_f_proposal(**_stage_f_inputs(tmp_path))

    assert proposal["status"] == "ready_for_execution"
    assert proposal["algorithm_level_holdout"] is False
    assert proposal["evidence_class"] == "development_reuse_pilot"
    assert proposal["independent_bo_authorized"] is False
    assert all(
        "independent_metrics" in record
        for record in proposal["record_panel"]
    )
    assert proposal["logical_task_count"] == 192
    assert proposal["planned_unique_identity_count"] == 192
    assert len(proposal["identities"]) == 192
    assert len(proposal["logical_tasks"]) == 192
    assert {
        (
            item["matrix_role"],
            item["recovery_candidate_id"],
            item["stage"],
        )
        for item in proposal["identities"]
    } == {
        (
            "provisional_recovery",
            "relative_gap_timeout_v1",
            "penalty_interaction",
        ),
        (
            "same_role_current_control",
            "current_fixed_floor_control_v1",
            "current_role_matrix",
        ),
    }
    for matrix_role in (
        "provisional_recovery",
        "same_role_current_control",
    ):
        lane = [
            item
            for item in proposal["logical_tasks"]
            if item["matrix_role"] == matrix_role
        ]
        assert len(lane) == 96
        assert len({item["record_id"] for item in lane}) == 12
        assert len({item["filter_profile_id"] for item in lane}) == 8
    assert proposal["proposal_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in proposal.items()
            if key != "proposal_sha256"
        }
    )


def test_stage_f_proposal_reuses_control_matrix_when_it_is_provisional(
    tmp_path: Path,
) -> None:
    inputs = _stage_f_inputs(tmp_path)
    completion = dict(inputs["stage_r_completion"])
    completion.pop("completion_sha256")
    completion["provisional_recovery_id"] = (
        "current_fixed_floor_control_v1"
    )
    completion["rollback_backup_id"] = "relative_gap_timeout_v1"
    completion["completion_sha256"] = canonical_sha256(completion)
    inputs["stage_r_completion"] = completion

    proposal = build_stage_f_proposal(**inputs)

    assert proposal["logical_task_count"] == 192
    assert proposal["planned_unique_identity_count"] == 96
    assert proposal["reused_logical_task_count"] == 96
    assert len(proposal["identities"]) == 96
    provisional = {
        (item["record_id"], item["filter_profile_id"]): item[
            "identity_sha256"
        ]
        for item in proposal["logical_tasks"]
        if item["matrix_role"] == "provisional_recovery"
    }
    current = {
        (item["record_id"], item["filter_profile_id"]): item[
            "identity_sha256"
        ]
        for item in proposal["logical_tasks"]
        if item["matrix_role"] == "same_role_current_control"
    }
    assert current == provisional
    provisional_tasks = [
        item
        for item in proposal["logical_tasks"]
        if item["matrix_role"] == "provisional_recovery"
    ]
    current_tasks = [
        item
        for item in proposal["logical_tasks"]
        if item["matrix_role"] == "same_role_current_control"
    ]
    assert {
        (
            item["logical_stage"],
            item["numerical_identity_stage"],
            item["numeric_source_role"],
        )
        for item in provisional_tasks
    } == {
        (
            "penalty_interaction",
            "penalty_interaction",
            "provisional_recovery",
        )
    }
    assert {
        (
            item["logical_stage"],
            item["numerical_identity_stage"],
            item["numeric_source_role"],
        )
        for item in current_tasks
    } == {
        (
            "current_role_matrix",
            "penalty_interaction",
            "provisional_recovery",
        )
    }
    assert all("stage" not in item for item in proposal["logical_tasks"])


def test_stage_f_proposal_stops_after_no_safe_stage_r_completion(
    tmp_path: Path,
) -> None:
    inputs = _stage_f_inputs(tmp_path)
    completion = dict(inputs["stage_r_completion"])
    completion.pop("completion_sha256")
    completion.update(
        {
            "status": "no_safe_recovery_candidate",
            "provisional_recovery_id": None,
            "rollback_backup_id": None,
            "next_state": "awaiting_human_independent_bo_decision",
        }
    )
    completion["completion_sha256"] = canonical_sha256(completion)
    inputs["stage_r_completion"] = completion

    with pytest.raises(
        StageFPlanError,
        match="stage_r_completion_not_ready_for_stage_f",
    ):
        build_stage_f_proposal(**inputs)


def test_stage_f_proposal_rejects_baseline_metrics_that_differ_from_stage_r(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _stage_f_inputs(tmp_path)
    baseline_metrics = deepcopy(inputs["baseline_metrics"])
    baseline_metrics["records"][0]["metrics"][
        "final_motion_mae_bpm"
    ] = 9.0
    inputs["baseline_metrics"] = baseline_metrics

    with pytest.raises(
        StageFPlanError,
        match="stage_f_stage_r_source_mismatch:baseline_metrics",
    ):
        _publish_stage_f_proposal(
            tmp_path,
            monkeypatch,
            inputs=inputs,
        )

    assert not (tmp_path / "stage_f_proposal").exists()


def test_stage_f_proposal_requires_the_completed_audited_profile_library(
    tmp_path: Path,
) -> None:
    inputs = _stage_f_inputs(tmp_path)
    completion = dict(inputs["profile_library_completion"])
    completion.pop("completion_sha256")
    completion["final_library_sha256"] = "f" * 64
    completion["completion_sha256"] = canonical_sha256(completion)
    inputs["profile_library_completion"] = completion

    with pytest.raises(
        StageFPlanError,
        match="stage_f_profile_library_completion_mismatch",
    ):
        build_stage_f_proposal(**inputs)


@pytest.mark.parametrize(
    ("mutator",),
    [
        (
            lambda completion: completion.update(
                receipt_version="forged_completion_v1",
            ),
        ),
        (
            lambda completion: completion.update(
                exploration_run_count=7,
            ),
        ),
        (
            lambda completion: completion[
                "candidate_profile_receipt_sha256"
            ].pop("p100-short-rate-normalized-low-40"),
        ),
        (
            lambda completion: completion[
                "final_profile_receipt_sha256"
            ].pop("p25-short-low"),
        ),
    ],
)
def test_stage_f_proposal_rejects_rehashed_incomplete_profile_completion(
    tmp_path: Path,
    mutator,
) -> None:
    inputs = _stage_f_inputs(tmp_path)
    completion = deepcopy(inputs["profile_library_completion"])
    completion.pop("completion_sha256")
    mutator(completion)
    completion["completion_sha256"] = canonical_sha256(completion)
    inputs["profile_library_completion"] = completion

    with pytest.raises(
        StageFPlanError,
        match="stage_f_profile_library_completion_mismatch",
    ):
        build_stage_f_proposal(**inputs)


def test_stage_f_proposal_freezes_the_profile_completion_content_hash(
    tmp_path: Path,
) -> None:
    inputs = _stage_f_inputs(tmp_path)

    proposal = build_stage_f_proposal(**inputs)

    assert proposal["upstream_completion_bindings"] == {
        "profile_library_completion_sha256": inputs[
            "profile_library_completion"
        ]["completion_sha256"],
    }


def test_stage_f_publication_binds_the_formal_baseline_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _stage_f_inputs(tmp_path)
    receipt = deepcopy(inputs["baseline_contract_receipt"])
    receipt["artifact_sha256"]["record_metrics.json"] = "f" * 64
    inputs["baseline_contract_receipt"] = receipt

    with pytest.raises(
        StageFPlanError,
        match="stage_f_baseline_metric_artifact_mismatch",
    ):
        _publish_stage_f_proposal(
            tmp_path,
            monkeypatch,
            inputs=inputs,
        )

    assert not (tmp_path / "stage_f_proposal").exists()


def test_stage_f_publication_rejects_a_library_different_from_stage_r(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _stage_f_inputs(tmp_path)
    library = deepcopy(inputs["profile_library"])
    library.pop("library_sha256")
    library["unexpected_profile_source_change"] = True
    library["library_sha256"] = canonical_sha256(library)
    inputs["profile_library"] = library

    with pytest.raises(
        StageFPlanError,
        match="stage_f_stage_r_source_mismatch:profile_library",
    ):
        _publish_stage_f_proposal(
            tmp_path,
            monkeypatch,
            inputs=inputs,
        )

    assert not (tmp_path / "stage_f_proposal").exists()


def test_stage_f_proposal_requires_the_three_frozen_penalty_candidates(
    tmp_path: Path,
) -> None:
    inputs = _stage_f_inputs(tmp_path)
    original_registry = dict(inputs["penalty_registry"])
    incomplete_registry = {
        key: value
        for key, value in original_registry.items()
        if key != "registry_sha256"
    }
    incomplete_registry["candidate_count"] = 1
    incomplete_registry["candidates"] = [
        original_registry["candidates"][0]
    ]
    incomplete_registry["registry_sha256"] = canonical_sha256(
        incomplete_registry
    )
    inputs["penalty_registry"] = incomplete_registry

    stage_r_proposal = dict(inputs["stage_r_proposal"])
    stage_r_proposal.pop("proposal_sha256")
    stage_r_proposal["frozen_contracts"] = {
        **stage_r_proposal["frozen_contracts"],
        "penalty_registry_hash": incomplete_registry[
            "registry_sha256"
        ],
    }
    stage_r_proposal["proposal_sha256"] = canonical_sha256(
        stage_r_proposal
    )
    inputs["stage_r_proposal"] = stage_r_proposal
    stage_r_completion = dict(inputs["stage_r_completion"])
    stage_r_completion.pop("completion_sha256")
    stage_r_completion["proposal_sha256"] = stage_r_proposal[
        "proposal_sha256"
    ]
    stage_r_completion["completion_sha256"] = canonical_sha256(
        stage_r_completion
    )
    inputs["stage_r_completion"] = stage_r_completion

    with pytest.raises(
        StageFPlanError,
        match="stage_f_penalty_registry_mismatch",
    ):
        build_stage_f_proposal(**inputs)


def test_stage_f_proposal_publication_is_atomic_and_zero_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = _publish_stage_f_proposal(
        tmp_path,
        monkeypatch,
    )
    receipt = read_json(
        destination / "proposal_receipt.json"
    )

    assert destination.is_dir()
    assert receipt["status"] == "ready_for_execution"
    assert receipt["formal_solver_run_count"] == 0
    assert receipt["diagnostic_solver_run_count"] == 0
    assert receipt["independent_bo_run_count"] == 0
    assert receipt["planned_unique_identity_count"] == 192
    evaluation = read_json(
        destination / "evaluation_source_identity.json"
    )
    assert evaluation["root_modules"] == [
        "ppg_hr.v2.recovery_stage_f_contracts",
        "ppg_hr.v2.recovery_stage_f_execution",
        "ppg_hr.v2.recovery_stage_f_experiment",
        "ppg_hr.v2.recovery_stage_f_plan",
        "ppg_hr.v2.recovery_stage_f_reporting",
        "ppg_hr.v2.recovery_stage_f_runner",
    ]
    for name, expected_hash in receipt["artifacts"].items():
        assert file_sha256(destination / name) == expected_hash


def test_stage_f_execution_rejects_tampered_proposal_receipt_before_governance_access(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proposal_dir = _publish_stage_f_proposal(tmp_path, monkeypatch)
    receipt_path = proposal_dir / "proposal_receipt.json"
    receipt = read_json(receipt_path)
    receipt["proposal_sha256"] = "f" * 64
    atomic_write_json(receipt_path, receipt)
    output_dir = tmp_path / "stage_f_execution"

    with pytest.raises(
        StageFPlanError,
        match="stage_f_proposal_receipt_mismatch",
    ):
        execute_stage_f_proposal(
            proposal_dir=proposal_dir,
            governance_dir=tmp_path / "governance-not-created",
            output_dir=output_dir,
            source_root=Path(__file__).parents[1] / "src",
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    (
        "provisional_recovery_id",
        "expected_unique_identity_count",
        "expected_spectral_numerical_reuse_count",
        "expected_spectral_logical_reuse_count",
    ),
    [
        (
            "relative_gap_timeout_v1",
            192,
            96,
            0,
        ),
        (
            "current_fixed_floor_control_v1",
            96,
            0,
            96,
        ),
    ],
)
def test_stage_f_execution_registers_two_matrices_and_is_resumable(
    tmp_path: Path,
    monkeypatch,
    provisional_recovery_id: str,
    expected_unique_identity_count: int,
    expected_spectral_numerical_reuse_count: int,
    expected_spectral_logical_reuse_count: int,
) -> None:
    inputs = _stage_f_inputs(tmp_path)
    if (
        provisional_recovery_id
        == "current_fixed_floor_control_v1"
    ):
        completion = dict(inputs["stage_r_completion"])
        completion.pop("completion_sha256")
        completion.update(
            {
                "provisional_recovery_id": (
                    provisional_recovery_id
                ),
                "rollback_backup_id": "relative_gap_timeout_v1",
            }
        )
        completion["completion_sha256"] = canonical_sha256(
            completion
        )
        inputs["stage_r_completion"] = completion
    proposal_dir = _publish_stage_f_proposal(
        tmp_path,
        monkeypatch,
        inputs=inputs,
    )
    governance = tmp_path / "governance"
    budget = BudgetContract.approved_v5()
    exploration = ExplorationRegistry.zero_budget_v1()
    atomic_write_json(
        governance / "budget_contract.json",
        budget.to_dict(),
    )
    atomic_write_json(
        governance / "exploration_registry.json",
        exploration.to_dict(),
    )
    AttemptRegistry.create(
        governance / "attempt_registry.json",
        budget_contract=budget,
        exploration_registry=exploration,
    )
    writes: list[str] = []
    original_writer = stage_f_reporting.atomic_write_json

    def recording_writer(path: Path, payload: object) -> None:
        writes.append(Path(path).name)
        original_writer(path, payload)

    monkeypatch.setattr(
        stage_f_reporting,
        "atomic_write_json",
        recording_writer,
    )
    failed_once: set[str] = set()

    def fake_numerical(
        identity: dict[str, object],
        _spectral_dir: Path,
    ) -> StageRNumericalResult:
        identity_hash = str(identity["identity_sha256"])
        if not failed_once:
            failed_once.add(identity_hash)
            raise RuntimeError("synthetic_first_attempt_failure")
        true_rise = bool(identity["true_rise_applicable"])
        return StageRNumericalResult(
            solver_result=_synthetic_solver_result(),
            metrics=asdict(
                _synthetic_recovery_metrics(
                    true_rise=true_rise,
                )
            ),
            spectral_audit=_synthetic_spectral_evidence(
                profile_id=str(identity["filter_profile_id"]),
                record_id=str(identity["record_id"]),
                include_audit_hash=True,
            ),
        )

    execution_options: dict[str, object] = {
        "_numerical_runner": fake_numerical,
    }
    if expected_unique_identity_count == 192:
        def fake_solve(config) -> V2SolverResult:
            if not failed_once:
                failed_once.add(str(config.data_path))
                raise RuntimeError(
                    "synthetic_first_attempt_failure"
                )
            result = _synthetic_solver_result()
            metadata = {
                **result.metadata,
                "true_rise": config.data_path.name.startswith(
                    ("kaihe", "run")
                ),
            }
            return V2SolverResult(
                HR=result.HR,
                err_stats=result.err_stats,
                metadata=metadata,
                window_table=result.window_table,
            )

        def fake_evaluate(
            result: V2SolverResult,
            *,
            ref_data,
            method_names,
        ) -> RecoveryProfileMetricResult:
            del ref_data, method_names
            return _synthetic_recovery_metrics(
                true_rise=bool(
                    result.metadata.get("true_rise")
                )
            )

        def fake_audit(
            profile,
            record,
            *,
            contract,
        ) -> dict[str, object]:
            del contract
            return _synthetic_spectral_evidence(
                profile_id=str(profile.profile_id),
                record_id=str(record.record_id),
                include_audit_hash=False,
            )

        monkeypatch.setattr(stage_f_execution, "solve_v2", fake_solve)
        monkeypatch.setattr(
            stage_f_execution,
            "load_v2_reference",
            lambda _path: np.asarray([[0.0, 60.0]]),
        )
        monkeypatch.setattr(
            stage_f_execution,
            "evaluate_recovery_profile_metrics",
            fake_evaluate,
        )
        monkeypatch.setattr(
            stage_f_execution,
            "audit_stage_r_profile_record",
            fake_audit,
        )
        execution_options = {}

    output_dir = tmp_path / "stage_f_execution"
    completion = execute_stage_f_proposal(
        proposal_dir=proposal_dir,
        governance_dir=governance,
        output_dir=output_dir,
        source_root=Path(__file__).parents[1] / "src",
        **execution_options,
    )

    assert completion["status"] == "complete"
    assert completion["logical_task_count"] == 192
    assert completion["logical_result_count"] == 192
    assert completion["formal_result_count"] == (
        expected_unique_identity_count
    )
    assert completion["formal_solver_run_count"] == (
        expected_unique_identity_count
    )
    assert completion["unique_spectral_audit_count"] == 96
    assert completion["spectral_audit_result_binding_count"] == (
        expected_unique_identity_count
    )
    assert completion[
        "spectral_audit_numerical_reuse_count"
    ] == expected_spectral_numerical_reuse_count
    assert completion[
        "spectral_audit_logical_reuse_count"
    ] == expected_spectral_logical_reuse_count
    assert completion["reused_logical_task_count"] == (
        expected_spectral_logical_reuse_count
    )
    assert completion["failed_attempt_count"] == 1
    assert completion["matrix_execution_summary"][
        "retry_count"
    ] == 1
    assert completion["matrix_execution_summary"][
        "total_attempt_count"
    ] == expected_unique_identity_count + 1
    assert completion["independent_bo_run_count"] == 0
    assert writes[-1] == "stage_f_completion.json"
    primary = read_json(
        output_dir / "profile_enumeration_matrix.json"
    )
    control = read_json(
        output_dir / "same_role_current_control_matrix.json"
    )
    upper = read_json(
        output_dir / "profile_sample_in_upper_bound.json"
    )
    assert len(primary["rows"]) == 96
    assert len(control["rows"]) == 96
    assert primary["unique_spectral_audit_count"] == 96
    assert control["unique_spectral_audit_count"] == 96
    assert {
        (
            row["stage"],
            row["logical_stage"],
            row["numerical_identity_stage"],
        )
        for row in primary["rows"]
    } == {
        (
            "penalty_interaction",
            "penalty_interaction",
            "penalty_interaction",
        )
    }
    expected_control_numerical_stage = (
        "penalty_interaction"
        if expected_unique_identity_count == 96
        else "current_role_matrix"
    )
    assert {
        (
            row["stage"],
            row["logical_stage"],
            row["numerical_identity_stage"],
        )
        for row in control["rows"]
    } == {
        (
            expected_control_numerical_stage,
            "current_role_matrix",
            expected_control_numerical_stage,
        )
    }
    assert len(upper["records"]) == 12
    assert {
        row["selected_profile_id"]
        for row in upper["records"]
    } == {"p25-short-low"}
    rerun = execute_stage_f_proposal(
        proposal_dir=proposal_dir,
        governance_dir=governance,
        output_dir=output_dir,
        source_root=Path(__file__).parents[1] / "src",
        _numerical_runner=lambda _item, _audit: (_ for _ in ()).throw(
            AssertionError("completed Stage F must not rerun")
        ),
    )
    assert rerun == completion
    completion_path = output_dir / "stage_f_completion.json"
    tampered = read_json(completion_path)
    tampered.pop("completion_sha256")
    tampered["independent_bo_run_count"] = 1
    tampered["completion_sha256"] = canonical_sha256(tampered)
    original_writer(completion_path, tampered)
    with pytest.raises(
        StageFPlanError,
        match="stage_f_completion_contract_mismatch",
    ):
        execute_stage_f_proposal(
            proposal_dir=proposal_dir,
            governance_dir=governance,
            output_dir=output_dir,
            source_root=Path(__file__).parents[1] / "src",
            _numerical_runner=lambda _item, _audit: (
                _ for _ in ()
            ).throw(
                AssertionError(
                    "tampered completion must fail before rerun"
                )
            ),
        )


def test_stage_f_runner_executes_exact_proposal_and_streams_progress(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured: dict[str, object] = {}

    def fake_execute(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        callback = kwargs["progress_callback"]
        assert callable(callback)
        callback(
            {
                "stage": "stage_f_filter_matrix",
                "completed": 1,
                "total": 192,
            }
        )
        return {
            "status": "complete",
            "proposal_sha256": "a" * 64,
        }

    monkeypatch.setattr(
        stage_f_runner,
        "execute_stage_f_proposal",
        fake_execute,
    )
    proposal_dir = tmp_path / "proposal"
    governance_dir = tmp_path / "governance"
    output_dir = tmp_path / "output"
    source_root = tmp_path / "src"

    exit_code = stage_f_runner.main(
        [
            "--proposal-dir",
            str(proposal_dir),
            "--governance-dir",
            str(governance_dir),
            "--output-dir",
            str(output_dir),
            "--source-root",
            str(source_root),
        ]
    )

    assert exit_code == 0
    assert captured == {
        "proposal_dir": proposal_dir,
        "governance_dir": governance_dir,
        "output_dir": output_dir,
        "source_root": source_root,
        "progress_callback": captured["progress_callback"],
    }
    output = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
    ]
    assert output == [
        {
            "stage": "stage_f_filter_matrix",
            "completed": 1,
            "total": 192,
        },
        {
            "proposal_sha256": "a" * 64,
            "status": "complete",
        },
    ]


def test_stage_f_proposal_cli_freezes_only_the_requested_inputs(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured: dict[str, object] = {}

    def fake_propose(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "status": "ready_for_execution",
            "formal_solver_run_count": 0,
        }

    monkeypatch.setattr(
        stage_f_module,
        "propose_stage_f_execution",
        fake_propose,
    )
    arguments = {
        "stage_r_proposal_path": tmp_path / "stage-r-proposal.json",
        "stage_r_completion_path": tmp_path / "stage-r-completion.json",
        "profile_library_path": tmp_path / "profiles.json",
        "profile_library_completion_path": (
            tmp_path / "profiles-completion.json"
        ),
        "baseline_metrics_path": tmp_path / "baselines.json",
        "baseline_contract_receipt_path": (
            tmp_path / "baseline-receipt.json"
        ),
        "recovery_registry_path": tmp_path / "recovery.json",
        "penalty_registry_path": tmp_path / "penalty.json",
        "budget_contract_path": tmp_path / "budget.json",
        "output_dir": tmp_path / "stage-f-proposal",
        "source_root": tmp_path / "src",
    }
    argv: list[str] = []
    for name, value in arguments.items():
        argv.extend(
            [
                f"--{name.removesuffix('_path').replace('_', '-')}",
                str(value),
            ]
        )
    argv.extend(["--parent-experiment-id", "lyx-v1"])

    exit_code = stage_f_module.main(argv)

    assert exit_code == 0
    assert captured == {
        **arguments,
        "parent_experiment_id": "lyx-v1",
    }
    assert json.loads(capsys.readouterr().out) == {
        "formal_solver_run_count": 0,
        "status": "ready_for_execution",
    }
