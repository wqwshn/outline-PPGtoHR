"""Historical-parameter recovery A/B and the pre-fold BO human gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .phase2_experiment_io import atomic_write_json, file_sha256, read_json
from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import AttemptIdentity
from .recovery_profile_upper_bound import (
    build_sample_in_upper_bound_payloads,
)
from .recovery_stage_p_contracts import (
    HISTORICAL_RECOVERY_AB_STAGE,
    StagePPlanError,
    require_hash,
    require_list,
    require_mapping,
    validate_recovery_candidate_registry,
    verify_embedded_hash,
)


def freeze_historical_parameter_replay_manifest(
    *,
    baseline_manifest_path: Path,
    archive_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze the 12 archived selected parameter sets without solver work."""

    baseline_path = Path(baseline_manifest_path).resolve()
    root = Path(archive_root).resolve()
    destination = Path(output_path).resolve()
    if destination.exists():
        raise StagePPlanError(f"historical_parameter_output_already_exists:{destination}")
    if not baseline_path.is_file() or not root.is_dir():
        raise StagePPlanError("historical_parameter_source_missing")
    baseline = read_json(baseline_path)
    records = [
        dict(require_mapping("historical_baseline_record", raw))
        for raw in require_list(
            "historical_baseline_records",
            baseline.get("records"),
        )
    ]
    if (
        baseline.get("manifest_version") != "lyx_recovery_profile_baseline_manifest_v1"
        or len(records) != 12
    ):
        raise StagePPlanError("historical_baseline_manifest_mismatch")
    frozen_records: list[dict[str, Any]] = []
    for record in records:
        relative = Path(str(record["selected_candidate"]))
        selected_path = (root / relative).resolve()
        try:
            selected_path.relative_to(root)
        except ValueError as error:
            raise StagePPlanError("historical_parameter_source_outside_archive") from error
        if not selected_path.is_file():
            raise StagePPlanError(f"historical_parameter_source_missing:{selected_path}")
        selected = read_json(selected_path)
        parameters = dict(
            require_mapping(
                "historical_selected_actual_params",
                selected.get("actual_params"),
            )
        )
        expected_parameters = {
            "analysis_scope",
            "fs_target",
            "lms_mu_base",
            "lms_mu_min",
            "max_order",
            "smooth_win_len",
            "spec_penalty_width",
            "time_bias",
        }
        if (
            selected.get("candidate_id") != record.get("candidate_id")
            or set(parameters) != expected_parameters
            or selected.get("arm") != "physical_new"
        ):
            raise StagePPlanError(f"historical_parameter_source_mismatch:{record.get('sample_id')}")
        frozen_records.append(
            {
                "record_id": record["sample_id"],
                "scene": record["scene"],
                "data_sha256": record["data_sha256"],
                "reference_sha256": record["reference_sha256"],
                "archive_candidate_id": record["candidate_id"],
                "parameter_source_relative_path": relative.as_posix(),
                "parameter_source_sha256": file_sha256(selected_path),
                "parameters": parameters,
            }
        )
    if len({record["record_id"] for record in frozen_records}) != 12 or {
        scene: sum(record["scene"] == scene for record in frozen_records)
        for scene in {"jianpan", "kaihe", "run", "xiezi"}
    } != {"jianpan": 3, "kaihe": 3, "run": 3, "xiezi": 3}:
        raise StagePPlanError("historical_parameter_panel_mismatch")
    manifest = {
        "manifest_version": "lyx_historical_parameter_replay_manifest_v1",
        "status": "frozen_zero_solver_runs",
        "archive_git_commit": baseline["archive_git_commit"],
        "baseline_manifest_file_sha256": file_sha256(baseline_path),
        "record_count": 12,
        "formal_solver_run_count": 0,
        "independent_bo_run_count": 0,
        "records": frozen_records,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write_json(destination, manifest)
    return manifest


def _historical_records(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    verify_embedded_hash(
        manifest,
        hash_field="manifest_sha256",
        artifact_name="historical_parameter_manifest",
    )
    records = {
        str(record["record_id"]): record
        for record in (
            dict(require_mapping("historical_parameter_record", raw))
            for raw in require_list(
                "historical_parameter_records",
                manifest.get("records"),
            )
        )
    }
    required_parameters = {
        "analysis_scope",
        "fs_target",
        "lms_mu_base",
        "lms_mu_min",
        "max_order",
        "smooth_win_len",
        "spec_penalty_width",
        "time_bias",
    }
    if (
        manifest.get("manifest_version") != "lyx_historical_parameter_replay_manifest_v1"
        or manifest.get("status", "frozen_zero_solver_runs") != "frozen_zero_solver_runs"
        or manifest.get("record_count") != 12
        or manifest.get("formal_solver_run_count", 0) != 0
        or manifest.get("independent_bo_run_count", 0) != 0
        or len(records) != 12
        or any(
            set(
                require_mapping(
                    f"historical_parameters:{record_id}",
                    record.get("parameters"),
                )
            )
            != required_parameters
            for record_id, record in records.items()
        )
        or any(
            require_hash(
                f"historical_parameter_source_sha256:{record_id}",
                record.get("parameter_source_sha256"),
            )
            == ""
            for record_id, record in records.items()
        )
        or any(
            (
                "data_sha256" in record
                and not require_hash(
                    f"historical_data_sha256:{record_id}",
                    record.get("data_sha256"),
                )
            )
            or (
                "reference_sha256" in record
                and not require_hash(
                    f"historical_reference_sha256:{record_id}",
                    record.get("reference_sha256"),
                )
            )
            for record_id, record in records.items()
        )
    ):
        raise StagePPlanError("historical_parameter_manifest_mismatch")
    return records


def _historical_identity(
    *,
    template: Mapping[str, Any],
    historical: Mapping[str, Any],
    recovery: Mapping[str, Any],
    penalty: Mapping[str, Any],
    arm: str,
    solver_hash: str,
    metric_contract_hash: str,
    evaluation_hash: str,
    parent_experiment_id: str,
) -> dict[str, Any]:
    config = deepcopy(
        dict(
            require_mapping(
                "historical_template_config",
                template.get("config"),
            )
        )
    )
    parameters = dict(
        require_mapping(
            "historical_template_parameters",
            config.get("parameters"),
        )
    )
    parameters.update(
        dict(
            require_mapping(
                "historical_parameters",
                historical.get("parameters"),
            )
        )
    )
    parameters.update(
        {
            "recovery_candidate_id": recovery["candidate_id"],
            "penalty_candidate_id": penalty["penalty_id"],
        }
    )
    config["parameters"] = parameters
    attempt = AttemptIdentity(
        solver_hash=solver_hash,
        config_hash=canonical_sha256(config),
        metric_contract_hash=metric_contract_hash,
        evaluation_hash=evaluation_hash,
        data_sha256=str(template["data_sha256"]),
        record_id=str(template["record_id"]),
        stage=HISTORICAL_RECOVERY_AB_STAGE,
        attempt_kind="formal",
        parent_experiment_id=parent_experiment_id,
    )
    constants = require_mapping(
        "historical_recovery_constants",
        recovery.get("constants"),
    )
    fs_target = int(parameters["fs_target"])
    actual_taps = int(parameters["max_order"])
    return {
        **attempt.to_dict(),
        "matrix_role": arm,
        "scene": template["scene"],
        "data_path": template["data_path"],
        "reference_path": template["reference_path"],
        "raw_data_sha256": template["raw_data_sha256"],
        "reference_sha256": template["reference_sha256"],
        "method_names": deepcopy(template["method_names"]),
        "true_rise_applicable": template["true_rise_applicable"],
        "config": config,
        "filter_profile_id": f"historical-{template['record_id']}",
        "filter_profile_sha256": canonical_sha256(
            {
                "record_id": template["record_id"],
                "parameters": historical["parameters"],
            }
        ),
        "filter_profile_design_role": "core",
        "physical_memory_ms": max(
            1,
            int(round(1000.0 * actual_taps / fs_target)),
        ),
        "actual_taps": actual_taps,
        "nominal_mu": float(parameters["lms_mu_base"]),
        "sentinel_role": None,
        "recovery_candidate_id": recovery["candidate_id"],
        "recovery_candidate_sha256": recovery["candidate_sha256"],
        "candidate_min_bpm": constants.get("candidate_min_bpm"),
        "penalty_candidate_id": penalty["penalty_id"],
        "penalty_candidate_sha256": penalty["candidate_sha256"],
        "historical_parameter_source_sha256": historical["parameter_source_sha256"],
    }


def build_historical_recovery_ab_proposal(
    *,
    stage_f_proposal: Mapping[str, Any],
    stage_p_proposal: Mapping[str, Any],
    rollback_receipt: Mapping[str, Any],
    historical_parameter_manifest: Mapping[str, Any],
    recovery_registry: Mapping[str, Any],
    budget_contract: Mapping[str, Any],
    parent_experiment_id: str,
    solver_hash: str,
    metric_contract_hash: str,
    evaluation_hash: str,
) -> dict[str, Any]:
    """Freeze same-code, same-parameter current/new recovery A/B."""

    for name, value in (
        ("solver_hash", solver_hash),
        ("metric_contract_hash", metric_contract_hash),
        ("evaluation_hash", evaluation_hash),
    ):
        require_hash(name, value)
    stage_f_sha = verify_embedded_hash(
        stage_f_proposal,
        hash_field="proposal_sha256",
        artifact_name="historical_stage_f_proposal",
    )
    stage_p_sha = verify_embedded_hash(
        stage_p_proposal,
        hash_field="proposal_sha256",
        artifact_name="historical_stage_p_proposal",
    )
    rollback_sha = verify_embedded_hash(
        rollback_receipt,
        hash_field="receipt_sha256",
        artifact_name="historical_rollback_receipt",
    )
    historical = _historical_records(historical_parameter_manifest)
    recoveries = validate_recovery_candidate_registry(
        recovery_registry,
        artifact_name="historical_recovery_registry",
    )
    penalties = {
        str(candidate["penalty_id"]): candidate
        for candidate in (
            dict(require_mapping("historical_penalty_candidate", raw))
            for raw in require_list(
                "historical_penalty_candidates",
                stage_p_proposal.get("penalties"),
            )
        )
    }
    current_id = str(recovery_registry.get("control_candidate_id"))
    final_id = rollback_receipt.get("final_recovery_id")
    selected_penalty_id = str(rollback_receipt.get("selected_penalty_id"))
    templates: dict[str, dict[str, Any]] = {}
    for raw in require_list(
        "historical_stage_f_identities",
        stage_f_proposal.get("identities"),
    ):
        template = dict(require_mapping("historical_stage_f_identity", raw))
        if template.get("matrix_role") == "provisional_recovery":
            templates.setdefault(str(template["record_id"]), template)
    limits = require_mapping(
        "historical_budget_stage_unique_limits",
        budget_contract.get("stage_unique_limits"),
    )
    stage_f_frozen = require_mapping(
        "historical_stage_f_frozen_contracts",
        stage_f_proposal.get("frozen_contracts"),
    )
    stage_p_frozen = require_mapping(
        "historical_stage_p_frozen_contracts",
        stage_p_proposal.get("frozen_contracts"),
    )
    if (
        rollback_receipt.get("status") not in {"retained", "rolled_back"}
        or rollback_receipt.get("next_state") != "ready_for_historical_recovery_ab_proposal"
        or rollback_receipt.get("rollback_count") not in {0, 1}
        or rollback_receipt.get("penalty_reselection_count") != 0
        or rollback_receipt.get("candidate_reselection_count") != 0
        or final_id not in recoveries
        or current_id not in recoveries
        or selected_penalty_id not in penalties
        or set(templates) != set(historical)
        or len(templates) != 12
        or limits.get(HISTORICAL_RECOVERY_AB_STAGE) != 24
        or stage_f_sha != stage_p_proposal.get("stage_f_proposal_sha256")
        or stage_p_sha != rollback_receipt.get("stage_p_proposal_sha256")
        or stage_f_frozen.get("recovery_candidate_registry_hash")
        != recovery_registry.get("registry_sha256")
        or stage_p_frozen.get("recovery_candidate_registry_hash")
        != recovery_registry.get("registry_sha256")
        or stage_f_frozen.get("budget_contract_hash") != canonical_sha256(budget_contract)
        or stage_p_frozen.get("budget_contract_hash") != canonical_sha256(budget_contract)
    ):
        raise StagePPlanError("historical_ab_upstream_contract_mismatch")
    if any(
        template.get("solver_hash") != solver_hash
        or template.get("metric_contract_hash") != metric_contract_hash
        or (
            "data_sha256" in historical[record_id]
            and historical[record_id]["data_sha256"] != template.get("raw_data_sha256")
        )
        or (
            "reference_sha256" in historical[record_id]
            and historical[record_id]["reference_sha256"] != template.get("reference_sha256")
        )
        for record_id, template in templates.items()
    ):
        raise StagePPlanError("historical_ab_runtime_or_data_mismatch")
    arms = [
        ("current_recovery", recoveries[current_id]),
        ("final_recovery", recoveries[str(final_id)]),
    ]
    numerical_by_hash: dict[str, dict[str, Any]] = {}
    logical_tasks: list[dict[str, Any]] = []
    for arm, recovery in arms:
        for record_id in sorted(templates):
            identity = _historical_identity(
                template=templates[record_id],
                historical=historical[record_id],
                recovery=recovery,
                penalty=penalties[selected_penalty_id],
                arm=arm,
                solver_hash=solver_hash,
                metric_contract_hash=metric_contract_hash,
                evaluation_hash=evaluation_hash,
                parent_experiment_id=parent_experiment_id,
            )
            numerical_by_hash.setdefault(
                str(identity["identity_sha256"]),
                identity,
            )
            logical_tasks.append(
                {
                    "arm": arm,
                    "record_id": record_id,
                    "scene": identity["scene"],
                    "recovery_candidate_id": recovery["candidate_id"],
                    "penalty_candidate_id": selected_penalty_id,
                    "identity_sha256": identity["identity_sha256"],
                }
            )
    identities = list(numerical_by_hash.values())
    expected_unique = 12 if current_id == final_id else 24
    if len(logical_tasks) != 24 or len(identities) != expected_unique:
        raise StagePPlanError("historical_ab_identity_matrix_mismatch")
    proposal = {
        "proposal_version": "lyx_historical_recovery_ab_proposal_v1",
        "status": "ready_for_execution",
        "parent_experiment_id": parent_experiment_id,
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "independent_bo_authorized": False,
        "stage_f_proposal_sha256": stage_f_sha,
        "stage_p_proposal_sha256": stage_p_sha,
        "rollback_receipt_sha256": rollback_sha,
        "historical_parameter_manifest_sha256": (historical_parameter_manifest["manifest_sha256"]),
        "current_recovery_id": current_id,
        "final_recovery_id": final_id,
        "selected_penalty_id": selected_penalty_id,
        "record_count": 12,
        "logical_task_count": 24,
        "planned_unique_identity_count": expected_unique,
        "reused_logical_task_count": 24 - expected_unique,
        "frozen_contracts": {
            "metric_contract_hash": metric_contract_hash,
            "recovery_candidate_registry_hash": recovery_registry["registry_sha256"],
            "penalty_registry_hash": stage_p_proposal["frozen_contracts"]["penalty_registry_hash"],
            "budget_contract_hash": canonical_sha256(budget_contract),
            "historical_ab_evaluation_hash": evaluation_hash,
        },
        "record_panel": deepcopy(stage_f_proposal["record_panel"]),
        "identities": identities,
        "logical_tasks": logical_tasks,
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def build_historical_recovery_ab_report(
    *,
    proposal: Mapping[str, Any],
    numerical_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Expand any identical-arm reuse and calculate paired deltas."""

    by_identity = {str(row["identity_sha256"]): dict(row) for row in numerical_rows}
    logical_rows: list[dict[str, Any]] = []
    for task in proposal["logical_tasks"]:
        numerical = by_identity.get(str(task["identity_sha256"]))
        if numerical is None:
            raise StagePPlanError("historical_ab_result_missing")
        logical_rows.append({**numerical, **task})
    if len(logical_rows) != 24:
        raise StagePPlanError("historical_ab_logical_result_mismatch")
    by_arm_record = {(str(row["arm"]), str(row["record_id"])): row for row in logical_rows}
    paired_records: list[dict[str, Any]] = []
    for record in proposal["record_panel"]:
        record_id = str(record["record_id"])
        current = by_arm_record[("current_recovery", record_id)]
        final = by_arm_record[("final_recovery", record_id)]
        current_metrics = require_mapping(
            "historical_current_metrics",
            current.get("metrics"),
        )
        final_metrics = require_mapping(
            "historical_final_metrics",
            final.get("metrics"),
        )
        paired_records.append(
            {
                "record_id": record_id,
                "scene": record["scene"],
                "current_identity_sha256": current["identity_sha256"],
                "final_identity_sha256": final["identity_sha256"],
                "current_metrics": dict(current_metrics),
                "final_metrics": dict(final_metrics),
                "delta_final_minus_current": {
                    "final_motion_mae_bpm": (
                        float(final_metrics["final_motion_mae_bpm"])
                        - float(current_metrics["final_motion_mae_bpm"])
                    ),
                    "longest_e10_run_windows": (
                        int(final_metrics["longest_e10_run_windows"])
                        - int(current_metrics["longest_e10_run_windows"])
                    ),
                    "longest_e20_run_windows": (
                        int(final_metrics["longest_e20_run_windows"])
                        - int(current_metrics["longest_e20_run_windows"])
                    ),
                },
                "spectral_gate_pass": (
                    final["spectral_audit"].get("stability_pass") is True
                    and final["spectral_audit"].get("spectral_gate_pass") is True
                ),
            }
        )
    report = {
        "report_version": "lyx_historical_recovery_ab_report_v1",
        "status": "complete",
        "proposal_sha256": proposal["proposal_sha256"],
        "logical_result_count": 24,
        "formal_result_count": len(numerical_rows),
        "reused_logical_task_count": proposal["reused_logical_task_count"],
        "records": paired_records,
        "independent_bo_run_count": 0,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def evaluate_pre_fold_independent_bo_gate(
    *,
    historical_ab_report: Mapping[str, Any],
    final_profile_rows: Sequence[Mapping[str, Any]],
    independent_metrics_by_record: Mapping[str, Mapping[str, Any]],
    review_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Mechanically evaluate all four pre-fold trigger conditions."""

    verify_embedded_hash(
        historical_ab_report,
        hash_field="report_sha256",
        artifact_name="pre_fold_historical_ab_report",
    )
    ab_records = [
        dict(require_mapping("pre_fold_ab_record", raw))
        for raw in require_list(
            "pre_fold_ab_records",
            historical_ab_report.get("records"),
        )
    ]
    if len(ab_records) != 12:
        raise StagePPlanError("pre_fold_ab_panel_incomplete")
    condition_1_records: list[str] = []
    condition_1_deltas: list[float] = []
    condition_3_records: list[str] = []
    trigger_evidence_by_record: dict[str, dict[str, Any]] = {}
    for record in ab_records:
        record_id = str(record["record_id"])
        scene = str(record["scene"])
        independent = require_mapping(
            f"pre_fold_independent_metrics:{record_id}",
            independent_metrics_by_record.get(record_id),
        )
        current = require_mapping(
            "pre_fold_current_metrics",
            record.get("current_metrics"),
        )
        final = require_mapping(
            "pre_fold_final_metrics",
            record.get("final_metrics"),
        )
        delta = float(final["final_motion_mae_bpm"]) - float(current["final_motion_mae_bpm"])
        condition_1_deltas.append(delta)
        final_l10_failure = int(final["longest_e10_run_windows"]) > max(
            10,
            int(independent["longest_e10_run_windows"]) + 2,
        )
        current_l10_pass = int(current["longest_e10_run_windows"]) <= max(
            10,
            int(independent["longest_e10_run_windows"]) + 2,
        )
        final_l20_failure = int(final["longest_e20_run_windows"]) > max(
            2,
            int(independent["longest_e20_run_windows"]),
        )
        current_l20_pass = int(current["longest_e20_run_windows"]) <= max(
            2,
            int(independent["longest_e20_run_windows"]),
        )
        new_l10_failure = final_l10_failure and current_l10_pass
        new_l20_failure = final_l20_failure and current_l20_pass
        reason_categories: list[str] = []
        if delta > 2.0:
            reason_categories.append("historical_parameter_mae_regression")
        if new_l10_failure:
            reason_categories.append("new_longest_e10_tail_failure")
        if new_l20_failure:
            reason_categories.append("new_longest_e20_tail_failure")
        if delta > 2.0 or new_l10_failure or new_l20_failure:
            condition_1_records.append(record_id)
        tail_pass = not final_l10_failure and not final_l20_failure
        if record.get("spectral_gate_pass") is True and tail_pass and delta > 2.0:
            condition_3_records.append(record_id)
            reason_categories.append("spectral_and_tail_pass_but_mae_regression")
        trigger_evidence_by_record[record_id] = {
            "record_id": record_id,
            "scene": scene,
            "historical_independent_bo_metrics": dict(independent),
            "historical_parameter_recovery_ab": {
                "current_metrics": dict(current),
                "final_metrics": dict(final),
                "delta_final_minus_current": {
                    "final_motion_mae_bpm": delta,
                    "longest_e10_run_windows": (
                        int(final["longest_e10_run_windows"])
                        - int(current["longest_e10_run_windows"])
                    ),
                    "longest_e20_run_windows": (
                        int(final["longest_e20_run_windows"])
                        - int(current["longest_e20_run_windows"])
                    ),
                },
            },
            "reason_categories": reason_categories,
        }
    mean_delta = sum(condition_1_deltas) / len(condition_1_deltas)
    if mean_delta > 0.5:
        for record_id, delta in zip(
            (str(record["record_id"]) for record in ab_records),
            condition_1_deltas,
            strict=True,
        ):
            if delta <= 0.0:
                continue
            condition_1_records.append(record_id)
            trigger_evidence_by_record[record_id]["reason_categories"].append(
                "historical_parameter_mean_mae_regression"
            )
    condition_1 = bool(condition_1_records) or mean_delta > 0.5
    final_coordinates = {
        (str(row["filter_profile_id"]), str(row["record_id"])) for row in final_profile_rows
    }
    final_record_ids = {str(row["record_id"]) for row in final_profile_rows}
    if (
        len(final_profile_rows) != 96
        or len(final_coordinates) != 96
        or final_record_ids != set(independent_metrics_by_record)
        or any(
            sum(str(row["record_id"]) == record_id for row in final_profile_rows) != 8
            for record_id in final_record_ids
        )
    ):
        raise StagePPlanError("pre_fold_final_profile_matrix_incomplete")
    scene_by_record = {str(record["record_id"]): str(record["scene"]) for record in ab_records}
    upper_bounds = build_sample_in_upper_bound_payloads(
        final_profile_rows=final_profile_rows,
        scene_by_record=scene_by_record,
    )
    condition_2_evidence: list[dict[str, Any]] = []
    for record in upper_bounds["sample_in_upper_bound"]["records"]:
        record_id = str(record["record_id"])
        metrics = require_mapping(
            "pre_fold_sample_in_selected_metrics",
            record.get("selected_metrics"),
        )
        independent_mae = float(independent_metrics_by_record[record_id]["final_motion_mae_bpm"])
        sample_in_mae = float(metrics["final_motion_mae_bpm"])
        delta = sample_in_mae - independent_mae
        sample_in_evidence = {
            "definition": upper_bounds["sample_in_upper_bound"]["definition"],
            "selected_profile_id": record["selected_profile_id"],
            "selected_identity_sha256": record["selected_identity_sha256"],
            "selected_qualified": record["selected_qualified"],
            "selected_metrics": dict(metrics),
            "historical_independent_bo_mae_bpm": independent_mae,
            "delta_mae_bpm": delta,
        }
        trigger_evidence_by_record[record_id]["sample_in_upper_bound"] = sample_in_evidence
        if delta <= 2.0:
            continue
        condition_2_evidence.append(
            {
                "record_id": record_id,
                "scene": record["scene"],
                **sample_in_evidence,
            }
        )
        trigger_evidence_by_record[record_id]["reason_categories"].append(
            "combination_library_raw_coverage_gap"
        )
        if record["selected_qualified"] is not True:
            trigger_evidence_by_record[record_id]["reason_categories"].append(
                "raw_coverage_best_fails_engineering_gate"
            )
    condition_2_records = [str(record["record_id"]) for record in condition_2_evidence]
    condition_2 = (
        len(set(condition_2_records)) >= 3
        and len({scene_by_record[record_id] for record_id in condition_2_records}) >= 2
    )
    condition_3 = len(set(condition_3_records)) >= 2
    condition_4 = False
    triggered = condition_1 or condition_2 or condition_3 or condition_4
    trigger_record_ids = (
        set(condition_1_records) | set(condition_2_records) | set(condition_3_records)
    )
    trigger_records = [
        {
            **trigger_evidence_by_record[record_id],
            "reason_categories": sorted(
                set(trigger_evidence_by_record[record_id]["reason_categories"])
            ),
        }
        for record_id in sorted(trigger_record_ids)
    ]
    context = dict(review_context)
    required_context = {
        "planned_search_space_hash",
        "planned_seed_manifest_hash",
        "planned_unique_budget",
        "estimated_runtime",
        "estimated_cache_size",
        "recommendation",
        "run_answers",
        "no_run_answers",
    }
    if triggered and (
        set(context) != required_context
        or not isinstance(context["planned_unique_budget"], int)
        or context["planned_unique_budget"] <= 0
        or any(not context[name] for name in required_context if name != "planned_unique_budget")
    ):
        raise StagePPlanError("pre_fold_review_context_incomplete")
    if triggered:
        require_hash(
            "pre_fold_planned_search_space_hash",
            context["planned_search_space_hash"],
        )
        require_hash(
            "pre_fold_planned_seed_manifest_hash",
            context["planned_seed_manifest_hash"],
        )
    receipt = {
        "receipt_version": "lyx_pre_fold_independent_bo_gate_v1",
        "status": (
            "awaiting_human_independent_bo_decision" if triggered else "ready_for_fold_replay"
        ),
        "triggered": triggered,
        "conditions": {
            "historical_ab_regression": {
                "triggered": condition_1,
                "record_ids": sorted(set(condition_1_records)),
                "mean_delta_mae_bpm": mean_delta,
            },
            "sample_in_upper_bound_gap": {
                "triggered": condition_2,
                "record_ids": sorted(set(condition_2_records)),
                "records": condition_2_evidence,
            },
            "spectral_tail_pass_but_mae_regression": {
                "triggered": condition_3,
                "record_ids": sorted(set(condition_3_records)),
            },
            "no_safe_recovery_candidate": {
                "triggered": condition_4,
            },
        },
        "trigger_records": trigger_records,
        "review_packet": context if triggered else None,
        "independent_bo_run_count": 0,
        "independent_bo_authorized": False,
        "next_state": (
            "awaiting_human_independent_bo_decision" if triggered else "ready_for_fold_replay"
        ),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def publish_stage_r_no_safe_independent_bo_gate(
    *,
    stage_r_proposal_path: Path,
    stage_r_completion_path: Path,
    stage_r_review_package_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Immediately publish the Stage R no-safe human gate and stop Stage F."""

    source_paths = {
        "stage_r_proposal": Path(stage_r_proposal_path).resolve(),
        "stage_r_completion": Path(stage_r_completion_path).resolve(),
        "stage_r_review_package": Path(stage_r_review_package_path).resolve(),
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise StagePPlanError(f"stage_r_no_safe_source_missing:{name}:{path}")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise StagePPlanError(f"stage_r_no_safe_output_already_exists:{destination}")
    proposal = read_json(source_paths["stage_r_proposal"])
    completion = read_json(source_paths["stage_r_completion"])
    review_package = read_json(source_paths["stage_r_review_package"])
    proposal_sha = verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="stage_r_no_safe_proposal",
    )
    completion_sha = verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="stage_r_no_safe_completion",
    )
    package_sha = verify_embedded_hash(
        review_package,
        hash_field="package_sha256",
        artifact_name="stage_r_no_safe_review_package",
    )
    completion_artifacts = require_mapping(
        "stage_r_no_safe_completion_artifacts",
        completion.get("artifacts"),
    )
    review_file_hash = file_sha256(source_paths["stage_r_review_package"])
    if (
        proposal.get("proposal_version") != "lyx_stage_r_execution_proposal_v1"
        or proposal.get("independent_bo_authorized") is not False
        or completion.get("completion_version") != "lyx_stage_r_completion_v2"
        or completion.get("status") != "no_safe_recovery_candidate"
        or completion.get("proposal_sha256") != proposal_sha
        or completion.get("next_state") != "awaiting_human_independent_bo_decision"
        or completion.get("independent_bo_run_count") != 0
        or completion_artifacts.get("independent_bo_review_package.json") != review_file_hash
        or review_package.get("package_version") != "lyx_stage_r_independent_bo_review_v1"
        or review_package.get("status") != "awaiting_human_independent_bo_decision"
        or review_package.get("trigger") != "no_safe_recovery_candidate"
        or review_package.get("proposal_sha256") != proposal_sha
        or review_package.get("authorization_sha256") != completion.get("authorization_sha256")
        or review_package.get("independent_bo_authorized") is not False
        or review_package.get("independent_bo_run_count") != 0
        or not isinstance(review_package.get("execution_identity_count"), int)
        or review_package["execution_identity_count"] <= 0
    ):
        raise StagePPlanError("stage_r_no_safe_gate_source_mismatch")
    receipt = {
        "receipt_version": "lyx_stage_r_no_safe_independent_bo_gate_v1",
        "status": "awaiting_human_independent_bo_decision",
        "triggered": True,
        "trigger_stage": "stage_r",
        "stage_r_proposal_sha256": proposal_sha,
        "stage_r_completion_sha256": completion_sha,
        "stage_r_review_package_sha256": package_sha,
        "conditions": {
            "historical_ab_regression": {
                "triggered": False,
                "status": "not_evaluated_stage_r_stop",
            },
            "sample_in_upper_bound_gap": {
                "triggered": False,
                "status": "not_evaluated_stage_r_stop",
            },
            "spectral_tail_pass_but_mae_regression": {
                "triggered": False,
                "status": "not_evaluated_stage_r_stop",
            },
            "no_safe_recovery_candidate": {
                "triggered": True,
            },
        },
        "review_packet": {
            "package_sha256": package_sha,
            "execution_identity_count": review_package["execution_identity_count"],
            "requested_human_decision": review_package.get("requested_human_decision"),
        },
        "source_artifacts": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in source_paths.items()
        },
        "independent_bo_run_count": 0,
        "independent_bo_authorized": False,
        "stage_f_allowed": False,
        "next_state": "awaiting_human_independent_bo_decision",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    destination.mkdir(parents=True)
    atomic_write_json(
        destination / "pre_fold_independent_bo_gate_receipt.json",
        receipt,
    )
    return receipt


def publish_pre_fold_independent_bo_gate(
    *,
    historical_ab_report_path: Path,
    final_interaction_audit_path: Path,
    review_context_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Publish the source-bound gate receipt without running any BO."""

    source_paths = {
        "historical_ab_report": Path(historical_ab_report_path).resolve(),
        "final_interaction_audit": Path(final_interaction_audit_path).resolve(),
        "review_context": Path(review_context_path).resolve(),
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise StagePPlanError(f"pre_fold_source_missing:{name}:{path}")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise StagePPlanError(f"pre_fold_output_already_exists:{destination}")
    historical_report = read_json(source_paths["historical_ab_report"])
    final_audit = read_json(source_paths["final_interaction_audit"])
    review_context = read_json(source_paths["review_context"])
    verify_embedded_hash(
        final_audit,
        hash_field="audit_sha256",
        artifact_name="pre_fold_final_interaction_audit",
    )
    final_rows = [
        require_mapping("pre_fold_final_profile_row", raw)
        for raw in require_list(
            "pre_fold_final_profile_rows",
            final_audit.get("rows"),
        )
    ]
    expected_upper_bounds = build_sample_in_upper_bound_payloads(
        final_profile_rows=final_rows,
        scene_by_record={
            str(record["record_id"]): str(record["scene"])
            for record in require_list(
                "pre_fold_historical_records",
                historical_report.get("records"),
            )
        },
    )
    if any(final_audit.get(name) != payload for name, payload in expected_upper_bounds.items()):
        raise StagePPlanError("pre_fold_sample_in_upper_bound_audit_mismatch")
    gate = evaluate_pre_fold_independent_bo_gate(
        historical_ab_report=historical_report,
        final_profile_rows=final_rows,
        independent_metrics_by_record={
            str(record_id): require_mapping(
                f"pre_fold_independent_metrics:{record_id}",
                metrics,
            )
            for record_id, metrics in require_mapping(
                "pre_fold_independent_metrics_by_record",
                final_audit.get("independent_metrics_by_record"),
            ).items()
        },
        review_context=require_mapping(
            "pre_fold_review_context",
            review_context,
        ),
    )
    gate.pop("receipt_sha256")
    gate["historical_ab_report_sha256"] = historical_report["report_sha256"]
    gate["final_interaction_audit_sha256"] = final_audit["audit_sha256"]
    gate["source_artifacts"] = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
        }
        for name, path in source_paths.items()
    }
    gate["receipt_sha256"] = canonical_sha256(gate)
    destination.mkdir(parents=True)
    atomic_write_json(
        destination / "pre_fold_independent_bo_gate_receipt.json",
        gate,
    )
    if gate["triggered"]:
        packet = {
            "packet_version": "lyx_independent_bo_review_packet_v1",
            "gate_receipt_sha256": gate["receipt_sha256"],
            "trigger_conditions": gate["conditions"],
            "trigger_records": gate["trigger_records"],
            "source_artifacts": gate["source_artifacts"],
            **dict(gate["review_packet"]),
            "independent_bo_run_count": 0,
            "independent_bo_authorized": False,
        }
        packet["packet_sha256"] = canonical_sha256(packet)
        atomic_write_json(
            destination / "independent_bo_review_packet.json",
            packet,
        )
    return gate
