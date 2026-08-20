"""Source-bound proposal and execution for the fixed rollback backup."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .experiment_freeze_utils import runtime_source_identity
from .phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    read_json,
)
from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import AttemptRegistry
from .recovery_interaction_resolution import (
    build_final_interaction_audit,
    build_rollback_backup_proposal,
    resolve_recovery_interaction,
)
from .recovery_stage_f_execution import (
    _execute_stage_f_identity_with_retry,
    _run_stage_f_numerical_identity,
)
from .recovery_stage_p_contracts import (
    StagePPlanError,
    StagePProgressCallback,
    attempt_identity_from_item,
    merge_identity_result_metadata,
    require_list,
    require_mapping,
    verify_embedded_hash,
)
from .recovery_stage_p_execution import (
    _budget_from_payload,
    _exploration_from_payload,
)
from .recovery_stage_r_common import StageRNumericalRunner
from .recovery_stage_r_experiment import stage_r_metric_contract_v1


def propose_rollback_backup_execution(
    *,
    stage_f_proposal_path: Path,
    stage_p_proposal_path: Path,
    stage_p_completion_path: Path,
    penalty_interaction_report_path: Path,
    stage_f_current_role_matrix_path: Path,
    recovery_registry_path: Path,
    budget_contract_path: Path,
    output_dir: Path,
    source_root: Path,
    parent_experiment_id: str,
) -> dict[str, Any]:
    """Publish an atomic zero-run proposal for the fixed backup matrix."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise StagePPlanError(f"rollback_output_already_exists:{destination}")
    source_paths = {
        "stage_f_proposal": Path(stage_f_proposal_path).resolve(),
        "stage_p_proposal": Path(stage_p_proposal_path).resolve(),
        "stage_p_completion": Path(stage_p_completion_path).resolve(),
        "penalty_interaction_report": Path(penalty_interaction_report_path).resolve(),
        "stage_f_current_role_matrix": Path(stage_f_current_role_matrix_path).resolve(),
        "recovery_registry": Path(recovery_registry_path).resolve(),
        "budget_contract": Path(budget_contract_path).resolve(),
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise StagePPlanError(f"rollback_source_missing:{name}:{path}")
    sources = {name: read_json(path) for name, path in source_paths.items()}
    stage_f_source_artifacts = sources["stage_f_proposal"].get("source_artifacts")
    if stage_f_source_artifacts is not None:
        recovery_binding = require_mapping(
            "rollback_stage_f_recovery_registry_binding",
            require_mapping(
                "rollback_stage_f_source_artifacts",
                stage_f_source_artifacts,
            ).get("recovery_registry"),
        )
        if recovery_binding.get("sha256") != file_sha256(source_paths["recovery_registry"]):
            raise StagePPlanError("rollback_recovery_registry_source_binding_mismatch")
    metric_contract = stage_r_metric_contract_v1()
    solver = runtime_source_identity(Path(source_root).resolve())
    evaluation_roots = (
        "ppg_hr.v2.recovery_interaction_execution",
        "ppg_hr.v2.recovery_interaction_resolution",
    )
    evaluation = runtime_source_identity(
        Path(source_root).resolve(),
        root_modules=evaluation_roots,
    )
    evaluation = {
        "root_modules": list(evaluation_roots),
        **evaluation,
        "evaluation_hash": evaluation["source_bundle_sha256"],
    }
    proposal = build_rollback_backup_proposal(
        stage_f_proposal=sources["stage_f_proposal"],
        stage_p_proposal=sources["stage_p_proposal"],
        stage_p_completion=sources["stage_p_completion"],
        penalty_interaction_report=sources["penalty_interaction_report"],
        recovery_registry=sources["recovery_registry"],
        budget_contract=sources["budget_contract"],
        parent_experiment_id=parent_experiment_id,
        solver_hash=solver["source_bundle_sha256"],
        metric_contract_hash=metric_contract["contract_sha256"],
        evaluation_hash=evaluation["evaluation_hash"],
    )
    proposal.pop("proposal_sha256")
    proposal["stage_f_current_role_matrix_sha256"] = sources["stage_f_current_role_matrix"][
        "matrix_sha256"
    ]
    proposal["source_artifacts"] = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
        }
        for name, path in source_paths.items()
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.staging")
    if staging.parent != destination.parent:
        raise StagePPlanError("rollback_staging_parent_mismatch")
    try:
        staging.mkdir(parents=True)
        atomic_write_json(staging / "metric_contract.json", metric_contract)
        atomic_write_json(staging / "solver_source_identity.json", solver)
        atomic_write_json(
            staging / "evaluation_source_identity.json",
            evaluation,
        )
        atomic_write_json(
            staging / "rollback_backup_proposal.json",
            proposal,
        )
        artifact_names = (
            "metric_contract.json",
            "solver_source_identity.json",
            "evaluation_source_identity.json",
            "rollback_backup_proposal.json",
        )
        receipt = {
            "receipt_version": "lyx_rollback_backup_proposal_receipt_v1",
            "status": proposal["status"],
            "formal_solver_run_count": 0,
            "diagnostic_solver_run_count": 0,
            "independent_bo_run_count": 0,
            "planned_unique_identity_count": proposal["planned_unique_identity_count"],
            "proposal_sha256": proposal["proposal_sha256"],
            "artifacts": {name: file_sha256(staging / name) for name in artifact_names},
        }
        atomic_write_json(staging / "proposal_receipt.json", receipt)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
        return receipt
    except Exception:
        if staging.exists() and staging.parent == destination.parent:
            shutil.rmtree(staging)
        raise


def _verify_preflight(
    *,
    proposal_root: Path,
    source_root: Path,
) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    proposal_path = proposal_root / "rollback_backup_proposal.json"
    receipt_path = proposal_root / "proposal_receipt.json"
    if not proposal_path.is_file() or not receipt_path.is_file():
        raise StagePPlanError("rollback_proposal_package_incomplete")
    proposal = read_json(proposal_path)
    verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="rollback_backup_proposal",
    )
    receipt = read_json(receipt_path)
    artifacts = require_mapping(
        "rollback_proposal_receipt_artifacts",
        receipt.get("artifacts"),
    )
    expected_artifacts = {
        "metric_contract.json",
        "solver_source_identity.json",
        "evaluation_source_identity.json",
        "rollback_backup_proposal.json",
    }
    if (
        receipt.get("receipt_version") != "lyx_rollback_backup_proposal_receipt_v1"
        or receipt.get("status") != proposal.get("status")
        or receipt.get("proposal_sha256") != proposal["proposal_sha256"]
        or receipt.get("formal_solver_run_count") != 0
        or receipt.get("diagnostic_solver_run_count") != 0
        or receipt.get("independent_bo_run_count") != 0
        or receipt.get("planned_unique_identity_count")
        != proposal.get("planned_unique_identity_count")
        or set(artifacts) != expected_artifacts
        or any(
            file_sha256(proposal_root / name) != expected_hash
            for name, expected_hash in artifacts.items()
        )
    ):
        raise StagePPlanError("rollback_proposal_receipt_mismatch")
    source_bindings = require_mapping(
        "rollback_source_artifacts",
        proposal.get("source_artifacts"),
    )
    expected_sources = {
        "stage_f_proposal",
        "stage_p_proposal",
        "stage_p_completion",
        "penalty_interaction_report",
        "stage_f_current_role_matrix",
        "recovery_registry",
        "budget_contract",
    }
    if set(source_bindings) != expected_sources:
        raise StagePPlanError("rollback_source_artifact_set_mismatch")
    sources: dict[str, Any] = {}
    for name in expected_sources:
        binding = require_mapping(
            f"rollback_source_binding:{name}",
            source_bindings[name],
        )
        path = Path(str(binding.get("path"))).resolve()
        if not path.is_file() or file_sha256(path) != binding.get("sha256"):
            raise StagePPlanError(f"rollback_source_changed:{name}")
        sources[name] = read_json(path)
    if sources["stage_f_current_role_matrix"].get("matrix_sha256") != proposal.get(
        "stage_f_current_role_matrix_sha256"
    ) or sources["penalty_interaction_report"].get("report_sha256") != proposal.get(
        "penalty_interaction_report_sha256"
    ):
        raise StagePPlanError("rollback_upstream_binding_mismatch")
    solver_identity = read_json(proposal_root / "solver_source_identity.json")
    if solver_identity != runtime_source_identity(source_root):
        raise StagePPlanError("rollback_solver_source_changed")
    evaluation_identity = read_json(proposal_root / "evaluation_source_identity.json")
    roots = tuple(
        str(value)
        for value in require_list(
            "rollback_evaluation_roots",
            evaluation_identity.get("root_modules"),
        )
    )
    current_evaluation = runtime_source_identity(
        source_root,
        root_modules=roots,
    )
    if (
        current_evaluation.get("source_files") != evaluation_identity.get("source_files")
        or current_evaluation.get("source_bundle_sha256")
        != evaluation_identity.get("source_bundle_sha256")
        or proposal["frozen_contracts"].get("rollback_evaluation_hash")
        != evaluation_identity.get("evaluation_hash")
    ):
        raise StagePPlanError("rollback_evaluation_source_changed")
    budget = _budget_from_payload(sources["budget_contract"])
    if budget.sha256 != proposal["frozen_contracts"].get("budget_contract_hash"):
        raise StagePPlanError("rollback_budget_hash_mismatch")
    identities = [
        attempt_identity_from_item(require_mapping("rollback_identity", raw))
        for raw in require_list(
            "rollback_identities",
            proposal.get("identities"),
        )
    ]
    expected_count = int(proposal["planned_unique_identity_count"])
    if (
        expected_count not in {0, 96}
        or len(identities) != expected_count
        or len({identity.sha256 for identity in identities}) != expected_count
    ):
        raise StagePPlanError("rollback_identity_matrix_mismatch")
    return (
        proposal,
        budget,
        sources["penalty_interaction_report"],
        sources["stage_f_current_role_matrix"],
    )


def _validate_completion(
    *,
    completion_path: Path,
    proposal: Mapping[str, Any],
    output_root: Path,
    governance_root: Path,
    registry: AttemptRegistry,
) -> dict[str, Any]:
    completion = read_json(completion_path)
    verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="rollback_completion",
    )
    artifacts = require_mapping(
        "rollback_completion_artifacts",
        completion.get("artifacts"),
    )
    status = completion.get("status")
    expected_artifacts = {
        "rollback_backup_matrix.json",
        "recovery_rollback_receipt.json",
        "attempt_registry_rollback_snapshot.json",
    }
    if status in {"retained", "rolled_back"}:
        expected_artifacts.add("final_interaction_audit.json")
    if (
        completion.get("completion_version") != "lyx_rollback_backup_completion_v1"
        or status
        not in {
            "retained",
            "rolled_back",
            "awaiting_human_interaction_decision",
        }
        or completion.get("proposal_sha256") != proposal.get("proposal_sha256")
        or completion.get("formal_result_count") != proposal.get("planned_unique_identity_count")
        or completion.get("rollback_count") not in {0, 1}
        or completion.get("independent_bo_run_count") != 0
        or (
            status == "awaiting_human_interaction_decision"
            and (
                completion.get("rollback_count") != 0
                or completion.get("final_recovery_id") is not None
                or completion.get("next_state") != "awaiting_human_interaction_decision"
            )
        )
        or (
            status in {"retained", "rolled_back"}
            and (
                not completion.get("final_recovery_id")
                or completion.get("next_state") != "ready_for_historical_recovery_ab_proposal"
            )
        )
        or set(artifacts) != expected_artifacts
        or any(
            not (output_root / name).is_file() or file_sha256(output_root / name) != expected_hash
            for name, expected_hash in artifacts.items()
        )
    ):
        raise StagePPlanError("rollback_completion_contract_mismatch")
    identities = tuple(
        attempt_identity_from_item(require_mapping("rollback_identity", raw))
        for raw in proposal["identities"]
    )
    if identities:
        registry.assert_complete_matrix(identities)
        registry.assert_matrix_matches_snapshot(
            identities,
            read_json(output_root / "attempt_registry_rollback_snapshot.json"),
        )
    governance_path = governance_root / "rollback_backup_governance_receipt.json"
    if not governance_path.is_file() or file_sha256(governance_path) != completion.get(
        "governance_receipt_file_sha256"
    ):
        raise StagePPlanError("rollback_governance_receipt_mismatch")
    return completion


def execute_rollback_backup_proposal(
    *,
    proposal_dir: Path,
    governance_dir: Path,
    output_dir: Path,
    source_root: Path,
    _numerical_runner: StageRNumericalRunner | None = None,
    progress_callback: StagePProgressCallback | None = None,
) -> dict[str, Any]:
    """Run or resume the fixed backup and apply the one-shot trigger."""

    proposal_root = Path(proposal_dir).resolve()
    proposal, source_budget, interaction_report, current_matrix = _verify_preflight(
        proposal_root=proposal_root,
        source_root=Path(source_root).resolve(),
    )
    governance_root = Path(governance_dir).resolve()
    governance_budget = _budget_from_payload(read_json(governance_root / "budget_contract.json"))
    if (
        governance_budget.sha256 != source_budget.sha256
        or governance_budget.to_dict() != source_budget.to_dict()
    ):
        raise StagePPlanError("rollback_governance_budget_mismatch")
    exploration = _exploration_from_payload(
        read_json(governance_root / "exploration_registry.json")
    )
    registry = AttemptRegistry.open(
        governance_root / "attempt_registry.json",
        budget_contract=governance_budget,
        exploration_registry=exploration,
    )
    output_root = Path(output_dir).resolve()
    completion_path = output_root / "rollback_backup_completion.json"
    if completion_path.is_file():
        return _validate_completion(
            completion_path=completion_path,
            proposal=proposal,
            output_root=output_root,
            governance_root=governance_root,
            registry=registry,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    raw_identities = [
        dict(require_mapping("rollback_identity", raw)) for raw in proposal["identities"]
    ]
    identities = tuple(attempt_identity_from_item(item) for item in raw_identities)
    for identity in identities:
        registry.register_identity(identity)
    runner = _run_stage_f_numerical_identity if _numerical_runner is None else _numerical_runner
    backup_rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw_identities, start=1):
        row = _execute_stage_f_identity_with_retry(
            registry=registry,
            item=item,
            numerical_runner=runner,
            spectral_audit_dir=output_root / "spectral_audits",
            retry_limit=governance_budget.retry_limit,
            progress_callback=progress_callback,
        )
        backup_rows.append(merge_identity_result_metadata(item=item, row=row))
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "rollback_backup_matrix",
                    "completed": index,
                    "total": len(raw_identities),
                    "identity_sha256": row["identity_sha256"],
                    "cache_hit": row["cache_hit"],
                }
            )
    provisional_rows = [
        dict(require_mapping("rollback_provisional_row", raw))
        for raw in interaction_report["rows"]
        if require_mapping(
            "rollback_provisional_row",
            raw,
        ).get("penalty_candidate_id")
        == proposal["selected_penalty_id"]
    ]
    current_rows = [
        dict(require_mapping("rollback_current_role_row", raw)) for raw in current_matrix["rows"]
    ]
    rollback_receipt = resolve_recovery_interaction(
        proposal=proposal,
        provisional_rows=provisional_rows,
        backup_rows=backup_rows,
        current_role_rows=current_rows,
    )
    final_interaction_audit = None
    if rollback_receipt["status"] in {"retained", "rolled_back"}:
        final_interaction_audit = build_final_interaction_audit(
            proposal=proposal,
            rollback_receipt=rollback_receipt,
            penalty_interaction_report=interaction_report,
            backup_rows=backup_rows,
            current_role_rows=current_rows,
        )
    matrix = {
        "matrix_version": "lyx_rollback_backup_matrix_v1",
        "matrix_role": "fixed_rollback_backup",
        "selected_penalty_id": proposal["selected_penalty_id"],
        "rollback_backup_id": proposal["rollback_backup_id"],
        "row_count": len(backup_rows),
        "rows": backup_rows,
    }
    matrix["matrix_sha256"] = canonical_sha256(matrix)
    atomic_write_json(
        output_root / "rollback_backup_matrix.json",
        matrix,
    )
    atomic_write_json(
        output_root / "recovery_rollback_receipt.json",
        rollback_receipt,
    )
    if final_interaction_audit is not None:
        atomic_write_json(
            output_root / "final_interaction_audit.json",
            final_interaction_audit,
        )
    if identities:
        snapshot = registry.matrix_snapshot(identities)
        summary = registry.matrix_execution_summary(identities)
    else:
        snapshot = {
            "snapshot_version": "lyx_empty_attempt_matrix_snapshot_v1",
            "identity_count": 0,
        }
        snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
        summary = {
            "identity_count": 0,
            "completed_identity_count": 0,
            "identity_with_solver_attempt_count": 0,
            "cache_only_identity_count": 0,
            "failed_attempt_count": 0,
        }
    atomic_write_json(
        output_root / "attempt_registry_rollback_snapshot.json",
        snapshot,
    )
    artifact_names = [
        "rollback_backup_matrix.json",
        "recovery_rollback_receipt.json",
        "attempt_registry_rollback_snapshot.json",
    ]
    if final_interaction_audit is not None:
        artifact_names.append("final_interaction_audit.json")
    artifacts = {name: file_sha256(output_root / name) for name in artifact_names}
    governance = {
        "receipt_version": "lyx_rollback_backup_governance_receipt_v1",
        "status": rollback_receipt["status"],
        "proposal_sha256": proposal["proposal_sha256"],
        "attempt_registry_matrix_snapshot_sha256": snapshot["snapshot_sha256"],
        "matrix_execution_summary": summary,
        "formal_result_count": len(identities),
        "independent_bo_run_count": 0,
        "artifacts": artifacts,
    }
    governance["receipt_sha256"] = canonical_sha256(governance)
    governance_path = governance_root / "rollback_backup_governance_receipt.json"
    atomic_write_json(governance_path, governance)
    completion = {
        "completion_version": "lyx_rollback_backup_completion_v1",
        "status": rollback_receipt["status"],
        "proposal_sha256": proposal["proposal_sha256"],
        "formal_result_count": len(identities),
        "formal_solver_run_count": summary["identity_with_solver_attempt_count"],
        "cache_hit_count": summary["cache_only_identity_count"],
        "failed_attempt_count": summary["failed_attempt_count"],
        "selected_penalty_id": proposal["selected_penalty_id"],
        "provisional_recovery_id": proposal["provisional_recovery_id"],
        "rollback_backup_id": proposal["rollback_backup_id"],
        "final_recovery_id": rollback_receipt["final_recovery_id"],
        "rollback_triggered": rollback_receipt["rollback_triggered"],
        "rollback_count": rollback_receipt["rollback_count"],
        "independent_bo_run_count": 0,
        "artifacts": artifacts,
        "governance_receipt_sha256": governance["receipt_sha256"],
        "governance_receipt_file_sha256": file_sha256(governance_path),
        "next_state": rollback_receipt["next_state"],
    }
    completion["completion_sha256"] = canonical_sha256(completion)
    atomic_write_json(completion_path, completion)
    return _validate_completion(
        completion_path=completion_path,
        proposal=proposal,
        output_root=output_root,
        governance_root=governance_root,
        registry=registry,
    )
