"""Source-bound execution for the historical-parameter recovery A/B."""

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
from .recovery_pre_fold_gate import (
    build_historical_recovery_ab_proposal,
    build_historical_recovery_ab_report,
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


def propose_historical_recovery_ab_execution(
    *,
    stage_f_proposal_path: Path,
    stage_p_proposal_path: Path,
    rollback_receipt_path: Path,
    historical_parameter_manifest_path: Path,
    recovery_registry_path: Path,
    budget_contract_path: Path,
    output_dir: Path,
    source_root: Path,
    parent_experiment_id: str,
) -> dict[str, Any]:
    """Publish the exact zero-run historical recovery A/B proposal."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise StagePPlanError(f"historical_ab_output_already_exists:{destination}")
    source_paths = {
        "stage_f_proposal": Path(stage_f_proposal_path).resolve(),
        "stage_p_proposal": Path(stage_p_proposal_path).resolve(),
        "rollback_receipt": Path(rollback_receipt_path).resolve(),
        "historical_parameter_manifest": Path(historical_parameter_manifest_path).resolve(),
        "recovery_registry": Path(recovery_registry_path).resolve(),
        "budget_contract": Path(budget_contract_path).resolve(),
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise StagePPlanError(f"historical_ab_source_missing:{name}:{path}")
    sources = {name: read_json(path) for name, path in source_paths.items()}
    stage_f_source_artifacts = sources["stage_f_proposal"].get("source_artifacts")
    if stage_f_source_artifacts is not None:
        recovery_binding = require_mapping(
            "historical_stage_f_recovery_registry_binding",
            require_mapping(
                "historical_stage_f_source_artifacts",
                stage_f_source_artifacts,
            ).get("recovery_registry"),
        )
        if recovery_binding.get("sha256") != file_sha256(source_paths["recovery_registry"]):
            raise StagePPlanError("historical_recovery_registry_source_binding_mismatch")
    metric_contract = stage_r_metric_contract_v1()
    solver = runtime_source_identity(Path(source_root).resolve())
    evaluation_roots = (
        "ppg_hr.v2.recovery_pre_fold_execution",
        "ppg_hr.v2.recovery_pre_fold_gate",
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
    proposal = build_historical_recovery_ab_proposal(
        stage_f_proposal=sources["stage_f_proposal"],
        stage_p_proposal=sources["stage_p_proposal"],
        rollback_receipt=sources["rollback_receipt"],
        historical_parameter_manifest=sources["historical_parameter_manifest"],
        recovery_registry=sources["recovery_registry"],
        budget_contract=sources["budget_contract"],
        parent_experiment_id=parent_experiment_id,
        solver_hash=solver["source_bundle_sha256"],
        metric_contract_hash=metric_contract["contract_sha256"],
        evaluation_hash=evaluation["evaluation_hash"],
    )
    proposal.pop("proposal_sha256")
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
        raise StagePPlanError("historical_ab_staging_parent_mismatch")
    try:
        staging.mkdir(parents=True)
        atomic_write_json(staging / "metric_contract.json", metric_contract)
        atomic_write_json(staging / "solver_source_identity.json", solver)
        atomic_write_json(
            staging / "evaluation_source_identity.json",
            evaluation,
        )
        atomic_write_json(
            staging / "historical_recovery_ab_proposal.json",
            proposal,
        )
        artifact_names = (
            "metric_contract.json",
            "solver_source_identity.json",
            "evaluation_source_identity.json",
            "historical_recovery_ab_proposal.json",
        )
        receipt = {
            "receipt_version": ("lyx_historical_recovery_ab_proposal_receipt_v1"),
            "status": "ready_for_execution",
            "formal_solver_run_count": 0,
            "diagnostic_solver_run_count": 0,
            "independent_bo_run_count": 0,
            "logical_task_count": 24,
            "planned_unique_identity_count": proposal["planned_unique_identity_count"],
            "reused_logical_task_count": proposal["reused_logical_task_count"],
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
) -> tuple[dict[str, Any], Any]:
    proposal_path = proposal_root / "historical_recovery_ab_proposal.json"
    receipt_path = proposal_root / "proposal_receipt.json"
    if not proposal_path.is_file() or not receipt_path.is_file():
        raise StagePPlanError("historical_ab_proposal_package_incomplete")
    proposal = read_json(proposal_path)
    verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="historical_ab_proposal",
    )
    receipt = read_json(receipt_path)
    artifacts = require_mapping(
        "historical_ab_proposal_receipt_artifacts",
        receipt.get("artifacts"),
    )
    expected_artifacts = {
        "metric_contract.json",
        "solver_source_identity.json",
        "evaluation_source_identity.json",
        "historical_recovery_ab_proposal.json",
    }
    if (
        receipt.get("receipt_version") != "lyx_historical_recovery_ab_proposal_receipt_v1"
        or receipt.get("status") != "ready_for_execution"
        or receipt.get("proposal_sha256") != proposal["proposal_sha256"]
        or receipt.get("formal_solver_run_count") != 0
        or receipt.get("diagnostic_solver_run_count") != 0
        or receipt.get("independent_bo_run_count") != 0
        or receipt.get("logical_task_count") != 24
        or receipt.get("planned_unique_identity_count")
        != proposal.get("planned_unique_identity_count")
        or receipt.get("reused_logical_task_count") != proposal.get("reused_logical_task_count")
        or set(artifacts) != expected_artifacts
        or any(
            file_sha256(proposal_root / name) != expected_hash
            for name, expected_hash in artifacts.items()
        )
    ):
        raise StagePPlanError("historical_ab_proposal_receipt_mismatch")
    source_bindings = require_mapping(
        "historical_ab_source_artifacts",
        proposal.get("source_artifacts"),
    )
    expected_sources = {
        "stage_f_proposal",
        "stage_p_proposal",
        "rollback_receipt",
        "historical_parameter_manifest",
        "recovery_registry",
        "budget_contract",
    }
    if set(source_bindings) != expected_sources:
        raise StagePPlanError("historical_ab_source_artifact_set_mismatch")
    sources: dict[str, Any] = {}
    for name in expected_sources:
        binding = require_mapping(
            f"historical_ab_source_binding:{name}",
            source_bindings[name],
        )
        path = Path(str(binding.get("path"))).resolve()
        if not path.is_file() or file_sha256(path) != binding.get("sha256"):
            raise StagePPlanError(f"historical_ab_source_changed:{name}")
        sources[name] = read_json(path)
    if sources["rollback_receipt"].get("receipt_sha256") != proposal.get(
        "rollback_receipt_sha256"
    ) or sources["historical_parameter_manifest"].get("manifest_sha256") != proposal.get(
        "historical_parameter_manifest_sha256"
    ):
        raise StagePPlanError("historical_ab_upstream_binding_mismatch")
    solver_identity = read_json(proposal_root / "solver_source_identity.json")
    if solver_identity != runtime_source_identity(source_root):
        raise StagePPlanError("historical_ab_solver_source_changed")
    evaluation_identity = read_json(proposal_root / "evaluation_source_identity.json")
    roots = tuple(
        str(value)
        for value in require_list(
            "historical_ab_evaluation_roots",
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
        or proposal["frozen_contracts"].get("historical_ab_evaluation_hash")
        != evaluation_identity.get("evaluation_hash")
    ):
        raise StagePPlanError("historical_ab_evaluation_source_changed")
    budget = _budget_from_payload(sources["budget_contract"])
    if budget.sha256 != proposal["frozen_contracts"].get("budget_contract_hash"):
        raise StagePPlanError("historical_ab_budget_hash_mismatch")
    identities = [
        attempt_identity_from_item(require_mapping("historical_ab_identity", raw))
        for raw in proposal["identities"]
    ]
    expected_count = int(proposal["planned_unique_identity_count"])
    if (
        expected_count not in {12, 24}
        or len(identities) != expected_count
        or len({identity.sha256 for identity in identities}) != expected_count
    ):
        raise StagePPlanError("historical_ab_identity_matrix_mismatch")
    return proposal, budget


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
        artifact_name="historical_ab_completion",
    )
    artifacts = require_mapping(
        "historical_ab_completion_artifacts",
        completion.get("artifacts"),
    )
    expected_artifacts = {
        "historical_recovery_ab_report.json",
        "attempt_registry_historical_ab_snapshot.json",
    }
    if (
        completion.get("completion_version") != "lyx_historical_recovery_ab_completion_v1"
        or completion.get("status") != "complete"
        or completion.get("proposal_sha256") != proposal.get("proposal_sha256")
        or completion.get("logical_result_count") != 24
        or completion.get("formal_result_count") != proposal.get("planned_unique_identity_count")
        or completion.get("independent_bo_run_count") != 0
        or completion.get("next_state") != "ready_for_pre_fold_independent_bo_gate"
        or set(artifacts) != expected_artifacts
        or any(
            not (output_root / name).is_file() or file_sha256(output_root / name) != expected_hash
            for name, expected_hash in artifacts.items()
        )
    ):
        raise StagePPlanError("historical_ab_completion_contract_mismatch")
    identities = tuple(
        attempt_identity_from_item(require_mapping("historical_ab_identity", raw))
        for raw in proposal["identities"]
    )
    registry.assert_complete_matrix(identities)
    registry.assert_matrix_matches_snapshot(
        identities,
        read_json(output_root / "attempt_registry_historical_ab_snapshot.json"),
    )
    governance_path = governance_root / "historical_ab_governance_receipt.json"
    if not governance_path.is_file() or file_sha256(governance_path) != completion.get(
        "governance_receipt_file_sha256"
    ):
        raise StagePPlanError("historical_ab_governance_receipt_mismatch")
    return completion


def execute_historical_recovery_ab_proposal(
    *,
    proposal_dir: Path,
    governance_dir: Path,
    output_dir: Path,
    source_root: Path,
    _numerical_runner: StageRNumericalRunner | None = None,
    progress_callback: StagePProgressCallback | None = None,
) -> dict[str, Any]:
    """Execute or resume the exact 12/24 numerical A/B identities."""

    proposal_root = Path(proposal_dir).resolve()
    proposal, source_budget = _verify_preflight(
        proposal_root=proposal_root,
        source_root=Path(source_root).resolve(),
    )
    governance_root = Path(governance_dir).resolve()
    governance_budget = _budget_from_payload(read_json(governance_root / "budget_contract.json"))
    if (
        governance_budget.sha256 != source_budget.sha256
        or governance_budget.to_dict() != source_budget.to_dict()
    ):
        raise StagePPlanError("historical_ab_governance_budget_mismatch")
    exploration = _exploration_from_payload(
        read_json(governance_root / "exploration_registry.json")
    )
    registry = AttemptRegistry.open(
        governance_root / "attempt_registry.json",
        budget_contract=governance_budget,
        exploration_registry=exploration,
    )
    output_root = Path(output_dir).resolve()
    completion_path = output_root / "historical_ab_completion.json"
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
        dict(require_mapping("historical_ab_identity", raw)) for raw in proposal["identities"]
    ]
    identities = tuple(attempt_identity_from_item(item) for item in raw_identities)
    for identity in identities:
        registry.register_identity(identity)
    runner = _run_stage_f_numerical_identity if _numerical_runner is None else _numerical_runner
    numerical_rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw_identities, start=1):
        row = _execute_stage_f_identity_with_retry(
            registry=registry,
            item=item,
            numerical_runner=runner,
            spectral_audit_dir=output_root / "spectral_audits",
            retry_limit=governance_budget.retry_limit,
            progress_callback=progress_callback,
        )
        numerical_rows.append(merge_identity_result_metadata(item=item, row=row))
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "historical_recovery_ab",
                    "completed": index,
                    "total": len(raw_identities),
                    "identity_sha256": row["identity_sha256"],
                    "cache_hit": row["cache_hit"],
                }
            )
    report = build_historical_recovery_ab_report(
        proposal=proposal,
        numerical_rows=numerical_rows,
    )
    atomic_write_json(
        output_root / "historical_recovery_ab_report.json",
        report,
    )
    snapshot = registry.matrix_snapshot(identities)
    atomic_write_json(
        output_root / "attempt_registry_historical_ab_snapshot.json",
        snapshot,
    )
    artifact_names = (
        "historical_recovery_ab_report.json",
        "attempt_registry_historical_ab_snapshot.json",
    )
    artifacts = {name: file_sha256(output_root / name) for name in artifact_names}
    summary = registry.matrix_execution_summary(identities)
    governance = {
        "receipt_version": "lyx_historical_ab_governance_receipt_v1",
        "status": "complete",
        "proposal_sha256": proposal["proposal_sha256"],
        "attempt_registry_matrix_snapshot_sha256": snapshot["snapshot_sha256"],
        "matrix_execution_summary": summary,
        "logical_result_count": 24,
        "formal_result_count": len(identities),
        "independent_bo_run_count": 0,
        "artifacts": artifacts,
    }
    governance["receipt_sha256"] = canonical_sha256(governance)
    governance_path = governance_root / "historical_ab_governance_receipt.json"
    atomic_write_json(governance_path, governance)
    completion = {
        "completion_version": "lyx_historical_recovery_ab_completion_v1",
        "status": "complete",
        "proposal_sha256": proposal["proposal_sha256"],
        "logical_result_count": 24,
        "formal_result_count": len(identities),
        "reused_logical_task_count": proposal["reused_logical_task_count"],
        "formal_solver_run_count": summary["identity_with_solver_attempt_count"],
        "cache_hit_count": summary["cache_only_identity_count"],
        "failed_attempt_count": summary["failed_attempt_count"],
        "independent_bo_run_count": 0,
        "artifacts": artifacts,
        "governance_receipt_sha256": governance["receipt_sha256"],
        "governance_receipt_file_sha256": file_sha256(governance_path),
        "next_state": "ready_for_pre_fold_independent_bo_gate",
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
