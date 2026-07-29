"""Execution and resumable reporting for a frozen LYX Stage P proposal."""

from __future__ import annotations

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
from .recovery_experiment_governance import (
    AttemptRegistry,
    BudgetContract,
    ExplorationRegistry,
)
from .recovery_stage_f_execution import (
    _execute_stage_f_identity_with_retry,
    _run_stage_f_numerical_identity,
)
from .recovery_stage_p_contracts import (
    EXPECTED_LOGICAL_RESULT_COUNT,
    EXPECTED_NEW_IDENTITY_COUNT,
    EXPECTED_REUSED_RESULT_COUNT,
    StagePPlanError,
    StagePProgressCallback,
    attempt_identity_from_item,
    merge_identity_result_metadata,
    require_list,
    require_mapping,
    verify_embedded_hash,
)
from .recovery_stage_p_reporting import build_penalty_interaction_report
from .recovery_stage_r_common import StageRNumericalRunner


def _budget_from_payload(payload: Mapping[str, Any]) -> BudgetContract:
    return BudgetContract(
        contract_version=str(payload["contract_version"]),
        stage_unique_limits=dict(
            require_mapping(
                "stage_p_stage_unique_limits",
                payload.get("stage_unique_limits"),
            )
        ),
        normal_unique_identity_limit=payload.get("normal_unique_identity_limit"),
        supplemental_stage=(
            None
            if payload.get("supplemental_stage") is None
            else str(payload["supplemental_stage"])
        ),
        stage_attempt_kinds=dict(
            require_mapping(
                "stage_p_stage_attempt_kinds",
                payload.get("stage_attempt_kinds"),
            )
        ),
        max_unique_identities=int(payload["max_unique_identities"]),
        max_attempts=int(payload["max_attempts"]),
        retry_limit=int(payload["retry_limit"]),
    )


def _exploration_from_payload(
    payload: Mapping[str, Any],
) -> ExplorationRegistry:
    return ExplorationRegistry(
        registry_version=str(payload["registry_version"]),
        unique_budget=int(payload["unique_budget"]),
        allowed_identity_sha256=tuple(
            str(value)
            for value in require_list(
                "stage_p_allowed_exploration_identities",
                payload.get("allowed_identity_sha256"),
            )
        ),
    )


def _verify_preflight(
    *,
    proposal_root: Path,
    source_root: Path,
) -> tuple[
    dict[str, Any],
    BudgetContract,
    dict[str, Any],
    dict[str, Any],
]:
    proposal_path = proposal_root / "stage_p_execution_proposal.json"
    receipt_path = proposal_root / "proposal_receipt.json"
    if not proposal_path.is_file() or not receipt_path.is_file():
        raise StagePPlanError("stage_p_proposal_package_incomplete")
    proposal = read_json(proposal_path)
    verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="stage_p_proposal",
    )
    receipt = read_json(receipt_path)
    artifact_hashes = require_mapping(
        "stage_p_proposal_receipt_artifacts",
        receipt.get("artifacts"),
    )
    expected_names = {
        "metric_contract.json",
        "solver_source_identity.json",
        "evaluation_source_identity.json",
        "stage_p_execution_proposal.json",
    }
    if (
        receipt.get("receipt_version") != "lyx_stage_p_proposal_receipt_v1"
        or receipt.get("status") != "ready_for_execution"
        or receipt.get("proposal_sha256") != proposal["proposal_sha256"]
        or receipt.get("formal_solver_run_count") != 0
        or receipt.get("diagnostic_solver_run_count") != 0
        or receipt.get("independent_bo_run_count") != 0
        or receipt.get("logical_task_count") != EXPECTED_LOGICAL_RESULT_COUNT
        or receipt.get("reused_stage_f_result_count") != EXPECTED_REUSED_RESULT_COUNT
        or receipt.get("planned_new_unique_identity_count") != EXPECTED_NEW_IDENTITY_COUNT
        or set(artifact_hashes) != expected_names
        or any(
            file_sha256(proposal_root / name) != expected_hash
            for name, expected_hash in artifact_hashes.items()
        )
    ):
        raise StagePPlanError("stage_p_proposal_receipt_mismatch")
    source_artifacts = require_mapping(
        "stage_p_source_artifacts",
        proposal.get("source_artifacts"),
    )
    expected_sources = {
        "stage_f_proposal",
        "stage_f_completion",
        "stage_f_profile_matrix",
        "stage_f_current_role_matrix",
        "penalty_registry",
        "budget_contract",
    }
    if set(source_artifacts) != expected_sources:
        raise StagePPlanError("stage_p_source_artifact_set_mismatch")
    sources: dict[str, Any] = {}
    for name in expected_sources:
        binding = require_mapping(
            f"stage_p_source_binding:{name}",
            source_artifacts[name],
        )
        path = Path(str(binding.get("path"))).resolve()
        if not path.is_file() or file_sha256(path) != binding.get("sha256"):
            raise StagePPlanError(f"stage_p_source_changed:{name}")
        sources[name] = read_json(path)
    if (
        sources["stage_f_proposal"].get("proposal_sha256")
        != proposal.get("stage_f_proposal_sha256")
        or sources["stage_f_completion"].get("completion_sha256")
        != proposal.get("stage_f_completion_sha256")
        or sources["stage_f_profile_matrix"].get("matrix_sha256")
        != proposal.get("stage_f_profile_matrix_sha256")
        or sources["stage_f_current_role_matrix"].get("matrix_sha256")
        != proposal.get("stage_f_current_role_matrix_sha256")
    ):
        raise StagePPlanError("stage_p_upstream_binding_mismatch")
    solver_identity = read_json(proposal_root / "solver_source_identity.json")
    if solver_identity != runtime_source_identity(source_root):
        raise StagePPlanError("stage_p_solver_source_changed")
    evaluation_identity = read_json(proposal_root / "evaluation_source_identity.json")
    roots = tuple(
        str(value)
        for value in require_list(
            "stage_p_evaluation_roots",
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
        or proposal["frozen_contracts"].get("stage_p_evaluation_hash")
        != evaluation_identity.get("evaluation_hash")
    ):
        raise StagePPlanError("stage_p_evaluation_source_changed")
    budget = _budget_from_payload(sources["budget_contract"])
    if budget.sha256 != proposal["frozen_contracts"].get("budget_contract_hash"):
        raise StagePPlanError("stage_p_budget_hash_mismatch")
    raw_identities = require_list(
        "stage_p_identities",
        proposal.get("identities"),
    )
    identities = [
        attempt_identity_from_item(require_mapping("stage_p_identity", raw))
        for raw in raw_identities
    ]
    if (
        len(identities) != EXPECTED_NEW_IDENTITY_COUNT
        or len({identity.sha256 for identity in identities}) != EXPECTED_NEW_IDENTITY_COUNT
        or [identity.sha256 for identity in identities]
        != [
            str(require_mapping("stage_p_identity", raw)["identity_sha256"])
            for raw in raw_identities
        ]
    ):
        raise StagePPlanError("stage_p_identity_matrix_mismatch")
    return (
        proposal,
        budget,
        sources["stage_f_profile_matrix"],
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
        artifact_name="stage_p_completion",
    )
    artifacts = require_mapping(
        "stage_p_completion_artifacts",
        completion.get("artifacts"),
    )
    expected_artifacts = {
        "penalty_interaction_matrix.json",
        "penalty_selection_receipt.json",
        "attempt_registry_stage_p_snapshot.json",
    }
    if (
        completion.get("completion_version") != "lyx_stage_p_completion_v1"
        or completion.get("status") != "selected"
        or completion.get("proposal_sha256") != proposal.get("proposal_sha256")
        or completion.get("logical_result_count") != EXPECTED_LOGICAL_RESULT_COUNT
        or completion.get("new_formal_result_count") != EXPECTED_NEW_IDENTITY_COUNT
        or completion.get("reused_stage_f_result_count") != EXPECTED_REUSED_RESULT_COUNT
        or completion.get("independent_bo_run_count") != 0
        or set(artifacts) != expected_artifacts
        or any(
            not (output_root / name).is_file() or file_sha256(output_root / name) != expected_hash
            for name, expected_hash in artifacts.items()
        )
    ):
        raise StagePPlanError("stage_p_completion_contract_mismatch")
    governance_path = governance_root / "stage_p_governance_receipt.json"
    if not governance_path.is_file() or file_sha256(governance_path) != completion.get(
        "governance_receipt_file_sha256"
    ):
        raise StagePPlanError("stage_p_governance_receipt_mismatch")
    identities = tuple(
        attempt_identity_from_item(require_mapping("stage_p_identity", raw))
        for raw in proposal["identities"]
    )
    registry.assert_complete_matrix(identities)
    snapshot = read_json(output_root / "attempt_registry_stage_p_snapshot.json")
    registry.assert_matrix_matches_snapshot(identities, snapshot)
    return completion


def _finalize(
    *,
    proposal: Mapping[str, Any],
    current_matrix: Mapping[str, Any],
    current_role_matrix: Mapping[str, Any],
    new_rows: list[dict[str, Any]],
    output_root: Path,
    governance_root: Path,
    registry: AttemptRegistry,
) -> dict[str, Any]:
    identities = tuple(
        attempt_identity_from_item(require_mapping("stage_p_identity", raw))
        for raw in proposal["identities"]
    )
    registry.assert_complete_matrix(identities)
    current_rows = [
        dict(require_mapping("stage_p_current_row", raw))
        for raw in require_list(
            "stage_p_current_rows",
            current_matrix.get("rows"),
        )
    ]
    report = build_penalty_interaction_report(
        proposal=proposal,
        current_rows=current_rows,
        current_role_rows=[
            require_mapping("stage_p_current_role_row", raw)
            for raw in require_list(
                "stage_p_current_role_rows",
                current_role_matrix.get("rows"),
            )
        ],
        new_rows=new_rows,
    )
    atomic_write_json(
        output_root / "penalty_interaction_matrix.json",
        report,
    )
    selection = {
        "receipt_version": "lyx_stage_p_penalty_selection_receipt_v1",
        "status": "selected",
        "proposal_sha256": proposal["proposal_sha256"],
        "interaction_report_sha256": report["report_sha256"],
        "selected_penalty_id": report["selected_penalty_id"],
        "selection_rule": report["penalty_selection_rule"],
        "penalty_scores": report["penalty_scores"],
        "no_fourth_strategy_after_freeze": True,
        "independent_bo_run_count": 0,
        "next_state": report["next_state"],
    }
    selection["receipt_sha256"] = canonical_sha256(selection)
    atomic_write_json(
        output_root / "penalty_selection_receipt.json",
        selection,
    )
    snapshot = registry.matrix_snapshot(identities)
    atomic_write_json(
        output_root / "attempt_registry_stage_p_snapshot.json",
        snapshot,
    )
    artifact_names = (
        "penalty_interaction_matrix.json",
        "penalty_selection_receipt.json",
        "attempt_registry_stage_p_snapshot.json",
    )
    artifacts = {name: file_sha256(output_root / name) for name in artifact_names}
    summary = registry.matrix_execution_summary(identities)
    governance = {
        "receipt_version": "lyx_stage_p_governance_receipt_v1",
        "status": "complete",
        "proposal_sha256": proposal["proposal_sha256"],
        "identity_matrix_sha256": canonical_sha256([identity.sha256 for identity in identities]),
        "attempt_registry_matrix_snapshot_sha256": snapshot["snapshot_sha256"],
        "matrix_execution_summary": summary,
        "logical_result_count": EXPECTED_LOGICAL_RESULT_COUNT,
        "new_formal_result_count": EXPECTED_NEW_IDENTITY_COUNT,
        "reused_stage_f_result_count": EXPECTED_REUSED_RESULT_COUNT,
        "independent_bo_run_count": 0,
        "artifacts": artifacts,
    }
    governance["receipt_sha256"] = canonical_sha256(governance)
    governance_path = governance_root / "stage_p_governance_receipt.json"
    atomic_write_json(governance_path, governance)
    completion = {
        "completion_version": "lyx_stage_p_completion_v1",
        "status": "selected",
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "proposal_sha256": proposal["proposal_sha256"],
        "logical_result_count": EXPECTED_LOGICAL_RESULT_COUNT,
        "new_formal_result_count": EXPECTED_NEW_IDENTITY_COUNT,
        "reused_stage_f_result_count": EXPECTED_REUSED_RESULT_COUNT,
        "formal_solver_run_count": summary["identity_with_solver_attempt_count"],
        "cache_hit_count": summary["cache_only_identity_count"],
        "failed_attempt_count": summary["failed_attempt_count"],
        "selected_penalty_id": report["selected_penalty_id"],
        "rollback_backup_id": proposal["rollback_backup_id"],
        "matrix_execution_summary": summary,
        "independent_bo_run_count": 0,
        "artifacts": artifacts,
        "governance_receipt_sha256": governance["receipt_sha256"],
        "governance_receipt_file_sha256": file_sha256(governance_path),
        "next_state": report["next_state"],
    }
    completion["completion_sha256"] = canonical_sha256(completion)
    completion_path = output_root / "stage_p_completion.json"
    atomic_write_json(completion_path, completion)
    return _validate_completion(
        completion_path=completion_path,
        proposal=proposal,
        output_root=output_root,
        governance_root=governance_root,
        registry=registry,
    )


def execute_stage_p_proposal(
    *,
    proposal_dir: Path,
    governance_dir: Path,
    output_dir: Path,
    source_root: Path,
    _numerical_runner: StageRNumericalRunner | None = None,
    progress_callback: StagePProgressCallback | None = None,
) -> dict[str, Any]:
    """Execute or resume exactly the 192 new Stage P identities."""

    proposal_root = Path(proposal_dir).resolve()
    (
        proposal,
        source_budget,
        current_matrix,
        current_role_matrix,
    ) = _verify_preflight(
        proposal_root=proposal_root,
        source_root=Path(source_root).resolve(),
    )
    governance_root = Path(governance_dir).resolve()
    governance_budget = _budget_from_payload(read_json(governance_root / "budget_contract.json"))
    if (
        governance_budget.sha256 != source_budget.sha256
        or governance_budget.to_dict() != source_budget.to_dict()
    ):
        raise StagePPlanError("stage_p_governance_budget_mismatch")
    exploration = _exploration_from_payload(
        read_json(governance_root / "exploration_registry.json")
    )
    registry = AttemptRegistry.open(
        governance_root / "attempt_registry.json",
        budget_contract=governance_budget,
        exploration_registry=exploration,
    )
    output_root = Path(output_dir).resolve()
    completion_path = output_root / "stage_p_completion.json"
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
        dict(require_mapping("stage_p_identity", raw)) for raw in proposal["identities"]
    ]
    identities = tuple(attempt_identity_from_item(item) for item in raw_identities)
    for identity in identities:
        registry.register_identity(identity)
    numerical_runner = (
        _run_stage_f_numerical_identity if _numerical_runner is None else _numerical_runner
    )
    spectral_dir = output_root / "spectral_audits"
    result_rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw_identities, start=1):
        row = _execute_stage_f_identity_with_retry(
            registry=registry,
            item=item,
            numerical_runner=numerical_runner,
            spectral_audit_dir=spectral_dir,
            retry_limit=governance_budget.retry_limit,
            progress_callback=progress_callback,
        )
        result_rows.append(merge_identity_result_metadata(item=item, row=row))
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "stage_p_penalty_interaction",
                    "completed": index,
                    "total": len(raw_identities),
                    "identity_sha256": row["identity_sha256"],
                    "cache_hit": row["cache_hit"],
                }
            )
    return _finalize(
        proposal=proposal,
        current_matrix=current_matrix,
        current_role_matrix=current_role_matrix,
        new_rows=result_rows,
        output_root=output_root,
        governance_root=governance_root,
        registry=registry,
    )
