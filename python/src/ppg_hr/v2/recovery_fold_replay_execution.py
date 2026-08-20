"""Execute the twelve leakage-safe fold-replay slots from frozen results."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .experiment_freeze_utils import runtime_source_identity
from .phase2_experiment_io import atomic_write_json, file_sha256, read_json
from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import (
    AttemptIdentity,
    AttemptRegistry,
    DataRoleManifest,
    ExplorationRegistry,
    FoldReadBarrier,
    RecordSource,
)
from .recovery_fold_replay_contracts import (
    EXPECTED_LOGICAL_SLOT_COUNT,
    FOLD_REPLAY_STAGE,
    TARGET_IDENTITY_SOURCE_FIELDS,
    TRAINING_SOURCE_FIELDS,
    FoldReplayError,
    budget_contract_from_payload,
    require_hash,
    require_list,
    require_mapping,
    selection_contract_v1,
    verify_embedded_hash,
)
from .recovery_fold_replay_plan import build_fold_replay_proposal
from .recovery_fold_replay_selection import (
    audit_selected_target,
    select_fold_profile,
)


def _safe_package_path(package_root: Path, relative: str) -> Path:
    path = (package_root / relative).resolve()
    if not path.is_relative_to(package_root):
        raise FoldReplayError(f"fold_replay_source_outside_package:{relative}")
    return path


def _verify_internal_sources(
    *,
    package_root: Path,
    source_index: Mapping[str, Any],
) -> dict[str, str]:
    verify_embedded_hash(
        source_index,
        hash_field="index_sha256",
        artifact_name="fold_replay_source_index",
    )
    sources = {
        str(relative): require_hash(
            f"fold_replay_source_hash:{relative}",
            value,
        )
        for relative, value in require_mapping(
            "fold_replay_index_sources",
            source_index.get("sources"),
        ).items()
    }
    if source_index.get("source_count") != len(sources) or len(sources) != 120:
        raise FoldReplayError("fold_replay_source_index_count_mismatch")
    for relative, expected_hash in sources.items():
        path = _safe_package_path(package_root, relative)
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise FoldReplayError(f"fold_replay_internal_source_changed:{relative}")
        payload = read_json(path)
        verify_embedded_hash(
            payload,
            hash_field="source_sha256",
            artifact_name=("fold_replay_internal_source:" + relative.replace("/", ":")),
        )
    return sources


def _verify_external_sources(
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    bindings = require_mapping(
        "fold_replay_source_artifacts",
        proposal.get("source_artifacts"),
    )
    required = {
        "final_interaction_audit",
        "pre_fold_gate_receipt",
        "budget_contract",
    }
    binding_names = set(bindings)
    if binding_names != required and binding_names != {
        *required,
        "pre_fold_human_decision",
    }:
        raise FoldReplayError("fold_replay_source_artifact_set_mismatch")
    sources: dict[str, Any] = {}
    for name, raw_binding in bindings.items():
        binding = require_mapping(
            f"fold_replay_source_binding:{name}",
            raw_binding,
        )
        path = Path(str(binding.get("path"))).resolve()
        if not path.is_file() or file_sha256(path) != binding.get("sha256"):
            raise FoldReplayError(f"fold_replay_external_source_changed:{name}")
        sources[name] = read_json(path)
    verify_embedded_hash(
        sources["final_interaction_audit"],
        hash_field="audit_sha256",
        artifact_name="fold_replay_external_final_audit",
    )
    verify_embedded_hash(
        sources["pre_fold_gate_receipt"],
        hash_field="receipt_sha256",
        artifact_name="fold_replay_external_pre_fold_gate",
    )
    if "pre_fold_human_decision" in sources:
        verify_embedded_hash(
            sources["pre_fold_human_decision"],
            hash_field="decision_sha256",
            artifact_name="fold_replay_external_human_decision",
        )
    if (
        sources["final_interaction_audit"].get("audit_sha256")
        != proposal.get("final_interaction_audit_sha256")
        or sources["pre_fold_gate_receipt"].get("receipt_sha256")
        != proposal.get("pre_fold_gate_receipt_sha256")
        or canonical_sha256(sources["budget_contract"]) != proposal.get("budget_contract_hash")
        or (
            "pre_fold_human_decision" in sources
            and sources["pre_fold_human_decision"].get("decision_sha256")
            != proposal.get("pre_fold_human_decision_sha256")
        )
    ):
        raise FoldReplayError("fold_replay_external_source_binding_mismatch")
    return sources


def _verify_preflight(
    *,
    package_root: Path,
    source_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
]:
    required_paths = {
        "proposal": package_root / "fold_replay_proposal.json",
        "receipt": package_root / "proposal_receipt.json",
        "contract": package_root / "fold_selection_contract.json",
        "manifest": package_root / "data_role_manifest.json",
        "source_index": package_root / "source_index.json",
        "evaluation": package_root / "evaluation_source_identity.json",
    }
    if any(not path.is_file() for path in required_paths.values()):
        raise FoldReplayError("fold_replay_proposal_package_incomplete")
    proposal = read_json(required_paths["proposal"])
    proposal_sha = verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="fold_replay_proposal",
    )
    receipt = read_json(required_paths["receipt"])
    verify_embedded_hash(
        receipt,
        hash_field="receipt_sha256",
        artifact_name="fold_replay_proposal_receipt",
    )
    artifacts = require_mapping(
        "fold_replay_proposal_artifacts",
        receipt.get("artifacts"),
    )
    expected_artifacts = {
        "fold_replay_proposal.json",
        "fold_selection_contract.json",
        "data_role_manifest.json",
        "source_index.json",
        "evaluation_source_identity.json",
    }
    if (
        receipt.get("receipt_version") != "lyx_fold_replay_proposal_receipt_v1"
        or receipt.get("status") != "ready_for_execution"
        or receipt.get("proposal_sha256") != proposal_sha
        or receipt.get("logical_task_count") != EXPECTED_LOGICAL_SLOT_COUNT
        or receipt.get("planned_unique_identity_count") != 0
        or receipt.get("formal_solver_run_count") != 0
        or receipt.get("independent_bo_run_count") != 0
        or set(artifacts) != expected_artifacts
        or any(
            file_sha256(package_root / name) != expected_hash
            for name, expected_hash in artifacts.items()
        )
    ):
        raise FoldReplayError("fold_replay_proposal_receipt_mismatch")
    contract = read_json(required_paths["contract"])
    verify_embedded_hash(
        contract,
        hash_field="contract_sha256",
        artifact_name="fold_replay_selection_contract",
    )
    if (
        contract != selection_contract_v1()
        or proposal.get("selection_contract_sha256") != contract["contract_sha256"]
    ):
        raise FoldReplayError("fold_replay_selection_contract_mismatch")
    manifest = read_json(required_paths["manifest"])
    verify_embedded_hash(
        manifest,
        hash_field="manifest_sha256",
        artifact_name="fold_replay_data_role_manifest",
    )
    if (
        manifest.get("proposal_sha256") != proposal_sha
        or manifest.get("fold_count") != EXPECTED_LOGICAL_SLOT_COUNT
        or manifest.get("algorithm_level_holdout") is not False
        or manifest.get("evidence_class") != "development_replay_audit"
    ):
        raise FoldReplayError("fold_replay_data_role_manifest_mismatch")
    evaluation = read_json(required_paths["evaluation"])
    roots = tuple(
        str(value)
        for value in require_list(
            "fold_replay_evaluation_roots",
            evaluation.get("root_modules"),
        )
    )
    current_evaluation = runtime_source_identity(
        source_root,
        root_modules=roots,
    )
    if (
        current_evaluation.get("source_files") != evaluation.get("source_files")
        or current_evaluation.get("source_bundle_sha256") != evaluation.get("source_bundle_sha256")
        or evaluation.get("evaluation_hash") != proposal.get("evaluation_hash")
    ):
        raise FoldReplayError("fold_replay_evaluation_source_changed")
    internal_sources = _verify_internal_sources(
        package_root=package_root,
        source_index=read_json(required_paths["source_index"]),
    )
    external_sources = _verify_external_sources(proposal)
    expected_proposal, _ = build_fold_replay_proposal(
        final_interaction_audit=external_sources["final_interaction_audit"],
        pre_fold_gate_receipt=external_sources["pre_fold_gate_receipt"],
        pre_fold_human_decision=external_sources.get("pre_fold_human_decision"),
        budget_contract=external_sources["budget_contract"],
        parent_experiment_id=str(proposal["parent_experiment_id"]),
        evaluation_hash=str(proposal["evaluation_hash"]),
    )
    actual_body = {
        key: value
        for key, value in proposal.items()
        if key not in {"proposal_sha256", "source_artifacts"}
    }
    expected_body = {
        key: value for key, value in expected_proposal.items() if key != "proposal_sha256"
    }
    if actual_body != expected_body:
        raise FoldReplayError("fold_replay_proposal_contract_mismatch")
    expected_internal_paths = {
        *(
            str(relative)
            for fold in proposal["folds"]
            for relative in require_mapping(
                "fold_replay_expected_training_sources",
                fold["training_source_relpaths"],
            ).values()
        ),
        *(str(fold["target_identity_source_relpath"]) for fold in proposal["folds"]),
        *(
            str(relative)
            for fold in proposal["folds"]
            for relative in require_mapping(
                "fold_replay_expected_target_sources",
                fold["target_result_source_relpath_by_profile"],
            ).values()
        ),
    }
    if set(internal_sources) != expected_internal_paths:
        raise FoldReplayError("fold_replay_internal_source_set_mismatch")
    return proposal, manifest, internal_sources, external_sources


def _manifest_folds_by_id(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    folds = {
        str(fold.get("fold_id", "")): fold
        for fold in (
            require_mapping("fold_replay_manifest_fold", raw)
            for raw in require_list(
                "fold_replay_manifest_folds",
                manifest.get("folds"),
            )
        )
    }
    if len(folds) != EXPECTED_LOGICAL_SLOT_COUNT or "" in folds:
        raise FoldReplayError("fold_replay_manifest_fold_set_mismatch")
    return folds


def _record_source(
    *,
    package_root: Path,
    raw: object,
    expected_hashes: Mapping[str, str],
) -> RecordSource:
    binding = require_mapping(
        "fold_replay_record_source_binding",
        raw,
    )
    relative = str(binding.get("path", ""))
    expected_hash = expected_hashes.get(relative)
    if not relative or binding.get("sha256") != expected_hash:
        raise FoldReplayError("fold_replay_record_source_binding_mismatch")
    return RecordSource(
        path=_safe_package_path(package_root, relative),
        sha256=str(expected_hash),
    )


def _target_result_payload(
    *,
    package_root: Path,
    source_binding: Mapping[str, Any],
    expected_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative = str(source_binding.get("path", ""))
    expected_hash = expected_hashes.get(relative)
    if not relative or source_binding.get("sha256") != expected_hash:
        raise FoldReplayError("fold_replay_target_source_binding_mismatch")
    path = _safe_package_path(package_root, relative)
    if file_sha256(path) != expected_hash:
        raise FoldReplayError("fold_replay_target_source_changed")
    payload = read_json(path)
    verify_embedded_hash(
        payload,
        hash_field="source_sha256",
        artifact_name="fold_replay_target_result_source",
    )
    return payload, {
        "phase": "post_selection_target_audit",
        "path": str(path),
        "path_sha256": expected_hash,
        "fields": ["record_id", "scene", "selected_row"],
        "selected_profile_result_count": 1,
    }


def _supplement_identity(
    *,
    proposal: Mapping[str, Any],
    target_result: Mapping[str, Any],
) -> dict[str, Any]:
    row = require_mapping(
        "fold_replay_supplement_source_row",
        target_result.get("selected_row"),
    )
    identity = AttemptIdentity(
        solver_hash=require_hash(
            "fold_replay_supplement_solver_hash",
            row.get("solver_hash"),
        ),
        config_hash=require_hash(
            "fold_replay_supplement_config_hash",
            row.get("config_hash"),
        ),
        metric_contract_hash=require_hash(
            "fold_replay_supplement_metric_contract_hash",
            row.get("metric_contract_hash"),
        ),
        evaluation_hash=require_hash(
            "fold_replay_supplement_evaluation_hash",
            proposal.get("evaluation_hash"),
        ),
        data_sha256=require_hash(
            "fold_replay_supplement_data_sha256",
            row.get("data_sha256"),
        ),
        record_id=str(row["record_id"]),
        stage=FOLD_REPLAY_STAGE,
        attempt_kind="formal",
        parent_experiment_id=str(proposal["parent_experiment_id"]),
    )
    material_fields = (
        "config",
        "data_path",
        "reference_path",
        "raw_data_sha256",
        "reference_sha256",
        "method_names",
        "scene",
        "true_rise_applicable",
        "filter_profile_id",
        "filter_profile_sha256",
        "filter_profile_design_role",
        "physical_memory_ms",
        "actual_taps",
        "nominal_mu",
        "recovery_candidate_id",
        "recovery_candidate_sha256",
        "candidate_min_bpm",
        "penalty_candidate_id",
        "penalty_candidate_sha256",
    )
    missing = [name for name in material_fields if name not in row]
    if missing:
        raise FoldReplayError(
            "fold_replay_supplement_execution_material_missing:" + ",".join(missing)
        )
    if canonical_sha256(row["config"]) != identity.config_hash:
        raise FoldReplayError("fold_replay_supplement_config_hash_mismatch")
    execution_item = {
        **identity.to_dict(),
        **{name: row[name] for name in material_fields},
        "matrix_role": "fold_replay_supplement",
    }
    return {
        **identity.to_dict(),
        "reason": "selected_target_numerical_identity_mismatch",
        "source_identity_sha256": row.get("identity_sha256"),
        "filter_profile_id": row.get("filter_profile_id"),
        "recovery_candidate_id": row.get("recovery_candidate_id"),
        "penalty_candidate_id": row.get("penalty_candidate_id"),
        "candidate_or_threshold_revision_allowed": False,
        "execution_item": execution_item,
    }


def _open_registry_for_supplements(
    *,
    governance_root: Path,
    source_budget_payload: Mapping[str, Any],
) -> AttemptRegistry:
    source_budget = budget_contract_from_payload(source_budget_payload)
    governance_budget = budget_contract_from_payload(
        read_json(governance_root / "budget_contract.json")
    )
    if (
        source_budget.sha256 != governance_budget.sha256
        or source_budget.to_dict() != governance_budget.to_dict()
    ):
        raise FoldReplayError("fold_replay_governance_budget_mismatch")
    exploration_payload = read_json(governance_root / "exploration_registry.json")
    exploration = ExplorationRegistry(
        registry_version=str(exploration_payload["registry_version"]),
        unique_budget=int(exploration_payload["unique_budget"]),
        allowed_identity_sha256=tuple(
            str(value)
            for value in require_list(
                "fold_replay_exploration_identities",
                exploration_payload.get("allowed_identity_sha256"),
            )
        ),
    )
    return AttemptRegistry.open(
        governance_root / "attempt_registry.json",
        budget_contract=governance_budget,
        exploration_registry=exploration,
    )


def _validate_completion(
    *,
    output_root: Path,
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    completion_path = output_root / "fold_replay_completion.json"
    completion = read_json(completion_path)
    verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="fold_replay_completion",
    )
    artifact_index_path = output_root / "artifact_index.json"
    status = completion.get("status")
    supplemental_count = completion.get("supplemental_identity_count")
    passed_count = completion.get("passed_slot_count")
    failed_count = completion.get("failed_slot_count")
    expected_next_state = (
        "ready_for_post_fold_independent_bo_gate"
        if status == "complete"
        else "awaiting_fold_replay_supplement_execution"
    )
    if (
        completion.get("completion_version") != "lyx_fold_replay_completion_v1"
        or status
        not in {
            "complete",
            "awaiting_fold_replay_supplement_execution",
        }
        or completion.get("proposal_sha256") != proposal.get("proposal_sha256")
        or completion.get("logical_slot_count") != EXPECTED_LOGICAL_SLOT_COUNT
        or completion.get("denominator_slot_count") != EXPECTED_LOGICAL_SLOT_COUNT
        or completion.get("independent_bo_run_count") != 0
        or completion.get("candidate_or_threshold_revision_count") != 0
        or completion.get("initial_planned_unique_identity_count") != 0
        or not isinstance(supplemental_count, int)
        or isinstance(supplemental_count, bool)
        or supplemental_count < 0
        or supplemental_count > 12
        or completion.get("planned_unique_identity_count") != supplemental_count
        or completion.get("registered_unique_identity_count") != supplemental_count
        or completion.get("actual_unique_run_count") != 0
        or (status == "complete" and supplemental_count != 0)
        or (status == "awaiting_fold_replay_supplement_execution" and supplemental_count == 0)
        or completion.get("formal_solver_run_count") != 0
        or completion.get("failed_attempt_count") != 0
        or completion.get("retry_count") != 0
        or completion.get("next_state") != expected_next_state
        or not isinstance(passed_count, int)
        or isinstance(passed_count, bool)
        or not isinstance(failed_count, int)
        or isinstance(failed_count, bool)
        or passed_count < 0
        or failed_count < 0
        or passed_count + failed_count != EXPECTED_LOGICAL_SLOT_COUNT
        or not artifact_index_path.is_file()
        or file_sha256(artifact_index_path) != completion.get("artifact_index_file_sha256")
    ):
        raise FoldReplayError("fold_replay_completion_contract_mismatch")
    artifact_index = read_json(artifact_index_path)
    verify_embedded_hash(
        artifact_index,
        hash_field="index_sha256",
        artifact_name="fold_replay_artifact_index",
    )
    artifacts = require_mapping(
        "fold_replay_completion_artifacts",
        artifact_index.get("artifacts"),
    )
    if artifact_index.get("artifact_count") != len(artifacts) or artifact_index.get(
        "index_sha256"
    ) != completion.get("artifact_index_sha256"):
        raise FoldReplayError("fold_replay_artifact_index_contract_mismatch")
    for relative, expected_hash in artifacts.items():
        path = _safe_package_path(output_root, str(relative))
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise FoldReplayError(f"fold_replay_completion_artifact_changed:{relative}")
    report = read_json(output_root / "fold_replay_report.json")
    report_sha = verify_embedded_hash(
        report,
        hash_field="report_sha256",
        artifact_name="fold_replay_report",
    )
    selection = read_json(output_root / "fold_selection_receipt.json")
    selection_sha = verify_embedded_hash(
        selection,
        hash_field="receipt_sha256",
        artifact_name="fold_replay_selection_aggregate",
    )
    if (
        report_sha != completion.get("report_sha256")
        or report.get("status") != status
        or report.get("logical_slot_count") != EXPECTED_LOGICAL_SLOT_COUNT
        or report.get("denominator_slot_count") != EXPECTED_LOGICAL_SLOT_COUNT
        or report.get("supplemental_identity_count") != supplemental_count
        or report.get("planned_unique_identity_count") != supplemental_count
        or report.get("registered_unique_identity_count") != supplemental_count
        or report.get("actual_unique_run_count") != 0
        or report.get("formal_solver_run_count") != 0
        or report.get("next_state") != expected_next_state
        or report.get("passed_slot_count") != passed_count
        or report.get("failed_slot_count") != failed_count
        or selection_sha != report.get("fold_selection_receipt_sha256")
        or selection.get("logical_slot_count") != EXPECTED_LOGICAL_SLOT_COUNT
        or selection.get("denominator_slot_count") != EXPECTED_LOGICAL_SLOT_COUNT
        or selection.get("folds") != report.get("folds")
        or len(require_list("fold_replay_report_folds", report.get("folds")))
        != EXPECTED_LOGICAL_SLOT_COUNT
    ):
        raise FoldReplayError("fold_replay_report_completion_mismatch")
    for raw_summary in report["folds"]:
        summary = require_mapping(
            "fold_replay_completion_fold_summary",
            raw_summary,
        )
        selection_receipt = read_json(
            _safe_package_path(
                output_root,
                str(summary["selection_receipt"]),
            )
        )
        fold_selection_sha = verify_embedded_hash(
            selection_receipt,
            hash_field="selection_sha256",
            artifact_name="fold_replay_completed_selection",
        )
        barrier_receipt = read_json(
            _safe_package_path(
                output_root,
                str(summary["read_barrier_receipt"]),
            )
        )
        verify_embedded_hash(
            barrier_receipt,
            hash_field="receipt_sha256",
            artifact_name="fold_replay_completed_read_barrier",
        )
        target_receipt = read_json(
            _safe_package_path(
                output_root,
                str(summary["target_audit_receipt"]),
            )
        )
        verify_embedded_hash(
            target_receipt,
            hash_field="receipt_sha256",
            artifact_name="fold_replay_completed_target_audit",
        )
        if (
            fold_selection_sha != summary.get("selection_sha256")
            or barrier_receipt.get("fold_id") != summary.get("fold_id")
            or target_receipt.get("fold_id") != summary.get("fold_id")
            or barrier_receipt.get("selection_sha256") != fold_selection_sha
            or target_receipt.get("selection_sha256") != fold_selection_sha
            or target_receipt.get("audit_pass") != summary.get("audit_pass")
            or target_receipt.get("failure_reasons") != summary.get("failure_reasons")
        ):
            raise FoldReplayError("fold_replay_completed_fold_receipt_mismatch")
    return completion


def execute_fold_replay_proposal(
    *,
    proposal_dir: Path,
    governance_dir: Path,
    output_dir: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Execute all 12 logical slots without reading target performance early."""

    package_root = Path(proposal_dir).resolve()
    proposal, manifest, internal_hashes, external_sources = _verify_preflight(
        package_root=package_root,
        source_root=Path(source_root).resolve(),
    )
    output_root = Path(output_dir).resolve()
    completion_path = output_root / "fold_replay_completion.json"
    if completion_path.is_file():
        return _validate_completion(
            output_root=output_root,
            proposal=proposal,
        )
    if output_root.exists():
        raise FoldReplayError(f"fold_replay_incomplete_output_exists:{output_root}")
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    if staging.parent != output_root.parent:
        raise FoldReplayError("fold_replay_output_staging_mismatch")
    manifest_folds = _manifest_folds_by_id(manifest)
    fold_summaries: list[dict[str, Any]] = []
    supplement_identities: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    try:
        staging.mkdir(parents=True)
        for raw_fold in proposal["folds"]:
            fold = require_mapping(
                "fold_replay_proposal_fold",
                raw_fold,
            )
            fold_id = str(fold["fold_id"])
            manifest_fold = manifest_folds.get(fold_id)
            if manifest_fold is None:
                raise FoldReplayError(f"fold_replay_manifest_fold_missing:{fold_id}")
            training_ids = tuple(
                str(value)
                for value in require_list(
                    "fold_replay_training_record_ids",
                    fold.get("training_record_ids"),
                )
            )
            target_id = str(fold["audit_target_record_id"])
            training_bindings = require_mapping(
                "fold_replay_manifest_training_sources",
                manifest_fold.get("training_sources"),
            )
            target_identity_binding = require_mapping(
                "fold_replay_target_identity_binding",
                manifest_fold.get("target_preselection_source"),
            )
            postselection_bindings = require_mapping(
                "fold_replay_target_postselection_sources",
                manifest_fold.get("target_postselection_sources_by_profile"),
            )
            if (
                manifest_fold.get("scene") != fold.get("scene")
                or manifest_fold.get("training_record_ids") != list(training_ids)
                or manifest_fold.get("audit_target_record_id") != target_id
                or set(training_bindings) != set(training_ids)
                or set(postselection_bindings) != set(proposal["profile_ids"])
            ):
                raise FoldReplayError(f"fold_replay_manifest_role_mismatch:{fold_id}")
            role_manifest = DataRoleManifest(
                fold_id=fold_id,
                training_record_ids=training_ids,
                audit_target_record_id=target_id,
                record_sources={
                    **{
                        record_id: _record_source(
                            package_root=package_root,
                            raw=training_bindings[record_id],
                            expected_hashes=internal_hashes,
                        )
                        for record_id in training_ids
                    },
                    target_id: _record_source(
                        package_root=package_root,
                        raw=target_identity_binding,
                        expected_hashes=internal_hashes,
                    ),
                },
            )
            barrier = FoldReadBarrier(role_manifest)
            training_payloads = [
                barrier.read_json_fields(
                    record_id=record_id,
                    fields=TRAINING_SOURCE_FIELDS,
                )
                for record_id in training_ids
            ]
            target_identity = barrier.read_json_fields(
                record_id=target_id,
                fields=TARGET_IDENTITY_SOURCE_FIELDS,
            )
            if (
                target_identity.get("record_id") != target_id
                or target_identity.get("sample_id") != target_id
            ):
                raise FoldReplayError("fold_replay_target_identity_source_mismatch")
            selection = select_fold_profile(
                fold_id=fold_id,
                scene=str(fold["scene"]),
                training_record_payloads=training_payloads,
                audit_target_record_id=target_id,
                profile_ids=tuple(str(value) for value in proposal["profile_ids"]),
            )
            fold_dir = staging / "folds" / fold_id
            selection_path = fold_dir / "fold_selection_receipt.json"
            atomic_write_json(selection_path, selection)
            frozen_selection = read_json(selection_path)
            verify_embedded_hash(
                frozen_selection,
                hash_field="selection_sha256",
                artifact_name=(f"fold_replay_selection_receipt:{fold_id}"),
            )
            barrier_receipt = barrier.receipt()
            barrier_receipt["selection_sha256"] = selection["selection_sha256"]
            barrier_receipt["target_performance_read_count"] = 0
            barrier_receipt["receipt_sha256"] = canonical_sha256(barrier_receipt)
            barrier_path = fold_dir / "read_barrier_receipt.json"
            atomic_write_json(barrier_path, barrier_receipt)

            target_access: dict[str, Any] | None = None
            if selection["status"] == "no_safe_shared_candidate":
                target_audit = {
                    "receipt_version": ("lyx_fold_target_audit_receipt_v1"),
                    "fold_id": fold_id,
                    "selection_sha256": selection["selection_sha256"],
                    "status": "failed",
                    "audit_pass": False,
                    "failure_reasons": ["no_safe_shared_candidate"],
                    "target_performance_read_count": 0,
                    "target_access": None,
                    "supplemental_identity_sha256": None,
                }
            else:
                profile_id = str(selection["selected_filter_profile_id"])
                result_payload, target_access = _target_result_payload(
                    package_root=package_root,
                    source_binding=require_mapping(
                        "fold_replay_selected_target_source",
                        postselection_bindings[profile_id],
                    ),
                    expected_hashes=internal_hashes,
                )
                expected_identity = require_hash(
                    "fold_replay_expected_target_identity",
                    require_mapping(
                        "fold_replay_expected_identity_by_profile",
                        fold["target_expected_identity_sha256_by_profile"],
                    )[profile_id],
                )
                audit = audit_selected_target(
                    selection_receipt=frozen_selection,
                    target_result_payload=result_payload,
                    expected_identity_sha256=expected_identity,
                )
                supplemental_identity = None
                if audit["status"] == "identity_mismatch_requires_supplement":
                    supplemental_identity = _supplement_identity(
                        proposal=proposal,
                        target_result=result_payload,
                    )
                    supplement_identities.append(supplemental_identity)
                target_audit = {
                    "receipt_version": ("lyx_fold_target_audit_receipt_v1"),
                    "fold_id": fold_id,
                    "selection_sha256": selection["selection_sha256"],
                    **audit,
                    "target_performance_read_count": 1,
                    "target_access": target_access,
                    "supplemental_identity_sha256": (
                        None
                        if supplemental_identity is None
                        else supplemental_identity["identity_sha256"]
                    ),
                }
            target_audit["receipt_sha256"] = canonical_sha256(target_audit)
            target_audit_path = fold_dir / "target_audit_receipt.json"
            atomic_write_json(target_audit_path, target_audit)
            fold_summaries.append(
                {
                    "fold_id": fold_id,
                    "scene": fold["scene"],
                    "training_record_ids": list(training_ids),
                    "audit_target_record_id": target_id,
                    "selection_status": selection["status"],
                    "selected_filter_profile_id": selection["selected_filter_profile_id"],
                    "selection_sha256": selection["selection_sha256"],
                    "target_audit_status": target_audit["status"],
                    "audit_pass": target_audit["audit_pass"],
                    "failure_reasons": target_audit["failure_reasons"],
                    "target_performance_read_count": target_audit["target_performance_read_count"],
                    "selection_receipt": str(
                        Path("folds") / fold_id / "fold_selection_receipt.json"
                    ),
                    "read_barrier_receipt": str(
                        Path("folds") / fold_id / "read_barrier_receipt.json"
                    ),
                    "target_audit_receipt": str(
                        Path("folds") / fold_id / "target_audit_receipt.json"
                    ),
                }
            )
        if len(supplement_identities) > 12:
            raise FoldReplayError("fold_replay_supplement_budget_exceeded")
        registry_snapshot = None
        if supplement_identities:
            registry = _open_registry_for_supplements(
                governance_root=Path(governance_dir).resolve(),
                source_budget_payload=external_sources["budget_contract"],
            )
            identities = tuple(
                AttemptIdentity(
                    **{
                        name: item[name]
                        for name in (
                            "solver_hash",
                            "config_hash",
                            "metric_contract_hash",
                            "evaluation_hash",
                            "data_sha256",
                            "record_id",
                            "stage",
                            "attempt_kind",
                            "parent_experiment_id",
                        )
                    }
                )
                for item in supplement_identities
            )
            for identity in identities:
                registry.register_identity(identity)
            registry_payload = read_json(Path(governance_dir).resolve() / "attempt_registry.json")
            registry_entries = require_mapping(
                "fold_replay_registered_entries",
                registry_payload.get("entries"),
            )
            registry_snapshot = {
                "snapshot_version": ("lyx_fold_replay_supplement_registration_v1"),
                "status": ("awaiting_fold_replay_supplement_execution"),
                "budget_contract_sha256": registry_payload["budget_contract_sha256"],
                "identity_count": len(identities),
                "identities": [
                    {
                        "identity_sha256": identity.sha256,
                        "identity": identity.to_dict(),
                        "registry_status": require_mapping(
                            "fold_replay_registered_entry",
                            registry_entries[identity.sha256],
                        )["status"],
                    }
                    for identity in identities
                ],
            }
            registry_snapshot["snapshot_sha256"] = canonical_sha256(registry_snapshot)
            atomic_write_json(
                staging / "fold_replay_supplement_registry_snapshot.json",
                registry_snapshot,
            )
        selection_aggregate = {
            "receipt_version": ("lyx_fold_selection_aggregate_receipt_v1"),
            "proposal_sha256": proposal["proposal_sha256"],
            "evidence_class": "development_replay_audit",
            "algorithm_level_holdout": False,
            "logical_slot_count": len(fold_summaries),
            "denominator_slot_count": len(fold_summaries),
            "folds": fold_summaries,
            "candidate_or_threshold_revision_count": 0,
        }
        selection_aggregate["receipt_sha256"] = canonical_sha256(selection_aggregate)
        atomic_write_json(
            staging / "fold_selection_receipt.json",
            selection_aggregate,
        )
        passed_count = sum(summary["audit_pass"] is True for summary in fold_summaries)
        selected_count = sum(
            summary["selection_status"] == "selected" for summary in fold_summaries
        )
        reused_result_count = selected_count - len(supplement_identities)
        report_status = (
            "awaiting_fold_replay_supplement_execution" if supplement_identities else "complete"
        )
        report = {
            "report_version": "lyx_fold_replay_report_v1",
            "status": report_status,
            "proposal_sha256": proposal["proposal_sha256"],
            "fold_selection_receipt_sha256": (selection_aggregate["receipt_sha256"]),
            "evidence_class": "development_replay_audit",
            "algorithm_level_holdout": False,
            "logical_slot_count": len(fold_summaries),
            "denominator_slot_count": len(fold_summaries),
            "passed_slot_count": passed_count,
            "failed_slot_count": len(fold_summaries) - passed_count,
            "no_safe_shared_candidate_count": sum(
                summary["selection_status"] == "no_safe_shared_candidate"
                for summary in fold_summaries
            ),
            "target_result_reuse_count": reused_result_count,
            "initial_planned_unique_identity_count": 0,
            "planned_unique_identity_count": len(supplement_identities),
            "registered_unique_identity_count": len(supplement_identities),
            "actual_unique_run_count": 0,
            "supplemental_identity_count": len(supplement_identities),
            "formal_solver_run_count": 0,
            "cache_hit_count": reused_result_count,
            "independent_bo_run_count": 0,
            "candidate_or_threshold_revision_count": 0,
            "folds": fold_summaries,
            "supplemental_identities": supplement_identities,
            "supplement_registry_snapshot_sha256": (
                None if registry_snapshot is None else registry_snapshot["snapshot_sha256"]
            ),
            "next_state": (
                "ready_for_post_fold_independent_bo_gate"
                if not supplement_identities
                else "awaiting_fold_replay_supplement_execution"
            ),
        }
        report["report_sha256"] = canonical_sha256(report)
        atomic_write_json(
            staging / "fold_replay_report.json",
            report,
        )
        for path in sorted(staging.rglob("*.json")):
            relative = str(path.relative_to(staging)).replace(
                "\\",
                "/",
            )
            artifact_hashes[relative] = file_sha256(path)
        artifact_index = {
            "index_version": "lyx_fold_replay_artifact_index_v1",
            "artifact_count": len(artifact_hashes),
            "artifacts": artifact_hashes,
        }
        artifact_index["index_sha256"] = canonical_sha256(artifact_index)
        atomic_write_json(
            staging / "artifact_index.json",
            artifact_index,
        )
        completion = {
            "completion_version": "lyx_fold_replay_completion_v1",
            "status": report_status,
            "proposal_sha256": proposal["proposal_sha256"],
            "report_sha256": report["report_sha256"],
            "logical_slot_count": len(fold_summaries),
            "denominator_slot_count": len(fold_summaries),
            "passed_slot_count": passed_count,
            "failed_slot_count": len(fold_summaries) - passed_count,
            "initial_planned_unique_identity_count": 0,
            "planned_unique_identity_count": len(supplement_identities),
            "registered_unique_identity_count": len(supplement_identities),
            "actual_unique_run_count": 0,
            "supplemental_identity_count": len(supplement_identities),
            "formal_solver_run_count": 0,
            "cache_hit_count": reused_result_count,
            "failed_attempt_count": 0,
            "retry_count": 0,
            "independent_bo_run_count": 0,
            "candidate_or_threshold_revision_count": 0,
            "artifact_index_sha256": artifact_index["index_sha256"],
            "artifact_index_file_sha256": file_sha256(staging / "artifact_index.json"),
            "next_state": report["next_state"],
        }
        completion["completion_sha256"] = canonical_sha256(completion)
        atomic_write_json(
            staging / "fold_replay_completion.json",
            completion,
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_root)
        return _validate_completion(
            output_root=output_root,
            proposal=proposal,
        )
    except Exception:
        if staging.exists() and staging.parent == output_root.parent:
            shutil.rmtree(staging)
        raise
