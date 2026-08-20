"""Zero-run proposal and leakage-safe source packaging for fold replay."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .experiment_freeze_utils import runtime_source_identity
from .phase2_experiment_io import atomic_write_json, file_sha256, read_json
from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import AttemptIdentity
from .recovery_fold_replay_contracts import (
    EXPECTED_LOGICAL_SLOT_COUNT,
    FOLD_REPLAY_STAGE,
    FoldReplayError,
    require_hash,
    require_list,
    require_mapping,
    selection_contract_v1,
    validate_scene_panel,
    verify_embedded_hash,
)

_EVALUATION_ROOTS = (
    "ppg_hr.v2.recovery_fold_replay_contracts",
    "ppg_hr.v2.recovery_fold_replay_selection",
    "ppg_hr.v2.recovery_fold_replay_plan",
    "ppg_hr.v2.recovery_fold_replay_execution",
)


def _with_hash(payload: dict[str, Any], field: str) -> dict[str, Any]:
    payload[field] = canonical_sha256(payload)
    return payload


def _validate_pre_fold_gate(
    gate: Mapping[str, Any],
    *,
    final_audit_sha256: str,
    human_decision: Mapping[str, Any] | None,
) -> str:
    gate_sha = verify_embedded_hash(
        gate,
        hash_field="receipt_sha256",
        artifact_name="fold_replay_pre_fold_gate",
    )
    if (
        gate.get("final_interaction_audit_sha256") != final_audit_sha256
        or gate.get("independent_bo_run_count") != 0
        or gate.get("independent_bo_authorized") is not False
    ):
        raise FoldReplayError("fold_replay_pre_fold_gate_binding_mismatch")
    if (
        gate.get("triggered") is False
        and gate.get("status") == "ready_for_fold_replay"
        and gate.get("next_state") == "ready_for_fold_replay"
    ):
        if human_decision is not None:
            raise FoldReplayError("fold_replay_unneeded_human_decision")
        return "pre_fold_gate_not_triggered"
    if human_decision is None:
        raise FoldReplayError("fold_replay_pre_fold_gate_awaiting_human_decision")
    decision_sha = verify_embedded_hash(
        human_decision,
        hash_field="decision_sha256",
        artifact_name="fold_replay_pre_fold_human_decision",
    )
    if (
        gate.get("triggered") is not True
        or gate.get("status") != "awaiting_human_independent_bo_decision"
        or human_decision.get("decision_version") != "lyx_pre_fold_independent_bo_human_decision_v1"
        or human_decision.get("gate_receipt_sha256") != gate_sha
        or human_decision.get("continue_current_non_bo_flow") is not True
        or human_decision.get("run_independent_bo_now") is not False
        or human_decision.get("independent_bo_run_count") != 0
        or not human_decision.get("decided_by")
        or not human_decision.get("decided_at")
    ):
        raise FoldReplayError(f"fold_replay_pre_fold_human_decision_mismatch:{decision_sha}")
    return "human_approved_current_non_bo_flow"


def _validate_budget(budget: Mapping[str, Any]) -> str:
    limits = require_mapping(
        "fold_replay_budget_stage_unique_limits",
        budget.get("stage_unique_limits"),
    )
    kinds = require_mapping(
        "fold_replay_budget_stage_attempt_kinds",
        budget.get("stage_attempt_kinds"),
    )
    if (
        limits.get(FOLD_REPLAY_STAGE) != 12
        or budget.get("supplemental_stage") != FOLD_REPLAY_STAGE
        or kinds.get(FOLD_REPLAY_STAGE) != "formal"
        or int(budget.get("retry_limit", -1)) != 1
    ):
        raise FoldReplayError("fold_replay_budget_contract_mismatch")
    return canonical_sha256(budget)


def build_fold_replay_proposal(
    *,
    final_interaction_audit: Mapping[str, Any],
    pre_fold_gate_receipt: Mapping[str, Any],
    budget_contract: Mapping[str, Any],
    parent_experiment_id: str,
    evaluation_hash: str,
    pre_fold_human_decision: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the exact 12-slot proposal without exposing target performance."""

    if not parent_experiment_id:
        raise FoldReplayError("fold_replay_parent_experiment_id_missing")
    require_hash("fold_replay_evaluation_hash", evaluation_hash)
    audit_sha = verify_embedded_hash(
        final_interaction_audit,
        hash_field="audit_sha256",
        artifact_name="fold_replay_final_interaction_audit",
    )
    if (
        final_interaction_audit.get("audit_version") != "lyx_final_interaction_audit_v1"
        or final_interaction_audit.get("status") != "complete"
        or final_interaction_audit.get("algorithm_level_holdout") is not False
        or final_interaction_audit.get("row_count") != 96
        or final_interaction_audit.get("independent_bo_run_count") != 0
    ):
        raise FoldReplayError("fold_replay_final_interaction_audit_mismatch")
    gate_resolution = _validate_pre_fold_gate(
        pre_fold_gate_receipt,
        final_audit_sha256=audit_sha,
        human_decision=pre_fold_human_decision,
    )
    budget_hash = _validate_budget(budget_contract)
    rows = [
        require_mapping("fold_replay_final_row", raw)
        for raw in require_list(
            "fold_replay_final_rows",
            final_interaction_audit.get("rows"),
        )
    ]
    scene_by_record, profile_ids = validate_scene_panel(rows)
    profile_receipts = require_mapping(
        "fold_replay_profile_receipts",
        final_interaction_audit.get("profile_receipts"),
    )
    if set(profile_receipts) != set(profile_ids):
        raise FoldReplayError("fold_replay_profile_receipt_set_mismatch")
    for profile_id in profile_ids:
        receipt = require_mapping(
            f"fold_replay_profile_receipt:{profile_id}",
            profile_receipts[profile_id],
        )
        verify_embedded_hash(
            receipt,
            hash_field="receipt_sha256",
            artifact_name=(f"fold_replay_profile_receipt:{profile_id}"),
        )
        expected_identities = sorted(
            str(row["identity_sha256"]) for row in rows if row["filter_profile_id"] == profile_id
        )
        profile_rows = [row for row in rows if row["filter_profile_id"] == profile_id]
        static_profiles = {
            (
                str(row.get("filter_profile_sha256")),
                str(row.get("filter_profile_design_role")),
                int(row.get("physical_memory_ms")),
                int(row.get("actual_taps")),
                float(row.get("nominal_mu")),
            )
            for row in profile_rows
        }
        if (
            receipt.get("receipt_version") != "lyx_final_filter_profile_receipt_v1"
            or receipt.get("filter_profile_id") != profile_id
            or receipt.get("record_count") != 12
            or receipt.get("identity_sha256") != expected_identities
            or len(static_profiles) != 1
        ):
            raise FoldReplayError(f"fold_replay_profile_receipt_mismatch:{profile_id}")
    final_recovery_id = str(final_interaction_audit.get("final_recovery_id", ""))
    selected_penalty_id = str(final_interaction_audit.get("selected_penalty_id", ""))
    if (
        not final_recovery_id
        or not selected_penalty_id
        or any(
            receipt.get("final_recovery_id") != final_recovery_id
            or receipt.get("selected_penalty_id") != selected_penalty_id
            for receipt in (
                require_mapping(
                    "fold_replay_profile_receipt",
                    profile_receipts[profile_id],
                )
                for profile_id in profile_ids
            )
        )
        or any(
            row.get("recovery_candidate_id") != final_recovery_id
            or row.get("penalty_candidate_id") != selected_penalty_id
            for row in rows
        )
    ):
        raise FoldReplayError("fold_replay_final_identity_matrix_mismatch")
    expected_identity_by_coordinate: dict[tuple[str, str], str] = {}
    for row in rows:
        require_hash(
            "fold_replay_source_identity_sha256",
            row.get("identity_sha256"),
        )
        try:
            expected = AttemptIdentity(
                solver_hash=str(row["solver_hash"]),
                config_hash=str(row["config_hash"]),
                metric_contract_hash=str(row["metric_contract_hash"]),
                evaluation_hash=str(row["evaluation_hash"]),
                data_sha256=str(row["data_sha256"]),
                record_id=str(row["record_id"]),
                stage=str(row["stage"]),
                attempt_kind=str(row["attempt_kind"]),
                parent_experiment_id=str(row["parent_experiment_id"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise FoldReplayError("fold_replay_source_attempt_identity_incomplete") from error
        execution_material_fields = {
            "config",
            "data_path",
            "reference_path",
            "raw_data_sha256",
            "reference_sha256",
            "method_names",
            "true_rise_applicable",
            "filter_profile_sha256",
            "filter_profile_design_role",
            "physical_memory_ms",
            "actual_taps",
            "nominal_mu",
            "recovery_candidate_sha256",
            "candidate_min_bpm",
            "penalty_candidate_sha256",
        }
        missing_material = sorted(execution_material_fields - set(row))
        if missing_material:
            raise FoldReplayError(
                "fold_replay_source_execution_material_missing:" + ",".join(missing_material)
            )
        if canonical_sha256(row["config"]) != expected.config_hash:
            raise FoldReplayError("fold_replay_source_config_hash_mismatch")
        expected_identity_by_coordinate[(str(row["record_id"]), str(row["filter_profile_id"]))] = (
            expected.sha256
        )
    folds: list[dict[str, Any]] = []
    for scene in sorted(set(scene_by_record.values())):
        scene_records = sorted(
            record_id
            for record_id, record_scene in scene_by_record.items()
            if record_scene == scene
        )
        if len(scene_records) != 3:
            raise FoldReplayError("fold_replay_scene_record_count_mismatch")
        for fold_index, target_record_id in enumerate(scene_records, start=1):
            training_ids = sorted(
                record_id for record_id in scene_records if record_id != target_record_id
            )
            fold_id = f"{scene}-fold-{fold_index}-{target_record_id}"
            folds.append(
                {
                    "fold_id": fold_id,
                    "scene": scene,
                    "fold_index": fold_index,
                    "training_record_ids": training_ids,
                    "audit_target_record_id": target_record_id,
                    "target_expected_identity_sha256_by_profile": {
                        profile_id: expected_identity_by_coordinate[(target_record_id, profile_id)]
                        for profile_id in profile_ids
                    },
                    "training_source_relpaths": {
                        record_id: (f"record_training_sources/{record_id}.json")
                        for record_id in training_ids
                    },
                    "target_identity_source_relpath": (
                        f"target_identity_sources/{target_record_id}.json"
                    ),
                    "target_result_source_relpath_by_profile": {
                        profile_id: (f"target_result_sources/{target_record_id}/{profile_id}.json")
                        for profile_id in profile_ids
                    },
                }
            )
    if len(folds) != EXPECTED_LOGICAL_SLOT_COUNT:
        raise FoldReplayError("fold_replay_fold_count_mismatch")
    contract = selection_contract_v1()
    proposal = {
        "proposal_version": "lyx_fold_replay_execution_proposal_v1",
        "status": "ready_for_execution",
        "parent_experiment_id": parent_experiment_id,
        "evidence_class": "development_replay_audit",
        "algorithm_level_holdout": False,
        "final_interaction_audit_sha256": audit_sha,
        "pre_fold_gate_receipt_sha256": pre_fold_gate_receipt["receipt_sha256"],
        "pre_fold_gate_resolution": gate_resolution,
        "pre_fold_human_decision_sha256": (
            None if pre_fold_human_decision is None else pre_fold_human_decision["decision_sha256"]
        ),
        "selection_contract_sha256": contract["contract_sha256"],
        "budget_contract_hash": budget_hash,
        "evaluation_hash": evaluation_hash,
        "final_recovery_id": final_recovery_id,
        "selected_penalty_id": selected_penalty_id,
        "profile_ids": list(profile_ids),
        "fold_count": len(folds),
        "logical_task_count": len(folds),
        "planned_unique_identity_count": 0,
        "maximum_supplemental_identity_count": 12,
        "planned_cached_result_read_count": len(folds),
        "independent_bo_run_count": 0,
        "candidate_or_threshold_revision_count": 0,
        "folds": folds,
        "next_state": "ready_for_fold_replay_execution",
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal, contract


def _write_internal_sources(
    *,
    staging: Path,
    rows: Sequence[Mapping[str, Any]],
    scene_by_record: Mapping[str, str],
    profile_ids: Sequence[str],
) -> dict[str, str]:
    rows_by_record: dict[str, list[dict[str, Any]]] = {
        record_id: [] for record_id in scene_by_record
    }
    for raw in rows:
        row = dict(raw)
        rows_by_record[str(row["record_id"])].append(row)
    source_hashes: dict[str, str] = {}
    for record_id in sorted(rows_by_record):
        record_rows = sorted(
            rows_by_record[record_id],
            key=lambda row: str(row["filter_profile_id"]),
        )
        training_relpath = f"record_training_sources/{record_id}.json"
        training_path = staging / training_relpath
        training_payload = _with_hash(
            {
                "source_version": "lyx_fold_training_source_v1",
                "record_id": record_id,
                "scene": scene_by_record[record_id],
                "profile_rows": record_rows,
            },
            "source_sha256",
        )
        atomic_write_json(training_path, training_payload)
        source_hashes[training_relpath] = file_sha256(training_path)

        identity_relpath = f"target_identity_sources/{record_id}.json"
        identity_path = staging / identity_relpath
        identity_payload = _with_hash(
            {
                "source_version": "lyx_fold_target_identity_source_v1",
                "sample_id": record_id,
                "record_id": record_id,
            },
            "source_sha256",
        )
        atomic_write_json(identity_path, identity_payload)
        source_hashes[identity_relpath] = file_sha256(identity_path)

        by_profile = {str(row["filter_profile_id"]): row for row in record_rows}
        if set(by_profile) != set(profile_ids):
            raise FoldReplayError(f"fold_replay_record_profile_set_mismatch:{record_id}")
        for profile_id in profile_ids:
            result_relpath = f"target_result_sources/{record_id}/{profile_id}.json"
            result_path = staging / result_relpath
            result_payload = _with_hash(
                {
                    "source_version": "lyx_fold_target_result_source_v1",
                    "record_id": record_id,
                    "scene": scene_by_record[record_id],
                    "selected_row": by_profile[profile_id],
                },
                "source_sha256",
            )
            atomic_write_json(result_path, result_payload)
            source_hashes[result_relpath] = file_sha256(result_path)
    return source_hashes


def _data_role_manifest(
    *,
    proposal: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for raw in proposal["folds"]:
        fold = require_mapping("fold_replay_manifest_fold", raw)
        training_sources = {
            record_id: {
                "path": relpath,
                "sha256": source_hashes[relpath],
                "selection_fields": [
                    "record_id",
                    "scene",
                    "profile_rows",
                ],
            }
            for record_id, relpath in require_mapping(
                "fold_replay_training_source_relpaths",
                fold["training_source_relpaths"],
            ).items()
        }
        target_identity_relpath = str(fold["target_identity_source_relpath"])
        target_results = {
            profile_id: {
                "path": relpath,
                "sha256": source_hashes[relpath],
                "postselection_fields": [
                    "record_id",
                    "scene",
                    "selected_row",
                ],
            }
            for profile_id, relpath in require_mapping(
                "fold_replay_target_result_relpaths",
                fold["target_result_source_relpath_by_profile"],
            ).items()
        }
        folds.append(
            {
                "fold_id": fold["fold_id"],
                "scene": fold["scene"],
                "training_record_ids": list(fold["training_record_ids"]),
                "audit_target_record_id": fold["audit_target_record_id"],
                "training_sources": training_sources,
                "target_preselection_source": {
                    "path": target_identity_relpath,
                    "sha256": source_hashes[target_identity_relpath],
                    "allowed_fields": ["sample_id", "record_id"],
                    "denied_field_classes": [
                        "mae",
                        "spectral",
                        "long_tail",
                        "independent_bo_parameter_summary",
                        "derived_performance",
                    ],
                },
                "target_postselection_sources_by_profile": target_results,
            }
        )
    manifest = {
        "manifest_version": "lyx_fold_data_role_manifest_v1",
        "evidence_class": "development_replay_audit",
        "algorithm_level_holdout": False,
        "proposal_sha256": proposal["proposal_sha256"],
        "fold_count": len(folds),
        "folds": folds,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def propose_fold_replay_execution(
    *,
    final_interaction_audit_path: Path,
    pre_fold_gate_receipt_path: Path,
    budget_contract_path: Path,
    output_dir: Path,
    source_root: Path,
    parent_experiment_id: str,
    pre_fold_human_decision_path: Path | None = None,
) -> dict[str, Any]:
    """Publish an atomic zero-solver proposal for all twelve fold slots."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FoldReplayError(f"fold_replay_output_already_exists:{destination}")
    source_paths = {
        "final_interaction_audit": Path(final_interaction_audit_path).resolve(),
        "pre_fold_gate_receipt": Path(pre_fold_gate_receipt_path).resolve(),
        "budget_contract": Path(budget_contract_path).resolve(),
    }
    if pre_fold_human_decision_path is not None:
        source_paths["pre_fold_human_decision"] = Path(pre_fold_human_decision_path).resolve()
    for name, path in source_paths.items():
        if not path.is_file():
            raise FoldReplayError(f"fold_replay_source_missing:{name}:{path}")
    sources = {name: read_json(path) for name, path in source_paths.items()}
    evaluation = runtime_source_identity(
        Path(source_root).resolve(),
        root_modules=_EVALUATION_ROOTS,
    )
    evaluation = {
        "root_modules": list(_EVALUATION_ROOTS),
        **evaluation,
        "evaluation_hash": evaluation["source_bundle_sha256"],
    }
    proposal, contract = build_fold_replay_proposal(
        final_interaction_audit=sources["final_interaction_audit"],
        pre_fold_gate_receipt=sources["pre_fold_gate_receipt"],
        budget_contract=sources["budget_contract"],
        parent_experiment_id=parent_experiment_id,
        evaluation_hash=evaluation["evaluation_hash"],
        pre_fold_human_decision=sources.get("pre_fold_human_decision"),
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
        raise FoldReplayError("fold_replay_staging_parent_mismatch")
    try:
        staging.mkdir(parents=True)
        rows = [
            require_mapping("fold_replay_final_row", raw)
            for raw in sources["final_interaction_audit"]["rows"]
        ]
        scene_by_record, profile_ids = validate_scene_panel(rows)
        source_hashes = _write_internal_sources(
            staging=staging,
            rows=rows,
            scene_by_record=scene_by_record,
            profile_ids=profile_ids,
        )
        source_index = _with_hash(
            {
                "index_version": "lyx_fold_replay_source_index_v1",
                "source_count": len(source_hashes),
                "sources": dict(sorted(source_hashes.items())),
            },
            "index_sha256",
        )
        manifest = _data_role_manifest(
            proposal=proposal,
            source_hashes=source_hashes,
        )
        atomic_write_json(
            staging / "fold_replay_proposal.json",
            proposal,
        )
        atomic_write_json(
            staging / "fold_selection_contract.json",
            contract,
        )
        atomic_write_json(
            staging / "data_role_manifest.json",
            manifest,
        )
        atomic_write_json(
            staging / "source_index.json",
            source_index,
        )
        atomic_write_json(
            staging / "evaluation_source_identity.json",
            evaluation,
        )
        artifact_names = (
            "fold_replay_proposal.json",
            "fold_selection_contract.json",
            "data_role_manifest.json",
            "source_index.json",
            "evaluation_source_identity.json",
        )
        receipt = {
            "receipt_version": ("lyx_fold_replay_proposal_receipt_v1"),
            "status": "ready_for_execution",
            "proposal_sha256": proposal["proposal_sha256"],
            "logical_task_count": EXPECTED_LOGICAL_SLOT_COUNT,
            "planned_unique_identity_count": 0,
            "planned_cached_result_read_count": (EXPECTED_LOGICAL_SLOT_COUNT),
            "formal_solver_run_count": 0,
            "independent_bo_run_count": 0,
            "artifacts": {name: file_sha256(staging / name) for name in artifact_names},
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_write_json(
            staging / "proposal_receipt.json",
            receipt,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
        return receipt
    except Exception:
        if staging.exists() and staging.parent == destination.parent:
            shutil.rmtree(staging)
        raise
