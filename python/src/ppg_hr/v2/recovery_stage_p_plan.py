"""Zero-run planning for the LYX Stage P penalty interaction matrix."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .experiment_freeze_utils import runtime_source_identity
from .phase2_experiment_io import atomic_write_json, file_sha256, read_json
from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import AttemptIdentity
from .recovery_stage_p_contracts import (
    EXPECTED_LOGICAL_RESULT_COUNT,
    EXPECTED_NEW_IDENTITY_COUNT,
    EXPECTED_PENALTY_IDS,
    EXPECTED_REUSED_RESULT_COUNT,
    EXPECTED_SELECTION_RANKING_KEY,
    PENALTY_INTERACTION_STAGE,
    StagePPlanError,
    require_hash,
    require_list,
    require_mapping,
    verify_embedded_hash,
)
from .recovery_stage_r_experiment import stage_r_metric_contract_v1


def _penalties(
    registry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    verify_embedded_hash(
        registry,
        hash_field="registry_sha256",
        artifact_name="stage_p_penalty_registry",
    )
    candidates = [
        dict(require_mapping("stage_p_penalty_candidate", raw))
        for raw in require_list(
            "stage_p_penalty_candidates",
            registry.get("candidates"),
        )
    ]
    by_id = {str(candidate.get("penalty_id")): candidate for candidate in candidates}
    control_id = str(registry.get("control_penalty_id"))
    if (
        registry.get("penalty_count", registry.get("candidate_count")) != 3
        or registry.get("new_penalty_count", 2) != 2
        or registry.get("no_fourth_strategy_after_freeze", True) is not True
        or set(by_id) != EXPECTED_PENALTY_IDS
        or control_id != "current_soft_penalty_control_v1"
        or len(candidates) != len(by_id)
        or any(
            require_hash(
                f"stage_p_penalty_candidate_sha256:{penalty_id}",
                candidate.get("candidate_sha256"),
            )
            != canonical_sha256(
                {key: value for key, value in candidate.items() if key != "candidate_sha256"}
            )
            for penalty_id, candidate in by_id.items()
        )
        or registry.get(
            "selection_ranking_key",
            EXPECTED_SELECTION_RANKING_KEY,
        )
        != EXPECTED_SELECTION_RANKING_KEY
    ):
        raise StagePPlanError("stage_p_penalty_registry_mismatch")
    return by_id[control_id], by_id


def _stage_f_sources(
    *,
    stage_f_proposal: Mapping[str, Any],
    stage_f_completion: Mapping[str, Any],
    stage_f_profile_matrix: Mapping[str, Any],
    stage_f_current_role_matrix: Mapping[str, Any],
    control_penalty_id: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    proposal_sha = verify_embedded_hash(
        stage_f_proposal,
        hash_field="proposal_sha256",
        artifact_name="stage_f_proposal",
    )
    verify_embedded_hash(
        stage_f_completion,
        hash_field="completion_sha256",
        artifact_name="stage_f_completion",
    )
    verify_embedded_hash(
        stage_f_profile_matrix,
        hash_field="matrix_sha256",
        artifact_name="stage_f_profile_matrix",
    )
    verify_embedded_hash(
        stage_f_current_role_matrix,
        hash_field="matrix_sha256",
        artifact_name="stage_f_current_role_matrix",
    )
    if (
        stage_f_proposal.get("proposal_version") != "lyx_stage_f_execution_proposal_v1"
        or stage_f_completion.get("completion_version") != "lyx_stage_f_completion_v1"
        or stage_f_completion.get("status") != "complete"
        or stage_f_completion.get("proposal_sha256") != proposal_sha
        or stage_f_completion.get("next_state") != "ready_for_penalty_interaction_completion"
        or stage_f_completion.get("independent_bo_run_count") != 0
        or stage_f_profile_matrix.get("matrix_version") != "lyx_stage_f_profile_matrix_v1"
        or stage_f_profile_matrix.get("matrix_role") != "provisional_recovery"
        or stage_f_profile_matrix.get("row_count") != 96
        or stage_f_current_role_matrix.get("matrix_version") != "lyx_stage_f_current_role_matrix_v1"
        or stage_f_current_role_matrix.get("matrix_role") != "same_role_current_control"
        or stage_f_current_role_matrix.get("row_count") != 96
    ):
        raise StagePPlanError("stage_p_stage_f_source_mismatch")
    identities = {
        str(item["identity_sha256"]): dict(item)
        for item in (
            require_mapping("stage_p_stage_f_identity", raw)
            for raw in require_list(
                "stage_p_stage_f_identities",
                stage_f_proposal.get("identities"),
            )
        )
        if item.get("matrix_role") == "provisional_recovery"
    }
    rows = [
        dict(require_mapping("stage_p_stage_f_matrix_row", raw))
        for raw in require_list(
            "stage_p_stage_f_matrix_rows",
            stage_f_profile_matrix.get("rows"),
        )
    ]
    by_coordinate: dict[tuple[str, str], dict[str, Any]] = {}
    templates: list[dict[str, Any]] = []
    for row in rows:
        coordinate = (
            str(row.get("filter_profile_id")),
            str(row.get("record_id")),
        )
        identity_hash = str(row.get("identity_sha256"))
        template = identities.get(identity_hash)
        if (
            coordinate in by_coordinate
            or template is None
            or row.get("penalty_candidate_id") != control_penalty_id
            or template.get("penalty_candidate_id") != control_penalty_id
            or template.get("stage") != PENALTY_INTERACTION_STAGE
        ):
            raise StagePPlanError("stage_p_stage_f_coordinate_mismatch")
        by_coordinate[coordinate] = row
        templates.append(template)
    if len(by_coordinate) != EXPECTED_REUSED_RESULT_COUNT:
        raise StagePPlanError("stage_p_stage_f_matrix_incomplete")
    current_role_coordinates = {
        (
            str(row.get("filter_profile_id")),
            str(row.get("record_id")),
        )
        for row in (
            require_mapping("stage_p_stage_f_current_role_row", raw)
            for raw in require_list(
                "stage_p_stage_f_current_role_rows",
                stage_f_current_role_matrix.get("rows"),
            )
        )
        if row.get("penalty_candidate_id") == control_penalty_id
    }
    if current_role_coordinates != set(by_coordinate):
        raise StagePPlanError("stage_p_stage_f_current_role_incomplete")
    return templates, by_coordinate


def _new_penalty_identity(
    *,
    template: Mapping[str, Any],
    penalty: Mapping[str, Any],
    solver_hash: str,
    metric_contract_hash: str,
    evaluation_hash: str,
    parent_experiment_id: str,
) -> dict[str, Any]:
    config = deepcopy(dict(require_mapping("stage_p_template_config", template.get("config"))))
    parameters = dict(
        require_mapping(
            "stage_p_template_parameters",
            config.get("parameters"),
        )
    )
    parameters["penalty_candidate_id"] = penalty["penalty_id"]
    config["parameters"] = parameters
    attempt = AttemptIdentity(
        solver_hash=solver_hash,
        config_hash=canonical_sha256(config),
        metric_contract_hash=metric_contract_hash,
        evaluation_hash=evaluation_hash,
        data_sha256=str(template["data_sha256"]),
        record_id=str(template["record_id"]),
        stage=PENALTY_INTERACTION_STAGE,
        attempt_kind="formal",
        parent_experiment_id=parent_experiment_id,
    )
    ignored = {
        "solver_hash",
        "config_hash",
        "metric_contract_hash",
        "evaluation_hash",
        "data_sha256",
        "record_id",
        "stage",
        "attempt_kind",
        "parent_experiment_id",
        "identity_sha256",
        "cache_identity_sha256",
        "matrix_role",
        "penalty_candidate_id",
        "penalty_candidate_sha256",
        "config",
    }
    return {
        **{key: deepcopy(value) for key, value in template.items() if key not in ignored},
        **attempt.to_dict(),
        "matrix_role": "provisional_recovery_penalty_interaction",
        "config": config,
        "penalty_candidate_id": penalty["penalty_id"],
        "penalty_candidate_sha256": penalty["candidate_sha256"],
    }


def build_stage_p_proposal(
    *,
    stage_f_proposal: Mapping[str, Any],
    stage_f_completion: Mapping[str, Any],
    stage_f_profile_matrix: Mapping[str, Any],
    stage_f_current_role_matrix: Mapping[str, Any],
    penalty_registry: Mapping[str, Any],
    budget_contract: Mapping[str, Any],
    parent_experiment_id: str,
    solver_hash: str,
    metric_contract_hash: str,
    evaluation_hash: str,
) -> dict[str, Any]:
    """Freeze the 96-reuse + 192-new Stage P matrix without solver work."""

    if not parent_experiment_id:
        raise StagePPlanError("parent_experiment_id_must_not_be_empty")
    for name, value in (
        ("solver_hash", solver_hash),
        ("metric_contract_hash", metric_contract_hash),
        ("evaluation_hash", evaluation_hash),
    ):
        require_hash(name, value)
    control, penalties = _penalties(penalty_registry)
    templates, current_rows = _stage_f_sources(
        stage_f_proposal=stage_f_proposal,
        stage_f_completion=stage_f_completion,
        stage_f_profile_matrix=stage_f_profile_matrix,
        stage_f_current_role_matrix=stage_f_current_role_matrix,
        control_penalty_id=str(control["penalty_id"]),
    )
    if any(
        template.get("solver_hash") != solver_hash
        or template.get("metric_contract_hash") != metric_contract_hash
        for template in templates
    ):
        raise StagePPlanError("stage_p_stage_f_runtime_identity_mismatch")
    frozen = require_mapping(
        "stage_p_stage_f_frozen_contracts",
        stage_f_proposal.get("frozen_contracts"),
    )
    limits = require_mapping(
        "stage_p_budget_stage_unique_limits",
        budget_contract.get("stage_unique_limits"),
    )
    if (
        limits.get(PENALTY_INTERACTION_STAGE) != 288
        or frozen.get("budget_contract_hash") != canonical_sha256(budget_contract)
        or frozen.get("penalty_registry_hash") != penalty_registry.get("registry_sha256")
        or frozen.get("metric_contract_hash") != metric_contract_hash
    ):
        raise StagePPlanError("stage_p_frozen_contract_mismatch")
    new_penalties = [
        penalties[penalty_id]
        for penalty_id in sorted(penalties)
        if penalty_id != control["penalty_id"]
    ]
    identities = [
        _new_penalty_identity(
            template=template,
            penalty=penalty,
            solver_hash=solver_hash,
            metric_contract_hash=metric_contract_hash,
            evaluation_hash=evaluation_hash,
            parent_experiment_id=parent_experiment_id,
        )
        for penalty in new_penalties
        for template in templates
    ]
    identity_hashes = [str(identity["identity_sha256"]) for identity in identities]
    if (
        len(identities) != EXPECTED_NEW_IDENTITY_COUNT
        or len(set(identity_hashes)) != EXPECTED_NEW_IDENTITY_COUNT
        or any(
            identity_hash in {str(row["identity_sha256"]) for row in current_rows.values()}
            for identity_hash in identity_hashes
        )
    ):
        raise StagePPlanError("stage_p_new_identity_matrix_mismatch")
    logical_tasks = [
        {
            "penalty_candidate_id": control["penalty_id"],
            "filter_profile_id": row["filter_profile_id"],
            "record_id": row["record_id"],
            "scene": row["scene"],
            "identity_sha256": row["identity_sha256"],
            "result_source": "stage_f_profile_matrix",
        }
        for row in current_rows.values()
    ]
    logical_tasks.extend(
        {
            "penalty_candidate_id": identity["penalty_candidate_id"],
            "filter_profile_id": identity["filter_profile_id"],
            "record_id": identity["record_id"],
            "scene": identity["scene"],
            "identity_sha256": identity["identity_sha256"],
            "result_source": "stage_p_execution",
        }
        for identity in identities
    )
    if len(logical_tasks) != EXPECTED_LOGICAL_RESULT_COUNT:
        raise StagePPlanError("stage_p_logical_matrix_mismatch")
    proposal = {
        "proposal_version": "lyx_stage_p_execution_proposal_v1",
        "status": "ready_for_execution",
        "parent_experiment_id": parent_experiment_id,
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "independent_bo_authorized": False,
        "stage_f_proposal_sha256": stage_f_proposal["proposal_sha256"],
        "stage_f_completion_sha256": stage_f_completion["completion_sha256"],
        "stage_f_profile_matrix_sha256": stage_f_profile_matrix["matrix_sha256"],
        "stage_f_current_role_matrix_sha256": (stage_f_current_role_matrix["matrix_sha256"]),
        "provisional_recovery_id": stage_f_proposal["provisional_recovery_id"],
        "rollback_backup_id": stage_f_proposal["rollback_backup_id"],
        "control_penalty_id": control["penalty_id"],
        "penalty_ids": sorted(penalties),
        "profile_count": 8,
        "record_count": 12,
        "logical_task_count": EXPECTED_LOGICAL_RESULT_COUNT,
        "reused_stage_f_result_count": EXPECTED_REUSED_RESULT_COUNT,
        "planned_new_unique_identity_count": (EXPECTED_NEW_IDENTITY_COUNT),
        "maximum_penalty_interaction_unique_identity_count": 288,
        "selection_ranking_key": EXPECTED_SELECTION_RANKING_KEY,
        "frozen_contracts": {
            **dict(frozen),
            "stage_p_evaluation_hash": evaluation_hash,
            "penalty_registry_hash": penalty_registry["registry_sha256"],
            "budget_contract_hash": canonical_sha256(budget_contract),
        },
        "penalties": [penalties[key] for key in sorted(penalties)],
        "profiles": deepcopy(stage_f_proposal["profiles"]),
        "record_panel": deepcopy(stage_f_proposal["record_panel"]),
        "identities": identities,
        "logical_tasks": logical_tasks,
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def propose_stage_p_execution(
    *,
    stage_f_proposal_path: Path,
    stage_f_completion_path: Path,
    stage_f_profile_matrix_path: Path,
    stage_f_current_role_matrix_path: Path,
    penalty_registry_path: Path,
    budget_contract_path: Path,
    output_dir: Path,
    source_root: Path,
    parent_experiment_id: str,
) -> dict[str, Any]:
    """Atomically publish a source-bound, zero-run Stage P proposal."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise StagePPlanError(f"stage_p_output_already_exists:{destination}")
    source_paths = {
        "stage_f_proposal": Path(stage_f_proposal_path).resolve(),
        "stage_f_completion": Path(stage_f_completion_path).resolve(),
        "stage_f_profile_matrix": Path(stage_f_profile_matrix_path).resolve(),
        "stage_f_current_role_matrix": Path(stage_f_current_role_matrix_path).resolve(),
        "penalty_registry": Path(penalty_registry_path).resolve(),
        "budget_contract": Path(budget_contract_path).resolve(),
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise StagePPlanError(f"stage_p_source_missing:{name}:{path}")
    sources = {name: read_json(path) for name, path in source_paths.items()}
    stage_f_source_artifacts = sources["stage_f_proposal"].get("source_artifacts")
    if stage_f_source_artifacts is not None:
        penalty_binding = require_mapping(
            "stage_p_stage_f_penalty_registry_binding",
            require_mapping(
                "stage_p_stage_f_source_artifacts",
                stage_f_source_artifacts,
            ).get("penalty_registry"),
        )
        if penalty_binding.get("sha256") != file_sha256(source_paths["penalty_registry"]):
            raise StagePPlanError("stage_p_penalty_registry_source_binding_mismatch")
    metric_contract = stage_r_metric_contract_v1()
    solver = runtime_source_identity(Path(source_root).resolve())
    evaluation_roots = (
        "ppg_hr.v2.recovery_stage_p_contracts",
        "ppg_hr.v2.recovery_stage_p_execution",
        "ppg_hr.v2.recovery_stage_p_experiment",
        "ppg_hr.v2.recovery_stage_p_plan",
        "ppg_hr.v2.recovery_stage_p_reporting",
        "ppg_hr.v2.recovery_stage_p_runner",
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
    proposal = build_stage_p_proposal(
        stage_f_proposal=sources["stage_f_proposal"],
        stage_f_completion=sources["stage_f_completion"],
        stage_f_profile_matrix=sources["stage_f_profile_matrix"],
        stage_f_current_role_matrix=sources["stage_f_current_role_matrix"],
        penalty_registry=sources["penalty_registry"],
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
        raise StagePPlanError("stage_p_staging_parent_mismatch")
    try:
        staging.mkdir(parents=True)
        atomic_write_json(staging / "metric_contract.json", metric_contract)
        atomic_write_json(staging / "solver_source_identity.json", solver)
        atomic_write_json(
            staging / "evaluation_source_identity.json",
            evaluation,
        )
        atomic_write_json(
            staging / "stage_p_execution_proposal.json",
            proposal,
        )
        artifact_names = (
            "metric_contract.json",
            "solver_source_identity.json",
            "evaluation_source_identity.json",
            "stage_p_execution_proposal.json",
        )
        receipt = {
            "receipt_version": "lyx_stage_p_proposal_receipt_v1",
            "status": "ready_for_execution",
            "formal_solver_run_count": 0,
            "diagnostic_solver_run_count": 0,
            "independent_bo_run_count": 0,
            "logical_task_count": EXPECTED_LOGICAL_RESULT_COUNT,
            "reused_stage_f_result_count": EXPECTED_REUSED_RESULT_COUNT,
            "planned_new_unique_identity_count": (EXPECTED_NEW_IDENTITY_COUNT),
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
