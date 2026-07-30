"""Governed per-record independent BO after a no-safe Stage R decision."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from .bo_space_generalization import (
    BOCandidate,
    SearchEvaluation,
    SearchExperimentIdentity,
    SearchRequestContext,
    SeedSearchBudget,
    build_seed_stability_audit,
    build_bo_search_space,
    run_seed_search,
)
from .experiment_freeze_utils import (
    V2_RUNTIME_ROOT_MODULES,
    file_sha256,
    runtime_source_identity,
)
from .phase2_experiment_io import atomic_write_json, read_json
from . import recovery_stage_r_cache as stage_r_cache
from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import (
    AttemptIdentity,
    AttemptRegistry,
    BudgetAmendmentRequest,
    BudgetContract,
    ExplorationRegistry,
    GovernanceError,
    IndependentBORequest,
    validate_budget_amendment_authorization,
    validate_independent_bo_authorization,
)
from .recovery_selection import (
    RecoveryCandidateEvaluation,
    RecoveryPanelRecord,
    RecoveryRecordEvaluation,
    select_rank1_recovery_candidate_evaluations,
)
from .recovery_stage_r_experiment import (
    run_stage_r_numerical_identity,
)


class RecoveryIndependentBOError(RuntimeError):
    """The recovery independent-BO experiment violates its frozen contract."""


class RecoveryIndependentBOAuthorizationError(
    RecoveryIndependentBOError
):
    """The exact recovery independent-BO proposal is not authorized."""


STAGE = "recovery_independent_bo"
PROPOSAL_VERSION = "lyx_recovery_independent_bo_proposal_v1"
EXPECTED_RECORD_COUNT = 12
EXPECTED_RECOVERY_COUNT = 3
EXPECTED_SEARCH_CELL_COUNT = 36
EXPECTED_CANDIDATE_COUNT = 300
EXPECTED_CELL_UNIQUE_BUDGET = 150
EXPECTED_UNIQUE_BUDGET = 5400
EXPECTED_SCENE_COUNTS = {
    "jianpan": 3,
    "kaihe": 3,
    "run": 3,
    "xiezi": 3,
}
CONTROL_RECOVERY_ID = "current_fixed_floor_control_v1"
_SOURCE_STAGE = "recovery_sentinel_rank1_replan"
_SOURCE_PROPOSAL_VERSION = "lyx_stage_r_rank1_replan_proposal_v2"
_SOURCE_COMPLETION_VERSION = "lyx_stage_r_rank1_replan_completion_v2"
_AUTHORIZATION_STATE = "awaiting_human_independent_bo_decision"
BLANKET_AUTHORIZATION_EXPIRES_AT = (
    "2026-07-31T10:00:00+08:00"
)
BLANKET_AUTHORIZATION_USER_TEXT = (
    "另外在7月31日10：00前，所有需要人工批准的proposal我都授权通过，"
    "目标进行不需要在被阻塞，你可以尽力提出想法并进行验证，最终完成整个大实验。"
)


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryIndependentBOError(f"{name}_must_be_object")
    return value


def _list(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise RecoveryIndependentBOError(f"{name}_must_be_list")
    return value


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    artifact_name: str,
) -> str:
    value = payload.get(hash_field)
    if not isinstance(value, str) or len(value) != 64:
        raise RecoveryIndependentBOError(
            f"{artifact_name}_{hash_field}_missing"
        )
    unsigned = {
        key: item for key, item in payload.items() if key != hash_field
    }
    if canonical_sha256(unsigned) != value:
        raise RecoveryIndependentBOError(
            f"{artifact_name}_{hash_field}_mismatch"
        )
    return value


def recovery_independent_bo_metric_contract_v1() -> dict[str, Any]:
    """Freeze the BO objective, constraints, and final upper-bound selector."""

    contract = {
        "contract_version": "lyx_recovery_independent_bo_metric_v1",
        "objective": "minimize_final_motion_mae_bpm",
        "per_record_constraints": [
            "spectral_gate_contract_v2",
            "l10_engineering_gate",
            "l20_engineering_gate",
            "mae_independent_delta_le_2_bpm",
            "no_new_right_censored_recovery",
            "true_rise_underestimate_delta_le_2_bpm",
            "current_l10_catastrophic_regression_gate",
            "mae_current_delta_le_2_bpm",
        ],
        "cross_record_final_gate": (
            "loo_training_pair_mean_independent_mae_delta_le_1_bpm"
        ),
        "search_candidate_order": [
            "eligible_first",
            "final_motion_mae_bpm",
            "candidate_id",
        ],
        "invalid_objective": 1.0e12,
        "evidence_class": "development_reuse_sample_in_upper_bound",
        "algorithm_level_holdout": False,
        "deployment_claim_allowed": False,
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def recovery_independent_bo_seed_manifest_v1() -> dict[str, Any]:
    """Freeze the existing three-lane independent-BO budget."""

    budget = SeedSearchBudget(
        objective_version="recovery_independent_bo_final_motion_v1",
        constraints_version="recovery_independent_bo_safety_v1",
    )
    serialized_budget = asdict(budget)
    serialized_budget["lane_seeds"] = list(budget.lane_seeds)
    manifest = {
        "manifest_version": "lyx_recovery_independent_bo_seed_manifest_v1",
        **serialized_budget,
        "search_cell_count": EXPECTED_SEARCH_CELL_COUNT,
        "unique_budget_per_cell": EXPECTED_CELL_UNIQUE_BUDGET,
        "total_unique_budget": EXPECTED_UNIQUE_BUDGET,
        "parallel_lanes": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _space_payload() -> dict[str, Any]:
    space = build_bo_search_space("physical_v1")
    candidates = [
        {
            "candidate_id": candidate.candidate_id,
            "coordinate": list(candidate.coordinate),
            "requested_params": dict(candidate.requested_params),
            "actual_params": dict(candidate.actual_params),
            "fixed_params": dict(candidate.fixed_params),
        }
        for candidate in space.candidates
    ]
    payload = {
        "space_name": space.name,
        "parameter_names": list(space.parameter_names),
        "option_values": [
            list(values) for values in space.option_values
        ],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    payload["search_space_sha256"] = canonical_sha256(payload)
    return payload


def _validate_stage_r_sources(
    *,
    stage_r_proposal: Mapping[str, Any],
    stage_r_completion: Mapping[str, Any],
    stage_r_selection: Mapping[str, Any],
    stage_r_result_index: Mapping[str, Any],
) -> tuple[
    str,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Mapping[str, Any]],
]:
    proposal_sha = _verify_embedded_hash(
        stage_r_proposal,
        hash_field="proposal_sha256",
        artifact_name="stage_r_proposal",
    )
    completion_sha = _verify_embedded_hash(
        stage_r_completion,
        hash_field="completion_sha256",
        artifact_name="stage_r_completion",
    )
    selection_sha = _verify_embedded_hash(
        stage_r_selection,
        hash_field="selection_sha256",
        artifact_name="stage_r_selection",
    )
    index_sha = _verify_embedded_hash(
        stage_r_result_index,
        hash_field="index_sha256",
        artifact_name="stage_r_result_index",
    )
    del completion_sha, selection_sha, index_sha
    if (
        stage_r_proposal.get("proposal_version")
        != _SOURCE_PROPOSAL_VERSION
        or stage_r_completion.get("completion_version")
        != _SOURCE_COMPLETION_VERSION
        or stage_r_completion.get("status")
        != "no_safe_recovery_candidate"
        or stage_r_completion.get("next_state")
        != "awaiting_human_independent_bo_decision"
        or stage_r_completion.get("proposal_sha256") != proposal_sha
        or stage_r_completion.get("independent_bo_run_count") != 0
        or stage_r_selection.get("status")
        != "no_safe_recovery_candidate"
        or stage_r_selection.get("provisional_recovery_id") is not None
        or stage_r_result_index.get("proposal_sha256") != proposal_sha
        or stage_r_result_index.get("result_count") != 36
    ):
        raise RecoveryIndependentBOError(
            "stage_r_no_safe_source_state_invalid"
        )
    records = tuple(
        dict(_mapping("stage_r_record", raw))
        for raw in _list(
            "stage_r_record_panel",
            stage_r_proposal.get("record_panel"),
        )
    )
    candidates = tuple(
        dict(_mapping("stage_r_recovery_candidate", raw))
        for raw in _list(
            "stage_r_recovery_candidates",
            stage_r_proposal.get("recovery_candidates"),
        )
    )
    identities = tuple(
        dict(_mapping("stage_r_identity", raw))
        for raw in _list(
            "stage_r_identities",
            stage_r_proposal.get("identities"),
        )
    )
    results = tuple(
        dict(_mapping("stage_r_result", raw))
        for raw in _list(
            "stage_r_results",
            stage_r_result_index.get("results"),
        )
    )
    record_ids = {str(record["record_id"]) for record in records}
    candidate_ids = {
        str(candidate["candidate_id"]) for candidate in candidates
    }
    coordinates = {
        (
            str(identity["recovery_candidate_id"]),
            str(identity["record_id"]),
        )
        for identity in identities
    }
    expected_coordinates = {
        (candidate_id, record_id)
        for candidate_id in candidate_ids
        for record_id in record_ids
    }
    scene_counts = {
        scene: sum(record.get("scene") == scene for record in records)
        for scene in EXPECTED_SCENE_COUNTS
    }
    if (
        len(records) != EXPECTED_RECORD_COUNT
        or len(record_ids) != EXPECTED_RECORD_COUNT
        or len(candidates) != EXPECTED_RECOVERY_COUNT
        or len(candidate_ids) != EXPECTED_RECOVERY_COUNT
        or len(identities) != EXPECTED_SEARCH_CELL_COUNT
        or coordinates != expected_coordinates
        or scene_counts != EXPECTED_SCENE_COUNTS
        or any(
            identity.get("stage") != _SOURCE_STAGE
            or identity.get("sentinel_role") != "fixed_rank1"
            or identity.get("adaptive_reference_stage_limit") != 1
            for identity in identities
        )
    ):
        raise RecoveryIndependentBOError(
            "stage_r_no_safe_source_panel_invalid"
        )
    result_by_coordinate = {
        (
            str(result["recovery_candidate_id"]),
            str(result["record_id"]),
        ): result
        for result in results
    }
    if set(result_by_coordinate) != expected_coordinates:
        raise RecoveryIndependentBOError(
            "stage_r_no_safe_result_panel_invalid"
        )
    current_by_record = {
        record_id: _mapping(
            f"stage_r_current_metrics:{record_id}",
            result_by_coordinate[
                (CONTROL_RECOVERY_ID, record_id)
            ].get("metrics"),
        )
        for record_id in sorted(record_ids)
    }
    return proposal_sha, records, candidates, current_by_record


def _baseline_source(
    *,
    stage_r_proposal: Mapping[str, Any],
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    sources = _mapping(
        "stage_r_source_artifacts",
        stage_r_proposal.get("source_artifacts"),
    )
    artifact = _mapping(
        "stage_r_baseline_metrics_source",
        sources.get("baseline_metrics"),
    )
    if artifact.get("path_base") != "repository_root":
        raise RecoveryIndependentBOError(
            "baseline_metrics_path_base_invalid"
        )
    path = (
        Path(repository_root).resolve()
        / str(artifact.get("path", ""))
    ).resolve()
    root = Path(repository_root).resolve()
    if (
        not path.is_relative_to(root)
        or not path.is_file()
        or file_sha256(path) != artifact.get("file_sha256")
    ):
        raise RecoveryIndependentBOError(
            "baseline_metrics_source_drift"
        )
    payload = read_json(path)
    records = {
        str(row["sample_id"]): _mapping(
            "independent_baseline_metrics",
            _mapping(
                "independent_baseline_record",
                row,
            ).get("metrics"),
        )
        for row in _list(
            "independent_baseline_records",
            payload.get("records"),
        )
    }
    if len(records) != EXPECTED_RECORD_COUNT:
        raise RecoveryIndependentBOError(
            "independent_baseline_panel_invalid"
        )
    frozen = {
        "path": path.relative_to(root).as_posix(),
        "path_base": "repository_root",
        "file_sha256": file_sha256(path),
    }
    return frozen, records


def build_recovery_independent_bo_proposal(
    *,
    stage_r_proposal: Mapping[str, Any],
    stage_r_completion: Mapping[str, Any],
    stage_r_selection: Mapping[str, Any],
    stage_r_result_index: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Build an exact zero-run proposal from the completed Stage R v2 stop."""

    (
        stage_r_proposal_sha,
        records,
        candidates,
        current_by_record,
    ) = _validate_stage_r_sources(
        stage_r_proposal=stage_r_proposal,
        stage_r_completion=stage_r_completion,
        stage_r_selection=stage_r_selection,
        stage_r_result_index=stage_r_result_index,
    )
    baseline_source, independent_by_record = _baseline_source(
        stage_r_proposal=stage_r_proposal,
        repository_root=repository_root,
    )
    if set(independent_by_record) != {
        str(record["record_id"]) for record in records
    }:
        raise RecoveryIndependentBOError(
            "independent_baseline_record_identity_mismatch"
        )
    identity_by_coordinate = {
        (
            str(identity["recovery_candidate_id"]),
            str(identity["record_id"]),
        ): dict(identity)
        for identity in _list(
            "stage_r_identities",
            stage_r_proposal.get("identities"),
        )
    }
    search_space = _space_payload()
    seed_manifest = recovery_independent_bo_seed_manifest_v1()
    metric_contract = recovery_independent_bo_metric_contract_v1()
    source_root = Path(repository_root).resolve() / "python" / "src"
    solver_source = runtime_source_identity(
        source_root,
        root_modules=V2_RUNTIME_ROOT_MODULES,
    )
    evaluation_source = runtime_source_identity(
        source_root,
        root_modules=(
            "ppg_hr.v2.recovery_independent_bo_experiment",
            "ppg_hr.v2.recovery_stage_r_experiment",
            "ppg_hr.v2.recovery_stage_r_cache",
            "ppg_hr.v2.recovery_selection",
            "ppg_hr.v2.bo_space_generalization",
        ),
    )
    search_cells: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        for record in sorted(
            records,
            key=lambda item: str(item["record_id"]),
        ):
            record_id = str(record["record_id"])
            template = identity_by_coordinate[
                (candidate_id, record_id)
            ]
            cell = {
                "recovery_candidate_id": candidate_id,
                "recovery_candidate_sha256": candidate[
                    "candidate_sha256"
                ],
                "mechanism_complexity": int(
                    candidate["mechanism_complexity"]
                ),
                "record_id": record_id,
                "scene": record["scene"],
                "true_rise_applicable": bool(
                    record["true_rise_applicable"]
                ),
                "unique_budget": EXPECTED_CELL_UNIQUE_BUDGET,
                "template_identity": template,
                "current_metrics": dict(
                    current_by_record[record_id]
                ),
                "independent_metrics": dict(
                    independent_by_record[record_id]
                ),
            }
            cell["cell_sha256"] = canonical_sha256(cell)
            search_cells.append(cell)
    if len(search_cells) != EXPECTED_SEARCH_CELL_COUNT:
        raise RecoveryIndependentBOError(
            "independent_bo_search_cell_count_invalid"
        )
    target_budget = BudgetContract.proposed_v13_recovery_independent_bo()
    record_manifest_hash = canonical_sha256(list(records))
    amendment = BudgetAmendmentRequest(
        stage=STAGE,
        profile_design_rule_hash=search_space[
            "search_space_sha256"
        ],
        record_manifest_hash=record_manifest_hash,
        added_unique_identities=EXPECTED_UNIQUE_BUDGET,
        normal_unique_identity_limit=int(
            target_budget.normal_unique_identity_limit or 0
        ),
        max_unique_identities=target_budget.max_unique_identities,
        max_attempts=target_budget.max_attempts,
    )
    independent_request = IndependentBORequest(
        solver_hash=solver_source["source_bundle_sha256"],
        search_space_hash=search_space["search_space_sha256"],
        metric_contract_hash=metric_contract["contract_sha256"],
        seed_manifest_hash=seed_manifest["manifest_sha256"],
        unique_budget=EXPECTED_UNIQUE_BUDGET,
    )
    proposal: dict[str, Any] = {
        "proposal_version": PROPOSAL_VERSION,
        "status": "frozen_zero_solver_runs",
        "stage": STAGE,
        "evidence_class": "development_reuse_sample_in_upper_bound",
        "algorithm_level_holdout": False,
        "parent_experiment_id": stage_r_proposal[
            "parent_experiment_id"
        ],
        "source_stage_r": {
            "proposal_sha256": stage_r_proposal_sha,
            "completion_sha256": stage_r_completion[
                "completion_sha256"
            ],
            "selection_sha256": stage_r_selection[
                "selection_sha256"
            ],
            "result_index_sha256": stage_r_result_index[
                "index_sha256"
            ],
            "status": "no_safe_recovery_candidate",
        },
        "baseline_metrics_source": baseline_source,
        "record_manifest_hash": record_manifest_hash,
        "record_panel": list(records),
        "recovery_candidates": list(candidates),
        "search_space": search_space,
        "seed_manifest": seed_manifest,
        "metric_contract": metric_contract,
        "solver_source_identity": solver_source,
        "evaluation_source_identity": evaluation_source,
        "search_cells": search_cells,
        "search_cell_count": len(search_cells),
        "unique_budget": EXPECTED_UNIQUE_BUDGET,
        "worst_case_attempt_budget": (
            EXPECTED_UNIQUE_BUDGET * 2
        ),
        "budget_contract": target_budget.to_dict(),
        "budget_contract_hash": target_budget.sha256,
        "budget_amendment_request": asdict(amendment),
        "independent_bo_request": asdict(independent_request),
        "formal_solver_run_count": 0,
        "independent_bo_run_count": 0,
        "automatic_stage_f_execution": False,
        "next_state": "awaiting_authorized_execution",
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def build_recovery_independent_bo_identity(
    *,
    proposal: Mapping[str, Any],
    cell: Mapping[str, Any],
    candidate: BOCandidate,
) -> dict[str, Any]:
    """Materialize one dynamically selected, budgeted solver identity."""

    if (
        proposal.get("proposal_version") != PROPOSAL_VERSION
        or candidate.space_name != "physical_v1"
        or cell.get("unique_budget")
        != EXPECTED_CELL_UNIQUE_BUDGET
    ):
        raise RecoveryIndependentBOError(
            "independent_bo_identity_source_invalid"
        )
    template = deepcopy(
        dict(
            _mapping(
                "independent_bo_template_identity",
                cell.get("template_identity"),
            )
        )
    )
    config = deepcopy(
        dict(
            _mapping(
                "independent_bo_template_config",
                template.get("config"),
            )
        )
    )
    parameters = dict(
        _mapping(
            "independent_bo_template_parameters",
            config.get("parameters"),
        )
    )
    parameters.update(dict(candidate.actual_params))
    parameters["adaptive_reference_stage_limit"] = 1
    config["parameters"] = parameters
    solver_hash = str(
        _mapping(
            "independent_bo_request",
            proposal.get("independent_bo_request"),
        )["solver_hash"]
    )
    metric_hash = str(
        _mapping(
            "independent_bo_metric_contract",
            proposal.get("metric_contract"),
        )["contract_sha256"]
    )
    evaluation_hash = str(
        _mapping(
            "independent_bo_evaluation_source",
            proposal.get("evaluation_source_identity"),
        )["source_bundle_sha256"]
    )
    attempt = AttemptIdentity(
        solver_hash=solver_hash,
        config_hash=canonical_sha256(config),
        metric_contract_hash=metric_hash,
        evaluation_hash=evaluation_hash,
        data_sha256=str(template["data_sha256"]),
        record_id=str(cell["record_id"]),
        stage=STAGE,
        attempt_kind="formal",
        parent_experiment_id=str(
            proposal["parent_experiment_id"]
        ),
    )
    candidate_digest = candidate.candidate_id.rsplit(":", 1)[-1]
    profile_id = f"bo-physical-v1-{candidate_digest[:20]}"
    profile_payload = {
        "profile_id": profile_id,
        "space_name": candidate.space_name,
        "candidate_id": candidate.candidate_id,
        "requested_params": dict(candidate.requested_params),
        "actual_params": dict(candidate.actual_params),
        "adaptive_reference_stage_limit": 1,
    }
    identity = {
        **attempt.to_dict(),
        "identity_sha256": attempt.sha256,
        "cache_identity_sha256": attempt.sha256,
        "scene": cell["scene"],
        "data_path": template["data_path"],
        "reference_path": template["reference_path"],
        "raw_data_sha256": template["raw_data_sha256"],
        "reference_sha256": template["reference_sha256"],
        "method_names": deepcopy(template["method_names"]),
        "true_rise_applicable": bool(
            cell["true_rise_applicable"]
        ),
        "config": config,
        "filter_profile_id": profile_id,
        "filter_profile_sha256": canonical_sha256(
            profile_payload
        ),
        "base_filter_profile_id": template[
            "base_filter_profile_id"
        ],
        "physical_memory_ms": int(
            candidate.requested_params["memory_ms"]
        ),
        "actual_taps": int(candidate.actual_params["max_order"]),
        "nominal_mu": float(
            candidate.actual_params["lms_mu_base"]
        ),
        "adaptive_reference_stage_limit": 1,
        "spectral_audit_required": True,
        "sentinel_role": "fixed_rank1",
        "recovery_candidate_id": cell[
            "recovery_candidate_id"
        ],
        "recovery_candidate_sha256": cell[
            "recovery_candidate_sha256"
        ],
        "mechanism_complexity": int(
            cell["mechanism_complexity"]
        ),
        "candidate_min_bpm": template.get(
            "candidate_min_bpm"
        ),
        "penalty_candidate_id": template[
            "penalty_candidate_id"
        ],
        "bo_candidate_id": candidate.candidate_id,
        "bo_candidate_coordinate": list(candidate.coordinate),
        "bo_requested_params": dict(candidate.requested_params),
        "bo_actual_params": dict(candidate.actual_params),
        "source_stage_r_identity_sha256": template[
            "identity_sha256"
        ],
        "search_cell_sha256": cell["cell_sha256"],
    }
    return identity


def validate_recovery_independent_bo_execution_authorization(
    proposal: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate both the exact BO request and the blanket-deadline binding."""

    try:
        proposal_sha = _verify_embedded_hash(
            proposal,
            hash_field="proposal_sha256",
            artifact_name="independent_bo_proposal",
        )
        request = IndependentBORequest(
            **dict(
                _mapping(
                    "independent_bo_request",
                    proposal.get("independent_bo_request"),
                )
            )
        )
        validated = validate_independent_bo_authorization(
            request,
            receipt=receipt,
        )
        if (
            validated.get("proposal_sha256") != proposal_sha
            or validated.get("budget_contract_hash")
            != proposal.get("budget_contract_hash")
            or validated.get("authorization_basis")
            != "blanket_proposal_authorization_until_deadline"
            or validated.get(
                "blanket_authorization_expires_at"
            )
            != BLANKET_AUTHORIZATION_EXPIRES_AT
            or validated.get("user_authorization")
            != BLANKET_AUTHORIZATION_USER_TEXT
        ):
            raise RecoveryIndependentBOAuthorizationError(
                "independent_bo_authorization_invalid"
            )
        approved_at = datetime.fromisoformat(
            str(validated["approved_at"])
        )
        expires_at = datetime.fromisoformat(
            str(validated["blanket_authorization_expires_at"])
        )
        if (
            approved_at.tzinfo is None
            or approved_at.utcoffset()
            != datetime.fromisoformat(
                BLANKET_AUTHORIZATION_EXPIRES_AT
            ).utcoffset()
            or expires_at
            != datetime.fromisoformat(
                BLANKET_AUTHORIZATION_EXPIRES_AT
            )
            or approved_at > expires_at
        ):
            raise RecoveryIndependentBOAuthorizationError(
                "independent_bo_authorization_invalid"
            )
    except (
        GovernanceError,
        KeyError,
        TypeError,
        ValueError,
        RecoveryIndependentBOError,
    ) as error:
        if isinstance(
            error,
            RecoveryIndependentBOAuthorizationError,
        ):
            raise
        raise RecoveryIndependentBOAuthorizationError(
            "independent_bo_authorization_invalid"
        ) from error
    return validated


def _budget_amendment_from_proposal(
    proposal: Mapping[str, Any],
) -> BudgetAmendmentRequest:
    return BudgetAmendmentRequest(
        **dict(
            _mapping(
                "independent_bo_budget_amendment",
                proposal.get("budget_amendment_request"),
            )
        )
    )


def _exploration_from_payload(
    payload: Mapping[str, Any],
) -> ExplorationRegistry:
    return ExplorationRegistry(
        unique_budget=int(payload["unique_budget"]),
        allowed_identity_sha256=tuple(
            str(value)
            for value in _list(
                "exploration_allowed_identities",
                payload.get("allowed_identity_sha256"),
            )
        ),
        registry_version=str(payload["registry_version"]),
    )


def validate_recovery_independent_bo_preflight(
    *,
    proposal: Mapping[str, Any],
    repository_root: Path,
) -> None:
    """Fail closed before governance migration or solver work."""

    _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="independent_bo_proposal",
    )
    if (
        proposal.get("proposal_version") != PROPOSAL_VERSION
        or proposal.get("status") != "frozen_zero_solver_runs"
        or proposal.get("stage") != STAGE
        or proposal.get("search_cell_count")
        != EXPECTED_SEARCH_CELL_COUNT
        or proposal.get("unique_budget")
        != EXPECTED_UNIQUE_BUDGET
        or proposal.get("formal_solver_run_count") != 0
        or proposal.get("independent_bo_run_count") != 0
        or proposal.get("automatic_stage_f_execution") is not False
    ):
        raise RecoveryIndependentBOError(
            "independent_bo_proposal_contract_invalid"
        )
    target_budget = BudgetContract.proposed_v13_recovery_independent_bo()
    if (
        proposal.get("budget_contract") != target_budget.to_dict()
        or proposal.get("budget_contract_hash")
        != target_budget.sha256
    ):
        raise RecoveryIndependentBOError(
            "independent_bo_budget_contract_drift"
        )
    current_space = _space_payload()
    if proposal.get("search_space") != current_space:
        raise RecoveryIndependentBOError(
            "independent_bo_search_space_drift"
        )
    if (
        proposal.get("seed_manifest")
        != recovery_independent_bo_seed_manifest_v1()
        or proposal.get("metric_contract")
        != recovery_independent_bo_metric_contract_v1()
    ):
        raise RecoveryIndependentBOError(
            "independent_bo_search_contract_drift"
        )
    source_root = Path(repository_root).resolve() / "python" / "src"
    current_solver = runtime_source_identity(
        source_root,
        root_modules=V2_RUNTIME_ROOT_MODULES,
    )
    current_evaluation = runtime_source_identity(
        source_root,
        root_modules=(
            "ppg_hr.v2.recovery_independent_bo_experiment",
            "ppg_hr.v2.recovery_stage_r_experiment",
            "ppg_hr.v2.recovery_stage_r_cache",
            "ppg_hr.v2.recovery_selection",
            "ppg_hr.v2.bo_space_generalization",
        ),
    )
    if (
        proposal.get("solver_source_identity") != current_solver
        or proposal.get("evaluation_source_identity")
        != current_evaluation
    ):
        raise RecoveryIndependentBOError(
            "independent_bo_source_identity_drift"
        )
    baseline = _mapping(
        "independent_bo_baseline_source",
        proposal.get("baseline_metrics_source"),
    )
    baseline_path = (
        Path(repository_root).resolve()
        / str(baseline.get("path", ""))
    ).resolve()
    if (
        baseline.get("path_base") != "repository_root"
        or not baseline_path.is_relative_to(
            Path(repository_root).resolve()
        )
        or not baseline_path.is_file()
        or file_sha256(baseline_path)
        != baseline.get("file_sha256")
    ):
        raise RecoveryIndependentBOError(
            "independent_bo_baseline_source_drift"
        )
    cells = [
        _mapping("independent_bo_search_cell", raw)
        for raw in _list(
            "independent_bo_search_cells",
            proposal.get("search_cells"),
        )
    ]
    if (
        len(cells) != EXPECTED_SEARCH_CELL_COUNT
        or len(
            {
                (
                    str(cell["recovery_candidate_id"]),
                    str(cell["record_id"]),
                )
                for cell in cells
            }
        )
        != EXPECTED_SEARCH_CELL_COUNT
        or any(
            canonical_sha256(
                {
                    key: value
                    for key, value in cell.items()
                    if key != "cell_sha256"
                }
            )
            != cell.get("cell_sha256")
            for cell in cells
        )
    ):
        raise RecoveryIndependentBOError(
            "independent_bo_search_cell_drift"
        )


def prepare_recovery_independent_bo_governance(
    *,
    proposal: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
    budget_authorization: Mapping[str, Any],
    source_governance_dir: Path,
    target_governance_dir: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Migrate v12 to v13 without registering or running BO identities."""

    validate_recovery_independent_bo_preflight(
        proposal=proposal,
        repository_root=repository_root,
    )
    validated_execution = (
        validate_recovery_independent_bo_execution_authorization(
            proposal,
            receipt=execution_authorization,
        )
    )
    amendment = _budget_amendment_from_proposal(proposal)
    validated_budget = validate_budget_amendment_authorization(
        amendment,
        receipt=budget_authorization,
    )
    source_root = Path(source_governance_dir).resolve()
    target_root = Path(target_governance_dir).resolve()
    source_budget = (
        BudgetContract.proposed_v12_stage_r_rank1_runtime_fix()
    )
    target_budget = (
        BudgetContract.proposed_v13_recovery_independent_bo()
    )
    if read_json(source_root / "budget_contract.json") != (
        source_budget.to_dict()
    ):
        raise RecoveryIndependentBOError(
            "independent_bo_source_governance_budget_mismatch"
        )
    exploration_payload = read_json(
        source_root / "exploration_registry.json"
    )
    exploration = _exploration_from_payload(exploration_payload)
    source_registry = AttemptRegistry.open(
        source_root / "attempt_registry.json",
        budget_contract=source_budget,
        exploration_registry=exploration,
    )
    authorization_sha = canonical_sha256(validated_execution)
    budget_authorization_sha = canonical_sha256(validated_budget)
    proposal_sha = str(proposal["proposal_sha256"])
    if target_root.exists():
        receipt = read_json(target_root / "governance_receipt.json")
        _verify_embedded_hash(
            receipt,
            hash_field="receipt_sha256",
            artifact_name="independent_bo_governance_receipt",
        )
        if (
            read_json(target_root / "budget_contract.json")
            != target_budget.to_dict()
            or read_json(
                target_root / "execution_authorization.json"
            )
            != validated_execution
            or read_json(target_root / "budget_authorization.json")
            != validated_budget
            or receipt.get("proposal_sha256") != proposal_sha
            or receipt.get("status") != "prepared_zero_runs"
        ):
            raise RecoveryIndependentBOError(
                "independent_bo_existing_governance_mismatch"
            )
        return receipt

    source_summary = source_registry.summary()

    def finalize(
        staging: Path,
        migrated: AttemptRegistry,
    ) -> None:
        atomic_write_json(
            staging / "budget_contract.json",
            target_budget.to_dict(),
        )
        atomic_write_json(
            staging / "exploration_registry.json",
            exploration.to_dict(),
        )
        atomic_write_json(
            staging / "execution_authorization.json",
            validated_execution,
        )
        atomic_write_json(
            staging / "budget_authorization.json",
            validated_budget,
        )
        receipt = {
            "receipt_version": (
                "lyx_recovery_independent_bo_governance_v1"
            ),
            "status": "prepared_zero_runs",
            "proposal_sha256": proposal_sha,
            "authorization_sha256": authorization_sha,
            "budget_authorization_sha256": (
                budget_authorization_sha
            ),
            "source_budget_contract_hash": source_budget.sha256,
            "target_budget_contract_hash": target_budget.sha256,
            "dynamic_identity_stage": STAGE,
            "dynamic_identity_limit": EXPECTED_UNIQUE_BUDGET,
            "registered_new_identity_count": 0,
            "independent_bo_run_count": 0,
            "source_attempt_registry_summary": source_summary,
            "attempt_registry_summary": migrated.summary(),
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_write_json(
            staging / "governance_receipt.json",
            receipt,
        )

    source_registry.migrate_to(
        target_root / "attempt_registry.json",
        budget_contract=target_budget,
        amendment_request=amendment,
        authorization_receipt=validated_budget,
        target_exploration_registry=exploration,
        finalize_staging=finalize,
    )
    return read_json(target_root / "governance_receipt.json")


def _attempt_identity_from_item(
    item: Mapping[str, Any],
) -> AttemptIdentity:
    names = {field.name for field in fields(AttemptIdentity)}
    return AttemptIdentity(
        **{name: item[name] for name in names}
    )


def _selection_recovery_delay(
    metrics: Mapping[str, Any],
) -> float:
    raw = metrics.get("max_recovered_delay_s")
    if raw is not None:
        value = float(raw)
    elif int(metrics.get("recovery_episode_count", 0)) == 0:
        value = 0.0
    else:
        value = float(metrics["total_window_count"])
    if not math.isfinite(value) or value < 0.0:
        raise RecoveryIndependentBOError(
            "independent_bo_recovery_delay_invalid"
        )
    return value


def _constraint_values(
    *,
    metrics: Mapping[str, Any],
    spectral: Mapping[str, Any],
    current: Mapping[str, Any],
    independent: Mapping[str, Any],
    true_rise_applicable: bool,
) -> tuple[float, ...]:
    l10 = float(metrics["longest_e10_run_windows"])
    l20 = float(metrics["longest_e20_run_windows"])
    mae = float(metrics["final_motion_mae_bpm"])
    current_l10 = float(current["longest_e10_run_windows"])
    values = [
        (
            0.0
            if spectral.get("stability_pass") is True
            and spectral.get("spectral_gate_pass") is True
            else 1.0
        ),
        l10
        - max(
            10.0,
            float(independent["longest_e10_run_windows"])
            + 2.0,
        ),
        l20
        - max(
            2.0,
            float(independent["longest_e20_run_windows"]),
        ),
        mae - float(independent["final_motion_mae_bpm"]) - 2.0,
        float(metrics["right_censored_recovery_count"])
        - float(current["right_censored_recovery_count"]),
        (
            l10 - 19.999999
            if current_l10 <= 10.0
            else 0.0
        ),
        mae - float(current["final_motion_mae_bpm"]) - 2.0,
    ]
    if true_rise_applicable:
        candidate_rise = metrics.get("max_rise_underestimate_bpm")
        current_rise = current.get("max_rise_underestimate_bpm")
        if candidate_rise is None or current_rise is None:
            values.append(1.0)
        else:
            values.append(
                float(candidate_rise)
                - float(current_rise)
                - 2.0
            )
    else:
        values.append(0.0)
    if any(not math.isfinite(value) for value in values):
        raise RecoveryIndependentBOError(
            "independent_bo_constraint_nonfinite"
        )
    return tuple(values)


def _seed_budget_from_manifest(
    manifest: Mapping[str, Any],
) -> SeedSearchBudget:
    return SeedSearchBudget(
        lane_seeds=tuple(int(v) for v in manifest["lane_seeds"]),
        lane_unique_budget=int(manifest["lane_unique_budget"]),
        global_unique_budget=int(
            manifest["global_unique_budget"]
        ),
        n_startup_trials=int(manifest["n_startup_trials"]),
        fill_seed=int(manifest["fill_seed"]),
        unique_stall_limit=int(manifest["unique_stall_limit"]),
        objective_version=str(manifest["objective_version"]),
        constraints_version=str(
            manifest["constraints_version"]
        ),
    )


def _execute_search_cell(
    *,
    proposal: Mapping[str, Any],
    cell: Mapping[str, Any],
    space: Any,
    registry: AttemptRegistry,
    output_dir: Path,
    parallel_lanes: bool,
    progress_callback: Any | None,
) -> dict[str, Any]:
    cell_dir = (
        Path(output_dir)
        / str(cell["recovery_candidate_id"])
        / str(cell["record_id"])
    )
    completion_path = cell_dir / "cell_completion.json"
    if completion_path.is_file():
        completion = read_json(completion_path)
        _verify_embedded_hash(
            completion,
            hash_field="completion_sha256",
            artifact_name="independent_bo_cell_completion",
        )
        if (
            completion.get("proposal_sha256")
            != proposal.get("proposal_sha256")
            or completion.get("cell_sha256")
            != cell.get("cell_sha256")
            or completion.get("unique_candidate_count")
            != EXPECTED_CELL_UNIQUE_BUDGET
        ):
            raise RecoveryIndependentBOError(
                "independent_bo_cell_completion_mismatch"
            )
        return completion

    candidates = {
        candidate.candidate_id: candidate
        for candidate in space.candidates
    }
    rows: dict[str, dict[str, Any]] = {}
    candidate_locks = {
        candidate_id: threading.Lock()
        for candidate_id in candidates
    }
    rows_lock = threading.Lock()
    spectral_dir = Path(output_dir) / "spectral_audits"
    current = _mapping(
        "independent_bo_current_metrics",
        cell.get("current_metrics"),
    )
    independent = _mapping(
        "independent_bo_baseline_metrics",
        cell.get("independent_metrics"),
    )

    def evaluate_candidate(
        candidate: BOCandidate,
    ) -> dict[str, Any]:
        with candidate_locks[candidate.candidate_id]:
            with rows_lock:
                existing = rows.get(candidate.candidate_id)
            if existing is not None:
                return existing
            item = build_recovery_independent_bo_identity(
                proposal=proposal,
                cell=cell,
                candidate=candidate,
            )
            identity = _attempt_identity_from_item(item)
            registry.register_identity(identity)
            before = registry.matrix_execution_summary((identity,))
            if (
                before["failed_attempt_count"] != 0
                or before["retry_count"] != 0
            ):
                raise RecoveryIndependentBOError(
                    "independent_bo_retry_requires_new_proposal:"
                    + identity.sha256
                )
            result = stage_r_cache.execute_stage_r_identity(
                registry=registry,
                item=item,
                numerical_runner=run_stage_r_numerical_identity,
                spectral_audit_dir=spectral_dir,
                allow_retry=False,
            )
            metrics = _mapping(
                "independent_bo_result_metrics",
                result.get("metrics"),
            )
            spectral = _mapping(
                "independent_bo_result_spectral",
                result.get("spectral_audit"),
            )
            constraints = _constraint_values(
                metrics=metrics,
                spectral=spectral,
                current=current,
                independent=independent,
                true_rise_applicable=bool(
                    cell["true_rise_applicable"]
                ),
            )
            row = {
                "candidate_id": candidate.candidate_id,
                "identity": item,
                "identity_sha256": identity.sha256,
                "cache_hit": bool(result["cache_hit"]),
                "metrics": dict(metrics),
                "spectral_audit": dict(spectral),
                "constraints": list(constraints),
                "eligible": all(
                    value <= 0.0 for value in constraints
                ),
                "objective": float(
                    metrics["final_motion_mae_bpm"]
                ),
            }
            with rows_lock:
                rows[candidate.candidate_id] = row
            return row

    def evaluate(
        candidate: BOCandidate,
        _context: SearchRequestContext,
    ) -> SearchEvaluation:
        row = evaluate_candidate(candidate)
        return SearchEvaluation(
            objective=float(row["objective"]),
            constraints=tuple(
                float(value) for value in row["constraints"]
            ),
            metric_valid=True,
            eligible=bool(row["eligible"]),
        )

    seed_manifest = _mapping(
        "independent_bo_seed_manifest",
        proposal.get("seed_manifest"),
    )
    budget = _seed_budget_from_manifest(seed_manifest)
    experiment_identity = SearchExperimentIdentity(
        input_sha256s=(
            str(
                _mapping(
                    "independent_bo_template",
                    cell["template_identity"],
                )["raw_data_sha256"]
            ),
        ),
        reference_sha256s=(
            str(
                _mapping(
                    "independent_bo_template",
                    cell["template_identity"],
                )["reference_sha256"]
            ),
        ),
        git_commit=str(
            _mapping(
                "independent_bo_evaluation_source",
                proposal["evaluation_source_identity"],
            )["source_bundle_sha256"]
        ),
        run_config={
            "proposal_sha256": proposal["proposal_sha256"],
            "cell_sha256": cell["cell_sha256"],
            "stage": STAGE,
            "recovery_candidate_id": cell[
                "recovery_candidate_id"
            ],
            "record_id": cell["record_id"],
            "space_name": "physical_v1",
        },
        evaluation_version=budget.objective_version,
    )
    result = run_seed_search(
        space=space,
        output_dir=cell_dir / "search",
        experiment_identity=experiment_identity,
        evaluate=evaluate,
        budget=budget,
        parallel_lanes=parallel_lanes,
    )
    for candidate_id in result.global_candidate_ids:
        evaluate_candidate(candidates[candidate_id])
    selected_rows = [
        rows[candidate_id]
        for candidate_id in result.global_candidate_ids
    ]
    if len(selected_rows) != EXPECTED_CELL_UNIQUE_BUDGET:
        raise RecoveryIndependentBOError(
            "independent_bo_cell_unique_budget_incomplete"
        )
    selected = min(
        selected_rows,
        key=lambda row: (
            not bool(row["eligible"]),
            float(row["objective"]),
            str(row["candidate_id"]),
        ),
    )
    candidate_results = {
        "result_version": (
            "lyx_recovery_independent_bo_cell_results_v1"
        ),
        "proposal_sha256": proposal["proposal_sha256"],
        "cell_sha256": cell["cell_sha256"],
        "result_count": len(selected_rows),
        "results": sorted(
            selected_rows,
            key=lambda row: str(row["candidate_id"]),
        ),
    }
    candidate_results["result_sha256"] = canonical_sha256(
        candidate_results
    )
    cell_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        cell_dir / "candidate_results.json",
        candidate_results,
    )
    seed_audit = build_seed_stability_audit(
        result,
        candidates=candidates,
    )
    atomic_write_json(
        cell_dir / "seed_stability_audit.json",
        seed_audit,
    )
    identities = tuple(
        _attempt_identity_from_item(row["identity"])
        for row in selected_rows
    )
    matrix = registry.matrix_execution_summary(identities)
    completion = {
        "completion_version": (
            "lyx_recovery_independent_bo_cell_completion_v1"
        ),
        "status": "complete",
        "proposal_sha256": proposal["proposal_sha256"],
        "cell_sha256": cell["cell_sha256"],
        "record_id": cell["record_id"],
        "scene": cell["scene"],
        "recovery_candidate_id": cell[
            "recovery_candidate_id"
        ],
        "unique_candidate_count": len(selected_rows),
        "eligible_candidate_count": sum(
            bool(row["eligible"]) for row in selected_rows
        ),
        "seed_stability_candidate_count": len(
            result.seed_stability_candidate_ids
        ),
        "selected": selected,
        "matrix_execution_summary": matrix,
        "candidate_results_sha256": candidate_results[
            "result_sha256"
        ],
        "seed_stability_audit_sha256": canonical_sha256(
            seed_audit
        ),
    }
    completion["completion_sha256"] = canonical_sha256(completion)
    atomic_write_json(completion_path, completion)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "independent_bo_cell_complete",
                "record_id": cell["record_id"],
                "recovery_candidate_id": cell[
                    "recovery_candidate_id"
                ],
                "eligible_candidate_count": completion[
                    "eligible_candidate_count"
                ],
                "selected_candidate_id": selected["candidate_id"],
            }
        )
    return completion


def _build_upper_bound_selection(
    *,
    proposal: Mapping[str, Any],
    cell_completions: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    completion_by_coordinate = {
        (
            str(item["recovery_candidate_id"]),
            str(item["record_id"]),
        ): item
        for item in cell_completions
    }
    panel = tuple(
        RecoveryPanelRecord(
            record_id=str(record["record_id"]),
            scene=str(record["scene"]),
            true_rise_applicable=bool(
                record["true_rise_applicable"]
            ),
        )
        for record in _list(
            "independent_bo_record_panel",
            proposal.get("record_panel"),
        )
    )
    evaluations: list[RecoveryCandidateEvaluation] = []
    serialized: list[dict[str, Any]] = []
    for candidate in _list(
        "independent_bo_recovery_candidates",
        proposal.get("recovery_candidates"),
    ):
        candidate_map = _mapping(
            "independent_bo_recovery_candidate",
            candidate,
        )
        candidate_id = str(candidate_map["candidate_id"])
        records: list[RecoveryRecordEvaluation] = []
        for panel_record in panel:
            cell = _mapping(
                "independent_bo_cell_completion",
                completion_by_coordinate[
                    (candidate_id, panel_record.record_id)
                ],
            )
            selected = _mapping(
                "independent_bo_selected_result",
                cell.get("selected"),
            )
            metrics = _mapping(
                "independent_bo_selected_metrics",
                selected.get("metrics"),
            )
            spectral = _mapping(
                "independent_bo_selected_spectral",
                selected.get("spectral_audit"),
            )
            source_cell = next(
                _mapping("independent_bo_source_cell", raw)
                for raw in _list(
                    "independent_bo_search_cells",
                    proposal.get("search_cells"),
                )
                if raw["recovery_candidate_id"] == candidate_id
                and raw["record_id"] == panel_record.record_id
            )
            current = _mapping(
                "independent_bo_current_metrics",
                source_cell.get("current_metrics"),
            )
            independent = _mapping(
                "independent_bo_baseline_metrics",
                source_cell.get("independent_metrics"),
            )
            records.append(
                RecoveryRecordEvaluation(
                    record_id=panel_record.record_id,
                    sentinel_id=(
                        "per_record_independent_bo_physical_v1"
                    ),
                    scene=panel_record.scene,
                    spectral_gate_passed=bool(
                        spectral.get("stability_pass")
                        and spectral.get("spectral_gate_pass")
                    ),
                    l10=float(
                        metrics["longest_e10_run_windows"]
                    ),
                    l20=float(
                        metrics["longest_e20_run_windows"]
                    ),
                    mae=float(metrics["final_motion_mae_bpm"]),
                    independent_l10=float(
                        independent["longest_e10_run_windows"]
                    ),
                    independent_l20=float(
                        independent["longest_e20_run_windows"]
                    ),
                    independent_mae=float(
                        independent["final_motion_mae_bpm"]
                    ),
                    current_l10=float(
                        current["longest_e10_run_windows"]
                    ),
                    current_mae=float(
                        current["final_motion_mae_bpm"]
                    ),
                    recovery_delay=_selection_recovery_delay(
                        metrics
                    ),
                    right_censored_recovery_count=int(
                        metrics["right_censored_recovery_count"]
                    ),
                    current_right_censored_recovery_count=int(
                        current[
                            "right_censored_recovery_count"
                        ]
                    ),
                    true_rise_underestimate=(
                        float(
                            metrics[
                                "max_rise_underestimate_bpm"
                            ]
                        )
                        if panel_record.true_rise_applicable
                        and metrics.get(
                            "max_rise_underestimate_bpm"
                        )
                        is not None
                        else None
                    ),
                    current_true_rise_underestimate=(
                        float(
                            current[
                                "max_rise_underestimate_bpm"
                            ]
                        )
                        if panel_record.true_rise_applicable
                        and current.get(
                            "max_rise_underestimate_bpm"
                        )
                        is not None
                        else None
                    ),
                )
            )
        evaluation = RecoveryCandidateEvaluation(
            candidate_id=candidate_id,
            mechanism_complexity=int(
                candidate_map["mechanism_complexity"]
            ),
            records=tuple(records),
        )
        evaluations.append(evaluation)
        serialized.append(asdict(evaluation))
    selection = select_rank1_recovery_candidate_evaluations(
        evaluations,
        expected_records=panel,
        expected_sentinel_ids=(
            "per_record_independent_bo_physical_v1",
        ),
    )
    return selection, serialized


def execute_recovery_independent_bo_proposal(
    *,
    proposal_path: Path,
    authorization_path: Path,
    governance_dir: Path,
    output_dir: Path,
    repository_root: Path,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Execute or resume the exact 36-cell recovery independent BO."""

    proposal = read_json(Path(proposal_path).resolve())
    validate_recovery_independent_bo_preflight(
        proposal=proposal,
        repository_root=repository_root,
    )
    authorization = (
        validate_recovery_independent_bo_execution_authorization(
            proposal,
            receipt=read_json(Path(authorization_path).resolve()),
        )
    )
    authorization_sha = canonical_sha256(authorization)
    governance_root = Path(governance_dir).resolve()
    budget = BudgetContract.proposed_v13_recovery_independent_bo()
    if (
        read_json(governance_root / "budget_contract.json")
        != budget.to_dict()
        or read_json(
            governance_root / "execution_authorization.json"
        )
        != authorization
    ):
        raise RecoveryIndependentBOError(
            "independent_bo_governance_binding_mismatch"
        )
    exploration = _exploration_from_payload(
        read_json(governance_root / "exploration_registry.json")
    )
    registry = AttemptRegistry.open(
        governance_root / "attempt_registry.json",
        budget_contract=budget,
        exploration_registry=exploration,
    )
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    binding = {
        "binding_version": (
            "lyx_recovery_independent_bo_execution_binding_v1"
        ),
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": authorization_sha,
        "budget_contract_hash": budget.sha256,
        "solver_source_bundle_sha256": proposal[
            "solver_source_identity"
        ]["source_bundle_sha256"],
        "evaluation_source_bundle_sha256": proposal[
            "evaluation_source_identity"
        ]["source_bundle_sha256"],
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    binding_path = destination / "execution_binding.json"
    if binding_path.is_file():
        if read_json(binding_path) != binding:
            raise RecoveryIndependentBOError(
                "independent_bo_execution_binding_mismatch"
            )
    else:
        atomic_write_json(binding_path, binding)
    completion_path = destination / "completion.json"
    if completion_path.is_file():
        completion = read_json(completion_path)
        _verify_embedded_hash(
            completion,
            hash_field="completion_sha256",
            artifact_name="independent_bo_completion",
        )
        return completion

    space = build_bo_search_space("physical_v1")
    seed_manifest = _mapping(
        "independent_bo_seed_manifest",
        proposal["seed_manifest"],
    )
    cells = [
        _mapping("independent_bo_search_cell", raw)
        for raw in _list(
            "independent_bo_search_cells",
            proposal.get("search_cells"),
        )
    ]
    cell_completions: list[Mapping[str, Any]] = []
    for index, cell in enumerate(cells, start=1):
        cell_completion = _execute_search_cell(
            proposal=proposal,
            cell=cell,
            space=space,
            registry=registry,
            output_dir=destination / "cells",
            parallel_lanes=bool(
                seed_manifest["parallel_lanes"]
            ),
            progress_callback=progress_callback,
        )
        cell_completions.append(cell_completion)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "independent_bo_progress",
                    "completed_cells": index,
                    "total_cells": len(cells),
                }
            )
    selection, evaluations = _build_upper_bound_selection(
        proposal=proposal,
        cell_completions=cell_completions,
    )
    evaluation_payload = {
        "evaluation_version": (
            "lyx_recovery_independent_bo_upper_bound_v1"
        ),
        "proposal_sha256": proposal["proposal_sha256"],
        "candidate_evaluations": evaluations,
    }
    evaluation_payload["evaluation_sha256"] = canonical_sha256(
        evaluation_payload
    )
    atomic_write_json(
        destination / "formal_candidate_evaluations.json",
        evaluation_payload,
    )
    atomic_write_json(
        destination / "upper_bound_selection.json",
        selection,
    )
    selected_index = {
        "index_version": (
            "lyx_recovery_independent_bo_selected_index_v1"
        ),
        "proposal_sha256": proposal["proposal_sha256"],
        "selected_result_count": len(cell_completions),
        "results": [
            {
                "cell_sha256": item["cell_sha256"],
                "record_id": item["record_id"],
                "scene": item["scene"],
                "recovery_candidate_id": item[
                    "recovery_candidate_id"
                ],
                "eligible_candidate_count": item[
                    "eligible_candidate_count"
                ],
                "selected": item["selected"],
            }
            for item in cell_completions
        ],
    }
    selected_index["index_sha256"] = canonical_sha256(
        selected_index
    )
    atomic_write_json(
        destination / "selected_result_index.json",
        selected_index,
    )
    all_identities: list[AttemptIdentity] = []
    for item in cell_completions:
        cell_dir = (
            destination
            / "cells"
            / str(item["recovery_candidate_id"])
            / str(item["record_id"])
        )
        cell_results = read_json(
            cell_dir / "candidate_results.json"
        )
        all_identities.extend(
            _attempt_identity_from_item(
                _mapping(
                    "independent_bo_result_identity",
                    _mapping(
                        "independent_bo_candidate_result",
                        row,
                    )["identity"],
                )
            )
            for row in _list(
                "independent_bo_candidate_results",
                cell_results.get("results"),
            )
        )
    if (
        len(all_identities) != EXPECTED_UNIQUE_BUDGET
        or len({identity.sha256 for identity in all_identities})
        != EXPECTED_UNIQUE_BUDGET
    ):
        raise RecoveryIndependentBOError(
            "independent_bo_identity_matrix_incomplete"
        )
    registry.assert_complete_matrix(tuple(all_identities))
    matrix = registry.matrix_execution_summary(
        tuple(all_identities)
    )
    if (
        matrix["planned_identity_count"]
        != EXPECTED_UNIQUE_BUDGET
        or matrix["identity_with_solver_attempt_count"]
        + matrix["cache_only_identity_count"]
        != EXPECTED_UNIQUE_BUDGET
        or matrix["failed_attempt_count"] != 0
        or matrix["retry_count"] != 0
    ):
        raise RecoveryIndependentBOError(
            "independent_bo_execution_matrix_invalid"
        )
    decision_status = (
        "independent_bo_no_safe_recovery"
        if selection["status"] == "no_safe_recovery_candidate"
        else "independent_bo_recovery_mechanism_rescuable"
    )
    decision = {
        "decision_version": (
            "lyx_recovery_independent_bo_decision_v1"
        ),
        "status": decision_status,
        "proposal_sha256": proposal["proposal_sha256"],
        "upper_bound_selection_sha256": selection[
            "selection_sha256"
        ],
        "sample_in_recovery_candidate_id": selection[
            "provisional_recovery_id"
        ],
        "sample_in_backup_candidate_id": selection[
            "rollback_backup_id"
        ],
        "deployable_shared_parameters_selected": False,
        "automatic_stage_f_execution": False,
        "next_state": (
            "experiment_terminated_no_safe_recovery"
            if selection["status"]
            == "no_safe_recovery_candidate"
            else "requires_shared_filter_profile_design"
        ),
        "interpretation": (
            "per_record_reference_guided_upper_bound_not_a_deployable_method"
        ),
    }
    decision["decision_sha256"] = canonical_sha256(decision)
    atomic_write_json(
        destination / "decision_receipt.json",
        decision,
    )
    artifact_names = [
        "execution_binding.json",
        "formal_candidate_evaluations.json",
        "upper_bound_selection.json",
        "selected_result_index.json",
        "decision_receipt.json",
    ]
    artifacts = {
        name: file_sha256(destination / name)
        for name in artifact_names
    }
    governance_receipt = {
        "receipt_version": (
            "lyx_recovery_independent_bo_execution_governance_v1"
        ),
        "status": decision_status,
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": authorization_sha,
        "unique_identity_count": EXPECTED_UNIQUE_BUDGET,
        "independent_bo_run_count": 1,
        "matrix_execution_summary": matrix,
        "attempt_registry_summary_at_completion": (
            registry.summary()
        ),
        "decision_sha256": decision["decision_sha256"],
        "artifacts": artifacts,
    }
    governance_receipt["receipt_sha256"] = canonical_sha256(
        governance_receipt
    )
    atomic_write_json(
        governance_root / "independent_bo_governance_receipt.json",
        governance_receipt,
    )
    completion = {
        "completion_version": (
            "lyx_recovery_independent_bo_completion_v1"
        ),
        "status": decision_status,
        "evidence_class": (
            "development_reuse_sample_in_upper_bound"
        ),
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": authorization_sha,
        "search_cell_count": EXPECTED_SEARCH_CELL_COUNT,
        "unique_identity_count": EXPECTED_UNIQUE_BUDGET,
        "physical_solver_identity_count": matrix[
            "identity_with_solver_attempt_count"
        ],
        "cache_only_identity_count": matrix[
            "cache_only_identity_count"
        ],
        "failed_attempt_count": 0,
        "retry_count": 0,
        "independent_bo_run_count": 1,
        "sample_in_recovery_candidate_id": decision[
            "sample_in_recovery_candidate_id"
        ],
        "automatic_stage_f_execution": False,
        "next_state": decision["next_state"],
        "matrix_execution_summary": matrix,
        "artifacts": artifacts,
        "governance_receipt_sha256": governance_receipt[
            "receipt_sha256"
        ],
    }
    completion["completion_sha256"] = canonical_sha256(
        completion
    )
    atomic_write_json(completion_path, completion)
    return completion
