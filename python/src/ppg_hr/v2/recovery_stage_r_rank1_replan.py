"""Governed Stage R repair for the validated rank-1 p25 filter."""

from __future__ import annotations

import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import recovery_stage_r_cache as stage_r_cache
from .experiment_freeze_utils import runtime_source_identity
from .phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    read_json,
)
from .recovery_contracts import canonical_sha256, require_sha256
from .recovery_experiment_governance import (
    AttemptIdentity,
    AttemptRegistry,
    BudgetAmendmentRequest,
    BudgetContract,
    ExplorationRegistry,
)
from .recovery_selection import (
    RecoveryCandidateEvaluation,
    RecoveryPanelRecord,
    RecoveryRecordEvaluation,
    recovery_selection_contract_rank1_replan_v1,
    select_rank1_recovery_candidate_evaluations,
)
from .recovery_stage_r_common import StageRPlanError
from .recovery_stage_r_experiment import (
    run_stage_r_numerical_identity,
    stage_r_metric_contract_v1,
    stage_r_spectral_gate_contract_v2,
)


class StageRRank1Error(StageRPlanError):
    """The rank-1 Stage R repair violates its frozen contract."""


class StageRRank1AuthorizationError(StageRRank1Error):
    """The exact 36-identity Stage R repair has not been approved."""


_STAGE = "recovery_sentinel_rank1_replan"
_ATTEMPT_KIND = "formal"
_AUTHORIZATION_STATE = "awaiting_human_budget_decision"
_PROFILE_ID = "p25-short-low-rank1-v1"
_BASE_PROFILE_ID = "p25-short-low"
_EXPECTED_RECORD_COUNT = 12
_EXPECTED_CANDIDATE_COUNT = 3
_EXPECTED_IDENTITY_COUNT = 36
_EXPECTED_SCENE_COUNTS = {
    "jianpan": 3,
    "kaihe": 3,
    "run": 3,
    "xiezi": 3,
}
_CONTROL_RECOVERY_ID = "current_fixed_floor_control_v1"
_CONTROL_PENALTY_ID = "current_soft_penalty_control_v1"
_SOURCE_SENTINEL_ROLE = "conservative"
_PROPOSAL_ARTIFACT_NAMES = {
    "budget_amendment_request.json",
    "budget_contract_v11.json",
    "metric_contract.json",
    "proposal_receipt.json",
    "recovery_selection_contract.json",
    "source_identity.json",
    "spectral_gate_contract.json",
    "stage_r_rank1_replan_proposal.json",
}


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageRRank1Error(f"{name}_must_be_mapping")
    return value


def _require_list(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise StageRRank1Error(f"{name}_must_be_list")
    return value


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    artifact_name: str,
) -> str:
    value = payload.get(hash_field)
    if not isinstance(value, str):
        raise StageRRank1Error(
            f"{artifact_name}_{hash_field}_missing"
        )
    try:
        require_sha256(hash_field, value)
    except ValueError as error:
        raise StageRRank1Error(
            f"{artifact_name}_{hash_field}_invalid"
        ) from error
    unsigned = {
        key: item for key, item in payload.items() if key != hash_field
    }
    if canonical_sha256(unsigned) != value:
        raise StageRRank1Error(
            f"{artifact_name}_{hash_field}_mismatch"
        )
    return value


def _candidate_registry_payload(
    registry: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    registry_sha = _verify_embedded_hash(
        registry,
        hash_field="registry_sha256",
        artifact_name="stage_r_rank1_recovery_registry",
    )
    del registry_sha
    raw_candidates = _require_list(
        "stage_r_rank1_recovery_candidates",
        registry.get("candidates"),
    )
    if (
        registry.get("control_candidate_id") != _CONTROL_RECOVERY_ID
        or len(raw_candidates) != _EXPECTED_CANDIDATE_COUNT
    ):
        raise StageRRank1Error(
            "stage_r_rank1_recovery_candidate_registry_invalid"
        )
    candidates: list[dict[str, Any]] = []
    for raw in raw_candidates:
        candidate = dict(
            _require_mapping(
                "stage_r_rank1_recovery_candidate",
                raw,
            )
        )
        candidate_sha = _verify_embedded_hash(
            candidate,
            hash_field="candidate_sha256",
            artifact_name=(
                "stage_r_rank1_recovery_candidate:"
                + str(candidate.get("candidate_id", ""))
            ),
        )
        candidates.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "candidate_sha256": candidate_sha,
                "mechanism_complexity": int(
                    candidate["mechanism_complexity"]
                ),
                "constants": dict(
                    _require_mapping(
                        "stage_r_rank1_candidate_constants",
                        candidate.get("constants"),
                    )
                ),
            }
        )
    candidates.sort(
        key=lambda item: (
            item["mechanism_complexity"],
            item["candidate_id"],
        )
    )
    if len({item["candidate_id"] for item in candidates}) != 3:
        raise StageRRank1Error(
            "stage_r_rank1_duplicate_recovery_candidate"
        )
    return tuple(candidates)


def _validate_penalty_registry(
    registry: Mapping[str, Any],
) -> str:
    _verify_embedded_hash(
        registry,
        hash_field="registry_sha256",
        artifact_name="stage_r_rank1_penalty_registry",
    )
    control = registry.get("control_penalty_id")
    if control != _CONTROL_PENALTY_ID:
        raise StageRRank1Error(
            "stage_r_rank1_control_penalty_id_mismatch"
        )
    candidates = _require_list(
        "stage_r_rank1_penalty_candidates",
        registry.get("candidates"),
    )
    if control not in {
        str(
            _require_mapping(
                "stage_r_rank1_penalty_candidate",
                item,
            ).get("penalty_id")
        )
        for item in candidates
    }:
        raise StageRRank1Error(
            "stage_r_rank1_control_penalty_invalid"
        )
    return control


def _validate_baseline_sources(
    *,
    baseline_manifest: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    source_by_record: Mapping[str, Mapping[str, Any]],
    prior_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    manifest_records = {
        str(item["sample_id"]): _require_mapping(
            "stage_r_rank1_baseline_manifest_record",
            item,
        )
        for item in _require_list(
            "stage_r_rank1_baseline_manifest_records",
            baseline_manifest.get("records"),
        )
    }
    metric_records = {
        str(item["sample_id"]): _require_mapping(
            "stage_r_rank1_baseline_metric_record",
            item,
        )
        for item in _require_list(
            "stage_r_rank1_baseline_metric_records",
            baseline_metrics.get("records"),
        )
    }
    expected_ids = set(source_by_record)
    if (
        set(manifest_records) != expected_ids
        or set(metric_records) != expected_ids
        or len(manifest_records) != 12
        or len(metric_records) != 12
    ):
        raise StageRRank1Error(
            "stage_r_rank1_baseline_record_panel_mismatch"
        )
    panel: list[dict[str, Any]] = []
    for record_id in sorted(expected_ids):
        source = source_by_record[record_id]
        prior = prior_records[record_id]
        manifest_record = manifest_records[record_id]
        metric_record = metric_records[record_id]
        expected = {
            "scene": source["scene"],
            "data_sha256": source["raw_data_sha256"],
            "reference_sha256": source["reference_sha256"],
        }
        if any(
            manifest_record.get(name) != value
            or metric_record.get(name) != value
            for name, value in expected.items()
        ) or list(manifest_record.get("method_names", [])) != list(
            prior["method_names"]
        ):
            raise StageRRank1Error(
                "stage_r_rank1_baseline_identity_drift:" + record_id
            )
        metrics = _require_mapping(
            "stage_r_rank1_baseline_metrics",
            metric_record.get("metrics"),
        )
        if (
            metrics.get("metric_contract_version")
            != "lyx_recovery_profile_metric_v1"
            or metrics.get("final_method")
            != prior["method_names"][-1]
        ):
            raise StageRRank1Error(
                "stage_r_rank1_baseline_metric_contract_drift:"
                + record_id
            )
        panel.append(
            {
                "record_id": record_id,
                "scene": source["scene"],
                "data_sha256": source["raw_data_sha256"],
                "reference_sha256": source["reference_sha256"],
                "method_names": list(prior["method_names"]),
                "true_rise_applicable": bool(
                    prior["true_rise_applicable"]
                ),
                "solver_identity_sha256": metric_record.get(
                    "solver_identity_sha256"
                ),
                "solver_result_sha256": metric_record.get(
                    "solver_result_sha256"
                ),
                "metrics_sha256": canonical_sha256(metrics),
            }
        )
    return tuple(panel)


def _validate_rank1_source(
    *,
    proposal: Mapping[str, Any],
    completion: Mapping[str, Any],
    decision: Mapping[str, Any],
    manifest: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="stage_r_rank1_source_proposal",
    )
    completion_sha = _verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="stage_r_rank1_source_completion",
    )
    decision_sha = _verify_embedded_hash(
        decision,
        hash_field="decision_sha256",
        artifact_name="stage_r_rank1_source_decision",
    )
    manifest_sha = _verify_embedded_hash(
        manifest,
        hash_field="manifest_sha256",
        artifact_name="stage_r_rank1_source_manifest",
    )
    if (
        completion.get("proposal_sha256") != proposal_sha
        or completion.get("status")
        != "rank1_filter_revision_validated"
        or completion.get("diagnostic_result_count") != 12
        or decision.get("proposal_sha256") != proposal_sha
        or decision.get("decision")
        != "rank1_filter_revision_validated"
        or decision.get("record_count") != 12
        or decision.get("exact_rank1_reproduction_count") != 12
        or decision.get("complete_gate_pass_count") != 12
        or decision.get("single_reference_stage_count") != 12
        or manifest.get("proposal_sha256") != proposal_sha
        or manifest.get("result_count") != 12
    ):
        raise StageRRank1Error(
            "stage_r_rank1_source_state_invalid"
        )
    raw_identities = _require_list(
        "stage_r_rank1_source_identities",
        proposal.get("identities"),
    )
    entries = _require_list(
        "stage_r_rank1_source_manifest_entries",
        manifest.get("results"),
    )
    identities_by_record = {
        str(item["record_id"]): _require_mapping(
            "stage_r_rank1_source_identity",
            item,
        )
        for item in raw_identities
        if isinstance(item, Mapping)
    }
    entries_by_record = {
        str(item["record_id"]): _require_mapping(
            "stage_r_rank1_source_manifest_entry",
            item,
        )
        for item in entries
        if isinstance(item, Mapping)
    }
    if (
        len(identities_by_record) != 12
        or len(entries_by_record) != 12
        or set(identities_by_record) != set(entries_by_record)
        or set(identities_by_record) != set(results)
    ):
        raise StageRRank1Error(
            "stage_r_rank1_source_panel_invalid"
        )
    for record_id, identity in identities_by_record.items():
        config = _require_mapping(
            "stage_r_rank1_source_identity_config",
            identity.get("config"),
        )
        source_result = _require_mapping(
            "stage_r_rank1_source_result",
            results[record_id],
        )
        result_sha = _verify_embedded_hash(
            source_result,
            hash_field="result_sha256",
            artifact_name=(
                "stage_r_rank1_source_result:" + record_id
            ),
        )
        entry = entries_by_record[record_id]
        if (
            config.get("adaptive_reference_stage_limit") != 1
            or source_result.get("proposal_sha256") != proposal_sha
            or source_result.get("identity_sha256")
            != identity.get("identity_sha256")
            or source_result.get("exact_rank1_reproduction_pass")
            is not True
            or source_result.get("all_gate_pass") is not True
            or source_result.get(
                "single_reference_stage_per_valid_window"
            )
            is not True
            or entry.get("identity_sha256")
            != identity.get("identity_sha256")
            or entry.get("result_sha256") != result_sha
        ):
            raise StageRRank1Error(
                "stage_r_rank1_source_result_invalid:" + record_id
            )
    scene_counts = {
        scene: sum(
            str(item["scene"]) == scene
            for item in identities_by_record.values()
        )
        for scene in _EXPECTED_SCENE_COUNTS
    }
    if scene_counts != _EXPECTED_SCENE_COUNTS:
        raise StageRRank1Error(
            "stage_r_rank1_source_scene_panel_invalid"
        )
    return identities_by_record, {
        "proposal_sha256": proposal_sha,
        "completion_sha256": completion_sha,
        "decision_sha256": decision_sha,
        "manifest_sha256": manifest_sha,
        "status": "rank1_filter_revision_validated",
        "record_count": 12,
    }


def _validate_prior_stage_r(
    proposal: Mapping[str, Any],
) -> tuple[
    dict[str, Mapping[str, Any]],
    tuple[Mapping[str, Any], ...],
    dict[str, Any],
]:
    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="stage_r_rank1_prior_stage_r_proposal",
    )
    if (
        proposal.get("diagnostic_unique_budget") != 60
        or proposal.get("formal_unique_budget") != 108
        or proposal.get("independent_bo_authorized") is not False
    ):
        raise StageRRank1Error(
            "stage_r_rank1_prior_stage_r_contract_invalid"
        )
    records = _require_list(
        "stage_r_rank1_prior_records",
        proposal.get("record_panel"),
    )
    if len(records) != 12:
        raise StageRRank1Error(
            "stage_r_rank1_prior_record_panel_invalid"
        )
    record_panel = {
        str(item["record_id"]): _require_mapping(
            "stage_r_rank1_prior_record",
            item,
        )
        for item in records
        if isinstance(item, Mapping)
    }
    identities = _require_list(
        "stage_r_rank1_prior_identities",
        proposal.get("identities"),
    )
    base_by_record: dict[str, Mapping[str, Any]] = {}
    for raw in identities:
        item = _require_mapping(
            "stage_r_rank1_prior_identity",
            raw,
        )
        if (
            item.get("stage") == "recovery_sentinel"
            and item.get("recovery_candidate_id")
            == _CONTROL_RECOVERY_ID
            and item.get("sentinel_role") == _SOURCE_SENTINEL_ROLE
        ):
            base_by_record[str(item["record_id"])] = item
    if (
        len(record_panel) != 12
        or set(base_by_record) != set(record_panel)
    ):
        raise StageRRank1Error(
            "stage_r_rank1_prior_base_config_panel_invalid"
        )
    candidates = tuple(
        _require_mapping(
            "stage_r_rank1_prior_candidate",
            item,
        )
        for item in _require_list(
            "stage_r_rank1_prior_candidates",
            proposal.get("recovery_candidates"),
        )
    )
    return base_by_record, candidates, {
        "proposal_sha256": proposal_sha,
        "historical_threshold_diagnostic_count": 60,
        "historical_formal_identity_count": 108,
        "historical_results_reusable_for_selection": False,
        "non_reuse_reason": (
            "solver_source_and_filter_structure_changed_to_rank1"
        ),
    }


def _identity_item(
    *,
    parent_experiment_id: str,
    solver_hash: str,
    evaluation_hash: str,
    metric_contract_hash: str,
    source_identity: Mapping[str, Any],
    prior_identity: Mapping[str, Any],
    prior_record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    penalty_candidate_id: str,
    filter_revision_contract_hash: str,
    spectral_gate_contract_hash: str,
) -> dict[str, Any]:
    prior_config = _require_mapping(
        "stage_r_rank1_prior_config",
        prior_identity.get("config"),
    )
    parameters = dict(
        _require_mapping(
            "stage_r_rank1_prior_parameters",
            prior_config.get("parameters"),
        )
    )
    parameters.update(
        {
            "analysis_scope": "full",
            "adaptive_filter": "lms",
            "algorithm_preset": "lite",
            "reference_groups_order": ["HF"],
            "adaptive_reference_stage_limit": 1,
            "fs_target": 25,
            "lms_mu_base": 0.008,
            "lms_mu_min": 1e-6,
            "max_order": 1,
            "recovery_candidate_id": candidate["candidate_id"],
            "penalty_candidate_id": penalty_candidate_id,
        }
    )
    config = {
        "data_path": source_identity["data_path"],
        "reference_path": source_identity["reference_path"],
        "method_names": list(prior_record["method_names"]),
        "parameters": parameters,
    }
    identity = AttemptIdentity(
        solver_hash=solver_hash,
        config_hash=canonical_sha256(config),
        metric_contract_hash=metric_contract_hash,
        evaluation_hash=evaluation_hash,
        data_sha256=str(source_identity["data_sha256"]),
        record_id=str(source_identity["record_id"]),
        stage=_STAGE,
        attempt_kind=_ATTEMPT_KIND,
        parent_experiment_id=parent_experiment_id,
    )
    return {
        **identity.to_dict(),
        "scene": source_identity["scene"],
        "data_path": source_identity["data_path"],
        "reference_path": source_identity["reference_path"],
        "raw_data_sha256": source_identity["raw_data_sha256"],
        "reference_sha256": source_identity["reference_sha256"],
        "method_names": list(prior_record["method_names"]),
        "true_rise_applicable": bool(
            prior_record["true_rise_applicable"]
        ),
        "config": config,
        "sentinel_role": "fixed_rank1",
        "filter_profile_id": _PROFILE_ID,
        "base_filter_profile_id": _BASE_PROFILE_ID,
        "filter_profile_sha256": filter_revision_contract_hash,
        "spectral_gate_contract_sha256": (
            spectral_gate_contract_hash
        ),
        "physical_memory_ms": 40,
        "actual_taps": 1,
        "nominal_mu": 0.008,
        "adaptive_reference_stage_limit": 1,
        "spectral_audit_required": True,
        "candidate_min_bpm": _require_mapping(
            "stage_r_rank1_candidate_constants",
            candidate["constants"],
        ).get("candidate_min_bpm"),
        "recovery_candidate_id": candidate["candidate_id"],
        "recovery_candidate_sha256": candidate[
            "candidate_sha256"
        ],
        "mechanism_complexity": candidate[
            "mechanism_complexity"
        ],
        "penalty_candidate_id": penalty_candidate_id,
        "source_rank1_identity_sha256": source_identity[
            "identity_sha256"
        ],
    }


def build_stage_r_rank1_replan_proposal(
    *,
    rank1_proposal: Mapping[str, Any],
    rank1_completion: Mapping[str, Any],
    rank1_decision: Mapping[str, Any],
    rank1_manifest: Mapping[str, Any],
    rank1_results: Mapping[str, Mapping[str, Any]],
    prior_stage_r_proposal: Mapping[str, Any],
    prior_stage_r_governance_receipt: Mapping[str, Any],
    recovery_registry: Mapping[str, Any],
    penalty_registry: Mapping[str, Any],
    baseline_manifest: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    parent_experiment_id: str,
    solver_hash: str,
    evaluation_hash: str,
    metric_contract_hash: str,
    spectral_gate_contract_hash: str,
    selection_contract_hash: str,
    budget_contract_hash: str,
    source_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the exact 36-identity proposal without running a solver."""

    if not parent_experiment_id:
        raise StageRRank1Error(
            "stage_r_rank1_parent_experiment_id_empty"
        )
    for name, value in (
        ("solver_hash", solver_hash),
        ("evaluation_hash", evaluation_hash),
        ("metric_contract_hash", metric_contract_hash),
        (
            "spectral_gate_contract_hash",
            spectral_gate_contract_hash,
        ),
        ("selection_contract_hash", selection_contract_hash),
        ("budget_contract_hash", budget_contract_hash),
    ):
        require_sha256(name, value)
    source_by_record, rank1_source = _validate_rank1_source(
        proposal=rank1_proposal,
        completion=rank1_completion,
        decision=rank1_decision,
        manifest=rank1_manifest,
        results=rank1_results,
    )
    rank1_frozen = _require_mapping(
        "stage_r_rank1_source_frozen_contracts",
        rank1_proposal.get("frozen_contracts"),
    )
    filter_revision_contract_hash = str(
        rank1_frozen.get("revision_contract_hash", "")
    )
    require_sha256(
        "filter_revision_contract_hash",
        filter_revision_contract_hash,
    )
    (
        prior_base_by_record,
        prior_candidates,
        prior_stage,
    ) = _validate_prior_stage_r(prior_stage_r_proposal)
    prior_receipt_sha = _verify_embedded_hash(
        prior_stage_r_governance_receipt,
        hash_field="receipt_sha256",
        artifact_name="stage_r_rank1_prior_governance_receipt",
    )
    if (
        prior_stage_r_governance_receipt.get("proposal_sha256")
        != prior_stage["proposal_sha256"]
        or prior_stage_r_governance_receipt.get("status")
        != "no_safe_recovery_candidate"
        or prior_stage_r_governance_receipt.get(
            "diagnostic_unique_identities"
        )
        != 60
        or prior_stage_r_governance_receipt.get(
            "formal_unique_identities"
        )
        != 108
    ):
        raise StageRRank1Error(
            "stage_r_rank1_prior_governance_state_invalid"
        )
    prior_stage["governance_receipt_sha256"] = prior_receipt_sha
    prior_stage["status"] = "no_safe_recovery_candidate"
    candidates = _candidate_registry_payload(recovery_registry)
    if [
        (
            str(item["candidate_id"]),
            str(item["candidate_sha256"]),
            int(item["mechanism_complexity"]),
        )
        for item in prior_candidates
    ] != [
        (
            item["candidate_id"],
            item["candidate_sha256"],
            item["mechanism_complexity"],
        )
        for item in candidates
    ]:
        raise StageRRank1Error(
            "stage_r_rank1_candidate_registry_drift"
        )
    penalty_candidate_id = _validate_penalty_registry(
        penalty_registry
    )
    prior_records = {
        str(item["record_id"]): item
        for item in _require_list(
            "stage_r_rank1_prior_record_panel",
            prior_stage_r_proposal.get("record_panel"),
        )
    }
    for record_id, source in source_by_record.items():
        prior = _require_mapping(
            "stage_r_rank1_prior_record",
            prior_records[record_id],
        )
        if (
            source.get("data_sha256")
            != prior.get("combined_data_sha256")
            or source.get("raw_data_sha256")
            != prior.get("data_sha256")
            or source.get("reference_sha256")
            != prior.get("reference_sha256")
            or source.get("scene") != prior.get("scene")
        ):
            raise StageRRank1Error(
                "stage_r_rank1_record_identity_drift:" + record_id
            )
    baseline_identity_panel = _validate_baseline_sources(
        baseline_manifest=baseline_manifest,
        baseline_metrics=baseline_metrics,
        source_by_record=source_by_record,
        prior_records={
            record_id: _require_mapping(
                "stage_r_rank1_prior_record",
                item,
            )
            for record_id, item in prior_records.items()
        },
    )
    identities = [
        _identity_item(
            parent_experiment_id=parent_experiment_id,
            solver_hash=solver_hash,
            evaluation_hash=evaluation_hash,
            metric_contract_hash=metric_contract_hash,
            source_identity=source_by_record[record_id],
            prior_identity=prior_base_by_record[record_id],
            prior_record=prior_records[record_id],
            candidate=candidate,
            penalty_candidate_id=penalty_candidate_id,
            filter_revision_contract_hash=(
                filter_revision_contract_hash
            ),
            spectral_gate_contract_hash=spectral_gate_contract_hash,
        )
        for candidate in candidates
        for record_id in sorted(source_by_record)
    ]
    identity_hashes = [
        str(item["identity_sha256"]) for item in identities
    ]
    if (
        len(identities) != _EXPECTED_IDENTITY_COUNT
        or len(set(identity_hashes)) != _EXPECTED_IDENTITY_COUNT
    ):
        raise StageRRank1Error(
            "stage_r_rank1_identity_matrix_invalid"
        )
    record_panel = [
        {
            "record_id": record_id,
            "scene": source_by_record[record_id]["scene"],
            "raw_data_sha256": source_by_record[record_id][
                "raw_data_sha256"
            ],
            "reference_sha256": source_by_record[record_id][
                "reference_sha256"
            ],
            "combined_data_sha256": source_by_record[record_id][
                "data_sha256"
            ],
            "method_names": list(
                prior_records[record_id]["method_names"]
            ),
            "true_rise_applicable": bool(
                prior_records[record_id]["true_rise_applicable"]
            ),
        }
        for record_id in sorted(source_by_record)
    ]
    proposal: dict[str, Any] = {
        "proposal_version": "lyx_stage_r_rank1_replan_proposal_v1",
        "status": "awaiting_human_execution_authorization",
        "authorization_state": _AUTHORIZATION_STATE,
        "parent_experiment_id": parent_experiment_id,
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "filter_revision_id": _PROFILE_ID,
        "base_filter_profile_id": _BASE_PROFILE_ID,
        "adaptive_reference_stage_limit": 1,
        "new_threshold_diagnostic_unique_budget": 0,
        "historical_threshold_diagnostic_count": 60,
        "historical_stage_r_formal_identity_count": 108,
        "formal_unique_budget": 36,
        "unique_budget": 36,
        "retry_limit": 1,
        "worst_case_attempt_budget": 72,
        "diagnostic_run_count": 0,
        "formal_run_count": 0,
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": True,
        "historical_stage_r": prior_stage,
        "upstream_rank1_revision": rank1_source,
        "frozen_contracts": {
            "solver_hash": solver_hash,
            "evaluation_hash": evaluation_hash,
            "metric_contract_hash": metric_contract_hash,
            "spectral_gate_contract_hash": (
                spectral_gate_contract_hash
            ),
            "selection_contract_hash": (
                selection_contract_hash
            ),
            "filter_revision_contract_hash": (
                filter_revision_contract_hash
            ),
            "budget_contract_hash": budget_contract_hash,
            "recovery_candidate_registry_hash": (
                recovery_registry["registry_sha256"]
            ),
            "penalty_registry_hash": (
                penalty_registry["registry_sha256"]
            ),
        },
        "record_panel": record_panel,
        "record_panel_sha256": canonical_sha256(record_panel),
        "baseline_identity_panel": list(
            baseline_identity_panel
        ),
        "baseline_identity_panel_sha256": canonical_sha256(
            baseline_identity_panel
        ),
        "recovery_candidates": list(candidates),
        "identity_sha256": identity_hashes,
        "identity_panel_sha256": canonical_sha256(
            identity_hashes
        ),
        "identities": identities,
    }
    if source_artifacts is not None:
        proposal["source_artifacts"] = {
            str(name): dict(value)
            for name, value in sorted(source_artifacts.items())
        }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def validate_stage_r_rank1_execution_authorization(
    proposal: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require approval of the exact matrix while keeping BO disabled."""

    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="stage_r_rank1_proposal",
    )
    if receipt is None or receipt.get("approved") is not True:
        raise StageRRank1AuthorizationError(
            "stage_r_rank1_execution_authorization_required"
        )
    frozen = _require_mapping(
        "stage_r_rank1_frozen_contracts",
        proposal.get("frozen_contracts"),
    )
    expected = {
        "decision_state": _AUTHORIZATION_STATE,
        "proposal_sha256": proposal_sha,
        "stage": _STAGE,
        "profile_design_rule_hash": frozen.get(
            "selection_contract_hash"
        ),
        "record_manifest_hash": proposal.get(
            "record_panel_sha256"
        ),
        "added_unique_identities": 36,
        "normal_unique_identity_limit": 888,
        "max_unique_identities": 900,
        "max_attempts": 1800,
        "budget_contract_hash": frozen.get(
            "budget_contract_hash"
        ),
        "identity_panel_sha256": proposal.get(
            "identity_panel_sha256"
        ),
        "baseline_identity_panel_sha256": proposal.get(
            "baseline_identity_panel_sha256"
        ),
        "solver_hash": frozen.get("solver_hash"),
        "evaluation_hash": frozen.get("evaluation_hash"),
        "metric_contract_hash": frozen.get(
            "metric_contract_hash"
        ),
        "spectral_gate_contract_hash": frozen.get(
            "spectral_gate_contract_hash"
        ),
        "selection_contract_hash": frozen.get(
            "selection_contract_hash"
        ),
        "filter_revision_contract_hash": frozen.get(
            "filter_revision_contract_hash"
        ),
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": True,
    }
    mismatched = sorted(
        name
        for name, value in expected.items()
        if receipt.get(name) != value
    )
    if mismatched:
        raise StageRRank1AuthorizationError(
            "stage_r_rank1_authorization_mismatch:"
            + ",".join(mismatched)
        )
    for name in ("approved_at", "approved_by"):
        if (
            not isinstance(receipt.get(name), str)
            or not receipt[name]
        ):
            raise StageRRank1AuthorizationError(
                "stage_r_rank1_authorization_"
                + name
                + "_invalid"
            )
    return dict(receipt)


def _repository_root_from_source_root(source_root: Path) -> Path:
    root = Path(source_root).resolve()
    if root.name != "src" or root.parent.name != "python":
        raise StageRRank1Error(
            "stage_r_rank1_source_root_invalid"
        )
    return root.parent.parent


def _source_artifact_payload(
    artifacts: Mapping[str, Path],
    *,
    repository_root: Path,
) -> dict[str, dict[str, str]]:
    payload: dict[str, dict[str, str]] = {}
    for name, path in artifacts.items():
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(repository_root)
        except ValueError as error:
            raise StageRRank1Error(
                "stage_r_rank1_source_outside_repository:" + name
            ) from error
        payload[name] = {
            "path": relative.as_posix(),
            "path_base": "repository_root",
            "file_sha256": file_sha256(resolved),
        }
    return payload


def _load_rank1_result_artifacts(
    *,
    manifest_path: Path,
    artifacts: dict[str, Path],
) -> dict[str, Mapping[str, Any]]:
    manifest = read_json(manifest_path)
    results: dict[str, Mapping[str, Any]] = {}
    for raw in _require_list(
        "stage_r_rank1_manifest_results",
        manifest.get("results"),
    ):
        entry = _require_mapping(
            "stage_r_rank1_manifest_entry",
            raw,
        )
        record_id = str(entry["record_id"])
        path = (
            manifest_path.parent / str(entry["path"])
        ).resolve()
        if (
            not path.is_file()
            or file_sha256(path) != entry.get("file_sha256")
        ):
            raise StageRRank1Error(
                "stage_r_rank1_source_result_file_mismatch:"
                + record_id
            )
        artifacts[f"rank1_result:{record_id}"] = path
        results[record_id] = read_json(path)
    return results


def propose_stage_r_rank1_replan(
    *,
    rank1_proposal_path: Path,
    rank1_completion_path: Path,
    rank1_decision_path: Path,
    rank1_manifest_path: Path,
    prior_stage_r_proposal_path: Path,
    prior_stage_r_governance_receipt_path: Path,
    recovery_registry_path: Path,
    penalty_registry_path: Path,
    baseline_manifest_path: Path,
    baseline_metrics_path: Path,
    source_budget_contract_path: Path,
    spec_path: Path,
    output_dir: Path,
    source_root: Path,
    parent_experiment_id: str,
) -> dict[str, Any]:
    """Publish an atomic zero-run proposal awaiting exact approval."""

    artifacts = {
        "rank1_proposal": Path(rank1_proposal_path).resolve(),
        "rank1_completion": Path(rank1_completion_path).resolve(),
        "rank1_decision": Path(rank1_decision_path).resolve(),
        "rank1_manifest": Path(rank1_manifest_path).resolve(),
        "prior_stage_r_proposal": Path(
            prior_stage_r_proposal_path
        ).resolve(),
        "prior_stage_r_governance_receipt": Path(
            prior_stage_r_governance_receipt_path
        ).resolve(),
        "recovery_registry": Path(
            recovery_registry_path
        ).resolve(),
        "penalty_registry": Path(
            penalty_registry_path
        ).resolve(),
        "baseline_manifest": Path(
            baseline_manifest_path
        ).resolve(),
        "baseline_metrics": Path(baseline_metrics_path).resolve(),
        "source_budget_contract": Path(
            source_budget_contract_path
        ).resolve(),
        "experiment_spec": Path(spec_path).resolve(),
    }
    missing = [
        name for name, path in artifacts.items() if not path.is_file()
    ]
    if missing:
        raise StageRRank1Error(
            "stage_r_rank1_source_missing:" + ",".join(missing)
        )
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise StageRRank1Error(
            "stage_r_rank1_destination_exists"
        )
    source_budget = BudgetContract.proposed_v10_rank1_filter_revision()
    if (
        read_json(artifacts["source_budget_contract"])
        != source_budget.to_dict()
    ):
        raise StageRRank1Error(
            "stage_r_rank1_source_budget_mismatch"
        )
    rank1_results = _load_rank1_result_artifacts(
        manifest_path=artifacts["rank1_manifest"],
        artifacts=artifacts,
    )
    source_root = Path(source_root).resolve()
    repository_root = _repository_root_from_source_root(source_root)
    source_artifacts = _source_artifact_payload(
        artifacts,
        repository_root=repository_root,
    )
    source_identity = runtime_source_identity(
        source_root,
        root_modules=(
            "ppg_hr.v2.recovery_stage_r_rank1_replan",
            "ppg_hr.v2.recovery_stage_r_rank1_replan_runner",
            "ppg_hr.v2.recovery_stage_r_experiment",
            "ppg_hr.v2.recovery_stage_r_cache",
            "ppg_hr.v2.recovery_selection",
            "ppg_hr.v2.solver",
        ),
    )
    bundle_hash = str(source_identity["source_bundle_sha256"])
    metric = stage_r_metric_contract_v1()
    spectral = stage_r_spectral_gate_contract_v2()
    selection = recovery_selection_contract_rank1_replan_v1()
    target_budget = BudgetContract.proposed_v11_stage_r_rank1_replan()
    proposal = build_stage_r_rank1_replan_proposal(
        rank1_proposal=read_json(artifacts["rank1_proposal"]),
        rank1_completion=read_json(artifacts["rank1_completion"]),
        rank1_decision=read_json(artifacts["rank1_decision"]),
        rank1_manifest=read_json(artifacts["rank1_manifest"]),
        rank1_results=rank1_results,
        prior_stage_r_proposal=read_json(
            artifacts["prior_stage_r_proposal"]
        ),
        prior_stage_r_governance_receipt=read_json(
            artifacts["prior_stage_r_governance_receipt"]
        ),
        recovery_registry=read_json(
            artifacts["recovery_registry"]
        ),
        penalty_registry=read_json(
            artifacts["penalty_registry"]
        ),
        baseline_manifest=read_json(
            artifacts["baseline_manifest"]
        ),
        baseline_metrics=read_json(
            artifacts["baseline_metrics"]
        ),
        parent_experiment_id=parent_experiment_id,
        solver_hash=bundle_hash,
        evaluation_hash=bundle_hash,
        metric_contract_hash=str(metric["contract_sha256"]),
        spectral_gate_contract_hash=str(
            spectral["contract_sha256"]
        ),
        selection_contract_hash=str(
            selection["contract_sha256"]
        ),
        budget_contract_hash=target_budget.sha256,
        source_artifacts=source_artifacts,
    )
    request: dict[str, Any] = {
        "request_version": (
            "lyx_stage_r_rank1_replan_budget_request_v1"
        ),
        "status": "awaiting_human_budget_decision",
        "approved": False,
        "decision_state": _AUTHORIZATION_STATE,
        "proposal_sha256": proposal["proposal_sha256"],
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "profile_design_rule_hash": selection[
            "contract_sha256"
        ],
        "record_manifest_hash": proposal[
            "record_panel_sha256"
        ],
        "added_unique_identities": 36,
        "normal_unique_identity_limit": 888,
        "max_unique_identities": 900,
        "max_attempts": 1800,
        "retry_limit": 1,
        "budget_contract_hash": target_budget.sha256,
        "identity_panel_sha256": proposal[
            "identity_panel_sha256"
        ],
        "baseline_identity_panel_sha256": proposal[
            "baseline_identity_panel_sha256"
        ],
        "solver_hash": bundle_hash,
        "evaluation_hash": bundle_hash,
        "metric_contract_hash": metric["contract_sha256"],
        "spectral_gate_contract_hash": spectral[
            "contract_sha256"
        ],
        "selection_contract_hash": selection[
            "contract_sha256"
        ],
        "filter_revision_contract_hash": proposal[
            "frozen_contracts"
        ]["filter_revision_contract_hash"],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": True,
    }
    request["request_sha256"] = canonical_sha256(request)
    receipt: dict[str, Any] = {
        "receipt_version": (
            "lyx_stage_r_rank1_replan_proposal_receipt_v1"
        ),
        "status": "awaiting_human_execution_authorization",
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_request_sha256": request["request_sha256"],
        "identity_count": 36,
        "diagnostic_run_count": 0,
        "formal_run_count": 0,
        "parameter_search_run_count": 0,
        "independent_bo_run_count": 0,
        "may_execute_without_new_authorization": False,
    }
    staging = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        staging.mkdir(parents=True)
        atomic_write_json(
            staging / "stage_r_rank1_replan_proposal.json",
            proposal,
        )
        atomic_write_json(
            staging / "metric_contract.json",
            metric,
        )
        atomic_write_json(
            staging / "spectral_gate_contract.json",
            spectral,
        )
        atomic_write_json(
            staging / "recovery_selection_contract.json",
            selection,
        )
        atomic_write_json(
            staging / "budget_contract_v11.json",
            target_budget.to_dict(),
        )
        atomic_write_json(
            staging / "budget_amendment_request.json",
            request,
        )
        atomic_write_json(
            staging / "source_identity.json",
            source_identity,
        )
        receipt["artifact_sha256"] = {
            name: file_sha256(staging / name)
            for name in sorted(
                _PROPOSAL_ARTIFACT_NAMES
                - {"proposal_receipt.json"}
            )
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_write_json(
            staging / "proposal_receipt.json",
            receipt,
        )
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return receipt


def _identity_from_item(
    item: Mapping[str, Any],
) -> AttemptIdentity:
    return AttemptIdentity(
        solver_hash=str(item["solver_hash"]),
        config_hash=str(item["config_hash"]),
        metric_contract_hash=str(item["metric_contract_hash"]),
        evaluation_hash=str(item["evaluation_hash"]),
        data_sha256=str(item["data_sha256"]),
        record_id=str(item["record_id"]),
        stage=str(item["stage"]),
        attempt_kind=str(item["attempt_kind"]),
        parent_experiment_id=str(item["parent_experiment_id"]),
    )


def _exploration_from_payload(
    payload: Mapping[str, Any],
) -> ExplorationRegistry:
    return ExplorationRegistry(
        registry_version=str(payload["registry_version"]),
        unique_budget=int(payload["unique_budget"]),
        allowed_identity_sha256=tuple(
            str(value)
            for value in payload["allowed_identity_sha256"]
        ),
    )


def _resolve_source_artifacts(
    proposal: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Path]:
    raw_artifacts = _require_mapping(
        "stage_r_rank1_source_artifacts",
        proposal.get("source_artifacts"),
    )
    resolved: dict[str, Path] = {}
    for name, raw in raw_artifacts.items():
        artifact = _require_mapping(
            "stage_r_rank1_source_artifact",
            raw,
        )
        if artifact.get("path_base") != "repository_root":
            raise StageRRank1Error(
                "stage_r_rank1_source_path_base_invalid:"
                + str(name)
            )
        relative = Path(str(artifact.get("path", "")))
        path = (repository_root / relative).resolve()
        if (
            not path.is_relative_to(repository_root)
            or not path.is_file()
            or file_sha256(path) != artifact.get("file_sha256")
        ):
            raise StageRRank1Error(
                "stage_r_rank1_source_artifact_mismatch:"
                + str(name)
            )
        resolved[str(name)] = path
    return resolved


def _validate_proposal_preflight(
    *,
    proposal_dir: Path,
    source_root: Path,
) -> tuple[
    dict[str, Any],
    tuple[dict[str, Any], ...],
    dict[str, Path],
]:
    proposal_root = Path(proposal_dir).resolve()
    proposal = read_json(
        proposal_root / "stage_r_rank1_replan_proposal.json"
    )
    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="stage_r_rank1_proposal",
    )
    receipt = read_json(proposal_root / "proposal_receipt.json")
    _verify_embedded_hash(
        receipt,
        hash_field="receipt_sha256",
        artifact_name="stage_r_rank1_proposal_receipt",
    )
    artifact_hashes = _require_mapping(
        "stage_r_rank1_proposal_artifact_hashes",
        receipt.get("artifact_sha256"),
    )
    expected_artifacts = _PROPOSAL_ARTIFACT_NAMES - {
        "proposal_receipt.json"
    }
    if set(artifact_hashes) != expected_artifacts:
        raise StageRRank1Error(
            "stage_r_rank1_proposal_artifact_set_mismatch"
        )
    for name in expected_artifacts:
        path = proposal_root / name
        if (
            not path.is_file()
            or file_sha256(path) != artifact_hashes.get(name)
        ):
            raise StageRRank1Error(
                "stage_r_rank1_proposal_artifact_mismatch:" + name
            )
    if (
        receipt.get("proposal_sha256") != proposal_sha
        or receipt.get("identity_count") != 36
        or receipt.get("diagnostic_run_count") != 0
        or receipt.get("formal_run_count") != 0
        or receipt.get("independent_bo_run_count") != 0
        or receipt.get("may_execute_without_new_authorization")
        is not False
    ):
        raise StageRRank1Error(
            "stage_r_rank1_proposal_receipt_mismatch"
        )
    source_root = Path(source_root).resolve()
    repository_root = _repository_root_from_source_root(source_root)
    current_source_identity = runtime_source_identity(
        source_root,
        root_modules=(
            "ppg_hr.v2.recovery_stage_r_rank1_replan",
            "ppg_hr.v2.recovery_stage_r_rank1_replan_runner",
            "ppg_hr.v2.recovery_stage_r_experiment",
            "ppg_hr.v2.recovery_stage_r_cache",
            "ppg_hr.v2.recovery_selection",
            "ppg_hr.v2.solver",
        ),
    )
    if (
        read_json(proposal_root / "source_identity.json")
        != current_source_identity
    ):
        raise StageRRank1Error(
            "stage_r_rank1_source_identity_mismatch"
        )
    metric = stage_r_metric_contract_v1()
    spectral = stage_r_spectral_gate_contract_v2()
    selection = recovery_selection_contract_rank1_replan_v1()
    target_budget = BudgetContract.proposed_v11_stage_r_rank1_replan()
    if (
        canonical_sha256(
            read_json(proposal_root / "metric_contract.json")
        )
        != canonical_sha256(metric)
        or canonical_sha256(
            read_json(
                proposal_root / "spectral_gate_contract.json"
            )
        )
        != canonical_sha256(spectral)
        or canonical_sha256(
            read_json(
                proposal_root / "recovery_selection_contract.json"
            )
        )
        != canonical_sha256(selection)
        or canonical_sha256(
            read_json(
                proposal_root / "budget_contract_v11.json"
            )
        )
        != canonical_sha256(target_budget.to_dict())
    ):
        raise StageRRank1Error(
            "stage_r_rank1_frozen_contract_mismatch"
        )
    resolved = _resolve_source_artifacts(
        proposal,
        repository_root=repository_root,
    )
    required_source_names = {
        "rank1_proposal",
        "rank1_completion",
        "rank1_decision",
        "rank1_manifest",
        "prior_stage_r_proposal",
        "prior_stage_r_governance_receipt",
        "recovery_registry",
        "penalty_registry",
        "baseline_manifest",
        "baseline_metrics",
        "source_budget_contract",
        "experiment_spec",
        *(
            f"rank1_result:{item['record_id']}"
            for item in _require_list(
                "stage_r_rank1_proposal_record_panel",
                proposal.get("record_panel"),
            )
        ),
    }
    if set(resolved) != required_source_names:
        raise StageRRank1Error(
            "stage_r_rank1_source_artifact_set_mismatch"
        )
    rank1_manifest = read_json(resolved["rank1_manifest"])
    rank1_results = {
        str(item["record_id"]): read_json(
            resolved[f"rank1_result:{item['record_id']}"]
        )
        for item in _require_list(
            "stage_r_rank1_manifest_results",
            rank1_manifest.get("results"),
        )
    }
    frozen = _require_mapping(
        "stage_r_rank1_frozen_contracts",
        proposal.get("frozen_contracts"),
    )
    rebuilt = build_stage_r_rank1_replan_proposal(
        rank1_proposal=read_json(resolved["rank1_proposal"]),
        rank1_completion=read_json(resolved["rank1_completion"]),
        rank1_decision=read_json(resolved["rank1_decision"]),
        rank1_manifest=rank1_manifest,
        rank1_results=rank1_results,
        prior_stage_r_proposal=read_json(
            resolved["prior_stage_r_proposal"]
        ),
        prior_stage_r_governance_receipt=read_json(
            resolved["prior_stage_r_governance_receipt"]
        ),
        recovery_registry=read_json(resolved["recovery_registry"]),
        penalty_registry=read_json(resolved["penalty_registry"]),
        baseline_manifest=read_json(
            resolved["baseline_manifest"]
        ),
        baseline_metrics=read_json(
            resolved["baseline_metrics"]
        ),
        parent_experiment_id=str(proposal["parent_experiment_id"]),
        solver_hash=str(frozen["solver_hash"]),
        evaluation_hash=str(frozen["evaluation_hash"]),
        metric_contract_hash=str(frozen["metric_contract_hash"]),
        spectral_gate_contract_hash=str(
            frozen["spectral_gate_contract_hash"]
        ),
        selection_contract_hash=str(
            frozen["selection_contract_hash"]
        ),
        budget_contract_hash=str(frozen["budget_contract_hash"]),
        source_artifacts={
            str(name): dict(
                _require_mapping(
                    "stage_r_rank1_source_artifact",
                    raw,
                )
            )
            for name, raw in _require_mapping(
                "stage_r_rank1_source_artifacts",
                proposal.get("source_artifacts"),
            ).items()
        },
    )
    if rebuilt != proposal:
        raise StageRRank1Error(
            "stage_r_rank1_proposal_rebuild_mismatch"
        )
    request = read_json(
        proposal_root / "budget_amendment_request.json"
    )
    request_sha = _verify_embedded_hash(
        request,
        hash_field="request_sha256",
        artifact_name="stage_r_rank1_budget_request",
    )
    expected_request = {
        "status": "awaiting_human_budget_decision",
        "approved": False,
        "decision_state": _AUTHORIZATION_STATE,
        "proposal_sha256": proposal_sha,
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "profile_design_rule_hash": frozen[
            "selection_contract_hash"
        ],
        "record_manifest_hash": proposal[
            "record_panel_sha256"
        ],
        "added_unique_identities": 36,
        "normal_unique_identity_limit": 888,
        "max_unique_identities": 900,
        "max_attempts": 1800,
        "retry_limit": 1,
        "budget_contract_hash": target_budget.sha256,
        "identity_panel_sha256": proposal[
            "identity_panel_sha256"
        ],
        "baseline_identity_panel_sha256": proposal[
            "baseline_identity_panel_sha256"
        ],
        "solver_hash": frozen["solver_hash"],
        "evaluation_hash": frozen["evaluation_hash"],
        "metric_contract_hash": frozen["metric_contract_hash"],
        "spectral_gate_contract_hash": frozen[
            "spectral_gate_contract_hash"
        ],
        "selection_contract_hash": frozen[
            "selection_contract_hash"
        ],
        "filter_revision_contract_hash": frozen[
            "filter_revision_contract_hash"
        ],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": True,
    }
    mismatched = sorted(
        name
        for name, value in expected_request.items()
        if request.get(name) != value
    )
    if mismatched or receipt.get("budget_request_sha256") != request_sha:
        raise StageRRank1Error(
            "stage_r_rank1_budget_request_mismatch:"
            + ",".join(mismatched)
        )
    raw_identities = tuple(
        dict(
            _require_mapping(
                "stage_r_rank1_identity",
                item,
            )
        )
        for item in _require_list(
            "stage_r_rank1_identities",
            proposal.get("identities"),
        )
    )
    identities = tuple(
        _identity_from_item(item) for item in raw_identities
    )
    identity_hashes = [
        identity.sha256 for identity in identities
    ]
    if (
        len(identities) != 36
        or len(set(identity_hashes)) != 36
        or identity_hashes != proposal.get("identity_sha256")
        or canonical_sha256(identity_hashes)
        != proposal.get("identity_panel_sha256")
    ):
        raise StageRRank1Error(
            "stage_r_rank1_identity_matrix_mismatch"
        )
    return proposal, raw_identities, resolved


def prepare_stage_r_rank1_replan_governance(
    *,
    proposal_dir: Path,
    authorization_receipt_path: Path,
    source_governance_dir: Path,
    governance_dir: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Migrate v10 and register only the approved 36 formal identities."""

    proposal, raw_identities, _resolved = _validate_proposal_preflight(
        proposal_dir=proposal_dir,
        source_root=source_root,
    )
    authorization = validate_stage_r_rank1_execution_authorization(
        proposal,
        receipt=read_json(
            Path(authorization_receipt_path).resolve()
        ),
    )
    target_root = Path(governance_dir).resolve()
    if target_root.exists():
        raise StageRRank1Error(
            "stage_r_rank1_governance_exists"
        )
    source_dir = Path(source_governance_dir).resolve()
    source_budget = BudgetContract.proposed_v10_rank1_filter_revision()
    if (
        read_json(source_dir / "budget_contract.json")
        != source_budget.to_dict()
    ):
        raise StageRRank1Error(
            "stage_r_rank1_source_governance_budget_mismatch"
        )
    exploration_payload = read_json(
        source_dir / "exploration_registry.json"
    )
    exploration = _exploration_from_payload(exploration_payload)
    if exploration.to_dict() != exploration_payload:
        raise StageRRank1Error(
            "stage_r_rank1_exploration_registry_mismatch"
        )
    registry = AttemptRegistry.open(
        source_dir / "attempt_registry.json",
        budget_contract=source_budget,
        exploration_registry=exploration,
    )
    target_budget = BudgetContract.proposed_v11_stage_r_rank1_replan()
    frozen = _require_mapping(
        "stage_r_rank1_frozen_contracts",
        proposal["frozen_contracts"],
    )
    if target_budget.sha256 != frozen.get(
        "budget_contract_hash"
    ):
        raise StageRRank1Error(
            "stage_r_rank1_target_budget_mismatch"
        )
    identities = tuple(
        _identity_from_item(item) for item in raw_identities
    )
    amendment = BudgetAmendmentRequest(
        stage=_STAGE,
        profile_design_rule_hash=str(
            frozen["selection_contract_hash"]
        ),
        record_manifest_hash=str(
            proposal["record_panel_sha256"]
        ),
        added_unique_identities=36,
        normal_unique_identity_limit=888,
        max_unique_identities=900,
        max_attempts=1800,
    )
    governance_receipt: dict[str, Any] = {}

    def finalize(
        staging: Path,
        staged: AttemptRegistry,
    ) -> None:
        nonlocal governance_receipt
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
            authorization,
        )
        governance_receipt = {
            "receipt_version": (
                "lyx_stage_r_rank1_replan_governance_v1"
            ),
            "status": "prepared_zero_runs",
            "proposal_sha256": proposal["proposal_sha256"],
            "authorization_sha256": canonical_sha256(
                authorization
            ),
            "source_budget_contract_hash": source_budget.sha256,
            "target_budget_contract_hash": target_budget.sha256,
            "new_unique_identity_count": 36,
            "historical_threshold_diagnostic_count": 60,
            "new_threshold_diagnostic_count": 0,
            "attempt_registry_summary": staged.summary(),
            "parameter_search_authorized": False,
            "independent_bo_authorized": False,
            "automatic_stage_f_execution": False,
        }
        governance_receipt["receipt_sha256"] = (
            canonical_sha256(governance_receipt)
        )
        atomic_write_json(
            staging / "governance_receipt.json",
            governance_receipt,
        )

    registry.migrate_to(
        target_root / "attempt_registry.json",
        budget_contract=target_budget,
        amendment_request=amendment,
        authorization_receipt=authorization,
        new_identities=identities,
        target_exploration_registry=exploration,
        finalize_staging=finalize,
    )
    return governance_receipt


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
        raise StageRRank1Error(
            "stage_r_rank1_recovery_delay_invalid"
        )
    return value


def _build_rank1_selection(
    *,
    proposal: Mapping[str, Any],
    result_rows: Sequence[Mapping[str, Any]],
    baseline_metrics_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = read_json(baseline_metrics_path)
    independent = {
        str(item["sample_id"]): _require_mapping(
            "stage_r_rank1_independent_metrics",
            _require_mapping(
                "stage_r_rank1_baseline_record",
                item,
            ).get("metrics"),
        )
        for item in _require_list(
            "stage_r_rank1_baseline_records",
            baseline.get("records"),
        )
    }
    if len(result_rows) != 36:
        raise StageRRank1Error(
            "stage_r_rank1_result_count_mismatch"
        )
    by_coordinate = {
        (
            str(row["recovery_candidate_id"]),
            str(row["record_id"]),
        ): row
        for row in result_rows
    }
    if len(by_coordinate) != 36:
        raise StageRRank1Error(
            "stage_r_rank1_result_coordinate_mismatch"
        )
    spectral_hashes: dict[str, set[str]] = {}
    for row in result_rows:
        spectral = _require_mapping(
            "stage_r_rank1_spectral_audit",
            row.get("spectral_audit"),
        )
        spectral_hashes.setdefault(
            str(row["record_id"]),
            set(),
        ).add(str(spectral.get("audit_sha256", "")))
    if (
        len(spectral_hashes) != 12
        or any(
            len(hashes) != 1 or "" in hashes
            for hashes in spectral_hashes.values()
        )
    ):
        raise StageRRank1Error(
            "stage_r_rank1_spectral_candidate_invariance_mismatch"
        )
    panel = [
        RecoveryPanelRecord(
            record_id=str(item["record_id"]),
            scene=str(item["scene"]),
            true_rise_applicable=bool(
                item["true_rise_applicable"]
            ),
        )
        for item in _require_list(
            "stage_r_rank1_record_panel",
            proposal.get("record_panel"),
        )
    ]
    if set(independent) != {
        item.record_id for item in panel
    }:
        raise StageRRank1Error(
            "stage_r_rank1_independent_baseline_panel_mismatch"
        )
    evaluations: list[RecoveryCandidateEvaluation] = []
    serialized: list[dict[str, Any]] = []
    for candidate in _require_list(
        "stage_r_rank1_recovery_candidates",
        proposal.get("recovery_candidates"),
    ):
        candidate_id = str(candidate["candidate_id"])
        records: list[RecoveryRecordEvaluation] = []
        for panel_record in panel:
            row = by_coordinate[
                (candidate_id, panel_record.record_id)
            ]
            current = by_coordinate[
                (_CONTROL_RECOVERY_ID, panel_record.record_id)
            ]
            metrics = _require_mapping(
                "stage_r_rank1_metrics",
                row.get("metrics"),
            )
            current_metrics = _require_mapping(
                "stage_r_rank1_current_metrics",
                current.get("metrics"),
            )
            independent_metrics = independent[
                panel_record.record_id
            ]
            spectral = _require_mapping(
                "stage_r_rank1_spectral_audit",
                row.get("spectral_audit"),
            )
            records.append(
                RecoveryRecordEvaluation(
                    record_id=panel_record.record_id,
                    sentinel_id=_PROFILE_ID,
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
                        independent_metrics[
                            "longest_e10_run_windows"
                        ]
                    ),
                    independent_l20=float(
                        independent_metrics[
                            "longest_e20_run_windows"
                        ]
                    ),
                    independent_mae=float(
                        independent_metrics[
                            "final_motion_mae_bpm"
                        ]
                    ),
                    current_l10=float(
                        current_metrics[
                            "longest_e10_run_windows"
                        ]
                    ),
                    current_mae=float(
                        current_metrics["final_motion_mae_bpm"]
                    ),
                    recovery_delay=_selection_recovery_delay(
                        metrics
                    ),
                    right_censored_recovery_count=int(
                        metrics["right_censored_recovery_count"]
                    ),
                    current_right_censored_recovery_count=int(
                        current_metrics[
                            "right_censored_recovery_count"
                        ]
                    ),
                    true_rise_underestimate=(
                        float(
                            metrics[
                                "max_rise_underestimate_bpm"
                            ]
                        )
                        if (
                            panel_record.true_rise_applicable
                            and metrics.get(
                                "max_rise_underestimate_bpm"
                            )
                            is not None
                        )
                        else None
                    ),
                    current_true_rise_underestimate=(
                        float(
                            current_metrics[
                                "max_rise_underestimate_bpm"
                            ]
                        )
                        if (
                            panel_record.true_rise_applicable
                            and current_metrics.get(
                                "max_rise_underestimate_bpm"
                            )
                            is not None
                        )
                        else None
                    ),
                )
            )
        evaluation = RecoveryCandidateEvaluation(
            candidate_id=candidate_id,
            mechanism_complexity=int(
                candidate["mechanism_complexity"]
            ),
            records=tuple(records),
        )
        evaluations.append(evaluation)
        serialized.append(asdict(evaluation))
    selection = select_rank1_recovery_candidate_evaluations(
        evaluations,
        expected_records=panel,
        expected_sentinel_ids=(_PROFILE_ID,),
    )
    return selection, serialized


def _independent_bo_review_package(
    *,
    proposal: Mapping[str, Any],
    authorization_sha256: str,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "package_version": (
            "lyx_stage_r_rank1_independent_bo_review_v1"
        ),
        "status": "awaiting_human_independent_bo_decision",
        "trigger": "no_safe_recovery_candidate",
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": authorization_sha256,
        "selection_sha256": selection["selection_sha256"],
        "independent_bo_authorized": False,
        "independent_bo_run_count": 0,
        "full_independent_bo_requires_separate_human_review": True,
        "main_comparison_baseline": (
            "per_record_independent_bo_lite"
        ),
        "trace_rescue_role": "historical_exploration_only",
        "automatic_stage_f_execution": False,
    }
    payload["package_sha256"] = canonical_sha256(payload)
    return payload


def _validate_completed_execution(
    *,
    completion_path: Path,
    proposal: Mapping[str, Any],
    authorization_sha256: str,
    destination: Path,
    registry: AttemptRegistry,
    identities: Sequence[AttemptIdentity],
    baseline_metrics_path: Path,
) -> dict[str, Any]:
    completion = read_json(completion_path)
    _verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="stage_r_rank1_completion",
    )
    matrix = registry.matrix_execution_summary(identities)
    if (
        completion.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or completion.get("authorization_sha256")
        != authorization_sha256
        or completion.get("formal_result_count") != 36
        or completion.get("diagnostic_result_count") != 0
        or completion.get("independent_bo_run_count") != 0
        or completion.get("automatic_stage_f_execution")
        is not False
        or completion.get("matrix_execution_summary") != matrix
        or matrix["planned_identity_count"] != 36
        or matrix["failed_attempt_count"] != 0
        or matrix["retry_count"] != 0
    ):
        raise StageRRank1Error(
            "stage_r_rank1_completion_mismatch"
        )
    artifacts = _require_mapping(
        "stage_r_rank1_completion_artifacts",
        completion.get("artifacts"),
    )
    expected_artifact_names = {
        "execution_binding.json",
        "identity_result_index.json",
        "formal_candidate_evaluations.json",
        "recovery_selection.json",
        "attempt_registry_snapshot.json",
    }
    if completion.get("status") == "no_safe_recovery_candidate":
        expected_artifact_names.add(
            "independent_bo_review_package.json"
        )
    if set(artifacts) != expected_artifact_names:
        raise StageRRank1Error(
            "stage_r_rank1_completion_artifact_set_mismatch"
        )
    resolved_destination = destination.resolve()
    for name in sorted(expected_artifact_names):
        expected_hash = artifacts[name]
        path = (resolved_destination / name).resolve()
        if (
            path.parent != resolved_destination
            or not path.is_file()
            or file_sha256(path) != expected_hash
        ):
            raise StageRRank1Error(
                "stage_r_rank1_completion_artifact_mismatch:"
                + str(name)
            )
    binding = read_json(
        resolved_destination / "execution_binding.json"
    )
    _verify_embedded_hash(
        binding,
        hash_field="binding_sha256",
        artifact_name="stage_r_rank1_execution_binding",
    )
    if (
        binding.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or binding.get("authorization_sha256")
        != authorization_sha256
        or binding.get("solver_source_bundle_sha256")
        != identities[0].solver_hash
        or binding.get("evaluation_hash")
        != identities[0].evaluation_hash
    ):
        raise StageRRank1Error(
            "stage_r_rank1_execution_binding_mismatch"
        )
    result_index = read_json(
        resolved_destination / "identity_result_index.json"
    )
    _verify_embedded_hash(
        result_index,
        hash_field="index_sha256",
        artifact_name="stage_r_rank1_result_index",
    )
    result_rows = _require_list(
        "stage_r_rank1_completion_results",
        result_index.get("results"),
    )
    if (
        result_index.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or result_index.get("result_count") != 36
        or len(result_rows) != 36
    ):
        raise StageRRank1Error(
            "stage_r_rank1_result_index_mismatch"
        )
    expected_selection, expected_evaluations = (
        _build_rank1_selection(
            proposal=proposal,
            result_rows=result_rows,
            baseline_metrics_path=baseline_metrics_path,
        )
    )
    selection = read_json(
        resolved_destination / "recovery_selection.json"
    )
    _verify_embedded_hash(
        selection,
        hash_field="selection_sha256",
        artifact_name="stage_r_rank1_selection",
    )
    if selection != expected_selection:
        raise StageRRank1Error(
            "stage_r_rank1_selection_recomputation_mismatch"
        )
    evaluation = read_json(
        resolved_destination
        / "formal_candidate_evaluations.json"
    )
    _verify_embedded_hash(
        evaluation,
        hash_field="evaluation_sha256",
        artifact_name="stage_r_rank1_candidate_evaluations",
    )
    expected_evaluation: dict[str, Any] = {
        "evaluation_version": (
            "lyx_stage_r_rank1_candidate_evaluations_v1"
        ),
        "proposal_sha256": proposal["proposal_sha256"],
        "candidate_evaluations": expected_evaluations,
    }
    expected_evaluation["evaluation_sha256"] = (
        canonical_sha256(expected_evaluation)
    )
    if canonical_sha256(evaluation) != canonical_sha256(
        expected_evaluation
    ):
        raise StageRRank1Error(
            "stage_r_rank1_evaluation_recomputation_mismatch"
        )
    snapshot = read_json(
        resolved_destination / "attempt_registry_snapshot.json"
    )
    registry.assert_matrix_matches_snapshot(
        identities,
        snapshot,
    )
    if snapshot != registry.matrix_snapshot(identities):
        raise StageRRank1Error(
            "stage_r_rank1_registry_snapshot_mismatch"
        )
    if selection["status"] == "no_safe_recovery_candidate":
        review_package = read_json(
            resolved_destination
            / "independent_bo_review_package.json"
        )
        _verify_embedded_hash(
            review_package,
            hash_field="package_sha256",
            artifact_name=(
                "stage_r_rank1_independent_bo_review_package"
            ),
        )
        if review_package != _independent_bo_review_package(
            proposal=proposal,
            authorization_sha256=authorization_sha256,
            selection=selection,
        ):
            raise StageRRank1Error(
                "stage_r_rank1_independent_bo_review_package_mismatch"
            )
    expected_next_state = (
        "awaiting_human_independent_bo_decision"
        if selection["status"] == "no_safe_recovery_candidate"
        else "awaiting_stage_f_rank1_replan_human_review"
    )
    if (
        completion.get("status") != selection["status"]
        or completion.get("next_state") != expected_next_state
        or completion.get("provisional_recovery_id")
        != selection["provisional_recovery_id"]
        or completion.get("rollback_backup_id")
        != selection["rollback_backup_id"]
        or completion.get("selection_sha256")
        != selection["selection_sha256"]
    ):
        raise StageRRank1Error(
            "stage_r_rank1_completion_selection_mismatch"
        )
    return completion


def execute_stage_r_rank1_replan(
    *,
    proposal_dir: Path,
    authorization_receipt_path: Path,
    governance_dir: Path,
    output_dir: Path,
    source_root: Path,
    numerical_runner: Any | None = None,
) -> dict[str, Any]:
    """Execute only the approved 3×1×12 rank-1 Stage R matrix."""

    proposal, raw_identities, resolved = _validate_proposal_preflight(
        proposal_dir=proposal_dir,
        source_root=source_root,
    )
    authorization = validate_stage_r_rank1_execution_authorization(
        proposal,
        receipt=read_json(
            Path(authorization_receipt_path).resolve()
        ),
    )
    authorization_sha256 = canonical_sha256(authorization)
    governance_root = Path(governance_dir).resolve()
    budget = BudgetContract.proposed_v11_stage_r_rank1_replan()
    if (
        read_json(governance_root / "budget_contract.json")
        != budget.to_dict()
    ):
        raise StageRRank1Error(
            "stage_r_rank1_governance_budget_mismatch"
        )
    governance_authorization = read_json(
        governance_root / "execution_authorization.json"
    )
    if governance_authorization != authorization:
        raise StageRRank1Error(
            "stage_r_rank1_governance_authorization_mismatch"
        )
    exploration_payload = read_json(
        governance_root / "exploration_registry.json"
    )
    exploration = _exploration_from_payload(exploration_payload)
    if exploration.to_dict() != exploration_payload:
        raise StageRRank1Error(
            "stage_r_rank1_exploration_registry_mismatch"
        )
    registry = AttemptRegistry.open(
        governance_root / "attempt_registry.json",
        budget_contract=budget,
        exploration_registry=exploration,
    )
    identities = tuple(
        _identity_from_item(item) for item in raw_identities
    )
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    binding = {
        "binding_version": (
            "lyx_stage_r_rank1_execution_binding_v1"
        ),
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": authorization_sha256,
        "solver_source_bundle_sha256": identities[0].solver_hash,
        "evaluation_hash": identities[0].evaluation_hash,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    binding_path = destination / "execution_binding.json"
    if binding_path.is_file():
        if read_json(binding_path) != binding:
            raise StageRRank1Error(
                "stage_r_rank1_execution_binding_mismatch"
            )
    else:
        atomic_write_json(binding_path, binding)
    completion_path = destination / "completion.json"
    if completion_path.is_file():
        registry.assert_complete_matrix(identities)
        return _validate_completed_execution(
            completion_path=completion_path,
            proposal=proposal,
            authorization_sha256=authorization_sha256,
            destination=destination,
            registry=registry,
            identities=identities,
            baseline_metrics_path=resolved["baseline_metrics"],
        )
    registry.register_identities(identities)
    runner = numerical_runner or run_stage_r_numerical_identity
    spectral_audit_dir = destination / "spectral_audits"
    results: list[dict[str, Any]] = []
    for item, identity in zip(
        raw_identities,
        identities,
        strict=True,
    ):
        before = registry.matrix_execution_summary((identity,))
        if (
            before["failed_attempt_count"] != 0
            or before["retry_count"] != 0
        ):
            raise StageRRank1Error(
                "stage_r_rank1_retry_requires_human_review:"
                + identity.record_id
            )
        results.append(
            stage_r_cache.execute_stage_r_identity(
                registry=registry,
                item=dict(item),
                numerical_runner=runner,
                spectral_audit_dir=spectral_audit_dir,
            )
        )
    registry.assert_complete_matrix(identities)
    matrix = registry.matrix_execution_summary(identities)
    if (
        matrix["planned_identity_count"] != 36
        or matrix["failed_attempt_count"] != 0
        or matrix["retry_count"] != 0
    ):
        raise StageRRank1Error(
            "stage_r_rank1_execution_summary_invalid"
        )
    result_index: dict[str, Any] = {
        "index_version": (
            "lyx_stage_r_rank1_result_index_v1"
        ),
        "proposal_sha256": proposal["proposal_sha256"],
        "result_count": 36,
        "results": results,
    }
    result_index["index_sha256"] = canonical_sha256(
        result_index
    )
    atomic_write_json(
        destination / "identity_result_index.json",
        result_index,
    )
    selection, evaluations = _build_rank1_selection(
        proposal=proposal,
        result_rows=results,
        baseline_metrics_path=resolved["baseline_metrics"],
    )
    evaluation_payload: dict[str, Any] = {
        "evaluation_version": (
            "lyx_stage_r_rank1_candidate_evaluations_v1"
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
        destination / "recovery_selection.json",
        selection,
    )
    artifact_names = [
        "execution_binding.json",
        "identity_result_index.json",
        "formal_candidate_evaluations.json",
        "recovery_selection.json",
    ]
    if selection["status"] == "no_safe_recovery_candidate":
        atomic_write_json(
            destination / "independent_bo_review_package.json",
            _independent_bo_review_package(
                proposal=proposal,
                authorization_sha256=authorization_sha256,
                selection=selection,
            ),
        )
        artifact_names.append(
            "independent_bo_review_package.json"
        )
    snapshot = registry.matrix_snapshot(identities)
    atomic_write_json(
        destination / "attempt_registry_snapshot.json",
        snapshot,
    )
    artifact_names.append("attempt_registry_snapshot.json")
    artifacts = {
        name: file_sha256(destination / name)
        for name in artifact_names
    }
    next_state = (
        "awaiting_human_independent_bo_decision"
        if selection["status"] == "no_safe_recovery_candidate"
        else "awaiting_stage_f_rank1_replan_human_review"
    )
    completion: dict[str, Any] = {
        "completion_version": (
            "lyx_stage_r_rank1_replan_completion_v1"
        ),
        "status": selection["status"],
        "next_state": next_state,
        "evidence_class": "development_reuse_pilot",
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": authorization_sha256,
        "diagnostic_result_count": 0,
        "formal_result_count": 36,
        "formal_solver_run_count": matrix[
            "identity_with_solver_attempt_count"
        ],
        "independent_bo_run_count": 0,
        "automatic_stage_f_execution": False,
        "provisional_recovery_id": selection[
            "provisional_recovery_id"
        ],
        "rollback_backup_id": selection[
            "rollback_backup_id"
        ],
        "matrix_execution_summary": matrix,
        "selection_sha256": selection["selection_sha256"],
        "artifacts": artifacts,
    }
    completion["completion_sha256"] = canonical_sha256(
        completion
    )
    atomic_write_json(completion_path, completion)
    return _validate_completed_execution(
        completion_path=completion_path,
        proposal=proposal,
        authorization_sha256=authorization_sha256,
        destination=destination,
        registry=registry,
        identities=identities,
        baseline_metrics_path=resolved["baseline_metrics"],
    )
