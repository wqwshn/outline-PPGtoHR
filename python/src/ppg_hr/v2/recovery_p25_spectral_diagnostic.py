"""Bounded p25 spectral diagnostic proposed after the Stage R stop.

This module first freezes a 3-profile by 12-record diagnostic matrix.  It does
not authorize execution, recovery nomination, Stage F, or independent BO.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
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
from .recovery_stage_r_common import StageRNumericalRunner


class P25SpectralDiagnosticError(RuntimeError):
    """The p25 diagnostic proposal violates its frozen contract."""


class P25SpectralDiagnosticAuthorizationError(P25SpectralDiagnosticError):
    """The exact p25 diagnostic proposal has not been approved."""


_STAGE = "filter_profile_p25_spectral_diagnostic"
_ATTEMPT_KIND = "diagnostic"
_AUTHORIZATION_STATE = "awaiting_human_p25_spectral_diagnostic_decision"
_CURRENT_RECOVERY_ID = "current_fixed_floor_control_v1"
_CURRENT_PENALTY_ID = "current_soft_penalty_control_v1"
_P25_PROFILE_IDS = (
    "p25-short-low",
    "p25-short-mid",
    "p25-long-mid",
)
_EXPECTED_SCENE_COUNTS = {
    "jianpan": 3,
    "kaihe": 3,
    "run": 3,
    "xiezi": 3,
}
_EXPECTED_RECORD_IDS = {
    "jianpan1_LYX_0708",
    "jianpan2_LYX_0708",
    "jianpan3_LYX_0708",
    "kaihe1_LYX_0613",
    "kaihe1_LYX_0617",
    "kaihe3_LYX_0613",
    "run1_LYX_0708",
    "run2_LYX_0708",
    "run3_LYX_0708",
    "xiezi2_LYX_0708",
    "xiezi3_LYX_0708",
    "xiezi4_LYX_0708",
}


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise P25SpectralDiagnosticError(f"{name}_must_be_object")
    return value


def _require_list(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise P25SpectralDiagnosticError(f"{name}_must_be_array")
    return value


def _require_hash(name: str, value: object) -> str:
    text = str(value)
    try:
        require_sha256(name, text)
    except ValueError as exc:
        raise P25SpectralDiagnosticError(str(exc)) from exc
    return text


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    artifact_name: str,
) -> str:
    declared = _require_hash(hash_field, payload.get(hash_field))
    unhashed = dict(payload)
    unhashed.pop(hash_field, None)
    if canonical_sha256(unhashed) != declared:
        raise P25SpectralDiagnosticError(f"{artifact_name}_hash_mismatch")
    return declared


def _budget_contract_from_payload(
    payload: Mapping[str, Any],
) -> BudgetContract:
    return BudgetContract(
        contract_version=str(payload["contract_version"]),
        stage_unique_limits=dict(
            _require_mapping(
                "stage_unique_limits",
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
            _require_mapping(
                "stage_attempt_kinds",
                payload.get("stage_attempt_kinds"),
            )
        ),
        max_unique_identities=int(payload["max_unique_identities"]),
        max_attempts=int(payload["max_attempts"]),
        retry_limit=int(payload["retry_limit"]),
    )


def _verify_budget_contract(
    payload: Mapping[str, Any],
) -> BudgetContract:
    contract = _budget_contract_from_payload(payload)
    if (
        contract.contract_version != "lyx_recovery_filter_budget_v6"
        or contract.stage_unique_limits.get(_STAGE) != 36
        or contract.stage_attempt_kinds.get(_STAGE) != _ATTEMPT_KIND
        or contract.normal_unique_identity_limit != 780
        or contract.max_unique_identities != 792
        or contract.max_attempts != 1584
        or contract.retry_limit != 1
        or contract.supplemental_stage != "fold_replay"
        or contract.stage_unique_limits.get("fold_replay") != 12
    ):
        raise P25SpectralDiagnosticError("p25_spectral_budget_contract_mismatch")
    return contract


def _verify_profile(
    profile: Mapping[str, Any],
) -> str:
    profile_id = str(profile.get("profile_id", ""))
    declared = _require_hash(
        f"profile_sha256:{profile_id}",
        profile.get("profile_sha256"),
    )
    identity = {
        "profile_id": profile.get("profile_id"),
        "design_role": profile.get("design_role"),
        "fs_target": profile.get("fs_target"),
        "memory_ms": profile.get("physical_memory_ms"),
        "nominal_mu": profile.get("nominal_mu"),
        "recovery_sentinel_role": profile.get("recovery_sentinel_role"),
        "actual_taps": profile.get("actual_taps"),
    }
    if canonical_sha256(identity) != declared:
        raise P25SpectralDiagnosticError(f"p25_spectral_profile_hash_mismatch:{profile_id}")
    return declared


def _p25_profiles_from_library(
    profile_library: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    _verify_embedded_hash(
        profile_library,
        hash_field="library_sha256",
        artifact_name="filter_profile_library",
    )
    profiles_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in _require_list(
        "filter_profiles",
        profile_library.get("profiles"),
    ):
        profile = _require_mapping("filter_profile", raw)
        profile_id = str(profile.get("profile_id", ""))
        if profile_id in _P25_PROFILE_IDS:
            if profile_id in profiles_by_id:
                raise P25SpectralDiagnosticError(f"duplicate_p25_profile:{profile_id}")
            _verify_profile(profile)
            profiles_by_id[profile_id] = profile
    if set(profiles_by_id) != set(_P25_PROFILE_IDS):
        raise P25SpectralDiagnosticError("p25_spectral_profile_set_mismatch")
    profiles = tuple(dict(profiles_by_id[profile_id]) for profile_id in _P25_PROFILE_IDS)
    for profile in profiles:
        if (
            profile.get("fs_target") != 25
            or profile.get("design_role") != "core"
            or profile.get("recovery_sentinel_role") is not None
        ):
            raise P25SpectralDiagnosticError(
                f"p25_spectral_profile_role_mismatch:{profile['profile_id']}"
            )
    return profiles


def _stage_r_templates(
    stage_r_proposal: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    proposal_sha = _verify_embedded_hash(
        stage_r_proposal,
        hash_field="proposal_sha256",
        artifact_name="stage_r_proposal",
    )
    if proposal_sha != "a661915b93b884cfaddc09ad00c43fb812bc64ea8878ed933e030e8f97947d1b":
        raise P25SpectralDiagnosticError("p25_spectral_unapproved_stage_r_proposal")
    candidates = [
        dict(_require_mapping("stage_r_identity", raw))
        for raw in _require_list(
            "stage_r_identities",
            stage_r_proposal.get("identities"),
        )
        if isinstance(raw, Mapping)
        and raw.get("stage") == "recovery_sentinel"
        and raw.get("recovery_candidate_id") == _CURRENT_RECOVERY_ID
        and raw.get("filter_profile_id") == "p50-short-low"
    ]
    if len(candidates) != 12:
        raise P25SpectralDiagnosticError("p25_spectral_stage_r_template_count_mismatch")
    record_ids = [str(item["record_id"]) for item in candidates]
    if len(set(record_ids)) != 12:
        raise P25SpectralDiagnosticError("p25_spectral_duplicate_stage_r_template_record")
    scene_counts = {
        scene: sum(item.get("scene") == scene for item in candidates)
        for scene in _EXPECTED_SCENE_COUNTS
    }
    if scene_counts != _EXPECTED_SCENE_COUNTS:
        raise P25SpectralDiagnosticError("p25_spectral_record_panel_mismatch")
    return tuple(sorted(candidates, key=lambda item: str(item["record_id"])))


def _validate_stage_r_completion(
    completion: Mapping[str, Any],
    *,
    expected_proposal_sha: str,
) -> str:
    completion_sha = _verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="stage_r_completion",
    )
    if (
        completion.get("proposal_sha256") != expected_proposal_sha
        or completion.get("status") != "no_safe_recovery_candidate"
        or completion.get("provisional_recovery_id") is not None
        or completion.get("rollback_backup_id") is not None
        or completion.get("independent_bo_run_count") != 0
        or completion.get("diagnostic_result_count") != 60
        or completion.get("formal_result_count") != 108
    ):
        raise P25SpectralDiagnosticError("p25_spectral_stage_r_completion_mismatch")
    return completion_sha


def _identity_item(
    *,
    template: Mapping[str, Any],
    profile: Mapping[str, Any],
    parent_experiment_id: str,
    solver_hash: str,
    evaluation_hash: str,
    metric_contract_hash: str,
) -> dict[str, Any]:
    config = deepcopy(
        dict(
            _require_mapping(
                "stage_r_template_config",
                template.get("config"),
            )
        )
    )
    parameters = dict(
        _require_mapping(
            "stage_r_template_parameters",
            config.get("parameters"),
        )
    )
    parameters.update(
        {
            "fs_target": int(profile["fs_target"]),
            "lms_mu_base": float(profile["nominal_mu"]),
            "lms_mu_min": 1e-6,
            "max_order": int(profile["actual_taps"]),
            "penalty_candidate_id": _CURRENT_PENALTY_ID,
            "recovery_candidate_id": _CURRENT_RECOVERY_ID,
        }
    )
    config["parameters"] = parameters
    config_hash = canonical_sha256(config)
    identity = AttemptIdentity(
        solver_hash=solver_hash,
        config_hash=config_hash,
        metric_contract_hash=metric_contract_hash,
        evaluation_hash=evaluation_hash,
        data_sha256=str(template["data_sha256"]),
        record_id=str(template["record_id"]),
        stage=_STAGE,
        attempt_kind=_ATTEMPT_KIND,
        parent_experiment_id=parent_experiment_id,
    )
    return {
        **identity.to_dict(),
        "scene": template["scene"],
        "data_path": template["data_path"],
        "reference_path": template["reference_path"],
        "raw_data_sha256": template["raw_data_sha256"],
        "reference_sha256": template["reference_sha256"],
        "method_names": list(template["method_names"]),
        "true_rise_applicable": template["true_rise_applicable"],
        "config": config,
        "filter_profile_id": profile["profile_id"],
        "filter_profile_sha256": profile["profile_sha256"],
        "physical_memory_ms": profile["physical_memory_ms"],
        "actual_taps": profile["actual_taps"],
        "nominal_mu": profile["nominal_mu"],
        "sentinel_role": None,
        "candidate_min_bpm": template["candidate_min_bpm"],
        "recovery_candidate_id": _CURRENT_RECOVERY_ID,
        "penalty_candidate_id": _CURRENT_PENALTY_ID,
        "spectral_audit_required": True,
        "source_stage_r_identity_sha256": template["identity_sha256"],
    }


def _decision_contract_v1() -> dict[str, Any]:
    payload = {
        "contract_version": "lyx_p25_spectral_decision_contract_v1",
        "reads_only": [
            "spectral_audit.stability_pass",
            "spectral_audit.stage_r_spectral_gate",
        ],
        "forbidden_decision_inputs": [
            "metrics",
            "mae",
            "l10",
            "l20",
            "recovery",
        ],
        "branches": {
            "stage_r_sentinel_revision_candidate": (
                "at least one profile passes the complete spectral gate on all 12 records"
            ),
            "spectral_metric_control_audit_required": (
                "all 36 coordinates fail pulse_power_retention_pass"
            ),
            "filter_mechanism_revision_required": (
                "no profile passes all 12 records and pulse-power failures are not universal"
            ),
        },
        "automatic_stage_r_execution": False,
        "automatic_independent_bo_execution": False,
        "may_nominate_recovery_candidate": False,
    }
    payload["contract_sha256"] = canonical_sha256(payload)
    return payload


def build_p25_spectral_diagnostic_proposal(
    *,
    stage_r_proposal: Mapping[str, Any],
    stage_r_completion: Mapping[str, Any],
    profile_library: Mapping[str, Any],
    budget_contract: Mapping[str, Any],
    parent_experiment_id: str,
    solver_hash: str,
    evaluation_hash: str,
) -> dict[str, Any]:
    """Build exactly 36 zero-run p25 spectral diagnostic identities."""

    if not parent_experiment_id:
        raise P25SpectralDiagnosticError("parent_experiment_id_must_not_be_empty")
    solver_hash = _require_hash("solver_hash", solver_hash)
    evaluation_hash = _require_hash("evaluation_hash", evaluation_hash)
    templates = _stage_r_templates(stage_r_proposal)
    stage_r_proposal_sha = str(stage_r_proposal["proposal_sha256"])
    completion_sha = _validate_stage_r_completion(
        stage_r_completion,
        expected_proposal_sha=stage_r_proposal_sha,
    )
    profiles = _p25_profiles_from_library(profile_library)
    budget = _verify_budget_contract(budget_contract)
    frozen_contracts = _require_mapping(
        "stage_r_frozen_contracts",
        stage_r_proposal.get("frozen_contracts"),
    )
    metric_contract_hash = _require_hash(
        "metric_contract_hash",
        frozen_contracts.get("metric_contract_hash"),
    )
    spectral_gate_contract_hash = _require_hash(
        "spectral_gate_contract_hash",
        frozen_contracts.get("spectral_gate_contract_hash"),
    )
    identities = [
        _identity_item(
            template=template,
            profile=profile,
            parent_experiment_id=parent_experiment_id,
            solver_hash=solver_hash,
            evaluation_hash=evaluation_hash,
            metric_contract_hash=metric_contract_hash,
        )
        for profile in profiles
        for template in templates
    ]
    identity_hashes = [str(item["identity_sha256"]) for item in identities]
    if len(identities) != 36 or len(set(identity_hashes)) != 36:
        raise P25SpectralDiagnosticError("p25_spectral_identity_matrix_mismatch")
    record_panel = [
        {
            key: template[key]
            for key in (
                "record_id",
                "scene",
                "raw_data_sha256",
                "reference_sha256",
                "data_sha256",
                "method_names",
                "true_rise_applicable",
            )
        }
        for template in templates
    ]
    profile_panel = [
        {
            key: profile[key]
            for key in (
                "profile_id",
                "profile_sha256",
                "fs_target",
                "physical_memory_ms",
                "actual_taps",
                "nominal_mu",
            )
        }
        for profile in profiles
    ]
    decision_contract = _decision_contract_v1()
    proposal = {
        "proposal_version": "lyx_p25_spectral_diagnostic_proposal_v1",
        "status": "awaiting_human_execution_authorization",
        "authorization_state": _AUTHORIZATION_STATE,
        "parent_experiment_id": parent_experiment_id,
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "unique_budget": 36,
        "retry_limit": 1,
        "worst_case_attempt_budget": 72,
        "independent_bo_authorized": False,
        "may_nominate_recovery_candidate": False,
        "automatic_stage_r_execution": False,
        "upstream_stage_r": {
            "proposal_sha256": stage_r_proposal_sha,
            "completion_sha256": completion_sha,
            "status": "no_safe_recovery_candidate",
        },
        "frozen_contracts": {
            "solver_hash": solver_hash,
            "evaluation_hash": evaluation_hash,
            "metric_contract_hash": metric_contract_hash,
            "spectral_gate_contract_hash": spectral_gate_contract_hash,
            "filter_profile_design_rule_hash": _require_hash(
                "filter_profile_design_rule_hash",
                profile_library.get("design_rule_sha256"),
            ),
            "filter_profile_library_hash": _require_hash(
                "filter_profile_library_hash",
                profile_library.get("library_sha256"),
            ),
            "budget_contract_hash": budget.sha256,
            "decision_contract_hash": decision_contract["contract_sha256"],
        },
        "profile_panel": profile_panel,
        "profile_panel_sha256": canonical_sha256(profile_panel),
        "record_panel": record_panel,
        "record_panel_sha256": canonical_sha256(record_panel),
        "decision_contract": decision_contract,
        "identity_sha256": identity_hashes,
        "identities": identities,
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def evaluate_p25_spectral_diagnostic_decision(
    result_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify the 36-coordinate panel from spectral evidence only."""

    if len(result_rows) != 36:
        raise P25SpectralDiagnosticError("p25_spectral_decision_result_count_mismatch")
    coordinates: set[tuple[str, str]] = set()
    profile_summaries: dict[str, dict[str, int]] = {
        profile_id: {
            "coordinate_count": 0,
            "complete_spectral_pass_count": 0,
            "pulse_power_retention_pass_count": 0,
        }
        for profile_id in _P25_PROFILE_IDS
    }
    for raw in result_rows:
        row = _require_mapping("p25_spectral_result_row", raw)
        profile_id = str(row.get("filter_profile_id", ""))
        record_id = str(row.get("record_id", ""))
        if profile_id not in profile_summaries or record_id not in _EXPECTED_RECORD_IDS:
            raise P25SpectralDiagnosticError("p25_spectral_decision_coordinate_outside_panel")
        coordinate = (profile_id, record_id)
        if coordinate in coordinates:
            raise P25SpectralDiagnosticError("p25_spectral_decision_duplicate_coordinate")
        coordinates.add(coordinate)
        audit = _require_mapping(
            "p25_spectral_result_audit",
            row.get("spectral_audit"),
        )
        spectral_gate = _require_mapping(
            "p25_spectral_result_stage_r_spectral_gate",
            audit.get("stage_r_spectral_gate"),
        )
        gates = _require_mapping(
            "p25_spectral_result_gates",
            spectral_gate.get("gates"),
        )
        stability_pass = audit.get("stability_pass")
        spectral_gate_pass = spectral_gate.get("spectral_gate_pass")
        pulse_pass = gates.get("pulse_power_retention_pass")
        if not all(
            isinstance(value, bool)
            for value in (
                stability_pass,
                spectral_gate_pass,
                pulse_pass,
            )
        ):
            raise P25SpectralDiagnosticError("p25_spectral_decision_gate_value_must_be_boolean")
        summary = profile_summaries[profile_id]
        summary["coordinate_count"] += 1
        summary["complete_spectral_pass_count"] += int(stability_pass and spectral_gate_pass)
        summary["pulse_power_retention_pass_count"] += int(pulse_pass)

    expected_coordinates = {
        (profile_id, record_id)
        for profile_id in _P25_PROFILE_IDS
        for record_id in _EXPECTED_RECORD_IDS
    }
    if coordinates != expected_coordinates:
        raise P25SpectralDiagnosticError("p25_spectral_decision_coordinate_matrix_mismatch")
    complete_profiles = [
        profile_id
        for profile_id in _P25_PROFILE_IDS
        if profile_summaries[profile_id]["complete_spectral_pass_count"] == 12
    ]
    pulse_pass_count = sum(
        summary["pulse_power_retention_pass_count"] for summary in profile_summaries.values()
    )
    if complete_profiles:
        decision = "stage_r_sentinel_revision_candidate"
    elif pulse_pass_count == 0:
        decision = "spectral_metric_control_audit_required"
    else:
        decision = "filter_mechanism_revision_required"
    result: dict[str, Any] = {
        "decision_version": "lyx_p25_spectral_decision_v1",
        "decision": decision,
        "result_count": 36,
        "complete_pass_profile_ids": complete_profiles,
        "pulse_power_retention_pass_count": pulse_pass_count,
        "profile_summaries": profile_summaries,
        "independent_bo_authorized": False,
        "may_nominate_recovery_candidate": False,
    }
    result["decision_sha256"] = canonical_sha256(result)
    return result


def propose_p25_spectral_diagnostic(
    *,
    stage_r_proposal_path: Path,
    stage_r_completion_path: Path,
    profile_library_path: Path,
    budget_contract_path: Path,
    metric_contract_path: Path,
    spectral_gate_contract_path: Path,
    output_dir: Path,
    source_root: Path,
    parent_experiment_id: str,
) -> dict[str, Any]:
    """Publish a content-addressed zero-run review package."""

    artifact_paths = {
        "stage_r_proposal": Path(stage_r_proposal_path).resolve(),
        "stage_r_completion": Path(stage_r_completion_path).resolve(),
        "profile_library": Path(profile_library_path).resolve(),
        "budget_contract": Path(budget_contract_path).resolve(),
        "metric_contract": Path(metric_contract_path).resolve(),
        "spectral_gate_contract": Path(spectral_gate_contract_path).resolve(),
    }
    missing = [name for name, path in artifact_paths.items() if not path.is_file()]
    if missing:
        raise P25SpectralDiagnosticError(
            "p25_spectral_source_artifact_missing:" + ",".join(missing)
        )
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise P25SpectralDiagnosticError("p25_spectral_proposal_destination_exists")

    source_root = Path(source_root).resolve()
    stage_r_proposal = read_json(artifact_paths["stage_r_proposal"])
    stage_r_completion = read_json(artifact_paths["stage_r_completion"])
    profile_library = read_json(artifact_paths["profile_library"])
    budget_contract = read_json(artifact_paths["budget_contract"])
    metric_contract = read_json(artifact_paths["metric_contract"])
    spectral_gate_contract = read_json(artifact_paths["spectral_gate_contract"])
    metric_contract_hash = _verify_embedded_hash(
        metric_contract,
        hash_field="contract_sha256",
        artifact_name="metric_contract",
    )
    spectral_gate_contract_hash = _verify_embedded_hash(
        spectral_gate_contract,
        hash_field="contract_sha256",
        artifact_name="spectral_gate_contract",
    )
    stage_r_frozen = _require_mapping(
        "stage_r_frozen_contracts",
        stage_r_proposal.get("frozen_contracts"),
    )
    if metric_contract_hash != stage_r_frozen.get(
        "metric_contract_hash"
    ) or spectral_gate_contract_hash != stage_r_frozen.get("spectral_gate_contract_hash"):
        raise P25SpectralDiagnosticError("p25_spectral_stage_r_contract_artifact_mismatch")

    solver_identity = runtime_source_identity(source_root)
    evaluation_identity = runtime_source_identity(
        source_root,
        root_modules=("ppg_hr.v2.recovery_p25_spectral_diagnostic",),
    )
    proposal = build_p25_spectral_diagnostic_proposal(
        stage_r_proposal=stage_r_proposal,
        stage_r_completion=stage_r_completion,
        profile_library=profile_library,
        budget_contract=budget_contract,
        parent_experiment_id=parent_experiment_id,
        solver_hash=str(solver_identity["source_bundle_sha256"]),
        evaluation_hash=str(evaluation_identity["source_bundle_sha256"]),
    )
    proposal.pop("proposal_sha256")
    proposal["source_artifacts"] = {
        name: {
            "path": str(path),
            "file_sha256": file_sha256(path),
        }
        for name, path in artifact_paths.items()
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)

    budget_request = {
        "request_version": ("lyx_p25_spectral_budget_amendment_request_v1"),
        "status": "awaiting_human_budget_and_execution_decision",
        "approved": False,
        "decision_state": _AUTHORIZATION_STATE,
        "proposal_sha256": proposal["proposal_sha256"],
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "added_unique_identities": 36,
        "normal_unique_identity_limit": 780,
        "max_unique_identities": 792,
        "max_attempts": 1584,
        "retry_limit": 1,
        "budget_contract_hash": proposal["frozen_contracts"]["budget_contract_hash"],
        "profile_panel_sha256": proposal["profile_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "independent_bo_authorized": False,
    }
    budget_request["request_sha256"] = canonical_sha256(budget_request)
    receipt: dict[str, Any] = {
        "receipt_version": ("lyx_p25_spectral_diagnostic_proposal_receipt_v1"),
        "status": "awaiting_human_execution_authorization",
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_request_sha256": budget_request["request_sha256"],
        "identity_count": 36,
        "diagnostic_solver_run_count": 0,
        "independent_bo_run_count": 0,
        "may_execute": False,
    }

    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        staging.mkdir(parents=True)
        atomic_write_json(
            staging / "p25_spectral_diagnostic_proposal.json",
            proposal,
        )
        atomic_write_json(
            staging / "budget_amendment_request.json",
            budget_request,
        )
        atomic_write_json(
            staging / "metric_contract.json",
            metric_contract,
        )
        atomic_write_json(
            staging / "spectral_gate_contract.json",
            spectral_gate_contract,
        )
        atomic_write_json(
            staging / "solver_source_identity.json",
            solver_identity,
        )
        atomic_write_json(
            staging / "evaluation_source_identity.json",
            evaluation_identity,
        )
        atomic_write_json(
            staging / "decision_contract.json",
            proposal["decision_contract"],
        )
        receipt["artifact_sha256"] = {
            path.name: file_sha256(path) for path in staging.iterdir() if path.is_file()
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_write_json(staging / "proposal_receipt.json", receipt)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return receipt


def _attempt_identity_from_item(
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


def _exploration_registry_from_payload(
    payload: Mapping[str, Any],
) -> ExplorationRegistry:
    return ExplorationRegistry(
        registry_version=str(payload["registry_version"]),
        unique_budget=int(payload["unique_budget"]),
        allowed_identity_sha256=tuple(
            str(value)
            for value in _require_list(
                "allowed_identity_sha256",
                payload.get("allowed_identity_sha256"),
            )
        ),
    )


def prepare_p25_spectral_diagnostic_governance(
    *,
    proposal_dir: Path,
    authorization_receipt_path: Path | None,
    source_governance_dir: Path,
    governance_dir: Path,
) -> dict[str, Any]:
    """Migrate v5 governance and register 36 identities after approval."""

    proposal_root = Path(proposal_dir).resolve()
    proposal = read_json(proposal_root / "p25_spectral_diagnostic_proposal.json")
    authorization = (
        None
        if authorization_receipt_path is None
        else read_json(Path(authorization_receipt_path).resolve())
    )
    validated = validate_p25_spectral_diagnostic_authorization(
        proposal,
        receipt=authorization,
    )
    target_root = Path(governance_dir).resolve()
    if target_root.exists():
        raise P25SpectralDiagnosticError("p25_spectral_governance_destination_exists")
    budget_request = read_json(proposal_root / "budget_amendment_request.json")
    _verify_embedded_hash(
        budget_request,
        hash_field="request_sha256",
        artifact_name="p25_spectral_budget_request",
    )
    frozen = _require_mapping(
        "p25_spectral_frozen_contracts",
        proposal.get("frozen_contracts"),
    )
    target_budget = BudgetContract.approved_v6_p25_diagnostic()
    if (
        target_budget.sha256 != frozen.get("budget_contract_hash")
        or budget_request.get("proposal_sha256") != proposal.get("proposal_sha256")
        or budget_request.get("budget_contract_hash") != target_budget.sha256
    ):
        raise P25SpectralDiagnosticError("p25_spectral_budget_request_mismatch")
    identities = tuple(
        _attempt_identity_from_item(_require_mapping("p25_spectral_identity", item))
        for item in _require_list(
            "p25_spectral_identities",
            proposal.get("identities"),
        )
    )
    if len(identities) != 36 or tuple(identity.sha256 for identity in identities) != tuple(
        str(value) for value in proposal["identity_sha256"]
    ):
        raise P25SpectralDiagnosticError("p25_spectral_governance_identity_matrix_mismatch")

    source_root = Path(source_governance_dir).resolve()
    source_budget_payload = read_json(source_root / "budget_contract.json")
    source_budget = BudgetContract.approved_v5()
    if source_budget_payload != source_budget.to_dict():
        raise P25SpectralDiagnosticError("p25_spectral_source_budget_mismatch")
    exploration_payload = read_json(source_root / "exploration_registry.json")
    exploration = _exploration_registry_from_payload(exploration_payload)
    if exploration.to_dict() != exploration_payload:
        raise P25SpectralDiagnosticError("p25_spectral_source_exploration_registry_mismatch")
    source_registry = AttemptRegistry.open(
        source_root / "attempt_registry.json",
        budget_contract=source_budget,
        exploration_registry=exploration,
    )
    amendment = BudgetAmendmentRequest(
        stage=_STAGE,
        profile_design_rule_hash=str(proposal["profile_panel_sha256"]),
        record_manifest_hash=str(proposal["record_panel_sha256"]),
        added_unique_identities=36,
        normal_unique_identity_limit=780,
        max_unique_identities=792,
        max_attempts=1584,
    )
    migration_authorization = {
        "approved": True,
        "decision_state": "awaiting_human_budget_decision",
        **{
            "stage": amendment.stage,
            "profile_design_rule_hash": (amendment.profile_design_rule_hash),
            "record_manifest_hash": amendment.record_manifest_hash,
            "added_unique_identities": amendment.added_unique_identities,
            "normal_unique_identity_limit": (amendment.normal_unique_identity_limit),
            "max_unique_identities": amendment.max_unique_identities,
            "max_attempts": amendment.max_attempts,
        },
        "independent_bo_authorized": False,
        "approved_at": validated["approved_at"],
        "approved_by": validated["approved_by"],
    }
    governance_receipt: dict[str, Any] = {}

    def finalize_governance(
        staging: Path,
        staged_registry: AttemptRegistry,
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
            validated,
        )
        governance_receipt = {
            "receipt_version": ("lyx_p25_spectral_governance_receipt_v1"),
            "status": "prepared_zero_runs",
            "proposal_sha256": proposal["proposal_sha256"],
            "source_budget_contract_hash": source_budget.sha256,
            "target_budget_contract_hash": target_budget.sha256,
            "new_unique_identity_count": 36,
            "attempt_registry_summary": staged_registry.summary(),
            "independent_bo_authorized": False,
            "artifacts": {
                name: file_sha256(staging / name)
                for name in (
                    "attempt_registry.json",
                    "budget_contract.json",
                    "exploration_registry.json",
                    "execution_authorization.json",
                )
            },
        }
        governance_receipt["receipt_sha256"] = canonical_sha256(governance_receipt)
        atomic_write_json(
            staging / "governance_receipt.json",
            governance_receipt,
        )

    source_registry.migrate_to(
        target_root / "attempt_registry.json",
        budget_contract=target_budget,
        amendment_request=amendment,
        authorization_receipt=migration_authorization,
        new_identities=identities,
        target_exploration_registry=exploration,
        finalize_staging=finalize_governance,
    )
    return governance_receipt


def _validate_p25_proposal_preflight(
    *,
    proposal_root: Path,
    proposal: Mapping[str, Any],
    source_root: Path,
) -> tuple[dict[str, Any], ...]:
    proposal_hash = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="p25_spectral_proposal",
    )
    receipt = read_json(proposal_root / "proposal_receipt.json")
    _verify_embedded_hash(
        receipt,
        hash_field="receipt_sha256",
        artifact_name="p25_spectral_proposal_receipt",
    )
    if receipt.get("proposal_sha256") != proposal_hash:
        raise P25SpectralDiagnosticError("p25_spectral_proposal_receipt_mismatch")
    for name, expected in _require_mapping(
        "p25_spectral_proposal_artifact_hashes",
        receipt.get("artifact_sha256"),
    ).items():
        path = proposal_root / str(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise P25SpectralDiagnosticError(f"p25_spectral_proposal_artifact_mismatch:{name}")
    source_artifacts = _require_mapping(
        "p25_spectral_source_artifacts",
        proposal.get("source_artifacts"),
    )
    for name, raw in source_artifacts.items():
        artifact = _require_mapping(
            f"p25_spectral_source_artifact:{name}",
            raw,
        )
        path = Path(str(artifact.get("path", ""))).resolve()
        if not path.is_file() or file_sha256(path) != artifact.get("file_sha256"):
            raise P25SpectralDiagnosticError(f"p25_spectral_source_artifact_mismatch:{name}")
    frozen = _require_mapping(
        "p25_spectral_frozen_contracts",
        proposal.get("frozen_contracts"),
    )
    current_solver = runtime_source_identity(Path(source_root).resolve())
    current_evaluation = runtime_source_identity(
        Path(source_root).resolve(),
        root_modules=("ppg_hr.v2.recovery_p25_spectral_diagnostic",),
    )
    if current_solver.get("source_bundle_sha256") != frozen.get(
        "solver_hash"
    ) or current_evaluation.get("source_bundle_sha256") != frozen.get("evaluation_hash"):
        raise P25SpectralDiagnosticError("p25_spectral_runtime_source_identity_mismatch")
    identities = tuple(
        dict(_require_mapping("p25_spectral_identity", item))
        for item in _require_list(
            "p25_spectral_identities",
            proposal.get("identities"),
        )
    )
    if len(identities) != 36 or tuple(str(item["identity_sha256"]) for item in identities) != tuple(
        str(value) for value in proposal["identity_sha256"]
    ):
        raise P25SpectralDiagnosticError("p25_spectral_execution_identity_matrix_mismatch")
    return identities


def _validate_completed_p25_execution(
    *,
    completion_path: Path,
    proposal: Mapping[str, Any],
    authorization_sha256: str,
    output_dir: Path,
    governance_dir: Path,
) -> dict[str, Any]:
    completion = read_json(completion_path)
    _verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="p25_spectral_completion",
    )
    if (
        completion.get("proposal_sha256") != proposal.get("proposal_sha256")
        or completion.get("authorization_sha256") != authorization_sha256
        or completion.get("diagnostic_result_count") != 36
        or completion.get("independent_bo_run_count") != 0
    ):
        raise P25SpectralDiagnosticError("p25_spectral_completion_identity_mismatch")
    for name, expected in _require_mapping(
        "p25_spectral_completion_artifacts",
        completion.get("artifacts"),
    ).items():
        path = output_dir / str(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise P25SpectralDiagnosticError(f"p25_spectral_completion_artifact_mismatch:{name}")
    governance_receipt_path = governance_dir / "p25_spectral_execution_receipt.json"
    if not governance_receipt_path.is_file() or file_sha256(
        governance_receipt_path
    ) != completion.get("governance_receipt_file_sha256"):
        raise P25SpectralDiagnosticError("p25_spectral_completion_governance_receipt_mismatch")
    governance_receipt = read_json(governance_receipt_path)
    if _verify_embedded_hash(
        governance_receipt,
        hash_field="receipt_sha256",
        artifact_name="p25_spectral_execution_receipt",
    ) != completion.get("governance_receipt_sha256"):
        raise P25SpectralDiagnosticError("p25_spectral_completion_governance_receipt_mismatch")
    return completion


def execute_p25_spectral_diagnostic(
    *,
    proposal_dir: Path,
    governance_dir: Path,
    output_dir: Path,
    source_root: Path,
    _numerical_runner: StageRNumericalRunner | None = None,
    progress_callback: (Callable[[Mapping[str, Any]], None] | None) = None,
) -> dict[str, Any]:
    """Execute only the 36 registered diagnostics and freeze one decision."""

    proposal_root = Path(proposal_dir).resolve()
    proposal = read_json(proposal_root / "p25_spectral_diagnostic_proposal.json")
    identities_raw = _validate_p25_proposal_preflight(
        proposal_root=proposal_root,
        proposal=proposal,
        source_root=source_root,
    )
    governance_root = Path(governance_dir).resolve()
    authorization = read_json(governance_root / "execution_authorization.json")
    authorization = validate_p25_spectral_diagnostic_authorization(
        proposal,
        receipt=authorization,
    )
    authorization_sha256 = canonical_sha256(authorization)
    budget_payload = read_json(governance_root / "budget_contract.json")
    budget = _budget_contract_from_payload(budget_payload)
    expected_budget = BudgetContract.approved_v6_p25_diagnostic()
    if (
        budget.to_dict() != expected_budget.to_dict()
        or budget.sha256 != proposal["frozen_contracts"]["budget_contract_hash"]
    ):
        raise P25SpectralDiagnosticError("p25_spectral_execution_budget_mismatch")
    exploration = _exploration_registry_from_payload(
        read_json(governance_root / "exploration_registry.json")
    )
    registry = AttemptRegistry.open(
        governance_root / "attempt_registry.json",
        budget_contract=budget,
        exploration_registry=exploration,
    )
    identities = tuple(_attempt_identity_from_item(item) for item in identities_raw)

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    binding = {
        "binding_version": "lyx_p25_spectral_execution_binding_v1",
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": authorization_sha256,
        "identity_matrix_sha256": canonical_sha256([identity.sha256 for identity in identities]),
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    binding_path = destination / "execution_binding.json"
    if binding_path.is_file():
        if read_json(binding_path) != binding:
            raise P25SpectralDiagnosticError("p25_spectral_execution_binding_mismatch")
    else:
        atomic_write_json(binding_path, binding)
    completion_path = destination / "completion.json"
    if completion_path.is_file():
        return _validate_completed_p25_execution(
            completion_path=completion_path,
            proposal=proposal,
            authorization_sha256=authorization_sha256,
            output_dir=destination,
            governance_dir=governance_root,
        )

    registered = registry.register_identities(identities)
    if registered != tuple(identity.sha256 for identity in identities):
        raise P25SpectralDiagnosticError("p25_spectral_bulk_registration_mismatch")
    if _numerical_runner is None:
        from .recovery_stage_r_experiment import (
            run_stage_r_numerical_identity,
        )

        runner = run_stage_r_numerical_identity
    else:
        runner = _numerical_runner
    results: list[dict[str, Any]] = []
    spectral_audit_dir = destination / "spectral_audits"
    for index, item in enumerate(identities_raw, start=1):
        result = stage_r_cache.execute_stage_r_identity(
            registry=registry,
            item=dict(item),
            numerical_runner=runner,
            spectral_audit_dir=spectral_audit_dir,
        )
        results.append(result)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "p25_spectral_identity_complete",
                    "completed": index,
                    "total": 36,
                    "identity_sha256": result["identity_sha256"],
                    "record_id": result["record_id"],
                    "filter_profile_id": result["filter_profile_id"],
                    "cache_hit": result["cache_hit"],
                }
            )
    registry.assert_complete_matrix(identities)
    result_index = {
        "index_version": "lyx_p25_spectral_result_index_v1",
        "proposal_sha256": proposal["proposal_sha256"],
        "result_count": 36,
        "results": results,
    }
    result_index["index_sha256"] = canonical_sha256(result_index)
    atomic_write_json(
        destination / "identity_result_index.json",
        result_index,
    )
    decision = evaluate_p25_spectral_diagnostic_decision(results)
    decision["proposal_sha256"] = proposal["proposal_sha256"]
    decision.pop("decision_sha256")
    decision["decision_sha256"] = canonical_sha256(decision)
    atomic_write_json(destination / "decision_receipt.json", decision)
    profile_summary = {
        "summary_version": "lyx_p25_profile_gate_summary_v1",
        "proposal_sha256": proposal["proposal_sha256"],
        "result_count": 36,
        "complete_pass_profile_ids": decision["complete_pass_profile_ids"],
        "pulse_power_retention_pass_count": decision["pulse_power_retention_pass_count"],
        "profiles": decision["profile_summaries"],
    }
    profile_summary["summary_sha256"] = canonical_sha256(profile_summary)
    atomic_write_json(
        destination / "profile_gate_summary.json",
        profile_summary,
    )
    matrix_summary = registry.matrix_execution_summary(identities)
    matrix_snapshot = registry.matrix_snapshot(identities)
    atomic_write_json(
        destination / "attempt_registry_p25_snapshot.json",
        matrix_snapshot,
    )
    artifacts = {
        name: file_sha256(destination / name)
        for name in (
            "execution_binding.json",
            "identity_result_index.json",
            "profile_gate_summary.json",
            "decision_receipt.json",
            "attempt_registry_p25_snapshot.json",
        )
    }
    execution_receipt = {
        "receipt_version": ("lyx_p25_spectral_execution_governance_receipt_v1"),
        "status": decision["decision"],
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": authorization_sha256,
        "matrix_execution_summary": matrix_summary,
        "attempt_registry_snapshot_sha256": matrix_snapshot["snapshot_sha256"],
        "independent_bo_run_count": 0,
        "artifacts": artifacts,
    }
    execution_receipt["receipt_sha256"] = canonical_sha256(execution_receipt)
    execution_receipt_path = governance_root / "p25_spectral_execution_receipt.json"
    atomic_write_json(execution_receipt_path, execution_receipt)
    next_states = {
        "stage_r_sentinel_revision_candidate": ("awaiting_human_stage_r_revision_decision"),
        "filter_mechanism_revision_required": ("awaiting_filter_mechanism_revision"),
        "spectral_metric_control_audit_required": ("awaiting_spectral_metric_control_audit"),
    }
    completion = {
        "completion_version": "lyx_p25_spectral_completion_v1",
        "status": decision["decision"],
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": authorization_sha256,
        "diagnostic_result_count": 36,
        "diagnostic_solver_run_count": matrix_summary["identity_with_solver_attempt_count"],
        "independent_bo_run_count": 0,
        "may_nominate_recovery_candidate": False,
        "next_state": next_states[str(decision["decision"])],
        "matrix_execution_summary": matrix_summary,
        "artifacts": artifacts,
        "governance_receipt_sha256": execution_receipt["receipt_sha256"],
        "governance_receipt_file_sha256": file_sha256(execution_receipt_path),
    }
    completion["completion_sha256"] = canonical_sha256(completion)
    atomic_write_json(completion_path, completion)
    return _validate_completed_p25_execution(
        completion_path=completion_path,
        proposal=proposal,
        authorization_sha256=authorization_sha256,
        output_dir=destination,
        governance_dir=governance_root,
    )


def validate_p25_spectral_diagnostic_authorization(
    proposal: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed unless the receipt approves this exact 36-identity plan."""

    declared_proposal_hash = _require_hash(
        "proposal_sha256",
        proposal.get("proposal_sha256"),
    )
    unhashed = dict(proposal)
    unhashed.pop("proposal_sha256", None)
    if canonical_sha256(unhashed) != declared_proposal_hash:
        raise P25SpectralDiagnosticAuthorizationError("p25_spectral_proposal_hash_mismatch")
    if receipt is None or receipt.get("approved") is not True:
        raise P25SpectralDiagnosticAuthorizationError(
            "p25_spectral_execution_authorization_required"
        )
    required = {
        "approved",
        "decision_state",
        "proposal_sha256",
        "budget_contract_hash",
        "unique_budget",
        "stage",
        "profile_panel_sha256",
        "record_panel_sha256",
        "solver_hash",
        "evaluation_hash",
        "metric_contract_hash",
        "spectral_gate_contract_hash",
        "decision_contract_hash",
        "independent_bo_authorized",
        "approved_at",
        "approved_by",
    }
    missing = sorted(required - set(receipt))
    if missing:
        raise P25SpectralDiagnosticAuthorizationError(
            "p25_spectral_authorization_missing_fields:" + ",".join(missing)
        )
    frozen = _require_mapping(
        "p25_spectral_frozen_contracts",
        proposal.get("frozen_contracts"),
    )
    expected = {
        "decision_state": _AUTHORIZATION_STATE,
        "proposal_sha256": declared_proposal_hash,
        "budget_contract_hash": frozen.get("budget_contract_hash"),
        "unique_budget": 36,
        "stage": _STAGE,
        "profile_panel_sha256": proposal.get("profile_panel_sha256"),
        "record_panel_sha256": proposal.get("record_panel_sha256"),
        "solver_hash": frozen.get("solver_hash"),
        "evaluation_hash": frozen.get("evaluation_hash"),
        "metric_contract_hash": frozen.get("metric_contract_hash"),
        "spectral_gate_contract_hash": frozen.get("spectral_gate_contract_hash"),
        "decision_contract_hash": frozen.get("decision_contract_hash"),
    }
    mismatched = sorted(
        name for name, expected_value in expected.items() if receipt.get(name) != expected_value
    )
    if mismatched:
        raise P25SpectralDiagnosticAuthorizationError(
            "p25_spectral_authorization_identity_mismatch:" + ",".join(mismatched)
        )
    if receipt.get("independent_bo_authorized") is not False:
        raise P25SpectralDiagnosticAuthorizationError(
            "p25_spectral_independent_bo_must_remain_unauthorized"
        )
    for name in (
        "proposal_sha256",
        "budget_contract_hash",
        "profile_panel_sha256",
        "record_panel_sha256",
        "solver_hash",
        "evaluation_hash",
        "metric_contract_hash",
        "spectral_gate_contract_hash",
        "decision_contract_hash",
    ):
        _require_hash(f"authorization_{name}", receipt.get(name))
    if not isinstance(receipt.get("approved_at"), str) or not receipt["approved_at"]:
        raise P25SpectralDiagnosticAuthorizationError(
            "p25_spectral_authorization_approved_at_invalid"
        )
    if not isinstance(receipt.get("approved_by"), str) or not receipt["approved_by"]:
        raise P25SpectralDiagnosticAuthorizationError(
            "p25_spectral_authorization_approved_by_invalid"
        )
    return dict(receipt)
