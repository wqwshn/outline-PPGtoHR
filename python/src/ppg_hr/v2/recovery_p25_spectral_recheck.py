"""Corrected p25 spectral recheck after the metric-scale control.

The proposal builder in this module is deliberately pure: it binds the exact
historical p25 panel, the completed scale-control evidence, the corrected v2
spectral contract, and a new 36-identity governance budget.  Building a
proposal does not authorize or execute any diagnostic.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .experiment_freeze_utils import runtime_source_identity
from .phase2_experiment_io import atomic_write_json, file_sha256, read_json
from .recovery_contracts import canonical_sha256, require_sha256
from .recovery_experiment_governance import (
    AttemptIdentity,
    AttemptRegistry,
    BudgetAmendmentRequest,
    BudgetContract,
    ExplorationRegistry,
    GovernanceError,
)
from .recovery_filter_profiles import FilterProfile
from .recovery_filter_stability import FilterAuditRecord
from .recovery_spectral_gate import (
    StageRSpectralGateContract,
    audit_stage_r_profile_record,
)


class P25SpectralRecheckError(RuntimeError):
    """The corrected p25 recheck violates a frozen evidence contract."""


class P25SpectralRecheckAuthorizationError(P25SpectralRecheckError):
    """The exact corrected panel and v8 budget have not been approved."""


_STAGE = "filter_profile_p25_spectral_recheck_v2"
_ATTEMPT_KIND = "diagnostic"
_AUTHORIZATION_STATE = "awaiting_human_p25_spectral_recheck_v2_decision"
_PRIOR_P25_PROPOSAL_SHA256 = (
    "db1f5d2278458592c08d7e6217d52090a8ab1f94b96262de99e84a466e2c6128"
)
_PRIOR_P25_COMPLETION_SHA256 = (
    "81171332c7c29e80f329c9140874480321bf101ca7f91c853c3ae453015251ac"
)
_PRIOR_P25_DECISION_SHA256 = (
    "29f042e080441ae979fea32e0b38c9de8ac778fc71d3bd5d99d78ba8ba763f55"
)
_SCALE_CONTROL_PROPOSAL_SHA256 = (
    "429233ecadf92cc2d669b59c8f2cc4516b3d767547d98919655a18918e1a60bd"
)
_SCALE_CONTROL_COMPLETION_SHA256 = (
    "338c73c2360b7bd0d7fad849fbdd84e6b991c1fa1eb823068e8a990ff937a5d0"
)
_SCALE_CONTROL_DECISION_SHA256 = (
    "4aa202167f754532e8e23318906ba3075ae4807f58080f4ae4a9eed00782f980"
)
_P25_PROFILE_IDS = (
    "p25-short-low",
    "p25-short-mid",
    "p25-long-mid",
)
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
_EXPECTED_SCENE_COUNTS = {
    "jianpan": 3,
    "kaihe": 3,
    "run": 3,
    "xiezi": 3,
}
_PROPOSAL_ARTIFACT_NAMES = {
    "budget_amendment_request.json",
    "budget_contract_v8.json",
    "decision_contract.json",
    "metric_contract.json",
    "source_identity.json",
    "spectral_gate_contract.json",
    "p25_spectral_recheck_proposal.json",
}
_SPECTRAL_GATE_NAMES = {
    "prominence_db_delta_pass",
    "visible_top3_rate_delta_pass",
    "hr_band_share_delta_pass",
    "pulse_power_retention_pass",
    "residual_artifact_corr_delta_pass",
    "complete_window_evidence_pass",
}


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise P25SpectralRecheckError(f"{name}_must_be_object")
    return value


def _require_list(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise P25SpectralRecheckError(f"{name}_must_be_array")
    return value


def _require_hash(name: str, value: object) -> str:
    text = str(value)
    try:
        require_sha256(name, text)
    except ValueError as error:
        raise P25SpectralRecheckError(str(error)) from error
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
        raise P25SpectralRecheckError(f"{artifact_name}_hash_mismatch")
    return declared


def _validate_prior_p25(
    *,
    proposal: Mapping[str, Any],
    completion: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="prior_p25_proposal",
    )
    completion_sha = _verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="prior_p25_completion",
    )
    decision_sha = _verify_embedded_hash(
        decision,
        hash_field="decision_sha256",
        artifact_name="prior_p25_decision",
    )
    if (
        proposal_sha != _PRIOR_P25_PROPOSAL_SHA256
        or completion_sha != _PRIOR_P25_COMPLETION_SHA256
        or decision_sha != _PRIOR_P25_DECISION_SHA256
        or completion.get("proposal_sha256") != proposal_sha
        or completion.get("status") != "spectral_metric_control_audit_required"
        or completion.get("diagnostic_result_count") != 36
        or completion.get("independent_bo_run_count") != 0
        or decision.get("proposal_sha256") != proposal_sha
        or decision.get("decision") != "spectral_metric_control_audit_required"
        or decision.get("result_count") != 36
        or decision.get("pulse_power_retention_pass_count") != 0
    ):
        raise P25SpectralRecheckError("prior_p25_evidence_mismatch")

    identities = tuple(
        dict(_require_mapping("prior_p25_identity", raw))
        for raw in _require_list("prior_p25_identities", proposal.get("identities"))
    )
    coordinates = {
        (str(item.get("filter_profile_id", "")), str(item.get("record_id", "")))
        for item in identities
    }
    expected = {
        (profile_id, record_id)
        for profile_id in _P25_PROFILE_IDS
        for record_id in _EXPECTED_RECORD_IDS
    }
    if len(identities) != 36 or coordinates != expected:
        raise P25SpectralRecheckError("prior_p25_identity_matrix_mismatch")
    scene_counts = Counter(
        str(item.get("scene", ""))
        for item in identities
        if item.get("filter_profile_id") == _P25_PROFILE_IDS[0]
    )
    if dict(scene_counts) != _EXPECTED_SCENE_COUNTS:
        raise P25SpectralRecheckError("prior_p25_scene_panel_mismatch")
    return tuple(
        sorted(
            identities,
            key=lambda item: (
                _P25_PROFILE_IDS.index(str(item["filter_profile_id"])),
                str(item["record_id"]),
            ),
        )
    )


def _validate_scale_control(
    *,
    proposal: Mapping[str, Any],
    completion: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="scale_control_proposal",
    )
    completion_sha = _verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="scale_control_completion",
    )
    decision_sha = _verify_embedded_hash(
        decision,
        hash_field="decision_sha256",
        artifact_name="scale_control_decision",
    )
    if (
        proposal_sha != _SCALE_CONTROL_PROPOSAL_SHA256
        or completion_sha != _SCALE_CONTROL_COMPLETION_SHA256
        or decision_sha != _SCALE_CONTROL_DECISION_SHA256
        or completion.get("proposal_sha256") != proposal_sha
        or completion.get("status") != "legacy_scale_mismatch_confirmed"
        or completion.get("diagnostic_result_count") != 12
        or completion.get("diagnostic_run_count") != 12
        or completion.get("parameter_search_run_count") != 0
        or completion.get("independent_bo_run_count") != 0
        or decision.get("proposal_sha256") != proposal_sha
        or decision.get("decision") != "legacy_scale_mismatch_confirmed"
        or decision.get("record_count") != 12
        or decision.get("direct_bypass_pass_count") != 12
        or decision.get("same_scale_zero_update_pass_count") != 12
        or decision.get("legacy_scale_mismatch_reproduced_count") != 12
    ):
        raise P25SpectralRecheckError("scale_control_evidence_mismatch")


def _decision_contract_v2() -> dict[str, Any]:
    payload = {
        "contract_version": "lyx_p25_spectral_recheck_decision_contract_v2",
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
                "at least one profile passes stability and the complete corrected "
                "spectral gate on all 12 records"
            ),
            "p25_failure_review_required": (
                "no profile passes the corrected complete gate on all 12 records; "
                "a human must choose mechanism diagnostics or whether to prepare "
                "an independent-BO review package"
            ),
        },
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "automatic_independent_bo_execution": False,
        "may_nominate_recovery_candidate": False,
    }
    payload["contract_sha256"] = canonical_sha256(payload)
    return payload


def _metric_contract(
    *,
    source_metric_contract_hash: str,
    spectral_gate_contract_hash: str,
    decision_contract_hash: str,
) -> dict[str, Any]:
    payload = {
        "contract_version": "lyx_p25_spectral_recheck_metric_contract_v2",
        "source_metric_contract_hash": source_metric_contract_hash,
        "spectral_gate_contract_hash": spectral_gate_contract_hash,
        "decision_contract_hash": decision_contract_hash,
        "spectral_signal_domain": "sample_zscore_ddof_1_per_window",
        "scale_control_decision_sha256": _SCALE_CONTROL_DECISION_SHA256,
        "evaluation_grain": "filter_profile_x_record",
        "parameter_search": False,
    }
    payload["contract_sha256"] = canonical_sha256(payload)
    return payload


def _identity_item(
    *,
    source: Mapping[str, Any],
    parent_experiment_id: str,
    solver_hash: str,
    evaluation_hash: str,
    metric_contract_hash: str,
    spectral_gate_contract_hash: str,
) -> dict[str, Any]:
    source_config = deepcopy(
        dict(_require_mapping("prior_p25_identity_config", source.get("config")))
    )
    source_parameters = _require_mapping(
        "prior_p25_identity_parameters",
        source_config.get("parameters"),
    )
    config = {
        "execution_mode": "prepared_window_spectral_reaudit",
        "source_config": source_config,
        "source_config_hash": _require_hash(
            "source_config_hash",
            source.get("config_hash"),
        ),
        "spectral_gate_contract_hash": spectral_gate_contract_hash,
        "parameter_search": False,
    }
    identity = AttemptIdentity(
        solver_hash=solver_hash,
        config_hash=canonical_sha256(config),
        metric_contract_hash=metric_contract_hash,
        evaluation_hash=evaluation_hash,
        data_sha256=_require_hash("data_sha256", source.get("data_sha256")),
        record_id=str(source["record_id"]),
        stage=_STAGE,
        attempt_kind=_ATTEMPT_KIND,
        parent_experiment_id=parent_experiment_id,
    )
    copied_fields = {
        key: deepcopy(source[key])
        for key in (
            "scene",
            "data_path",
            "reference_path",
            "raw_data_sha256",
            "reference_sha256",
            "method_names",
            "true_rise_applicable",
            "filter_profile_id",
            "filter_profile_sha256",
            "physical_memory_ms",
            "actual_taps",
            "nominal_mu",
            "candidate_min_bpm",
            "recovery_candidate_id",
            "penalty_candidate_id",
        )
    }
    return {
        **identity.to_dict(),
        **copied_fields,
        "fs_target": int(source_parameters["fs_target"]),
        "config": config,
        "spectral_audit_required": True,
        "source_p25_identity_sha256": _require_hash(
            "source_p25_identity_sha256",
            source.get("identity_sha256"),
        ),
    }


def build_p25_spectral_recheck_proposal(
    *,
    prior_p25_proposal: Mapping[str, Any],
    prior_p25_completion: Mapping[str, Any],
    prior_p25_decision: Mapping[str, Any],
    scale_control_proposal: Mapping[str, Any],
    scale_control_completion: Mapping[str, Any],
    scale_control_decision: Mapping[str, Any],
    source_budget_contract: Mapping[str, Any],
    parent_experiment_id: str,
    solver_hash: str,
    evaluation_hash: str,
) -> dict[str, Any]:
    """Build the exact corrected 3-profile by 12-record zero-run proposal."""

    if not parent_experiment_id:
        raise P25SpectralRecheckError("parent_experiment_id_must_not_be_empty")
    solver_hash = _require_hash("solver_hash", solver_hash)
    evaluation_hash = _require_hash("evaluation_hash", evaluation_hash)
    sources = _validate_prior_p25(
        proposal=prior_p25_proposal,
        completion=prior_p25_completion,
        decision=prior_p25_decision,
    )
    _validate_scale_control(
        proposal=scale_control_proposal,
        completion=scale_control_completion,
        decision=scale_control_decision,
    )
    if dict(source_budget_contract) != BudgetContract.proposed_v7_spectral_metric_control().to_dict():
        raise P25SpectralRecheckError("source_budget_contract_mismatch")

    frozen_prior = _require_mapping(
        "prior_p25_frozen_contracts",
        prior_p25_proposal.get("frozen_contracts"),
    )
    source_metric_hash = _require_hash(
        "source_metric_contract_hash",
        frozen_prior.get("metric_contract_hash"),
    )
    spectral = StageRSpectralGateContract()
    decision_contract = _decision_contract_v2()
    metric_contract = _metric_contract(
        source_metric_contract_hash=source_metric_hash,
        spectral_gate_contract_hash=spectral.sha256,
        decision_contract_hash=decision_contract["contract_sha256"],
    )
    budget = BudgetContract.proposed_v8_p25_spectral_recheck()
    identities = [
        _identity_item(
            source=source,
            parent_experiment_id=parent_experiment_id,
            solver_hash=solver_hash,
            evaluation_hash=evaluation_hash,
            metric_contract_hash=metric_contract["contract_sha256"],
            spectral_gate_contract_hash=spectral.sha256,
        )
        for source in sources
    ]
    identity_hashes = [str(item["identity_sha256"]) for item in identities]
    if len(identities) != 36 or len(set(identity_hashes)) != 36:
        raise P25SpectralRecheckError("p25_spectral_recheck_identity_matrix_mismatch")
    if set(identity_hashes).intersection(
        str(item["identity_sha256"])
        for item in _require_list(
            "prior_p25_identities",
            prior_p25_proposal.get("identities"),
        )
    ):
        raise P25SpectralRecheckError("p25_spectral_recheck_reuses_prior_identity")

    profile_panel = [
        {
            key: item[key]
            for key in (
                "filter_profile_id",
                "filter_profile_sha256",
                "fs_target",
                "physical_memory_ms",
                "actual_taps",
                "nominal_mu",
            )
        }
        for item in identities
        if item["record_id"] == sorted(_EXPECTED_RECORD_IDS)[0]
    ]
    record_panel = [
        {
            key: source[key]
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
        for source in sources
        if source["filter_profile_id"] == _P25_PROFILE_IDS[0]
    ]
    proposal = {
        "proposal_version": "lyx_p25_spectral_recheck_proposal_v2",
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
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "may_nominate_recovery_candidate": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "upstream_p25": {
            "proposal_sha256": _PRIOR_P25_PROPOSAL_SHA256,
            "completion_sha256": _PRIOR_P25_COMPLETION_SHA256,
            "decision_sha256": _PRIOR_P25_DECISION_SHA256,
            "status": "spectral_metric_control_audit_required",
        },
        "upstream_scale_control": {
            "proposal_sha256": _SCALE_CONTROL_PROPOSAL_SHA256,
            "completion_sha256": _SCALE_CONTROL_COMPLETION_SHA256,
            "decision_sha256": _SCALE_CONTROL_DECISION_SHA256,
            "status": "legacy_scale_mismatch_confirmed",
        },
        "frozen_contracts": {
            "solver_hash": solver_hash,
            "evaluation_hash": evaluation_hash,
            "source_metric_contract_hash": source_metric_hash,
            "metric_contract_hash": metric_contract["contract_sha256"],
            "spectral_gate_contract_hash": spectral.sha256,
            "decision_contract_hash": decision_contract["contract_sha256"],
            "budget_contract_hash": budget.sha256,
        },
        "metric_contract": metric_contract,
        "spectral_gate_contract": spectral.to_dict(),
        "decision_contract": decision_contract,
        "profile_panel": profile_panel,
        "profile_panel_sha256": canonical_sha256(profile_panel),
        "record_panel": record_panel,
        "record_panel_sha256": canonical_sha256(record_panel),
        "identity_sha256": identity_hashes,
        "identity_panel_sha256": canonical_sha256(identity_hashes),
        "identities": identities,
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def evaluate_p25_spectral_recheck_decision(
    result_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify the corrected panel from stability and spectral gates only."""

    if len(result_rows) != 36:
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_decision_result_count_mismatch"
        )
    coordinates: set[tuple[str, str]] = set()
    profile_summaries: dict[str, dict[str, Any]] = {
        profile_id: {
            "coordinate_count": 0,
            "stability_pass_count": 0,
            "spectral_gate_pass_count": 0,
            "complete_pass_count": 0,
            "gate_pass_counts": {
                name: 0 for name in sorted(_SPECTRAL_GATE_NAMES)
            },
        }
        for profile_id in _P25_PROFILE_IDS
    }
    for raw in result_rows:
        row = _require_mapping("p25_spectral_recheck_result_row", raw)
        profile_id = str(row.get("filter_profile_id", ""))
        record_id = str(row.get("record_id", ""))
        if profile_id not in profile_summaries or record_id not in _EXPECTED_RECORD_IDS:
            raise P25SpectralRecheckError(
                "p25_spectral_recheck_coordinate_outside_panel"
            )
        coordinate = (profile_id, record_id)
        if coordinate in coordinates:
            raise P25SpectralRecheckError(
                "p25_spectral_recheck_duplicate_coordinate"
            )
        coordinates.add(coordinate)
        audit = _require_mapping(
            "p25_spectral_recheck_audit",
            row.get("spectral_audit"),
        )
        spectral_gate = _require_mapping(
            "p25_spectral_recheck_stage_r_spectral_gate",
            audit.get("stage_r_spectral_gate"),
        )
        gates = _require_mapping(
            "p25_spectral_recheck_gates",
            spectral_gate.get("gates"),
        )
        if set(gates) != _SPECTRAL_GATE_NAMES:
            raise P25SpectralRecheckError(
                "p25_spectral_recheck_gate_set_mismatch"
            )
        stability_pass = audit.get("stability_pass")
        spectral_gate_pass = spectral_gate.get("spectral_gate_pass")
        gate_values = [gates[name] for name in sorted(_SPECTRAL_GATE_NAMES)]
        if not all(
            isinstance(value, bool)
            for value in (stability_pass, spectral_gate_pass, *gate_values)
        ):
            raise P25SpectralRecheckError(
                "p25_spectral_recheck_gate_value_must_be_boolean"
            )
        if spectral_gate_pass != all(gate_values):
            raise P25SpectralRecheckError(
                "p25_spectral_recheck_gate_summary_mismatch"
            )
        summary = profile_summaries[profile_id]
        summary["coordinate_count"] += 1
        summary["stability_pass_count"] += int(stability_pass)
        summary["spectral_gate_pass_count"] += int(spectral_gate_pass)
        summary["complete_pass_count"] += int(
            stability_pass and spectral_gate_pass
        )
        gate_pass_counts = _require_mapping(
            "p25_spectral_recheck_gate_pass_counts",
            summary["gate_pass_counts"],
        )
        for name in _SPECTRAL_GATE_NAMES:
            gate_pass_counts[name] += int(gates[name])

    expected_coordinates = {
        (profile_id, record_id)
        for profile_id in _P25_PROFILE_IDS
        for record_id in _EXPECTED_RECORD_IDS
    }
    if coordinates != expected_coordinates:
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_coordinate_matrix_mismatch"
        )
    complete_profiles = [
        profile_id
        for profile_id in _P25_PROFILE_IDS
        if profile_summaries[profile_id]["complete_pass_count"] == 12
    ]
    if complete_profiles:
        decision = "stage_r_sentinel_revision_candidate"
        next_state = "awaiting_human_stage_r_revision_decision"
    else:
        decision = "p25_failure_review_required"
        next_state = (
            "awaiting_human_filter_mechanism_or_independent_bo_review"
        )
    global_gate_pass_counts = {
        name: sum(
            int(summary["gate_pass_counts"][name])
            for summary in profile_summaries.values()
        )
        for name in sorted(_SPECTRAL_GATE_NAMES)
    }
    result: dict[str, Any] = {
        "decision_version": "lyx_p25_spectral_recheck_decision_v2",
        "decision": decision,
        "next_state": next_state,
        "result_count": 36,
        "complete_pass_profile_ids": complete_profiles,
        "profile_summaries": profile_summaries,
        "global_gate_pass_counts": global_gate_pass_counts,
        "global_gate_failure_counts": {
            name: 36 - count
            for name, count in global_gate_pass_counts.items()
        },
        "independent_bo_review_package_generated": False,
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "may_nominate_recovery_candidate": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
    }
    result["decision_sha256"] = canonical_sha256(result)
    return result


def _repository_root_from_source_root(source_root: Path) -> Path:
    source = Path(source_root).resolve()
    if source.name != "src" or source.parent.name != "python":
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_source_root_must_be_python_src"
        )
    return source.parents[1]


def propose_p25_spectral_recheck(
    *,
    prior_p25_proposal_path: Path,
    prior_p25_completion_path: Path,
    prior_p25_decision_path: Path,
    scale_control_proposal_path: Path,
    scale_control_completion_path: Path,
    scale_control_decision_path: Path,
    source_budget_contract_path: Path,
    output_dir: Path,
    source_root: Path,
    parent_experiment_id: str,
) -> dict[str, Any]:
    """Publish a content-addressed v2 proposal without running diagnostics."""

    artifact_paths = {
        "prior_p25_proposal": Path(prior_p25_proposal_path).resolve(),
        "prior_p25_completion": Path(prior_p25_completion_path).resolve(),
        "prior_p25_decision": Path(prior_p25_decision_path).resolve(),
        "scale_control_proposal": Path(scale_control_proposal_path).resolve(),
        "scale_control_completion": Path(scale_control_completion_path).resolve(),
        "scale_control_decision": Path(scale_control_decision_path).resolve(),
        "source_budget_contract": Path(source_budget_contract_path).resolve(),
    }
    missing = [name for name, path in artifact_paths.items() if not path.is_file()]
    if missing:
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_source_missing:" + ",".join(missing)
        )
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_destination_exists"
        )
    source_root = Path(source_root).resolve()
    repository_root = _repository_root_from_source_root(source_root)
    relative_artifacts: dict[str, Path] = {}
    for name, path in artifact_paths.items():
        try:
            relative_artifacts[name] = path.relative_to(repository_root)
        except ValueError as error:
            raise P25SpectralRecheckError(
                f"p25_spectral_recheck_source_outside_repository:{name}"
            ) from error
    source_identity = runtime_source_identity(
        source_root,
        root_modules=("ppg_hr.v2.recovery_p25_spectral_recheck",),
    )
    bundle_hash = _require_hash(
        "source_bundle_sha256",
        source_identity.get("source_bundle_sha256"),
    )
    proposal = build_p25_spectral_recheck_proposal(
        prior_p25_proposal=read_json(
            artifact_paths["prior_p25_proposal"]
        ),
        prior_p25_completion=read_json(
            artifact_paths["prior_p25_completion"]
        ),
        prior_p25_decision=read_json(
            artifact_paths["prior_p25_decision"]
        ),
        scale_control_proposal=read_json(
            artifact_paths["scale_control_proposal"]
        ),
        scale_control_completion=read_json(
            artifact_paths["scale_control_completion"]
        ),
        scale_control_decision=read_json(
            artifact_paths["scale_control_decision"]
        ),
        source_budget_contract=read_json(
            artifact_paths["source_budget_contract"]
        ),
        parent_experiment_id=parent_experiment_id,
        solver_hash=bundle_hash,
        evaluation_hash=bundle_hash,
    )
    proposal.pop("proposal_sha256")
    proposal["source_artifacts"] = {
        name: {
            "path": relative_artifacts[name].as_posix(),
            "path_base": "repository_root",
            "file_sha256": file_sha256(path),
        }
        for name, path in artifact_paths.items()
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)

    budget = BudgetContract.proposed_v8_p25_spectral_recheck()
    frozen = _require_mapping(
        "p25_spectral_recheck_frozen_contracts",
        proposal["frozen_contracts"],
    )
    request: dict[str, Any] = {
        "request_version": "lyx_p25_spectral_recheck_budget_request_v2",
        "status": "awaiting_human_budget_and_execution_decision",
        "approved": False,
        "decision_state": _AUTHORIZATION_STATE,
        "proposal_sha256": proposal["proposal_sha256"],
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "added_unique_identities": 36,
        "normal_unique_identity_limit": 828,
        "max_unique_identities": 840,
        "max_attempts": 1680,
        "retry_limit": 1,
        "budget_contract_hash": budget.sha256,
        "identity_panel_sha256": proposal["identity_panel_sha256"],
        "profile_panel_sha256": proposal["profile_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "metric_contract_hash": frozen["metric_contract_hash"],
        "spectral_gate_contract_hash": frozen[
            "spectral_gate_contract_hash"
        ],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
    }
    request["request_sha256"] = canonical_sha256(request)
    receipt: dict[str, Any] = {
        "receipt_version": "lyx_p25_spectral_recheck_proposal_receipt_v2",
        "status": "awaiting_human_execution_authorization",
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_request_sha256": request["request_sha256"],
        "identity_count": 36,
        "diagnostic_run_count": 0,
        "parameter_search_run_count": 0,
        "independent_bo_run_count": 0,
        "may_execute": False,
    }
    staging = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        staging.mkdir(parents=True)
        atomic_write_json(
            staging / "p25_spectral_recheck_proposal.json",
            proposal,
        )
        atomic_write_json(
            staging / "budget_amendment_request.json",
            request,
        )
        atomic_write_json(
            staging / "budget_contract_v8.json",
            budget.to_dict(),
        )
        atomic_write_json(
            staging / "metric_contract.json",
            dict(_require_mapping("metric_contract", proposal["metric_contract"])),
        )
        atomic_write_json(
            staging / "decision_contract.json",
            dict(
                _require_mapping(
                    "decision_contract",
                    proposal["decision_contract"],
                )
            ),
        )
        spectral_artifact = {
            **StageRSpectralGateContract().to_dict(),
            "contract_sha256": StageRSpectralGateContract().sha256,
        }
        atomic_write_json(
            staging / "spectral_gate_contract.json",
            spectral_artifact,
        )
        atomic_write_json(staging / "source_identity.json", source_identity)
        receipt["artifact_sha256"] = {
            path.name: file_sha256(path)
            for path in staging.iterdir()
            if path.is_file()
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_write_json(staging / "proposal_receipt.json", receipt)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return receipt


def validate_p25_spectral_recheck_authorization(
    proposal: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require approval bound to the exact 36 identities and v8 budget."""

    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="p25_spectral_recheck_proposal",
    )
    if receipt is None or receipt.get("approved") is not True:
        raise P25SpectralRecheckAuthorizationError(
            "p25_spectral_recheck_execution_authorization_required"
        )
    frozen = _require_mapping(
        "p25_spectral_recheck_frozen_contracts",
        proposal.get("frozen_contracts"),
    )
    expected = {
        "decision_state": _AUTHORIZATION_STATE,
        "proposal_sha256": proposal_sha,
        "budget_contract_hash": frozen.get("budget_contract_hash"),
        "unique_budget": 36,
        "stage": _STAGE,
        "identity_panel_sha256": proposal.get("identity_panel_sha256"),
        "profile_panel_sha256": proposal.get("profile_panel_sha256"),
        "record_panel_sha256": proposal.get("record_panel_sha256"),
        "solver_hash": frozen.get("solver_hash"),
        "evaluation_hash": frozen.get("evaluation_hash"),
        "metric_contract_hash": frozen.get("metric_contract_hash"),
        "spectral_gate_contract_hash": frozen.get(
            "spectral_gate_contract_hash"
        ),
        "decision_contract_hash": frozen.get("decision_contract_hash"),
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
    }
    mismatched = sorted(
        name
        for name, value in expected.items()
        if receipt.get(name) != value
    )
    if mismatched:
        raise P25SpectralRecheckAuthorizationError(
            "p25_spectral_recheck_authorization_mismatch:"
            + ",".join(mismatched)
        )
    for name in ("approved_at", "approved_by"):
        if not isinstance(receipt.get(name), str) or not receipt[name]:
            raise P25SpectralRecheckAuthorizationError(
                f"p25_spectral_recheck_authorization_{name}_invalid"
            )
    return dict(receipt)


def _identity_from_item(item: Mapping[str, Any]) -> AttemptIdentity:
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
            str(value) for value in payload["allowed_identity_sha256"]
        ),
    )


def _validate_proposal_preflight(
    *,
    proposal_dir: Path,
    source_root: Path,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    proposal_root = Path(proposal_dir).resolve()
    proposal = read_json(
        proposal_root / "p25_spectral_recheck_proposal.json"
    )
    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="p25_spectral_recheck_proposal",
    )
    receipt = read_json(proposal_root / "proposal_receipt.json")
    _verify_embedded_hash(
        receipt,
        hash_field="receipt_sha256",
        artifact_name="p25_spectral_recheck_proposal_receipt",
    )
    if receipt.get("proposal_sha256") != proposal_sha:
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_proposal_receipt_mismatch"
        )
    artifact_hashes = _require_mapping(
        "p25_spectral_recheck_artifacts",
        receipt.get("artifact_sha256"),
    )
    if set(artifact_hashes) != _PROPOSAL_ARTIFACT_NAMES:
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_artifact_set_mismatch"
        )
    for name, expected_hash in artifact_hashes.items():
        path = proposal_root / str(name)
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise P25SpectralRecheckError(
                f"p25_spectral_recheck_artifact_mismatch:{name}"
            )

    source_root = Path(source_root).resolve()
    repository_root = _repository_root_from_source_root(source_root)
    source_artifacts = _require_mapping(
        "p25_spectral_recheck_source_artifacts",
        proposal.get("source_artifacts"),
    )
    expected_source_names = {
        "prior_p25_proposal",
        "prior_p25_completion",
        "prior_p25_decision",
        "scale_control_proposal",
        "scale_control_completion",
        "scale_control_decision",
        "source_budget_contract",
    }
    if set(source_artifacts) != expected_source_names:
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_source_artifact_set_mismatch"
        )
    resolved_sources: dict[str, Path] = {}
    for name, raw in source_artifacts.items():
        artifact = _require_mapping(f"source_artifact:{name}", raw)
        relative = Path(str(artifact.get("path", "")))
        path = (repository_root / relative).resolve()
        if (
            artifact.get("path_base") != "repository_root"
            or relative.is_absolute()
            or not path.is_relative_to(repository_root)
            or not path.is_file()
            or file_sha256(path) != artifact.get("file_sha256")
        ):
            raise P25SpectralRecheckError(
                f"p25_spectral_recheck_source_artifact_mismatch:{name}"
            )
        resolved_sources[name] = path

    current_source_identity = runtime_source_identity(
        source_root,
        root_modules=("ppg_hr.v2.recovery_p25_spectral_recheck",),
    )
    if read_json(proposal_root / "source_identity.json") != current_source_identity:
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_source_identity_artifact_mismatch"
        )
    bundle_hash = _require_hash(
        "source_bundle_sha256",
        current_source_identity.get("source_bundle_sha256"),
    )
    rebuilt = build_p25_spectral_recheck_proposal(
        prior_p25_proposal=read_json(resolved_sources["prior_p25_proposal"]),
        prior_p25_completion=read_json(
            resolved_sources["prior_p25_completion"]
        ),
        prior_p25_decision=read_json(resolved_sources["prior_p25_decision"]),
        scale_control_proposal=read_json(
            resolved_sources["scale_control_proposal"]
        ),
        scale_control_completion=read_json(
            resolved_sources["scale_control_completion"]
        ),
        scale_control_decision=read_json(
            resolved_sources["scale_control_decision"]
        ),
        source_budget_contract=read_json(
            resolved_sources["source_budget_contract"]
        ),
        parent_experiment_id=str(proposal["parent_experiment_id"]),
        solver_hash=bundle_hash,
        evaluation_hash=bundle_hash,
    )
    rebuilt.pop("proposal_sha256")
    rebuilt["source_artifacts"] = dict(source_artifacts)
    rebuilt["proposal_sha256"] = canonical_sha256(rebuilt)
    if proposal_sha != rebuilt["proposal_sha256"]:
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_proposal_reconstruction_mismatch"
        )

    budget = BudgetContract.proposed_v8_p25_spectral_recheck()
    if read_json(proposal_root / "budget_contract_v8.json") != budget.to_dict():
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_budget_artifact_mismatch"
        )
    metric = _require_mapping("metric_contract", proposal["metric_contract"])
    decision_contract = _require_mapping(
        "decision_contract",
        proposal["decision_contract"],
    )
    if read_json(proposal_root / "metric_contract.json") != metric:
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_metric_contract_artifact_mismatch"
        )
    if (
        read_json(proposal_root / "decision_contract.json")
        != decision_contract
    ):
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_decision_contract_artifact_mismatch"
        )
    spectral_artifact = read_json(
        proposal_root / "spectral_gate_contract.json"
    )
    spectral_artifact_hash = _verify_embedded_hash(
        spectral_artifact,
        hash_field="contract_sha256",
        artifact_name="p25_spectral_recheck_spectral_contract",
    )
    if spectral_artifact_hash != StageRSpectralGateContract().sha256:
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_spectral_contract_artifact_mismatch"
        )
    budget_request = read_json(
        proposal_root / "budget_amendment_request.json"
    )
    _verify_embedded_hash(
        budget_request,
        hash_field="request_sha256",
        artifact_name="p25_spectral_recheck_budget_request",
    )
    expected_budget_fields = {
        "proposal_sha256": proposal_sha,
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "added_unique_identities": 36,
        "normal_unique_identity_limit": 828,
        "max_unique_identities": 840,
        "max_attempts": 1680,
        "retry_limit": 1,
        "budget_contract_hash": budget.sha256,
        "identity_panel_sha256": proposal["identity_panel_sha256"],
        "profile_panel_sha256": proposal["profile_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "metric_contract_hash": metric["contract_sha256"],
        "spectral_gate_contract_hash": StageRSpectralGateContract().sha256,
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
    }
    if any(
        budget_request.get(name) != value
        for name, value in expected_budget_fields.items()
    ):
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_budget_request_mismatch"
        )
    identity_items = tuple(
        dict(_require_mapping("p25_spectral_recheck_identity", raw))
        for raw in _require_list(
            "p25_spectral_recheck_identities",
            proposal.get("identities"),
        )
    )
    identities = tuple(_identity_from_item(item) for item in identity_items)
    if (
        len(identities) != 36
        or len({identity.sha256 for identity in identities}) != 36
        or [identity.sha256 for identity in identities]
        != proposal.get("identity_sha256")
    ):
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_identity_matrix_mismatch"
        )
    return proposal, identity_items


def prepare_p25_spectral_recheck_governance(
    *,
    proposal_dir: Path,
    authorization_receipt_path: Path,
    source_governance_dir: Path,
    governance_dir: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Migrate v7 and register the approved identities without running them."""

    proposal, identity_items = _validate_proposal_preflight(
        proposal_dir=Path(proposal_dir).resolve(),
        source_root=source_root,
    )
    authorization = validate_p25_spectral_recheck_authorization(
        proposal,
        receipt=read_json(Path(authorization_receipt_path).resolve()),
    )
    source_governance = Path(source_governance_dir).resolve()
    target_governance = Path(governance_dir).resolve()
    if target_governance.exists():
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_governance_exists"
        )
    source_budget = BudgetContract.proposed_v7_spectral_metric_control()
    if (
        read_json(source_governance / "budget_contract.json")
        != source_budget.to_dict()
    ):
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_source_budget_mismatch"
        )
    exploration_payload = read_json(
        source_governance / "exploration_registry.json"
    )
    exploration = _exploration_from_payload(exploration_payload)
    if exploration.to_dict() != exploration_payload:
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_exploration_registry_mismatch"
        )
    source_registry = AttemptRegistry.open(
        source_governance / "attempt_registry.json",
        budget_contract=source_budget,
        exploration_registry=exploration,
    )
    target_budget = BudgetContract.proposed_v8_p25_spectral_recheck()
    frozen = _require_mapping(
        "p25_spectral_recheck_frozen_contracts",
        proposal["frozen_contracts"],
    )
    if target_budget.sha256 != frozen.get("budget_contract_hash"):
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_target_budget_mismatch"
        )
    identities = tuple(_identity_from_item(item) for item in identity_items)
    amendment = BudgetAmendmentRequest(
        stage=_STAGE,
        profile_design_rule_hash=str(frozen["metric_contract_hash"]),
        record_manifest_hash=str(proposal["record_panel_sha256"]),
        added_unique_identities=36,
        normal_unique_identity_limit=828,
        max_unique_identities=840,
        max_attempts=1680,
    )
    migration_authorization = {
        "approved": True,
        "decision_state": "awaiting_human_budget_decision",
        **amendment.__dict__,
        "independent_bo_authorized": False,
        "approved_at": authorization["approved_at"],
        "approved_by": authorization["approved_by"],
    }
    governance_receipt: dict[str, Any] = {}

    def finalize(staging: Path, staged: AttemptRegistry) -> None:
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
                "lyx_p25_spectral_recheck_governance_receipt_v2"
            ),
            "status": "prepared_zero_runs",
            "proposal_sha256": proposal["proposal_sha256"],
            "source_budget_contract_hash": source_budget.sha256,
            "target_budget_contract_hash": target_budget.sha256,
            "new_unique_identity_count": 36,
            "attempt_registry_summary": staged.summary(),
            "parameter_search_authorized": False,
            "independent_bo_authorized": False,
            "automatic_stage_r_execution": False,
            "automatic_stage_f_execution": False,
        }
        governance_receipt["receipt_sha256"] = canonical_sha256(
            governance_receipt
        )
        atomic_write_json(
            staging / "governance_receipt.json",
            governance_receipt,
        )

    source_registry.migrate_to(
        target_governance / "attempt_registry.json",
        budget_contract=target_budget,
        amendment_request=amendment,
        authorization_receipt=migration_authorization,
        new_identities=identities,
        target_exploration_registry=exploration,
        finalize_staging=finalize,
    )
    return governance_receipt


def _profile_and_record(
    item: Mapping[str, Any],
) -> tuple[FilterProfile, FilterAuditRecord]:
    config = _require_mapping(
        "p25_spectral_recheck_identity_config",
        item.get("config"),
    )
    source_config = _require_mapping(
        "p25_spectral_recheck_source_config",
        config.get("source_config"),
    )
    parameters = _require_mapping(
        "p25_spectral_recheck_source_parameters",
        source_config.get("parameters"),
    )
    profile = FilterProfile(
        profile_id=str(item["filter_profile_id"]),
        design_role="core",
        fs_target=int(parameters["fs_target"]),
        memory_ms=int(item["physical_memory_ms"]),
        nominal_mu=float(item["nominal_mu"]),
        recovery_sentinel_role=None,
    )
    if (
        profile.sha256 != item.get("filter_profile_sha256")
        or profile.actual_taps != item.get("actual_taps")
    ):
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_profile_reconstruction_mismatch"
        )
    record = FilterAuditRecord(
        record_id=str(item["record_id"]),
        scene=str(item["scene"]),
        data_path=str(item["data_path"]),
        reference_path=str(item["reference_path"]),
        data_sha256=str(item["raw_data_sha256"]),
        reference_sha256=str(item["reference_sha256"]),
    )
    return profile, record


def execute_p25_spectral_recheck(
    *,
    proposal_dir: Path,
    governance_dir: Path,
    output_dir: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Execute only the approved corrected 36-coordinate audit matrix."""

    proposal, identity_items = _validate_proposal_preflight(
        proposal_dir=Path(proposal_dir).resolve(),
        source_root=source_root,
    )
    governance_root = Path(governance_dir).resolve()
    authorization = validate_p25_spectral_recheck_authorization(
        proposal,
        receipt=read_json(
            governance_root / "execution_authorization.json"
        ),
    )
    budget = BudgetContract.proposed_v8_p25_spectral_recheck()
    if read_json(governance_root / "budget_contract.json") != budget.to_dict():
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_execution_budget_mismatch"
        )
    exploration = _exploration_from_payload(
        read_json(governance_root / "exploration_registry.json")
    )
    registry = AttemptRegistry.open(
        governance_root / "attempt_registry.json",
        budget_contract=budget,
        exploration_registry=exploration,
    )
    identities = tuple(_identity_from_item(item) for item in identity_items)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    completion_path = destination / "completion.json"
    if completion_path.exists():
        registry.assert_complete_matrix(identities)
        return _validate_completed_p25_spectral_recheck(
            completion_path=completion_path,
            proposal=proposal,
            authorization=authorization,
            output_dir=destination,
            identities=identities,
            registry=registry,
        )

    registry.register_identities(identities)
    rows: list[dict[str, Any]] = []
    result_entries: list[dict[str, Any]] = []
    results_dir = destination / "profile_record_audits"
    contract = StageRSpectralGateContract()
    for item, identity in zip(identity_items, identities, strict=True):
        profile_id = str(item["filter_profile_id"])
        result_path = (
            results_dir / profile_id / f"{identity.record_id}.json"
        )

        def run(
            *,
            _item: Mapping[str, Any] = item,
            _identity: AttemptIdentity = identity,
            _result_path: Path = result_path,
        ) -> dict[str, Any]:
            profile, record = _profile_and_record(_item)
            audit = audit_stage_r_profile_record(
                profile,
                record,
                contract=contract,
            )
            payload = {
                "result_version": "lyx_p25_spectral_recheck_result_v2",
                "proposal_sha256": proposal["proposal_sha256"],
                "identity_sha256": _identity.sha256,
                "filter_profile_id": profile.profile_id,
                "filter_profile_sha256": profile.sha256,
                "record_id": record.record_id,
                "scene": record.scene,
                "spectral_gate_contract_sha256": contract.sha256,
                "spectral_audit": audit,
            }
            payload["result_sha256"] = canonical_sha256(payload)
            atomic_write_json(_result_path, payload)
            return payload

        try:
            registry.assert_complete_matrix((identity,))
        except GovernanceError as error:
            if str(error).startswith("matrix_identity_still_running:"):
                registry.reconcile_interrupted_attempt(identity, evidence=None)
            row = registry.execute_registered(identity, run)
        else:
            row = _validate_p25_spectral_recheck_result(
                path=result_path,
                proposal_sha256=str(proposal["proposal_sha256"]),
                identity=identity,
                profile_id=profile_id,
            )
        rows.append(row)
        result_entries.append(
            {
                "filter_profile_id": profile_id,
                "record_id": identity.record_id,
                "identity_sha256": identity.sha256,
                "path": result_path.relative_to(destination).as_posix(),
                "file_sha256": file_sha256(result_path),
                "result_sha256": row["result_sha256"],
            }
        )
    registry.assert_complete_matrix(identities)
    decision = evaluate_p25_spectral_recheck_decision(rows)
    decision["proposal_sha256"] = proposal["proposal_sha256"]
    decision.pop("decision_sha256")
    decision["decision_sha256"] = canonical_sha256(decision)
    atomic_write_json(destination / "decision_receipt.json", decision)
    manifest = {
        "manifest_version": "lyx_p25_spectral_recheck_manifest_v2",
        "proposal_sha256": proposal["proposal_sha256"],
        "result_count": 36,
        "results": result_entries,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write_json(destination / "result_manifest.json", manifest)
    matrix_summary = registry.matrix_execution_summary(identities)
    completion = {
        "completion_version": "lyx_p25_spectral_recheck_completion_v2",
        "status": decision["decision"],
        "next_state": decision["next_state"],
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": canonical_sha256(authorization),
        "diagnostic_result_count": 36,
        "diagnostic_run_count": matrix_summary[
            "identity_with_solver_attempt_count"
        ],
        "parameter_search_run_count": 0,
        "independent_bo_run_count": 0,
        "may_nominate_recovery_candidate": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "matrix_execution_summary": matrix_summary,
        "decision_sha256": decision["decision_sha256"],
        "artifacts": {
            "decision_receipt.json": file_sha256(
                destination / "decision_receipt.json"
            ),
            "result_manifest.json": file_sha256(
                destination / "result_manifest.json"
            ),
        },
    }
    completion["completion_sha256"] = canonical_sha256(completion)
    atomic_write_json(completion_path, completion)
    return _validate_completed_p25_spectral_recheck(
        completion_path=completion_path,
        proposal=proposal,
        authorization=authorization,
        output_dir=destination,
        identities=identities,
        registry=registry,
    )


def _validate_p25_spectral_recheck_result(
    *,
    path: Path,
    proposal_sha256: str,
    identity: AttemptIdentity,
    profile_id: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise P25SpectralRecheckError(
            f"p25_spectral_recheck_result_missing:{profile_id}:{identity.record_id}"
        )
    payload = read_json(path)
    _verify_embedded_hash(
        payload,
        hash_field="result_sha256",
        artifact_name="p25_spectral_recheck_result",
    )
    audit = _require_mapping(
        "p25_spectral_recheck_result_audit",
        payload.get("spectral_audit"),
    )
    if (
        payload.get("proposal_sha256") != proposal_sha256
        or payload.get("identity_sha256") != identity.sha256
        or payload.get("record_id") != identity.record_id
        or payload.get("filter_profile_id") != profile_id
        or payload.get("spectral_gate_contract_sha256")
        != StageRSpectralGateContract().sha256
        or audit.get("stage_r_spectral_gate_contract_sha256")
        != StageRSpectralGateContract().sha256
    ):
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_result_identity_mismatch:"
            f"{profile_id}:{identity.record_id}"
        )
    return payload


def _validate_completed_p25_spectral_recheck(
    *,
    completion_path: Path,
    proposal: Mapping[str, Any],
    authorization: Mapping[str, Any],
    output_dir: Path,
    identities: Sequence[AttemptIdentity],
    registry: AttemptRegistry,
) -> dict[str, Any]:
    completion = read_json(completion_path)
    _verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="p25_spectral_recheck_completion",
    )
    matrix_summary = registry.matrix_execution_summary(identities)
    if (
        completion.get("proposal_sha256") != proposal.get("proposal_sha256")
        or completion.get("authorization_sha256")
        != canonical_sha256(authorization)
        or completion.get("diagnostic_result_count") != 36
        or completion.get("diagnostic_run_count")
        != matrix_summary["identity_with_solver_attempt_count"]
        or completion.get("parameter_search_run_count") != 0
        or completion.get("independent_bo_run_count") != 0
        or completion.get("may_nominate_recovery_candidate") is not False
        or completion.get("automatic_stage_r_execution") is not False
        or completion.get("automatic_stage_f_execution") is not False
        or completion.get("matrix_execution_summary") != matrix_summary
    ):
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_completion_identity_mismatch"
        )
    artifacts = _require_mapping(
        "p25_spectral_recheck_completion_artifacts",
        completion.get("artifacts"),
    )
    for name in ("decision_receipt.json", "result_manifest.json"):
        path = output_dir / name
        if not path.is_file() or file_sha256(path) != artifacts.get(name):
            raise P25SpectralRecheckError(
                f"p25_spectral_recheck_completion_artifact_mismatch:{name}"
            )
    decision = read_json(output_dir / "decision_receipt.json")
    decision_sha = _verify_embedded_hash(
        decision,
        hash_field="decision_sha256",
        artifact_name="p25_spectral_recheck_decision",
    )
    if (
        decision_sha != completion.get("decision_sha256")
        or decision.get("proposal_sha256") != proposal.get("proposal_sha256")
        or decision.get("decision") != completion.get("status")
        or decision.get("next_state") != completion.get("next_state")
    ):
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_completion_decision_mismatch"
        )
    manifest = read_json(output_dir / "result_manifest.json")
    _verify_embedded_hash(
        manifest,
        hash_field="manifest_sha256",
        artifact_name="p25_spectral_recheck_manifest",
    )
    entries = _require_list(
        "p25_spectral_recheck_manifest_results",
        manifest.get("results"),
    )
    identities_by_hash = {
        identity.sha256: identity for identity in identities
    }
    if (
        manifest.get("proposal_sha256") != proposal.get("proposal_sha256")
        or manifest.get("result_count") != 36
        or len(entries) != 36
    ):
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_manifest_mismatch"
        )
    observed: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw in entries:
        entry = _require_mapping(
            "p25_spectral_recheck_manifest_entry",
            raw,
        )
        identity_hash = str(entry.get("identity_sha256", ""))
        identity = identities_by_hash.get(identity_hash)
        profile_id = str(entry.get("filter_profile_id", ""))
        relative = Path(str(entry.get("path", "")))
        path = (output_dir / relative).resolve()
        expected = (
            Path("profile_record_audits")
            / profile_id
            / f"{entry.get('record_id')}.json"
        )
        if (
            identity is None
            or identity_hash in observed
            or entry.get("record_id") != identity.record_id
            or relative.as_posix() != expected.as_posix()
            or not path.is_relative_to(output_dir)
            or not path.is_file()
            or file_sha256(path) != entry.get("file_sha256")
        ):
            raise P25SpectralRecheckError(
                "p25_spectral_recheck_manifest_result_mismatch"
            )
        result = _validate_p25_spectral_recheck_result(
            path=path,
            proposal_sha256=str(proposal["proposal_sha256"]),
            identity=identity,
            profile_id=profile_id,
        )
        if result.get("result_sha256") != entry.get("result_sha256"):
            raise P25SpectralRecheckError(
                "p25_spectral_recheck_manifest_result_mismatch"
            )
        rows.append(result)
        observed.add(identity_hash)
    if observed != set(identities_by_hash):
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_manifest_identity_set_mismatch"
        )
    expected_decision = evaluate_p25_spectral_recheck_decision(rows)
    expected_decision["proposal_sha256"] = proposal["proposal_sha256"]
    expected_decision.pop("decision_sha256")
    expected_decision["decision_sha256"] = canonical_sha256(
        expected_decision
    )
    if decision != expected_decision:
        raise P25SpectralRecheckError(
            "p25_spectral_recheck_decision_recomputation_mismatch"
        )
    return completion
