"""Bounded, paired decomposition of the frozen p25-short-low LMS mechanism."""

from __future__ import annotations

import math
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ppg_hr.core.lms_filter import lms_filter

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
    StageRPreparedWindow,
    StageRSpectralGateContract,
    evaluate_stage_r_spectral_gate_windows,
    prepare_stage_r_record_windows,
)


class FilterMechanismDecompositionError(RuntimeError):
    """The filter-mechanism package violates its frozen contract."""


class FilterMechanismDecompositionAuthorizationError(
    FilterMechanismDecompositionError
):
    """The exact mechanism package has not been approved."""


_STAGE = "filter_mechanism_decomposition_diagnostic"
_ATTEMPT_KIND = "diagnostic"
_AUTHORIZATION_STATE = (
    "awaiting_human_filter_mechanism_decomposition_decision"
)
_P25_PROPOSAL_SHA256 = (
    "83081d8081def70b356b2737864b552c3fc2376817bbc595db32f4f650616c26"
)
_P25_COMPLETION_SHA256 = (
    "f66ac4bde326ce3b2cc25738176ea5613cec9c9b9ee91862557ddf44d9009076"
)
_P25_DECISION_SHA256 = (
    "fd05114792fd04b276a114a6ac9e9292089da544883d1b98a6f7e91832fd5684"
)
_P25_MANIFEST_SHA256 = (
    "4a11dde3a3cc70804cc0f40cec747cd6ff0cc40bd20a5217050e12e5be232850"
)
_P25_MANIFEST_FILE_SHA256 = (
    "5acc4555ae2b5f36b4006b892ee49b13e537b69dffcf58a7bd6cc5c522ef6aeb"
)
_P25_SOURCE_FILE_SHA256 = {
    "p25_proposal": (
        "8db3770656b90b2e3ad2ae7b0bacef5e3ec8813730d7be6d812eaa48dbd2f9f9"
    ),
    "p25_completion": (
        "5c2f6c6904530b1a4658befcc8df22a4f8d2d6fafe25f12b26ce4db7515f7eb1"
    ),
    "p25_decision": (
        "f53448a1e1430f803130d1f6c00d4455e20b45357cf3d284255e7daab28595bb"
    ),
    "p25_manifest": _P25_MANIFEST_FILE_SHA256,
    "source_budget_contract": (
        "c5114f8497821f8d090133a5b55a12ecc0cf0baa4ffa294efeaff24691e8b0ed"
    ),
    "spectral_gate_contract": (
        "bf2ee4deca720850b9030293b36c3db7b87bec621d9f4827388166da18460c19"
    ),
}
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
_CONTROL_PROFILE = FilterProfile(
    profile_id="p25-short-low",
    design_role="core",
    fs_target=25,
    memory_ms=40,
    nominal_mu=0.008,
)
CONTROL_LANES = (
    "raw_bypass",
    "two_stage_zero_update",
)
CANDIDATE_LANES = (
    "rank1_only_adaptive",
    "rank2_only_adaptive",
    "reverse_cascade_adaptive",
)
_FORWARD_LANE = "ranked_cascade_adaptive"
_ALL_LANES = (
    *CONTROL_LANES,
    "rank1_only_adaptive",
    "rank2_only_adaptive",
    _FORWARD_LANE,
    "reverse_cascade_adaptive",
)
_GATE_NAMES = (
    "prominence_db_delta_pass",
    "visible_top3_rate_delta_pass",
    "hr_band_share_delta_pass",
    "pulse_power_retention_pass",
    "residual_artifact_corr_delta_pass",
    "complete_window_evidence_pass",
)
_GATE_SUMMARY_FIELDS = (
    "spectral_gate_pass",
    "valid_window_count",
    "invalid_window_count",
    "prominence_db_delta_median",
    "visible_top3_rate_delta",
    "hr_band_share_delta_median",
    "pulse_power_retention_median",
    "residual_artifact_corr_delta_median",
    "gates",
    "failure_reasons",
)
_PROPOSAL_ARTIFACT_NAMES = {
    "budget_amendment_request.json",
    "budget_contract_v9.json",
    "filter_mechanism_decomposition_proposal.json",
    "mechanism_contract.json",
    "source_identity.json",
    "spectral_gate_contract.json",
}


@dataclass(frozen=True)
class FilterMechanismDecompositionContract:
    """Pre-data, six-lane mechanism decomposition for one frozen profile."""

    expected_record_count: int = 12
    expected_reference_count: int = 2
    profile_id: str = "p25-short-low"
    fs_target: int = 25
    memory_ms: int = 40
    actual_taps: int = 1
    nominal_mu: float = 0.008
    lms_mu_min: float = 1e-6
    require_complete_windows: bool = True
    contract_version: str = "lyx_filter_mechanism_decomposition_v1"

    def __post_init__(self) -> None:
        if (
            self.expected_record_count != 12
            or self.expected_reference_count != 2
            or self.profile_id != _CONTROL_PROFILE.profile_id
            or self.fs_target != _CONTROL_PROFILE.fs_target
            or self.memory_ms != _CONTROL_PROFILE.memory_ms
            or self.actual_taps != _CONTROL_PROFILE.actual_taps
            or self.nominal_mu != float(_CONTROL_PROFILE.nominal_mu)
            or self.lms_mu_min != 1e-6
        ):
            raise ValueError(
                "invalid_filter_mechanism_decomposition_contract"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lanes"] = {
            "raw_bypass": {
                "reference_rank_sequence": [],
                "adaptive_updates": 0,
                "mu_rule": "none",
            },
            "two_stage_zero_update": {
                "reference_rank_sequence": [1, 2],
                "adaptive_updates": 0,
                "mu_rule": "forced_zero",
            },
            "rank1_only_adaptive": {
                "reference_rank_sequence": [1],
                "adaptive_updates": 1,
                "mu_rule": "current_effective_mu",
            },
            "rank2_only_adaptive": {
                "reference_rank_sequence": [2],
                "adaptive_updates": 1,
                "mu_rule": "current_effective_mu",
            },
            "ranked_cascade_adaptive": {
                "reference_rank_sequence": [1, 2],
                "adaptive_updates": 2,
                "mu_rule": "current_effective_mu",
                "role": "committed_baseline_reproduction",
            },
            "reverse_cascade_adaptive": {
                "reference_rank_sequence": [2, 1],
                "adaptive_updates": 2,
                "mu_rule": "current_effective_mu",
            },
        }
        payload["effective_mu_formula"] = (
            "max(lms_mu_min, nominal_mu - "
            "abs(corrcoef(current_desired, reference)) / 100)"
        )
        payload["decision_precedence"] = [
            "control_invalid",
            "baseline_reproduction_invalid",
            "rank1_single_stage_mechanism_candidate",
            "rank2_reference_selection_mechanism_candidate",
            "reverse_order_mechanism_candidate",
            "partial_mechanism_relief_requires_factorial",
            "no_mechanism_relief_requires_factorial_or_bo_review",
        ]
        payload["identity_grain"] = "record"
        payload["paired_within_record"] = True
        payload["no_parameter_search"] = True
        payload["independent_bo_authorized"] = False
        payload["may_nominate_recovery_candidate"] = False
        payload["automatic_stage_r_execution"] = False
        payload["automatic_stage_f_execution"] = False
        return payload

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FilterMechanismDecompositionError(f"{name}_must_be_object")
    return value


def _require_list(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise FilterMechanismDecompositionError(f"{name}_must_be_array")
    return value


def _require_hash(name: str, value: object) -> str:
    text = str(value)
    try:
        require_sha256(name, text)
    except ValueError as error:
        raise FilterMechanismDecompositionError(str(error)) from error
    return text


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    artifact_name: str,
) -> str:
    declared = _require_hash(hash_field, payload.get(hash_field))
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    if canonical_sha256(unsigned) != declared:
        raise FilterMechanismDecompositionError(
            f"{artifact_name}_hash_mismatch"
        )
    return declared


def _repository_root_from_source_root(source_root: Path) -> Path:
    resolved = Path(source_root).resolve()
    if (
        resolved.name != "src"
        or resolved.parent.name != "python"
        or not (resolved / "ppg_hr").is_dir()
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_source_root_mismatch"
        )
    return resolved.parent.parent


def _profile_payload() -> dict[str, Any]:
    return {
        **asdict(_CONTROL_PROFILE),
        "actual_taps": _CONTROL_PROFILE.actual_taps,
        "profile_sha256": _CONTROL_PROFILE.sha256,
    }


def _identity_config(
    contract: FilterMechanismDecompositionContract,
    spectral: StageRSpectralGateContract,
) -> dict[str, Any]:
    return {
        "execution_mode": "prepared_window_filter_mechanism_decomposition",
        "profile_id": contract.profile_id,
        "profile_sha256": _CONTROL_PROFILE.sha256,
        "fs_target": contract.fs_target,
        "memory_ms": contract.memory_ms,
        "actual_taps": contract.actual_taps,
        "nominal_mu": contract.nominal_mu,
        "lms_mu_min": contract.lms_mu_min,
        "lanes": list(_ALL_LANES),
        "mechanism_contract_sha256": contract.sha256,
        "spectral_gate_contract_sha256": spectral.sha256,
        "parameter_search": False,
    }


def _gate_summary(gate: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        name: gate.get(name)
        for name in _GATE_SUMMARY_FIELDS
        if name in gate
    }
    gates = _require_mapping(
        "filter_mechanism_decomposition_gate_values",
        summary.get("gates"),
    )
    if set(gates) != set(_GATE_NAMES) or any(
        not isinstance(value, bool) for value in gates.values()
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_gate_set_mismatch"
        )
    return summary


def _validate_upstream(
    *,
    p25_proposal: Mapping[str, Any],
    p25_completion: Mapping[str, Any],
    p25_decision: Mapping[str, Any],
    p25_manifest: Mapping[str, Any],
    anchor_results: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    proposal_sha = _verify_embedded_hash(
        p25_proposal,
        hash_field="proposal_sha256",
        artifact_name="filter_mechanism_p25_proposal",
    )
    completion_sha = _verify_embedded_hash(
        p25_completion,
        hash_field="completion_sha256",
        artifact_name="filter_mechanism_p25_completion",
    )
    decision_sha = _verify_embedded_hash(
        p25_decision,
        hash_field="decision_sha256",
        artifact_name="filter_mechanism_p25_decision",
    )
    manifest_sha = _verify_embedded_hash(
        p25_manifest,
        hash_field="manifest_sha256",
        artifact_name="filter_mechanism_p25_manifest",
    )
    if (
        proposal_sha != _P25_PROPOSAL_SHA256
        or completion_sha != _P25_COMPLETION_SHA256
        or decision_sha != _P25_DECISION_SHA256
        or manifest_sha != _P25_MANIFEST_SHA256
        or p25_completion.get("proposal_sha256") != proposal_sha
        or p25_completion.get("decision_sha256") != decision_sha
        or p25_completion.get("status") != "p25_failure_review_required"
        or p25_completion.get("next_state")
        != "awaiting_human_filter_mechanism_or_independent_bo_review"
        or p25_completion.get("diagnostic_result_count") != 36
        or p25_completion.get("parameter_search_run_count") != 0
        or p25_completion.get("independent_bo_run_count") != 0
        or _require_mapping(
            "filter_mechanism_p25_completion_artifacts",
            p25_completion.get("artifacts"),
        ).get("result_manifest.json")
        != _P25_MANIFEST_FILE_SHA256
        or p25_decision.get("proposal_sha256") != proposal_sha
        or p25_decision.get("decision") != "p25_failure_review_required"
        or p25_manifest.get("proposal_sha256") != proposal_sha
        or p25_manifest.get("result_count") != 36
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_upstream_mismatch"
        )

    templates = tuple(
        dict(_require_mapping("filter_mechanism_p25_identity", item))
        for item in _require_list(
            "filter_mechanism_p25_identities",
            p25_proposal.get("identities"),
        )
        if isinstance(item, Mapping)
        and item.get("filter_profile_id") == _CONTROL_PROFILE.profile_id
    )
    if len(templates) != 12:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_template_count_mismatch"
        )
    templates = tuple(
        sorted(templates, key=lambda item: str(item["record_id"]))
    )
    if (
        {str(item["record_id"]) for item in templates}
        != _EXPECTED_RECORD_IDS
        or dict(Counter(str(item["scene"]) for item in templates))
        != _EXPECTED_SCENE_COUNTS
        or any(
            item.get("filter_profile_sha256") != _CONTROL_PROFILE.sha256
            for item in templates
        )
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_record_panel_mismatch"
        )

    entries = {
        str(item["record_id"]): dict(
            _require_mapping("filter_mechanism_p25_manifest_entry", item)
        )
        for item in _require_list(
            "filter_mechanism_p25_manifest_results",
            p25_manifest.get("results"),
        )
        if isinstance(item, Mapping)
        and item.get("filter_profile_id") == _CONTROL_PROFILE.profile_id
    }
    if set(entries) != _EXPECTED_RECORD_IDS or set(anchor_results) != set(
        entries
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_anchor_panel_mismatch"
        )
    anchor_panel: list[dict[str, Any]] = []
    for template in templates:
        record_id = str(template["record_id"])
        entry = entries[record_id]
        result = _require_mapping(
            "filter_mechanism_anchor_result",
            anchor_results[record_id],
        )
        result_sha = _verify_embedded_hash(
            result,
            hash_field="result_sha256",
            artifact_name="filter_mechanism_anchor_result",
        )
        audit = _require_mapping(
            "filter_mechanism_anchor_audit",
            result.get("spectral_audit"),
        )
        gate = _require_mapping(
            "filter_mechanism_anchor_gate",
            audit.get("stage_r_spectral_gate"),
        )
        summary = _gate_summary(gate)
        if (
            result_sha != entry.get("result_sha256")
            or result.get("proposal_sha256") != proposal_sha
            or result.get("record_id") != record_id
            or result.get("filter_profile_id") != _CONTROL_PROFILE.profile_id
            or result.get("filter_profile_sha256")
            != _CONTROL_PROFILE.sha256
        ):
            raise FilterMechanismDecompositionError(
                "filter_mechanism_decomposition_anchor_result_mismatch"
            )
        anchor_panel.append(
            {
                "record_id": record_id,
                "scene": template["scene"],
                "source_identity_sha256": entry["identity_sha256"],
                "source_result_sha256": result_sha,
                "source_file_sha256": entry["file_sha256"],
                "source_path": entry["path"],
                "spectral_gate_summary": summary,
                "spectral_gate_summary_sha256": canonical_sha256(summary),
            }
        )
    return templates, tuple(anchor_panel)


def _identity_item(
    *,
    template: Mapping[str, Any],
    anchor: Mapping[str, Any],
    parent_experiment_id: str,
    solver_hash: str,
    evaluation_hash: str,
    contract: FilterMechanismDecompositionContract,
    spectral: StageRSpectralGateContract,
) -> dict[str, Any]:
    config = _identity_config(contract, spectral)
    identity = AttemptIdentity(
        solver_hash=solver_hash,
        config_hash=canonical_sha256(config),
        metric_contract_hash=contract.sha256,
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
        "config": config,
        "source_p25_identity_sha256": template["identity_sha256"],
        "anchor_source_result_sha256": anchor["source_result_sha256"],
        "anchor_spectral_gate_summary": anchor["spectral_gate_summary"],
        "anchor_spectral_gate_summary_sha256": anchor[
            "spectral_gate_summary_sha256"
        ],
    }


def build_filter_mechanism_decomposition_proposal(
    *,
    p25_proposal: Mapping[str, Any],
    p25_completion: Mapping[str, Any],
    p25_decision: Mapping[str, Any],
    p25_manifest: Mapping[str, Any],
    anchor_results: Mapping[str, Mapping[str, Any]],
    parent_experiment_id: str,
    solver_hash: str,
    evaluation_hash: str,
    source_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build exactly 12 zero-run identities with six paired lanes each."""

    if not parent_experiment_id:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_parent_experiment_id_must_not_be_empty"
        )
    templates, anchors = _validate_upstream(
        p25_proposal=p25_proposal,
        p25_completion=p25_completion,
        p25_decision=p25_decision,
        p25_manifest=p25_manifest,
        anchor_results=anchor_results,
    )
    solver_hash = _require_hash("solver_hash", solver_hash)
    evaluation_hash = _require_hash("evaluation_hash", evaluation_hash)
    contract = FilterMechanismDecompositionContract()
    spectral = StageRSpectralGateContract()
    budget = BudgetContract.proposed_v9_filter_mechanism_decomposition()
    anchors_by_record = {
        str(anchor["record_id"]): anchor for anchor in anchors
    }
    identities = [
        _identity_item(
            template=template,
            anchor=anchors_by_record[str(template["record_id"])],
            parent_experiment_id=parent_experiment_id,
            solver_hash=solver_hash,
            evaluation_hash=evaluation_hash,
            contract=contract,
            spectral=spectral,
        )
        for template in templates
    ]
    identity_hashes = [str(item["identity_sha256"]) for item in identities]
    if len(set(identity_hashes)) != 12:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_identity_collision"
        )
    record_panel = [
        {
            "record_id": item["record_id"],
            "scene": item["scene"],
            "raw_data_sha256": item["raw_data_sha256"],
            "reference_sha256": item["reference_sha256"],
            "data_sha256": item["data_sha256"],
        }
        for item in identities
    ]
    proposal: dict[str, Any] = {
        "proposal_version": "lyx_filter_mechanism_decomposition_proposal_v1",
        "status": "awaiting_human_execution_authorization",
        "authorization_state": _AUTHORIZATION_STATE,
        "parent_experiment_id": parent_experiment_id,
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "unique_budget": 12,
        "retry_limit": 1,
        "worst_case_attempt_budget": 24,
        "deterministic_lane_count_per_identity": 6,
        "diagnostic_run_count": 0,
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "may_nominate_recovery_candidate": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "upstream_p25_recheck": {
            "proposal_sha256": _P25_PROPOSAL_SHA256,
            "completion_sha256": _P25_COMPLETION_SHA256,
            "decision_sha256": _P25_DECISION_SHA256,
            "manifest_sha256": _P25_MANIFEST_SHA256,
            "manifest_file_sha256": _P25_MANIFEST_FILE_SHA256,
            "status": "p25_failure_review_required",
        },
        "frozen_contracts": {
            "solver_hash": solver_hash,
            "evaluation_hash": evaluation_hash,
            "mechanism_contract_hash": contract.sha256,
            "spectral_gate_contract_hash": spectral.sha256,
            "budget_contract_hash": budget.sha256,
            "control_profile_hash": _CONTROL_PROFILE.sha256,
        },
        "mechanism_contract": contract.to_dict(),
        "control_profile": _profile_payload(),
        "decision_inputs": {
            "allowed": [
                "per_record_lane_spectral_gates",
                "per_record_anchor_reproduction",
                "complete_window_evidence",
            ],
            "forbidden": [
                "mae",
                "l10",
                "l20",
                "recovery",
                "penalty",
            ],
        },
        "record_panel": record_panel,
        "record_panel_sha256": canonical_sha256(record_panel),
        "anchor_panel": list(anchors),
        "anchor_panel_sha256": canonical_sha256(list(anchors)),
        "identity_sha256": identity_hashes,
        "identity_panel_sha256": canonical_sha256(identity_hashes),
        "identities": identities,
    }
    if source_artifacts is not None:
        proposal["source_artifacts"] = {
            str(name): dict(value)
            for name, value in sorted(source_artifacts.items())
        }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def _spectral_window(
    prepared: StageRPreparedWindow,
    *,
    after: np.ndarray,
) -> dict[str, Any]:
    return {
        "before": prepared.original,
        "after": after,
        "motion_reference": prepared.primary_reference,
        "fs": prepared.fs,
        "reference_hr_bpm": prepared.reference_hr_bpm,
        "window_center_s": prepared.window_center_s,
    }


def _apply_reference_sequence(
    prepared: StageRPreparedWindow,
    *,
    ranks: Sequence[int],
    contract: FilterMechanismDecompositionContract,
    forced_mu: float | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if len(prepared.ranked_references) != contract.expected_reference_count:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_reference_count_mismatch"
        )
    current = np.asarray(prepared.original, dtype=float).copy()
    trace: list[dict[str, Any]] = []
    for rank in ranks:
        channel, reference, archive_corr = prepared.ranked_references[
            int(rank) - 1
        ]
        reference_arr = np.asarray(reference, dtype=float)
        live_corr = float(abs(np.corrcoef(current, reference_arr)[0, 1]))
        if not math.isfinite(live_corr):
            live_corr = 0.0
        effective_mu = (
            float(forced_mu)
            if forced_mu is not None
            else max(
                contract.lms_mu_min,
                contract.nominal_mu - live_corr / 100.0,
            )
        )
        current, weights, _unused = lms_filter(
            effective_mu,
            prepared.order,
            0,
            reference_arr,
            current,
        )
        trace.append(
            {
                "reference_rank": int(rank),
                "channel": channel,
                "archive_style_abs_corr": float(archive_corr),
                "live_abs_corr": live_corr,
                "effective_mu": effective_mu,
                "weight_norm": float(np.linalg.norm(weights)),
                "nonfinite_count": int(
                    current.size
                    + weights.size
                    - np.count_nonzero(np.isfinite(current))
                    - np.count_nonzero(np.isfinite(weights))
                ),
            }
        )
    return current, trace


def evaluate_filter_mechanism_lanes(
    prepared_windows: Sequence[StageRPreparedWindow | None],
    *,
    spectral_contract: StageRSpectralGateContract,
    mechanism_contract: FilterMechanismDecompositionContract,
) -> dict[str, Any]:
    """Evaluate all six deterministic lanes on one prepared record."""

    lane_windows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in _ALL_LANES
    }
    lane_traces: dict[str, list[dict[str, Any]]] = {
        name: [] for name in _ALL_LANES
    }
    sequences = {
        "rank1_only_adaptive": (1,),
        "rank2_only_adaptive": (2,),
        "ranked_cascade_adaptive": (1, 2),
        "reverse_cascade_adaptive": (2, 1),
    }
    zero_update_weight_max_abs = 0.0
    for prepared in prepared_windows:
        if prepared is None:
            for name in _ALL_LANES:
                lane_windows[name].append({})
            continue
        original = np.asarray(prepared.original, dtype=float)
        lane_windows["raw_bypass"].append(
            _spectral_window(prepared, after=original.copy())
        )
        lane_traces["raw_bypass"].append(
            {
                "window_center_s": prepared.window_center_s,
                "stages": [],
            }
        )
        zero_after, zero_trace = _apply_reference_sequence(
            prepared,
            ranks=(1, 2),
            contract=mechanism_contract,
            forced_mu=0.0,
        )
        zero_update_weight_max_abs = max(
            zero_update_weight_max_abs,
            *(
                float(stage["weight_norm"])
                for stage in zero_trace
            ),
        )
        lane_windows["two_stage_zero_update"].append(
            _spectral_window(prepared, after=zero_after)
        )
        lane_traces["two_stage_zero_update"].append(
            {
                "window_center_s": prepared.window_center_s,
                "stages": zero_trace,
            }
        )
        for lane, ranks in sequences.items():
            after, trace = _apply_reference_sequence(
                prepared,
                ranks=ranks,
                contract=mechanism_contract,
            )
            lane_windows[lane].append(
                _spectral_window(prepared, after=after)
            )
            lane_traces[lane].append(
                {
                    "window_center_s": prepared.window_center_s,
                    "stages": trace,
                }
            )

    lanes = {
        name: evaluate_stage_r_spectral_gate_windows(
            windows,
            contract=spectral_contract,
        )
        for name, windows in lane_windows.items()
    }
    controls_complete = all(
        lanes[name]["invalid_window_count"] == 0
        and lanes[name]["valid_window_count"]
        >= spectral_contract.minimum_valid_window_count
        for name in CONTROL_LANES
    )
    control_valid = bool(
        controls_complete
        and zero_update_weight_max_abs == 0.0
        and all(lanes[name]["spectral_gate_pass"] for name in CONTROL_LANES)
    )
    return {
        "mechanism_contract_sha256": mechanism_contract.sha256,
        "spectral_gate_contract_sha256": spectral_contract.sha256,
        "prepared_window_count": len(prepared_windows),
        "zero_update_weight_max_abs": zero_update_weight_max_abs,
        "control_valid": control_valid,
        "lanes": lanes,
        "lane_traces": lane_traces,
    }


def audit_filter_mechanism_record(
    record: FilterAuditRecord,
    *,
    anchor_spectral_gate_summary: Mapping[str, Any],
    spectral_contract: StageRSpectralGateContract | None = None,
    mechanism_contract: FilterMechanismDecompositionContract | None = None,
) -> dict[str, Any]:
    """Run the frozen six-lane audit once for one LYX development record."""

    spectral = spectral_contract or StageRSpectralGateContract()
    mechanism = mechanism_contract or FilterMechanismDecompositionContract()
    prepared = prepare_stage_r_record_windows(_CONTROL_PROFILE, record)
    result = evaluate_filter_mechanism_lanes(
        prepared,
        spectral_contract=spectral,
        mechanism_contract=mechanism,
    )
    forward_summary = _gate_summary(
        _require_mapping(
            "filter_mechanism_forward_gate",
            _require_mapping(
                "filter_mechanism_lanes",
                result["lanes"],
            )[_FORWARD_LANE],
        )
    )
    anchor_summary = _gate_summary(anchor_spectral_gate_summary)
    return {
        "record_id": record.record_id,
        "scene": record.scene,
        "profile_id": _CONTROL_PROFILE.profile_id,
        "profile_sha256": _CONTROL_PROFILE.sha256,
        "anchor_spectral_gate_summary_sha256": canonical_sha256(
            anchor_summary
        ),
        "forward_spectral_gate_summary_sha256": canonical_sha256(
            forward_summary
        ),
        "anchor_reproduction_pass": forward_summary == anchor_summary,
        **result,
    }


def _lane_complete(gate: Mapping[str, Any]) -> bool:
    gates = _require_mapping(
        "filter_mechanism_decision_gates",
        gate.get("gates"),
    )
    if set(gates) != set(_GATE_NAMES) or any(
        not isinstance(value, bool) for value in gates.values()
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_gate_set_mismatch"
        )
    return bool(gate.get("spectral_gate_pass") is True and all(gates.values()))


def _strictly_dominates(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_lane: str,
) -> bool:
    improved = False
    for row in rows:
        lanes = _require_mapping(
            "filter_mechanism_decision_lanes",
            row.get("lanes"),
        )
        baseline = _require_mapping(
            "filter_mechanism_forward_lane",
            lanes.get(_FORWARD_LANE),
        )
        candidate = _require_mapping(
            "filter_mechanism_candidate_lane",
            lanes.get(candidate_lane),
        )
        baseline_gates = _require_mapping(
            "filter_mechanism_forward_gates",
            baseline.get("gates"),
        )
        candidate_gates = _require_mapping(
            "filter_mechanism_candidate_gates",
            candidate.get("gates"),
        )
        if set(baseline_gates) != set(_GATE_NAMES) or set(
            candidate_gates
        ) != set(_GATE_NAMES):
            raise FilterMechanismDecompositionError(
                "filter_mechanism_decomposition_gate_set_mismatch"
            )
        for name in _GATE_NAMES:
            before = baseline_gates[name]
            after = candidate_gates[name]
            if not isinstance(before, bool) or not isinstance(after, bool):
                raise FilterMechanismDecompositionError(
                    "filter_mechanism_decomposition_gate_value_invalid"
                )
            if before and not after:
                return False
            if not before and after:
                improved = True
    return improved


def evaluate_filter_mechanism_decomposition_decision(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen, mutually exclusive precedence to 12 paired rows."""

    if len(rows) != 12:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_result_count_mismatch"
        )
    record_ids = [str(row.get("record_id", "")) for row in rows]
    if len(set(record_ids)) != 12 or any(not value for value in record_ids):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_record_panel_mismatch"
        )
    for row in rows:
        lanes = _require_mapping(
            "filter_mechanism_decision_lanes",
            row.get("lanes"),
        )
        if set(lanes) != set(_ALL_LANES):
            raise FilterMechanismDecompositionError(
                "filter_mechanism_decomposition_lane_set_mismatch"
            )
        for lane in _ALL_LANES:
            _lane_complete(
                _require_mapping(
                    "filter_mechanism_decision_lane",
                    lanes[lane],
                )
            )
        recomputed_control_valid = bool(
            all(
                _lane_complete(
                    _require_mapping(
                        "filter_mechanism_control_lane",
                        lanes[lane],
                    )
                )
                for lane in CONTROL_LANES
            )
            and float(row.get("zero_update_weight_max_abs", math.nan))
            == 0.0
        )
        anchor_hash = _require_hash(
            "anchor_spectral_gate_summary_sha256",
            row.get("anchor_spectral_gate_summary_sha256"),
        )
        forward_hash = _require_hash(
            "forward_spectral_gate_summary_sha256",
            row.get("forward_spectral_gate_summary_sha256"),
        )
        if (
            row.get("control_valid") is not recomputed_control_valid
            or row.get("anchor_reproduction_pass")
            is not (anchor_hash == forward_hash)
        ):
            raise FilterMechanismDecompositionError(
                "filter_mechanism_decomposition_result_consistency_mismatch"
            )

    control_valid_count = sum(
        row.get("control_valid") is True for row in rows
    )
    reproduction_count = sum(
        row.get("anchor_reproduction_pass") is True for row in rows
    )
    complete_counts = {
        lane: sum(
            _lane_complete(
                _require_mapping(
                    "filter_mechanism_decision_lane",
                    _require_mapping(
                        "filter_mechanism_decision_lanes",
                        row["lanes"],
                    )[lane],
                )
            )
            for row in rows
        )
        for lane in _ALL_LANES
    }
    dominant_lanes = [
        lane
        for lane in CANDIDATE_LANES
        if _strictly_dominates(rows, candidate_lane=lane)
    ]
    if control_valid_count < 12:
        decision = "control_invalid"
        next_state = "awaiting_filter_mechanism_control_revision"
    elif reproduction_count < 12:
        decision = "baseline_reproduction_invalid"
        next_state = "awaiting_filter_mechanism_reproduction_revision"
    elif complete_counts["rank1_only_adaptive"] == 12:
        decision = "rank1_single_stage_mechanism_candidate"
        next_state = "awaiting_rank1_filter_revision_proposal"
    elif complete_counts["rank2_only_adaptive"] == 12:
        decision = "rank2_reference_selection_mechanism_candidate"
        next_state = "awaiting_reference_ranking_revision_proposal"
    elif complete_counts["reverse_cascade_adaptive"] == 12:
        decision = "reverse_order_mechanism_candidate"
        next_state = "awaiting_cascade_order_revision_proposal"
    elif dominant_lanes:
        decision = "partial_mechanism_relief_requires_factorial"
        next_state = "awaiting_memory_mu_factorial_proposal"
    else:
        decision = (
            "no_mechanism_relief_requires_factorial_or_bo_review"
        )
        next_state = "awaiting_factorial_or_independent_bo_review"
    payload = {
        "decision_version": (
            "lyx_filter_mechanism_decomposition_decision_v1"
        ),
        "decision": decision,
        "next_state": next_state,
        "record_count": 12,
        "control_valid_count": control_valid_count,
        "anchor_reproduction_pass_count": reproduction_count,
        "lane_complete_pass_counts": complete_counts,
        "strictly_dominant_candidate_lanes": dominant_lanes,
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "may_nominate_recovery_candidate": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
    }
    payload["decision_sha256"] = canonical_sha256(payload)
    return payload


def propose_filter_mechanism_decomposition(
    *,
    p25_proposal_path: Path,
    p25_completion_path: Path,
    p25_decision_path: Path,
    p25_manifest_path: Path,
    source_budget_contract_path: Path,
    spectral_gate_contract_path: Path,
    spec_path: Path,
    output_dir: Path,
    source_root: Path,
    parent_experiment_id: str,
) -> dict[str, Any]:
    """Publish a content-addressed, zero-run review package."""

    artifacts = {
        "p25_proposal": Path(p25_proposal_path).resolve(),
        "p25_completion": Path(p25_completion_path).resolve(),
        "p25_decision": Path(p25_decision_path).resolve(),
        "p25_manifest": Path(p25_manifest_path).resolve(),
        "source_budget_contract": Path(
            source_budget_contract_path
        ).resolve(),
        "spectral_gate_contract": Path(
            spectral_gate_contract_path
        ).resolve(),
        "experiment_spec": Path(spec_path).resolve(),
    }
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    if missing:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_source_missing:"
            + ",".join(missing)
        )
    drifted = [
        name
        for name, expected in _P25_SOURCE_FILE_SHA256.items()
        if file_sha256(artifacts[name]) != expected
    ]
    if drifted:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_source_file_hash_mismatch:"
            + ",".join(drifted)
        )
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_destination_exists"
        )
    if (
        read_json(artifacts["source_budget_contract"])
        != BudgetContract.proposed_v8_p25_spectral_recheck().to_dict()
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_source_budget_mismatch"
        )
    spectral_payload = read_json(artifacts["spectral_gate_contract"])
    if (
        _verify_embedded_hash(
            spectral_payload,
            hash_field="contract_sha256",
            artifact_name="filter_mechanism_spectral_contract",
        )
        != StageRSpectralGateContract().sha256
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_spectral_contract_mismatch"
        )
    manifest = read_json(artifacts["p25_manifest"])
    anchor_paths = {
        f"anchor:{entry['record_id']}": (
            artifacts["p25_manifest"].parent / str(entry["path"])
        ).resolve()
        for entry in _require_list(
            "filter_mechanism_p25_manifest_results",
            manifest.get("results"),
        )
        if isinstance(entry, Mapping)
        and entry.get("filter_profile_id") == _CONTROL_PROFILE.profile_id
    }
    artifacts.update(anchor_paths)
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    if missing:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_anchor_missing:"
            + ",".join(missing)
        )
    manifest_entries = {
        str(entry["record_id"]): entry
        for entry in _require_list(
            "filter_mechanism_p25_manifest_results",
            manifest.get("results"),
        )
        if isinstance(entry, Mapping)
        and entry.get("filter_profile_id") == _CONTROL_PROFILE.profile_id
    }
    drifted_anchors = [
        record_id
        for record_id, entry in manifest_entries.items()
        if file_sha256(anchor_paths[f"anchor:{record_id}"])
        != entry.get("file_sha256")
    ]
    if drifted_anchors:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_anchor_file_hash_mismatch:"
            + ",".join(sorted(drifted_anchors))
        )
    source_root = Path(source_root).resolve()
    repository_root = _repository_root_from_source_root(source_root)
    relative_artifacts: dict[str, Path] = {}
    for name, path in artifacts.items():
        try:
            relative_artifacts[name] = path.relative_to(repository_root)
        except ValueError as error:
            raise FilterMechanismDecompositionError(
                "filter_mechanism_decomposition_source_outside_repository:"
                + name
            ) from error
    source_artifacts = {
        name: {
            "path": relative_artifacts[name].as_posix(),
            "path_base": "repository_root",
            "file_sha256": file_sha256(path),
        }
        for name, path in artifacts.items()
    }
    source_identity = runtime_source_identity(
        source_root,
        root_modules=(
            "ppg_hr.v2.recovery_filter_mechanism_decomposition",
            "ppg_hr.v2.recovery_filter_mechanism_decomposition_runner",
        ),
    )
    bundle_hash = str(source_identity["source_bundle_sha256"])
    anchors = {
        name.split(":", 1)[1]: read_json(path)
        for name, path in anchor_paths.items()
    }
    proposal = build_filter_mechanism_decomposition_proposal(
        p25_proposal=read_json(artifacts["p25_proposal"]),
        p25_completion=read_json(artifacts["p25_completion"]),
        p25_decision=read_json(artifacts["p25_decision"]),
        p25_manifest=manifest,
        anchor_results=anchors,
        parent_experiment_id=parent_experiment_id,
        solver_hash=bundle_hash,
        evaluation_hash=bundle_hash,
        source_artifacts=source_artifacts,
    )
    budget = BudgetContract.proposed_v9_filter_mechanism_decomposition()
    request = {
        "request_version": (
            "lyx_filter_mechanism_decomposition_budget_request_v1"
        ),
        "status": "approved_by_prior_scope_bound_user_authorization",
        "approved": True,
        "decision_state": _AUTHORIZATION_STATE,
        "proposal_sha256": proposal["proposal_sha256"],
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "added_unique_identities": 12,
        "normal_unique_identity_limit": 840,
        "max_unique_identities": 852,
        "max_attempts": 1704,
        "retry_limit": 1,
        "budget_contract_hash": budget.sha256,
        "identity_panel_sha256": proposal["identity_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "anchor_panel_sha256": proposal["anchor_panel_sha256"],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": False,
    }
    request["request_sha256"] = canonical_sha256(request)
    receipt: dict[str, Any] = {
        "receipt_version": (
            "lyx_filter_mechanism_decomposition_proposal_receipt_v1"
        ),
        "status": "frozen_zero_runs",
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_request_sha256": request["request_sha256"],
        "identity_count": 12,
        "deterministic_lane_count_per_identity": 6,
        "diagnostic_run_count": 0,
        "parameter_search_run_count": 0,
        "independent_bo_run_count": 0,
        "may_execute_under_scope_bound_user_authorization": True,
    }
    staging = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        staging.mkdir(parents=True)
        atomic_write_json(
            staging / "filter_mechanism_decomposition_proposal.json",
            proposal,
        )
        atomic_write_json(
            staging / "budget_amendment_request.json",
            request,
        )
        atomic_write_json(
            staging / "budget_contract_v9.json",
            budget.to_dict(),
        )
        mechanism = FilterMechanismDecompositionContract()
        atomic_write_json(
            staging / "mechanism_contract.json",
            {
                **mechanism.to_dict(),
                "contract_sha256": mechanism.sha256,
            },
        )
        atomic_write_json(
            staging / "spectral_gate_contract.json",
            spectral_payload,
        )
        atomic_write_json(staging / "source_identity.json", source_identity)
        receipt["artifact_sha256"] = {
            name: file_sha256(staging / name)
            for name in sorted(_PROPOSAL_ARTIFACT_NAMES)
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_write_json(staging / "proposal_receipt.json", receipt)
        os.replace(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return receipt


def validate_filter_mechanism_decomposition_authorization(
    proposal: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require an exact approval for the 12 identities and v9 budget."""

    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="filter_mechanism_decomposition_proposal",
    )
    if receipt is None or receipt.get("approved") is not True:
        raise FilterMechanismDecompositionAuthorizationError(
            "filter_mechanism_decomposition_execution_authorization_required"
        )
    frozen = _require_mapping(
        "filter_mechanism_decomposition_frozen_contracts",
        proposal.get("frozen_contracts"),
    )
    expected = {
        "decision_state": _AUTHORIZATION_STATE,
        "proposal_sha256": proposal_sha,
        "budget_contract_hash": frozen.get("budget_contract_hash"),
        "unique_budget": 12,
        "stage": _STAGE,
        "identity_panel_sha256": proposal.get("identity_panel_sha256"),
        "record_panel_sha256": proposal.get("record_panel_sha256"),
        "anchor_panel_sha256": proposal.get("anchor_panel_sha256"),
        "solver_hash": frozen.get("solver_hash"),
        "evaluation_hash": frozen.get("evaluation_hash"),
        "mechanism_contract_hash": frozen.get(
            "mechanism_contract_hash"
        ),
        "spectral_gate_contract_hash": frozen.get(
            "spectral_gate_contract_hash"
        ),
        "control_profile_hash": frozen.get("control_profile_hash"),
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": False,
    }
    mismatched = sorted(
        name for name, value in expected.items() if receipt.get(name) != value
    )
    if mismatched:
        raise FilterMechanismDecompositionAuthorizationError(
            "filter_mechanism_decomposition_authorization_mismatch:"
            + ",".join(mismatched)
        )
    for name in ("approved_at", "approved_by"):
        if not isinstance(receipt.get(name), str) or not receipt[name]:
            raise FilterMechanismDecompositionAuthorizationError(
                "filter_mechanism_decomposition_authorization_"
                + name
                + "_invalid"
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
    proposal = read_json(
        proposal_dir / "filter_mechanism_decomposition_proposal.json"
    )
    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="filter_mechanism_decomposition_proposal",
    )
    receipt = read_json(proposal_dir / "proposal_receipt.json")
    _verify_embedded_hash(
        receipt,
        hash_field="receipt_sha256",
        artifact_name="filter_mechanism_decomposition_proposal_receipt",
    )
    if receipt.get("proposal_sha256") != proposal_sha:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_proposal_receipt_mismatch"
        )
    artifact_hashes = _require_mapping(
        "filter_mechanism_decomposition_artifacts",
        receipt.get("artifact_sha256"),
    )
    if set(artifact_hashes) != _PROPOSAL_ARTIFACT_NAMES:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_artifact_set_mismatch"
        )
    for name, expected in artifact_hashes.items():
        path = proposal_dir / str(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise FilterMechanismDecompositionError(
                f"filter_mechanism_decomposition_artifact_mismatch:{name}"
            )

    source_root = Path(source_root).resolve()
    repository_root = _repository_root_from_source_root(source_root)
    source_artifacts = _require_mapping(
        "filter_mechanism_decomposition_source_artifacts",
        proposal.get("source_artifacts"),
    )
    resolved_artifacts: dict[str, Path] = {}
    for name, raw in source_artifacts.items():
        artifact = _require_mapping(
            f"filter_mechanism_source_artifact:{name}",
            raw,
        )
        relative = Path(str(artifact.get("path", "")))
        path = (repository_root / relative).resolve()
        if (
            artifact.get("path_base") != "repository_root"
            or relative.is_absolute()
            or not path.is_relative_to(repository_root)
            or not path.is_file()
            or file_sha256(path) != artifact.get("file_sha256")
        ):
            raise FilterMechanismDecompositionError(
                "filter_mechanism_decomposition_source_artifact_mismatch:"
                + str(name)
            )
        resolved_artifacts[str(name)] = path
    expected_source_keys = {
        "p25_proposal",
        "p25_completion",
        "p25_decision",
        "p25_manifest",
        "source_budget_contract",
        "spectral_gate_contract",
        "experiment_spec",
        *(f"anchor:{record_id}" for record_id in _EXPECTED_RECORD_IDS),
    }
    if set(resolved_artifacts) != expected_source_keys:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_source_artifact_set_mismatch"
        )
    current = runtime_source_identity(
        source_root,
        root_modules=(
            "ppg_hr.v2.recovery_filter_mechanism_decomposition",
            "ppg_hr.v2.recovery_filter_mechanism_decomposition_runner",
        ),
    )
    if read_json(proposal_dir / "source_identity.json") != current:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_source_identity_mismatch"
        )
    mechanism = FilterMechanismDecompositionContract()
    spectral = StageRSpectralGateContract()
    budget = BudgetContract.proposed_v9_filter_mechanism_decomposition()
    mechanism_artifact = read_json(
        proposal_dir / "mechanism_contract.json"
    )
    if (
        _verify_embedded_hash(
            mechanism_artifact,
            hash_field="contract_sha256",
            artifact_name="filter_mechanism_contract",
        )
        != mechanism.sha256
        or {
            key: value
            for key, value in mechanism_artifact.items()
            if key != "contract_sha256"
        }
        != mechanism.to_dict()
        or read_json(proposal_dir / "budget_contract_v9.json")
        != budget.to_dict()
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_frozen_artifact_mismatch"
        )
    spectral_artifact = read_json(
        proposal_dir / "spectral_gate_contract.json"
    )
    if (
        _verify_embedded_hash(
            spectral_artifact,
            hash_field="contract_sha256",
            artifact_name="filter_mechanism_spectral_contract",
        )
        != spectral.sha256
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_spectral_contract_mismatch"
        )
    frozen = _require_mapping(
        "filter_mechanism_decomposition_frozen_contracts",
        proposal.get("frozen_contracts"),
    )
    if (
        current.get("source_bundle_sha256") != frozen.get("solver_hash")
        or current.get("source_bundle_sha256")
        != frozen.get("evaluation_hash")
        or frozen.get("mechanism_contract_hash") != mechanism.sha256
        or frozen.get("spectral_gate_contract_hash") != spectral.sha256
        or frozen.get("budget_contract_hash") != budget.sha256
        or frozen.get("control_profile_hash") != _CONTROL_PROFILE.sha256
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_frozen_contract_mismatch"
        )
    anchors = {
        record_id: read_json(
            resolved_artifacts[f"anchor:{record_id}"]
        )
        for record_id in _EXPECTED_RECORD_IDS
    }
    rebuilt = build_filter_mechanism_decomposition_proposal(
        p25_proposal=read_json(resolved_artifacts["p25_proposal"]),
        p25_completion=read_json(resolved_artifacts["p25_completion"]),
        p25_decision=read_json(resolved_artifacts["p25_decision"]),
        p25_manifest=read_json(resolved_artifacts["p25_manifest"]),
        anchor_results=anchors,
        parent_experiment_id=str(proposal["parent_experiment_id"]),
        solver_hash=str(frozen["solver_hash"]),
        evaluation_hash=str(frozen["evaluation_hash"]),
        source_artifacts={
            str(name): dict(_require_mapping("source_artifact", raw))
            for name, raw in source_artifacts.items()
        },
    )
    if rebuilt != proposal:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_proposal_reconstruction_mismatch"
        )
    request = read_json(
        proposal_dir / "budget_amendment_request.json"
    )
    _verify_embedded_hash(
        request,
        hash_field="request_sha256",
        artifact_name="filter_mechanism_budget_request",
    )
    expected_request = {
        "proposal_sha256": proposal_sha,
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "added_unique_identities": 12,
        "normal_unique_identity_limit": 840,
        "max_unique_identities": 852,
        "max_attempts": 1704,
        "retry_limit": 1,
        "budget_contract_hash": budget.sha256,
        "identity_panel_sha256": proposal["identity_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "anchor_panel_sha256": proposal["anchor_panel_sha256"],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": False,
    }
    if any(request.get(name) != value for name, value in expected_request.items()):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_budget_request_mismatch"
        )
    identities = tuple(
        dict(_require_mapping("filter_mechanism_identity", item))
        for item in _require_list(
            "filter_mechanism_identities",
            proposal.get("identities"),
        )
    )
    expected_config = _identity_config(mechanism, spectral)
    actual_hashes: list[str] = []
    scenes: Counter[str] = Counter()
    for item in identities:
        identity = _identity_from_item(item)
        actual_hashes.append(identity.sha256)
        config = _require_mapping(
            "filter_mechanism_identity_config",
            item.get("config"),
        )
        if (
            item.get("identity_sha256") != identity.sha256
            or item.get("cache_identity_sha256") != identity.sha256
            or canonical_sha256(config) != identity.config_hash
            or dict(config) != expected_config
            or identity.stage != _STAGE
            or identity.attempt_kind != _ATTEMPT_KIND
            or identity.metric_contract_hash != mechanism.sha256
            or identity.solver_hash != current["source_bundle_sha256"]
            or identity.evaluation_hash != current["source_bundle_sha256"]
            or item.get("scene") not in _EXPECTED_SCENE_COUNTS
        ):
            raise FilterMechanismDecompositionError(
                "filter_mechanism_decomposition_identity_contract_mismatch"
            )
        scenes[str(item["scene"])] += 1
    if (
        len(identities) != 12
        or len(set(actual_hashes)) != 12
        or actual_hashes != proposal.get("identity_sha256")
        or canonical_sha256(actual_hashes)
        != proposal.get("identity_panel_sha256")
        or dict(scenes) != _EXPECTED_SCENE_COUNTS
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_identity_matrix_mismatch"
        )
    return proposal, identities


def prepare_filter_mechanism_decomposition_governance(
    *,
    proposal_dir: Path,
    authorization_receipt_path: Path,
    source_governance_dir: Path,
    governance_dir: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Migrate v8 and register 12 identities only after exact approval."""

    proposal, identity_items = _validate_proposal_preflight(
        proposal_dir=Path(proposal_dir).resolve(),
        source_root=source_root,
    )
    authorization = validate_filter_mechanism_decomposition_authorization(
        proposal,
        receipt=read_json(Path(authorization_receipt_path).resolve()),
    )
    target_root = Path(governance_dir).resolve()
    if target_root.exists():
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_governance_exists"
        )
    source_dir = Path(source_governance_dir).resolve()
    source_budget = BudgetContract.proposed_v8_p25_spectral_recheck()
    if (
        read_json(source_dir / "budget_contract.json")
        != source_budget.to_dict()
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_source_budget_mismatch"
        )
    exploration_payload = read_json(
        source_dir / "exploration_registry.json"
    )
    exploration = _exploration_from_payload(exploration_payload)
    if exploration.to_dict() != exploration_payload:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_exploration_registry_mismatch"
        )
    registry = AttemptRegistry.open(
        source_dir / "attempt_registry.json",
        budget_contract=source_budget,
        exploration_registry=exploration,
    )
    target_budget = (
        BudgetContract.proposed_v9_filter_mechanism_decomposition()
    )
    frozen = _require_mapping(
        "filter_mechanism_decomposition_frozen_contracts",
        proposal["frozen_contracts"],
    )
    if target_budget.sha256 != frozen.get("budget_contract_hash"):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_target_budget_mismatch"
        )
    identities = tuple(_identity_from_item(item) for item in identity_items)
    amendment = BudgetAmendmentRequest(
        stage=_STAGE,
        profile_design_rule_hash=str(
            frozen["mechanism_contract_hash"]
        ),
        record_manifest_hash=str(proposal["record_panel_sha256"]),
        added_unique_identities=12,
        normal_unique_identity_limit=840,
        max_unique_identities=852,
        max_attempts=1704,
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
                "lyx_filter_mechanism_decomposition_governance_v1"
            ),
            "status": "prepared_zero_runs",
            "proposal_sha256": proposal["proposal_sha256"],
            "source_budget_contract_hash": source_budget.sha256,
            "target_budget_contract_hash": target_budget.sha256,
            "new_unique_identity_count": 12,
            "attempt_registry_summary": staged.summary(),
            "parameter_search_authorized": False,
            "independent_bo_authorized": False,
            "automatic_stage_r_execution": False,
        }
        governance_receipt["receipt_sha256"] = canonical_sha256(
            governance_receipt
        )
        atomic_write_json(
            staging / "governance_receipt.json",
            governance_receipt,
        )

    registry.migrate_to(
        target_root / "attempt_registry.json",
        budget_contract=target_budget,
        amendment_request=amendment,
        authorization_receipt=migration_authorization,
        new_identities=identities,
        target_exploration_registry=exploration,
        finalize_staging=finalize,
    )
    return governance_receipt


def _validate_result_file(
    *,
    path: Path,
    proposal_sha256: str,
    identity: AttemptIdentity,
) -> dict[str, Any]:
    if not path.is_file():
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_result_missing:"
            + identity.record_id
        )
    payload = read_json(path)
    _verify_embedded_hash(
        payload,
        hash_field="result_sha256",
        artifact_name="filter_mechanism_decomposition_result",
    )
    if (
        payload.get("proposal_sha256") != proposal_sha256
        or payload.get("identity_sha256") != identity.sha256
        or payload.get("record_id") != identity.record_id
        or payload.get("mechanism_contract_sha256")
        != FilterMechanismDecompositionContract().sha256
        or payload.get("spectral_gate_contract_sha256")
        != StageRSpectralGateContract().sha256
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_result_identity_mismatch:"
            + identity.record_id
        )
    return payload


def execute_filter_mechanism_decomposition(
    *,
    proposal_dir: Path,
    governance_dir: Path,
    output_dir: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Execute only the approved 12-record, six-lane diagnostic panel."""

    proposal, identity_items = _validate_proposal_preflight(
        proposal_dir=Path(proposal_dir).resolve(),
        source_root=source_root,
    )
    governance_root = Path(governance_dir).resolve()
    authorization = validate_filter_mechanism_decomposition_authorization(
        proposal,
        receipt=read_json(
            governance_root / "execution_authorization.json"
        ),
    )
    budget = BudgetContract.proposed_v9_filter_mechanism_decomposition()
    if (
        read_json(governance_root / "budget_contract.json")
        != budget.to_dict()
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_execution_budget_mismatch"
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
        return _validate_completed_execution(
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
    results_dir = destination / "record_mechanism_audits"
    for item, identity in zip(identity_items, identities, strict=True):
        result_path = results_dir / f"{identity.record_id}.json"

        def run(
            *,
            _item: Mapping[str, Any] = item,
            _identity: AttemptIdentity = identity,
            _path: Path = result_path,
        ) -> dict[str, Any]:
            record = FilterAuditRecord(
                record_id=_identity.record_id,
                scene=str(_item["scene"]),
                data_path=str(_item["data_path"]),
                reference_path=str(_item["reference_path"]),
                data_sha256=str(_item["raw_data_sha256"]),
                reference_sha256=str(_item["reference_sha256"]),
            )
            audit = audit_filter_mechanism_record(
                record,
                anchor_spectral_gate_summary=_require_mapping(
                    "filter_mechanism_identity_anchor_summary",
                    _item["anchor_spectral_gate_summary"],
                ),
            )
            payload = {
                "result_version": (
                    "lyx_filter_mechanism_decomposition_result_v1"
                ),
                "proposal_sha256": proposal["proposal_sha256"],
                "identity_sha256": _identity.sha256,
                **audit,
            }
            payload["result_sha256"] = canonical_sha256(payload)
            atomic_write_json(_path, payload)
            return payload

        try:
            registry.assert_complete_matrix((identity,))
        except GovernanceError as error:
            if str(error).startswith("matrix_identity_still_running:"):
                raise FilterMechanismDecompositionError(
                    "filter_mechanism_decomposition_interrupted_attempt_"
                    "requires_human_review:"
                    + identity.record_id
                ) from error
            prior_summary = registry.matrix_execution_summary((identity,))
            if (
                prior_summary["total_attempt_count"] != 0
                or prior_summary["failed_attempt_count"] != 0
                or prior_summary["cache_only_identity_count"] != 0
            ):
                raise FilterMechanismDecompositionError(
                    "filter_mechanism_decomposition_retry_requires_"
                    "human_review:"
                    + identity.record_id
                ) from error
            row = registry.execute_registered(identity, run)
        else:
            row = _validate_result_file(
                path=result_path,
                proposal_sha256=str(proposal["proposal_sha256"]),
                identity=identity,
            )
        rows.append(row)
        result_entries.append(
            {
                "record_id": identity.record_id,
                "identity_sha256": identity.sha256,
                "path": result_path.relative_to(destination).as_posix(),
                "file_sha256": file_sha256(result_path),
                "result_sha256": row["result_sha256"],
            }
        )
    registry.assert_complete_matrix(identities)
    decision = evaluate_filter_mechanism_decomposition_decision(rows)
    decision["proposal_sha256"] = proposal["proposal_sha256"]
    decision.pop("decision_sha256")
    decision["decision_sha256"] = canonical_sha256(decision)
    atomic_write_json(destination / "decision_receipt.json", decision)
    manifest = {
        "manifest_version": (
            "lyx_filter_mechanism_decomposition_manifest_v1"
        ),
        "proposal_sha256": proposal["proposal_sha256"],
        "result_count": 12,
        "results": result_entries,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write_json(destination / "result_manifest.json", manifest)
    matrix_summary = registry.matrix_execution_summary(identities)
    expected_matrix_summary = {
        "planned_identity_count": 12,
        "identity_with_solver_attempt_count": 12,
        "cache_only_identity_count": 0,
        "total_attempt_count": 12,
        "failed_attempt_count": 0,
        "retry_count": 0,
    }
    if matrix_summary != expected_matrix_summary:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_execution_summary_mismatch"
        )
    completion = {
        "completion_version": (
            "lyx_filter_mechanism_decomposition_completion_v1"
        ),
        "status": decision["decision"],
        "next_state": decision["next_state"],
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": canonical_sha256(authorization),
        "diagnostic_result_count": 12,
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
    return _validate_completed_execution(
        completion_path=completion_path,
        proposal=proposal,
        authorization=authorization,
        output_dir=destination,
        identities=identities,
        registry=registry,
    )


def _validate_completed_execution(
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
        artifact_name="filter_mechanism_decomposition_completion",
    )
    matrix_summary = registry.matrix_execution_summary(identities)
    expected_matrix_summary = {
        "planned_identity_count": 12,
        "identity_with_solver_attempt_count": 12,
        "cache_only_identity_count": 0,
        "total_attempt_count": 12,
        "failed_attempt_count": 0,
        "retry_count": 0,
    }
    if (
        completion.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or completion.get("authorization_sha256")
        != canonical_sha256(authorization)
        or completion.get("diagnostic_result_count") != 12
        or completion.get("diagnostic_run_count")
        != matrix_summary["identity_with_solver_attempt_count"]
        or completion.get("parameter_search_run_count") != 0
        or completion.get("independent_bo_run_count") != 0
        or completion.get("may_nominate_recovery_candidate") is not False
        or completion.get("automatic_stage_r_execution") is not False
        or completion.get("automatic_stage_f_execution") is not False
        or completion.get("matrix_execution_summary") != matrix_summary
        or matrix_summary != expected_matrix_summary
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_completion_mismatch"
        )
    artifacts = _require_mapping(
        "filter_mechanism_decomposition_completion_artifacts",
        completion.get("artifacts"),
    )
    for name in ("decision_receipt.json", "result_manifest.json"):
        path = output_dir / name
        if not path.is_file() or file_sha256(path) != artifacts.get(name):
            raise FilterMechanismDecompositionError(
                "filter_mechanism_decomposition_completion_artifact_mismatch:"
                + name
            )
    decision = read_json(output_dir / "decision_receipt.json")
    decision_sha = _verify_embedded_hash(
        decision,
        hash_field="decision_sha256",
        artifact_name="filter_mechanism_decomposition_decision",
    )
    if (
        decision_sha != completion.get("decision_sha256")
        or decision.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or decision.get("decision") != completion.get("status")
        or decision.get("next_state") != completion.get("next_state")
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_completion_decision_mismatch"
        )
    manifest = read_json(output_dir / "result_manifest.json")
    _verify_embedded_hash(
        manifest,
        hash_field="manifest_sha256",
        artifact_name="filter_mechanism_decomposition_manifest",
    )
    entries = _require_list(
        "filter_mechanism_decomposition_manifest_results",
        manifest.get("results"),
    )
    identities_by_hash = {
        identity.sha256: identity for identity in identities
    }
    if (
        manifest.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or manifest.get("result_count") != 12
        or len(entries) != 12
    ):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_manifest_mismatch"
        )
    observed: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw in entries:
        entry = _require_mapping(
            "filter_mechanism_decomposition_manifest_entry",
            raw,
        )
        identity_hash = str(entry.get("identity_sha256", ""))
        identity = identities_by_hash.get(identity_hash)
        relative = Path(str(entry.get("path", "")))
        path = (output_dir / relative).resolve()
        expected = (
            Path("record_mechanism_audits")
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
            raise FilterMechanismDecompositionError(
                "filter_mechanism_decomposition_manifest_result_mismatch"
            )
        result = _validate_result_file(
            path=path,
            proposal_sha256=str(proposal["proposal_sha256"]),
            identity=identity,
        )
        if result.get("result_sha256") != entry.get("result_sha256"):
            raise FilterMechanismDecompositionError(
                "filter_mechanism_decomposition_manifest_result_mismatch"
            )
        rows.append(result)
        observed.add(identity_hash)
    if observed != set(identities_by_hash):
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_manifest_identity_set_mismatch"
        )
    expected_decision = evaluate_filter_mechanism_decomposition_decision(
        rows
    )
    expected_decision["proposal_sha256"] = proposal["proposal_sha256"]
    expected_decision.pop("decision_sha256")
    expected_decision["decision_sha256"] = canonical_sha256(
        expected_decision
    )
    if decision != expected_decision:
        raise FilterMechanismDecompositionError(
            "filter_mechanism_decomposition_decision_recomputation_mismatch"
        )
    return completion
