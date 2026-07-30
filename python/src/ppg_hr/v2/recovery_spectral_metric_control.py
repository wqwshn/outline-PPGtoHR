"""Record-level controls for the Stage R spectral metric scale."""

from __future__ import annotations

import os
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ppg_hr.core.lms_filter import lms_filter, standardize_lms_signal

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


class SpectralMetricControlError(RuntimeError):
    """The spectral scale-control package violates its frozen contract."""


class SpectralMetricControlAuthorizationError(SpectralMetricControlError):
    """The exact spectral scale-control package has not been approved."""


_STAGE = "spectral_metric_scale_control_diagnostic"
_ATTEMPT_KIND = "diagnostic"
_AUTHORIZATION_STATE = "awaiting_human_spectral_metric_scale_control_decision"
_UPSTREAM_PROPOSAL_SHA256 = (
    "db1f5d2278458592c08d7e6217d52090a8ab1f94b96262de99e84a466e2c6128"
)
_UPSTREAM_COMPLETION_SHA256 = (
    "81171332c7c29e80f329c9140874480321bf101ca7f91c853c3ae453015251ac"
)
_UPSTREAM_DECISION_SHA256 = (
    "29f042e080441ae979fea32e0b38c9de8ac778fc71d3bd5d99d78ba8ba763f55"
)
_EXPECTED_SCENE_COUNTS = {
    "jianpan": 3,
    "kaihe": 3,
    "run": 3,
    "xiezi": 3,
}
_CONTROL_PROFILE = FilterProfile(
    profile_id="p25-short-low",
    design_role="core",
    fs_target=25,
    memory_ms=40,
    nominal_mu=0.008,
)
_PROPOSAL_ARTIFACT_NAMES = {
    "budget_amendment_request.json",
    "budget_contract_v7.json",
    "control_contract.json",
    "source_identity.json",
    "spectral_metric_control_proposal.json",
}


@dataclass(frozen=True)
class SpectralMetricScaleControlContract:
    """Pre-data thresholds for three deterministic, within-window controls."""

    direct_bypass_retention_min: float = 0.999999
    direct_bypass_retention_max: float = 1.000001
    same_scale_zero_update_retention_min: float = 0.95
    same_scale_zero_update_retention_max: float = 1.05
    legacy_to_same_scale_ratio_max: float = 0.10
    expected_record_count: int = 12
    require_complete_windows: bool = True
    profile_id: str = "p25-short-low"
    forced_mu: float = 0.0
    control_version: str = "lyx_spectral_metric_scale_control_v1"

    def __post_init__(self) -> None:
        if (
            self.direct_bypass_retention_min
            > self.direct_bypass_retention_max
            or self.same_scale_zero_update_retention_min
            > self.same_scale_zero_update_retention_max
            or not 0.0 < self.legacy_to_same_scale_ratio_max < 1.0
            or self.expected_record_count != 12
            or self.profile_id != _CONTROL_PROFILE.profile_id
            or self.forced_mu != 0.0
        ):
            raise ValueError("invalid_spectral_metric_scale_control_contract")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lanes"] = {
            "direct_raw_bypass": {
                "before_scale": "raw_pre_filter",
                "after_scale": "raw_pre_filter",
                "filter_updates": 0,
            },
            "legacy_raw_vs_zero_update_lms": {
                "before_scale": "raw_pre_filter",
                "after_scale": "sample_zscore",
                "filter_updates": 0,
            },
            "same_scale_zero_update_lms": {
                "before_scale": "sample_zscore",
                "after_scale": "sample_zscore",
                "filter_updates": 0,
            },
        }
        payload["decision_precedence"] = [
            "spectral_evaluator_invalid",
            "zero_update_path_invalid",
            "legacy_scale_mismatch_confirmed",
            "legacy_scale_mismatch_partial",
            "legacy_scale_mismatch_not_reproduced",
        ]
        payload["identity_grain"] = "record"
        payload["no_parameter_search"] = True
        payload["independent_bo_authorized"] = False
        payload["may_nominate_recovery_candidate"] = False
        return payload

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpectralMetricControlError(f"{name}_must_be_object")
    return value


def _require_list(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise SpectralMetricControlError(f"{name}_must_be_array")
    return value


def _require_hash(name: str, value: object) -> str:
    text = str(value)
    try:
        require_sha256(name, text)
    except ValueError as error:
        raise SpectralMetricControlError(str(error)) from error
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
        raise SpectralMetricControlError(f"{artifact_name}_hash_mismatch")
    return declared


def _repository_root_from_source_root(source_root: Path) -> Path:
    resolved = Path(source_root).resolve()
    if (
        resolved.name != "src"
        or resolved.parent.name != "python"
        or not (resolved / "ppg_hr").is_dir()
    ):
        raise SpectralMetricControlError(
            "spectral_metric_control_source_root_mismatch"
        )
    return resolved.parent.parent


def _spectral_window(
    prepared: StageRPreparedWindow,
    *,
    before: np.ndarray,
    after: np.ndarray,
) -> dict[str, Any]:
    return {
        "before": before,
        "after": after,
        "motion_reference": prepared.primary_reference,
        "fs": prepared.fs,
        "reference_hr_bpm": prepared.reference_hr_bpm,
        "window_center_s": prepared.window_center_s,
    }


def _control_profile_payload() -> dict[str, Any]:
    return {
        **asdict(_CONTROL_PROFILE),
        "actual_taps": _CONTROL_PROFILE.actual_taps,
        "profile_sha256": _CONTROL_PROFILE.sha256,
    }


def _control_identity_config() -> dict[str, Any]:
    return {
        "control_version": "lyx_spectral_metric_scale_control_v1",
        "profile_id": _CONTROL_PROFILE.profile_id,
        "profile_sha256": _CONTROL_PROFILE.sha256,
        "fs_target": _CONTROL_PROFILE.fs_target,
        "memory_ms": _CONTROL_PROFILE.memory_ms,
        "actual_taps": _CONTROL_PROFILE.actual_taps,
        "forced_mu": 0.0,
        "lanes": [
            "direct_raw_bypass",
            "legacy_raw_vs_zero_update_lms",
            "same_scale_zero_update_lms",
        ],
        "parameter_search": False,
    }


def evaluate_spectral_metric_scale_controls(
    prepared_windows: Sequence[StageRPreparedWindow | None],
    *,
    spectral_contract: StageRSpectralGateContract,
    control_contract: SpectralMetricScaleControlContract,
) -> dict[str, Any]:
    """Evaluate three deterministic lanes on one frozen record window set."""

    direct: list[dict[str, Any]] = []
    legacy: list[dict[str, Any]] = []
    same_scale: list[dict[str, Any]] = []
    zero_weight_max_abs = 0.0
    for prepared in prepared_windows:
        if prepared is None:
            direct.append({})
            legacy.append({})
            same_scale.append({})
            continue
        original = np.asarray(prepared.original, dtype=float)
        current = original.copy()
        for _channel, reference, _corr in prepared.ranked_references:
            current, weights, _unused = lms_filter(
                control_contract.forced_mu,
                prepared.order,
                0,
                reference,
                current,
            )
            if weights.size:
                zero_weight_max_abs = max(
                    zero_weight_max_abs,
                    float(np.max(np.abs(weights))),
                )
        standardized = standardize_lms_signal(original)
        direct.append(
            _spectral_window(
                prepared,
                before=original,
                after=original.copy(),
            )
        )
        legacy.append(
            _spectral_window(
                prepared,
                before=original,
                after=current,
            )
        )
        same_scale.append(
            _spectral_window(
                prepared,
                before=standardized,
                after=current,
            )
        )

    lanes = {
        "direct_raw_bypass": evaluate_stage_r_spectral_gate_windows(
            direct,
            contract=spectral_contract,
        ),
        "legacy_raw_vs_zero_update_lms": (
            evaluate_stage_r_spectral_gate_windows(
                legacy,
                contract=spectral_contract,
            )
        ),
        "same_scale_zero_update_lms": (
            evaluate_stage_r_spectral_gate_windows(
                same_scale,
                contract=spectral_contract,
            )
        ),
    }
    retention = {
        name: value.get("pulse_power_retention_median")
        for name, value in lanes.items()
    }
    complete = all(
        value["invalid_window_count"] == 0
        and value["valid_window_count"] >= spectral_contract.minimum_valid_window_count
        for value in lanes.values()
    )
    direct_value = retention["direct_raw_bypass"]
    legacy_value = retention["legacy_raw_vs_zero_update_lms"]
    same_value = retention["same_scale_zero_update_lms"]
    finite = all(
        value is not None and np.isfinite(float(value))
        for value in (direct_value, legacy_value, same_value)
    )
    direct_pass = bool(
        finite
        and control_contract.direct_bypass_retention_min
        <= float(direct_value)
        <= control_contract.direct_bypass_retention_max
    )
    same_pass = bool(
        finite
        and control_contract.same_scale_zero_update_retention_min
        <= float(same_value)
        <= control_contract.same_scale_zero_update_retention_max
        and zero_weight_max_abs == 0.0
    )
    ratio = (
        None
        if not finite or float(same_value) <= 0.0
        else float(legacy_value) / float(same_value)
    )
    mismatch = bool(
        ratio is not None
        and ratio <= control_contract.legacy_to_same_scale_ratio_max
    )
    if control_contract.require_complete_windows and not complete:
        direct_pass = False
        same_pass = False
        mismatch = False
    return {
        "control_version": control_contract.control_version,
        "control_contract_sha256": control_contract.sha256,
        "spectral_gate_contract_sha256": spectral_contract.sha256,
        "prepared_window_count": len(prepared_windows),
        "complete_window_evidence": complete,
        "zero_update_weight_max_abs": zero_weight_max_abs,
        "direct_bypass_pass": direct_pass,
        "same_scale_zero_update_pass": same_pass,
        "legacy_scale_mismatch_reproduced": mismatch,
        "legacy_to_same_scale_retention_ratio": ratio,
        "pulse_power_retention_median": retention,
        "lanes": lanes,
    }


def audit_spectral_metric_scale_record(
    record: FilterAuditRecord,
    *,
    spectral_contract: StageRSpectralGateContract | None = None,
    control_contract: SpectralMetricScaleControlContract | None = None,
) -> dict[str, Any]:
    """Run the frozen control lanes once for one LYX development record."""

    spectral = (
        spectral_contract
        or StageRSpectralGateContract.legacy_v1()
    )
    control = control_contract or SpectralMetricScaleControlContract()
    prepared = prepare_stage_r_record_windows(_CONTROL_PROFILE, record)
    result = evaluate_spectral_metric_scale_controls(
        prepared,
        spectral_contract=spectral,
        control_contract=control,
    )
    return {
        "record_id": record.record_id,
        "scene": record.scene,
        "profile_id": _CONTROL_PROFILE.profile_id,
        "profile_sha256": _CONTROL_PROFILE.sha256,
        **result,
    }


def evaluate_spectral_metric_control_decision(
    rows: Sequence[Mapping[str, Any]],
    *,
    control_contract: SpectralMetricScaleControlContract,
) -> dict[str, Any]:
    """Apply the frozen precedence to the complete 12-record control panel."""

    if len(rows) != control_contract.expected_record_count:
        raise SpectralMetricControlError("spectral_metric_control_result_count_mismatch")
    record_ids = [str(row.get("record_id", "")) for row in rows]
    if len(set(record_ids)) != len(record_ids) or any(not value for value in record_ids):
        raise SpectralMetricControlError("spectral_metric_control_record_panel_mismatch")
    direct_count = sum(row.get("direct_bypass_pass") is True for row in rows)
    same_count = sum(row.get("same_scale_zero_update_pass") is True for row in rows)
    mismatch_count = sum(
        row.get("legacy_scale_mismatch_reproduced") is True for row in rows
    )
    complete_count = sum(row.get("complete_window_evidence") is True for row in rows)
    expected = control_contract.expected_record_count
    if direct_count < expected or complete_count < expected:
        decision = "spectral_evaluator_invalid"
        next_state = "awaiting_spectral_evaluator_revision"
    elif same_count < expected:
        decision = "zero_update_path_invalid"
        next_state = "awaiting_zero_update_path_revision"
    elif mismatch_count == expected:
        decision = "legacy_scale_mismatch_confirmed"
        next_state = "awaiting_spectral_metric_scale_correction"
    elif mismatch_count:
        decision = "legacy_scale_mismatch_partial"
        next_state = "awaiting_record_level_scale_mismatch_review"
    else:
        decision = "legacy_scale_mismatch_not_reproduced"
        next_state = "awaiting_filter_mechanism_revision"
    payload = {
        "decision_version": "lyx_spectral_metric_scale_control_decision_v1",
        "decision": decision,
        "next_state": next_state,
        "record_count": expected,
        "direct_bypass_pass_count": direct_count,
        "same_scale_zero_update_pass_count": same_count,
        "legacy_scale_mismatch_reproduced_count": mismatch_count,
        "complete_window_evidence_count": complete_count,
        "independent_bo_authorized": False,
        "may_nominate_recovery_candidate": False,
        "automatic_stage_r_execution": False,
    }
    payload["decision_sha256"] = canonical_sha256(payload)
    return payload


def _validate_upstream(
    *,
    proposal: Mapping[str, Any],
    completion: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="p25_spectral_proposal",
    )
    completion_sha = _verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="p25_spectral_completion",
    )
    decision_sha = _verify_embedded_hash(
        decision,
        hash_field="decision_sha256",
        artifact_name="p25_spectral_decision",
    )
    if (
        proposal_sha != _UPSTREAM_PROPOSAL_SHA256
        or completion_sha != _UPSTREAM_COMPLETION_SHA256
        or decision_sha != _UPSTREAM_DECISION_SHA256
        or completion.get("proposal_sha256") != proposal_sha
        or completion.get("status") != "spectral_metric_control_audit_required"
        or completion.get("next_state") != "awaiting_spectral_metric_control_audit"
        or completion.get("diagnostic_result_count") != 36
        or completion.get("independent_bo_run_count") != 0
        or decision.get("proposal_sha256") != proposal_sha
        or decision.get("decision") != "spectral_metric_control_audit_required"
        or decision.get("pulse_power_retention_pass_count") != 0
        or decision.get("result_count") != 36
    ):
        raise SpectralMetricControlError("spectral_metric_control_upstream_mismatch")
    templates = tuple(
        dict(_require_mapping("p25_control_identity", item))
        for item in _require_list("p25_identities", proposal.get("identities"))
        if isinstance(item, Mapping)
        and item.get("filter_profile_id") == _CONTROL_PROFILE.profile_id
    )
    if len(templates) != 12:
        raise SpectralMetricControlError("spectral_metric_control_template_count_mismatch")
    scene_counts = Counter(str(item.get("scene", "")) for item in templates)
    if dict(scene_counts) != _EXPECTED_SCENE_COUNTS:
        raise SpectralMetricControlError("spectral_metric_control_scene_panel_mismatch")
    return tuple(sorted(templates, key=lambda item: str(item["record_id"])))


def _identity_item(
    *,
    template: Mapping[str, Any],
    parent_experiment_id: str,
    solver_hash: str,
    evaluation_hash: str,
    metric_contract_hash: str,
) -> dict[str, Any]:
    config = _control_identity_config()
    identity = AttemptIdentity(
        solver_hash=solver_hash,
        config_hash=canonical_sha256(config),
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
        "config": config,
        "source_p25_identity_sha256": template["identity_sha256"],
    }


def build_spectral_metric_control_proposal(
    *,
    p25_proposal: Mapping[str, Any],
    p25_completion: Mapping[str, Any],
    p25_decision: Mapping[str, Any],
    parent_experiment_id: str,
    solver_hash: str,
    evaluation_hash: str,
) -> dict[str, Any]:
    """Build exactly 12 record identities without authorizing execution."""

    if not parent_experiment_id:
        raise SpectralMetricControlError("parent_experiment_id_must_not_be_empty")
    templates = _validate_upstream(
        proposal=p25_proposal,
        completion=p25_completion,
        decision=p25_decision,
    )
    solver_hash = _require_hash("solver_hash", solver_hash)
    evaluation_hash = _require_hash("evaluation_hash", evaluation_hash)
    control = SpectralMetricScaleControlContract()
    spectral = StageRSpectralGateContract.legacy_v1()
    budget = BudgetContract.proposed_v7_spectral_metric_control()
    identities = [
        _identity_item(
            template=template,
            parent_experiment_id=parent_experiment_id,
            solver_hash=solver_hash,
            evaluation_hash=evaluation_hash,
            metric_contract_hash=control.sha256,
        )
        for template in templates
    ]
    identity_hashes = [str(item["identity_sha256"]) for item in identities]
    if len(set(identity_hashes)) != 12:
        raise SpectralMetricControlError("spectral_metric_control_identity_matrix_mismatch")
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
    proposal = {
        "proposal_version": "lyx_spectral_metric_scale_control_proposal_v1",
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
        "deterministic_lane_count_per_identity": 3,
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "may_nominate_recovery_candidate": False,
        "automatic_stage_r_execution": False,
        "upstream_p25": {
            "proposal_sha256": _UPSTREAM_PROPOSAL_SHA256,
            "completion_sha256": _UPSTREAM_COMPLETION_SHA256,
            "decision_sha256": _UPSTREAM_DECISION_SHA256,
            "status": "spectral_metric_control_audit_required",
        },
        "frozen_contracts": {
            "solver_hash": solver_hash,
            "evaluation_hash": evaluation_hash,
            "control_contract_hash": control.sha256,
            "spectral_gate_contract_hash": spectral.sha256,
            "budget_contract_hash": budget.sha256,
            "control_profile_hash": _CONTROL_PROFILE.sha256,
        },
        "control_contract": control.to_dict(),
        "control_profile": _control_profile_payload(),
        "decision_branches": {
            "spectral_evaluator_invalid": "direct bypass or complete-window control fails",
            "zero_update_path_invalid": "direct bypass passes but same-scale zero-update fails",
            "legacy_scale_mismatch_confirmed": "all 12 records reproduce only the legacy mismatch",
            "legacy_scale_mismatch_partial": "1-11 records reproduce the legacy mismatch",
            "legacy_scale_mismatch_not_reproduced": "no record reproduces the legacy mismatch",
        },
        "record_panel": record_panel,
        "record_panel_sha256": canonical_sha256(record_panel),
        "identity_sha256": identity_hashes,
        "identity_panel_sha256": canonical_sha256(identity_hashes),
        "identities": identities,
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def propose_spectral_metric_control(
    *,
    p25_proposal_path: Path,
    p25_completion_path: Path,
    p25_decision_path: Path,
    source_budget_contract_path: Path,
    spectral_gate_contract_path: Path,
    output_dir: Path,
    source_root: Path,
    parent_experiment_id: str,
) -> dict[str, Any]:
    """Publish a content-addressed, zero-run review package."""

    artifacts = {
        "p25_proposal": Path(p25_proposal_path).resolve(),
        "p25_completion": Path(p25_completion_path).resolve(),
        "p25_decision": Path(p25_decision_path).resolve(),
        "source_budget_contract": Path(source_budget_contract_path).resolve(),
        "spectral_gate_contract": Path(spectral_gate_contract_path).resolve(),
    }
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    if missing:
        raise SpectralMetricControlError(
            "spectral_metric_control_source_missing:" + ",".join(missing)
        )
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise SpectralMetricControlError("spectral_metric_control_destination_exists")
    source_budget = read_json(artifacts["source_budget_contract"])
    if source_budget != BudgetContract.approved_v6_p25_diagnostic().to_dict():
        raise SpectralMetricControlError("spectral_metric_control_source_budget_mismatch")
    spectral_payload = read_json(artifacts["spectral_gate_contract"])
    if _verify_embedded_hash(
        spectral_payload,
        hash_field="contract_sha256",
        artifact_name="stage_r_spectral_gate_contract",
    ) != StageRSpectralGateContract.legacy_v1().sha256:
        raise SpectralMetricControlError("spectral_metric_control_spectral_contract_mismatch")

    source_root = Path(source_root).resolve()
    repository_root = _repository_root_from_source_root(source_root)
    relative_artifacts: dict[str, Path] = {}
    for name, path in artifacts.items():
        try:
            relative_artifacts[name] = path.relative_to(repository_root)
        except ValueError as error:
            raise SpectralMetricControlError(
                f"spectral_metric_control_source_outside_repository:{name}"
            ) from error
    source_identity = runtime_source_identity(
        source_root,
        root_modules=("ppg_hr.v2.recovery_spectral_metric_control",),
    )
    bundle_hash = str(source_identity["source_bundle_sha256"])
    proposal = build_spectral_metric_control_proposal(
        p25_proposal=read_json(artifacts["p25_proposal"]),
        p25_completion=read_json(artifacts["p25_completion"]),
        p25_decision=read_json(artifacts["p25_decision"]),
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
        for name, path in artifacts.items()
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    budget = BudgetContract.proposed_v7_spectral_metric_control()
    request = {
        "request_version": "lyx_spectral_metric_scale_control_budget_request_v1",
        "status": "awaiting_human_budget_and_execution_decision",
        "approved": False,
        "decision_state": _AUTHORIZATION_STATE,
        "proposal_sha256": proposal["proposal_sha256"],
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "added_unique_identities": 12,
        "normal_unique_identity_limit": 792,
        "max_unique_identities": 804,
        "max_attempts": 1608,
        "retry_limit": 1,
        "budget_contract_hash": budget.sha256,
        "identity_panel_sha256": proposal["identity_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
    }
    request["request_sha256"] = canonical_sha256(request)
    receipt: dict[str, Any] = {
        "receipt_version": "lyx_spectral_metric_scale_control_proposal_receipt_v1",
        "status": "awaiting_human_execution_authorization",
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_request_sha256": request["request_sha256"],
        "identity_count": 12,
        "diagnostic_run_count": 0,
        "independent_bo_run_count": 0,
        "may_execute": False,
    }
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        staging.mkdir(parents=True)
        atomic_write_json(staging / "spectral_metric_control_proposal.json", proposal)
        atomic_write_json(staging / "budget_amendment_request.json", request)
        atomic_write_json(staging / "budget_contract_v7.json", budget.to_dict())
        atomic_write_json(
            staging / "control_contract.json",
            {
                **SpectralMetricScaleControlContract().to_dict(),
                "contract_sha256": SpectralMetricScaleControlContract().sha256,
            },
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


def validate_spectral_metric_control_authorization(
    proposal: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require an exact approval for all 12 identities and v7 budget."""

    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="spectral_metric_control_proposal",
    )
    if receipt is None or receipt.get("approved") is not True:
        raise SpectralMetricControlAuthorizationError(
            "spectral_metric_control_execution_authorization_required"
        )
    frozen = _require_mapping(
        "spectral_metric_control_frozen_contracts",
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
        "solver_hash": frozen.get("solver_hash"),
        "evaluation_hash": frozen.get("evaluation_hash"),
        "control_contract_hash": frozen.get("control_contract_hash"),
        "spectral_gate_contract_hash": frozen.get("spectral_gate_contract_hash"),
        "control_profile_hash": frozen.get("control_profile_hash"),
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
    }
    mismatched = sorted(
        name for name, value in expected.items() if receipt.get(name) != value
    )
    if mismatched:
        raise SpectralMetricControlAuthorizationError(
            "spectral_metric_control_authorization_mismatch:"
            + ",".join(mismatched)
        )
    for name in (
        "approved_at",
        "approved_by",
    ):
        if not isinstance(receipt.get(name), str) or not receipt[name]:
            raise SpectralMetricControlAuthorizationError(
                f"spectral_metric_control_authorization_{name}_invalid"
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


def _exploration_from_payload(payload: Mapping[str, Any]) -> ExplorationRegistry:
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
    proposal = read_json(proposal_dir / "spectral_metric_control_proposal.json")
    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="spectral_metric_control_proposal",
    )
    receipt = read_json(proposal_dir / "proposal_receipt.json")
    _verify_embedded_hash(
        receipt,
        hash_field="receipt_sha256",
        artifact_name="spectral_metric_control_proposal_receipt",
    )
    if receipt.get("proposal_sha256") != proposal_sha:
        raise SpectralMetricControlError("spectral_metric_control_proposal_receipt_mismatch")
    artifact_hashes = _require_mapping(
        "spectral_metric_control_artifacts",
        receipt.get("artifact_sha256"),
    )
    if set(artifact_hashes) != _PROPOSAL_ARTIFACT_NAMES:
        raise SpectralMetricControlError(
            "spectral_metric_control_artifact_set_mismatch"
        )
    for name, expected in artifact_hashes.items():
        path = proposal_dir / str(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise SpectralMetricControlError(
                f"spectral_metric_control_artifact_mismatch:{name}"
            )
    repository_root = _repository_root_from_source_root(source_root)
    for name, raw in _require_mapping(
        "spectral_metric_control_source_artifacts",
        proposal.get("source_artifacts"),
    ).items():
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
            raise SpectralMetricControlError(
                f"spectral_metric_control_source_artifact_mismatch:{name}"
            )
    current = runtime_source_identity(
        Path(source_root).resolve(),
        root_modules=("ppg_hr.v2.recovery_spectral_metric_control",),
    )
    if read_json(proposal_dir / "source_identity.json") != current:
        raise SpectralMetricControlError(
            "spectral_metric_control_source_identity_artifact_mismatch"
        )
    control = SpectralMetricScaleControlContract()
    spectral = StageRSpectralGateContract.legacy_v1()
    budget = BudgetContract.proposed_v7_spectral_metric_control()
    control_artifact = read_json(proposal_dir / "control_contract.json")
    if (
        _verify_embedded_hash(
            control_artifact,
            hash_field="contract_sha256",
            artifact_name="spectral_metric_control_contract",
        )
        != control.sha256
        or {
            key: value
            for key, value in control_artifact.items()
            if key != "contract_sha256"
        }
        != control.to_dict()
    ):
        raise SpectralMetricControlError(
            "spectral_metric_control_contract_artifact_mismatch"
        )
    if read_json(proposal_dir / "budget_contract_v7.json") != budget.to_dict():
        raise SpectralMetricControlError(
            "spectral_metric_control_budget_artifact_mismatch"
        )
    budget_request = read_json(proposal_dir / "budget_amendment_request.json")
    _verify_embedded_hash(
        budget_request,
        hash_field="request_sha256",
        artifact_name="spectral_metric_control_budget_request",
    )
    frozen = _require_mapping(
        "spectral_metric_control_frozen_contracts",
        proposal.get("frozen_contracts"),
    )
    if (
        current.get("source_bundle_sha256") != frozen.get("solver_hash")
        or current.get("source_bundle_sha256") != frozen.get("evaluation_hash")
        or frozen.get("control_contract_hash") != control.sha256
        or frozen.get("spectral_gate_contract_hash") != spectral.sha256
        or frozen.get("budget_contract_hash") != budget.sha256
        or frozen.get("control_profile_hash") != _CONTROL_PROFILE.sha256
    ):
        raise SpectralMetricControlError(
            "spectral_metric_control_frozen_contract_mismatch"
        )
    if (
        proposal.get("status") != "awaiting_human_execution_authorization"
        or proposal.get("authorization_state") != _AUTHORIZATION_STATE
        or proposal.get("stage") != _STAGE
        or proposal.get("attempt_kind") != _ATTEMPT_KIND
        or proposal.get("unique_budget") != 12
        or proposal.get("retry_limit") != 1
        or proposal.get("worst_case_attempt_budget") != 24
        or proposal.get("deterministic_lane_count_per_identity") != 3
        or proposal.get("parameter_search_authorized") is not False
        or proposal.get("independent_bo_authorized") is not False
        or proposal.get("may_nominate_recovery_candidate") is not False
        or proposal.get("automatic_stage_r_execution") is not False
        or proposal.get("control_contract") != control.to_dict()
        or proposal.get("control_profile") != _control_profile_payload()
    ):
        raise SpectralMetricControlError(
            "spectral_metric_control_proposal_contract_mismatch"
        )
    expected_budget_request = {
        "proposal_sha256": proposal_sha,
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "added_unique_identities": 12,
        "normal_unique_identity_limit": 792,
        "max_unique_identities": 804,
        "max_attempts": 1608,
        "retry_limit": 1,
        "budget_contract_hash": budget.sha256,
        "identity_panel_sha256": proposal.get("identity_panel_sha256"),
        "record_panel_sha256": proposal.get("record_panel_sha256"),
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
    }
    if any(
        budget_request.get(name) != value
        for name, value in expected_budget_request.items()
    ):
        raise SpectralMetricControlError(
            "spectral_metric_control_budget_request_mismatch"
        )
    identities = tuple(
        dict(_require_mapping("spectral_metric_control_identity", item))
        for item in _require_list(
            "spectral_metric_control_identities",
            proposal.get("identities"),
        )
    )
    hashes = [str(item["identity_sha256"]) for item in identities]
    actual_hashes: list[str] = []
    record_panel: list[dict[str, Any]] = []
    scenes: Counter[str] = Counter()
    for item in identities:
        identity = _identity_from_item(item)
        actual_hashes.append(identity.sha256)
        config = _require_mapping(
            "spectral_metric_control_identity_config",
            item.get("config"),
        )
        if (
            item.get("identity_sha256") != identity.sha256
            or item.get("cache_identity_sha256") != identity.sha256
            or canonical_sha256(config) != identity.config_hash
            or dict(config) != _control_identity_config()
            or identity.stage != _STAGE
            or identity.attempt_kind != _ATTEMPT_KIND
            or identity.metric_contract_hash != control.sha256
            or identity.solver_hash
            != current.get("source_bundle_sha256")
            or identity.evaluation_hash
            != current.get("source_bundle_sha256")
            or item.get("scene") not in _EXPECTED_SCENE_COUNTS
        ):
            raise SpectralMetricControlError(
                "spectral_metric_control_identity_contract_mismatch"
            )
        scenes[str(item["scene"])] += 1
        record_panel.append(
            {
                "record_id": identity.record_id,
                "scene": item["scene"],
                "raw_data_sha256": item["raw_data_sha256"],
                "reference_sha256": item["reference_sha256"],
                "data_sha256": identity.data_sha256,
            }
        )
    if (
        len(identities) != 12
        or len(set(actual_hashes)) != 12
        or hashes != proposal.get("identity_sha256")
        or hashes != actual_hashes
        or canonical_sha256(hashes) != proposal.get("identity_panel_sha256")
        or dict(scenes) != _EXPECTED_SCENE_COUNTS
        or record_panel != proposal.get("record_panel")
        or canonical_sha256(record_panel)
        != proposal.get("record_panel_sha256")
    ):
        raise SpectralMetricControlError("spectral_metric_control_identity_matrix_mismatch")
    return proposal, identities


def prepare_spectral_metric_control_governance(
    *,
    proposal_dir: Path,
    authorization_receipt_path: Path,
    source_governance_dir: Path,
    governance_dir: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Migrate v6 and register 12 identities only after exact approval."""

    proposal_root = Path(proposal_dir).resolve()
    proposal, identity_items = _validate_proposal_preflight(
        proposal_dir=proposal_root,
        source_root=source_root,
    )
    authorization = validate_spectral_metric_control_authorization(
        proposal,
        receipt=read_json(Path(authorization_receipt_path).resolve()),
    )
    target_root = Path(governance_dir).resolve()
    if target_root.exists():
        raise SpectralMetricControlError("spectral_metric_control_governance_exists")
    source_root_dir = Path(source_governance_dir).resolve()
    source_budget = BudgetContract.approved_v6_p25_diagnostic()
    if read_json(source_root_dir / "budget_contract.json") != source_budget.to_dict():
        raise SpectralMetricControlError("spectral_metric_control_source_budget_mismatch")
    exploration_payload = read_json(source_root_dir / "exploration_registry.json")
    exploration = _exploration_from_payload(exploration_payload)
    if exploration.to_dict() != exploration_payload:
        raise SpectralMetricControlError(
            "spectral_metric_control_exploration_registry_mismatch"
        )
    registry = AttemptRegistry.open(
        source_root_dir / "attempt_registry.json",
        budget_contract=source_budget,
        exploration_registry=exploration,
    )
    target_budget = BudgetContract.proposed_v7_spectral_metric_control()
    frozen = _require_mapping(
        "spectral_metric_control_frozen_contracts",
        proposal["frozen_contracts"],
    )
    if target_budget.sha256 != frozen.get("budget_contract_hash"):
        raise SpectralMetricControlError("spectral_metric_control_target_budget_mismatch")
    identities = tuple(_identity_from_item(item) for item in identity_items)
    amendment = BudgetAmendmentRequest(
        stage=_STAGE,
        profile_design_rule_hash=str(frozen["control_contract_hash"]),
        record_manifest_hash=str(proposal["record_panel_sha256"]),
        added_unique_identities=12,
        normal_unique_identity_limit=792,
        max_unique_identities=804,
        max_attempts=1608,
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
        atomic_write_json(staging / "budget_contract.json", target_budget.to_dict())
        atomic_write_json(staging / "exploration_registry.json", exploration.to_dict())
        atomic_write_json(staging / "execution_authorization.json", authorization)
        governance_receipt = {
            "receipt_version": "lyx_spectral_metric_scale_control_governance_v1",
            "status": "prepared_zero_runs",
            "proposal_sha256": proposal["proposal_sha256"],
            "source_budget_contract_hash": source_budget.sha256,
            "target_budget_contract_hash": target_budget.sha256,
            "new_unique_identity_count": 12,
            "attempt_registry_summary": staged.summary(),
            "parameter_search_authorized": False,
            "independent_bo_authorized": False,
        }
        governance_receipt["receipt_sha256"] = canonical_sha256(governance_receipt)
        atomic_write_json(staging / "governance_receipt.json", governance_receipt)

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


def execute_spectral_metric_control(
    *,
    proposal_dir: Path,
    governance_dir: Path,
    output_dir: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Execute only the registered 12-record deterministic control panel."""

    proposal, identity_items = _validate_proposal_preflight(
        proposal_dir=Path(proposal_dir).resolve(),
        source_root=source_root,
    )
    governance_root = Path(governance_dir).resolve()
    authorization = validate_spectral_metric_control_authorization(
        proposal,
        receipt=read_json(governance_root / "execution_authorization.json"),
    )
    budget = BudgetContract.proposed_v7_spectral_metric_control()
    if read_json(governance_root / "budget_contract.json") != budget.to_dict():
        raise SpectralMetricControlError("spectral_metric_control_execution_budget_mismatch")
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
        return _validate_completed_control_execution(
            completion_path=completion_path,
            proposal=proposal,
            authorization=authorization,
            output_dir=destination,
            identities=identities,
            registry=registry,
        )

    registry.register_identities(identities)
    rows: list[dict[str, Any]] = []
    result_files: list[dict[str, Any]] = []
    results_dir = destination / "record_controls"
    for item, identity in zip(identity_items, identities, strict=True):
        path = results_dir / f"{identity.record_id}.json"

        def run(
            *,
            _item: Mapping[str, Any] = item,
            _identity: AttemptIdentity = identity,
            _path: Path = path,
        ) -> dict[str, Any]:
            record = FilterAuditRecord(
                record_id=_identity.record_id,
                scene=str(_item["scene"]),
                data_path=str(_item["data_path"]),
                reference_path=str(_item["reference_path"]),
                data_sha256=str(_item["raw_data_sha256"]),
                reference_sha256=str(_item["reference_sha256"]),
            )
            audit = audit_spectral_metric_scale_record(record)
            payload = {
                "result_version": "lyx_spectral_metric_scale_control_result_v1",
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
                registry.reconcile_interrupted_attempt(
                    identity,
                    evidence=None,
                )
            row = registry.execute_registered(identity, run)
        else:
            row = _validate_control_result_file(
                path=path,
                proposal_sha256=str(proposal["proposal_sha256"]),
                identity=identity,
            )
        rows.append(row)
        result_files.append(
            {
                "record_id": identity.record_id,
                "identity_sha256": identity.sha256,
                "path": path.relative_to(destination).as_posix(),
                "file_sha256": file_sha256(path),
                "result_sha256": row["result_sha256"],
            }
        )
    registry.assert_complete_matrix(identities)
    decision = evaluate_spectral_metric_control_decision(
        rows,
        control_contract=SpectralMetricScaleControlContract(),
    )
    decision["proposal_sha256"] = proposal["proposal_sha256"]
    decision.pop("decision_sha256")
    decision["decision_sha256"] = canonical_sha256(decision)
    atomic_write_json(destination / "decision_receipt.json", decision)
    manifest = {
        "manifest_version": "lyx_spectral_metric_scale_control_manifest_v1",
        "proposal_sha256": proposal["proposal_sha256"],
        "result_count": 12,
        "results": result_files,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write_json(destination / "result_manifest.json", manifest)
    matrix_summary = registry.matrix_execution_summary(identities)
    completion = {
        "completion_version": "lyx_spectral_metric_scale_control_completion_v1",
        "status": decision["decision"],
        "next_state": decision["next_state"],
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": canonical_sha256(authorization),
        "diagnostic_result_count": 12,
        "diagnostic_run_count": matrix_summary["identity_with_solver_attempt_count"],
        "parameter_search_run_count": 0,
        "independent_bo_run_count": 0,
        "may_nominate_recovery_candidate": False,
        "matrix_execution_summary": matrix_summary,
        "decision_sha256": decision["decision_sha256"],
        "artifacts": {
            "decision_receipt.json": file_sha256(
                destination / "decision_receipt.json"
            ),
            "result_manifest.json": file_sha256(destination / "result_manifest.json"),
        },
    }
    completion["completion_sha256"] = canonical_sha256(completion)
    atomic_write_json(completion_path, completion)
    return _validate_completed_control_execution(
        completion_path=completion_path,
        proposal=proposal,
        authorization=authorization,
        output_dir=destination,
        identities=identities,
        registry=registry,
    )


def _validate_control_result_file(
    *,
    path: Path,
    proposal_sha256: str,
    identity: AttemptIdentity,
) -> dict[str, Any]:
    if not path.is_file():
        raise SpectralMetricControlError(
            f"spectral_metric_control_result_missing:{identity.record_id}"
        )
    payload = read_json(path)
    _verify_embedded_hash(
        payload,
        hash_field="result_sha256",
        artifact_name="spectral_metric_control_result",
    )
    if (
        payload.get("proposal_sha256") != proposal_sha256
        or payload.get("identity_sha256") != identity.sha256
        or payload.get("record_id") != identity.record_id
        or payload.get("control_contract_sha256")
        != SpectralMetricScaleControlContract().sha256
        or payload.get("spectral_gate_contract_sha256")
        != StageRSpectralGateContract.legacy_v1().sha256
    ):
        raise SpectralMetricControlError(
            f"spectral_metric_control_result_identity_mismatch:{identity.record_id}"
        )
    return payload


def _validate_completed_control_execution(
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
        artifact_name="spectral_metric_control_completion",
    )
    matrix_summary = registry.matrix_execution_summary(identities)
    if (
        completion.get("proposal_sha256") != proposal.get("proposal_sha256")
        or completion.get("authorization_sha256")
        != canonical_sha256(authorization)
        or completion.get("diagnostic_result_count") != 12
        or completion.get("diagnostic_run_count")
        != matrix_summary["identity_with_solver_attempt_count"]
        or completion.get("parameter_search_run_count") != 0
        or completion.get("independent_bo_run_count") != 0
        or completion.get("may_nominate_recovery_candidate") is not False
        or completion.get("matrix_execution_summary") != matrix_summary
    ):
        raise SpectralMetricControlError(
            "spectral_metric_control_completion_identity_mismatch"
        )
    artifacts = _require_mapping(
        "spectral_metric_control_completion_artifacts",
        completion.get("artifacts"),
    )
    for name in ("decision_receipt.json", "result_manifest.json"):
        path = output_dir / name
        if not path.is_file() or file_sha256(path) != artifacts.get(name):
            raise SpectralMetricControlError(
                f"spectral_metric_control_completion_artifact_mismatch:{name}"
            )
    decision = read_json(output_dir / "decision_receipt.json")
    decision_sha = _verify_embedded_hash(
        decision,
        hash_field="decision_sha256",
        artifact_name="spectral_metric_control_decision",
    )
    if (
        decision_sha != completion.get("decision_sha256")
        or decision.get("proposal_sha256") != proposal.get("proposal_sha256")
        or decision.get("decision") != completion.get("status")
        or decision.get("next_state") != completion.get("next_state")
    ):
        raise SpectralMetricControlError(
            "spectral_metric_control_completion_decision_mismatch"
        )
    manifest = read_json(output_dir / "result_manifest.json")
    _verify_embedded_hash(
        manifest,
        hash_field="manifest_sha256",
        artifact_name="spectral_metric_control_manifest",
    )
    entries = _require_list(
        "spectral_metric_control_manifest_results",
        manifest.get("results"),
    )
    identities_by_hash = {identity.sha256: identity for identity in identities}
    if (
        manifest.get("proposal_sha256") != proposal.get("proposal_sha256")
        or manifest.get("result_count") != 12
        or len(entries) != 12
    ):
        raise SpectralMetricControlError(
            "spectral_metric_control_manifest_mismatch"
        )
    observed: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw in entries:
        entry = _require_mapping("spectral_metric_control_manifest_entry", raw)
        identity_hash = str(entry.get("identity_sha256", ""))
        identity = identities_by_hash.get(identity_hash)
        relative = Path(str(entry.get("path", "")))
        path = (output_dir / relative).resolve()
        expected = Path("record_controls") / f"{entry.get('record_id')}.json"
        if (
            identity is None
            or identity_hash in observed
            or entry.get("record_id") != identity.record_id
            or relative.as_posix() != expected.as_posix()
            or not path.is_relative_to(output_dir)
            or not path.is_file()
            or file_sha256(path) != entry.get("file_sha256")
        ):
            raise SpectralMetricControlError(
                "spectral_metric_control_manifest_result_mismatch"
            )
        result = _validate_control_result_file(
            path=path,
            proposal_sha256=str(proposal["proposal_sha256"]),
            identity=identity,
        )
        if result.get("result_sha256") != entry.get("result_sha256"):
            raise SpectralMetricControlError(
                "spectral_metric_control_manifest_result_mismatch"
            )
        rows.append(result)
        observed.add(identity_hash)
    if observed != set(identities_by_hash):
        raise SpectralMetricControlError(
            "spectral_metric_control_manifest_identity_set_mismatch"
        )
    expected_decision = evaluate_spectral_metric_control_decision(
        rows,
        control_contract=SpectralMetricScaleControlContract(),
    )
    expected_decision["proposal_sha256"] = proposal["proposal_sha256"]
    expected_decision.pop("decision_sha256")
    expected_decision["decision_sha256"] = canonical_sha256(
        expected_decision
    )
    if decision != expected_decision:
        raise SpectralMetricControlError(
            "spectral_metric_control_decision_recomputation_mismatch"
        )
    return completion
