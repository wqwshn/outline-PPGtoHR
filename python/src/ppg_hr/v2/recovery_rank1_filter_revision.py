"""Governed rank-1-only revision of the frozen p25-short-low filter."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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


class Rank1FilterRevisionError(RuntimeError):
    """The rank-1 revision package violates its frozen contract."""


class Rank1FilterRevisionAuthorizationError(Rank1FilterRevisionError):
    """The exact rank-1 revision package has not been approved."""


_STAGE = "filter_profile_rank1_revision_diagnostic"
_ATTEMPT_KIND = "diagnostic"
_AUTHORIZATION_STATE = "authorized_rank1_filter_revision_diagnostic"
_EXPECTED_RECORD_COUNT = 12
_EXPECTED_SCENE_COUNTS = {
    "jianpan": 3,
    "kaihe": 3,
    "run": 3,
    "xiezi": 3,
}
_PROFILE = FilterProfile(
    profile_id="p25-short-low",
    design_role="core",
    fs_target=25,
    memory_ms=40,
    nominal_mu=0.008,
)
_PROPOSAL_ARTIFACT_NAMES = {
    "budget_amendment_request.json",
    "budget_contract_v10.json",
    "execution_authorization.json",
    "rank1_filter_revision_contract.json",
    "rank1_filter_revision_proposal.json",
    "source_identity.json",
    "spectral_gate_contract.json",
}


@dataclass(frozen=True)
class Rank1FilterRevisionContract:
    """One fixed, non-searchable rank-1 revision."""

    revision_id: str = "p25-short-low-rank1-v1"
    base_profile_id: str = "p25-short-low"
    reference_groups_order: tuple[str, ...] = ("HF",)
    adaptive_reference_stage_limit: int = 1
    fs_target: int = 25
    memory_ms: int = 40
    actual_taps: int = 1
    nominal_mu: float = 0.008
    lms_mu_min: float = 1e-6
    parameter_search: bool = False
    contract_version: str = "lyx_rank1_filter_revision_v1"

    def __post_init__(self) -> None:
        if (
            self.revision_id != "p25-short-low-rank1-v1"
            or self.base_profile_id != _PROFILE.profile_id
            or self.reference_groups_order != ("HF",)
            or self.adaptive_reference_stage_limit != 1
            or self.fs_target != _PROFILE.fs_target
            or self.memory_ms != _PROFILE.memory_ms
            or self.actual_taps != _PROFILE.actual_taps
            or self.nominal_mu != float(_PROFILE.nominal_mu)
            or self.lms_mu_min != 1e-6
            or self.parameter_search is not False
        ):
            raise ValueError("invalid_rank1_filter_revision_contract")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reference_groups_order"] = list(
            self.reference_groups_order
        )
        payload["selection_rule"] = (
            "first_reference_after_existing_absolute_correlation_ranking"
        )
        return payload

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Rank1FilterRevisionError(f"{name}_must_be_mapping")
    return value


def _require_list(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise Rank1FilterRevisionError(f"{name}_must_be_list")
    return value


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    artifact_name: str,
) -> str:
    value = payload.get(hash_field)
    if not isinstance(value, str):
        raise Rank1FilterRevisionError(
            f"{artifact_name}_{hash_field}_missing"
        )
    try:
        require_sha256(hash_field, value)
    except ValueError as error:
        raise Rank1FilterRevisionError(
            f"{artifact_name}_{hash_field}_invalid"
        ) from error
    unsigned = {
        key: item for key, item in payload.items() if key != hash_field
    }
    if canonical_sha256(unsigned) != value:
        raise Rank1FilterRevisionError(
            f"{artifact_name}_{hash_field}_mismatch"
        )
    return value


def _profile_payload() -> dict[str, Any]:
    return {
        "profile_id": _PROFILE.profile_id,
        "design_role": _PROFILE.design_role,
        "fs_target": _PROFILE.fs_target,
        "memory_ms": _PROFILE.memory_ms,
        "actual_taps": _PROFILE.actual_taps,
        "nominal_mu": float(_PROFILE.nominal_mu),
        "profile_sha256": _PROFILE.sha256,
    }


def _validate_upstream(
    *,
    mechanism_proposal: Mapping[str, Any],
    mechanism_completion: Mapping[str, Any],
    mechanism_decision: Mapping[str, Any],
    mechanism_manifest: Mapping[str, Any],
    mechanism_results: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[dict[str, Any], ...]]:
    proposal_sha = _verify_embedded_hash(
        mechanism_proposal,
        hash_field="proposal_sha256",
        artifact_name="rank1_revision_source_proposal",
    )
    completion_sha = _verify_embedded_hash(
        mechanism_completion,
        hash_field="completion_sha256",
        artifact_name="rank1_revision_source_completion",
    )
    decision_sha = _verify_embedded_hash(
        mechanism_decision,
        hash_field="decision_sha256",
        artifact_name="rank1_revision_source_decision",
    )
    manifest_sha = _verify_embedded_hash(
        mechanism_manifest,
        hash_field="manifest_sha256",
        artifact_name="rank1_revision_source_manifest",
    )
    expected_counts = {
        "raw_bypass": 12,
        "two_stage_zero_update": 12,
        "rank1_only_adaptive": 12,
        "rank2_only_adaptive": 12,
        "ranked_cascade_adaptive": 10,
        "reverse_cascade_adaptive": 10,
    }
    if (
        mechanism_completion.get("proposal_sha256") != proposal_sha
        or mechanism_completion.get("status")
        != "rank1_single_stage_mechanism_candidate"
        or mechanism_completion.get("diagnostic_result_count") != 12
        or mechanism_decision.get("proposal_sha256") != proposal_sha
        or mechanism_decision.get("decision")
        != "rank1_single_stage_mechanism_candidate"
        or mechanism_decision.get("record_count") != 12
        or mechanism_decision.get("lane_complete_pass_counts")
        != expected_counts
        or mechanism_manifest.get("proposal_sha256") != proposal_sha
        or mechanism_manifest.get("result_count") != 12
    ):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_source_state_invalid"
        )
    identities = _require_list(
        "rank1_revision_source_identities",
        mechanism_proposal.get("identities"),
    )
    entries = _require_list(
        "rank1_revision_source_results",
        mechanism_manifest.get("results"),
    )
    identities_by_record = {
        str(item["record_id"]): _require_mapping(
            "rank1_revision_source_identity",
            item,
        )
        for item in identities
        if isinstance(item, Mapping)
    }
    entries_by_record = {
        str(item["record_id"]): _require_mapping(
            "rank1_revision_source_manifest_entry",
            item,
        )
        for item in entries
        if isinstance(item, Mapping)
    }
    if (
        len(identities) != 12
        or len(identities_by_record) != 12
        or len(entries) != 12
        or len(entries_by_record) != 12
        or set(mechanism_results) != set(identities_by_record)
        or set(entries_by_record) != set(identities_by_record)
    ):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_source_panel_invalid"
        )
    source_panel: list[dict[str, Any]] = []
    for record_id in sorted(identities_by_record):
        identity = identities_by_record[record_id]
        entry = entries_by_record[record_id]
        result = _require_mapping(
            "rank1_revision_source_result",
            mechanism_results[record_id],
        )
        result_sha = _verify_embedded_hash(
            result,
            hash_field="result_sha256",
            artifact_name="rank1_revision_source_result",
        )
        lanes = _require_mapping(
            "rank1_revision_source_lanes",
            result.get("lanes"),
        )
        rank1 = _require_mapping(
            "rank1_revision_source_rank1_lane",
            lanes.get("rank1_only_adaptive"),
        )
        cascade = _require_mapping(
            "rank1_revision_source_cascade_lane",
            lanes.get("ranked_cascade_adaptive"),
        )
        if (
            result.get("proposal_sha256") != proposal_sha
            or result.get("record_id") != record_id
            or entry.get("identity_sha256")
            != result.get("identity_sha256")
            or entry.get("result_sha256") != result_sha
            or rank1.get("spectral_gate_pass") is not True
        ):
            raise Rank1FilterRevisionError(
                "rank1_filter_revision_source_result_invalid:"
                + record_id
            )
        source_panel.append(
            {
                "record_id": record_id,
                "scene": identity["scene"],
                "source_identity_sha256": entry["identity_sha256"],
                "source_result_sha256": result_sha,
                "source_result_file_sha256": entry["file_sha256"],
                "source_result_path": entry["path"],
                "expected_rank1_lane_sha256": canonical_sha256(rank1),
                "source_ranked_cascade_spectral_gate_pass": bool(
                    cascade.get("spectral_gate_pass")
                ),
            }
        )
    scene_counts = {
        scene: sum(item["scene"] == scene for item in source_panel)
        for scene in _EXPECTED_SCENE_COUNTS
    }
    if scene_counts != _EXPECTED_SCENE_COUNTS:
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_source_scene_coverage_invalid"
        )
    upstream = {
        "proposal_sha256": proposal_sha,
        "completion_sha256": completion_sha,
        "decision_sha256": decision_sha,
        "manifest_sha256": manifest_sha,
        "status": "rank1_single_stage_mechanism_candidate",
    }
    return (
        tuple(identities_by_record[item["record_id"]] for item in source_panel),
        tuple([{**item, "upstream": upstream} for item in source_panel]),
    )


def build_rank1_filter_revision_proposal(
    *,
    mechanism_proposal: Mapping[str, Any],
    mechanism_completion: Mapping[str, Any],
    mechanism_decision: Mapping[str, Any],
    mechanism_manifest: Mapping[str, Any],
    mechanism_results: Mapping[str, Mapping[str, Any]],
    parent_experiment_id: str,
    solver_hash: str,
    evaluation_hash: str,
    source_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the exact 12-identity, zero-run rank-1 revision proposal."""

    if not parent_experiment_id:
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_parent_experiment_id_empty"
        )
    require_sha256("solver_hash", solver_hash)
    require_sha256("evaluation_hash", evaluation_hash)
    source_identities, source_panel = _validate_upstream(
        mechanism_proposal=mechanism_proposal,
        mechanism_completion=mechanism_completion,
        mechanism_decision=mechanism_decision,
        mechanism_manifest=mechanism_manifest,
        mechanism_results=mechanism_results,
    )
    contract = Rank1FilterRevisionContract()
    spectral = StageRSpectralGateContract()
    budget = BudgetContract.proposed_v10_rank1_filter_revision()
    source_by_record = {
        str(item["record_id"]): item for item in source_panel
    }
    config = {
        "execution_mode": "rank1_filter_revision_spectral_audit",
        "revision_contract_sha256": contract.sha256,
        "spectral_gate_contract_sha256": spectral.sha256,
        "adaptive_reference_stage_limit": 1,
        "profile": _profile_payload(),
        "parameter_search": False,
    }
    identities: list[dict[str, Any]] = []
    for source in source_identities:
        record_id = str(source["record_id"])
        panel = source_by_record[record_id]
        identity = AttemptIdentity(
            solver_hash=solver_hash,
            config_hash=canonical_sha256(config),
            metric_contract_hash=contract.sha256,
            evaluation_hash=evaluation_hash,
            data_sha256=str(source["data_sha256"]),
            record_id=record_id,
            stage=_STAGE,
            attempt_kind=_ATTEMPT_KIND,
            parent_experiment_id=parent_experiment_id,
        )
        identities.append(
            {
                **identity.to_dict(),
                "scene": source["scene"],
                "data_path": source["data_path"],
                "reference_path": source["reference_path"],
                "raw_data_sha256": source["raw_data_sha256"],
                "reference_sha256": source["reference_sha256"],
                "config": config,
                "source_mechanism_identity_sha256": panel[
                    "source_identity_sha256"
                ],
                "source_mechanism_result_sha256": panel[
                    "source_result_sha256"
                ],
                "source_mechanism_result_file_sha256": panel[
                    "source_result_file_sha256"
                ],
                "source_mechanism_result_path": panel[
                    "source_result_path"
                ],
                "expected_rank1_lane_sha256": panel[
                    "expected_rank1_lane_sha256"
                ],
                "source_ranked_cascade_spectral_gate_pass": panel[
                    "source_ranked_cascade_spectral_gate_pass"
                ],
            }
        )
    hashes = [str(item["identity_sha256"]) for item in identities]
    if len(set(hashes)) != 12:
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_identity_collision"
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
    compact_source_panel = [
        {
            key: value
            for key, value in panel.items()
            if key != "upstream"
        }
        for panel in source_panel
    ]
    proposal: dict[str, Any] = {
        "proposal_version": "lyx_rank1_filter_revision_proposal_v1",
        "status": "authorized_scope_frozen_zero_runs",
        "authorization_state": _AUTHORIZATION_STATE,
        "parent_experiment_id": parent_experiment_id,
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "unique_budget": 12,
        "retry_limit": 1,
        "worst_case_attempt_budget": 24,
        "diagnostic_run_count": 0,
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "may_nominate_recovery_candidate": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "upstream_filter_mechanism_decomposition": source_panel[0][
            "upstream"
        ],
        "frozen_contracts": {
            "solver_hash": solver_hash,
            "evaluation_hash": evaluation_hash,
            "revision_contract_hash": contract.sha256,
            "spectral_gate_contract_hash": spectral.sha256,
            "budget_contract_hash": budget.sha256,
            "profile_hash": _PROFILE.sha256,
        },
        "revision_contract": contract.to_dict(),
        "profile": _profile_payload(),
        "record_panel": record_panel,
        "record_panel_sha256": canonical_sha256(record_panel),
        "source_result_panel": compact_source_panel,
        "source_result_panel_sha256": canonical_sha256(
            compact_source_panel
        ),
        "identity_sha256": hashes,
        "identity_panel_sha256": canonical_sha256(hashes),
        "identities": identities,
    }
    if source_artifacts is not None:
        proposal["source_artifacts"] = {
            str(name): dict(value)
            for name, value in sorted(source_artifacts.items())
        }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def validate_rank1_filter_revision_authorization(
    proposal: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require an exact approval while keeping Stage R outside scope."""

    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="rank1_filter_revision_proposal",
    )
    if receipt is None or receipt.get("approved") is not True:
        raise Rank1FilterRevisionAuthorizationError(
            "rank1_filter_revision_execution_authorization_required"
        )
    frozen = _require_mapping(
        "rank1_filter_revision_frozen_contracts",
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
        "source_result_panel_sha256": proposal.get(
            "source_result_panel_sha256"
        ),
        "solver_hash": frozen.get("solver_hash"),
        "evaluation_hash": frozen.get("evaluation_hash"),
        "revision_contract_hash": frozen.get("revision_contract_hash"),
        "spectral_gate_contract_hash": frozen.get(
            "spectral_gate_contract_hash"
        ),
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
        raise Rank1FilterRevisionAuthorizationError(
            "rank1_filter_revision_authorization_mismatch:"
            + ",".join(mismatched)
        )
    for name in ("approved_at", "approved_by"):
        if not isinstance(receipt.get(name), str) or not receipt[name]:
            raise Rank1FilterRevisionAuthorizationError(
                f"rank1_filter_revision_authorization_{name}_invalid"
            )
    return dict(receipt)


def evaluate_rank1_filter_revision_decision(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless all 12 results exactly reproduce rank-1."""

    record_ids = [str(row.get("record_id", "")) for row in rows]
    exact_count = sum(
        row.get("exact_rank1_reproduction_pass") is True for row in rows
    )
    complete_count = sum(
        row.get("spectral_gate_pass") is True
        and row.get("all_gate_pass") is True
        for row in rows
    )
    stage_count = sum(
        row.get("single_reference_stage_per_valid_window") is True
        for row in rows
    )
    cascade_pass_count = sum(
        row.get("source_ranked_cascade_spectral_gate_pass") is True
        for row in rows
    )
    recovered_count = sum(
        row.get("source_ranked_cascade_spectral_gate_pass") is False
        and row.get("spectral_gate_pass") is True
        for row in rows
    )
    valid = (
        len(rows) == 12
        and len(set(record_ids)) == 12
        and all(record_ids)
        and exact_count == 12
        and complete_count == 12
        and stage_count == 12
        and cascade_pass_count == 10
        and recovered_count == 2
    )
    decision = (
        "rank1_filter_revision_validated"
        if valid
        else "rank1_revision_reproduction_invalid"
    )
    payload: dict[str, Any] = {
        "decision_version": "lyx_rank1_filter_revision_decision_v1",
        "decision": decision,
        "next_state": (
            "awaiting_stage_r_replan_human_review"
            if valid
            else "rank1_filter_revision_requires_human_review"
        ),
        "record_count": len(rows),
        "exact_rank1_reproduction_count": exact_count,
        "complete_gate_pass_count": complete_count,
        "single_reference_stage_count": stage_count,
        "source_ranked_cascade_pass_count": cascade_pass_count,
        "rank1_recovered_from_cascade_failure_count": recovered_count,
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": False,
    }
    payload["decision_sha256"] = canonical_sha256(payload)
    return payload


def _repository_root_from_source_root(source_root: Path) -> Path:
    source_root = Path(source_root).resolve()
    if source_root.name != "src" or source_root.parent.name != "python":
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_source_root_invalid"
        )
    return source_root.parent.parent


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
            raise Rank1FilterRevisionError(
                "rank1_filter_revision_source_outside_repository:" + name
            ) from error
        payload[name] = {
            "path": relative.as_posix(),
            "path_base": "repository_root",
            "file_sha256": file_sha256(resolved),
        }
    return payload


def _authorization_for_proposal(
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = _require_mapping(
        "rank1_filter_revision_frozen_contracts",
        proposal["frozen_contracts"],
    )
    return {
        "approved": True,
        "decision_state": _AUTHORIZATION_STATE,
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_contract_hash": frozen["budget_contract_hash"],
        "unique_budget": 12,
        "stage": _STAGE,
        "identity_panel_sha256": proposal["identity_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "source_result_panel_sha256": proposal[
            "source_result_panel_sha256"
        ],
        "solver_hash": frozen["solver_hash"],
        "evaluation_hash": frozen["evaluation_hash"],
        "revision_contract_hash": frozen["revision_contract_hash"],
        "spectral_gate_contract_hash": frozen[
            "spectral_gate_contract_hash"
        ],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": False,
        "approved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "approved_by": "user",
        "approval_scope_quote": (
            "我同意进行A：撰写滤波机制分解的零运行 spec/proposal，"
            "并且允许之后可以推进执行这个子实验，"
            "基于子实验的结论继续完成这一轮大实验。"
        ),
    }


def propose_rank1_filter_revision(
    *,
    mechanism_proposal_path: Path,
    mechanism_completion_path: Path,
    mechanism_decision_path: Path,
    mechanism_manifest_path: Path,
    source_budget_contract_path: Path,
    spectral_gate_contract_path: Path,
    spec_path: Path,
    output_dir: Path,
    source_root: Path,
    parent_experiment_id: str,
) -> dict[str, Any]:
    """Publish the authorized, content-addressed zero-run package."""

    artifacts = {
        "mechanism_proposal": Path(mechanism_proposal_path).resolve(),
        "mechanism_completion": Path(
            mechanism_completion_path
        ).resolve(),
        "mechanism_decision": Path(mechanism_decision_path).resolve(),
        "mechanism_manifest": Path(mechanism_manifest_path).resolve(),
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
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_source_missing:" + ",".join(missing)
        )
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_destination_exists"
        )
    source_budget = BudgetContract.proposed_v9_filter_mechanism_decomposition()
    if read_json(artifacts["source_budget_contract"]) != source_budget.to_dict():
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_source_budget_mismatch"
        )
    spectral_payload = read_json(artifacts["spectral_gate_contract"])
    if (
        _verify_embedded_hash(
            spectral_payload,
            hash_field="contract_sha256",
            artifact_name="rank1_filter_revision_spectral_contract",
        )
        != StageRSpectralGateContract().sha256
    ):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_spectral_contract_mismatch"
        )
    manifest = read_json(artifacts["mechanism_manifest"])
    entries = _require_list(
        "rank1_filter_revision_manifest_results",
        manifest.get("results"),
    )
    source_results: dict[str, Mapping[str, Any]] = {}
    for raw in entries:
        entry = _require_mapping(
            "rank1_filter_revision_manifest_entry",
            raw,
        )
        record_id = str(entry["record_id"])
        path = (
            artifacts["mechanism_manifest"].parent / str(entry["path"])
        ).resolve()
        if not path.is_file() or file_sha256(path) != entry.get(
            "file_sha256"
        ):
            raise Rank1FilterRevisionError(
                "rank1_filter_revision_source_result_file_mismatch:"
                + record_id
            )
        artifacts[f"source_result:{record_id}"] = path
        source_results[record_id] = read_json(path)
    source_root = Path(source_root).resolve()
    repository_root = _repository_root_from_source_root(source_root)
    source_artifacts = _source_artifact_payload(
        artifacts,
        repository_root=repository_root,
    )
    source_identity = runtime_source_identity(
        source_root,
        root_modules=(
            "ppg_hr.v2.recovery_rank1_filter_revision",
            "ppg_hr.v2.recovery_rank1_filter_revision_runner",
            "ppg_hr.v2.solver",
        ),
    )
    bundle_hash = str(source_identity["source_bundle_sha256"])
    proposal = build_rank1_filter_revision_proposal(
        mechanism_proposal=read_json(artifacts["mechanism_proposal"]),
        mechanism_completion=read_json(artifacts["mechanism_completion"]),
        mechanism_decision=read_json(artifacts["mechanism_decision"]),
        mechanism_manifest=manifest,
        mechanism_results=source_results,
        parent_experiment_id=parent_experiment_id,
        solver_hash=bundle_hash,
        evaluation_hash=bundle_hash,
        source_artifacts=source_artifacts,
    )
    target_budget = BudgetContract.proposed_v10_rank1_filter_revision()
    contract = Rank1FilterRevisionContract()
    request: dict[str, Any] = {
        "request_version": "lyx_rank1_filter_revision_budget_request_v1",
        "status": "approved_by_scope_bound_user_authorization",
        "approved": True,
        "decision_state": _AUTHORIZATION_STATE,
        "proposal_sha256": proposal["proposal_sha256"],
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "added_unique_identities": 12,
        "normal_unique_identity_limit": 852,
        "max_unique_identities": 864,
        "max_attempts": 1728,
        "retry_limit": 1,
        "budget_contract_hash": target_budget.sha256,
        "identity_panel_sha256": proposal["identity_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "source_result_panel_sha256": proposal[
            "source_result_panel_sha256"
        ],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": False,
    }
    request["request_sha256"] = canonical_sha256(request)
    authorization = _authorization_for_proposal(proposal)
    validate_rank1_filter_revision_authorization(
        proposal,
        receipt=authorization,
    )
    receipt: dict[str, Any] = {
        "receipt_version": "lyx_rank1_filter_revision_proposal_receipt_v1",
        "status": "authorized_scope_frozen_zero_runs",
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_request_sha256": request["request_sha256"],
        "identity_count": 12,
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
            staging / "rank1_filter_revision_proposal.json",
            proposal,
        )
        atomic_write_json(
            staging / "rank1_filter_revision_contract.json",
            {
                **contract.to_dict(),
                "contract_sha256": contract.sha256,
            },
        )
        atomic_write_json(
            staging / "spectral_gate_contract.json",
            spectral_payload,
        )
        atomic_write_json(
            staging / "budget_contract_v10.json",
            target_budget.to_dict(),
        )
        atomic_write_json(
            staging / "budget_amendment_request.json",
            request,
        )
        atomic_write_json(staging / "source_identity.json", source_identity)
        atomic_write_json(
            staging / "execution_authorization.json",
            authorization,
        )
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


def _validate_proposal_preflight(
    *,
    proposal_dir: Path,
    source_root: Path,
) -> tuple[
    dict[str, Any],
    tuple[dict[str, Any], ...],
    dict[str, Path],
]:
    proposal_dir = Path(proposal_dir).resolve()
    proposal = read_json(proposal_dir / "rank1_filter_revision_proposal.json")
    proposal_sha = _verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="rank1_filter_revision_proposal",
    )
    receipt = read_json(proposal_dir / "proposal_receipt.json")
    _verify_embedded_hash(
        receipt,
        hash_field="receipt_sha256",
        artifact_name="rank1_filter_revision_proposal_receipt",
    )
    artifact_hashes = _require_mapping(
        "rank1_filter_revision_artifact_hashes",
        receipt.get("artifact_sha256"),
    )
    if (
        receipt.get("proposal_sha256") != proposal_sha
        or set(artifact_hashes) != _PROPOSAL_ARTIFACT_NAMES
    ):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_proposal_receipt_mismatch"
        )
    for name, expected in artifact_hashes.items():
        path = proposal_dir / str(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise Rank1FilterRevisionError(
                "rank1_filter_revision_proposal_artifact_mismatch:"
                + str(name)
            )
    source_root = Path(source_root).resolve()
    repository_root = _repository_root_from_source_root(source_root)
    source_artifacts = _require_mapping(
        "rank1_filter_revision_source_artifacts",
        proposal.get("source_artifacts"),
    )
    resolved: dict[str, Path] = {}
    for name, raw in source_artifacts.items():
        artifact = _require_mapping(
            f"rank1_filter_revision_source_artifact:{name}",
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
            raise Rank1FilterRevisionError(
                "rank1_filter_revision_source_artifact_mismatch:"
                + str(name)
            )
        resolved[str(name)] = path
    identity_items = tuple(
        dict(
            _require_mapping(
                "rank1_filter_revision_identity",
                item,
            )
        )
        for item in _require_list(
            "rank1_filter_revision_identities",
            proposal.get("identities"),
        )
    )
    expected_keys = {
        "mechanism_proposal",
        "mechanism_completion",
        "mechanism_decision",
        "mechanism_manifest",
        "source_budget_contract",
        "spectral_gate_contract",
        "experiment_spec",
        *(
            f"source_result:{item['record_id']}"
            for item in identity_items
        ),
    }
    if set(resolved) != expected_keys:
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_source_artifact_set_mismatch"
        )
    current_identity = runtime_source_identity(
        source_root,
        root_modules=(
            "ppg_hr.v2.recovery_rank1_filter_revision",
            "ppg_hr.v2.recovery_rank1_filter_revision_runner",
            "ppg_hr.v2.solver",
        ),
    )
    if read_json(proposal_dir / "source_identity.json") != current_identity:
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_source_identity_mismatch"
        )
    contract = Rank1FilterRevisionContract()
    contract_artifact = read_json(
        proposal_dir / "rank1_filter_revision_contract.json"
    )
    contract_sha = _verify_embedded_hash(
        contract_artifact,
        hash_field="contract_sha256",
        artifact_name="rank1_filter_revision_contract",
    )
    spectral = StageRSpectralGateContract()
    spectral_artifact = read_json(
        proposal_dir / "spectral_gate_contract.json"
    )
    spectral_sha = _verify_embedded_hash(
        spectral_artifact,
        hash_field="contract_sha256",
        artifact_name="rank1_filter_revision_spectral_contract",
    )
    target_budget = BudgetContract.proposed_v10_rank1_filter_revision()
    if (
        contract_sha != contract.sha256
        or {
            key: value
            for key, value in contract_artifact.items()
            if key != "contract_sha256"
        }
        != contract.to_dict()
        or spectral_sha != spectral.sha256
        or read_json(proposal_dir / "budget_contract_v10.json")
        != target_budget.to_dict()
    ):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_contract_mismatch"
        )
    mechanism_manifest = read_json(resolved["mechanism_manifest"])
    mechanism_results = {
        str(item["record_id"]): read_json(
            resolved[f"source_result:{item['record_id']}"]
        )
        for item in identity_items
    }
    frozen = _require_mapping(
        "rank1_filter_revision_frozen_contracts",
        proposal["frozen_contracts"],
    )
    rebuilt = build_rank1_filter_revision_proposal(
        mechanism_proposal=read_json(resolved["mechanism_proposal"]),
        mechanism_completion=read_json(resolved["mechanism_completion"]),
        mechanism_decision=read_json(resolved["mechanism_decision"]),
        mechanism_manifest=mechanism_manifest,
        mechanism_results=mechanism_results,
        parent_experiment_id=str(proposal["parent_experiment_id"]),
        solver_hash=str(frozen["solver_hash"]),
        evaluation_hash=str(frozen["evaluation_hash"]),
        source_artifacts={
            str(name): dict(
                _require_mapping("rank1_source_artifact", raw)
            )
            for name, raw in source_artifacts.items()
        },
    )
    if rebuilt != proposal:
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_proposal_rebuild_mismatch"
        )
    request = read_json(proposal_dir / "budget_amendment_request.json")
    _verify_embedded_hash(
        request,
        hash_field="request_sha256",
        artifact_name="rank1_filter_revision_budget_request",
    )
    expected_request = {
        "proposal_sha256": proposal_sha,
        "stage": _STAGE,
        "attempt_kind": _ATTEMPT_KIND,
        "added_unique_identities": 12,
        "normal_unique_identity_limit": 852,
        "max_unique_identities": 864,
        "max_attempts": 1728,
        "retry_limit": 1,
        "budget_contract_hash": target_budget.sha256,
        "identity_panel_sha256": proposal["identity_panel_sha256"],
        "record_panel_sha256": proposal["record_panel_sha256"],
        "source_result_panel_sha256": proposal[
            "source_result_panel_sha256"
        ],
        "parameter_search_authorized": False,
        "independent_bo_authorized": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "may_nominate_recovery_candidate": False,
    }
    mismatched = sorted(
        name
        for name, value in expected_request.items()
        if request.get(name) != value
    )
    if mismatched:
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_budget_request_mismatch:"
            + ",".join(mismatched)
        )
    authorization = read_json(
        proposal_dir / "execution_authorization.json"
    )
    validate_rank1_filter_revision_authorization(
        proposal,
        receipt=authorization,
    )
    identities = tuple(_identity_from_item(item) for item in identity_items)
    hashes = [identity.sha256 for identity in identities]
    if (
        len(identities) != 12
        or len(set(hashes)) != 12
        or hashes != proposal.get("identity_sha256")
        or canonical_sha256(hashes) != proposal.get(
            "identity_panel_sha256"
        )
    ):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_identity_matrix_mismatch"
        )
    return proposal, identity_items, resolved


def prepare_rank1_filter_revision_governance(
    *,
    proposal_dir: Path,
    source_governance_dir: Path,
    governance_dir: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Migrate v9 and register only the 12 approved diagnostics."""

    proposal, identity_items, _resolved = _validate_proposal_preflight(
        proposal_dir=proposal_dir,
        source_root=source_root,
    )
    authorization = validate_rank1_filter_revision_authorization(
        proposal,
        receipt=read_json(
            Path(proposal_dir).resolve()
            / "execution_authorization.json"
        ),
    )
    target_root = Path(governance_dir).resolve()
    if target_root.exists():
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_governance_exists"
        )
    source_dir = Path(source_governance_dir).resolve()
    source_budget = (
        BudgetContract.proposed_v9_filter_mechanism_decomposition()
    )
    if (
        read_json(source_dir / "budget_contract.json")
        != source_budget.to_dict()
    ):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_source_governance_budget_mismatch"
        )
    exploration_payload = read_json(
        source_dir / "exploration_registry.json"
    )
    exploration = _exploration_from_payload(exploration_payload)
    if exploration.to_dict() != exploration_payload:
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_exploration_registry_mismatch"
        )
    registry = AttemptRegistry.open(
        source_dir / "attempt_registry.json",
        budget_contract=source_budget,
        exploration_registry=exploration,
    )
    target_budget = BudgetContract.proposed_v10_rank1_filter_revision()
    frozen = _require_mapping(
        "rank1_filter_revision_frozen_contracts",
        proposal["frozen_contracts"],
    )
    if target_budget.sha256 != frozen.get("budget_contract_hash"):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_target_budget_mismatch"
        )
    identities = tuple(_identity_from_item(item) for item in identity_items)
    amendment = BudgetAmendmentRequest(
        stage=_STAGE,
        profile_design_rule_hash=str(
            frozen["revision_contract_hash"]
        ),
        record_manifest_hash=str(proposal["record_panel_sha256"]),
        added_unique_identities=12,
        normal_unique_identity_limit=852,
        max_unique_identities=864,
        max_attempts=1728,
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
                "lyx_rank1_filter_revision_governance_v1"
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
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_result_missing:" + identity.record_id
        )
    payload = read_json(path)
    _verify_embedded_hash(
        payload,
        hash_field="result_sha256",
        artifact_name="rank1_filter_revision_result",
    )
    if (
        payload.get("proposal_sha256") != proposal_sha256
        or payload.get("identity_sha256") != identity.sha256
        or payload.get("record_id") != identity.record_id
        or payload.get("revision_contract_sha256")
        != Rank1FilterRevisionContract().sha256
        or payload.get("spectral_gate_contract_sha256")
        != StageRSpectralGateContract().sha256
        or payload.get("reference_stage_limit") != 1
    ):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_result_identity_mismatch:"
            + identity.record_id
        )
    return payload


def _run_rank1_record(
    *,
    item: Mapping[str, Any],
    identity: AttemptIdentity,
    proposal_sha256: str,
    source_result_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    if (
        file_sha256(source_result_path)
        != item["source_mechanism_result_file_sha256"]
    ):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_source_result_file_drift:"
            + identity.record_id
        )
    source_result = read_json(source_result_path)
    source_result_sha = _verify_embedded_hash(
        source_result,
        hash_field="result_sha256",
        artifact_name="rank1_filter_revision_source_result",
    )
    source_lanes = _require_mapping(
        "rank1_filter_revision_source_lanes",
        source_result.get("lanes"),
    )
    expected_rank1 = dict(
        _require_mapping(
            "rank1_filter_revision_expected_rank1_lane",
            source_lanes.get("rank1_only_adaptive"),
        )
    )
    source_cascade = _require_mapping(
        "rank1_filter_revision_source_cascade_lane",
        source_lanes.get("ranked_cascade_adaptive"),
    )
    if (
        source_result_sha != item["source_mechanism_result_sha256"]
        or canonical_sha256(expected_rank1)
        != item["expected_rank1_lane_sha256"]
    ):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_source_result_drift:"
            + identity.record_id
        )
    record = FilterAuditRecord(
        record_id=identity.record_id,
        scene=str(item["scene"]),
        data_path=str(item["data_path"]),
        reference_path=str(item["reference_path"]),
        data_sha256=str(item["raw_data_sha256"]),
        reference_sha256=str(item["reference_sha256"]),
    )
    audit = audit_stage_r_profile_record(
        _PROFILE,
        record,
        contract=StageRSpectralGateContract(),
        reference_stage_limit=1,
    )
    observed_rank1 = dict(
        _require_mapping(
            "rank1_filter_revision_observed_rank1_lane",
            audit.get("stage_r_spectral_gate"),
        )
    )
    exact = observed_rank1 == expected_rank1
    valid_window_count = int(observed_rank1["valid_window_count"])
    single_stage = (
        audit.get("reference_stage_limit") == 1
        and audit.get("lms_stage_count") == valid_window_count
    )
    all_gate_pass = bool(
        audit.get("stability_pass") is True
        and observed_rank1.get("spectral_gate_pass") is True
        and all(
            value is True
            for value in _require_mapping(
                "rank1_filter_revision_observed_gates",
                observed_rank1.get("gates"),
            ).values()
        )
    )
    payload: dict[str, Any] = {
        "result_version": "lyx_rank1_filter_revision_result_v1",
        "proposal_sha256": proposal_sha256,
        "identity_sha256": identity.sha256,
        "record_id": identity.record_id,
        "scene": item["scene"],
        "profile_id": _PROFILE.profile_id,
        "profile_sha256": _PROFILE.sha256,
        "revision_contract_sha256": Rank1FilterRevisionContract().sha256,
        "spectral_gate_contract_sha256": (
            StageRSpectralGateContract().sha256
        ),
        "reference_stage_limit": 1,
        "source_mechanism_result_sha256": source_result_sha,
        "source_expected_rank1_lane_sha256": canonical_sha256(
            expected_rank1
        ),
        "observed_rank1_lane_sha256": canonical_sha256(observed_rank1),
        "exact_rank1_reproduction_pass": exact,
        "single_reference_stage_per_valid_window": single_stage,
        "lms_stage_count": audit["lms_stage_count"],
        "valid_window_count": valid_window_count,
        "stability_pass": audit["stability_pass"],
        "spectral_gate_pass": observed_rank1["spectral_gate_pass"],
        "all_gate_pass": all_gate_pass,
        "source_ranked_cascade_spectral_gate_pass": bool(
            source_cascade.get("spectral_gate_pass")
        ),
        "rank1_spectral_gate": observed_rank1,
        "stability_summary": {
            key: value
            for key, value in audit.items()
            if key
            not in {
                "runtime_seconds",
                "stage_r_spectral_gate",
            }
        },
    }
    payload["result_sha256"] = canonical_sha256(payload)
    atomic_write_json(result_path, payload)
    return payload


def execute_rank1_filter_revision(
    *,
    proposal_dir: Path,
    governance_dir: Path,
    output_dir: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Execute only the approved 12 exact-reproduction diagnostics."""

    proposal, identity_items, resolved = _validate_proposal_preflight(
        proposal_dir=proposal_dir,
        source_root=source_root,
    )
    governance_root = Path(governance_dir).resolve()
    authorization = validate_rank1_filter_revision_authorization(
        proposal,
        receipt=read_json(
            governance_root / "execution_authorization.json"
        ),
    )
    budget = BudgetContract.proposed_v10_rank1_filter_revision()
    if (
        read_json(governance_root / "budget_contract.json")
        != budget.to_dict()
    ):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_execution_budget_mismatch"
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
    entries: list[dict[str, Any]] = []
    results_dir = destination / "record_rank1_revision_audits"
    for item, identity in zip(identity_items, identities, strict=True):
        result_path = results_dir / f"{identity.record_id}.json"

        def run(
            *,
            _item: Mapping[str, Any] = item,
            _identity: AttemptIdentity = identity,
            _result_path: Path = result_path,
        ) -> dict[str, Any]:
            return _run_rank1_record(
                item=_item,
                identity=_identity,
                proposal_sha256=str(proposal["proposal_sha256"]),
                source_result_path=resolved[
                    f"source_result:{_identity.record_id}"
                ],
                result_path=_result_path,
            )

        try:
            registry.assert_complete_matrix((identity,))
        except GovernanceError as error:
            if str(error).startswith("matrix_identity_still_running:"):
                raise Rank1FilterRevisionError(
                    "rank1_filter_revision_interrupted_attempt_"
                    "requires_human_review:"
                    + identity.record_id
                ) from error
            prior = registry.matrix_execution_summary((identity,))
            if (
                prior["total_attempt_count"] != 0
                or prior["failed_attempt_count"] != 0
                or prior["cache_only_identity_count"] != 0
            ):
                raise Rank1FilterRevisionError(
                    "rank1_filter_revision_retry_requires_human_review:"
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
        entries.append(
            {
                "record_id": identity.record_id,
                "identity_sha256": identity.sha256,
                "path": result_path.relative_to(destination).as_posix(),
                "file_sha256": file_sha256(result_path),
                "result_sha256": row["result_sha256"],
            }
        )
    registry.assert_complete_matrix(identities)
    decision = evaluate_rank1_filter_revision_decision(rows)
    decision["proposal_sha256"] = proposal["proposal_sha256"]
    decision.pop("decision_sha256")
    decision["decision_sha256"] = canonical_sha256(decision)
    atomic_write_json(destination / "decision_receipt.json", decision)
    manifest: dict[str, Any] = {
        "manifest_version": "lyx_rank1_filter_revision_manifest_v1",
        "proposal_sha256": proposal["proposal_sha256"],
        "result_count": 12,
        "results": entries,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write_json(destination / "result_manifest.json", manifest)
    matrix = registry.matrix_execution_summary(identities)
    expected_matrix = {
        "planned_identity_count": 12,
        "identity_with_solver_attempt_count": 12,
        "cache_only_identity_count": 0,
        "total_attempt_count": 12,
        "failed_attempt_count": 0,
        "retry_count": 0,
    }
    if matrix != expected_matrix:
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_execution_summary_mismatch"
        )
    completion: dict[str, Any] = {
        "completion_version": "lyx_rank1_filter_revision_completion_v1",
        "status": decision["decision"],
        "next_state": decision["next_state"],
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": canonical_sha256(authorization),
        "diagnostic_result_count": 12,
        "diagnostic_run_count": matrix[
            "identity_with_solver_attempt_count"
        ],
        "parameter_search_run_count": 0,
        "independent_bo_run_count": 0,
        "may_nominate_recovery_candidate": False,
        "automatic_stage_r_execution": False,
        "automatic_stage_f_execution": False,
        "matrix_execution_summary": matrix,
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
        artifact_name="rank1_filter_revision_completion",
    )
    matrix = registry.matrix_execution_summary(identities)
    expected_matrix = {
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
        or completion.get("diagnostic_run_count") != 12
        or completion.get("parameter_search_run_count") != 0
        or completion.get("independent_bo_run_count") != 0
        or completion.get("may_nominate_recovery_candidate") is not False
        or completion.get("automatic_stage_r_execution") is not False
        or completion.get("automatic_stage_f_execution") is not False
        or completion.get("matrix_execution_summary") != matrix
        or matrix != expected_matrix
    ):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_completion_mismatch"
        )
    artifacts = _require_mapping(
        "rank1_filter_revision_completion_artifacts",
        completion.get("artifacts"),
    )
    for name in ("decision_receipt.json", "result_manifest.json"):
        path = output_dir / name
        if not path.is_file() or file_sha256(path) != artifacts.get(name):
            raise Rank1FilterRevisionError(
                "rank1_filter_revision_completion_artifact_mismatch:"
                + name
            )
    decision = read_json(output_dir / "decision_receipt.json")
    decision_sha = _verify_embedded_hash(
        decision,
        hash_field="decision_sha256",
        artifact_name="rank1_filter_revision_decision",
    )
    if (
        decision_sha != completion.get("decision_sha256")
        or decision.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or decision.get("decision") != completion.get("status")
        or decision.get("next_state") != completion.get("next_state")
    ):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_completion_decision_mismatch"
        )
    manifest = read_json(output_dir / "result_manifest.json")
    _verify_embedded_hash(
        manifest,
        hash_field="manifest_sha256",
        artifact_name="rank1_filter_revision_manifest",
    )
    entries = _require_list(
        "rank1_filter_revision_manifest_results",
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
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_manifest_mismatch"
        )
    rows: list[dict[str, Any]] = []
    observed: set[str] = set()
    for raw in entries:
        entry = _require_mapping(
            "rank1_filter_revision_manifest_entry",
            raw,
        )
        identity_hash = str(entry.get("identity_sha256", ""))
        identity = identities_by_hash.get(identity_hash)
        relative = Path(str(entry.get("path", "")))
        path = (output_dir / relative).resolve()
        expected = (
            Path("record_rank1_revision_audits")
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
            raise Rank1FilterRevisionError(
                "rank1_filter_revision_manifest_result_mismatch"
            )
        result = _validate_result_file(
            path=path,
            proposal_sha256=str(proposal["proposal_sha256"]),
            identity=identity,
        )
        if result.get("result_sha256") != entry.get("result_sha256"):
            raise Rank1FilterRevisionError(
                "rank1_filter_revision_manifest_result_mismatch"
            )
        rows.append(result)
        observed.add(identity_hash)
    if observed != set(identities_by_hash):
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_manifest_identity_set_mismatch"
        )
    expected_decision = evaluate_rank1_filter_revision_decision(rows)
    expected_decision["proposal_sha256"] = proposal["proposal_sha256"]
    expected_decision.pop("decision_sha256")
    expected_decision["decision_sha256"] = canonical_sha256(
        expected_decision
    )
    if decision != expected_decision:
        raise Rank1FilterRevisionError(
            "rank1_filter_revision_decision_recomputation_mismatch"
        )
    return completion
