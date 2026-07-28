"""Prepare and execute the bounded LYX filter-profile stability audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from ppg_hr.core.choose_delay import choose_delay
from ppg_hr.core.lms_filter import lms_filter

from .phase2_experiment_io import atomic_write_json, file_sha256, read_json
from .recovery_experiment_governance import (
    AttemptIdentity,
    AttemptRegistry,
    BudgetAmendmentRequest,
    BudgetContract,
    CacheEvidence,
    ExplorationRegistry,
    validate_budget_amendment_authorization,
)
from .recovery_filter_profiles import (
    ArchivedProfileEvidence,
    FilterProfile,
    freeze_filter_profile_library,
)
from .recovery_filter_stability import (
    FilterAuditRecord,
    StabilityAuditContract,
    StabilityAuditError,
    audit_lms_stage,
    build_filter_profile_receipt,
    plan_filter_audit_identities,
    plan_replacement_filter_audit_identities,
    plan_spec_gate_supplement_identities,
    reclassify_cached_record_audit,
    summarize_record_audit,
)
from .signal_preparation import prepare_v2_signals
from .types import V2RunConfig

_PARENT_EXPERIMENT_ID = "lyx_recovery_filter_profile_v1"
_EXPECTED_SCENES = ("jianpan", "kaihe", "run", "xiezi")
_PREPARATION_MARKER = "preparation_transaction.json"


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _verify_embedded_sha256(payload: dict[str, Any], field: str) -> None:
    declared = payload.get(field)
    unsigned = {key: value for key, value in payload.items() if key != field}
    if declared != _canonical_sha256(unsigned):
        raise StabilityAuditError(f"embedded_sha256_mismatch:{field}")


def _validate_spec_gate_source_artifacts(
    *,
    source_plan: dict[str, Any],
    source_completion: dict[str, Any],
    source_record_manifest: dict[str, Any],
    proposal: dict[str, Any],
    design_rule: dict[str, Any],
) -> None:
    """Bind the approved supplement to the exact source artifacts."""

    _verify_embedded_sha256(source_plan, "plan_sha256")
    _verify_embedded_sha256(source_completion, "completion_sha256")
    _verify_embedded_sha256(
        source_record_manifest,
        "record_manifest_sha256",
    )
    if (
        source_completion.get("completion_sha256")
        != proposal.get("source_completion_sha256")
        or source_completion.get("status")
        != "blocked_insufficient_eligible_profiles"
        or source_record_manifest.get("record_manifest_sha256")
        != design_rule.get("record_manifest_sha256")
    ):
        raise StabilityAuditError("spec_gate_supplement_source_state_mismatch")


def _validate_frozen_spec_gate_sources(
    *,
    plan: dict[str, Any],
    proposal: dict[str, Any],
    authorization: dict[str, Any],
    design_rule: dict[str, Any],
    candidate_evidence: dict[str, Any],
    source_plan: dict[str, Any],
    source_completion: dict[str, Any],
    source_evidence: dict[str, Any],
    source_reclassification: dict[str, Any],
) -> None:
    _verify_embedded_sha256(plan, "plan_sha256")
    _verify_embedded_sha256(proposal, "proposal_sha256")
    _verify_embedded_sha256(design_rule, "design_rule_sha256")
    _verify_embedded_sha256(candidate_evidence, "evidence_sha256")
    bindings = (
        ("source_plan_artifact_sha256", source_plan),
        ("source_completion_artifact_sha256", source_completion),
        ("source_archive_evidence_artifact_sha256", source_evidence),
        ("source_reclassification_artifact_sha256", source_reclassification),
    )
    if any(
        _canonical_sha256(payload) != proposal.get(field)
        for field, payload in bindings
    ) or (
        authorization.get("proposal_sha256") != proposal.get("proposal_sha256")
        or plan.get("proposal_sha256") != proposal.get("proposal_sha256")
        or proposal.get("design_rule_sha256")
        != design_rule.get("design_rule_sha256")
        or proposal.get("archive_candidate_evidence_sha256")
        != candidate_evidence.get("evidence_sha256")
        or source_completion.get("completion_sha256")
        != proposal.get("source_completion_sha256")
    ):
        raise StabilityAuditError("spec_gate_supplement_source_state_mismatch")


def _validate_spec_gate_audit_contract(
    contract: StabilityAuditContract,
    *,
    plan: dict[str, Any],
    proposal: dict[str, Any],
) -> None:
    if not (
        contract.sha256
        == plan.get("audit_contract_sha256")
        == proposal.get("audit_contract_sha256")
    ):
        raise StabilityAuditError("spec_gate_supplement_contract_mismatch")


def _write_preparation_marker(
    directory: Path,
    transaction_id: str,
    *,
    role: str,
    status: str = "staging",
) -> None:
    if role not in {"output", "governance"}:
        raise ValueError(f"invalid_preparation_role:{role}")
    atomic_write_json(
        directory / _PREPARATION_MARKER,
        {
            "marker_version": "lyx_filter_preparation_transaction_v1",
            "transaction_id": transaction_id,
            "role": role,
            "status": status,
        },
    )


def _commit_preparation_pair(
    output_dir: Path,
    governance_dir: Path,
    transaction_id: str,
) -> None:
    _write_preparation_marker(
        output_dir,
        transaction_id,
        role="output",
        status="committed",
    )
    _write_preparation_marker(
        governance_dir,
        transaction_id,
        role="governance",
        status="committed",
    )


def _require_committed_preparation_pair(
    output_dir: Path,
    governance_dir: Path,
) -> None:
    try:
        output_marker = read_json(output_dir / _PREPARATION_MARKER)
        governance_marker = read_json(governance_dir / _PREPARATION_MARKER)
    except (FileNotFoundError, ValueError) as exc:
        raise StabilityAuditError("preparation_pair_not_committed") from exc
    if (
        output_marker.get("status") != "committed"
        or governance_marker.get("status") != "committed"
        or output_marker.get("role") != "output"
        or governance_marker.get("role") != "governance"
        or not output_marker.get("transaction_id")
        or output_marker.get("transaction_id")
        != governance_marker.get("transaction_id")
    ):
        raise StabilityAuditError("preparation_pair_not_committed")


def _long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt":
        if resolved.startswith("\\\\?\\"):
            return resolved
        if resolved.startswith("\\\\"):
            return "\\\\?\\UNC\\" + resolved[2:]
        return "\\\\?\\" + resolved
    return resolved


def _path_exists(path: Path) -> bool:
    return os.path.exists(_long_path(path))


def _read_json_path(path: Path) -> dict[str, Any]:
    with open(_long_path(path), encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise StabilityAuditError(f"json_root_must_be_object:{path}")
    return payload


def _file_sha256_long(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_long_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_filter_profiles() -> tuple[FilterProfile, ...]:
    """The eight archive-derived coordinates frozen before new diagnostics."""

    return (
        FilterProfile("p25-short-low", "core", 25, 40, 0.008),
        FilterProfile("p25-short-mid", "core", 25, 40, 0.012),
        FilterProfile("p25-long-mid", "core", 25, 200, 0.010),
        FilterProfile(
            "p50-short-low",
            "core",
            50,
            80,
            0.006,
            recovery_sentinel_role="conservative",
        ),
        FilterProfile(
            "p50-long-mid",
            "core",
            50,
            200,
            0.012,
            recovery_sentinel_role="intermediate",
        ),
        FilterProfile("p50-long-high", "coverage_boundary", 50, 200, 0.016),
        FilterProfile(
            "p100-short-high",
            "core",
            100,
            40,
            0.016,
            recovery_sentinel_role="aggressive",
        ),
        FilterProfile("p100-long-low", "coverage_boundary", 100, 200, 0.006),
    )


def revised_filter_profiles_v2() -> tuple[FilterProfile, ...]:
    """Replace only the two profiles rejected by non-pathological hard gates."""

    return (
        FilterProfile("p25-short-low", "core", 25, 40, 0.008),
        FilterProfile("p25-short-mid", "core", 25, 40, 0.012),
        FilterProfile("p25-long-mid", "core", 25, 200, 0.010),
        FilterProfile(
            "p50-short-low",
            "core",
            50,
            80,
            0.006,
            recovery_sentinel_role="conservative",
        ),
        FilterProfile(
            "p50-long-mid",
            "core",
            50,
            200,
            0.012,
            recovery_sentinel_role="intermediate",
        ),
        FilterProfile("p50-boundary-high", "coverage_boundary", 50, 120, 0.016),
        FilterProfile(
            "p100-short-high",
            "core",
            100,
            40,
            0.016,
            recovery_sentinel_role="aggressive",
        ),
        FilterProfile("p100-boundary-low", "coverage_boundary", 100, 120, 0.006),
    )


def _select_audit_records(
    *,
    baseline_manifest_path: Path,
    metrics_table_path: Path,
) -> tuple[tuple[FilterAuditRecord, ...], dict[str, Any]]:
    baseline = read_json(baseline_manifest_path)
    records = baseline.get("records")
    if not isinstance(records, list):
        raise StabilityAuditError("invalid_baseline_manifest_records")
    by_id = {
        str(record["sample_id"]): record
        for record in records
        if isinstance(record, dict)
    }
    metrics_by_scene: dict[str, list[tuple[float, str]]] = {
        scene: [] for scene in _EXPECTED_SCENES
    }
    with metrics_table_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scene = str(row.get("scene", ""))
            sample_id = str(row.get("sample_id", ""))
            if scene not in metrics_by_scene or sample_id not in by_id:
                continue
            value = float(row["physical_reliable_motion_mae_bpm"])
            if math.isfinite(value):
                metrics_by_scene[scene].append((value, sample_id))

    chosen: list[FilterAuditRecord] = []
    selection_rows: list[dict[str, Any]] = []
    for scene in _EXPECTED_SCENES:
        candidates = metrics_by_scene[scene]
        if not candidates:
            raise StabilityAuditError(f"missing_record_selection_evidence:{scene}")
        metric, sample_id = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
        source = by_id[sample_id]
        chosen.append(
            FilterAuditRecord(
                record_id=sample_id,
                scene=scene,
                data_path=str(source["sensor_path"]),
                reference_path=str(source["reference_path"]),
                data_sha256=str(source["data_sha256"]),
                reference_sha256=str(source["reference_sha256"]),
            )
        )
        selection_rows.append(
            {
                "scene": scene,
                "record_id": sample_id,
                "physical_reliable_motion_mae_bpm": metric,
            }
        )
    payload = {
        "manifest_version": "lyx_filter_stability_record_manifest_v1",
        "selection_rule": (
            "each_scene_max_physical_reliable_motion_mae_then_record_id"
        ),
        "source_baseline_manifest": str(baseline_manifest_path),
        "source_baseline_manifest_sha256": file_sha256(baseline_manifest_path),
        "source_metrics_table": str(metrics_table_path),
        "source_metrics_table_sha256": file_sha256(metrics_table_path),
        "records": [
            {
                **asdict(record),
                "combined_data_sha256": record.combined_data_sha256,
                "selection_metric": next(
                    row["physical_reliable_motion_mae_bpm"]
                    for row in selection_rows
                    if row["record_id"] == record.record_id
                ),
            }
            for record in chosen
        ],
    }
    payload["record_manifest_sha256"] = _canonical_sha256(payload)
    return tuple(chosen), payload


def _scan_archive_evidence(
    *,
    archive_root: Path,
    baseline_manifest_path: Path,
    profiles: tuple[FilterProfile, ...],
) -> tuple[tuple[ArchivedProfileEvidence, ...], dict[str, Any]]:
    baseline = read_json(baseline_manifest_path)
    records = baseline.get("records")
    if not isinstance(records, list):
        raise StabilityAuditError("invalid_baseline_manifest_records")
    wanted = {profile.coordinate: profile.profile_id for profile in profiles}
    matches: dict[tuple[int, int, float], list[dict[str, Any]]] = {
        coordinate: [] for coordinate in wanted
    }
    for record in records:
        if not isinstance(record, dict):
            continue
        cache_entry = Path(str(record["cache_entry"]))
        cache_root = archive_root / cache_entry.parent
        for child in cache_root.iterdir():
            if not child.is_dir():
                continue
            reservation_path = child / "reservation.json"
            try:
                reservation = _read_json_path(reservation_path)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            identity = reservation.get("identity")
            if not isinstance(identity, dict) or not str(
                identity.get("candidate_id", "")
            ).startswith("physical_v1:"):
                continue
            requested = identity.get("requested_params")
            if not isinstance(requested, dict):
                continue
            coordinate = (
                int(requested["fs_target"]),
                int(requested["memory_ms"]),
                float(requested["mu_base"]),
            )
            if coordinate not in matches:
                continue
            matches[coordinate].append(
                {
                    "profile_id": wanted[coordinate],
                    "record_id": str(record["sample_id"]),
                    "scene": str(record["scene"]),
                    "candidate_id": str(identity["candidate_id"]),
                    "cache_key": str(reservation["cache_key"]),
                    "exclusion_half_width_bpm": float(
                        requested["exclusion_half_width_bpm"]
                    ),
                    "candidate_identity_sha256": _canonical_sha256(identity),
                    "reservation_sha256": _file_sha256_long(reservation_path),
                }
            )

    evidence_payload: dict[str, Any] = {
        "evidence_version": "lyx_filter_profile_archive_evidence_v1",
        "archive_root": str(archive_root),
        "archive_git_commit": baseline.get("archive_git_commit"),
        "baseline_manifest_sha256": file_sha256(baseline_manifest_path),
        "profiles": [],
    }
    for profile in profiles:
        entries = sorted(
            matches[profile.coordinate],
            key=lambda item: (
                item["record_id"],
                item["candidate_id"],
                item["cache_key"],
            ),
        )
        scenes = sorted({str(item["scene"]) for item in entries})
        evidence_payload["profiles"].append(
            {
                "profile_id": profile.profile_id,
                "coordinate": {
                    "fs_target": profile.fs_target,
                    "memory_ms": profile.memory_ms,
                    "nominal_mu": float(profile.nominal_mu),
                },
                "occurrence_count": len(entries),
                "scenes": scenes,
                "entries": entries,
            }
        )
    evidence_payload["evidence_index_sha256"] = _canonical_sha256(evidence_payload)
    evidence = tuple(
        ArchivedProfileEvidence(
            fs_target=profile.fs_target,
            memory_ms=profile.memory_ms,
            nominal_mu=profile.nominal_mu,
            occurrence_count=len(matches[profile.coordinate]),
            scenes=tuple(
                sorted(
                    {
                        str(item["scene"])
                        for item in matches[profile.coordinate]
                    }
                )
            ),
            archive_manifest_sha256=file_sha256(baseline_manifest_path),
            archive_table_sha256=str(evidence_payload["evidence_index_sha256"]),
        )
        for profile in profiles
    )
    return evidence, evidence_payload


def _source_bundle_hash(paths: tuple[Path, ...]) -> str:
    return _canonical_sha256(
        {
            str(path): file_sha256(path)
            for path in sorted(paths, key=lambda item: str(item))
        }
    )


def _current_solver_hash() -> str:
    source_root = Path(__file__).resolve().parents[2]
    return _source_bundle_hash(
        (
            source_root / "ppg_hr" / "core" / "choose_delay.py",
            source_root / "ppg_hr" / "core" / "lms_filter.py",
            source_root / "ppg_hr" / "v2" / "signal_preparation.py",
            Path(__file__).resolve(),
        )
    )


def _current_code_hash() -> str:
    source_root = Path(__file__).resolve().parents[2]
    return _source_bundle_hash(
        (
            source_root / "ppg_hr" / "v2" / "recovery_filter_profiles.py",
            source_root / "ppg_hr" / "v2" / "recovery_filter_stability.py",
            Path(__file__).resolve(),
        )
    )


def prepare_filter_profile_audit(
    *,
    archive_root: Path,
    baseline_manifest_path: Path,
    metrics_table_path: Path,
    output_dir: Path,
    governance_dir: Path,
    authorization_receipt_path: Path,
) -> dict[str, Any]:
    """Freeze inputs, validate user authorization, and register zero-run identities."""

    if output_dir.exists() or governance_dir.exists():
        raise StabilityAuditError("filter_audit_output_already_exists")
    profiles = default_filter_profiles()
    contract = StabilityAuditContract.frozen_v1()
    records, record_manifest = _select_audit_records(
        baseline_manifest_path=baseline_manifest_path,
        metrics_table_path=metrics_table_path,
    )
    evidence, evidence_index = _scan_archive_evidence(
        archive_root=archive_root,
        baseline_manifest_path=baseline_manifest_path,
        profiles=profiles,
    )
    design_rule: dict[str, Any] = {
        "design_rule_version": "lyx_filter_profile_design_rule_v1",
        "status": "frozen_before_new_diagnostics",
        "source_policy": "existing_archived_trial_distribution_only",
        "profile_selection_policy": (
            "preserve_3_3_2_sampling_quota_and_cover_short_long_low_mid_high_regions"
        ),
        "audit_policy": "qualify_or_reject_only_never_tune_or_replace",
        "independent_bo_authorized": False,
        "profile_coordinates": [asdict(profile) for profile in profiles],
        "record_selection_rule": record_manifest["selection_rule"],
        "audit_contract": contract.to_dict(),
        "audit_contract_sha256": contract.sha256,
        "archive_evidence_index_sha256": evidence_index["evidence_index_sha256"],
    }
    design_rule["design_rule_sha256"] = _canonical_sha256(design_rule)
    library = freeze_filter_profile_library(
        profiles,
        evidence,
        design_rule_sha256=str(design_rule["design_rule_sha256"]),
    )

    authorization = read_json(authorization_receipt_path)
    validate_budget_amendment_authorization(
        BudgetAmendmentRequest(
            stage="filter_profile_stability_audit",
            profile_design_rule_hash=str(design_rule["design_rule_sha256"]),
            record_manifest_hash=str(record_manifest["record_manifest_sha256"]),
            added_unique_identities=32,
            normal_unique_identity_limit=704,
            max_unique_identities=716,
            max_attempts=1432,
        ),
        receipt=authorization,
    )

    solver_hash = _current_solver_hash()
    code_hash = _current_code_hash()
    evaluation_hash = _canonical_sha256(
        {
            "design_rule_sha256": design_rule["design_rule_sha256"],
            "record_manifest_sha256": record_manifest["record_manifest_sha256"],
            "audit_contract_sha256": contract.sha256,
            "produces_hr_tracking_trajectory": False,
            "uses_bayesian_optimization": False,
        }
    )
    identities = plan_filter_audit_identities(
        profiles=profiles,
        records=records,
        parent_experiment_id=_PARENT_EXPERIMENT_ID,
        solver_hash=solver_hash,
        metric_contract_hash=contract.sha256,
        evaluation_hash=evaluation_hash,
        design_rule_sha256=str(design_rule["design_rule_sha256"]),
        record_manifest_sha256=str(record_manifest["record_manifest_sha256"]),
        authorization_receipt=authorization,
    )

    budget = BudgetContract.approved_v2()
    exploration = ExplorationRegistry.zero_budget_v1()
    plan = {
        "plan_version": "lyx_filter_stability_audit_plan_v1",
        "status": "prepared_zero_run",
        "parent_experiment_id": _PARENT_EXPERIMENT_ID,
        "profile_library_sha256": library["library_sha256"],
        "design_rule_sha256": design_rule["design_rule_sha256"],
        "record_manifest_sha256": record_manifest["record_manifest_sha256"],
        "audit_contract": contract.to_dict(),
        "audit_contract_sha256": contract.sha256,
        "solver_hash": solver_hash,
        "code_hash": code_hash,
        "evaluation_hash": evaluation_hash,
        "identity_count": len(identities),
        "identity_sha256": [identity.sha256 for identity in identities],
        "independent_bo_authorized": False,
        "produces_hr_tracking_trajectory": False,
        "profiles": [asdict(profile) for profile in profiles],
        "records": [asdict(record) for record in records],
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    output_staging = output_dir.with_name(
        f".{output_dir.name}.{uuid.uuid4().hex}.staging"
    )
    governance_staging = governance_dir.with_name(
        f".{governance_dir.name}.{uuid.uuid4().hex}.staging"
    )
    preparation_transaction_id = uuid.uuid4().hex
    governance_receipt: dict[str, Any] = {}
    try:
        os.makedirs(_long_path(output_staging))
        os.makedirs(_long_path(governance_staging))
        _write_preparation_marker(
            output_staging,
            preparation_transaction_id,
            role="output",
        )
        _write_preparation_marker(
            governance_staging,
            preparation_transaction_id,
            role="governance",
        )
        atomic_write_json(output_staging / "profile_design_rule.json", design_rule)
        atomic_write_json(
            output_staging / "stability_audit_record_manifest.json",
            record_manifest,
        )
        atomic_write_json(
            output_staging / "archive_profile_evidence_index.json",
            evidence_index,
        )
        atomic_write_json(
            output_staging / "filter_profile_library_freeze.json",
            library,
        )
        atomic_write_json(output_staging / "stability_audit_plan.json", plan)
        atomic_write_json(
            governance_staging / "budget_amendment_authorization.json",
            authorization,
        )
        atomic_write_json(
            governance_staging / "budget_contract.json",
            budget.to_dict(),
        )
        atomic_write_json(
            governance_staging / "exploration_registry.json",
            exploration.to_dict(),
        )
        registry = AttemptRegistry.create(
            governance_staging / "attempt_registry.json",
            budget_contract=budget,
            exploration_registry=exploration,
        )
        for identity in identities:
            registry.register_identity(identity)
        (governance_staging / ".attempt_registry.json.lock").unlink(
            missing_ok=True
        )
        governance_receipt = {
            "receipt_version": "lyx_recovery_governance_receipt_v2",
            "status": "prepared_zero_run",
            "parent_experiment_id": _PARENT_EXPERIMENT_ID,
            "planned_unique_identity_limit": budget.max_unique_identities,
            "normal_unique_identity_limit": budget.normal_unique_identity_limit,
            "worst_case_attempt_limit": budget.max_attempts,
            "filter_profile_stability_audit_unique_identities": 32,
            "attempt_registry_summary": registry.summary(),
            "exploration_unique_budget": exploration.unique_budget,
            "independent_bo_authorized": False,
            "budget_authorization_sha256": file_sha256(
                governance_staging / "budget_amendment_authorization.json"
            ),
            "artifacts": {
                name: file_sha256(governance_staging / name)
                for name in (
                    "attempt_registry.json",
                    "budget_contract.json",
                    "exploration_registry.json",
                    "budget_amendment_authorization.json",
                )
            },
        }
        atomic_write_json(
            governance_staging / "governance_receipt.json",
            governance_receipt,
        )
        os.replace(_long_path(governance_staging), _long_path(governance_dir))
        os.replace(_long_path(output_staging), _long_path(output_dir))
        _commit_preparation_pair(
            output_dir,
            governance_dir,
            preparation_transaction_id,
        )
    except BaseException:
        for path in (output_staging, governance_staging, governance_dir):
            if path.exists():
                shutil.rmtree(_long_path(path))
        raise
    return {
        "plan": plan,
        "governance_receipt": governance_receipt,
    }


def _reference_hr_at_center(ref_data: np.ndarray, center_s: float) -> float:
    ref = np.asarray(ref_data, dtype=float)
    if ref.ndim != 2 or ref.shape[1] < 2 or ref.shape[0] == 0:
        raise StabilityAuditError("missing_reference_hr_for_spectral_audit")
    times = ref[:, 0]
    values = ref[:, 1]
    finite = np.isfinite(times) & np.isfinite(values)
    if not np.any(finite):
        raise StabilityAuditError("nonfinite_reference_hr_for_spectral_audit")
    return float(np.interp(center_s, times[finite], values[finite]))


def _audit_profile_record(
    profile: FilterProfile,
    record: FilterAuditRecord,
    *,
    contract: StabilityAuditContract,
) -> dict[str, Any]:
    started = time.perf_counter()
    data_path = Path(record.data_path)
    reference_path = Path(record.reference_path)
    if file_sha256(data_path) != record.data_sha256:
        raise StabilityAuditError(f"record_data_hash_mismatch:{record.record_id}")
    if file_sha256(reference_path) != record.reference_sha256:
        raise StabilityAuditError(f"record_reference_hash_mismatch:{record.record_id}")
    cfg = V2RunConfig(
        data_path=data_path,
        ref_path=reference_path,
        adaptive_filter="lms",
        algorithm_preset="lite",
        reference_groups_order=("HF",),
        fs_target=profile.fs_target,
        lms_mu_base=float(profile.nominal_mu),
        lms_mu_min=1e-6,
        max_order=profile.actual_taps,
        smooth_win_len=5,
        time_bias=5.0,
    )
    prepared = prepare_v2_signals(cfg)
    fs = int(prepared.fs)
    window_samples = int(round(cfg.window_seconds * fs))
    step_samples = int(round(cfg.window_step_seconds * fs))
    start_sample = max(0, int(round(float(prepared.params.time_start) * fs)))
    end_sample = min(
        prepared.ppg.size,
        int(
            round(
                (
                    prepared.ppg_ori.size / fs
                    - float(prepared.params.time_buffer)
                )
                * fs
            )
        ),
    )
    references = list(prepared.references)
    if len(references) != 2:
        raise StabilityAuditError("filter_audit_requires_two_hf_references")
    stage_audits: list[dict[str, Any]] = []
    for idx_s in range(start_sample, max(start_sample, end_sample - window_samples + 1), step_samples):
        idx_e = idx_s + window_samples
        if idx_e > prepared.ppg.size:
            break
        time_1 = idx_s / fs
        center_s = time_1 + float(cfg.window_seconds) / 2.0
        true_hr_bpm = _reference_hr_at_center(prepared.ref_data, center_s)
        signals = [np.asarray(item["signal"], dtype=float) for item in references]
        corr_arr, _empty, delay, _acc_delay = choose_delay(
            fs,
            time_1,
            prepared.ppg,
            [],
            signals,
        )
        if corr_arr.size == 0:
            continue
        current = np.asarray(prepared.ppg[idx_s:idx_e], dtype=float)
        order = max(1, min(profile.actual_taps, int(abs(delay)) or 1))
        for ref_idx in np.argsort(corr_arr)[::-1]:
            reference = signals[int(ref_idx)][idx_s:idx_e]
            stage = audit_lms_stage(
                desired=current,
                reference=reference,
                fs=fs,
                nominal_mu=float(profile.nominal_mu),
                order=order,
                K=0,
                true_hr_bpm=true_hr_bpm,
            )
            stage.update(
                {
                    "window_center_s": center_s,
                    "channel": str(references[int(ref_idx)]["channel"]),
                    "delay_samples": int(delay),
                    "archive_style_abs_corr": float(corr_arr[int(ref_idx)]),
                }
            )
            stage_audits.append(stage)
            current, _weights, _unused = lms_filter(
                float(stage["effective_mu"]),
                order,
                0,
                reference,
                current,
            )
    return summarize_record_audit(
        record_id=record.record_id,
        scene=record.scene,
        stage_audits=stage_audits,
        configured_max_taps=profile.actual_taps,
        runtime_seconds=time.perf_counter() - started,
        contract=contract,
    )


def _execute_registered_profile_audits(
    *,
    registry: AttemptRegistry,
    identities: tuple[AttemptIdentity, ...],
    profiles: tuple[FilterProfile, ...],
    records: tuple[FilterAuditRecord, ...],
    contract: StabilityAuditContract,
    output_dir: Path,
    cache_prefix: str,
) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {
        profile.profile_id: [] for profile in profiles
    }
    pairs = tuple(
        (profile, record)
        for profile in profiles
        for record in records
    )
    if len(identities) != len(pairs):
        raise StabilityAuditError("profile_audit_identity_matrix_mismatch")
    for identity, (profile, record) in zip(identities, pairs, strict=True):
        result_dir = (
            registry.trusted_cache_root
            / f"{cache_prefix}-{identity.sha256[:16]}"
        )
        result_path = result_dir / "result.json"
        cache_receipt_path = result_dir / "cache_receipt.json"

        def operation(
            *,
            current_profile: FilterProfile = profile,
            current_record: FilterAuditRecord = record,
            current_identity: Any = identity,
            current_result_dir: Path = result_dir,
            current_result_path: Path = result_path,
            current_cache_receipt_path: Path = cache_receipt_path,
        ) -> dict[str, Any]:
            audit = _audit_profile_record(
                current_profile,
                current_record,
                contract=contract,
            )
            os.makedirs(_long_path(current_result_dir), exist_ok=True)
            payload = {
                "producer": "content_addressed_solver_cache_v1",
                "status": "complete",
                "valid": True,
                "identity": current_identity.to_dict(),
                "profile_id": current_profile.profile_id,
                "record_id": current_record.record_id,
                "audit": audit,
            }
            atomic_write_json(current_result_path, payload)
            atomic_write_json(
                current_cache_receipt_path,
                {
                    "identity_sha256": current_identity.sha256,
                    "result_path": current_result_path.name,
                    "result_sha256": file_sha256(current_result_path),
                },
            )
            return audit

        if _path_exists(cache_receipt_path):
            result_payload = read_json(result_path)
            audit = dict(result_payload["audit"])
        else:
            audit = registry.execute_registered(identity, operation)
        evidence = CacheEvidence.from_path(
            cache_receipt_path,
            expected_identity=identity,
            trusted_cache_root=registry.trusted_cache_root,
        )
        registry.record_cache_hit(identity, evidence=evidence)
        completed = {
            **audit,
            "identity_sha256": identity.sha256,
            "result_sha256": evidence.result_sha256,
            "data_sha256": record.data_sha256,
            "reference_sha256": record.reference_sha256,
        }
        results[profile.profile_id].append(completed)
        atomic_write_json(
            output_dir
            / "record_audits"
            / profile.profile_id
            / f"{record.record_id}.json",
            completed,
        )
    return results


def spec_gate_supplement_profiles_v1() -> tuple[FilterProfile, ...]:
    """Archive-observed conservative coordinates frozen before approval."""

    return (
        FilterProfile("p50-short-low-40", "core", 50, 40, 0.006),
        FilterProfile("p50-short-midlow-40", "core", 50, 40, 0.008),
        FilterProfile("p100-short-low-40", "coverage_boundary", 100, 40, 0.006),
        FilterProfile("p100-short-midlow-40", "coverage_boundary", 100, 40, 0.008),
        FilterProfile(
            "p100-medium-low-80",
            "coverage_boundary",
            100,
            80,
            0.006,
        ),
        FilterProfile(
            "p100-medium-midlow-80",
            "coverage_boundary",
            100,
            80,
            0.008,
        ),
    )


def select_spec_gate_supplement_profiles(
    profile_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen per-rate ranking without borrowing slots across rates."""

    def rank(receipt: dict[str, Any]) -> tuple[float, float, float, str]:
        retention = min(
            float(item["true_peak_retention_ratio_median"])
            for item in receipt["spectral_evidence"]["record_results"]
        )
        weight_norm = max(
            float(item["weight_norm_max"])
            for item in receipt["stability"]["record_results"]
        )
        runtime = float(
            np.median(
                [
                    float(item["runtime_seconds"])
                    for item in receipt["stability"]["record_results"]
                ]
            )
        )
        return (-retention, weight_norm, runtime, str(receipt["profile_id"]))

    eligible = [
        receipt
        for receipt in profile_receipts
        if receipt.get("may_enter_formal_matrix") is True
    ]
    p50 = sorted(
        (receipt for receipt in eligible if int(receipt["fs_target"]) == 50),
        key=rank,
    )
    p100 = sorted(
        (receipt for receipt in eligible if int(receipt["fs_target"]) == 100),
        key=rank,
    )
    if len(p50) < 2 or len(p100) < 2:
        return {
            "status": "blocked_insufficient_rate_coverage",
            "selected_p50_profile_ids": [
                str(receipt["profile_id"]) for receipt in p50[:2]
            ],
            "selected_p100_profile_ids": [
                str(receipt["profile_id"]) for receipt in p100[:2]
            ],
            "selected_profile_ids": [],
        }
    selected_p50 = [str(receipt["profile_id"]) for receipt in p50[:2]]
    selected_p100 = [str(receipt["profile_id"]) for receipt in p100[:2]]
    return {
        "status": "complete",
        "selected_p50_profile_ids": selected_p50,
        "selected_p100_profile_ids": selected_p100,
        "selected_profile_ids": [*selected_p50, *selected_p100],
    }


def _profile_from_dict(payload: dict[str, Any]) -> FilterProfile:
    return FilterProfile(
        profile_id=str(payload["profile_id"]),
        design_role=str(payload["design_role"]),  # type: ignore[arg-type]
        fs_target=int(payload["fs_target"]),
        memory_ms=int(payload["memory_ms"]),
        nominal_mu=float(payload["nominal_mu"]),
        recovery_sentinel_role=payload.get("recovery_sentinel_role"),  # type: ignore[arg-type]
    )


def execute_filter_profile_audit(
    *,
    output_dir: Path,
    governance_dir: Path,
) -> dict[str, Any]:
    """Execute only the 32 pre-registered diagnostic identities."""

    _require_committed_preparation_pair(output_dir, governance_dir)
    plan = read_json(output_dir / "stability_audit_plan.json")
    _verify_embedded_sha256(plan, "plan_sha256")
    if plan.get("status") != "prepared_zero_run":
        raise StabilityAuditError("filter_audit_plan_not_prepared")
    if plan.get("solver_hash") != _current_solver_hash():
        raise StabilityAuditError("filter_audit_solver_source_changed_after_freeze")
    if plan.get("code_hash") != _current_code_hash():
        raise StabilityAuditError("filter_audit_code_source_changed_after_freeze")
    profiles = tuple(_profile_from_dict(item) for item in plan["profiles"])
    records = tuple(FilterAuditRecord(**item) for item in plan["records"])
    contract = StabilityAuditContract(**plan["audit_contract"])
    authorization = read_json(governance_dir / "budget_amendment_authorization.json")
    identities = plan_filter_audit_identities(
        profiles=profiles,
        records=records,
        parent_experiment_id=str(plan["parent_experiment_id"]),
        solver_hash=str(plan["solver_hash"]),
        metric_contract_hash=str(plan["audit_contract_sha256"]),
        evaluation_hash=str(plan["evaluation_hash"]),
        design_rule_sha256=str(plan["design_rule_sha256"]),
        record_manifest_sha256=str(plan["record_manifest_sha256"]),
        authorization_receipt=authorization,
    )
    if [identity.sha256 for identity in identities] != plan["identity_sha256"]:
        raise StabilityAuditError("filter_audit_plan_identity_mismatch")
    budget = BudgetContract.approved_v2()
    exploration = ExplorationRegistry.zero_budget_v1()
    registry = AttemptRegistry.open(
        governance_dir / "attempt_registry.json",
        budget_contract=budget,
        exploration_registry=exploration,
    )
    all_results: dict[str, list[dict[str, Any]]] = {
        profile.profile_id: [] for profile in profiles
    }
    for identity, (profile, record) in zip(
        identities,
        (
            (profile, record)
            for profile in profiles
            for record in records
        ),
        strict=True,
    ):
        result_dir = registry.trusted_cache_root / identity.sha256
        result_path = result_dir / "result.json"
        cache_receipt_path = result_dir / "cache_receipt.json"

        def operation(
            *,
            current_profile: FilterProfile = profile,
            current_record: FilterAuditRecord = record,
            current_identity: Any = identity,
            current_result_dir: Path = result_dir,
            current_result_path: Path = result_path,
            current_cache_receipt_path: Path = cache_receipt_path,
        ) -> dict[str, Any]:
            audit = _audit_profile_record(
                current_profile,
                current_record,
                contract=contract,
            )
            os.makedirs(_long_path(current_result_dir), exist_ok=True)
            payload = {
                "producer": "content_addressed_solver_cache_v1",
                "status": "complete",
                "valid": True,
                "identity": current_identity.to_dict(),
                "profile_id": current_profile.profile_id,
                "record_id": current_record.record_id,
                "audit": audit,
            }
            atomic_write_json(current_result_path, payload)
            atomic_write_json(
                current_cache_receipt_path,
                {
                    "identity_sha256": current_identity.sha256,
                    "result_path": current_result_path.name,
                    "result_sha256": file_sha256(current_result_path),
                },
            )
            return audit

        audit = registry.execute_registered(identity, operation)
        evidence = CacheEvidence.from_path(
            cache_receipt_path,
            expected_identity=identity,
            trusted_cache_root=registry.trusted_cache_root,
        )
        registry.record_cache_hit(identity, evidence=evidence)
        audit = {
            **audit,
            "identity_sha256": identity.sha256,
            "result_sha256": evidence.result_sha256,
            "data_sha256": record.data_sha256,
            "reference_sha256": record.reference_sha256,
        }
        all_results[profile.profile_id].append(audit)
        atomic_write_json(
            output_dir
            / "record_audits"
            / profile.profile_id
            / f"{record.record_id}.json",
            audit,
        )

    library = read_json(output_dir / "filter_profile_library_freeze.json")
    receipts: list[dict[str, Any]] = []
    for profile in profiles:
        receipt = build_filter_profile_receipt(
            profile,
            all_results[profile.profile_id],
            audit_contract=contract,
            library_sha256=str(library["library_sha256"]),
            solver_hash=str(plan["solver_hash"]),
            code_hash=str(plan["code_hash"]),
            evaluation_hash=str(plan["evaluation_hash"]),
            design_rule_sha256=str(plan["design_rule_sha256"]),
            record_manifest_sha256=str(plan["record_manifest_sha256"]),
        )
        receipts.append(receipt)
        atomic_write_json(
            output_dir / "filter_profile_receipts" / f"{profile.profile_id}.json",
            receipt,
        )
    eligible = [
        str(receipt["profile_id"])
        for receipt in receipts
        if receipt["may_enter_formal_matrix"] is True
    ]
    final = {
        "receipt_version": "lyx_filter_profile_audit_completion_v1",
        "status": "complete" if len(eligible) == 8 else "incomplete_profile_library",
        "profile_count": 8,
        "eligible_profile_count": len(eligible),
        "eligible_profile_ids": eligible,
        "rejected_profile_ids": [
            str(receipt["profile_id"])
            for receipt in receipts
            if receipt["may_enter_formal_matrix"] is not True
        ],
        "recovery_sentinels": library["recovery_sentinels"],
        "all_recovery_sentinels_eligible": all(
            profile_id in eligible
            for profile_id in library["recovery_sentinels"].values()
        ),
        "attempt_registry_summary": registry.summary(),
        "planned_identity_count": 32,
        "actual_hr_tracking_trajectory_count": 0,
        "independent_bo_run_count": 0,
        "exploration_run_count": 0,
        "profile_receipt_sha256": {
            str(receipt["profile_id"]): str(receipt["receipt_sha256"])
            for receipt in receipts
        },
    }
    final["completion_sha256"] = _canonical_sha256(final)
    atomic_write_json(output_dir / "filter_profile_audit_completion.json", final)
    governance_receipt_path = governance_dir / "governance_receipt.json"
    governance_receipt = read_json(governance_receipt_path)
    governance_receipt.update(
        {
            "status": "complete",
            "attempt_registry_summary": registry.summary(),
            "filter_profile_audit_completion_sha256": file_sha256(
                output_dir / "filter_profile_audit_completion.json"
            ),
        }
    )
    governance_receipt["artifacts"]["attempt_registry.json"] = file_sha256(
        governance_dir / "attempt_registry.json"
    )
    atomic_write_json(governance_receipt_path, governance_receipt)
    return final


def build_filter_audit_revision_proposal(
    *,
    archive_root: Path,
    baseline_manifest_path: Path,
    source_output_dir: Path,
    proposal_dir: Path,
) -> dict[str, Any]:
    """Build a zero-run correction proposal without recording user approval."""

    if proposal_dir.exists():
        raise StabilityAuditError("filter_audit_revision_proposal_already_exists")
    source_plan = read_json(source_output_dir / "stability_audit_plan.json")
    source_completion = read_json(
        source_output_dir / "filter_profile_audit_completion.json"
    )
    if (
        source_completion.get("status") != "incomplete_profile_library"
        or source_completion.get("planned_identity_count") != 32
        or source_completion.get("independent_bo_run_count") != 0
    ):
        raise StabilityAuditError("unexpected_source_filter_audit_state")
    record_manifest = read_json(
        source_output_dir / "stability_audit_record_manifest.json"
    )
    if (
        record_manifest.get("record_manifest_sha256")
        != source_plan.get("record_manifest_sha256")
    ):
        raise StabilityAuditError("source_record_manifest_hash_mismatch")

    profiles = revised_filter_profiles_v2()
    evidence, evidence_index = _scan_archive_evidence(
        archive_root=archive_root,
        baseline_manifest_path=baseline_manifest_path,
        profiles=profiles,
    )
    corrected_contract = StabilityAuditContract.corrected_v2()
    unchanged_profile_ids = {
        "p25-short-low",
        "p25-short-mid",
        "p25-long-mid",
        "p50-short-low",
        "p50-long-mid",
        "p100-short-high",
    }
    reclassified_records: list[dict[str, Any]] = []
    for profile_id in sorted(unchanged_profile_ids):
        profile_dir = source_output_dir / "record_audits" / profile_id
        for path in sorted(profile_dir.glob("*.json")):
            cached = read_json(path)
            revised = reclassify_cached_record_audit(
                cached,
                corrected_contract=corrected_contract,
                source_metric_contract_sha256=str(
                    source_plan["audit_contract_sha256"]
                ),
                source_result_sha256=str(cached["result_sha256"]),
            )
            revised["profile_id"] = profile_id
            revised["source_record_audit_sha256"] = file_sha256(path)
            reclassified_records.append(revised)
    if len(reclassified_records) != 24:
        raise StabilityAuditError("expected_twenty_four_cached_reclassifications")
    if any(
        not item["stability_pass"] or not item["spectral_pass"]
        for item in reclassified_records
    ):
        raise StabilityAuditError("unchanged_profile_fails_corrected_hard_gates")

    design_rule: dict[str, Any] = {
        "design_rule_version": "lyx_filter_profile_design_rule_v2",
        "status": "proposed_awaiting_human_budget_decision",
        "source_v1_completion_sha256": file_sha256(
            source_output_dir / "filter_profile_audit_completion.json"
        ),
        "source_v1_metric_contract_sha256": source_plan[
            "audit_contract_sha256"
        ],
        "corrected_metric_contract": corrected_contract.to_dict(),
        "corrected_metric_contract_sha256": corrected_contract.sha256,
        "instrument_defect": {
            "id": "cold_start_tail_head_max_denominator_instability",
            "reproduction": (
                "one near-zero cold-start head among otherwise stable stages "
                "forces the record maximum above the hard gate"
            ),
            "correction": (
                "retain tail/head maximum and p95 as descriptive evidence; "
                "use stability load, weight norm, residual RMS p95 and "
                "nonfinite count as independent stability hard gates"
            ),
        },
        "reclassification_policy": (
            "reuse_immutable_v1_numeric_summaries_for_six_unchanged_profiles"
        ),
        "replacement_policy": (
            "replace_only_profiles_that_fail_non_pathological_weight_or_residual_gates"
        ),
        "replacements": [
            {
                "rejected_profile_id": "p50-long-high",
                "rejected_coordinate": [50, 200, 0.016],
                "replacement_profile_id": "p50-boundary-high",
                "replacement_coordinate": [50, 120, 0.016],
                "reason": "reduce_mu_times_tap_span_while_retaining_high_mu_boundary",
            },
            {
                "rejected_profile_id": "p100-long-low",
                "rejected_coordinate": [100, 200, 0.006],
                "replacement_profile_id": "p100-boundary-low",
                "replacement_coordinate": [100, 120, 0.006],
                "reason": "reduce_high_rate_tap_span_while_retaining_low_mu_boundary",
            },
        ],
        "profile_coordinates": [asdict(profile) for profile in profiles],
        "record_manifest_sha256": record_manifest["record_manifest_sha256"],
        "archive_evidence_index_sha256": evidence_index["evidence_index_sha256"],
        "new_diagnostic_identity_count": 8,
        "independent_bo_authorized": False,
    }
    design_rule["design_rule_sha256"] = _canonical_sha256(design_rule)
    library = freeze_filter_profile_library(
        profiles,
        evidence,
        design_rule_sha256=str(design_rule["design_rule_sha256"]),
    )
    reclassification = {
        "receipt_version": "lyx_filter_metric_reclassification_proposal_v1",
        "status": "proposed_not_yet_authorized",
        "source_metric_contract_sha256": source_plan["audit_contract_sha256"],
        "corrected_metric_contract_sha256": corrected_contract.sha256,
        "numerical_rerun_count": 0,
        "record_count": len(reclassified_records),
        "all_six_unchanged_profiles_pass": True,
        "records": reclassified_records,
    }
    reclassification["proposal_sha256"] = _canonical_sha256(reclassification)
    budget_request = {
        "request_version": "lyx_filter_audit_budget_amendment_request_v2",
        "status": "awaiting_human_budget_decision",
        "approved": False,
        "decision_state": "awaiting_human_budget_decision",
        "stage": "filter_profile_stability_audit",
        "profile_design_rule_hash": design_rule["design_rule_sha256"],
        "record_manifest_hash": record_manifest["record_manifest_sha256"],
        "added_unique_identities": 8,
        "normal_unique_identity_limit": 712,
        "max_unique_identities": 724,
        "max_attempts": 1448,
        "independent_bo_authorized": False,
        "reclassifies_cached_identity_count": 24,
        "replacement_profile_count": 2,
        "replacement_record_count": 4,
    }
    budget_request["request_sha256"] = _canonical_sha256(budget_request)

    os.makedirs(_long_path(proposal_dir))
    atomic_write_json(proposal_dir / "profile_design_rule.json", design_rule)
    atomic_write_json(proposal_dir / "archive_profile_evidence_index.json", evidence_index)
    atomic_write_json(proposal_dir / "filter_profile_library_freeze.json", library)
    atomic_write_json(
        proposal_dir / "cached_metric_reclassification.json",
        reclassification,
    )
    atomic_write_json(proposal_dir / "budget_amendment_request.json", budget_request)
    proposal = {
        "proposal_version": "lyx_filter_audit_revision_proposal_v1",
        "status": "awaiting_human_budget_decision",
        "design_rule_sha256": design_rule["design_rule_sha256"],
        "record_manifest_sha256": record_manifest["record_manifest_sha256"],
        "library_sha256": library["library_sha256"],
        "corrected_metric_contract_sha256": corrected_contract.sha256,
        "budget_request_sha256": budget_request["request_sha256"],
        "new_unique_identity_count": 8,
        "reused_numeric_result_count": 24,
        "independent_bo_authorized": False,
        "may_execute": False,
    }
    proposal["proposal_sha256"] = _canonical_sha256(proposal)
    atomic_write_json(proposal_dir / "revision_proposal.json", proposal)
    return proposal


def prepare_filter_audit_revision(
    *,
    source_output_dir: Path,
    source_governance_dir: Path,
    proposal_dir: Path,
    output_dir: Path,
    governance_dir: Path,
    authorization_receipt_path: Path,
) -> dict[str, Any]:
    """Apply explicit approval, migrate the ledger, and register eight identities."""

    if output_dir.exists() or governance_dir.exists():
        raise StabilityAuditError("filter_audit_revision_output_already_exists")
    proposal = read_json(proposal_dir / "revision_proposal.json")
    budget_request = read_json(proposal_dir / "budget_amendment_request.json")
    design_rule = read_json(proposal_dir / "profile_design_rule.json")
    if (
        proposal.get("status") != "awaiting_human_budget_decision"
        or proposal.get("may_execute") is not False
        or budget_request.get("approved") is not False
        or proposal.get("budget_request_sha256")
        != budget_request.get("request_sha256")
        or proposal.get("design_rule_sha256")
        != design_rule.get("design_rule_sha256")
    ):
        raise StabilityAuditError("invalid_filter_audit_revision_proposal")
    authorization = read_json(authorization_receipt_path)
    if authorization.get("proposal_sha256") != proposal.get("proposal_sha256"):
        raise StabilityAuditError("budget_authorization_proposal_mismatch")
    amendment_request = BudgetAmendmentRequest(
        stage="filter_profile_stability_audit",
        profile_design_rule_hash=str(design_rule["design_rule_sha256"]),
        record_manifest_hash=str(design_rule["record_manifest_sha256"]),
        added_unique_identities=8,
        normal_unique_identity_limit=712,
        max_unique_identities=724,
        max_attempts=1448,
    )
    validate_budget_amendment_authorization(
        amendment_request,
        receipt=authorization,
    )
    _require_committed_preparation_pair(
        source_output_dir,
        source_governance_dir,
    )

    source_plan = read_json(source_output_dir / "stability_audit_plan.json")
    source_record_manifest = read_json(
        source_output_dir / "stability_audit_record_manifest.json"
    )
    _verify_embedded_sha256(source_plan, "plan_sha256")
    _verify_embedded_sha256(
        source_record_manifest,
        "record_manifest_sha256",
    )
    if (
        source_record_manifest.get("record_manifest_sha256")
        != design_rule.get("record_manifest_sha256")
    ):
        raise StabilityAuditError("filter_audit_revision_source_manifest_mismatch")
    records = tuple(
        FilterAuditRecord(
            record_id=str(item["record_id"]),
            scene=str(item["scene"]),
            data_path=str(item["data_path"]),
            reference_path=str(item["reference_path"]),
            data_sha256=str(item["data_sha256"]),
            reference_sha256=str(item["reference_sha256"]),
        )
        for item in source_record_manifest["records"]
    )
    profiles = revised_filter_profiles_v2()
    replacements = tuple(
        profile
        for profile in profiles
        if profile.profile_id in {"p50-boundary-high", "p100-boundary-low"}
    )
    contract = StabilityAuditContract.corrected_v2()
    solver_hash = _current_solver_hash()
    code_hash = _current_code_hash()
    evaluation_hash = _canonical_sha256(
        {
            "design_rule_sha256": design_rule["design_rule_sha256"],
            "record_manifest_sha256": design_rule["record_manifest_sha256"],
            "audit_contract_sha256": contract.sha256,
            "source_v1_plan_sha256": source_plan["plan_sha256"],
            "produces_hr_tracking_trajectory": False,
            "uses_bayesian_optimization": False,
        }
    )
    identities = plan_replacement_filter_audit_identities(
        profiles=replacements,
        records=records,
        parent_experiment_id=_PARENT_EXPERIMENT_ID,
        solver_hash=solver_hash,
        metric_contract_hash=contract.sha256,
        evaluation_hash=evaluation_hash,
        design_rule_sha256=str(design_rule["design_rule_sha256"]),
        record_manifest_sha256=str(design_rule["record_manifest_sha256"]),
        authorization_receipt=authorization,
    )

    source_budget = BudgetContract.approved_v2()
    target_budget = BudgetContract.approved_v3()
    exploration = ExplorationRegistry.zero_budget_v1()
    source_registry = AttemptRegistry.open(
        source_governance_dir / "attempt_registry.json",
        budget_contract=source_budget,
        exploration_registry=exploration,
    )
    plan = {
        "plan_version": "lyx_filter_stability_audit_revision_plan_v1",
        "status": "prepared_zero_new_runs",
        "parent_experiment_id": _PARENT_EXPERIMENT_ID,
        "source_v1_plan_sha256": source_plan["plan_sha256"],
        "source_v1_attempt_registry_sha256": file_sha256(
            source_governance_dir / "attempt_registry.json"
        ),
        "proposal_sha256": proposal["proposal_sha256"],
        "profile_library_sha256": proposal["library_sha256"],
        "design_rule_sha256": design_rule["design_rule_sha256"],
        "record_manifest_sha256": design_rule["record_manifest_sha256"],
        "audit_contract": contract.to_dict(),
        "audit_contract_sha256": contract.sha256,
        "solver_hash": solver_hash,
        "code_hash": code_hash,
        "evaluation_hash": evaluation_hash,
        "new_identity_count": 8,
        "new_identity_sha256": [identity.sha256 for identity in identities],
        "replacement_profiles": [asdict(profile) for profile in replacements],
        "profiles": [asdict(profile) for profile in profiles],
        "records": [asdict(record) for record in records],
        "reused_numeric_result_count": 24,
        "independent_bo_authorized": False,
        "produces_hr_tracking_trajectory": False,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    output_staging = output_dir.with_name(
        f".{output_dir.name}.{uuid.uuid4().hex}.staging"
    )
    preparation_transaction_id = uuid.uuid4().hex
    os.makedirs(_long_path(output_staging))
    _write_preparation_marker(
        output_staging,
        preparation_transaction_id,
        role="output",
    )
    atomic_write_json(output_staging / "profile_design_rule.json", design_rule)
    atomic_write_json(
        output_staging / "stability_audit_record_manifest.json",
        source_record_manifest,
    )
    for name in (
        "archive_profile_evidence_index.json",
        "filter_profile_library_freeze.json",
        "cached_metric_reclassification.json",
        "revision_proposal.json",
    ):
        atomic_write_json(output_staging / name, read_json(proposal_dir / name))
    atomic_write_json(output_staging / "stability_audit_revision_plan.json", plan)
    governance_receipt: dict[str, Any] = {}

    def finalize_governance(
        staging: Path,
        staged_registry: AttemptRegistry,
    ) -> None:
        nonlocal governance_receipt
        _write_preparation_marker(
            staging,
            preparation_transaction_id,
            role="governance",
        )
        atomic_write_json(
            staging / "budget_amendment_authorization.json",
            authorization,
        )
        atomic_write_json(staging / "budget_contract.json", target_budget.to_dict())
        atomic_write_json(
            staging / "exploration_registry.json",
            exploration.to_dict(),
        )
        governance_receipt = {
            "receipt_version": "lyx_recovery_governance_receipt_v3",
            "status": "prepared_zero_new_runs",
            "parent_experiment_id": _PARENT_EXPERIMENT_ID,
            "parent_governance_receipt_sha256": file_sha256(
                source_governance_dir / "governance_receipt.json"
            ),
            "planned_unique_identity_limit": target_budget.max_unique_identities,
            "normal_unique_identity_limit": (
                target_budget.normal_unique_identity_limit
            ),
            "worst_case_attempt_limit": target_budget.max_attempts,
            "filter_profile_stability_audit_unique_identities": 40,
            "new_replacement_unique_identities": 8,
            "attempt_registry_summary": staged_registry.summary(),
            "exploration_unique_budget": 0,
            "independent_bo_authorized": False,
            "artifacts": {
                name: file_sha256(staging / name)
                for name in (
                    "attempt_registry.json",
                    "budget_contract.json",
                    "exploration_registry.json",
                    "budget_amendment_authorization.json",
                )
            },
        }
        atomic_write_json(staging / "governance_receipt.json", governance_receipt)

    try:
        source_registry.migrate_to(
            governance_dir / "attempt_registry.json",
            budget_contract=target_budget,
            amendment_request=amendment_request,
            authorization_receipt=authorization,
            new_identities=identities,
            finalize_staging=finalize_governance,
        )
        os.replace(_long_path(output_staging), _long_path(output_dir))
        _commit_preparation_pair(
            output_dir,
            governance_dir,
            preparation_transaction_id,
        )
    except BaseException:
        if output_staging.exists():
            shutil.rmtree(_long_path(output_staging))
        if governance_dir.exists():
            shutil.rmtree(_long_path(governance_dir))
        raise
    return {"plan": plan, "governance_receipt": governance_receipt}


def execute_filter_audit_revision(
    *,
    output_dir: Path,
    governance_dir: Path,
) -> dict[str, Any]:
    """Run only the eight approved replacement-profile diagnostics."""

    _require_committed_preparation_pair(output_dir, governance_dir)
    plan = read_json(output_dir / "stability_audit_revision_plan.json")
    _verify_embedded_sha256(plan, "plan_sha256")
    if plan.get("status") != "prepared_zero_new_runs":
        raise StabilityAuditError("filter_audit_revision_plan_not_prepared")
    if plan.get("solver_hash") != _current_solver_hash():
        raise StabilityAuditError("filter_audit_solver_source_changed_after_freeze")
    if plan.get("code_hash") != _current_code_hash():
        raise StabilityAuditError("filter_audit_code_source_changed_after_freeze")
    replacements = tuple(_profile_from_dict(item) for item in plan["replacement_profiles"])
    profiles = tuple(_profile_from_dict(item) for item in plan["profiles"])
    records = tuple(FilterAuditRecord(**item) for item in plan["records"])
    contract = StabilityAuditContract(**plan["audit_contract"])
    authorization = read_json(
        governance_dir / "budget_amendment_authorization.json"
    )
    identities = plan_replacement_filter_audit_identities(
        profiles=replacements,
        records=records,
        parent_experiment_id=str(plan["parent_experiment_id"]),
        solver_hash=str(plan["solver_hash"]),
        metric_contract_hash=str(plan["audit_contract_sha256"]),
        evaluation_hash=str(plan["evaluation_hash"]),
        design_rule_sha256=str(plan["design_rule_sha256"]),
        record_manifest_sha256=str(plan["record_manifest_sha256"]),
        authorization_receipt=authorization,
    )
    if [identity.sha256 for identity in identities] != plan["new_identity_sha256"]:
        raise StabilityAuditError("replacement_audit_plan_identity_mismatch")
    registry = AttemptRegistry.open(
        governance_dir / "attempt_registry.json",
        budget_contract=BudgetContract.approved_v3(),
        exploration_registry=ExplorationRegistry.zero_budget_v1(),
    )
    replacement_results = _execute_registered_profile_audits(
        registry=registry,
        identities=identities,
        profiles=replacements,
        records=records,
        contract=contract,
        output_dir=output_dir,
        cache_prefix="replacement",
    )

    reclassification = read_json(
        output_dir / "cached_metric_reclassification.json"
    )
    all_results: dict[str, list[dict[str, Any]]] = {
        profile.profile_id: [] for profile in profiles
    }
    for item in reclassification["records"]:
        all_results[str(item["profile_id"])].append(item)
    all_results.update(replacement_results)
    library = read_json(output_dir / "filter_profile_library_freeze.json")
    receipts: list[dict[str, Any]] = []
    for profile in profiles:
        receipt = build_filter_profile_receipt(
            profile,
            all_results[profile.profile_id],
            audit_contract=contract,
            library_sha256=str(library["library_sha256"]),
            solver_hash=str(plan["solver_hash"]),
            code_hash=str(plan["code_hash"]),
            evaluation_hash=str(plan["evaluation_hash"]),
            design_rule_sha256=str(plan["design_rule_sha256"]),
            record_manifest_sha256=str(plan["record_manifest_sha256"]),
        )
        receipts.append(receipt)
        atomic_write_json(
            output_dir / "filter_profile_receipts" / f"{profile.profile_id}.json",
            receipt,
        )
    eligible = [
        str(receipt["profile_id"])
        for receipt in receipts
        if receipt["may_enter_formal_matrix"] is True
    ]
    final = {
        "receipt_version": "lyx_filter_profile_audit_completion_v2",
        "status": "complete" if len(eligible) == 8 else "incomplete_profile_library",
        "profile_count": 8,
        "eligible_profile_count": len(eligible),
        "eligible_profile_ids": eligible,
        "rejected_profile_ids": [
            str(receipt["profile_id"])
            for receipt in receipts
            if receipt["may_enter_formal_matrix"] is not True
        ],
        "recovery_sentinels": library["recovery_sentinels"],
        "all_recovery_sentinels_eligible": all(
            profile_id in eligible
            for profile_id in library["recovery_sentinels"].values()
        ),
        "attempt_registry_summary": registry.summary(),
        "new_replacement_run_count": 8,
        "reused_numeric_result_count": 24,
        "actual_hr_tracking_trajectory_count": 0,
        "independent_bo_run_count": 0,
        "exploration_run_count": 0,
        "profile_receipt_sha256": {
            str(receipt["profile_id"]): str(receipt["receipt_sha256"])
            for receipt in receipts
        },
    }
    final["completion_sha256"] = _canonical_sha256(final)
    atomic_write_json(output_dir / "filter_profile_audit_completion.json", final)
    governance_receipt_path = governance_dir / "governance_receipt.json"
    governance_receipt = read_json(governance_receipt_path)
    governance_receipt.update(
        {
            "status": "complete",
            "attempt_registry_summary": registry.summary(),
            "filter_profile_audit_completion_sha256": file_sha256(
                output_dir / "filter_profile_audit_completion.json"
            ),
        }
    )
    governance_receipt["artifacts"]["attempt_registry.json"] = file_sha256(
        governance_dir / "attempt_registry.json"
    )
    atomic_write_json(governance_receipt_path, governance_receipt)
    return final


def reclassify_filter_audit_to_frozen_spec(
    *,
    output_dir: Path,
    governance_dir: Path,
) -> dict[str, Any]:
    """Restore the frozen 0.80 spectral gate without running the solver."""

    plan = read_json(output_dir / "stability_audit_revision_plan.json")
    source_completion_path = output_dir / "filter_profile_audit_completion.json"
    source_completion = read_json(source_completion_path)
    if source_completion.get("independent_bo_run_count") != 0:
        raise StabilityAuditError("cannot_reclassify_audit_with_independent_bo_runs")
    profiles = tuple(_profile_from_dict(item) for item in plan["profiles"])
    contract = StabilityAuditContract.corrected_v2()
    source_contract_sha256 = str(plan["audit_contract_sha256"])
    cached = read_json(output_dir / "cached_metric_reclassification.json")
    cached_by_profile: dict[str, list[dict[str, Any]]] = {}
    for raw in cached["records"]:
        cached_by_profile.setdefault(str(raw["profile_id"]), []).append(dict(raw))

    all_results: dict[str, list[dict[str, Any]]] = {}
    reclassified_records: list[dict[str, Any]] = []
    for profile in profiles:
        source_records = cached_by_profile.get(profile.profile_id)
        if source_records is None:
            source_records = [
                {
                    **read_json(
                        output_dir
                        / "record_audits"
                        / profile.profile_id
                        / f"{record['record_id']}.json"
                    ),
                    "profile_id": profile.profile_id,
                }
                for record in plan["records"]
            ]
        revised_records = [
            reclassify_cached_record_audit(
                dict(record),
                corrected_contract=contract,
                source_metric_contract_sha256=str(
                    record.get("audit_contract_sha256", source_contract_sha256)
                ),
                source_result_sha256=str(record["result_sha256"]),
                reclassification_reason=(
                    "restore_frozen_pulse_power_retention_gate"
                ),
            )
            for record in source_records
        ]
        all_results[profile.profile_id] = revised_records
        reclassified_records.extend(revised_records)

    evaluation_hash = _canonical_sha256(
        {
            "design_rule_sha256": plan["design_rule_sha256"],
            "record_manifest_sha256": plan["record_manifest_sha256"],
            "audit_contract_sha256": contract.sha256,
            "source_evaluation_hash": plan["evaluation_hash"],
            "source_completion_sha256": source_completion["completion_sha256"],
            "classification_only": True,
            "produces_hr_tracking_trajectory": False,
            "uses_bayesian_optimization": False,
        }
    )
    reclassification = {
        "receipt_version": "lyx_filter_spec_gate_reclassification_v1",
        "status": "complete_zero_new_runs",
        "source_completion_sha256": source_completion["completion_sha256"],
        "source_audit_contract_sha256": source_contract_sha256,
        "corrected_audit_contract": contract.to_dict(),
        "corrected_audit_contract_sha256": contract.sha256,
        "source_evaluation_hash": plan["evaluation_hash"],
        "evaluation_hash": evaluation_hash,
        "record_count": len(reclassified_records),
        "numerical_rerun_count": 0,
        "independent_bo_run_count": 0,
        "records": reclassified_records,
    }
    reclassification["receipt_sha256"] = _canonical_sha256(reclassification)
    atomic_write_json(
        output_dir / "spec_gate_reclassification.json",
        reclassification,
    )

    library = read_json(output_dir / "filter_profile_library_freeze.json")
    code_hash = _current_code_hash()
    receipts: list[dict[str, Any]] = []
    for profile in profiles:
        receipt = build_filter_profile_receipt(
            profile,
            all_results[profile.profile_id],
            audit_contract=contract,
            library_sha256=str(library["library_sha256"]),
            solver_hash=str(plan["solver_hash"]),
            code_hash=code_hash,
            evaluation_hash=evaluation_hash,
            design_rule_sha256=str(plan["design_rule_sha256"]),
            record_manifest_sha256=str(plan["record_manifest_sha256"]),
        )
        receipts.append(receipt)
        atomic_write_json(
            output_dir / "filter_profile_receipts" / f"{profile.profile_id}.json",
            receipt,
        )
    eligible = [
        str(receipt["profile_id"])
        for receipt in receipts
        if receipt["may_enter_formal_matrix"] is True
    ]
    sentinels = {
        str(profile.recovery_sentinel_role): profile.profile_id
        for profile in profiles
        if profile.recovery_sentinel_role is not None
    }
    registry = AttemptRegistry.open(
        governance_dir / "attempt_registry.json",
        budget_contract=BudgetContract.approved_v3(),
        exploration_registry=ExplorationRegistry.zero_budget_v1(),
    )
    final = {
        "receipt_version": "lyx_filter_profile_audit_completion_v3",
        "status": (
            "complete" if len(eligible) == len(profiles)
            else "blocked_insufficient_eligible_profiles"
        ),
        "profile_count": len(profiles),
        "eligible_profile_count": len(eligible),
        "eligible_profile_ids": eligible,
        "rejected_profile_ids": sorted(
            profile.profile_id
            for profile in profiles
            if profile.profile_id not in eligible
        ),
        "recovery_sentinels": sentinels,
        "all_recovery_sentinels_eligible": all(
            profile_id in eligible for profile_id in sentinels.values()
        ),
        "audit_contract_sha256": contract.sha256,
        "evaluation_hash": evaluation_hash,
        "attempt_registry_summary": registry.summary(),
        "new_replacement_run_count": 8,
        "reused_numeric_result_count": 24,
        "spec_gate_reclassified_result_count": len(reclassified_records),
        "spec_gate_reclassification_numerical_rerun_count": 0,
        "actual_hr_tracking_trajectory_count": 0,
        "independent_bo_run_count": 0,
        "exploration_run_count": 0,
        "profile_receipt_sha256": {
            str(receipt["profile_id"]): str(receipt["receipt_sha256"])
            for receipt in receipts
        },
    }
    final["completion_sha256"] = _canonical_sha256(final)
    atomic_write_json(source_completion_path, final)

    governance_receipt_path = governance_dir / "governance_receipt.json"
    governance_receipt = read_json(governance_receipt_path)
    governance_receipt.update(
        {
            "status": (
                "complete"
                if final["status"] == "complete"
                else "blocked_filter_profile_library"
            ),
            "attempt_registry_summary": registry.summary(),
            "filter_profile_audit_completion_sha256": file_sha256(
                source_completion_path
            ),
            "spec_gate_reclassification_sha256": file_sha256(
                output_dir / "spec_gate_reclassification.json"
            ),
        }
    )
    atomic_write_json(governance_receipt_path, governance_receipt)
    return final


def prepare_spec_gate_supplement(
    *,
    source_output_dir: Path,
    source_governance_dir: Path,
    proposal_dir: Path,
    output_dir: Path,
    governance_dir: Path,
    authorization_receipt_path: Path,
) -> dict[str, Any]:
    """Register the 24-identity supplement only after an external approval receipt."""

    if output_dir.exists() or governance_dir.exists():
        raise StabilityAuditError("spec_gate_supplement_output_already_exists")
    proposal = read_json(proposal_dir / "proposal.json")
    design_rule = read_json(proposal_dir / "profile_design_rule.json")
    evidence = read_json(proposal_dir / "archive_candidate_evidence.json")
    budget_request = read_json(proposal_dir / "budget_amendment_request.json")
    for payload, field in (
        (proposal, "proposal_sha256"),
        (design_rule, "design_rule_sha256"),
        (evidence, "evidence_sha256"),
        (budget_request, "request_sha256"),
    ):
        _verify_embedded_sha256(payload, field)
    if (
        proposal.get("status") != "awaiting_human_budget_decision"
        or proposal.get("may_execute") is not False
        or budget_request.get("approved") is not False
        or proposal.get("design_rule_sha256")
        != design_rule.get("design_rule_sha256")
        or proposal.get("archive_candidate_evidence_sha256")
        != evidence.get("evidence_sha256")
        or proposal.get("budget_request_sha256")
        != budget_request.get("request_sha256")
    ):
        raise StabilityAuditError("invalid_spec_gate_supplement_proposal")

    authorization = read_json(authorization_receipt_path)
    if authorization.get("proposal_sha256") != proposal.get("proposal_sha256"):
        raise StabilityAuditError("budget_authorization_proposal_mismatch")
    amendment_request = BudgetAmendmentRequest(
        stage="filter_profile_stability_audit",
        profile_design_rule_hash=str(design_rule["design_rule_sha256"]),
        record_manifest_hash=str(design_rule["record_manifest_sha256"]),
        added_unique_identities=24,
        normal_unique_identity_limit=736,
        max_unique_identities=748,
        max_attempts=1496,
    )
    validate_budget_amendment_authorization(
        amendment_request,
        receipt=authorization,
    )
    _require_committed_preparation_pair(
        source_output_dir,
        source_governance_dir,
    )

    source_plan = read_json(source_output_dir / "stability_audit_revision_plan.json")
    source_completion = read_json(
        source_output_dir / "filter_profile_audit_completion.json"
    )
    source_record_manifest = read_json(
        source_output_dir / "stability_audit_record_manifest.json"
    )
    source_archive_evidence = read_json(
        source_output_dir / "archive_profile_evidence_index.json"
    )
    source_reclassification = read_json(
        source_output_dir / "spec_gate_reclassification.json"
    )
    _validate_spec_gate_source_artifacts(
        source_plan=source_plan,
        source_completion=source_completion,
        source_record_manifest=source_record_manifest,
        proposal=proposal,
        design_rule=design_rule,
    )
    approved_source_bindings = {
        "source_plan_artifact_sha256": _canonical_sha256(source_plan),
        "source_completion_artifact_sha256": _canonical_sha256(
            source_completion
        ),
        "source_archive_evidence_artifact_sha256": _canonical_sha256(
            source_archive_evidence
        ),
        "source_reclassification_artifact_sha256": _canonical_sha256(
            source_reclassification
        ),
    }
    if any(
        proposal.get(field) != artifact_sha256
        for field, artifact_sha256 in approved_source_bindings.items()
    ):
        raise StabilityAuditError("spec_gate_supplement_source_state_mismatch")
    records = tuple(
        FilterAuditRecord(
            record_id=str(item["record_id"]),
            scene=str(item["scene"]),
            data_path=str(item["data_path"]),
            reference_path=str(item["reference_path"]),
            data_sha256=str(item["data_sha256"]),
            reference_sha256=str(item["reference_sha256"]),
        )
        for item in source_record_manifest["records"]
    )
    profiles = spec_gate_supplement_profiles_v1()
    expected_coordinates = [
        {
            "profile_id": profile.profile_id,
            "design_role": profile.design_role,
            "fs_target": profile.fs_target,
            "memory_ms": profile.memory_ms,
            "nominal_mu": float(profile.nominal_mu),
        }
        for profile in profiles
    ]
    if design_rule.get("candidate_profiles") != expected_coordinates:
        raise StabilityAuditError("spec_gate_supplement_profile_mismatch")
    contract = StabilityAuditContract.corrected_v2()
    if proposal.get("audit_contract_sha256") != contract.sha256:
        raise StabilityAuditError("spec_gate_supplement_contract_mismatch")
    solver_hash = _current_solver_hash()
    code_hash = _current_code_hash()
    evaluation_hash = _canonical_sha256(
        {
            "design_rule_sha256": design_rule["design_rule_sha256"],
            "record_manifest_sha256": design_rule["record_manifest_sha256"],
            "audit_contract_sha256": contract.sha256,
            "source_completion_sha256": source_completion["completion_sha256"],
            "selection_rule": design_rule["selection_rule"],
            "produces_hr_tracking_trajectory": False,
            "uses_bayesian_optimization": False,
        }
    )
    identities = plan_spec_gate_supplement_identities(
        profiles=profiles,
        records=records,
        parent_experiment_id=_PARENT_EXPERIMENT_ID,
        solver_hash=solver_hash,
        metric_contract_hash=contract.sha256,
        evaluation_hash=evaluation_hash,
        design_rule_sha256=str(design_rule["design_rule_sha256"]),
        record_manifest_sha256=str(design_rule["record_manifest_sha256"]),
        authorization_receipt=authorization,
    )
    plan = {
        "plan_version": "lyx_filter_spec_gate_supplement_plan_v1",
        "status": "prepared_zero_new_runs",
        "parent_experiment_id": _PARENT_EXPERIMENT_ID,
        "source_plan_sha256": source_plan["plan_sha256"],
        "source_completion_sha256": source_completion["completion_sha256"],
        "frozen_source_plan_sha256": _canonical_sha256(source_plan),
        "frozen_source_completion_sha256": _canonical_sha256(source_completion),
        "frozen_source_archive_evidence_sha256": _canonical_sha256(
            source_archive_evidence
        ),
        "frozen_source_reclassification_sha256": _canonical_sha256(
            source_reclassification
        ),
        "proposal_sha256": proposal["proposal_sha256"],
        "design_rule_sha256": design_rule["design_rule_sha256"],
        "record_manifest_sha256": design_rule["record_manifest_sha256"],
        "audit_contract": contract.to_dict(),
        "audit_contract_sha256": contract.sha256,
        "solver_hash": solver_hash,
        "code_hash": code_hash,
        "evaluation_hash": evaluation_hash,
        "new_identity_count": 24,
        "new_identity_sha256": [identity.sha256 for identity in identities],
        "candidate_profiles": [asdict(profile) for profile in profiles],
        "records": [asdict(record) for record in records],
        "source_eligible_profile_ids": source_completion["eligible_profile_ids"],
        "selection_rule": design_rule["selection_rule"],
        "sentinel_rule": design_rule["sentinel_rule"],
        "independent_bo_authorized": False,
        "produces_hr_tracking_trajectory": False,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)

    source_budget = BudgetContract.approved_v3()
    target_budget = BudgetContract.approved_v4()
    exploration = ExplorationRegistry.zero_budget_v1()
    source_registry = AttemptRegistry.open(
        source_governance_dir / "attempt_registry.json",
        budget_contract=source_budget,
        exploration_registry=exploration,
    )
    output_staging = output_dir.with_name(
        f".{output_dir.name}.{uuid.uuid4().hex}.staging"
    )
    preparation_transaction_id = uuid.uuid4().hex
    governance_receipt: dict[str, Any] = {}

    def finalize_governance(
        staging: Path,
        staged_registry: AttemptRegistry,
    ) -> None:
        nonlocal governance_receipt
        _write_preparation_marker(
            staging,
            preparation_transaction_id,
            role="governance",
        )
        atomic_write_json(
            staging / "budget_amendment_authorization.json",
            authorization,
        )
        atomic_write_json(staging / "budget_contract.json", target_budget.to_dict())
        atomic_write_json(
            staging / "exploration_registry.json",
            exploration.to_dict(),
        )
        governance_receipt = {
            "receipt_version": "lyx_recovery_governance_receipt_v4",
            "status": "prepared_zero_new_runs",
            "parent_experiment_id": _PARENT_EXPERIMENT_ID,
            "parent_governance_receipt_sha256": file_sha256(
                source_governance_dir / "governance_receipt.json"
            ),
            "planned_unique_identity_limit": target_budget.max_unique_identities,
            "normal_unique_identity_limit": (
                target_budget.normal_unique_identity_limit
            ),
            "worst_case_attempt_limit": target_budget.max_attempts,
            "filter_profile_stability_audit_unique_identities": 64,
            "new_spec_gate_supplement_unique_identities": 24,
            "attempt_registry_summary": staged_registry.summary(),
            "exploration_unique_budget": 0,
            "independent_bo_authorized": False,
            "artifacts": {
                name: file_sha256(staging / name)
                for name in (
                    "attempt_registry.json",
                    "budget_contract.json",
                    "exploration_registry.json",
                    "budget_amendment_authorization.json",
                )
            },
        }
        atomic_write_json(staging / "governance_receipt.json", governance_receipt)

    try:
        os.makedirs(_long_path(output_staging))
        _write_preparation_marker(
            output_staging,
            preparation_transaction_id,
            role="output",
        )
        for name in (
            "proposal.json",
            "profile_design_rule.json",
            "archive_candidate_evidence.json",
            "budget_amendment_request.json",
        ):
            atomic_write_json(output_staging / name, read_json(proposal_dir / name))
        atomic_write_json(
            output_staging / "stability_audit_record_manifest.json",
            source_record_manifest,
        )
        for name, payload in (
            ("frozen_source_revision_plan.json", source_plan),
            ("frozen_source_completion.json", source_completion),
            ("frozen_source_archive_evidence.json", source_archive_evidence),
            ("frozen_source_reclassification.json", source_reclassification),
        ):
            atomic_write_json(output_staging / name, payload)
        atomic_write_json(
            output_staging / "spec_gate_supplement_plan.json",
            plan,
        )
        source_registry.migrate_to(
            governance_dir / "attempt_registry.json",
            budget_contract=target_budget,
            amendment_request=amendment_request,
            authorization_receipt=authorization,
            new_identities=identities,
            finalize_staging=finalize_governance,
        )
        os.replace(_long_path(output_staging), _long_path(output_dir))
        _commit_preparation_pair(
            output_dir,
            governance_dir,
            preparation_transaction_id,
        )
    except BaseException:
        if output_staging.exists():
            shutil.rmtree(_long_path(output_staging))
        if governance_dir.exists():
            shutil.rmtree(_long_path(governance_dir))
        raise
    return {"plan": plan, "governance_receipt": governance_receipt}


def execute_spec_gate_supplement(
    *,
    source_output_dir: Path,
    output_dir: Path,
    governance_dir: Path,
) -> dict[str, Any]:
    """Run only the 24 approved diagnostics and apply the frozen selector."""

    _require_committed_preparation_pair(output_dir, governance_dir)
    plan = read_json(output_dir / "spec_gate_supplement_plan.json")
    _verify_embedded_sha256(plan, "plan_sha256")
    if plan.get("status") != "prepared_zero_new_runs":
        raise StabilityAuditError("spec_gate_supplement_plan_not_prepared")
    if plan.get("solver_hash") != _current_solver_hash():
        raise StabilityAuditError("filter_audit_solver_source_changed_after_freeze")
    if plan.get("code_hash") != _current_code_hash():
        raise StabilityAuditError("filter_audit_code_source_changed_after_freeze")
    frozen_source_plan = read_json(
        output_dir / "frozen_source_revision_plan.json"
    )
    source_completion = read_json(output_dir / "frozen_source_completion.json")
    frozen_source_evidence = read_json(
        output_dir / "frozen_source_archive_evidence.json"
    )
    frozen_source_reclassification = read_json(
        output_dir / "frozen_source_reclassification.json"
    )
    proposal = read_json(output_dir / "proposal.json")
    design_rule = read_json(output_dir / "profile_design_rule.json")
    candidate_evidence = read_json(
        output_dir / "archive_candidate_evidence.json"
    )
    authorization = read_json(
        governance_dir / "budget_amendment_authorization.json"
    )
    _validate_frozen_spec_gate_sources(
        plan=plan,
        proposal=proposal,
        authorization=authorization,
        design_rule=design_rule,
        candidate_evidence=candidate_evidence,
        source_plan=frozen_source_plan,
        source_completion=source_completion,
        source_evidence=frozen_source_evidence,
        source_reclassification=frozen_source_reclassification,
    )
    profiles = tuple(_profile_from_dict(item) for item in plan["candidate_profiles"])
    records = tuple(FilterAuditRecord(**item) for item in plan["records"])
    contract = StabilityAuditContract(**plan["audit_contract"])
    _validate_spec_gate_audit_contract(
        contract,
        plan=plan,
        proposal=proposal,
    )
    identities = plan_spec_gate_supplement_identities(
        profiles=profiles,
        records=records,
        parent_experiment_id=str(plan["parent_experiment_id"]),
        solver_hash=str(plan["solver_hash"]),
        metric_contract_hash=str(plan["audit_contract_sha256"]),
        evaluation_hash=str(plan["evaluation_hash"]),
        design_rule_sha256=str(plan["design_rule_sha256"]),
        record_manifest_sha256=str(plan["record_manifest_sha256"]),
        authorization_receipt=authorization,
    )
    if [identity.sha256 for identity in identities] != plan["new_identity_sha256"]:
        raise StabilityAuditError("spec_gate_supplement_plan_identity_mismatch")
    registry = AttemptRegistry.open(
        governance_dir / "attempt_registry.json",
        budget_contract=BudgetContract.approved_v4(),
        exploration_registry=ExplorationRegistry.zero_budget_v1(),
    )
    candidate_results = _execute_registered_profile_audits(
        registry=registry,
        identities=identities,
        profiles=profiles,
        records=records,
        contract=contract,
        output_dir=output_dir,
        cache_prefix="spec-gate",
    )
    candidate_set_sha256 = _canonical_sha256(
        {
            "design_rule_sha256": plan["design_rule_sha256"],
            "candidate_profiles": plan["candidate_profiles"],
        }
    )
    candidate_receipts: list[dict[str, Any]] = []
    for profile in profiles:
        receipt = build_filter_profile_receipt(
            profile,
            candidate_results[profile.profile_id],
            audit_contract=contract,
            library_sha256=candidate_set_sha256,
            solver_hash=str(plan["solver_hash"]),
            code_hash=str(plan["code_hash"]),
            evaluation_hash=str(plan["evaluation_hash"]),
            design_rule_sha256=str(plan["design_rule_sha256"]),
            record_manifest_sha256=str(plan["record_manifest_sha256"]),
        )
        candidate_receipts.append(receipt)
        atomic_write_json(
            output_dir / "candidate_profile_receipts" / f"{profile.profile_id}.json",
            receipt,
        )
    selection = select_spec_gate_supplement_profiles(candidate_receipts)
    atomic_write_json(output_dir / "supplement_selection_receipt.json", selection)

    final_library: dict[str, Any] | None = None
    final_receipts: list[dict[str, Any]] = []
    if selection["status"] == "complete":
        source_profiles = {
            profile.profile_id: profile
            for profile in (
                _profile_from_dict(item)
                for item in frozen_source_plan["profiles"]
            )
        }
        existing_profiles = [
            source_profiles[profile_id]
            for profile_id in plan["source_eligible_profile_ids"]
        ]
        candidate_by_id = {profile.profile_id: profile for profile in profiles}
        selected_p50 = [
            candidate_by_id[profile_id]
            for profile_id in selection["selected_p50_profile_ids"]
        ]
        selected_p100 = [
            candidate_by_id[profile_id]
            for profile_id in selection["selected_p100_profile_ids"]
        ]
        existing_profiles = [
            replace(
                profile,
                recovery_sentinel_role=(
                    "conservative"
                    if profile.profile_id == "p50-short-low"
                    else None
                ),
            )
            for profile in existing_profiles
        ]
        selected_p50 = [
            replace(
                profile,
                recovery_sentinel_role="intermediate" if index == 0 else None,
            )
            for index, profile in enumerate(selected_p50)
        ]
        selected_p100 = [
            replace(
                profile,
                recovery_sentinel_role="aggressive" if index == 0 else None,
            )
            for index, profile in enumerate(selected_p100)
        ]
        final_profiles = tuple(
            sorted(
                [*existing_profiles, *selected_p50, *selected_p100],
                key=lambda profile: (
                    profile.fs_target,
                    profile.memory_ms,
                    float(profile.nominal_mu),
                    profile.profile_id,
                ),
            )
        )
        source_evidence = frozen_source_evidence
        supplement_evidence = read_json(
            output_dir / "archive_candidate_evidence.json"
        )
        evidence_by_coordinate: dict[
            tuple[int, int, float],
            ArchivedProfileEvidence,
        ] = {}
        for item in source_evidence["profiles"]:
            coordinate = item["coordinate"]
            evidence_by_coordinate[
                (
                    int(coordinate["fs_target"]),
                    int(coordinate["memory_ms"]),
                    float(coordinate["nominal_mu"]),
                )
            ] = ArchivedProfileEvidence(
                fs_target=int(coordinate["fs_target"]),
                memory_ms=int(coordinate["memory_ms"]),
                nominal_mu=float(coordinate["nominal_mu"]),
                occurrence_count=int(item["occurrence_count"]),
                scenes=tuple(str(scene) for scene in item["scenes"]),
                archive_manifest_sha256=str(
                    source_evidence["baseline_manifest_sha256"]
                ),
                archive_table_sha256=str(
                    source_evidence["evidence_index_sha256"]
                ),
            )
        for item in supplement_evidence["candidate_profiles"]:
            evidence_by_coordinate[
                (
                    int(item["fs_target"]),
                    int(item["memory_ms"]),
                    float(item["nominal_mu"]),
                )
            ] = ArchivedProfileEvidence(
                fs_target=int(item["fs_target"]),
                memory_ms=int(item["memory_ms"]),
                nominal_mu=float(item["nominal_mu"]),
                occurrence_count=int(item["archive_occurrence_count"]),
                scenes=tuple(str(scene) for scene in item["archive_scenes"]),
                archive_manifest_sha256=str(
                    supplement_evidence["baseline_manifest_sha256"]
                ),
                archive_table_sha256=str(
                    supplement_evidence["evidence_sha256"]
                ),
            )
        final_library = freeze_filter_profile_library(
            final_profiles,
            tuple(
                evidence_by_coordinate[profile.coordinate]
                for profile in final_profiles
            ),
            design_rule_sha256=str(plan["design_rule_sha256"]),
        )
        atomic_write_json(
            output_dir / "filter_profile_library_freeze.json",
            final_library,
        )
        source_reclassification = frozen_source_reclassification
        final_results: dict[str, list[dict[str, Any]]] = {
            profile.profile_id: [] for profile in final_profiles
        }
        for item in source_reclassification["records"]:
            profile_id = str(item["profile_id"])
            if profile_id in final_results:
                final_results[profile_id].append(item)
        for profile in [*selected_p50, *selected_p100]:
            final_results[profile.profile_id] = candidate_results[profile.profile_id]
        for profile in final_profiles:
            receipt = build_filter_profile_receipt(
                profile,
                final_results[profile.profile_id],
                audit_contract=contract,
                library_sha256=str(final_library["library_sha256"]),
                solver_hash=str(plan["solver_hash"]),
                code_hash=str(plan["code_hash"]),
                evaluation_hash=str(plan["evaluation_hash"]),
                design_rule_sha256=str(plan["design_rule_sha256"]),
                record_manifest_sha256=str(plan["record_manifest_sha256"]),
            )
            final_receipts.append(receipt)
            atomic_write_json(
                output_dir / "filter_profile_receipts" / f"{profile.profile_id}.json",
                receipt,
            )

    final = {
        "receipt_version": "lyx_filter_spec_gate_supplement_completion_v1",
        "status": (
            "complete"
            if selection["status"] == "complete"
            else "blocked_insufficient_eligible_profiles"
        ),
        "candidate_profile_count": 6,
        "candidate_eligible_profile_ids": sorted(
            str(receipt["profile_id"])
            for receipt in candidate_receipts
            if receipt["may_enter_formal_matrix"] is True
        ),
        "selection": selection,
        "final_profile_count": len(final_receipts),
        "final_profile_ids": [
            str(receipt["profile_id"]) for receipt in final_receipts
        ],
        "final_library_sha256": (
            None if final_library is None else final_library["library_sha256"]
        ),
        "attempt_registry_summary": registry.summary(),
        "new_spec_gate_supplement_run_count": 24,
        "actual_hr_tracking_trajectory_count": 0,
        "independent_bo_run_count": 0,
        "exploration_run_count": 0,
        "candidate_profile_receipt_sha256": {
            str(receipt["profile_id"]): str(receipt["receipt_sha256"])
            for receipt in candidate_receipts
        },
        "final_profile_receipt_sha256": {
            str(receipt["profile_id"]): str(receipt["receipt_sha256"])
            for receipt in final_receipts
        },
    }
    final["completion_sha256"] = _canonical_sha256(final)
    completion_path = output_dir / "spec_gate_supplement_completion.json"
    atomic_write_json(completion_path, final)
    governance_receipt_path = governance_dir / "governance_receipt.json"
    governance_receipt = read_json(governance_receipt_path)
    governance_receipt.update(
        {
            "status": (
                "complete"
                if final["status"] == "complete"
                else "blocked_filter_profile_library"
            ),
            "attempt_registry_summary": registry.summary(),
            "spec_gate_supplement_completion_sha256": file_sha256(
                completion_path
            ),
        }
    )
    governance_receipt["artifacts"]["attempt_registry.json"] = file_sha256(
        governance_dir / "attempt_registry.json"
    )
    atomic_write_json(governance_receipt_path, governance_receipt)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--archive-root", type=Path, required=True)
    prepare.add_argument("--baseline-manifest", type=Path, required=True)
    prepare.add_argument("--metrics-table", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--governance-dir", type=Path, required=True)
    prepare.add_argument("--authorization-receipt", type=Path, required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--governance-dir", type=Path, required=True)
    proposal = subparsers.add_parser("propose-revision")
    proposal.add_argument("--archive-root", type=Path, required=True)
    proposal.add_argument("--baseline-manifest", type=Path, required=True)
    proposal.add_argument("--source-output-dir", type=Path, required=True)
    proposal.add_argument("--proposal-dir", type=Path, required=True)
    prepare_revision = subparsers.add_parser("prepare-revision")
    prepare_revision.add_argument("--source-output-dir", type=Path, required=True)
    prepare_revision.add_argument("--source-governance-dir", type=Path, required=True)
    prepare_revision.add_argument("--proposal-dir", type=Path, required=True)
    prepare_revision.add_argument("--output-dir", type=Path, required=True)
    prepare_revision.add_argument("--governance-dir", type=Path, required=True)
    prepare_revision.add_argument("--authorization-receipt", type=Path, required=True)
    execute_revision = subparsers.add_parser("execute-revision")
    execute_revision.add_argument("--output-dir", type=Path, required=True)
    execute_revision.add_argument("--governance-dir", type=Path, required=True)
    reclassify = subparsers.add_parser("reclassify-spec-gate")
    reclassify.add_argument("--output-dir", type=Path, required=True)
    reclassify.add_argument("--governance-dir", type=Path, required=True)
    prepare_supplement = subparsers.add_parser("prepare-spec-gate-supplement")
    prepare_supplement.add_argument("--source-output-dir", type=Path, required=True)
    prepare_supplement.add_argument("--source-governance-dir", type=Path, required=True)
    prepare_supplement.add_argument("--proposal-dir", type=Path, required=True)
    prepare_supplement.add_argument("--output-dir", type=Path, required=True)
    prepare_supplement.add_argument("--governance-dir", type=Path, required=True)
    prepare_supplement.add_argument(
        "--authorization-receipt",
        type=Path,
        required=True,
    )
    execute_supplement = subparsers.add_parser("execute-spec-gate-supplement")
    execute_supplement.add_argument("--source-output-dir", type=Path, required=True)
    execute_supplement.add_argument("--output-dir", type=Path, required=True)
    execute_supplement.add_argument("--governance-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_filter_profile_audit(
            archive_root=args.archive_root,
            baseline_manifest_path=args.baseline_manifest,
            metrics_table_path=args.metrics_table,
            output_dir=args.output_dir,
            governance_dir=args.governance_dir,
            authorization_receipt_path=args.authorization_receipt,
        )
    elif args.command == "execute":
        result = execute_filter_profile_audit(
            output_dir=args.output_dir,
            governance_dir=args.governance_dir,
        )
    elif args.command == "propose-revision":
        result = build_filter_audit_revision_proposal(
            archive_root=args.archive_root,
            baseline_manifest_path=args.baseline_manifest,
            source_output_dir=args.source_output_dir,
            proposal_dir=args.proposal_dir,
        )
    elif args.command == "prepare-revision":
        result = prepare_filter_audit_revision(
            source_output_dir=args.source_output_dir,
            source_governance_dir=args.source_governance_dir,
            proposal_dir=args.proposal_dir,
            output_dir=args.output_dir,
            governance_dir=args.governance_dir,
            authorization_receipt_path=args.authorization_receipt,
        )
    elif args.command == "execute-revision":
        result = execute_filter_audit_revision(
            output_dir=args.output_dir,
            governance_dir=args.governance_dir,
        )
    elif args.command == "reclassify-spec-gate":
        result = reclassify_filter_audit_to_frozen_spec(
            output_dir=args.output_dir,
            governance_dir=args.governance_dir,
        )
    elif args.command == "prepare-spec-gate-supplement":
        result = prepare_spec_gate_supplement(
            source_output_dir=args.source_output_dir,
            source_governance_dir=args.source_governance_dir,
            proposal_dir=args.proposal_dir,
            output_dir=args.output_dir,
            governance_dir=args.governance_dir,
            authorization_receipt_path=args.authorization_receipt,
        )
    else:
        result = execute_spec_gate_supplement(
            source_output_dir=args.source_output_dir,
            output_dir=args.output_dir,
            governance_dir=args.governance_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
