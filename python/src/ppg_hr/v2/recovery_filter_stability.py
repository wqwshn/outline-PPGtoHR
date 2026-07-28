"""Bounded LMS stability and spectral audit used to qualify frozen profiles."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy.signal import periodogram

from ppg_hr.core.lms_filter import lms_filter

from .recovery_experiment_governance import (
    AttemptIdentity,
    BudgetAmendmentRequest,
    validate_budget_amendment_authorization,
)
from .recovery_filter_profiles import FilterProfile

_EXPECTED_SCENES = ("jianpan", "kaihe", "run", "xiezi")
_EPS = np.finfo(float).eps


class StabilityAuditError(ValueError):
    """The diagnostic evidence is incomplete or violates the frozen contract."""


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise StabilityAuditError(f"{name}_must_be_lowercase_sha256")


@dataclass(frozen=True)
class StabilityAuditContract:
    """Thresholds frozen before the 32 diagnostic identities are executed."""

    max_stability_load: float
    max_weight_norm: float
    max_residual_rms_ratio_p95: float
    max_residual_tail_head_ratio: float
    min_true_peak_retention_ratio_median: float
    min_motion_artifact_suppression_db_median: float
    max_residual_artifact_abs_corr_median: float
    contract_version: str = "lyx_filter_stability_audit_v1"
    tail_head_gate: Literal["maximum", "descriptive_only"] = "maximum"

    @classmethod
    def frozen_v1(cls) -> StabilityAuditContract:
        return cls(
            max_stability_load=1.0,
            max_weight_norm=25.0,
            max_residual_rms_ratio_p95=2.0,
            max_residual_tail_head_ratio=2.0,
            min_true_peak_retention_ratio_median=0.80,
            min_motion_artifact_suppression_db_median=-1.0,
            max_residual_artifact_abs_corr_median=0.95,
        )

    @classmethod
    def corrected_v2(cls) -> StabilityAuditContract:
        return cls(
            max_stability_load=1.0,
            max_weight_norm=25.0,
            max_residual_rms_ratio_p95=2.0,
            max_residual_tail_head_ratio=2.0,
            min_true_peak_retention_ratio_median=0.80,
            min_motion_artifact_suppression_db_median=-1.0,
            max_residual_artifact_abs_corr_median=0.95,
            contract_version="lyx_filter_stability_audit_v2",
            tail_head_gate="descriptive_only",
        )

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.contract_version == "lyx_filter_stability_audit_v1":
            payload.pop("tail_head_gate")
        return payload


@dataclass(frozen=True)
class FilterAuditRecord:
    record_id: str
    scene: str
    data_path: str
    reference_path: str
    data_sha256: str
    reference_sha256: str

    def __post_init__(self) -> None:
        if not self.record_id or self.scene not in _EXPECTED_SCENES:
            raise ValueError("invalid_filter_audit_record_identity")
        if not self.data_path or not self.reference_path:
            raise ValueError("filter_audit_record_paths_must_not_be_empty")
        _require_sha256("data_sha256", self.data_sha256)
        _require_sha256("reference_sha256", self.reference_sha256)

    @property
    def combined_data_sha256(self) -> str:
        return _canonical_sha256(
            {
                "data_sha256": self.data_sha256,
                "reference_sha256": self.reference_sha256,
            }
        )


def plan_filter_audit_identities(
    *,
    profiles: tuple[FilterProfile, ...],
    records: tuple[FilterAuditRecord, ...],
    parent_experiment_id: str,
    solver_hash: str,
    metric_contract_hash: str,
    evaluation_hash: str,
    design_rule_sha256: str,
    record_manifest_sha256: str,
    authorization_receipt: dict[str, Any] | None,
) -> tuple[AttemptIdentity, ...]:
    """Authorize and deterministically plan the bounded 8 × 4 diagnostic matrix."""

    if len(profiles) != 8 or len(records) != 4:
        raise StabilityAuditError("filter_audit_plan_must_be_eight_by_four")
    if len({profile.profile_id for profile in profiles}) != len(profiles):
        raise StabilityAuditError("duplicate_filter_audit_profile")
    if {record.scene for record in records} != set(_EXPECTED_SCENES):
        raise StabilityAuditError("filter_audit_record_scene_coverage_mismatch")
    for name, value in (
        ("solver_hash", solver_hash),
        ("metric_contract_hash", metric_contract_hash),
        ("evaluation_hash", evaluation_hash),
        ("design_rule_sha256", design_rule_sha256),
        ("record_manifest_sha256", record_manifest_sha256),
    ):
        _require_sha256(name, value)
    request = BudgetAmendmentRequest(
        stage="filter_profile_stability_audit",
        profile_design_rule_hash=design_rule_sha256,
        record_manifest_hash=record_manifest_sha256,
        added_unique_identities=32,
        normal_unique_identity_limit=704,
        max_unique_identities=716,
        max_attempts=1432,
    )
    validate_budget_amendment_authorization(
        request,
        receipt=authorization_receipt,
    )

    identities = tuple(
        AttemptIdentity(
            solver_hash=solver_hash,
            config_hash=_canonical_sha256(
                {
                    "profile_sha256": profile.sha256,
                    "fs_target": profile.fs_target,
                    "memory_ms": profile.memory_ms,
                    "actual_taps": profile.actual_taps,
                    "nominal_mu": float(profile.nominal_mu),
                }
            ),
            metric_contract_hash=metric_contract_hash,
            evaluation_hash=evaluation_hash,
            data_sha256=record.combined_data_sha256,
            record_id=record.record_id,
            stage="filter_profile_stability_audit",
            attempt_kind="diagnostic",
            parent_experiment_id=parent_experiment_id,
        )
        for profile in profiles
        for record in records
    )
    if len({identity.sha256 for identity in identities}) != 32:
        raise StabilityAuditError("filter_audit_identity_collision")
    return identities


def plan_replacement_filter_audit_identities(
    *,
    profiles: tuple[FilterProfile, ...],
    records: tuple[FilterAuditRecord, ...],
    parent_experiment_id: str,
    solver_hash: str,
    metric_contract_hash: str,
    evaluation_hash: str,
    design_rule_sha256: str,
    record_manifest_sha256: str,
    authorization_receipt: dict[str, Any] | None,
) -> tuple[AttemptIdentity, ...]:
    """Authorize exactly two replacement profiles across the four audit records."""

    if len(profiles) != 2 or len(records) != 4:
        raise StabilityAuditError("replacement_audit_plan_must_be_two_by_four")
    if len({profile.profile_id for profile in profiles}) != 2:
        raise StabilityAuditError("duplicate_replacement_audit_profile")
    if {record.scene for record in records} != set(_EXPECTED_SCENES):
        raise StabilityAuditError("filter_audit_record_scene_coverage_mismatch")
    for name, value in (
        ("solver_hash", solver_hash),
        ("metric_contract_hash", metric_contract_hash),
        ("evaluation_hash", evaluation_hash),
        ("design_rule_sha256", design_rule_sha256),
        ("record_manifest_sha256", record_manifest_sha256),
    ):
        _require_sha256(name, value)
    validate_budget_amendment_authorization(
        BudgetAmendmentRequest(
            stage="filter_profile_stability_audit",
            profile_design_rule_hash=design_rule_sha256,
            record_manifest_hash=record_manifest_sha256,
            added_unique_identities=8,
            normal_unique_identity_limit=712,
            max_unique_identities=724,
            max_attempts=1448,
        ),
        receipt=authorization_receipt,
    )
    identities = tuple(
        AttemptIdentity(
            solver_hash=solver_hash,
            config_hash=_canonical_sha256(
                {
                    "profile_sha256": profile.sha256,
                    "fs_target": profile.fs_target,
                    "memory_ms": profile.memory_ms,
                    "actual_taps": profile.actual_taps,
                    "nominal_mu": float(profile.nominal_mu),
                }
            ),
            metric_contract_hash=metric_contract_hash,
            evaluation_hash=evaluation_hash,
            data_sha256=record.combined_data_sha256,
            record_id=record.record_id,
            stage="filter_profile_stability_audit",
            attempt_kind="diagnostic",
            parent_experiment_id=parent_experiment_id,
        )
        for profile in profiles
        for record in records
    )
    if len({identity.sha256 for identity in identities}) != 8:
        raise StabilityAuditError("replacement_audit_identity_collision")
    return identities


def plan_spec_gate_supplement_identities(
    *,
    profiles: tuple[FilterProfile, ...],
    records: tuple[FilterAuditRecord, ...],
    parent_experiment_id: str,
    solver_hash: str,
    metric_contract_hash: str,
    evaluation_hash: str,
    design_rule_sha256: str,
    record_manifest_sha256: str,
    authorization_receipt: dict[str, Any] | None,
) -> tuple[AttemptIdentity, ...]:
    """Authorize the six-profile, four-record frozen-gate supplement."""

    if len(profiles) != 6 or len(records) != 4:
        raise StabilityAuditError("spec_gate_supplement_plan_must_be_six_by_four")
    if len({profile.profile_id for profile in profiles}) != 6:
        raise StabilityAuditError("duplicate_spec_gate_supplement_profile")
    if {record.scene for record in records} != set(_EXPECTED_SCENES):
        raise StabilityAuditError("filter_audit_record_scene_coverage_mismatch")
    for name, value in (
        ("solver_hash", solver_hash),
        ("metric_contract_hash", metric_contract_hash),
        ("evaluation_hash", evaluation_hash),
        ("design_rule_sha256", design_rule_sha256),
        ("record_manifest_sha256", record_manifest_sha256),
    ):
        _require_sha256(name, value)
    validate_budget_amendment_authorization(
        BudgetAmendmentRequest(
            stage="filter_profile_stability_audit",
            profile_design_rule_hash=design_rule_sha256,
            record_manifest_hash=record_manifest_sha256,
            added_unique_identities=24,
            normal_unique_identity_limit=736,
            max_unique_identities=748,
            max_attempts=1496,
        ),
        receipt=authorization_receipt,
    )
    identities = tuple(
        AttemptIdentity(
            solver_hash=solver_hash,
            config_hash=_canonical_sha256(
                {
                    "profile_sha256": profile.sha256,
                    "fs_target": profile.fs_target,
                    "memory_ms": profile.memory_ms,
                    "actual_taps": profile.actual_taps,
                    "nominal_mu": float(profile.nominal_mu),
                }
            ),
            metric_contract_hash=metric_contract_hash,
            evaluation_hash=evaluation_hash,
            data_sha256=record.combined_data_sha256,
            record_id=record.record_id,
            stage="filter_profile_stability_audit",
            attempt_kind="diagnostic",
            parent_experiment_id=parent_experiment_id,
        )
        for profile in profiles
        for record in records
    )
    if len({identity.sha256 for identity in identities}) != 24:
        raise StabilityAuditError("spec_gate_supplement_identity_collision")
    return identities


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    centered = values - float(np.mean(values)) if values.size else values.copy()
    return centered / std if std > 0.0 and math.isfinite(std) else centered


def _power_at_frequency(values: np.ndarray, fs: int, frequency_hz: float) -> float:
    frequencies, power = periodogram(np.asarray(values, dtype=float), fs=float(fs))
    if power.size == 0 or not math.isfinite(float(frequency_hz)):
        return 0.0
    index = int(np.argmin(np.abs(frequencies - float(frequency_hz))))
    low = max(0, index - 1)
    high = min(power.size, index + 2)
    return float(np.sum(power[low:high]))


def _dominant_frequency(values: np.ndarray, fs: int) -> float:
    frequencies, power = periodogram(np.asarray(values, dtype=float), fs=float(fs))
    mask = (frequencies >= 0.5) & (frequencies <= min(4.0, 0.45 * fs))
    if not np.any(mask):
        return float("nan")
    bounded_frequencies = frequencies[mask]
    bounded_power = power[mask]
    return float(bounded_frequencies[int(np.argmax(bounded_power))])


def _maximum_input_eigenvalue(reference: np.ndarray, span: int) -> float:
    normalized = _zscore(reference)
    if span <= 0 or normalized.size < span + 1:
        return 0.0
    rows = np.lib.stride_tricks.sliding_window_view(normalized, span)
    covariance = (rows.T @ rows) / max(1, rows.shape[0])
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(max(0.0, eigenvalues[-1])) if eigenvalues.size else 0.0


def audit_lms_stage(
    *,
    desired: np.ndarray,
    reference: np.ndarray,
    fs: int,
    nominal_mu: float,
    order: int,
    K: int,
    true_hr_bpm: float,
    lms_mu_min: float = 1e-6,
) -> dict[str, Any]:
    """Audit one LMS stage without running heart-rate tracking or BO."""

    started = time.perf_counter()
    desired_arr = np.asarray(desired, dtype=float).ravel()
    reference_arr = np.asarray(reference, dtype=float).ravel()
    if desired_arr.size < 8 or reference_arr.size < desired_arr.size:
        raise StabilityAuditError("insufficient_lms_stage_samples")
    if order < 1 or K < 0:
        raise StabilityAuditError("invalid_lms_stage_shape")
    common = min(desired_arr.size, reference_arr.size)
    desired_arr = desired_arr[:common]
    reference_arr = reference_arr[:common]
    finite = np.isfinite(desired_arr) & np.isfinite(reference_arr)
    if not np.all(finite):
        raise StabilityAuditError("nonfinite_lms_stage_input")

    corr = float(abs(np.corrcoef(desired_arr, reference_arr)[0, 1]))
    if not math.isfinite(corr):
        corr = 0.0
    effective_mu = max(float(lms_mu_min), float(nominal_mu) - corr / 100.0)
    residual, weights, _ = lms_filter(
        effective_mu,
        int(order),
        int(K),
        reference_arr,
        desired_arr,
    )
    warmup = min(max(0, int(order) - 1), residual.size)
    residual_eval = residual[warmup:]
    desired_eval = _zscore(desired_arr)[warmup : warmup + residual_eval.size]
    reference_eval = _zscore(reference_arr)[warmup : warmup + residual_eval.size]
    if residual_eval.size < 4:
        raise StabilityAuditError("insufficient_residual_samples")

    desired_rms = float(np.sqrt(np.mean(desired_eval**2)))
    residual_rms = float(np.sqrt(np.mean(residual_eval**2)))
    quarter = max(1, residual_eval.size // 4)
    head_rms = float(np.sqrt(np.mean(residual_eval[:quarter] ** 2)))
    tail_rms = float(np.sqrt(np.mean(residual_eval[-quarter:] ** 2)))
    true_frequency_hz = float(true_hr_bpm) / 60.0
    artifact_frequency_hz = _dominant_frequency(reference_eval, fs)
    desired_true_power = _power_at_frequency(desired_eval, fs, true_frequency_hz)
    residual_true_power = _power_at_frequency(residual_eval, fs, true_frequency_hz)
    desired_artifact_power = _power_at_frequency(
        desired_eval,
        fs,
        artifact_frequency_hz,
    )
    residual_artifact_power = _power_at_frequency(
        residual_eval,
        fs,
        artifact_frequency_hz,
    )
    residual_corr = float(abs(np.corrcoef(residual_eval, reference_eval)[0, 1]))
    if not math.isfinite(residual_corr):
        residual_corr = 0.0
    lambda_max = _maximum_input_eigenvalue(reference_arr, int(order) + int(K))
    stability_load = effective_mu * lambda_max
    numeric_values = np.concatenate((residual_eval, weights))

    return {
        "sample_count": int(common),
        "order": int(order),
        "K": int(K),
        "nominal_mu": float(nominal_mu),
        "effective_mu": float(effective_mu),
        "input_energy": float(np.mean((desired_arr - np.mean(desired_arr)) ** 2)),
        "reference_energy": float(
            np.mean((reference_arr - np.mean(reference_arr)) ** 2)
        ),
        "input_covariance_lambda_max": lambda_max,
        "stability_load": stability_load,
        "weight_norm": float(np.linalg.norm(weights)),
        "residual_rms_ratio": residual_rms / max(desired_rms, _EPS),
        "residual_tail_head_ratio": tail_rms / max(head_rms, _EPS),
        "true_peak_retention_ratio": residual_true_power
        / max(desired_true_power, _EPS),
        "motion_artifact_frequency_hz": artifact_frequency_hz,
        "motion_artifact_suppression_db": 10.0
        * math.log10(
            max(desired_artifact_power, _EPS)
            / max(residual_artifact_power, _EPS)
        ),
        "residual_artifact_abs_corr": residual_corr,
        "nonfinite_count": int(
            numeric_values.size - np.count_nonzero(np.isfinite(numeric_values))
        ),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def summarize_record_audit(
    *,
    record_id: str,
    scene: str,
    stage_audits: list[dict[str, Any]],
    configured_max_taps: int,
    runtime_seconds: float,
    contract: StabilityAuditContract,
) -> dict[str, Any]:
    """Apply each frozen hard gate independently to one record's stage evidence."""

    if not record_id or scene not in _EXPECTED_SCENES:
        raise StabilityAuditError("invalid_record_audit_identity")
    if not stage_audits:
        raise StabilityAuditError("record_audit_has_no_lms_stages")
    required = {
        "order",
        "stability_load",
        "weight_norm",
        "residual_rms_ratio",
        "residual_tail_head_ratio",
        "true_peak_retention_ratio",
        "motion_artifact_suppression_db",
        "residual_artifact_abs_corr",
        "input_energy",
        "reference_energy",
        "effective_mu",
        "nonfinite_count",
    }
    if any(not required <= set(stage) for stage in stage_audits):
        raise StabilityAuditError("incomplete_lms_stage_audit")

    def values(field: str) -> np.ndarray:
        return np.asarray([float(stage[field]) for stage in stage_audits], dtype=float)

    stability_load_max = float(np.max(values("stability_load")))
    weight_norm_max = float(np.max(values("weight_norm")))
    residual_rms_ratio_p95 = float(np.quantile(values("residual_rms_ratio"), 0.95))
    residual_tail_head_ratio_max = float(
        np.max(values("residual_tail_head_ratio"))
    )
    residual_tail_head_ratio_p95 = float(
        np.quantile(values("residual_tail_head_ratio"), 0.95)
    )
    nonfinite_count = int(
        sum(int(stage["nonfinite_count"]) for stage in stage_audits)
    )
    true_peak_retention_median = float(
        np.median(values("true_peak_retention_ratio"))
    )
    artifact_suppression_median = float(
        np.median(values("motion_artifact_suppression_db"))
    )
    residual_artifact_corr_median = float(
        np.median(values("residual_artifact_abs_corr"))
    )
    tail_head_pass = (
        contract.tail_head_gate == "descriptive_only"
        or residual_tail_head_ratio_max
        <= contract.max_residual_tail_head_ratio
    )
    stability_pass = (
        nonfinite_count == 0
        and stability_load_max < contract.max_stability_load
        and weight_norm_max <= contract.max_weight_norm
        and residual_rms_ratio_p95 <= contract.max_residual_rms_ratio_p95
        and tail_head_pass
    )
    spectral_pass = (
        true_peak_retention_median
        >= contract.min_true_peak_retention_ratio_median
        and artifact_suppression_median
        >= contract.min_motion_artifact_suppression_db_median
        and residual_artifact_corr_median
        <= contract.max_residual_artifact_abs_corr_median
    )
    effective_mu = values("effective_mu")
    return {
        "record_id": record_id,
        "scene": scene,
        "lms_stage_count": len(stage_audits),
        "configured_max_taps": int(configured_max_taps),
        "max_tap_hit_count": int(
            sum(
                int(stage["order"]) >= int(configured_max_taps)
                for stage in stage_audits
            )
        ),
        "input_energy_median": float(np.median(values("input_energy"))),
        "reference_energy_median": float(np.median(values("reference_energy"))),
        "effective_mu_min": float(np.min(effective_mu)),
        "effective_mu_median": float(np.median(effective_mu)),
        "effective_mu_max": float(np.max(effective_mu)),
        "stability_load_max": stability_load_max,
        "weight_norm_max": weight_norm_max,
        "residual_rms_ratio_p95": residual_rms_ratio_p95,
        "residual_tail_head_ratio_max": residual_tail_head_ratio_max,
        "residual_tail_head_ratio_p95": residual_tail_head_ratio_p95,
        "tail_head_gate": contract.tail_head_gate,
        "nonfinite_count": nonfinite_count,
        "true_peak_retention_ratio_median": true_peak_retention_median,
        "motion_artifact_suppression_db_median": artifact_suppression_median,
        "residual_artifact_abs_corr_median": residual_artifact_corr_median,
        "runtime_seconds": float(runtime_seconds),
        "stability_pass": bool(stability_pass),
        "spectral_pass": bool(spectral_pass),
    }


def reclassify_cached_record_audit(
    cached_audit: dict[str, object],
    *,
    corrected_contract: StabilityAuditContract,
    source_metric_contract_sha256: str,
    source_result_sha256: str,
    reclassification_reason: str = (
        "remove_pathological_cold_start_tail_head_max_gate"
    ),
) -> dict[str, Any]:
    """Reapply a corrected decision gate to immutable cached numeric summaries."""

    _require_sha256(
        "source_metric_contract_sha256",
        source_metric_contract_sha256,
    )
    _require_sha256("source_result_sha256", source_result_sha256)
    if corrected_contract.tail_head_gate != "descriptive_only":
        raise StabilityAuditError("reclassification_requires_descriptive_tail_gate")
    required = {
        "stability_load_max",
        "weight_norm_max",
        "residual_rms_ratio_p95",
        "nonfinite_count",
        "true_peak_retention_ratio_median",
        "motion_artifact_suppression_db_median",
        "residual_artifact_abs_corr_median",
    }
    missing = sorted(required - cached_audit.keys())
    if missing:
        raise StabilityAuditError(
            "cached_audit_missing_fields:" + ",".join(missing)
        )
    stability_pass = (
        int(cached_audit["nonfinite_count"]) == 0
        and float(cached_audit["stability_load_max"])
        < corrected_contract.max_stability_load
        and float(cached_audit["weight_norm_max"])
        <= corrected_contract.max_weight_norm
        and float(cached_audit["residual_rms_ratio_p95"])
        <= corrected_contract.max_residual_rms_ratio_p95
    )
    spectral_pass = (
        float(cached_audit["true_peak_retention_ratio_median"])
        >= corrected_contract.min_true_peak_retention_ratio_median
        and float(cached_audit["motion_artifact_suppression_db_median"])
        >= corrected_contract.min_motion_artifact_suppression_db_median
        and float(cached_audit["residual_artifact_abs_corr_median"])
        <= corrected_contract.max_residual_artifact_abs_corr_median
    )
    return {
        **cached_audit,
        "stability_pass": bool(stability_pass),
        "spectral_pass": bool(spectral_pass),
        "audit_contract_sha256": corrected_contract.sha256,
        "source_metric_contract_sha256": source_metric_contract_sha256,
        "source_result_sha256": source_result_sha256,
        "numerical_result_reused": True,
        "reclassification_reason": reclassification_reason,
    }


def build_filter_profile_receipt(
    profile: FilterProfile,
    record_audits: list[dict[str, object]],
    *,
    audit_contract: StabilityAuditContract,
    library_sha256: str,
    solver_hash: str,
    code_hash: str,
    evaluation_hash: str,
    design_rule_sha256: str,
    record_manifest_sha256: str,
) -> dict[str, Any]:
    """Aggregate four pre-registered record diagnostics into one fail-closed receipt."""

    for name, value in (
        ("library_sha256", library_sha256),
        ("solver_hash", solver_hash),
        ("code_hash", code_hash),
        ("evaluation_hash", evaluation_hash),
        ("design_rule_sha256", design_rule_sha256),
        ("record_manifest_sha256", record_manifest_sha256),
    ):
        _require_sha256(name, value)
    if len(record_audits) != 4:
        raise StabilityAuditError("profile_audit_requires_four_records")
    by_scene = {str(item.get("scene")): item for item in record_audits}
    if set(by_scene) != set(_EXPECTED_SCENES):
        raise StabilityAuditError("profile_audit_scene_coverage_mismatch")
    if len(by_scene) != len(record_audits):
        raise StabilityAuditError("duplicate_profile_audit_scene")
    for item in record_audits:
        _require_sha256("identity_sha256", str(item.get("identity_sha256", "")))
        _require_sha256("result_sha256", str(item.get("result_sha256", "")))
        _require_sha256("data_sha256", str(item.get("data_sha256", "")))
        _require_sha256("reference_sha256", str(item.get("reference_sha256", "")))

    stability_pass = all(bool(item.get("stability_pass")) for item in record_audits)
    spectral_pass = all(bool(item.get("spectral_pass")) for item in record_audits)
    eligible = stability_pass and spectral_pass
    payload: dict[str, Any] = {
        "receipt_version": "lyx_filter_profile_receipt_v1",
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "design_role": profile.design_role,
        "fs_target": profile.fs_target,
        "physical_memory_ms": profile.memory_ms,
        "actual_taps": profile.actual_taps,
        "nominal_mu": float(profile.nominal_mu),
        "effective_mu_formula": "max(lms_mu_min, nominal_mu - abs_corr / 100)",
        "recovery_sentinel_role": profile.recovery_sentinel_role,
        "library_sha256": library_sha256,
        "audit_contract_sha256": audit_contract.sha256,
        "solver_hash": solver_hash,
        "code_hash": code_hash,
        "evaluation_hash": evaluation_hash,
        "design_rule_sha256": design_rule_sha256,
        "record_manifest_sha256": record_manifest_sha256,
        "diagnostic_identity_sha256": sorted(
            str(item["identity_sha256"]) for item in record_audits
        ),
        "diagnostic_result_sha256": sorted(
            str(item["result_sha256"]) for item in record_audits
        ),
        "record_identity_hashes": sorted(
            (
                {
                    "record_id": str(item["record_id"]),
                    "data_sha256": str(item["data_sha256"]),
                    "reference_sha256": str(item["reference_sha256"]),
                    "diagnostic_identity_sha256": str(item["identity_sha256"]),
                    "diagnostic_result_sha256": str(item["result_sha256"]),
                }
                for item in record_audits
            ),
            key=lambda item: item["record_id"],
        ),
        "stability": {
            "all_records_pass": stability_pass,
            "record_results": [
                {
                    "record_id": item["record_id"],
                    "scene": item["scene"],
                    "max_tap_hit_count": item["max_tap_hit_count"],
                    "input_energy_median": item["input_energy_median"],
                    "reference_energy_median": item["reference_energy_median"],
                    "weight_norm_max": item["weight_norm_max"],
                    "residual_rms_ratio_p95": item["residual_rms_ratio_p95"],
                    "runtime_seconds": item["runtime_seconds"],
                    "passed": item["stability_pass"],
                }
                for item in record_audits
            ],
        },
        "spectral_evidence": {
            "all_records_pass": spectral_pass,
            "record_results": [
                {
                    "record_id": item["record_id"],
                    "scene": item["scene"],
                    "true_peak_retention_ratio_median": item[
                        "true_peak_retention_ratio_median"
                    ],
                    "motion_artifact_suppression_db_median": item[
                        "motion_artifact_suppression_db_median"
                    ],
                    "residual_artifact_abs_corr_median": item[
                        "residual_artifact_abs_corr_median"
                    ],
                    "passed": item["spectral_pass"],
                }
                for item in record_audits
            ],
        },
        "status": "eligible" if eligible else "rejected",
        "may_enter_formal_matrix": eligible,
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    return payload
