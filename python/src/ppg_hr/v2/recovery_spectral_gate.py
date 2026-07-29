"""Frozen relative spectral qualification for the LYX Stage R sentinels."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks, periodogram

from ppg_hr.core.choose_delay import choose_delay
from ppg_hr.core.lms_filter import lms_filter

from .phase2_experiment_io import file_sha256
from .recovery_contracts import canonical_sha256
from .recovery_filter_profiles import FilterProfile
from .recovery_filter_stability import (
    FilterAuditRecord,
    StabilityAuditContract,
    StabilityAuditError,
    audit_lms_stage,
    summarize_record_audit,
)
from .signal_preparation import prepare_v2_signals
from .types import V2RunConfig

_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class StageRSpectralGateContract:
    """Exact five-metric, per-record relative spectral gate from spec §7.5."""

    analysis_band_low_bpm: float = 30.0
    analysis_band_high_bpm: float = 210.0
    reference_neighborhood_half_width_bpm: float = 10.0
    visible_top_k: int = 3
    min_prominence_db_delta_median: float = -0.5
    min_visible_top3_rate_delta: float = -0.05
    min_hr_band_share_delta_median: float = -0.02
    min_pulse_power_retention_median: float = 0.80
    max_residual_artifact_corr_delta_median: float = 0.05
    minimum_valid_window_count: int = 3
    fail_on_any_uncomputable_window: bool = True
    record_aggregation: str = "equal_weight_window_then_record"
    cross_record_window_pooling: bool = False
    contract_version: str = "lyx_stage_r_spectral_gate_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        stability = StabilityAuditContract.corrected_v2()
        payload["filter_stability_contract"] = stability.to_dict()
        payload["filter_stability_contract_sha256"] = stability.sha256
        payload["metrics"] = (
            "visible_top3",
            "prominence_db",
            "hr_band_share",
            "pulse_power_retention",
            "residual_artifact_corr",
        )
        payload["failure_rule"] = (
            "any_uncomputable_metric_or_insufficient_window_fails_closed"
        )
        payload["relative_control"] = (
            "same_window_pre_filter_signal_shared_by_the_same_sentinel_"
            "current_recovery_control"
        )
        payload["motion_reference_rule"] = (
            "largest_archive_style_absolute_correlation_after_"
            "frozen_choose_delay"
        )
        payload["evaluation_grain"] = "sentinel_profile_x_record"
        payload["candidate_invariant"] = True
        payload["reuse_within_same_sentinel_record"] = True
        return payload

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _reference_hr_at_center(ref_data: np.ndarray, center_s: float) -> float:
    ref = np.asarray(ref_data, dtype=float)
    if ref.ndim != 2 or ref.shape[1] < 2 or ref.shape[0] == 0:
        raise StabilityAuditError(
            "missing_reference_hr_for_stage_r_spectral_gate"
        )
    times = ref[:, 0]
    values = ref[:, 1]
    finite = np.isfinite(times) & np.isfinite(values)
    if not np.any(finite):
        raise StabilityAuditError(
            "nonfinite_reference_hr_for_stage_r_spectral_gate"
        )
    return float(np.interp(center_s, times[finite], values[finite]))


def _aligned_reference(reference: np.ndarray, delay_samples: int) -> np.ndarray:
    values = np.asarray(reference, dtype=float).ravel()
    if not delay_samples:
        return values
    if abs(delay_samples) >= values.size:
        raise StabilityAuditError(
            "stage_r_spectral_delay_exceeds_window"
        )
    aligned = np.empty_like(values)
    if delay_samples > 0:
        aligned[:delay_samples] = values[0]
        aligned[delay_samples:] = values[:-delay_samples]
    else:
        shift = abs(delay_samples)
        aligned[-shift:] = values[-1]
        aligned[:-shift] = values[shift:]
    return aligned


def _spectrum_features(
    values: np.ndarray,
    *,
    fs: int,
    reference_hr_bpm: float,
    contract: StageRSpectralGateContract,
) -> dict[str, float | bool]:
    signal = np.asarray(values, dtype=float).ravel()
    if signal.size < 8 or not np.all(np.isfinite(signal)):
        raise StabilityAuditError("uncomputable_stage_r_spectrum")
    frequencies, power = periodogram(
        signal,
        fs=float(fs),
        window="hann",
        detrend="constant",
        scaling="spectrum",
    )
    bpm = frequencies * 60.0
    analysis = (
        (bpm >= contract.analysis_band_low_bpm)
        & (bpm <= contract.analysis_band_high_bpm)
    )
    neighborhood = analysis & (
        np.abs(bpm - float(reference_hr_bpm))
        <= contract.reference_neighborhood_half_width_bpm
    )
    competitor = analysis & ~neighborhood
    if (
        np.count_nonzero(analysis) < contract.visible_top_k
        or not np.any(neighborhood)
        or not np.any(competitor)
    ):
        raise StabilityAuditError(
            "insufficient_stage_r_spectral_bins"
        )
    analysis_indices = np.flatnonzero(analysis)
    bounded_power = power[analysis_indices]
    peak_offsets = find_peaks(bounded_power)[0]
    strongest_offset = int(np.argmax(bounded_power))
    candidate_offsets = np.unique(
        np.append(peak_offsets, strongest_offset)
    )
    if candidate_offsets.size < contract.visible_top_k:
        candidate_offsets = np.unique(
            np.append(
                candidate_offsets,
                np.argsort(bounded_power)[-contract.visible_top_k :],
            )
        )
    ranked_offsets = candidate_offsets[
        np.argsort(bounded_power[candidate_offsets])[::-1]
    ][: contract.visible_top_k]
    top_indices = analysis_indices[ranked_offsets]
    reference_power = float(np.sum(power[neighborhood]))
    analysis_power = float(np.sum(power[analysis]))
    reference_peak_power = float(np.max(power[neighborhood]))
    competitor_peak_power = float(np.max(power[competitor]))
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (
            reference_power,
            analysis_power,
            reference_peak_power,
            competitor_peak_power,
        )
    ):
        raise StabilityAuditError(
            "nonfinite_stage_r_spectral_power"
        )
    return {
        "visible_top3": bool(np.any(neighborhood[top_indices])),
        "prominence_db": 10.0
        * math.log10(
            max(reference_peak_power, _EPS)
            / max(competitor_peak_power, _EPS)
        ),
        "hr_band_share": reference_power / max(analysis_power, _EPS),
        "reference_band_power": reference_power,
    }


def evaluate_stage_r_spectral_gate_windows(
    windows: Sequence[dict[str, Any]],
    *,
    contract: StageRSpectralGateContract,
) -> dict[str, Any]:
    """Aggregate complete per-window evidence without cross-record pooling."""

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, window in enumerate(windows):
        try:
            before = np.asarray(window["before"], dtype=float)
            after = np.asarray(window["after"], dtype=float)
            motion_reference = np.asarray(
                window["motion_reference"],
                dtype=float,
            )
            if (
                before.size != after.size
                or before.size != motion_reference.size
                or before.size < 8
                or not np.all(np.isfinite(motion_reference))
            ):
                raise StabilityAuditError(
                    "invalid_stage_r_spectral_window_shape"
                )
            before_features = _spectrum_features(
                before,
                fs=int(window["fs"]),
                reference_hr_bpm=float(window["reference_hr_bpm"]),
                contract=contract,
            )
            after_features = _spectrum_features(
                after,
                fs=int(window["fs"]),
                reference_hr_bpm=float(window["reference_hr_bpm"]),
                contract=contract,
            )
            before_corr = float(
                abs(np.corrcoef(before, motion_reference)[0, 1])
            )
            after_corr = float(
                abs(np.corrcoef(after, motion_reference)[0, 1])
            )
            if not math.isfinite(before_corr) or not math.isfinite(after_corr):
                raise StabilityAuditError(
                    "uncomputable_stage_r_artifact_correlation"
                )
            rows.append(
                {
                    "window_center_s": float(window["window_center_s"]),
                    "reference_hr_bpm": float(window["reference_hr_bpm"]),
                    "visible_top3_before": bool(
                        before_features["visible_top3"]
                    ),
                    "visible_top3_after": bool(
                        after_features["visible_top3"]
                    ),
                    "prominence_db_delta": float(
                        after_features["prominence_db"]
                    )
                    - float(before_features["prominence_db"]),
                    "hr_band_share_delta": float(
                        after_features["hr_band_share"]
                    )
                    - float(before_features["hr_band_share"]),
                    "pulse_power_retention": float(
                        after_features["reference_band_power"]
                    )
                    / max(
                        float(before_features["reference_band_power"]),
                        _EPS,
                    ),
                    "residual_artifact_corr_before": before_corr,
                    "residual_artifact_corr_after": after_corr,
                    "residual_artifact_corr_delta": (
                        after_corr - before_corr
                    ),
                }
            )
        except (KeyError, TypeError, ValueError, StabilityAuditError) as error:
            failures.append(f"window_{index}:{type(error).__name__}:{error}")

    valid_count = len(rows)
    invalid_count = len(failures)
    enough = valid_count >= contract.minimum_valid_window_count
    complete = not contract.fail_on_any_uncomputable_window or invalid_count == 0
    if not enough:
        return {
            "spectral_gate_pass": False,
            "valid_window_count": valid_count,
            "invalid_window_count": invalid_count,
            "failure_reasons": [
                *failures,
                "insufficient_valid_spectral_windows",
            ],
            "window_metrics": rows,
        }

    def median(name: str) -> float:
        return float(np.median([float(row[name]) for row in rows]))

    prominence_delta = median("prominence_db_delta")
    visible_delta = float(
        np.mean(
            [
                float(row["visible_top3_after"])
                - float(row["visible_top3_before"])
                for row in rows
            ]
        )
    )
    share_delta = median("hr_band_share_delta")
    retention = median("pulse_power_retention")
    corr_delta = median("residual_artifact_corr_delta")
    gates = {
        "prominence_db_delta_pass": (
            prominence_delta
            >= contract.min_prominence_db_delta_median
        ),
        "visible_top3_rate_delta_pass": (
            visible_delta >= contract.min_visible_top3_rate_delta
        ),
        "hr_band_share_delta_pass": (
            share_delta
            >= contract.min_hr_band_share_delta_median
        ),
        "pulse_power_retention_pass": (
            retention >= contract.min_pulse_power_retention_median
        ),
        "residual_artifact_corr_delta_pass": (
            corr_delta
            <= contract.max_residual_artifact_corr_delta_median
        ),
        "complete_window_evidence_pass": complete,
    }
    return {
        "spectral_gate_pass": bool(all(gates.values())),
        "valid_window_count": valid_count,
        "invalid_window_count": invalid_count,
        "prominence_db_delta_median": prominence_delta,
        "visible_top3_rate_delta": visible_delta,
        "hr_band_share_delta_median": share_delta,
        "pulse_power_retention_median": retention,
        "residual_artifact_corr_delta_median": corr_delta,
        "gates": gates,
        "failure_reasons": failures,
        "window_metrics": rows,
    }


def audit_stage_r_profile_record(
    profile: FilterProfile,
    record: FilterAuditRecord,
    *,
    contract: StageRSpectralGateContract,
) -> dict[str, Any]:
    """Run LMS stability plus the exact five relative spectral gates once."""

    started = time.perf_counter()
    data_path = Path(record.data_path)
    reference_path = Path(record.reference_path)
    if file_sha256(data_path) != record.data_sha256:
        raise StabilityAuditError(
            f"record_data_hash_mismatch:{record.record_id}"
        )
    if file_sha256(reference_path) != record.reference_sha256:
        raise StabilityAuditError(
            f"record_reference_hash_mismatch:{record.record_id}"
        )
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
    start_sample = max(
        0,
        int(round(float(prepared.params.time_start) * fs)),
    )
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
        raise StabilityAuditError(
            "stage_r_spectral_gate_requires_two_hf_references"
        )
    stage_audits: list[dict[str, Any]] = []
    spectral_windows: list[dict[str, Any]] = []
    for idx_s in range(
        start_sample,
        max(start_sample, end_sample - window_samples + 1),
        step_samples,
    ):
        idx_e = idx_s + window_samples
        if idx_e > prepared.ppg.size:
            break
        time_1 = idx_s / fs
        center_s = time_1 + float(cfg.window_seconds) / 2.0
        true_hr_bpm = _reference_hr_at_center(
            prepared.ref_data,
            center_s,
        )
        signals = [
            np.asarray(item["signal"], dtype=float)
            for item in references
        ]
        corr_arr, _empty, delay, _acc_delay = choose_delay(
            fs,
            time_1,
            prepared.ppg,
            [],
            signals,
        )
        if corr_arr.size == 0:
            spectral_windows.append({})
            continue
        original = np.asarray(
            prepared.ppg[idx_s:idx_e],
            dtype=float,
        )
        current = original.copy()
        order = max(
            1,
            min(profile.actual_taps, int(abs(delay)) or 1),
        )
        ranked_reference_indices = np.argsort(corr_arr)[::-1]
        primary = signals[int(ranked_reference_indices[0])][idx_s:idx_e]
        primary = _aligned_reference(primary, int(delay))
        for ref_idx in ranked_reference_indices:
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
                    "channel": str(
                        references[int(ref_idx)]["channel"]
                    ),
                    "delay_samples": int(delay),
                    "archive_style_abs_corr": float(
                        corr_arr[int(ref_idx)]
                    ),
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
        spectral_windows.append(
            {
                "before": original,
                "after": current,
                "motion_reference": primary,
                "fs": fs,
                "reference_hr_bpm": true_hr_bpm,
                "window_center_s": center_s,
            }
        )
    stability = summarize_record_audit(
        record_id=record.record_id,
        scene=record.scene,
        stage_audits=stage_audits,
        configured_max_taps=profile.actual_taps,
        runtime_seconds=time.perf_counter() - started,
        contract=StabilityAuditContract.corrected_v2(),
    )
    spectral = evaluate_stage_r_spectral_gate_windows(
        spectral_windows,
        contract=contract,
    )
    return {
        **stability,
        "stage_r_spectral_gate": spectral,
        "spectral_gate_pass": bool(
            stability["stability_pass"]
            and spectral["spectral_gate_pass"]
        ),
        "stage_r_spectral_gate_contract_sha256": contract.sha256,
    }
