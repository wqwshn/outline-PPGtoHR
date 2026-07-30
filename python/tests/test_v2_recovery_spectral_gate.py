from __future__ import annotations

import numpy as np
import pytest

from ppg_hr.v2.recovery_spectral_gate import (
    StageRSpectralGateContract,
    evaluate_stage_r_spectral_gate_windows,
)


def _window(*, retain_pulse: bool) -> dict[str, object]:
    fs = 25
    time_s = np.arange(0.0, 8.0, 1.0 / fs)
    pulse = np.sin(2.0 * np.pi * 1.5 * time_s)
    artifact = np.sin(2.0 * np.pi * 2.5 * time_s)
    before = pulse + 1.4 * artifact
    after = (
        (0.95 * pulse if retain_pulse else 0.1 * pulse)
        + 0.2 * artifact
    )
    return {
        "before": before,
        "after": after,
        "motion_reference": artifact,
        "fs": fs,
        "reference_hr_bpm": 90.0,
        "window_center_s": 4.0,
    }


def test_stage_r_spectral_gate_reports_all_five_relative_metrics() -> None:
    result = evaluate_stage_r_spectral_gate_windows(
        [_window(retain_pulse=True) for _ in range(3)],
        contract=StageRSpectralGateContract(),
    )

    assert result["spectral_gate_pass"] is True
    assert result["valid_window_count"] == 3
    assert result["invalid_window_count"] == 0
    assert result["visible_top3_rate_delta"] >= 0.0
    assert result["prominence_db_delta_median"] > 0.0
    assert result["hr_band_share_delta_median"] > 0.0
    assert result["pulse_power_retention_median"] >= 0.80
    assert result["residual_artifact_corr_delta_median"] < 0.0
    assert set(result["gates"]) == {
        "prominence_db_delta_pass",
        "visible_top3_rate_delta_pass",
        "hr_band_share_delta_pass",
        "pulse_power_retention_pass",
        "residual_artifact_corr_delta_pass",
        "complete_window_evidence_pass",
    }


def test_stage_r_spectral_gate_compares_lms_inputs_in_standardized_domain() -> None:
    fs = 25
    time_s = np.arange(0.0, 8.0, 1.0 / fs)
    pulse = np.sin(2.0 * np.pi * 1.5 * time_s)
    artifact = 0.2 * np.sin(2.0 * np.pi * 2.5 * time_s)
    raw_before = 120.0 + 25.0 * (pulse + artifact)
    lms_after = (raw_before - raw_before.mean()) / raw_before.std(ddof=1)
    window = {
        "before": raw_before,
        "after": lms_after,
        "motion_reference": artifact,
        "fs": fs,
        "reference_hr_bpm": 90.0,
        "window_center_s": 4.0,
    }

    result = evaluate_stage_r_spectral_gate_windows(
        [window for _ in range(3)],
        contract=StageRSpectralGateContract(),
    )

    assert result["pulse_power_retention_median"] == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert result["gates"]["pulse_power_retention_pass"] is True


def test_stage_r_spectral_gate_contract_freezes_standardized_v2_domain() -> None:
    payload = StageRSpectralGateContract().to_dict()

    assert payload["contract_version"] == "lyx_stage_r_spectral_gate_v2"
    assert payload["spectral_signal_domain"] == {
        "before": "sample_zscore_ddof_1_per_window",
        "after": "sample_zscore_ddof_1_per_window",
        "implementation": "ppg_hr.core.lms_filter.standardize_lms_signal",
    }


def test_stage_r_spectral_gate_can_reconstruct_historical_v1_contract() -> None:
    contract = StageRSpectralGateContract.legacy_v1()

    assert contract.sha256 == (
        "46ccbaebe3f7eafcc14853066a5173cb3ff5ce0b8400b2a1fc1c8e6f59f0ee08"
    )
    assert "spectral_signal_domain" not in contract.to_dict()


def test_stage_r_spectral_gate_fails_closed_on_pulse_damage_or_bad_window() -> None:
    damaged = evaluate_stage_r_spectral_gate_windows(
        [_window(retain_pulse=False) for _ in range(3)],
        contract=StageRSpectralGateContract(),
    )
    incomplete = evaluate_stage_r_spectral_gate_windows(
        [_window(retain_pulse=True) for _ in range(3)] + [{}],
        contract=StageRSpectralGateContract(),
    )

    assert damaged["spectral_gate_pass"] is False
    assert damaged["gates"]["pulse_power_retention_pass"] is False
    assert incomplete["spectral_gate_pass"] is False
    assert incomplete["invalid_window_count"] == 1
    assert (
        incomplete["gates"]["complete_window_evidence_pass"]
        is False
    )


def test_stage_r_spectral_gate_keeps_six_false_gates_when_evidence_is_insufficient() -> None:
    result = evaluate_stage_r_spectral_gate_windows(
        [],
        contract=StageRSpectralGateContract(),
    )

    assert result["spectral_gate_pass"] is False
    assert set(result["gates"]) == {
        "prominence_db_delta_pass",
        "visible_top3_rate_delta_pass",
        "hr_band_share_delta_pass",
        "pulse_power_retention_pass",
        "residual_artifact_corr_delta_pass",
        "complete_window_evidence_pass",
    }
    assert not any(result["gates"].values())
