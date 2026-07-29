from __future__ import annotations

import numpy as np

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
