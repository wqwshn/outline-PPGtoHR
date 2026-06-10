from __future__ import annotations

import numpy as np

from spo2_pressure_recovery.metrics import (
    beat_metrics,
    decide_candidate,
    waveform_metrics,
)
from spo2_pressure_recovery.types import DecisionThresholds


def test_waveform_metrics_identical_signal_has_perfect_score() -> None:
    values = np.sin(np.linspace(0.0, 10.0, 500))

    metrics = waveform_metrics(values, values.copy())

    assert metrics["corr"] == 1.0
    assert metrics["nrmse"] == 0.0


def test_beat_metrics_reports_false_peaks() -> None:
    reference = np.array([100, 200, 300, 400])
    estimate = np.array([100, 200, 250, 300, 400])

    metrics = beat_metrics(reference, estimate, fs_hz=100.0)

    assert metrics["false_peak_rate"] > 0.0
    assert metrics["missed_peak_rate"] == 0.0


def test_decide_candidate_rejects_rest_damage_false_peak_and_ratio_instability() -> None:
    thresholds = DecisionThresholds(
        maximum_rest_nrmse=0.02,
        maximum_false_peak_increase=0.05,
        maximum_ratio_relative_error=0.15,
    )

    rest = decide_candidate({"rest_nrmse": 0.08}, thresholds)
    peaks = decide_candidate({"false_peak_increase": 0.25}, thresholds)
    ratio = decide_candidate({"ratio_relative_error": 0.30}, thresholds)

    assert rest.accepted is False
    assert "rest_damage" in rest.rejection_reasons
    assert peaks.accepted is False
    assert "false_peak_increase" in peaks.rejection_reasons
    assert ratio.accepted is False
    assert "ratio_instability" in ratio.rejection_reasons
