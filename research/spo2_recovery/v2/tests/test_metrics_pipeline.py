from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spo2_pressure_recovery.metrics import (
    beat_metrics,
    decide_candidate,
    peak_interval_stability,
    spo2_event_metrics,
    spo2_from_ratio,
    waveform_metrics,
)
from spo2_pressure_recovery.pipeline import ExperimentConfig, run_experiment, save_experiment
from spo2_pressure_recovery.plotting import render_experiment_figures
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


def test_spo2_from_ratio_uses_max30101_formula() -> None:
    value = spo2_from_ratio(0.6)
    expected = 1.5958422 * 0.6**2 - 34.6596622 * 0.6 + 112.6898759

    assert value == pytest.approx(expected)


def test_peak_interval_metrics_rejects_extra_spike() -> None:
    fs = 100.0
    clean = np.array([100, 200, 300, 400, 500])
    noisy = np.array([100, 200, 235, 300, 400, 500])

    metrics = peak_interval_stability(clean, noisy, fs_hz=fs)

    assert metrics["peak_interval_cv"] >= 0.0
    assert metrics["extra_peak_count"] == 1.0
    assert metrics["min_interval_s"] < 0.5


def test_spo2_event_metrics_are_near_stable_for_constant_ratio() -> None:
    fs = 100.0
    t = np.arange(0.0, 10.0, 1.0 / fs)
    ir = 1500.0 + 20.0 * np.sin(2.0 * np.pi * 1.0 * t)
    red = 1000.0 + 10.0 * np.sin(2.0 * np.pi * 1.0 * t)
    mask = (t >= 3.0) & (t <= 7.0)

    metrics = spo2_event_metrics(red, ir, mask, fs_hz=fs)

    assert metrics["valid_beat_count"] >= 3.0
    assert np.isfinite(metrics["r_median"])
    assert np.isfinite(metrics["spo2_median"])


def test_run_experiment_outputs_minimum_end_to_end_result(tmp_path) -> None:
    fs = 100.0
    t = np.arange(0.0, 24.0, 1.0 / fs)
    ut1 = 2000.0 + np.zeros_like(t)
    ut2 = 1720.0 + np.zeros_like(t)
    for center in (8.0, 16.0):
        pulse = 2.0 * np.exp(-0.5 * ((t - center) / 0.7) ** 4)
        ut1 += pulse
        ut2 += 0.8 * pulse
    event_signal = ((t > 7.0) & (t < 9.0)) | ((t > 15.0) & (t < 17.0))
    red = 1000.0 + 8.0 * np.sin(2.0 * np.pi * 1.0 * t)
    ir = 1500.0 + 12.0 * np.sin(2.0 * np.pi * 1.0 * t + 0.05)
    red[event_signal] += 30.0
    ir[event_signal] += 50.0
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(
        {
            "Time(s)": t,
            "Ut1(mV)": ut1,
            "Ut2(mV)": ut2,
            "PPG_Red": red,
            "PPG_IR": ir,
        }
    ).to_csv(path, index=False)

    result = run_experiment(path, ExperimentConfig())
    files = save_experiment(result, tmp_path / "out")

    assert result.events.shape[0] == 2
    assert not result.pseudo_quality.empty
    assert {"event_id", "usable"}.issubset(result.pseudo_quality.columns)
    assert not result.candidate_metrics.empty
    assert result.best_candidate["accepted"] is True
    assert set(result.waveforms) >= {
        "time_s",
        "red_observed",
        "ir_observed",
        "red_recovered",
        "ir_recovered",
        "red_pseudo",
        "ir_pseudo",
    }
    assert np.isfinite(result.waveforms["red_pseudo"]).any()
    assert np.isfinite(result.waveforms["ir_pseudo"]).any()
    assert files["candidate_metrics"].exists()
    assert files["pseudo_quality"].exists()
    assert files["summary"].exists()


def test_render_experiment_figures_writes_png_files(tmp_path) -> None:
    fs = 100.0
    t = np.arange(0.0, 24.0, 1.0 / fs)
    ut1 = 2000.0 + np.zeros_like(t)
    ut2 = 1720.0 + np.zeros_like(t)
    for center in (8.0, 16.0):
        pulse = 2.0 * np.exp(-0.5 * ((t - center) / 0.7) ** 4)
        ut1 += pulse
        ut2 += 0.8 * pulse
    event_signal = ((t > 7.0) & (t < 9.0)) | ((t > 15.0) & (t < 17.0))
    red = 1000.0 + 8.0 * np.sin(2.0 * np.pi * 1.0 * t)
    ir = 1500.0 + 12.0 * np.sin(2.0 * np.pi * 1.0 * t + 0.05)
    red[event_signal] += 30.0
    ir[event_signal] += 50.0
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(
        {
            "Time(s)": t,
            "Ut1(mV)": ut1,
            "Ut2(mV)": ut2,
            "PPG_Red": red,
            "PPG_IR": ir,
        }
    ).to_csv(path, index=False)
    result = run_experiment(path, ExperimentConfig())

    files = render_experiment_figures(result, tmp_path / "figures")

    expected = {
        "01-full-trace-events.png",
        "02-candidate-comparison.png",
        "03-best-model-diagnostics.png",
        "04-pseudo-truth-event-zoom.png",
        "05-pseudo-truth-dc-envelope-quality.png",
    }
    assert expected <= {file.name for file in files}
    assert all(file.suffix == ".png" and file.stat().st_size > 10_000 for file in files)
