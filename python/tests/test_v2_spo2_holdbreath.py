"""Tests for Red/IR-only hold-breath SpO2 evaluation."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ppg_hr.v2.spo2_holdbreath import (
    HoldBreathSpO2Config,
    HoldBreathSpO2Result,
    PulseOximeterModel,
    _estimate_holdbreath_band_seconds,
    apply_or_fit_device_model,
    compute_holdbreath_metrics,
    find_holdbreath_truth_path,
    load_holdbreath_truth,
    save_holdbreath_report,
    solve_spo2_holdbreath,
)


def write_synthetic_red_ir_sensor_csv(path: Path, seconds: int = 70) -> None:
    fs = 100
    n = seconds * fs
    t = np.arange(n, dtype=float) / fs
    pulse = np.sin(2 * np.pi * 1.15 * t)
    slow = 0.15 * np.sin(2 * np.pi * t / 35.0)
    frame = pd.DataFrame(
        {
            "Time(s)": t,
            "Uc1(mV)": 1026.0 + 0.01 * slow,
            "Uc2(mV)": 886.0 + 0.01 * slow,
            "Ut1(mV)": 1976.0 + slow,
            "Ut2(mV)": 1707.0 - 0.5 * slow,
            "PPG_Green": 1000.0 + 10.0 * pulse,
            "PPG_Red": 103000.0 + 180.0 * pulse + 35.0 * slow,
            "PPG_IR": 157000.0 + 250.0 * pulse + 30.0 * slow,
            "AccX(g)": np.zeros(n),
            "AccY(g)": np.zeros(n),
            "AccZ(g)": np.ones(n),
            "GyroX(dps)": np.zeros(n),
            "GyroY(dps)": np.zeros(n),
            "GyroZ(dps)": np.zeros(n),
        }
    )
    frame.to_csv(path, index=False)


def test_find_holdbreath_truth_path_uses_stem_ref_suffix(tmp_path: Path) -> None:
    data = tmp_path / "Spo2_HB1.csv"
    truth = tmp_path / "Spo2_HB1_ref.csv"
    data.write_text("Time(s),PPG_Red,PPG_IR\n0,1,1\n", encoding="utf-8")
    truth.write_bytes(b"placeholder")

    assert find_holdbreath_truth_path(data) == truth


def test_load_holdbreath_truth_accepts_excel_content_with_csv_suffix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "Spo2_HB1_ref.csv"
    buffer = io.BytesIO()
    pd.DataFrame({"clock": ["21:30:16", "21:30:17"], "spo2": [98, 99]}).to_excel(
        buffer,
        index=False,
    )
    path.write_bytes(buffer.getvalue())

    truth = load_holdbreath_truth(path)

    assert truth.time_s.tolist() == [0.0, 1.0]
    assert truth.spo2.tolist() == [98.0, 99.0]


def test_holdbreath_metrics_use_analysis_slice_and_mae_primary() -> None:
    time_s = np.arange(0, 8, dtype=float)
    calculated = np.array([80, 97, 98, 97, 95, 98, 97, 70], dtype=float)
    truth = np.array([99, 98, 98, 96, 96, 98, 99, 99], dtype=float)

    metrics = compute_holdbreath_metrics(
        time_s,
        calculated,
        truth,
        analysis_start_s=1.0,
        analysis_end_s=6.0,
    )

    assert metrics["sample_count"] == 6
    assert metrics["mae"] == pytest.approx(
        np.mean(np.abs(calculated[1:7] - truth[1:7]))
    )
    assert metrics["rmse"] == pytest.approx(
        np.sqrt(np.mean((calculated[1:7] - truth[1:7]) ** 2))
    )
    assert metrics["mean_bias"] == pytest.approx(np.mean(calculated[1:7] - truth[1:7]))


def test_solve_holdbreath_red_ir_only_has_no_ut_columns(tmp_path: Path) -> None:
    data = tmp_path / "Spo2_HB1.csv"
    write_synthetic_red_ir_sensor_csv(data, seconds=70)
    truth_path = tmp_path / "Spo2_HB1_ref.csv"
    pd.DataFrame({"time_s": np.arange(70), "spo2": np.full(70, 98.0)}).to_csv(
        truth_path,
        index=False,
    )

    result = solve_spo2_holdbreath(
        HoldBreathSpO2Config(
            data_path=data,
            truth_path=truth_path,
            trim_seconds=5.0,
            fit_device_model=False,
        )
    )

    assert result.spo2_table
    assert "spo2_calculated" in result.aligned_table[0]
    assert all(
        "spo2_ut1" not in row and "spo2_ut2" not in row
        for row in result.spo2_table
    )


def test_fixed_device_model_does_not_refit_bias_or_lag() -> None:
    time_s = np.arange(20, dtype=float)
    raw = np.linspace(99, 94, 20)
    truth = raw + 10.0
    fixed = PulseOximeterModel(smooth_seconds=1.0, lag_seconds=2.0, bias=0.0)

    modeled, model, metrics = apply_or_fit_device_model(
        time_s,
        raw,
        truth,
        fit=False,
        fixed_model=fixed,
    )

    assert model == fixed
    assert metrics["device_model_fit"] is False
    assert np.isfinite(modeled).all()


def test_default_device_model_lag_search_matches_design_range() -> None:
    cfg = HoldBreathSpO2Config(data_path=Path("sample.csv"))

    assert cfg.lag_grid_seconds[0] == -20.0
    assert cfg.lag_grid_seconds[-1] == 20.0


def test_model_search_prefers_trend_shape_without_excessive_smoothing() -> None:
    time_s = np.arange(0, 31, dtype=float)
    raw = np.r_[
        np.full(8, 99.0),
        np.linspace(99, 92, 8),
        np.linspace(92, 98, 8),
        np.full(7, 98.0),
    ]
    truth = np.r_[
        np.full(10, 99.0),
        np.linspace(99, 92, 8),
        np.linspace(92, 98, 8),
        np.full(5, 98.0),
    ]

    modeled, model, metrics = apply_or_fit_device_model(
        time_s,
        raw,
        truth,
        fit=True,
        smooth_grid_seconds=(1.0, 3.0, 9.0, 15.0),
        lag_grid_seconds=(-1.0, 0.0, 1.0, 2.0, 3.0),
        fit_bias=False,
    )

    assert model.smooth_seconds <= 3.0
    assert abs(model.lag_seconds - 2.0) <= 1.0
    assert metrics["mae"] < compute_holdbreath_metrics(time_s, raw, truth)["mae"]
    assert np.nanmin(modeled) <= 93.0


def test_save_holdbreath_report_writes_csv_json_and_figures(tmp_path: Path) -> None:
    data = tmp_path / "Spo2_HB1.csv"
    write_synthetic_red_ir_sensor_csv(data, seconds=75)
    truth = tmp_path / "Spo2_HB1_ref.csv"
    pd.DataFrame({"time_s": np.arange(75), "spo2": np.full(75, 98.0)}).to_csv(
        truth,
        index=False,
    )
    result = solve_spo2_holdbreath(
        HoldBreathSpO2Config(
            data_path=data,
            truth_path=truth,
            trim_seconds=5.0,
            fit_device_model=False,
        )
    )

    outputs = save_holdbreath_report(
        result,
        out_dir=tmp_path / "out",
        output_prefix="Spo2_HB1",
    )

    assert outputs["json"].is_file()
    assert outputs["csv"].is_file()
    assert outputs["png"].is_file()
    assert outputs["svg"].is_file()
    assert outputs["pdf"].is_file()
    csv_rows = pd.read_csv(outputs["csv"])
    assert {"time_s", "spo2_calculated", "spo2_truth", "error"}.issubset(
        csv_rows.columns
    )


def test_holdbreath_band_estimate_uses_main_calculated_nadir_not_all_98_plateaus() -> None:
    time_s = np.arange(0, 181, dtype=float)
    calculated = np.r_[
        np.full(80, 99.0),
        np.linspace(99, 94, 20),
        np.linspace(94, 99, 20),
        np.full(61, 98.5),
    ]
    truth = np.r_[np.full(80, 98.0), np.linspace(98, 94, 20), np.full(81, 98.0)]
    result = HoldBreathSpO2Result(
        spo2_table=[],
        aligned_table=[
            {
                "time_s": float(t),
                "spo2_calculated": float(calc),
                "spo2_truth": float(ref),
                "error": float(calc - ref),
                "spo2_raw": float(calc),
                "device_model_lag_s": 0.0,
                "device_model_smooth_s": 1.0,
            }
            for t, calc, ref in zip(time_s, calculated, truth, strict=True)
        ],
        metrics={},
        metadata={},
    )

    band = _estimate_holdbreath_band_seconds(result)

    assert band is not None
    assert 70.0 <= band[0] <= 95.0
    assert 100.0 <= band[1] <= 125.0
    assert band[1] - band[0] < 60.0
