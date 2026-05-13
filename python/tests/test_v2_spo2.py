from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ppg_hr.v2.spo2 import (
    V2SpO2Coefficients,
    V2SpO2Config,
    _apply_rest_adaptive_policy,
    _amplitude_preserving_lms,
    _calc_ac_dc_by_valley_line,
    _clean_red_ir_adaptive,
    _compute_spo2_window,
    _delay_to_order,
    _load_spo2_raw_signals,
    _rank_references_for_window,
    _smooth_spo2_table,
    load_spo2_report,
    save_spo2_report,
    solve_spo2_v2,
    spo2_from_r,
)


def test_spo2_config_defaults_use_100hz_and_causal_lms(tmp_path: Path) -> None:
    cfg = V2SpO2Config(data_path=tmp_path / "sample.csv", output_dir=tmp_path)

    assert cfg.fs_origin == 100
    assert cfg.window_seconds == pytest.approx(4.0)
    assert cfg.window_step_seconds == pytest.approx(1.0)
    assert cfg.delay_search_samples == 20
    assert cfg.max_order == 20
    assert cfg.lms_mu_base == pytest.approx(0.01)
    assert cfg.lms_mu_min == pytest.approx(1e-6)
    assert cfg.reference_groups_order == ("HF", "CF", "ACC")
    assert cfg.adaptive_enabled is True


def test_spo2_from_r_uses_max30101_quadratic_coefficients() -> None:
    coeffs = V2SpO2Coefficients()
    r = np.array([0.5, 1.0, 2.0])

    out = spo2_from_r(r, coeffs)

    expected = coeffs.a * r**2 + coeffs.b * r + coeffs.c
    assert np.allclose(out, np.clip(expected, 0.0, 100.0))


def _write_spo2_sensor(path: Path, seconds: int = 12) -> None:
    fs = 100
    n = seconds * fs
    t = np.arange(n, dtype=float) / fs
    motion = 0.4 * np.sin(2 * np.pi * 1.5 * t)
    frame = pd.DataFrame(
        {
            "Time(s)": t,
            "Uc1(mV)": 1.0 + 0.01 * np.sin(t),
            "Uc2(mV)": 1.2 + 0.01 * np.cos(t),
            "Ut1(mV)": 5.0 + motion,
            "Ut2(mV)": 5.3 + 0.5 * motion,
            "PPG_Green": 1000.0 + 10.0 * np.sin(2 * np.pi * 1.2 * t),
            "PPG_Red": 900.0 + 20.0 * np.sin(2 * np.pi * 1.2 * t) + 6.0 * motion,
            "PPG_IR": 800.0 + 24.0 * np.sin(2 * np.pi * 1.2 * t) + 5.0 * motion,
            "AccX(g)": motion,
            "AccY(g)": np.zeros(n),
            "AccZ(g)": np.ones(n),
            "GyroX(dps)": np.zeros(n),
            "GyroY(dps)": np.zeros(n),
            "GyroZ(dps)": np.zeros(n),
        }
    )
    frame.to_csv(path, index=False)


def test_load_spo2_raw_signals_reads_red_ir_and_references(tmp_path: Path) -> None:
    data = tmp_path / "sample.csv"
    _write_spo2_sensor(data)
    cfg = V2SpO2Config(data_path=data)

    signals = _load_spo2_raw_signals(cfg)

    assert signals.fs == 100
    assert signals.red.shape == signals.ir.shape
    assert signals.red.size == 1200
    assert {"hf1", "hf2", "cf1", "cf2", "accx", "accy", "accz"}.issubset(
        signals.references
    )
    assert np.isfinite(signals.red).all()
    assert np.isfinite(signals.ir).all()


def test_delay_to_order_clips_to_maximum_20_samples() -> None:
    cfg = V2SpO2Config(data_path=Path("x.csv"), max_order=20, min_order=1)

    assert _delay_to_order(0, cfg) == 1
    assert _delay_to_order(7, cfg) == 7
    assert _delay_to_order(-9, cfg) == 9
    assert _delay_to_order(30, cfg) == 20


def test_rank_references_uses_100hz_plus_minus_20_sample_delay(
    tmp_path: Path,
) -> None:
    fs = 100
    n = 10 * fs
    t = np.arange(n, dtype=float) / fs
    ppg = np.sin(2 * np.pi * 1.4 * t)
    ref = np.roll(ppg, 8)
    cfg = V2SpO2Config(data_path=tmp_path / "sample.csv", delay_search_samples=20)

    ranked = _rank_references_for_window(
        target=ppg,
        references={"accx": ref, "accy": np.zeros_like(ref)},
        start=0,
        end=n,
        cfg=cfg,
    )

    assert ranked[0]["channel"] == "accx"
    assert abs(int(ranked[0]["delay_samples"])) <= 20
    assert ranked[0]["order"] == _delay_to_order(
        int(ranked[0]["delay_samples"]),
        cfg,
    )
    assert ranked[0]["corr"] > 0.9


def test_amplitude_preserving_lms_reduces_reference_correlated_artifact() -> None:
    fs = 100
    t = np.arange(12 * fs, dtype=float) / fs
    pulse_with_dc = 900.0 + 20.0 * np.sin(2 * np.pi * 1.2 * t)
    artifact_ref = np.sin(2 * np.pi * 2.0 * t)
    contaminated = pulse_with_dc + 15.0 * artifact_ref
    cfg = V2SpO2Config(data_path=Path("x.csv"), lms_mu_base=0.01, max_order=20)

    cleaned = _amplitude_preserving_lms(
        desired=contaminated,
        reference=artifact_ref,
        order=10,
        corr=80.0,
        cfg=cfg,
    )

    before_corr = abs(
        np.corrcoef(contaminated - np.mean(contaminated), artifact_ref)[0, 1]
    )
    after_corr = abs(np.corrcoef(cleaned - np.mean(cleaned), artifact_ref)[0, 1])
    assert after_corr < before_corr
    assert np.median(cleaned) == pytest.approx(np.median(contaminated), abs=1.0)


def test_clean_red_ir_adaptive_uses_same_reference_order_for_both_channels() -> None:
    fs = 100
    t = np.arange(12 * fs, dtype=float) / fs
    artifact = np.sin(2 * np.pi * 1.5 * t)
    red = 900.0 + 18.0 * np.sin(2 * np.pi * 1.2 * t) + 10.0 * artifact
    ir = 800.0 + 24.0 * np.sin(2 * np.pi * 1.2 * t) + 8.0 * artifact
    refs = {"accx": artifact, "accy": np.zeros_like(artifact)}
    cfg = V2SpO2Config(data_path=Path("x.csv"))

    cleaned = _clean_red_ir_adaptive(red, ir, refs, start=0, end=red.size, cfg=cfg)

    assert cleaned.red_clean.shape == red.shape
    assert cleaned.ir_clean.shape == ir.shape
    assert cleaned.stages[0]["channel"] == "accx"
    assert cleaned.stages[0]["order"] <= 20
    assert np.median(cleaned.red_clean) == pytest.approx(np.median(red), abs=1.0)
    assert np.median(cleaned.ir_clean) == pytest.approx(np.median(ir), abs=1.0)


def test_calc_ac_dc_by_valley_line_uses_peak_baseline() -> None:
    adc = np.array([100.0, 110.0, 120.0, 110.0, 100.0])

    ac, dc = _calc_ac_dc_by_valley_line(adc, 0, 2, 4)

    assert dc == pytest.approx(100.0)
    assert ac == pytest.approx(20.0)


def test_compute_spo2_window_detects_ratio_from_ir_cycles() -> None:
    fs = 100
    t = np.arange(4 * fs, dtype=float) / fs
    ir = 800.0 - 24.0 * np.cos(2 * np.pi * 1.2 * t)
    red = 900.0 - 20.0 * np.cos(2 * np.pi * 1.2 * t)
    cfg = V2SpO2Config(data_path=Path("x.csv"))

    out = _compute_spo2_window(red=red, ir=ir, fs=fs, cfg=cfg, scheme="raw")

    expected_r = (20.0 / 900.0) / (24.0 / 800.0)
    assert out["valid_beat_count"] >= 2
    assert out["r_median"] == pytest.approx(expected_r, rel=0.15)
    assert 0.0 <= out["spo2"] <= 100.0
    assert {row["scheme"] for row in out["beat_rows"]} == {"raw"}


def test_solve_spo2_v2_outputs_one_second_spo2_windows(tmp_path: Path) -> None:
    data = tmp_path / "sample.csv"
    _write_spo2_sensor(data, seconds=12)
    cfg = V2SpO2Config(data_path=data, output_dir=tmp_path)

    result = solve_spo2_v2(cfg)

    assert len(result.spo2_table) == 9
    first_valid = next(
        row for row in result.spo2_table if np.isfinite(row["adaptive_spo2"])
    )
    assert 0.0 <= first_valid["raw_spo2"] <= 100.0
    assert 0.0 <= first_valid["adaptive_spo2"] <= 100.0
    assert first_valid["spo2"] == first_valid["adaptive_spo2"]
    assert first_valid["raw_valid_beat_count"] >= 1
    assert first_valid["adaptive_valid_beat_count"] >= 1
    assert result.waveforms["red_raw"].shape == result.waveforms["red_clean"].shape
    assert result.waveforms["ir_raw"].shape == result.waveforms["ir_clean"].shape
    assert result.metadata["fs"] == 100
    assert result.metadata["spo2_smooth_seconds"] == pytest.approx(7.0)


def test_spo2_table_applies_7s_average_to_remove_spikes() -> None:
    rows = [
        {
            "raw_spo2": value,
            "adaptive_spo2": value,
            "spo2": value,
            "motion_score": 1.0,
        }
        for value in [96.0, 96.0, 96.0, 80.0, 96.0, 96.0, 96.0]
    ]
    cfg = V2SpO2Config(data_path=Path("x.csv"), spo2_smooth_seconds=7.0)

    _smooth_spo2_table(rows, cfg)

    assert rows[3]["adaptive_spo2_unsmoothed"] == pytest.approx(80.0)
    assert rows[3]["adaptive_spo2"] == pytest.approx(np.mean([96.0] * 6 + [80.0]))
    assert rows[3]["spo2"] == rows[3]["adaptive_spo2"]


def test_rest_windows_use_raw_spo2_without_adaptive_comparison() -> None:
    rows = [
        {
            "raw_spo2": 96.0,
            "adaptive_spo2": 94.0,
            "spo2": 94.0,
            "raw_r_median": 0.7,
            "adaptive_r_median": 0.8,
            "raw_valid_beat_count": 3,
            "adaptive_valid_beat_count": 2,
            "raw_carried_forward": False,
            "adaptive_carried_forward": False,
            "motion_score": 0.001,
            "reliable": True,
        },
        {
            "raw_spo2": 92.0,
            "adaptive_spo2": 95.0,
            "spo2": 95.0,
            "raw_r_median": 0.9,
            "adaptive_r_median": 0.6,
            "raw_valid_beat_count": 3,
            "adaptive_valid_beat_count": 3,
            "raw_carried_forward": False,
            "adaptive_carried_forward": False,
            "motion_score": 0.2,
            "reliable": True,
        },
    ]
    cfg = V2SpO2Config(data_path=Path("x.csv"), rest_motion_score_threshold=0.02)

    _apply_rest_adaptive_policy(rows, cfg)

    assert rows[0]["adaptive_applied"] is False
    assert rows[0]["adaptive_spo2"] == rows[0]["raw_spo2"]
    assert rows[0]["spo2"] == rows[0]["raw_spo2"]
    assert rows[0]["adaptive_r_median"] == rows[0]["raw_r_median"]
    assert rows[0]["adaptive_valid_beat_count"] == rows[0]["raw_valid_beat_count"]
    assert rows[1]["adaptive_applied"] is True
    assert rows[1]["spo2"] == rows[1]["adaptive_spo2"]


def test_solver_skips_adaptive_filtering_for_static_rest_windows(tmp_path: Path) -> None:
    data = tmp_path / "rest.csv"
    fs = 100
    n = 8 * fs
    t = np.arange(n, dtype=float) / fs
    artifact = np.sin(2 * np.pi * 1.5 * t)
    frame = pd.DataFrame(
        {
            "Time(s)": t,
            "Uc1(mV)": np.ones(n),
            "Uc2(mV)": np.ones(n) * 1.2,
            "Ut1(mV)": 5.0 + artifact,
            "Ut2(mV)": np.ones(n) * 5.2,
            "PPG_Green": 1000.0 + 10.0 * np.sin(2 * np.pi * 1.2 * t),
            "PPG_Red": 900.0 - 20.0 * np.cos(2 * np.pi * 1.2 * t) + 8.0 * artifact,
            "PPG_IR": 800.0 - 24.0 * np.cos(2 * np.pi * 1.2 * t) + 7.0 * artifact,
            "AccX(g)": np.zeros(n),
            "AccY(g)": np.zeros(n),
            "AccZ(g)": np.ones(n),
            "GyroX(dps)": np.zeros(n),
            "GyroY(dps)": np.zeros(n),
            "GyroZ(dps)": np.zeros(n),
        }
    )
    frame.to_csv(data, index=False)

    result = solve_spo2_v2(V2SpO2Config(data_path=data, output_dir=tmp_path))

    assert all(row["adaptive_applied"] is False for row in result.spo2_table)
    assert all(not stages for stages in result.metadata["adaptive_stage_rows"])
    assert np.allclose(result.waveforms["red_clean"], result.waveforms["red_raw"])
    assert np.allclose(result.waveforms["ir_clean"], result.waveforms["ir_raw"])


def test_save_and_load_spo2_report_writes_json_csv_and_waveforms(
    tmp_path: Path,
) -> None:
    data = tmp_path / "sample.csv"
    _write_spo2_sensor(data, seconds=8)
    result = solve_spo2_v2(V2SpO2Config(data_path=data, output_dir=tmp_path))

    outputs = save_spo2_report(result, out_dir=tmp_path, output_prefix="sample")
    payload = load_spo2_report(outputs["json"])
    csv_frame = pd.read_csv(outputs["csv"])

    assert outputs["json"].is_file()
    assert outputs["csv"].is_file()
    assert payload["schema_version"] == "v2_spo2"
    assert len(payload["spo2_table"]) == len(result.spo2_table)
    assert len(payload["waveforms"]["red_raw"]) == result.waveforms["red_raw"].size
    assert {"raw_spo2", "adaptive_spo2", "motion_score"}.issubset(csv_frame.columns)
    assert json.loads(outputs["json"].read_text(encoding="utf-8"))["metadata"]["fs"] == 100
