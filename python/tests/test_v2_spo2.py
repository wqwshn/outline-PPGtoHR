from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ppg_hr.v2.spo2 import (
    V2SpO2Coefficients,
    V2SpO2Config,
    _delay_to_order,
    _load_spo2_raw_signals,
    _rank_references_for_window,
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
