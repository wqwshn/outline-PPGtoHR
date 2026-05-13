from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.spo2 import (
    V2SpO2Coefficients,
    V2SpO2Config,
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
