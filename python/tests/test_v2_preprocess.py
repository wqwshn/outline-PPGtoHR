from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ppg_hr.v2.preprocess import V2_CHANNELS, load_v2_dataset, safe_cf_ratio


def _write_ref(path: Path) -> None:
    path.write_text(
        "header1\nheader2\nheader3\n0,00:00:00,75\n1,00:00:01,76\n",
        encoding="utf-8",
    )


def _raw_frame(n: int = 120) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "Uc1(mV)": 1.0 + 0.01 * t,
            "Uc2(mV)": 2.0 + 0.01 * t,
            "Ut1(mV)": 5.0 + 0.02 * t,
            "Ut2(mV)": 7.0 + 0.02 * t,
            "PPG_Green": 1000.0 + np.sin(t / 10.0),
            "PPG_Red": 900.0 + np.sin(t / 11.0),
            "PPG_IR": 800.0 + np.sin(t / 12.0),
            "AccX(g)": 0.1 * np.sin(t / 8.0),
            "AccY(g)": 0.1 * np.cos(t / 8.0),
            "AccZ(g)": 1.0 + 0.01 * np.sin(t / 9.0),
            "GyroX(dps)": 0.01 * t,
            "GyroY(dps)": 0.02 * t,
            "GyroZ(dps)": 0.03 * t,
        }
    )


def test_safe_cf_ratio_outputs_finite_values() -> None:
    uc = np.array([1.0, 2.0, 3.0, np.nan])
    ut = np.array([2.0, 2.0, 6.0, 8.0])

    out = safe_cf_ratio(uc, ut)

    assert out.shape == uc.shape
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(1.0)
    assert out[2] == pytest.approx(1.0)


def test_load_v2_dataset_derives_cf_and_keeps_13_protocol_channels(
    tmp_path: Path,
) -> None:
    sensor = tmp_path / "sample.csv"
    ref = tmp_path / "sample_ref.csv"
    _raw_frame().to_csv(sensor, index=False)
    _write_ref(ref)

    ds = load_v2_dataset(sensor, ref, fs_origin=100)

    assert tuple(ds.data.columns) == ("time_s", *V2_CHANNELS)
    assert ds.fs == 100
    assert ds.sample_stem == "sample"
    assert np.isfinite(ds.data.to_numpy(dtype=float)).all()
    assert ds.data["cf1"].iloc[0] == pytest.approx(1.0 / 4.0)
    assert ds.data["cf2"].iloc[0] == pytest.approx(2.0 / 5.0)
    assert ds.ref_data.shape[1] == 2


def test_load_v2_dataset_requires_standard_columns(tmp_path: Path) -> None:
    sensor = tmp_path / "bad.csv"
    ref = tmp_path / "bad_ref.csv"
    frame = _raw_frame().drop(columns=["Ut1(mV)"])
    frame.to_csv(sensor, index=False)
    _write_ref(ref)

    with pytest.raises(KeyError, match="Ut1"):
        load_v2_dataset(sensor, ref)
