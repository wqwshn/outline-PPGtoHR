from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ppg_hr.v2.qc import quality_filter_sample_v2


def _write_sample(path: Path, *, spike: bool = False) -> None:
    n = 1200
    t = np.arange(n, dtype=float) / 100.0
    ut1 = 4.0 + 0.01 * t
    ut2 = 5.0 + 0.01 * t
    if spike:
        ut1[100:300] += np.sin(np.arange(200)) * 8.0
    df = pd.DataFrame(
        {
            "Uc1(mV)": 1.0,
            "Uc2(mV)": 1.2,
            "Ut1(mV)": ut1,
            "Ut2(mV)": ut2,
            "PPG_Green": 1000.0,
            "PPG_Red": 900.0,
            "PPG_IR": 800.0,
            "AccX(g)": 0.0,
            "AccY(g)": 0.0,
            "AccZ(g)": 1.0,
            "GyroX(dps)": 0.0,
            "GyroY(dps)": 0.0,
            "GyroZ(dps)": 0.0,
        }
    )
    df.to_csv(path, index=False)


def test_quality_filter_good_sample(tmp_path: Path) -> None:
    sample = tmp_path / "good.csv"
    ref = tmp_path / "good_ref.csv"
    _write_sample(sample)
    ref.write_text("ref", encoding="utf-8")

    qc = quality_filter_sample_v2(sample, ref_csv=ref)

    assert qc.status == "good"
    assert qc.reason == "ok"
    assert qc.ref_file == str(ref)
    assert qc.is_good


def test_quality_filter_bad_sample_is_marked_not_blocking(tmp_path: Path) -> None:
    sample = tmp_path / "bad.csv"
    _write_sample(sample, spike=True)

    qc = quality_filter_sample_v2(sample)

    assert qc.status == "bad"
    assert not qc.is_good
    assert "STD" in qc.reason
    assert qc.data_file == str(sample)


def test_quality_filter_missing_columns_returns_bad(tmp_path: Path) -> None:
    sample = tmp_path / "missing.csv"
    pd.DataFrame({"Ut1(mV)": [1.0, 2.0]}).to_csv(sample, index=False)

    qc = quality_filter_sample_v2(sample)

    assert qc.status == "bad"
    assert "missing required columns" in qc.reason
