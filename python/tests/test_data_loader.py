"""Tests for the sensor + reference CSV loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.io.golden import load_golden
from ppg_hr.preprocess import SENSOR_COLUMNS, ProcessedDataset, load_dataset

SCENARIO = "multi_tiaosheng1"


@pytest.fixture(scope="module")
def loaded(dataset_dir: Path) -> ProcessedDataset:
    sensor = dataset_dir / f"{SCENARIO}.csv"
    gt = dataset_dir / f"{SCENARIO}_ref.csv"
    if not sensor.is_file() or not gt.is_file():
        pytest.skip(f"Missing test CSV under {dataset_dir}")
    return load_dataset(sensor, gt)


def test_dataframe_shape(loaded: ProcessedDataset) -> None:
    n = len(loaded.data)
    assert n > 0
    expected_cols = (
        ["Time_s"]
        + list(SENSOR_COLUMNS)
        + [f"{short}_Filt" for short in SENSOR_COLUMNS]
    )
    assert list(loaded.data.columns) == expected_cols


def test_time_axis_is_100hz(loaded: ProcessedDataset) -> None:
    t = loaded.data["Time_s"].to_numpy()
    np.testing.assert_allclose(np.diff(t), 0.01, atol=1e-12)
    assert t[0] == 0.0


def test_ref_data_layout(loaded: ProcessedDataset) -> None:
    ref = loaded.ref_data
    assert ref.ndim == 2 and ref.shape[1] == 2
    assert ref.shape[0] > 0
    assert np.all(ref[:, 1] > 30) and np.all(ref[:, 1] < 220)
    assert np.all(np.diff(ref[:, 0]) >= 0)


def test_filtered_signals_zero_mean(loaded: ProcessedDataset) -> None:
    for short in SENSOR_COLUMNS:
        filtered = loaded.data[f"{short}_Filt"].to_numpy()
        assert abs(filtered.mean()) < max(abs(loaded.data[short].mean()) * 0.1, 5.0)


def test_no_negative_ppg_after_correction(loaded: ProcessedDataset) -> None:
    for short in ("PPG_Green", "PPG_Red", "PPG_IR"):
        assert loaded.data[short].min() >= 0


def test_unknown_column_raises(dataset_dir: Path) -> None:
    sensor = dataset_dir / f"{SCENARIO}.csv"
    gt = dataset_dir / f"{SCENARIO}_ref.csv"
    if not sensor.is_file() or not gt.is_file():
        pytest.skip("Missing test CSV")
    with pytest.raises(KeyError):
        load_dataset(sensor, gt, columns=["unknown_channel"])


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv", tmp_path / "ref.csv")


# --- Golden-snapshot alignment (skipped if .mat missing) ----------------------


def _struct_to_dict(struct_obj) -> dict[str, np.ndarray]:
    """Convert scipy.io.loadmat struct (mat_struct) to a flat dict of numpy arrays."""
    if hasattr(struct_obj, "_fieldnames"):
        return {name: np.asarray(getattr(struct_obj, name)) for name in struct_obj._fieldnames}
    if isinstance(struct_obj, dict):
        return {k: np.asarray(v) for k, v in struct_obj.items()}
    raise TypeError(f"Cannot interpret struct object of type {type(struct_obj)}")


def test_data_loader_matches_golden(
    loaded: ProcessedDataset, golden_dir: Path
) -> None:
    mat_path = golden_dir / "data_loader.mat"
    if not mat_path.is_file():
        pytest.skip(
            "Run MATLAB/gen_golden_all.m to produce data_loader.mat before "
            "running this strict-alignment test"
        )
    snap = load_golden(mat_path)
    expected = _struct_to_dict(snap["data_struct"])
    actual = loaded.data

    # MATLAB column name -> Python column name
    name_map = {"Time_s": "Time_s"}
    for short in SENSOR_COLUMNS:
        name_map[short] = short
        name_map[f"{short}_Filt"] = f"{short}_Filt"

    for matlab_name, py_name in name_map.items():
        if matlab_name not in expected:
            continue
        exp = np.asarray(expected[matlab_name]).squeeze()
        act = actual[py_name].to_numpy()
        assert exp.shape == act.shape, f"{py_name} shape mismatch"
        # tolerance: filtfilt + filloutliers introduce floating accumulation
        np.testing.assert_allclose(
            act, exp, atol=1e-6, rtol=1e-6, err_msg=f"column {py_name}"
        )

    if "ref_data_dl" in snap:
        exp_ref = np.asarray(snap["ref_data_dl"])
        np.testing.assert_allclose(loaded.ref_data, exp_ref, atol=1e-9)
