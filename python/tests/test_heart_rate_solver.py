"""Tests for ``heart_rate_solver``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppg_hr.core.heart_rate_solver import SolverResult, solve
from ppg_hr.io.golden import assert_array_close, load_golden
from ppg_hr.params import SolverParams


SCENARIO = "multi_tiaosheng1"


@pytest.fixture(scope="module")
def csv_pair(dataset_dir: Path) -> tuple[Path, Path]:
    sensor = dataset_dir / f"{SCENARIO}.csv"
    gt = dataset_dir / f"{SCENARIO}_ref.csv"
    if not sensor.is_file() or not gt.is_file():
        pytest.skip(f"CSV files for {SCENARIO} not found")
    return sensor, gt


@pytest.fixture(scope="module")
def result(csv_pair) -> SolverResult:
    sensor, gt = csv_pair
    params = SolverParams(file_name=sensor, ref_file=gt)
    return solve(params)


def test_hr_matrix_shape(result: SolverResult) -> None:
    assert result.HR.shape[1] == 9
    assert result.HR.shape[0] > 50  # at least 50 windows for a 4-minute scenario


def test_err_stats_shape(result: SolverResult) -> None:
    assert result.err_stats.shape == (5, 3)


def test_motion_threshold_positive(result: SolverResult) -> None:
    assert result.motion_threshold[0] > 0
    assert result.motion_threshold[1] == result.motion_threshold[0]


def test_aae_within_reasonable_range(result: SolverResult) -> None:
    # Fusion-HF total AAE should be sensible (≤ 25 BPM in worst case)
    aae_total = result.err_stats[3, 0]
    assert np.isfinite(aae_total)
    assert 0 < aae_total < 25, f"unexpected AAE: {aae_total}"


def test_t_pred_shifted_by_time_bias(result: SolverResult) -> None:
    np.testing.assert_allclose(result.T_Pred, result.HR[:, 0] + 5.0, atol=1e-12)


def test_motion_flag_consistency(result: SolverResult) -> None:
    # Cols 8 and 9 (motion_acc / motion_hf) are forced equal in MATLAB code
    np.testing.assert_array_equal(result.HR[:, 7], result.HR[:, 8])


def test_ref_csv_must_exist(tmp_path: Path) -> None:
    fake_csv = tmp_path / "fake.csv"
    fake_csv.write_text("Time(s)\n0\n")
    with pytest.raises(FileNotFoundError):
        solve(SolverParams(file_name=fake_csv))


def test_matches_golden_e2e(dataset_dir: Path, golden_dir: Path) -> None:
    scenario = SCENARIO  # 单一典型场景足以验证重构等价性
    mat_path = golden_dir / f"e2e_{scenario}.mat"
    if not mat_path.is_file():
        pytest.skip(f"Run MATLAB/gen_golden_all.m to produce e2e_{scenario}.mat")
    sensor = dataset_dir / f"{scenario}.csv"
    gt = dataset_dir / f"{scenario}_ref.csv"
    if not sensor.is_file() or not gt.is_file():
        pytest.skip(f"Missing CSV input for {scenario}")

    snap = load_golden(mat_path)
    expected_struct = snap["Res"]
    if hasattr(expected_struct, "_fieldnames"):
        expected_HR = np.asarray(expected_struct.HR, dtype=float)
        expected_err = np.asarray(expected_struct.err_stats, dtype=float)
    else:
        expected_HR = np.asarray(expected_struct["HR"], dtype=float)
        expected_err = np.asarray(expected_struct["err_stats"], dtype=float)

    res = solve(SolverParams(file_name=sensor, ref_file=gt))
    assert res.HR.shape == expected_HR.shape, "HR matrix shape mismatch"
    assert_array_close(res.HR, expected_HR, atol=1e-3, rtol=1e-3, err_msg="HR matrix")
    assert_array_close(
        res.err_stats, expected_err, atol=5e-3, rtol=5e-3, err_msg="err_stats"
    )
