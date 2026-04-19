"""Behavioural tests for the result-viewer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ppg_hr.params import SolverParams
from ppg_hr.visualization import load_report, render

SCENARIO = "multi_tiaosheng1"


@pytest.fixture(scope="module")
def base_params(dataset_dir: Path) -> SolverParams:
    sensor = dataset_dir / f"{SCENARIO}.csv"
    gt = dataset_dir / f"{SCENARIO}_ref.csv"
    if not sensor.is_file() or not gt.is_file():
        pytest.skip(f"CSV files for {SCENARIO} not found")
    return SolverParams(file_name=sensor, ref_file=gt)


@pytest.fixture()
def fake_json_report(tmp_path: Path) -> Path:
    # Minimal report using solver defaults; stress-tests the viewer without
    # needing a real Bayesian search to have run.
    payload = {
        "min_err_hf": 8.0,
        "min_err_acc": 7.5,
        "best_para_hf": {
            "fs_target": 100, "max_order": 16, "spec_penalty_width": 0.2,
            "hr_range_hz": 25 / 60, "slew_limit_bpm": 10, "slew_step_bpm": 7,
            "hr_range_rest": 30 / 60, "slew_limit_rest": 6, "slew_step_rest": 4,
            "smooth_win_len": 7, "time_bias": 5,
        },
        "best_para_acc": {
            "fs_target": 100, "max_order": 20, "spec_penalty_width": 0.3,
            "hr_range_hz": 25 / 60, "slew_limit_bpm": 12, "slew_step_bpm": 9,
            "hr_range_rest": 30 / 60, "slew_limit_rest": 7, "slew_step_rest": 5,
            "smooth_win_len": 5, "time_bias": 6,
        },
        "importance_hf": None,
        "search_space": {},
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_report_json(fake_json_report: Path) -> None:
    rep = load_report(fake_json_report)
    assert rep["min_err_hf"] == 8.0
    assert rep["best_para_acc"]["max_order"] == 20


def test_load_report_unknown_extension(tmp_path: Path) -> None:
    bogus = tmp_path / "x.txt"
    bogus.write_text("noop")
    with pytest.raises(ValueError):
        load_report(bogus)


def test_load_report_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_report(tmp_path / "nope.json")


def test_render_emits_figure_and_csvs(
    base_params: SolverParams,
    fake_json_report: Path,
    tmp_path: Path,
) -> None:
    artefacts = render(
        fake_json_report, base_params, out_dir=tmp_path / "out", show=False
    )
    assert artefacts.figure is not None and artefacts.figure.is_file()
    assert artefacts.error_csv is not None and artefacts.error_csv.is_file()
    assert artefacts.param_csv is not None and artefacts.param_csv.is_file()

    with artefacts.error_csv.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert header == ["case", "method", "total_aae", "rest_aae", "motion_aae"]
    assert len(rows) == 1 + 2 * 5  # header + 5 methods × 2 cases

    with artefacts.param_csv.open(encoding="utf-8") as f:
        prows = list(csv.reader(f))
    first_col = [r[0] for r in prows]
    assert "target_aae" in first_col
    assert "motion_aae" in first_col
    assert "fs_target" in first_col
    assert "time_bias" in first_col
