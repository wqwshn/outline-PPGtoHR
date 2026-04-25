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


def _make_strategy_report(tmp_path: Path, strategy: str) -> Path:
    payload = {
        "adaptive_filter": strategy,
        "min_err_hf": 1.0,
        "min_err_acc": 1.0,
        "best_para_hf": {"fs_target": 100, "max_order": 16},
        "best_para_acc": {"fs_target": 100, "max_order": 16},
        "importance_hf": None,
        "search_space": {},
    }
    path = tmp_path / f"report_{strategy}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("strategy", ["lms", "klms", "volterra"])
def test_render_honours_report_strategy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, strategy: str,
) -> None:
    """Reports produced for klms/volterra must replay with that strategy."""
    seen_strategies: list[str] = []

    class _FakeRes:
        err_stats = [[0] * 3] * 5
        HR = [[0] * 9] * 1
        T_Pred = [0]
        HR_Ref_Interp = [0]

    def _fake_solve(params):
        seen_strategies.append(params.adaptive_filter)
        # Return an object minimally compatible with the parts render()
        # / write_*_csv touch.
        import numpy as np
        res = type("R", (), {})()
        res.err_stats = np.zeros((5, 3))
        res.HR = np.zeros((1, 9))
        res.T_Pred = np.zeros((1,))
        res.HR_Ref_Interp = np.zeros((1,))
        res.motion_threshold = (0.0, 0.0)
        return res

    monkeypatch.setattr("ppg_hr.visualization.result_viewer.solve", _fake_solve)
    report = _make_strategy_report(tmp_path, strategy)
    base = SolverParams(file_name=tmp_path / "dummy.csv")  # adaptive_filter="lms"
    render(report, base, out_dir=tmp_path / "out", show=False)

    # render() calls solve() twice (HF + ACC) — both must use the report strategy.
    assert seen_strategies == [strategy, strategy]


def test_render_defaults_to_base_strategy_when_report_lacks_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Legacy reports (no adaptive_filter) keep the base SolverParams choice."""
    seen: list[str] = []

    def _fake_solve(params):
        seen.append(params.adaptive_filter)
        import numpy as np
        res = type("R", (), {})()
        res.err_stats = np.zeros((5, 3))
        res.HR = np.zeros((1, 9))
        res.T_Pred = np.zeros((1,))
        res.HR_Ref_Interp = np.zeros((1,))
        res.motion_threshold = (0.0, 0.0)
        return res

    monkeypatch.setattr("ppg_hr.visualization.result_viewer.solve", _fake_solve)
    legacy = {
        "min_err_hf": 1.0, "min_err_acc": 1.0,
        "best_para_hf": {}, "best_para_acc": {},
        "importance_hf": None, "search_space": {},
    }
    report = tmp_path / "legacy.json"
    report.write_text(json.dumps(legacy), encoding="utf-8")
    base = SolverParams(file_name=tmp_path / "x.csv").replace(
        adaptive_filter="volterra"
    )
    render(report, base, out_dir=tmp_path / "out", show=False)
    assert seen == ["volterra", "volterra"]


def test_render_honours_report_delay_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen_modes: list[str] = []

    def _fake_solve(params):
        seen_modes.append(params.delay_search_mode)
        import numpy as np
        res = type("R", (), {})()
        res.err_stats = np.zeros((5, 3))
        res.HR = np.zeros((1, 9))
        res.T_Pred = np.zeros((1,))
        res.HR_Ref_Interp = np.zeros((1,))
        res.motion_threshold = (0.0, 0.0)
        res.delay_profile = None
        return res

    monkeypatch.setattr("ppg_hr.visualization.result_viewer.solve", _fake_solve)
    report = {
        "min_err_hf": 1.0,
        "min_err_acc": 1.0,
        "best_para_hf": {},
        "best_para_acc": {},
        "delay_search": {"delay_search_mode": "fixed"},
        "importance_hf": None,
        "search_space": {},
    }
    report_path = tmp_path / "delay.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    base = SolverParams(file_name=tmp_path / "x.csv")
    render(report_path, base, out_dir=tmp_path / "out", show=False)
    assert seen_modes == ["fixed", "fixed"]


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


def test_render_can_prefix_output_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_solve(params):
        import numpy as np
        res = type("R", (), {})()
        res.err_stats = np.zeros((5, 3))
        res.HR = np.zeros((1, 9))
        res.T_Pred = np.zeros((1,))
        res.HR_Ref_Interp = np.zeros((1,))
        res.motion_threshold = (0.0, 0.0)
        res.delay_profile = None
        return res

    monkeypatch.setattr("ppg_hr.visualization.result_viewer.solve", _fake_solve)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({
            "min_err_hf": 1.0,
            "min_err_acc": 1.0,
            "best_para_hf": {},
            "best_para_acc": {},
        }),
        encoding="utf-8",
    )
    artefacts = render(
        report,
        SolverParams(file_name=tmp_path / "multi_bobi1.csv"),
        out_dir=tmp_path / "viewer_out" / "multi_bobi1",
        output_prefix="multi_bobi1",
        show=False,
    )

    assert artefacts.figure == tmp_path / "viewer_out" / "multi_bobi1" / "multi_bobi1-figure.png"
    assert artefacts.error_csv == tmp_path / "viewer_out" / "multi_bobi1" / "multi_bobi1-error_table.csv"
    assert artefacts.param_csv == tmp_path / "viewer_out" / "multi_bobi1" / "multi_bobi1-param_table.csv"
