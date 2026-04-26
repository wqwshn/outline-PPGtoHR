"""Behavioural tests for the result-viewer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ppg_hr.params import SolverParams
from ppg_hr.visualization import load_report, render
from ppg_hr.visualization import result_viewer

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


def test_render_uses_one_motion_mask_for_both_panels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import numpy as np

    report = {
        "min_err_hf": 1.0,
        "min_err_acc": 1.0,
        "best_para_hf": {"fs_target": 50},
        "best_para_acc": {"fs_target": 100},
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    hf = type("R", (), {})()
    hf.err_stats = np.zeros((5, 3))
    hf.HR = np.zeros((5, 9), dtype=float)
    hf.HR[:, 0] = np.arange(5, dtype=float)
    hf.HR[:, 7] = [0, 1, 1, 0, 0]
    hf.HR[:, 8] = hf.HR[:, 7]
    hf.T_Pred = hf.HR[:, 0]
    hf.HR_Ref_Interp = np.zeros((5,))
    hf.delay_profile = None

    acc = type("R", (), {})()
    acc.err_stats = np.zeros((5, 3))
    acc.HR = np.zeros((5, 9), dtype=float)
    acc.HR[:, 0] = np.arange(5, dtype=float)
    acc.HR[:, 7] = [0, 0, 1, 1, 1]
    acc.HR[:, 8] = acc.HR[:, 7]
    acc.T_Pred = acc.HR[:, 0]
    acc.HR_Ref_Interp = np.zeros((5,))
    acc.delay_profile = None

    results = [hf, acc]
    plotted_masks: list[list[float]] = []

    def _fake_solve(params):
        return results.pop(0)

    def _fake_plot_panel(ax, res, label, min_err, **kwargs):
        plotted_masks.append(res.HR[:, 7].tolist())

    monkeypatch.setattr("ppg_hr.visualization.result_viewer.solve", _fake_solve)
    monkeypatch.setattr("ppg_hr.visualization.result_viewer._plot_panel", _fake_plot_panel)

    render(
        report_path,
        SolverParams(file_name=tmp_path / "multi.csv"),
        out_dir=tmp_path / "out",
        show=False,
    )

    assert plotted_masks == [[0, 1, 1, 0, 0], [0, 1, 1, 0, 0]]


def test_plot_panel_labels_core_methods_with_errors() -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    res = type("R", (), {})()
    res.err_stats = np.array([
        [2.345, 0.0, 0.0],
        [4.567, 0.0, 0.0],
        [6.789, 0.0, 0.0],
        [1.111, 0.0, 0.0],
        [1.222, 0.0, 0.0],
    ])
    res.HR = np.array([
        [0.0, 1.30, 1.31, 1.32, 1.33, 1.30, 1.31, 0.0, 0.0],
        [1.0, 1.40, 1.41, 1.42, 1.43, 1.40, 1.41, 1.0, 1.0],
        [2.0, 1.50, 1.51, 1.52, 1.53, 1.50, 1.51, 0.0, 0.0],
    ])
    res.T_Pred = res.HR[:, 0]

    fig, ax = plt.subplots()
    result_viewer._plot_panel(ax, res, "HF best", 1.111)

    labels = ax.get_legend_handles_labels()[1]
    assert labels == [
        "Reference",
        "Pure FFT (AAE=6.79 BPM)",
        "HF-LMS (AAE=1.11 BPM)",
        "ACC-LMS (AAE=1.22 BPM)",
    ]

    lines = {line.get_label(): line for line in ax.lines}
    hf_line = lines["HF-LMS (AAE=1.11 BPM)"]
    acc_line = lines["ACC-LMS (AAE=1.22 BPM)"]
    assert hf_line.get_color().lower() == "#e07a8a"
    assert hf_line.get_linewidth() > acc_line.get_linewidth()
    assert "Motion MAE:" in ax.get_title(loc="left")
    plt.close(fig)


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

    out_dir = tmp_path / "viewer_out" / "multi_bobi1"
    assert artefacts.figure == out_dir / "multi_bobi1-full-hf-best.png"
    assert artefacts.extras["figure_hf"] == out_dir / "multi_bobi1-full-hf-best.png"
    assert artefacts.extras["figure_acc"] == out_dir / "multi_bobi1-full-acc-best.png"
    assert artefacts.error_csv == out_dir / "multi_bobi1-full-error_table.csv"
    assert artefacts.param_csv == out_dir / "multi_bobi1-full-param_table.csv"


def test_render_exports_two_png_figures_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_solve(params):
        import numpy as np
        res = type("R", (), {})()
        res.err_stats = np.ones((5, 3))
        res.HR = np.array([
            [0.0, 1.30, 1.31, 1.32, 1.33, 1.30, 1.31, 0.0, 0.0],
            [1.0, 1.40, 1.41, 1.42, 1.43, 1.40, 1.41, 1.0, 1.0],
            [2.0, 1.50, 1.51, 1.52, 1.53, 1.50, 1.51, 0.0, 0.0],
        ])
        res.T_Pred = res.HR[:, 0]
        res.HR_Ref_Interp = res.HR[:, 1]
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
        SolverParams(file_name=tmp_path / "multi_bobi1.csv", analysis_scope="motion"),
        out_dir=tmp_path / "viewer_out",
        output_prefix="multi_bobi1",
        show=False,
    )

    assert artefacts.figure == tmp_path / "viewer_out" / "multi_bobi1-motion-hf-best.png"
    assert artefacts.extras["figure_hf"] == artefacts.figure
    assert artefacts.extras["figure_acc"] == tmp_path / "viewer_out" / "multi_bobi1-motion-acc-best.png"
    assert artefacts.extras["figure_hf"].is_file()
    assert artefacts.extras["figure_acc"].is_file()
    assert not list((tmp_path / "viewer_out").glob("*.pdf"))
    assert not list((tmp_path / "viewer_out").glob("*.svg"))
