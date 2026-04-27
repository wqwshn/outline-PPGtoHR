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
    legend_locs: list[str] = []

    def _fake_solve(params):
        return results.pop(0)

    def _fake_plot_panel(ax, res, **kwargs):
        plotted_masks.append(res.HR[:, 7].tolist())
        legend_locs.append(kwargs["legend_loc"])

    monkeypatch.setattr("ppg_hr.visualization.result_viewer.solve", _fake_solve)
    monkeypatch.setattr("ppg_hr.visualization.result_viewer._plot_panel", _fake_plot_panel)

    render(
        report_path,
        SolverParams(file_name=tmp_path / "multi.csv"),
        out_dir=tmp_path / "out",
        show=False,
    )

    assert plotted_masks == [[0, 1, 1, 0, 0], [0, 1, 1, 0, 0]]
    assert legend_locs == ["upper right", "upper right"]

    plotted_masks.clear()
    legend_locs.clear()
    results = [hf, acc]
    render(
        report_path,
        SolverParams(file_name=tmp_path / "multi.csv", analysis_scope="motion"),
        out_dir=tmp_path / "out-motion",
        show=False,
    )
    assert plotted_masks == [[0, 1, 1, 0, 0], [0, 1, 1, 0, 0]]
    assert legend_locs == ["lower right", "lower right"]


def test_plot_panel_uses_nature_single_column_style() -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    res = type("R", (), {})()
    res.err_stats = np.array([
        [2.345, 0.0, 0.0],
        [4.567, 0.0, 0.0],
        [6.789, 5.432, 7.654],
        [1.111, 0.987, 1.234],
        [1.222, 1.111, 1.333],
    ])
    t = np.arange(30, dtype=float)
    res.HR = np.zeros((30, 9), dtype=float)
    res.HR[:, 0] = t
    res.HR[:, 1] = np.linspace(56.0, 148.0, 30) / 60.0
    res.HR[:, 4] = np.linspace(57.0, 149.0, 30) / 60.0
    res.HR[:, 5] = np.linspace(56.5, 149.0, 30) / 60.0
    res.HR[:, 6] = np.linspace(56.2, 148.5, 30) / 60.0
    res.HR[10:15, 7] = 1.0
    res.HR[:, 8] = res.HR[:, 7]
    res.T_Pred = res.HR[:, 0]

    fig, ax = plt.subplots()
    result_viewer._plot_panel(
        ax,
        res,
        fill_reference_to_t_pred_end=False,
        legend_loc="lower right",
    )

    labels = ax.get_legend_handles_labels()[1]
    assert labels == ["Reference", "FFT", "HF-LMS", "ACC-LMS"]
    assert ax.get_title() == ""
    assert ax.get_ylabel() == "Heart rate (BPM)"
    # MAE table is drawn as individual text cells (not a single block).
    cell_texts = [t.get_text() for t in ax.texts]
    assert "MAE (BPM)" in cell_texts
    assert "all" in cell_texts
    assert "motion" in cell_texts
    assert "rest" not in cell_texts
    # Data cells exist and are correct
    assert "HF-LMS" in cell_texts
    assert "ACC-LMS" in cell_texts
    assert "FFT" in cell_texts
    assert "6.8" in cell_texts and "7.7" in cell_texts
    assert "1.1" in cell_texts and "1.2" in cell_texts
    # All data cells use ha='center'
    data_cells = [t for t in ax.texts if t.get_text() not in ("", )]
    assert all(t.get_ha() == "center" for t in data_cells)
    # Row order check: cells are positioned by y-coordinate, verify order
    non_empty = [t for t in ax.texts if t.get_text()]
    non_empty_sorted = sorted(non_empty, key=lambda t: -t.get_position()[1])
    row_labels = [t.get_text() for t in non_empty_sorted]
    assert row_labels.index("HF-LMS") < row_labels.index("ACC-LMS") < row_labels.index("FFT")

    lines = {line.get_label(): line for line in ax.lines}
    ref_line = lines["Reference"]
    hf_line = lines["HF-LMS"]
    acc_line = lines["ACC-LMS"]
    fft_line = lines["FFT"]
    assert ref_line.get_color().lower() == "#2b2b2b"
    assert ref_line.get_linestyle() == "-"
    assert hf_line.get_color().lower() == "#e68653"
    assert hf_line.get_linewidth() > acc_line.get_linewidth()
    assert hf_line.get_marker() == "o"
    assert acc_line.get_color().lower() == "#5da9c9"
    assert fft_line.get_color().lower() == "#a8adb3"
    assert fft_line.get_linestyle() != "-"
    legend = ax.get_legend()
    assert legend is not None
    assert legend._loc == 2  # upper left
    assert legend._ncols == 1
    assert ax.collections
    assert ax.collections[0].get_alpha() == pytest.approx(0.24)
    y_gridlines = ax.get_ygridlines()
    x_gridlines = ax.get_xgridlines()
    assert y_gridlines
    assert any(line.get_visible() for line in y_gridlines)
    assert all(line.get_alpha() == pytest.approx(0.12) for line in y_gridlines)
    assert all(line.get_linewidth() == pytest.approx(0.45) for line in y_gridlines)
    assert not any(line.get_visible() for line in x_gridlines)
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    ymin, ymax = ax.get_ylim()
    assert ymin == 55
    assert ymax == 150
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


def test_render_exports_unique_nature_single_column_pngs_only(
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
    seen_figsizes: list[tuple[float, float]] = []
    savefig_kwargs: list[dict[str, object]] = []
    real_subplots = result_viewer.plt_subplots_for_test()

    def _spy_subplots(*args, **kwargs):
        seen_figsizes.append(tuple(kwargs.get("figsize")))
        fig, ax = real_subplots(*args, **kwargs)
        real_savefig = fig.savefig

        def _spy_savefig(*save_args, **save_kwargs):
            savefig_kwargs.append(dict(save_kwargs))
            return real_savefig(*save_args, **save_kwargs)

        fig.savefig = _spy_savefig
        import matplotlib as mpl
        mpl.rcParams["savefig.bbox"] = "tight"
        return fig, ax

    monkeypatch.setattr("ppg_hr.visualization.result_viewer._plt_subplots", _spy_subplots)
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
    out_dir = tmp_path / "viewer_out"
    out_dir.mkdir()
    (out_dir / "multi_bobi1-full-hf-best.png").write_bytes(b"existing")
    (out_dir / "multi_bobi1-full-acc-best.png").write_bytes(b"existing")
    (out_dir / "multi_bobi1-full-error_table.csv").write_text("existing", encoding="utf-8")
    (out_dir / "multi_bobi1-full-param_table.csv").write_text("existing", encoding="utf-8")

    artefacts = render(
        report,
        SolverParams(file_name=tmp_path / "multi_bobi1.csv"),
        out_dir=out_dir,
        output_prefix="multi_bobi1",
        show=False,
    )

    assert seen_figsizes == [(3.54, 2.6), (3.54, 2.6)]
    assert [kwargs.get("dpi") for kwargs in savefig_kwargs] == [600, 600]
    assert all("bbox_inches" in kwargs for kwargs in savefig_kwargs)
    assert [kwargs["bbox_inches"] for kwargs in savefig_kwargs] == [None, None]
    assert artefacts.figure == out_dir / "multi_bobi1-full-hf-best-2.png"
    assert artefacts.extras["figure_hf"] == artefacts.figure
    assert artefacts.extras["figure_acc"] == out_dir / "multi_bobi1-full-acc-best-2.png"
    assert artefacts.error_csv == out_dir / "multi_bobi1-full-error_table-2.csv"
    assert artefacts.param_csv == out_dir / "multi_bobi1-full-param_table-2.csv"
    assert artefacts.extras["figure_hf"].is_file()
    assert artefacts.extras["figure_acc"].is_file()
    import matplotlib.image as mpimg
    assert mpimg.imread(artefacts.extras["figure_hf"]).shape[:2] == (1560, 2124)
    assert mpimg.imread(artefacts.extras["figure_acc"]).shape[:2] == (1560, 2124)
    assert not list(out_dir.glob("*.pdf"))
    assert not list(out_dir.glob("*.svg"))


def test_viewer_accepts_legacy_report_with_fs_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Legacy reports with fs_target should replay correctly."""
    seen_targets: list[int] = []

    def _fake_solve(params):
        seen_targets.append(params.fs_target)
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
    legacy_report = {
        "min_err_hf": 1.0,
        "min_err_acc": 1.0,
        "best_para_hf": {"fs_target": 100, "max_order": 16},
        "best_para_acc": {"fs_target": 100, "max_order": 16},
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy_report), encoding="utf-8")
    render(path, SolverParams(file_name=tmp_path / "x.csv"), out_dir=tmp_path / "out", show=False)
    assert seen_targets == [100, 100]


def test_viewer_accepts_new_report_without_fs_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """New reports without fs_target should use default 25."""
    seen_targets: list[int] = []

    def _fake_solve(params):
        seen_targets.append(params.fs_target)
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
    new_report = {
        "min_err_hf": 1.0,
        "min_err_acc": 1.0,
        "best_para_hf": {"max_order": 16},
        "best_para_acc": {"max_order": 16},
    }
    path = tmp_path / "new.json"
    path.write_text(json.dumps(new_report), encoding="utf-8")
    render(path, SolverParams(file_name=tmp_path / "x.csv"), out_dir=tmp_path / "out", show=False)
    assert seen_targets == [25, 25]
