from __future__ import annotations

import json
from pathlib import Path

from ppg_hr.visualization.batch_viewer import discover_report_jobs, render_report_batch


def _write_report(path: Path, **extra) -> Path:
    payload = {
        "min_err_hf": 1.0,
        "min_err_acc": 1.0,
        "best_para_hf": {},
        "best_para_acc": {},
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_discover_report_jobs_matches_batch_best_params_name(tmp_path: Path) -> None:
    data = tmp_path / "sample1.csv"
    ref = tmp_path / "sample1_ref.csv"
    data.write_text("data\n", encoding="utf-8")
    ref.write_text("ref\n", encoding="utf-8")
    report = _write_report(
        tmp_path
        / "batch_outputs"
        / "batch_runs"
        / "sample1-green-lms-full"
        / "sample1-green-lms-full-best_params.json"
    )

    jobs = discover_report_jobs(tmp_path, analysis_scope="full")

    assert len(jobs) == 1
    assert jobs[0].report_path == report
    assert jobs[0].data_path == data
    assert jobs[0].ref_path == ref
    assert jobs[0].error is None


def test_discover_report_jobs_reports_missing_data(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "Best_Params_Result_missing-full.json")

    jobs = discover_report_jobs(tmp_path, analysis_scope="full")

    assert len(jobs) == 1
    assert jobs[0].report_path == report
    assert jobs[0].data_path is None
    assert jobs[0].status == "missing"
    assert jobs[0].error is not None
    assert "data" in jobs[0].error.lower()


def test_render_report_batch_defaults_output_next_to_each_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data = tmp_path / "sample1.csv"
    ref = tmp_path / "sample1_ref.csv"
    data.write_text("data\n", encoding="utf-8")
    ref.write_text("ref\n", encoding="utf-8")
    good = _write_report(tmp_path / "reports" / "Best_Params_Result_sample1-full.json")
    bad = _write_report(tmp_path / "reports" / "Best_Params_Result_missing-full.json")
    calls: list[tuple[Path, Path, Path, Path, str]] = []

    def _fake_render(report_path, params, *, out_dir, output_prefix, show):
        from ppg_hr.visualization.result_viewer import ViewerArtefacts

        out_dir = Path(out_dir)
        calls.append(
            (
                Path(report_path),
                Path(params.file_name),
                Path(params.ref_file),
                out_dir,
                output_prefix,
            )
        )
        figure = out_dir / f"{output_prefix}-hf-best.png"
        figure_acc = out_dir / f"{output_prefix}-acc-best.png"
        error_csv = out_dir / f"{output_prefix}-error_table.csv"
        param_csv = out_dir / f"{output_prefix}-param_table.csv"
        return ViewerArtefacts(
            figure=figure,
            error_csv=error_csv,
            param_csv=param_csv,
            extras={"figure_hf": figure, "figure_acc": figure_acc},
        )

    monkeypatch.setattr("ppg_hr.visualization.batch_viewer.render", _fake_render)

    result = render_report_batch(tmp_path, out_dir=None, analysis_scope="full")

    by_report = {item.report_path: item for item in result.items}
    assert set(by_report) == {good, bad}
    assert by_report[good].status == "ok"
    assert by_report[good].figure_hf == good.parent / "sample1-full-hf-best.png"
    assert by_report[bad].status == "missing"
    assert by_report[bad].error is not None
    assert calls == [(good, data, ref, good.parent, "sample1-full")]


def test_render_report_batch_uses_explicit_output_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data = tmp_path / "sample1.csv"
    ref = tmp_path / "sample1_ref.csv"
    data.write_text("data\n", encoding="utf-8")
    ref.write_text("ref\n", encoding="utf-8")
    report = _write_report(tmp_path / "Best_Params_Result_sample1-full.json")
    out_dir = tmp_path / "custom_out"
    seen_out_dirs: list[Path] = []

    def _fake_render(report_path, params, *, out_dir, output_prefix, show):
        from ppg_hr.visualization.result_viewer import ViewerArtefacts

        seen_out_dirs.append(Path(out_dir))
        figure = Path(out_dir) / f"{output_prefix}-hf-best.png"
        return ViewerArtefacts(figure=figure, extras={"figure_hf": figure})

    monkeypatch.setattr("ppg_hr.visualization.batch_viewer.render", _fake_render)

    result = render_report_batch(tmp_path, out_dir=out_dir, analysis_scope="full")

    assert result.items[0].report_path == report
    assert result.items[0].figure_hf == out_dir / "sample1-full-hf-best.png"
    assert seen_out_dirs == [out_dir]


def test_render_report_batch_passes_num_cascade_hf_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    data = tmp_path / "sample.csv"
    ref = tmp_path / "sample_ref.csv"
    data.write_text("dummy", encoding="utf-8")
    ref.write_text("dummy", encoding="utf-8")
    report = _write_report(
        tmp_path / "sample-best_params.json",
        file_name=str(data),
        ref_file=str(ref),
    )

    seen: list[int] = []

    def fake_render(report_path, params, *, out_dir, output_prefix, show):
        from ppg_hr.visualization.result_viewer import ViewerArtefacts

        seen.append(int(params.num_cascade_hf))
        return ViewerArtefacts()

    monkeypatch.setattr("ppg_hr.visualization.batch_viewer.render", fake_render)
    render_report_batch(
        tmp_path,
        out_dir=None,
        analysis_scope="full",
        num_cascade_hf=4,
    )

    assert report.is_file()
    assert seen == [4]
