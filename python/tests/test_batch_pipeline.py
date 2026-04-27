from __future__ import annotations

import csv
from pathlib import Path

from ppg_hr.batch_pipeline import BatchRunRecord, QcRow, QcThresholds, run_batch_pipeline
from ppg_hr.optimization import BayesConfig, BayesResult
from ppg_hr.visualization.result_viewer import ViewerArtefacts


def test_run_batch_pipeline_reports_fine_grained_stages_and_interleaves_render(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    sample = input_dir / "sample01.csv"
    sample.write_text("dummy\n", encoding="utf-8")
    ref = input_dir / "sample01_ref.csv"
    ref.write_text("dummy\n", encoding="utf-8")

    call_order: list[str] = []
    seen_hf_counts: list[int] = []
    progress_stages: list[str] = []

    def fake_quality_scan(input_dir, thresholds, *, on_file_scanned=None):
        assert input_dir == sample.parent
        assert isinstance(thresholds, QcThresholds)
        if on_file_scanned is not None:
            on_file_scanned(1, 1, sample.name)
        return [QcRow(sample.name, "好采样", "无", sample)], []

    def fake_plot(file_path, out_path, *, fs=100.0):
        call_order.append(f"plot:{file_path.name}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("png", encoding="utf-8")

    def fake_optimise(base, *, config, out_path, verbose, on_trial_step=None):
        call_order.append(f"optimise:{base.ppg_mode}")
        seen_hf_counts.append(int(base.num_cascade_hf))
        if on_trial_step is not None:
            on_trial_step(
                {
                    "mode": "HF",
                    "repeat_idx": 1,
                    "repeat_total": 1,
                    "trial_idx": 1,
                    "trial_total": 2,
                    "global_trial": 1,
                    "global_total": 2,
                    "value": 1.5,
                    "best_in_repeat": 1.5,
                    "best_overall": 1.5,
                }
            )
        Path(out_path).write_text("{}", encoding="utf-8")
        return BayesResult(
            min_err_hf=1.0,
            best_para_hf={"fs_target": 100},
            min_err_acc=2.0,
            best_para_acc={"fs_target": 100},
            importance_hf=None,
            ppg_mode=base.ppg_mode,
            num_cascade_hf=int(base.num_cascade_hf),
        )

    def fake_render(report_path, base_params, *, out_dir, output_prefix, show):
        call_order.append(f"render:{base_params.ppg_mode}")
        seen_hf_counts.append(int(base_params.num_cascade_hf))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        figure = out_dir / f"{output_prefix}-hf-best.png"
        figure_acc = out_dir / f"{output_prefix}-acc-best.png"
        error_csv = out_dir / f"{output_prefix}-error_table.csv"
        param_csv = out_dir / f"{output_prefix}-param_table.csv"
        hr_csv = out_dir / f"{output_prefix}-hr_results.csv"
        figure.write_text("fig", encoding="utf-8")
        figure_acc.write_text("fig", encoding="utf-8")
        error_csv.write_text("err", encoding="utf-8")
        param_csv.write_text("par", encoding="utf-8")
        hr_csv.write_text("hr", encoding="utf-8")
        return ViewerArtefacts(
            figure=figure,
            error_csv=error_csv,
            param_csv=param_csv,
            hr_csv=hr_csv,
            extras={
                "figure_hf": figure,
                "figure_acc": figure_acc,
            },
        )

    monkeypatch.setattr("ppg_hr.batch_pipeline.quality_scan", fake_quality_scan)
    monkeypatch.setattr("ppg_hr.batch_pipeline.save_motion_segment_plot", fake_plot)
    monkeypatch.setattr("ppg_hr.batch_pipeline.optimise", fake_optimise)
    monkeypatch.setattr("ppg_hr.batch_pipeline.render", fake_render)

    payload = run_batch_pipeline(
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        modes=["green", "red", "ir"],
        adaptive_filter="lms",
        bayes_cfg=BayesConfig(max_iterations=2, num_seed_points=1, num_repeats=1),
        thresholds=QcThresholds(),
        num_cascade_hf=4,
        on_progress=lambda info: progress_stages.append(str(info["stage"])),
    )

    assert call_order == [
        "plot:sample01.csv",
        "optimise:green",
        "render:green",
        "optimise:red",
        "render:red",
        "optimise:ir",
        "render:ir",
    ]
    assert progress_stages[:2] == ["qc", "segment_plot"]
    assert progress_stages.count("optimise") >= 3
    assert progress_stages.count("visualise") >= 3
    records = payload["records"]
    assert len(records) == 3
    assert all(isinstance(r, BatchRunRecord) for r in records)

    # Output names must follow the short dash-separated convention so each
    # artefact stays unambiguous when users drag it outside the folder.
    run_root = payload["output_dir"] / "batch_runs"
    for mode in ("green", "red", "ir"):
        prefix = f"sample01-{mode}-lms-full-hf4"
        run_dir = run_root / prefix
        assert (run_dir / f"{prefix}-best_params.json").is_file()
        assert (run_dir / f"{prefix}-hf-best.png").is_file()
        assert (run_dir / f"{prefix}-acc-best.png").is_file()
        assert (run_dir / f"{prefix}-error_table.csv").is_file()
        assert (run_dir / f"{prefix}-param_table.csv").is_file()
        assert (run_dir / f"{prefix}-hr_results.csv").is_file()
    rec_by_mode = {r.mode: r for r in records}
    for mode in ("green", "red", "ir"):
        rec = rec_by_mode[mode]
        assert rec.figure_path is not None and rec.figure_path.name == f"sample01-{mode}-lms-full-hf4-hf-best.png"
        assert rec.error_csv is not None and rec.error_csv.name == f"sample01-{mode}-lms-full-hf4-error_table.csv"
        assert rec.param_csv is not None and rec.param_csv.name == f"sample01-{mode}-lms-full-hf4-param_table.csv"
        assert rec.hr_csv is not None and rec.hr_csv.name == f"sample01-{mode}-lms-full-hf4-hr_results.csv"
        assert rec.report_path.name == f"sample01-{mode}-lms-full-hf4-best_params.json"
    assert seen_hf_counts == [4, 4, 4, 4, 4, 4]
    with payload["summary_csv"].open("r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    assert "hr_csv" in header


def test_run_batch_pipeline_runs_bad_quality_rows_with_reference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    good_sample = input_dir / "good01.csv"
    bad_sample = input_dir / "bad01.csv"
    good_sample.write_text("dummy\n", encoding="utf-8")
    bad_sample.write_text("dummy\n", encoding="utf-8")
    good_sample.with_name("good01_ref.csv").write_text("dummy\n", encoding="utf-8")
    bad_sample.with_name("bad01_ref.csv").write_text("dummy\n", encoding="utf-8")

    optimised: list[str] = []

    def fake_quality_scan(input_dir, thresholds, *, on_file_scanned=None):
        return (
            [QcRow(good_sample.name, "好采样", "无", good_sample)],
            [QcRow(bad_sample.name, "坏采样", "STD过大", bad_sample)],
        )

    def fake_plot(file_path, out_path, *, fs=100.0):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("png", encoding="utf-8")

    def fake_optimise(base, *, config, out_path, verbose, on_trial_step=None):
        optimised.append(Path(base.file_name).name)
        Path(out_path).write_text("{}", encoding="utf-8")
        return BayesResult(
            min_err_hf=1.0,
            best_para_hf={"fs_target": 100},
            min_err_acc=2.0,
            best_para_acc={"fs_target": 100},
            importance_hf=None,
            ppg_mode=base.ppg_mode,
        )

    def fake_render(report_path, base_params, *, out_dir, output_prefix, show):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        figure = out_dir / f"{output_prefix}-hf-best.png"
        error_csv = out_dir / f"{output_prefix}-error_table.csv"
        param_csv = out_dir / f"{output_prefix}-param_table.csv"
        figure.write_text("fig", encoding="utf-8")
        error_csv.write_text("err", encoding="utf-8")
        param_csv.write_text("par", encoding="utf-8")
        return ViewerArtefacts(figure=figure, error_csv=error_csv, param_csv=param_csv)

    monkeypatch.setattr("ppg_hr.batch_pipeline.quality_scan", fake_quality_scan)
    monkeypatch.setattr("ppg_hr.batch_pipeline.save_motion_segment_plot", fake_plot)
    monkeypatch.setattr("ppg_hr.batch_pipeline.optimise", fake_optimise)
    monkeypatch.setattr("ppg_hr.batch_pipeline.render", fake_render)

    payload = run_batch_pipeline(
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        modes=["green"],
        adaptive_filter="lms",
        bayes_cfg=BayesConfig(max_iterations=2, num_seed_points=1, num_repeats=1),
        thresholds=QcThresholds(),
    )

    assert optimised == ["good01.csv", "bad01.csv"]
    assert len(payload["good_rows"]) == 1
    assert len(payload["bad_rows"]) == 1
    assert len(payload["records"]) == 2


def test_run_batch_pipeline_writes_one_bom_encoded_qc_table(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    good_sample = input_dir / "good01.csv"
    bad_sample = input_dir / "bad01.csv"
    good_sample.write_text("dummy\n", encoding="utf-8")
    bad_sample.write_text("dummy\n", encoding="utf-8")
    good_sample.with_name("good01_ref.csv").write_text("dummy\n", encoding="utf-8")
    bad_sample.with_name("bad01_ref.csv").write_text("dummy\n", encoding="utf-8")

    def fake_quality_scan(input_dir, thresholds, *, on_file_scanned=None):
        return (
            [QcRow(good_sample.name, "好采样", "无", good_sample)],
            [QcRow(bad_sample.name, "坏采样", "STD过大", bad_sample)],
        )

    def fake_plot(file_path, out_path, *, fs=100.0):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("png", encoding="utf-8")

    def fake_optimise(base, *, config, out_path, verbose, on_trial_step=None):
        Path(out_path).write_text("{}", encoding="utf-8")
        return BayesResult(
            min_err_hf=1.0,
            best_para_hf={"fs_target": 100},
            min_err_acc=2.0,
            best_para_acc={"fs_target": 100},
            importance_hf=None,
            ppg_mode=base.ppg_mode,
        )

    def fake_render(report_path, base_params, *, out_dir, output_prefix, show):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        figure = out_dir / f"{output_prefix}-hf-best.png"
        error_csv = out_dir / f"{output_prefix}-error_table.csv"
        param_csv = out_dir / f"{output_prefix}-param_table.csv"
        figure.write_text("fig", encoding="utf-8")
        error_csv.write_text("err", encoding="utf-8")
        param_csv.write_text("par", encoding="utf-8")
        return ViewerArtefacts(figure=figure, error_csv=error_csv, param_csv=param_csv)

    monkeypatch.setattr("ppg_hr.batch_pipeline.quality_scan", fake_quality_scan)
    monkeypatch.setattr("ppg_hr.batch_pipeline.save_motion_segment_plot", fake_plot)
    monkeypatch.setattr("ppg_hr.batch_pipeline.optimise", fake_optimise)
    monkeypatch.setattr("ppg_hr.batch_pipeline.render", fake_render)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for legacy_name in ("good_samples.csv", "bad_samples.csv", "qc_summary.csv"):
        (out_dir / legacy_name).write_text("stale\n", encoding="utf-8")

    run_batch_pipeline(
        input_dir=input_dir,
        output_dir=out_dir,
        modes=["green"],
        adaptive_filter="lms",
        bayes_cfg=BayesConfig(max_iterations=2, num_seed_points=1, num_repeats=1),
        thresholds=QcThresholds(),
    )

    qc_path = out_dir / "qc_samples.csv"
    assert qc_path.is_file()
    assert not (out_dir / "good_samples.csv").exists()
    assert not (out_dir / "bad_samples.csv").exists()
    assert not (out_dir / "qc_summary.csv").exists()
    assert qc_path.read_bytes().startswith(b"\xef\xbb\xbf")
    text = qc_path.read_text(encoding="utf-8-sig")
    assert "文件名,状态,原因,文件路径" in text
    assert "good01.csv,好采样,无" in text
    assert "bad01.csv,坏采样,STD过大" in text


def test_qc_threshold_defaults_are_relaxed() -> None:
    thresholds = QcThresholds()
    assert thresholds.std_max_threshold == 5.0
    assert thresholds.std_ratio_threshold == 3.0
    assert thresholds.outlier_std_multiplier == 3.0
    assert thresholds.outlier_ratio_threshold == 4.0
