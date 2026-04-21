from __future__ import annotations

from pathlib import Path

from ppg_hr.batch_pipeline import BatchRunRecord, QcRow, QcThresholds, run_batch_pipeline
from ppg_hr.optimization import BayesConfig, BayesResult


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
        )

    class FakeArtefact:
        def __init__(self, out_dir: Path):
            self.figure = out_dir / "figure.png"
            self.error_csv = out_dir / "error_table.csv"
            self.param_csv = out_dir / "param_table.csv"
            out_dir.mkdir(parents=True, exist_ok=True)
            self.figure.write_text("fig", encoding="utf-8")
            self.error_csv.write_text("err", encoding="utf-8")
            self.param_csv.write_text("par", encoding="utf-8")

    def fake_render(report_path, base_params, *, out_dir, show):
        call_order.append(f"render:{base_params.ppg_mode}")
        return FakeArtefact(Path(out_dir))

    monkeypatch.setattr("ppg_hr.batch_pipeline.quality_scan", fake_quality_scan)
    monkeypatch.setattr("ppg_hr.batch_pipeline.save_motion_segment_plot", fake_plot)
    monkeypatch.setattr("ppg_hr.batch_pipeline.optimise", fake_optimise)
    monkeypatch.setattr("ppg_hr.batch_pipeline.render", fake_render)

    payload = run_batch_pipeline(
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        modes=["green", "tri"],
        adaptive_filter="lms",
        bayes_cfg=BayesConfig(max_iterations=2, num_seed_points=1, num_repeats=1),
        thresholds=QcThresholds(),
        on_progress=lambda info: progress_stages.append(str(info["stage"])),
    )

    assert call_order == [
        "plot:sample01.csv",
        "optimise:green",
        "render:green",
        "optimise:tri",
        "render:tri",
    ]
    assert progress_stages[:2] == ["qc", "segment_plot"]
    assert progress_stages.count("optimise") >= 2
    assert progress_stages.count("visualise") >= 2
    records = payload["records"]
    assert len(records) == 2
    assert all(isinstance(r, BatchRunRecord) for r in records)
