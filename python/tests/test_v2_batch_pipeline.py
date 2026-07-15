from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ppg_hr.v2.batch_pipeline import (
    default_v2_batch_output_dir,
    run_v2_batch_pipeline,
)
from ppg_hr.v2.optimizer import V2BayesConfig
from ppg_hr.v2.search_space import V2SearchSpace


def _write_pair(root: Path, stem: str) -> None:
    fs = 100
    n = 40 * fs
    t = np.arange(n, dtype=float) / fs
    pd.DataFrame(
        {
            "Uc1(mV)": 1.0,
            "Uc2(mV)": 1.2,
            "Ut1(mV)": 5.0,
            "Ut2(mV)": 5.5,
            "PPG_Green": 1000 + 20 * np.sin(2 * np.pi * 1.2 * t),
            "PPG_Red": 900 + 20 * np.sin(2 * np.pi * 1.2 * t),
            "PPG_IR": 800 + 20 * np.sin(2 * np.pi * 1.2 * t),
            "AccX(g)": 0.0,
            "AccY(g)": 0.0,
            "AccZ(g)": 1.0,
            "GyroX(dps)": 0.0,
            "GyroY(dps)": 0.0,
            "GyroZ(dps)": 0.0,
        }
    ).to_csv(root / f"{stem}.csv", index=False)
    (root / f"{stem}_ref.csv").write_text(
        "h1\nh2\nh3\n0,00:00:00,72\n1,00:00:01,72\n",
        encoding="utf-8",
    )


def test_run_v2_batch_pipeline_processes_bad_qc_when_ref_exists(
    tmp_path: Path,
) -> None:
    _write_pair(tmp_path, "sample")
    out = tmp_path / "out"

    payload = run_v2_batch_pipeline(
        input_dir=tmp_path,
        output_dir=out,
        ppg_modes=["green"],
        adaptive_filter="noncausal_lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        bayes_cfg=V2BayesConfig(max_iterations=1, num_seed_points=1, random_state=1),
    )

    assert payload["summary_csv"].is_file()
    assert len(payload["records"]) == 1
    assert payload["records"][0].report_path.is_file()


def test_run_v2_batch_pipeline_writes_json_png_csv_layout(tmp_path: Path) -> None:
    _write_pair(tmp_path, "sample")
    logs: list[str] = []
    progress: list[dict] = []

    payload = run_v2_batch_pipeline(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        ppg_modes=["green"],
        adaptive_filter="lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        bayes_cfg=V2BayesConfig(
            max_iterations=1,
            num_seed_points=1,
            num_repeats=2,
            random_state=1,
        ),
        on_log=logs.append,
        on_progress=progress.append,
    )

    out = tmp_path / "out"
    prefix = "sample-green-raw_bandpass-lms-full-HF"
    assert (out / "json" / f"{prefix}-v2.json").is_file()
    assert (out / "png" / f"{prefix}-v2-hr.png").is_file()
    assert (out / "csv" / f"{prefix}-v2-hr.csv").is_file()
    assert (out / "csv" / f"{prefix}-v2-error.csv").is_file()
    assert (out / "csv" / f"{prefix}-v2-window-trace.csv").is_file()
    assert (out / "csv" / f"{prefix}-v2-history.csv").is_file()
    assert payload["summary_csv"] == out / "csv" / "v2_batch_summary.csv"
    assert payload["summary_csv"].is_file()
    record = payload["records"][0]
    assert record.figure_png == out / "png" / f"{prefix}-v2-hr.png"
    assert record.hr_csv == out / "csv" / f"{prefix}-v2-hr.csv"
    assert record.ppg_input_transform == "raw_bandpass"
    assert any("repeat 1/2" in msg for msg in logs)
    assert any(item.get("stage") == "optimise" for item in progress)
    assert any(item.get("stage") == "visualise" for item in progress)


def test_run_v2_batch_pipeline_names_and_records_log_absorbance_transform(
    tmp_path: Path,
) -> None:
    _write_pair(tmp_path, "sample")
    payload = run_v2_batch_pipeline(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        ppg_modes=["green"],
        ppg_input_transform="log_absorbance",
        adaptive_filter="lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        bayes_cfg=V2BayesConfig(max_iterations=1, num_seed_points=1, random_state=1),
    )

    out = tmp_path / "out"
    prefix = "sample-green-log_absorbance-lms-full-HF"
    report = out / "json" / f"{prefix}-v2.json"
    assert report.is_file()
    assert payload["records"][0].ppg_input_transform == "log_absorbance"
    assert "log_absorbance" in payload["summary_csv"].read_text(encoding="utf-8-sig")


def test_run_v2_batch_pipeline_writes_algorithm_preset_metadata(
    tmp_path: Path,
) -> None:
    _write_pair(tmp_path, "sample")

    payload = run_v2_batch_pipeline(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        ppg_modes=["green"],
        adaptive_filter="lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        bayes_cfg=V2BayesConfig(max_iterations=1, num_seed_points=1, random_state=1),
        algorithm_preset="lite",
    )

    report = json.loads(payload["records"][0].report_path.read_text(encoding="utf-8"))
    assert report["algorithm_preset"] == "lite"


def test_run_v2_batch_pipeline_passes_comparison_groups_to_render(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ppg_hr.v2 import batch_pipeline

    _write_pair(tmp_path, "sample")
    seen: list[tuple[tuple[str, ...], ...]] = []

    def fake_render_v2_report(
        report_path,
        out_dir,
        *,
        csv_dir=None,
        output_prefix=None,
        comparison_groups=(),
        **_kwargs,
    ):
        seen.append(comparison_groups)
        png_dir = Path(out_dir)
        csv_out = Path(csv_dir)
        png_dir.mkdir(parents=True, exist_ok=True)
        csv_out.mkdir(parents=True, exist_ok=True)
        prefix = output_prefix or Path(report_path).stem
        figure = png_dir / f"{prefix}-v2-hr.png"
        err = csv_out / f"{prefix}-v2-error.csv"
        hr = csv_out / f"{prefix}-v2-hr.csv"
        figure.write_text("png", encoding="utf-8")
        err.write_text("err", encoding="utf-8")
        hr.write_text("hr", encoding="utf-8")
        return SimpleNamespace(figure_png=figure, error_csv=err, hr_csv=hr)

    monkeypatch.setattr(batch_pipeline, "render_v2_report", fake_render_v2_report)

    payload = run_v2_batch_pipeline(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        ppg_modes=["green"],
        adaptive_filter="lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        bayes_cfg=V2BayesConfig(max_iterations=1, num_seed_points=1, random_state=1),
        comparison_groups=(("ACC",),),
    )

    assert seen == [(("ACC",),)]
    assert payload["records"][0].figure_png is not None


def test_default_v2_batch_output_dir_includes_algorithm_preset(
    tmp_path: Path,
) -> None:
    lite = default_v2_batch_output_dir(
        tmp_path,
        ppg_input_transform="raw_bandpass",
        analysis_scope="full",
        adaptive_filter="lms",
        reference_groups_order=("HF",),
        algorithm_preset="Lite",
    )
    dynamic = default_v2_batch_output_dir(
        tmp_path,
        ppg_input_transform="raw_bandpass",
        analysis_scope="full",
        adaptive_filter="lms",
        reference_groups_order=("HF",),
        algorithm_preset="dynamic_rest_bo",
    )

    assert lite != dynamic
    assert "lite" in lite.name
    assert "dynamic_rest_bo" in dynamic.name


def test_run_v2_batch_pipeline_accepts_custom_search_space(tmp_path: Path) -> None:
    _write_pair(tmp_path, "sample")
    compact_space = V2SearchSpace(
        fs_target=[25],
        max_order=None,
        lms_mu_base=None,
        smooth_win_len=None,
        spec_penalty_width=None,
        hr_range_hz=None,
        slew_limit_bpm=None,
        slew_step_bpm=None,
        hr_range_rest=None,
        slew_limit_rest=None,
        slew_step_rest=None,
        time_bias=None,
    )

    payload = run_v2_batch_pipeline(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        ppg_modes=["green"],
        adaptive_filter="lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        bayes_cfg=V2BayesConfig(max_iterations=1, num_seed_points=1, random_state=1),
        algorithm_preset="lite",
        search_space=compact_space,
    )

    report_text = payload["records"][0].report_path.read_text(encoding="utf-8")
    assert '"fs_target": 25' in report_text


def test_run_v2_batch_pipeline_filters_requested_sample_stems(tmp_path: Path) -> None:
    _write_pair(tmp_path, "bobi1_HB_0711")
    _write_pair(tmp_path, "bobi2_HB_0711")

    payload = run_v2_batch_pipeline(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        ppg_modes=["green"],
        adaptive_filter="lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        algorithm_preset="lite",
        sample_stems=("bobi2",),
        bayes_cfg=V2BayesConfig(max_iterations=1, num_seed_points=1, num_repeats=1),
    )

    assert [record.sample for record in payload["records"]] == ["bobi2_HB_0711.csv"]


def test_run_v2_batch_pipeline_rejects_missing_requested_sample(tmp_path: Path) -> None:
    _write_pair(tmp_path, "bobi1_HB_0711")

    with pytest.raises(ValueError, match="requested batch samples not found"):
        run_v2_batch_pipeline(
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            ppg_modes=["green"],
            adaptive_filter="lms",
            analysis_scope="full",
            reference_groups_order=("HF",),
            algorithm_preset="lite",
            sample_stems=("bobi2",),
            bayes_cfg=V2BayesConfig(max_iterations=1),
        )


def test_run_v2_batch_pipeline_uses_default_dynamic_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample = tmp_path / "sample.csv"
    ref = tmp_path / "sample_ref.csv"
    sample.write_text("sensor", encoding="utf-8")
    ref.write_text("ref", encoding="utf-8")
    seen = []

    monkeypatch.setattr(
        "ppg_hr.v2.batch_pipeline.quality_filter_sample_v2",
        lambda *args, **kwargs: type(
            "Qc",
            (),
            {"status": "good", "to_dict": lambda self: {"status": "good"}},
        )(),
    )

    def fake_optimise_v2(cfg, *args, **kwargs):
        seen.append(cfg)
        path = kwargs["out_path"]
        path.write_text("{}", encoding="utf-8")
        return type("Result", (), {"best_error": 0.0, "report_path": path})()

    monkeypatch.setattr("ppg_hr.v2.batch_pipeline.optimise_v2", fake_optimise_v2)
    monkeypatch.setattr(
        "ppg_hr.v2.batch_pipeline.render_v2_report",
        lambda *args, **kwargs: type(
            "Arte",
            (),
            {
                "figure_png": tmp_path / "x.png",
                "error_csv": tmp_path / "e.csv",
                "hr_csv": tmp_path / "h.csv",
            },
        )(),
    )

    run_v2_batch_pipeline(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        ppg_modes=["green"],
        adaptive_filter="lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        algorithm_preset="lite",
        bayes_cfg=V2BayesConfig(max_iterations=1, num_repeats=1),
    )

    assert seen[0].post_motion_dynamic_guard_enable is True
    assert seen[0].post_motion_dual_reset_enable is False
    assert seen[0].post_motion_dynamic_guard_crossover_gap_bpm == 2.0
    assert seen[0].post_motion_dynamic_guard_rescue_gap_bpm == 20.0


def test_run_v2_batch_pipeline_applies_run_config_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample = tmp_path / "multi_fuwo1_0613.csv"
    ref = tmp_path / "multi_fuwo1_0613_HR_ref.csv"
    sample.write_text("sensor", encoding="utf-8")
    ref.write_text("ref", encoding="utf-8")
    seen = []

    monkeypatch.setattr(
        "ppg_hr.v2.batch_pipeline.quality_filter_sample_v2",
        lambda *args, **kwargs: type(
            "Qc",
            (),
            {"status": "good", "to_dict": lambda self: {"status": "good"}},
        )(),
    )

    def fake_optimise_v2(cfg, *args, **kwargs):
        seen.append(cfg)
        path = kwargs["out_path"]
        path.write_text("{}", encoding="utf-8")
        return type("Result", (), {"best_error": 0.0, "report_path": path})()

    monkeypatch.setattr("ppg_hr.v2.batch_pipeline.optimise_v2", fake_optimise_v2)
    monkeypatch.setattr(
        "ppg_hr.v2.batch_pipeline.render_v2_report",
        lambda *args, **kwargs: type(
            "Arte",
            (),
            {
                "figure_png": tmp_path / "x.png",
                "error_csv": tmp_path / "e.csv",
                "hr_csv": tmp_path / "h.csv",
            },
        )(),
    )

    run_v2_batch_pipeline(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        ppg_modes=["green"],
        adaptive_filter="lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        algorithm_preset="lite",
        bayes_cfg=V2BayesConfig(max_iterations=1, num_repeats=1),
        run_config_overrides={
            "post_motion_dynamic_guard_enable": True,
            "post_motion_dynamic_guard_stable_windows": 2,
        },
    )

    assert seen[0].post_motion_dynamic_guard_enable is True
    assert seen[0].post_motion_dynamic_guard_stable_windows == 2
