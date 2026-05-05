from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ppg_hr.v2.plotting import (
    discover_v2_plot_jobs,
    render_v2_report,
    render_v2_report_batch,
)


def _write_report(
    path: Path, order: list[str], *, time_bias: float = 5.0
) -> None:
    payload = {
        "schema_version": "v2",
        "data_path": "sample.csv",
        "ref_path": str(path.parent / "sample_ref.csv"),
        "ppg_mode": "green",
        "analysis_scope": "full",
        "adaptive_filter": "noncausal_lms",
        "reference_groups_order": order,
        "err_stats": {"fft_aae_bpm": 2.0, "final_aae_bpm": 1.0},
        "hr": [
            [4.0, 75.0, 74.0, 75.5, 0.0, 0.0],
            [5.0, 76.0, 75.0, 76.2, 0.0, 0.0],
        ],
        "metadata": {
            "time_bias": time_bias,
            "ref_path": str(path.parent / "sample_ref.csv"),
            "adaptive_filter": "noncausal_lms",
        },
        "best_params": {"max_order": 16},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    ref_path = path.parent / "sample_ref.csv"
    ref_path.write_text(
        "h1,h2,h3\n"
        + "".join(
            f"{i},00:00:{i:02d},{75.0 + 0.1 * i:.1f}\n" for i in range(30)
        ),
        encoding="utf-8",
    )


def test_discover_v2_plot_jobs_skips_old_json(tmp_path: Path) -> None:
    _write_report(tmp_path / "new.json", ["HF", "ACC"])
    (tmp_path / "old.json").write_text(
        json.dumps({"adaptive_filter": "lms"}),
        encoding="utf-8",
    )

    jobs = discover_v2_plot_jobs(tmp_path)

    assert [j.report_path.name for j in jobs] == ["new.json"]


def test_render_v2_report_outputs_png_and_csv_with_reference_key(
    tmp_path: Path,
) -> None:
    report = tmp_path / "new.json"
    _write_report(report, ["HF", "ACC"])

    arte = render_v2_report(report, out_dir=tmp_path / "figures")

    assert arte.figure_png.is_file()
    assert arte.error_csv.is_file()
    assert arte.reference_order_key == "HF+ACC"


def test_render_batch_records_reference_order(tmp_path: Path) -> None:
    _write_report(tmp_path / "a.json", ["HF", "ACC"])
    _write_report(tmp_path / "b.json", ["ACC", "HF"])

    result = render_v2_report_batch(tmp_path, tmp_path / "out")

    keys = {item.reference_order_key for item in result.items}
    assert keys == {"HF+ACC", "ACC+HF"}


def test_render_v2_report_can_split_png_and_csv_outputs(tmp_path: Path) -> None:
    report = tmp_path / "new.json"
    _write_report(report, ["HF"])

    arte = render_v2_report(
        report,
        out_dir=tmp_path / "png",
        csv_dir=tmp_path / "csv",
        output_prefix="sample-green-lms-full-HF",
    )

    assert arte.figure_png == tmp_path / "png" / "sample-green-lms-full-HF-v2-hr.png"
    assert arte.hr_csv == tmp_path / "csv" / "sample-green-lms-full-HF-v2-hr.csv"
    assert arte.error_csv == tmp_path / "csv" / "sample-green-lms-full-HF-v2-error.csv"
    assert arte.figure_png.is_file()
    assert arte.hr_csv.is_file()
    assert arte.error_csv.is_file()
    assert not arte.figure_png.with_suffix(".pdf").exists()
    assert not arte.figure_png.with_suffix(".svg").exists()


def test_render_v2_report_hr_csv_has_aligned_times(tmp_path: Path) -> None:
    """验证 HR CSV 中的时间列包含 time_bias 对齐后的值."""
    import csv

    report = tmp_path / "new.json"
    _write_report(report, ["HF"], time_bias=3.0)
    arte = render_v2_report(report, out_dir=tmp_path / "out")
    with arte.hr_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert header[0] == "time_s"
    times = [float(r[0]) for r in rows]
    assert times[0] == 4.0 + 3.0
    assert times[1] == 5.0 + 3.0


def test_render_v2_report_error_csv_has_v1_style_format(tmp_path: Path) -> None:
    """验证错误 CSV 采用 v1 格式：含 total/rest/motion AAE 和 hit rate."""
    import csv

    report = tmp_path / "new.json"
    _write_report(report, ["HF"])
    arte = render_v2_report(report, out_dir=tmp_path / "out")
    with arte.error_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert header == [
        "method", "total_aae", "rest_aae", "motion_aae",
        "total_hit_rate_5bpm", "rest_hit_rate_5bpm", "motion_hit_rate_5bpm",
    ]
    methods = [r[0] for r in rows]
    assert "FFT" in methods
    assert any("NC-LMS+H" in m for m in methods)


def test_render_v2_report_motion_scope_crops_to_analysis_window(
    tmp_path: Path,
) -> None:
    times = np.arange(0, 100.0, 1.0, dtype=float)
    hr_rows = []
    for t in times:
        in_motion = 1.0 if 30.0 <= t <= 60.0 else 0.0
        hr_rows.append([t - 5.0, 75.0, 73.0, 74.0, in_motion, in_motion])
    payload = {
        "schema_version": "v2",
        "data_path": "sample.csv",
        "ref_path": str(tmp_path / "sample_ref.csv"),
        "ppg_mode": "green",
        "analysis_scope": "motion",
        "adaptive_filter": "lms",
        "reference_groups_order": ["HF"],
        "err_stats": {"fft_aae_bpm": 2.0, "final_aae_bpm": 1.0},
        "hr": hr_rows,
        "metadata": {
            "time_bias": 5.0,
            "analysis_scope": "motion",
            "adaptive_filter": "lms",
            "pre_motion_context_seconds": 30.0,
            "motion_segment": {"start_s": 30.0, "end_s": 60.0},
            "ref_path": str(tmp_path / "sample_ref.csv"),
        },
        "best_params": {"max_order": 16},
    }
    report = tmp_path / "m.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    ref_path = tmp_path / "sample_ref.csv"
    ref_path.write_text(
        "h1,h2,h3\n"
        + "".join(f"{i},00:00:{i:02d},{75.0:.1f}\n" for i in range(100)),
        encoding="utf-8",
    )

    arte = render_v2_report(report, out_dir=tmp_path / "figures")
    assert arte.figure_png.is_file()


def test_render_v2_report_full_scope_uses_all_data(tmp_path: Path) -> None:
    times = np.arange(0, 50.0, 1.0, dtype=float)
    hr_rows = [[t - 5.0, 75.0, 73.0, 74.0, 0.0, 0.0] for t in times]
    payload = {
        "schema_version": "v2",
        "data_path": "sample.csv",
        "ref_path": str(tmp_path / "sample_ref.csv"),
        "ppg_mode": "green",
        "analysis_scope": "full",
        "adaptive_filter": "lms",
        "reference_groups_order": ["HF"],
        "err_stats": {"fft_aae_bpm": 2.0, "final_aae_bpm": 1.0},
        "hr": hr_rows,
        "metadata": {
            "time_bias": 5.0,
            "analysis_scope": "full",
            "adaptive_filter": "lms",
            "ref_path": str(tmp_path / "sample_ref.csv"),
        },
        "best_params": {"max_order": 16},
    }
    report = tmp_path / "f.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    ref_path = tmp_path / "sample_ref.csv"
    ref_path.write_text(
        "h1,h2,h3\n"
        + "".join(f"{i},00:00:{i:02d},{75.0:.1f}\n" for i in range(60)),
        encoding="utf-8",
    )

    arte = render_v2_report(report, out_dir=tmp_path / "figures")
    assert arte.figure_png.is_file()
