from __future__ import annotations

import json
from pathlib import Path

from ppg_hr.v2.plotting import (
    discover_v2_plot_jobs,
    render_v2_report,
    render_v2_report_batch,
)


def _write_report(path: Path, order: list[str]) -> None:
    payload = {
        "schema_version": "v2",
        "data_path": "sample.csv",
        "ref_path": "sample_ref.csv",
        "ppg_mode": "green",
        "analysis_scope": "full",
        "adaptive_filter": "noncausal_lms",
        "reference_groups_order": order,
        "err_stats": {"fft_aae_bpm": 2.0, "final_aae_bpm": 1.0},
        "hr": [
            [0.0, 75.0, 74.0, 75.5, 0.0, 0.0],
            [1.0, 76.0, 75.0, 76.2, 0.0, 0.0],
        ],
        "best_params": {"max_order": 16},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


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
