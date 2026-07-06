from __future__ import annotations

import csv
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_reference_rows_read_hf_and_acc_from_error_csv(tmp_path: Path) -> None:
    from ppg_hr.v2.post_motion_gap_rescue_figures import (
        reference_comparison_rows_from_summary,
    )

    err = tmp_path / "sample-error.csv"
    _write_csv(
        err,
        [
            {"method": "reset FFT", "total_aae": 8.0},
            {"method": "LMS+H", "total_aae": 2.0},
            {"method": "LMS+A", "total_aae": 3.0},
        ],
    )
    summary = tmp_path / "v2_generalization_summary.csv"
    _write_csv(
        summary,
        [
            {
                "motion_type": "bobi",
                "fold_id": "fold_01",
                "split": "test",
                "sample_stem": "multi_bobi1",
                "final_aae_bpm": 2.0,
                "error_csv": str(err),
            }
        ],
    )

    rows = reference_comparison_rows_from_summary(summary)

    assert rows == [
        {
            "motion_type": "bobi",
            "fold_id": "fold_01",
            "split": "test",
            "sample_stem": "multi_bobi1",
            "reference": "HF",
            "final_aae_bpm": 2.0,
        },
        {
            "motion_type": "bobi",
            "fold_id": "fold_01",
            "split": "test",
            "sample_stem": "multi_bobi1",
            "reference": "ACC",
            "final_aae_bpm": 3.0,
        },
    ]


def test_render_reference_figures_create_pngs(tmp_path: Path) -> None:
    from ppg_hr.v2.post_motion_gap_rescue_figures import (
        render_cross_motion_reference_comparison,
        render_train_vs_eval_gap_reference,
    )

    rows = [
        {
            "motion_type": "bobi",
            "split": "train",
            "reference": "HF",
            "final_aae_bpm": 2.0,
        },
        {
            "motion_type": "bobi",
            "split": "train",
            "reference": "ACC",
            "final_aae_bpm": 2.5,
        },
        {
            "motion_type": "bobi",
            "split": "test",
            "reference": "HF",
            "final_aae_bpm": 3.0,
        },
        {
            "motion_type": "bobi",
            "split": "test",
            "reference": "ACC",
            "final_aae_bpm": 4.0,
        },
    ]
    cross = render_cross_motion_reference_comparison(
        rows,
        tmp_path / "cross_motion_reference_comparison.png",
    )
    gap = render_train_vs_eval_gap_reference(
        rows,
        tmp_path / "train_vs_eval_gap_reference.png",
    )

    assert cross.is_file()
    assert cross.stat().st_size > 1000
    assert gap.is_file()
    assert gap.stat().st_size > 1000
