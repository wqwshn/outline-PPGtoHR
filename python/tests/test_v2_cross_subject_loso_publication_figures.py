from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ppg_hr.v2.cross_subject_loso_figures import (
    build_publication_tables,
    parse_coordinate_id,
    save_publication_png,
)

SCENES = ("bobi", "jianpan", "kaihe", "quanji", "run", "tiaosheng", "woli", "xiezi")
SUBJECTS = ("CGX", "LYX", "LZJ", "PJY", "QYC", "TS", "HB")


def _coordinate(index: int) -> str:
    return f"physical4d:fs{25 + index:03d}:m{80 + index}:mu0006:w006"


def _fold_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scene_index, scene in enumerate(SCENES):
        roster = SUBJECTS[:6] if scene != "run" else (*SUBJECTS[:4], SUBJECTS[5], SUBJECTS[6])
        for subject_index, subject in enumerate(roster):
            coordinate_index = (scene_index * 6 + subject_index) % 23
            rows.append(
                {
                    "fold_id": f"{scene}__holdout_{subject}",
                    "scene": scene,
                    "holdout_subject_id": subject,
                    "selected_coordinate_id": _coordinate(coordinate_index),
                    "minimum_training_subject_pass_fraction": ("0", "1/3", "2/3")[
                        (scene_index + subject_index) % 3
                    ],
                    "candidate_mean_mae_bpm": 2.0 + scene_index + subject_index / 10,
                    "baseline_mean_mae_bpm": 1.0 + scene_index / 10,
                }
            )
    return pd.DataFrame(rows)


def _record_rows(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_index, fold in folds.iterrows():
        repeat_count = 2 if fold_index == 0 else 3
        for repeat_index in range(repeat_count):
            value = float(fold["candidate_mean_mae_bpm"]) + repeat_index / 10
            failed = ["G2", "G3"] if (fold_index + repeat_index) % 2 else []
            rows.append(
                {
                    "fold_id": fold["fold_id"],
                    "holdout_subject_id": fold["holdout_subject_id"],
                    "scene": fold["scene"],
                    "record_id": f"record_{fold_index:02d}_{repeat_index}",
                    "candidate_mae_bpm": value,
                    "baseline_mae_bpm": value - 0.5,
                    "mae_delta_vs_baseline_bpm": 0.5,
                    "g1i_pass": True,
                    "g2_pass": "G2" not in failed,
                    "g3_pass": "G3" not in failed,
                    "g4_pass": True,
                    "g5_pass": True,
                    "g7_pass": True,
                    "qualified": not failed,
                    "failed_gates_json": '["G2", "G3"]' if failed else "[]",
                }
            )
    return pd.DataFrame(rows)


def test_parse_coordinate_id_exposes_all_four_physical_axes() -> None:
    assert parse_coordinate_id("physical4d:fs100:m200:mu0008:w006") == {
        "fs_target_hz": 100,
        "memory_ms": 200,
        "mu_base": pytest.approx(0.008),
        "exclusion_half_width_bpm": 6,
    }


def test_publication_tables_preserve_panel_and_observation_contract() -> None:
    folds = _fold_rows()
    records = _record_rows(folds)
    assert len(records) == 143

    tables = build_publication_tables(folds, records)

    assert tables.fold_pairs.shape[0] == 48
    assert tables.record_deltas.shape[0] == 143
    assert tables.fold_mae_matrix.shape == (7, 8)
    assert tables.fold_mae_matrix.notna().sum().sum() == 48
    assert tables.training_fraction_matrix.notna().sum().sum() == 48
    assert tables.coordinate_scene_counts.shape == (8, 23)
    assert tables.coordinate_parameters.shape == (4, 23)
    assert int(tables.gate_summary.loc["All six", "passed_records"]) == int(
        records["qualified"].sum()
    )


def test_publication_png_has_fixed_600_dpi_canvas_without_extra_formats(tmp_path: Path) -> None:
    fig = plt.figure(figsize=(7.2, 1.0))
    target = tmp_path / "figure.png"

    save_publication_png(fig, target)

    rendered = mpimg.imread(target)
    assert rendered.shape[:2] == (600, 4320)
    assert [path.suffix for path in tmp_path.iterdir()] == [".png"]
