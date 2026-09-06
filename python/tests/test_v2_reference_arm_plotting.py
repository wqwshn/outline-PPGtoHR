from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from ppg_hr.v2.reference_arm_plotting import (
    FIGURE_HEIGHT_MM,
    FIGURE_WIDTH_MM,
    ROUTE_COLORS,
    load_reference_arm_figure_data,
    render_reference_arm_figure,
)

ROUTES = ("HF", "ACC", "HF_ACC")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_analysis_root(tmp_path: Path) -> Path:
    analysis_root = tmp_path / "analysis"
    diagonals = []
    effects = []
    matrices = []
    for scene_index in range(8):
        scene = f"scene{scene_index + 1}"
        for record_index in range(3):
            fold = f"{scene}::r{record_index + 1}"
            holdout = f"{scene}_r{record_index + 1}"
            values = {
                "HF": 2.0 + scene_index * 0.4 + record_index * 0.1,
                "ACC": 3.1 + scene_index * 0.45 + record_index * 0.1,
                "HF_ACC": 1.7 + scene_index * 0.35 + record_index * 0.1,
            }
            for route, value in values.items():
                diagonals.append(
                    {
                        "scene": scene,
                        "fold_id": fold,
                        "holdout_record_id": holdout,
                        "route_id": route,
                        "mae_bpm": value,
                    }
                )
            for contrast, difference in (
                ("ACC_minus_HF", values["ACC"] - values["HF"]),
                ("HF_minus_HF_ACC", values["HF"] - values["HF_ACC"]),
                ("ACC_minus_HF_ACC", values["ACC"] - values["HF_ACC"]),
            ):
                effects.append(
                    {
                        "scene": scene,
                        "fold_id": fold,
                        "holdout_record_id": holdout,
                        "contrast_id": contrast,
                        "difference_bpm": difference,
                    }
                )
            for actual_index, actual in enumerate(ROUTES):
                for source_index, source in enumerate(ROUTES):
                    penalty = 0.0 if actual == source else 0.2 * (source_index - actual_index)
                    matrices.append(
                        {
                            "scene": scene,
                            "fold_id": fold,
                            "holdout_record_id": holdout,
                            "actual_route_id": actual,
                            "coordinate_source_route_id": source,
                            "coordinate_id": f"c{source_index}",
                            "coordinate_index": source_index,
                            "mae_bpm": values[actual] + penalty,
                            "is_diagonal": actual == source,
                        }
                    )
    _write_csv(analysis_root / "diagonal_rows.csv", diagonals)
    _write_csv(analysis_root / "paired_effect_rows.csv", effects)
    _write_csv(analysis_root / "cross_matrix_rows.csv", matrices)
    return analysis_root


def test_figure_data_preserves_points_and_row_relative_transfer(tmp_path: Path) -> None:
    data = load_reference_arm_figure_data(_synthetic_analysis_root(tmp_path))

    assert len(data.diagonal_rows) == 72
    assert len(data.effect_rows) == 72
    assert len(data.matrix_rows) == 216
    assert len(data.scene_order) == 8
    assert all(data.transfer_delta[index][index] == 0.0 for index in range(3))
    assert data.transfer_delta[0][1] == pytest.approx(0.2)
    assert data.transfer_delta[1][0] == pytest.approx(-0.2)


def test_render_exports_exact_journal_size_editable_svg_and_qa(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    analysis_root = _synthetic_analysis_root(tmp_path)
    output_root = tmp_path / "figures"
    monkeypatch.setitem(plt.rcParams, "axes.titleweight", "600")

    receipt = render_reference_arm_figure(analysis_root, output_root)

    assert receipt["status"] == "PASS"
    assert receipt["diagonal_point_count"] == 72
    assert receipt["effect_point_count"] == 72
    assert receipt["matrix_cell_count"] == 216
    assert receipt["scene_mean_count_panel_a"] == 24
    assert receipt["scene_mean_count_panel_b"] == 24
    assert receipt["range_whisker_count_panel_a"] == 24
    assert receipt["raw_point_count_panel_b"] == 0
    assert receipt["panel_b_legend_labels"] == [
        "Scene mean (n=3)",
        "Overall mean (n=24)",
        "Overall median",
    ]
    assert ROUTE_COLORS == {
        "HF": "#D97706",
        "ACC": "#4C78A8",
        "HF_ACC": "#8B6FAF",
    }
    png = output_root / "reference_arm_threefold_main.png"
    svg = output_root / "reference_arm_threefold_main.svg"
    pdf = output_root / "reference_arm_threefold_main.pdf"
    with Image.open(png) as image:
        assert image.size == (
            int(FIGURE_WIDTH_MM / 25.4 * 600),
            int(FIGURE_HEIGHT_MM / 25.4 * 600),
        )
    assert "<text" in svg.read_text(encoding="utf-8")
    assert pdf.stat().st_size > 10_000
    qa = json.loads((output_root / "figure_qa.json").read_text(encoding="utf-8"))
    assert qa["status"] == "PASS"
    assert qa["editable_svg_text"] is True
    assert qa["panel_labels"] == ["a", "b", "c"]
