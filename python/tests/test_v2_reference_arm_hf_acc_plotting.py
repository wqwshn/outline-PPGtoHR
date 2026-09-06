from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from ppg_hr.v2.reference_arm_hf_acc_plotting import (
    FIGURE_DPI,
    FIGURE_HEIGHT_MM,
    FIGURE_WIDTH_MM,
    ROUTE_COLORS,
    load_hf_acc_figure_data,
    render_hf_acc_cross_wear_figure,
)


def _write_source(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for scene_index in range(8):
        scene = f"scene{scene_index + 1}"
        scene_delta = 7.0 - scene_index
        if scene_index == 7:
            scene_delta = -0.2
        for fold_index in range(3):
            fold_id = f"{scene}::fold{fold_index + 1}"
            hf = 1.4 + 0.12 * scene_index + 0.08 * fold_index
            acc = hf + scene_delta + 0.05 * fold_index
            for route_id, value in (("HF", hf), ("ACC", acc), ("HF_ACC", hf + 0.4)):
                rows.append(
                    {
                        "scene": scene,
                        "fold_id": fold_id,
                        "holdout_record_id": f"{scene}_record{fold_index + 1}",
                        "route_id": route_id,
                        "coordinate_id": f"{route_id}_coordinate_{fold_index + 1}",
                        "time_bias_s": 4.0 + 0.5 * fold_index,
                        "mae_bpm": value,
                        "window_count": 40 + fold_index,
                        "evaluation_contract": "matched_hf_delay_and_common_windows",
                        "data_source": "synthetic",
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_loader_keeps_paired_hf_acc_rows_and_orders_scenes_by_advantage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "time_tuned_diagonal_rows.csv"
    _write_source(source)

    data = load_hf_acc_figure_data(source)

    assert len(data.rows) == 48
    assert len(data.scene_summaries) == 8
    assert data.scene_order == tuple(f"scene{index}" for index in range(1, 9))
    assert data.positive_scene_count == 7
    assert data.positive_fold_count == 21
    assert data.time_bias_matched is True
    assert data.window_support_matched is True
    assert data.scene_summaries[0].delta_mae_bpm == pytest.approx(7.05)
    assert data.scene_summaries[-1].delta_mae_bpm == pytest.approx(-0.15)


def test_render_exports_two_panel_journal_figure_and_qa(tmp_path: Path) -> None:
    source = tmp_path / "time_tuned_diagonal_rows.csv"
    output = tmp_path / "figures"
    _write_source(source)

    receipt = render_hf_acc_cross_wear_figure(source, output)

    assert receipt["status"] == "PASS"
    assert receipt["panel_labels"] == ["a", "b"]
    assert receipt["route_point_count"] == 48
    assert receipt["scene_dumbbell_count"] == 8
    assert receipt["fold_connector_count"] == 0
    assert receipt["range_glyph_count"] == 0
    assert receipt["positive_scene_count"] == 7
    assert receipt["time_bias_matched"] is True
    assert receipt["window_support_matched"] is True
    assert ROUTE_COLORS == {"HF": "#D55E00", "ACC": "#0072B2"}

    png = output / "hf_acc_cross_wear_time_tuned.png"
    svg = output / "hf_acc_cross_wear_time_tuned.svg"
    pdf = output / "hf_acc_cross_wear_time_tuned.pdf"
    with Image.open(png) as image:
        assert image.size == (
            int(FIGURE_WIDTH_MM / 25.4 * FIGURE_DPI),
            int(FIGURE_HEIGHT_MM / 25.4 * FIGURE_DPI),
        )
        assert all(abs(value - FIGURE_DPI) <= 1.0 for value in image.info["dpi"])
    assert "<text" in svg.read_text(encoding="utf-8")
    assert pdf.stat().st_size > 10_000

    qa = json.loads(
        (output / "hf_acc_cross_wear_time_tuned_qa.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output / "hf_acc_cross_wear_time_tuned_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert qa == receipt
    assert manifest["primary_figure"] == "hf_acc_cross_wear_time_tuned.svg"
    assert manifest["figure_contract"]["archetype"] == "asymmetric_quantitative_grid"
    assert manifest["figure_contract"]["panel_a"].startswith("Scene-level HF–ACC")
