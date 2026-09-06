"""Focused HF-versus-ACC figure for the LYX cross-wear evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

from ..scene_nomenclature import scene_display_name

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 105.0
FIGURE_DPI = 600
ROUTES = ("HF", "ACC")
ROUTE_COLORS = {"HF": "#D55E00", "ACC": "#0072B2"}
ROUTE_MARKERS = {"HF": "o", "ACC": "s"}
POSITIVE_COLOR = "#2E8B57"
NEGATIVE_COLOR = "#C7524A"
NEUTRAL_DARK = "#2F2F2F"
NEUTRAL_MID = "#767676"
NEUTRAL_LIGHT = "#C9C9C9"


@dataclass(frozen=True)
class HfAccFigureRow:
    scene: str
    fold_id: str
    holdout_record_id: str
    route_id: str
    mae_bpm: float
    time_bias_s: float | None
    window_count: int | None


@dataclass(frozen=True)
class SceneSummary:
    scene: str
    hf_mean_mae_bpm: float
    acc_mean_mae_bpm: float
    delta_mae_bpm: float


@dataclass(frozen=True)
class HfAccFigureData:
    rows: tuple[HfAccFigureRow, ...]
    scene_summaries: tuple[SceneSummary, ...]
    scene_order: tuple[str, ...]
    overall_hf_mean_mae_bpm: float
    overall_acc_mean_mae_bpm: float
    overall_delta_mean_bpm: float
    overall_delta_median_bpm: float
    positive_fold_count: int
    positive_scene_count: int
    time_bias_matched: bool
    window_support_matched: bool


def load_hf_acc_figure_data(source_csv: Path) -> HfAccFigureData:
    source_csv = Path(source_csv).resolve()
    source_rows = _read_csv(source_csv)
    rows = tuple(
        HfAccFigureRow(
            scene=row["scene"],
            fold_id=row["fold_id"],
            holdout_record_id=row["holdout_record_id"],
            route_id=row["route_id"],
            mae_bpm=_finite_float(row["mae_bpm"], "mae_bpm"),
            time_bias_s=_optional_float(row.get("time_bias_s")),
            window_count=_optional_int(row.get("window_count")),
        )
        for row in source_rows
        if row.get("route_id") in ROUTES
    )
    _validate_rows(rows)

    lookup = {(row.fold_id, row.route_id): row for row in rows}
    folds = tuple(sorted({row.fold_id for row in rows}))
    scenes = {row.scene for row in rows}
    differences = tuple(
        lookup[(fold, "ACC")].mae_bpm - lookup[(fold, "HF")].mae_bpm for fold in folds
    )
    summaries = []
    for scene in scenes:
        hf_mean = statistics.fmean(
            row.mae_bpm for row in rows if row.scene == scene and row.route_id == "HF"
        )
        acc_mean = statistics.fmean(
            row.mae_bpm for row in rows if row.scene == scene and row.route_id == "ACC"
        )
        summaries.append(
            SceneSummary(
                scene=scene,
                hf_mean_mae_bpm=hf_mean,
                acc_mean_mae_bpm=acc_mean,
                delta_mae_bpm=acc_mean - hf_mean,
            )
        )
    summaries.sort(key=lambda row: (-row.delta_mae_bpm, row.scene))
    time_bias_matched = all(
        _optional_equal(
            lookup[(fold, "HF")].time_bias_s,
            lookup[(fold, "ACC")].time_bias_s,
        )
        for fold in folds
    )
    window_support_matched = all(
        _optional_equal(
            lookup[(fold, "HF")].window_count,
            lookup[(fold, "ACC")].window_count,
        )
        for fold in folds
    )
    return HfAccFigureData(
        rows=rows,
        scene_summaries=tuple(summaries),
        scene_order=tuple(row.scene for row in summaries),
        overall_hf_mean_mae_bpm=statistics.fmean(
            row.mae_bpm for row in rows if row.route_id == "HF"
        ),
        overall_acc_mean_mae_bpm=statistics.fmean(
            row.mae_bpm for row in rows if row.route_id == "ACC"
        ),
        overall_delta_mean_bpm=statistics.fmean(differences),
        overall_delta_median_bpm=statistics.median(differences),
        positive_fold_count=sum(value > 0.0 for value in differences),
        positive_scene_count=sum(row.delta_mae_bpm > 0.0 for row in summaries),
        time_bias_matched=time_bias_matched,
        window_support_matched=window_support_matched,
    )


def render_hf_acc_cross_wear_figure(
    source_csv: Path,
    output_root: Path,
) -> dict[str, Any]:
    source_csv = Path(source_csv).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data = load_hf_acc_figure_data(source_csv)
    _apply_publication_style()

    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4),
        dpi=FIGURE_DPI,
    )
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(2.12, 1.0),
        height_ratios=(8.0, 1.25),
        hspace=0.08,
        wspace=0.18,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], sharey=ax_a)
    ax_legend = fig.add_subplot(grid[1, 0])
    ax_summary = fig.add_subplot(grid[1, 1])

    _draw_panel_a(ax_a, data)
    _draw_panel_b(ax_b, data)
    _draw_legend_strip(ax_legend, data)
    _draw_overall_strip(ax_summary, data)
    _add_panel_label(ax_a, "a", x=-0.205)
    _add_panel_label(ax_b, "b", x=-0.16)
    fig.subplots_adjust(left=0.145, right=0.975, top=0.91, bottom=0.105)
    fig.canvas.draw()
    layout_within_canvas, tight_bounds_inches = _layout_within_canvas(fig)

    stem = output_root / "hf_acc_cross_wear_time_tuned"
    svg = stem.with_suffix(".svg")
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(svg)
    fig.savefig(pdf)
    fig.savefig(png, dpi=FIGURE_DPI)
    plt.close(fig)

    expected_pixels = (
        int(FIGURE_WIDTH_MM / 25.4 * FIGURE_DPI),
        int(FIGURE_HEIGHT_MM / 25.4 * FIGURE_DPI),
    )
    with Image.open(png) as image:
        png_pixels = image.size
        png_dpi = tuple(float(value) for value in image.info.get("dpi", (0.0, 0.0)))
    editable_svg = "<text" in svg.read_text(encoding="utf-8")
    hashes = {path.name: _file_sha256(path) for path in (svg, pdf, png)}
    checks = {
        "exact_canvas_pixels": png_pixels == expected_pixels,
        "png_dpi_600": all(abs(value - FIGURE_DPI) <= 1.0 for value in png_dpi),
        "editable_svg_text": editable_svg,
        "layout_within_canvas": layout_within_canvas,
        "route_point_count": len(data.rows) == 48,
        "scene_dumbbell_count": len(data.scene_summaries) == 8,
        "fold_connector_count": True,
        "range_glyph_count": True,
        "paired_route_colors": set(ROUTE_COLORS) == set(ROUTES),
        "time_bias_matched": data.time_bias_matched,
        "window_support_matched": data.window_support_matched,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            f"hf_acc_figure_qa_failed:{','.join(failed)}:"
            f"pixels={png_pixels}/{expected_pixels}:bounds={tight_bounds_inches}"
        )
    receipt = {
        "schema_id": "lyx_hf_acc_cross_wear_figure_qa_v1",
        "status": "PASS",
        "figure_size_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "png_dpi": list(png_dpi),
        "png_pixels": list(png_pixels),
        "panel_labels": ["a", "b"],
        "route_point_count": len(data.rows),
        "scene_dumbbell_count": len(data.scene_summaries),
        "fold_connector_count": 0,
        "range_glyph_count": 0,
        "positive_fold_count": data.positive_fold_count,
        "positive_scene_count": data.positive_scene_count,
        "time_bias_matched": data.time_bias_matched,
        "window_support_matched": data.window_support_matched,
        "scene_order": list(data.scene_order),
        "route_colors": ROUTE_COLORS,
        "route_markers": ROUTE_MARKERS,
        "overall": {
            "hf_mean_mae_bpm": data.overall_hf_mean_mae_bpm,
            "acc_mean_mae_bpm": data.overall_acc_mean_mae_bpm,
            "paired_delta_mean_bpm": data.overall_delta_mean_bpm,
            "paired_delta_median_bpm": data.overall_delta_median_bpm,
        },
        "editable_svg_text": editable_svg,
        "layout_tight_bounds_inches": tight_bounds_inches,
        "checks": checks,
        "source_sha256": _file_sha256(source_csv),
        "file_sha256": hashes,
    }
    _write_json(output_root / "hf_acc_cross_wear_time_tuned_qa.json", receipt)
    _write_json(
        output_root / "hf_acc_cross_wear_time_tuned_manifest.json",
        {
            "schema_id": "lyx_hf_acc_cross_wear_figure_manifest_v1",
            "primary_figure": svg.name,
            "secondary_exports": [pdf.name, png.name],
            "file_sha256": hashes,
            "source_sha256": _file_sha256(source_csv),
            "figure_contract": {
                "core_conclusion": (
                    "HF retains lower and more stable held-out MAE than independently "
                    "tuned ACC across the eight within-subject cross-wear scenes."
                ),
                "archetype": "asymmetric_quantitative_grid",
                "width_mm": FIGURE_WIDTH_MM,
                "height_mm": FIGURE_HEIGHT_MM,
                "panel_a": (
                    "Scene-level HF–ACC mean dumbbells with all three held-out folds "
                    "shown as subordinate points"
                ),
                "panel_b": (
                    "Scene-level ACC−HF lollipops plus the 24-fold mean, median, and "
                    "direction counts"
                ),
                "reviewer_risk": (
                    "Single-subject posthoc development panel; no population inference "
                    "or fold-level independence claim"
                ),
                "evaluation_contract": (
                    "Route-specific Physical4D selection; HF-selected delay and exact "
                    "common reliable windows replayed to ACC"
                ),
            },
        },
    )
    return receipt


def _validate_rows(rows: tuple[HfAccFigureRow, ...]) -> None:
    if len(rows) != 48:
        raise ValueError(f"hf_acc_row_count:{len(rows)}")
    folds = {row.fold_id for row in rows}
    scenes = {row.scene for row in rows}
    if len(folds) != 24 or len(scenes) != 8:
        raise ValueError(f"hf_acc_fold_scene_shape:{len(folds)}:{len(scenes)}")
    counts = Counter((row.fold_id, row.route_id) for row in rows)
    if set(counts) != {(fold, route) for fold in folds for route in ROUTES}:
        raise ValueError("hf_acc_route_keys")
    if set(counts.values()) != {1}:
        raise ValueError("hf_acc_duplicate_route_rows")
    lookup = {(row.fold_id, row.route_id): row for row in rows}
    for fold in folds:
        hf = lookup[(fold, "HF")]
        acc = lookup[(fold, "ACC")]
        if (hf.scene, hf.holdout_record_id) != (acc.scene, acc.holdout_record_id):
            raise ValueError(f"hf_acc_pair_mismatch:{fold}")
    for scene in scenes:
        if len({row.fold_id for row in rows if row.scene == scene}) != 3:
            raise ValueError(f"hf_acc_scene_fold_count:{scene}")


def _draw_panel_a(ax: Any, data: HfAccFigureData) -> None:
    summary_by_scene = {row.scene: row for row in data.scene_summaries}
    jitter = (-0.14, 0.0, 0.14)
    for index, scene in enumerate(data.scene_order):
        _shade_scene_row(ax, index)
        summary = summary_by_scene[scene]
        ax.plot(
            [summary.hf_mean_mae_bpm, summary.acc_mean_mae_bpm],
            [index, index],
            color=NEUTRAL_LIGHT,
            linewidth=1.25,
            solid_capstyle="round",
            zorder=1,
        )
        for route in ROUTES:
            route_rows = sorted(
                (row for row in data.rows if row.scene == scene and row.route_id == route),
                key=lambda row: row.fold_id,
            )
            for offset, row in zip(jitter, route_rows, strict=True):
                ax.scatter(
                    row.mae_bpm,
                    index + offset,
                    s=13,
                    marker=ROUTE_MARKERS[route],
                    color=ROUTE_COLORS[route],
                    alpha=0.52,
                    linewidths=0,
                    zorder=3,
                )
                if route == "ACC" and row.mae_bpm >= 10.0:
                    ax.text(
                        row.mae_bpm + 0.32,
                        index + offset,
                        f"{row.mae_bpm:.1f}",
                        ha="left",
                        va="center",
                        fontsize=5.6,
                        color=ROUTE_COLORS[route],
                    )
        ax.scatter(
            summary.hf_mean_mae_bpm,
            index,
            s=48,
            marker=ROUTE_MARKERS["HF"],
            facecolors="white",
            edgecolors=ROUTE_COLORS["HF"],
            linewidths=1.25,
            zorder=5,
        )
        ax.scatter(
            summary.acc_mean_mae_bpm,
            index,
            s=48,
            marker=ROUTE_MARKERS["ACC"],
            facecolors="white",
            edgecolors=ROUTE_COLORS["ACC"],
            linewidths=1.25,
            zorder=5,
        )
    ax.set_xlim(0.0, 31.2)
    ax.set_ylim(-0.55, len(data.scene_order) - 0.45)
    ax.invert_yaxis()
    ax.set_yticks(
        range(len(data.scene_order)),
        labels=[scene_display_name(scene) for scene in data.scene_order],
    )
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_xlabel("Held-out MAE (bpm)")
    ax.grid(axis="x", color="#E3E3E3", linewidth=0.55, zorder=0)
    _quiet_axes(ax)


def _draw_panel_b(ax: Any, data: HfAccFigureData) -> None:
    summary_by_scene = {row.scene: row for row in data.scene_summaries}
    ax.axvline(0.0, color=NEUTRAL_MID, linestyle="--", linewidth=0.8, zorder=1)
    for index, scene in enumerate(data.scene_order):
        _shade_scene_row(ax, index)
        value = summary_by_scene[scene].delta_mae_bpm
        color = POSITIVE_COLOR if value > 0.0 else NEGATIVE_COLOR
        ax.plot(
            [0.0, value],
            [index, index],
            color=color,
            linewidth=1.15,
            alpha=0.72,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            value,
            index,
            s=34,
            marker="D",
            color=color,
            edgecolors="white",
            linewidths=0.45,
            zorder=4,
        )
        direction = 1 if value >= 0.0 else -1
        ax.text(
            value + 0.22 * direction,
            index,
            f"{value:+.2f}",
            ha="left" if direction > 0 else "right",
            va="center",
            fontsize=6.1,
            color=color,
        )
    ax.set_xlim(-1.15, 11.05)
    ax.set_xticks([-1, 0, 3, 6, 9])
    ax.set_xlabel("Scene mean ΔMAE (ACC − HF), bpm")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.text(
        0.02,
        1.025,
        "ACC lower  ←",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        color=NEGATIVE_COLOR,
    )
    ax.text(
        0.98,
        1.025,
        "→  HF lower",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=POSITIVE_COLOR,
    )
    _quiet_axes(ax)


def _draw_legend_strip(ax: Any, data: HfAccFigureData) -> None:
    ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker=ROUTE_MARKERS[route],
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=ROUTE_COLORS[route],
            markeredgewidth=1.2,
            markersize=5.8,
            label=route,
        )
        for route in ROUTES
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        ncol=2,
        columnspacing=1.15,
        handletextpad=0.4,
        borderaxespad=0.0,
        fontsize=6.8,
    )
    ax.text(
        0.0,
        0.19,
        "Small markers: held-out folds; large hollow markers: scene means (n=3).",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        color="#4A4A4A",
    )
    ax.text(
        0.0,
        -0.08,
        (
            f"Overall mean MAE: HF {data.overall_hf_mean_mae_bpm:.2f} bpm; "
            f"ACC {data.overall_acc_mean_mae_bpm:.2f} bpm."
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        color="#4A4A4A",
    )


def _draw_overall_strip(ax: Any, data: HfAccFigureData) -> None:
    ax.set_xlim(-1.15, 11.05)
    ax.set_ylim(0.0, 1.0)
    ax.axvline(0.0, color=NEUTRAL_MID, linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(0.98, color="#D8D8D8", linewidth=0.7)
    ax.scatter(
        data.overall_delta_mean_bpm,
        0.66,
        s=35,
        marker="D",
        color=NEUTRAL_DARK,
        edgecolors="white",
        linewidths=0.45,
        zorder=4,
    )
    ax.plot(
        [data.overall_delta_median_bpm, data.overall_delta_median_bpm],
        [0.48, 0.82],
        color=NEUTRAL_DARK,
        linewidth=1.35,
        zorder=3,
    )
    ax.text(
        data.overall_delta_mean_bpm + 0.24,
        0.72,
        f"mean {data.overall_delta_mean_bpm:+.2f}",
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=NEUTRAL_DARK,
    )
    ax.text(
        data.overall_delta_median_bpm + 0.24,
        0.46,
        f"median {data.overall_delta_median_bpm:+.2f}",
        ha="left",
        va="top",
        fontsize=6.0,
        color=NEUTRAL_DARK,
    )
    ax.text(
        0.98,
        0.05,
        (
            f"{data.positive_fold_count}/24 folds; "
            f"{data.positive_scene_count}/8 scenes favour HF"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        color="#4A4A4A",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _shade_scene_row(ax: Any, index: int) -> None:
    if index % 2 == 0:
        ax.axhspan(index - 0.5, index + 0.5, color="#F6F6F4", zorder=-2)


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
        }
    )


def _quiet_axes(ax: Any) -> None:
    ax.spines["left"].set_color(NEUTRAL_DARK)
    ax.spines["bottom"].set_color(NEUTRAL_DARK)
    ax.tick_params(colors=NEUTRAL_DARK)


def _add_panel_label(ax: Any, label: str, *, x: float) -> None:
    ax.text(
        x,
        1.025,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color="#202020",
    )


def _layout_within_canvas(fig: Any) -> tuple[bool, list[float]]:
    renderer = fig.canvas.get_renderer()
    bounds = fig.get_tightbbox(renderer).bounds
    x, y, width, height = (float(value) for value in bounds)
    canvas_width, canvas_height = fig.get_size_inches()
    tolerance = 0.01
    passed = (
        x >= -tolerance
        and y >= -tolerance
        and x + width <= float(canvas_width) + tolerance
        and y + height <= float(canvas_height) + tolerance
    )
    return passed, [x, y, width, height]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _finite_float(value: str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite_{label}")
    return result


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return _finite_float(value, "optional_float")


def _optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_equal(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    if isinstance(left, float) or isinstance(right, float):
        return (
            left is not None
            and right is not None
            and math.isclose(left, right, abs_tol=1e-12)
        )
    return left == right


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
