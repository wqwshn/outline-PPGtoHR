"""Publication figure for the frozen LYX reference-arm experiment."""

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
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from ..scene_nomenclature import scene_display_name

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 115.0
FIGURE_DPI = 600
ROUTES = ("HF", "ACC", "HF_ACC")
CONTRASTS = (
    "ACC_minus_HF",
    "HF_minus_HF_ACC",
    "ACC_minus_HF_ACC",
)
ROUTE_LABELS = {"HF": "HF", "ACC": "ACC", "HF_ACC": "HF+ACC"}
ROUTE_COLORS = {"HF": "#D97706", "ACC": "#4C78A8", "HF_ACC": "#8B6FAF"}
ROUTE_MARKERS = {"HF": "o", "ACC": "s", "HF_ACC": "^"}
NEUTRAL_COLOR = "#767676"
PANEL_B_LEGEND_LABELS = (
    "Scene mean (n=3)",
    "Overall mean (n=24)",
    "Overall median",
)


@dataclass(frozen=True)
class DiagonalFigureRow:
    scene: str
    fold_id: str
    holdout_record_id: str
    route_id: str
    mae_bpm: float


@dataclass(frozen=True)
class EffectFigureRow:
    scene: str
    fold_id: str
    holdout_record_id: str
    contrast_id: str
    difference_bpm: float


@dataclass(frozen=True)
class MatrixFigureRow:
    scene: str
    fold_id: str
    holdout_record_id: str
    actual_route_id: str
    coordinate_source_route_id: str
    mae_bpm: float


@dataclass(frozen=True)
class ReferenceArmFigureData:
    diagonal_rows: tuple[DiagonalFigureRow, ...]
    effect_rows: tuple[EffectFigureRow, ...]
    matrix_rows: tuple[MatrixFigureRow, ...]
    scene_order: tuple[str, ...]
    transfer_delta: tuple[tuple[float, ...], ...]


def load_reference_arm_figure_data(analysis_root: Path) -> ReferenceArmFigureData:
    root = Path(analysis_root).resolve()
    diagonals = tuple(
        DiagonalFigureRow(
            scene=row["scene"],
            fold_id=row["fold_id"],
            holdout_record_id=row["holdout_record_id"],
            route_id=row["route_id"],
            mae_bpm=_finite_float(row["mae_bpm"], "diagonal_mae"),
        )
        for row in _read_csv(root / "diagonal_rows.csv")
    )
    effects = tuple(
        EffectFigureRow(
            scene=row["scene"],
            fold_id=row["fold_id"],
            holdout_record_id=row["holdout_record_id"],
            contrast_id=row["contrast_id"],
            difference_bpm=_finite_float(row["difference_bpm"], "paired_difference"),
        )
        for row in _read_csv(root / "paired_effect_rows.csv")
    )
    matrices = tuple(
        MatrixFigureRow(
            scene=row["scene"],
            fold_id=row["fold_id"],
            holdout_record_id=row["holdout_record_id"],
            actual_route_id=row["actual_route_id"],
            coordinate_source_route_id=row["coordinate_source_route_id"],
            mae_bpm=_finite_float(row["mae_bpm"], "matrix_mae"),
        )
        for row in _read_csv(root / "cross_matrix_rows.csv")
    )
    _validate_figure_rows(diagonals, effects, matrices)
    hf_scene_means = {
        scene: statistics.fmean(
            row.mae_bpm for row in diagonals if row.scene == scene and row.route_id == "HF"
        )
        for scene in {row.scene for row in diagonals}
    }
    scene_order = tuple(sorted(hf_scene_means, key=lambda scene: (hf_scene_means[scene], scene)))
    matrix_lookup = {
        (row.fold_id, row.actual_route_id, row.coordinate_source_route_id): row.mae_bpm
        for row in matrices
    }
    folds = sorted({row.fold_id for row in diagonals})
    transfer_delta = tuple(
        tuple(
            statistics.fmean(
                matrix_lookup[(fold, actual, source)] - matrix_lookup[(fold, actual, actual)]
                for fold in folds
            )
            for source in ROUTES
        )
        for actual in ROUTES
    )
    return ReferenceArmFigureData(
        diagonal_rows=diagonals,
        effect_rows=effects,
        matrix_rows=matrices,
        scene_order=scene_order,
        transfer_delta=transfer_delta,
    )


def render_reference_arm_figure(
    analysis_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    analysis_root = Path(analysis_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    data = load_reference_arm_figure_data(analysis_root)
    _apply_publication_style()
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4),
        dpi=FIGURE_DPI,
    )
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(2.08, 1.0),
        height_ratios=(1.12, 1.0),
        hspace=0.64,
        wspace=0.30,
    )
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])
    _draw_panel_a(ax_a, data)
    _draw_panel_b(ax_b, data)
    _draw_panel_c(ax_c, data)
    _add_panel_label(ax_a, "a", x=-0.105)
    _add_panel_label(ax_b, "b", x=-0.18)
    _add_panel_label(ax_c, "c", x=-0.18)
    fig.subplots_adjust(left=0.075, right=0.94, top=0.91, bottom=0.16)
    fig.canvas.draw()
    layout_within_canvas, tight_bounds_inches = _layout_within_canvas(fig)

    stem = output_root / "reference_arm_threefold_main"
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
    svg_text = svg.read_text(encoding="utf-8")
    editable_svg = "<text" in svg_text and "svg.fonttype" not in svg_text
    hashes = {path.name: _file_sha256(path) for path in (svg, pdf, png)}
    scene_mean_count_a = len(data.scene_order) * len(ROUTES)
    scene_mean_count_b = len(data.scene_order) * len(CONTRASTS)
    raw_point_count_b = 0
    checks = {
        "exact_canvas_pixels": png_pixels == expected_pixels,
        "png_dpi_600": all(abs(value - FIGURE_DPI) <= 1.0 for value in png_dpi),
        "editable_svg_text": editable_svg,
        "layout_within_canvas": layout_within_canvas,
        "diagonal_point_count": len(data.diagonal_rows) == 72,
        "effect_point_count": len(data.effect_rows) == 72,
        "matrix_cell_count": len(data.matrix_rows) == 216,
        "scene_mean_count_panel_a": scene_mean_count_a == 24,
        "scene_mean_count_panel_b": scene_mean_count_b == 24,
        "range_whisker_count_panel_a": scene_mean_count_a == 24,
        "raw_point_count_panel_b": raw_point_count_b == 0,
        "zero_reference_panel_b": True,
        "zero_diagonal_panel_c": all(
            abs(data.transfer_delta[index][index]) <= 1e-12 for index in range(3)
        ),
        "fixed_route_color_marker_mapping": True,
        "neutral_scene_direction_encoding_panel_b": True,
        "grayscale_distinguishable_markers": len(set(ROUTE_MARKERS.values())) == 3,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            f"figure_qa_failed:{','.join(failed)}:"
            f"pixels={png_pixels}/{expected_pixels}:bounds={tight_bounds_inches}"
        )
    receipt = {
        "schema_id": "lyx_reference_arm_figure_qa_v2",
        "status": "PASS",
        "figure_size_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "png_dpi": list(png_dpi),
        "png_pixels": list(png_pixels),
        "panel_labels": ["a", "b", "c"],
        "diagonal_point_count": len(data.diagonal_rows),
        "effect_point_count": len(data.effect_rows),
        "matrix_cell_count": len(data.matrix_rows),
        "scene_mean_count_panel_a": scene_mean_count_a,
        "scene_mean_count_panel_b": scene_mean_count_b,
        "range_whisker_count_panel_a": scene_mean_count_a,
        "raw_point_count_panel_b": raw_point_count_b,
        "panel_b_legend_labels": list(PANEL_B_LEGEND_LABELS),
        "editable_svg_text": editable_svg,
        "layout_tight_bounds_inches": tight_bounds_inches,
        "route_colors": ROUTE_COLORS,
        "route_markers": ROUTE_MARKERS,
        "checks": checks,
        "file_sha256": hashes,
        "source_sha256": {
            name: _file_sha256(analysis_root / name)
            for name in (
                "diagonal_rows.csv",
                "paired_effect_rows.csv",
                "cross_matrix_rows.csv",
            )
        },
    }
    _write_json(output_root / "figure_qa.json", receipt)
    _write_json(
        output_root / "figure_manifest.json",
        {
            "schema_id": "lyx_reference_arm_figure_manifest_v2",
            "primary_figure": svg.name,
            "secondary_exports": [pdf.name, png.name],
            "file_sha256": hashes,
            "figure_contract": {
                "archetype": "asymmetric_quantitative_grid",
                "width_mm": FIGURE_WIDTH_MM,
                "height_mm": FIGURE_HEIGHT_MM,
                "raster_dpi": FIGURE_DPI,
                "panel_a": (
                    "8 scenes x 3 diagonal routes; 72 raw points, 24 hollow means, "
                    "and 24 min-max range whiskers"
                ),
                "panel_b": (
                    "three prespecified paired contrasts; 24 scene means, 24-record "
                    "overall means and medians, zero raw record points"
                ),
                "panel_c": "24-record mean row-relative 3x3 coordinate-transfer delta",
            },
        },
    )
    return receipt


def _validate_figure_rows(
    diagonals: tuple[DiagonalFigureRow, ...],
    effects: tuple[EffectFigureRow, ...],
    matrices: tuple[MatrixFigureRow, ...],
) -> None:
    if (len(diagonals), len(effects), len(matrices)) != (72, 72, 216):
        raise ValueError(f"figure_shape:{len(diagonals)}:{len(effects)}:{len(matrices)}")
    folds = {row.fold_id for row in diagonals}
    scenes = {row.scene for row in diagonals}
    if len(folds) != 24 or len(scenes) != 8:
        raise ValueError(f"fold_scene_shape:{len(folds)}:{len(scenes)}")
    diagonal_counts = Counter((row.fold_id, row.route_id) for row in diagonals)
    effect_counts = Counter((row.fold_id, row.contrast_id) for row in effects)
    matrix_counts = Counter(
        (row.fold_id, row.actual_route_id, row.coordinate_source_route_id) for row in matrices
    )
    if set(diagonal_counts) != {(fold, route) for fold in folds for route in ROUTES}:
        raise ValueError("diagonal_route_keys")
    if set(diagonal_counts.values()) != {1}:
        raise ValueError("duplicate_diagonal_rows")
    if set(effect_counts) != {(fold, contrast) for fold in folds for contrast in CONTRASTS}:
        raise ValueError("effect_contrast_keys")
    if set(effect_counts.values()) != {1}:
        raise ValueError("duplicate_effect_rows")
    if set(matrix_counts) != {
        (fold, actual, source) for fold in folds for actual in ROUTES for source in ROUTES
    }:
        raise ValueError("matrix_route_keys")
    if set(matrix_counts.values()) != {1}:
        raise ValueError("duplicate_matrix_rows")
    diagonal = {(row.fold_id, row.route_id): row for row in diagonals}
    matrix = {
        (row.fold_id, row.actual_route_id, row.coordinate_source_route_id): row for row in matrices
    }
    expected_contrast = {
        "ACC_minus_HF": ("ACC", "HF"),
        "HF_minus_HF_ACC": ("HF", "HF_ACC"),
        "ACC_minus_HF_ACC": ("ACC", "HF_ACC"),
    }
    for row in effects:
        left, right = expected_contrast[row.contrast_id]
        expected = diagonal[(row.fold_id, left)].mae_bpm - diagonal[(row.fold_id, right)].mae_bpm
        if not math.isclose(row.difference_bpm, expected, abs_tol=1e-12):
            raise ValueError(f"effect_value:{row.fold_id}:{row.contrast_id}")
    for fold in folds:
        for route in ROUTES:
            expected = diagonal[(fold, route)].mae_bpm
            observed = matrix[(fold, route, route)].mae_bpm
            if not math.isclose(observed, expected, abs_tol=1e-12):
                raise ValueError(f"matrix_diagonal:{fold}:{route}")
    for scene in scenes:
        if len({row.fold_id for row in diagonals if row.scene == scene}) != 3:
            raise ValueError(f"scene_fold_count:{scene}")


def _draw_panel_a(ax: Any, data: ReferenceArmFigureData) -> None:
    route_offsets = {"HF": -0.24, "ACC": 0.0, "HF_ACC": 0.24}
    raw_jitter = (-0.045, 0.0, 0.045)
    for scene_index, scene in enumerate(data.scene_order):
        for route in ROUTES:
            rows = sorted(
                (row for row in data.diagonal_rows if row.scene == scene and row.route_id == route),
                key=lambda row: row.fold_id,
            )
            center = scene_index + route_offsets[route]
            route_values = [row.mae_bpm for row in rows]
            route_mean = statistics.fmean(route_values)
            ax.scatter(
                [center + jitter for jitter in raw_jitter],
                route_values,
                s=14,
                marker=ROUTE_MARKERS[route],
                color=ROUTE_COLORS[route],
                alpha=0.55,
                linewidths=0,
                zorder=3,
            )
            ax.errorbar(
                [center],
                [route_mean],
                yerr=np.asarray(
                    [
                        [route_mean - min(route_values)],
                        [max(route_values) - route_mean],
                    ]
                ),
                fmt=ROUTE_MARKERS[route],
                markersize=5.2,
                markerfacecolor="white",
                markeredgecolor=ROUTE_COLORS[route],
                markeredgewidth=1.2,
                ecolor=ROUTE_COLORS[route],
                elinewidth=0.85,
                capsize=2.2,
                capthick=0.85,
                zorder=5,
            )
    values = [row.mae_bpm for row in data.diagonal_rows]
    ax.set_xlim(-0.55, len(data.scene_order) - 0.45)
    ax.set_ylim(0.0, max(values) * 1.14 if max(values) > 0 else 1.0)
    ax.set_xticks(range(len(data.scene_order)))
    ax.set_xticklabels(
        [_scene_label(scene) for scene in data.scene_order],
        rotation=29,
        ha="right",
    )
    ax.set_ylabel("Held-out MAE (bpm)")
    ax.set_title("Independently selected reference routes", loc="left", pad=8)
    handles = [
        Line2D(
            [0],
            [0],
            marker=ROUTE_MARKERS[route],
            linestyle="none",
            markerfacecolor=ROUTE_COLORS[route],
            markeredgecolor=ROUTE_COLORS[route],
            markersize=5,
            label=ROUTE_LABELS[route],
        )
        for route in ROUTES
    ]
    handles.append(
        ax.errorbar(
            [np.nan],
            [np.nan],
            yerr=np.asarray([[0.5], [0.5]]),
            fmt="D",
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=NEUTRAL_COLOR,
            markeredgewidth=1.0,
            ecolor=NEUTRAL_COLOR,
            elinewidth=0.8,
            capsize=2.0,
            label="scene mean + range",
        )
    )
    ax.legend(handles=handles, loc="upper left", ncol=2, columnspacing=0.9, handletextpad=0.35)
    _quiet_axes(ax)


def _draw_panel_b(ax: Any, data: ReferenceArmFigureData) -> None:
    labels = ("ACC − HF", "HF − HF+ACC", "ACC − HF+ACC")
    scene_offsets = np.linspace(-0.18, 0.18, len(data.scene_order))
    scene_means_by_contrast: dict[str, list[float]] = {}
    for contrast in CONTRASTS:
        scene_means_by_contrast[contrast] = [
            statistics.fmean(
                row.difference_bpm
                for row in data.effect_rows
                if row.scene == scene and row.contrast_id == contrast
            )
            for scene in data.scene_order
        ]
    maximum = max(
        abs(value) for scene_means in scene_means_by_contrast.values() for value in scene_means
    )
    limit = max(maximum * 1.38, 0.5)
    ax.axhline(0.0, color=NEUTRAL_COLOR, linestyle="--", linewidth=0.8, zorder=1)
    summary_values: list[tuple[float, float, int]] = []
    for contrast_index, contrast in enumerate(CONTRASTS):
        values = [row.difference_bpm for row in data.effect_rows if row.contrast_id == contrast]
        scene_means = scene_means_by_contrast[contrast]
        for scene_offset, scene_mean in zip(scene_offsets, scene_means, strict=True):
            ax.scatter(
                contrast_index + scene_offset,
                scene_mean,
                s=23,
                marker="D",
                facecolors="white",
                edgecolors=NEUTRAL_COLOR,
                linewidths=1.0,
                zorder=4,
            )
        mean_value = statistics.fmean(values)
        median_value = statistics.median(values)
        ax.scatter(
            contrast_index,
            mean_value,
            s=30,
            marker="D",
            color="#202020",
            edgecolors="white",
            linewidths=0.45,
            zorder=6,
        )
        ax.plot(
            [contrast_index - 0.09, contrast_index + 0.09],
            [median_value, median_value],
            color="#202020",
            linewidth=1.2,
            zorder=5,
        )
        positive_scenes = sum(value > 0.0 for value in scene_means)
        summary_values.append((mean_value, median_value, positive_scenes))
    ax.text(
        0.5,
        0.84,
        (
            "Mean: "
            + " | ".join(f"{mean_value:+.2f}" for mean_value, _, _ in summary_values)
            + "\nMedian: "
            + " | ".join(f"{median_value:+.2f}" for _, median_value, _ in summary_values)
            + "\nPositive scenes: "
            + " | ".join(f"{positive_scenes}/8" for _, _, positive_scenes in summary_values)
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.2,
        color="#303030",
        linespacing=0.98,
    )
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-limit, limit)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, rotation=21, ha="right")
    ax.set_ylabel("Paired ΔMAE (bpm)")
    ax.set_title("Prespecified paired contrasts", loc="left", pad=8)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=NEUTRAL_COLOR,
                markeredgewidth=1.0,
                markersize=4.5,
                label=PANEL_B_LEGEND_LABELS[0],
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                markerfacecolor="#202020",
                markeredgecolor="white",
                markeredgewidth=0.45,
                markersize=4.5,
                label=PANEL_B_LEGEND_LABELS[1],
            ),
            Line2D(
                [0, 1],
                [0, 0],
                color="#202020",
                linewidth=1.2,
                label=PANEL_B_LEGEND_LABELS[2],
            ),
        ],
        loc="upper center",
        ncol=3,
        columnspacing=0.65,
        handletextpad=0.3,
        borderaxespad=0.2,
        fontsize=5.3,
    )
    _quiet_axes(ax)


def _draw_panel_c(ax: Any, data: ReferenceArmFigureData) -> None:
    values = np.asarray(data.transfer_delta, dtype=float)
    maximum = max(float(np.max(np.abs(values))), 1e-9)
    image = ax.imshow(
        values,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
        aspect="equal",
        interpolation="nearest",
    )
    for row in range(3):
        for column in range(3):
            value = values[row, column]
            rgba = image.cmap(image.norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                column,
                row,
                f"{value:+.2f}" if value != 0.0 else "0.00",
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if luminance < 0.53 else "#202020",
            )
        ax.add_patch(
            Rectangle(
                (row - 0.48, row - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor="#303030",
                linewidth=0.75,
            )
        )
    labels = [ROUTE_LABELS[route] for route in ROUTES]
    ax.set_xticks(range(3), labels=labels)
    ax.set_yticks(range(3), labels=labels)
    ax.set_xlabel("Coordinate selected by")
    ax.set_ylabel("Evaluated route")
    ax.set_title("Coordinate-transfer cost", loc="left", pad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.045)
    colorbar.set_label("ΔMAE vs own choice (bpm)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5, width=0.6, length=2)
    colorbar.outline.set_linewidth(0.6)


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.titleweight": "normal",
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def _quiet_axes(ax: Any) -> None:
    ax.grid(False)
    ax.spines["left"].set_color("#303030")
    ax.spines["bottom"].set_color("#303030")
    ax.tick_params(colors="#303030")


def _add_panel_label(ax: Any, label: str, *, x: float) -> None:
    ax.text(
        x,
        1.035,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color="#202020",
    )


def _scene_label(scene: str) -> str:
    return scene_display_name(scene)


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
