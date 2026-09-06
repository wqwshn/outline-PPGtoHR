from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle

SCENE_ORDER = ("bobi", "jianpan", "kaihe", "quanji", "run", "tiaosheng", "woli", "xiezi")
SUBJECT_ORDER = ("CGX", "LYX", "LZJ", "PJY", "QYC", "TS", "HB")
GATE_COLUMNS = (
    ("G1-I", "g1i_pass"),
    ("G2", "g2_pass"),
    ("G3", "g3_pass"),
    ("G4", "g4_pass"),
    ("G5", "g5_pass"),
    ("G7", "g7_pass"),
    ("All six", "qualified"),
)
FAILURE_GATE_ORDER = ("G1-I", "G2", "G3", "G4", "G5", "G7")
FIGURE_WIDTH_IN = 7.2
EXPORT_DPI = 600

COLORS = {
    "candidate": "#D55E00",
    "candidate_dark": "#9C3D00",
    "baseline": "#666666",
    "improved": "#2878A5",
    "text": "#222222",
    "grid": "#D8D8D8",
    "missing": "#ECECEC",
}

_COORDINATE_RE = re.compile(
    r"^physical4d:fs(?P<fs>\d+):m(?P<memory>\d+):mu(?P<mu>\d+):w(?P<width>\d+)$"
)


@dataclass(frozen=True)
class PublicationTables:
    fold_pairs: pd.DataFrame
    record_deltas: pd.DataFrame
    fold_mae_matrix: pd.DataFrame
    training_fraction_matrix: pd.DataFrame
    gate_summary: pd.DataFrame
    coordinate_scene_counts: pd.DataFrame
    coordinate_parameters: pd.DataFrame
    coordinate_labels: dict[str, str]
    failure_combinations: pd.DataFrame


def parse_coordinate_id(coordinate_id: str) -> dict[str, int | float]:
    match = _COORDINATE_RE.fullmatch(str(coordinate_id))
    if match is None:
        raise ValueError(f"Unsupported Physical4D coordinate id: {coordinate_id!r}")
    return {
        "fs_target_hz": int(match.group("fs")),
        "memory_ms": int(match.group("memory")),
        "mu_base": int(match.group("mu")) / 1000,
        "exclusion_half_width_bpm": int(match.group("width")),
    }


def _ordered(existing: Iterable[str], preferred: tuple[str, ...]) -> list[str]:
    values = {str(value) for value in existing}
    return [value for value in preferred if value in values] + sorted(values.difference(preferred))


def _fraction_to_float(value: Any) -> float:
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(text)


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    unknown = sorted(set(normalized).difference(mapping))
    if unknown:
        raise ValueError(f"Unsupported boolean values: {unknown}")
    return normalized.map(mapping).astype(bool)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _canonical_failure_combination(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        decoded = json.loads(raw)
    elif isinstance(raw, (list, tuple)):
        decoded = raw
    else:
        raise ValueError(f"Unsupported failed-gate value: {raw!r}")
    decoded_set = {str(item) for item in decoded}
    unknown = decoded_set.difference(FAILURE_GATE_ORDER)
    if unknown:
        raise ValueError(f"Unsupported failed gates: {sorted(unknown)}")
    return tuple(gate for gate in FAILURE_GATE_ORDER if gate in decoded_set)


def build_publication_tables(folds: pd.DataFrame, records: pd.DataFrame) -> PublicationTables:
    _require_columns(
        folds,
        (
            "fold_id",
            "scene",
            "holdout_subject_id",
            "selected_coordinate_id",
            "minimum_training_subject_pass_fraction",
            "candidate_mean_mae_bpm",
            "baseline_mean_mae_bpm",
        ),
        "fold table",
    )
    _require_columns(
        records,
        (
            "fold_id",
            "scene",
            "holdout_subject_id",
            "record_id",
            "candidate_mae_bpm",
            "baseline_mae_bpm",
            "mae_delta_vs_baseline_bpm",
            "failed_gates_json",
            *(column for _, column in GATE_COLUMNS),
        ),
        "record table",
    )

    fold_pairs = folds.copy()
    record_deltas = records.copy()
    for column in ("candidate_mean_mae_bpm", "baseline_mean_mae_bpm"):
        fold_pairs[column] = pd.to_numeric(fold_pairs[column], errors="raise")
    for column in ("candidate_mae_bpm", "baseline_mae_bpm", "mae_delta_vs_baseline_bpm"):
        record_deltas[column] = pd.to_numeric(record_deltas[column], errors="raise")
    fold_pairs["minimum_training_subject_pass_fraction_numeric"] = fold_pairs[
        "minimum_training_subject_pass_fraction"
    ].map(_fraction_to_float)
    for _, column in GATE_COLUMNS:
        record_deltas[column] = _boolean_series(record_deltas[column])

    scenes = _ordered(fold_pairs["scene"], SCENE_ORDER)
    subjects = _ordered(fold_pairs["holdout_subject_id"], SUBJECT_ORDER)
    fold_mae_matrix = fold_pairs.pivot(
        index="holdout_subject_id", columns="scene", values="candidate_mean_mae_bpm"
    ).reindex(index=subjects, columns=scenes)
    training_fraction_matrix = fold_pairs.pivot(
        index="holdout_subject_id",
        columns="scene",
        values="minimum_training_subject_pass_fraction_numeric",
    ).reindex(index=subjects, columns=scenes)

    gate_rows = []
    for label, column in GATE_COLUMNS:
        passed = int(record_deltas[column].sum())
        gate_rows.append(
            {
                "gate": label,
                "passed_records": passed,
                "total_records": len(record_deltas),
                "pass_fraction": passed / len(record_deltas),
            }
        )
    gate_summary = pd.DataFrame(gate_rows).set_index("gate")

    frequencies = Counter(fold_pairs["selected_coordinate_id"].astype(str))
    coordinate_ids = sorted(frequencies, key=lambda item: (-frequencies[item], item))
    coordinate_labels = {
        coordinate_id: f"C{index:02d}" for index, coordinate_id in enumerate(coordinate_ids, 1)
    }
    coordinate_scene_counts = pd.crosstab(
        fold_pairs["scene"], fold_pairs["selected_coordinate_id"]
    ).reindex(index=scenes, columns=coordinate_ids, fill_value=0)

    parameter_rows: dict[str, list[int | float]] = {
        "fs_target_hz": [],
        "memory_ms": [],
        "mu_base": [],
        "exclusion_half_width_bpm": [],
    }
    for coordinate_id in coordinate_ids:
        parsed = parse_coordinate_id(coordinate_id)
        for key in parameter_rows:
            parameter_rows[key].append(parsed[key])
    coordinate_parameters = pd.DataFrame(parameter_rows, index=coordinate_ids).T

    failed_combinations = [
        _canonical_failure_combination(value) for value in record_deltas["failed_gates_json"]
    ]
    combination_counts = Counter(combo for combo in failed_combinations if combo)
    failure_rows = [
        {"failed_gates": combo, "count": count}
        for combo, count in sorted(combination_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    failure_combinations = pd.DataFrame(failure_rows, columns=("failed_gates", "count"))

    return PublicationTables(
        fold_pairs=fold_pairs,
        record_deltas=record_deltas,
        fold_mae_matrix=fold_mae_matrix,
        training_fraction_matrix=training_fraction_matrix,
        gate_summary=gate_summary,
        coordinate_scene_counts=coordinate_scene_counts,
        coordinate_parameters=coordinate_parameters,
        coordinate_labels=coordinate_labels,
        failure_combinations=failure_combinations,
    )


def validate_experiment_contract(tables: PublicationTables) -> dict[str, int]:
    checks = {
        "fold_count": len(tables.fold_pairs),
        "record_count": len(tables.record_deltas),
        "physical_subject_count": len(tables.fold_mae_matrix.index),
        "scene_count": len(tables.fold_mae_matrix.columns),
        "observed_subject_scene_cells": int(tables.fold_mae_matrix.notna().sum().sum()),
        "selected_coordinate_count": len(tables.coordinate_scene_counts.columns),
    }
    expected = {
        "fold_count": 48,
        "record_count": 143,
        "physical_subject_count": 7,
        "scene_count": 8,
        "observed_subject_scene_cells": 48,
        "selected_coordinate_count": 23,
    }
    if checks != expected:
        raise ValueError(
            f"Publication data contract mismatch: observed={checks}, expected={expected}"
        )
    return checks


def _publication_style() -> dict[str, Any]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.titlesize": 7.3,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.8,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "axes.edgecolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }


def _panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        clip_on=False,
    )


def _scene_labels(scenes: Iterable[str]) -> list[str]:
    return [str(scene).capitalize() for scene in scenes]


def _draw_missing_cells(ax: mpl.axes.Axes, matrix: pd.DataFrame) -> None:
    for row_index, column_index in np.argwhere(matrix.isna().to_numpy()):
        ax.add_patch(
            Rectangle(
                (column_index - 0.5, row_index - 0.5),
                1,
                1,
                facecolor=COLORS["missing"],
                edgecolor="#A8A8A8",
                hatch="////",
                linewidth=0.35,
            )
        )


def _annotate_heatmap(
    ax: mpl.axes.Axes,
    matrix: pd.DataFrame,
    formatter: Any,
    threshold: float,
) -> None:
    values = matrix.to_numpy(dtype=float)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if not np.isfinite(value):
                continue
            ax.text(
                column_index,
                row_index,
                formatter(value),
                ha="center",
                va="center",
                fontsize=4.7,
                color="white" if value >= threshold else COLORS["text"],
            )


def render_figure_1(tables: PublicationTables) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_style()):
        fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 6.15), layout="constrained")
        grid = fig.add_gridspec(2, 2, width_ratios=(1.62, 1.0), height_ratios=(1.02, 1.0))
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1])
        ax_c = fig.add_subplot(grid[1, 0])
        ax_d = fig.add_subplot(grid[1, 1])

        scenes = list(tables.fold_mae_matrix.columns)
        scene_x = {scene: index for index, scene in enumerate(scenes)}
        for scene in scenes:
            rows = tables.fold_pairs.loc[tables.fold_pairs["scene"] == scene].sort_values(
                "holdout_subject_id"
            )
            jitters = np.linspace(-0.045, 0.045, len(rows))
            for jitter, (_, row) in zip(jitters, rows.iterrows(), strict=True):
                center = scene_x[scene] + jitter
                baseline_x = center - 0.15
                candidate_x = center + 0.15
                baseline = float(row["baseline_mean_mae_bpm"])
                candidate = float(row["candidate_mean_mae_bpm"])
                ax_a.plot(
                    (baseline_x, candidate_x),
                    (baseline, candidate),
                    color="#B6B6B6",
                    linewidth=0.55,
                    alpha=0.85,
                    zorder=1,
                )
                ax_a.scatter(
                    baseline_x, baseline, s=11, color=COLORS["baseline"], marker="s", zorder=2
                )
                ax_a.scatter(
                    candidate_x, candidate, s=13, color=COLORS["candidate"], marker="o", zorder=3
                )
            ax_a.plot(
                (scene_x[scene] - 0.21, scene_x[scene] - 0.09),
                (rows["baseline_mean_mae_bpm"].mean(),) * 2,
                color="#111111",
                linewidth=1.5,
                solid_capstyle="butt",
            )
            ax_a.plot(
                (scene_x[scene] + 0.09, scene_x[scene] + 0.21),
                (rows["candidate_mean_mae_bpm"].mean(),) * 2,
                color="#111111",
                linewidth=1.5,
                solid_capstyle="butt",
            )
        ax_a.set_xticks(range(len(scenes)), _scene_labels(scenes), rotation=32, ha="right")
        ax_a.set_ylabel("Fold mean MAE (bpm)")
        ax_a.set_title("Held-out fold performance (n = 48 folds)", loc="left", pad=4)
        ax_a.grid(axis="y", color=COLORS["grid"], linewidth=0.45, zorder=0)
        ax_a.set_axisbelow(True)
        max_fold = float(
            tables.fold_pairs[["candidate_mean_mae_bpm", "baseline_mean_mae_bpm"]].max().max()
        )
        ax_a.set_ylim(0, max_fold * 1.13)
        baseline_handle = ax_a.scatter(
            [], [], s=13, color=COLORS["baseline"], marker="s", label="Historical Lite"
        )
        candidate_handle = ax_a.scatter(
            [], [], s=14, color=COLORS["candidate"], marker="o", label="Shared HF"
        )
        ax_a.legend(
            handles=(baseline_handle, candidate_handle), frameon=False, ncol=2, loc="upper left"
        )
        _panel_label(ax_a, "a")

        mae_matrix = tables.fold_mae_matrix
        finite = mae_matrix.to_numpy(dtype=float)
        cmap = LinearSegmentedColormap.from_list(
            "candidate_mae", ("#FFF7EC", "#FDBB84", "#D94701", "#7F2704")
        )
        cmap.set_bad(COLORS["missing"])
        ax_b.imshow(finite, cmap=cmap, vmin=0, vmax=np.nanmax(finite), aspect="auto")
        _draw_missing_cells(ax_b, mae_matrix)
        _annotate_heatmap(
            ax_b, mae_matrix, lambda value: f"{value:.1f}", threshold=np.nanmax(finite) * 0.57
        )
        ax_b.set_xticks(range(len(scenes)), _scene_labels(scenes), rotation=45, ha="right")
        ax_b.set_yticks(range(len(mae_matrix.index)), mae_matrix.index)
        ax_b.set_title("Shared-HF fold MAE (bpm)", loc="left", pad=4)
        for spine in ax_b.spines.values():
            spine.set_visible(False)
        _panel_label(ax_b, "b", x=-0.15)

        record_rows = tables.record_deltas.copy()
        point_positions: dict[int, float] = {}
        for scene in scenes:
            rows = record_rows.loc[record_rows["scene"] == scene].sort_values("record_id")
            jitters = np.linspace(-0.22, 0.22, len(rows))
            for jitter, (row_index, row) in zip(jitters, rows.iterrows(), strict=True):
                x = scene_x[scene] + jitter
                point_positions[int(row_index)] = x
                delta = float(row["mae_delta_vs_baseline_bpm"])
                ax_c.scatter(
                    x,
                    delta,
                    s=10,
                    color=COLORS["candidate"] if delta > 0 else COLORS["improved"],
                    alpha=0.72,
                    edgecolor="white",
                    linewidth=0.22,
                    zorder=3,
                )
        ax_c.axhline(0, color=COLORS["baseline"], linestyle=(0, (3, 2)), linewidth=0.75, zorder=1)
        ax_c.set_xticks(range(len(scenes)), _scene_labels(scenes), rotation=32, ha="right")
        ax_c.set_ylabel("Record MAE difference (bpm)\nShared HF − historical Lite")
        ax_c.set_title("Record-level differences (n = 143 records)", loc="left", pad=4)
        ax_c.grid(axis="y", color=COLORS["grid"], linewidth=0.45, zorder=0)
        ax_c.set_axisbelow(True)
        worst_rows = record_rows.nlargest(5, "candidate_mae_bpm")
        delta_min = float(record_rows["mae_delta_vs_baseline_bpm"].min())
        delta_max = float(record_rows["mae_delta_vs_baseline_bpm"].max())
        ax_c.set_ylim(min(-5.0, delta_min * 1.15), delta_max * 1.23)
        offsets = ((8, 8), (8, -16), (8, 12), (-8, -16), (-8, -14))
        for offset, (row_index, row) in zip(offsets, worst_rows.iterrows(), strict=True):
            x = point_positions[int(row_index)]
            y = float(row["mae_delta_vs_baseline_bpm"])
            ax_c.annotate(
                str(row["record_id"]),
                xy=(x, y),
                xytext=offset,
                textcoords="offset points",
                ha="left" if offset[0] > 0 else "right",
                va="bottom" if offset[1] > 0 else "top",
                fontsize=4.6,
                color=COLORS["text"],
                arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.4},
            )
        _panel_label(ax_c, "c")

        gate_summary = tables.gate_summary.reindex([label for label, _ in GATE_COLUMNS])
        gate_y = np.arange(len(gate_summary))
        percentages = gate_summary["pass_fraction"].to_numpy(dtype=float) * 100
        ax_d.hlines(gate_y, 0, percentages, color="#A8A8A8", linewidth=1.25)
        ax_d.scatter(percentages, gate_y, s=24, color=COLORS["candidate"], zorder=3)
        for y, percentage, passed in zip(
            gate_y, percentages, gate_summary["passed_records"], strict=True
        ):
            ax_d.text(
                min(percentage + 2.2, 97),
                y,
                f"{int(passed)}/143",
                va="center",
                ha="left" if percentage < 88 else "right",
                fontsize=5.4,
            )
        ax_d.set_yticks(gate_y, gate_summary.index)
        ax_d.set_xlim(0, 100)
        ax_d.set_xlabel("Passing records (%)")
        ax_d.set_title("Frozen gate coverage", loc="left", pad=4)
        ax_d.grid(axis="x", color=COLORS["grid"], linewidth=0.45)
        ax_d.invert_yaxis()
        _panel_label(ax_d, "d", x=-0.15)
        return fig


def _fraction_text(value: float) -> str:
    if np.isclose(value, 0):
        return "0"
    if np.isclose(value, 1 / 3):
        return "1/3"
    if np.isclose(value, 2 / 3):
        return "2/3"
    return f"{value:.2f}"


def render_figure_2(
    tables: PublicationTables, top_failure_combinations: int = 8
) -> mpl.figure.Figure:
    with mpl.rc_context(_publication_style()):
        fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 6.6), layout="constrained")
        outer = fig.add_gridspec(2, 1, height_ratios=(1.72, 1.0))
        upper = outer[0].subgridspec(1, 2, width_ratios=(1.03, 1.87))
        ax_a = fig.add_subplot(upper[0, 0])
        right = upper[0, 1].subgridspec(2, 1, height_ratios=(1.35, 0.92), hspace=0.08)
        ax_b = fig.add_subplot(right[0, 0])
        ax_c = fig.add_subplot(right[1, 0])
        lower = outer[1].subgridspec(1, 2, width_ratios=(0.85, 1.55), wspace=0.03)
        ax_d_bar = fig.add_subplot(lower[0, 0])
        ax_d_matrix = fig.add_subplot(lower[0, 1], sharey=ax_d_bar)

        fraction_matrix = tables.training_fraction_matrix
        fraction_values = fraction_matrix.to_numpy(dtype=float)
        categorical = np.full_like(fraction_values, np.nan)
        categorical[np.isclose(fraction_values, 0, equal_nan=False)] = 0
        categorical[np.isclose(fraction_values, 1 / 3, equal_nan=False)] = 1
        categorical[np.isclose(fraction_values, 2 / 3, equal_nan=False)] = 2
        fraction_cmap = ListedColormap(("#B95A4A", "#E2B45A", "#4F8588"))
        fraction_cmap.set_bad(COLORS["missing"])
        ax_a.imshow(categorical, cmap=fraction_cmap, vmin=-0.5, vmax=2.5, aspect="auto")
        _draw_missing_cells(ax_a, fraction_matrix)
        for row_index in range(fraction_values.shape[0]):
            for column_index in range(fraction_values.shape[1]):
                value = fraction_values[row_index, column_index]
                if np.isfinite(value):
                    ax_a.text(
                        column_index,
                        row_index,
                        _fraction_text(value),
                        ha="center",
                        va="center",
                        fontsize=5.0,
                        color="white" if np.isclose(value, 0) else COLORS["text"],
                    )
        scenes = list(fraction_matrix.columns)
        ax_a.set_xticks(range(len(scenes)), _scene_labels(scenes), rotation=45, ha="right")
        ax_a.set_yticks(range(len(fraction_matrix.index)), fraction_matrix.index)
        ax_a.set_title("Minimum training-subject pass fraction", loc="left", pad=4)
        for spine in ax_a.spines.values():
            spine.set_visible(False)
        _panel_label(ax_a, "a", x=-0.17, y=1.05)

        selection_counts = tables.coordinate_scene_counts
        selection_values = selection_counts.to_numpy(dtype=float)
        selection_cmap = LinearSegmentedColormap.from_list(
            "selection", ("#FFFFFF", "#F6C59F", "#D55E00")
        )
        ax_b.imshow(
            selection_values,
            cmap=selection_cmap,
            vmin=0,
            vmax=max(1, np.max(selection_values)),
            aspect="auto",
        )
        for row_index, column_index in np.argwhere(selection_values > 0):
            value = int(selection_values[row_index, column_index])
            ax_b.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=4.6,
                color="white"
                if value >= max(2, np.max(selection_values) * 0.6)
                else COLORS["text"],
            )
        ax_b.set_yticks(range(len(selection_counts.index)), _scene_labels(selection_counts.index))
        ax_b.set_xticks([])
        ax_b.set_title(
            "Scene-specific selection frequency across 23 coordinates", loc="left", pad=4
        )
        for spine in ax_b.spines.values():
            spine.set_visible(False)
        _panel_label(ax_b, "b", x=-0.10)

        parameters = tables.coordinate_parameters
        parameter_labels = {
            "fs_target_hz": "fs (Hz)",
            "memory_ms": "Memory (ms)",
            "mu_base": "mu",
            "exclusion_half_width_bpm": "Width (bpm)",
        }
        row_colormaps = (
            LinearSegmentedColormap.from_list("fs", ("#E8F1F7", "#2878A5")),
            LinearSegmentedColormap.from_list("memory", ("#FFF1E2", "#C45A11")),
            LinearSegmentedColormap.from_list("mu", ("#E9F5F2", "#368878")),
            LinearSegmentedColormap.from_list("width", ("#F1EBF5", "#7B5AA6")),
        )
        for row_index, (parameter_name, row_values) in enumerate(parameters.iterrows()):
            numeric = row_values.to_numpy(dtype=float)
            value_min = float(np.min(numeric))
            value_max = float(np.max(numeric))
            normalized = (
                np.zeros_like(numeric)
                if np.isclose(value_min, value_max)
                else (numeric - value_min) / (value_max - value_min)
            )
            for column_index, (value, norm_value) in enumerate(
                zip(numeric, normalized, strict=True)
            ):
                ax_c.add_patch(
                    Rectangle(
                        (column_index - 0.5, row_index - 0.5),
                        1,
                        1,
                        facecolor=row_colormaps[row_index](0.18 + 0.75 * norm_value),
                        edgecolor="white",
                        linewidth=0.35,
                    )
                )
                label = f"{value:.3f}" if parameter_name == "mu_base" else f"{value:g}"
                ax_c.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=3.9,
                    color="white" if norm_value > 0.62 else COLORS["text"],
                )
        coordinate_ids = list(parameters.columns)
        ax_c.set_xlim(-0.5, len(coordinate_ids) - 0.5)
        ax_c.set_ylim(len(parameters.index) - 0.5, -0.5)
        ax_c.set_yticks(
            range(len(parameters.index)), [parameter_labels[item] for item in parameters.index]
        )
        ax_c.set_xticks(
            range(len(coordinate_ids)),
            [tables.coordinate_labels[item] for item in coordinate_ids],
            rotation=90,
        )
        ax_c.tick_params(axis="x", pad=1)
        ax_c.set_title("Coordinate encoding (columns aligned with b)", loc="left", pad=3)
        for spine in ax_c.spines.values():
            spine.set_visible(False)
        _panel_label(ax_c, "c", x=-0.10)

        combinations = tables.failure_combinations.head(top_failure_combinations).copy()
        y_positions = np.arange(len(combinations))
        counts = combinations["count"].to_numpy(dtype=int)
        ax_d_bar.barh(y_positions, counts, color=COLORS["candidate"], height=0.58)
        for y, count in zip(y_positions, counts, strict=True):
            ax_d_bar.text(count + 0.4, y, str(count), va="center", ha="left", fontsize=5.5)
        ax_d_bar.set_yticks(y_positions, [f"#{index}" for index in range(1, len(combinations) + 1)])
        ax_d_bar.set_xlabel("Records")
        ax_d_bar.set_title("Leading failure combinations", loc="left", pad=4)
        ax_d_bar.grid(axis="x", color=COLORS["grid"], linewidth=0.45)
        ax_d_bar.set_axisbelow(True)
        ax_d_bar.invert_yaxis()
        ax_d_bar.set_xlim(0, max(counts) * 1.18 if len(counts) else 1)
        _panel_label(ax_d_bar, "d", x=-0.20)

        for y, combination in zip(y_positions, combinations["failed_gates"], strict=True):
            failed = set(combination)
            failed_positions = []
            for x, gate in enumerate(FAILURE_GATE_ORDER):
                is_failed = gate in failed
                ax_d_matrix.scatter(
                    x,
                    y,
                    s=22 if is_failed else 13,
                    facecolor=COLORS["candidate"] if is_failed else "#D9D9D9",
                    edgecolor="none",
                    zorder=3,
                )
                if is_failed:
                    failed_positions.append(x)
            if len(failed_positions) > 1:
                ax_d_matrix.plot(
                    (min(failed_positions), max(failed_positions)),
                    (y, y),
                    color=COLORS["candidate_dark"],
                    linewidth=0.75,
                    zorder=2,
                )
        ax_d_matrix.set_xticks(range(len(FAILURE_GATE_ORDER)), FAILURE_GATE_ORDER)
        ax_d_matrix.tick_params(axis="y", left=False, labelleft=False)
        ax_d_matrix.set_xlim(-0.55, len(FAILURE_GATE_ORDER) - 0.45)
        ax_d_matrix.set_title("Failed gates (orange = failed)", loc="left", pad=4)
        for spine in ax_d_matrix.spines.values():
            spine.set_visible(False)
        ax_d_matrix.grid(axis="x", color="#EEEEEE", linewidth=0.4)
        return fig


def audit_figure_layout(fig: mpl.figure.Figure, tolerance_px: float = 3.0) -> dict[str, Any]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    outside: list[str] = []
    for text_artist in fig.findobj(mpl.text.Text):
        if not text_artist.get_visible() or not text_artist.get_text().strip():
            continue
        bbox = text_artist.get_window_extent(renderer=renderer)
        if (
            bbox.x0 < canvas.x0 - tolerance_px
            or bbox.y0 < canvas.y0 - tolerance_px
            or bbox.x1 > canvas.x1 + tolerance_px
            or bbox.y1 > canvas.y1 + tolerance_px
        ):
            outside.append(text_artist.get_text())
    return {
        "canvas_width_px_at_display_dpi": int(round(canvas.width)),
        "canvas_height_px_at_display_dpi": int(round(canvas.height)),
        "outside_text_count": len(outside),
        "outside_text": outside,
        "passed": not outside,
    }


def save_publication_png(fig: mpl.figure.Figure, target: Path) -> Path:
    target = Path(target)
    if target.suffix.lower() != ".png":
        raise ValueError("Publication figures are currently restricted to PNG output")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        target,
        dpi=EXPORT_DPI,
        facecolor="white",
        edgecolor="none",
        transparent=False,
        bbox_inches=None,
        metadata={"Software": "Matplotlib; cross_subject_loso_figures.py"},
    )
    return target


def write_source_tables(tables: PublicationTables, output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exports = {
        "figure_1_fold_pairs.csv": tables.fold_pairs,
        "figure_1_record_deltas.csv": tables.record_deltas,
        "figure_1_gate_summary.csv": tables.gate_summary.reset_index(),
        "figure_2_training_fraction_matrix.csv": tables.training_fraction_matrix.reset_index(),
        "figure_2_coordinate_scene_counts.csv": tables.coordinate_scene_counts.reset_index(),
        "figure_2_coordinate_parameters.csv": tables.coordinate_parameters.reset_index(
            names="parameter"
        ),
        "figure_2_failure_combinations.csv": tables.failure_combinations.assign(
            failed_gates=lambda frame: frame["failed_gates"].map(json.dumps)
        ),
    }
    paths = []
    for filename, frame in exports.items():
        path = output_dir / filename
        frame.to_csv(path, index=False, encoding="utf-8")
        paths.append(path)
    coordinate_key = output_dir / "figure_2_coordinate_key.csv"
    pd.DataFrame(
        [
            {"coordinate_label": label, "coordinate_id": coordinate_id}
            for coordinate_id, label in tables.coordinate_labels.items()
        ]
    ).to_csv(coordinate_key, index=False, encoding="utf-8")
    paths.append(coordinate_key)
    return paths
