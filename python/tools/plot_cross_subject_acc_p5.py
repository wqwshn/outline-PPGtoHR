"""Render the frozen three-panel cross-subject HF-ACC comparison as 600 dpi PNG."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from ppg_hr.v2.cross_subject_acc_experiment import ACC_EXPERIMENT_ID

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

SCENE_ORDER = (
    "tiaosheng",
    "bobi",
    "run",
    "quanji",
    "jianpan",
    "xiezi",
    "kaihe",
    "woli",
)
SCENE_DISPLAY = {
    "tiaosheng": "Rope Skipping",
    "bobi": "Burpees",
    "run": "Running",
    "quanji": "Punching",
    "jianpan": "Typing",
    "xiezi": "Handwriting",
    "kaihe": "Jumping Jacks",
    "woli": "Handgrip",
}
SUBJECT_ORDER = ("CGX", "HB", "LYX", "LZJ", "PJY", "QYC", "TS")
SUBJECT_COLORS = {
    "CGX": "#4C78A8",
    "HB": "#9D755D",
    "LYX": "#F2A541",
    "LZJ": "#59A14F",
    "PJY": "#B279A2",
    "QYC": "#76B7B2",
    "TS": "#E15759",
}
HF_COLOR = "#D97824"
ACC_COLOR = "#3F7CAC"
NEUTRAL = "#6D6D6D"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--experiment-root", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    experiment_root = (
        args.experiment_root.resolve()
        if args.experiment_root
        else repo_root / "data" / "experiments" / ACC_EXPERIMENT_ID
    )
    p4_root = experiment_root / "p4"
    p5_root = experiment_root / "p5"
    figure_root = p5_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    validation = _read_json(p5_root / "p5_validation_receipt.json")
    if validation.get("status") != "pass":
        raise ValueError("p5_validation_not_pass")
    record_rows = _read_csv(p4_root / "paired_record_results.csv")
    fold_rows = _read_csv(p4_root / "paired_fold_results.csv")
    matrix_rows = _read_csv(p4_root / "paired_2x2_matrix.csv")
    if len(record_rows) != 143 or len(fold_rows) != 48 or len(matrix_rows) != 4:
        raise ValueError("p5_figure_source_count_mismatch")
    if {row["scene"] for row in record_rows} != set(SCENE_ORDER):
        raise ValueError("p5_figure_scene_set_mismatch")

    panel_a_rows = _panel_a_source(record_rows, fold_rows)
    panel_b_rows = _panel_b_source(fold_rows)
    panel_c_rows = _panel_c_source(matrix_rows)
    source_hashes = {
        "figure_panel_a_source.csv": _write_csv(
            figure_root / "figure_panel_a_source.csv", panel_a_rows
        ),
        "figure_panel_b_source.csv": _write_csv(
            figure_root / "figure_panel_b_source.csv", panel_b_rows
        ),
        "figure_panel_c_source.csv": _write_csv(
            figure_root / "figure_panel_c_source.csv", panel_c_rows
        ),
    }
    contract = {
        "schema_id": "cross_subject_acc_main_figure_contract_v1",
        "core_conclusion": (
            "Objectively display the paired HF-ACC end-to-end difference on identical "
            "common support and decompose reference-route versus coordinate-source effects."
        ),
        "figure_archetype": "quantitative_grid",
        "backend": "python_matplotlib",
        "output": "600_dpi_png_only",
        "final_size_inches": [10.8, 7.2],
        "panel_map": {
            "a": "143 paired records and 48 paired fold means across eight frozen scenes",
            "b": "48 end-to-end fold differences with subject identity, median, and IQR",
            "c": "48-fold-equal 2x2 route-by-coordinate-source mean MAE",
        },
        "statistics": "descriptive_only_no_independent_fold_significance_test",
        "scene_order": [SCENE_DISPLAY[scene] for scene in SCENE_ORDER],
        "review_risks": [
            "Forty-eight folds are repeated scene-level evaluations of seven people.",
            "All four cells must use each record's exact common reliable-window support.",
            "Off-diagonal cells explain mismatch and are not additional selected routes.",
        ],
    }
    contract_sha = _write_json(figure_root / "figure_contract.json", contract)
    output_path = figure_root / "cross_subject_acc_four_cell_comparison_600dpi.png"
    _render(record_rows, fold_rows, matrix_rows, output_path)
    _validate_png(output_path)
    receipt = {
        "schema_id": "cross_subject_acc_p5_figure_receipt_v1",
        "experiment_id": ACC_EXPERIMENT_ID,
        "status": "pass",
        "backend": "python_matplotlib",
        "dpi": 600,
        "format": "PNG",
        "record_count": len(record_rows),
        "fold_count": len(fold_rows),
        "scene_count": len(SCENE_ORDER),
        "figure_file": output_path.name,
        "figure_sha256": _file_sha256(output_path),
        "figure_contract_sha256": contract_sha,
        "source_data_sha256": source_hashes,
        "validation_receipt_sha256": _file_sha256(p5_root / "p5_validation_receipt.json"),
    }
    receipt_sha = _write_json(figure_root / "figure_receipt.json", receipt)
    print(
        json.dumps(
            {**receipt, "receipt_sha256": receipt_sha},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


def _render(
    records: list[dict[str, str]],
    folds: list[dict[str, str]],
    matrix_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(10.8, 7.2), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1.28, 1.0),
        width_ratios=(1.0, 1.0, 0.72),
        hspace=0.47,
        wspace=0.38,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, :2])
    ax_c = fig.add_subplot(grid[1, 2])
    _draw_panel_a(ax_a, records, folds)
    _draw_panel_b(ax_b, folds)
    _draw_panel_c(ax_c, matrix_rows)
    _panel_label(ax_a, "a")
    _panel_label(ax_b, "b")
    _panel_label(ax_c, "c")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.96, bottom=0.15)
    fig.savefig(
        output_path,
        dpi=600,
        format="png",
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)


def _draw_panel_a(ax: Any, records: list[dict[str, str]], folds: list[dict[str, str]]) -> None:
    by_scene_records = defaultdict(list)
    by_scene_folds = defaultdict(list)
    for row in records:
        by_scene_records[row["scene"]].append(row)
    for row in folds:
        by_scene_folds[row["scene"]].append(row)
    route_specs = (
        ("hf_theta_hf_mae_bpm", -0.18, HF_COLOR),
        ("acc_theta_acc_mae_bpm", 0.18, ACC_COLOR),
    )
    for scene_index, scene in enumerate(SCENE_ORDER):
        scene_records = by_scene_records[scene]
        for column, offset, color in route_specs:
            values = np.asarray([float(row[column]) for row in scene_records])
            violin = ax.violinplot(
                values,
                positions=[scene_index + offset],
                widths=0.28,
                showextrema=False,
                showmeans=False,
                showmedians=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.16)
                body.set_linewidth(0.65)
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            ax.plot(
                [scene_index + offset, scene_index + offset],
                [q1, q3],
                color=color,
                linewidth=2.2,
                solid_capstyle="round",
                zorder=4,
            )
            ax.plot(
                [scene_index + offset - 0.035, scene_index + offset + 0.035],
                [med, med],
                color="white",
                linewidth=1.0,
                zorder=5,
            )
        for row in scene_records:
            jitter = _stable_jitter(row["record_id"], 0.045)
            x_hf = scene_index - 0.18 + jitter
            x_acc = scene_index + 0.18 + jitter
            y_hf = float(row["hf_theta_hf_mae_bpm"])
            y_acc = float(row["acc_theta_acc_mae_bpm"])
            ax.plot([x_hf, x_acc], [y_hf, y_acc], color="#9B9B9B", lw=0.35, alpha=0.28)
            ax.scatter(x_hf, y_hf, s=7, color=HF_COLOR, alpha=0.58, linewidths=0, zorder=3)
            ax.scatter(x_acc, y_acc, s=7, color=ACC_COLOR, alpha=0.58, linewidths=0, zorder=3)
        for fold_index, row in enumerate(
            sorted(by_scene_folds[scene], key=lambda value: value["holdout_subject_id"])
        ):
            jitter = np.linspace(-0.065, 0.065, 6)[fold_index]
            x_hf = scene_index - 0.18 + jitter
            x_acc = scene_index + 0.18 + jitter
            y_hf = float(row["hf_theta_hf_mae_bpm"])
            y_acc = float(row["acc_theta_acc_mae_bpm"])
            ax.plot([x_hf, x_acc], [y_hf, y_acc], color="#4C4C4C", lw=0.58, alpha=0.68)
            ax.scatter(
                [x_hf, x_acc],
                [y_hf, y_acc],
                s=24,
                c=[HF_COLOR, ACC_COLOR],
                edgecolors="#252525",
                linewidths=0.55,
                zorder=6,
            )
    ax.set_xlim(-0.55, len(SCENE_ORDER) - 0.45)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("MAE (bpm)")
    ax.set_xticks(range(len(SCENE_ORDER)))
    ax.set_xticklabels([SCENE_DISPLAY[scene] for scene in SCENE_ORDER], rotation=22, ha="right")
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    legend = (
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=HF_COLOR,
            markersize=4,
            label="HF($\\theta_{HF}$)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ACC_COLOR,
            markersize=4,
            label="ACC($\\theta_{ACC}$)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#252525",
            markersize=4.8,
            label="Fold mean",
        ),
    )
    ax.legend(handles=legend, loc="upper left", ncol=3, handletextpad=0.35, columnspacing=1.0)


def _draw_panel_b(ax: Any, folds: list[dict[str, str]]) -> None:
    by_scene = defaultdict(list)
    for row in folds:
        by_scene[row["scene"]].append(row)
    subject_offsets = dict(
        zip(SUBJECT_ORDER, np.linspace(-0.18, 0.18, len(SUBJECT_ORDER)), strict=True)
    )
    all_values = []
    for scene_index, scene in enumerate(SCENE_ORDER):
        rows = by_scene[scene]
        values = np.asarray([float(row["delta_end_bpm"]) for row in rows])
        all_values.extend(values.tolist())
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        ax.add_patch(
            Rectangle(
                (scene_index - 0.27, q1),
                0.54,
                q3 - q1,
                facecolor="#D8D8D8",
                edgecolor="#8A8A8A",
                linewidth=0.55,
                alpha=0.55,
                zorder=1,
            )
        )
        ax.plot(
            [scene_index - 0.27, scene_index + 0.27],
            [med, med],
            color="#3A3A3A",
            linewidth=1.15,
            zorder=2,
        )
        for row in rows:
            subject = row["holdout_subject_id"]
            ax.scatter(
                scene_index + subject_offsets[subject],
                float(row["delta_end_bpm"]),
                s=28,
                color=SUBJECT_COLORS[subject],
                edgecolors="white",
                linewidths=0.45,
                zorder=4,
            )
    limit = max(abs(min(all_values)), abs(max(all_values))) * 1.10
    if limit == 0:
        limit = 1.0
    ax.axhline(0, color="#4F4F4F", linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
    ax.set_ylim(-limit, limit)
    ax.set_xlim(-0.55, len(SCENE_ORDER) - 0.45)
    ax.set_ylabel("ACC($\\theta_{ACC}$) − HF($\\theta_{HF}$) MAE (bpm)")
    ax.set_xticks(range(len(SCENE_ORDER)))
    ax.set_xticklabels([SCENE_DISPLAY[scene] for scene in SCENE_ORDER], rotation=25, ha="right")
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SUBJECT_COLORS[subject],
            markeredgecolor="white",
            markersize=4.2,
            label=subject,
        )
        for subject in SUBJECT_ORDER
    ]
    handles.append(Patch(facecolor="#D8D8D8", edgecolor="#8A8A8A", label="IQR / median"))
    ax.legend(
        handles=handles,
        loc="upper left",
        ncol=4,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.2,
    )


def _draw_panel_c(ax: Any, matrix_rows: list[dict[str, str]]) -> None:
    by_label = {row["cell_label"]: row for row in matrix_rows}
    matrix = np.asarray(
        [
            [
                float(by_label["hf_theta_hf"]["mae_bpm__mean"]),
                float(by_label["hf_theta_acc"]["mae_bpm__mean"]),
            ],
            [
                float(by_label["acc_theta_hf"]["mae_bpm__mean"]),
                float(by_label["acc_theta_acc"]["mae_bpm__mean"]),
            ],
        ]
    )
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "neutral_blue", ("#F5F2EC", "#AFC1D2", "#536F89")
    )
    image = ax.imshow(matrix, cmap=cmap, aspect="equal")
    norm = image.norm
    for row_index in range(2):
        for column_index in range(2):
            value = matrix[row_index, column_index]
            color = "white" if norm(value) > 0.58 else "#202020"
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight="bold",
                color=color,
            )
    ax.set_xticks((0, 1), ("$\\theta_{HF}$", "$\\theta_{ACC}$"))
    ax.set_yticks((0, 1), ("HF", "ACC"))
    ax.set_xlabel("Parameter source")
    ax.set_ylabel("Reference route")
    ax.tick_params(which="both", length=0)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _panel_a_source(
    records: list[dict[str, str]], folds: list[dict[str, str]]
) -> list[dict[str, Any]]:
    output = []
    for level, rows, subject_column in (
        ("record", records, "physical_subject_id"),
        ("fold", folds, "holdout_subject_id"),
    ):
        for row in rows:
            for route, column in (
                ("HF(theta_HF)", "hf_theta_hf_mae_bpm"),
                ("ACC(theta_ACC)", "acc_theta_acc_mae_bpm"),
            ):
                output.append(
                    {
                        "level": level,
                        "scene": row["scene"],
                        "scene_display": SCENE_DISPLAY[row["scene"]],
                        "fold_id": row["fold_id"],
                        "subject_id": row[subject_column],
                        "record_id": row.get("record_id", ""),
                        "route": route,
                        "mae_bpm": row[column],
                    }
                )
    return output


def _panel_b_source(folds: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "scene": row["scene"],
            "scene_display": SCENE_DISPLAY[row["scene"]],
            "fold_id": row["fold_id"],
            "holdout_subject_id": row["holdout_subject_id"],
            "delta_end_bpm": row["delta_end_bpm"],
        }
        for row in folds
    ]


def _panel_c_source(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "route_id": row["route_id"],
            "coordinate_source": row["coordinate_source"],
            "cell_label": row["cell_label"],
            "fold_equal_mean_mae_bpm": row["mae_bpm__mean"],
        }
        for row in rows
    ]


def _stable_jitter(value: str, width: float) -> float:
    integer = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)
    return ((integer / 0xFFFFFFFF) * 2.0 - 1.0) * width


def _panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.055,
        1.025,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def _validate_png(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as image:
        if image.format != "PNG" or image.width < 5000 or image.height < 3000:
            raise ValueError(f"p5_png_contract:{image.format}:{image.size}")
        dpi = image.info.get("dpi")
        if dpi is None or min(float(value) for value in dpi) < 599.0:
            raise ValueError(f"p5_png_dpi:{dpi}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("p5_empty_figure_source")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return _file_sha256(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
