"""Render two matched-minimax cross-subject violin figures as 600 dpi PNG."""

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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ppg_hr.v2.cross_subject_matched_minimax import EXPERIMENT_ID, file_sha256

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]

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
HF_COLOR = "#D97824"
ACC_COLOR = "#3F7CAC"
HF_COLUMN = "hf_theta_hf_minimax_mae_bpm"
FIGURES = (
    {
        "figure_id": "independent_minimax",
        "acc_column": "acc_theta_acc_minimax_mae_bpm",
        "hf_label": r"HF ($\theta_{HF}^{MM}$)",
        "acc_label": r"ACC ($\theta_{ACC}^{MM}$)",
        "filename": "cross_subject_hf_acc_independent_minimax_violin_600dpi.png",
        "source_filename": "independent_minimax_violin_source.csv",
    },
    {
        "figure_id": "acc_replay_hf_minimax",
        "acc_column": "acc_theta_hf_minimax_mae_bpm",
        "hf_label": r"HF ($\theta_{HF}^{MM}$)",
        "acc_label": r"ACC ($\theta_{HF}^{MM}$)",
        "filename": "cross_subject_hf_acc_replay_hf_minimax_violin_600dpi.png",
        "source_filename": "acc_replay_hf_minimax_violin_source.csv",
    },
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--experiment-root", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = (
        args.experiment_root.resolve()
        if args.experiment_root
        else repo_root / "data" / "experiments" / EXPERIMENT_ID
    )
    p2_root = root / "p2"
    p3_root = root / "p3"
    figure_root = p3_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    validation_path = p3_root / "p3_validation_receipt.json"
    validation = _read_json(validation_path)
    if validation.get("status") != "pass":
        raise ValueError("matched_figure_validation_not_pass")
    records = _read_csv(p2_root / "matched_record_results.csv")
    folds = _read_csv(p2_root / "matched_fold_results.csv")
    if len(records) != 143 or len(folds) != 48:
        raise ValueError("matched_figure_source_count")
    if {row["scene"] for row in records} != set(SCENE_ORDER):
        raise ValueError("matched_figure_scene_set")

    all_values = [
        float(row[column])
        for row in records
        for column in (HF_COLUMN, *(spec["acc_column"] for spec in FIGURES))
    ]
    y_limit = float(np.ceil(max(all_values) / 5.0) * 5.0)
    source_hashes = {}
    figure_hashes = {}
    dimensions = {}
    for spec in FIGURES:
        source_rows = _figure_source(records, folds, spec)
        source_path = figure_root / spec["source_filename"]
        source_hashes[spec["source_filename"]] = _write_csv(source_path, source_rows)
        output_path = figure_root / spec["filename"]
        _render(records, folds, spec, y_limit, output_path)
        dimensions[spec["filename"]] = _validate_png(output_path)
        figure_hashes[spec["filename"]] = file_sha256(output_path)

    forbidden = sorted(
        path.name
        for path in figure_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".svg", ".tif", ".tiff"}
    )
    if forbidden:
        raise ValueError(f"matched_figure_forbidden_outputs:{forbidden}")
    png_files = sorted(path.name for path in figure_root.glob("*.png"))
    expected_png = sorted(spec["filename"] for spec in FIGURES)
    if png_files != expected_png:
        raise ValueError(f"matched_figure_png_set:{png_files}")

    contract = {
        "schema_id": "cross_subject_matched_minimax_figure_contract_v1",
        "core_conclusion": (
            "Under one training-side subject-balanced MAE minimax rule, objectively compare "
            "HF and ACC with independent coordinates, then compare the two references at the "
            "same HF-selected coordinate."
        ),
        "figure_archetype": "two_standalone_paired_violin_figures",
        "backend": "python_matplotlib",
        "output": "600_dpi_png_only",
        "final_size_inches_each": [10.8, 4.6],
        "shared_y_range_bpm": [0.0, y_limit],
        "scene_order": [SCENE_DISPLAY[scene] for scene in SCENE_ORDER],
        "visual_layers": [
            "143 paired record points and lines",
            "violin density with median and IQR",
            "48 fold means with black outlines",
        ],
        "statistics": "descriptive_only_no_independent_fold_significance_test",
        "text_policy": "axes_scene_labels_and_legend_only",
        "review_risks": [
            "The 48 folds are repeated scene-level evaluations of seven physical subjects.",
            "Both figures use the same five-cell record-level exact common support.",
            "The scene order is frozen from the parent HF report and is not reordered here.",
        ],
    }
    contract_sha = _write_json(figure_root / "figure_contract.json", contract)
    receipt = {
        "schema_id": "cross_subject_matched_minimax_figure_receipt_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass",
        "backend": "python_matplotlib",
        "dpi": 600,
        "format": "PNG",
        "record_count": len(records),
        "fold_count": len(folds),
        "scene_count": len(SCENE_ORDER),
        "shared_y_limit_bpm": y_limit,
        "figure_sha256": figure_hashes,
        "source_data_sha256": source_hashes,
        "image_dimensions_px": dimensions,
        "figure_contract_sha256": contract_sha,
        "validation_receipt_sha256": file_sha256(validation_path),
        "forbidden_output_files": forbidden,
    }
    sha = _write_json(figure_root / "figure_receipt.json", receipt)
    print(json.dumps({**receipt, "receipt_sha256": sha}, ensure_ascii=False, indent=2))
    return 0


def _render(
    records: list[dict[str, str]],
    folds: list[dict[str, str]],
    spec: dict[str, str],
    y_limit: float,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 9.2,
            "axes.linewidth": 0.85,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 7.9,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 7.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 4.6), constrained_layout=False)
    by_scene_records = defaultdict(list)
    by_scene_folds = defaultdict(list)
    for row in records:
        by_scene_records[row["scene"]].append(row)
    for row in folds:
        by_scene_folds[row["scene"]].append(row)
    route_specs = (
        (HF_COLUMN, -0.18, HF_COLOR),
        (spec["acc_column"], 0.18, ACC_COLOR),
    )
    for scene_index, scene in enumerate(SCENE_ORDER):
        scene_records = by_scene_records[scene]
        for column, offset, color in route_specs:
            values = np.asarray([float(row[column]) for row in scene_records])
            violin = ax.violinplot(
                values,
                positions=[scene_index + offset],
                widths=0.30,
                showextrema=False,
                showmeans=False,
                showmedians=False,
                points=160,
                bw_method="scott",
            )
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.18)
                body.set_linewidth(0.7)
            q1, med, q3 = np.percentile(values, (25, 50, 75))
            ax.plot(
                [scene_index + offset, scene_index + offset],
                [q1, q3],
                color=color,
                linewidth=2.6,
                solid_capstyle="round",
                zorder=4,
            )
            ax.plot(
                [scene_index + offset - 0.04, scene_index + offset + 0.04],
                [med, med],
                color="white",
                linewidth=1.15,
                zorder=5,
            )
        for row in scene_records:
            jitter = _stable_jitter(row["record_id"], 0.05)
            x_hf = scene_index - 0.18 + jitter
            x_acc = scene_index + 0.18 + jitter
            y_hf = float(row[HF_COLUMN])
            y_acc = float(row[spec["acc_column"]])
            ax.plot([x_hf, x_acc], [y_hf, y_acc], color="#919191", lw=0.38, alpha=0.30)
            ax.scatter(x_hf, y_hf, s=8.5, color=HF_COLOR, alpha=0.62, linewidths=0, zorder=3)
            ax.scatter(x_acc, y_acc, s=8.5, color=ACC_COLOR, alpha=0.62, linewidths=0, zorder=3)
        scene_folds = sorted(by_scene_folds[scene], key=lambda row: row["holdout_subject_id"])
        for offset_index, row in enumerate(scene_folds):
            jitter = np.linspace(-0.067, 0.067, len(scene_folds))[offset_index]
            x_hf = scene_index - 0.18 + jitter
            x_acc = scene_index + 0.18 + jitter
            y_hf = float(row[HF_COLUMN])
            y_acc = float(row[spec["acc_column"]])
            ax.plot([x_hf, x_acc], [y_hf, y_acc], color="#363636", lw=0.62, alpha=0.72)
            ax.scatter(
                (x_hf, x_acc),
                (y_hf, y_acc),
                s=28,
                c=(HF_COLOR, ACC_COLOR),
                edgecolors="#202020",
                linewidths=0.58,
                zorder=6,
            )
    ax.set_xlim(-0.55, len(SCENE_ORDER) - 0.45)
    ax.set_ylim(0.0, y_limit)
    ax.set_ylabel("MAE (bpm)")
    ax.set_xticks(range(len(SCENE_ORDER)))
    ax.set_xticklabels([SCENE_DISPLAY[scene] for scene in SCENE_ORDER], rotation=22, ha="right")
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10.0))
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.45, alpha=0.50, zorder=0)
    legend = (
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=HF_COLOR,
            markeredgecolor="none",
            markersize=4.6,
            label=spec["hf_label"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ACC_COLOR,
            markeredgecolor="none",
            markersize=4.6,
            label=spec["acc_label"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#202020",
            markersize=5.1,
            label="Fold mean",
        ),
    )
    ax.legend(
        handles=legend,
        loc="upper left",
        ncol=3,
        handletextpad=0.35,
        columnspacing=1.15,
        borderaxespad=0.25,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.965, bottom=0.20)
    fig.savefig(
        output_path,
        dpi=600,
        format="png",
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)


def _figure_source(
    records: list[dict[str, str]], folds: list[dict[str, str]], spec: dict[str, str]
) -> list[dict[str, Any]]:
    output = []
    for level, rows, subject_column in (
        ("record", records, "physical_subject_id"),
        ("fold", folds, "holdout_subject_id"),
    ):
        for row in rows:
            for route, column in (
                (spec["hf_label"], HF_COLUMN),
                (spec["acc_label"], spec["acc_column"]),
            ):
                output.append(
                    {
                        "figure_id": spec["figure_id"],
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


def _stable_jitter(value: str, width: float) -> float:
    integer = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)
    return ((integer / 0xFFFFFFFF) * 2.0 - 1.0) * width


def _validate_png(path: Path) -> list[int]:
    from PIL import Image

    with Image.open(path) as image:
        dpi = image.info.get("dpi")
        if image.format != "PNG" or image.width < 5500 or image.height < 2200:
            raise ValueError(f"matched_png_contract:{image.format}:{image.size}")
        if dpi is None or min(float(value) for value in dpi) < 599.0:
            raise ValueError(f"matched_png_dpi:{dpi}")
        return [image.width, image.height]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("matched_empty_figure_source")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return file_sha256(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
