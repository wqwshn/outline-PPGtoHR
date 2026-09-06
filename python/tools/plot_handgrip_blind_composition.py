"""Plot the sealed D24 Handgrip blind-composition experiment outcome."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

EXPERIMENT_ID = "cross_subject_multirecord_d24_handgrip_blind_composition_v1"
HF_COLOR = "#E28E2C"
ACC_COLOR = "#5B8FD6"
BASELINE_COLOR = "#4D4D4D"
IMPROVED_COLOR = "#2E9E44"
WORSENED_COLOR = "#B64342"
NEUTRAL_COLOR = "#8F8F8F"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    experiment_root = root / "data" / "experiments" / EXPERIMENT_ID
    figure_root = experiment_root / "p5" / "figures"
    verifier = _read_json(experiment_root / "p5" / "verifier" / "receipt.json")
    contract = _read_json(figure_root / "figure_contract.json")
    if verifier.get("status") != "pass":
        raise ValueError("handgrip_plot_verifier_not_passed")
    if contract.get("backend") != "python":
        raise ValueError("handgrip_plot_backend_contract")

    baseline = _read_json(experiment_root / "p0" / "baseline_snapshot.json")
    primary_rows = _read_csv(
        experiment_root / "p3" / "d24_15_primary" / "main" / "record_results.csv"
    )
    fallback_rows = _read_csv(
        experiment_root / "p4" / "d24_18_fallback" / "main" / "record_results.csv"
    )
    lyx_rows = _read_csv(experiment_root / "p4" / "lyx_handgrip" / "record_results.csv")
    primary_by_record = {row["record_id"]: row for row in primary_rows}
    fallback_by_record = {row["record_id"]: row for row in fallback_rows}
    if set(primary_by_record) != set(fallback_by_record) or len(primary_rows) != 15:
        raise ValueError("handgrip_plot_record_identity")

    stage_data = _stage_data(primary_rows, fallback_rows, baseline)
    grouping_data = _grouping_data(experiment_root)
    source_rows = _source_rows(
        stage_data, grouping_data, primary_by_record, fallback_by_record, lyx_rows
    )
    _write_csv(figure_root / "source_data.csv", source_rows)

    _apply_style()
    width_in = float(contract["final_size_mm"][0]) / 25.4
    height_in = float(contract["final_size_mm"][1]) / 25.4
    fig = plt.figure(figsize=(width_in, height_in), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.15, 1.15, 0.95),
        height_ratios=(1.0, 1.2),
        left=0.09,
        right=0.98,
        bottom=0.10,
        top=0.97,
        wspace=0.58,
        hspace=0.43,
    )
    ax_a = fig.add_subplot(grid[0, :2])
    ax_b = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[1, :2])
    ax_d = fig.add_subplot(grid[1, 2])

    _plot_stage_summary(ax_a, stage_data)
    _plot_grouping(ax_b, grouping_data)
    _plot_record_deltas(ax_c, primary_by_record, fallback_by_record)
    _plot_lyx(ax_d, lyx_rows)
    for label, axis in zip("abcd", (ax_a, ax_b, ax_c, ax_d), strict=True):
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    output_base = figure_root / "handgrip_blind_composition_outcome"
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    primary_hf = stage_data[1]["hf_mean"]
    fallback_hf = stage_data[2]["hf_mean"]
    lyx_hf = mean(float(row["hf_mae_bpm"]) for row in lyx_rows)
    lyx_acc = mean(float(row["acc_mae_bpm"]) for row in lyx_rows)
    lyx_independent = mean(float(row["independent_baseline_mae_bpm"]) for row in lyx_rows)
    caption = (
        "**Figure | Performance-blind Handgrip training composition does not pass the "
        "frozen D24 gate.** (a) Record-level mean MAE with sample SD (n=15 records; "
        "six LOSO subject folds). D24-15 leaves HF unchanged at "
        f"{primary_hf:.4f} BPM, whereas D24-18 increases HF to {fallback_hf:.4f} BPM; "
        "ACC is independently selected from the same training core. (b) Selected "
        "cluster count and retained training-core size for each fold; cell labels are "
        "k/core. (c) Per-record HF change after adding the three fallback training "
        "candidates; diamonds mark the two pre-locked challenge records. Negative "
        "values indicate improvement. (d) Overlapping one-subject LYX development "
        f"check (n=3 records): HF {lyx_hf:.4f}, ACC {lyx_acc:.4f}, and the existing "
        f"independent baseline {lyx_independent:.4f} BPM. Error bars are descriptive "
        "sample SD; no inferential test was used because the decision followed the "
        "pre-frozen deterministic gates. LYX is a development consistency check, not "
        "independent validation.\n"
    )
    (figure_root / "figure_caption.md").write_text(caption, encoding="utf-8")


def _apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 6.5
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = 0.7
    plt.rcParams["ytick.major.width"] = 0.7


def _stage_data(
    primary_rows: list[dict[str, str]],
    fallback_rows: list[dict[str, str]],
    baseline: dict,
) -> list[dict[str, float | str]]:
    primary_hf = [float(row["hf_mae_bpm"]) for row in primary_rows]
    primary_acc = [float(row["acc_mae_bpm"]) for row in primary_rows]
    fallback_hf = [float(row["hf_mae_bpm"]) for row in fallback_rows]
    fallback_acc = [float(row["acc_mae_bpm"]) for row in fallback_rows]
    return [
        {
            "stage": "Frozen\nbaseline",
            "hf_mean": float(baseline["hf_mean_mae_bpm"]),
            "hf_sd": stdev(primary_hf),
            "acc_mean": float(baseline["acc_mean_mae_bpm"]),
            "acc_sd": stdev(primary_acc),
        },
        {
            "stage": "D24-15",
            "hf_mean": mean(primary_hf),
            "hf_sd": stdev(primary_hf),
            "acc_mean": mean(primary_acc),
            "acc_sd": stdev(primary_acc),
        },
        {
            "stage": "D24-18",
            "hf_mean": mean(fallback_hf),
            "hf_sd": stdev(fallback_hf),
            "acc_mean": mean(fallback_acc),
            "acc_sd": stdev(fallback_acc),
        },
    ]


def _plot_stage_summary(ax: plt.Axes, stage_data: list[dict]) -> None:
    x = np.arange(len(stage_data), dtype=float)
    width = 0.34
    hf = np.asarray([row["hf_mean"] for row in stage_data], dtype=float)
    acc = np.asarray([row["acc_mean"] for row in stage_data], dtype=float)
    hf_sd = np.asarray([row["hf_sd"] for row in stage_data], dtype=float)
    acc_sd = np.asarray([row["acc_sd"] for row in stage_data], dtype=float)
    bars_hf = ax.bar(
        x - width / 2,
        hf,
        width,
        yerr=hf_sd,
        color=HF_COLOR,
        edgecolor="#333333",
        linewidth=0.6,
        capsize=2.5,
        label="HF",
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    bars_acc = ax.bar(
        x + width / 2,
        acc,
        width,
        yerr=acc_sd,
        color=ACC_COLOR,
        edgecolor="#333333",
        linewidth=0.6,
        capsize=2.5,
        label="ACC",
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    ax.axhline(
        float(stage_data[0]["hf_mean"]),
        color=BASELINE_COLOR,
        linestyle="--",
        linewidth=0.9,
        alpha=0.8,
        label="Frozen HF gate",
    )
    for bars in (bars_hf, bars_acc):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.18,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
            )
    ax.set_xticks(x, [row["stage"] for row in stage_data])
    ax.set_ylabel("MAE (BPM)")
    ax.set_ylim(0.0, max((hf + hf_sd).max(), (acc + acc_sd).max()) * 1.13)
    ax.set_title("Frozen D24 decision outcome", loc="left", fontsize=7, fontweight="bold")
    ax.legend(ncols=3, loc="upper left", fontsize=5.7, handlelength=1.5, columnspacing=1.0)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _grouping_data(experiment_root: Path) -> list[dict[str, str | int]]:
    subjects = ("CGX", "LYX", "LZJ", "PJY", "QYC", "TS")
    roots = (
        ("D24-15 main", experiment_root / "p2" / "d24_15_primary", False),
        ("D24-15 30 s", experiment_root / "p2" / "d24_15_primary", True),
        ("D24-18 main", experiment_root / "p4" / "d24_18_fallback", False),
        ("D24-18 30 s", experiment_root / "p4" / "d24_18_fallback", True),
    )
    rows = []
    for label, stage_root, sensitivity in roots:
        for subject in subjects:
            composition = _read_json(stage_root / "compositions" / f"woli__holdout_{subject}.json")
            selected = composition["sensitivity_30s"] if sensitivity else composition
            rows.append(
                {
                    "row": label,
                    "subject": subject,
                    "cluster_count": int(selected["selected_cluster_count"]),
                    "core_size": len(selected["training_core_record_ids"]),
                }
            )
    return rows


def _plot_grouping(ax: plt.Axes, rows: list[dict]) -> None:
    row_labels = list(dict.fromkeys(str(row["row"]) for row in rows))
    subjects = list(dict.fromkeys(str(row["subject"]) for row in rows))
    matrix = np.asarray(
        [
            [
                next(
                    int(row["cluster_count"])
                    for row in rows
                    if row["row"] == row_label and row["subject"] == subject
                )
                for subject in subjects
            ]
            for row_label in row_labels
        ]
    )
    core = np.asarray(
        [
            [
                next(
                    int(row["core_size"])
                    for row in rows
                    if row["row"] == row_label and row["subject"] == subject
                )
                for subject in subjects
            ]
            for row_label in row_labels
        ]
    )
    cmap = ListedColormap(("#E3E3E3", "#A9C7E8", "#F0C58B"))
    norm = BoundaryNorm((0.5, 1.5, 2.5, 3.5), cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, f"{matrix[y, x]}/{core[y, x]}", ha="center", va="center", fontsize=5.0)
    ax.set_xticks(range(len(subjects)), subjects, rotation=45, ha="right", fontsize=5.3)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=5.3)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Fold grouping (k/core)", loc="left", fontsize=7, fontweight="bold")


def _plot_record_deltas(
    ax: plt.Axes,
    primary: dict[str, dict[str, str]],
    fallback: dict[str, dict[str, str]],
) -> None:
    challenge = {"woli2_LYX_0708", "woli2_LZJ_0711"}
    values = sorted(
        (
            float(fallback[record_id]["hf_mae_bpm"]) - float(primary[record_id]["hf_mae_bpm"]),
            record_id,
        )
        for record_id in primary
    )
    y = np.arange(len(values))
    deltas = np.asarray([value for value, _ in values], dtype=float)
    colors = [
        IMPROVED_COLOR if value < -1e-12 else WORSENED_COLOR if value > 1e-12 else NEUTRAL_COLOR
        for value in deltas
    ]
    ax.axvline(0.0, color=BASELINE_COLOR, linewidth=0.8)
    ax.hlines(y, 0.0, deltas, color=colors, linewidth=1.0)
    for index, ((delta, record_id), color) in enumerate(zip(values, colors, strict=True)):
        marker = "D" if record_id in challenge else "o"
        ax.scatter(
            delta,
            index,
            s=20 if record_id in challenge else 13,
            marker=marker,
            color=color,
            edgecolor="#222222" if record_id in challenge else color,
            linewidth=0.55,
            zorder=3,
        )
    ax.set_yticks(y, [_short_record(record_id) for _, record_id in values], fontsize=5.2)
    ax.set_xlabel("HF MAE change, D24-18 minus D24-15 (BPM)")
    ax.set_title("Fallback effect by evaluation record", loc="left", fontsize=7, fontweight="bold")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(
        0.99,
        0.02,
        "diamond: locked challenge",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color=BASELINE_COLOR,
    )


def _plot_lyx(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    x = np.arange(len(rows))
    series = (
        (
            "Independent baseline",
            [float(row["independent_baseline_mae_bpm"]) for row in rows],
            BASELINE_COLOR,
            "o",
        ),
        ("HF", [float(row["hf_mae_bpm"]) for row in rows], HF_COLOR, "s"),
        ("ACC", [float(row["acc_mae_bpm"]) for row in rows], ACC_COLOR, "^"),
    )
    offsets = (-0.16, 0.0, 0.16)
    for offset, (label, values, color, marker) in zip(offsets, series, strict=True):
        ax.scatter(x + offset, values, color=color, marker=marker, s=22, label=label, zorder=3)
        ax.plot(x + offset, values, color=color, linewidth=0.7, alpha=0.55)
    ax.set_xticks(x, ("Wear 1", "Wear 2", "Wear 3"), rotation=25, ha="right", fontsize=5.5)
    ax.set_ylabel("MAE (BPM)")
    ax.set_title("Overlapping LYX development check", loc="left", fontsize=7, fontweight="bold")
    ax.legend(fontsize=5.0, loc="upper right", handletextpad=0.4)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    ax.set_axisbelow(True)


def _source_rows(
    stage_data: list[dict],
    grouping_data: list[dict],
    primary: dict[str, dict[str, str]],
    fallback: dict[str, dict[str, str]],
    lyx_rows: list[dict[str, str]],
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for stage in stage_data:
        for route in ("hf", "acc"):
            rows.append(
                {
                    "panel": "a",
                    "group": str(stage["stage"]).replace("\n", " "),
                    "item": route.upper(),
                    "metric": "mean_mae_bpm",
                    "value": float(stage[f"{route}_mean"]),
                    "spread": float(stage[f"{route}_sd"]),
                }
            )
    for row in grouping_data:
        rows.append(
            {
                "panel": "b",
                "group": str(row["row"]),
                "item": str(row["subject"]),
                "metric": "selected_cluster_count",
                "value": int(row["cluster_count"]),
                "spread": int(row["core_size"]),
            }
        )
    for record_id in sorted(primary):
        rows.append(
            {
                "panel": "c",
                "group": "D24-18 minus D24-15",
                "item": record_id,
                "metric": "hf_mae_delta_bpm",
                "value": float(fallback[record_id]["hf_mae_bpm"])
                - float(primary[record_id]["hf_mae_bpm"]),
                "spread": "",
            }
        )
    for row in lyx_rows:
        for item, column in (
            ("Independent baseline", "independent_baseline_mae_bpm"),
            ("HF", "hf_mae_bpm"),
            ("ACC", "acc_mae_bpm"),
        ):
            rows.append(
                {
                    "panel": "d",
                    "group": row["record_id"],
                    "item": item,
                    "metric": "mae_bpm",
                    "value": float(row[column]),
                    "spread": "",
                }
            )
    return rows


def _short_record(record_id: str) -> str:
    parts = record_id.split("_")
    repeat = parts[0].replace("woli", "")
    subject = parts[1]
    return f"HG{repeat}-{subject}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
