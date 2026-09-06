from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


PARENT = "#4D4D4D"
MAIN = "#D97732"
MAIN_LIGHT = "#E8B07A"
SECONDARY = "#4C78A8"
TEAL = "#4A9D8E"
WORSE = "#B65A50"
BETTER = "#4A8C68"
NEUTRAL = "#A6A6A6"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    report = root / "report"
    source_dir = report / "source_data"
    figure_dir = report / "figures"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader((root / "panel_summary.csv").open(encoding="utf-8-sig")))
    by_id = {row["panel_id"]: row for row in rows}
    receipt = json.loads((root / "run_receipt.json").read_text(encoding="utf-8"))
    local_id = str(receipt["optimized_record_panel_source_id"])

    trajectory = _trajectory_rows(rows)
    decomposition = _decomposition_rows(by_id, local_id)
    selector_rows = _selector_rows(by_id, local_id)
    scene_rows = _scene_rows(root, ("lyx_synced_full143", local_id))
    _write_csv(source_dir / "figure_1a_deletion_trajectory.csv", trajectory)
    _write_csv(source_dir / "figure_1b_gain_decomposition.csv", decomposition)
    _write_csv(source_dir / "figure_1c_selector_gate_ablation.csv", selector_rows)
    _write_csv(source_dir / "figure_1d_scene_folds.csv", scene_rows)

    _apply_style()
    fig = plt.figure(figsize=(183 / 25.4, 152 / 25.4))
    gs = fig.add_gridspec(
        3,
        6,
        height_ratios=(1.0, 1.0, 0.95),
        left=0.075,
        right=0.985,
        bottom=0.09,
        top=0.97,
        hspace=0.48,
        wspace=0.8,
    )
    ax_a = fig.add_subplot(gs[0:2, 0:4])
    ax_b = fig.add_subplot(gs[0, 4:6])
    ax_c = fig.add_subplot(gs[1, 4:6])
    ax_d = fig.add_subplot(gs[2, 0:6])
    _plot_trajectory(ax_a, trajectory, local_id)
    _plot_decomposition(ax_b, decomposition)
    _plot_selector_ablation(ax_c, selector_rows)
    _plot_scene_folds(ax_d, scene_rows)
    for label, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d), strict=True):
        ax.text(
            -0.12 if ax is not ax_d else -0.035,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    fig.text(
        0.985,
        0.012,
        "Post-hoc development backtest; no statistical inference",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=PARENT,
    )

    stem = figure_dir / "hf_loso_curation_optimization"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    legend = _legend_text(local_id, by_id)
    (report / "figure_legend_zh.md").write_text(legend, encoding="utf-8")
    qa = _qa_bundle(stem, source_dir)
    (report / "figure_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"complete local={local_id} svg_text={qa['svg_text_element_count']} "
        f"png_dpi={qa['png_dpi'][0]:.0f}"
    )


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 6,
        }
    )


def _trajectory_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        panel_id = row["panel_id"]
        if panel_id == "parent_full143":
            stage, order = "Historical anchor", -2
        elif panel_id == "lyx_synced_full143":
            stage, order = "LYX-synced anchor", -1
        elif panel_id.startswith("balanced_record_d"):
            stage, order = "Balanced deletion", int(panel_id.rsplit("d", 1)[1])
        elif panel_id.startswith("unbalanced_record_d"):
            stage, order = "Unbalanced extension", int(panel_id.rsplit("d", 1)[1])
        elif panel_id.startswith("local_swap_d42_i"):
            stage = "Local swap"
            order = 100 + int(panel_id.rsplit("i", 1)[1])
        else:
            continue
        selected.append(
            {
                "panel_id": panel_id,
                "stage": stage,
                "plot_order": order,
                "excluded_record_count": int(row["excluded_record_count"]),
                "retained_record_count": int(row["record_count"]),
                "mean_mae_bpm": float(row["mean_mae_bpm"]),
                "qualified_record_count": int(row["qualified_record_count"]),
            }
        )
    return sorted(selected, key=lambda row: int(row["plot_order"]))


def _decomposition_rows(by_id: dict[str, dict[str, str]], local_id: str) -> list[dict[str, Any]]:
    rows = []
    for panel_id, label in (
        ("balanced_record_d24", "R119"),
        ("balanced_record_d40", "R103"),
        (local_id, "D42 local"),
    ):
        row = by_id[panel_id]
        composition = float(row["composition_effect_vs_lyx_synced_bpm"])
        reselection = float(row["reselection_or_selector_effect_bpm"])
        rows.append(
            {
                "panel_id": panel_id,
                "label": label,
                "composition_effect_bpm": composition,
                "reselection_effect_bpm": reselection,
                "total_effect_bpm": composition + reselection,
            }
        )
    return rows


def _selector_rows(by_id: dict[str, dict[str, str]], local_id: str) -> list[dict[str, Any]]:
    full_reference = float(by_id["lyx_synced_full143"]["mean_mae_bpm"])
    local_reference = float(by_id[local_id]["mean_mae_bpm"])
    specs = (
        ("Full", "4D gate proxy", "lyx_synced_best4d_core_gate_proxy_full143", full_reference),
        ("Full", "Mean-first", "lyx_synced_mean_first_full143", full_reference),
        ("Full", "Consensus", "lyx_synced_consensus_full143", full_reference),
        ("D42", "Mean-first", "optimized_record_mean_first", local_reference),
        ("D42", "Consensus", "optimized_record_consensus", local_reference),
    )
    return [
        {
            "roster": roster,
            "variant": variant,
            "panel_id": panel_id,
            "mean_mae_bpm": float(by_id[panel_id]["mean_mae_bpm"]),
            "legacy_reference_mae_bpm": reference,
            "delta_vs_legacy_bpm": float(by_id[panel_id]["mean_mae_bpm"]) - reference,
        }
        for roster, variant, panel_id, reference in specs
    ]


def _scene_rows(root: Path, panel_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    labels = {panel_ids[0]: "LYX-synced full", panel_ids[1]: "D42 local"}
    for panel_id in panel_ids:
        folds = csv.DictReader(
            (root / "panels" / panel_id / "folds.csv").open(encoding="utf-8-sig")
        )
        for fold in folds:
            rows.append(
                {
                    "panel_id": panel_id,
                    "panel_label": labels[panel_id],
                    "scene": fold["scene"],
                    "holdout_subject_id": fold["holdout_subject_id"],
                    "holdout_record_count": int(fold["holdout_record_count"]),
                    "holdout_mean_mae_bpm": float(fold["holdout_mean_mae_bpm"]),
                }
            )
    return rows


def _plot_trajectory(ax, rows: list[dict[str, Any]], local_id: str) -> None:
    by_stage = defaultdict(list)
    for row in rows:
        by_stage[row["stage"]].append(row)
    parent = by_stage["Historical anchor"][0]
    synced = by_stage["LYX-synced anchor"][0]
    ax.scatter(0, parent["mean_mae_bpm"], marker="s", s=25, color=PARENT, zorder=5)
    ax.scatter(0.7, synced["mean_mae_bpm"], marker="o", s=28, color=MAIN, zorder=6)
    ax.plot(
        [0.7] + [row["excluded_record_count"] for row in by_stage["Balanced deletion"]],
        [synced["mean_mae_bpm"]] + [row["mean_mae_bpm"] for row in by_stage["Balanced deletion"]],
        color=MAIN,
        lw=1.8,
        marker="o",
        ms=4,
        label="Balanced deletion",
    )
    unbalanced = by_stage["Unbalanced extension"]
    start = next(row for row in by_stage["Balanced deletion"] if row["excluded_record_count"] == 40)
    ax.plot(
        [start["excluded_record_count"]] + [row["excluded_record_count"] for row in unbalanced],
        [start["mean_mae_bpm"]] + [row["mean_mae_bpm"] for row in unbalanced],
        color=SECONDARY,
        lw=1.5,
        marker="o",
        ms=3.5,
        label="Unbalanced extension",
    )
    local = by_stage["Local swap"]
    d42 = next(row for row in unbalanced if row["excluded_record_count"] == 42)
    local_x = [42 + 0.16 * index for index in range(len(local) + 1)]
    ax.plot(
        local_x,
        [d42["mean_mae_bpm"]] + [row["mean_mae_bpm"] for row in local],
        color=TEAL,
        lw=1.5,
        marker="D",
        ms=3.5,
        label="Local swaps",
        zorder=7,
    )
    final = next(row for row in local if row["panel_id"] == local_id)
    ax.annotate(
        f"{final['mean_mae_bpm']:.3f} bpm\n101 records",
        xy=(local_x[-1], final["mean_mae_bpm"]),
        xytext=(33, 3.30),
        textcoords="data",
        fontsize=6,
        color=TEAL,
        arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 0.8},
    )
    ax.text(-0.5, parent["mean_mae_bpm"] + 0.15, "Historical 143", color=PARENT, fontsize=6)
    ax.text(1.2, synced["mean_mae_bpm"] - 0.28, "LYX-synced 143", color=MAIN, fontsize=6)
    ax.set_xlabel("Excluded records")
    ax.set_ylabel("HF LOSO mean MAE (bpm)")
    ax.set_xlim(-2, 49)
    ax.set_ylim(3.15, 7.25)
    ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    ax.legend(loc="upper right", handlelength=2.2)


def _plot_decomposition(ax, rows: list[dict[str, Any]]) -> None:
    y = np.arange(len(rows))[::-1]
    composition = np.asarray([row["composition_effect_bpm"] for row in rows])
    reselection = np.asarray([row["reselection_effect_bpm"] for row in rows])
    ax.barh(y, composition, color=MAIN_LIGHT, height=0.58, label="Composition")
    ax.barh(y, reselection, left=composition, color=TEAL, height=0.58, label="Reselection")
    for yi, row in zip(y, rows, strict=True):
        ax.text(
            row["total_effect_bpm"] - 0.04,
            yi,
            f"{row['total_effect_bpm']:.2f}",
            ha="right",
            va="center",
            fontsize=5.5,
        )
    ax.axvline(0, color=PARENT, lw=0.8)
    ax.set_yticks(y, [row["label"] for row in rows])
    ax.set_xlabel("Δ MAE vs LYX-synced full (bpm)")
    ax.set_xlim(-3.2, 0.1)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=5.2,
        handlelength=1.2,
        columnspacing=0.9,
    )


def _plot_selector_ablation(ax, rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['roster']} {row['variant']}" for row in rows]
    values = [row["delta_vs_legacy_bpm"] for row in rows]
    y = np.arange(len(rows))[::-1]
    colors = [BETTER if value < 0 else WORSE for value in values]
    ax.barh(y, values, color=colors, height=0.62)
    ax.axvline(0, color=PARENT, lw=0.8)
    for yi, value in zip(y, values, strict=True):
        if -0.03 < value < 0:
            label_x = 0.012
            label_ha = "left"
        else:
            label_x = value + (0.018 if value >= 0 else -0.018)
            label_ha = "left" if value >= 0 else "right"
        ax.text(
            label_x,
            yi,
            f"{value:+.3f}",
            ha=label_ha,
            va="center",
            fontsize=5.2,
        )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Δ MAE vs legacy selector (bpm)")
    ax.set_xlim(-0.09, 0.84)


def _plot_scene_folds(ax, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["panel_label"], row["scene"])].append(row["holdout_mean_mae_bpm"])
    scenes = sorted(
        {row["scene"] for row in rows},
        key=lambda scene: mean(grouped[("LYX-synced full", scene)]),
        reverse=True,
    )
    x = np.arange(len(scenes))
    rng = np.random.default_rng(42)
    for label, offset, color, marker in (
        ("LYX-synced full", -0.16, PARENT, "o"),
        ("D42 local", 0.16, MAIN, "D"),
    ):
        means = [mean(grouped[(label, scene)]) for scene in scenes]
        spreads = [stdev(grouped[(label, scene)]) for scene in scenes]
        for index, scene in enumerate(scenes):
            values = grouped[(label, scene)]
            jitter = rng.normal(0, 0.018, size=len(values))
            ax.scatter(
                np.full(len(values), x[index] + offset) + jitter,
                values,
                s=7,
                color=color,
                alpha=0.38,
                linewidths=0,
                zorder=2,
            )
        ax.errorbar(
            x + offset,
            means,
            yerr=spreads,
            fmt=marker,
            color=color,
            markersize=3.8,
            capsize=2,
            elinewidth=0.9,
            lw=0,
            label=label,
            zorder=4,
        )
    ax.set_xticks(x, scenes, rotation=25, ha="right")
    ax.set_ylabel("Fold mean MAE (bpm)")
    ax.set_xlabel("Scene")
    ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    ax.legend(loc="upper right", ncol=2)


def _legend_text(local_id: str, by_id: dict[str, dict[str, str]]) -> str:
    local = by_id[local_id]
    consensus = by_id["optimized_record_consensus"]
    return (
        "# 图 1 | 后验记录策展主导 HF-LOSO 误差下降\n\n"
        "**a** 在旧 143 条权威锚点、同步 7 条 LYX 替换后的 143 条锚点，以及均衡删减、"
        "非均衡扩展和局部交换阶段，展示逐记录 HF-LOSO MAE 均值。每个三记录的受试者—场景格最多删除 1 条；"
        f"最终原六门选择器的局部最优面板保留 101 条，MAE 为 {float(local['mean_mae_bpm']):.3f} bpm。"
        "**b** 将关键面板相对 LYX 同步全量锚点的变化分为固定原坐标的组成效应与在删减后重新选参的效应。"
        "**c** 给出选择器和 best-4D 核心门基线代理相对同一清单原六门选择器的 MAE 变化；正值表示恶化。"
        f"最优 101 条清单上的共识选择器为 {float(consensus['mean_mae_bpm']):.3f} bpm，"
        "但其在全量清单上明显恶化，因此不替代正式选择器。"
        "**d** 显示八个场景的六个受试者留出折均值；大符号和误差棒分别为六折算术均值和样本标准差，"
        "半透明小点为全部折。所有结果均为已揭盲响应面上的后验开发回放，不执行统计推断，也不构成独立外部验证。"
        "Source data are provided in the report source-data directory.\n"
    )


def _qa_bundle(stem: Path, source_dir: Path) -> dict[str, Any]:
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    with Image.open(png_path) as image:
        dpi = image.info.get("dpi", (0.0, 0.0))
        pixel_size = image.size
    svg_text = svg_path.read_text(encoding="utf-8")
    text_count = len(re.findall(r"<text\b", svg_text))
    if text_count < 20:
        raise AssertionError("svg_editable_text_check_failed")
    pdf_bytes = pdf_path.read_bytes()
    font_subtypes = sorted(
        {
            match.decode("ascii")
            for match in re.findall(rb"/Subtype\s*/(Type0|TrueType|Type1)", pdf_bytes)
        }
    )
    if b"/Font" not in pdf_bytes or not font_subtypes:
        raise AssertionError("pdf_font_resource_check_failed")
    pdf_page_count = len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))
    return {
        "status": "automated_checks_pass_visual_review_pending",
        "backend": "Python",
        "backend_exclusive": True,
        "png_dpi": [float(dpi[0]), float(dpi[1])],
        "png_pixel_size": list(pixel_size),
        "svg_text_element_count": text_count,
        "pdf_page_count": pdf_page_count,
        "pdf_font_subtypes": font_subtypes,
        "exports": {
            path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
            for path in (svg_path, pdf_path, png_path)
        },
        "source_data": {
            path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
            for path in sorted(source_dir.glob("*.csv"))
        },
        "statistical_inference_performed": False,
        "image_manipulation_performed": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty_source_data:{path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
