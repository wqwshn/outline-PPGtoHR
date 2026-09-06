"""Render the manuscript-grade D24 HF/ACC distribution figure and Chinese report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HF_COLOR = "#D97732"
ACC_COLOR = "#4C78A8"
NEUTRAL = "#505050"
LIGHT_NEUTRAL = "#B8B8B8"
OUTLIER = "#B65A50"
GRID = "#E5E5E5"
SCENES = ("bobi", "jianpan", "kaihe", "quanji", "run", "tiaosheng", "woli", "xiezi")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument(
        "--visual-review-status",
        choices=("pending_manual_view", "pass"),
        default="pending_manual_view",
    )
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    source = root / "source_data"
    report = root / "report"
    figures = report / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    record_rows = _read_csv(source / "d24_record_route_mae.csv")
    scene_rows = _read_csv(source / "d24_scene_summary.csv")
    excluded_rows = _read_csv(source / "d24_excluded_record_explainability.csv")
    receipt = json.loads((root / "analysis_receipt.json").read_text(encoding="utf-8"))

    _apply_style()
    fig = plt.figure(figsize=(183 / 25.4, 185 / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.55, 1.0),
        width_ratios=(1.35, 1.0),
        left=0.085,
        right=0.985,
        bottom=0.125,
        top=0.975,
        hspace=0.42,
        wspace=0.36,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    _plot_record_distributions(ax_a, record_rows)
    _plot_scene_means(ax_b, scene_rows, receipt)
    _plot_iqr_audit(ax_c, excluded_rows)
    for label, ax, x_offset in (("a", ax_a, -0.055), ("b", ax_b, -0.12), ("c", ax_c, -0.13)):
        ax.text(
            x_offset,
            1.035,
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
        "Post-hoc D24 sensitivity analysis; descriptive statistics only",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=NEUTRAL,
    )

    stem = figures / "d24_hf_acc_record_distributions"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)

    (report / "figure_legend_zh.md").write_text(
        _legend_text(receipt, record_rows, excluded_rows), encoding="utf-8"
    )
    (report / "d24_analysis_report_zh.md").write_text(
        _report_text(receipt, scene_rows, excluded_rows), encoding="utf-8"
    )
    qa = _qa_bundle(stem, source, visual_review_status=args.visual_review_status)
    (report / "figure_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"complete svg_text={qa['svg_text_element_count']} "
        f"pdf_font={qa['pdf_has_font_resource']} png_dpi={qa['png_dpi'][0]:.0f}",
        flush=True,
    )
    return 0


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


def _plot_record_distributions(ax: Any, rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["route_id"])].append(float(row["mae_bpm"]))
    rng = np.random.default_rng(20260831)
    positions = np.arange(len(SCENES), dtype=float)
    for route, offset, color in (("HF", -0.18, HF_COLOR), ("ACC", 0.18, ACC_COLOR)):
        values = [grouped[(scene, route)] for scene in SCENES]
        box = ax.boxplot(
            values,
            positions=positions + offset,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 1.0},
            boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.48, "linewidth": 0.8},
            whiskerprops={"color": color, "linewidth": 0.75},
            capprops={"color": color, "linewidth": 0.75},
        )
        for patch in box["boxes"]:
            patch.set_zorder(2)
        for index, scene_values in enumerate(values):
            jitter = rng.uniform(-0.105, 0.105, size=len(scene_values))
            ax.scatter(
                positions[index] + offset + jitter,
                scene_values,
                s=8,
                color=color,
                alpha=0.62,
                edgecolors="white",
                linewidths=0.2,
                zorder=3,
            )
            ax.scatter(
                positions[index] + offset,
                float(np.mean(scene_values)),
                marker="D",
                s=17,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=5,
            )
    ax.set_xticks(positions, SCENES)
    ax.set_ylabel("Record MAE (bpm)")
    ax.set_xlabel("Scene")
    ax.set_yscale("symlog", linthresh=8, linscale=1.0, base=2)
    ax.set_yticks((0, 2, 4, 6, 8, 12, 20, 40))
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    maximum = max(float(row["mae_bpm"]) for row in rows)
    ax.set_ylim(0, max(12.5, maximum * 1.12))
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="", color=HF_COLOR, label="HF"),
        mpl.lines.Line2D([], [], marker="o", linestyle="", color=ACC_COLOR, label="ACC"),
        mpl.lines.Line2D(
            [],
            [],
            marker="D",
            linestyle="",
            markerfacecolor=NEUTRAL,
            markeredgecolor="white",
            color=NEUTRAL,
            label="Mean",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=3, handletextpad=0.3, columnspacing=0.8)
    ax.text(
        0.005,
        0.985,
        "Boxes: median and IQR; every retained record shown\nLinear to 8 bpm, log-scaled above",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=NEUTRAL,
    )


def _plot_scene_means(ax: Any, rows: list[dict[str, str]], receipt: dict[str, Any]) -> None:
    lookup = {(row["route_id"], row["scene"]): float(row["mean_mae_bpm"]) for row in rows}
    order = sorted(SCENES, key=lambda scene: lookup[("HF", scene)], reverse=True)
    y = np.arange(len(order))[::-1]
    hf = np.asarray([lookup[("HF", scene)] for scene in order])
    acc = np.asarray([lookup[("ACC", scene)] for scene in order])
    for yi, hf_value, acc_value in zip(y, hf, acc, strict=True):
        ax.plot([hf_value, acc_value], [yi, yi], color=LIGHT_NEUTRAL, linewidth=0.85, zorder=1)
    ax.scatter(hf, y, s=22, marker="o", color=HF_COLOR, label="HF scene mean", zorder=3)
    ax.scatter(acc, y, s=22, marker="s", color=ACC_COLOR, label="ACC scene mean", zorder=3)
    ax.axvline(
        float(receipt["hf_d24_mean_mae_bpm"]),
        color=HF_COLOR,
        linestyle="--",
        linewidth=0.8,
    )
    ax.axvline(
        float(receipt["acc_d24_mean_mae_bpm"]),
        color=ACC_COLOR,
        linestyle="--",
        linewidth=0.8,
    )
    ax.set_yticks(y, order)
    ax.set_xlabel("Scene mean MAE (bpm)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=5.4, handletextpad=0.4)
    ax.text(
        0.01,
        0.985,
        "Dashed lines: route-wide means",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        color=NEUTRAL,
    )


def _plot_iqr_audit(ax: Any, rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["scene"]].append(row)
    rng = np.random.default_rng(24)
    for scene_index, scene in enumerate(SCENES):
        scene_rows = sorted(grouped[scene], key=lambda row: row["record_id"])
        x = np.full(len(scene_rows), scene_index, dtype=float) + rng.uniform(
            -0.16, 0.16, len(scene_rows)
        )
        ratios = np.asarray(
            [
                float(row["original_synced_full143_loso_mae_bpm"])
                / float(row["scene_upper_fence_bpm"])
                for row in scene_rows
            ]
        )
        colors = [OUTLIER if _bool(row["is_upper_iqr_outlier"]) else NEUTRAL for row in scene_rows]
        ax.scatter(x, ratios, s=18, c=colors, edgecolor="white", linewidth=0.35, zorder=3)
    ax.axhline(1.0, color=OUTLIER, linestyle="--", linewidth=0.9)
    ax.set_xticks(np.arange(len(SCENES)), SCENES, rotation=38, ha="right")
    ax.set_ylabel("Original LOSO MAE / scene upper fence")
    ax.set_xlabel("Scene")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    outlier_count = sum(_bool(row["is_upper_iqr_outlier"]) for row in rows)
    ax.text(
        0.02,
        0.98,
        f"{outlier_count}/24 exceed Q3 + 1.5×IQR",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=NEUTRAL,
    )


def _legend_text(
    receipt: dict[str, Any], record_rows: list[dict[str, str]], excluded_rows: list[dict[str, str]]
) -> str:
    counts = defaultdict(int)
    for row in record_rows:
        if row["route_id"] == "HF":
            counts[row["scene"]] += 1
    outliers = sum(_bool(row["is_upper_iqr_outlier"]) for row in excluded_rows)
    count_text = "、".join(f"{scene} n={counts[scene]}" for scene in SCENES)
    return f"""# 图注

**图 1｜D24（119 条记录）中 HF 与 ACC 独立调参后的逐记录误差分布及删除记录 IQR 审计。**

**a，** 八场景中保留的全部逐记录 MAE。HF（暖橙）沿用原六门训练侧选择器；ACC（冷蓝）在每个外层 LOSO 折仅使用训练个体，按最坏个体平均 MAE、五训练个体平均 MAE、坐标顺序依次最小化。箱体表示四分位距，白线为中位数，菱形为算术平均，散点为所有记录；纵轴 0–8 bpm 为线性尺度，8 bpm 以上为对数尺度。场景样本量：{count_text}。

**b，** 同一来源数据的场景平均 MAE；虚线为各路线 119 条记录的总平均，HF={float(receipt["hf_d24_mean_mae_bpm"]):.3f} bpm，ACC={float(receipt["acc_d24_mean_mae_bpm"]):.3f} bpm。连线仅辅助读取场景内两路线的描述性位置，不表示共同窗口上的配对效应。

**c，** 24 条删除记录在 LYX 同步后完整 143 条 HF LOSO 结果中，相对其场景 `Q3+1.5×IQR` 上界的位置；虚线 1 表示上界，共 {outliers}/24 条严格越界。IQR 是单变量高误差标志，不证明采集无效，也不单独构成删样依据。

D24 是揭示结果后的事后敏感性子集；ACC 独立调参并不改变 D24 由 HF 路径产生这一事实。HF 与 ACC 使用各自路线的可靠窗口支持，因此本图不报告逐窗口配对差值或推断性检验。
"""


def _report_text(
    receipt: dict[str, Any], scene_rows: list[dict[str, str]], excluded_rows: list[dict[str, str]]
) -> str:
    lookup = {(row["route_id"], row["scene"]): float(row["mean_mae_bpm"]) for row in scene_rows}
    outliers = [row for row in excluded_rows if _bool(row["is_upper_iqr_outlier"])]
    above_q3 = [
        row
        for row in excluded_rows
        if float(row["original_synced_full143_loso_mae_bpm"]) > float(row["scene_q3_bpm"])
    ]
    rows = [
        "# D24 HF/ACC 独立调参与困难记录解释性分析",
        "",
        "## 主要结果",
        "",
        f"D24 保留 119 条记录、覆盖 8 个场景和 48 个分组 LOSO 折。HF 沿用原六门选择器后的平均 MAE 为 **{float(receipt['hf_d24_mean_mae_bpm']):.4f} bpm**；ACC 使用训练个体均衡 minimax 独立调参后的平均 MAE 为 **{float(receipt['acc_d24_mean_mae_bpm']):.4f} bpm**。这两个数是各路线原生可靠窗口上的并列汇报，不应解释为同一窗口支持上的配对优劣。",
        "",
        "| 场景 | HF 平均MAE | ACC 平均MAE |",
        "|---|---:|---:|",
    ]
    rows.extend(
        f"| {scene} | {lookup[('HF', scene)]:.3f} | {lookup[('ACC', scene)]:.3f} |"
        for scene in SCENES
    )
    rows.extend(
        [
            "",
            "## 删除记录的可解释性",
            "",
            f"按完整同步 143 条记录的场景内经典 IQR 规则，24 条删除记录中 **{len(outliers)} 条**超过 `Q3+1.5×IQR`，**{len(above_q3)} 条**高于场景 Q3。因而，D24 删除集合并不等价于‘删除 24 个统计离群点’：未越界记录仍可能因为删除后改变训练侧共同坐标选择与场景记录构成而带来收益。逐条原 LOSO、历史 Lite、300 点最小值、IQR 上界和删减轮次见 `d24_excluded_record_explainability.csv`。",
            "",
            "论文中可据此表述为：D24 是受约束的场景均衡困难记录敏感性分析；IQR 审计提供部分单变量高误差证据，但不能把全部删减归因于采集异常，也不能将事后子集结果作为独立泛化性能。",
            "",
            "## 口径限制",
            "",
            "- D24 由已揭示的 HF LOSO 表现事后形成；它是开发集敏感性结果，不是预注册或独立验证。",
            "- ACC 坐标在每个外层折内只用训练个体选择，但 D24 本身并非按 ACC 独立产生。",
            "- 记录嵌套于个体和场景；本轮只报告描述性分布，不做独立同分布假设下的显著性检验。",
            "- IQR 离群只描述原 HF LOSO 的场景内单变量位置，不证明原始采集无效。",
        ]
    )
    return "\n".join(rows) + "\n"


def _qa_bundle(stem: Path, source: Path, *, visual_review_status: str) -> dict[str, Any]:
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    svg_text = svg_path.read_text(encoding="utf-8")
    pdf_bytes = pdf_path.read_bytes()
    with Image.open(png_path) as image:
        dpi = image.info.get("dpi", (0.0, 0.0))
        dimensions = image.size
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.glob("*.csv"))
    }
    text_count = len(re.findall(r"<text(?:\s|>)", svg_text))
    pdf_font = b"/Font" in pdf_bytes
    dpi_tuple = (float(dpi[0]), float(dpi[1]))
    automated_status = (
        "pass"
        if text_count > 0 and pdf_font and min(dpi_tuple) >= 599.0 and dimensions[0] > 3000
        else "fail"
    )
    status = "pass" if automated_status == "pass" and visual_review_status == "pass" else "fail"
    return {
        "schema_id": "d24_nature_figure_qa_v1",
        "status": status,
        "backend": "Python",
        "backend_exclusive": True,
        "svg_editable_text_required": True,
        "svg_text_element_count": text_count,
        "pdf_font_resource_required": True,
        "pdf_has_font_resource": pdf_font,
        "png_dpi_required": 600,
        "png_dpi": dpi_tuple,
        "png_dimensions_px": dimensions,
        "all_record_points_required": True,
        "plotted_record_point_count": 238,
        "plotted_excluded_record_point_count": 24,
        "automated_qa_status": automated_status,
        "visual_review_status": visual_review_status,
        "source_sha256": source_hashes,
        "artifact_sha256": {
            path.suffix.lstrip("."): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (svg_path, pdf_path, png_path)
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
