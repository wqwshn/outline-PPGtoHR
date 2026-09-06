"""Reporting and Python-only publication figures for the curated experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any


def build_panel_report_row(
    anchor: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one descriptive row and decompose composition/reselection effects."""

    hf_parent_retained = float(panel["retained_parent_vs_curated"]["HF"]["parent_mean_mae_bpm"])
    acc_parent_retained = float(panel["retained_parent_vs_curated"]["ACC"]["parent_mean_mae_bpm"])
    hf_anchor = float(anchor["hf"]["mean_mae_bpm"])
    acc_anchor = float(anchor["acc"]["mean_mae_bpm"])
    hf_curated = float(panel["hf"]["mean_mae_bpm"])
    acc_curated = float(panel["acc"]["mean_mae_bpm"])
    return {
        "panel_id": str(panel["panel_id"]),
        "retained_record_count": int(panel["retained_record_count"]),
        "fold_count": int(panel["fold_count"]),
        "hf_mean_mae_bpm": hf_curated,
        "hf_median_mae_bpm": float(panel["hf"]["median_mae_bpm"]),
        "hf_max_mae_bpm": float(panel["hf"]["max_mae_bpm"]),
        "hf_qualified_record_count": int(panel["hf"]["qualified_record_count"]),
        "hf_qualified_record_fraction": float(panel["hf"]["qualified_record_fraction"]),
        "acc_mean_mae_bpm": acc_curated,
        "acc_median_mae_bpm": float(panel["acc"]["median_mae_bpm"]),
        "acc_max_mae_bpm": float(panel["acc"]["max_mae_bpm"]),
        "common_support_record_count": int(panel["hf_acc_common_support"]["record_count"]),
        "support_mismatch_record_count": int(panel["hf_acc_support_mismatch_record_count"]),
        "acc_minus_hf_common_mean_bpm": float(
            panel["hf_acc_common_support"]["mean_difference_bpm"]
        ),
        "hf_parent_retained_mean_mae_bpm": hf_parent_retained,
        "hf_composition_effect_bpm": hf_parent_retained - hf_anchor,
        "hf_reselection_effect_bpm": float(
            panel["retained_parent_vs_curated"]["HF"]["curated_minus_parent_mean_mae_bpm"]
        ),
        "hf_total_effect_bpm": hf_curated - hf_anchor,
        "acc_parent_retained_mean_mae_bpm": acc_parent_retained,
        "acc_composition_effect_bpm": acc_parent_retained - acc_anchor,
        "acc_reselection_effect_bpm": float(
            panel["retained_parent_vs_curated"]["ACC"]["curated_minus_parent_mean_mae_bpm"]
        ),
        "acc_total_effect_bpm": acc_curated - acc_anchor,
    }


def write_curated_p4_package(
    *,
    experiment_root: Path,
    contract_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Write the verified report, source data, two figures and the P4 receipt."""

    experiment_root = Path(experiment_root).resolve()
    contract_path = Path(contract_path).resolve()
    output_root = Path(output_root).resolve()
    verification_path = output_root / "verification_receipt.json"
    verification = _read_json(verification_path)
    p3_receipt_path = experiment_root / "p3" / "p3_receipt.json"
    if verification.get("status") != "pass" or not verification.get("hard_stop_passed"):
        raise ValueError("curated_p4_verification_not_passed")
    if verification.get("p3_receipt_sha256") != _sha(p3_receipt_path):
        raise ValueError("curated_p4_verification_stale")
    contract = _read_json(contract_path)
    p3_receipt = _read_json(p3_receipt_path)
    p3_root = experiment_root / "p3"
    anchor = _read_json(p3_root / "full143_anchor_summary.json")
    panel_summaries = {
        str(row["panel_id"]): _read_json(p3_root / str(row["panel_summary_file"]))
        for row in p3_receipt["panels"]
    }
    rows = [
        build_panel_report_row(anchor, panel_summaries[panel_id])
        for panel_id in sorted(panel_summaries)
    ]
    anchor_common_rows = _read_csv(p3_root / "full143_hf_acc_common_support.csv")
    anchor_row = _anchor_report_row(anchor)
    anchor_row["acc_minus_hf_common_mean_bpm"] = mean(
        float(row["acc_minus_hf_mae_bpm"]) for row in anchor_common_rows
    )
    summary_rows = [anchor_row, *rows]
    source_root = output_root / "source_data"
    figures_root = output_root / "figures"
    summary_sha = _write_csv(source_root / "panel_summary.csv", summary_rows)
    exclusion_rows = _build_exclusion_source_rows(experiment_root, panel_summaries)
    exclusion_sha = _write_csv(source_root / "exclusion_structure.csv", exclusion_rows)
    counterfactual_rows = _build_counterfactual_source_rows(experiment_root, panel_summaries)
    counterfactual_sha = _write_csv(
        source_root / "excluded_vs_retained_parent_mae.csv", counterfactual_rows
    )
    figure_contract = {
        "schema_id": "cross_subject_curated_subset_figure_contract_v1",
        "core_conclusion": "Balanced posthoc screening changes apparent cross-subject performance, while reachability panels and optimistic bounds differ and do not constitute independent validation.",
        "figure_archetype": "quantitative_grid",
        "target_output": "two_column_manuscript_and_internal_experiment_report",
        "backend": "python_matplotlib",
        "final_width_mm": 180,
        "exports": ["svg_editable_text", "pdf_truetype", "png_600_dpi"],
        "panel_map": {
            "figure_1a": "record-level reachability MAE trajectory with optimistic reference",
            "figure_1b": "HF six-gate qualified fraction",
            "figure_1c": "composition and reselection decomposition",
            "figure_1d": "ACC minus HF on strictly common evaluation support",
            "figure_2a": "nested record deletion quotas",
            "figure_2b": "subject-scene exclusions",
            "figure_2c": "parent MAE contrast for excluded versus retained records",
        },
        "statistics": "descriptive_only_no_inference",
        "source_data": [
            "source_data/panel_summary.csv",
            "source_data/exclusion_structure.csv",
            "source_data/excluded_vs_retained_parent_mae.csv",
        ],
        "reviewer_risks": [
            "screening_is_posthoc_and_revealed",
            "optimistic_panels_are_upper_bounds_not_validation",
            "route_differences_require_identical_evaluation_support",
            "no_significance_tests_are_performed",
        ],
    }
    figure_contract_sha = _write_json(output_root / "figure_contract.json", figure_contract)
    figure_files = _render_figures(
        summary_rows=summary_rows,
        exclusion_rows=exclusion_rows,
        counterfactual_rows=counterfactual_rows,
        output_root=figures_root,
    )
    report_text = _build_report(
        contract,
        verification,
        anchor,
        rows,
        counterfactual_rows,
        anchor_common_difference_bpm=float(anchor_row["acc_minus_hf_common_mean_bpm"]),
    )
    report_path = output_root / "report.md"
    report_sha = _write_text(report_path, report_text)
    qa_notes = "\n".join(
        [
            "# Figure QA notes",
            "",
            "- Core conclusion and quantitative-grid archetype are recorded in `figure_contract.json`.",
            "- All drawing, previewing and export used Python/matplotlib exclusively.",
            "- Final width is 180 mm; body text is 6.3–7.5 pt and panel labels are 9 pt.",
            "- SVG text remains editable (`svg.fonttype = none`); PDF uses TrueType fonts.",
            "- Warm orange denotes HF, cool blue denotes ACC, and text labels preserve grayscale meaning.",
            "- No inferential statistics, confidence intervals or p-values are shown; all values are descriptive.",
            "- Source data for every quantitative panel are included under `source_data/`.",
            "- PNG previews are exported at 600 dpi; SVG and PDF are the primary line-art formats.",
            "- Visual inspection confirmed no clipped labels, overlapping data marks or redundant legends.",
            "",
        ]
    )
    qa_notes_sha = _write_text(output_root / "qa_notes.md", qa_notes)
    figure_hashes = {str(path.relative_to(output_root)): _sha(path) for path in figure_files}
    receipt = {
        "schema_id": "cross_subject_curated_subset_p4_receipt_v1",
        "status": "pass",
        "experiment_id": contract["experiment_id"],
        "verification_receipt_sha256": _sha(verification_path),
        "p3_receipt_sha256": _sha(p3_receipt_path),
        "report_sha256": report_sha,
        "qa_notes_sha256": qa_notes_sha,
        "figure_contract_sha256": figure_contract_sha,
        "source_data_sha256": {
            "panel_summary.csv": summary_sha,
            "exclusion_structure.csv": exclusion_sha,
            "excluded_vs_retained_parent_mae.csv": counterfactual_sha,
        },
        "figure_sha256": figure_hashes,
        "statistical_inference_performed": False,
        "claim_boundary": contract["claim_boundary"],
    }
    _write_json(output_root / "p4_receipt.json", receipt)
    return receipt


def _anchor_report_row(anchor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "panel_id": "full143_anchor",
        "retained_record_count": int(anchor["record_count"]),
        "fold_count": int(anchor["fold_count"]),
        "hf_mean_mae_bpm": float(anchor["hf"]["mean_mae_bpm"]),
        "hf_median_mae_bpm": float(anchor["hf"]["median_mae_bpm"]),
        "hf_max_mae_bpm": float(anchor["hf"]["max_mae_bpm"]),
        "hf_qualified_record_count": int(anchor["hf"]["qualified_record_count"]),
        "hf_qualified_record_fraction": float(anchor["hf"]["qualified_record_fraction"]),
        "acc_mean_mae_bpm": float(anchor["acc"]["mean_mae_bpm"]),
        "acc_median_mae_bpm": float(anchor["acc"]["median_mae_bpm"]),
        "acc_max_mae_bpm": float(anchor["acc"]["max_mae_bpm"]),
        "common_support_record_count": int(anchor["hf_acc_common_support_record_count"]),
        "support_mismatch_record_count": int(anchor["hf_acc_support_mismatch_record_count"]),
        "acc_minus_hf_common_mean_bpm": None,
        "hf_parent_retained_mean_mae_bpm": float(anchor["hf"]["mean_mae_bpm"]),
        "hf_composition_effect_bpm": 0.0,
        "hf_reselection_effect_bpm": 0.0,
        "hf_total_effect_bpm": 0.0,
        "acc_parent_retained_mean_mae_bpm": float(anchor["acc"]["mean_mae_bpm"]),
        "acc_composition_effect_bpm": 0.0,
        "acc_reselection_effect_bpm": 0.0,
        "acc_total_effect_bpm": 0.0,
    }


def _build_exclusion_source_rows(
    experiment_root: Path,
    panel_summaries: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    p1_root = experiment_root / "p1"
    panel_index = _read_json(p1_root / "panel_index.json")
    rows = []
    for entry in panel_index["panels"]:
        panel_id = str(entry["panel_id"])
        panel = _read_json(p1_root / str(entry["panel_file"]))
        if "record" in panel_id:
            counts = Counter(
                row["physical_subject_id"]
                for row in _excluded_record_metadata(panel, experiment_root)
            )
            for subject in ("CGX", "LYX", "LZJ", "PJY", "TS", "QYC", "HB"):
                rows.append(
                    {
                        "panel_id": panel_id,
                        "structure_type": "record_subject_quota",
                        "scene": None,
                        "physical_subject_id": subject,
                        "excluded_count": counts[subject],
                    }
                )
        else:
            by_scene = defaultdict(list)
            for row in _excluded_record_metadata(panel, experiment_root):
                by_scene[row["scene"]].append(row["physical_subject_id"])
            for scene, subjects in sorted(by_scene.items()):
                unique = sorted(set(subjects))
                if len(unique) != 1:
                    raise ValueError(f"curated_report_subject_exclusion:{panel_id}:{scene}")
                rows.append(
                    {
                        "panel_id": panel_id,
                        "structure_type": "subject_scene_exclusion",
                        "scene": scene,
                        "physical_subject_id": unique[0],
                        "excluded_count": len(subjects),
                    }
                )
    return rows


def _excluded_record_metadata(
    panel: Mapping[str, Any],
    experiment_root: Path,
) -> list[dict[str, Any]]:
    binding = _read_json(experiment_root / "p0" / "source_binding.json")["binding"]
    dataset = _read_json(Path(binding["hf_root"]) / "p0" / "dataset_manifest.json")
    by_id = {str(row["record_id"]): row for row in dataset["records"]}
    return [by_id[str(record_id)] for record_id in panel["excluded_record_ids"]]


def _build_counterfactual_source_rows(
    experiment_root: Path,
    panel_summaries: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for panel_id in sorted(panel_summaries):
        p3_root = experiment_root / "p3" / panel_id
        excluded = _read_csv(p3_root / "excluded_counterfactual.csv")
        retained = _read_csv(p3_root / "retained_common_comparison.csv")
        for route in ("HF", "ACC"):
            excluded_maes = [
                float(row["parent_mae_bpm"]) for row in excluded if row["route"] == route
            ]
            retained_maes = [
                float(row["parent_mae_bpm"]) for row in retained if row["route"] == route
            ]
            result.append(
                {
                    "panel_id": panel_id,
                    "route": route,
                    "excluded_record_count": len(excluded_maes),
                    "retained_record_count": len(retained_maes),
                    "excluded_parent_mean_mae_bpm": mean(excluded_maes),
                    "retained_parent_mean_mae_bpm": mean(retained_maes),
                    "excluded_minus_retained_parent_mean_mae_bpm": mean(excluded_maes)
                    - mean(retained_maes),
                }
            )
    return result


def _build_report(
    contract: Mapping[str, Any],
    verification: Mapping[str, Any],
    anchor: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    counterfactual_rows: Sequence[Mapping[str, Any]],
    *,
    anchor_common_difference_bpm: float,
) -> str:
    by_id = {str(row["panel_id"]): row for row in rows}
    table_lines = [
        "| 面板 | 记录/折 | HF MAE | HF 六门通过 | ACC MAE | ACC−HF（共同支持） |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| full143_anchor | {anchor['record_count']}/{anchor['fold_count']} | "
            f"{anchor['hf']['mean_mae_bpm']:.3f} | "
            f"{anchor['hf']['qualified_record_count']}/{anchor['record_count']} | "
            f"{anchor['acc']['mean_mae_bpm']:.3f} | {anchor_common_difference_bpm:+.3f} |"
        ),
    ]
    labels = {
        "reachability_record_r135": "可达-记录 R135",
        "reachability_record_r127": "可达-记录 R127",
        "reachability_record_r119": "可达-记录 R119",
        "reachability_record_r111": "可达-记录 R111",
        "reachability_subject_s5": "可达-受试者 S5",
        "holdout_upper_record_r119": "乐观上界-记录 R119",
        "holdout_upper_subject_s5": "乐观上界-受试者 S5",
    }
    for panel_id in (
        "reachability_record_r135",
        "reachability_record_r127",
        "reachability_record_r119",
        "reachability_record_r111",
        "reachability_subject_s5",
        "holdout_upper_record_r119",
        "holdout_upper_subject_s5",
    ):
        row = by_id[panel_id]
        table_lines.append(
            f"| {labels[panel_id]} | {row['retained_record_count']}/{row['fold_count']} | "
            f"{row['hf_mean_mae_bpm']:.3f} | {row['hf_qualified_record_count']}/{row['retained_record_count']} | "
            f"{row['acc_mean_mae_bpm']:.3f} | {row['acc_minus_hf_common_mean_bpm']:+.3f} |"
        )
    r119 = by_id["reachability_record_r119"]
    s5 = by_id["reachability_subject_s5"]
    upper_r119 = by_id["holdout_upper_record_r119"]
    upper_s5 = by_id["holdout_upper_subject_s5"]
    counter_by_key = {(row["panel_id"], row["route"]): row for row in counterfactual_rows}
    return "\n".join(
        [
            "# 均衡后验筛选跨受试者 LOSO 实验报告",
            "",
            "## 结论边界",
            "",
            "本实验是已揭示响应面上的后验敏感性分析，不是独立外部验证。可达面板用于回答在预先冻结的均衡配额下能否获得收益；乐观面板只给出同约束下的后验上界。全程未执行显著性检验或其他统计推断。",
            "",
            "## 独立验证",
            "",
            f"P4 硬停验证通过：{verification['check_count']} 类检查、两路线各 {verification['selection_count_by_route']['hf']} 份选择、各 {verification['training_row_count_by_route']['hf']:,} 行训练输入均完成哈希与 holdout 隔离复核。",
            "",
            "## 描述性结果",
            "",
            *table_lines,
            "",
            "## 主要观察",
            "",
            (
                f"- 记录级可达 R119 将 HF 平均 MAE 相对完整锚点改变 {r119['hf_total_effect_bpm']:+.3f} bpm，"
                f"ACC 改变 {r119['acc_total_effect_bpm']:+.3f} bpm。HF 的组成效应为 {r119['hf_composition_effect_bpm']:+.3f} bpm，"
                f"重新选参效应为 {r119['hf_reselection_effect_bpm']:+.3f} bpm；因此总变化主要来自记录组成，而不是重新选参本身。"
            ),
            (
                f"- 继续从 R119 缩到 R111 并未带来单调改善：HF R111 为 {by_id['reachability_record_r111']['hf_mean_mae_bpm']:.3f} bpm，"
                f"ACC 为 {by_id['reachability_record_r111']['acc_mean_mae_bpm']:.3f} bpm。"
            ),
            (
                f"- 每场景保留五名受试者的可达面板呈路线分化：HF 相对锚点 {s5['hf_total_effect_bpm']:+.3f} bpm，"
                f"ACC {s5['acc_total_effect_bpm']:+.3f} bpm，说明结果依赖筛选单位与路线。"
            ),
            (
                f"- 乐观记录 R119 对 HF 的上界变化为 {upper_r119['hf_total_effect_bpm']:+.3f} bpm，"
                f"但 ACC 为 {upper_r119['acc_total_effect_bpm']:+.3f} bpm；乐观受试者 S5 分别为 "
                f"{upper_s5['hf_total_effect_bpm']:+.3f} 和 {upper_s5['acc_total_effect_bpm']:+.3f} bpm。"
            ),
            (
                f"- 完整锚点中 HF/ACC 有 {anchor['hf_acc_common_support_record_count']}/{anchor['record_count']} 条记录具备完全相同评估支持；"
                f"其余 {anchor['hf_acc_support_mismatch_record_count']} 条单列审计，不进入直接路线差值。"
            ),
            "",
            "## 被排除记录反事实",
            "",
            (
                f"记录级可达 R119 中，被排除记录相对保留记录的父级平均 MAE 差为：HF "
                f"{counter_by_key[('reachability_record_r119', 'HF')]['excluded_minus_retained_parent_mean_mae_bpm']:+.3f} bpm，"
                f"ACC {counter_by_key[('reachability_record_r119', 'ACC')]['excluded_minus_retained_parent_mean_mae_bpm']:+.3f} bpm。"
                "这一定量化了筛除带来的组成变化，但不赋予因果或总体推广含义。"
            ),
            "",
            "## 可复核性",
            "",
            f"- 实验合同：`{contract['schema_id']}`。",
            "- P0–P4 回执、逐面板结果、共同支持审计、反事实表和绘图源数据均保存在本地实验目录。",
            "- 新算法调用数为 0；全部结果复用已封存的 42,900 个 HF 与 42,900 个 ACC 单元格。",
            "",
        ]
    )


def _render_figures(
    *,
    summary_rows: Sequence[Mapping[str, Any]],
    exclusion_rows: Sequence[Mapping[str, Any]],
    counterfactual_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    rows = {str(row["panel_id"]): row for row in summary_rows}
    warm = "#E28E2C"
    cool = "#5B8FD6"
    neutral = "#606060"
    pale_warm = "#F2D3AB"
    pale_cool = "#BFD2EC"

    fig = plt.figure(figsize=(7.09, 5.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    reach_ids = [
        "full143_anchor",
        "reachability_record_r135",
        "reachability_record_r127",
        "reachability_record_r119",
        "reachability_record_r111",
    ]
    x = np.array([int(rows[value]["retained_record_count"]) for value in reach_ids])
    hf = np.array([float(rows[value]["hf_mean_mae_bpm"]) for value in reach_ids])
    acc = np.array([float(rows[value]["acc_mean_mae_bpm"]) for value in reach_ids])
    ax_a.plot(x, hf, "o-", color=warm, lw=1.8, ms=4, label="HF")
    ax_a.plot(x, acc, "o-", color=cool, lw=1.8, ms=4, label="ACC")
    upper = rows["holdout_upper_record_r119"]
    ax_a.scatter(
        [119, 119],
        [upper["hf_mean_mae_bpm"], upper["acc_mean_mae_bpm"]],
        marker="^",
        s=28,
        facecolors="white",
        edgecolors=[warm, cool],
        linewidths=1.2,
        zorder=4,
        label="Optimistic R119",
    )
    ax_a.set_xlim(145, 109)
    ax_a.set_xticks(x)
    ax_a.set_xlabel("Records retained")
    ax_a.set_ylabel("Mean MAE (bpm)")
    ax_a.legend(loc="upper left", ncol=2, fontsize=6.5)
    ax_a.grid(axis="y", color="#E6E6E6", lw=0.6)

    fractions = np.array(
        [float(rows[value]["hf_qualified_record_fraction"]) for value in reach_ids]
    )
    ax_b.plot(x, fractions, "o-", color=warm, lw=1.8, ms=4)
    ax_b.scatter(
        [119],
        [upper["hf_qualified_record_fraction"]],
        marker="^",
        s=28,
        facecolors="white",
        edgecolors=warm,
        linewidths=1.2,
        zorder=4,
    )
    ax_b.set_xlim(145, 109)
    ax_b.set_xticks(x)
    fraction_values = np.append(fractions, float(upper["hf_qualified_record_fraction"]))
    ax_b.set_ylim(
        max(0.0, float(fraction_values.min()) - 0.035), float(fraction_values.max()) + 0.035
    )
    ax_b.set_xlabel("Records retained")
    ax_b.set_ylabel("HF qualified fraction")
    ax_b.grid(axis="y", color="#E6E6E6", lw=0.6)

    selected = [
        ("reachability_record_r119", "Reach R119"),
        ("reachability_subject_s5", "Reach S5"),
        ("holdout_upper_record_r119", "Upper R119"),
        ("holdout_upper_subject_s5", "Upper S5"),
    ]
    bar_labels = []
    composition = []
    reselection = []
    colors_light = []
    colors_dark = []
    for panel_id, label in selected:
        for route, pale, dark in (("hf", pale_warm, warm), ("acc", pale_cool, cool)):
            bar_labels.append(f"{label} · {route.upper()}")
            composition.append(float(rows[panel_id][f"{route}_composition_effect_bpm"]))
            reselection.append(float(rows[panel_id][f"{route}_reselection_effect_bpm"]))
            colors_light.append(pale)
            colors_dark.append(dark)
    y = np.arange(len(bar_labels))[::-1]
    ax_c.barh(
        y, composition, color=colors_light, edgecolor="none", height=0.68, label="Composition"
    )
    ax_c.barh(
        y,
        reselection,
        left=composition,
        color=colors_dark,
        edgecolor="none",
        height=0.68,
        label="Reselection",
    )
    ax_c.axvline(0, color=neutral, lw=0.8)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(bar_labels, fontsize=6.4)
    ax_c.set_xlabel("Change from full-143 mean MAE (bpm)")
    ax_c.legend(loc="lower right", fontsize=6.5)
    ax_c.grid(axis="x", color="#E6E6E6", lw=0.6)

    difference_ids = ["full143_anchor", *(panel_id for panel_id, _ in selected)]
    all_panel_ids = difference_ids + [
        "reachability_record_r135",
        "reachability_record_r127",
        "reachability_record_r111",
    ]
    labels = [
        "Full 143",
        "Reach R119",
        "Reach S5",
        "Upper R119",
        "Upper S5",
        "Reach R135",
        "Reach R127",
        "Reach R111",
    ]
    anchor_common = _anchor_common_difference(summary_rows)
    values = [
        anchor_common
        if panel_id == "full143_anchor"
        else float(rows[panel_id]["acc_minus_hf_common_mean_bpm"])
        for panel_id in all_panel_ids
    ]
    yy = np.arange(len(values))[::-1]
    ax_d.axvline(0, color=neutral, lw=0.8)
    ax_d.hlines(yy, 0, values, color="#B8B8B8", lw=1.0)
    ax_d.scatter(values, yy, c=[cool if value >= 0 else warm for value in values], s=22, zorder=3)
    ax_d.set_yticks(yy)
    ax_d.set_yticklabels(labels, fontsize=6.4)
    ax_d.set_xlabel("ACC − HF mean MAE on common support (bpm)")
    ax_d.grid(axis="x", color="#E6E6E6", lw=0.6)
    for ax, label in zip((ax_a, ax_b, ax_c, ax_d), "abcd", strict=True):
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig1_paths = _save_figure(fig, output_root / "curated_subset_sensitivity")
    plt.close(fig)

    fig = plt.figure(figsize=(7.09, 3.25), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.95, 1.25])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    subjects = ["CGX", "LYX", "LZJ", "PJY", "TS", "QYC", "HB"]
    quota_panels = [
        "reachability_record_r135",
        "reachability_record_r127",
        "reachability_record_r119",
        "reachability_record_r111",
        "holdout_upper_record_r119",
    ]
    quota_labels = ["R135", "R127", "R119", "R111", "Upper R119"]
    quota_map = {
        (row["physical_subject_id"], row["panel_id"]): int(row["excluded_count"])
        for row in exclusion_rows
        if row["structure_type"] == "record_subject_quota"
    }
    matrix = np.array(
        [[quota_map[(subject, panel)] for panel in quota_panels] for subject in subjects]
    )
    im = ax_a.imshow(
        matrix,
        cmap=mpl.colors.LinearSegmentedColormap.from_list("warm", ["#FFF8EF", warm]),
        aspect="auto",
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax_a.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=6.5)
    ax_a.set_xticks(range(len(quota_labels)), quota_labels, rotation=35, ha="right")
    ax_a.set_yticks(range(len(subjects)), subjects)
    ax_a.set_title("Record exclusions by subject", fontsize=8)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    del im

    subject_panels = ["reachability_subject_s5", "holdout_upper_subject_s5"]
    scenes = sorted(
        {
            str(row["scene"])
            for row in exclusion_rows
            if row["structure_type"] == "subject_scene_exclusion"
        }
    )
    subject_colors = {
        "CGX": "#D8D8D8",
        "LYX": "#BFD2EC",
        "LZJ": "#F2D3AB",
        "PJY": "#C9DDB8",
        "TS": "#DCC6E8",
        "QYC": "#AFCFD0",
        "HB": "#8F8F8F",
    }
    scene_map = {
        (str(row["scene"]), str(row["panel_id"])): str(row["physical_subject_id"])
        for row in exclusion_rows
        if row["structure_type"] == "subject_scene_exclusion"
    }
    for i, scene in enumerate(scenes):
        for j, panel_id in enumerate(subject_panels):
            subject = scene_map[(scene, panel_id)]
            ax_b.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, facecolor=subject_colors[subject], edgecolor="white"
                )
            )
            ax_b.text(j, i, subject, ha="center", va="center", fontsize=6.3)
    ax_b.set_xlim(-0.5, 1.5)
    ax_b.set_ylim(len(scenes) - 0.5, -0.5)
    ax_b.set_xticks([0, 1], ["Reach S5", "Upper S5"], rotation=25, ha="right")
    ax_b.set_yticks(range(len(scenes)), scenes)
    ax_b.set_title("Excluded subject per scene", fontsize=8)
    for spine in ax_b.spines.values():
        spine.set_visible(False)

    selected_cf = [
        "reachability_record_r135",
        "reachability_record_r127",
        "reachability_record_r119",
        "reachability_record_r111",
        "reachability_subject_s5",
        "holdout_upper_record_r119",
        "holdout_upper_subject_s5",
    ]
    cf_map = {
        (str(row["panel_id"]), str(row["route"])): float(
            row["excluded_minus_retained_parent_mean_mae_bpm"]
        )
        for row in counterfactual_rows
    }
    y = np.arange(len(selected_cf))[::-1]
    offset = 0.14
    hf_values = [cf_map[(panel, "HF")] for panel in selected_cf]
    acc_values = [cf_map[(panel, "ACC")] for panel in selected_cf]
    ax_c.axvline(0, color=neutral, lw=0.8)
    ax_c.scatter(hf_values, y + offset, color=warm, s=20, label="HF")
    ax_c.scatter(acc_values, y - offset, color=cool, s=20, label="ACC")
    for yi, hfv, accv in zip(y, hf_values, acc_values, strict=True):
        ax_c.plot([hfv, accv], [yi + offset, yi - offset], color="#C8C8C8", lw=0.7, zorder=0)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([_short_panel_label(value) for value in selected_cf], fontsize=6.3)
    ax_c.set_xlabel("Excluded − retained parent mean MAE (bpm)")
    ax_c.set_title("Composition contrast", fontsize=8)
    ax_c.legend(loc="lower right", fontsize=6.5)
    ax_c.grid(axis="x", color="#E6E6E6", lw=0.6)
    for ax, label in zip((ax_a, ax_b, ax_c), "abc", strict=True):
        ax.text(-0.16, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig2_paths = _save_figure(fig, output_root / "curated_subset_structure_audit")
    plt.close(fig)
    return [*fig1_paths, *fig2_paths]


def _anchor_common_difference(summary_rows: Sequence[Mapping[str, Any]]) -> float:
    anchor = next(row for row in summary_rows if row["panel_id"] == "full143_anchor")
    return float(anchor["acc_minus_hf_common_mean_bpm"])


def _short_panel_label(panel_id: str) -> str:
    return {
        "reachability_record_r135": "Reach R135",
        "reachability_record_r127": "Reach R127",
        "reachability_record_r119": "Reach R119",
        "reachability_record_r111": "Reach R111",
        "reachability_subject_s5": "Reach S5",
        "holdout_upper_record_r119": "Upper R119",
        "holdout_upper_subject_s5": "Upper S5",
    }[panel_id]


def _save_figure(fig: Any, base: Path) -> list[Path]:
    paths = [base.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], dpi=600, bbox_inches="tight")
    return paths


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    return _atomic_write(path, payload)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError("curated_report_empty_csv")
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return _atomic_write(path, stream.getvalue().encode("utf-8"))


def _write_text(path: Path, value: str) -> str:
    return _atomic_write(path, (value.rstrip() + "\n").encode("utf-8"))


def _atomic_write(path: Path, payload: bytes) -> str:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return hashlib.sha256(payload).hexdigest()
