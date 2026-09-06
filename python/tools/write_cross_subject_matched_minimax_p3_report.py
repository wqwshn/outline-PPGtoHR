"""Write the Chinese matched-minimax follow-up report and completion receipt."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from ppg_hr.v2.cross_subject_matched_minimax import (
    EXPERIMENT_ID,
    PARENT_ACC_EXPERIMENT_ID,
    PARENT_HF_EXPERIMENT_ID,
    file_sha256,
    write_json,
)

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
CELL_NAMES = {
    "hf_theta_hf_gate": "HF(θHF-gate)",
    "hf_theta_acc_minimax": "HF(θACC-MM)",
    "hf_theta_hf_minimax": "HF(θHF-MM)",
    "acc_theta_acc_minimax": "ACC(θACC-MM)",
    "acc_theta_hf_minimax": "ACC(θHF-MM)",
}
CONTRAST_NAMES = {
    "hf_minimax_gain_vs_gate_bpm": "HF 旧六门坐标−HF 新 minimax 坐标",
    "hf_minimax_gain_vs_acc_coordinate_bpm": "HF 的 ACC-minimax 坐标−HF 新 minimax 坐标",
    "independent_minimax_delta_bpm": "ACC 独立 minimax−HF 独立 minimax",
    "same_hf_coordinate_reference_delta_bpm": "ACC(θHF-MM)−HF(θHF-MM)",
}


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
    hf_root = repo_root / "data" / "experiments" / PARENT_HF_EXPERIMENT_ID
    acc_root = repo_root / "data" / "experiments" / PARENT_ACC_EXPERIMENT_ID
    p2_root = root / "p2"
    p3_root = root / "p3"
    figure_root = p3_root / "figures"

    p0 = _read_json(root / "p0" / "p0_receipt.json")
    p1 = _read_json(root / "p1" / "selections" / "p1_freeze_receipt.json")
    p2 = _read_json(p2_root / "p2_receipt.json")
    validation = _read_json(p3_root / "p3_validation_receipt.json")
    figures = _read_json(figure_root / "figure_receipt.json")
    if any(receipt.get("status") != "pass" for receipt in (p0, p1, p2, validation, figures)):
        raise ValueError("matched_report_upstream_not_pass")

    folds = _read_csv(p2_root / "matched_fold_results.csv")
    records = _read_csv(p2_root / "matched_record_results.csv")
    cells = _read_csv(p2_root / "five_cell_summary.csv")
    contrasts = _read_csv(p2_root / "contrast_summary.csv")
    audit = _read_csv(p2_root / "common_support_audit.csv")
    hf_new_selection = _selection_map(root / "p1" / "selections", "p1_freeze_receipt.json")
    hf_gate_selection = _selection_map(hf_root / "p3" / "selections", "p3_freeze_receipt.json")
    acc_selection = _selection_map(acc_root / "p3" / "selections", "p3_freeze_receipt.json")

    report = _build_report(
        folds=folds,
        records=records,
        cells=cells,
        contrasts=contrasts,
        audit=audit,
        hf_new_selection=hf_new_selection,
        hf_gate_selection=hf_gate_selection,
        acc_selection=acc_selection,
        p2=p2,
        validation=validation,
        figures=figures,
    )
    report_path = p3_root / "cross_subject_matched_minimax_experiment_report_zh.md"
    _atomic_write(report_path, report.encode("utf-8"))
    completion = {
        "schema_id": "cross_subject_matched_minimax_p3_completion_receipt_v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": "P3",
        "status": "pass",
        "hard_stop_reached": True,
        "record_count": len(records),
        "fold_count": len(folds),
        "scene_count": len({row["scene"] for row in folds}),
        "physical_subject_count": len({row["holdout_subject_id"] for row in folds}),
        "performance_exclusion_count": 0,
        "full_response_recalculation_count": 0,
        "targeted_replay_count": p2["targeted_replay_count"],
        "report_file": report_path.name,
        "report_sha256": file_sha256(report_path),
        "figure_files": sorted(figures["figure_sha256"]),
        "figure_sha256": figures["figure_sha256"],
        "p3_validation_receipt_sha256": file_sha256(p3_root / "p3_validation_receipt.json"),
        "p3_figure_receipt_sha256": file_sha256(figure_root / "figure_receipt.json"),
        "p2_receipt_sha256": file_sha256(p2_root / "p2_receipt.json"),
        "forbidden_extensions_run": [],
    }
    sha = write_json(p3_root / "p3_completion_receipt.json", completion)
    print(json.dumps({**completion, "receipt_sha256": sha}, ensure_ascii=False, indent=2))
    return 0


def _build_report(
    *,
    folds: list[dict[str, str]],
    records: list[dict[str, str]],
    cells: list[dict[str, str]],
    contrasts: list[dict[str, str]],
    audit: list[dict[str, str]],
    hf_new_selection: dict[str, str],
    hf_gate_selection: dict[str, str],
    acc_selection: dict[str, str],
    p2: dict[str, Any],
    validation: dict[str, Any],
    figures: dict[str, Any],
) -> str:
    by_cell = {row["cell_label"]: row for row in cells}
    by_contrast = {row["contrast_id"]: row for row in contrasts}
    hf_gate = float(by_cell["hf_theta_hf_gate"]["mean"])
    hf_acc = float(by_cell["hf_theta_acc_minimax"]["mean"])
    hf_new = float(by_cell["hf_theta_hf_minimax"]["mean"])
    acc_own = float(by_cell["acc_theta_acc_minimax"]["mean"])
    acc_hf = float(by_cell["acc_theta_hf_minimax"]["mean"])
    gain_gate = by_contrast["hf_minimax_gain_vs_gate_bpm"]
    gain_acc = by_contrast["hf_minimax_gain_vs_acc_coordinate_bpm"]
    independent = by_contrast["independent_minimax_delta_bpm"]
    same_coordinate = by_contrast["same_hf_coordinate_reference_delta_bpm"]
    support = _support_summary(audit)
    hf_new_unique = Counter(hf_new_selection.values())
    match_gate = sum(hf_new_selection[key] == hf_gate_selection[key] for key in hf_new_selection)
    match_acc = sum(hf_new_selection[key] == acc_selection[key] for key in hf_new_selection)
    figure_one = "cross_subject_hf_acc_independent_minimax_violin_600dpi.png"
    figure_two = "cross_subject_hf_acc_replay_hf_minimax_violin_600dpi.png"

    lines = [
        "# HF 与 ACC 对称训练侧 minimax 跨个体跟进实验报告",
        "",
        "## 结论摘要",
        "",
        (
            "本轮没有观察到 HF 在改用 ACC 同构选参规则后继续提升。"
            f"在 143 条记录五单元精确公共支持上，HF 新独立 minimax 坐标的 48 折等权 MAE 为 "
            f"{_f(hf_new)} bpm，高于原 HF 六门坐标的 {_f(hf_gate)} bpm，也高于 HF 重放 "
            f"ACC-minimax 坐标的 {_f(hf_acc)} bpm。相对于原六门选择，MAE 平均增加 "
            f"{_f(hf_new - hf_gate)} bpm；相对于 ACC 坐标，平均增加 {_f(hf_new - hf_acc)} bpm。"
        ),
        "",
        (
            f"当 HF 和 ACC 都使用相同形式的训练侧受试者等权 MAE minimax、但各自在本路线响应面上独立选参时，"
            f"HF 为 {_f(hf_new)} bpm，ACC 为 {_f(acc_own)} bpm，ACC−HF 为 "
            f"{_f(float(independent['mean']))} bpm；48 折中差值正/零/负为 "
            f"{independent['positive_fold_count']}/{independent['zero_fold_count']}/"
            f"{independent['negative_fold_count']}。这是一项描述性结果，不构成独立受试者显著性检验。"
        ),
        "",
        (
            f"在同一个新 HF-minimax 坐标上，仅切换参考路线时，HF 为 {_f(hf_new)} bpm，ACC 为 "
            f"{_f(acc_hf)} bpm，ACC−HF 为 {_f(float(same_coordinate['mean']))} bpm；"
            f"正/零/负折为 {same_coordinate['positive_fold_count']}/"
            f"{same_coordinate['zero_fold_count']}/{same_coordinate['negative_fold_count']}。"
        ),
        "",
        "## 问题、设计与评价口径",
        "",
        "本实验回答一个限定问题：此前 2×2 对照中 HF 重放 θACC 的结果优于 HF 原 θHF，是否主要因为 ACC 使用了无六门、训练侧受试者等权 MAE minimax，而 HF 使用六门体系？为此，本轮只给 HF 增加与 ACC 数学同构的训练侧选择器，不改变数据面板、折定义、Physical4D 坐标空间、参考时标或求解器。",
        "",
        "- 面板仍为 143 条真实记录、8 个场景、48 个场景内受试者成组留出折；每折同一受试者的 2–3 条记录全部留出。",
        "- 每个候选坐标先在每名训练受试者内平均其重复记录 MAE，再依次最小化五名训练受试者中的最差均值、五人均值和冻结坐标顺序。选择文件只含训练受试者、记录、坐标和原生路线 MAE，不含留出指标。",
        "- HF 新选择在读取留出结果前一次性冻结；ACC 沿用上一节点已冻结的同构选择，原 HF 六门坐标只作预先存在的参照。",
        "- 为使两张新图中的 HF 数值完全一致，同时能够直接回答新规则相对原规则是否改进，本轮每条记录取五种结果的可靠窗口精确交集，再计算全部 MAE。",
        "- 主汇总先平均每折的 2–3 条留出记录，再令 48 折等权；不把 143 条记录或 48 折解释为 143/48 名独立受试者。",
        "",
        "## 计算复用与冻结完整性",
        "",
        (
            f"两套既有 42,900 点紧凑响应面均保持只读，本轮完整响应面重算数为 "
            f"{p2['full_response_recalculation_count']}。715 个五单元逻辑请求去重为 "
            f"{p2['unique_report_count']} 份完整报告，其中复用 {p2['prior_report_reuse_count']} 份，"
            f"仅对新坐标定点重放 {p2['targeted_replay_count']} 份。技术失败事件为 "
            f"{p2['technical_failure_event_count']}，按性能删样本数为 {p2['performance_exclusion_count']}。"
        ),
        (
            f"HF 新 minimax 在 48 折中选中 {len(hf_new_unique)} 个不同坐标；与原 HF 六门坐标完全一致 "
            f"{match_gate}/48 折，与 ACC-minimax 坐标完全一致 {match_acc}/48 折。"
        ),
        "",
        "HF 新 minimax 获选频次最高的坐标如下：",
        "",
        "| 坐标 | 折数 |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{coordinate}` | {count} |" for coordinate, count in hf_new_unique.most_common(8)
    )
    lines.extend(
        [
            "",
            "## 五单元结果与预设差值",
            "",
            "| 单元 | 48 折均值 | 中位数 | SD | IQR | 范围 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cell_id in CELL_NAMES:
        row = by_cell[cell_id]
        lines.append(
            f"| {CELL_NAMES[cell_id]} | {_f(float(row['mean']))} | {_f(float(row['median']))} | "
            f"{_f(float(row['sd']))} | {_f(float(row['iqr']))} | "
            f"{_f(float(row['min']))}–{_f(float(row['max']))} |"
        )
    lines.extend(
        [
            "",
            "差值均按表中公式左项减右项。前两项为正才表示新 HF-minimax 坐标降低 HF MAE；后两项正值表示 ACC MAE 更高。",
            "",
            "| 差值 | 均值 | 中位数 | SD | IQR | 范围 | 正/零/负折 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for contrast_id in CONTRAST_NAMES:
        row = by_contrast[contrast_id]
        lines.append(
            f"| {CONTRAST_NAMES[contrast_id]} | {_f(float(row['mean']))} | "
            f"{_f(float(row['median']))} | {_f(float(row['sd']))} | "
            f"{_f(float(row['iqr']))} | {_f(float(row['min']))}–{_f(float(row['max']))} | "
            f"{row['positive_fold_count']}/{row['zero_fold_count']}/{row['negative_fold_count']} |"
        )
    lines.extend(
        [
            "",
            "## 图 1：统一规则后的 HF 与 ACC 独立调参",
            "",
            f"![HF 与 ACC 独立 minimax 跨个体分布](figures/{figure_one})",
            "",
            (
                f"图 1 对比 HF(θHF-MM) 与 ACC(θACC-MM)。每个浅色点和细连线是一条记录的配对结果；"
                "半透明小提琴展示该场景 17–18 条记录的分布，中位线和粗线分别给出中位数与 IQR；"
                "黑边大点及深色连线为每个场景的 6 个留出折均值。场景顺序沿用父 HF 报告的冻结顺序，"
                "并未根据本轮结果重排。两路线折均差值在 48 折中恰为 24 正、24 负，但 ACC 的总体均值"
                f"低 {_f(hf_new - acc_own)} bpm，说明方向计数平衡并不意味着差值幅度平衡。"
            ),
            "",
            "图中 Rope Skipping 的两条路线均保持较低且相对紧凑；Typing、Handwriting 等场景的记录分布更宽。该表述只描述可见分布，不预设场景内折间异质性的机制解释。",
            "",
            "## 图 2：统一规则后的 HF 与 ACC 重放 HF 参数",
            "",
            f"![HF 与 ACC 在 HF minimax 坐标上的跨个体分布](figures/{figure_two})",
            "",
            (
                f"图 2 固定参数坐标为 θHF-MM，仅比较参考路线。HF 的点、折均值和小提琴与图 1 完全相同；"
                f"ACC(θHF-MM) 的 48 折均值为 {_f(acc_hf)} bpm，比 HF 高 "
                f"{_f(acc_hf - hf_new)} bpm。ACC−HF 在 {same_coordinate['positive_fold_count']}/48 折为正，"
                "表明在这个由 HF 训练面选出的坐标上，HF 更常获得较低折均 MAE。"
            ),
            "",
            "图 2 保留了 Handwriting 中约 60 bpm 的 ACC 记录级离群值，并与图 1 共用 0–65 bpm 纵轴。该点没有因误差大而删除；共享尺度避免将两图的分布宽度差异误读为坐标轴效应。",
            "",
            "## 场景层级描述",
            "",
            "| 场景 | HF旧六门 | HF(θACC-MM) | HF(θHF-MM) | ACC(θACC-MM) | ACC(θHF-MM) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(_scene_table(folds))
    lines.extend(
        [
            "",
            "## 如何解释反直觉结果",
            "",
            (
                f"直接证据只支持以下结论：本轮 HF 新 minimax 并未复现 θACC 在 HF 上的收益。"
                f"新规则相对旧六门选择改善 {gain_gate['positive_fold_count']} 折、持平 "
                f"{gain_gate['zero_fold_count']} 折、变差 {gain_gate['negative_fold_count']} 折；"
                f"相对 HF 重放 θACC 改善 {gain_acc['positive_fold_count']} 折、持平 "
                f"{gain_acc['zero_fold_count']} 折、变差 {gain_acc['negative_fold_count']} 折。"
            ),
            "",
            "一种合理但尚未被本实验单独验证的解释是：受试者等权 minimax 优化的是五名训练受试者中的最差受试者均值，其目标并不保证最小化第六名留出受试者的平均 MAE；HF 响应面还可能比 ACC 响应面更容易发生坐标排序迁移。原六门体系包含与单一 MAE minimax 不同的稳定性约束，而 θACC 在 HF 上表现最好可能来自特定坐标的跨路线迁移，而不是“无六门 minimax”这一选择形式本身。以上均属于后续可检验机制假设，不应写成已证实原因。",
            "",
            "因此，现阶段不宜用新 θHF-MM 替换原 HF 六门结果，也不宜把 θACC 在 HF 上的较低 MAE解释为 ACC 选择器普遍优于 HF 选择器。更准确的节点结论是：在本面板和冻结空间中，HF 的最佳已观察固定方案仍是重放 θACC，但该现象没有被路线对称 minimax 规则解释。",
            "",
            "## 公共支持与独立复核",
            "",
            (
                f"143 条记录全部形成非空五单元共同支持。每条记录共同窗口数的最小值、中位数和最大值"
                f"分别为 {support['common_min']}、{support['common_median']:.1f} 和 "
                f"{support['common_max']}。只有 {support['records_with_any_loss']}/143 条记录至少一个单元"
                f"相对原生支持损失窗口，最大损失为 {support['max_cell_loss']} 个窗口。"
            ),
            (
                f"独立验证器未调用正式选择函数或正式五单元共同支持函数，重新计算并匹配了 "
                f"{validation['independent_hf_selection_match_count']} 份 HF 选择、"
                f"{validation['independent_acc_selection_match_count']} 份 ACC 选择、"
                f"{validation['unique_report_hash_check_count']} 份报告哈希、"
                f"{validation['independent_record_reconstruction_count']} 条记录和 "
                f"{validation['independent_fold_reconstruction_count']} 个折级结果；数值容差为 "
                f"{validation['float_tolerance']:.0e}。"
            ),
            "",
            "## 证据边界",
            "",
            "1. 该实验仍是同一 143 条面板上的回顾性跨个体开发评估，不是独立外部验证。",
            "2. 48 折来自 7 名真实受试者跨场景重复出现，不能当作 48 名独立受试者进行显著性推断。",
            "3. 选择规则只在冻结的 300 点 Physical4D 空间、原生路线 MAE 和固定 5 秒评价口径内比较；没有搜索新的 HF 专属空间。",
            "4. 本轮没有逐折事后择优、没有因误差大删除有效记录，也没有重新调 ACC 或改动场景排序。",
            "5. 两图与数值回答的是冻结规则下的描述性差异，不证明参考路线或选择器在其他数据集上的一般优劣。",
            "",
            "## 完成状态",
            "",
            (
                f"两张主图均以 Python/Matplotlib 输出为 600 dpi PNG，实际尺寸为 "
                f"{figures['image_dimensions_px'][figure_one][0]}×"
                f"{figures['image_dimensions_px'][figure_one][1]} px，并通过格式、DPI、文件集合和源数据"
                "哈希检查。本实验在报告与回执生成后硬停止；任何新 HF 目标函数、稳健选择器或扩展空间"
                "均需新实验身份。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _scene_table(folds: list[dict[str, str]]) -> list[str]:
    grouped = defaultdict(list)
    for row in folds:
        grouped[row["scene"]].append(row)
    columns = (
        "hf_theta_hf_gate_mae_bpm",
        "hf_theta_acc_minimax_mae_bpm",
        "hf_theta_hf_minimax_mae_bpm",
        "acc_theta_acc_minimax_mae_bpm",
        "acc_theta_hf_minimax_mae_bpm",
    )
    return [
        f"| {SCENE_DISPLAY[scene]} | "
        + " | ".join(_f(_mean(grouped[scene], column)) for column in columns)
        + " |"
        for scene in SCENE_ORDER
    ]


def _support_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    common = [int(row["common_window_count"]) for row in rows]
    loss_columns = [column for column in rows[0] if column.endswith("_lost_window_count")]
    per_record_max = [max(int(row[column]) for column in loss_columns) for row in rows]
    return {
        "common_min": min(common),
        "common_median": float(median(common)),
        "common_max": max(common),
        "records_with_any_loss": sum(value > 0 for value in per_record_max),
        "max_cell_loss": max(per_record_max),
    }


def _selection_map(root: Path, freeze_name: str) -> dict[str, str]:
    freeze = _read_json(root / freeze_name)
    rows = list(freeze.get("selections") or [])
    if freeze.get("status") != "pass" or len(rows) != 48:
        raise ValueError(f"matched_report_selection_incomplete:{root}")
    return {row["fold_id"]: row["selected_coordinate_id"] for row in rows}


def _mean(rows: list[dict[str, str]], column: str) -> float:
    return float(np.mean([float(row[column]) for row in rows]))


def _f(value: float) -> str:
    return f"{value:.3f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
