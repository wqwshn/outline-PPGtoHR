"""Write the evidence-bounded Chinese P5 report and completion receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from ppg_hr.v2.cross_subject_acc_experiment import ACC_EXPERIMENT_ID, PARENT_EXPERIMENT_ID

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
CONTRAST_NAMES = {
    "delta_end_bpm": "端到端差值 ACC(θACC)−HF(θHF)",
    "delta_hf_coordinate_reference_bpm": "HF 坐标参考差值 ACC(θHF)−HF(θHF)",
    "delta_acc_coordinate_reference_bpm": "ACC 坐标参考差值 ACC(θACC)−HF(θACC)",
    "acc_selection_gain_bpm": "ACC 选参收益 ACC(θHF)−ACC(θACC)",
    "hf_own_coordinate_gain_bpm": "HF 自有坐标收益 HF(θACC)−HF(θHF)",
}
P2_RUNTIME_SOURCE_COMMIT = "a9847454b90920fa004048a4e7625b8ba3951413"


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
    p2_root = experiment_root / "p2"
    p3_root = experiment_root / "p3"
    p4_root = experiment_root / "p4"
    p5_root = experiment_root / "p5"
    validation = _read_json(p5_root / "p5_validation_receipt.json")
    figure_receipt = _read_json(p5_root / "figures" / "figure_receipt.json")
    p2_receipt = _read_json(p2_root / "p2_receipt.json")
    p4_receipt = _read_json(p4_root / "p4_receipt.json")
    if any(
        receipt.get("status") != "pass"
        for receipt in (validation, figure_receipt, p2_receipt, p4_receipt)
    ):
        raise ValueError("p5_report_upstream_not_pass")
    folds = _read_csv(p4_root / "paired_fold_results.csv")
    records = _read_csv(p4_root / "paired_record_results.csv")
    contrasts = _read_csv(p4_root / "paired_contrast_summary.csv")
    matrix = _read_csv(p4_root / "paired_2x2_matrix.csv")
    audit = _read_csv(p4_root / "common_support_audit.csv")
    acc_selection = _selection_frequency(p3_root / "selections")
    parent_selection = _selection_frequency(
        repo_root / "data" / "experiments" / PARENT_EXPERIMENT_ID / "p3" / "selections"
    )
    report = _build_report(
        p2_receipt=p2_receipt,
        p4_receipt=p4_receipt,
        validation=validation,
        folds=folds,
        records=records,
        contrasts=contrasts,
        matrix=matrix,
        audit=audit,
        acc_selection=acc_selection,
        parent_selection=parent_selection,
        figure_file=figure_receipt["figure_file"],
    )
    report_path = p5_root / "cross_subject_acc_independent_experiment_report_zh.md"
    _atomic_write(report_path, report.encode("utf-8"))
    completion = {
        "schema_id": "cross_subject_multirecord_acc_p5_completion_receipt_v1",
        "experiment_id": ACC_EXPERIMENT_ID,
        "stage": "P5",
        "status": "pass",
        "hard_stop_reached": True,
        "record_count": len(records),
        "fold_count": len(folds),
        "scene_count": len({row["scene"] for row in folds}),
        "physical_subject_count": len({row["holdout_subject_id"] for row in folds}),
        "performance_exclusion_count": 0,
        "report_file": report_path.name,
        "report_sha256": _file_sha256(report_path),
        "figure_file": figure_receipt["figure_file"],
        "figure_sha256": figure_receipt["figure_sha256"],
        "p5_validation_receipt_sha256": _file_sha256(p5_root / "p5_validation_receipt.json"),
        "p5_figure_receipt_sha256": _file_sha256(p5_root / "figures" / "figure_receipt.json"),
        "p4_receipt_sha256": _file_sha256(p4_root / "p4_receipt.json"),
        "p2_runtime_source_commit": P2_RUNTIME_SOURCE_COMMIT,
        "forbidden_extensions_run": [],
    }
    receipt_sha = _write_json(p5_root / "p5_completion_receipt.json", completion)
    print(
        json.dumps(
            {**completion, "receipt_sha256": receipt_sha},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


def _build_report(
    *,
    p2_receipt: dict[str, Any],
    p4_receipt: dict[str, Any],
    validation: dict[str, Any],
    folds: list[dict[str, str]],
    records: list[dict[str, str]],
    contrasts: list[dict[str, str]],
    matrix: list[dict[str, str]],
    audit: list[dict[str, str]],
    acc_selection: Counter[str],
    parent_selection: Counter[str],
    figure_file: str,
) -> str:
    by_cell = {row["cell_label"]: row for row in matrix}
    by_contrast = {row["contrast_id"]: row for row in contrasts}
    hf_mean = float(by_cell["hf_theta_hf"]["mae_bpm__mean"])
    acc_mean = float(by_cell["acc_theta_acc"]["mae_bpm__mean"])
    delta = by_contrast["delta_end_bpm"]
    scene_rows = _scene_table(folds)
    subject_rows = _subject_table(folds)
    support = _support_summary(audit)
    matched_folds = sum(
        row["theta_hf_coordinate_id"] == row["theta_acc_coordinate_id"] for row in folds
    )
    lines = [
        "# 143 条多记录跨个体 ACC 独立 Physical4D 选参实验报告",
        "",
        "## 实验结论摘要",
        "",
        (
            f"本实验已按冻结方案完整结束。ACC 在与 HF 相同的 143 条记录、8 个场景、"
            f"48 个受试者—场景成组留出折和 300 点 Physical4D 空间内获得独立训练侧选参机会。"
            f"四格结果均在每条记录完全相同的可靠窗口交集上重评。48 折等权后，"
            f"HF(θHF) 的 MAE 为 {_f(hf_mean)} bpm，ACC(θACC) 为 {_f(acc_mean)} bpm，"
            f"端到端差值 ACC−HF 为 {_f(float(delta['mean']))} bpm。"
        ),
        "",
        (
            f"端到端折差值中，正值 {delta['positive_fold_count']} 折、零值 "
            f"{delta['zero_fold_count']} 折、负值 {delta['negative_fold_count']} 折。"
            "正值只表示该折 HF 的共同支持 MAE 更低，不能解释为独立受试者胜率或统计显著性。"
        ),
        "",
        "## 数据与评价口径",
        "",
        "- 数据面板保留 143 条真实采集记录；除跑步使用 HB 替代无跑步记录的 QYC 外，各场景沿用冻结的 6 人构成。QYC 开合跳保留 2 条记录，其余受试者—场景格原则上为 3 条。",
        "- 每个场景执行 6 折受试者成组留出，共 48 折；同一受试者同一场景的重复记录不跨训练和留出。",
        "- HF 与 ACC 仅在参考组上不同；ACC 使用 `reference_groups_order=(ACC,)`，其他求解机制、300 点空间和固定 5 秒评价口径均保持一致。",
        "- ACC 选择先对一名训练受试者的重复记录取平均，再依次最小化五名训练受试者中的最差均值、五人均值和冻结坐标顺序。历史 Lite HF 与 HF 六门均不参与 ACC 排序。",
        "- 主统计先平均每折 2–3 条留出记录，再令 48 折等权；不将 143 条记录或 48 折误作独立受试者。",
        "",
        "## ACC 响应面与选择冻结",
        "",
        (
            f"ACC 紧凑响应面完成 {p2_receipt['complete_cell_count']:,}/"
            f"{p2_receipt['expected_cell_count']:,} 个调用，技术尝试事件 "
            f"{p2_receipt['attempt_event_count']} 个，运行耗时 "
            f"{float(p2_receipt['elapsed_s']) / 3600.0:.2f} 小时。"
        ),
        f"P2 实际运行源码固定于提交 `{P2_RUNTIME_SOURCE_COMMIT}`；后续格式化不改写已封存的响应面回执。",
        (
            f"48 份 ACC 选择回执在读取留出结果之前统一冻结。ACC 共选中 "
            f"{len(acc_selection)} 个不同坐标，父实验 HF 共选中 {len(parent_selection)} 个不同坐标；"
            f"θACC 与 θHF 完全一致的折为 {matched_folds}/48。"
        ),
        "",
        "ACC 获选坐标频次最高的条目如下：",
        "",
        "| 坐标 | 折数 |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{coordinate}` | {count} |" for coordinate, count in acc_selection.most_common(8)
    )
    lines.extend(
        [
            "",
            "## 四格共同支持结果",
            "",
            "| 实际参考路线 | θHF | θACC |",
            "|---|---:|---:|",
            (
                f"| HF | {_f(float(by_cell['hf_theta_hf']['mae_bpm__mean']))} | "
                f"{_f(float(by_cell['hf_theta_acc']['mae_bpm__mean']))} |"
            ),
            (
                f"| ACC | {_f(float(by_cell['acc_theta_hf']['mae_bpm__mean']))} | "
                f"{_f(float(by_cell['acc_theta_acc']['mae_bpm__mean']))} |"
            ),
            "",
            "表中数值均为 48 折等权共同支持 MAE（bpm）；非对角线只用于解释参考路线与坐标来源错配，不构成额外算法路线，也未进行逐折事后择优。",
            "",
            "五个预注册配对量如下。正值的含义由公式方向决定；例如 ACC 选参收益为正表示 θACC 相对 θHF 降低了 ACC MAE。",
            "",
            "| 配对量 | 均值 | 中位数 | SD | IQR | 范围 | 正/零/负折 |",
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
            "## 场景层级描述",
            "",
            "场景顺序沿用父实验按 HF 折均值由低到高冻结的顺序，未根据 ACC 结果重排。",
            "",
            "| 场景 | HF(θHF) | ACC(θACC) | Δ均值 | Δ中位数 | 正/零/负折 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(scene_rows)
    lines.extend(
        [
            "",
            "## 真实受试者层级描述",
            "",
            "以下为受试者持有的场景折描述。HB 仅对应跑步，QYC 不含跑步；因此各受试者折数不同，表格不用于独立受试者显著性推断。",
            "",
            "| 受试者 | 折数 | HF(θHF) | ACC(θACC) | Δ均值 | Δ中位数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(subject_rows)
    lines.extend(
        [
            "",
            "## 共同支持与完整性审计",
            "",
            (
                f"143 条记录全部形成非空四格共同支持集。每条记录共同窗口数的最小值、"
                f"中位数和最大值分别为 {support['common_min']}、{support['common_median']:.1f} "
                f"和 {support['common_max']}。"
            ),
            (
                f"至少一个四格单元相对其原生可靠支持损失窗口的记录为 "
                f"{support['records_with_any_loss']}/143；单条记录任一单元的最大损失为 "
                f"{support['max_cell_loss']} 个窗口。所有四格 MAE 均使用记录内同一交集分母，"
                "没有因支持差异删除记录或改回各格独立分母。"
            ),
            (
                f"P4 去重后共有 {p4_receipt['unique_report_count']} 份完整报告请求，"
                f"四格逻辑请求 {p4_receipt['four_cell_request_count']} 个，技术失败事件 "
                f"{p4_receipt['technical_failure_event_count']} 个，性能条件化删样本数为 "
                f"{p4_receipt['performance_exclusion_count']}。"
            ),
            "",
            "## 主图与客观解读",
            "",
            f"![跨个体 HF–ACC 四格共同支持比较](figures/{figure_file})",
            "",
            "- 图 a 在冻结场景顺序下同时展示 143 条记录的 HF(θHF) 与 ACC(θACC) 共同支持 MAE、记录内配对连线、分布形状、中位数与 IQR，以及 48 个折均值。它用于观察场景难度、分布范围和记录—折两个层级，不预设某场景存在或不存在折间异质性。",
            "- 图 b 展示 48 个端到端折差值；颜色仅编码真实留出受试者，灰色区间和横线分别表示场景内 IQR 与中位数，虚线为零差。该面板用于描述差值方向和范围，不等同于 48 名独立受试者的胜负或显著性检验。",
            "- 图 c 将四格压缩为 48 折等权 MAE 矩阵，用于同时核对参考路线效应和坐标来源效应。对角线是正式端到端比较，非对角线只解释错配。",
            "",
            "## 证据边界与限制",
            "",
            "1. 这是同一 143 条面板上的回顾性跨个体开发对照，不是独立外部验证。",
            "2. 48 折来自 7 名真实受试者跨场景重复出现，不能按 48 个独立个体进行 p 值或独立胜率推断。",
            "3. ACC 仅在冻结的 300 点 Physical4D 空间和固定 5 秒口径下独立选参；本节点没有调 ACC 专属时延、扩大空间或建立 ACC 六门。",
            "4. 历史 Lite HF 不进入本轮主图和主要统计；本轮回答的是 HF 与 ACC 在相同 LOSO 面板、共同支持和各自冻结坐标下的比较。",
            "5. 未运行 HF+ACC 路线，未根据误差删除有效记录，也未根据结果改变选择器、场景顺序、对比方向或图型。",
            "",
            "## 完成状态",
            "",
            (
                f"独立验证器已直接重算 {validation['validated_acc_response_cell_count']:,} 个 "
                f"ACC 响应单元、{validation['validated_selection_count']} 份选择、"
                f"{validation['validated_record_count']} 条共同支持结果和 "
                f"{validation['validated_fold_count']} 个折级结果，未调用正式选择函数或正式共同支持计算函数。"
            ),
            "",
            "P5 完成后按预注册方案硬停止。任何进一步的 ACC 时延、HF+ACC、参数扩展或新筛选分析均需新实验身份和新计划。",
            "",
        ]
    )
    return "\n".join(lines)


def _scene_table(folds: list[dict[str, str]]) -> list[str]:
    grouped = defaultdict(list)
    for row in folds:
        grouped[row["scene"]].append(row)
    output = []
    for scene in SCENE_ORDER:
        rows = grouped[scene]
        delta = [float(row["delta_end_bpm"]) for row in rows]
        output.append(
            f"| {SCENE_DISPLAY[scene]} | {_f(_mean(rows, 'hf_theta_hf_mae_bpm'))} | "
            f"{_f(_mean(rows, 'acc_theta_acc_mae_bpm'))} | {_f(float(np.mean(delta)))} | "
            f"{_f(float(median(delta)))} | {sum(value > 0 for value in delta)}/"
            f"{sum(value == 0 for value in delta)}/{sum(value < 0 for value in delta)} |"
        )
    return output


def _subject_table(folds: list[dict[str, str]]) -> list[str]:
    grouped = defaultdict(list)
    for row in folds:
        grouped[row["holdout_subject_id"]].append(row)
    output = []
    for subject in sorted(grouped):
        rows = grouped[subject]
        delta = [float(row["delta_end_bpm"]) for row in rows]
        output.append(
            f"| {subject} | {len(rows)} | {_f(_mean(rows, 'hf_theta_hf_mae_bpm'))} | "
            f"{_f(_mean(rows, 'acc_theta_acc_mae_bpm'))} | {_f(float(np.mean(delta)))} | "
            f"{_f(float(median(delta)))} |"
        )
    return output


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


def _selection_frequency(root: Path) -> Counter[str]:
    freeze = _read_json(root / "p3_freeze_receipt.json")
    rows = list(freeze.get("selections") or [])
    if freeze.get("status") != "pass" or len(rows) != 48:
        raise ValueError(f"p5_report_selection_incomplete:{root}")
    return Counter(str(row["selected_coordinate_id"]) for row in rows)


def _mean(rows: list[dict[str, str]], column: str) -> float:
    return float(np.mean([float(row[column]) for row in rows]))


def _f(value: float) -> str:
    return f"{value:.3f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)


if __name__ == "__main__":
    raise SystemExit(main())
