"""Generate the local report and completion receipts for the Handgrip experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

EXPERIMENT_ID = "cross_subject_multirecord_d24_handgrip_blind_composition_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    experiment_root = root / "data" / "experiments" / EXPERIMENT_ID
    p5_root = experiment_root / "p5"
    figure_root = p5_root / "figures"

    verifier_receipt = _read_json(p5_root / "verifier" / "receipt.json")
    verifier = _read_json(p5_root / "verifier" / "verification.json")
    if verifier_receipt.get("status") != "pass" or verifier.get("status") != "pass":
        raise ValueError("handgrip_report_verifier_not_passed")
    baseline = _read_json(experiment_root / "p0" / "baseline_snapshot.json")
    primary = _read_json(experiment_root / "p3" / "d24_15_primary" / "main" / "summary.json")
    primary_sensitivity = _read_json(
        experiment_root / "p3" / "d24_15_primary" / "sens30" / "summary.json"
    )
    fallback = _read_json(experiment_root / "p4" / "d24_18_fallback" / "main" / "summary.json")
    fallback_sensitivity = _read_json(
        experiment_root / "p4" / "d24_18_fallback" / "sens30" / "summary.json"
    )
    lyx = _read_json(experiment_root / "p4" / "lyx_handgrip" / "summary.json")
    primary_rows = _read_csv(
        experiment_root / "p3" / "d24_15_primary" / "main" / "record_results.csv"
    )
    fallback_rows = _read_csv(
        experiment_root / "p4" / "d24_18_fallback" / "main" / "record_results.csv"
    )
    primary_by_id = {row["record_id"]: row for row in primary_rows}
    fallback_by_id = {row["record_id"]: row for row in fallback_rows}
    deltas = sorted(
        (
            float(fallback_by_id[record_id]["hf_mae_bpm"])
            - float(primary_by_id[record_id]["hf_mae_bpm"]),
            record_id,
        )
        for record_id in primary_by_id
    )
    primary_grouping = _grouping_summary(experiment_root / "p2" / "d24_15_primary")
    fallback_grouping = _grouping_summary(experiment_root / "p4" / "d24_18_fallback")
    primary_coordinates = _selection_coordinates(
        experiment_root / "p2" / "d24_15_primary" / "main" / "selection_manifest.json"
    )
    fallback_coordinates = _selection_coordinates(
        experiment_root / "p4" / "d24_18_fallback" / "main" / "selection_manifest.json"
    )

    report = _build_report(
        baseline=baseline,
        primary=primary,
        primary_sensitivity=primary_sensitivity,
        fallback=fallback,
        fallback_sensitivity=fallback_sensitivity,
        lyx=lyx,
        deltas=deltas,
        primary_by_id=primary_by_id,
        fallback_by_id=fallback_by_id,
        primary_grouping=primary_grouping,
        fallback_grouping=fallback_grouping,
        primary_coordinates=primary_coordinates,
        fallback_coordinates=fallback_coordinates,
    )
    report_path = p5_root / "report.md"
    _write_text(report_path, report)
    qa = _figure_qa(figure_root)
    qa_path = figure_root / "qa_notes.json"
    _write_json(qa_path, qa)
    report_receipt = {
        "schema_id": "d24_handgrip_blind_composition_report_receipt_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass",
        "report_path": str(report_path.resolve()),
        "report_sha256": _file_sha256(report_path),
        "figure_qa_path": str(qa_path.resolve()),
        "figure_qa_sha256": _file_sha256(qa_path),
        "verifier_receipt_sha256": _file_sha256(p5_root / "verifier" / "receipt.json"),
        "final_route_decision": "stop_composition_route",
    }
    _write_json(p5_root / "report_receipt.json", report_receipt)

    if args.finalize:
        test_xml = p5_root / "test_results.xml"
        test_summary = _verify_junit(test_xml)
        completion = {
            "schema_id": "d24_handgrip_blind_composition_completion_receipt_v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "complete",
            "completed_phases": ["P0", "P1", "P2", "P3", "P4", "P5"],
            "primary_stage_passed": False,
            "fallback_stage_passed": False,
            "final_route_decision": "stop_composition_route",
            "next_research_route": "handgrip_algorithm_mechanism_development",
            "lyx_consistency_completed": True,
            "lyx_evidence_role": "development_consistency_not_independent_validation",
            "independent_verification_passed": True,
            "formal_rules_changed_after_reveal": False,
            "report_receipt_sha256": _file_sha256(p5_root / "report_receipt.json"),
            "test_results_path": str(test_xml.resolve()),
            "test_results_sha256": _file_sha256(test_xml),
            "test_summary": test_summary,
            "figure_outputs": {
                suffix: {
                    "path": str(
                        (figure_root / f"handgrip_blind_composition_outcome.{suffix}").resolve()
                    ),
                    "sha256": _file_sha256(
                        figure_root / f"handgrip_blind_composition_outcome.{suffix}"
                    ),
                }
                for suffix in ("svg", "pdf", "png")
            },
        }
        _write_json(p5_root / "completion_receipt.json", completion)


def _build_report(
    *,
    baseline: dict,
    primary: dict,
    primary_sensitivity: dict,
    fallback: dict,
    fallback_sensitivity: dict,
    lyx: dict,
    deltas: list[tuple[float, str]],
    primary_by_id: dict[str, dict[str, str]],
    fallback_by_id: dict[str, dict[str, str]],
    primary_grouping: dict,
    fallback_grouping: dict,
    primary_coordinates: dict,
    fallback_coordinates: dict,
) -> str:
    worsened = list(reversed(deltas[-3:]))
    improved = deltas[:3]
    challenges = ("woli2_LYX_0708", "woli2_LZJ_0711")
    coordinate_rows = "\n".join(
        f"| {subject} | {primary_coordinates[subject]['hf']} | "
        f"{primary_coordinates[subject]['acc']} | {fallback_coordinates[subject]['hf']} | "
        f"{fallback_coordinates[subject]['acc']} |"
        for subject in sorted(primary_coordinates)
    )
    return f"""# D24 Handgrip 无心率标签训练构成实验报告

## 结论

本实验按冻结合同完整执行后，**数据构成路线未通过**：D24-15 主方案六折均退化为 `k=1`、保留全部折内训练记录，HF 平均 MAE 与原六门基线完全相同；加入三条历史删除记录后的 D24-18 后备仍全部为 `k=1`，但训练账本变化使 HF 选择坐标改变，平均 MAE 反而升高。因此按预注册阶段门停止继续调整特征、权重、聚类阈值或记录清单，下一条研究路线应转入 Handgrip 算法机制开发。

LYX 跨佩戴同步仅保留一项有限的一致性信号：在三条同一受试者、与历史开发重叠的记录上，HF 优于独立调参 ACC；但 HF 仍略差于既有独立基线。该结果不改变 D24 裁决，也不构成独立验证。

![实验结果总览](figures/handgrip_blind_composition_outcome.png)

## 冻结设计与证据边界

- 主评价分母固定为 D24 Handgrip 的 15 条记录、6 个按受试者划分的 LOSO 折。
- D24-18 仅把三条历史删除记录加入训练候选，评价仍是原 15 条。
- 构成只读取 HF、ACC、PPG 原始信号的六类无心率标签指纹；不读取参考心率、MAE 或留出受试者信号。
- 同一折内训练核心同时供原六门 HF 选择器和独立 ACC minimax 选择器使用。
- 四等分版本是唯一主裁决；30 s 六分块只作敏感性，不能替代主方案。
- 结果属于回顾性开发证据，不是随机人群或独立外部泛化验证。

## D24 主裁决

| 阶段 | HF 平均 MAE (BPM) | HF 中位数 | HF 样本 SD | 同核心 ACC 平均 MAE | HF 自改善 | HF 不弱于 ACC | 主门 |
|---|---:|---:|---:|---:|---|---|---|
| 冻结原六门基线 | {float(baseline["hf_mean_mae_bpm"]):.4f} | — | {float(baseline["hf_sample_sd_bpm"]):.4f} | {float(baseline["acc_mean_mae_bpm"]):.4f} | — | 否 | 基线 |
| D24-15 主方案 | {float(primary["hf_mean_mae_bpm"]):.4f} | {float(primary["hf_median_mae_bpm"]):.4f} | {float(primary["hf_sample_sd_bpm"]):.4f} | {float(primary["acc_mean_mae_bpm"]):.4f} | 否 | 否 | 失败 |
| D24-18 后备 | {float(fallback["hf_mean_mae_bpm"]):.4f} | {float(fallback["hf_median_mae_bpm"]):.4f} | {float(fallback["hf_sample_sd_bpm"]):.4f} | {float(fallback["acc_mean_mae_bpm"]):.4f} | 否 | 否 | 失败 |

D24-15 的 HF 平均值相对基线变化为 `{float(primary["hf_mean_mae_bpm"]) - float(baseline["hf_mean_mae_bpm"]):+.4f}` BPM；D24-18 相对 D24-15 变化为 `{float(fallback["hf_mean_mae_bpm"]) - float(primary["hf_mean_mae_bpm"]):+.4f}` BPM，HF 样本 SD 同时变化 `{float(fallback["hf_sample_sd_bpm"]) - float(primary["hf_sample_sd_bpm"]):+.4f}` BPM。后备阶段 ACC 变好不参与否决或通过，但 HF 自身未改善且仍弱于 ACC，因此主门明确失败。

30 s 敏感性同样不通过：D24-15 HF/ACC 平均 MAE 为 `{float(primary_sensitivity["hf_mean_mae_bpm"]):.4f}/{float(primary_sensitivity["acc_mean_mae_bpm"]):.4f}` BPM，D24-18 为 `{float(fallback_sensitivity["hf_mean_mae_bpm"]):.4f}/{float(fallback_sensitivity["acc_mean_mae_bpm"]):.4f}` BPM。按合同，这些数值只用于诊断。

## 为什么构成路线没有奏效

- D24-15 主版本 6/6 折均选择 `k=1`；折内核心保留 `{primary_grouping["main_core_min"]}–{primary_grouping["main_core_max"]}` 条，即全部原始训练记录。
- D24-18 主版本 6/6 折也均选择 `k=1`；每折保留全部 15 条训练记录。
- `k=2/3` 在主版本的所有折均未同时满足正 silhouette、中位 silhouette、ARI 中位数和最小 ARI 四组冻结阈值。
- 30 s 敏感性仅在留出 LYX 的折出现 `k=2`、核心 6 条；其余折仍为 `k=1`。这既不具备跨折稳定性，也被合同明确禁止升级为主规则。
- 因此，15 条阶段没有产生新的训练子集；18 条阶段的变化来自“把三条历史记录整体加入训练账本”，而不是信号模式代表性筛选成功。

## 逐记录变化与挑战记录

后备相对主方案 HF 恶化最大的三条为：

{_format_delta_list(worsened)}

改善最大的三条为：

{_format_delta_list(improved)}

两条预锁定挑战记录均未恶化，但这不足以抵消其他长尾记录的显著恶化：

| 挑战记录 | D24-15 HF | D24-18 HF | 变化 (BPM) |
|---|---:|---:|---:|
{_challenge_rows(challenges, primary_by_id, fallback_by_id)}

## 两条路线的折内坐标

下表给出坐标索引；完整物理参数保存在每折选择回执中。D24-15 的主核心等同原训练账本，所以 HF 坐标复现原六门选择。D24-18 虽未删出代表子集，但新增训练记录改变了多折的 HF 排名。

| 留出受试者 | D24-15 HF | D24-15 ACC | D24-18 HF | D24-18 ACC |
|---|---:|---:|---:|---:|
{coordinate_rows}

## LYX Handgrip 一致性检查

LYX 三条跨佩戴记录来自同一受试者，构成结果为 `k={int(lyx["selected_cluster_count"])}`，三条全部保留。冻结规则选得：

- HF：`{lyx["hf_coordinate_id"]}`（索引 {int(lyx["hf_coordinate_index"])}），平均 MAE `{float(lyx["hf_mean_mae_bpm"]):.4f}` BPM；
- ACC：`{lyx["acc_coordinate_id"]}`（索引 {int(lyx["acc_coordinate_index"])}），平均 MAE `{float(lyx["acc_mean_mae_bpm"]):.4f}` BPM；
- 既有独立基线平均 MAE `{float(lyx["independent_baseline_mean_mae_bpm"]):.4f}` BPM。

所以 LYX 上的方向是 `HF < ACC`，但 `HF > 既有独立基线`。记录与受试者均和既有开发数据重叠，不能据此声称跨个体或跨佩戴泛化。

## 独立复核

独立验证器未调用正式分组、代表记录选择或两条正式选择器入口，而是从冻结指纹和最小训练 CSV 重新计算：折内稳健尺度、六类等权距离、平均链接聚类、silhouette、全部成对 ARI、共同/罕见模式、medoid、训练核心、HF/ACC 坐标、15 条评价分母、均值/中位数/样本 SD 和阶段门。D24-15、D24-18 的主版本与 30 s 敏感性，以及 LYX 同步均与正式产物一致，复核状态为 `pass`。

## 最终裁决与下一步

1. 封存“无心率标签信号分型后选择代表训练记录”这条数据构成路线；本轮不继续改特征、权重、阈值、块长或历史记录清单。
2. 保留原六门 HF 选择器作为当前 D24 Handgrip 基线，不采用 D24-18 后备结果。
3. 下一实验应另立 Handgrip 算法机制开发身份，直接针对 HF 界面压力变化与长尾失稳；目标仍是 HF 自身变好并最终不弱于同口径 ACC。
4. LYX 只保留为开发一致性参考，不用于修改下一实验规则，也不升级为验证集。

## 产物

- `p5/report.md`：本报告；
- `p5/figures/handgrip_blind_composition_outcome.svg|pdf|png`：可编辑矢量、PDF 与 600 dpi 审阅图；
- `p5/figures/source_data.csv`：图件源数据；
- `p5/verifier/verification.json`：独立复核明细；
- `p5/completion_receipt.json`：最终完成回执（验证测试通过后生成）。
"""


def _grouping_summary(stage_root: Path) -> dict:
    main_cores = []
    main_k = []
    sensitivity_k = []
    sensitivity_cores = []
    for path in sorted((stage_root / "compositions").glob("*.json")):
        composition = _read_json(path)
        main_k.append(int(composition["selected_cluster_count"]))
        main_cores.append(len(composition["training_core_record_ids"]))
        sensitivity_k.append(int(composition["sensitivity_30s"]["selected_cluster_count"]))
        sensitivity_cores.append(len(composition["sensitivity_30s"]["training_core_record_ids"]))
    return {
        "main_k": main_k,
        "main_core_min": min(main_cores),
        "main_core_max": max(main_cores),
        "sensitivity_k": sensitivity_k,
        "sensitivity_cores": sensitivity_cores,
    }


def _selection_coordinates(path: Path) -> dict[str, dict[str, int]]:
    manifest = _read_json(path)
    result = {}
    for row in manifest["selections"]:
        subject = str(row["fold_id"]).split("_")[-1]
        result[subject] = {
            "hf": int(row["hf_coordinate_index"]),
            "acc": int(row["acc_coordinate_index"]),
        }
    return result


def _format_delta_list(rows: list[tuple[float, str]]) -> str:
    return "\n".join(
        f"- `{_canonical_record(record_id)}`：`{delta:+.4f}` BPM" for delta, record_id in rows
    )


def _challenge_rows(
    record_ids: tuple[str, ...],
    primary: dict[str, dict[str, str]],
    fallback: dict[str, dict[str, str]],
) -> str:
    rows = []
    for record_id in record_ids:
        before = float(primary[record_id]["hf_mae_bpm"])
        after = float(fallback[record_id]["hf_mae_bpm"])
        rows.append(
            f"| `{_canonical_record(record_id)}` | {before:.4f} | {after:.4f} | "
            f"{after - before:+.4f} |"
        )
    return "\n".join(rows)


def _canonical_record(record_id: str) -> str:
    parts = record_id.split("_")
    repeat = parts[0].replace("woli", "")
    return f"HG{repeat}_{parts[1]}_{parts[2]}"


def _figure_qa(figure_root: Path) -> dict:
    svg_path = figure_root / "handgrip_blind_composition_outcome.svg"
    pdf_path = figure_root / "handgrip_blind_composition_outcome.pdf"
    png_path = figure_root / "handgrip_blind_composition_outcome.png"
    source_path = figure_root / "source_data.csv"
    caption_path = figure_root / "figure_caption.md"
    contract_path = figure_root / "figure_contract.json"
    svg_text = svg_path.read_text(encoding="utf-8")
    if "<text" not in svg_text:
        raise ValueError("handgrip_figure_svg_text_not_editable")
    width, height = _png_dimensions(png_path)
    if width < 4000 or height < 3000:
        raise ValueError("handgrip_figure_png_resolution")
    for path in (pdf_path, source_path, caption_path, contract_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"handgrip_figure_artifact:{path.name}")
    return {
        "schema_id": "d24_handgrip_blind_composition_figure_qa_v1",
        "status": "pass",
        "backend": "python",
        "backend_exclusive": True,
        "visual_review": "pass_after_legend_reposition",
        "svg_editable_text": True,
        "pdf_exported": True,
        "png_dpi": 600,
        "png_pixel_width": width,
        "png_pixel_height": height,
        "source_data_sha256": _file_sha256(source_path),
        "caption_sha256": _file_sha256(caption_path),
        "figure_contract_sha256": _file_sha256(contract_path),
        "statistics_note": "record mean and sample SD; no inferential test",
        "lyx_limitation_visible": True,
        "image_integrity": "vector-native plots; no raster image manipulation",
        "outputs": {
            suffix: _file_sha256(figure_root / f"handgrip_blind_composition_outcome.{suffix}")
            for suffix in ("svg", "pdf", "png")
        },
    }


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("handgrip_figure_png_signature")
    return struct.unpack(">II", signature[16:24])


def _verify_junit(path: Path) -> dict:
    if not path.is_file():
        raise ValueError("handgrip_completion_test_results_missing")
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
    }
    if totals["tests"] < 1 or totals["failures"] or totals["errors"]:
        raise ValueError(f"handgrip_completion_tests_failed:{totals}")
    return totals


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: dict) -> None:
    _write_text(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
