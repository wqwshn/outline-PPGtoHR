"""Build canonical D24 HF/ACC distributions and excluded-record evidence tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from run_cross_subject_hf_optimization import LYX_REPLACEMENTS

from ppg_hr.v2.cross_subject_acc_selection import load_acc_compact_cell_csv
from ppg_hr.v2.cross_subject_d24_analysis import (
    AccPanelResult,
    acc_records_from_cells,
    evaluate_acc_panel,
    nested_deletion_rounds,
    scene_iqr_audit,
)
from ppg_hr.v2.cross_subject_hf_optimization import (
    PanelResult,
    ResponseTable,
    evaluate_panel,
    load_lyx_partition,
    load_parent_hf_cells,
)

EXPERIMENT_ID = "cross_subject_multirecord_d24_hf_acc_analysis_20260831_v1"
OPTIMIZATION_EXPERIMENT_ID = "cross_subject_multirecord_hf_loso_optimization_20260831_v4"
PARENT_HF_EXPERIMENT_ID = "cross_subject_multirecord_hf_loso_v1"
PARENT_ACC_EXPERIMENT_ID = "cross_subject_multirecord_acc_independent_physical4d_v1"
EXPECTED_D24_HF_MEAN = 3.9361675656762425


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--parent-worktree", type=Path)
    parser.add_argument("--lyx-worktree", type=Path)
    parser.add_argument("--optimization-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    worktrees_root = repo_root.parent
    parent_worktree = (
        args.parent_worktree.resolve()
        if args.parent_worktree
        else worktrees_root / "cross-subject-multirecord-hf-loso"
    )
    lyx_worktree = (
        args.lyx_worktree.resolve()
        if args.lyx_worktree
        else worktrees_root / "lyx-bo-space-generalization"
    )
    optimization_root = (
        args.optimization_root.resolve()
        if args.optimization_root
        else repo_root / "data" / "experiments" / OPTIMIZATION_EXPERIMENT_ID
    )
    experiment_root = repo_root / "data" / "experiments" / EXPERIMENT_ID
    output_root = (
        args.output_root.resolve() if args.output_root else experiment_root / "source_data"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    parent_hf_root = parent_worktree / "data" / "experiments" / PARENT_HF_EXPERIMENT_ID
    parent_acc_root = parent_worktree / "data" / "experiments" / PARENT_ACC_EXPERIMENT_ID
    parent_hf_csv = parent_hf_root / "p2" / "hf_cell_metrics.csv"
    parent_acc_csv = parent_acc_root / "p2" / "acc_cell_metrics.csv"
    supplement_acc_csv = experiment_root / "acc_supplement" / "acc_cell_metrics.csv"

    print("stage=load_hf_response", flush=True)
    parent_hf = load_parent_hf_cells(parent_hf_csv)
    lyx_partitions = _lyx_partitions(lyx_worktree)
    additions = [load_lyx_partition(path) for path in lyx_partitions.values()]
    synced_hf = parent_hf.with_replacements(
        remove_record_ids=[old_id for _, _, _, old_id in LYX_REPLACEMENTS],
        additions=additions,
    )
    d24_payload = _read_json(optimization_root / "panels" / "balanced_record_d24" / "panel.json")
    retained = tuple(str(value) for value in d24_payload["retained_record_ids"])
    excluded = tuple(str(value) for value in d24_payload["excluded_record_ids"])
    if len(retained) != 119 or len(excluded) != 24:
        raise RuntimeError("d24_panel_size_mismatch")

    hf_full = evaluate_panel(synced_hf)
    hf_d24 = evaluate_panel(synced_hf, retained_record_ids=retained)
    _assert_close(hf_d24.mean_mae_bpm, EXPECTED_D24_HF_MEAN)
    _assert_close(hf_d24.mean_mae_bpm, float(d24_payload["mean_mae_bpm"]))

    print("stage=load_acc_response", flush=True)
    parent_acc_cells = load_acc_compact_cell_csv(parent_acc_csv)
    supplement_acc_cells = load_acc_compact_cell_csv(supplement_acc_csv)
    old_ids = {old_id for _, _, _, old_id in LYX_REPLACEMENTS}
    synced_acc_cells = (
        tuple(cell for cell in parent_acc_cells if cell.record_id not in old_ids)
        + supplement_acc_cells
    )
    parent_acc_records = acc_records_from_cells(parent_acc_cells)
    synced_acc_records = acc_records_from_cells(synced_acc_cells)
    if len(synced_acc_records) != 143 or len(supplement_acc_cells) != 2_100:
        raise RuntimeError("synced_acc_response_size_mismatch")
    acc_parent = evaluate_acc_panel(parent_acc_records)
    acc_full = evaluate_acc_panel(synced_acc_records)
    acc_d24 = evaluate_acc_panel(synced_acc_records, retained_record_ids=retained)
    authoritative_acc_mean = _authoritative_acc_mean(
        parent_acc_root / "p3" / "holdout_record_results.csv"
    )
    _assert_close(acc_parent.mean_mae_bpm, authoritative_acc_mean)

    print("stage=write_distributions", flush=True)
    route_rows = _hf_route_rows(hf_d24) + _acc_route_rows(acc_d24)
    route_rows.sort(
        key=lambda row: (str(row["scene"]), str(row["route_id"]), str(row["record_id"]))
    )
    scene_rows = _scene_summary(route_rows)
    acc_fold_rows = _acc_fold_rows(acc_d24)
    _write_csv(output_root / "d24_record_route_mae.csv", route_rows)
    _write_csv(output_root / "d24_scene_summary.csv", scene_rows)
    _write_csv(output_root / "d24_acc_fold_selections.csv", acc_fold_rows)

    print("stage=iqr_and_explainability", flush=True)
    full_hf_rows = _hf_route_rows(hf_full)
    audited_rows, iqr_summaries = scene_iqr_audit(full_hf_rows)
    audited_by_id = {str(row["record_id"]): row for row in audited_rows}
    baseline_by_id = _historical_baselines(parent_hf_csv, lyx_partitions)
    deletion_rounds, level_results = _deletion_levels(optimization_root, synced_hf)
    full_by_id = {str(row["record_id"]): row for row in full_hf_rows}
    explainability_rows = []
    for record_id in sorted(excluded, key=lambda item: (synced_hf.records[item].scene, item)):
        record = synced_hf.records[record_id]
        original = full_by_id[record_id]
        audit = audited_by_id[record_id]
        round_index = deletion_rounds[record_id]
        before_result = hf_full if round_index == 1 else level_results[round_index - 1]
        after_result = level_results[round_index]
        before_scene_mean = _panel_scene_means(before_result)[record.scene]
        after_scene_mean = _panel_scene_means(after_result)[record.scene]
        best_index = int(np.argmin(record.mae_bpm))
        original_mae = float(original["mae_bpm"])
        best_mae = float(record.mae_bpm[best_index])
        scene_values = np.asarray(
            [float(row["mae_bpm"]) for row in full_hf_rows if str(row["scene"]) == record.scene],
            dtype=float,
        )
        percentile = 100.0 * float(np.mean(scene_values <= original_mae))
        reason = _deletion_reason(audit, round_index)
        explainability_rows.append(
            {
                "scene": record.scene,
                "record_id": record_id,
                "physical_subject_id": record.subject_id,
                "deletion_round": round_index,
                "original_synced_full143_loso_mae_bpm": original_mae,
                "original_selected_coordinate_id": original["selected_coordinate_id"],
                "historical_lite_baseline_mae_bpm": baseline_by_id[record_id],
                "physical4d_300point_min_mae_bpm": best_mae,
                "physical4d_best_coordinate_id": record.coordinate_ids[best_index],
                "physical4d_fullspace_median_mae_bpm": float(np.median(record.mae_bpm)),
                "physical4d_qualified_coordinate_count": int(np.count_nonzero(record.qualified)),
                "loso_minus_best4d_bpm": original_mae - best_mae,
                "scene_loso_percentile_pct": percentile,
                "scene_q1_bpm": audit["scene_q1_bpm"],
                "scene_q3_bpm": audit["scene_q3_bpm"],
                "scene_iqr_bpm": audit["scene_iqr_bpm"],
                "scene_upper_fence_bpm": audit["scene_upper_fence_bpm"],
                "is_upper_iqr_outlier": audit["is_upper_iqr_outlier"],
                "scene_mean_before_deletion_bpm": before_scene_mean,
                "scene_mean_after_deletion_bpm": after_scene_mean,
                "scene_mean_improvement_bpm": before_scene_mean - after_scene_mean,
                "deletion_reason_zh": reason,
            }
        )
    _write_csv(output_root / "d24_full143_iqr_audit.csv", audited_rows)
    _write_csv(
        output_root / "d24_full143_iqr_summary.csv",
        [asdict(row) for row in iqr_summaries],
    )
    _write_csv(output_root / "d24_excluded_record_explainability.csv", explainability_rows)
    _write_markdown_table(
        experiment_root / "report" / "d24_excluded_record_explainability_zh.md",
        explainability_rows,
    )

    iqr_outlier_count = sum(bool(row["is_upper_iqr_outlier"]) for row in explainability_rows)
    upper_quartile_count = sum(
        float(row["original_synced_full143_loso_mae_bpm"]) > float(row["scene_q3_bpm"])
        for row in explainability_rows
    )
    summary = {
        "schema_id": "d24_hf_acc_analysis_receipt_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass",
        "claim_boundary": "posthoc_curated_D24_sensitivity_analysis_not_independent_validation",
        "d24_record_count": len(retained),
        "d24_excluded_record_count": len(excluded),
        "fold_count_per_route": 48,
        "hf_selector": "original_six_gate_subject_balanced_lexicographic_physical4d_v1",
        "acc_selector": "training_subject_balanced_acc_mae_minimax_v1",
        "hf_d24_mean_mae_bpm": hf_d24.mean_mae_bpm,
        "acc_d24_mean_mae_bpm": acc_d24.mean_mae_bpm,
        "hf_synced_full143_mean_mae_bpm": hf_full.mean_mae_bpm,
        "acc_synced_full143_mean_mae_bpm": acc_full.mean_mae_bpm,
        "acc_parent_authoritative_mean_mae_bpm": authoritative_acc_mean,
        "acc_parent_reproduced_mean_mae_bpm": acc_parent.mean_mae_bpm,
        "deleted_upper_iqr_outlier_count": iqr_outlier_count,
        "deleted_above_scene_q3_count": upper_quartile_count,
        "iqr_rule": "scene-wise NumPy linear Q1/Q3; strict MAE > Q3 + 1.5*IQR",
        "iqr_interpretation": "descriptive high-error flag, not proof of acquisition invalidity",
        "route_comparison_boundary": (
            "HF and ACC are independently tuned on route-native reliable supports; "
            "juxtaposition is descriptive and not a paired per-window effect estimate."
        ),
        "source_sha256": {
            "parent_hf_cells": _file_sha256(parent_hf_csv),
            "parent_acc_cells": _file_sha256(parent_acc_csv),
            "supplement_acc_cells": _file_sha256(supplement_acc_csv),
            "d24_panel": _file_sha256(
                optimization_root / "panels" / "balanced_record_d24" / "panel.json"
            ),
        },
        "artifact_sha256": {
            path.name: _file_sha256(path) for path in sorted(output_root.glob("*.csv"))
        },
    }
    _write_json(experiment_root / "analysis_receipt.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0


def _lyx_partitions(lyx_worktree: Path) -> dict[str, Path]:
    roots = lyx_worktree / "data" / "experiments"
    result = {}
    for experiment_id, scene, new_record_id, _ in LYX_REPLACEMENTS:
        result[new_record_id] = (
            roots
            / experiment_id
            / "response"
            / "partitions"
            / scene
            / new_record_id
            / "cell_rows.csv"
        )
    return result


def _hf_route_rows(result: PanelResult) -> list[dict[str, Any]]:
    rows = []
    for fold in result.folds:
        for record_id, mae, qualified in zip(
            fold.holdout_record_ids,
            fold.holdout_mae_bpm,
            fold.holdout_qualified,
            strict=True,
        ):
            rows.append(
                {
                    "route_id": "HF",
                    "scene": fold.scene,
                    "physical_subject_id": fold.holdout_subject_id,
                    "record_id": record_id,
                    "selected_coordinate_id": fold.coordinate_id,
                    "selected_coordinate_index": fold.coordinate_index,
                    "mae_bpm": mae,
                    "route_native_qualified": qualified,
                    "route_native_evaluation_window_sha256": "",
                    "repeat_index": _repeat_index(record_id),
                }
            )
    return rows


def _acc_route_rows(result: AccPanelResult) -> list[dict[str, Any]]:
    rows = []
    for fold in result.folds:
        for record_id, mae, window_sha in zip(
            fold.holdout_record_ids,
            fold.holdout_mae_bpm,
            fold.holdout_evaluation_window_sha256,
            strict=True,
        ):
            rows.append(
                {
                    "route_id": "ACC",
                    "scene": fold.scene,
                    "physical_subject_id": fold.holdout_subject_id,
                    "record_id": record_id,
                    "selected_coordinate_id": fold.coordinate_id,
                    "selected_coordinate_index": fold.coordinate_index,
                    "mae_bpm": mae,
                    "route_native_qualified": "",
                    "route_native_evaluation_window_sha256": window_sha,
                    "repeat_index": _repeat_index(record_id),
                }
            )
    return rows


def _scene_summary(route_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in route_rows:
        buckets[(str(row["route_id"]), str(row["scene"]))].append(float(row["mae_bpm"]))
    rows = []
    for (route, scene), values_list in sorted(buckets.items()):
        values = np.asarray(values_list, dtype=float)
        rows.append(
            {
                "route_id": route,
                "scene": scene,
                "record_count": len(values),
                "mean_mae_bpm": float(np.mean(values)),
                "median_mae_bpm": float(np.median(values)),
                "q1_mae_bpm": float(np.quantile(values, 0.25, method="linear")),
                "q3_mae_bpm": float(np.quantile(values, 0.75, method="linear")),
                "min_mae_bpm": float(np.min(values)),
                "max_mae_bpm": float(np.max(values)),
            }
        )
    for route in ("HF", "ACC"):
        values = np.asarray(
            [float(row["mae_bpm"]) for row in route_rows if row["route_id"] == route],
            dtype=float,
        )
        rows.append(
            {
                "route_id": route,
                "scene": "OVERALL",
                "record_count": len(values),
                "mean_mae_bpm": float(np.mean(values)),
                "median_mae_bpm": float(np.median(values)),
                "q1_mae_bpm": float(np.quantile(values, 0.25, method="linear")),
                "q3_mae_bpm": float(np.quantile(values, 0.75, method="linear")),
                "min_mae_bpm": float(np.min(values)),
                "max_mae_bpm": float(np.max(values)),
            }
        )
    return rows


def _acc_fold_rows(result: AccPanelResult) -> list[dict[str, Any]]:
    return [
        {
            "scene": fold.scene,
            "holdout_subject_id": fold.holdout_subject_id,
            "coordinate_id": fold.coordinate_id,
            "coordinate_index": fold.coordinate_index,
            "holdout_record_count": len(fold.holdout_record_ids),
            "holdout_mean_mae_bpm": float(np.mean(fold.holdout_mae_bpm)),
            "holdout_record_ids_json": json.dumps(fold.holdout_record_ids, ensure_ascii=False),
            "holdout_mae_bpm_json": json.dumps(fold.holdout_mae_bpm),
        }
        for fold in result.folds
    ]


def _historical_baselines(parent_hf_csv: Path, lyx_partitions: dict[str, Path]) -> dict[str, float]:
    values: dict[str, float] = {}
    with parent_hf_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(str(row["record_id"]), float(row["baseline_mae_bpm"]))
    for record_id, path in lyx_partitions.items():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            first = next(csv.DictReader(handle))
        values[record_id] = float(first["independent_mae_full_bpm"])
    return values


def _deletion_levels(
    optimization_root: Path, table: ResponseTable
) -> tuple[dict[str, int], dict[int, PanelResult]]:
    levels = []
    results = {}
    for index, count in enumerate((8, 16, 24), start=1):
        payload = _read_json(
            optimization_root / "panels" / f"balanced_record_d{count}" / "panel.json"
        )
        levels.append(set(str(value) for value in payload["excluded_record_ids"]))
        result = evaluate_panel(table, retained_record_ids=payload["retained_record_ids"])
        _assert_close(result.mean_mae_bpm, float(payload["mean_mae_bpm"]))
        results[index] = result
    return nested_deletion_rounds(tuple(levels), expected_increment=8), results


def _panel_scene_means(result: PanelResult) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for fold in result.folds:
        values[fold.scene].extend(float(value) for value in fold.holdout_mae_bpm)
    return {scene: float(np.mean(rows)) for scene, rows in values.items()}


def _deletion_reason(audit: dict[str, Any], round_index: int) -> str:
    prefix = f"第{round_index}轮场景内均衡贪心删减：在每个三记录格最多删1条约束下，使该场景删后LOSO均值最低"
    if bool(audit["is_upper_iqr_outlier"]):
        return prefix + "；原LOSO误差超过场景Q3+1.5IQR上界"
    if float(audit["mae_bpm"]) > float(audit["scene_q3_bpm"]):
        return prefix + "；原LOSO误差位于场景上四分位，但未越过1.5IQR上界"
    return prefix + "；并非单变量IQR离群点，收益来自场景构成变化与训练侧共同坐标重选"


def _write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# D24 删除记录逐条证据表",
        "",
        "口径：原 LOSO 为 LYX 同步后的完整 143 条 HF 六门选择器结果；离群判定在各场景内按严格 `MAE > Q3 + 1.5×IQR`。IQR 只表示单变量高误差，不等同于采集无效。",
        "",
        "| 场景 | 记录 | 轮次 | 原LOSO | 历史Lite | 300点最小 | IQR离群 | 场景收益 | 简要原因 |",
        "|---|---|---:|---:|---:|---:|:---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {scene} | {record_id} | {deletion_round} | {original:.3f} | "
            "{baseline:.3f} | {best:.3f} | {outlier} | {gain:.3f} | {reason} |".format(
                scene=row["scene"],
                record_id=row["record_id"],
                deletion_round=row["deletion_round"],
                original=float(row["original_synced_full143_loso_mae_bpm"]),
                baseline=float(row["historical_lite_baseline_mae_bpm"]),
                best=float(row["physical4d_300point_min_mae_bpm"]),
                outlier="是" if row["is_upper_iqr_outlier"] else "否",
                gain=float(row["scene_mean_improvement_bpm"]),
                reason=row["deletion_reason_zh"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _authoritative_acc_mean(path: Path) -> float:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        values = [float(row["native_mae_bpm"]) for row in csv.DictReader(handle)]
    return float(np.mean(np.asarray(values, dtype=float)))


def _repeat_index(record_id: str) -> int:
    prefix = record_id.split("_", 1)[0]
    digits = "".join(character for character in prefix if character.isdigit())
    return int(digits) if digits else 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot_write_empty_csv:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_close(actual: float, expected: float) -> None:
    if abs(actual - expected) > 1e-12:
        raise RuntimeError(f"float_mismatch:{actual}:{expected}")


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
