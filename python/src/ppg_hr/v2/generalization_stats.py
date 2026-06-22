"""Statistics helpers for v2 generalization outputs."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .solver import solve_v2
from .types import V2RunConfig


@dataclass(frozen=True)
class V2GeneralizationStatsResult:
    fold_stats_csv: Path
    aggregate_stats_csv: Path
    analysis_tables_dir: Path


TEST_SPLITS = {"test", "external_test"}
TABLE_SPECS = (
    ("table_full_mae.csv", "total", "aae"),
    ("table_full_r5.csv", "total", "hit_rate_5bpm"),
    ("table_motion_mae.csv", "motion", "aae"),
    ("table_motion_r5.csv", "motion", "hit_rate_5bpm"),
)


def write_generalization_statistics(
    output_dir: Path,
    records: Iterable[Any],
    *,
    on_progress=None,
) -> V2GeneralizationStatsResult:
    output_dir = Path(output_dir)
    rows = list(records)
    fold_stats_csv = output_dir / "v2_generalization_fold_stats.csv"
    aggregate_stats_csv = output_dir / "v2_generalization_aggregate_stats.csv"
    analysis_dir = output_dir / "analysis_tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_fold_stats(fold_stats_csv, rows)
    _progress(on_progress, event="stats_fold", stage="stats", stage_label="统计fold指标", detail=str(fold_stats_csv))
    _write_aggregate_stats(aggregate_stats_csv, rows)
    _progress(on_progress, event="stats_aggregate", stage="stats", stage_label="统计整体指标", detail=str(aggregate_stats_csv))
    _write_analysis_tables(analysis_dir, rows)
    _write_markdown(analysis_dir / "generalization_stat_tables.md", rows)
    _progress(on_progress, event="stats_tables", stage="stats", stage_label="生成统计表", detail=str(analysis_dir))
    return V2GeneralizationStatsResult(fold_stats_csv, aggregate_stats_csv, analysis_dir)


def _test_records(records: list[Any]) -> list[Any]:
    selected: list[Any] = []
    for record in records:
        if getattr(record, "status", "ok") != "ok":
            continue
        mode = str(getattr(record, "evaluation_mode", ""))
        split = str(getattr(record, "split", ""))
        if mode == "all_train" and split == "train_test":
            selected.append(record)
        elif mode == "cross_person" and split == "external_test":
            selected.append(record)
        elif mode in {"k_fold_holdout", "leave_one_group_out"} and split == "test":
            selected.append(record)
        elif not mode and split in TEST_SPLITS:
            selected.append(record)
    if selected:
        return selected
    return [r for r in records if getattr(r, "status", "ok") == "ok" and getattr(r, "split", "") in TEST_SPLITS]


def _write_fold_stats(path: Path, records: list[Any]) -> None:
    grouped: dict[tuple[str, str, str], list[Any]] = {}
    for record in _test_records(records):
        key = (str(record.motion_type), str(record.evaluation_mode), str(record.fold_id))
        grouped.setdefault(key, []).append(record)
    rows: list[dict[str, Any]] = []
    for (motion_type, evaluation_mode, fold_id), items in sorted(grouped.items()):
        final = [_num(getattr(r, "final_aae_bpm", math.nan)) for r in items]
        fft = [_num(getattr(r, "fft_aae_bpm", math.nan)) for r in items]
        rows.append({
            "motion_type": motion_type,
            "evaluation_mode": evaluation_mode,
            "fold_id": fold_id,
            "test_sample_count": len(items),
            "mean_final_aae_bpm": _mean(final),
            "std_final_aae_bpm": _std(final),
            "mean_fft_aae_bpm": _mean(fft),
            "std_fft_aae_bpm": _std(fft),
        })
    _write_dict_csv(path, rows, [
        "motion_type", "evaluation_mode", "fold_id", "test_sample_count",
        "mean_final_aae_bpm", "std_final_aae_bpm", "mean_fft_aae_bpm", "std_fft_aae_bpm",
    ])


def _write_aggregate_stats(path: Path, records: list[Any]) -> None:
    grouped: dict[tuple[str, str], list[Any]] = {}
    fold_ids: dict[tuple[str, str], set[str]] = {}
    for record in _test_records(records):
        key = (str(record.motion_type), str(record.evaluation_mode))
        grouped.setdefault(key, []).append(record)
        fold_ids.setdefault(key, set()).add(str(record.fold_id))
    rows: list[dict[str, Any]] = []
    for (motion_type, evaluation_mode), items in sorted(grouped.items()):
        final = [_num(getattr(r, "final_aae_bpm", math.nan)) for r in items]
        fft = [_num(getattr(r, "fft_aae_bpm", math.nan)) for r in items]
        rows.append({
            "motion_type": motion_type,
            "evaluation_mode": evaluation_mode,
            "fold_count": len(fold_ids.get((motion_type, evaluation_mode), set())),
            "test_sample_count": len(items),
            "mean_final_aae_bpm": _mean(final),
            "std_final_aae_bpm": _std(final),
            "mean_fft_aae_bpm": _mean(fft),
            "std_fft_aae_bpm": _std(fft),
        })
    _write_dict_csv(path, rows, [
        "motion_type", "evaluation_mode", "fold_count", "test_sample_count",
        "mean_final_aae_bpm", "std_final_aae_bpm", "mean_fft_aae_bpm", "std_fft_aae_bpm",
    ])


def _write_analysis_tables(output_dir: Path, records: list[Any]) -> None:
    table_rows = [_analysis_row(record) for record in _test_records(records)]
    for file_name, scope, metric_name in TABLE_SPECS:
        rows = []
        for row in table_rows:
            final = _metric(row["final"], scope, metric_name)
            acc = _metric(row["acc"], scope, metric_name)
            fft = _metric(row["fft"], scope, metric_name)
            if metric_name == "hit_rate_5bpm":
                final *= 100.0
                acc *= 100.0
                fft *= 100.0
                delta = final - acc if math.isfinite(final) and math.isfinite(acc) else math.nan
            else:
                delta = 100.0 * (acc - final) / acc if math.isfinite(final) and math.isfinite(acc) and abs(acc) > 1e-12 else math.nan
            rows.append({
                "data_file": row["data_file"],
                "motion_type": row["motion_type"],
                "evaluation_mode": row["evaluation_mode"],
                "fold_id": row["fold_id"],
                "klms_hf": final,
                "klms_acc": acc,
                "klms_fft": fft,
                "klms_hf_vs_acc_delta": delta,
            })
        rows.append(_average_row(rows))
        _write_dict_csv(output_dir / file_name, rows, [
            "data_file", "motion_type", "evaluation_mode", "fold_id",
            "klms_hf", "klms_acc", "klms_fft", "klms_hf_vs_acc_delta",
        ])


def _analysis_row(record: Any) -> dict[str, Any]:
    final_metrics, fft_metrics = _metrics_from_error_csv(record)
    acc_metrics = _acc_replay_metrics(record)
    return {
        "data_file": str(getattr(record, "sample", "")),
        "motion_type": str(getattr(record, "motion_type", "")),
        "evaluation_mode": str(getattr(record, "evaluation_mode", "")),
        "fold_id": str(getattr(record, "fold_id", "")),
        "final": final_metrics,
        "fft": fft_metrics,
        "acc": acc_metrics,
    }


def _metrics_from_error_csv(record: Any) -> tuple[dict[str, float], dict[str, float]]:
    fallback_final = _metric_bundle(_num(getattr(record, "final_aae_bpm", math.nan)))
    fallback_fft = _metric_bundle(_num(getattr(record, "fft_aae_bpm", math.nan)))
    path = Path(str(getattr(record, "error_csv", "")))
    if not path.is_file():
        return fallback_final, fallback_fft
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            method_rows = {str(row.get("method", "")).strip(): row for row in csv.DictReader(fh)}
        fft = _metrics_from_error_row(method_rows.get("FFT", {})) if "FFT" in method_rows else fallback_fft
        final_key = next((key for key in method_rows if key != "FFT"), "")
        final = _metrics_from_error_row(method_rows.get(final_key, {})) if final_key else fallback_final
        return final, fft
    except Exception:
        return fallback_final, fallback_fft


def _acc_replay_metrics(record: Any) -> dict[str, float]:
    report_path = Path(str(getattr(record, "report_path", "")))
    params_path = Path(str(getattr(record, "params_report_path", "")))
    if not report_path.is_file() or not params_path.is_file():
        return _empty_metrics()
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        params_payload = json.loads(params_path.read_text(encoding="utf-8"))
        cfg = _config_from_payload(payload, params_payload.get("best_params", {}) or {}, ("ACC",))
        result = solve_v2(cfg)
        return _metrics_from_hr(np.asarray(result.HR, dtype=float), column=3)
    except Exception as exc:
        metrics = _empty_metrics()
        metrics["error"] = str(exc)  # type: ignore[index]
        return metrics


def _config_from_payload(payload: dict[str, Any], params: dict[str, Any], reference_groups_order: tuple[str, ...]) -> V2RunConfig:
    field_names = {f.name for f in dataclasses.fields(V2RunConfig)}
    cfg_dict: dict[str, Any] = {
        "data_path": Path(str(payload.get("data_path", ""))),
        "ref_path": Path(str(payload.get("ref_path", ""))),
        "adaptive_filter": payload.get("adaptive_filter"),
        "reference_groups_order": reference_groups_order,
    }
    for name in field_names:
        if name in cfg_dict or name == "extras":
            continue
        if name == "ppg_input_baseline_seconds":
            transform_params = payload.get("ppg_input_transform_params", {}) or {}
            if isinstance(transform_params, dict) and "baseline_seconds" in transform_params:
                cfg_dict[name] = transform_params["baseline_seconds"]
            continue
        if name in payload:
            cfg_dict[name] = payload[name]
    for key, value in params.items():
        if key in field_names and key not in {"data_path", "ref_path", "adaptive_filter", "reference_groups_order", "extras"}:
            cfg_dict[key] = value
    return V2RunConfig(**{k: v for k, v in cfg_dict.items() if k in field_names})


def _metrics_from_hr(hr: np.ndarray, *, column: int) -> dict[str, float]:
    if hr.size == 0 or hr.ndim != 2 or hr.shape[1] <= column:
        return _empty_metrics()
    ref = hr[:, 1]
    pred = hr[:, column]
    motion = hr[:, 4] > 0.5 if hr.shape[1] > 4 else np.zeros_like(ref, dtype=bool)
    valid = np.isfinite(ref) & np.isfinite(pred)
    return {
        "total_aae": _aae(ref, pred, valid),
        "rest_aae": _aae(ref, pred, valid & ~motion),
        "motion_aae": _aae(ref, pred, valid & motion),
        "total_hit_rate_5bpm": _hit(ref, pred, valid),
        "rest_hit_rate_5bpm": _hit(ref, pred, valid & ~motion),
        "motion_hit_rate_5bpm": _hit(ref, pred, valid & motion),
    }


def _metrics_from_error_row(row: dict[str, Any]) -> dict[str, float]:
    return {
        "total_aae": _num(row.get("total_aae")),
        "rest_aae": _num(row.get("rest_aae")),
        "motion_aae": _num(row.get("motion_aae")),
        "total_hit_rate_5bpm": _num(row.get("total_hit_rate_5bpm")),
        "rest_hit_rate_5bpm": _num(row.get("rest_hit_rate_5bpm")),
        "motion_hit_rate_5bpm": _num(row.get("motion_hit_rate_5bpm")),
    }


def _metric_bundle(aae: float) -> dict[str, float]:
    return {
        "total_aae": aae,
        "rest_aae": math.nan,
        "motion_aae": aae,
        "total_hit_rate_5bpm": math.nan,
        "rest_hit_rate_5bpm": math.nan,
        "motion_hit_rate_5bpm": math.nan,
    }


def _empty_metrics() -> dict[str, float]:
    return _metric_bundle(math.nan)


def _metric(metrics: dict[str, Any], scope: str, metric_name: str) -> float:
    return _num(metrics.get(f"{scope}_{metric_name}"))


def _aae(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if not bool(np.any(mask)):
        return math.nan
    return float(np.mean(np.abs(pred[mask] - ref[mask])))


def _hit(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if not bool(np.any(mask)):
        return math.nan
    return float(np.mean(np.abs(pred[mask] - ref[mask]) <= 5.0))


def _average_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rec: dict[str, Any] = {"data_file": "平均", "motion_type": "", "evaluation_mode": "", "fold_id": ""}
    for key in ("klms_hf", "klms_acc", "klms_fft", "klms_hf_vs_acc_delta"):
        rec[key] = _mean([_num(row.get(key)) for row in rows])
    return rec


def _write_markdown(path: Path, records: list[Any]) -> None:
    lines = [
        "# v2 泛化评估统计表",
        "",
        f"- 统计样本数：{len(_test_records(records))}",
        "- HF/FFT 指标优先来自每条记录的 error CSV；缺失时回退到 summary 指标。",
        "- ACC 指标通过读取参数报告 best_params，并将 reference_groups_order 改为 ACC 后重放得到。",
        "",
    ]
    for title, file_name in (
        ("全段 MAE", "table_full_mae.csv"),
        ("全段 R5", "table_full_r5.csv"),
        ("运动段 MAE", "table_motion_mae.csv"),
        ("运动段 R5", "table_motion_r5.csv"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"详见 {file_name}。")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dict_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key, "")) for key in fieldnames})


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.6g}"
    return value


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return math.nan
    return float(np.mean(vals))


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return math.nan
    return float(np.std(vals))


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _progress(callback, **info: Any) -> None:
    if callback is not None:
        callback(info)
