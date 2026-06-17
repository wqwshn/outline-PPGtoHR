from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ppg_hr.v2.solver import solve_v2
from ppg_hr.v2.types import V2RunConfig


TABLE_SPECS = (
    ("table_full_mae.csv", "total", "mae"),
    ("table_full_r5.csv", "total", "r5"),
    ("table_motion_mae.csv", "motion", "mae"),
    ("table_motion_r5.csv", "motion", "r5"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 generalization MAE/R5 tables.")
    parser.add_argument("root", type=Path, help="v2_generalization_outputs run directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated tables. Default: <root>/analysis_tables",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or (root / "analysis_tables")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = root / "v2_generalization_summary.csv"
    rows = _load_summary(summary_path)
    cache_path = output_dir / "acc_replay_cache.json"
    cache = _load_cache(cache_path)

    result_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        report_path = Path(row["report_path"])
        error_path = Path(row["error_csv"])
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        error_metrics = _metrics_from_error_csv(error_path, payload)

        acc_key = _acc_cache_key(report_path, payload)
        if acc_key not in cache or _cache_failed(cache[acc_key]):
            cache[acc_key] = _solve_acc_metrics(payload)
            _save_cache(cache_path, cache)

        result_rows.append({
            "data_file": row.get("sample") or Path(str(payload.get("data_path", ""))).name,
            "motion_type": row.get("motion_type", ""),
            "hf": error_metrics["hf"],
            "fft": error_metrics["fft"],
            "acc": cache[acc_key],
        })

    _write_all_tables(output_dir, result_rows)
    _write_markdown(output_dir / "generalization_stat_tables.md", result_rows)
    _save_cache(cache_path, cache)


def _load_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_cache(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_cache(path: Path, cache: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _metrics_from_error_csv(path: Path, payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    method_rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            method_rows[str(row.get("method", "")).strip()] = row

    adaptive_filter = str(payload.get("adaptive_filter", "")).lower()
    hf_method = "K-LMS+H" if adaptive_filter == "klms" else "LMS+H"
    if hf_method not in method_rows:
        raise KeyError(f"{path} missing method {hf_method}")
    if "FFT" not in method_rows:
        raise KeyError(f"{path} missing method FFT")
    return {
        "hf": _metrics_from_error_row(method_rows[hf_method]),
        "fft": _metrics_from_error_row(method_rows["FFT"]),
    }


def _metrics_from_error_row(row: dict[str, str]) -> dict[str, float]:
    return {
        "total_aae": _num(row.get("total_aae")),
        "rest_aae": _num(row.get("rest_aae")),
        "motion_aae": _num(row.get("motion_aae")),
        "total_hit_rate_5bpm": _num(row.get("total_hit_rate_5bpm")),
        "rest_hit_rate_5bpm": _num(row.get("rest_hit_rate_5bpm")),
        "motion_hit_rate_5bpm": _num(row.get("motion_hit_rate_5bpm")),
    }


def _acc_cache_key(report_path: Path, payload: dict[str, Any]) -> str:
    params = payload.get("best_params", {}) or {}
    return "::".join([
        "acc-v1",
        str(report_path),
        str(payload.get("data_path", "")),
        str(payload.get("ref_path", "")),
        json.dumps(params, sort_keys=True, ensure_ascii=False),
    ])


def _cache_failed(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get("error"))


def _solve_acc_metrics(payload: dict[str, Any]) -> dict[str, float] | dict[str, str]:
    try:
        cfg = _config_from_payload(payload, payload.get("best_params", {}) or {}, ("ACC",))
        result = solve_v2(cfg)
        return _metrics_from_hr(np.asarray(result.HR, dtype=float), column=3)
    except Exception as exc:
        metrics: dict[str, Any] = _empty_metrics()
        metrics["error"] = str(exc)
        return metrics


def _config_from_payload(
    payload: dict[str, Any],
    params: dict[str, Any],
    reference_groups_order: tuple[str, ...],
) -> V2RunConfig:
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
        if key in field_names and key not in {
            "data_path",
            "ref_path",
            "adaptive_filter",
            "reference_groups_order",
            "extras",
        }:
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


def _empty_metrics() -> dict[str, float]:
    return {
        "total_aae": math.nan,
        "rest_aae": math.nan,
        "motion_aae": math.nan,
        "total_hit_rate_5bpm": math.nan,
        "rest_hit_rate_5bpm": math.nan,
        "motion_hit_rate_5bpm": math.nan,
    }


def _aae(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if not bool(np.any(mask)):
        return math.nan
    return float(np.mean(np.abs(pred[mask] - ref[mask])))


def _hit(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if not bool(np.any(mask)):
        return math.nan
    return float(np.mean(np.abs(pred[mask] - ref[mask]) <= 5.0))


def _write_all_tables(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    for file_name, scope, metric_kind in TABLE_SPECS:
        table_rows = _stat_rows(rows, scope=scope, metric_kind=metric_kind)
        _write_table_csv(output_dir / file_name, table_rows)


def _stat_rows(rows: list[dict[str, Any]], *, scope: str, metric_kind: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: _sort_key(str(item["data_file"]))):
        rec: dict[str, Any] = {"data_file": row["data_file"]}
        if metric_kind == "mae":
            rec["klms_hf"] = _metric(row["hf"], scope, "aae")
            rec["klms_acc"] = _metric(row["acc"], scope, "aae")
            rec["klms_fft"] = _metric(row["fft"], scope, "aae")
            rec["klms_hf_vs_acc_delta"] = _mae_reduction(rec["klms_hf"], rec["klms_acc"])
        else:
            rec["klms_hf"] = 100.0 * _metric(row["hf"], scope, "hit_rate_5bpm")
            rec["klms_acc"] = 100.0 * _metric(row["acc"], scope, "hit_rate_5bpm")
            rec["klms_fft"] = 100.0 * _metric(row["fft"], scope, "hit_rate_5bpm")
            rec["klms_hf_vs_acc_delta"] = _rate_delta_pp(rec["klms_hf"], rec["klms_acc"])
        out.append(rec)
    out.append(_average_row(out))
    return out


def _metric(metrics: dict[str, Any], scope: str, metric_name: str) -> float:
    if metric_name == "aae":
        return _num(metrics.get(f"{scope}_aae"))
    return _num(metrics.get(f"{scope}_{metric_name}"))


def _sort_key(data_file: str) -> tuple[int, str]:
    order = {"tiaosheng": 0, "bobi": 1, "fuwo": 2, "kaihe": 3, "wanju": 4}
    motion = data_file.split("_", 2)[1] if data_file.startswith("multi_") else data_file
    motion = "".join(ch for ch in motion if not ch.isdigit())
    return (order.get(motion, 99), data_file)


def _mae_reduction(new_value: Any, baseline_value: Any) -> float:
    new = _num(new_value)
    baseline = _num(baseline_value)
    if not math.isfinite(new) or not math.isfinite(baseline) or abs(baseline) < 1e-12:
        return math.nan
    return 100.0 * (baseline - new) / baseline


def _rate_delta_pp(new_value: Any, baseline_value: Any) -> float:
    new = _num(new_value)
    baseline = _num(baseline_value)
    if not math.isfinite(new) or not math.isfinite(baseline):
        return math.nan
    return new - baseline


def _average_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rec: dict[str, Any] = {"data_file": "平均"}
    for key in ("klms_hf", "klms_acc", "klms_fft", "klms_hf_vs_acc_delta"):
        rec[key] = _mean(rows, key)
    return rec


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_num(row.get(key)) for row in rows]
    vals = [value for value in vals if math.isfinite(value)]
    if not vals:
        return math.nan
    return float(np.mean(vals))


def _write_table_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["data_file", "klms_hf", "klms_acc", "klms_fft", "klms_hf_vs_acc_delta"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v2 泛化评估统计表",
        "",
        f"- 样本数：{len(rows)}",
        "- HF/FFT 指标来源：泛化输出目录中的 `*-error.csv`。",
        "- ACC 指标来源：复用对应 JSON 的 `best_params`，仅将 `reference_groups_order` 改为 `ACC` 后重放。",
        "- CSV 输出：`table_full_mae.csv`、`table_full_r5.csv`、`table_motion_mae.csv`、`table_motion_r5.csv`。",
        "",
    ]
    for title, file_name, metric_kind in (
        ("全段 MAE", "table_full_mae.csv", "mae"),
        ("全段 R5", "table_full_r5.csv", "r5"),
        ("运动段 MAE", "table_motion_mae.csv", "mae"),
        ("运动段 R5", "table_motion_r5.csv", "r5"),
    ):
        table_rows = list(csv.DictReader((path.parent / file_name).open("r", encoding="utf-8-sig")))
        lines.append(f"## {title}")
        lines.append("")
        lines.extend(_markdown_table(table_rows, metric_kind=metric_kind))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(rows: list[dict[str, Any]], *, metric_kind: str) -> list[str]:
    delta_label = "HF较ACC降低(%)" if metric_kind == "mae" else "HF较ACC提高(pp)"
    columns = [
        ("data_file", "数据文件"),
        ("klms_hf", "KLMS-HF"),
        ("klms_acc", "KLMS-ACC"),
        ("klms_fft", "FFT"),
        ("klms_hf_vs_acc_delta", delta_label),
    ]
    out = ["| " + " | ".join(label for _, label in columns) + " |"]
    out.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        out.append("| " + " | ".join(_format_cell(row.get(key, ""), key=key) for key, _ in columns) + " |")
    return out


def _format_cell(value: Any, *, key: str) -> str:
    if key == "data_file":
        return str(value)
    num = _num(value)
    if not math.isfinite(num):
        return "--"
    return f"{num:.2f}"


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


if __name__ == "__main__":
    main()
