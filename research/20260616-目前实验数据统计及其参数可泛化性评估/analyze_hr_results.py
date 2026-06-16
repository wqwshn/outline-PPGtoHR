from __future__ import annotations

import csv
import dataclasses
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ppg_hr.v2.solver import solve_v2
from ppg_hr.v2.types import V2RunConfig


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
CACHE_SCHEMA_VERSION = "20260616-path-aware"
OUT = ROOT / "analysis_outputs"
CACHE = OUT / f"replay_cache_{CACHE_SCHEMA_VERSION}.json"

MOTION_LABELS = {
    "tiaosheng": "跳绳",
    "bobi": "波比跳",
    "fuwo": "俯卧撑",
    "kaihe": "开合跳",
    "wanju": "弯举",
}
MOTION_ORDER = {
    "tiaosheng": 0,
    "bobi": 1,
    "fuwo": 2,
    "kaihe": 3,
    "wanju": 4,
}

PARAM_KEYS = (
    "fs_target",
    "max_order",
    "lms_mu_base",
    "klms_step_size",
    "klms_sigma",
    "klms_epsilon",
    "smooth_win_len",
    "spec_penalty_width",
    "hr_range_hz",
    "slew_limit_bpm",
    "slew_step_bpm",
    "hr_range_rest",
    "slew_limit_rest",
    "slew_step_rest",
    "time_bias",
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()
    reports = _collect_reports()

    rows: list[dict[str, Any]] = []
    for report_path, payload in reports:
        row = _base_row(report_path, payload)
        hr = np.asarray(payload.get("hr", []), dtype=float)
        fft_metrics = _metrics_from_hr(hr, column=2)
        hf_metrics = _metrics_from_hr(hr, column=3)
        row.update(_prefix_metrics("fft", fft_metrics))
        row.update(_prefix_metrics("hf", hf_metrics))

        acc_key = _cache_key("acc", row["report_rel"], row["data_path"], row["ref_path"])
        if acc_key not in cache or _cache_failed(cache[acc_key]):
            cache[acc_key] = _solve_with_params(payload, payload.get("best_params", {}), ("ACC",))
            _save_cache(cache)
        row.update(_prefix_metrics("acc", cache[acc_key]))

        for key in PARAM_KEYS:
            row[key] = payload.get("best_params", {}).get(key)
        rows.append(row)

    shared_rows = _evaluate_scene_shared_candidates(reports, cache)
    _save_cache(cache)

    _write_csv(OUT / "all_result_summary.csv", rows)
    _write_csv(OUT / "scene_shared_parameter_replay.csv", shared_rows)
    scene_rows = _scene_summary(rows)
    _write_csv(OUT / "scene_summary.csv", scene_rows)
    subject_rows = _subject_summary(rows)
    _write_csv(OUT / "subject_summary.csv", subject_rows)
    param_rows = _parameter_summary(rows)
    _write_csv(OUT / "parameter_summary.csv", param_rows)

    _write_result_table(rows, scene_rows, shared_rows)
    _write_report(rows, scene_rows, subject_rows, param_rows, shared_rows)


def _load_cache() -> dict[str, Any]:
    if CACHE.is_file():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    tmp_path = CACHE.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(CACHE)


def _cache_key(kind: str, *parts: Any) -> str:
    return "::".join([kind, CACHE_SCHEMA_VERSION, *(str(part) for part in parts)])


def _collect_reports() -> list[tuple[Path, dict[str, Any]]]:
    reports: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(ROOT.rglob("*.json")):
        if OUT in path.parents:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("schema_version") != "v2":
            continue
        if payload.get("reference_order_key") != "HF":
            continue
        if payload.get("adaptive_filter") not in {"lms", "klms"}:
            continue
        payload["_report_path"] = str(path)
        reports.append((path, payload))
    return reports


def _base_row(report_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    raw_data_path = Path(str(payload.get("data_path", "")))
    data_path = _resolve_report_path(payload, "data_path")
    ref_path = _resolve_report_path(payload, "ref_path")
    sample = _resolved_sample_name(report_path, data_path)
    sample_stem = Path(sample).stem
    motion_type = _motion_type(sample_stem)
    data_date = _date_from_path(data_path) or _date_from_path(report_path)
    subject = _subject_from_payload(sample_stem, data_date)
    rel = report_path.relative_to(ROOT).as_posix()
    return {
        "report_rel": rel,
        "raw_data_path": str(raw_data_path),
        "data_path": str(data_path),
        "ref_path": str(ref_path),
        "data_date": data_date,
        "research_bucket": report_path.relative_to(ROOT).parts[0],
        "subject": subject,
        "sample": sample,
        "sample_stem": sample_stem,
        "motion_type": motion_type,
        "motion_label": MOTION_LABELS.get(motion_type, motion_type),
        "adaptive_filter": str(payload.get("adaptive_filter", "")),
        "reference_order_key": str(payload.get("reference_order_key", "")),
        "analysis_scope": str(payload.get("analysis_scope", "")),
        "motion_start_s": _safe_get(payload, ("motion_segment", "start_s")),
        "motion_end_s": _safe_get(payload, ("motion_segment", "end_s")),
        "used_adaptive_windows": payload.get("used_adaptive_windows"),
        "unreliable_windows": payload.get("unreliable_windows"),
    }


def _motion_type(sample_stem: str) -> str:
    m = re.match(r"multi_([A-Za-z]+)\d+", sample_stem)
    return m.group(1).lower() if m else "unknown"


def _date_from_path(path: Path) -> str:
    for part in path.parts:
        if re.fullmatch(r"20\d{6}", part):
            return part
    return ""


def _subject_from_payload(sample_stem: str, data_date: str) -> str:
    if "_TS" in sample_stem or data_date == "20260615":
        return "TS"
    return "LYX"


def _safe_get(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


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


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def _solve_with_params(
    payload: dict[str, Any],
    params: dict[str, Any],
    reference_groups_order: tuple[str, ...],
) -> dict[str, float] | dict[str, str]:
    try:
        cfg = _config_from_payload(payload, params, reference_groups_order)
        result = solve_v2(cfg)
        return _metrics_from_hr(np.asarray(result.HR, dtype=float), column=3)
    except Exception as exc:
        metrics: dict[str, Any] = _empty_metrics()
        metrics["error"] = str(exc)
        return metrics


def _cache_failed(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get("error"))


def _config_from_payload(
    payload: dict[str, Any],
    params: dict[str, Any],
    reference_groups_order: tuple[str, ...],
) -> V2RunConfig:
    field_names = {f.name for f in dataclasses.fields(V2RunConfig)}
    cfg_dict: dict[str, Any] = {
        "data_path": _resolve_report_path(payload, "data_path"),
        "ref_path": _resolve_report_path(payload, "ref_path"),
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


def _resolve_report_path(payload: dict[str, Any], key: str) -> Path:
    path = Path(str(payload.get(key, "")))
    report_path = Path(str(payload.get("_report_path", "")))
    report_sample = _report_sample_stem(report_path)
    report_date = _date_from_path(report_path)
    if report_sample and report_date:
        if key == "data_path":
            preferred_names = [f"{report_sample}.csv"]
        elif key == "ref_path":
            preferred_names = [f"{report_sample}_HR_ref.csv"]
        else:
            preferred_names = []
        resolved = _find_first_existing_path([WORKSPACE / "data" / report_date], preferred_names)
        if resolved is not None:
            return resolved
    if path.is_file():
        return path
    date = _date_from_path(path)
    names = [path.name] if path.name else []
    if report_sample:
        if key == "data_path":
            names.insert(0, f"{report_sample}.csv")
        elif key == "ref_path":
            names.insert(0, f"{report_sample}_HR_ref.csv")
    names = [name for i, name in enumerate(names) if name and name not in names[:i]]
    if not names:
        return path
    roots: list[Path] = []
    if report_date:
        roots.append(WORKSPACE / "data" / report_date)
    if date:
        roots.append(WORKSPACE / "data" / date)
    roots.append(WORKSPACE / "data")
    return _find_first_existing_path(roots, names) or path


def _find_first_existing_path(roots: list[Path], names: list[str]) -> Path | None:
    seen_roots: set[Path] = set()
    for root in roots:
        if root in seen_roots or not root.is_dir():
            continue
        seen_roots.add(root)
        for name in names:
            matches = sorted(root.rglob(name))
            if matches:
                return matches[0]
    return None


def _resolved_sample_name(report_path: Path, data_path: Path) -> str:
    if data_path.is_file():
        return data_path.name
    report_sample = _report_sample_stem(report_path)
    if report_sample:
        return f"{report_sample}.csv"
    return data_path.name or report_path.stem


def _report_sample_stem(report_path: Path) -> str:
    stem = report_path.stem
    for marker in ("-green-", "-red-", "-ir-"):
        if marker in stem:
            return stem.split(marker, 1)[0]
    return ""


def _evaluate_scene_shared_candidates(
    reports: list[tuple[Path, dict[str, Any]]],
    cache: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path, payload in reports:
        base = _base_row(path, payload)
        grouped[(str(base["motion_type"]), str(payload.get("adaptive_filter", "")))].append((path, payload))

    rows: list[dict[str, Any]] = []
    for (motion_type, adaptive_filter), group_reports in sorted(grouped.items()):
        if len(group_reports) < 2:
            continue
        candidate_path, candidate_payload = _medoid_report(group_reports)
        candidate_base = _base_row(candidate_path, candidate_payload)
        candidate_sample = str(candidate_base["sample_stem"])
        candidate_data_path = str(candidate_base["data_path"])
        candidate_params = candidate_payload.get("best_params", {})
        for report_path, payload in group_reports:
            base = _base_row(report_path, payload)
            key = _cache_key(
                "shared",
                motion_type,
                adaptive_filter,
                candidate_sample,
                candidate_data_path,
                base["report_rel"],
                base["data_path"],
                base["ref_path"],
            )
            if key not in cache or _cache_failed(cache[key]):
                cache[key] = _solve_with_params(payload, candidate_params, ("HF",))
                _save_cache(cache)
            metrics = cache[key]
            row = {
                **base,
                "candidate_sample": candidate_sample,
                "candidate_report_rel": candidate_path.relative_to(ROOT).as_posix(),
                "independent_hf_total_aae": _metrics_from_hr(
                    np.asarray(payload.get("hr", []), dtype=float), column=3
                )["total_aae"],
            }
            row.update(_prefix_metrics("shared_hf", metrics))
            row["shared_minus_independent_aae"] = _num(row.get("shared_hf_total_aae")) - _num(
                row.get("independent_hf_total_aae")
            )
            rows.append(row)
    return rows


def _medoid_report(
    reports: list[tuple[Path, dict[str, Any]]],
) -> tuple[Path, dict[str, Any]]:
    vectors = [(_param_vector(payload.get("best_params", {})), path, payload) for path, payload in reports]
    best: tuple[float, Path, dict[str, Any]] | None = None
    for vec, path, payload in vectors:
        dist = 0.0
        count = 0
        for other, _, _ in vectors:
            d = _vector_distance(vec, other)
            if math.isfinite(d):
                dist += d
                count += 1
        score = dist / max(count, 1)
        if best is None or score < best[0]:
            best = (score, path, payload)
    assert best is not None
    return best[1], best[2]


def _param_vector(params: dict[str, Any]) -> dict[str, float]:
    vector: dict[str, float] = {}
    for key in PARAM_KEYS:
        value = params.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            vector[key] = float(value)
    return vector


def _vector_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    if not keys:
        return math.nan
    vals = []
    for key in keys:
        scale = max(abs(a[key]), abs(b[key]), 1.0)
        vals.append(abs(a[key] - b[key]) / scale)
    return float(np.mean(vals))


def _scene_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["motion_type"], row["adaptive_filter"])].append(row)
    out: list[dict[str, Any]] = []
    for (motion_type, adaptive_filter), items in sorted(grouped.items()):
        out.append({
            "motion_type": motion_type,
            "motion_label": MOTION_LABELS.get(motion_type, motion_type),
            "adaptive_filter": adaptive_filter,
            "n": len(items),
            "subjects": "/".join(sorted({str(x["subject"]) for x in items})),
            "dates": "/".join(sorted({str(x["data_date"]) for x in items})),
            "hf_total_aae_mean": _mean(items, "hf_total_aae"),
            "hf_motion_aae_mean": _mean(items, "hf_motion_aae"),
            "hf_r5_mean": _mean(items, "hf_total_hit_rate_5bpm"),
            "acc_total_aae_mean": _mean(items, "acc_total_aae"),
            "acc_r5_mean": _mean(items, "acc_total_hit_rate_5bpm"),
            "fft_total_aae_mean": _mean(items, "fft_total_aae"),
            "fft_r5_mean": _mean(items, "fft_total_hit_rate_5bpm"),
        })
    return out


def _parameter_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["motion_type"], row["adaptive_filter"])].append(row)
    out: list[dict[str, Any]] = []
    for (motion_type, adaptive_filter), items in sorted(grouped.items()):
        rec: dict[str, Any] = {
            "motion_type": motion_type,
            "motion_label": MOTION_LABELS.get(motion_type, motion_type),
            "adaptive_filter": adaptive_filter,
            "n": len(items),
            "unique_param_vectors": len({_param_signature(x) for x in items}),
        }
        for key in PARAM_KEYS:
            values = [x.get(key) for x in items if x.get(key) not in {None, ""}]
            if not values:
                continue
            counts = Counter(str(v) for v in values)
            mode_value, mode_count = counts.most_common(1)[0]
            rec[f"{key}_mode"] = mode_value
            rec[f"{key}_mode_share"] = mode_count / len(values)
            numeric = [float(v) for v in values if isinstance(v, (int, float))]
            if numeric:
                rec[f"{key}_median"] = float(np.median(numeric))
        out.append(rec)
    return out


def _subject_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["subject"], row["adaptive_filter"])].append(row)
    out: list[dict[str, Any]] = []
    for (subject, adaptive_filter), items in sorted(grouped.items()):
        out.append({
            "subject": subject,
            "adaptive_filter": adaptive_filter,
            "n": len(items),
            "motions": "/".join(sorted({str(x["motion_label"]) for x in items})),
            "dates": "/".join(sorted({str(x["data_date"]) for x in items})),
            "hf_total_aae_mean": _mean(items, "hf_total_aae"),
            "hf_r5_mean": _mean(items, "hf_total_hit_rate_5bpm"),
            "acc_total_aae_mean": _mean(items, "acc_total_aae"),
            "fft_total_aae_mean": _mean(items, "fft_total_aae"),
        })
    return out


def _param_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in PARAM_KEYS)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_num(row.get(key)) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_result_table(
    rows: list[dict[str, Any]],
    scene_rows: list[dict[str, Any]],
    shared_rows: list[dict[str, Any]],
) -> None:
    del scene_rows, shared_rows
    table_specs = [
        (
            "full_mae",
            "total",
            "mae",
            "不同运动场景下心率解算全段 MAE 统计",
            "tab:hr-full-mae",
        ),
        (
            "full_r5",
            "total",
            "r5",
            "不同运动场景下心率解算全段 \\(R_5\\) 命中率统计",
            "tab:hr-full-r5",
        ),
        (
            "motion_mae",
            "motion",
            "mae",
            "不同运动场景下心率解算运动段 MAE 统计",
            "tab:hr-motion-mae",
        ),
        (
            "motion_r5",
            "motion",
            "r5",
            "不同运动场景下心率解算运动段 \\(R_5\\) 命中率统计",
            "tab:hr-motion-r5",
        ),
    ]

    lines: list[str] = []
    lines.append("# 目前实验数据心率解算结果汇总表")
    lines.append("")
    lines.append(f"- 统计时间：2026-06-16")
    lines.append(f"- v2 HF 独立优化报告数：{len(rows)}")
    lines.append("- ACC 对比：复用每个 HF 报告的 `best_params`，仅将 `reference_groups_order` 改为 `ACC` 后重放。")
    lines.append("- 行标签格式：`运动编号(日期-受试者)`；缺少 KLMS 独立优化结果的样本以 `--` 表示。")
    lines.append("- 同口径 CSV：`analysis_outputs/table_full_mae.csv`、`table_full_r5.csv`、`table_motion_mae.csv`、`table_motion_r5.csv`。")
    lines.append("")

    for file_stem, scope, metric_kind, caption, label in table_specs:
        table_rows = _paired_stat_rows(rows, scope=scope, metric_kind=metric_kind)
        _write_stat_csv(OUT / f"table_{file_stem}.csv", table_rows)
        lines.append(f"## {caption}")
        lines.append("")
        lines.append("```latex")
        lines.extend(_latex_stat_table(
            table_rows,
            caption=caption,
            label=label,
            metric_kind=metric_kind,
        ))
        lines.append("```")
        lines.append("")

    (ROOT / "结果汇总统计表_20260616.md").write_text("\n".join(lines), encoding="utf-8")


def _paired_stat_rows(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    metric_kind: str,
) -> list[dict[str, Any]]:
    pairs: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["data_date"]),
            str(row["subject"]),
            str(row["motion_type"]),
            str(row["sample_stem"]),
        )
        pairs[key][str(row["adaptive_filter"])] = row

    stat_rows: list[dict[str, Any]] = []
    for key, by_filter in sorted(pairs.items(), key=lambda item: _paired_sort_key(item[0])):
        data_date, subject, motion_type, sample_stem = key
        lms = by_filter.get("lms")
        klms = by_filter.get("klms")
        rec: dict[str, Any] = {
            "scenario": _scenario_label(motion_type, sample_stem, data_date, subject),
            "motion_type": motion_type,
            "sample_stem": sample_stem,
            "data_date": data_date,
            "subject": subject,
        }
        rec.update(_stat_values_for_filter(lms, prefix="lms", scope=scope, metric_kind=metric_kind))
        rec.update(_stat_values_for_filter(klms, prefix="klms", scope=scope, metric_kind=metric_kind))
        if metric_kind == "mae":
            rec["lms_hf_vs_acc_delta"] = _mae_reduction(rec["lms_hf"], rec["lms_acc"])
            rec["klms_vs_lms_delta"] = _mae_reduction(rec["klms_hf"], rec["lms_hf"])
        else:
            rec["lms_hf_vs_acc_delta"] = _rate_delta_pp(rec["lms_hf"], rec["lms_acc"])
            rec["klms_vs_lms_delta"] = _rate_delta_pp(rec["klms_hf"], rec["lms_hf"])
        stat_rows.append(rec)

    stat_rows.append(_average_stat_row(stat_rows))
    return stat_rows


def _write_stat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scenario",
        "motion_type",
        "sample_stem",
        "data_date",
        "subject",
        "lms_hf",
        "lms_acc",
        "lms_fft",
        "lms_hf_vs_acc_delta",
        "klms_hf",
        "klms_acc",
        "klms_fft",
        "klms_vs_lms_delta",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _paired_sort_key(key: tuple[str, str, str, str]) -> tuple[Any, ...]:
    data_date, subject, motion_type, sample_stem = key
    return (
        MOTION_ORDER.get(motion_type, 99),
        _sample_number(sample_stem),
        data_date,
        subject,
        sample_stem,
    )


def _sample_number(sample_stem: str) -> int:
    m = re.match(r"multi_[A-Za-z]+(\d+)", sample_stem)
    return int(m.group(1)) if m else 999


def _scenario_label(motion_type: str, sample_stem: str, data_date: str, subject: str) -> str:
    label = MOTION_LABELS.get(motion_type, motion_type)
    suffix = data_date[4:] if len(data_date) == 8 else data_date
    sample_no = _sample_number(sample_stem)
    sample_tag = str(sample_no) if sample_no != 999 else sample_stem
    return f"{label}{sample_tag}({suffix}-{subject})"


def _stat_values_for_filter(
    row: dict[str, Any] | None,
    *,
    prefix: str,
    scope: str,
    metric_kind: str,
) -> dict[str, float]:
    if row is None:
        return {f"{prefix}_{name}": math.nan for name in ("hf", "acc", "fft")}
    if metric_kind == "mae":
        return {
            f"{prefix}_hf": _num(row.get(f"hf_{scope}_aae")),
            f"{prefix}_acc": _num(row.get(f"acc_{scope}_aae")),
            f"{prefix}_fft": _num(row.get(f"fft_{scope}_aae")),
        }
    metric_name = f"{scope}_hit_rate_5bpm"
    return {
        f"{prefix}_hf": 100.0 * _num(row.get(f"hf_{metric_name}")),
        f"{prefix}_acc": 100.0 * _num(row.get(f"acc_{metric_name}")),
        f"{prefix}_fft": 100.0 * _num(row.get(f"fft_{metric_name}")),
    }


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


def _average_stat_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "scenario": "平均",
        "motion_type": "",
        "sample_stem": "",
        "data_date": "",
        "subject": "",
    }
    for key in (
        "lms_hf",
        "lms_acc",
        "lms_fft",
        "lms_hf_vs_acc_delta",
        "klms_hf",
        "klms_acc",
        "klms_fft",
        "klms_vs_lms_delta",
    ):
        rec[key] = _mean(rows, key)
    return rec


def _latex_stat_table(
    rows: list[dict[str, Any]],
    *,
    caption: str,
    label: str,
    metric_kind: str,
) -> list[str]:
    if metric_kind == "mae":
        unit = "bpm"
        lms_delta = "\\makecell{HF 较 ACC\\\\降低(\\%)}"
        klms_delta = "\\makecell{KLMS 较 LMS\\\\降低(\\%)}"
    else:
        unit = "\\%"
        lms_delta = "\\makecell{HF 较 ACC\\\\提高(pp)}"
        klms_delta = "\\makecell{KLMS 较 LMS\\\\提高(pp)}"

    out = [
        "\\begin{table*}[!t]",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\centering",
        "  \\xiaowuhao",
        "  \\setlength{\\tabcolsep}{3.0pt}",
        "  \\renewcommand{\\arraystretch}{1.15}",
        "  \\begin{tabular}{lrrrrrrrr}",
        "    \\toprule",
        "    \\makecell{实验\\\\场景} &",
        "    \\multicolumn{4}{c}{LMS 参考信号对比} &",
        "    \\multicolumn{4}{c}{KLMS 结果与提升} \\\\",
        "    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        f"    & \\makecell{{HF-LMS\\\\({unit})}}",
        f"    & \\makecell{{ACC-LMS\\\\({unit})}}",
        f"    & \\makecell{{FFT\\\\({unit})}}",
        f"    & {lms_delta}",
        f"    & \\makecell{{HF-KLMS\\\\({unit})}}",
        f"    & \\makecell{{ACC-KLMS\\\\({unit})}}",
        f"    & \\makecell{{FFT\\\\({unit})}}",
        f"    & {klms_delta} \\\\",
        "    \\midrule",
    ]
    for row in rows:
        out.append(
            "    "
            + _latex_scenario_cell(str(row["scenario"]))
            + " & "
            + " & ".join(
                _latex_num(row.get(key))
                for key in (
                    "lms_hf",
                    "lms_acc",
                    "lms_fft",
                    "lms_hf_vs_acc_delta",
                    "klms_hf",
                    "klms_acc",
                    "klms_fft",
                    "klms_vs_lms_delta",
                )
            )
            + " \\\\"
        )
    out.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table*}",
    ])
    return out


def _latex_scenario_cell(value: str) -> str:
    if value == "平均":
        return "平均"
    m = re.match(r"(.+?)\\((.+)\\)", value)
    if not m:
        return value
    return f"\\makecell{{{m.group(1)}\\\\{m.group(2)}}}"


def _latex_num(value: Any) -> str:
    num = _num(value)
    if not math.isfinite(num):
        return "--"
    return f"{num:.2f}"


def _shared_summary(shared_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in shared_rows:
        grouped[(row["motion_type"], row["adaptive_filter"])].append(row)
    out = []
    for (motion_type, adaptive_filter), items in sorted(grouped.items()):
        out.append({
            "motion_type": motion_type,
            "motion_label": MOTION_LABELS.get(motion_type, motion_type),
            "adaptive_filter": adaptive_filter,
            "n": len(items),
            "candidate_sample": items[0].get("candidate_sample"),
            "independent_mean_aae": _mean(items, "independent_hf_total_aae"),
            "shared_mean_aae": _mean(items, "shared_hf_total_aae"),
            "delta_aae": _mean(items, "shared_minus_independent_aae"),
        })
    return out


def _write_report(
    rows: list[dict[str, Any]],
    scene_rows: list[dict[str, Any]],
    subject_rows: list[dict[str, Any]],
    param_rows: list[dict[str, Any]],
    shared_rows: list[dict[str, Any]],
) -> None:
    overall = _overall_stats(rows)
    shared_summary = _shared_summary(shared_rows)
    lines: list[str] = []
    lines.append("# 参数泛化性评估报告")
    lines.append("")
    lines.append("## 技术摘要")
    lines.append("")
    lines.append(
        f"本次共解析 {len(rows)} 个 v2 HF 独立贝叶斯优化报告，其中 LMS {overall['lms_n']} 个、KLMS {overall['klms_n']} 个。"
        f"HF 路径总体 MAE 为 {overall['hf_mae']:.2f} bpm，ACC 参考组在复用同一参数后总体 MAE 为 {overall['acc_mae']:.2f} bpm，"
        f"FFT 基线为 {overall['fft_mae']:.2f} bpm。"
    )
    lines.append("")
    lines.append(
        "结论上，HF 仍是当前数据中最值得优先研究和固化的参考信号组；ACC 作为单独参考组在多数场景下不如 HF，"
        "更适合作为后续多参考级联或运动分割诊断信号，而不是直接替代 HF。KLMS 的主要价值出现在 LMS 易失败的强运动/非线性残差样本上，"
        "但不是所有场景都稳定优于 LMS。"
    )
    lines.append("")
    lines.append(
        "参数泛化方面，独立最优参数的唯一组合数几乎等于样本数，说明当前 BO 最优点仍明显带有数据个体性。"
        "直接从独立最优结果中挑一套 medoid 作为场景共享参数时，跳绳和弯举损失较小，俯卧撑处于可继续优化区间，"
        "波比跳和开合跳损失很大，不能直接固化某个单样本最优参数。"
    )
    lines.append("")
    lines.append("## 泛化性概念边界")
    lines.append("")
    lines.append("- 场景内泛化性：同一种运动类型内，不同采集批次或重复样本共用一套参数。")
    lines.append("- 跨时间泛化性：同一受试者在不同采集日期之间共用参数。")
    lines.append("- 跨个体泛化性：LYX 与 TS 之间共用参数；当前样本量仍小，只能做早期迹象判断。")
    lines.append("- 多场景参数复用：少于运动场景数量的参数组覆盖多类运动，例如跳绳/开合跳共享一套、波比跳/俯卧撑共享另一套。")
    lines.append("")
    lines.append("## 独立优化结果说明了什么")
    lines.append("")
    lines.extend(_markdown_table(
        scene_rows,
        [
            ("motion_label", "运动"),
            ("adaptive_filter", "滤波器"),
            ("n", "n"),
            ("hf_total_aae_mean", "HF MAE"),
            ("acc_total_aae_mean", "ACC MAE"),
            ("fft_total_aae_mean", "FFT MAE"),
            ("hf_r5_mean", "HF R5"),
        ],
    ))
    lines.append("")
    lines.append(
        "这些数字是“每个数据单独 BO 后”的上限式结果，不能直接等价为上线参数表现。"
        "它们更适合回答两个问题：第一，某运动场景在当前机制下是否可解；第二，独立最优参数是否呈现足够一致的规律，"
        "值得收敛成场景共享参数。"
    )
    lines.append("")
    lines.append("## 受试者与时间分层")
    lines.append("")
    lines.extend(_markdown_table(
        subject_rows,
        [
            ("subject", "受试者"),
            ("adaptive_filter", "滤波器"),
            ("n", "n"),
            ("dates", "采集日期"),
            ("motions", "覆盖运动"),
            ("hf_total_aae_mean", "HF MAE"),
            ("hf_r5_mean", "HF R5"),
            ("acc_total_aae_mean", "ACC MAE"),
            ("fft_total_aae_mean", "FFT MAE"),
        ],
    ))
    lines.append("")
    lines.append(
        "当前跨个体证据仍偏早期：除 20260615 数据外，其余数据均按 LYX 处理，LYX 样本数量多于 TS，且弯举仅 TS 有样本。"
        "因此跨个体泛化不能只看总体均值，后续应在每个运动类型内做 LYX->TS、TS->LYX 的 leave-subject-out 重放。"
    )
    lines.append("")
    lines.append("## 参数集中趋势")
    lines.append("")
    lines.extend(_markdown_table(
        param_rows,
        [
            ("motion_label", "运动"),
            ("adaptive_filter", "滤波器"),
            ("n", "n"),
            ("unique_param_vectors", "唯一参数组数"),
            ("fs_target_mode", "fs_target众数"),
            ("fs_target_mode_share", "占比"),
            ("max_order_mode", "max_order众数"),
            ("smooth_win_len_mode", "smooth众数"),
            ("spec_penalty_width_mode", "惩罚宽度众数"),
            ("hr_range_hz_mode", "运动搜索宽度众数"),
            ("slew_limit_bpm_mode", "运动限幅众数"),
        ],
    ))
    lines.append("")
    lines.append(
        "如果一个场景内 `fs_target`、`smooth_win_len`、`spec_penalty_width` 和运动段搜索/限幅参数的众数占比高，"
        "说明该场景具备参数固化条件；反之，若唯一参数组数接近样本数，泛化风险主要来自参数机制仍在补偿样本差异。"
    )
    lines.append("")
    lines.append("## 场景共享候选参数重放")
    lines.append("")
    lines.extend(_markdown_table(
        shared_summary,
        [
            ("motion_label", "运动"),
            ("adaptive_filter", "滤波器"),
            ("n", "n"),
            ("candidate_sample", "候选参数来源"),
            ("independent_mean_aae", "独立 MAE"),
            ("shared_mean_aae", "共享候选 MAE"),
            ("delta_aae", "损失"),
        ],
    ))
    lines.append("")
    lines.append(
        "这里的共享参数候选是组内独立最优参数的 medoid，不是重新训练出的 all-train 最优参数。"
        "它的意义是快速估计“从已有 BO 结果中挑一套场景参数”会付出多少误差代价。"
        "正式发布参数前仍应运行 `run_v2_generalization` 的 all_train 与 leave_one_group_out。"
    )
    lines.append("")
    lines.append(
        "按这张表做决策：跳绳 KLMS、弯举 LMS/KLMS 可以优先尝试固化；俯卧撑 LMS/KLMS 和跳绳 LMS 需要正式 all-train 共享 BO 验证；"
        "波比跳和开合跳不宜直接复用单样本参数，说明参数之间存在较强耦合，或者当前后处理机制仍在用参数补偿样本级失效。"
    )
    lines.append("")
    lines.append("## 后续工作建议")
    lines.append("")
    lines.append("1. 优先把泛化目标从“单数据最优”切换为“场景内共享参数最优”，评价指标使用场景均值 MAE、最差样本 MAE 与 R5 下限三者共同约束。")
    lines.append("2. 对参数集中度高的场景先固化一套 LMS-HF 参数；对 LMS 离群失败明显的场景保留 KLMS-HF 作为二级方案。")
    lines.append("3. ACC 单参考不宜作为默认替代 HF；更合理的方向是 HF 主路径 + ACC/陀螺仪用于运动分割、窗口质量或级联补充。")
    lines.append("4. 跨个体泛化现在只能做探索性判断。TS 样本应继续补齐同运动重复次数，尤其是波比跳、开合跳和弯举。")
    lines.append("5. 多场景共用参数可以先按运动伪影频谱相似性分组，而不是按动作名称硬分组；跳绳/开合跳、波比跳/俯卧撑可能分别形成候选簇。")
    lines.append("")
    lines.append("## 数据与可复现性")
    lines.append("")
    lines.append("- 输入：本目录下 59 个 `schema_version=v2`、`reference_order_key=HF` 的独立优化 JSON。")
    lines.append("- 生成文件：`analysis_outputs/all_result_summary.csv`、`scene_summary.csv`、`subject_summary.csv`、`parameter_summary.csv`、`scene_shared_parameter_replay.csv`。")
    lines.append("- ACC 对比：用每个报告 `best_params` 重放 `reference_groups_order=(\"ACC\",)`。")
    lines.append("- 共享候选重放：每个运动类型 × 滤波器选择参数 medoid 后重放组内样本。")
    (ROOT / "参数泛化性评估报告_20260616.md").write_text("\n".join(lines), encoding="utf-8")


def _overall_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "lms_n": sum(1 for row in rows if row["adaptive_filter"] == "lms"),
        "klms_n": sum(1 for row in rows if row["adaptive_filter"] == "klms"),
        "hf_mae": _mean(rows, "hf_total_aae"),
        "acc_mae": _mean(rows, "acc_total_aae"),
        "fft_mae": _mean(rows, "fft_total_aae"),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    out.append("| " + " | ".join(label for _, label in columns) + " |")
    out.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        cells = [_format_cell(row.get(key), key=key) for key, _ in columns]
        out.append("| " + " | ".join(cells) + " |")
    return out


def _format_cell(value: Any, *, key: str) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        if ("rate" in key or "share" in key or "r5" in key) and 0 <= value <= 1:
            return f"{value * 100:.1f}%"
        return f"{value:.2f}"
    return str(value)


if __name__ == "__main__":
    main()
