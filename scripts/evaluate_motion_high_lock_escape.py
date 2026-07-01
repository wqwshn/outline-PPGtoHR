from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from ppg_hr.v2.high_lock_replay import RESCUE_ATTENTION_STEMS
from ppg_hr.v2.solver import solve_v2
from ppg_hr.v2.types import V2RunConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--all", action="store_true", help="evaluate every JSON report in the batch")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for report_path in _report_paths(args.batch_dir):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        sample = _sample_stem(report_path, payload)
        for enabled in (False, True):
            cfg = _config_from_payload(
                payload,
                payload.get("best_params", {}) or {},
                tuple(payload.get("reference_groups_order", ("HF",))),
                high_lock_enabled=enabled,
            )
            result = solve_v2(cfg)
            rows.append(_metrics_row(sample, enabled, result.HR, result.metadata, result.window_table))

    csv_path = args.output_dir / "solver_motion_high_lock_escape_eval.csv"
    md_path = args.output_dir / "solver_motion_high_lock_escape_eval.md"
    aggregate_csv = args.output_dir / "solver_motion_high_lock_escape_aggregate.csv"
    _write_csv(csv_path, rows)
    aggregate = _aggregate_rows(rows)
    _write_csv(aggregate_csv, aggregate)
    _write_markdown(md_path, aggregate)
    print(f"csv={csv_path}")
    print(f"aggregate_csv={aggregate_csv}")
    print(f"summary={md_path}")
    return 0


def _report_paths(batch_dir: Path) -> list[Path]:
    root = batch_dir / "json" if (batch_dir / "json").is_dir() else batch_dir
    return sorted(root.glob("*.json"))


def _config_from_payload(
    payload: dict[str, Any],
    params: dict[str, Any],
    reference_groups_order: tuple[str, ...],
    *,
    high_lock_enabled: bool,
) -> V2RunConfig:
    field_names = {f.name for f in dataclasses.fields(V2RunConfig)}
    cfg_dict: dict[str, Any] = {
        "data_path": Path(str(payload.get("data_path", ""))),
        "ref_path": Path(str(payload.get("ref_path", ""))),
        "adaptive_filter": payload.get("adaptive_filter"),
        "reference_groups_order": reference_groups_order,
        "high_lock_escape_enable": bool(high_lock_enabled),
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
    cfg_dict["high_lock_escape_enable"] = bool(high_lock_enabled)
    return V2RunConfig(**{k: v for k, v in cfg_dict.items() if k in field_names})


def _metrics_row(
    sample: str,
    enabled: bool,
    hr: np.ndarray,
    metadata: dict[str, Any],
    window_table: list[dict[str, Any]],
) -> dict[str, Any]:
    mode = "high_lock_escape" if enabled else "legacy"
    metrics = _metrics_from_hr(np.asarray(hr, dtype=float))
    post_errors = [
        abs(float(row["final_hr_bpm"]) - float(row["ref_hr_bpm"]))
        for row in window_table
        if _is_post_motion(row) and _finite(row.get("final_hr_bpm")) and _finite(row.get("ref_hr_bpm"))
    ]
    trigger_count = int(metadata.get("high_lock_escape", {}).get("trigger_count", 0))
    return {
        "sample": sample,
        "cohort": "rescue_candidates" if sample in RESCUE_ATTENTION_STEMS else "non_regression_candidates",
        "mode": mode,
        "total_aae_bpm": metrics["total_aae"],
        "rest_aae_bpm": metrics["rest_aae"],
        "motion_aae_bpm": metrics["motion_aae"],
        "motion_hit_rate_5bpm": metrics["motion_hit_rate_5bpm"],
        "post_motion_aae_bpm": _mean(post_errors),
        "high_lock_trigger_count": trigger_count,
    }


def _metrics_from_hr(hr: np.ndarray) -> dict[str, float]:
    if hr.size == 0 or hr.ndim != 2:
        return {
            "total_aae": float("nan"),
            "rest_aae": float("nan"),
            "motion_aae": float("nan"),
            "motion_hit_rate_5bpm": float("nan"),
        }
    ref = hr[:, 1]
    pred = hr[:, 3]
    motion = hr[:, 4] > 0.5 if hr.shape[1] > 4 else np.zeros_like(ref, dtype=bool)
    valid = np.isfinite(ref) & np.isfinite(pred)
    return {
        "total_aae": _aae(ref, pred, valid),
        "rest_aae": _aae(ref, pred, valid & ~motion),
        "motion_aae": _aae(ref, pred, valid & motion),
        "motion_hit_rate_5bpm": _hit(ref, pred, valid & motion),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample = {(row["sample"], row["mode"]): row for row in rows}
    deltas: list[dict[str, Any]] = []
    for row in rows:
        if row["mode"] != "high_lock_escape":
            continue
        legacy = by_sample.get((row["sample"], "legacy"))
        if legacy is None:
            continue
        deltas.append(
            {
                "sample": row["sample"],
                "cohort": row["cohort"],
                "motion_delta_aae_bpm": row["motion_aae_bpm"] - legacy["motion_aae_bpm"],
                "post_motion_delta_aae_bpm": row["post_motion_aae_bpm"] - legacy["post_motion_aae_bpm"],
                "high_lock_trigger_count": row["high_lock_trigger_count"],
                "legacy_motion_aae_bpm": legacy["motion_aae_bpm"],
                "motion_aae_bpm": row["motion_aae_bpm"],
                "legacy_post_motion_aae_bpm": legacy["post_motion_aae_bpm"],
                "post_motion_aae_bpm": row["post_motion_aae_bpm"],
            }
        )
    out: list[dict[str, Any]] = []
    for cohort in sorted(set(row["cohort"] for row in deltas) | {"full_batch"}):
        cohort_rows = deltas if cohort == "full_batch" else [row for row in deltas if row["cohort"] == cohort]
        if not cohort_rows:
            continue
        out.append(
            {
                "cohort": cohort,
                "sample_count": len(cohort_rows),
                "legacy_motion_aae_bpm": _mean([row["legacy_motion_aae_bpm"] for row in cohort_rows]),
                "motion_aae_bpm": _mean([row["motion_aae_bpm"] for row in cohort_rows]),
                "motion_delta_aae_bpm": _mean([row["motion_delta_aae_bpm"] for row in cohort_rows]),
                "legacy_post_motion_aae_bpm": _mean([row["legacy_post_motion_aae_bpm"] for row in cohort_rows]),
                "post_motion_aae_bpm": _mean([row["post_motion_aae_bpm"] for row in cohort_rows]),
                "post_motion_delta_aae_bpm": _mean([row["post_motion_delta_aae_bpm"] for row in cohort_rows]),
                "high_lock_trigger_count": sum(int(row["high_lock_trigger_count"]) for row in cohort_rows),
            }
        )
    out.extend(deltas)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    summary_rows = [row for row in rows if "sample" not in row]
    lines = [
        "# Solver Motion High-Lock Escape Evaluation",
        "",
        "| Cohort | N | Legacy motion AAE | Escape motion AAE | Delta | Legacy post-motion AAE | Escape post-motion AAE | Post delta | Triggers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {cohort} | {sample_count} | {legacy_motion_aae_bpm:.3f} | {motion_aae_bpm:.3f} | "
            "{motion_delta_aae_bpm:.3f} | {legacy_post_motion_aae_bpm:.3f} | "
            "{post_motion_aae_bpm:.3f} | {post_motion_delta_aae_bpm:.3f} | "
            "{high_lock_trigger_count} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sample_stem(report_path: Path, report: dict[str, Any]) -> str:
    data_path = str(report.get("data_path") or "")
    if data_path:
        return Path(data_path).stem
    marker = "-green-"
    return report_path.stem.split(marker, 1)[0] if marker in report_path.stem else report_path.stem


def _is_post_motion(row: dict[str, Any]) -> bool:
    stage = str(row.get("window_stage") or "")
    return stage.startswith("post_motion")


def _finite(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return parsed == parsed


def _aae(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.abs(pred[mask] - ref[mask]))) if np.any(mask) else float("nan")


def _hit(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - ref[mask]) <= 5.0))


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
