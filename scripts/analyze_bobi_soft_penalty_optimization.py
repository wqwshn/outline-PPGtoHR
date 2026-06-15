"""Re-optimise bobi1/bobi2 with continuity-protected soft spectrum penalty."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from ppg_hr.v2.optimizer import V2BayesConfig, optimise_v2
from ppg_hr.v2.plotting import render_v2_report
from ppg_hr.v2.report import load_v2_report
from ppg_hr.v2.solver import solve_v2
from ppg_hr.v2.types import V2RunConfig


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures" / "bobi_soft_penalty_optimization_20260615"
WIDTH_SCAN_VALUES = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35)


def main() -> None:
    data_dir = _find_bobi_bug_dir()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered_dir = OUTPUT_DIR / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "source_dir": str(data_dir),
        "search_space_changed": False,
        "width_scan_values": list(WIDTH_SCAN_VALUES),
        "samples": {},
    }
    for sample in ("multi_bobi1", "multi_bobi2"):
        print(f"[bobi-soft-penalty] start {sample}", flush=True)
        old_report = data_dir / f"{sample}-green-raw_bandpass-lms-full-HF-v2.json"
        old_payload = load_v2_report(old_report)
        base_cfg = _base_config_from_payload(old_payload, data_dir, sample)
        out_report = OUTPUT_DIR / f"{sample}-green-raw_bandpass-lms-full-HF-v2-soft.json"

        result = optimise_v2(
            base_cfg,
            V2BayesConfig(),
            out_path=out_report,
            on_trial_step=lambda row, sample=sample: _print_trial(sample, row),
        )
        new_payload = load_v2_report(result.report_path)
        render_v2_report(
            result.report_path,
            out_dir=rendered_dir / sample,
        )

        best_cfg = V2RunConfig(
            **{
                **base_cfg.__dict__,
                **result.best_params,
            }
        )
        width_scan = _scan_penalty_width(best_cfg)

        summary["samples"][sample] = {
            "baseline_report": str(old_report),
            "new_report": str(result.report_path),
            "old_best_params": old_payload.get("best_params", {}),
            "new_best_params": result.best_params,
            "old_err_stats": old_payload.get("err_stats", {}),
            "new_err_stats": new_payload.get("err_stats", {}),
            "old_metrics": _metrics_from_payload(old_payload),
            "new_metrics": _metrics_from_payload(new_payload),
            "width_scan": width_scan,
        }
        print(f"[bobi-soft-penalty] done {sample}", flush=True)

    comparison_path = OUTPUT_DIR / "bobi_soft_penalty_comparison.json"
    comparison_path.write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"[bobi-soft-penalty] wrote {comparison_path}", flush=True)


def _find_bobi_bug_dir() -> Path:
    bug_root = ROOT / "bug"
    for candidate in bug_root.iterdir():
        if not candidate.is_dir():
            continue
        names = {p.name for p in candidate.iterdir()}
        if {"multi_bobi1.csv", "multi_bobi2.csv"}.issubset(names):
            return candidate
    raise FileNotFoundError("Could not find bobi bug data directory")


def _base_config_from_payload(
    payload: dict[str, Any],
    data_dir: Path,
    sample: str,
) -> V2RunConfig:
    field_names = {f.name for f in fields(V2RunConfig)}
    cfg: dict[str, Any] = {
        name: payload[name]
        for name in field_names
        if name in payload
    }
    cfg["data_path"] = data_dir / f"{sample}.csv"
    cfg["ref_path"] = data_dir / f"{sample}_HR_ref.csv"
    cfg["reference_groups_order"] = tuple(payload.get("reference_groups_order", ("HF",)))
    # The optimiser owns the parameters in best_params; the base config only
    # carries protocol fields and fixed non-search knobs.
    for name in payload.get("best_params", {}):
        cfg.pop(name, None)
    return V2RunConfig(**cfg)


def _print_trial(sample: str, row: dict[str, Any]) -> None:
    global_trial = int(row["global_trial"])
    global_total = int(row["global_total"])
    if global_trial == 1 or global_trial == global_total or global_trial % 25 == 0:
        print(
            f"[{sample}] trial {global_trial}/{global_total} "
            f"value={float(row['value']):.4f} "
            f"best={float(row['best_overall']):.4f}",
            flush=True,
        )


def _scan_penalty_width(base_cfg: V2RunConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for width in WIDTH_SCAN_VALUES:
        cfg = V2RunConfig(**{**base_cfg.__dict__, "spec_penalty_width": float(width)})
        result = solve_v2(cfg)
        payload = {
            "err_stats": result.err_stats,
            "window_table": result.window_table,
        }
        rows.append(
            {
                "spec_penalty_width": float(width),
                "err_stats": result.err_stats,
                "metrics": _metrics_from_payload(payload),
            }
        )
    return rows


def _metrics_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("window_table", [])
    motion = [
        row
        for row in rows
        if row.get("window_kind") == "motion"
        and np.isfinite(_float(row.get("final_hr_bpm")))
        and np.isfinite(_float(row.get("ref_hr_bpm")))
    ]
    all_errors = np.asarray(
        [
            _float(row.get("final_hr_bpm")) - _float(row.get("ref_hr_bpm"))
            for row in motion
        ],
        dtype=float,
    )
    covered_errors = np.asarray(
        [
            err
            for row, err in zip(motion, all_errors, strict=True)
            if _ref_inside_penalty(row)
        ],
        dtype=float,
    )
    protected_overlap_errors = np.asarray(
        [
            err
            for row, err in zip(motion, all_errors, strict=True)
            if bool(row.get("spectrum_tracking", {}).get("protected_penalty_overlap", False))
        ],
        dtype=float,
    )
    return {
        "motion_window_count": int(len(motion)),
        "motion_mae_bpm": _mae(all_errors),
        "motion_p80_abs_error_bpm": _quantile_abs(all_errors, 0.80),
        "motion_p95_abs_error_bpm": _quantile_abs(all_errors, 0.95),
        "ref_in_penalty_count": int(covered_errors.size),
        "ref_in_penalty_mae_bpm": _mae(covered_errors),
        "protected_overlap_count": int(protected_overlap_errors.size),
        "protected_overlap_mae_bpm": _mae(protected_overlap_errors),
    }


def _ref_inside_penalty(row: dict[str, Any]) -> bool:
    ref = _float(row.get("ref_hr_bpm"))
    tracking = row.get("spectrum_tracking", {})
    centers = tracking.get("penalty_centers_bpm", ())
    half_width = _float(tracking.get("penalty_half_width_bpm"))
    if not np.isfinite(ref) or not np.isfinite(half_width):
        return False
    return any(abs(ref - _float(center)) <= half_width for center in centers)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mae(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(np.abs(arr))) if arr.size else float("nan")


def _quantile_abs(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = np.abs(arr[np.isfinite(arr)])
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
