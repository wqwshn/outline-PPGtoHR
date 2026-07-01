from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

from ppg_hr.v2.output_paths import prepare_output_dir, safe_output_path
from ppg_hr.v2.plotting import render_v2_report
from ppg_hr.v2.reference_groups import reference_order_key
from ppg_hr.v2.report import save_v2_report
from ppg_hr.v2.solver import solve_v2
from ppg_hr.v2.types import V2RunConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_batch_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--comparison",
        action="append",
        default=["ACC"],
        help="comparison reference group, can be repeated; default: ACC",
    )
    args = parser.parse_args()

    output_dir = prepare_output_dir(args.output_dir)
    json_dir = prepare_output_dir(output_dir / "json")
    png_dir = prepare_output_dir(output_dir / "png")
    csv_dir = prepare_output_dir(output_dir / "csv")
    comparison_groups = tuple((str(group),) for group in args.comparison)
    records: list[dict[str, Any]] = []

    for src_report in sorted((args.source_batch_dir / "json").glob("*.json")):
        payload = json.loads(src_report.read_text(encoding="utf-8"))
        cfg = _config_from_payload(
            payload,
            payload.get("best_params", {}) or {},
            tuple(payload.get("reference_groups_order", ("HF",))),
        )
        result = solve_v2(cfg)
        prefix = src_report.stem.removesuffix("-v2")
        report_path = save_v2_report(
            safe_output_path(json_dir, f"{prefix}-v2.json"),
            result,
            best_params=payload.get("best_params", {}) or {},
            history=payload.get("history", []) or [],
            qc=payload.get("qc", {}) or {},
            artefacts=payload.get("artefacts", {}) or {},
        )
        arte = render_v2_report(
            report_path,
            out_dir=png_dir,
            csv_dir=csv_dir,
            output_prefix=prefix,
            comparison_groups=comparison_groups,
        )
        records.append(
            {
                "sample": Path(str(payload.get("data_path", src_report.stem))).name,
                "ppg_mode": payload.get("ppg_mode", ""),
                "ppg_input_transform": payload.get("ppg_input_transform", ""),
                "adaptive_filter": payload.get("adaptive_filter", ""),
                "analysis_scope": payload.get("analysis_scope", ""),
                "reference_order_key": reference_order_key(tuple(payload.get("reference_groups_order", ()))),
                "qc_status": (payload.get("qc") or {}).get("status", ""),
                "status": "ok",
                "best_error": payload.get("err_stats", {}).get("final_aae_bpm", ""),
                "report_path": str(report_path),
                "figure_png": str(arte.figure_png),
                "error_csv": str(arte.error_csv),
                "hr_csv": str(arte.hr_csv),
                "high_lock_trigger_count": result.metadata.get("high_lock_escape", {}).get("trigger_count", 0),
                "error": "",
            }
        )

    summary = _write_summary(csv_dir / "v2_batch_summary.csv", records)
    print(f"output_dir={output_dir}")
    print(f"summary_csv={summary}")
    return 0


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
    cfg_dict["high_lock_escape_enable"] = True
    return V2RunConfig(**{k: v for k, v in cfg_dict.items() if k in field_names})


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
