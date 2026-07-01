from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ppg_hr.v2.solver import solve_v2
from ppg_hr.v2.types import V2RunConfig


DEFAULT_SAMPLES = [
    "multi_fuwo1_0613",
    "multi_fuwo2_0613",
    "multi_tiaosheng1_0613",
    "multi_tiaosheng1_0617",
    "multi_wanju1_0613",
    "multi_wanju1_0617",
    "multi_kaihe2_0519",
    "multi_bobi2_0613",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--samples", nargs="*", default=DEFAULT_SAMPLES)
    parser.add_argument("--all", action="store_true", help="evaluate every JSON report in the batch")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    samples = _all_samples(args.batch_dir) if args.all else args.samples
    for stem in samples:
        report_path = _find_report(args.batch_dir, stem)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        for enabled in (False, True):
            cfg = V2RunConfig(
                data_path=Path(payload["data_path"]),
                ref_path=Path(payload["ref_path"]),
                ppg_mode=str(payload.get("ppg_mode", "green")),
                ppg_input_transform=str(payload.get("ppg_input_transform", "raw_bandpass")),
                analysis_scope="full",
                adaptive_filter=str(payload.get("adaptive_filter", "lms")),
                algorithm_preset=str(payload.get("algorithm_preset", "trace_rescue")),
                reference_groups_order=tuple(payload.get("reference_groups_order", ("HF",))),
                post_motion_reacquire_enable=enabled,
            )
            result = solve_v2(cfg)
            rows.append(_metrics_row(stem, enabled, result.metadata, result.window_table))

    csv_path = args.output_dir / "solver_post_motion_reacquire_eval.csv"
    _write_csv(csv_path, rows)
    md_path = args.output_dir / "solver_post_motion_reacquire_eval.md"
    _write_markdown(md_path, rows)
    print(f"csv={csv_path}")
    print(f"summary={md_path}")
    return 0


def _find_report(batch_dir: Path, stem: str) -> Path:
    matches = sorted((batch_dir / "json").glob(f"{stem}-*.json"))
    if not matches:
        raise FileNotFoundError(stem)
    return matches[0]


def _all_samples(batch_dir: Path) -> list[str]:
    samples = []
    for path in sorted((batch_dir / "json").glob("*.json")):
        marker = "-green-"
        samples.append(path.stem.split(marker, 1)[0] if marker in path.stem else path.stem)
    return samples


def _metrics_row(
    sample: str,
    enabled: bool,
    metadata: dict[str, Any],
    window_table: list[dict[str, Any]],
) -> dict[str, Any]:
    motion = metadata.get("motion_segment") or {}
    guard = float(metadata.get("post_motion_reacquire", {}).get("guard_seconds", 20.0))
    start = float(motion.get("end_s", 0.0)) + guard
    rows = [
        row
        for row in window_table
        if float(row.get("center_s", 0.0)) > start
        and _finite(row.get("final_hr_bpm"))
        and _finite(row.get("ref_hr_bpm"))
    ]
    errors = [abs(float(row["final_hr_bpm"]) - float(row["ref_hr_bpm"])) for row in rows]
    stage_counts: dict[str, int] = {}
    for row in window_table:
        stage = str(row.get("window_stage", row.get("window_kind", "")))
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    switch_idx = metadata.get("post_motion_reacquire", {}).get("switch_idx")
    return {
        "sample": sample,
        "mode": "reacquire" if enabled else "legacy",
        "post_motion_rest_window_count": len(rows),
        "post_motion_rest_aae_bpm": _mean(errors),
        "post_motion_rest_hit_rate_5bpm": _hit_rate(errors),
        "switch_idx": "" if switch_idx is None else switch_idx,
        "post_motion_guard_windows": stage_counts.get("post_motion_guard", 0),
        "post_motion_reacquire_windows": stage_counts.get("post_motion_reacquire", 0),
        "used_adaptive_windows": metadata.get("used_adaptive_windows", 0),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Solver Post-Motion Reacquire Evaluation",
        "",
        "| Sample | Mode | N | AAE | Hit <=5 BPM | Switch | Guard | Reacquire |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {sample} | {mode} | {post_motion_rest_window_count} | "
            "{post_motion_rest_aae_bpm:.3f} | {post_motion_rest_hit_rate_5bpm:.3f} | "
            "{switch_idx} | {post_motion_guard_windows} | {post_motion_reacquire_windows} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finite(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return parsed == parsed


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _hit_rate(errors: list[float]) -> float:
    return sum(1 for value in errors if value <= 5.0) / len(errors) if errors else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
