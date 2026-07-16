"""Audited HB Lite 1x40 batch runner for the frozen dual-reset mechanism."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .algorithm_presets import v2_search_space_for_preset
from .batch_pipeline import V2BatchRecord, run_v2_batch_pipeline
from .optimizer import V2BayesConfig
from .post_motion_dual_reset_runtime import FrozenDualResetConfig
from .post_motion_minimal_pipeline_gate import (
    frozen_minimal_run_overrides,
    require_fixed_validation_go,
)
from .report import load_v2_report

HB_LITE_BAYES_CONFIG = V2BayesConfig(
    max_iterations=40,
    num_seed_points=10,
    num_repeats=1,
    random_state=42,
)
HB24_SAMPLE_STEMS = (
    "bobi1", "bobi2", "bobi3",
    "jianpan1", "jianpan2", "jianpan3",
    "kaihe1", "kaihe2", "kaihe3",
    "quanji1", "quanji2", "quanji3",
    "run1", "run2", "run3",
    "tiaosheng1", "tiaosheng2", "tiaosheng3",
    "woli1", "woli2", "woli3",
    "xiezi1", "xiezi2", "xiezi3",
)


def run_audited_hb_lite_batch(
    *,
    input_dir: Path,
    output_dir: Path,
    sample_stems: tuple[str, ...],
    fixed_validation_decision_path: Path,
    bayes_cfg: V2BayesConfig = HB_LITE_BAYES_CONFIG,
) -> dict[str, Any]:
    if not sample_stems:
        raise ValueError("sample_stems must not be empty")
    normalised = tuple(str(sample).strip().lower() for sample in sample_stems)
    if len(set(normalised)) != len(normalised):
        raise ValueError("sample_stems contains duplicate samples")
    if bayes_cfg != HB_LITE_BAYES_CONFIG:
        raise ValueError(
            "HB Lite N5 requires exactly 1x40, 10 seed points, random_state=42"
        )
    if set(normalised) != set(HB24_SAMPLE_STEMS):
        missing = sorted(set(HB24_SAMPLE_STEMS) - set(normalised))
        extra = sorted(set(normalised) - set(HB24_SAMPLE_STEMS))
        raise ValueError(
            f"HB Lite batch requires exact HB24 manifest; missing={missing}, extra={extra}"
        )
    fixed_decision = require_fixed_validation_go(fixed_validation_decision_path)
    mechanism_overrides = frozen_minimal_run_overrides(fixed_decision)
    result = run_v2_batch_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        ppg_modes=["green"],
        ppg_input_transform="raw_bandpass",
        adaptive_filter="lms",
        analysis_scope="full",
        reference_groups_order=("HF",),
        comparison_groups=(("ACC",),),
        bayes_cfg=bayes_cfg,
        algorithm_preset="lite",
        sample_stems=sample_stems,
        run_config_overrides=mechanism_overrides,
    )
    audit = audit_hb_lite_batch(
        records=result["records"],
        requested_samples=sample_stems,
        bayes_cfg=bayes_cfg,
        output_dir=Path(result["output_dir"]),
        mechanism_overrides=mechanism_overrides,
    )
    audit_path = Path(result["output_dir"]) / "batch_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if audit["status"] != "pass":
        raise RuntimeError(f"HB Lite batch audit failed: {audit['failures']}")
    return {**result, "audit": audit, "audit_path": audit_path}


def audit_hb_lite_batch(
    *,
    records: list[V2BatchRecord],
    requested_samples: tuple[str, ...],
    bayes_cfg: V2BayesConfig,
    output_dir: Path,
    mechanism_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if bayes_cfg != HB_LITE_BAYES_CONFIG:
        failures.append("non_frozen_bayes_config")
    requested_list = [name.lower() for name in requested_samples]
    actual_list = [record.sample.lower().split("_hb_")[0] for record in records]
    requested = set(requested_list)
    actual = set(actual_list)
    if Counter(actual_list) != Counter(requested_list):
        failures.append(
            "sample_multiset_mismatch: "
            f"requested={sorted(requested_list)}, actual={sorted(actual_list)}"
        )
    expected_trials = max(1, int(bayes_cfg.num_repeats)) * max(1, int(bayes_cfg.max_iterations))
    sample_audits: list[dict[str, Any]] = []
    for record in records:
        sample_audits.append(
            _audit_record(record, expected_trials=expected_trials, failures=failures)
        )
    summary = Path(output_dir) / "csv" / "v2_batch_summary.csv"
    if not summary.is_file():
        failures.append("missing_batch_summary")
    else:
        _audit_summary_samples(
            summary_path=summary,
            requested_samples=requested_list,
            failures=failures,
        )
    _audit_artifact_sets(
        output_dir=Path(output_dir),
        requested_samples=requested_list,
        failures=failures,
    )
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "protocol": {
            "algorithm_preset": "lite",
            "ppg_mode": "green",
            "ppg_input_transform": "raw_bandpass",
            "adaptive_filter": "lms",
            "analysis_scope": "full",
            "reference_groups_order": ["HF"],
            "comparison_groups": [["ACC"]],
            "bayes": asdict(bayes_cfg),
            "search_space": asdict(v2_search_space_for_preset("lms", "lite")),
            "frozen_dual_reset": asdict(FrozenDualResetConfig()),
            "minimal_handoff_overrides": mechanism_overrides,
        },
        "code": _code_provenance(),
        "requested_samples": sorted(requested),
        "actual_samples": sorted(actual),
        "samples": sample_audits,
        "summary_csv": _file_descriptor(summary),
    }


def _audit_record(
    record: V2BatchRecord,
    *,
    expected_trials: int,
    failures: list[str],
) -> dict[str, Any]:
    prefix = record.sample.lower().split("_hb_")[0]
    required = {
        "report": record.report_path,
        "figure": record.figure_png,
        "error_csv": record.error_csv,
        "hr_csv": record.hr_csv,
        "window_trace_csv": record.window_trace_csv,
        "history_csv": record.history_csv,
    }
    for label, path in required.items():
        if path is None or not Path(path).is_file():
            failures.append(f"{prefix}:missing_{label}")
    acc_comparison = _audit_acc_comparison(record, prefix=prefix, failures=failures)
    payload: dict[str, Any] = {}
    if record.report_path.is_file():
        try:
            payload = load_v2_report(record.report_path)
        except Exception as exc:
            failures.append(f"{prefix}:invalid_report:{exc}")
    history = payload.get("history", []) if payload else []
    if len(history) != expected_trials:
        failures.append(f"{prefix}:trial_count={len(history)} expected={expected_trials}")
    if not math.isfinite(float(record.best_error)):
        failures.append(f"{prefix}:nonfinite_best_error")
    for index, trial in enumerate(history):
        try:
            value = float(trial["value"])
        except (KeyError, TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value):
            failures.append(f"{prefix}:nonfinite_trial_{index}")
            break
    dual = payload.get("post_motion_dual_reset", {}) if payload else {}
    if not bool(dual.get("enabled")):
        failures.append(f"{prefix}:dual_reset_not_enabled")
    trace_rows = payload.get("window_table", []) if payload else []
    post_rows = [row for row in trace_rows if "independent_reset_bpm" in row]
    if payload.get("motion_segment") is not None and not post_rows:
        failures.append(f"{prefix}:missing_dual_reset_window_trace")
    trace_fields = {
        "independent_reset_bpm",
        "handoff_reset_bpm",
        "candidate_qualified",
        "switch_target_ready",
        "switch_state",
        "switch_reason_detail",
    }
    if post_rows and not trace_fields.issubset(post_rows[0]):
        failures.append(f"{prefix}:incomplete_dual_reset_window_trace")
    if str((payload.get("qc") or {}).get("status", "")).lower() != "good":
        failures.append(f"{prefix}:qc_not_good")
    for key, value in (payload.get("err_stats") or {}).items():
        if isinstance(value, int | float) and not math.isfinite(float(value)):
            failures.append(f"{prefix}:nonfinite_err_stat:{key}")
    for row_index, row in enumerate(payload.get("hr", [])):
        if any(not math.isfinite(float(value)) for value in row):
            failures.append(f"{prefix}:nonfinite_hr_row:{row_index}")
            break
    inputs = {
        "data": _file_descriptor(Path(str(payload.get("data_path", "")))),
        "reference": _file_descriptor(Path(str(payload.get("ref_path", "")))),
    }
    return {
        "sample": prefix,
        "best_error": float(record.best_error),
        "trial_count": len(history),
        "best_params": payload.get("best_params", {}),
        "inputs": inputs,
        "artifacts": {
            label: _file_descriptor(Path(path)) if path is not None else None
            for label, path in required.items()
        },
        "dual_reset": dual,
        "acc_comparison": acc_comparison,
    }


def _audit_acc_comparison(
    record: V2BatchRecord,
    *,
    prefix: str,
    failures: list[str],
) -> dict[str, Any]:
    label = "LMS+A"
    column = f"{label}_bpm"
    timeline_count = 0
    metric_count = 0
    if record.hr_csv is not None and Path(record.hr_csv).is_file():
        with Path(record.hr_csv).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if column not in (rows[0].keys() if rows else ()):
            failures.append(f"{prefix}:missing_acc_timeline_column")
        else:
            for row in rows:
                try:
                    if math.isfinite(float(row[column])):
                        timeline_count += 1
                except (KeyError, TypeError, ValueError):
                    continue
            if timeline_count == 0:
                failures.append(f"{prefix}:empty_acc_timeline")
    if record.error_csv is not None and Path(record.error_csv).is_file():
        with Path(record.error_csv).open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        acc_rows = [row for row in rows if row.get("method") == label]
        for row in acc_rows:
            try:
                if math.isfinite(float(row["total_aae"])):
                    metric_count += 1
            except (KeyError, TypeError, ValueError):
                continue
        if metric_count == 0:
            failures.append(f"{prefix}:missing_acc_metrics")
    return {
        "label": label,
        "timeline_finite_windows": timeline_count,
        "metric_rows": metric_count,
    }


def _file_descriptor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None, "bytes": None}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "exists": True,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def _audit_artifact_sets(
    *,
    output_dir: Path,
    requested_samples: list[str],
    failures: list[str],
) -> None:
    patterns = {
        "json": (output_dir / "json", "*-v2.json"),
        "figure": (output_dir / "png", "*-v2-hr.png"),
        "hr_csv": (output_dir / "csv", "*-v2-hr.csv"),
        "error_csv": (output_dir / "csv", "*-v2-error.csv"),
        "window_trace_csv": (output_dir / "csv", "*-v2-window-trace.csv"),
        "history_csv": (output_dir / "csv", "*-v2-history.csv"),
    }
    expected = Counter(requested_samples)
    for label, (directory, pattern) in patterns.items():
        paths = sorted(directory.glob(pattern))
        samples = [path.name.lower().split("_hb_")[0] for path in paths]
        if Counter(samples) != expected:
            failures.append(
                f"{label}_artifact_multiset_mismatch: "
                f"expected={sorted(requested_samples)}, actual={sorted(samples)}"
            )


def _audit_summary_samples(
    *,
    summary_path: Path,
    requested_samples: list[str],
    failures: list[str],
) -> None:
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    samples = [str(row.get("sample", "")).lower().split("_hb_")[0] for row in rows]
    if Counter(samples) != Counter(requested_samples):
        failures.append(
            "summary_sample_multiset_mismatch: "
            f"expected={sorted(requested_samples)}, actual={sorted(samples)}"
        )


def _code_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
        diff = git("diff", "--binary", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "diff_sha256": None, "error": str(exc)}
    return {
        "commit": commit,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument(
        "--fixed-validation-decision",
        type=Path,
        required=True,
        help="Machine decision that must explicitly allow this BO batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_audited_hb_lite_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_stems=tuple(args.samples),
        fixed_validation_decision_path=args.fixed_validation_decision,
        bayes_cfg=HB_LITE_BAYES_CONFIG,
    )
    print(result["audit_path"])


if __name__ == "__main__":
    main()
