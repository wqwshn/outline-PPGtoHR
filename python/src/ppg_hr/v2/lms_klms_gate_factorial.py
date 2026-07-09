"""LMS/KLMS motion-gate factorial experiment runner."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from .algorithm_presets import V2_ALGORITHM_PRESET_LITE
from .batch_pipeline import safe_run_prefix
from .optimizer import V2BayesConfig, optimise_v2
from .output_paths import prepare_output_dir, safe_name, safe_output_path
from .plotting import render_v2_report
from .qc import quality_filter_sample_v2
from .reference_groups import normalise_reference_order, reference_order_key
from .types import V2RunConfig

DEFAULT_DATA_ROOT = Path(
    r"D:\data\PPG_HeartRate\Algorithm\Algorithm\outline-PPGtoHR\data\202607-multiperson\0708-LYX"
)
DEFAULT_SMOKE_SAMPLE = "xiezi2_LYX_0708"
SCENARIO_PREFIXES = {
    "xiezi": "写字",
    "jianpan": "敲键盘",
    "woli": "握力计",
    "quanji": "拳击",
    "kaihe": "kaihe",
    "bobi": "bobi",
    "fuwo": "fuwo",
    "tiaosheng": "tiaosheng",
    "wanju": "wanju",
}
LMS_GATE_ALLOWLIST = ("lms", "noncausal_lms")
KLMS_EXPERIMENT_GATE_ALLOWLIST = ("lms", "noncausal_lms", "klms")


@dataclass(frozen=True)
class GateFactorialCondition:
    name: str
    adaptive_filter: str
    low_reacquire: bool
    high_escape: bool


CONDITIONS: tuple[GateFactorialCondition, ...] = (
    GateFactorialCondition("lms_gate_off", "lms", False, False),
    GateFactorialCondition("lms_low_reacquire_only", "lms", True, False),
    GateFactorialCondition("lms_high_escape_only", "lms", False, True),
    GateFactorialCondition("lms_gate_full", "lms", True, True),
    GateFactorialCondition("klms_gate_off", "klms", False, False),
    GateFactorialCondition("klms_low_reacquire_only", "klms", True, False),
    GateFactorialCondition("klms_high_escape_only", "klms", False, True),
    GateFactorialCondition("klms_gate_full", "klms", True, True),
)
CONDITIONS_BY_NAME = {condition.name: condition for condition in CONDITIONS}


@dataclass(frozen=True)
class GateFactorialSample:
    sample_id: str
    scenario: str
    data_path: Path
    ref_path: Path


@dataclass(frozen=True)
class PlannedGateRun:
    sample: GateFactorialSample
    condition: GateFactorialCondition
    output_dir: Path


@dataclass
class CompletedGateRun:
    sample: GateFactorialSample
    condition: GateFactorialCondition
    report_path: Path
    best_error: float
    figure_png: Path | None
    error_csv: Path | None
    hr_csv: Path | None


@dataclass
class GateFactorialResult:
    output_root: Path
    planned_runs: list[PlannedGateRun]
    completed_runs: list[CompletedGateRun]
    summary_csv: Path | None = None


def discover_samples(data_root: Path | str) -> list[GateFactorialSample]:
    root = Path(data_root)
    samples: list[GateFactorialSample] = []
    for data_path in sorted(root.glob("*.csv")):
        if data_path.name.endswith(("_ref.csv", "_HR_ref.csv")):
            continue
        scenario = scenario_for_sample(data_path.stem)
        if scenario is None:
            continue
        ref_path = data_path.with_name(f"{data_path.stem}_HR_ref.csv")
        if not ref_path.is_file():
            ref_path = data_path.with_name(f"{data_path.stem}_ref.csv")
        if not ref_path.is_file():
            continue
        samples.append(
            GateFactorialSample(
                sample_id=data_path.stem,
                scenario=scenario,
                data_path=data_path,
                ref_path=ref_path,
            )
        )
    return samples


def scenario_for_sample(sample_id: str) -> str | None:
    stem = str(sample_id).strip().lower()
    for prefix in SCENARIO_PREFIXES:
        if stem.startswith(prefix) or stem.startswith(f"multi_{prefix}"):
            return prefix
    return None


def condition_run_config_overrides(condition_name: str) -> dict[str, object]:
    condition = condition_by_name(condition_name)
    allowlist = (
        KLMS_EXPERIMENT_GATE_ALLOWLIST
        if condition.adaptive_filter == "klms"
        else LMS_GATE_ALLOWLIST
    )
    return {
        "adaptive_filter": condition.adaptive_filter,
        "reacquire_enable": condition.low_reacquire,
        "high_lock_escape_enable": condition.high_escape,
        "motion_gate_filter_allowlist": allowlist,
    }


def condition_by_name(condition_name: str) -> GateFactorialCondition:
    try:
        return CONDITIONS_BY_NAME[str(condition_name)]
    except KeyError as exc:
        valid = ", ".join(CONDITIONS_BY_NAME)
        raise ValueError(f"Unknown gate factorial condition: {condition_name}. Valid: {valid}") from exc


def run_gate_factorial_experiment(
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    output_root: Path | str | None = None,
    sample_ids: Sequence[str] = (),
    condition_names: Sequence[str] = (),
    reference_groups_order: Sequence[str] = ("HF",),
    dry_run: bool = False,
    bayes_cfg: V2BayesConfig | None = None,
    render: bool = True,
    resume: bool = False,
    on_log: Callable[[str], None] | None = None,
) -> GateFactorialResult:
    data_root = Path(data_root)
    output_root_path = Path(output_root) if output_root is not None else _default_output_root(data_root)
    reference_order = normalise_reference_order(tuple(reference_groups_order))
    samples = _select_samples(discover_samples(data_root), sample_ids)
    conditions = _select_conditions(condition_names)
    planned = [
        PlannedGateRun(
            sample=sample,
            condition=condition,
            output_dir=output_root_path / condition.name,
        )
        for condition in conditions
        for sample in samples
    ]
    if dry_run:
        return GateFactorialResult(output_root=output_root_path, planned_runs=planned, completed_runs=[])

    prepare_output_dir(output_root_path)
    completed: list[CompletedGateRun] = []
    cfg_bayes = bayes_cfg or V2BayesConfig()
    for run in planned:
        existing = _existing_completed_run(run, reference_order) if resume else None
        if existing is not None:
            _log(on_log, f"跳过已存在结果 {run.condition.name}: {run.sample.sample_id}")
            completed.append(existing)
            continue
        completed.append(
            _run_one(
                run,
                bayes_cfg=cfg_bayes,
                render=render,
                reference_groups_order=reference_order,
                on_log=on_log,
            )
        )
    summary_csv = _write_summary(output_root_path, completed)
    return GateFactorialResult(
        output_root=output_root_path,
        planned_runs=planned,
        completed_runs=completed,
        summary_csv=summary_csv,
    )


def _run_one(
    run: PlannedGateRun,
    *,
    bayes_cfg: V2BayesConfig,
    render: bool,
    reference_groups_order: tuple[str, ...],
    on_log: Callable[[str], None] | None,
) -> CompletedGateRun:
    condition_dir = prepare_output_dir(run.output_dir)
    json_dir = prepare_output_dir(condition_dir / "json")
    png_dir = prepare_output_dir(condition_dir / "png")
    csv_dir = prepare_output_dir(condition_dir / "csv")
    overrides = condition_run_config_overrides(run.condition.name)
    cfg = V2RunConfig(
        data_path=run.sample.data_path,
        ref_path=run.sample.ref_path,
        ppg_mode="green",
        ppg_input_transform="raw_bandpass",
        analysis_scope="full",
        algorithm_preset=V2_ALGORITHM_PRESET_LITE,
        reference_groups_order=reference_groups_order,
        **overrides,
    )
    prefix = safe_run_prefix(
        run.sample.sample_id,
        "green",
        "raw_bandpass",
        run.condition.adaptive_filter,
        "full",
        reference_groups_order,
    )
    report_path = safe_output_path(json_dir, f"{prefix}-v2.json")
    qc = quality_filter_sample_v2(run.sample.data_path, ref_csv=run.sample.ref_path)
    _log(on_log, f"开始 {run.condition.name}: {run.sample.sample_id}")
    result = optimise_v2(
        cfg,
        bayes_cfg,
        out_path=report_path,
        qc=qc.to_dict(),
    )
    figure_png = None
    error_csv = None
    hr_csv = None
    if render:
        artefacts = render_v2_report(
            result.report_path,
            out_dir=png_dir,
            csv_dir=csv_dir,
            output_prefix=prefix,
        )
        figure_png = artefacts.figure_png
        error_csv = artefacts.error_csv
        hr_csv = artefacts.hr_csv
    _log(
        on_log,
        f"完成 {run.condition.name}: {run.sample.sample_id} best={result.best_error:.3f}",
    )
    return CompletedGateRun(
        sample=run.sample,
        condition=run.condition,
        report_path=result.report_path,
        best_error=float(result.best_error),
        figure_png=figure_png,
        error_csv=error_csv,
        hr_csv=hr_csv,
    )


def _existing_completed_run(
    run: PlannedGateRun,
    reference_groups_order: tuple[str, ...],
) -> CompletedGateRun | None:
    report_path = _planned_report_path(run, reference_groups_order)
    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    best_error = _best_error_from_payload(payload)
    prefix = report_path.name[: -len("-v2.json")] if report_path.name.endswith("-v2.json") else report_path.stem
    png = run.output_dir / "png" / f"{prefix}-v2-hr.png"
    err = run.output_dir / "csv" / f"{prefix}-v2-error.csv"
    hr = run.output_dir / "csv" / f"{prefix}-v2-hr.csv"
    return CompletedGateRun(
        sample=run.sample,
        condition=run.condition,
        report_path=report_path,
        best_error=best_error,
        figure_png=png if png.is_file() else None,
        error_csv=err if err.is_file() else None,
        hr_csv=hr if hr.is_file() else None,
    )


def _planned_report_path(
    run: PlannedGateRun,
    reference_groups_order: tuple[str, ...],
) -> Path:
    prefix = safe_run_prefix(
        run.sample.sample_id,
        "green",
        "raw_bandpass",
        run.condition.adaptive_filter,
        "full",
        reference_groups_order,
    )
    return safe_output_path(run.output_dir / "json", f"{prefix}-v2.json")


def _best_error_from_payload(payload: dict[str, object]) -> float:
    err_stats = payload.get("err_stats")
    if isinstance(err_stats, dict):
        value = err_stats.get("final_aae_bpm")
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    history = payload.get("history")
    if isinstance(history, list):
        values = []
        for row in history:
            if isinstance(row, dict):
                try:
                    values.append(float(row.get("value")))
                except (TypeError, ValueError):
                    pass
        if values:
            return min(values)
    return float("nan")


def _select_samples(
    discovered: list[GateFactorialSample],
    sample_ids: Sequence[str],
) -> list[GateFactorialSample]:
    if not sample_ids:
        return discovered
    wanted = {str(sample_id) for sample_id in sample_ids}
    selected = [sample for sample in discovered if sample.sample_id in wanted]
    found = {sample.sample_id for sample in selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"Unknown sample ids: {', '.join(missing)}")
    return selected


def _select_conditions(condition_names: Sequence[str]) -> list[GateFactorialCondition]:
    if not condition_names:
        return list(CONDITIONS)
    return [condition_by_name(name) for name in condition_names]


def _default_output_root(data_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return data_root / "v2_gate_factorial_outputs" / f"{stamp}_lms_klms_gate_factorial"


def _write_summary(output_root: Path, runs: list[CompletedGateRun]) -> Path:
    path = safe_output_path(output_root, "gate_factorial_summary.csv")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "sample",
                "scenario",
                "adaptive_filter",
                "low_reacquire",
                "high_escape",
                "reference_order",
                "best_error",
                "report_path",
                "figure_png",
                "error_csv",
                "hr_csv",
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run.condition.name,
                    run.sample.sample_id,
                    run.sample.scenario,
                    run.condition.adaptive_filter,
                    int(run.condition.low_reacquire),
                    int(run.condition.high_escape),
                    _reference_order_from_report(run.report_path),
                    f"{run.best_error:.6g}",
                    str(run.report_path),
                    str(run.figure_png or ""),
                    str(run.error_csv or ""),
                    str(run.hr_csv or ""),
                ]
            )
    return path


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _reference_order_from_report(report_path: Path) -> str:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    groups = ()
    raw = payload.get("reference_groups_order")
    if isinstance(raw, (list, tuple)):
        groups = tuple(str(item) for item in raw)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata_raw = metadata.get("reference_groups_order")
        if not groups and isinstance(metadata_raw, (list, tuple)):
            groups = tuple(str(item) for item in metadata_raw)
    try:
        return reference_order_key(normalise_reference_order(groups))
    except ValueError:
        return ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--all", action="store_true", help="Run all included motion samples.")
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument(
        "--reference-group",
        action="append",
        default=None,
        help="Reference groups in cascade order; repeat, e.g. --reference-group HF --reference-group ACC.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=75)
    parser.add_argument("--num-seed-points", type=int, default=10)
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    samples = tuple(args.sample)
    if not args.all and not samples:
        samples = (DEFAULT_SMOKE_SAMPLE,)
    bayes_cfg = V2BayesConfig(
        max_iterations=int(args.max_iterations),
        num_seed_points=int(args.num_seed_points),
        num_repeats=int(args.num_repeats),
        random_state=int(args.random_state),
    )
    result = run_gate_factorial_experiment(
        data_root=args.data_root,
        output_root=args.output_root,
        sample_ids=() if args.all else samples,
        condition_names=tuple(args.condition),
        reference_groups_order=tuple(args.reference_group or ("HF",)),
        dry_run=bool(args.dry_run),
        bayes_cfg=bayes_cfg,
        render=not bool(args.skip_render),
        resume=bool(args.resume),
        on_log=print,
    )
    print(f"output_root={result.output_root}")
    for run in result.planned_runs:
        print(
            "plan: {condition} {sample} scenario={scenario}".format(
                condition=run.condition.name,
                sample=run.sample.sample_id,
                scenario=run.sample.scenario,
            )
        )
    if result.summary_csv is not None:
        print(f"summary_csv={result.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
