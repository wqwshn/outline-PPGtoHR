"""Frozen A0/A1/A2 replay for handoff-reset observability experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from .post_motion_dual_reset_experiment import (
    SampleReplay,
    load_hb_manifest,
    load_sample_replay,
)
from .post_motion_dual_reset_runtime import (
    DualResetRuntimeWindow,
    FrozenDualResetConfig,
    apply_frozen_dual_reset,
)


@dataclass(frozen=True)
class ObservabilityCandidate:
    name: str
    mode: str
    periodicity_min: float = 0.5
    peak_competition_min: float = 1.3
    recovery_hits: int = 2
    hits_required: int = 3
    qualification_windows: int = 4


@dataclass(frozen=True)
class ObservabilityReplayResult:
    candidate: ObservabilityCandidate
    window_rows: tuple[dict[str, Any], ...]
    sample_metrics: dict[str, Any]
    independent_reset_bpm: tuple[float, ...]


def build_predeclared_candidates() -> tuple[ObservabilityCandidate, ...]:
    """Return the frozen nested observability configurations for Track A."""

    candidates = [ObservabilityCandidate(name="a0", mode="a0")]
    levels = (
        ("loose", 0.40, 1.10, 2),
        ("central", 0.50, 1.20, 2),
        ("strict", 0.60, 1.30, 3),
    )
    for mode in ("a1", "a2"):
        candidates.extend(
            ObservabilityCandidate(
                name=f"{mode}_{name}",
                mode=mode,
                periodicity_min=periodicity,
                peak_competition_min=competition,
                recovery_hits=hits,
            )
            for name, periodicity, competition, hits in levels
        )
    return tuple(candidates)


def evaluate_candidate_matrix(
    replay: SampleReplay,
    candidates: tuple[ObservabilityCandidate, ...],
) -> dict[str, ObservabilityReplayResult]:
    """Evaluate candidates and fail closed on independent-reset drift."""

    results = {
        candidate.name: evaluate_observability_candidate(replay, candidate)
        for candidate in candidates
    }
    independent = {
        result.independent_reset_bpm for result in results.values()
    }
    if len(independent) != 1:
        raise ValueError(f"{replay.sample}: independent reset drift across A0/A1/A2")
    return results


def evaluate_observability_candidate(
    replay: SampleReplay,
    candidate: ObservabilityCandidate,
) -> ObservabilityReplayResult:
    """Replay one frozen sample without rerunning Bayesian optimisation."""

    if len(replay.evidence) != len(replay.offline):
        raise ValueError(f"{replay.sample}: evidence/offline length mismatch")
    baseline = np.asarray(
        [window.archived_final_bpm for window in replay.offline],
        dtype=float,
    )
    windows = tuple(
        DualResetRuntimeWindow(
            window_idx=index,
            center_s=evidence.center_s,
            start_s=evidence.start_s,
            reliable=evidence.reliable,
            archived_final_bpm=offline.archived_final_bpm,
            archived_final_history=evidence.archived_final_history,
            candidates=evidence.candidates,
            periodicity=evidence.periodicity,
            peak_competition=evidence.peak_competition,
        )
        for index, (evidence, offline) in enumerate(
            zip(replay.evidence, replay.offline, strict=True)
        )
    )
    runtime = apply_frozen_dual_reset(
        windows,
        motion_end_s=replay.motion_end_s,
        baseline_final_bpm=baseline,
        config=FrozenDualResetConfig(
            experiment_mode=candidate.mode,
            observability_periodicity_min=candidate.periodicity_min,
            observability_peak_competition_min=candidate.peak_competition_min,
            observability_recovery_hits=candidate.recovery_hits,
            hits_required=candidate.hits_required,
            qualification_windows=candidate.qualification_windows,
        ),
    )
    rows: list[dict[str, Any]] = []
    for runtime_row, offline in zip(runtime.window_rows, replay.offline, strict=True):
        final_bpm = float(runtime_row["switch_final_bpm"])
        ref_bpm = float(offline.ref_bpm)
        row = {
            **runtime_row,
            "sample": replay.sample,
            "candidate": candidate.name,
            "mode": candidate.mode,
            "ref_bpm": ref_bpm,
            "final_error_bpm": abs(final_bpm - ref_bpm),
            "handoff_error_bpm": abs(
                float(runtime_row["handoff_reset_bpm"]) - ref_bpm
            ),
            "independent_error_bpm": abs(
                float(runtime_row["independent_reset_bpm"]) - ref_bpm
            ),
            "in_post60": float(runtime_row["center_s"])
            <= replay.motion_end_s + 60.0,
        }
        rows.append(row)
    metrics = _summarise(replay, rows)
    return ObservabilityReplayResult(
        candidate=candidate,
        window_rows=tuple(rows),
        sample_metrics=metrics,
        independent_reset_bpm=tuple(
            float(row["independent_reset_bpm"]) for row in rows
        ),
    )


def _summarise(replay: SampleReplay, rows: list[dict[str, Any]]) -> dict[str, Any]:
    post60 = [row for row in rows if bool(row["in_post60"])]
    first_recovery = next(
        (row for row in rows if row["observability_state"] == "recovered"),
        None,
    )
    first_ready = next(
        (row for row in rows if bool(row["switch_target_ready"])),
        None,
    )
    first_consumed = next(
        (
            row
            for row in rows
            if row["switch_state"]
            in {"gap_rescue", "stable_crossover", "handoff_active"}
        ),
        None,
    )
    ready_rows = (
        []
        if first_ready is None
        else [
            row
            for row in post60
            if float(row["center_s"]) >= float(first_ready["center_s"])
        ]
    )
    pre_ready_rows = (
        post60
        if first_ready is None
        else [
            row
            for row in post60
            if float(row["center_s"]) < float(first_ready["center_s"])
        ]
    )
    final_errors = [float(row["final_error_bpm"]) for row in post60]
    handoff_errors = [float(row["handoff_error_bpm"]) for row in ready_rows]
    switch_values = [float(row["switch_final_bpm"]) for row in rows]
    return {
        "sample": replay.sample,
        "post60_window_count": len(post60),
        "post60_final_mae_bpm": _mean_or_nan(final_errors),
        "post60_e10_count": sum(error > 10.0 for error in final_errors),
        "post60_e20_count": sum(error > 20.0 for error in final_errors),
        "first_recovery_delay_s": _delay(first_recovery, replay.motion_end_s),
        "first_ready_delay_s": _delay(first_ready, replay.motion_end_s),
        "first_consumed_delay_s": _delay(first_consumed, replay.motion_end_s),
        "ready_handoff_mae_bpm": _mean_or_nan(handoff_errors),
        "ready_handoff_e20_count": sum(error > 20.0 for error in handoff_errors),
        "ready_pre_e20_count": sum(
            float(row["final_error_bpm"]) > 20.0 for row in pre_ready_rows
        ),
        "reinitialization_count": max(
            (int(row["handoff_reinitialization_count"]) for row in rows),
            default=0,
        ),
        "max_switch_jump_bpm": max(
            (abs(current - previous) for previous, current in zip(
                switch_values,
                switch_values[1:],
                strict=False,
            )),
            default=0.0,
        ),
        "safe_abstain": any(row["switch_state"] == "safe_abstain" for row in rows),
    }


def _delay(row: dict[str, Any] | None, motion_end_s: float) -> float:
    return (
        float("nan")
        if row is None
        else float(row["center_s"]) - float(motion_end_s)
    )


def _mean_or_nan(values: list[float]) -> float:
    return fmean(values) if values else float("nan")


def run_observability_experiment(
    *,
    manifest_path: Path,
    lite_batch_dir: Path,
    output_dir: Path,
    samples: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the frozen HB Track A matrix and write auditable artifacts."""

    manifest = load_hb_manifest(manifest_path)
    selected = tuple(samples or manifest.all_samples)
    unknown = sorted(set(selected) - set(manifest.all_samples))
    if unknown:
        raise ValueError(f"unknown HB samples: {unknown}")
    cohort = _cohort_by_sample(manifest)
    candidates = build_predeclared_candidates()
    matrices: dict[str, dict[str, ObservabilityReplayResult]] = {}
    for sample in selected:
        matrices[sample] = evaluate_candidate_matrix(
            load_sample_replay(sample, lite_batch_dir),
            candidates,
        )

    sample_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for sample in selected:
        baseline = matrices[sample]["a0"].sample_metrics
        for candidate in candidates:
            result = matrices[sample][candidate.name]
            metrics = result.sample_metrics
            consumed_e20 = sum(
                bool(row["in_post60"])
                and row["switch_state"]
                in {"gap_rescue", "stable_crossover", "handoff_active"}
                and float(row["final_error_bpm"]) > 20.0
                for row in result.window_rows
            )
            row = {
                "sample": sample,
                "cohort": cohort[sample],
                "candidate": candidate.name,
                "mode": candidate.mode,
                "periodicity_min": candidate.periodicity_min,
                "peak_competition_min": candidate.peak_competition_min,
                "recovery_hits": candidate.recovery_hits,
                **metrics,
                "post60_mae_delta_vs_a0_bpm": float(
                    metrics["post60_final_mae_bpm"]
                )
                - float(baseline["post60_final_mae_bpm"]),
                "new_e10_vs_a0_count": max(
                    0,
                    int(metrics["post60_e10_count"])
                    - int(baseline["post60_e10_count"]),
                ),
                "new_e20_vs_a0_count": max(
                    0,
                    int(metrics["post60_e20_count"])
                    - int(baseline["post60_e20_count"]),
                ),
                "wrong_switch_e20_count": consumed_e20,
            }
            row.update(_sample_gates(row))
            sample_rows.append(row)
            window_rows.extend(result.window_rows)

    summaries = [
        _candidate_summary(candidate, sample_rows)
        for candidate in candidates
    ]
    passing = [row for row in summaries if bool(row["track_a_pass"])]
    selected_candidate = passing[0]["candidate"] if passing else None
    decision = {
        "verdict": "GO" if selected_candidate is not None else "NO_GO",
        "selected_candidate": selected_candidate,
        "yzy_allowed": selected_candidate is not None,
        "reason": (
            "minimal_candidate_passed_all_HB_gates"
            if selected_candidate is not None
            else "no_A1_A2_candidate_passed_HB_absolute_and_nonregression_gates"
        ),
        "independent_reset_invariant": True,
        "candidate_order": [candidate.name for candidate in candidates],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "track_a_sample_metrics.csv", sample_rows)
    _write_rows(output_dir / "track_a_window_metrics.csv", window_rows)
    _write_rows(output_dir / "track_a_candidate_summary.csv", summaries)
    (output_dir / "track_a_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "track_a_report.md").write_text(
        _markdown_report(decision, summaries, sample_rows, lite_batch_dir),
        encoding="utf-8",
    )
    return {
        "decision": decision,
        "candidate_summary": summaries,
        "sample_metrics": sample_rows,
        "output_dir": str(output_dir),
    }


def _cohort_by_sample(manifest: Any) -> dict[str, str]:
    cohorts: dict[str, str] = {}
    for name in manifest.development_failures:
        cohorts[name] = "D1"
    for name in manifest.development_controls:
        cohorts[name] = "D2"
    for name in manifest.frozen_normal_gate:
        cohorts[name] = "G1"
    for name in manifest.hard_switch_sentinels:
        cohorts[name] = "S1"
    for name in manifest.full_batch_only:
        cohorts[name] = "C1"
    return cohorts


def _sample_gates(row: dict[str, Any]) -> dict[str, bool]:
    final_pass = bool(
        float(row["post60_final_mae_bpm"]) <= 3.0
        and int(row["post60_e20_count"]) == 0
    )
    ready_delay = float(row["first_ready_delay_s"])
    ready_mae = float(row["ready_handoff_mae_bpm"])
    target_pass = bool(
        np.isfinite(ready_delay)
        and ready_delay <= 20.0
        and np.isfinite(ready_mae)
        and ready_mae <= 3.0
        and int(row["ready_handoff_e20_count"]) == 0
    )
    normal_pass = bool(
        float(row["post60_mae_delta_vs_a0_bpm"]) <= 1.0
        and int(row["new_e20_vs_a0_count"]) == 0
        and int(row["wrong_switch_e20_count"]) == 0
    )
    safe_abstain_pass = bool(
        row["safe_abstain"]
        and float(row["post60_mae_delta_vs_a0_bpm"]) <= 1.0
        and int(row["new_e10_vs_a0_count"]) == 0
        and int(row["new_e20_vs_a0_count"]) == 0
    )
    return {
        "final_rescue_pass": final_pass,
        "target_ready_pass": target_pass,
        "normal_nonregression_pass": normal_pass,
        "safe_abstain_pass": safe_abstain_pass,
    }


def _candidate_summary(
    candidate: ObservabilityCandidate,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    scoped = [row for row in rows if row["candidate"] == candidate.name]
    d1 = [row for row in scoped if row["cohort"] == "D1"]
    normal = [row for row in scoped if row["cohort"] != "D1"]
    rescued = [row["sample"] for row in d1 if row["final_rescue_pass"]]
    target_pass = [row["sample"] for row in d1 if row["target_ready_pass"]]
    unresolved = [
        row["sample"]
        for row in d1
        if not row["final_rescue_pass"] and not row["safe_abstain_pass"]
    ]
    normal_failures = [
        row["sample"] for row in normal if not row["normal_nonregression_pass"]
    ]
    track_a_pass = bool(
        candidate.mode != "a0"
        and len(rescued) >= 3
        and len(target_pass) >= 3
        and not unresolved
        and not normal_failures
    )
    return {
        "candidate": candidate.name,
        "mode": candidate.mode,
        "d1_rescued_count": len(rescued),
        "d1_rescued_samples": ",".join(rescued),
        "d1_target_pass_count": len(target_pass),
        "d1_target_pass_samples": ",".join(target_pass),
        "d1_unresolved_samples": ",".join(unresolved),
        "normal_failure_count": len(normal_failures),
        "normal_failure_samples": ",".join(normal_failures),
        "mean_post60_mae_bpm": fmean(
            float(row["post60_final_mae_bpm"]) for row in scoped
        ),
        "track_a_pass": track_a_pass,
    }


def _markdown_report(
    decision: dict[str, Any],
    summaries: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    source_dir: Path,
) -> str:
    lines = [
        "# 交接 reset 可观测性恢复 Track A",
        "",
        f"**结论：`{decision['verdict']}`。**",
        "",
        f"冻结输入：`{source_dir}`。本实验未重新运行 BO，YZY 是否允许开启：`{decision['yzy_allowed']}`。",
        "",
        "## 候选汇总",
        "",
        "| 候选 | D1 Final 通过 | D1 target 通过 | 正常失败 | Track A |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['candidate']} | {row['d1_rescued_count']}/4 | "
            f"{row['d1_target_pass_count']}/4 | {row['normal_failure_count']} | "
            f"{'PASS' if row['track_a_pass'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## D1 逐样本",
        "",
        "| 候选 | 样本 | Final MAE | E20 | 恢复延迟 | ready 延迟 | 首次消费 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in sample_rows:
        if row["cohort"] != "D1":
            continue
        lines.append(
            f"| {row['candidate']} | {row['sample']} | "
            f"{float(row['post60_final_mae_bpm']):.3f} | "
            f"{row['post60_e20_count']} | {_fmt(row['first_recovery_delay_s'])} | "
            f"{_fmt(row['first_ready_delay_s'])} | {_fmt(row['first_consumed_delay_s'])} |"
        )
    lines.extend([
        "",
        "独立 reset 在 A0/A1/A2 间逐窗完全一致；任何平均改善均未用于覆盖单样本硬门槛。",
        "",
    ])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = float(value)
    return "—" if not np.isfinite(number) else f"{number:.1f}"


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lite-batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", nargs="*")
    args = parser.parse_args(argv)
    payload = run_observability_experiment(
        manifest_path=args.manifest,
        lite_batch_dir=args.lite_batch_dir,
        output_dir=args.output_dir,
        samples=args.samples,
    )
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))
    return 0 if payload["decision"]["verdict"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
