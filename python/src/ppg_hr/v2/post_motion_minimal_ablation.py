"""Fixed-parameter relocation ablation for the minimal handoff adapter."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from .handoff_only_switch_experiment import (
    DEFAULT_SAMPLES,
    FAILURE_SAMPLES,
    count_down_up_bounces,
)
from .post_motion_minimal_diagnostics import analyse_archived_report
from .post_motion_reset_fft_reacquire import load_lite_report_config
from .reference_overlap import aligned_reference_bpm
from .solver import V2SolverResult, solve_v2
from .types import V2RunConfig


@dataclass(frozen=True)
class MinimalRelocationCandidate:
    name: str
    relocation_mode: str
    mechanism_count: int
    runtime_eligible: bool = True


def build_relocation_candidates() -> tuple[MinimalRelocationCandidate, ...]:
    return (
        MinimalRelocationCandidate("minimal_none", "none", 0),
        MinimalRelocationCandidate("minimal_a2", "a2", 1),
        MinimalRelocationCandidate(
            "minimal_reanchor", "controlled_reanchor", 1
        ),
        MinimalRelocationCandidate(
            "minimal_a2_reanchor", "a2_reanchor", 2, runtime_eligible=False
        ),
    )


def build_ablation_configs(base: V2RunConfig) -> dict[str, V2RunConfig]:
    """Build four configs whose only candidate factor is relocation."""

    return {
        candidate.name: replace(
            base,
            post_motion_dual_reset_enable=True,
            post_motion_dual_reset_experiment_mode="a0",
            post_motion_dual_reset_handoff_only_switch=False,
            post_motion_minimal_handoff_enable=True,
            post_motion_minimal_relocation_mode=candidate.relocation_mode,
            post_motion_dual_reset_prior_invalidation_enable=False,
            post_motion_dual_reset_post_switch_hold_actual_final=False,
            post_motion_dual_reset_gap_rescue_gap_bpm=18.0,
        )
        for candidate in build_relocation_candidates()
    }


def build_provisional_configs(base: V2RunConfig) -> dict[str, V2RunConfig]:
    common = replace(
        base,
        post_motion_dual_reset_enable=True,
        post_motion_dual_reset_experiment_mode="a0",
        post_motion_dual_reset_handoff_only_switch=False,
        post_motion_minimal_handoff_enable=True,
        post_motion_minimal_relocation_mode="controlled_reanchor",
        post_motion_dual_reset_prior_invalidation_enable=False,
        post_motion_dual_reset_post_switch_hold_actual_final=False,
        post_motion_dual_reset_gap_rescue_gap_bpm=18.0,
    )
    return {
        "minimal_reanchor": common,
        "minimal_provisional_reanchor": replace(
            common,
            post_motion_minimal_provisional_enable=True,
        ),
    }


def build_provisional_candidates() -> tuple[MinimalRelocationCandidate, ...]:
    return (
        MinimalRelocationCandidate(
            "minimal_reanchor", "controlled_reanchor", 1
        ),
        MinimalRelocationCandidate(
            "minimal_provisional_reanchor", "controlled_reanchor", 2
        ),
    )


def select_relocation_candidate(
    candidates: Sequence[MinimalRelocationCandidate],
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select by the frozen continuity→normal→complexity→benefit order."""

    by_name = {str(row["candidate"]): row for row in summaries}
    eligible = [
        candidate
        for candidate in candidates
        if candidate.name in by_name
        and candidate.runtime_eligible
        and bool(by_name[candidate.name].get("acceptance_pass", True))
        and bool(by_name[candidate.name]["independent_reset_invariant"])
        and int(by_name[candidate.name]["bounce_count"]) == 0
        and int(by_name[candidate.name]["wrong_hard_switch_count"]) == 0
    ]
    if not eligible:
        failed_gates = {
            candidate.name: str(
                by_name.get(candidate.name, {}).get(
                    "failed_gates", "missing_or_failed_acceptance"
                )
            )
            for candidate in candidates
            if candidate.runtime_eligible
        }
        return {
            "verdict": "NO_GO",
            "selected_candidate": None,
            "selected_relocation_mode": None,
            "reason": "all_runtime_candidates_failed_frozen_acceptance",
            "failed_gates_by_candidate": failed_gates,
        }

    def rank(candidate: MinimalRelocationCandidate) -> tuple[float, ...]:
        row = by_name[candidate.name]
        return (
            float(row["normal_mean_post60_mae_bpm"]),
            float(row["relocation_mechanism_count"]),
            float(row["failure_mean_post60_mae_bpm"]),
            float(row["all_mean_post60_mae_bpm"]),
        )

    selected = min(eligible, key=rank)
    return {
        "verdict": "GO",
        "selected_candidate": selected.name,
        "selected_relocation_mode": selected.relocation_mode,
        "reason": "single_relocation_lexicographic_selection",
        "ranking_order": [
            "continuity_and_independent_reset",
            "normal_pool_mae",
            "relocation_complexity",
            "failure_pool_mae",
            "all_sample_mae",
        ],
    }


def run_relocation_ablation(
    report_dir: str | Path,
    main_report_dir: str | Path,
    output_dir: str | Path,
    samples: Sequence[str] = DEFAULT_SAMPLES,
    provisional_experiment: bool = False,
) -> dict[str, Any]:
    source = Path(report_dir)
    main_source = Path(main_report_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = (
        build_provisional_candidates()
        if provisional_experiment
        else build_relocation_candidates()
    )
    sample_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for sample in samples:
        matches = sorted(source.glob(f"{sample}_*-v2.json"))
        if len(matches) != 1:
            raise ValueError(f"{sample}: expected one report, found {len(matches)}")
        main_matches = sorted(main_source.glob(f"{sample}_*-v2.json"))
        if len(main_matches) != 1:
            raise ValueError(
                f"{sample}: expected one main report, found {len(main_matches)}"
            )
        source_payload = json.loads(matches[0].read_text(encoding="utf-8"))
        main_payload = json.loads(main_matches[0].read_text(encoding="utf-8"))
        source_post60_mae = float(
            analyse_archived_report(source_payload)["post60_mae_bpm"]
        )
        main_post60_mae = float(
            analyse_archived_report(main_payload)["post60_mae_bpm"]
        )
        base = load_lite_report_config(matches[0])
        configs = (
            build_provisional_configs(base)
            if provisional_experiment
            else build_ablation_configs(base)
        )
        results = {name: solve_v2(config) for name, config in configs.items()}
        baseline = results[candidates[0].name]
        experiment_baseline_post60_mae = float(
            baseline.err_stats["post_motion_60s_mae_bpm"]
        )
        for candidate in candidates:
            result = results[candidate.name]
            row, windows = _evaluate_result(
                sample,
                candidate,
                configs[candidate.name],
                baseline,
                result,
                main_post60_mae=main_post60_mae,
                source_post60_mae=source_post60_mae,
                experiment_baseline_post60_mae=(
                    experiment_baseline_post60_mae
                ),
            )
            sample_rows.append(row)
            window_rows.extend(windows)

    summaries = [
        _candidate_summary(
            candidate,
            sample_rows,
            provisional_experiment=provisional_experiment,
        )
        for candidate in candidates
    ]
    decision = select_relocation_candidate(candidates, summaries)
    decision["sample_order"] = list(samples)
    _write_csv(output / "sample_metrics.csv", sample_rows)
    _write_csv(output / "candidate_summary.csv", summaries)
    _write_csv(output / "window_metrics.csv", window_rows)
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "candidate_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "decision": decision,
        "candidate_summary": summaries,
        "sample_metrics": sample_rows,
        "output_dir": str(output),
    }


def _evaluate_result(
    sample: str,
    candidate: MinimalRelocationCandidate,
    config: V2RunConfig,
    baseline: V2SolverResult,
    result: V2SolverResult,
    main_post60_mae: float,
    source_post60_mae: float,
    experiment_baseline_post60_mae: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    motion_end_s = float(result.metadata["motion_segment"]["end_s"])
    reference = aligned_reference_bpm(result.HR, float(config.time_bias))
    times = result.HR[:, 0]
    tail = np.flatnonzero(
        (times > motion_end_s)
        & (times <= motion_end_s + 60.0)
        & np.isfinite(reference)
    )
    final = result.HR[:, 3]
    tail_final = final[tail]
    trace_rows = {
        int(row["window_idx"]): row
        for row in result.window_table
        if "handoff_reset_bpm" in row
    }
    first_switch = next(
        (
            int(row["window_idx"])
            for row in result.window_table
            if bool(row.get("handoff_consumed"))
        ),
        None,
    )
    wrong_hard = int(
        first_switch is not None
        and first_switch > 0
        and str(result.window_table[first_switch].get("switch_state"))
        == "gap_rescue"
        and abs(final[first_switch] - reference[first_switch]) > 20.0
        and abs(final[first_switch] - reference[first_switch])
        > abs(final[first_switch - 1] - reference[first_switch - 1])
    )
    invariant = _independent_invariance(baseline.window_table, result.window_table)
    states = [
        str(row.get("switch_state", ""))
        for row in result.window_table
        if float(row["center_s"]) > motion_end_s
    ]
    sample_row = {
        "sample": sample,
        "cohort": "failure" if sample in FAILURE_SAMPLES else "normal",
        "candidate": candidate.name,
        "relocation_mode": candidate.relocation_mode,
        "relocation_mechanism_count": candidate.mechanism_count,
        "post60_mae_bpm": float(result.err_stats["post_motion_60s_mae_bpm"]),
        "main_post60_mae_bpm": main_post60_mae,
        "delta_vs_main_post60_mae_bpm": float(
            result.err_stats["post_motion_60s_mae_bpm"] - main_post60_mae
        ),
        "source_post60_mae_bpm": source_post60_mae,
        "experiment_baseline_post60_mae_bpm": (
            experiment_baseline_post60_mae
        ),
        "delta_vs_experiment_baseline_post60_mae_bpm": float(
            result.err_stats["post_motion_60s_mae_bpm"]
            - experiment_baseline_post60_mae
        ),
        "lost_existing_sub3_rescue": bool(
            source_post60_mae < 3.0
            and float(result.err_stats["post_motion_60s_mae_bpm"]) >= 3.0
        ),
        "post60_e10_count": int(result.err_stats["post_motion_60s_e10_count"]),
        "post60_e20_count": int(result.err_stats["post_motion_60s_e20_count"]),
        "bounce_count": count_down_up_bounces(tail_final),
        "wrong_hard_switch_count": wrong_hard,
        "first_switch_center_s": (
            None if first_switch is None else float(times[first_switch])
        ),
        "first_switch_state": (
            ""
            if first_switch is None
            else str(result.window_table[first_switch].get("switch_state", ""))
        ),
        "control_state_count": len(set(states)),
        "control_transition_count": sum(
            current != previous
            for previous, current in zip(states, states[1:], strict=False)
        ),
        **invariant,
    }
    windows = []
    for index in tail:
        trace = trace_rows.get(int(index), {})
        windows.append(
            {
                "sample": sample,
                "candidate": candidate.name,
                "window_idx": int(index),
                "center_s": float(times[index]),
                "reference_bpm": float(reference[index]),
                "final_bpm": float(final[index]),
                "independent_reset_bpm": trace.get("independent_reset_bpm"),
                "handoff_target_bpm": trace.get("handoff_reset_bpm"),
                "candidate_stable": bool(trace.get("candidate_stable")),
                "tracker_converged": bool(trace.get("tracker_converged")),
                "target_consumable": bool(trace.get("target_consumable")),
                "switch_state": str(trace.get("switch_state", "")),
                "final_source": str(trace.get("final_source", "")),
                "reinitialization_count": int(
                    trace.get("handoff_reinitialization_count", 0)
                ),
                "reanchor_event": bool(
                    (trace.get("handoff_reset_trace") or {}).get(
                        "reanchor_event"
                    )
                ),
            }
        )
    return sample_row, windows


def _independent_invariance(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = {
        int(row["window_idx"]): row
        for row in baseline_rows
        if "independent_reset_bpm" in row
    }
    candidate = {
        int(row["window_idx"]): row
        for row in candidate_rows
        if "independent_reset_bpm" in row
    }
    shared = sorted(set(baseline) & set(candidate))
    value_mismatch = sum(
        float(baseline[index]["independent_reset_bpm"])
        != float(candidate[index]["independent_reset_bpm"])
        for index in shared
    )
    raw_mismatch = sum(
        baseline[index].get("raw_top5") != candidate[index].get("raw_top5")
        for index in shared
    )
    trace_mismatch = sum(
        baseline[index].get("independent_reset_trace")
        != candidate[index].get("independent_reset_trace")
        for index in shared
    )
    return {
        "independent_reset_window_count": len(shared),
        "independent_reset_value_mismatch_count": value_mismatch,
        "independent_reset_raw_top5_mismatch_count": raw_mismatch,
        "independent_reset_trace_mismatch_count": trace_mismatch,
        "independent_reset_invariant": bool(
            shared and not value_mismatch and not raw_mismatch and not trace_mismatch
        ),
    }


def _candidate_summary(
    candidate: MinimalRelocationCandidate,
    rows: Sequence[Mapping[str, Any]],
    *,
    provisional_experiment: bool = False,
) -> dict[str, Any]:
    selected = [row for row in rows if row["candidate"] == candidate.name]
    failures = [row for row in selected if row["cohort"] == "failure"]
    normals = [row for row in selected if row["cohort"] == "normal"]
    delta_key = (
        "delta_vs_experiment_baseline_post60_mae_bpm"
        if provisional_experiment
        else "delta_vs_main_post60_mae_bpm"
    )
    normal_delta = fmean(float(row[delta_key]) for row in normals)
    failure_delta = fmean(float(row[delta_key]) for row in failures)
    normal_delta_vs_main = fmean(
        float(row["delta_vs_main_post60_mae_bpm"]) for row in normals
    )
    failure_delta_vs_main = fmean(
        float(row["delta_vs_main_post60_mae_bpm"]) for row in failures
    )
    normal_review = sum(
        float(row[delta_key]) > 2.0 for row in normals
    )
    lost_rescues = sum(
        bool(row["lost_existing_sub3_rescue"]) for row in failures
    )
    invariant = all(
        bool(row["independent_reset_invariant"]) for row in selected
    )
    bounce_count = sum(int(row["bounce_count"]) for row in selected)
    wrong_switches = sum(
        int(row["wrong_hard_switch_count"]) for row in selected
    )
    bobi2_mae = next(
        (
            float(row["post60_mae_bpm"])
            for row in failures
            if row["sample"] == "bobi2"
        ),
        float("inf"),
    )
    switch_centers = [
        float(row["first_switch_center_s"])
        for row in selected
        if row["first_switch_center_s"] is not None
    ]
    acceptance_pass = bool(
        invariant
        and bounce_count == 0
        and wrong_switches == 0
        and normal_delta <= 0.5
        and normal_review == 0
        and (provisional_experiment or failure_delta < 0.0)
        and lost_rescues == 0
        and (not provisional_experiment or bobi2_mae < 3.0)
    )
    failed_gates = []
    if not invariant:
        failed_gates.append("independent_reset_invariance")
    if bounce_count:
        failed_gates.append("down_up_bounce")
    if wrong_switches:
        failed_gates.append("wrong_hard_switch")
    if normal_delta > 0.5:
        failed_gates.append("normal_pool_delta_over_0.5_bpm")
    if normal_review:
        failed_gates.append("normal_sample_regression_over_2_bpm")
    if not provisional_experiment and failure_delta >= 0.0:
        failed_gates.append("failure_pool_no_improvement")
    if lost_rescues:
        failed_gates.append("lost_existing_sub3_rescue")
    if provisional_experiment and bobi2_mae >= 3.0:
        failed_gates.append("bobi2_not_below_3_bpm")
    return {
        "candidate": candidate.name,
        "relocation_mode": candidate.relocation_mode,
        "relocation_mechanism_count": candidate.mechanism_count,
        "sample_count": len(selected),
        "failure_mean_post60_mae_bpm": fmean(
            float(row["post60_mae_bpm"]) for row in failures
        ),
        "normal_mean_post60_mae_bpm": fmean(
            float(row["post60_mae_bpm"]) for row in normals
        ),
        "all_mean_post60_mae_bpm": fmean(
            float(row["post60_mae_bpm"]) for row in selected
        ),
        "normal_mean_delta_vs_main_bpm": normal_delta_vs_main,
        "acceptance_baseline": (
            "minimal_reanchor" if provisional_experiment else "main"
        ),
        "normal_mean_delta_vs_acceptance_baseline_bpm": normal_delta,
        "normal_regression_over_2bpm_count": normal_review,
        "failure_mean_delta_vs_main_bpm": failure_delta_vs_main,
        "failure_mean_delta_vs_acceptance_baseline_bpm": failure_delta,
        "bobi2_post60_mae_bpm": bobi2_mae,
        "lost_existing_sub3_rescue_count": lost_rescues,
        "bounce_count": bounce_count,
        "wrong_hard_switch_count": wrong_switches,
        "independent_reset_invariant": invariant,
        "first_switch_min_center_s": min(switch_centers) if switch_centers else None,
        "first_switch_max_center_s": max(switch_centers) if switch_centers else None,
        "no_switch_sample_count": len(selected) - len(switch_centers),
        "acceptance_pass": acceptance_pass,
        "failed_gates": ";".join(failed_gates),
        "mean_control_state_count": fmean(
            int(row["control_state_count"]) for row in selected
        ),
        "mean_control_transition_count": fmean(
            int(row["control_transition_count"]) for row in selected
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty result: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("main_report_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--samples", nargs="*")
    parser.add_argument("--provisional-experiment", action="store_true")
    args = parser.parse_args()
    result = run_relocation_ablation(
        args.report_dir,
        args.main_report_dir,
        args.output_dir,
        samples=tuple(args.samples or DEFAULT_SAMPLES),
        provisional_experiment=bool(args.provisional_experiment),
    )
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
