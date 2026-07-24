"""Phase2 单记录独立 BO 的双基线编排与审计产物。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .bo_space_generalization import (
    METRIC_CONTRACT_VERSION,
    BOCandidate,
    CandidateSolveOutcome,
    ContentAddressedSolverCache,
    FormalMetricResult,
    SearchEvaluation,
    SearchExperimentIdentity,
    SearchRequestContext,
    SearchTrialRecord,
    SeedSearchBudget,
    SeedSearchResult,
    SolverCacheIdentity,
    build_bo_search_space,
    evaluate_formal_metrics,
    run_seed_search,
)
from .phase2_experiment_io import atomic_temp_path
from .phase2_solver_diagnostics import collect_solver_diagnostics
from .plotting import render_v2_report
from .post_motion_reset_fft_reacquire import load_lite_report_config
from .preprocess import load_v2_dataset
from .reference_groups import method_label
from .report import load_v2_report, save_v2_report
from .solver import V2SolverResult, solve_v2

_INVALID_OBJECTIVE = 1e9


class IndependentInputIdentityMismatchError(ValueError):
    """历史报告输入与 preflight 冻结身份不一致。"""


class IndependentMethodIdentityMismatchError(ValueError):
    """正式比较所需的方法名不精确或不完整。"""


@dataclass(frozen=True)
class IndependentStudyConfig:
    historical_report_path: Path
    historical_error_csv: Path
    output_dir: Path
    git_commit: str
    expected_data_path: Path | None = None
    expected_reference_path: Path | None = None
    scene: str = ""
    legacy_budget: SeedSearchBudget = SeedSearchBudget()
    physical_budget: SeedSearchBudget = SeedSearchBudget(
        objective_version="phase2_independent_physical_v1"
    )
    parallel_lanes: bool = False


@dataclass(frozen=True)
class IndependentRecordRuntime:
    """一条记录的已审计输入和可替换求解/绘图边界。"""

    sample_id: str
    data_sha256: str
    reference_sha256: str
    run_config: Mapping[str, Any]
    historical_metrics: FormalMetricResult
    historical_method_names: tuple[str, ...]
    historical_plot: Path
    solve_candidate: Callable[[BOCandidate], CandidateSolveOutcome]
    render_selected: Callable[
        [str, BOCandidate, CandidateSolveOutcome, Path],
        Path,
    ]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class IndependentArmResult:
    arm: str
    selected_candidate_id: str
    selected_metrics: FormalMetricResult
    candidate_history: Path
    seed_stability: Path
    classic_plot: Path
    search_result: SeedSearchResult
    cache_summary: Mapping[str, Any]


@dataclass(frozen=True)
class IndependentStudyResult:
    sample_id: str
    historical_metrics: FormalMetricResult
    historical_plot: Path
    legacy: IndependentArmResult
    physical: IndependentArmResult
    comparison: Mapping[str, float | bool]
    comparison_table: Path
    acceptance_preview: Path


def run_independent_bo_study(
    config: IndependentStudyConfig,
    *,
    runtime: IndependentRecordRuntime | None = None,
) -> IndependentStudyResult:
    """完成历史锚点、同代码完整旧空间和新物理空间的单记录闭环。"""

    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    normalized_config = replace(config, output_dir=output)
    active_runtime = runtime or _build_default_runtime(normalized_config)
    legacy = _run_arm(
        arm="legacy_same_code",
        space_name="legacy_full_v1",
        budget=config.legacy_budget,
        config=normalized_config,
        runtime=active_runtime,
    )
    physical = _run_arm(
        arm="physical_new",
        space_name="physical_v1",
        budget=config.physical_budget,
        config=normalized_config,
        runtime=active_runtime,
    )
    comparison = _comparison_values(
        historical=active_runtime.historical_metrics,
        legacy=legacy.selected_metrics,
        physical=physical.selected_metrics,
    )
    comparison_table = output / "independent_dual_baseline.csv"
    _write_comparison_table(
        comparison_table,
        historical=active_runtime.historical_metrics,
        legacy=legacy.selected_metrics,
        physical=physical.selected_metrics,
    )
    acceptance_preview = output / "acceptance_preview.json"
    _atomic_write_json(
        acceptance_preview,
        {
            "sample_id": active_runtime.sample_id,
            "scope": "single_record_preview_not_stage_2_1_decision",
            "metric_contract_version": METRIC_CONTRACT_VERSION,
            **comparison,
        },
    )
    _atomic_write_json(
        output / "independent_study_manifest.json",
        {
            "sample_id": active_runtime.sample_id,
            "git_commit": config.git_commit,
            "data_sha256": active_runtime.data_sha256,
            "reference_sha256": active_runtime.reference_sha256,
            "historical_method_names": active_runtime.historical_method_names,
            "historical_plot": str(active_runtime.historical_plot),
            "legacy_plot": str(legacy.classic_plot),
            "physical_plot": str(physical.classic_plot),
            "comparison_table": str(comparison_table),
            "acceptance_preview": str(acceptance_preview),
            "evidence_level": "development_reuse_pilot",
        },
    )
    return IndependentStudyResult(
        sample_id=active_runtime.sample_id,
        historical_metrics=active_runtime.historical_metrics,
        historical_plot=active_runtime.historical_plot,
        legacy=legacy,
        physical=physical,
        comparison=comparison,
        comparison_table=comparison_table,
        acceptance_preview=acceptance_preview,
    )


def _run_arm(
    *,
    arm: str,
    space_name: str,
    budget: SeedSearchBudget,
    config: IndependentStudyConfig,
    runtime: IndependentRecordRuntime,
) -> IndependentArmResult:
    arm_dir = Path(config.output_dir) / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    space = build_bo_search_space(space_name)
    candidates = {
        candidate.candidate_id: candidate for candidate in space.candidates
    }
    cache = ContentAddressedSolverCache(Path(config.output_dir) / "cache")
    trial_audit_dir = arm_dir / "trial_audit"

    def cache_identity(candidate: BOCandidate) -> SolverCacheIdentity:
        return SolverCacheIdentity(
            data_sha256=runtime.data_sha256,
            reference_sha256=runtime.reference_sha256,
            git_commit=config.git_commit,
            run_config={
                **_json_ready(runtime.run_config),
                "arm": arm,
                "space_name": space_name,
            },
            candidate=candidate,
            reference_groups_order=("HF",),
        )

    def evaluate(
        candidate: BOCandidate,
        context: SearchRequestContext,
    ) -> SearchEvaluation:
        lookup = cache.get_or_solve(
            cache_identity(candidate),
            lambda: runtime.solve_candidate(candidate),
            logical_reference={"arm": arm, **asdict(context)},
        )
        _atomic_write_json(
            _trial_audit_path(trial_audit_dir, context),
            {
                **asdict(context),
                "candidate_id": candidate.candidate_id,
                "cache_key": lookup.cache_key,
                "cache_hit": lookup.cache_hit,
                "physical_solve_performed": (
                    lookup.physical_solve_performed
                ),
                "outcome_status": lookup.outcome.status,
                "failure_reason": lookup.outcome.failure_reason,
                "formal_metrics": (
                    asdict(lookup.outcome.formal_metrics)
                    if lookup.outcome.formal_metrics is not None
                    else {}
                ),
                "diagnostics": dict(lookup.outcome.diagnostics),
            },
        )
        if lookup.outcome.status != "valid":
            return SearchEvaluation(
                objective=_INVALID_OBJECTIVE,
                metric_valid=False,
                eligible=False,
                failure_reason=lookup.outcome.failure_reason,
            )
        metrics = lookup.outcome.formal_metrics
        if metrics is None:
            raise RuntimeError("valid 候选缺少 formal_metrics")
        return SearchEvaluation(objective=metrics.full_final_mae_bpm)

    identity = SearchExperimentIdentity(
        input_sha256s=(runtime.data_sha256,),
        reference_sha256s=(runtime.reference_sha256,),
        git_commit=config.git_commit,
        run_config={
            **_json_ready(runtime.run_config),
            "arm": arm,
            "space_name": space_name,
        },
        evaluation_version=budget.objective_version,
    )
    search_result = run_seed_search(
        space=space,
        output_dir=arm_dir / "search",
        experiment_identity=identity,
        evaluate=evaluate,
        budget=budget,
        parallel_lanes=config.parallel_lanes,
    )
    selected_candidate, selected_outcome = _select_final_candidate(
        candidate_ids=search_result.global_candidate_ids,
        candidates=candidates,
        cache=cache,
        cache_identity=cache_identity,
        solve_candidate=runtime.solve_candidate,
        arm=arm,
    )
    if selected_outcome.formal_metrics is None:
        raise RuntimeError("最终候选缺少 formal_metrics")
    _atomic_write_json(
        arm_dir / "selected_candidate.json",
        {
            "arm": arm,
            "candidate_id": selected_candidate.candidate_id,
            "requested_params": selected_candidate.requested_params,
            "actual_params": selected_candidate.actual_params,
            "fixed_params": selected_candidate.fixed_params,
            "formal_metrics": asdict(selected_outcome.formal_metrics),
            "diagnostics": dict(selected_outcome.diagnostics),
        },
    )
    history_path = arm_dir / "candidate_history.csv"
    _write_candidate_history(
        history_path,
        arm=arm,
        scene=config.scene or _scene_from_sample_id(runtime.sample_id),
        search_result=search_result,
        candidates=candidates,
        audit_dir=trial_audit_dir,
    )
    cache_summary = _arm_cache_summary(cache, arm)
    stability_path = arm_dir / "seed_stability.json"
    _write_seed_stability(
        stability_path,
        search_result,
        candidates=candidates,
        cache_statistics=cache_summary,
    )
    classic_plot = runtime.render_selected(
        arm,
        selected_candidate,
        selected_outcome,
        arm_dir,
    )
    _atomic_write_json(arm_dir / "cache_summary.json", cache_summary)
    return IndependentArmResult(
        arm=arm,
        selected_candidate_id=selected_candidate.candidate_id,
        selected_metrics=selected_outcome.formal_metrics,
        candidate_history=history_path,
        seed_stability=stability_path,
        classic_plot=classic_plot,
        search_result=search_result,
        cache_summary=cache_summary,
    )


def _select_final_candidate(
    *,
    candidate_ids: Sequence[str],
    candidates: Mapping[str, BOCandidate],
    cache: ContentAddressedSolverCache,
    cache_identity: Callable[[BOCandidate], SolverCacheIdentity],
    solve_candidate: Callable[[BOCandidate], CandidateSolveOutcome],
    arm: str,
) -> tuple[BOCandidate, CandidateSolveOutcome]:
    valid: list[tuple[BOCandidate, CandidateSolveOutcome]] = []
    for candidate_id in candidate_ids:
        candidate = candidates[candidate_id]
        lookup = cache.get_or_solve(
            cache_identity(candidate),
            lambda candidate=candidate: solve_candidate(candidate),
            logical_reference={
                "arm": arm,
                "stage": "final_selection",
                "candidate_id": candidate_id,
            },
        )
        if (
            lookup.outcome.status == "valid"
            and lookup.outcome.formal_metrics is not None
        ):
            valid.append((candidate, lookup.outcome))
    if not valid:
        raise RuntimeError(f"{arm} 没有正式指标有效候选")
    return min(
        valid,
        key=lambda item: (
            item[1].formal_metrics.full_final_mae_bpm,
            item[1].formal_metrics.reliable_motion_final_mae_bpm,
            item[0].candidate_id,
        ),
    )


def _write_candidate_history(
    path: Path,
    *,
    arm: str,
    scene: str,
    search_result: SeedSearchResult,
    candidates: Mapping[str, BOCandidate],
    audit_dir: Path,
) -> None:
    history = [
        *(row for lane in search_result.lanes for row in lane.history),
        *search_result.fill_history,
    ]
    parameter_keys = sorted(
        {
            key
            for candidate in candidates.values()
            for key in (
                *candidate.requested_params.keys(),
                *candidate.actual_params.keys(),
                *candidate.fixed_params.keys(),
            )
        }
    )
    rows: list[dict[str, Any]] = []
    for row in history:
        context = SearchRequestContext(
            lane=row.lane,
            seed=row.seed,
            trial_number=row.trial_number,
            stage=row.stage,
            suggestion_index=row.suggestion_index,
            unique_index=row.unique_index,
            is_duplicate=row.is_duplicate,
        )
        audit = _read_json(_trial_audit_path(audit_dir, context))
        candidate = candidates[row.candidate_id]
        metrics = audit.get("formal_metrics", {})
        diagnostics = audit.get("diagnostics", {})
        output = {
            "arm": arm,
            "scene": scene,
            "fold": "independent",
            "lane": row.lane,
            "seed": row.seed,
            "trial_number": row.trial_number,
            "suggestion_index": row.suggestion_index,
            "unique_index": row.unique_index,
            "candidate_id": row.candidate_id,
            "is_duplicate": row.is_duplicate,
            "cache_hit": audit["cache_hit"],
            "cache_key": audit["cache_key"],
            "physical_solve_performed": audit["physical_solve_performed"],
            "objective": row.objective,
            "metric_valid": row.metric_valid,
            "eligible": row.eligible,
            "failure_reason": row.failure_reason,
            "stage": row.stage,
            **metrics,
            **{
                f"diagnostic_{key}": value
                for key, value in diagnostics.items()
            },
        }
        for key in parameter_keys:
            output[f"requested_{key}"] = candidate.requested_params.get(key)
            output[f"actual_{key}"] = candidate.actual_params.get(key)
            output[f"fixed_{key}"] = candidate.fixed_params.get(key)
        rows.append(output)
    _write_csv(path, rows)


def _write_seed_stability(
    path: Path,
    result: SeedSearchResult,
    *,
    candidates: Mapping[str, BOCandidate],
    cache_statistics: Mapping[str, Any],
) -> None:
    lanes = []
    best_by_seed: dict[int, SearchTrialRecord] = {}
    lane_candidate_ids: dict[int, set[str]] = {}
    for lane in result.lanes:
        best = min(
            lane.history,
            key=lambda row: (
                not row.eligible,
                row.objective,
                row.candidate_id,
            ),
        )
        best_by_seed[lane.seed] = best
        lane_candidate_ids[lane.seed] = set(lane.unique_candidate_ids)
        best_so_far = []
        incumbent: SearchTrialRecord | None = None
        for row in lane.history:
            if incumbent is None or (
                not row.eligible,
                row.objective,
                row.candidate_id,
            ) < (
                not incumbent.eligible,
                incumbent.objective,
                incumbent.candidate_id,
            ):
                incumbent = row
            best_so_far.append(
                {
                    "trial_number": row.trial_number,
                    "suggestion_index": row.suggestion_index,
                    "candidate_id": incumbent.candidate_id,
                    "objective": incumbent.objective,
                    "eligible": incumbent.eligible,
                }
            )
        lanes.append(
            {
                "seed": lane.seed,
                "logical_suggestion_count": len(lane.history),
                "unique_candidate_count": lane.unique_candidate_count,
                "duplicate_suggestion_count": sum(
                    row.is_duplicate for row in lane.history
                ),
                "best_candidate_id": best.candidate_id,
                "best_objective": best.objective,
                "best_requested_params": candidates[
                    best.candidate_id
                ].requested_params,
                "best_actual_params": candidates[
                    best.candidate_id
                ].actual_params,
                "best_so_far": best_so_far,
            }
        )
    pairwise_overlaps = []
    best_parameter_differences = []
    seeds = sorted(lane_candidate_ids)
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1 :]:
            overlap = sorted(
                lane_candidate_ids[left_seed]
                & lane_candidate_ids[right_seed]
            )
            pairwise_overlaps.append(
                {
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "overlap_count": len(overlap),
                    "candidate_ids": overlap,
                }
            )
            left_candidate = candidates[
                best_by_seed[left_seed].candidate_id
            ]
            right_candidate = candidates[
                best_by_seed[right_seed].candidate_id
            ]
            requested_keys = sorted(
                set(left_candidate.requested_params)
                | set(right_candidate.requested_params)
            )
            actual_keys = sorted(
                set(left_candidate.actual_params)
                | set(right_candidate.actual_params)
            )
            best_parameter_differences.append(
                {
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "left_candidate_id": left_candidate.candidate_id,
                    "right_candidate_id": right_candidate.candidate_id,
                    "differing_requested_params": [
                        key
                        for key in requested_keys
                        if left_candidate.requested_params.get(key)
                        != right_candidate.requested_params.get(key)
                    ],
                    "differing_actual_params": [
                        key
                        for key in actual_keys
                        if left_candidate.actual_params.get(key)
                        != right_candidate.actual_params.get(key)
                    ],
                }
            )
    candidate_lane_counts: dict[str, int] = {}
    for candidate_ids in lane_candidate_ids.values():
        for candidate_id in candidate_ids:
            candidate_lane_counts[candidate_id] = (
                candidate_lane_counts.get(candidate_id, 0) + 1
            )
    cache_counts = {
        key: cache_statistics[key]
        for key in (
            "logical_request_count",
            "physical_solve_count",
            "cache_hit_count",
            "reservation_conflict_count",
            "infrastructure_failure_count",
        )
    }
    _atomic_write_json(
        path,
        {
            "lanes": lanes,
            "cross_lane_overlap_count": sum(
                count > 1 for count in candidate_lane_counts.values()
            ),
            "cross_lane_overlap_candidate_ids": sorted(
                candidate_id
                for candidate_id, count in candidate_lane_counts.items()
                if count > 1
            ),
            "pairwise_lane_overlap_counts": pairwise_overlaps,
            "seed_best_parameter_differences": (
                best_parameter_differences
            ),
            "cache_statistics": cache_counts,
            "seed_stability_candidate_ids": (
                result.seed_stability_candidate_ids
            ),
            "fill_unique_candidate_count": result.fill_unique_candidate_count,
            "global_candidate_count": len(result.global_candidate_ids),
            "requested_global_unique_budget": (
                result.requested_global_unique_budget
            ),
            "effective_global_unique_budget": (
                result.effective_global_unique_budget
            ),
            "space_exhausted": result.space_exhausted,
        },
    )


def _arm_cache_summary(
    cache: ContentAddressedSolverCache,
    arm: str,
) -> dict[str, Any]:
    events = [
        event
        for event in cache.audit_summary()["events"]
        if event.get("logical_reference", {}).get("arm") == arm
    ]
    return {
        "logical_request_count": len(events),
        "physical_solve_count": sum(
            bool(event.get("physical_solve_performed")) for event in events
        ),
        "cache_hit_count": sum(
            bool(event.get("cache_hit")) for event in events
        ),
        "reservation_conflict_count": sum(
            event.get("event_type") == "reservation_conflict"
            for event in events
        ),
        "infrastructure_failure_count": sum(
            event.get("event_type") == "infrastructure_failure"
            for event in events
        ),
        "events": events,
    }


def _comparison_values(
    *,
    historical: FormalMetricResult,
    legacy: FormalMetricResult,
    physical: FormalMetricResult,
) -> dict[str, float | bool]:
    historical_delta = (
        physical.classic_motion_final_mae_bpm
        - historical.classic_motion_final_mae_bpm
    )
    legacy_reliable_delta = (
        physical.reliable_motion_final_mae_bpm
        - legacy.reliable_motion_final_mae_bpm
    )
    legacy_classic_delta = (
        physical.classic_motion_final_mae_bpm
        - legacy.classic_motion_final_mae_bpm
    )
    return {
        "physical_vs_historical_classic_delta_bpm": historical_delta,
        "physical_vs_legacy_reliable_delta_bpm": legacy_reliable_delta,
        "physical_vs_legacy_classic_delta_bpm": legacy_classic_delta,
        "per_record_delta_limit_bpm": 2.0,
        "historical_classic_preview_pass": historical_delta <= 2.0,
        "legacy_reliable_preview_pass": legacy_reliable_delta <= 2.0,
        "legacy_classic_preview_pass": legacy_classic_delta <= 2.0,
        "new_disaster_vs_historical": (
            historical.classic_motion_final_mae_bpm <= 5.0
            and physical.classic_motion_final_mae_bpm >= 10.0
        ),
        "new_disaster_vs_legacy_reliable": (
            legacy.reliable_motion_final_mae_bpm <= 5.0
            and physical.reliable_motion_final_mae_bpm >= 10.0
        ),
        "new_disaster_vs_legacy_classic": (
            legacy.classic_motion_final_mae_bpm <= 5.0
            and physical.classic_motion_final_mae_bpm >= 10.0
        ),
    }


def _write_comparison_table(
    path: Path,
    *,
    historical: FormalMetricResult,
    legacy: FormalMetricResult,
    physical: FormalMetricResult,
) -> None:
    rows = []
    for arm, metrics in (
        ("historical_anchor", historical),
        ("legacy_same_code", legacy),
        ("physical_new", physical),
    ):
        rows.append(
            {
                "arm": arm,
                "full_final_mae_bpm": (
                    ""
                    if arm == "historical_anchor"
                    else metrics.full_final_mae_bpm
                ),
                "reliable_motion_final_mae_bpm": (
                    ""
                    if arm == "historical_anchor"
                    else metrics.reliable_motion_final_mae_bpm
                ),
                "classic_motion_final_mae_bpm": (
                    metrics.classic_motion_final_mae_bpm
                ),
                "base_full_window_count": (
                    ""
                    if arm == "historical_anchor"
                    else metrics.base_full_window_count
                ),
                "base_motion_window_count": (
                    ""
                    if arm == "historical_anchor"
                    else metrics.base_motion_window_count
                ),
                "classic_motion_window_count": (
                    metrics.classic_motion_window_count
                ),
                "base_motion_window_sha256": (
                    ""
                    if arm == "historical_anchor"
                    else metrics.base_motion_window_sha256
                ),
                "classic_motion_window_sha256": (
                    metrics.classic_motion_window_sha256
                ),
                "final_method": metrics.final_method,
                "reset_fft_method": metrics.reset_fft_method,
            }
        )
    _write_csv(path, rows)


def _build_default_runtime(
    config: IndependentStudyConfig,
) -> IndependentRecordRuntime:
    report_path = Path(config.historical_report_path).resolve()
    error_csv = Path(config.historical_error_csv).resolve()
    payload = load_v2_report(report_path)
    base = load_lite_report_config(payload)
    if (
        config.expected_data_path is not None
        and Path(base.data_path).resolve()
        != Path(config.expected_data_path).resolve()
    ):
        raise IndependentInputIdentityMismatchError(
            "历史报告 data_path 与 preflight 冻结路径不一致: "
            f"{Path(base.data_path).resolve()} != "
            f"{Path(config.expected_data_path).resolve()}"
        )
    if (
        config.expected_reference_path is not None
        and Path(base.ref_path).resolve()
        != Path(config.expected_reference_path).resolve()
    ):
        raise IndependentInputIdentityMismatchError(
            "历史报告 ref_path 与 preflight 冻结路径不一致: "
            f"{Path(base.ref_path).resolve()} != "
            f"{Path(config.expected_reference_path).resolve()}"
        )
    base = replace(
        base,
        analysis_scope="full",
        adaptive_filter="lms",
        reference_groups_order=("HF",),
        lms_mu_min=1e-6,
    )
    dataset = load_v2_dataset(
        base.data_path,
        base.ref_path,
        fs_origin=base.fs_origin,
    )
    historical_result = V2SolverResult(
        HR=np.asarray(payload.get("hr", []), dtype=float),
        err_stats=dict(payload.get("err_stats", {})),
        metadata=dict(payload),
        window_table=list(payload.get("window_table", [])),
    )
    method_names = _method_names_from_error_csv(error_csv)
    _validate_historical_method_names(method_names)
    historical_metrics = evaluate_formal_metrics(
        historical_result,
        ref_data=dataset.ref_data,
        time_bias=float(payload.get("time_bias", base.time_bias)),
        method_names=method_names,
    )

    def solve_candidate(candidate: BOCandidate) -> CandidateSolveOutcome:
        candidate_config = replace(base, **dict(candidate.actual_params))
        started_at = perf_counter()
        result = solve_v2(candidate_config)
        solver_runtime_seconds = perf_counter() - started_at
        metrics = evaluate_formal_metrics(
            result,
            ref_data=dataset.ref_data,
            time_bias=candidate_config.time_bias,
            method_names=(
                "reset FFT",
                method_label("lms", ("HF",)),
            ),
        )
        return CandidateSolveOutcome.valid(
            result,
            metrics,
            diagnostics=collect_solver_diagnostics(
                result,
                max_order=candidate_config.max_order,
                solver_runtime_seconds=solver_runtime_seconds,
            ),
        )

    def render_selected(
        arm: str,
        candidate: BOCandidate,
        outcome: CandidateSolveOutcome,
        output_dir: Path,
    ) -> Path:
        if outcome.solver_result is None:
            raise RuntimeError("最终候选缺少 solver_result")
        report = save_v2_report(
            output_dir / "json" / f"{arm}-{dataset.sample_stem}.json",
            outcome.solver_result,
            best_params=dict(candidate.actual_params),
            artefacts={
                "candidate_id": candidate.candidate_id,
                "requested_params": dict(candidate.requested_params),
                "actual_params": dict(candidate.actual_params),
                "fixed_params": dict(candidate.fixed_params),
            },
        )
        rendered = render_v2_report(
            report,
            out_dir=output_dir / "png",
            csv_dir=output_dir / "csv",
            comparison_groups=(("ACC",),),
        )
        _validate_classic_plot_methods(rendered.error_csv)
        return rendered.figure_png

    historical_render = render_v2_report(
        report_path,
        out_dir=Path(config.output_dir) / "historical_anchor" / "png",
        csv_dir=Path(config.output_dir) / "historical_anchor" / "csv",
        comparison_groups=(("ACC",),),
    )
    _validate_classic_plot_methods(historical_render.error_csv)
    return IndependentRecordRuntime(
        sample_id=dataset.sample_stem,
        data_sha256=_file_sha256(base.data_path),
        reference_sha256=_file_sha256(base.ref_path),
        run_config=_json_ready(asdict(base)),
        historical_metrics=historical_metrics,
        historical_method_names=method_names,
        historical_plot=historical_render.figure_png,
        solve_candidate=solve_candidate,
        render_selected=render_selected,
        diagnostics={},
    )


def _method_names_from_error_csv(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = tuple(str(row.get("method", "")).strip() for row in rows)
    if not names or any(not name for name in names):
        raise IndependentMethodIdentityMismatchError(
            f"历史 error CSV 缺少方法身份: {path}"
        )
    return names


def _validate_historical_method_names(names: Sequence[str]) -> None:
    required = {"reset FFT", method_label("lms", ("HF",))}
    missing = sorted(required - set(names))
    if missing:
        raise IndependentMethodIdentityMismatchError(
            "历史 error CSV 缺少精确方法身份: " + ", ".join(missing)
        )


def _scene_from_sample_id(sample_id: str) -> str:
    prefix = str(sample_id).split("_", maxsplit=1)[0]
    return prefix.rstrip("0123456789") or prefix


def _validate_classic_plot_methods(path: Path) -> None:
    names = set(_method_names_from_error_csv(path))
    required = {
        "reset FFT",
        method_label("lms", ("HF",)),
        method_label("lms", ("ACC",)),
    }
    missing = sorted(required - names)
    if missing:
        raise IndependentMethodIdentityMismatchError(
            "经典心率图缺少必需方法曲线: " + ", ".join(missing)
        )


def _trial_audit_path(
    root: Path,
    context: SearchRequestContext,
) -> Path:
    safe_lane = context.lane.replace("/", "_")
    return root / f"{safe_lane}-{context.trial_number}.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"不能写入空 CSV: {path}")
    fieldnames = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )
    temp = atomic_temp_path(path)
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_json_ready(row) for row in rows)
    os.replace(temp, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = atomic_temp_path(path)
    temp.write_text(
        json.dumps(
            _json_ready(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(nested)
            for key, nested in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("审计产物不得包含非有限数")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"不支持的审计类型: {type(value).__name__}")
