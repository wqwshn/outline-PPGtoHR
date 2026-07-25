"""Phase2 场景内三折的 K0 完整旧空间简单平均流程基线。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
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
    SeedSearchBudget,
    SeedSearchResult,
    SolverCacheIdentity,
    build_bo_search_space,
    run_seed_search,
)
from .phase2_experiment_io import (
    all_search_rows as _all_search_rows,
)
from .phase2_experiment_io import (
    atomic_write_json as _atomic_write_json,
)
from .phase2_experiment_io import (
    cache_summary as _cache_summary,
)
from .phase2_experiment_io import (
    file_sha256 as _file_sha256,
)
from .phase2_experiment_io import (
    json_ready as _json_ready,
)
from .phase2_experiment_io import (
    read_json as _read_json,
)
from .phase2_experiment_io import (
    space_sha256 as _space_sha256,
)
from .phase2_experiment_io import (
    trial_audit_path as _trial_audit_path,
)
from .phase2_experiment_io import (
    write_csv as _write_csv,
)
from .phase2_kfold_runtime import (
    ClassicPlotArtifact as _ClassicPlotArtifact,
)
from .phase2_kfold_runtime import (
    KFoldRecordInput,
    KFoldRuntime,
    KFoldTrainingRecordRuntime,
    build_default_kfold_runtime,
    kfold_plot_title,
)
from .phase2_receipt import (
    NeighborhoodEvidence,
    ReplayIdentity,
    SearchBudgetEvidence,
    SelectionEvidence,
    TrainingMetricEvidence,
    freeze_selection,
    replay_frozen_selection,
)
from .types import V2RunConfig

K0_FLOW_LABEL = "完整旧空间简单平均流程基线"
_INVALID_OBJECTIVE = 1e9

ClassicPlotArtifact = _ClassicPlotArtifact
K0TrainingRecordRuntime = KFoldTrainingRecordRuntime
K0RecordInput = KFoldRecordInput
K0FoldRuntime = KFoldRuntime


@dataclass(frozen=True)
class K0FoldConfig:
    output_dir: Path
    scene: str
    fold: int
    git_commit: str
    budget: SeedSearchBudget = SeedSearchBudget(
        objective_version="phase2_k0_mean_full_final_v1"
    )
    parallel_lanes: bool = False
    code_dirty: bool = False

    def __post_init__(self) -> None:
        if not self.scene or not self.git_commit:
            raise ValueError("K0 必须冻结 scene 和 git_commit")
        if type(self.fold) is not int or self.fold < 0:
            raise ValueError("fold 必须是非负整数")
        if self.budget.lane_seeds != (42, 43, 44):
            raise ValueError("K0 固定使用 seed 42/43/44")
        if self.budget.objective_version != "phase2_k0_mean_full_final_v1":
            raise ValueError("K0 objective_version 与简单平均目标合同不一致")


@dataclass(frozen=True)
class K0FoldResult:
    arm: str
    flow_label: str
    selected_candidate_id: str
    selected_mean_train_mae_bpm: float
    candidate_history: Path
    selected_params: Path
    training_metrics: Path
    selection_receipt: Path
    replay_receipt: Path
    replay_status: str
    training_plots: tuple[Path, Path]
    cache_summary: Path
    failure_classification: Path
    manifest: Path
    search_result: SeedSearchResult


def build_k0_default_runtime(
    *,
    base_config: V2RunConfig,
    training_records: tuple[K0RecordInput, K0RecordInput],
    heldout_record: K0RecordInput,
    output_dir: Path | str,
) -> K0FoldRuntime:
    return build_default_kfold_runtime(
        arm="K0",
        base_config=base_config,
        training_records=training_records,
        heldout_record=heldout_record,
        output_dir=output_dir,
    )


def run_k0_fold_study(
    config: K0FoldConfig,
    *,
    runtime: K0FoldRuntime,
) -> K0FoldResult:
    """只用两条训练记录选参，冻结后再回放一条留出记录。"""

    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    space = build_bo_search_space("legacy_full_v1")
    candidates = {
        candidate.candidate_id: candidate for candidate in space.candidates
    }
    cache = ContentAddressedSolverCache(output / "cache")
    trial_audit_dir = output / "trial_audit"

    def cache_identity(
        record: K0TrainingRecordRuntime,
        candidate: BOCandidate,
    ) -> SolverCacheIdentity:
        return SolverCacheIdentity(
            data_sha256=record.identity.data_sha256,
            reference_sha256=record.identity.reference_sha256,
            git_commit=config.git_commit,
            run_config={
                **_json_ready(record.run_config),
                "arm": "K0",
                "scene": config.scene,
                "fold": config.fold,
                "space_name": space.name,
            },
            candidate=candidate,
            reference_groups_order=("HF",),
        )

    def evaluate(
        candidate: BOCandidate,
        context: SearchRequestContext,
    ) -> SearchEvaluation:
        outcomes: list[dict[str, Any]] = []
        valid_metrics: list[FormalMetricResult] = []
        for record_index, record in enumerate(runtime.training_records):
            lookup = cache.get_or_solve(
                cache_identity(record, candidate),
                lambda record=record, candidate=candidate: (
                    record.solve_candidate(candidate)
                ),
                logical_reference={
                    "arm": "K0",
                    "scene": config.scene,
                    "fold": config.fold,
                    "record_id": record.identity.record_id,
                    "record_index": record_index,
                    **asdict(context),
                },
            )
            outcomes.append(
                {
                    "record_id": record.identity.record_id,
                    "cache_key": lookup.cache_key,
                    "cache_hit": lookup.cache_hit,
                    "physical_solve_performed": (
                        lookup.physical_solve_performed
                    ),
                    "status": lookup.outcome.status,
                    "failure_reason": lookup.outcome.failure_reason,
                    "formal_metrics": (
                        asdict(lookup.outcome.formal_metrics)
                        if lookup.outcome.formal_metrics is not None
                        else {}
                    ),
                }
            )
            if (
                lookup.outcome.status == "valid"
                and lookup.outcome.formal_metrics is not None
            ):
                valid_metrics.append(lookup.outcome.formal_metrics)
        audit_payload = {
            **asdict(context),
            "candidate_id": candidate.candidate_id,
            "training_outcomes": outcomes,
        }
        if len(valid_metrics) != 2:
            failure_reason = next(
                (
                    str(outcome["failure_reason"])
                    for outcome in outcomes
                    if outcome["status"] != "valid"
                ),
                "metric_window_contract_failed",
            )
            audit_payload.update(
                {
                    "metric_valid": False,
                    "mean_train_full_final_mae_bpm": _INVALID_OBJECTIVE,
                    "failure_reason": failure_reason,
                }
            )
            _atomic_write_json(
                _trial_audit_path(trial_audit_dir, context),
                audit_payload,
            )
            return SearchEvaluation(
                objective=_INVALID_OBJECTIVE,
                metric_valid=False,
                eligible=False,
                failure_reason=failure_reason,
            )
        mean_mae = float(
            np.mean(
                [metrics.full_final_mae_bpm for metrics in valid_metrics]
            )
        )
        audit_payload.update(
            {
                "metric_valid": True,
                "mean_train_full_final_mae_bpm": mean_mae,
                "failure_reason": "",
            }
        )
        _atomic_write_json(
            _trial_audit_path(trial_audit_dir, context),
            audit_payload,
        )
        return SearchEvaluation(objective=mean_mae)

    experiment_identity = SearchExperimentIdentity(
        input_sha256s=tuple(
            record.identity.data_sha256 for record in runtime.training_records
        ),
        reference_sha256s=tuple(
            record.identity.reference_sha256
            for record in runtime.training_records
        ),
        git_commit=config.git_commit,
        run_config={
            "arm": "K0",
            "flow_label": K0_FLOW_LABEL,
            "scene": config.scene,
            "fold": config.fold,
            "space_name": space.name,
            "training_records": [
                {
                    "identity": asdict(record.identity),
                    "run_config": _json_ready(record.run_config),
                }
                for record in runtime.training_records
            ],
            "heldout_record_identity": asdict(runtime.heldout_record),
        },
        evaluation_version=config.budget.objective_version,
    )
    search_result = run_seed_search(
        space=space,
        output_dir=output / "search",
        experiment_identity=experiment_identity,
        evaluate=evaluate,
        budget=config.budget,
        parallel_lanes=config.parallel_lanes,
    )
    selected_candidate, selected_outcomes, selected_metrics = (
        _select_k0_candidate(
            candidate_ids=search_result.global_candidate_ids,
            candidates=candidates,
            training_records=runtime.training_records,
            cache=cache,
            cache_identity=cache_identity,
            config=config,
        )
    )
    mean_train_mae = float(
        np.mean(
            [metrics.full_final_mae_bpm for metrics in selected_metrics]
        )
    )
    selected_params_path = output / "params.json"
    _atomic_write_json(
        selected_params_path,
        {
            "arm": "K0",
            "flow_label": K0_FLOW_LABEL,
            "candidate_id": selected_candidate.candidate_id,
            "requested_params": selected_candidate.requested_params,
            "actual_params": selected_candidate.actual_params,
            "fixed_params": selected_candidate.fixed_params,
            "mean_train_full_final_mae_bpm": mean_train_mae,
        },
    )
    history_path = output / "candidate_history.csv"
    _write_candidate_history(
        history_path,
        search_result=search_result,
        candidates=candidates,
        audit_dir=trial_audit_dir,
    )
    training_metrics_path = output / "training_metrics.csv"
    _write_training_metrics(
        training_metrics_path,
        records=runtime.training_records,
        metrics=selected_metrics,
    )
    search_identity = _read_json(output / "search" / "search_identity.json")
    config_hash = str(search_identity["config_hash"])
    selection_evidence = SelectionEvidence(
        experiment_name=(
            f"{config.scene}-fold-{config.fold}-k0"
        ),
        arm="K0",
        scene=config.scene,
        fold=config.fold,
        code_commit=config.git_commit,
        code_dirty=config.code_dirty,
        training_records=(
            runtime.training_records[0].identity,
            runtime.training_records[1].identity,
        ),
        heldout_record=runtime.heldout_record,
        space_name=space.name,
        space_sha256=_space_sha256(space.candidates),
        metric_contract_version=METRIC_CONTRACT_VERSION,
        study_identities=(
            f"seed_42:{config_hash}",
            f"seed_43:{config_hash}",
            f"seed_44:{config_hash}",
            f"fill:{config_hash}",
        ),
        budget=SearchBudgetEvidence(
            lane_unique_budget=config.budget.lane_unique_budget,
            requested_global_unique_budget=(
                config.budget.global_unique_budget
            ),
            actual_global_unique_count=len(
                search_result.global_candidate_ids
            ),
            requested_neighborhood_budget=0,
            actual_neighborhood_count=0,
        ),
        selected_candidate_id=selected_candidate.candidate_id,
        selected_requested_params=selected_candidate.requested_params,
        selected_actual_params=selected_candidate.actual_params,
        selected_fixed_params=selected_candidate.fixed_params,
        training_metrics=TrainingMetricEvidence(
            eligible=True,
            common_window_counts=tuple(
                metrics.base_motion_common_finite_count
                for metrics in selected_metrics
            ),
            common_window_sha256s=tuple(
                metrics.base_motion_window_sha256
                for metrics in selected_metrics
            ),
            worst_train_mae_bpm=max(
                metrics.full_final_mae_bpm
                for metrics in selected_metrics
            ),
            mean_train_mae_bpm=mean_train_mae,
            nonharm_deltas_bpm=tuple(
                metrics.reliable_motion_final_mae_bpm
                - metrics.reliable_motion_reset_fft_mae_bpm
                for metrics in selected_metrics
            ),
        ),
        neighborhood_evidence=NeighborhoodEvidence(
            status="not_required",
            reviewed_neighbor_count=0,
            support_ratio=0.0,
            has_cliff=False,
            truncated_center_count=0,
        ),
        candidate_history_sha256=_file_sha256(history_path),
        evidence_level="development_reuse_pilot",
    )
    selection_receipt_path = output / "selection_receipt.json"
    selection_receipt = freeze_selection(
        selection_receipt_path,
        selection_evidence,
    )

    training_plots = tuple(
        record.render_selected(
            selected_candidate,
            outcome,
            output / "training" / record.identity.record_id / "png",
        ).figure_png
        for record, outcome in zip(
            runtime.training_records,
            selected_outcomes,
            strict=True,
        )
    )
    replay_receipt_path = output / "replay_receipt.json"
    replay_receipt = replay_frozen_selection(
        receipt_path=selection_receipt_path,
        expected_selection_hash=selection_receipt.selection_hash,
        replay_identity=ReplayIdentity(
            heldout_record=runtime.heldout_record,
            reference_groups_order=("HF", "ACC"),
        ),
        replay_receipt_path=replay_receipt_path,
        replay=runtime.replay_heldout,
    )
    cache_summary_payload = _cache_summary(cache)
    cache_summary_path = output / "cache_summary.json"
    _atomic_write_json(cache_summary_path, cache_summary_payload)
    invalid_candidate_ids = {
        row.candidate_id
        for row in _all_search_rows(search_result)
        if not row.metric_valid
    }
    failure_classification_path = output / "failure_classification.json"
    _atomic_write_json(
        failure_classification_path,
        {
            "invalid_candidate_count": len(invalid_candidate_ids),
            "invalid_candidate_ids": sorted(invalid_candidate_ids),
            "infrastructure_failure_count": cache_summary_payload[
                "infrastructure_failure_count"
            ],
            "replay_status": replay_receipt.status,
            "replay_failure_reason": replay_receipt.failure_reason,
        },
    )
    manifest_path = output / "k0_fold_manifest.json"
    _atomic_write_json(
        manifest_path,
        {
            "arm": "K0",
            "flow_label": K0_FLOW_LABEL,
            "causal_claim_allowed": False,
            "comparison_scope": (
                "旧平均目标流程与后续完整流程的操作性比较，"
                "不得解释为单个规则的因果作用"
            ),
            "scene": config.scene,
            "fold": config.fold,
            "git_commit": config.git_commit,
            "evidence_level": "development_reuse_pilot",
            "confirmatory_claim_allowed": False,
            "selected_candidate_id": selected_candidate.candidate_id,
            "selection_hash": selection_receipt.selection_hash,
            "candidate_history": str(history_path),
            "selection_receipt": str(selection_receipt_path),
            "replay_receipt": str(replay_receipt_path),
            "training_plots": training_plots,
        },
    )
    return K0FoldResult(
        arm="K0",
        flow_label=K0_FLOW_LABEL,
        selected_candidate_id=selected_candidate.candidate_id,
        selected_mean_train_mae_bpm=mean_train_mae,
        candidate_history=history_path,
        selected_params=selected_params_path,
        training_metrics=training_metrics_path,
        selection_receipt=selection_receipt_path,
        replay_receipt=replay_receipt_path,
        replay_status=replay_receipt.status,
        training_plots=training_plots,
        cache_summary=cache_summary_path,
        failure_classification=failure_classification_path,
        manifest=manifest_path,
        search_result=search_result,
    )


def _select_k0_candidate(
    *,
    candidate_ids: Sequence[str],
    candidates: Mapping[str, BOCandidate],
    training_records: tuple[
        K0TrainingRecordRuntime,
        K0TrainingRecordRuntime,
    ],
    cache: ContentAddressedSolverCache,
    cache_identity: Callable[
        [K0TrainingRecordRuntime, BOCandidate],
        SolverCacheIdentity,
    ],
    config: K0FoldConfig,
) -> tuple[
    BOCandidate,
    tuple[CandidateSolveOutcome, CandidateSolveOutcome],
    tuple[FormalMetricResult, FormalMetricResult],
]:
    valid: list[
        tuple[
            float,
            str,
            BOCandidate,
            tuple[CandidateSolveOutcome, CandidateSolveOutcome],
            tuple[FormalMetricResult, FormalMetricResult],
        ]
    ] = []
    for candidate_id in candidate_ids:
        candidate = candidates[candidate_id]
        outcomes = tuple(
            cache.get_or_solve(
                cache_identity(record, candidate),
                lambda record=record, candidate=candidate: (
                    record.solve_candidate(candidate)
                ),
                logical_reference={
                    "arm": "K0",
                    "scene": config.scene,
                    "fold": config.fold,
                    "stage": "final_selection",
                    "record_id": record.identity.record_id,
                    "candidate_id": candidate_id,
                },
            ).outcome
            for record in training_records
        )
        if any(
            outcome.status != "valid" or outcome.formal_metrics is None
            for outcome in outcomes
        ):
            continue
        metrics = tuple(
            outcome.formal_metrics for outcome in outcomes
        )
        if len(metrics) != 2 or any(metric is None for metric in metrics):
            continue
        typed_metrics = (metrics[0], metrics[1])
        typed_outcomes = (outcomes[0], outcomes[1])
        objective = float(
            np.mean(
                [metric.full_final_mae_bpm for metric in typed_metrics]
            )
        )
        valid.append(
            (
                objective,
                candidate_id,
                candidate,
                typed_outcomes,
                typed_metrics,
            )
        )
    if not valid:
        raise RuntimeError("K0 没有两条训练记录均有效的候选")
    _, _, candidate, outcomes, metrics = min(
        valid,
        key=lambda item: (item[0], item[1]),
    )
    return candidate, outcomes, metrics


def _k0_plot_title(
    *,
    training_record_ids: tuple[str, str],
    heldout_record_id: str,
    view_role: str,
    view_record_id: str,
    actual_params: Mapping[str, Any],
) -> str:
    return kfold_plot_title(
        arm="K0",
        training_record_ids=training_record_ids,
        heldout_record_id=heldout_record_id,
        view_role=view_role,
        view_record_id=view_record_id,
        actual_params=actual_params,
    )


def _write_candidate_history(
    path: Path,
    *,
    search_result: SeedSearchResult,
    candidates: Mapping[str, BOCandidate],
    audit_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    parameter_keys = sorted(
        {
            key
            for candidate in candidates.values()
            for key in (
                *candidate.requested_params,
                *candidate.actual_params,
                *candidate.fixed_params,
            )
        }
    )
    for trial in _all_search_rows(search_result):
        context = SearchRequestContext(
            lane=trial.lane,
            seed=trial.seed,
            trial_number=trial.trial_number,
            stage=trial.stage,
            suggestion_index=trial.suggestion_index,
            unique_index=trial.unique_index,
            is_duplicate=trial.is_duplicate,
        )
        audit = _read_json(_trial_audit_path(audit_dir, context))
        candidate = candidates[trial.candidate_id]
        output: dict[str, Any] = {
            "arm": "K0",
            "flow_label": K0_FLOW_LABEL,
            "lane": trial.lane,
            "seed": trial.seed,
            "trial_number": trial.trial_number,
            "stage": trial.stage,
            "suggestion_index": trial.suggestion_index,
            "unique_index": trial.unique_index,
            "candidate_id": trial.candidate_id,
            "is_duplicate": trial.is_duplicate,
            "selection_source": trial.selection_source,
            "objective": trial.objective,
            "metric_valid": trial.metric_valid,
            "eligible": trial.eligible,
            "failure_reason": trial.failure_reason,
            "mean_train_full_final_mae_bpm": audit[
                "mean_train_full_final_mae_bpm"
            ],
        }
        for record_index, record_outcome in enumerate(
            audit["training_outcomes"]
        ):
            metrics = record_outcome.get("formal_metrics", {})
            output[f"train_{record_index}_record_id"] = record_outcome[
                "record_id"
            ]
            output[f"cache_hit_train_{record_index}"] = record_outcome[
                "cache_hit"
            ]
            output[f"cache_key_train_{record_index}"] = record_outcome[
                "cache_key"
            ]
            output[f"physical_solve_train_{record_index}"] = (
                record_outcome["physical_solve_performed"]
            )
            output[f"train_{record_index}_status"] = record_outcome["status"]
            output[f"train_{record_index}_failure_reason"] = (
                record_outcome["failure_reason"]
            )
            for key, value in metrics.items():
                output[f"train_{record_index}_{key}"] = value
        for key in parameter_keys:
            output[f"requested_{key}"] = candidate.requested_params.get(key)
            output[f"actual_{key}"] = candidate.actual_params.get(key)
            output[f"fixed_{key}"] = candidate.fixed_params.get(key)
        rows.append(output)
    _write_csv(path, rows)


def _write_training_metrics(
    path: Path,
    *,
    records: Sequence[K0TrainingRecordRuntime],
    metrics: Sequence[FormalMetricResult],
) -> None:
    _write_csv(
        path,
        [
            {
                "arm": "K0",
                "record_id": record.identity.record_id,
                **asdict(metric),
            }
            for record, metric in zip(records, metrics, strict=True)
        ],
    )
