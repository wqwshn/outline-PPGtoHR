"""Phase2 K1/K3 稳健共享选参折驱动。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .bo_space_generalization import (
    METRIC_CONTRACT_VERSION,
    BOCandidate,
    BOSearchSpace,
    CandidateSolveOutcome,
    ContentAddressedSolverCache,
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
    all_search_rows,
    atomic_write_json,
    cache_summary,
    file_sha256,
    json_ready,
    read_json,
    space_sha256,
    trial_audit_path,
    write_csv,
)
from .phase2_kfold_robust_common import (
    RobustFoldAuditIntegrityError,
    annotate_robust_history,
    evaluate_training_candidate,
    formal_metrics,
    history_row_from_audit,
    terminal_artifact_manifest,
    training_evidence_from_audit,
    validate_terminal_artifact_manifest,
)
from .phase2_kfold_runtime import (
    KFoldRecordInput,
    KFoldRuntime,
    KFoldTrainingRecordRuntime,
    build_default_kfold_runtime,
)
from .phase2_receipt import (
    NeighborhoodEvidence,
    ReplayIdentity,
    SearchBudgetEvidence,
    SelectionEvidence,
    TrainingMetricEvidence,
    freeze_selection,
    load_replay_receipt,
    load_selection_receipt,
    replay_frozen_selection,
)
from .phase2_robust_selection import (
    RobustBands,
    RobustNeighborhoodPlan,
    RobustSelection,
    RobustSelectionError,
    RobustTrainingEvidence,
    build_robust_bands,
    plan_robust_neighborhood,
    select_robust_center,
)
from .types import V2RunConfig

K1_FLOW_LABEL = "完整旧空间稳健选参流程"
K3_FLOW_LABEL = "新物理四维空间稳健选参流程"
_OBJECTIVE_VERSION = "phase2_robust_worst_motion_v1"
_CONSTRAINTS_VERSION = "phase2_nonharm_per_record_v1"


class K1DriverIdentityConflictError(RuntimeError):
    """输出目录中的折级配置身份与当前请求不同。"""


class K3DriverIdentityConflictError(RuntimeError):
    """输出目录中的物理空间折级配置与当前请求不同。"""


K1AuditIntegrityError = RobustFoldAuditIntegrityError
K3AuditIntegrityError = RobustFoldAuditIntegrityError


@dataclass(frozen=True)
class _RobustFoldVariant:
    arm: str
    flow_label: str
    space_name: str
    manifest_name: str
    driver_schema_version: str
    identity_error: type[RuntimeError]


_K1_VARIANT = _RobustFoldVariant(
    arm="K1",
    flow_label=K1_FLOW_LABEL,
    space_name="legacy_full_v1",
    manifest_name="k1_fold_manifest.json",
    driver_schema_version="phase2_k1_driver_identity_v1",
    identity_error=K1DriverIdentityConflictError,
)
_K3_VARIANT = _RobustFoldVariant(
    arm="K3",
    flow_label=K3_FLOW_LABEL,
    space_name="physical_v1",
    manifest_name="k3_fold_manifest.json",
    driver_schema_version="phase2_k3_driver_identity_v1",
    identity_error=K3DriverIdentityConflictError,
)


@dataclass(frozen=True)
class K1FoldConfig:
    output_dir: Path
    scene: str
    fold: int
    git_commit: str
    budget: SeedSearchBudget = SeedSearchBudget(
        lane_unique_budget=40,
        global_unique_budget=120,
        n_startup_trials=10,
        objective_version=_OBJECTIVE_VERSION,
        constraints_version=_CONSTRAINTS_VERSION,
    )
    neighborhood_budget: int = 30
    parallel_lanes: bool = False
    code_dirty: bool = False

    def __post_init__(self) -> None:
        _validate_robust_fold_config(self, arm="K1")


@dataclass(frozen=True)
class K3FoldConfig:
    output_dir: Path
    scene: str
    fold: int
    git_commit: str
    budget: SeedSearchBudget = SeedSearchBudget(
        lane_unique_budget=40,
        global_unique_budget=120,
        n_startup_trials=10,
        objective_version=_OBJECTIVE_VERSION,
        constraints_version=_CONSTRAINTS_VERSION,
    )
    neighborhood_budget: int = 30
    parallel_lanes: bool = False
    code_dirty: bool = False

    def __post_init__(self) -> None:
        _validate_robust_fold_config(self, arm="K3")


def _validate_robust_fold_config(
    config: K1FoldConfig | K3FoldConfig,
    *,
    arm: str,
) -> None:
    if not config.scene or not config.git_commit:
        raise ValueError(f"{arm} 必须冻结 scene 和 git_commit")
    if type(config.fold) is not int or config.fold < 0:
        raise ValueError("fold 必须是非负整数")
    if config.budget.lane_seeds != (42, 43, 44):
        raise ValueError(f"{arm} 固定使用 seed 42/43/44")
    if config.budget.objective_version != _OBJECTIVE_VERSION:
        raise ValueError(f"{arm} objective_version 不匹配")
    if config.budget.constraints_version != _CONSTRAINTS_VERSION:
        raise ValueError(f"{arm} constraints_version 不匹配")
    if (
        type(config.neighborhood_budget) is not int
        or not 0 <= config.neighborhood_budget <= 30
    ):
        raise ValueError(f"{arm} 邻域预算必须位于 [0, 30]")


@dataclass(frozen=True)
class K1FoldResult:
    arm: str
    flow_label: str
    selected_candidate_id: str
    selected_worst_train_mae_bpm: float
    selected_mean_train_mae_bpm: float
    candidate_history: Path
    selected_params: Path
    training_metrics: Path
    neighborhood_evidence: Path
    selection_receipt: Path
    replay_receipt: Path
    replay_status: str
    training_plots: tuple[Path, Path]
    cache_summary: Path
    failure_classification: Path
    manifest: Path
    neighborhood_candidate_count: int
    search_result: SeedSearchResult


K3FoldResult = K1FoldResult


def build_k1_default_runtime(
    *,
    base_config: V2RunConfig,
    training_records: tuple[
        KFoldRecordInput,
        KFoldRecordInput,
    ],
    heldout_record: KFoldRecordInput,
    output_dir: Path | str,
) -> KFoldRuntime:
    return build_default_kfold_runtime(
        arm="K1",
        base_config=base_config,
        training_records=training_records,
        heldout_record=heldout_record,
        output_dir=output_dir,
    )


def build_k3_default_runtime(
    *,
    base_config: V2RunConfig,
    training_records: tuple[
        KFoldRecordInput,
        KFoldRecordInput,
    ],
    heldout_record: KFoldRecordInput,
    output_dir: Path | str,
) -> KFoldRuntime:
    return build_default_kfold_runtime(
        arm="K3",
        base_config=base_config,
        training_records=training_records,
        heldout_record=heldout_record,
        output_dir=output_dir,
    )


def run_k1_fold_study(
    config: K1FoldConfig,
    *,
    runtime: KFoldRuntime,
) -> K1FoldResult:
    """运行 K1：稳健目标、逐记录约束、完整邻域、冻结回放。"""

    return _run_robust_fold_study(
        config,
        runtime=runtime,
        variant=_K1_VARIANT,
    )


def run_k3_fold_study(
    config: K3FoldConfig,
    *,
    runtime: KFoldRuntime,
) -> K3FoldResult:
    """运行 K3：物理四维空间上的稳健共享选参和冻结回放。"""

    return _run_robust_fold_study(
        config,
        runtime=runtime,
        variant=_K3_VARIANT,
    )


def _run_robust_fold_study(
    config: K1FoldConfig | K3FoldConfig,
    *,
    runtime: KFoldRuntime,
    variant: _RobustFoldVariant,
) -> K1FoldResult:
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    space = build_bo_search_space(variant.space_name)
    candidates = {
        candidate.candidate_id: candidate
        for candidate in space.candidates
    }
    driver_identity_path = output / "driver_identity.json"
    driver_identity_hash = _freeze_driver_identity(
        driver_identity_path,
        config=config,
        runtime=runtime,
        space=space,
        variant=variant,
    )
    cache = ContentAddressedSolverCache(output / "cache")
    trial_audit_dir = output / "trial_audit"

    def cache_identity(
        record: KFoldTrainingRecordRuntime,
        candidate: BOCandidate,
    ) -> SolverCacheIdentity:
        return SolverCacheIdentity(
            data_sha256=record.identity.data_sha256,
            reference_sha256=record.identity.reference_sha256,
            git_commit=config.git_commit,
            run_config={
                **json_ready(record.run_config),
                "arm": variant.arm,
                "scene": config.scene,
                "fold": config.fold,
                "space_name": space.name,
                "objective_version": _OBJECTIVE_VERSION,
                "constraints_version": _CONSTRAINTS_VERSION,
            },
            candidate=candidate,
            reference_groups_order=("HF",),
        )

    def evaluate_candidate(
        candidate: BOCandidate,
        *,
        logical_reference: Mapping[str, Any],
        audit_path: Path,
    ) -> tuple[
        RobustTrainingEvidence,
        tuple[CandidateSolveOutcome, CandidateSolveOutcome],
    ]:
        return evaluate_training_candidate(
            cache=cache,
            candidate=candidate,
            training_records=runtime.training_records,
            cache_identity=cache_identity,
            arm=variant.arm,
            scene=config.scene,
            fold=config.fold,
            logical_reference=logical_reference,
            audit_path=audit_path,
        )

    def evaluate_search(
        candidate: BOCandidate,
        context: SearchRequestContext,
    ) -> SearchEvaluation:
        evidence, _ = evaluate_candidate(
            candidate,
            logical_reference=asdict(context),
            audit_path=trial_audit_path(
                trial_audit_dir,
                context,
            ),
        )
        return SearchEvaluation(
            objective=evidence.objective_bpm,
            constraints=evidence.constraints_bpm,
            metric_valid=evidence.metric_valid,
            eligible=evidence.eligible,
            failure_reason=evidence.failure_reason,
        )

    experiment_identity = SearchExperimentIdentity(
        input_sha256s=tuple(
            record.identity.data_sha256
            for record in runtime.training_records
        ),
        reference_sha256s=tuple(
            record.identity.reference_sha256
            for record in runtime.training_records
        ),
        git_commit=config.git_commit,
        run_config={
            "arm": variant.arm,
            "flow_label": variant.flow_label,
            "scene": config.scene,
            "fold": config.fold,
            "space_name": space.name,
            "driver_identity_sha256": driver_identity_hash,
            "neighborhood_budget": config.neighborhood_budget,
            "training_records": [
                {
                    "identity": asdict(record.identity),
                    "run_config": json_ready(record.run_config),
                }
                for record in runtime.training_records
            ],
            "heldout_record_identity": asdict(
                runtime.heldout_record
            ),
        },
        evaluation_version=_OBJECTIVE_VERSION,
    )
    search_result = run_seed_search(
        space=space,
        output_dir=output / "search",
        experiment_identity=experiment_identity,
        evaluate=evaluate_search,
        budget=config.budget,
        parallel_lanes=config.parallel_lanes,
    )
    completed = _load_completed_result(
        output,
        search_result=search_result,
        expected_driver_identity_sha256=driver_identity_hash,
        variant=variant,
    )
    if completed is not None:
        return completed

    search_evidence: dict[str, RobustTrainingEvidence] = {}
    for index, candidate_id in enumerate(
        search_result.global_candidate_ids
    ):
        audit_path = (
            trial_audit_dir
            / f"band-evidence-{index:03d}.json"
        )
        if audit_path.is_file():
            evidence = training_evidence_from_audit(
                read_json(audit_path),
                expected_candidate=candidates[candidate_id],
                expected_stage="band_evidence",
                expected_index_name="band_index",
                expected_index=index,
                expected_record_ids=tuple(
                    record.identity.record_id
                    for record in runtime.training_records
                ),
                arm=variant.arm,
            )
        else:
            evidence, _ = evaluate_candidate(
                candidates[candidate_id],
                logical_reference={
                    "stage": "band_evidence",
                    "band_index": index,
                    "candidate_id": candidate_id,
                },
                audit_path=audit_path,
            )
        search_evidence[candidate_id] = evidence

    history_path = output / "candidate_history.csv"
    try:
        bands = build_robust_bands(search_evidence.values())
    except RobustSelectionError as exc:
        _write_candidate_history(
            history_path,
            scene=config.scene,
            fold=config.fold,
            search_result=search_result,
            candidates=candidates,
            trial_audit_dir=trial_audit_dir,
            neighborhood_rows=(),
            space=space,
            evidence_by_candidate_id=search_evidence,
            variant=variant,
        )
        _fail_closed(
            output,
            reason=exc.reason,
            search_result=search_result,
            candidate_history=history_path,
            variant=variant,
        )
        raise RuntimeError(exc.reason) from exc

    plan = plan_robust_neighborhood(
        space=space,
        bands=bands,
        reviewed_candidate_ids=frozenset(search_evidence),
        max_new_candidates=config.neighborhood_budget,
    )
    all_evidence = dict(search_evidence)
    neighborhood_rows: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(
        plan.candidate_ids_to_evaluate
    ):
        audit_path = (
            trial_audit_dir
            / f"neighborhood-{index:03d}.json"
        )
        if audit_path.is_file():
            evidence = training_evidence_from_audit(
                read_json(audit_path),
                expected_candidate=candidates[candidate_id],
                expected_stage="neighborhood",
                expected_index_name="neighborhood_index",
                expected_index=index,
                expected_record_ids=tuple(
                    record.identity.record_id
                    for record in runtime.training_records
                ),
                arm=variant.arm,
            )
        else:
            evidence, _ = evaluate_candidate(
                candidates[candidate_id],
                logical_reference={
                    "stage": "neighborhood",
                    "neighborhood_index": index,
                    "candidate_id": candidate_id,
                },
                audit_path=audit_path,
            )
        all_evidence[candidate_id] = evidence
        neighborhood_rows.append(
            history_row_from_audit(
                arm=variant.arm,
                scene=config.scene,
                fold=config.fold,
                stage="neighborhood",
                lane="enumeration",
                seed=-1,
                trial_number=index,
                suggestion_index=index + 1,
                unique_index=index + 1,
                candidate=candidates[candidate_id],
                audit=read_json(audit_path),
                is_duplicate=False,
            )
        )
    try:
        selection = select_robust_center(
            space=space,
            bands=bands,
            plan=plan,
            evidence_by_candidate_id=all_evidence,
        )
    except RobustSelectionError as exc:
        _write_candidate_history(
            history_path,
            scene=config.scene,
            fold=config.fold,
            search_result=search_result,
            candidates=candidates,
            trial_audit_dir=trial_audit_dir,
            neighborhood_rows=neighborhood_rows,
            space=space,
            bands=bands,
            evidence_by_candidate_id=all_evidence,
            variant=variant,
        )
        _fail_closed(
            output,
            reason=exc.reason,
            search_result=search_result,
            candidate_history=history_path,
            plan=plan,
            variant=variant,
        )
        raise RuntimeError(exc.reason) from exc

    selected_candidate = candidates[selection.candidate_id]
    selected_evidence, selected_outcomes = evaluate_candidate(
        selected_candidate,
        logical_reference={
            "stage": "final_selection",
            "candidate_id": selection.candidate_id,
        },
        audit_path=trial_audit_dir / "final-selection.json",
    )
    selected_metrics = formal_metrics(
        selected_outcomes,
        arm=variant.arm,
    )
    selected_diagnostics = (
        {
            "training_records": [
                {
                    "record_id": record.identity.record_id,
                    **dict(outcome.diagnostics),
                }
                for record, outcome in zip(
                    runtime.training_records,
                    selected_outcomes,
                    strict=True,
                )
            ]
        }
        if variant.arm == "K3"
        else {}
    )
    _write_candidate_history(
        history_path,
        scene=config.scene,
        fold=config.fold,
        search_result=search_result,
        candidates=candidates,
        trial_audit_dir=trial_audit_dir,
        neighborhood_rows=neighborhood_rows,
        space=space,
        bands=bands,
        evidence_by_candidate_id=all_evidence,
        selection=selection,
        variant=variant,
    )
    selected_center = next(
        center
        for center in selection.center_evidence
        if center.candidate_id == selection.candidate_id
    )
    neighborhood_path = output / "neighborhood_evidence.json"
    atomic_write_json(
        neighborhood_path,
        {
            "arm": variant.arm,
            "bands": asdict(bands),
            "plan": asdict(plan),
            "selected_candidate_id": selection.candidate_id,
            "primary_center_evidence": [
                asdict(center)
                for center in selection.primary_center_evidence
            ],
            "diagnostic_center_evidence": [
                asdict(center)
                for center
                in selection.diagnostic_center_evidence
            ],
            "center_evidence": [
                asdict(center)
                for center in selection.center_evidence
            ],
        },
    )
    params_path = output / "params.json"
    params_payload = {
        "arm": variant.arm,
        "flow_label": variant.flow_label,
        "candidate_id": selected_candidate.candidate_id,
        "requested_params": selected_candidate.requested_params,
        "actual_params": selected_candidate.actual_params,
        "fixed_params": selected_candidate.fixed_params,
        "worst_train_mae_bpm": (
            selected_evidence.worst_train_mae_bpm
        ),
        "mean_train_mae_bpm": (
            selected_evidence.mean_train_mae_bpm
        ),
    }
    if selected_diagnostics:
        params_payload["selected_diagnostics"] = selected_diagnostics
    atomic_write_json(params_path, params_payload)
    training_metrics_path = output / "training_metrics.csv"
    write_csv(
        training_metrics_path,
        [
            {
                "arm": variant.arm,
                "record_id": record.identity.record_id,
                **asdict(metric),
            }
            for record, metric in zip(
                runtime.training_records,
                selected_metrics,
                strict=True,
            )
        ],
    )
    search_identity = read_json(
        output / "search" / "search_identity.json"
    )
    config_hash = str(search_identity["config_hash"])
    selection_receipt_path = output / "selection_receipt.json"
    selection_receipt = freeze_selection(
        selection_receipt_path,
        SelectionEvidence(
            experiment_name=(
                f"{config.scene}-fold-{config.fold}-"
                f"{variant.arm.lower()}"
            ),
            arm=variant.arm,
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
            space_sha256=space_sha256(space.candidates),
            metric_contract_version=METRIC_CONTRACT_VERSION,
            study_identities=(
                f"seed_42:{config_hash}",
                f"seed_43:{config_hash}",
                f"seed_44:{config_hash}",
                f"fill:{config_hash}|neighborhood:"
                f"{file_sha256(neighborhood_path)}",
            ),
            budget=SearchBudgetEvidence(
                lane_unique_budget=(
                    config.budget.lane_unique_budget
                ),
                requested_global_unique_budget=(
                    config.budget.global_unique_budget
                ),
                actual_global_unique_count=len(
                    search_result.global_candidate_ids
                ),
                requested_neighborhood_budget=(
                    config.neighborhood_budget
                ),
                actual_neighborhood_count=len(
                    plan.candidate_ids_to_evaluate
                ),
            ),
            selected_candidate_id=selected_candidate.candidate_id,
            selected_requested_params=(
                selected_candidate.requested_params
            ),
            selected_actual_params=selected_candidate.actual_params,
            selected_fixed_params=selected_candidate.fixed_params,
            training_metrics=TrainingMetricEvidence(
                eligible=True,
                common_window_counts=tuple(
                    metric.base_motion_common_finite_count
                    for metric in selected_metrics
                ),
                common_window_sha256s=tuple(
                    metric.base_motion_window_sha256
                    for metric in selected_metrics
                ),
                worst_train_mae_bpm=(
                    selected_evidence.worst_train_mae_bpm
                ),
                mean_train_mae_bpm=(
                    selected_evidence.mean_train_mae_bpm
                ),
                nonharm_deltas_bpm=tuple(
                    metric.reliable_motion_final_mae_bpm
                    - metric.reliable_motion_reset_fft_mae_bpm
                    for metric in selected_metrics
                ),
            ),
            neighborhood_evidence=NeighborhoodEvidence(
                status="complete",
                reviewed_neighbor_count=(
                    selected_center.reviewed_neighbor_count
                ),
                support_ratio=selected_center.support_ratio,
                has_cliff=selected_center.has_cliff,
                truncated_center_count=len(
                    plan.truncated_primary_center_ids
                ),
            ),
            candidate_history_sha256=file_sha256(history_path),
            evidence_level="development_reuse_pilot",
            selected_diagnostics=selected_diagnostics,
        ),
    )
    training_plots = tuple(
        record.render_selected(
            selected_candidate,
            outcome,
            output
            / "training"
            / record.identity.record_id
            / "png",
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
    cache_summary_payload = cache_summary(cache)
    cache_summary_path = output / "cache_summary.json"
    atomic_write_json(cache_summary_path, cache_summary_payload)
    failure_classification_path = (
        output / "failure_classification.json"
    )
    atomic_write_json(
        failure_classification_path,
        {
            "failure_reason": "",
            "invalid_candidate_count": sum(
                not evidence.metric_valid
                for evidence in all_evidence.values()
            ),
            "unsafe_candidate_count": sum(
                evidence.metric_valid and not evidence.eligible
                for evidence in all_evidence.values()
            ),
            "truncated_primary_center_count": len(
                plan.truncated_primary_center_ids
            ),
            "replay_status": replay_receipt.status,
            "replay_failure_reason": (
                replay_receipt.failure_reason
            ),
            "infrastructure_failure_count": (
                cache_summary_payload[
                    "infrastructure_failure_count"
                ]
            ),
        },
    )
    manifest_path = output / variant.manifest_name
    atomic_write_json(
        manifest_path,
        {
            "arm": variant.arm,
            "flow_label": variant.flow_label,
            "causal_claim_allowed": False,
            "comparison_scope": "operational_workflow_only",
            "space_candidate_count": len(space.candidates),
            "global_search_candidate_count": len(
                search_result.global_candidate_ids
            ),
            "neighborhood_candidate_count": len(
                plan.candidate_ids_to_evaluate
            ),
            "reviewed_unique_candidate_count": len(all_evidence),
            "coverage_ratio": (
                len(all_evidence) / len(space.candidates)
            ),
            **(
                {
                    "k2_k3_comparison_context": {
                        "k2_space_candidate_count": 108,
                        "k2_max_reviewed_candidate_count": 108,
                        "k2_max_coverage_ratio": 1.0,
                        "k2_neighborhood_geometry": (
                            "all_direct_neighbors_already_enumerated"
                        ),
                        "k3_space_candidate_count": 300,
                        "k3_max_global_search_candidate_count": 120,
                        "k3_max_neighborhood_candidate_count": 30,
                        "k3_max_reviewed_candidate_count": 150,
                        "k3_max_coverage_ratio": 0.5,
                        "k3_neighborhood_geometry": (
                            "budgeted_direct_neighbors_primary_band_first_"
                            "then_diagnostic_band_if_budget_remains"
                        ),
                        "single_factor_causal_attribution_allowed": False,
                    }
                }
                if variant.arm == "K3"
                else {}
            ),
            "scene": config.scene,
            "fold": config.fold,
            "git_commit": config.git_commit,
            "driver_identity_sha256": driver_identity_hash,
            "selected_candidate_id": (
                selected_candidate.candidate_id
            ),
            "selection_hash": selection_receipt.selection_hash,
            "candidate_history": str(history_path),
            "neighborhood_evidence": str(neighborhood_path),
            "selection_receipt": str(selection_receipt_path),
            "replay_receipt": str(replay_receipt_path),
            "training_plots": training_plots,
            "artifacts": terminal_artifact_manifest(
                required={
                    "candidate_history": history_path,
                    "selected_params": params_path,
                    "training_metrics": training_metrics_path,
                    "neighborhood_evidence": neighborhood_path,
                    "selection_receipt": selection_receipt_path,
                    "replay_receipt": replay_receipt_path,
                    "cache_summary": cache_summary_path,
                    "failure_classification": (
                        failure_classification_path
                    ),
                },
                training_plots=training_plots,
            ),
            "evidence_level": "development_reuse_pilot",
            "confirmatory_claim_allowed": False,
        },
    )
    return K1FoldResult(
        arm=variant.arm,
        flow_label=variant.flow_label,
        selected_candidate_id=selected_candidate.candidate_id,
        selected_worst_train_mae_bpm=(
            selected_evidence.worst_train_mae_bpm
        ),
        selected_mean_train_mae_bpm=(
            selected_evidence.mean_train_mae_bpm
        ),
        candidate_history=history_path,
        selected_params=params_path,
        training_metrics=training_metrics_path,
        neighborhood_evidence=neighborhood_path,
        selection_receipt=selection_receipt_path,
        replay_receipt=replay_receipt_path,
        replay_status=replay_receipt.status,
        training_plots=training_plots,
        cache_summary=cache_summary_path,
        failure_classification=failure_classification_path,
        manifest=manifest_path,
        neighborhood_candidate_count=len(
            plan.candidate_ids_to_evaluate
        ),
        search_result=search_result,
    )


def _freeze_driver_identity(
    path: Path,
    *,
    config: K1FoldConfig | K3FoldConfig,
    runtime: KFoldRuntime,
    space: BOSearchSpace,
    variant: _RobustFoldVariant,
) -> str:
    identity = {
        "arm": variant.arm,
        "scene": config.scene,
        "fold": config.fold,
        "git_commit": config.git_commit,
        "code_dirty": config.code_dirty,
        "space_name": space.name,
        "space_sha256": space_sha256(space.candidates),
        "budget": asdict(config.budget),
        "neighborhood_budget": config.neighborhood_budget,
        "parallel_lanes": config.parallel_lanes,
        "objective_version": _OBJECTIVE_VERSION,
        "constraints_version": _CONSTRAINTS_VERSION,
        "robust_rule_version": "phase2_robust_selection_v1",
        "training_records": [
            asdict(record.identity)
            for record in runtime.training_records
        ],
        "heldout_record": asdict(runtime.heldout_record),
    }
    identity_sha256 = _mapping_sha256(identity)
    payload = {
        "schema_version": variant.driver_schema_version,
        "identity_sha256": identity_sha256,
        "identity": identity,
    }
    if path.is_file():
        existing = read_json(path)
        if existing != json_ready(payload):
            raise variant.identity_error(
                f"{variant.arm} 输出目录已绑定其他折级配置: {path}"
            )
        return identity_sha256
    atomic_write_json(path, payload)
    return identity_sha256


def _load_completed_result(
    output: Path,
    *,
    search_result: SeedSearchResult,
    expected_driver_identity_sha256: str,
    variant: _RobustFoldVariant,
) -> K1FoldResult | None:
    selection_path = output / "selection_receipt.json"
    replay_path = output / "replay_receipt.json"
    manifest_path = output / variant.manifest_name
    if not (
        selection_path.is_file()
        and replay_path.is_file()
        and manifest_path.is_file()
    ):
        return None
    required = {
        "candidate_history": output / "candidate_history.csv",
        "selected_params": output / "params.json",
        "training_metrics": output / "training_metrics.csv",
        "neighborhood_evidence": (
            output / "neighborhood_evidence.json"
        ),
        "cache_summary": output / "cache_summary.json",
        "failure_classification": (
            output / "failure_classification.json"
        ),
    }
    missing = [
        name
        for name, path in required.items()
        if not path.is_file()
    ]
    if missing:
        raise RobustFoldAuditIntegrityError(
            f"{variant.arm} 终态产物不完整: " + ", ".join(missing)
        )
    selection = load_selection_receipt(selection_path)
    replay = load_replay_receipt(replay_path)
    manifest = read_json(manifest_path)
    params = read_json(required["selected_params"])
    neighborhood = read_json(
        required["neighborhood_evidence"]
    )
    if (
        manifest.get("driver_identity_sha256")
        != expected_driver_identity_sha256
        or manifest.get("selection_hash")
        != selection.selection_hash
        or replay.selection_hash != selection.selection_hash
    ):
        raise RobustFoldAuditIntegrityError(
            f"{variant.arm} 终态回执、manifest 或 driver 身份不一致"
        )
    if (
        selection.evidence.candidate_history_sha256
        != file_sha256(required["candidate_history"])
    ):
        raise RobustFoldAuditIntegrityError(
            f"{variant.arm} 终态 candidate_history 哈希不匹配"
        )
    neighborhood_identity = (
        "neighborhood:"
        f"{file_sha256(required['neighborhood_evidence'])}"
    )
    if not any(
        neighborhood_identity in identity
        for identity in selection.evidence.study_identities
    ):
        raise RobustFoldAuditIntegrityError(
            f"{variant.arm} 选择回执未绑定完整邻域证据哈希"
        )
    selected_candidate_id = str(
        params.get("candidate_id", "")
    )
    if (
        selected_candidate_id
        != selection.evidence.selected_candidate_id
    ):
        raise RobustFoldAuditIntegrityError(
            f"{variant.arm} params 与选择回执候选不一致"
        )
    expected_params = {
        "arm": variant.arm,
        "flow_label": variant.flow_label,
        "candidate_id": selection.evidence.selected_candidate_id,
        "requested_params": (
            selection.evidence.selected_requested_params
        ),
        "actual_params": selection.evidence.selected_actual_params,
        "fixed_params": selection.evidence.selected_fixed_params,
        "worst_train_mae_bpm": (
            selection.evidence.training_metrics.worst_train_mae_bpm
        ),
        "mean_train_mae_bpm": (
            selection.evidence.training_metrics.mean_train_mae_bpm
        ),
    }
    if selection.evidence.selected_diagnostics:
        expected_params["selected_diagnostics"] = (
            selection.evidence.selected_diagnostics
        )
    if params != json_ready(expected_params):
        raise RobustFoldAuditIntegrityError(
            f"{variant.arm} params 参数或训练汇总与选择回执不一致"
        )
    if (
        manifest.get("selected_candidate_id")
        != selection.evidence.selected_candidate_id
    ):
        raise RobustFoldAuditIntegrityError(
            f"{variant.arm} manifest 与选择回执候选不一致"
        )
    plot_values = manifest.get("training_plots")
    if (
        not isinstance(plot_values, list)
        or len(plot_values) != 2
    ):
        raise RobustFoldAuditIntegrityError(
            f"{variant.arm} manifest 缺少两张训练经典图"
        )
    training_plots = (
        Path(str(plot_values[0])),
        Path(str(plot_values[1])),
    )
    if any(not path.is_file() for path in training_plots):
        raise RobustFoldAuditIntegrityError(
            f"{variant.arm} 训练经典图不存在"
        )
    validate_terminal_artifact_manifest(
        manifest,
        required={
            **required,
            "selection_receipt": selection_path,
            "replay_receipt": replay_path,
        },
        training_plots=training_plots,
        arm=variant.arm,
    )
    plan = neighborhood.get("plan")
    if not isinstance(plan, Mapping):
        raise RobustFoldAuditIntegrityError(
            f"{variant.arm} 邻域证据缺少 plan"
        )
    neighborhood_candidates = plan.get(
        "candidate_ids_to_evaluate"
    )
    if not isinstance(neighborhood_candidates, list):
        raise RobustFoldAuditIntegrityError(
            f"{variant.arm} 邻域候选列表无效"
        )
    if replay.status == "infrastructure_failed":
        return None
    return K1FoldResult(
        arm=variant.arm,
        flow_label=variant.flow_label,
        selected_candidate_id=selected_candidate_id,
        selected_worst_train_mae_bpm=float(
            params["worst_train_mae_bpm"]
        ),
        selected_mean_train_mae_bpm=float(
            params["mean_train_mae_bpm"]
        ),
        candidate_history=required["candidate_history"],
        selected_params=required["selected_params"],
        training_metrics=required["training_metrics"],
        neighborhood_evidence=required[
            "neighborhood_evidence"
        ],
        selection_receipt=selection_path,
        replay_receipt=replay_path,
        replay_status=replay.status,
        training_plots=training_plots,
        cache_summary=required["cache_summary"],
        failure_classification=required[
            "failure_classification"
        ],
        manifest=manifest_path,
        neighborhood_candidate_count=len(
            neighborhood_candidates
        ),
        search_result=search_result,
    )


def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            json_ready(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()



def _write_candidate_history(
    path: Path,
    *,
    scene: str,
    fold: int,
    search_result: SeedSearchResult,
    candidates: Mapping[str, BOCandidate],
    trial_audit_dir: Path,
    neighborhood_rows: Sequence[Mapping[str, Any]],
    space: BOSearchSpace,
    bands: RobustBands | None = None,
    evidence_by_candidate_id: Mapping[
        str,
        RobustTrainingEvidence,
    ] | None = None,
    selection: RobustSelection | None = None,
    variant: _RobustFoldVariant,
) -> None:
    rows: list[dict[str, Any]] = []
    for trial in all_search_rows(search_result):
        context = SearchRequestContext(
            lane=trial.lane,
            seed=trial.seed,
            trial_number=trial.trial_number,
            stage=trial.stage,
            suggestion_index=trial.suggestion_index,
            unique_index=trial.unique_index,
            is_duplicate=trial.is_duplicate,
        )
        rows.append(
            history_row_from_audit(
                arm=variant.arm,
                scene=scene,
                fold=fold,
                stage=trial.stage,
                lane=trial.lane,
                seed=trial.seed,
                trial_number=trial.trial_number,
                suggestion_index=trial.suggestion_index,
                unique_index=trial.unique_index,
                candidate=candidates[trial.candidate_id],
                audit=read_json(
                    trial_audit_path(
                        trial_audit_dir,
                        context,
                    )
                ),
                is_duplicate=trial.is_duplicate,
            )
        )
    rows.extend(dict(row) for row in neighborhood_rows)
    annotate_robust_history(
        rows,
        space=space,
        bands=bands,
        evidence_by_candidate_id=(
            evidence_by_candidate_id or {}
        ),
        selection=selection,
    )
    write_csv(path, rows)



def _fail_closed(
    output: Path,
    *,
    reason: str,
    search_result: SeedSearchResult,
    candidate_history: Path,
    plan: RobustNeighborhoodPlan | None = None,
    variant: _RobustFoldVariant,
) -> None:
    atomic_write_json(
        output / "failure_classification.json",
        {
            "failure_reason": reason,
            "replay_status": "not_started",
            "global_search_candidate_count": len(
                search_result.global_candidate_ids
            ),
            "neighborhood_candidate_count": (
                len(plan.candidate_ids_to_evaluate)
                if plan is not None
                else 0
            ),
            "truncated_primary_center_count": (
                len(plan.truncated_primary_center_ids)
                if plan is not None
                else 0
            ),
        },
    )
    atomic_write_json(
        output / variant.manifest_name,
        {
            "arm": variant.arm,
            "flow_label": variant.flow_label,
            "status": "failed_closed",
            "failure_reason": reason,
            "candidate_history": str(candidate_history),
            "selection_receipt": "",
            "replay_receipt": "",
        },
    )
