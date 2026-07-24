"""Phase2 K1 完整旧空间稳健共享选参折驱动。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .bo_space_generalization import (
    METRIC_CONTRACT_VERSION,
    BOCandidate,
    BOSearchSpace,
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
    build_robust_training_evidence,
    direct_neighbor_ids,
    plan_robust_neighborhood,
    select_robust_center,
)
from .types import V2RunConfig

K1_FLOW_LABEL = "完整旧空间稳健选参流程"
_OBJECTIVE_VERSION = "phase2_robust_worst_motion_v1"
_CONSTRAINTS_VERSION = "phase2_nonharm_per_record_v1"


class K1DriverIdentityConflictError(RuntimeError):
    """输出目录中的折级配置身份与当前请求不同。"""


class K1AuditIntegrityError(RuntimeError):
    """不可变 K1 审计文件身份或内容不匹配。"""


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
        if not self.scene or not self.git_commit:
            raise ValueError("K1 必须冻结 scene 和 git_commit")
        if type(self.fold) is not int or self.fold < 0:
            raise ValueError("fold 必须是非负整数")
        if self.budget.lane_seeds != (42, 43, 44):
            raise ValueError("K1 固定使用 seed 42/43/44")
        if self.budget.objective_version != _OBJECTIVE_VERSION:
            raise ValueError("K1 objective_version 不匹配")
        if self.budget.constraints_version != _CONSTRAINTS_VERSION:
            raise ValueError("K1 constraints_version 不匹配")
        if (
            type(self.neighborhood_budget) is not int
            or not 0 <= self.neighborhood_budget <= 30
        ):
            raise ValueError("K1 邻域预算必须位于 [0, 30]")


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


def run_k1_fold_study(
    config: K1FoldConfig,
    *,
    runtime: KFoldRuntime,
) -> K1FoldResult:
    """运行 K1：稳健目标、逐记录约束、完整邻域、冻结回放。"""

    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    space = build_bo_search_space("legacy_full_v1")
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
                "arm": "K1",
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
        started_at = perf_counter()
        outcomes: list[CandidateSolveOutcome] = []
        audit_outcomes: list[dict[str, Any]] = []
        for record_index, record in enumerate(
            runtime.training_records
        ):
            lookup = cache.get_or_solve(
                cache_identity(record, candidate),
                lambda record=record, candidate=candidate: (
                    record.solve_candidate(candidate)
                ),
                logical_reference={
                    "arm": "K1",
                    "scene": config.scene,
                    "fold": config.fold,
                    "record_id": record.identity.record_id,
                    "record_index": record_index,
                    **logical_reference,
                },
            )
            outcomes.append(lookup.outcome)
            audit_outcomes.append(
                {
                    "record_id": record.identity.record_id,
                    "cache_key": lookup.cache_key,
                    "cache_hit": lookup.cache_hit,
                    "physical_solve_performed": (
                        lookup.physical_solve_performed
                    ),
                    "status": lookup.outcome.status,
                    "failure_reason": (
                        lookup.outcome.failure_reason
                    ),
                    "formal_metrics": (
                        asdict(lookup.outcome.formal_metrics)
                        if lookup.outcome.formal_metrics is not None
                        else {}
                    ),
                }
            )
        typed_outcomes = (outcomes[0], outcomes[1])
        if any(
            outcome.status != "valid"
            or outcome.formal_metrics is None
            for outcome in typed_outcomes
        ):
            reason = next(
                (
                    outcome.failure_reason
                    for outcome in typed_outcomes
                    if outcome.status != "valid"
                ),
                "metric_window_contract_failed",
            )
            evidence = build_robust_training_evidence(
                candidate_id=candidate.candidate_id,
                final_motion_mae_bpm=None,
                reset_motion_mae_bpm=None,
                failure_reason=reason,
            )
        else:
            metrics = (
                typed_outcomes[0].formal_metrics,
                typed_outcomes[1].formal_metrics,
            )
            if metrics[0] is None or metrics[1] is None:
                raise AssertionError("有效 K1 outcome 缺少正式指标")
            evidence = build_robust_training_evidence(
                candidate_id=candidate.candidate_id,
                final_motion_mae_bpm=(
                    metrics[0].reliable_motion_final_mae_bpm,
                    metrics[1].reliable_motion_final_mae_bpm,
                ),
                reset_motion_mae_bpm=(
                    metrics[0].reliable_motion_reset_fft_mae_bpm,
                    metrics[1].reliable_motion_reset_fft_mae_bpm,
                ),
            )
        atomic_write_json(
            audit_path,
            {
                **dict(logical_reference),
                "candidate_id": candidate.candidate_id,
                "candidate_identity": {
                    "requested_params": candidate.requested_params,
                    "actual_params": candidate.actual_params,
                    "fixed_params": candidate.fixed_params,
                },
                "training_outcomes": audit_outcomes,
                "robust_evidence": asdict(evidence),
                "runtime_seconds": perf_counter() - started_at,
            },
        )
        return evidence, typed_outcomes

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
            "arm": "K1",
            "flow_label": K1_FLOW_LABEL,
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
            evidence = _training_evidence_from_audit(
                read_json(audit_path),
                expected_candidate=candidates[candidate_id],
                expected_stage="band_evidence",
                expected_index_name="band_index",
                expected_index=index,
                expected_record_ids=tuple(
                    record.identity.record_id
                    for record in runtime.training_records
                ),
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
        )
        _fail_closed(
            output,
            reason=exc.reason,
            search_result=search_result,
            candidate_history=history_path,
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
            evidence = _training_evidence_from_audit(
                read_json(audit_path),
                expected_candidate=candidates[candidate_id],
                expected_stage="neighborhood",
                expected_index_name="neighborhood_index",
                expected_index=index,
                expected_record_ids=tuple(
                    record.identity.record_id
                    for record in runtime.training_records
                ),
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
            _history_row_from_audit(
                arm="K1",
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
        )
        _fail_closed(
            output,
            reason=exc.reason,
            search_result=search_result,
            candidate_history=history_path,
            plan=plan,
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
    selected_metrics = _formal_metrics(selected_outcomes)
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
            "arm": "K1",
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
    atomic_write_json(
        params_path,
        {
            "arm": "K1",
            "flow_label": K1_FLOW_LABEL,
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
        },
    )
    training_metrics_path = output / "training_metrics.csv"
    write_csv(
        training_metrics_path,
        [
            {
                "arm": "K1",
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
                f"{config.scene}-fold-{config.fold}-k1"
            ),
            arm="K1",
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
    manifest_path = output / "k1_fold_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "arm": "K1",
            "flow_label": K1_FLOW_LABEL,
            "causal_claim_allowed": False,
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
            "evidence_level": "development_reuse_pilot",
            "confirmatory_claim_allowed": False,
        },
    )
    return K1FoldResult(
        arm="K1",
        flow_label=K1_FLOW_LABEL,
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
    config: K1FoldConfig,
    runtime: KFoldRuntime,
    space: BOSearchSpace,
) -> str:
    identity = {
        "arm": "K1",
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
        "schema_version": "phase2_k1_driver_identity_v1",
        "identity_sha256": identity_sha256,
        "identity": identity,
    }
    if path.is_file():
        existing = read_json(path)
        if existing != json_ready(payload):
            raise K1DriverIdentityConflictError(
                f"K1 输出目录已绑定其他折级配置: {path}"
            )
        return identity_sha256
    atomic_write_json(path, payload)
    return identity_sha256


def _load_completed_result(
    output: Path,
    *,
    search_result: SeedSearchResult,
    expected_driver_identity_sha256: str,
) -> K1FoldResult | None:
    selection_path = output / "selection_receipt.json"
    replay_path = output / "replay_receipt.json"
    manifest_path = output / "k1_fold_manifest.json"
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
        raise K1AuditIntegrityError(
            "K1 终态产物不完整: " + ", ".join(missing)
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
        raise K1AuditIntegrityError(
            "K1 终态回执、manifest 或 driver 身份不一致"
        )
    if (
        selection.evidence.candidate_history_sha256
        != file_sha256(required["candidate_history"])
    ):
        raise K1AuditIntegrityError(
            "K1 终态 candidate_history 哈希不匹配"
        )
    neighborhood_identity = (
        "neighborhood:"
        f"{file_sha256(required['neighborhood_evidence'])}"
    )
    if not any(
        neighborhood_identity in identity
        for identity in selection.evidence.study_identities
    ):
        raise K1AuditIntegrityError(
            "K1 选择回执未绑定完整邻域证据哈希"
        )
    selected_candidate_id = str(
        params.get("candidate_id", "")
    )
    if (
        selected_candidate_id
        != selection.evidence.selected_candidate_id
    ):
        raise K1AuditIntegrityError(
            "K1 params 与选择回执候选不一致"
        )
    plot_values = manifest.get("training_plots")
    if (
        not isinstance(plot_values, list)
        or len(plot_values) != 2
    ):
        raise K1AuditIntegrityError(
            "K1 manifest 缺少两张训练经典图"
        )
    training_plots = (
        Path(str(plot_values[0])),
        Path(str(plot_values[1])),
    )
    if any(not path.is_file() for path in training_plots):
        raise K1AuditIntegrityError("K1 训练经典图不存在")
    plan = neighborhood.get("plan")
    if not isinstance(plan, Mapping):
        raise K1AuditIntegrityError(
            "K1 邻域证据缺少 plan"
        )
    neighborhood_candidates = plan.get(
        "candidate_ids_to_evaluate"
    )
    if not isinstance(neighborhood_candidates, list):
        raise K1AuditIntegrityError(
            "K1 邻域候选列表无效"
        )
    return K1FoldResult(
        arm="K1",
        flow_label=K1_FLOW_LABEL,
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


def _formal_metrics(
    outcomes: tuple[
        CandidateSolveOutcome,
        CandidateSolveOutcome,
    ],
) -> tuple[FormalMetricResult, FormalMetricResult]:
    metrics = (
        outcomes[0].formal_metrics,
        outcomes[1].formal_metrics,
    )
    if metrics[0] is None or metrics[1] is None:
        raise RuntimeError("K1 最终候选缺少两条训练指标")
    return metrics[0], metrics[1]


def _training_evidence_from_audit(
    audit: Mapping[str, Any],
    *,
    expected_candidate: BOCandidate,
    expected_stage: str,
    expected_index_name: str,
    expected_index: int,
    expected_record_ids: tuple[str, ...],
) -> RobustTrainingEvidence:
    expected_candidate_id = expected_candidate.candidate_id
    if (
        audit.get("candidate_id") != expected_candidate_id
        or audit.get("stage") != expected_stage
        or audit.get(expected_index_name) != expected_index
    ):
        raise K1AuditIntegrityError(
            "K1 审计的候选、stage 或索引身份不匹配"
        )
    expected_identity = json_ready(
        {
            "requested_params": (
                expected_candidate.requested_params
            ),
            "actual_params": expected_candidate.actual_params,
            "fixed_params": expected_candidate.fixed_params,
        }
    )
    if audit.get("candidate_identity") != expected_identity:
        raise K1AuditIntegrityError(
            "K1 审计的候选参数身份不匹配"
        )
    outcomes = audit.get("training_outcomes")
    if not isinstance(outcomes, list) or tuple(
        str(outcome.get("record_id", ""))
        for outcome in outcomes
        if isinstance(outcome, Mapping)
    ) != expected_record_ids:
        raise K1AuditIntegrityError(
            "K1 审计的训练记录身份不匹配"
        )
    payload = audit.get("robust_evidence")
    if not isinstance(payload, Mapping):
        raise K1AuditIntegrityError(
            "K1 审计缺少 robust_evidence"
        )
    candidate_id = str(payload.get("candidate_id", ""))
    if candidate_id != expected_candidate_id:
        raise K1AuditIntegrityError(
            "K1 审计内外 candidate_id 不一致"
        )
    if not bool(payload.get("metric_valid")):
        return build_robust_training_evidence(
            candidate_id=candidate_id,
            final_motion_mae_bpm=None,
            reset_motion_mae_bpm=None,
            failure_reason=str(payload.get("failure_reason", "")),
        )
    finals = payload.get("final_motion_mae_bpm")
    resets = payload.get("reset_motion_mae_bpm")
    if (
        not isinstance(finals, list)
        or len(finals) != 2
        or not isinstance(resets, list)
        or len(resets) != 2
    ):
        raise K1AuditIntegrityError(
            "K1 审计的训练指标不是两记录结构"
        )
    return build_robust_training_evidence(
        candidate_id=candidate_id,
        final_motion_mae_bpm=(
            float(finals[0]),
            float(finals[1]),
        ),
        reset_motion_mae_bpm=(
            float(resets[0]),
            float(resets[1]),
        ),
    )


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
            _history_row_from_audit(
                arm="K1",
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
    _annotate_robust_history(
        rows,
        space=space,
        bands=bands,
        evidence_by_candidate_id=(
            evidence_by_candidate_id or {}
        ),
        selection=selection,
    )
    write_csv(path, rows)


def _history_row_from_audit(
    *,
    arm: str,
    scene: str,
    fold: int,
    stage: str,
    lane: str,
    seed: int,
    trial_number: int,
    suggestion_index: int,
    unique_index: int | None,
    candidate: BOCandidate,
    audit: Mapping[str, Any],
    is_duplicate: bool,
) -> dict[str, Any]:
    evidence = audit["robust_evidence"]
    row: dict[str, Any] = {
        "arm": arm,
        "scene": scene,
        "fold": fold,
        "stage": stage,
        "lane": lane,
        "seed": seed,
        "trial_number": trial_number,
        "suggestion_index": suggestion_index,
        "unique_index": unique_index,
        "candidate_id": candidate.candidate_id,
        "is_duplicate": is_duplicate,
        "objective": evidence["objective_bpm"],
        "tpe_objective": evidence["objective_bpm"],
        "metric_valid": evidence["metric_valid"],
        "eligible": evidence["eligible"],
        "failure_reason": evidence["failure_reason"],
        "worst_train_mae_bpm": (
            evidence["worst_train_mae_bpm"]
        ),
        "worst_train_mae": evidence["worst_train_mae_bpm"],
        "mean_train_mae_bpm": (
            evidence["mean_train_mae_bpm"]
        ),
        "mean_train_mae": evidence["mean_train_mae_bpm"],
        "constraint_train_0_bpm": (
            evidence["constraints_bpm"][0]
        ),
        "constraint_r1": evidence["constraints_bpm"][0],
        "constraint_train_1_bpm": (
            evidence["constraints_bpm"][1]
        ),
        "constraint_r2": evidence["constraints_bpm"][1],
        "nonharm_delta_train_0_bpm": (
            evidence["constraints_bpm"][0] + 2.0
        ),
        "nonharm_delta_train_1_bpm": (
            evidence["constraints_bpm"][1] + 2.0
        ),
        "runtime_seconds": audit.get("runtime_seconds", ""),
    }
    training_outcomes = audit["training_outcomes"]
    row["cache_hit"] = all(
        bool(outcome["cache_hit"])
        for outcome in training_outcomes
    )
    row["cache_key"] = "|".join(
        str(outcome["cache_key"])
        for outcome in training_outcomes
    )
    for record_index, outcome in enumerate(training_outcomes):
        row[f"train_{record_index}_record_id"] = outcome[
            "record_id"
        ]
        row[f"cache_hit_train_{record_index}"] = outcome[
            "cache_hit"
        ]
        row[f"cache_key_train_{record_index}"] = outcome[
            "cache_key"
        ]
        row[f"physical_solve_train_{record_index}"] = outcome[
            "physical_solve_performed"
        ]
        for key, value in outcome.get(
            "formal_metrics",
            {},
        ).items():
            row[f"train_{record_index}_{key}"] = value
    parameter_keys = sorted(
        {
            *candidate.requested_params,
            *candidate.actual_params,
            *candidate.fixed_params,
        }
    )
    for key in parameter_keys:
        row[f"requested_{key}"] = (
            candidate.requested_params.get(key)
        )
        row[f"actual_{key}"] = candidate.actual_params.get(key)
        row[f"fixed_{key}"] = candidate.fixed_params.get(key)
    return row


def _annotate_robust_history(
    rows: Sequence[dict[str, Any]],
    *,
    space: BOSearchSpace,
    bands: RobustBands | None,
    evidence_by_candidate_id: Mapping[
        str,
        RobustTrainingEvidence,
    ],
    selection: RobustSelection | None,
) -> None:
    selected_center_evidence = {
        center.candidate_id: center
        for center in (
            selection.center_evidence
            if selection is not None
            else ()
        )
    }
    primary_ids = (
        frozenset(bands.primary_candidate_ids)
        if bands is not None
        else frozenset()
    )
    diagnostic_ids = (
        frozenset(bands.diagnostic_candidate_ids)
        if bands is not None
        else frozenset()
    )
    center_ids = tuple(
        (
            *bands.primary_candidate_ids,
            *bands.diagnostic_candidate_ids,
        )
        if bands is not None
        else ()
    )
    neighbor_sets = {
        center_id: frozenset(
            direct_neighbor_ids(space, center_id)
        )
        for center_id in center_ids
    }
    for row in rows:
        candidate_id = str(row["candidate_id"])
        related_centers = tuple(
            center_id
            for center_id in center_ids
            if candidate_id in neighbor_sets[center_id]
        )
        candidate_evidence = evidence_by_candidate_id.get(
            candidate_id
        )
        supporting_centers: list[str] = []
        cliff_centers: list[str] = []
        for center_id in related_centers:
            center = evidence_by_candidate_id.get(center_id)
            if center is None or candidate_evidence is None:
                continue
            if (
                candidate_evidence.metric_valid
                and candidate_evidence.eligible
                and candidate_evidence.worst_train_mae_bpm
                <= center.worst_train_mae_bpm + 1.0
            ):
                supporting_centers.append(center_id)
            if (
                center.worst_train_mae_bpm <= 5.0
                and candidate_evidence.metric_valid
                and candidate_evidence.worst_train_mae_bpm
                >= 10.0
            ):
                cliff_centers.append(center_id)
        own_center = selected_center_evidence.get(candidate_id)
        row.update(
            {
                "w_star_bpm": (
                    bands.w_star_bpm
                    if bands is not None
                    else ""
                ),
                "w_star": (
                    bands.w_star_bpm
                    if bands is not None
                    else ""
                ),
                "in_primary_band": candidate_id in primary_ids,
                "in_diagnostic_band": (
                    candidate_id in diagnostic_ids
                ),
                "center_candidate_id": (
                    candidate_id
                    if candidate_id
                    in primary_ids | diagnostic_ids
                    else "|".join(related_centers)
                ),
                "is_direct_neighbor": bool(related_centers),
                "support_neighbor": bool(supporting_centers),
                "support_center_ids": "|".join(
                    supporting_centers
                ),
                "parameter_cliff": (
                    own_center.has_cliff
                    if own_center is not None
                    else bool(cliff_centers)
                ),
                "cliff_center_ids": "|".join(cliff_centers),
            }
        )


def _fail_closed(
    output: Path,
    *,
    reason: str,
    search_result: SeedSearchResult,
    candidate_history: Path,
    plan: RobustNeighborhoodPlan | None = None,
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
        output / "k1_fold_manifest.json",
        {
            "arm": "K1",
            "flow_label": K1_FLOW_LABEL,
            "status": "failed_closed",
            "failure_reason": reason,
            "candidate_history": str(candidate_history),
            "selection_receipt": "",
            "replay_receipt": "",
        },
    )
