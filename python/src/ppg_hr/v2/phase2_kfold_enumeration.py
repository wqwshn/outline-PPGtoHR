"""Phase2 K2 精简旧空间 108/108 全枚举稳健共享选参。"""

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
    ContentAddressedSolverCache,
    SolverCacheIdentity,
    build_bo_search_space,
)
from .phase2_experiment_io import (
    atomic_write_json,
    cache_summary,
    file_sha256,
    json_ready,
    read_json,
    space_sha256,
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
    RobustSelectionError,
    RobustTrainingEvidence,
    build_robust_bands,
    plan_robust_neighborhood,
    select_robust_center,
)
from .types import V2RunConfig

K2_FLOW_LABEL = "精简旧四维空间全枚举稳健选参流程"
_SPACE_NAME = "legacy_reduced_v1"
_ENUMERATION_COUNT = 108

K2AuditIntegrityError = RobustFoldAuditIntegrityError


class K2DriverIdentityConflictError(RuntimeError):
    """K2 输出目录已绑定其他折级配置。"""


@dataclass(frozen=True)
class K2FoldConfig:
    output_dir: Path
    scene: str
    fold: int
    git_commit: str
    code_dirty: bool = False

    def __post_init__(self) -> None:
        if not self.scene or not self.git_commit:
            raise ValueError("K2 必须冻结 scene 和 git_commit")
        if type(self.fold) is not int or self.fold < 0:
            raise ValueError("fold 必须是非负整数")
        if type(self.code_dirty) is not bool:
            raise ValueError("code_dirty 必须是布尔值")


@dataclass(frozen=True)
class K2FoldResult:
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
    enumeration_count: int
    coverage_ratio: float


def build_k2_default_runtime(
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
        arm="K2",
        base_config=base_config,
        training_records=training_records,
        heldout_record=heldout_record,
        output_dir=output_dir,
    )


def run_k2_fold_study(
    config: K2FoldConfig,
    *,
    runtime: KFoldRuntime,
    enumeration_order: Sequence[str] | None = None,
) -> K2FoldResult:
    """全枚举精简旧空间并用完整空间邻域证据冻结一折。"""

    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    space = build_bo_search_space(_SPACE_NAME)
    if len(space.candidates) != _ENUMERATION_COUNT:
        raise AssertionError("K2 精简旧空间不再是 108 个候选")
    candidates = {
        candidate.candidate_id: candidate
        for candidate in space.candidates
    }
    ordered_ids = _resolve_enumeration_order(
        space,
        enumeration_order,
    )
    driver_identity_path = output / "driver_identity.json"
    driver_identity_hash = _freeze_driver_identity(
        driver_identity_path,
        config=config,
        runtime=runtime,
        space=space,
        ordered_ids=ordered_ids,
    )
    completed = _load_completed_result(
        output,
        expected_driver_identity_sha256=driver_identity_hash,
    )
    if completed is not None:
        return completed

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
                "arm": "K2",
                "scene": config.scene,
                "fold": config.fold,
                "space_name": space.name,
                "enumeration_contract": "complete_108_v1",
            },
            candidate=candidate,
            reference_groups_order=("HF",),
        )

    all_evidence: dict[str, RobustTrainingEvidence] = {}
    rows: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(ordered_ids):
        candidate = candidates[candidate_id]
        audit_path = (
            trial_audit_dir
            / f"enumeration-{index:03d}.json"
        )
        if audit_path.is_file():
            evidence = training_evidence_from_audit(
                read_json(audit_path),
                expected_candidate=candidate,
                expected_stage="enumeration",
                expected_index_name="enumeration_index",
                expected_index=index,
                expected_record_ids=tuple(
                    record.identity.record_id
                    for record in runtime.training_records
                ),
                arm="K2",
            )
        else:
            evidence, _ = evaluate_training_candidate(
                cache=cache,
                candidate=candidate,
                training_records=runtime.training_records,
                cache_identity=cache_identity,
                arm="K2",
                scene=config.scene,
                fold=config.fold,
                logical_reference={
                    "stage": "enumeration",
                    "enumeration_index": index,
                    "candidate_id": candidate_id,
                },
                audit_path=audit_path,
            )
        all_evidence[candidate_id] = evidence
        rows.append(
            history_row_from_audit(
                arm="K2",
                scene=config.scene,
                fold=config.fold,
                stage="enumeration",
                lane="enumeration",
                seed=-1,
                trial_number=index,
                suggestion_index=index + 1,
                unique_index=index + 1,
                candidate=candidate,
                audit=read_json(audit_path),
                is_duplicate=False,
                selection_source="enumeration",
            )
        )

    history_path = output / "candidate_history.csv"
    try:
        bands = build_robust_bands(all_evidence.values())
    except RobustSelectionError as exc:
        annotate_robust_history(
            rows,
            space=space,
            bands=None,
            evidence_by_candidate_id=all_evidence,
            selection=None,
        )
        write_csv(history_path, rows)
        _fail_closed(
            output,
            reason=exc.reason,
            cache=cache,
            candidate_history=history_path,
        )
        raise RuntimeError(exc.reason) from exc

    plan = plan_robust_neighborhood(
        space=space,
        bands=bands,
        reviewed_candidate_ids=frozenset(all_evidence),
        max_new_candidates=0,
    )
    if plan.candidate_ids_to_evaluate:
        raise AssertionError("K2 全枚举后不应再请求邻域候选")
    try:
        selection = select_robust_center(
            space=space,
            bands=bands,
            plan=plan,
            evidence_by_candidate_id=all_evidence,
        )
    except RobustSelectionError as exc:
        annotate_robust_history(
            rows,
            space=space,
            bands=bands,
            evidence_by_candidate_id=all_evidence,
            selection=None,
        )
        write_csv(history_path, rows)
        _fail_closed(
            output,
            reason=exc.reason,
            cache=cache,
            candidate_history=history_path,
        )
        raise RuntimeError(exc.reason) from exc

    annotate_robust_history(
        rows,
        space=space,
        bands=bands,
        evidence_by_candidate_id=all_evidence,
        selection=selection,
    )
    write_csv(history_path, rows)
    selected_candidate = candidates[selection.candidate_id]
    selected_evidence, selected_outcomes = (
        evaluate_training_candidate(
            cache=cache,
            candidate=selected_candidate,
            training_records=runtime.training_records,
            cache_identity=cache_identity,
            arm="K2",
            scene=config.scene,
            fold=config.fold,
            logical_reference={
                "stage": "final_selection",
                "candidate_id": selection.candidate_id,
            },
            audit_path=trial_audit_dir / "final-selection.json",
        )
    )
    if selected_evidence != all_evidence[selection.candidate_id]:
        raise K2AuditIntegrityError(
            "K2 最终候选复核与枚举证据不一致"
        )
    selected_metrics = formal_metrics(
        selected_outcomes,
        arm="K2",
    )
    selected_center = next(
        center
        for center in selection.primary_center_evidence
        if center.candidate_id == selection.candidate_id
    )

    params_path = output / "params.json"
    atomic_write_json(
        params_path,
        {
            "arm": "K2",
            "flow_label": K2_FLOW_LABEL,
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
                "arm": "K2",
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
    neighborhood_path = output / "neighborhood_evidence.json"
    atomic_write_json(
        neighborhood_path,
        {
            "arm": "K2",
            "coverage": {
                "enumerated_candidate_count": len(all_evidence),
                "space_candidate_count": len(space.candidates),
                "coverage_ratio": 1.0,
                "additional_neighborhood_candidate_count": 0,
            },
            "bands": asdict(bands),
            "plan": asdict(plan),
            "primary_center_evidence": [
                asdict(center)
                for center in selection.primary_center_evidence
            ],
            "diagnostic_center_evidence": [
                asdict(center)
                for center in selection.diagnostic_center_evidence
            ],
            "selected_candidate_id": selection.candidate_id,
        },
    )
    selection_receipt_path = output / "selection_receipt.json"
    enumeration_identity = (
        f"enumeration:{driver_identity_hash}|neighborhood:"
        f"{file_sha256(neighborhood_path)}"
    )
    selection_receipt = freeze_selection(
        selection_receipt_path,
        SelectionEvidence(
            experiment_name=(
                f"{config.scene}-fold-{config.fold}-k2"
            ),
            arm="K2",
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
            study_identities=(enumeration_identity,),
            budget=SearchBudgetEvidence(
                lane_unique_budget=_ENUMERATION_COUNT,
                requested_global_unique_budget=_ENUMERATION_COUNT,
                actual_global_unique_count=len(all_evidence),
                requested_neighborhood_budget=0,
                actual_neighborhood_count=0,
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
                truncated_center_count=0,
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
    cache_summary_path = output / "cache_summary.json"
    atomic_write_json(cache_summary_path, cache_summary(cache))
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
            "replay_status": replay_receipt.status,
            "replay_failure_reason": replay_receipt.failure_reason,
        },
    )
    manifest_path = output / "k2_fold_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "arm": "K2",
            "flow_label": K2_FLOW_LABEL,
            "causal_claim_allowed": False,
            "confirmatory_claim_allowed": False,
            "comparison_scope": "operational_workflow_only",
            "scene": config.scene,
            "fold": config.fold,
            "git_commit": config.git_commit,
            "driver_identity_sha256": driver_identity_hash,
            "selected_candidate_id": (
                selected_candidate.candidate_id
            ),
            "selection_hash": selection_receipt.selection_hash,
            "enumeration_count": len(all_evidence),
            "space_candidate_count": len(space.candidates),
            "coverage_ratio": 1.0,
            "additional_neighborhood_candidate_count": 0,
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
        },
    )
    return K2FoldResult(
        arm="K2",
        flow_label=K2_FLOW_LABEL,
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
        enumeration_count=len(all_evidence),
        coverage_ratio=1.0,
    )


def _resolve_enumeration_order(
    space: BOSearchSpace,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    canonical = tuple(
        candidate.candidate_id
        for candidate in space.candidates
    )
    if requested is None:
        return canonical
    resolved = tuple(requested)
    if (
        len(resolved) != len(canonical)
        or len(set(resolved)) != len(canonical)
        or set(resolved) != set(canonical)
    ):
        raise ValueError(
            "K2 enumeration_order 必须是 108 个空间候选的完整排列"
        )
    return resolved


def _freeze_driver_identity(
    path: Path,
    *,
    config: K2FoldConfig,
    runtime: KFoldRuntime,
    space: BOSearchSpace,
    ordered_ids: tuple[str, ...],
) -> str:
    identity = {
        "arm": "K2",
        "flow_label": K2_FLOW_LABEL,
        "scene": config.scene,
        "fold": config.fold,
        "git_commit": config.git_commit,
        "code_dirty": config.code_dirty,
        "space_name": space.name,
        "space_sha256": space_sha256(space.candidates),
        "enumeration_count": len(ordered_ids),
        "enumeration_order_sha256": _mapping_sha256(
            {"candidate_ids": ordered_ids}
        ),
        "training_records": [
            asdict(record.identity)
            for record in runtime.training_records
        ],
        "heldout_record": asdict(runtime.heldout_record),
    }
    identity_sha256 = _mapping_sha256(identity)
    payload = {
        "schema_version": "phase2_k2_driver_identity_v1",
        "identity_sha256": identity_sha256,
        "identity": identity,
    }
    if path.is_file():
        if read_json(path) != json_ready(payload):
            raise K2DriverIdentityConflictError(
                f"K2 输出目录已绑定其他折级配置: {path}"
            )
        return identity_sha256
    atomic_write_json(path, payload)
    return identity_sha256


def _load_completed_result(
    output: Path,
    *,
    expected_driver_identity_sha256: str,
) -> K2FoldResult | None:
    selection_path = output / "selection_receipt.json"
    replay_path = output / "replay_receipt.json"
    manifest_path = output / "k2_fold_manifest.json"
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
    if any(not path.is_file() for path in required.values()):
        raise K2AuditIntegrityError("K2 终态产物不完整")
    selection = load_selection_receipt(selection_path)
    replay = load_replay_receipt(replay_path)
    manifest = read_json(manifest_path)
    params = read_json(required["selected_params"])
    if (
        manifest.get("driver_identity_sha256")
        != expected_driver_identity_sha256
        or manifest.get("selection_hash")
        != selection.selection_hash
        or replay.selection_hash != selection.selection_hash
    ):
        raise K2AuditIntegrityError(
            "K2 终态回执、manifest 或 driver 身份不一致"
        )
    if (
        selection.evidence.candidate_history_sha256
        != file_sha256(required["candidate_history"])
    ):
        raise K2AuditIntegrityError(
            "K2 candidate_history 哈希不匹配"
        )
    neighborhood_hash = file_sha256(
        required["neighborhood_evidence"]
    )
    if not any(
        f"neighborhood:{neighborhood_hash}" in identity
        for identity in selection.evidence.study_identities
    ):
        raise K2AuditIntegrityError(
            "K2 选择回执未绑定完整枚举邻域证据"
        )
    expected_params = {
        "arm": "K2",
        "flow_label": K2_FLOW_LABEL,
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
    if params != json_ready(expected_params):
        raise K2AuditIntegrityError(
            "K2 params 与选择回执不一致"
        )
    if (
        manifest.get("selected_candidate_id")
        != selection.evidence.selected_candidate_id
        or manifest.get("enumeration_count")
        != _ENUMERATION_COUNT
        or manifest.get("space_candidate_count")
        != _ENUMERATION_COUNT
        or manifest.get("coverage_ratio") != 1.0
        or manifest.get(
            "additional_neighborhood_candidate_count"
        )
        != 0
    ):
        raise K2AuditIntegrityError(
            "K2 manifest 的候选或覆盖率不一致"
        )
    plot_values = manifest.get("training_plots")
    if (
        not isinstance(plot_values, list)
        or len(plot_values) != 2
    ):
        raise K2AuditIntegrityError(
            "K2 manifest 缺少两张训练经典图"
        )
    training_plots = (
        Path(str(plot_values[0])),
        Path(str(plot_values[1])),
    )
    if any(not path.is_file() for path in training_plots):
        raise K2AuditIntegrityError("K2 训练经典图不存在")
    validate_terminal_artifact_manifest(
        manifest,
        required={
            **required,
            "selection_receipt": selection_path,
            "replay_receipt": replay_path,
        },
        training_plots=training_plots,
        arm="K2",
    )
    if replay.status == "infrastructure_failed":
        return None
    return K2FoldResult(
        arm="K2",
        flow_label=K2_FLOW_LABEL,
        selected_candidate_id=(
            selection.evidence.selected_candidate_id
        ),
        selected_worst_train_mae_bpm=(
            selection.evidence.training_metrics.worst_train_mae_bpm
        ),
        selected_mean_train_mae_bpm=(
            selection.evidence.training_metrics.mean_train_mae_bpm
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
        enumeration_count=_ENUMERATION_COUNT,
        coverage_ratio=1.0,
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


def _fail_closed(
    output: Path,
    *,
    reason: str,
    cache: ContentAddressedSolverCache,
    candidate_history: Path,
) -> None:
    cache_summary_path = output / "cache_summary.json"
    atomic_write_json(cache_summary_path, cache_summary(cache))
    failure_path = output / "failure_classification.json"
    atomic_write_json(
        failure_path,
        {"failure_reason": reason},
    )
    atomic_write_json(
        output / "k2_fold_manifest.json",
        {
            "arm": "K2",
            "flow_label": K2_FLOW_LABEL,
            "status": "failed_closed",
            "failure_reason": reason,
            "enumeration_count": _ENUMERATION_COUNT,
            "space_candidate_count": _ENUMERATION_COUNT,
            "coverage_ratio": 1.0,
            "candidate_history": str(candidate_history),
            "cache_summary": str(cache_summary_path),
            "failure_classification": str(failure_path),
            "causal_claim_allowed": False,
            "comparison_scope": "operational_workflow_only",
        },
    )
