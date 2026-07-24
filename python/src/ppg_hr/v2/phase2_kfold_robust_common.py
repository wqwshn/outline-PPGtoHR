"""Phase2 稳健 K 折流程共享的审计、历史与终态产物合同。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from .bo_space_generalization import (
    BOCandidate,
    BOSearchSpace,
    CandidateSolveOutcome,
    ContentAddressedSolverCache,
    FormalMetricResult,
    SolverCacheIdentity,
)
from .phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    json_ready,
)
from .phase2_kfold_runtime import KFoldTrainingRecordRuntime
from .phase2_robust_selection import (
    RobustBands,
    RobustSelection,
    RobustTrainingEvidence,
    build_robust_training_evidence,
    direct_neighbor_ids,
)


class RobustFoldAuditIntegrityError(RuntimeError):
    """稳健 K 折不可变证据的身份或内容不匹配。"""


def evaluate_training_candidate(
    *,
    cache: ContentAddressedSolverCache,
    candidate: BOCandidate,
    training_records: tuple[
        KFoldTrainingRecordRuntime,
        KFoldTrainingRecordRuntime,
    ],
    cache_identity: Callable[
        [KFoldTrainingRecordRuntime, BOCandidate],
        SolverCacheIdentity,
    ],
    arm: str,
    scene: str,
    fold: int,
    logical_reference: Mapping[str, Any],
    audit_path: Path,
) -> tuple[
    RobustTrainingEvidence,
    tuple[CandidateSolveOutcome, CandidateSolveOutcome],
]:
    started_at = perf_counter()
    outcomes: list[CandidateSolveOutcome] = []
    audit_outcomes: list[dict[str, Any]] = []
    for record_index, record in enumerate(training_records):
        lookup = cache.get_or_solve(
            cache_identity(record, candidate),
            lambda record=record, candidate=candidate: (
                record.solve_candidate(candidate)
            ),
            logical_reference={
                "arm": arm,
                "scene": scene,
                "fold": fold,
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
                "failure_reason": lookup.outcome.failure_reason,
                "diagnostics": lookup.outcome.diagnostics,
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
            raise AssertionError(
                f"有效 {arm} outcome 缺少正式指标"
            )
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


def terminal_artifact_manifest(
    *,
    required: Mapping[str, Path],
    training_plots: Sequence[Path],
) -> dict[str, Any]:
    return {
        "files": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in sorted(required.items())
        },
        "training_plots": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for path in training_plots
        ],
    }


def validate_terminal_artifact_manifest(
    manifest: Mapping[str, Any],
    *,
    required: Mapping[str, Path],
    training_plots: Sequence[Path],
    arm: str,
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RobustFoldAuditIntegrityError(
            f"{arm} manifest 缺少终态产物哈希"
        )
    files = artifacts.get("files")
    plot_entries = artifacts.get("training_plots")
    if not isinstance(files, Mapping):
        raise RobustFoldAuditIntegrityError(
            f"{arm} manifest 终态文件清单无效"
        )
    if set(files) != set(required):
        raise RobustFoldAuditIntegrityError(
            f"{arm} manifest 终态文件集合不匹配"
        )
    for name, path in required.items():
        entry = files.get(name)
        if (
            not isinstance(entry, Mapping)
            or Path(str(entry.get("path", ""))).resolve()
            != path.resolve()
            or entry.get("sha256") != file_sha256(path)
        ):
            raise RobustFoldAuditIntegrityError(
                f"{arm} 终态产物身份或哈希不匹配: {name}"
            )
    if (
        not isinstance(plot_entries, list)
        or len(plot_entries) != len(training_plots)
    ):
        raise RobustFoldAuditIntegrityError(
            f"{arm} manifest 训练图哈希清单无效"
        )
    for index, (entry, path) in enumerate(
        zip(plot_entries, training_plots, strict=True)
    ):
        if (
            not isinstance(entry, Mapping)
            or Path(str(entry.get("path", ""))).resolve()
            != path.resolve()
            or entry.get("sha256") != file_sha256(path)
        ):
            raise RobustFoldAuditIntegrityError(
                f"{arm} 训练图身份或哈希不匹配: index={index}"
            )


def formal_metrics(
    outcomes: tuple[
        CandidateSolveOutcome,
        CandidateSolveOutcome,
    ],
    *,
    arm: str,
) -> tuple[FormalMetricResult, FormalMetricResult]:
    metrics = (
        outcomes[0].formal_metrics,
        outcomes[1].formal_metrics,
    )
    if metrics[0] is None or metrics[1] is None:
        raise RuntimeError(f"{arm} 最终候选缺少两条训练指标")
    return metrics[0], metrics[1]


def training_evidence_from_audit(
    audit: Mapping[str, Any],
    *,
    expected_candidate: BOCandidate,
    expected_stage: str,
    expected_index_name: str,
    expected_index: int,
    expected_record_ids: tuple[str, ...],
    arm: str,
) -> RobustTrainingEvidence:
    expected_candidate_id = expected_candidate.candidate_id
    if (
        audit.get("candidate_id") != expected_candidate_id
        or audit.get("stage") != expected_stage
        or audit.get(expected_index_name) != expected_index
    ):
        raise RobustFoldAuditIntegrityError(
            f"{arm} 审计的候选、stage 或索引身份不匹配"
        )
    expected_identity = json_ready(
        {
            "requested_params": expected_candidate.requested_params,
            "actual_params": expected_candidate.actual_params,
            "fixed_params": expected_candidate.fixed_params,
        }
    )
    if audit.get("candidate_identity") != expected_identity:
        raise RobustFoldAuditIntegrityError(
            f"{arm} 审计的候选参数身份不匹配"
        )
    outcomes = audit.get("training_outcomes")
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != 2
        or not all(
            isinstance(outcome, Mapping)
            for outcome in outcomes
        )
        or tuple(
            str(outcome.get("record_id", ""))
            for outcome in outcomes
        )
        != expected_record_ids
    ):
        raise RobustFoldAuditIntegrityError(
            f"{arm} 审计的训练记录身份不匹配"
        )
    payload = audit.get("robust_evidence")
    if not isinstance(payload, Mapping):
        raise RobustFoldAuditIntegrityError(
            f"{arm} 审计缺少 robust_evidence"
        )
    candidate_id = str(payload.get("candidate_id", ""))
    if candidate_id != expected_candidate_id:
        raise RobustFoldAuditIntegrityError(
            f"{arm} 审计内外 candidate_id 不一致"
        )
    typed_outcomes = tuple(outcomes)
    statuses = tuple(
        str(outcome.get("status", ""))
        for outcome in typed_outcomes
    )
    if any(status not in {"valid", "invalid"} for status in statuses):
        raise RobustFoldAuditIntegrityError(
            f"{arm} 审计包含未知训练 outcome 状态"
        )
    try:
        parsed_metrics: list[FormalMetricResult | None] = []
        for outcome, status in zip(
            typed_outcomes,
            statuses,
            strict=True,
        ):
            raw_metrics = outcome.get("formal_metrics")
            if status == "valid":
                if not isinstance(raw_metrics, Mapping):
                    raise TypeError("valid outcome 缺少 formal_metrics")
                parsed_metrics.append(
                    FormalMetricResult(**dict(raw_metrics))
                )
            else:
                if raw_metrics != {}:
                    raise ValueError(
                        "invalid outcome 不得包含 formal_metrics"
                    )
                if not str(outcome.get("failure_reason", "")):
                    raise ValueError(
                        "invalid outcome 缺少 failure_reason"
                    )
                parsed_metrics.append(None)
        if statuses == ("valid", "valid"):
            if (
                parsed_metrics[0] is None
                or parsed_metrics[1] is None
            ):
                raise AssertionError("有效训练指标解析失败")
            derived = build_robust_training_evidence(
                candidate_id=candidate_id,
                final_motion_mae_bpm=(
                    parsed_metrics[
                        0
                    ].reliable_motion_final_mae_bpm,
                    parsed_metrics[
                        1
                    ].reliable_motion_final_mae_bpm,
                ),
                reset_motion_mae_bpm=(
                    parsed_metrics[
                        0
                    ].reliable_motion_reset_fft_mae_bpm,
                    parsed_metrics[
                        1
                    ].reliable_motion_reset_fft_mae_bpm,
                ),
            )
        else:
            invalid_index = statuses.index("invalid")
            derived = build_robust_training_evidence(
                candidate_id=candidate_id,
                final_motion_mae_bpm=None,
                reset_motion_mae_bpm=None,
                failure_reason=str(
                    typed_outcomes[invalid_index][
                        "failure_reason"
                    ]
                ),
            )
    except (TypeError, ValueError) as exc:
        raise RobustFoldAuditIntegrityError(
            f"{arm} 审计的训练 outcome 指标无效"
        ) from exc
    if json_ready(asdict(derived)) != json_ready(payload):
        raise RobustFoldAuditIntegrityError(
            f"{arm} 审计的 robust_evidence 与训练 outcome 不一致"
        )
    return derived


def history_row_from_audit(
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
        "worst_train_mae_bpm": evidence["worst_train_mae_bpm"],
        "worst_train_mae": evidence["worst_train_mae_bpm"],
        "mean_train_mae_bpm": evidence["mean_train_mae_bpm"],
        "mean_train_mae": evidence["mean_train_mae_bpm"],
        "constraint_train_0_bpm": evidence[
            "constraints_bpm"
        ][0],
        "constraint_r1": evidence["constraints_bpm"][0],
        "constraint_train_1_bpm": evidence[
            "constraints_bpm"
        ][1],
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
        for key, value in outcome.get("diagnostics", {}).items():
            row[f"diagnostic_train_{record_index}_{key}"] = value
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


def annotate_robust_history(
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
                "in_primary_band": candidate_id
                in primary_ids,
                "in_diagnostic_band": candidate_id
                in diagnostic_ids,
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
