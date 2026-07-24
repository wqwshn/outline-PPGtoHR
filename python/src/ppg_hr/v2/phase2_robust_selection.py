"""Phase2 K1/K2/K3 的稳健共享参数选择纯规则。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from .bo_space_generalization import BOSearchSpace

_INVALID_OBJECTIVE = 1e9
_INVALID_CONSTRAINT = 1e6
_NONHARM_ALLOWANCE_BPM = 2.0
_PRIMARY_BAND_BPM = 0.25
_DIAGNOSTIC_BAND_BPM = 0.5
_SUPPORT_DELTA_BPM = 1.0
_CLIFF_CENTER_MAX_BPM = 5.0
_CLIFF_NEIGHBOR_MIN_BPM = 10.0


class RobustSelectionError(RuntimeError):
    """稳健共享参数规则无法产生可冻结结果。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class RobustTrainingEvidence:
    candidate_id: str
    metric_valid: bool
    eligible: bool
    objective_bpm: float
    constraints_bpm: tuple[float, float]
    final_motion_mae_bpm: tuple[float, float] | None
    reset_motion_mae_bpm: tuple[float, float] | None
    worst_train_mae_bpm: float
    mean_train_mae_bpm: float
    failure_reason: str = ""


@dataclass(frozen=True)
class RobustBands:
    w_star_bpm: float
    primary_candidate_ids: tuple[str, ...]
    diagnostic_candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class RobustNeighborhoodPlan:
    candidate_ids_to_evaluate: tuple[str, ...]
    complete_primary_center_ids: tuple[str, ...]
    truncated_primary_center_ids: tuple[str, ...]
    complete_diagnostic_center_ids: tuple[str, ...]
    diagnostic_candidate_ids_started: tuple[str, ...]
    unused_budget: int


@dataclass(frozen=True)
class RobustCenterEvidence:
    candidate_id: str
    direct_neighbor_ids: tuple[str, ...]
    supporting_neighbor_ids: tuple[str, ...]
    reviewed_neighbor_count: int
    support_ratio: float
    has_cliff: bool
    worst_train_mae_bpm: float
    mean_train_mae_bpm: float


@dataclass(frozen=True)
class RobustSelection:
    candidate_id: str
    center_evidence: tuple[RobustCenterEvidence, ...]


def build_robust_training_evidence(
    *,
    candidate_id: str,
    final_motion_mae_bpm: tuple[float, float] | None,
    reset_motion_mae_bpm: tuple[float, float] | None,
    failure_reason: str = "",
) -> RobustTrainingEvidence:
    """把两条训练记录的严格运动指标转换为冻结 TPE 反馈。"""

    if not candidate_id:
        raise ValueError("candidate_id 不得为空")
    if final_motion_mae_bpm is None or reset_motion_mae_bpm is None:
        if not failure_reason:
            raise ValueError("无效稳健指标必须提供 failure_reason")
        return RobustTrainingEvidence(
            candidate_id=candidate_id,
            metric_valid=False,
            eligible=False,
            objective_bpm=_INVALID_OBJECTIVE,
            constraints_bpm=(
                _INVALID_CONSTRAINT,
                _INVALID_CONSTRAINT,
            ),
            final_motion_mae_bpm=None,
            reset_motion_mae_bpm=None,
            worst_train_mae_bpm=_INVALID_OBJECTIVE,
            mean_train_mae_bpm=_INVALID_OBJECTIVE,
            failure_reason=failure_reason,
        )
    finals = _finite_pair(final_motion_mae_bpm, "final_motion_mae_bpm")
    resets = _finite_pair(reset_motion_mae_bpm, "reset_motion_mae_bpm")
    constraints = (
        finals[0] - resets[0] - _NONHARM_ALLOWANCE_BPM,
        finals[1] - resets[1] - _NONHARM_ALLOWANCE_BPM,
    )
    worst = max(finals)
    mean = float(np.mean(finals))
    return RobustTrainingEvidence(
        candidate_id=candidate_id,
        metric_valid=True,
        eligible=all(value <= 0.0 for value in constraints),
        objective_bpm=worst,
        constraints_bpm=constraints,
        final_motion_mae_bpm=finals,
        reset_motion_mae_bpm=resets,
        worst_train_mae_bpm=worst,
        mean_train_mae_bpm=mean,
    )


def build_robust_bands(
    evidence: Iterable[RobustTrainingEvidence],
) -> RobustBands:
    """用一次全局最小值构造传递的主带与扩展诊断带。"""

    safe = tuple(
        item
        for item in evidence
        if item.metric_valid and item.eligible
    )
    if not safe:
        raise RobustSelectionError("no_safe_shared_candidate")
    w_star = min(item.worst_train_mae_bpm for item in safe)

    def training_key(
        item: RobustTrainingEvidence,
    ) -> tuple[float, float, str]:
        return (
            item.mean_train_mae_bpm,
            item.worst_train_mae_bpm,
            item.candidate_id,
        )

    primary = tuple(
        item.candidate_id
        for item in sorted(
            (
                item
                for item in safe
                if item.worst_train_mae_bpm
                <= w_star + _PRIMARY_BAND_BPM
            ),
            key=training_key,
        )
    )
    primary_set = frozenset(primary)
    diagnostic = tuple(
        item.candidate_id
        for item in sorted(
            (
                item
                for item in safe
                if (
                    item.candidate_id not in primary_set
                    and item.worst_train_mae_bpm
                    <= w_star + _DIAGNOSTIC_BAND_BPM
                )
            ),
            key=training_key,
        )
    )
    return RobustBands(
        w_star_bpm=w_star,
        primary_candidate_ids=primary,
        diagnostic_candidate_ids=diagnostic,
    )


def plan_robust_neighborhood(
    *,
    space: BOSearchSpace,
    bands: RobustBands,
    reviewed_candidate_ids: frozenset[str],
    max_new_candidates: int = 30,
) -> RobustNeighborhoodPlan:
    """主带优先安排直接邻居；预算不足时保留明确截断状态。"""

    if type(max_new_candidates) is not int or max_new_candidates < 0:
        raise ValueError("max_new_candidates 必须是非负整数")
    candidate_ids = frozenset(
        candidate.candidate_id for candidate in space.candidates
    )
    unknown = reviewed_candidate_ids - candidate_ids
    if unknown:
        raise ValueError(
            "reviewed_candidate_ids 含空间外候选: "
            + ", ".join(sorted(unknown))
        )
    center_sequence = (
        *bands.primary_candidate_ids,
        *bands.diagnostic_candidate_ids,
    )
    if any(center not in candidate_ids for center in center_sequence):
        raise ValueError("稳健中心不属于给定参数空间")

    available = set(reviewed_candidate_ids)
    scheduled: list[str] = []
    diagnostic_started: list[str] = []
    budget_remaining = max_new_candidates
    for center_id in center_sequence:
        is_diagnostic = center_id in bands.diagnostic_candidate_ids
        neighbors = _direct_neighbor_ids(space, center_id)
        missing = [
            candidate_id
            for candidate_id in neighbors
            if candidate_id not in available
        ]
        if is_diagnostic and (not missing or budget_remaining > 0):
            diagnostic_started.append(center_id)
        if not missing:
            continue
        take_count = min(len(missing), budget_remaining)
        selected = missing[:take_count]
        scheduled.extend(selected)
        available.update(selected)
        budget_remaining -= take_count
        if take_count < len(missing):
            break

    complete_primary, truncated_primary = _center_completion(
        space,
        bands.primary_candidate_ids,
        frozenset(available),
    )
    complete_diagnostic, _ = _center_completion(
        space,
        bands.diagnostic_candidate_ids,
        frozenset(available),
    )
    return RobustNeighborhoodPlan(
        candidate_ids_to_evaluate=tuple(scheduled),
        complete_primary_center_ids=complete_primary,
        truncated_primary_center_ids=truncated_primary,
        complete_diagnostic_center_ids=complete_diagnostic,
        diagnostic_candidate_ids_started=tuple(diagnostic_started),
        unused_budget=budget_remaining,
    )


def select_robust_center(
    *,
    space: BOSearchSpace,
    bands: RobustBands,
    plan: RobustNeighborhoodPlan,
    evidence_by_candidate_id: Mapping[str, RobustTrainingEvidence],
) -> RobustSelection:
    """只在搜索主带内、合格且实际完整复核的中心中固定排序。"""

    complete: list[RobustCenterEvidence] = []
    for center_id in bands.primary_candidate_ids:
        if center_id not in plan.complete_primary_center_ids:
            continue
        center = evidence_by_candidate_id.get(center_id)
        if center is None or not center.metric_valid or not center.eligible:
            continue
        neighbor_ids = _direct_neighbor_ids(space, center_id)
        neighbors = tuple(
            evidence_by_candidate_id.get(candidate_id)
            for candidate_id in neighbor_ids
        )
        if any(item is None for item in neighbors):
            continue
        typed_neighbors = tuple(
            item
            for item in neighbors
            if item is not None
        )
        supporting_ids = tuple(
            candidate_id
            for candidate_id, neighbor in zip(
                neighbor_ids,
                typed_neighbors,
                strict=True,
            )
            if (
                neighbor.metric_valid
                and neighbor.eligible
                and neighbor.worst_train_mae_bpm
                <= center.worst_train_mae_bpm + _SUPPORT_DELTA_BPM
            )
        )
        has_cliff = (
            center.worst_train_mae_bpm <= _CLIFF_CENTER_MAX_BPM
            and any(
                neighbor.metric_valid
                and neighbor.worst_train_mae_bpm
                >= _CLIFF_NEIGHBOR_MIN_BPM
                for neighbor in typed_neighbors
            )
        )
        complete.append(
            RobustCenterEvidence(
                candidate_id=center_id,
                direct_neighbor_ids=neighbor_ids,
                supporting_neighbor_ids=supporting_ids,
                reviewed_neighbor_count=len(neighbor_ids),
                support_ratio=(
                    len(supporting_ids) / len(neighbor_ids)
                    if neighbor_ids
                    else 1.0
                ),
                has_cliff=has_cliff,
                worst_train_mae_bpm=center.worst_train_mae_bpm,
                mean_train_mae_bpm=center.mean_train_mae_bpm,
            )
        )
    if not complete:
        raise RobustSelectionError(
            "no_fully_reviewed_primary_center"
        )
    ordered = tuple(
        sorted(
            complete,
            key=lambda center: (
                center.has_cliff,
                -center.support_ratio,
                -center.reviewed_neighbor_count,
                center.worst_train_mae_bpm,
                center.mean_train_mae_bpm,
                center.candidate_id,
            ),
        )
    )
    return RobustSelection(
        candidate_id=ordered[0].candidate_id,
        center_evidence=ordered,
    )


def _direct_neighbor_ids(
    space: BOSearchSpace,
    center_id: str,
) -> tuple[str, ...]:
    candidates = {
        candidate.candidate_id: candidate for candidate in space.candidates
    }
    center = candidates[center_id]
    coordinate_to_id = {
        candidate.coordinate: candidate.candidate_id
        for candidate in space.candidates
    }
    neighbors: set[str] = set()
    for dimension, coordinate in enumerate(center.coordinate):
        for delta in (-1, 1):
            moved = coordinate + delta
            if not 0 <= moved < len(space.option_values[dimension]):
                continue
            neighbor_coordinate = list(center.coordinate)
            neighbor_coordinate[dimension] = moved
            neighbor_id = coordinate_to_id.get(tuple(neighbor_coordinate))
            if neighbor_id is not None:
                neighbors.add(neighbor_id)
    return tuple(sorted(neighbors))


def _center_completion(
    space: BOSearchSpace,
    center_ids: tuple[str, ...],
    available: frozenset[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    complete: list[str] = []
    truncated: list[str] = []
    for center_id in center_ids:
        neighbors = _direct_neighbor_ids(space, center_id)
        if all(candidate_id in available for candidate_id in neighbors):
            complete.append(center_id)
        else:
            truncated.append(center_id)
    return tuple(complete), tuple(truncated)


def _finite_pair(
    values: tuple[float, float],
    name: str,
) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} 必须恰好包含两条训练记录")
    pair = (float(values[0]), float(values[1]))
    if any(not math.isfinite(value) for value in pair):
        raise ValueError(f"{name} 必须全部有限")
    return pair
