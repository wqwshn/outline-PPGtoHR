"""Phase2 训练选择与冻结测试回放之间的不可变回执边界。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

SELECTION_RECEIPT_SCHEMA_VERSION = "phase2_selection_receipt_v1"
REPLAY_RECEIPT_SCHEMA_VERSION = "phase2_frozen_replay_receipt_v1"
EvidenceLevel = Literal["development_reuse_pilot"]
ReplayOutcomeStatus = Literal["success", "invalid"]
ReplayReceiptStatus = Literal["success", "invalid", "infrastructure_failed"]
InfrastructureFailureReason = Literal[
    "solver_timeout",
    "cache_io_failed",
    "worker_interrupted",
]
_INFRASTRUCTURE_FAILURE_REASONS = frozenset(
    {"solver_timeout", "cache_io_failed", "worker_interrupted"}
)


class ReceiptIntegrityError(ValueError):
    """回执内容、类型或声明哈希不一致。"""

    failure_reason = "receipt_integrity_failed"


class ReceiptConflictError(RuntimeError):
    """目标路径已有另一份不可变回执。"""

    failure_reason = "receipt_identity_conflict"


class SelectionReceiptMismatchError(RuntimeError):
    """测试回放请求未绑定当前冻结选择。"""

    failure_reason = "selection_receipt_mismatch"


class ReplayAlreadyRunningError(RuntimeError):
    """同一冻结回放已有另一个进程正在执行。"""

    failure_reason = "replay_already_running"


class ReplayInfrastructureError(RuntimeError):
    """唯一允许同一 selection_hash 重试的设施故障。"""

    def __init__(self, reason: InfrastructureFailureReason) -> None:
        if reason not in _INFRASTRUCTURE_FAILURE_REASONS:
            raise ValueError(f"未知基础设施失败原因: {reason}")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class RecordIdentity:
    """不含任何指标的记录路径与内容身份。"""

    record_id: str
    data_path: str
    data_sha256: str
    reference_path: str
    reference_sha256: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.record_id, "record_id")
        _require_nonempty_string(self.data_path, "data_path")
        _require_sha256(self.data_sha256, "data_sha256")
        _require_nonempty_string(self.reference_path, "reference_path")
        _require_sha256(self.reference_sha256, "reference_sha256")


@dataclass(frozen=True)
class SearchBudgetEvidence:
    lane_unique_budget: int
    requested_global_unique_budget: int
    actual_global_unique_count: int
    requested_neighborhood_budget: int
    actual_neighborhood_count: int

    def __post_init__(self) -> None:
        values = (
            self.lane_unique_budget,
            self.requested_global_unique_budget,
            self.actual_global_unique_count,
            self.requested_neighborhood_budget,
            self.actual_neighborhood_count,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("预算与实际数量必须是非负整数")
        if self.lane_unique_budget <= 0:
            raise ValueError("lane_unique_budget 必须大于零")
        if self.requested_global_unique_budget <= 0:
            raise ValueError("requested_global_unique_budget 必须大于零")
        if (
            self.actual_global_unique_count
            > self.requested_global_unique_budget
        ):
            raise ValueError("实际全局候选数不得超过请求预算")
        if self.actual_neighborhood_count > self.requested_neighborhood_budget:
            raise ValueError("实际邻域数不得超过请求预算")


@dataclass(frozen=True)
class TrainingMetricEvidence:
    eligible: bool
    common_window_counts: tuple[int, int]
    common_window_sha256s: tuple[str, str]
    worst_train_mae_bpm: float
    mean_train_mae_bpm: float
    nonharm_deltas_bpm: tuple[float, float]

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise ValueError("eligible 必须是布尔值")
        if (
            len(self.common_window_counts) != 2
            or any(
                type(value) is not int or value <= 0
                for value in self.common_window_counts
            )
        ):
            raise ValueError("必须提供两条训练记录的正共同窗口数")
        if len(self.common_window_sha256s) != 2:
            raise ValueError("必须提供两条训练记录的共同窗口哈希")
        for value in self.common_window_sha256s:
            _require_sha256(value, "common_window_sha256")
        _require_finite_float(
            self.worst_train_mae_bpm,
            "worst_train_mae_bpm",
        )
        _require_finite_float(
            self.mean_train_mae_bpm,
            "mean_train_mae_bpm",
        )
        if len(self.nonharm_deltas_bpm) != 2:
            raise ValueError("必须提供两条训练记录的非伤害差值")
        for value in self.nonharm_deltas_bpm:
            _require_finite_float(value, "nonharm_delta_bpm")


@dataclass(frozen=True)
class NeighborhoodEvidence:
    status: Literal["not_required", "complete"]
    reviewed_neighbor_count: int
    support_ratio: float
    has_cliff: bool
    truncated_center_count: int

    def __post_init__(self) -> None:
        if self.status not in ("not_required", "complete"):
            raise ValueError("未知邻域证据状态")
        if (
            type(self.reviewed_neighbor_count) is not int
            or self.reviewed_neighbor_count < 0
            or type(self.truncated_center_count) is not int
            or self.truncated_center_count < 0
        ):
            raise ValueError("邻域计数必须是非负整数")
        _require_finite_float(self.support_ratio, "support_ratio")
        if not 0.0 <= self.support_ratio <= 1.0:
            raise ValueError("support_ratio 必须位于 [0, 1]")
        if type(self.has_cliff) is not bool:
            raise ValueError("has_cliff 必须是布尔值")


@dataclass(frozen=True)
class SelectionEvidence:
    """训练接口的完整事实；留出记录只有身份，没有结果指标。"""

    experiment_name: str
    arm: str
    scene: str
    fold: int
    code_commit: str
    code_dirty: bool
    training_records: tuple[RecordIdentity, RecordIdentity]
    heldout_record: RecordIdentity
    space_name: str
    space_sha256: str
    metric_contract_version: str
    study_identities: tuple[str, ...]
    budget: SearchBudgetEvidence
    selected_candidate_id: str
    selected_requested_params: Mapping[str, Any]
    selected_actual_params: Mapping[str, Any]
    selected_fixed_params: Mapping[str, Any]
    training_metrics: TrainingMetricEvidence
    neighborhood_evidence: NeighborhoodEvidence
    candidate_history_sha256: str
    evidence_level: EvidenceLevel = "development_reuse_pilot"
    selected_diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("experiment_name", self.experiment_name),
            ("arm", self.arm),
            ("scene", self.scene),
            ("code_commit", self.code_commit),
            ("space_name", self.space_name),
            ("metric_contract_version", self.metric_contract_version),
            ("selected_candidate_id", self.selected_candidate_id),
        ):
            _require_nonempty_string(value, name)
        if type(self.fold) is not int or self.fold < 0:
            raise ValueError("fold 必须是非负整数")
        if type(self.code_dirty) is not bool:
            raise ValueError("code_dirty 必须是布尔值")
        if (
            type(self.training_records) is not tuple
            or len(self.training_records) != 2
            or any(
                not isinstance(record, RecordIdentity)
                for record in self.training_records
            )
            or not isinstance(self.heldout_record, RecordIdentity)
        ):
            raise ValueError("共享参数回执必须恰好包含两条训练记录")
        record_ids = {
            self.training_records[0].record_id,
            self.training_records[1].record_id,
            self.heldout_record.record_id,
        }
        if len(record_ids) != 3:
            raise ValueError("两条训练记录与留出记录必须互不相同")
        data_hashes = {
            self.training_records[0].data_sha256,
            self.training_records[1].data_sha256,
            self.heldout_record.data_sha256,
        }
        if len(data_hashes) != 3:
            raise ValueError("训练记录与留出记录不得指向相同数据内容")
        _require_sha256(self.space_sha256, "space_sha256")
        expected_study_count = 1 if self.arm == "K2" else 4
        if (
            type(self.study_identities) is not tuple
            or len(self.study_identities) != expected_study_count
            or any(
                type(value) is not str or not value
                for value in self.study_identities
            )
        ):
            if self.arm == "K2":
                raise ValueError(
                    "K2 必须包含一个完整枚举证据身份"
                )
            raise ValueError(
                "必须包含三个 seed lane 与 fill 的 study 身份"
            )
        if not isinstance(self.budget, SearchBudgetEvidence):
            raise ValueError("budget 必须是 SearchBudgetEvidence")
        if not isinstance(self.training_metrics, TrainingMetricEvidence):
            raise ValueError(
                "training_metrics 必须是 TrainingMetricEvidence"
            )
        if not isinstance(self.neighborhood_evidence, NeighborhoodEvidence):
            raise ValueError(
                "neighborhood_evidence 必须是 NeighborhoodEvidence"
            )
        _require_sha256(
            self.candidate_history_sha256,
            "candidate_history_sha256",
        )
        if self.evidence_level != "development_reuse_pilot":
            raise ValueError("本阶段证据等级固定为 development_reuse_pilot")
        for field_name in (
            "selected_requested_params",
            "selected_actual_params",
            "selected_fixed_params",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{field_name} 必须是非空对象")
            object.__setattr__(self, field_name, _freeze_json(value))
        if not isinstance(self.selected_diagnostics, Mapping):
            raise ValueError("selected_diagnostics 必须是对象")
        object.__setattr__(
            self,
            "selected_diagnostics",
            _freeze_json(self.selected_diagnostics),
        )


@dataclass(frozen=True)
class SelectionReceipt:
    schema_version: str
    selection_hash: str
    evidence: SelectionEvidence


@dataclass(frozen=True)
class ReplayIdentity:
    """测试回放必须精确匹配选择回执中预先冻结的留出记录。"""

    heldout_record: RecordIdentity
    reference_groups_order: tuple[str, ...] = ("HF", "ACC")

    def __post_init__(self) -> None:
        if self.reference_groups_order != ("HF", "ACC"):
            raise ValueError("冻结回放 reference group 顺序固定为 HF/ACC")


@dataclass(frozen=True)
class FrozenReplayContext:
    """测试求解器能看到的全部冻结参数与留出记录身份。"""

    selection_hash: str
    candidate_id: str
    requested_params: Mapping[str, Any]
    actual_params: Mapping[str, Any]
    fixed_params: Mapping[str, Any]
    heldout_record: RecordIdentity
    reference_groups_order: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "requested_params",
            "actual_params",
            "fixed_params",
        ):
            object.__setattr__(self, name, _freeze_json(getattr(self, name)))


@dataclass(frozen=True)
class FrozenReplayOutcome:
    status: ReplayOutcomeStatus
    metrics: Mapping[str, Any]
    artifact_sha256s: Mapping[str, str]
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in ("success", "invalid"):
            raise ValueError("回放回调只能返回 success 或 invalid")
        _validate_terminal_payload(
            status=self.status,
            metrics=self.metrics,
            artifact_sha256s=self.artifact_sha256s,
            failure_reason=self.failure_reason,
        )
        object.__setattr__(self, "metrics", _freeze_json(self.metrics))
        object.__setattr__(
            self,
            "artifact_sha256s",
            _freeze_json(self.artifact_sha256s),
        )

    @classmethod
    def success(
        cls,
        *,
        metrics: Mapping[str, Any],
        artifact_sha256s: Mapping[str, str],
    ) -> FrozenReplayOutcome:
        return cls(
            status="success",
            metrics=metrics,
            artifact_sha256s=artifact_sha256s,
        )

    @classmethod
    def invalid(cls, reason: str) -> FrozenReplayOutcome:
        _require_nonempty_string(reason, "failure_reason")
        return cls(
            status="invalid",
            metrics={},
            artifact_sha256s={},
            failure_reason=reason,
        )


@dataclass(frozen=True)
class FrozenReplayReceipt:
    schema_version: str
    replay_hash: str
    selection_hash: str
    replay_identity: ReplayIdentity
    status: ReplayReceiptStatus
    metrics: Mapping[str, Any]
    artifact_sha256s: Mapping[str, str]
    failure_reason: str

    def __post_init__(self) -> None:
        _validate_terminal_payload(
            status=self.status,
            metrics=self.metrics,
            artifact_sha256s=self.artifact_sha256s,
            failure_reason=self.failure_reason,
        )
        object.__setattr__(self, "metrics", _freeze_json(self.metrics))
        object.__setattr__(
            self,
            "artifact_sha256s",
            _freeze_json(self.artifact_sha256s),
        )


def freeze_selection(
    path: Path | str,
    evidence: SelectionEvidence,
) -> SelectionReceipt:
    """写入训练侧不可变选择回执；相同内容幂等，不同内容拒绝覆盖。"""

    target = Path(path)
    payload = _selection_evidence_payload(evidence)
    selection_hash = _sha256(payload)
    receipt = SelectionReceipt(
        schema_version=SELECTION_RECEIPT_SCHEMA_VERSION,
        selection_hash=selection_hash,
        evidence=evidence,
    )
    if target.exists():
        existing = load_selection_receipt(target)
        if existing != receipt:
            raise ReceiptConflictError(f"选择回执已冻结: {target}")
        return existing
    try:
        _atomic_create_json(target, _selection_receipt_payload(receipt))
    except FileExistsError:
        existing = load_selection_receipt(target)
        if existing != receipt:
            raise ReceiptConflictError(f"选择回执已冻结: {target}") from None
        return existing
    return receipt


def load_selection_receipt(path: Path | str) -> SelectionReceipt:
    payload = _read_json(Path(path))
    _require_exact_keys(
        payload,
        {"schema_version", "selection_hash", "evidence"},
        "选择回执",
    )
    if _require_string(payload, "schema_version") != (
        SELECTION_RECEIPT_SCHEMA_VERSION
    ):
        raise ReceiptIntegrityError("选择回执 schema_version 不匹配")
    try:
        evidence = _selection_evidence_from_payload(
            _require_mapping(payload, "evidence")
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ReceiptIntegrityError):
            raise
        raise ReceiptIntegrityError("选择回执字段验证失败") from exc
    actual_hash = _require_string(payload, "selection_hash")
    _require_sha256(actual_hash, "selection_hash")
    expected_hash = _sha256(_selection_evidence_payload(evidence))
    if actual_hash != expected_hash:
        raise ReceiptIntegrityError("选择回执 selection_hash 校验失败")
    return SelectionReceipt(
        schema_version=SELECTION_RECEIPT_SCHEMA_VERSION,
        selection_hash=actual_hash,
        evidence=evidence,
    )


def load_replay_receipt(
    path: Path | str,
) -> FrozenReplayReceipt:
    """读取并完整校验一个终态冻结回放回执。"""

    return _load_replay_receipt(Path(path))


def replay_frozen_selection(
    *,
    receipt_path: Path | str,
    expected_selection_hash: str,
    replay_identity: ReplayIdentity,
    replay_receipt_path: Path | str,
    replay: Callable[[FrozenReplayContext], FrozenReplayOutcome],
) -> FrozenReplayReceipt:
    """只用回执中的冻结参数回放留出记录，禁止回放时重新选参。"""

    target = Path(replay_receipt_path)
    with _exclusive_replay(target.with_name(f".{target.name}.lock")):
        return _replay_frozen_selection_locked(
            receipt_path=receipt_path,
            expected_selection_hash=expected_selection_hash,
            replay_identity=replay_identity,
            target=target,
            replay=replay,
        )


def _replay_frozen_selection_locked(
    *,
    receipt_path: Path | str,
    expected_selection_hash: str,
    replay_identity: ReplayIdentity,
    target: Path,
    replay: Callable[[FrozenReplayContext], FrozenReplayOutcome],
) -> FrozenReplayReceipt:
    selection = load_selection_receipt(receipt_path)
    if selection.selection_hash != expected_selection_hash:
        raise SelectionReceiptMismatchError(
            "请求的 selection_hash 与冻结回执不一致"
        )
    if replay_identity.heldout_record != selection.evidence.heldout_record:
        raise SelectionReceiptMismatchError(
            "回放留出记录与选择回执冻结 fold 不一致"
        )
    identity_payload = _replay_identity_payload(
        selection_hash=selection.selection_hash,
        identity=replay_identity,
    )
    if target.exists():
        existing = _load_replay_receipt(target)
        if (
            existing.selection_hash != selection.selection_hash
            or _replay_identity_payload(
                selection_hash=existing.selection_hash,
                identity=existing.replay_identity,
            )
            != identity_payload
        ):
            raise ReceiptConflictError(f"测试回放回执身份冲突: {target}")
        if existing.status != "infrastructure_failed":
            return existing

    evidence = selection.evidence
    context = FrozenReplayContext(
        selection_hash=selection.selection_hash,
        candidate_id=evidence.selected_candidate_id,
        requested_params=evidence.selected_requested_params,
        actual_params=evidence.selected_actual_params,
        fixed_params=evidence.selected_fixed_params,
        heldout_record=evidence.heldout_record,
        reference_groups_order=replay_identity.reference_groups_order,
    )
    try:
        outcome = replay(context)
        if not isinstance(outcome, FrozenReplayOutcome):
            raise TypeError("回放回调必须返回 FrozenReplayOutcome")
        status: ReplayReceiptStatus = outcome.status
        metrics = outcome.metrics
        artifacts = outcome.artifact_sha256s
        failure_reason = outcome.failure_reason
    except ReplayInfrastructureError as exc:
        status = "infrastructure_failed"
        metrics = {}
        artifacts = {}
        failure_reason = exc.reason
    except Exception as exc:  # noqa: BLE001 - 固化未知算法异常，禁止重试旁路
        status = "invalid"
        metrics = {}
        artifacts = {}
        failure_reason = f"unclassified_replay_exception:{type(exc).__name__}"

    receipt = _build_replay_receipt(
        selection_hash=selection.selection_hash,
        replay_identity=replay_identity,
        status=status,
        metrics=metrics,
        artifact_sha256s=artifacts,
        failure_reason=failure_reason,
    )
    _atomic_write_json(target, _replay_receipt_payload(receipt))
    return receipt


def _build_replay_receipt(
    *,
    selection_hash: str,
    replay_identity: ReplayIdentity,
    status: ReplayReceiptStatus,
    metrics: Mapping[str, Any],
    artifact_sha256s: Mapping[str, str],
    failure_reason: str,
) -> FrozenReplayReceipt:
    if status not in ("success", "invalid", "infrastructure_failed"):
        raise ReceiptIntegrityError("未知回放回执状态")
    if (
        status == "infrastructure_failed"
        and failure_reason not in _INFRASTRUCTURE_FAILURE_REASONS
    ):
        raise ReceiptIntegrityError("回放回执含未知基础设施失败原因")
    try:
        _validate_terminal_payload(
            status=status,
            metrics=metrics,
            artifact_sha256s=artifact_sha256s,
            failure_reason=failure_reason,
        )
    except ValueError as exc:
        raise ReceiptIntegrityError("回放终态载荷不一致") from exc
    content = {
        **_replay_identity_payload(
            selection_hash=selection_hash,
            identity=replay_identity,
        ),
        "status": status,
        "metrics": _json_ready(metrics),
        "artifact_sha256s": _json_ready(artifact_sha256s),
        "failure_reason": failure_reason,
    }
    return FrozenReplayReceipt(
        schema_version=REPLAY_RECEIPT_SCHEMA_VERSION,
        replay_hash=_sha256(content),
        selection_hash=selection_hash,
        replay_identity=replay_identity,
        status=status,
        metrics=metrics,
        artifact_sha256s=artifact_sha256s,
        failure_reason=failure_reason,
    )


def _load_replay_receipt(path: Path) -> FrozenReplayReceipt:
    payload = _read_json(path)
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "replay_hash",
            "selection_hash",
            "replay_identity",
            "status",
            "metrics",
            "artifact_sha256s",
            "failure_reason",
        },
        "测试回放回执",
    )
    if _require_string(payload, "schema_version") != (
        REPLAY_RECEIPT_SCHEMA_VERSION
    ):
        raise ReceiptIntegrityError("测试回放 schema_version 不匹配")
    selection_hash = _require_string(payload, "selection_hash")
    _require_sha256(selection_hash, "selection_hash")
    try:
        identity = _replay_identity_from_payload(
            _require_mapping(payload, "replay_identity")
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ReceiptIntegrityError):
            raise
        raise ReceiptIntegrityError("回放身份字段验证失败") from exc
    status = _require_literal(
        payload,
        "status",
        {"success", "invalid", "infrastructure_failed"},
    )
    try:
        receipt = _build_replay_receipt(
            selection_hash=selection_hash,
            replay_identity=identity,
            status=status,
            metrics=_require_mapping(payload, "metrics"),
            artifact_sha256s=_require_string_mapping(
                payload,
                "artifact_sha256s",
            ),
            failure_reason=_require_string(
                payload,
                "failure_reason",
                allow_empty=True,
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ReceiptIntegrityError):
            raise
        raise ReceiptIntegrityError("回放回执字段验证失败") from exc
    replay_hash = _require_string(payload, "replay_hash")
    _require_sha256(replay_hash, "replay_hash")
    if replay_hash != receipt.replay_hash:
        raise ReceiptIntegrityError("测试回放 replay_hash 校验失败")
    return receipt


def _selection_receipt_payload(
    receipt: SelectionReceipt,
) -> dict[str, Any]:
    return {
        "schema_version": receipt.schema_version,
        "selection_hash": receipt.selection_hash,
        "evidence": _selection_evidence_payload(receipt.evidence),
    }


def _selection_evidence_payload(
    evidence: SelectionEvidence,
) -> dict[str, Any]:
    payload = {
        "experiment_name": evidence.experiment_name,
        "arm": evidence.arm,
        "scene": evidence.scene,
        "fold": evidence.fold,
        "code_commit": evidence.code_commit,
        "code_dirty": evidence.code_dirty,
        "training_records": [
            _record_identity_payload(record)
            for record in evidence.training_records
        ],
        "heldout_record": _record_identity_payload(
            evidence.heldout_record
        ),
        "space_name": evidence.space_name,
        "space_sha256": evidence.space_sha256,
        "metric_contract_version": evidence.metric_contract_version,
        "study_identities": list(evidence.study_identities),
        "budget": {
            "lane_unique_budget": evidence.budget.lane_unique_budget,
            "requested_global_unique_budget": (
                evidence.budget.requested_global_unique_budget
            ),
            "actual_global_unique_count": (
                evidence.budget.actual_global_unique_count
            ),
            "requested_neighborhood_budget": (
                evidence.budget.requested_neighborhood_budget
            ),
            "actual_neighborhood_count": (
                evidence.budget.actual_neighborhood_count
            ),
        },
        "selected_candidate_id": evidence.selected_candidate_id,
        "selected_requested_params": _json_ready(
            evidence.selected_requested_params
        ),
        "selected_actual_params": _json_ready(
            evidence.selected_actual_params
        ),
        "selected_fixed_params": _json_ready(
            evidence.selected_fixed_params
        ),
        "training_metrics": {
            "eligible": evidence.training_metrics.eligible,
            "common_window_counts": list(
                evidence.training_metrics.common_window_counts
            ),
            "common_window_sha256s": list(
                evidence.training_metrics.common_window_sha256s
            ),
            "worst_train_mae_bpm": (
                evidence.training_metrics.worst_train_mae_bpm
            ),
            "mean_train_mae_bpm": (
                evidence.training_metrics.mean_train_mae_bpm
            ),
            "nonharm_deltas_bpm": list(
                evidence.training_metrics.nonharm_deltas_bpm
            ),
        },
        "neighborhood_evidence": {
            "status": evidence.neighborhood_evidence.status,
            "reviewed_neighbor_count": (
                evidence.neighborhood_evidence.reviewed_neighbor_count
            ),
            "support_ratio": evidence.neighborhood_evidence.support_ratio,
            "has_cliff": evidence.neighborhood_evidence.has_cliff,
            "truncated_center_count": (
                evidence.neighborhood_evidence.truncated_center_count
            ),
        },
        "candidate_history_sha256": evidence.candidate_history_sha256,
        "evidence_level": evidence.evidence_level,
    }
    if evidence.selected_diagnostics:
        payload["selected_diagnostics"] = _json_ready(
            evidence.selected_diagnostics
        )
    return payload


def _selection_evidence_from_payload(
    payload: Mapping[str, Any],
) -> SelectionEvidence:
    _require_keys_with_optional(
        payload,
        {
            "experiment_name",
            "arm",
            "scene",
            "fold",
            "code_commit",
            "code_dirty",
            "training_records",
            "heldout_record",
            "space_name",
            "space_sha256",
            "metric_contract_version",
            "study_identities",
            "budget",
            "selected_candidate_id",
            "selected_requested_params",
            "selected_actual_params",
            "selected_fixed_params",
            "training_metrics",
            "neighborhood_evidence",
            "candidate_history_sha256",
            "evidence_level",
        },
        optional={"selected_diagnostics"},
        label="选择证据",
    )
    records = _require_sequence(payload, "training_records")
    if len(records) != 2:
        raise ReceiptIntegrityError("training_records 必须恰好有两条")
    budget = _require_mapping(payload, "budget")
    metrics = _require_mapping(payload, "training_metrics")
    neighborhood = _require_mapping(payload, "neighborhood_evidence")
    return SelectionEvidence(
        experiment_name=_require_string(payload, "experiment_name"),
        arm=_require_string(payload, "arm"),
        scene=_require_string(payload, "scene"),
        fold=_require_int(payload, "fold"),
        code_commit=_require_string(payload, "code_commit"),
        code_dirty=_require_bool(payload, "code_dirty"),
        training_records=(
            _record_identity_from_payload(_as_mapping(records[0])),
            _record_identity_from_payload(_as_mapping(records[1])),
        ),
        heldout_record=_record_identity_from_payload(
            _require_mapping(payload, "heldout_record")
        ),
        space_name=_require_string(payload, "space_name"),
        space_sha256=_require_string(payload, "space_sha256"),
        metric_contract_version=_require_string(
            payload,
            "metric_contract_version",
        ),
        study_identities=_nonempty_string_tuple(
            payload,
            "study_identities",
        ),
        budget=SearchBudgetEvidence(
            lane_unique_budget=_require_int(
                budget,
                "lane_unique_budget",
            ),
            requested_global_unique_budget=_require_int(
                budget,
                "requested_global_unique_budget",
            ),
            actual_global_unique_count=_require_int(
                budget,
                "actual_global_unique_count",
            ),
            requested_neighborhood_budget=_require_int(
                budget,
                "requested_neighborhood_budget",
            ),
            actual_neighborhood_count=_require_int(
                budget,
                "actual_neighborhood_count",
            ),
        ),
        selected_candidate_id=_require_string(
            payload,
            "selected_candidate_id",
        ),
        selected_requested_params=_require_mapping(
            payload,
            "selected_requested_params",
        ),
        selected_actual_params=_require_mapping(
            payload,
            "selected_actual_params",
        ),
        selected_fixed_params=_require_mapping(
            payload,
            "selected_fixed_params",
        ),
        selected_diagnostics=(
            _require_mapping(payload, "selected_diagnostics")
            if "selected_diagnostics" in payload
            else {}
        ),
        training_metrics=TrainingMetricEvidence(
            eligible=_require_bool(metrics, "eligible"),
            common_window_counts=_two_int_tuple(
                metrics,
                "common_window_counts",
            ),
            common_window_sha256s=_two_string_tuple(
                metrics,
                "common_window_sha256s",
            ),
            worst_train_mae_bpm=_require_float(
                metrics,
                "worst_train_mae_bpm",
            ),
            mean_train_mae_bpm=_require_float(
                metrics,
                "mean_train_mae_bpm",
            ),
            nonharm_deltas_bpm=_two_float_tuple(
                metrics,
                "nonharm_deltas_bpm",
            ),
        ),
        neighborhood_evidence=NeighborhoodEvidence(
            status=_require_literal(
                neighborhood,
                "status",
                {"not_required", "complete"},
            ),
            reviewed_neighbor_count=_require_int(
                neighborhood,
                "reviewed_neighbor_count",
            ),
            support_ratio=_require_float(
                neighborhood,
                "support_ratio",
            ),
            has_cliff=_require_bool(neighborhood, "has_cliff"),
            truncated_center_count=_require_int(
                neighborhood,
                "truncated_center_count",
            ),
        ),
        candidate_history_sha256=_require_string(
            payload,
            "candidate_history_sha256",
        ),
        evidence_level=_require_literal(
            payload,
            "evidence_level",
            {"development_reuse_pilot"},
        ),
    )


def _record_identity_payload(record: RecordIdentity) -> dict[str, str]:
    return {
        "record_id": record.record_id,
        "data_path": record.data_path,
        "data_sha256": record.data_sha256,
        "reference_path": record.reference_path,
        "reference_sha256": record.reference_sha256,
    }


def _record_identity_from_payload(
    payload: Mapping[str, Any],
) -> RecordIdentity:
    _require_exact_keys(
        payload,
        {
            "record_id",
            "data_path",
            "data_sha256",
            "reference_path",
            "reference_sha256",
        },
        "记录身份",
    )
    return RecordIdentity(
        record_id=_require_string(payload, "record_id"),
        data_path=_require_string(payload, "data_path"),
        data_sha256=_require_string(payload, "data_sha256"),
        reference_path=_require_string(payload, "reference_path"),
        reference_sha256=_require_string(payload, "reference_sha256"),
    )


def _replay_receipt_payload(
    receipt: FrozenReplayReceipt,
) -> dict[str, Any]:
    return {
        "schema_version": receipt.schema_version,
        "replay_hash": receipt.replay_hash,
        "selection_hash": receipt.selection_hash,
        "replay_identity": _replay_identity_only_payload(
            receipt.replay_identity
        ),
        "status": receipt.status,
        "metrics": _json_ready(receipt.metrics),
        "artifact_sha256s": _json_ready(receipt.artifact_sha256s),
        "failure_reason": receipt.failure_reason,
    }


def _replay_identity_payload(
    *,
    selection_hash: str,
    identity: ReplayIdentity,
) -> dict[str, Any]:
    return {
        "selection_hash": selection_hash,
        "replay_identity": _replay_identity_only_payload(identity),
    }


def _replay_identity_only_payload(
    identity: ReplayIdentity,
) -> dict[str, Any]:
    return {
        "heldout_record": _record_identity_payload(
            identity.heldout_record
        ),
        "reference_groups_order": list(identity.reference_groups_order),
    }


def _replay_identity_from_payload(
    payload: Mapping[str, Any],
) -> ReplayIdentity:
    _require_exact_keys(
        payload,
        {"heldout_record", "reference_groups_order"},
        "回放身份",
    )
    return ReplayIdentity(
        heldout_record=_record_identity_from_payload(
            _require_mapping(payload, "heldout_record")
        ),
        reference_groups_order=_two_string_tuple(
            payload,
            "reference_groups_order",
        ),
    )


@contextmanager
def _exclusive_replay(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except OSError as exc:
            raise ReplayAlreadyRunningError(
                f"冻结回放正在执行: {lock_path.parent}"
            ) from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _require_finite_numbers(value: Any) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _require_finite_numbers(nested)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _require_finite_numbers(nested)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("回执数值必须有限")


def _validate_terminal_payload(
    *,
    status: ReplayReceiptStatus,
    metrics: Mapping[str, Any],
    artifact_sha256s: Mapping[str, str],
    failure_reason: str,
) -> None:
    if status == "success":
        if failure_reason:
            raise ValueError("成功回放不得携带 failure_reason")
        if not metrics:
            raise ValueError("成功回放必须携带指标")
        if set(artifact_sha256s) != {"hf", "reset_fft", "acc"}:
            raise ValueError("成功回放必须包含 HF/reset FFT/ACC 产物")
    else:
        if not failure_reason:
            raise ValueError("失败回放必须携带 failure_reason")
        if metrics or artifact_sha256s:
            raise ValueError("失败回放不得携带指标或产物")
    _require_finite_numbers(metrics)
    for name, value in artifact_sha256s.items():
        _require_nonempty_string(name, "artifact_name")
        _require_sha256(value, f"artifact_sha256s[{name}]")


def _freeze_json(value: Any) -> Any:
    ready = _json_ready(value)
    if isinstance(ready, dict):
        return MappingProxyType(
            {key: _freeze_json(nested) for key, nested in ready.items()}
        )
    if isinstance(ready, list):
        return tuple(_freeze_json(item) for item in ready)
    return ready


def _sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        raise ReceiptIntegrityError(
            f"{label} 字段不匹配: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _require_keys_with_optional(
    payload: Mapping[str, Any],
    required: set[str],
    *,
    optional: set[str],
    label: str,
) -> None:
    actual = set(payload)
    allowed = required | optional
    if not required <= actual or not actual <= allowed:
        raise ReceiptIntegrityError(
            f"{label} 字段不匹配: missing={sorted(required - actual)}, "
            f"unknown={sorted(actual - allowed)}"
        )


def _require_mapping(
    payload: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ReceiptIntegrityError(f"{key} 必须是对象")
    return {str(name): nested for name, nested in value.items()}


def _as_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptIntegrityError("数组元素必须是对象")
    return {str(name): nested for name, nested in value.items()}


def _require_sequence(
    payload: Mapping[str, Any],
    key: str,
) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ReceiptIntegrityError(f"{key} 必须是数组")
    return value


def _require_string(
    payload: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(key)
    if type(value) is not str or (not allow_empty and not value):
        raise ReceiptIntegrityError(f"{key} 必须是字符串")
    return value


def _require_string_mapping(
    payload: Mapping[str, Any],
    key: str,
) -> dict[str, str]:
    value = _require_mapping(payload, key)
    if any(type(item) is not str for item in value.values()):
        raise ReceiptIntegrityError(f"{key} 的值必须是字符串")
    return {name: item for name, item in value.items()}


def _require_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if type(value) is not bool:
        raise ReceiptIntegrityError(f"{key} 必须是布尔值")
    return value


def _require_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise ReceiptIntegrityError(f"{key} 必须是整数")
    return value


def _require_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ReceiptIntegrityError(f"{key} 必须是数值")
    result = float(value)
    if not math.isfinite(result):
        raise ReceiptIntegrityError(f"{key} 必须有限")
    return result


def _require_literal(
    payload: Mapping[str, Any],
    key: str,
    allowed: set[str],
):
    value = _require_string(payload, key)
    if value not in allowed:
        raise ReceiptIntegrityError(f"{key} 值不在冻结词汇中")
    return value


def _four_string_tuple(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[str, str, str, str]:
    values = _string_list(payload, key, expected_length=4)
    return (values[0], values[1], values[2], values[3])


def _nonempty_string_tuple(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[str, ...]:
    values = _require_sequence(payload, key)
    if not values or any(
        type(value) is not str or not value
        for value in values
    ):
        raise ReceiptIntegrityError(
            f"{key} 必须是非空字符串数组"
        )
    return tuple(values)


def _two_string_tuple(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[str, str]:
    values = _string_list(payload, key, expected_length=2)
    return (values[0], values[1])


def _two_int_tuple(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[int, int]:
    values = _require_sequence(payload, key)
    if len(values) != 2 or any(type(value) is not int for value in values):
        raise ReceiptIntegrityError(f"{key} 必须是两个整数")
    return (values[0], values[1])


def _two_float_tuple(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[float, float]:
    values = _require_sequence(payload, key)
    if len(values) != 2:
        raise ReceiptIntegrityError(f"{key} 必须是两个数值")
    converted = []
    for value in values:
        if type(value) not in (int, float) or isinstance(value, bool):
            raise ReceiptIntegrityError(f"{key} 必须是两个数值")
        converted.append(float(value))
    return (converted[0], converted[1])


def _string_list(
    payload: Mapping[str, Any],
    key: str,
    *,
    expected_length: int,
) -> list[str]:
    values = _require_sequence(payload, key)
    if len(values) != expected_length or any(
        type(value) is not str or not value for value in values
    ):
        raise ReceiptIntegrityError(
            f"{key} 必须是 {expected_length} 个非空字符串"
        )
    return values


def _require_nonempty_string(value: Any, field_name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} 必须是非空字符串")


def _require_sha256(value: Any, field_name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} 必须是 64 位小写 SHA-256")


def _require_finite_float(value: Any, field_name: str) -> None:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是数值")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} 必须有限")


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
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("回执数值必须有限")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"回执包含不支持的类型: {type(value).__name__}")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
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


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
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
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptIntegrityError(f"无法读取回执: {path}") from exc
    if not isinstance(payload, dict):
        raise ReceiptIntegrityError("回执根节点必须是对象")
    return payload
