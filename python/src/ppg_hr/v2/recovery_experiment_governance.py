"""LYX 恢复/滤波档位实验的身份、预算与人工门控。

本模块不执行求解。它只负责在任何新求解发生前建立可机读的失败关闭边界。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    read_json,
)

_HASH_LENGTH = 64
_ATTEMPT_KINDS = frozenset({"exploration", "diagnostic", "formal"})
_TERMINAL_ATTEMPT_STATUSES = frozenset({"succeeded", "failed"})
_HUMAN_GATE_STATES = frozenset(
    {
        "awaiting_human_interaction_decision",
        "awaiting_human_budget_decision",
        "awaiting_human_independent_bo_decision",
    }
)
_AUDIT_TARGET_IDENTITY_FIELDS = frozenset(
    {
        "sample_id",
        "record_id",
        "data_sha256",
        "raw_data_sha256",
        "reference_sha256",
        "source_path",
    }
)


class GovernanceError(RuntimeError):
    """实验治理合同被违反。"""


class HumanGateRequiredError(GovernanceError):
    """流程必须暂停并等待指定人工决策。"""

    def __init__(self, state: str, message: str | None = None) -> None:
        if state not in _HUMAN_GATE_STATES:
            raise ValueError(f"unknown_human_gate_state:{state}")
        self.state = state
        super().__init__(message or state)


def _require_sha256(name: str, value: str) -> None:
    if len(value) != _HASH_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name}_must_be_lowercase_sha256")


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AttemptIdentity:
    """一个唯一的 solver/config/metric/data/record 求解身份。"""

    solver_hash: str
    config_hash: str
    metric_contract_hash: str
    data_sha256: str
    record_id: str
    stage: str
    attempt_kind: Literal["exploration", "diagnostic", "formal"]
    parent_experiment_id: str

    def __post_init__(self) -> None:
        for name in (
            "solver_hash",
            "config_hash",
            "metric_contract_hash",
            "data_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in ("record_id", "stage", "parent_experiment_id"):
            if not getattr(self, name):
                raise ValueError(f"{name}_must_not_be_empty")
        if self.attempt_kind not in _ATTEMPT_KINDS:
            raise ValueError(f"unknown_attempt_kind:{self.attempt_kind}")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "identity_sha256": self.sha256}


@dataclass(frozen=True)
class AttemptToken:
    identity_sha256: str
    attempt_number: int
    token: str


@dataclass(frozen=True)
class BudgetContract:
    """按阶段冻结的唯一身份及最坏尝试数合同。"""

    stage_unique_limits: Mapping[str, int]
    max_unique_identities: int
    max_attempts: int
    retry_limit: int
    contract_version: str = "lyx_recovery_filter_budget_v1"
    normal_unique_identity_limit: int | None = None

    def __post_init__(self) -> None:
        if not self.stage_unique_limits:
            raise ValueError("stage_unique_limits_must_not_be_empty")
        if any(
            not stage or not isinstance(limit, int) or limit < 0
            for stage, limit in self.stage_unique_limits.items()
        ):
            raise ValueError("invalid_stage_unique_limit")
        if self.max_unique_identities < 0 or self.max_attempts < 0:
            raise ValueError("budget_limits_must_be_non_negative")
        if self.retry_limit < 0:
            raise ValueError("retry_limit_must_be_non_negative")
        if sum(self.stage_unique_limits.values()) > self.max_unique_identities:
            raise ValueError("stage_limits_exceed_total_unique_budget")
        if self.max_attempts < self.max_unique_identities * (self.retry_limit + 1):
            raise ValueError("max_attempts_cannot_cover_retry_contract")
        if (
            self.normal_unique_identity_limit is not None
            and self.normal_unique_identity_limit > self.max_unique_identities
        ):
            raise ValueError("normal_limit_exceeds_absolute_limit")

    @classmethod
    def frozen_v1(cls) -> BudgetContract:
        return cls(
            stage_unique_limits={
                "fixed_lower_bound_diagnostic": 60,
                "historical_recovery_ab": 24,
                "recovery_sentinel": 108,
                "penalty_interaction": 288,
                "current_role_matrix": 96,
                "rollback_backup_matrix": 96,
                "fold_replay": 12,
            },
            normal_unique_identity_limit=672,
            max_unique_identities=684,
            max_attempts=1368,
            retry_limit=1,
        )

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "stage_unique_limits": dict(self.stage_unique_limits),
            "normal_unique_identity_limit": self.normal_unique_identity_limit,
            "max_unique_identities": self.max_unique_identities,
            "max_attempts": self.max_attempts,
            "retry_limit": self.retry_limit,
        }


@dataclass(frozen=True)
class ExplorationRegistry:
    """人工批准的探索身份白名单。"""

    unique_budget: int
    allowed_identity_sha256: tuple[str, ...] = ()
    registry_version: str = "lyx_recovery_exploration_registry_v1"

    def __post_init__(self) -> None:
        if self.unique_budget < 0:
            raise ValueError("exploration_budget_must_be_non_negative")
        if len(set(self.allowed_identity_sha256)) != len(self.allowed_identity_sha256):
            raise ValueError("duplicate_exploration_identity")
        if len(self.allowed_identity_sha256) > self.unique_budget:
            raise ValueError("exploration_allowlist_exceeds_budget")
        for value in self.allowed_identity_sha256:
            _require_sha256("exploration_identity", value)

    @classmethod
    def zero_budget_v1(cls) -> ExplorationRegistry:
        return cls(unique_budget=0)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_version": self.registry_version,
            "unique_budget": self.unique_budget,
            "allowed_identity_sha256": list(self.allowed_identity_sha256),
        }


class AttemptRegistry:
    """持久化的先登记后求解状态机。"""

    def __init__(
        self,
        path: Path,
        *,
        budget_contract: BudgetContract,
        exploration_registry: ExplorationRegistry,
        entries: dict[str, dict[str, Any]],
    ) -> None:
        self.path = path
        self.budget_contract = budget_contract
        self.exploration_registry = exploration_registry
        self._entries = entries

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        budget_contract: BudgetContract,
        exploration_registry: ExplorationRegistry,
    ) -> AttemptRegistry:
        if path.exists():
            raise GovernanceError(f"attempt_registry_already_exists:{path}")
        registry = cls(
            path,
            budget_contract=budget_contract,
            exploration_registry=exploration_registry,
            entries={},
        )
        registry._persist()
        return registry

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        budget_contract: BudgetContract,
        exploration_registry: ExplorationRegistry,
    ) -> AttemptRegistry:
        payload = read_json(path)
        if payload.get("budget_contract_sha256") != budget_contract.sha256:
            raise GovernanceError("budget_contract_mismatch")
        if payload.get("exploration_registry_sha256") != exploration_registry.sha256:
            raise GovernanceError("exploration_registry_mismatch")
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise GovernanceError("invalid_attempt_registry_entries")
        registry = cls(
            path,
            budget_contract=budget_contract,
            exploration_registry=exploration_registry,
            entries=entries,
        )
        registry._validate_entries()
        if payload.get("summary") != registry.summary():
            raise GovernanceError("attempt_registry_summary_mismatch")
        return registry

    def register_identity(self, identity: AttemptIdentity) -> str:
        identity_hash = identity.sha256
        if identity_hash in self._entries:
            return identity_hash
        if identity.attempt_kind == "exploration":
            if (
                self.exploration_registry.unique_budget == 0
                or identity_hash not in self.exploration_registry.allowed_identity_sha256
            ):
                raise GovernanceError(f"exploration_not_authorized:{identity_hash}")
        limit = self.budget_contract.stage_unique_limits.get(identity.stage)
        if limit is None:
            raise GovernanceError(f"unbudgeted_stage:{identity.stage}")
        stage_count = sum(
            entry["identity"]["stage"] == identity.stage for entry in self._entries.values()
        )
        if stage_count >= limit or len(self._entries) >= self.budget_contract.max_unique_identities:
            raise HumanGateRequiredError(
                "awaiting_human_budget_decision",
                f"unique_budget_exceeded:{identity.stage}",
            )
        self._entries[identity_hash] = {
            "identity": identity.to_dict(),
            "attempts": [],
            "cache_hits": 0,
            "status": "registered",
        }
        self._persist()
        return identity_hash

    def begin_attempt(self, identity: AttemptIdentity) -> AttemptToken:
        identity_hash = identity.sha256
        entry = self._entries.get(identity_hash)
        if entry is None:
            raise GovernanceError(f"unregistered_identity:{identity_hash}")
        if entry["identity"] != identity.to_dict():
            raise GovernanceError(f"identity_payload_mismatch:{identity_hash}")
        attempts = entry["attempts"]
        if any(attempt["status"] == "running" for attempt in attempts):
            raise GovernanceError(f"attempt_already_running:{identity_hash}")
        if any(attempt["status"] == "succeeded" for attempt in attempts):
            raise GovernanceError(f"identity_already_succeeded:{identity_hash}")
        if len(attempts) >= self.budget_contract.retry_limit + 1:
            raise GovernanceError(f"retry_limit_exceeded:{identity_hash}")
        total_attempts = sum(len(item["attempts"]) for item in self._entries.values())
        if total_attempts >= self.budget_contract.max_attempts:
            raise GovernanceError("attempt_budget_exceeded")
        token = AttemptToken(
            identity_sha256=identity_hash,
            attempt_number=len(attempts) + 1,
            token=uuid.uuid4().hex,
        )
        attempts.append(
            {
                "attempt_number": token.attempt_number,
                "token": token.token,
                "status": "running",
                "failure_reason": None,
            }
        )
        entry["status"] = "running"
        self._persist()
        return token

    def finish_attempt(
        self,
        token: AttemptToken,
        *,
        status: Literal["succeeded", "failed"],
        failure_reason: str | None = None,
    ) -> None:
        if status not in _TERMINAL_ATTEMPT_STATUSES:
            raise GovernanceError(f"invalid_attempt_status:{status}")
        entry = self._entries.get(token.identity_sha256)
        if entry is None:
            raise GovernanceError(f"unknown_attempt_token_identity:{token.identity_sha256}")
        try:
            attempt = entry["attempts"][token.attempt_number - 1]
        except IndexError as error:
            raise GovernanceError("unknown_attempt_token") from error
        if (
            attempt["token"] != token.token
            or attempt["attempt_number"] != token.attempt_number
            or attempt["status"] != "running"
        ):
            raise GovernanceError("attempt_token_mismatch")
        if status == "failed" and not failure_reason:
            raise GovernanceError("failed_attempt_requires_reason")
        attempt["status"] = status
        attempt["failure_reason"] = failure_reason
        entry["status"] = status
        self._persist()

    def record_cache_hit(self, identity: AttemptIdentity) -> None:
        entry = self._entries.get(identity.sha256)
        if entry is None:
            raise GovernanceError(f"unregistered_identity:{identity.sha256}")
        entry["cache_hits"] += 1
        self._persist()

    def summary(self) -> dict[str, int]:
        attempts = [attempt for entry in self._entries.values() for attempt in entry["attempts"]]
        return {
            "logical_task_count": len(attempts)
            + sum(entry["cache_hits"] for entry in self._entries.values()),
            "planned_unique_identity_count": len(self._entries),
            "actual_unique_run_count": sum(
                bool(entry["attempts"]) for entry in self._entries.values()
            ),
            "cache_hit_count": sum(entry["cache_hits"] for entry in self._entries.values()),
            "failed_attempt_count": sum(attempt["status"] == "failed" for attempt in attempts),
            "retry_count": sum(
                max(0, len(entry["attempts"]) - 1) for entry in self._entries.values()
            ),
        }

    def _persist(self) -> None:
        atomic_write_json(
            self.path,
            {
                "registry_version": "lyx_recovery_attempt_registry_v1",
                "budget_contract_sha256": self.budget_contract.sha256,
                "exploration_registry_sha256": (self.exploration_registry.sha256),
                "entries": self._entries,
                "summary": self.summary(),
            },
        )

    def _validate_entries(self) -> None:
        for identity_hash, entry in self._entries.items():
            try:
                identity_payload = dict(entry["identity"])
                stored_hash = identity_payload.pop("identity_sha256")
                identity = AttemptIdentity(**identity_payload)
                attempts = entry["attempts"]
                cache_hits = entry["cache_hits"]
            except (KeyError, TypeError, ValueError) as error:
                raise GovernanceError(f"invalid_attempt_registry_entry:{identity_hash}") from error
            if identity_hash != identity.sha256 or stored_hash != identity_hash:
                raise GovernanceError(f"attempt_identity_hash_mismatch:{identity_hash}")
            if not isinstance(attempts, list) or not isinstance(cache_hits, int) or cache_hits < 0:
                raise GovernanceError(f"invalid_attempt_registry_entry:{identity_hash}")


@dataclass(frozen=True)
class DataRoleManifest:
    """一次场景内折叠的训练记录与开发内审计目标角色。"""

    fold_id: str
    training_record_ids: tuple[str, ...]
    audit_target_record_id: str

    def __post_init__(self) -> None:
        if not self.fold_id or not self.audit_target_record_id:
            raise ValueError("fold_and_target_must_not_be_empty")
        if not self.training_record_ids:
            raise ValueError("training_records_must_not_be_empty")
        if len(set(self.training_record_ids)) != len(self.training_record_ids):
            raise ValueError("duplicate_training_record")
        if self.audit_target_record_id in self.training_record_ids:
            raise ValueError("audit_target_cannot_be_training_record")


class FoldReadBarrier:
    """记录实际字段读取，并阻断对审计目标性能字段的访问。"""

    def __init__(
        self,
        manifest: DataRoleManifest,
        *,
        target_identity_fields: Sequence[str] = tuple(sorted(_AUDIT_TARGET_IDENTITY_FIELDS)),
    ) -> None:
        self.manifest = manifest
        self._target_identity_fields = frozenset(target_identity_fields)
        self._accesses: list[dict[str, Any]] = []

    def read_json_fields(
        self,
        *,
        record_id: str,
        path: Path,
        fields: Sequence[str],
    ) -> dict[str, Any]:
        if record_id not in {
            *self.manifest.training_record_ids,
            self.manifest.audit_target_record_id,
        }:
            raise GovernanceError(f"record_outside_fold:{record_id}")
        requested = tuple(fields)
        if not requested:
            raise GovernanceError("empty_field_request")
        role = "audit_target" if record_id == self.manifest.audit_target_record_id else "training"
        if role == "audit_target":
            denied = sorted(set(requested) - self._target_identity_fields)
            if denied:
                raise GovernanceError("audit_target_field_denied:" + ",".join(denied))
        payload = read_json(path)
        missing = [field for field in requested if field not in payload]
        if missing:
            raise GovernanceError("requested_field_missing:" + ",".join(missing))
        result = {field: payload[field] for field in requested}
        self._accesses.append(
            {
                "record_id": record_id,
                "role": role,
                "path": str(path.resolve()),
                "path_sha256": file_sha256(path),
                "fields": list(requested),
            }
        )
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "receipt_version": "lyx_fold_read_barrier_v1",
            "fold_id": self.manifest.fold_id,
            "training_record_ids": list(self.manifest.training_record_ids),
            "audit_target_record_id": (self.manifest.audit_target_record_id),
            "algorithm_level_holdout": False,
            "evidence_class": "development_replay_audit",
            "accesses": list(self._accesses),
        }


@dataclass(frozen=True)
class IndependentBORequest:
    solver_hash: str
    search_space_hash: str
    metric_contract_hash: str
    seed_manifest_hash: str
    unique_budget: int

    def __post_init__(self) -> None:
        for name in (
            "solver_hash",
            "search_space_hash",
            "metric_contract_hash",
            "seed_manifest_hash",
        ):
            _require_sha256(name, getattr(self, name))
        if self.unique_budget <= 0:
            raise ValueError("independent_bo_budget_must_be_positive")


def validate_human_gate(
    *,
    state: str,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """校验一般人工门回执；缺失或未批准时保持等待状态。"""

    if state not in _HUMAN_GATE_STATES:
        raise ValueError(f"unknown_human_gate_state:{state}")
    if receipt is None or receipt.get("approved") is not True:
        raise HumanGateRequiredError(state)
    required = {"approved_at", "approved_by"}
    missing = sorted(required - receipt.keys())
    if missing or not all(receipt.get(field) for field in required):
        raise GovernanceError("authorization_missing_fields:" + ",".join(missing))
    return dict(receipt)


def validate_independent_bo_authorization(
    request: IndependentBORequest,
    *,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """严格绑定机器身份、种子清单和唯一候选预算。"""

    state = "awaiting_human_independent_bo_decision"
    validated = validate_human_gate(state=state, receipt=receipt)
    expected = asdict(request)
    required = {
        "approved",
        *expected.keys(),
        "approved_at",
        "approved_by",
    }
    missing = sorted(required - validated.keys())
    if missing:
        raise GovernanceError("authorization_missing_fields:" + ",".join(missing))
    mismatched = sorted(
        field
        for field, expected_value in expected.items()
        if validated.get(field) != expected_value
    )
    if mismatched:
        raise GovernanceError("authorization_identity_mismatch:" + ",".join(mismatched))
    return validated


def initialize_recovery_experiment_governance(
    *,
    output_dir: Path,
    parent_experiment_id: str,
) -> dict[str, Any]:
    """原子发布零运行治理骨架，不登记或执行任何求解。"""

    if not parent_experiment_id:
        raise ValueError("parent_experiment_id_must_not_be_empty")
    if output_dir.exists():
        raise GovernanceError(f"output_dir_already_exists:{output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.staging")
    budget = BudgetContract.frozen_v1()
    exploration = ExplorationRegistry.zero_budget_v1()
    try:
        staging.mkdir()
        atomic_write_json(staging / "budget_contract.json", budget.to_dict())
        atomic_write_json(
            staging / "exploration_registry.json",
            exploration.to_dict(),
        )
        AttemptRegistry.create(
            staging / "attempt_registry.json",
            budget_contract=budget,
            exploration_registry=exploration,
        )
        receipt = {
            "receipt_version": "lyx_recovery_governance_receipt_v1",
            "status": "complete",
            "parent_experiment_id": parent_experiment_id,
            "planned_unique_identity_limit": budget.max_unique_identities,
            "normal_unique_identity_limit": (budget.normal_unique_identity_limit),
            "worst_case_attempt_limit": budget.max_attempts,
            "exploration_unique_budget": exploration.unique_budget,
            "independent_bo_authorized": False,
            "human_gate_states": sorted(_HUMAN_GATE_STATES),
            "artifacts": {
                name: file_sha256(staging / name)
                for name in (
                    "attempt_registry.json",
                    "budget_contract.json",
                    "exploration_registry.json",
                )
            },
        }
        atomic_write_json(staging / "governance_receipt.json", receipt)
        os.replace(staging, output_dir)
        return receipt
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
