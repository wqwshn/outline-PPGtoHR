"""LYX 恢复/滤波档位实验的身份、预算与人工门控。

本模块不执行求解。它只负责在任何新求解发生前建立可机读的失败关闭边界。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from .phase2_experiment_io import (
    atomic_write_json,
    file_sha256,
    read_json,
)
from .recovery_contracts import (
    RECOVERY_PREFLIGHT_HASH_FIELDS,
)
from .recovery_contracts import (
    canonical_sha256 as _canonical_sha256,
)
from .recovery_contracts import (
    require_sha256 as _require_sha256,
)

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


@dataclass(frozen=True)
class FrozenExperimentContractHashes:
    """The seven contracts that must agree before any formal solve."""

    metric_contract_hash: str
    spectral_gate_contract_hash: str
    recovery_candidate_registry_hash: str
    recovery_selection_contract_hash: str
    penalty_registry_hash: str
    filter_profile_design_rule_hash: str
    budget_contract_hash: str

    def __post_init__(self) -> None:
        for name in RECOVERY_PREFLIGHT_HASH_FIELDS:
            _require_sha256(name, getattr(self, name))

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in RECOVERY_PREFLIGHT_HASH_FIELDS}


def validate_recovery_experiment_preflight(
    *,
    expected: FrozenExperimentContractHashes,
    actual: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless every frozen experiment contract is present and equal."""

    actual_keys = set(actual)
    required_keys = set(RECOVERY_PREFLIGHT_HASH_FIELDS)
    missing = sorted(required_keys - actual_keys)
    if missing:
        raise GovernanceError(f"preflight_contract_missing:{','.join(missing)}")
    unexpected = sorted(actual_keys - required_keys)
    if unexpected:
        raise GovernanceError(f"preflight_contract_unexpected:{','.join(unexpected)}")
    expected_hashes = expected.to_dict()
    verified: dict[str, str] = {}
    for name in RECOVERY_PREFLIGHT_HASH_FIELDS:
        value = str(actual[name])
        try:
            _require_sha256(name, value)
        except ValueError as exc:
            raise GovernanceError(f"preflight_contract_invalid:{name}") from exc
        if value != expected_hashes[name]:
            raise GovernanceError(f"preflight_contract_mismatch:{name}")
        verified[name] = value
    receipt = {
        "receipt_version": "lyx_recovery_preflight_contract_receipt_v1",
        "status": "preflight_contracts_verified",
        "required_contract_count": len(RECOVERY_PREFLIGHT_HASH_FIELDS),
        "contracts": verified,
    }
    receipt["preflight_sha256"] = _canonical_sha256(receipt)
    return receipt


def _filesystem_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt":
        if resolved.startswith("\\\\?\\"):
            return resolved
        if resolved.startswith("\\\\"):
            return "\\\\?\\UNC\\" + resolved[2:]
        return "\\\\?\\" + resolved
    return resolved


def _read_bytes(path: Path) -> bytes:
    with open(_filesystem_path(path), "rb") as stream:
        return stream.read()


@dataclass(frozen=True)
class AttemptIdentity:
    """一个唯一的 solver/config/metric/data/record 求解身份。"""

    solver_hash: str
    config_hash: str
    metric_contract_hash: str
    evaluation_hash: str
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
            "evaluation_hash",
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

    @property
    def cache_identity_sha256(self) -> str:
        return self.sha256

    def to_identity_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_identity_dict(),
            "identity_sha256": self.sha256,
            "cache_identity_sha256": self.cache_identity_sha256,
        }


@dataclass(frozen=True)
class AttemptToken:
    identity_sha256: str
    attempt_number: int
    token: str


@dataclass(frozen=True)
class CacheEvidence:
    """绑定完整求解身份与结果哈希的缓存命中回执。"""

    path: Path
    result_path: Path
    receipt_sha256: str
    identity_sha256: str
    result_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", self.path.resolve())
        object.__setattr__(
            self,
            "result_path",
            self.result_path.resolve(),
        )
        for name in (
            "receipt_sha256",
            "identity_sha256",
            "result_sha256",
        ):
            _require_sha256(name, getattr(self, name))

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        expected_identity: AttemptIdentity,
        trusted_cache_root: Path,
    ) -> CacheEvidence:
        path = path.resolve()
        trusted_cache_root = trusted_cache_root.resolve()
        if not path.is_relative_to(trusted_cache_root):
            raise GovernanceError(f"cache_evidence_outside_trusted_root:{path}")
        raw = _read_bytes(path)
        receipt_hash = hashlib.sha256(raw).hexdigest()
        try:
            payload = json.loads(raw.decode("utf-8"))
            identity_hash = payload["identity_sha256"]
            declared_result_path = Path(payload["result_path"])
            result_hash = payload["result_sha256"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise GovernanceError(f"invalid_cache_evidence:{path}") from error
        result_path = (
            declared_result_path
            if declared_result_path.is_absolute()
            else path.parent / declared_result_path
        ).resolve()
        if not result_path.is_relative_to(trusted_cache_root):
            raise GovernanceError(f"cache_result_outside_trusted_root:{result_path}")
        result_raw = _read_bytes(result_path)
        if hashlib.sha256(result_raw).hexdigest() != result_hash:
            raise GovernanceError(f"cache_result_hash_mismatch:{result_path}")
        try:
            result_payload = json.loads(result_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GovernanceError(f"invalid_cache_result:{result_path}") from error
        if (
            not isinstance(result_payload, dict)
            or identity_hash != expected_identity.sha256
            or result_payload.get("identity") != expected_identity.to_dict()
            or result_payload.get("status") != "complete"
            or result_payload.get("valid") is not True
            or result_payload.get("producer") != "content_addressed_solver_cache_v1"
        ):
            raise GovernanceError(f"cache_result_contract_mismatch:{result_path}")
        return cls(
            path=path,
            result_path=result_path,
            receipt_sha256=receipt_hash,
            identity_sha256=identity_hash,
            result_sha256=result_hash,
        )

    def to_dict(self, *, trusted_cache_root: Path) -> dict[str, str]:
        root = trusted_cache_root.resolve()
        if not self.path.is_relative_to(root) or not self.result_path.is_relative_to(root):
            raise GovernanceError("cache_evidence_outside_trusted_root")
        return {
            "path": str(self.path.relative_to(root)),
            "result_path": str(self.result_path.relative_to(root)),
            "receipt_sha256": self.receipt_sha256,
            "identity_sha256": self.identity_sha256,
            "result_sha256": self.result_sha256,
        }


@dataclass(frozen=True)
class BudgetContract:
    """按阶段冻结的唯一身份及最坏尝试数合同。"""

    stage_unique_limits: Mapping[str, int]
    max_unique_identities: int
    max_attempts: int
    retry_limit: int
    contract_version: str = "lyx_recovery_filter_budget_v1"
    normal_unique_identity_limit: int | None = None
    supplemental_stage: str | None = None
    stage_attempt_kinds: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        frozen_limits = MappingProxyType(dict(self.stage_unique_limits))
        object.__setattr__(self, "stage_unique_limits", frozen_limits)
        kinds = self.stage_attempt_kinds
        if kinds is None:
            kinds = {
                stage: ("diagnostic" if "diagnostic" in stage else "formal")
                for stage in frozen_limits
            }
        frozen_kinds = MappingProxyType(dict(kinds))
        object.__setattr__(self, "stage_attempt_kinds", frozen_kinds)
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
        if set(self.stage_attempt_kinds) != set(self.stage_unique_limits):
            raise ValueError("stage_attempt_kinds_must_cover_all_stages")
        if any(kind not in _ATTEMPT_KINDS for kind in self.stage_attempt_kinds.values()):
            raise ValueError("invalid_stage_attempt_kind")
        if (
            self.supplemental_stage is not None
            and self.supplemental_stage not in self.stage_unique_limits
        ):
            raise ValueError("supplemental_stage_not_budgeted")

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
            supplemental_stage="fold_replay",
            max_unique_identities=684,
            max_attempts=1368,
            retry_limit=1,
        )

    @classmethod
    def approved_v2(cls) -> BudgetContract:
        """Return the user-approved contract with one bounded filter audit lane."""
        return cls(
            stage_unique_limits={
                "fixed_lower_bound_diagnostic": 60,
                "historical_recovery_ab": 24,
                "recovery_sentinel": 108,
                "filter_profile_stability_audit": 32,
                "penalty_interaction": 288,
                "current_role_matrix": 96,
                "rollback_backup_matrix": 96,
                "fold_replay": 12,
            },
            normal_unique_identity_limit=704,
            supplemental_stage="fold_replay",
            max_unique_identities=716,
            max_attempts=1432,
            retry_limit=1,
            contract_version="lyx_recovery_filter_budget_v2",
            stage_attempt_kinds={
                "fixed_lower_bound_diagnostic": "diagnostic",
                "historical_recovery_ab": "formal",
                "recovery_sentinel": "formal",
                "filter_profile_stability_audit": "diagnostic",
                "penalty_interaction": "formal",
                "current_role_matrix": "formal",
                "rollback_backup_matrix": "formal",
                "fold_replay": "formal",
            },
        )

    @classmethod
    def approved_v3(cls) -> BudgetContract:
        """Return the contract with eight bounded replacement diagnostics."""
        return cls(
            stage_unique_limits={
                "fixed_lower_bound_diagnostic": 60,
                "historical_recovery_ab": 24,
                "recovery_sentinel": 108,
                "filter_profile_stability_audit": 40,
                "penalty_interaction": 288,
                "current_role_matrix": 96,
                "rollback_backup_matrix": 96,
                "fold_replay": 12,
            },
            normal_unique_identity_limit=712,
            supplemental_stage="fold_replay",
            max_unique_identities=724,
            max_attempts=1448,
            retry_limit=1,
            contract_version="lyx_recovery_filter_budget_v3",
            stage_attempt_kinds={
                "fixed_lower_bound_diagnostic": "diagnostic",
                "historical_recovery_ab": "formal",
                "recovery_sentinel": "formal",
                "filter_profile_stability_audit": "diagnostic",
                "penalty_interaction": "formal",
                "current_role_matrix": "formal",
                "rollback_backup_matrix": "formal",
                "fold_replay": "formal",
            },
        )

    @classmethod
    def approved_v4(cls) -> BudgetContract:
        """Return the contract for the 24-identity frozen-gate supplement."""

        prior = cls.approved_v3()
        return cls(
            stage_unique_limits={
                **prior.stage_unique_limits,
                "filter_profile_stability_audit": 64,
            },
            normal_unique_identity_limit=736,
            supplemental_stage=prior.supplemental_stage,
            max_unique_identities=748,
            max_attempts=1496,
            retry_limit=prior.retry_limit,
            contract_version="lyx_recovery_filter_budget_v4",
            stage_attempt_kinds=prior.stage_attempt_kinds,
        )

    @classmethod
    def approved_v5(cls) -> BudgetContract:
        """Return the contract for eight allowlisted rate-normalization probes."""

        prior = cls.approved_v4()
        return cls(
            stage_unique_limits={
                **prior.stage_unique_limits,
                "filter_profile_rate_normalization_exploration": 8,
            },
            normal_unique_identity_limit=744,
            supplemental_stage=prior.supplemental_stage,
            max_unique_identities=756,
            max_attempts=1512,
            retry_limit=prior.retry_limit,
            contract_version="lyx_recovery_filter_budget_v5",
            stage_attempt_kinds={
                **prior.stage_attempt_kinds,
                "filter_profile_rate_normalization_exploration": (
                    "exploration"
                ),
            },
        )

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "stage_unique_limits": dict(self.stage_unique_limits),
            "normal_unique_identity_limit": self.normal_unique_identity_limit,
            "supplemental_stage": self.supplemental_stage,
            "stage_attempt_kinds": dict(self.stage_attempt_kinds),
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


@contextmanager
def _exclusive_registry_lock(path: Path) -> Iterator[None]:
    """用系统文件锁串行化更新；进程退出时由系统自动释放。"""

    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.touch(exist_ok=True)
    deadline = time.monotonic() + 10.0
    handle = lock_path.open("r+b")
    if lock_path.stat().st_size == 0:
        handle.write(b"\0")
        handle.flush()
    acquired = False
    if os.name == "nt":
        import msvcrt

        while not acquired:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError as error:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise GovernanceError(f"attempt_registry_lock_timeout:{lock_path}") from error
                time.sleep(0.05)
    else:
        import fcntl

        while not acquired:
            try:
                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                acquired = True
            except BlockingIOError as error:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise GovernanceError(f"attempt_registry_lock_timeout:{lock_path}") from error
                time.sleep(0.05)
    try:
        yield
    finally:
        if os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


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
        self.trusted_cache_root = (path.parent / "solver_cache").resolve()
        self._entries = entries

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        budget_contract: BudgetContract,
        exploration_registry: ExplorationRegistry,
    ) -> AttemptRegistry:
        os.makedirs(_filesystem_path(path.parent), exist_ok=True)
        with _exclusive_registry_lock(path):
            if path.exists():
                raise GovernanceError(f"attempt_registry_already_exists:{path}")
            registry = cls(
                path,
                budget_contract=budget_contract,
                exploration_registry=exploration_registry,
                entries={},
            )
            registry._write_entries({})
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
        registry_version = payload.get("registry_version")
        if registry_version not in {
            "lyx_recovery_attempt_registry_v2",
            "lyx_recovery_attempt_registry_v3",
        }:
            raise GovernanceError("attempt_registry_version_mismatch")
        if payload.get("budget_contract_sha256") != budget_contract.sha256:
            raise GovernanceError("budget_contract_mismatch")
        if payload.get("exploration_registry_sha256") != exploration_registry.sha256:
            raise GovernanceError("exploration_registry_mismatch")
        if payload.get("trusted_cache_root_relative") != "solver_cache":
            raise GovernanceError("trusted_cache_root_mismatch")
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, dict):
            raise GovernanceError("invalid_attempt_registry_entries")
        entries = deepcopy(raw_entries)
        if registry_version == "lyx_recovery_attempt_registry_v2":
            if payload.get("summary") != cls._legacy_summary_v2(entries):
                raise GovernanceError("attempt_registry_summary_mismatch")
            for entry in entries.values():
                entry["cache_hit_count"] = (
                    len(entry.get("cache_evidence", []))
                    if not entry.get("attempts")
                    else 0
                )
        registry = cls(
            path,
            budget_contract=budget_contract,
            exploration_registry=exploration_registry,
            entries=entries,
        )
        registry._validate_entries(entries)
        if (
            registry_version == "lyx_recovery_attempt_registry_v3"
            and payload.get("summary") != registry._summary(entries)
        ):
            raise GovernanceError("attempt_registry_summary_mismatch")
        return registry

    def migrate_to(
        self,
        path: Path,
        *,
        budget_contract: BudgetContract,
        amendment_request: BudgetAmendmentRequest,
        authorization_receipt: Mapping[str, Any] | None,
        new_identities: Sequence[AttemptIdentity] = (),
        target_exploration_registry: ExplorationRegistry | None = None,
        exploration_amendment_request: (
            ExplorationBudgetAmendmentRequest | None
        ) = None,
        finalize_staging: Callable[[Path, AttemptRegistry], None] | None = None,
    ) -> AttemptRegistry:
        """Clone the complete ledger and cache into one expanded budget contract."""

        validate_budget_amendment_authorization(
            amendment_request,
            receipt=authorization_receipt,
        )
        expected_stage_limit = (
            self.budget_contract.stage_unique_limits.get(amendment_request.stage, 0)
            + amendment_request.added_unique_identities
        )
        unchanged_stages = {
            stage: limit
            for stage, limit in self.budget_contract.stage_unique_limits.items()
            if stage != amendment_request.stage
        }
        target_unchanged_stages = {
            stage: limit
            for stage, limit in budget_contract.stage_unique_limits.items()
            if stage != amendment_request.stage
        }
        source_stage_kind = self.budget_contract.stage_attempt_kinds.get(amendment_request.stage)
        target_stage_kind = budget_contract.stage_attempt_kinds.get(
            amendment_request.stage
        )
        expected_stage_kind = source_stage_kind or target_stage_kind or "diagnostic"
        source_unchanged_stage_kinds = {
            stage: kind
            for stage, kind in self.budget_contract.stage_attempt_kinds.items()
            if stage != amendment_request.stage
        }
        target_unchanged_stage_kinds = {
            stage: kind
            for stage, kind in budget_contract.stage_attempt_kinds.items()
            if stage != amendment_request.stage
        }
        target_exploration = (
            self.exploration_registry
            if target_exploration_registry is None
            else target_exploration_registry
        )
        known_transitions = {
            BudgetContract.frozen_v1().sha256: BudgetContract.approved_v2().sha256,
            BudgetContract.approved_v2().sha256: BudgetContract.approved_v3().sha256,
            BudgetContract.approved_v3().sha256: BudgetContract.approved_v4().sha256,
            BudgetContract.approved_v4().sha256: BudgetContract.approved_v5().sha256,
        }
        expected_known_target = known_transitions.get(self.budget_contract.sha256)
        if (
            budget_contract.stage_unique_limits.get(amendment_request.stage) != expected_stage_limit
            or target_unchanged_stages != unchanged_stages
            or budget_contract.normal_unique_identity_limit
            != amendment_request.normal_unique_identity_limit
            or budget_contract.max_unique_identities != amendment_request.max_unique_identities
            or budget_contract.max_attempts != amendment_request.max_attempts
            or budget_contract.retry_limit != self.budget_contract.retry_limit
            or budget_contract.supplemental_stage != self.budget_contract.supplemental_stage
            or budget_contract.stage_attempt_kinds.get(amendment_request.stage)
            != expected_stage_kind
            or target_unchanged_stage_kinds != source_unchanged_stage_kinds
            or (
                expected_known_target is not None
                and budget_contract.sha256 != expected_known_target
            )
        ):
            raise GovernanceError("budget_amendment_contract_mismatch")
        new_identity_sha256 = tuple(identity.sha256 for identity in new_identities)
        if target_stage_kind == "exploration":
            if (
                exploration_amendment_request is None
                or exploration_amendment_request.stage
                != amendment_request.stage
                or exploration_amendment_request.profile_design_rule_hash
                != amendment_request.profile_design_rule_hash
                or exploration_amendment_request.record_manifest_hash
                != amendment_request.record_manifest_hash
                or exploration_amendment_request.added_unique_identities
                != amendment_request.added_unique_identities
                or exploration_amendment_request.normal_unique_identity_limit
                != amendment_request.normal_unique_identity_limit
                or exploration_amendment_request.max_unique_identities
                != amendment_request.max_unique_identities
                or exploration_amendment_request.max_attempts
                != amendment_request.max_attempts
            ):
                raise GovernanceError(
                    "exploration_amendment_request_mismatch"
                )
            validate_exploration_budget_amendment_authorization(
                exploration_amendment_request,
                receipt=authorization_receipt,
            )
            expected_allowlist = {
                *self.exploration_registry.allowed_identity_sha256,
                *new_identity_sha256,
            }
            if (
                len(new_identity_sha256)
                != amendment_request.added_unique_identities
                or len(set(new_identity_sha256)) != len(new_identity_sha256)
                or target_exploration.unique_budget
                != self.exploration_registry.unique_budget
                + amendment_request.added_unique_identities
                or set(target_exploration.allowed_identity_sha256)
                != expected_allowlist
                or len(expected_allowlist)
                != len(target_exploration.allowed_identity_sha256)
                or any(
                    identity.stage != amendment_request.stage
                    or identity.attempt_kind != "exploration"
                    for identity in new_identities
                )
            ):
                raise GovernanceError("exploration_amendment_registry_mismatch")
        elif (
            target_exploration.sha256 != self.exploration_registry.sha256
            or exploration_amendment_request is not None
        ):
            raise GovernanceError("unexpected_exploration_registry_change")
        source = self.open(
            self.path,
            budget_contract=self.budget_contract,
            exploration_registry=self.exploration_registry,
        )
        target_dir = path.parent
        if target_dir.exists():
            raise GovernanceError(f"migration_target_already_exists:{target_dir}")
        os.makedirs(_filesystem_path(target_dir.parent), exist_ok=True)
        staging = target_dir.with_name(f".{target_dir.name}.{uuid.uuid4().hex}.staging")

        try:
            os.makedirs(_filesystem_path(staging))
            if source.trusted_cache_root.exists():
                shutil.copytree(
                    _filesystem_path(source.trusted_cache_root),
                    _filesystem_path(staging / "solver_cache"),
                )
            migrated_path = staging / path.name
            migrated = type(self)(
                migrated_path,
                budget_contract=budget_contract,
                exploration_registry=target_exploration,
                entries=deepcopy(source._entries),
            )
            migrated._write_entries(migrated._entries)
            for identity in new_identities:
                migrated.register_identity(identity)
            migrated._validate_entries(migrated._entries)
            migrated._write_entries(migrated._entries)
            if finalize_staging is not None:
                finalize_staging(staging, migrated)
            os.replace(_filesystem_path(staging), _filesystem_path(target_dir))
        except Exception:
            if staging.exists():
                shutil.rmtree(_filesystem_path(staging))
            raise
        return self.open(
            target_dir / path.name,
            budget_contract=budget_contract,
            exploration_registry=target_exploration,
        )

    def register_identity(self, identity: AttemptIdentity) -> str:
        return self.register_identities((identity,))[0]

    def register_identities(
        self,
        identities: Sequence[AttemptIdentity],
    ) -> tuple[str, ...]:
        """Atomically register a complete bounded matrix or no new identity."""

        frozen = tuple(identities)

        def mutate(
            entries: dict[str, dict[str, Any]],
        ) -> tuple[str, ...]:
            return tuple(
                self._register_into_entries(entries, identity)
                for identity in frozen
            )

        return self._transaction(mutate)

    def _register_into_entries(
        self,
        entries: dict[str, dict[str, Any]],
        identity: AttemptIdentity,
    ) -> str:
        identity_hash = identity.sha256
        if identity_hash in entries:
            if entries[identity_hash].get("identity") != identity.to_dict():
                raise GovernanceError(
                    f"identity_payload_mismatch:{identity_hash}"
                )
            return identity_hash
        if identity.attempt_kind == "exploration" and (
            self.exploration_registry.unique_budget == 0
            or identity_hash
            not in self.exploration_registry.allowed_identity_sha256
        ):
            raise GovernanceError(
                f"exploration_not_authorized:{identity_hash}"
            )
        expected_kind = self.budget_contract.stage_attempt_kinds.get(
            identity.stage
        )
        if expected_kind is None:
            raise GovernanceError(f"unbudgeted_stage:{identity.stage}")
        if identity.attempt_kind != expected_kind:
            raise GovernanceError(
                "stage_attempt_kind_mismatch:"
                f"{identity.stage}:{identity.attempt_kind}"
            )
        stage_count = sum(
            entry["identity"]["stage"] == identity.stage
            for entry in entries.values()
        )
        stage_limit = self.budget_contract.stage_unique_limits[
            identity.stage
        ]
        normal_limit = self.budget_contract.normal_unique_identity_limit
        normal_count = sum(
            entry["identity"]["stage"]
            != self.budget_contract.supplemental_stage
            for entry in entries.values()
        )
        normal_overflow = (
            normal_limit is not None
            and identity.stage != self.budget_contract.supplemental_stage
            and normal_count >= normal_limit
        )
        if (
            stage_count >= stage_limit
            or len(entries) >= self.budget_contract.max_unique_identities
            or normal_overflow
        ):
            raise HumanGateRequiredError(
                "awaiting_human_budget_decision",
                f"unique_budget_exceeded:{identity.stage}",
            )
        entries[identity_hash] = {
            "identity": identity.to_dict(),
            "attempts": [],
            "cache_evidence": [],
            "cache_hit_count": 0,
            "status": "registered",
        }
        return identity_hash

    def begin_attempt(self, identity: AttemptIdentity) -> AttemptToken:
        def mutate(entries: dict[str, dict[str, Any]]) -> AttemptToken:
            identity_hash = identity.sha256
            entry = entries.get(identity_hash)
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
            total_attempts = sum(len(item["attempts"]) for item in entries.values())
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
            return token

        return self._transaction(mutate)

    def finish_attempt(
        self,
        token: AttemptToken,
        *,
        status: Literal["succeeded", "failed"],
        failure_reason: str | None = None,
    ) -> None:
        if status not in _TERMINAL_ATTEMPT_STATUSES:
            raise GovernanceError(f"invalid_attempt_status:{status}")

        def mutate(entries: dict[str, dict[str, Any]]) -> None:
            entry = entries.get(token.identity_sha256)
            if entry is None:
                raise GovernanceError(f"unknown_attempt_token_identity:{token.identity_sha256}")
            if token.attempt_number <= 0 or token.attempt_number > len(entry["attempts"]):
                raise GovernanceError("unknown_attempt_token")
            attempt = entry["attempts"][token.attempt_number - 1]
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

        self._transaction(mutate)

    def bind_cache_evidence(
        self,
        identity: AttemptIdentity,
        *,
        evidence: CacheEvidence,
    ) -> None:
        def mutate(entries: dict[str, dict[str, Any]]) -> None:
            entry = entries.get(identity.sha256)
            if entry is None:
                raise GovernanceError(f"unregistered_identity:{identity.sha256}")
            if entry["identity"] != identity.to_dict():
                raise GovernanceError(f"identity_payload_mismatch:{identity.sha256}")
            if evidence.identity_sha256 != identity.sha256:
                raise GovernanceError(f"cache_identity_mismatch:{identity.sha256}")
            if (
                CacheEvidence.from_path(
                    evidence.path,
                    expected_identity=identity,
                    trusted_cache_root=self.trusted_cache_root,
                )
                != evidence
            ):
                raise GovernanceError(f"cache_evidence_changed:{evidence.path}")
            serialized = evidence.to_dict(trusted_cache_root=self.trusted_cache_root)
            if serialized not in entry["cache_evidence"]:
                entry["cache_evidence"].append(serialized)

        self._transaction(mutate)

    def record_cache_hit(
        self,
        identity: AttemptIdentity,
        *,
        evidence: CacheEvidence,
    ) -> None:
        def mutate(entries: dict[str, dict[str, Any]]) -> None:
            entry = entries.get(identity.sha256)
            if entry is None:
                raise GovernanceError(f"unregistered_identity:{identity.sha256}")
            if entry["identity"] != identity.to_dict():
                raise GovernanceError(f"identity_payload_mismatch:{identity.sha256}")
            if evidence.identity_sha256 != identity.sha256:
                raise GovernanceError(f"cache_identity_mismatch:{identity.sha256}")
            if (
                CacheEvidence.from_path(
                    evidence.path,
                    expected_identity=identity,
                    trusted_cache_root=self.trusted_cache_root,
                )
                != evidence
            ):
                raise GovernanceError(f"cache_evidence_changed:{evidence.path}")
            serialized = evidence.to_dict(trusted_cache_root=self.trusted_cache_root)
            if serialized not in entry["cache_evidence"]:
                entry["cache_evidence"].append(serialized)
            entry["cache_hit_count"] += 1

        self._transaction(mutate)

    def execute_registered(
        self,
        identity: AttemptIdentity,
        operation: Callable[[], Any],
    ) -> Any:
        """唯一受支持的新求解入口；账本先落 running 再调用求解。"""

        token = self.begin_attempt(identity)
        try:
            result = operation()
        except Exception as error:
            self.finish_attempt(
                token,
                status="failed",
                failure_reason=f"{type(error).__name__}:{error}",
            )
            raise
        self.finish_attempt(token, status="succeeded")
        return result

    def assert_nominatable(self, identity: AttemptIdentity) -> None:
        """候选只有存在成功求解或已登记缓存证据时才可被提名。"""

        fresh = self.open(
            self.path,
            budget_contract=self.budget_contract,
            exploration_registry=self.exploration_registry,
        )
        entry = fresh._entries.get(identity.sha256)
        if entry is None:
            raise GovernanceError(f"unregistered_nomination:{identity.sha256}")
        has_success = any(attempt["status"] == "succeeded" for attempt in entry["attempts"])
        if not has_success and not entry["cache_evidence"]:
            raise GovernanceError(f"nomination_without_evidence:{identity.sha256}")

    def summary(self) -> dict[str, int]:
        fresh = self.open(
            self.path,
            budget_contract=self.budget_contract,
            exploration_registry=self.exploration_registry,
        )
        self._entries = fresh._entries
        return self._summary(self._entries)

    def rewrite_current_schema(self) -> dict[str, int]:
        """Persist a validated legacy ledger using the current accounting schema."""

        with _exclusive_registry_lock(self.path):
            fresh = self.open(
                self.path,
                budget_contract=self.budget_contract,
                exploration_registry=self.exploration_registry,
            )
            entries = deepcopy(fresh._entries)
            self._validate_entries(entries)
            self._write_entries(entries)
            self._entries = entries
        return self._summary(self._entries)

    @staticmethod
    def _summary(
        entries: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, int]:
        attempts = [attempt for entry in entries.values() for attempt in entry["attempts"]]
        cache_hit_count = sum(int(entry["cache_hit_count"]) for entry in entries.values())
        return {
            "logical_task_count": len(attempts) + cache_hit_count,
            "planned_unique_identity_count": len(entries),
            "actual_unique_run_count": sum(bool(entry["attempts"]) for entry in entries.values()),
            "cache_evidence_count": sum(
                len(entry["cache_evidence"]) for entry in entries.values()
            ),
            "cache_hit_count": cache_hit_count,
            "failed_attempt_count": sum(attempt["status"] == "failed" for attempt in attempts),
            "retry_count": sum(max(0, len(entry["attempts"]) - 1) for entry in entries.values()),
        }

    @staticmethod
    def _legacy_summary_v2(
        entries: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, int]:
        attempts = [attempt for entry in entries.values() for attempt in entry["attempts"]]
        cache_evidence_count = sum(
            len(entry["cache_evidence"]) for entry in entries.values()
        )
        return {
            "logical_task_count": len(attempts) + cache_evidence_count,
            "planned_unique_identity_count": len(entries),
            "actual_unique_run_count": sum(bool(entry["attempts"]) for entry in entries.values()),
            "cache_hit_count": cache_evidence_count,
            "failed_attempt_count": sum(attempt["status"] == "failed" for attempt in attempts),
            "retry_count": sum(max(0, len(entry["attempts"]) - 1) for entry in entries.values()),
        }

    def _transaction(self, mutate: Callable[[dict[str, dict[str, Any]]], Any]) -> Any:
        with _exclusive_registry_lock(self.path):
            fresh = self.open(
                self.path,
                budget_contract=self.budget_contract,
                exploration_registry=self.exploration_registry,
            )
            entries = deepcopy(fresh._entries)
            result = mutate(entries)
            self._validate_entries(entries)
            self._write_entries(entries)
            self._entries = entries
            return result

    def _write_entries(
        self,
        entries: dict[str, dict[str, Any]],
    ) -> None:
        atomic_write_json(
            self.path,
            {
                "registry_version": "lyx_recovery_attempt_registry_v3",
                "budget_contract_sha256": self.budget_contract.sha256,
                "exploration_registry_sha256": (self.exploration_registry.sha256),
                "trusted_cache_root_relative": "solver_cache",
                "entries": entries,
                "summary": self._summary(entries),
            },
        )

    def _validate_entries(
        self,
        entries: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if len(entries) > self.budget_contract.max_unique_identities:
            raise GovernanceError("unique_budget_exceeded_in_registry")
        total_attempts = 0
        stage_counts: dict[str, int] = {}
        normal_count = 0
        for identity_hash, entry in entries.items():
            try:
                identity_payload = dict(entry["identity"])
                stored_hash = identity_payload.pop("identity_sha256")
                cache_hash = identity_payload.pop("cache_identity_sha256")
                identity = AttemptIdentity(**identity_payload)
                attempts = entry["attempts"]
                cache_evidence = entry["cache_evidence"]
                cache_hit_count = entry["cache_hit_count"]
                entry_status = entry["status"]
            except (KeyError, TypeError, ValueError) as error:
                raise GovernanceError(f"invalid_attempt_registry_entry:{identity_hash}") from error
            if (
                identity_hash != identity.sha256
                or stored_hash != identity_hash
                or cache_hash != identity.cache_identity_sha256
            ):
                raise GovernanceError(f"attempt_identity_hash_mismatch:{identity_hash}")
            expected_kind = self.budget_contract.stage_attempt_kinds.get(identity.stage)
            if expected_kind != identity.attempt_kind:
                raise GovernanceError(f"stage_attempt_kind_mismatch:{identity.stage}")
            if identity.attempt_kind == "exploration" and (
                identity_hash not in self.exploration_registry.allowed_identity_sha256
            ):
                raise GovernanceError(f"exploration_not_authorized:{identity_hash}")
            stage_counts[identity.stage] = stage_counts.get(identity.stage, 0) + 1
            if identity.stage != self.budget_contract.supplemental_stage:
                normal_count += 1
            if not isinstance(attempts, list) or not isinstance(cache_evidence, list):
                raise GovernanceError(f"invalid_attempt_registry_entry:{identity_hash}")
            if (
                not isinstance(cache_hit_count, int)
                or isinstance(cache_hit_count, bool)
                or cache_hit_count < 0
                or (cache_hit_count > 0 and not cache_evidence)
            ):
                raise GovernanceError(f"invalid_cache_hit_count:{identity_hash}")
            self._validate_attempt_state(
                identity_hash,
                attempts,
                entry_status,
            )
            for evidence_payload in cache_evidence:
                try:
                    receipt_relative = Path(evidence_payload["path"])
                    result_relative = Path(evidence_payload["result_path"])
                    if (
                        receipt_relative.is_absolute()
                        or result_relative.is_absolute()
                        or ".." in receipt_relative.parts
                        or ".." in result_relative.parts
                    ):
                        raise ValueError("cache evidence paths must be relative")
                    evidence = CacheEvidence(
                        path=self.trusted_cache_root / receipt_relative,
                        result_path=self.trusted_cache_root / result_relative,
                        receipt_sha256=evidence_payload["receipt_sha256"],
                        identity_sha256=evidence_payload["identity_sha256"],
                        result_sha256=evidence_payload["result_sha256"],
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise GovernanceError(f"invalid_cache_evidence:{identity_hash}") from error
                if evidence.identity_sha256 != identity_hash:
                    raise GovernanceError(f"cache_identity_mismatch:{identity_hash}")
                if (
                    CacheEvidence.from_path(
                        evidence.path,
                        expected_identity=identity,
                        trusted_cache_root=self.trusted_cache_root,
                    )
                    != evidence
                ):
                    raise GovernanceError(f"cache_evidence_changed:{evidence.path}")
            total_attempts += len(attempts)
        for stage, count in stage_counts.items():
            if count > self.budget_contract.stage_unique_limits[stage]:
                raise GovernanceError(f"stage_budget_exceeded_in_registry:{stage}")
        normal_limit = self.budget_contract.normal_unique_identity_limit
        if normal_limit is not None and normal_count > normal_limit:
            raise GovernanceError("normal_budget_exceeded_in_registry")
        if total_attempts > self.budget_contract.max_attempts:
            raise GovernanceError("attempt_budget_exceeded_in_registry")

    def _validate_attempt_state(
        self,
        identity_hash: str,
        attempts: Sequence[Mapping[str, Any]],
        entry_status: Any,
    ) -> None:
        if len(attempts) > self.budget_contract.retry_limit + 1:
            raise GovernanceError(f"invalid_attempt_state:{identity_hash}:retry_limit")
        tokens: set[str] = set()
        for index, attempt in enumerate(attempts, start=1):
            token = attempt.get("token")
            status = attempt.get("status")
            reason = attempt.get("failure_reason")
            if (
                attempt.get("attempt_number") != index
                or not isinstance(token, str)
                or not token
                or token in tokens
                or status not in {"running", *_TERMINAL_ATTEMPT_STATUSES}
                or (status == "failed" and not reason)
                or (status != "failed" and reason is not None)
                or (status in {"running", "succeeded"} and index != len(attempts))
            ):
                raise GovernanceError(f"invalid_attempt_state:{identity_hash}:{index}")
            tokens.add(token)
        expected_status = attempts[-1]["status"] if attempts else "registered"
        if entry_status != expected_status:
            raise GovernanceError(f"invalid_attempt_state:{identity_hash}:entry_status")


@dataclass(frozen=True)
class RecordSource:
    """一个记录在读取屏障中的冻结物理来源。"""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        _require_sha256("record_source", self.sha256)
        object.__setattr__(self, "path", self.path.resolve())

    @classmethod
    def from_path(cls, path: Path) -> RecordSource:
        return cls(path=path, sha256=file_sha256(path))


@dataclass(frozen=True)
class DataRoleManifest:
    """一次场景内折叠的训练记录与开发内审计目标角色。"""

    fold_id: str
    training_record_ids: tuple[str, ...]
    audit_target_record_id: str
    record_sources: Mapping[str, RecordSource]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "record_sources",
            MappingProxyType(dict(self.record_sources)),
        )
        if not self.fold_id or not self.audit_target_record_id:
            raise ValueError("fold_and_target_must_not_be_empty")
        if not self.training_record_ids:
            raise ValueError("training_records_must_not_be_empty")
        if len(set(self.training_record_ids)) != len(self.training_record_ids):
            raise ValueError("duplicate_training_record")
        if self.audit_target_record_id in self.training_record_ids:
            raise ValueError("audit_target_cannot_be_training_record")
        expected_records = {
            *self.training_record_ids,
            self.audit_target_record_id,
        }
        if set(self.record_sources) != expected_records:
            raise ValueError("record_sources_must_match_fold_records")


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
        if not self._target_identity_fields <= _AUDIT_TARGET_IDENTITY_FIELDS:
            raise ValueError("target_identity_fields_cannot_expand_whitelist")
        self._accesses: list[dict[str, Any]] = []

    def read_json_fields(
        self,
        *,
        record_id: str,
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
        source = self.manifest.record_sources[record_id]
        raw = source.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source.sha256:
            raise GovernanceError(f"record_source_hash_mismatch:{record_id}")
        role = "audit_target" if record_id == self.manifest.audit_target_record_id else "training"
        if role == "audit_target":
            denied = sorted(set(requested) - self._target_identity_fields)
            if denied:
                raise GovernanceError("audit_target_field_denied:" + ",".join(denied))
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GovernanceError(f"record_source_invalid_json:{record_id}") from error
        if not isinstance(payload, dict):
            raise GovernanceError(f"record_source_root_must_be_object:{record_id}")
        missing = [field for field in requested if field not in payload]
        if missing:
            raise GovernanceError("requested_field_missing:" + ",".join(missing))
        result = {field: payload[field] for field in requested}
        self._accesses.append(
            {
                "record_id": record_id,
                "role": role,
                "path": str(source.path),
                "path_sha256": source.sha256,
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
            "record_sources": {
                record_id: {
                    "path": str(source.path),
                    "sha256": source.sha256,
                }
                for record_id, source in self.manifest.record_sources.items()
            },
            "accesses": list(self._accesses),
        }

    def write_receipt(self, path: Path) -> dict[str, Any]:
        receipt = self.receipt()
        atomic_write_json(path, receipt)
        return receipt


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


@dataclass(frozen=True)
class BudgetAmendmentRequest:
    """Exact identity of the bounded, non-BO filter stability budget amendment."""

    stage: str
    profile_design_rule_hash: str
    record_manifest_hash: str
    added_unique_identities: int
    normal_unique_identity_limit: int
    max_unique_identities: int
    max_attempts: int

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("budget_amendment_stage_must_not_be_empty")
        for name in ("profile_design_rule_hash", "record_manifest_hash"):
            _require_sha256(name, getattr(self, name))
        for name in (
            "added_unique_identities",
            "normal_unique_identity_limit",
            "max_unique_identities",
            "max_attempts",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name}_must_be_positive")


@dataclass(frozen=True)
class ExplorationBudgetAmendmentRequest:
    """Exact bounded exploration amendment for mechanism-derived coordinates."""

    stage: str
    profile_design_rule_hash: str
    record_manifest_hash: str
    added_unique_identities: int
    normal_unique_identity_limit: int
    max_unique_identities: int
    max_attempts: int
    exploration_unique_budget: int

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("budget_amendment_stage_must_not_be_empty")
        for name in ("profile_design_rule_hash", "record_manifest_hash"):
            _require_sha256(name, getattr(self, name))
        for name in (
            "added_unique_identities",
            "normal_unique_identity_limit",
            "max_unique_identities",
            "max_attempts",
            "exploration_unique_budget",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name}_must_be_positive")
        if self.exploration_unique_budget != self.added_unique_identities:
            raise ValueError("exploration_budget_must_equal_added_identities")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "profile_design_rule_hash": self.profile_design_rule_hash,
            "record_manifest_hash": self.record_manifest_hash,
            "added_unique_identities": self.added_unique_identities,
            "normal_unique_identity_limit": self.normal_unique_identity_limit,
            "max_unique_identities": self.max_unique_identities,
            "max_attempts": self.max_attempts,
            "attempt_kind": "exploration",
            "exploration_unique_budget": self.exploration_unique_budget,
        }


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
    required = {"decision_state", "approved_at", "approved_by"}
    missing = sorted(required - receipt.keys())
    if missing or not all(receipt.get(field) for field in required):
        raise GovernanceError("authorization_missing_fields:" + ",".join(missing))
    if receipt["decision_state"] != state:
        raise GovernanceError(f"authorization_state_mismatch:{receipt['decision_state']}:{state}")
    if not isinstance(receipt["approved_at"], str) or not isinstance(receipt["approved_by"], str):
        raise GovernanceError("authorization_metadata_must_be_strings")
    return dict(receipt)


def validate_budget_amendment_authorization(
    request: BudgetAmendmentRequest,
    *,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the exact non-BO budget amendment approved by the user."""

    state = "awaiting_human_budget_decision"
    validated = validate_human_gate(state=state, receipt=receipt)
    expected = asdict(request)
    required = {
        "approved",
        "decision_state",
        *expected.keys(),
        "independent_bo_authorized",
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
    if validated["independent_bo_authorized"] is not False:
        raise GovernanceError("independent_bo_must_remain_unauthorized")
    return validated


def validate_exploration_budget_amendment_authorization(
    request: ExplorationBudgetAmendmentRequest,
    *,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the exact allowlisted exploration budget approved by the user."""

    state = "awaiting_human_budget_decision"
    validated = validate_human_gate(state=state, receipt=receipt)
    expected = request.to_dict()
    required = {
        "approved",
        "decision_state",
        *expected.keys(),
        "independent_bo_authorized",
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
        raise GovernanceError(
            "authorization_identity_mismatch:" + ",".join(mismatched)
        )
    if validated["independent_bo_authorized"] is not False:
        raise GovernanceError("independent_bo_must_remain_unauthorized")
    return validated


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
        "decision_state",
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
    os.makedirs(_filesystem_path(output_dir.parent), exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.staging")
    budget = BudgetContract.frozen_v1()
    exploration = ExplorationRegistry.zero_budget_v1()
    try:
        os.makedirs(_filesystem_path(staging))
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
        (staging / ".attempt_registry.json.lock").unlink(missing_ok=True)
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
