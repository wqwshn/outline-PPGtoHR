"""LYX BO 参数空间泛化第二阶段的冻结实验合同。

本模块为第二阶段实验提供一组可审计的公开边界。现有 GUI 使用的
``optimise_v2`` 与 ``optimise_v2_shared_params`` 不受影响。
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import sleep
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import optuna

from .reference_groups import method_label, normalise_reference_order
from .solver import V2SolverResult

SpaceName = Literal["legacy_full_v1", "legacy_reduced_v1", "physical_v1"]
METRIC_CONTRACT_VERSION = "lyx_bo_formal_metric_v1"
FORMAL_MIN_WINDOW_COUNT = 10
_FORMAL_ADAPTIVE_FILTERS = frozenset(
    {
        "lms",
        "noncausal_lms",
        "rff_lms",
        "klms",
        "as_lms",
        "volterra",
    }
)
_CANDIDATE_INVALID_FAILURE_REASONS = frozenset(
    {
        "method_identity_mismatch",
        "metric_window_contract_failed",
        "nonfinite_solver_output",
    }
)
_CACHE_NONFINITE_FLOAT_KEY = "__ppg_hr_cache_nonfinite_float_v1__"
_METHOD_IDENTITY_CONTRACT_REASONS = frozenset(
    {
        "invalid_method_identity",
        "duplicate_method_identity",
        "missing_expected_method_identity",
        "invalid_expected_method_identity",
        "missing_final_method_identity",
        "missing_reset_fft_method_identity",
        "invalid_adaptive_filter_identity",
    }
)

_LEGACY_FULL_OPTIONS: tuple[tuple[str, tuple[int | float, ...]], ...] = (
    ("fs_target", (25, 50, 100)),
    ("max_order", (8, 12, 16, 20)),
    ("lms_mu_base", (0.008, 0.010, 0.012)),
    ("smooth_win_len", (5, 7, 9)),
    ("spec_penalty_width", (0.1, 0.2, 0.3)),
    ("time_bias", (4.0, 4.5, 5.0, 5.5, 6.0)),
)
_LEGACY_REDUCED_OPTIONS: tuple[tuple[str, tuple[int | float, ...]], ...] = (
    ("fs_target", (25, 50, 100)),
    ("max_order", (8, 12, 16, 20)),
    ("lms_mu_base", (0.008, 0.010, 0.012)),
    ("spec_penalty_width", (0.1, 0.2, 0.3)),
)
_PHYSICAL_OPTIONS: tuple[tuple[str, tuple[int | float, ...]], ...] = (
    ("fs_target", (25, 50, 100)),
    ("memory_ms", (40, 80, 120, 160, 200)),
    ("mu_base", (0.006, 0.008, 0.010, 0.012, 0.016)),
    ("exclusion_half_width_bpm", (3, 6, 12, 18)),
)
_COMMON_FIXED_PARAMS: Mapping[str, Any] = MappingProxyType(
    {
        "analysis_scope": "full",
        "smooth_win_len": 5,
        "time_bias": 5.0,
        "lms_mu_min": 1e-6,
    }
)


@dataclass(frozen=True)
class BOCandidate:
    """一组可解释请求参数及其实际求解器参数。"""

    space_name: SpaceName
    candidate_id: str
    coordinate: tuple[int, ...]
    requested_params: Mapping[str, Any]
    actual_params: Mapping[str, Any]
    fixed_params: Mapping[str, Any]


@dataclass(frozen=True)
class BOSearchSpace:
    """冻结离散空间及其确定性候选顺序。"""

    name: SpaceName
    parameter_names: tuple[str, ...]
    option_values: tuple[tuple[int | float, ...], ...]
    candidates: tuple[BOCandidate, ...]


class FormalMetricContractError(ValueError):
    """窗口级正式指标无法按冻结合同计算。"""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        message = reason if not detail else f"{reason}: {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class FormalMetricResult:
    """同一组冻结窗口分母上的正式候选指标。"""

    metric_contract_version: str
    final_method: str
    reset_fft_method: str
    base_full_window_count: int
    base_motion_window_count: int
    classic_motion_window_count: int
    base_full_final_finite_count: int
    base_motion_final_finite_count: int
    base_motion_reset_fft_finite_count: int
    base_motion_common_finite_count: int
    classic_motion_final_finite_count: int
    classic_motion_reset_fft_finite_count: int
    classic_motion_common_finite_count: int
    base_full_window_sha256: str
    base_motion_window_sha256: str
    classic_motion_window_sha256: str
    full_final_mae_bpm: float
    reliable_motion_final_mae_bpm: float
    reliable_motion_reset_fft_mae_bpm: float
    classic_motion_final_mae_bpm: float
    classic_motion_reset_fft_mae_bpm: float


@dataclass(frozen=True)
class SolverCacheIdentity:
    """足以唯一识别一次候选物理求解的输入事实。"""

    data_sha256: str
    reference_sha256: str
    git_commit: str
    run_config: Mapping[str, Any]
    candidate: BOCandidate
    reference_groups_order: tuple[str, ...]


@dataclass(frozen=True)
class SolverCacheKey:
    """内容寻址键及其可审计规范化载荷。"""

    key: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateSolveOutcome:
    """一次物理求解及正式指标评价的可缓存结果。"""

    status: Literal["valid", "invalid"]
    solver_result: V2SolverResult | None = None
    formal_metrics: FormalMetricResult | None = None
    failure_reason: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status == "valid":
            if self.solver_result is None or self.formal_metrics is None:
                raise ValueError(
                    "valid 候选必须同时包含 solver_result 和 formal_metrics"
                )
            if self.failure_reason:
                raise ValueError("valid 候选不能包含 failure_reason")
            return
        if self.status != "invalid":
            raise ValueError(f"未知 CandidateSolveOutcome 状态: {self.status!r}")
        if self.failure_reason not in _CANDIDATE_INVALID_FAILURE_REASONS:
            raise ValueError(
                "failure_reason 必须使用冻结候选失败分类，"
                f"实际为 {self.failure_reason!r}"
            )
        if self.formal_metrics is not None:
            raise ValueError("invalid 候选不能包含 formal_metrics")

    @classmethod
    def valid(
        cls,
        solver_result: V2SolverResult,
        formal_metrics: FormalMetricResult,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> CandidateSolveOutcome:
        return cls(
            status="valid",
            solver_result=solver_result,
            formal_metrics=formal_metrics,
            diagnostics=MappingProxyType(dict(diagnostics or {})),
        )

    @classmethod
    def invalid(
        cls,
        failure_reason: str,
        *,
        solver_result: V2SolverResult | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> CandidateSolveOutcome:
        reason = str(failure_reason).strip()
        if reason not in _CANDIDATE_INVALID_FAILURE_REASONS:
            raise ValueError(
                "failure_reason 必须使用冻结候选失败分类，"
                f"实际为 {reason!r}"
            )
        return cls(
            status="invalid",
            solver_result=solver_result,
            failure_reason=reason,
            diagnostics=MappingProxyType(dict(diagnostics or {})),
        )


@dataclass(frozen=True)
class SolverCacheLookup:
    """一次逻辑候选对内容寻址缓存的读取结果。"""

    cache_key: str
    cache_hit: bool
    physical_solve_performed: bool
    outcome: CandidateSolveOutcome


class CachedInfrastructureError(RuntimeError):
    """缓存中已有明确的基础设施失败。"""


class CacheReservationConflictError(TimeoutError):
    """等待另一个 worker 的候选预占超时。"""


class InfrastructureSolveError(RuntimeError):
    """调用方已确认的求解基础设施故障。"""


def _process_is_demonstrably_dead(pid: Any) -> bool:
    """仅在本机明确报告进程不存在时返回真。"""

    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError as exc:
        # Windows 对不存在的 PID 返回 ERROR_INVALID_PARAMETER。
        return getattr(exc, "winerror", None) == 87
    return False


@contextmanager
def _try_exclusive_file_lock(
    lock_path: Path,
    *,
    blocking: bool = False,
) -> Iterator[bool]:
    """尝试持有一个由操作系统随进程退出自动释放的文件锁。"""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
                msvcrt.locking(handle.fileno(), mode, 1)
            else:
                import fcntl

                operation = fcntl.LOCK_EX
                if not blocking:
                    operation |= fcntl.LOCK_NB
                fcntl.flock(
                    handle.fileno(),
                    operation,
                )
            acquired = True
        except OSError:
            pass
        yield acquired
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class ContentAddressedSolverCache:
    """使用原子目录预占的本地确定性候选求解缓存。"""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def entry_state(
        self,
        cache_key: str,
    ) -> Literal["missing", "reserved", "complete", "failed"]:
        entry = self._entry_path(cache_key)
        if not entry.is_dir():
            return "missing"
        if (entry / "complete.json").is_file():
            return "complete"
        if (entry / "failed.json").is_file():
            return "failed"
        return "reserved"

    def entry_audit(self, cache_key: str) -> Mapping[str, Any]:
        """返回不包含大型数值数组的缓存状态审计。"""

        entry = self._entry_path(cache_key)
        state = self.entry_state(cache_key)
        audit: dict[str, Any] = {
            "cache_key": cache_key,
            "state": state,
        }
        reservation_path = entry / "reservation.json"
        if reservation_path.is_file():
            reservation = _read_json(reservation_path)
            audit["pid"] = reservation.get("pid")
            audit["identity"] = reservation.get("identity")
        if state == "failed":
            audit.update(_read_json(entry / "failed.json"))
        elif state == "complete":
            outcome = _cache_json_restore(_read_json(entry / "outcome.json"))
            audit["outcome_status"] = outcome.get("status")
            audit["failure_reason"] = outcome.get("failure_reason", "")
            audit["diagnostics"] = outcome.get("diagnostics", {})
        return MappingProxyType(audit)

    def audit_summary(self) -> Mapping[str, Any]:
        """汇总物理求解、命中、冲突和失败，并保留逻辑引用。"""

        event_dir = self.root / "_audit_events"
        events = (
            [
                _read_json(path)
                for path in sorted(event_dir.glob("*.json"))
                if path.is_file()
            ]
            if event_dir.is_dir()
            else []
        )
        return MappingProxyType(
            {
                "logical_request_count": sum(
                    event.get("event_type")
                    != "abandoned_reservation_recovered"
                    for event in events
                ),
                "physical_solve_count": sum(
                    bool(event.get("physical_solve_performed"))
                    for event in events
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
                "unclassified_error_count": sum(
                    event.get("event_type") == "unclassified_error"
                    for event in events
                ),
                "abandoned_reservation_recovery_count": sum(
                    event.get("event_type")
                    == "abandoned_reservation_recovered"
                    for event in events
                ),
                "events": events,
            }
        )

    def get_or_solve(
        self,
        identity: SolverCacheIdentity,
        solve: Callable[[], CandidateSolveOutcome],
        *,
        logical_reference: Mapping[str, Any] | None = None,
        wait_timeout_s: float = 60.0,
        poll_interval_s: float = 0.05,
    ) -> SolverCacheLookup:
        cache_key = build_solver_cache_key(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        entry = self._entry_path(cache_key.key)
        reservation = {
            "cache_key": cache_key.key,
            "pid": os.getpid(),
            "identity": cache_key.payload,
        }
        while True:
            if self._try_claim_entry(
                cache_key.key,
                reservation=reservation,
                logical_reference=logical_reference,
            ):
                break
            try:
                lookup = self._wait_for_existing(
                    cache_key.key,
                    logical_reference=logical_reference,
                    reservation=reservation,
                    wait_timeout_s=wait_timeout_s,
                    poll_interval_s=poll_interval_s,
                )
            except CacheReservationConflictError:
                self._record_event(
                    cache_key=cache_key.key,
                    event_type="reservation_conflict",
                    logical_reference=logical_reference,
                    cache_hit=False,
                    physical_solve_performed=False,
                )
                raise
            except CachedInfrastructureError as exc:
                self._record_event(
                    cache_key=cache_key.key,
                    event_type="infrastructure_failure",
                    logical_reference=logical_reference,
                    cache_hit=True,
                    physical_solve_performed=False,
                    message=str(exc),
                )
                raise
            if lookup is None:
                break
            self._record_event(
                cache_key=cache_key.key,
                event_type="lookup_complete",
                logical_reference=logical_reference,
                cache_hit=True,
                physical_solve_performed=False,
                outcome_status=lookup.outcome.status,
                failure_reason=lookup.outcome.failure_reason,
            )
            return lookup

        try:
            outcome = solve()
        except FormalMetricContractError as exc:
            failure_reason = _formal_metric_failure_category(exc.reason)
            outcome = CandidateSolveOutcome.invalid(
                failure_reason,
                diagnostics={
                    "formal_metric_contract_reason": exc.reason,
                    "formal_metric_contract_message": str(exc),
                },
            )
        except InfrastructureSolveError as exc:
            self._write_infrastructure_failure(
                entry,
                cache_key=cache_key.key,
                exc=exc,
            )
            self._record_event(
                cache_key=cache_key.key,
                event_type="infrastructure_failure",
                logical_reference=logical_reference,
                cache_hit=False,
                physical_solve_performed=True,
                message=str(exc),
            )
            raise
        except Exception as exc:
            self._record_event(
                cache_key=cache_key.key,
                event_type="unclassified_error",
                logical_reference=logical_reference,
                cache_hit=False,
                physical_solve_performed=True,
                message=str(exc),
                exception_type=type(exc).__name__,
            )
            raise

        try:
            if not isinstance(outcome, CandidateSolveOutcome):
                raise TypeError("solve 必须返回 CandidateSolveOutcome")
            self._publish_completed_outcome(
                entry,
                cache_key=cache_key.key,
                outcome=outcome,
            )
        except OSError as exc:
            wrapped = InfrastructureSolveError(str(exc))
            self._write_infrastructure_failure(
                entry,
                cache_key=cache_key.key,
                exc=wrapped,
            )
            self._record_event(
                cache_key=cache_key.key,
                event_type="infrastructure_failure",
                logical_reference=logical_reference,
                cache_hit=False,
                physical_solve_performed=True,
                message=str(exc),
            )
            raise wrapped from exc
        except Exception as exc:
            self._record_event(
                cache_key=cache_key.key,
                event_type="unclassified_error",
                logical_reference=logical_reference,
                cache_hit=False,
                physical_solve_performed=True,
                message=str(exc),
                exception_type=type(exc).__name__,
            )
            raise
        lookup = SolverCacheLookup(
            cache_key=cache_key.key,
            cache_hit=False,
            physical_solve_performed=True,
            outcome=outcome,
        )
        self._record_event(
            cache_key=cache_key.key,
            event_type="lookup_complete",
            logical_reference=logical_reference,
            cache_hit=False,
            physical_solve_performed=True,
            outcome_status=outcome.status,
            failure_reason=outcome.failure_reason,
        )
        return lookup

    def _write_infrastructure_failure(
        self,
        entry: Path,
        *,
        cache_key: str,
        exc: InfrastructureSolveError,
    ) -> None:
        with _try_exclusive_file_lock(
            self._claim_lock_path(cache_key),
            blocking=True,
        ) as acquired:
            if not acquired:
                raise AssertionError("阻塞缓存锁必须成功获取")
            _atomic_write_json(
                entry / "failed.json",
                {
                    "cache_key": cache_key,
                    "failure_class": "infrastructure_failure",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )

    def _publish_completed_outcome(
        self,
        entry: Path,
        *,
        cache_key: str,
        outcome: CandidateSolveOutcome,
    ) -> None:
        with _try_exclusive_file_lock(
            self._claim_lock_path(cache_key),
            blocking=True,
        ) as acquired:
            if not acquired:
                raise AssertionError("阻塞缓存锁必须成功获取")
            _write_cached_outcome(entry, outcome)
            _atomic_write_json(
                entry / "complete.json",
                {
                    "cache_key": cache_key,
                    "status": outcome.status,
                },
            )

    def _record_event(
        self,
        *,
        cache_key: str,
        event_type: str,
        logical_reference: Mapping[str, Any] | None,
        cache_hit: bool,
        physical_solve_performed: bool,
        **details: Any,
    ) -> None:
        event_dir = self.root / "_audit_events"
        event_id = uuid.uuid4().hex
        _atomic_write_json(
            event_dir / f"{event_id}.json",
            {
                "event_id": event_id,
                "event_type": event_type,
                "cache_key": cache_key,
                "logical_reference": _json_ready(logical_reference or {}),
                "cache_hit": bool(cache_hit),
                "physical_solve_performed": bool(
                    physical_solve_performed
                ),
                **_json_ready(details),
            },
        )

    def _wait_for_existing(
        self,
        cache_key: str,
        *,
        logical_reference: Mapping[str, Any] | None,
        reservation: Mapping[str, Any],
        wait_timeout_s: float,
        poll_interval_s: float,
    ) -> SolverCacheLookup | None:
        import time

        deadline = time.monotonic() + max(0.0, float(wait_timeout_s))
        while True:
            if self._try_claim_entry(
                cache_key,
                reservation=reservation,
                logical_reference=logical_reference,
            ):
                return None
            state = self.entry_state(cache_key)
            if state == "complete":
                return SolverCacheLookup(
                    cache_key=cache_key,
                    cache_hit=True,
                    physical_solve_performed=False,
                    outcome=_read_cached_outcome(self._entry_path(cache_key)),
                )
            if state == "failed":
                failure = _read_json(self._entry_path(cache_key) / "failed.json")
                raise CachedInfrastructureError(
                    str(failure.get("message", "cached infrastructure failure"))
                )
            if time.monotonic() >= deadline:
                raise CacheReservationConflictError(
                    f"等待缓存预占超时: {cache_key}"
                )
            time.sleep(max(0.001, float(poll_interval_s)))

    def _try_claim_entry(
        self,
        cache_key: str,
        *,
        reservation: Mapping[str, Any],
        logical_reference: Mapping[str, Any] | None,
    ) -> bool:
        claim_lock = self._claim_lock_path(cache_key)
        with _try_exclusive_file_lock(claim_lock) as acquired:
            if not acquired:
                return False
            entry = self._entry_path(cache_key)
            state = self.entry_state(cache_key)
            recovery_reason = ""
            owner_pid: Any = None
            abandoned: Path | None = None
            if state == "reserved":
                reservation_path = entry / "reservation.json"
                if not reservation_path.is_file():
                    recovery_reason = "missing_reservation"
                else:
                    try:
                        existing = _read_json(reservation_path)
                    except (OSError, ValueError, json.JSONDecodeError):
                        return False
                    if existing.get("cache_key") != cache_key:
                        return False
                    owner_pid = existing.get("pid")
                    if not _process_is_demonstrably_dead(owner_pid):
                        return False
                    recovery_reason = "dead_owner"
                abandoned_root = self.root / "_abandoned_reservations"
                abandoned_root.mkdir(parents=True, exist_ok=True)
                abandoned = (
                    abandoned_root / f"{cache_key}.{uuid.uuid4().hex}"
                )
                try:
                    entry.rename(abandoned)
                except OSError:
                    return False
            elif state != "missing":
                return False

            try:
                entry.mkdir()
                _atomic_write_json(entry / "reservation.json", reservation)
            except FileExistsError:
                return False
            if abandoned is not None:
                self._record_event(
                    cache_key=cache_key,
                    event_type="abandoned_reservation_recovered",
                    logical_reference=logical_reference,
                    cache_hit=False,
                    physical_solve_performed=False,
                    abandoned_entry=str(abandoned.relative_to(self.root)),
                    owner_pid=owner_pid,
                    recovery_reason=recovery_reason,
                )
            return True

    def _claim_lock_path(self, cache_key: str) -> Path:
        self._entry_path(cache_key)
        return self.root / "_claim_locks" / f"{cache_key}.lock"

    def _entry_path(self, cache_key: str) -> Path:
        if (
            len(cache_key) != 64
            or any(character not in "0123456789abcdef" for character in cache_key)
        ):
            raise ValueError("cache_key 必须是 64 位小写 SHA-256")
        return self.root / cache_key


@dataclass(frozen=True)
class SearchEvaluation:
    """一个候选反馈给 Optuna 及最终排序的冻结数值。"""

    objective: float
    constraints: tuple[float, ...] = ()
    metric_valid: bool = True
    eligible: bool = True
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.objective)):
            raise ValueError("SearchEvaluation.objective 必须有限")
        if any(not math.isfinite(float(value)) for value in self.constraints):
            raise ValueError("SearchEvaluation.constraints 必须全部有限")


@dataclass(frozen=True)
class SearchRequestContext:
    """候选求解与缓存审计所绑定的精确逻辑 trial。"""

    lane: str
    seed: int
    trial_number: int
    stage: Literal["search", "fill"]
    suggestion_index: int
    unique_index: int | None
    is_duplicate: bool


@dataclass(frozen=True)
class SearchExperimentIdentity:
    """禁止跨输入、代码或求解配置复用持久化搜索的冻结身份。"""

    input_sha256s: tuple[str, ...]
    reference_sha256s: tuple[str, ...]
    git_commit: str
    run_config: Mapping[str, Any]
    evaluation_version: str

    def __post_init__(self) -> None:
        if not self.input_sha256s or not self.reference_sha256s:
            raise ValueError("搜索身份必须包含输入与参考文件 SHA-256")
        if not self.git_commit or not self.evaluation_version:
            raise ValueError("搜索身份必须包含 commit 与评价版本")


@dataclass(frozen=True)
class SeedSearchBudget:
    """独立 seed lane 与确定性 fill 的不重复候选预算。"""

    lane_seeds: tuple[int, ...] = (42, 43, 44)
    lane_unique_budget: int = 50
    global_unique_budget: int = 150
    n_startup_trials: int = 10
    fill_seed: int = 20260724
    unique_stall_limit: int = 200
    objective_version: str = "phase2_independent_full_final_v1"
    constraints_version: str = "none_v1"

    def __post_init__(self) -> None:
        if not self.lane_seeds or len(set(self.lane_seeds)) != len(
            self.lane_seeds
        ):
            raise ValueError("lane_seeds 必须非空且不重复")
        if self.lane_unique_budget <= 0 or self.global_unique_budget <= 0:
            raise ValueError("候选预算必须为正整数")
        if self.n_startup_trials <= 0 or self.unique_stall_limit <= 0:
            raise ValueError("startup 与 stall 预算必须为正整数")


@dataclass(frozen=True)
class SearchTrialRecord:
    """一个逻辑 Optuna trial 的稳定审计行。"""

    lane: str
    seed: int
    trial_number: int
    suggestion_index: int
    unique_index: int | None
    candidate_id: str
    is_duplicate: bool
    objective: float
    constraints: tuple[float, ...]
    metric_valid: bool
    eligible: bool
    failure_reason: str
    stage: Literal["search", "fill"]


@dataclass(frozen=True)
class SeedLaneResult:
    seed: int
    history: tuple[SearchTrialRecord, ...]
    unique_candidate_ids: tuple[str, ...]

    @property
    def unique_candidate_count(self) -> int:
        return len(self.unique_candidate_ids)


@dataclass(frozen=True)
class SeedSearchResult:
    lanes: tuple[SeedLaneResult, ...]
    fill_history: tuple[SearchTrialRecord, ...]
    global_candidate_ids: tuple[str, ...]
    seed_stability_candidate_ids: tuple[str, ...]
    requested_global_unique_budget: int
    effective_global_unique_budget: int
    space_exhausted: bool

    @property
    def fill_unique_candidate_count(self) -> int:
        return sum(not row.is_duplicate for row in self.fill_history)


class StudyStateMismatchError(RuntimeError):
    """持久化 study 与当前冻结驱动配置不一致。"""


class UniqueBudgetStalledError(RuntimeError):
    """离散空间未耗尽但连续建议无法补足不同候选。"""


class SearchAlreadyRunningError(RuntimeError):
    """同一输出目录已有另一个搜索驱动器持有独占锁。"""


def run_seed_search(
    *,
    space: BOSearchSpace,
    output_dir: Path | str,
    experiment_identity: SearchExperimentIdentity,
    evaluate: Callable[
        [BOCandidate, SearchRequestContext],
        SearchEvaluation,
    ],
    budget: SeedSearchBudget,
    parallel_lanes: bool = False,
) -> SeedSearchResult:
    """运行独立 seed lane，并以独立 fill 补足全局不同候选。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive_search_directory(output / ".driver.lock"):
        return _run_seed_search_locked(
            space=space,
            output=output,
            experiment_identity=experiment_identity,
            evaluate=evaluate,
            budget=budget,
            parallel_lanes=parallel_lanes,
        )


def _run_seed_search_locked(
    *,
    space: BOSearchSpace,
    output: Path,
    experiment_identity: SearchExperimentIdentity,
    evaluate: Callable[
        [BOCandidate, SearchRequestContext],
        SearchEvaluation,
    ],
    budget: SeedSearchBudget,
    parallel_lanes: bool,
) -> SeedSearchResult:
    _validate_seed_search(space, budget)
    studies_dir = output / "studies"
    studies_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _seed_search_config_hash(
        space,
        budget,
        experiment_identity,
    )
    state_path = output / "driver_state.json"
    _ensure_search_identity(
        output / "search_identity.json",
        config_hash=config_hash,
        experiment_identity=experiment_identity,
    )
    _read_driver_state(state_path, config_hash=config_hash)
    state_lock = threading.Lock()
    effective_global_budget = min(
        budget.global_unique_budget,
        len(space.candidates),
    )
    enumerate_entire_space = (
        effective_global_budget < budget.global_unique_budget
    )

    def run_lane(seed: int) -> SeedLaneResult:
        return _run_seed_lane(
            space=space,
            studies_dir=studies_dir,
            state_path=state_path,
            state_lock=state_lock,
            config_hash=config_hash,
            seed=seed,
            unique_budget=budget.lane_unique_budget,
            n_startup_trials=budget.n_startup_trials,
            unique_stall_limit=budget.unique_stall_limit,
            objective_version=budget.objective_version,
            constraints_version=budget.constraints_version,
            evaluate=evaluate,
        )

    if parallel_lanes and len(budget.lane_seeds) > 1:
        with ThreadPoolExecutor(max_workers=len(budget.lane_seeds)) as executor:
            futures = {
                seed: executor.submit(run_lane, seed)
                for seed in budget.lane_seeds
            }
            lane_by_seed = {
                seed: future.result() for seed, future in futures.items()
            }
        lanes = tuple(lane_by_seed[seed] for seed in budget.lane_seeds)
    else:
        lanes = tuple(run_lane(seed) for seed in budget.lane_seeds)

    evaluation_by_candidate = _evaluations_from_lanes(lanes)
    seed_union = tuple(
        sorted(
            {
                candidate_id
                for lane in lanes
                for candidate_id in lane.unique_candidate_ids
            }
        )
    )
    fill_history = _run_fill_study(
        space=space,
        studies_dir=studies_dir,
        state_path=state_path,
        state_lock=state_lock,
        config_hash=config_hash,
        seed=budget.fill_seed,
        global_unique_budget=effective_global_budget,
        enumerate_entire_space=enumerate_entire_space,
        n_startup_trials=budget.n_startup_trials,
        unique_stall_limit=budget.unique_stall_limit,
        objective_version=budget.objective_version,
        constraints_version=budget.constraints_version,
        seed_union=seed_union,
        evaluation_by_candidate=evaluation_by_candidate,
        evaluate=evaluate,
    )
    global_candidate_ids = tuple(
        sorted(
            {
                *seed_union,
                *(
                    row.candidate_id
                    for row in fill_history
                    if not row.is_duplicate
                ),
            }
        )
    )
    result = SeedSearchResult(
        lanes=lanes,
        fill_history=fill_history,
        global_candidate_ids=global_candidate_ids,
        seed_stability_candidate_ids=seed_union,
        requested_global_unique_budget=budget.global_unique_budget,
        effective_global_unique_budget=effective_global_budget,
        space_exhausted=enumerate_entire_space,
    )
    _atomic_write_json(
        state_path,
        {
            "config_hash": config_hash,
            "stage": "complete",
            "lane_unique_counts": {
                f"seed_{lane.seed}": lane.unique_candidate_count
                for lane in lanes
            },
            "seed_union_candidate_ids": seed_union,
            "fill_unique_candidate_count": (
                result.fill_unique_candidate_count
            ),
            "global_candidate_ids": global_candidate_ids,
            "requested_global_unique_budget": budget.global_unique_budget,
            "effective_global_unique_budget": effective_global_budget,
            "space_exhausted": enumerate_entire_space,
            "unresolved_trials": [],
        },
    )
    return result


def _run_seed_lane(
    *,
    space: BOSearchSpace,
    studies_dir: Path,
    state_path: Path,
    state_lock: threading.Lock,
    config_hash: str,
    seed: int,
    unique_budget: int,
    n_startup_trials: int,
    unique_stall_limit: int,
    objective_version: str,
    constraints_version: str,
    evaluate: Callable[
        [BOCandidate, SearchRequestContext],
        SearchEvaluation,
    ],
) -> SeedLaneResult:
    lane = f"seed_{seed}"
    study = _open_seed_study(
        path=studies_dir / f"{lane}.sqlite3",
        name=f"phase2_{config_hash}_{lane}",
        seed=seed,
        n_startup_trials=n_startup_trials,
        config_hash=config_hash,
        objective_version=objective_version,
        constraints_version=constraints_version,
    )
    distributions = _coordinate_distributions(space)
    candidate_by_coordinate = {
        candidate.coordinate: candidate for candidate in space.candidates
    }
    completed = _study_history(study, lane=lane, seed=seed, stage="search")
    seen = {
        row.candidate_id for row in completed if not row.is_duplicate
    }
    duplicate_streak = _trailing_duplicate_count(completed)
    if len(seen) < unique_budget and duplicate_streak >= unique_stall_limit:
        raise UniqueBudgetStalledError(
            f"{lane} 连续 {duplicate_streak} 次未产生新候选"
        )

    running = sorted(
        study.get_trials(
            deepcopy=False,
            states=(optuna.trial.TrialState.RUNNING,),
        ),
        key=lambda trial: trial.number,
    )
    for frozen in running:
        candidate = _candidate_from_trial(
            frozen,
            space=space,
            candidate_by_coordinate=candidate_by_coordinate,
        )
        duplicate = candidate.candidate_id in seen
        unique_index = None if duplicate else len(seen) + 1
        _write_running_trial_state(
            state_path,
            config_hash=config_hash,
            lane=lane,
            stage="search",
            trial_number=frozen.number,
            candidate_id=candidate.candidate_id,
            state_lock=state_lock,
        )
        evaluation = evaluate(
            candidate,
            SearchRequestContext(
                lane=lane,
                seed=seed,
                trial_number=frozen.number,
                stage="search",
                suggestion_index=frozen.number + 1,
                unique_index=unique_index,
                is_duplicate=duplicate,
            ),
        )
        _finish_trial(
            study,
            trial=frozen,
            candidate=candidate,
            evaluation=evaluation,
            lane=lane,
            seed=seed,
            stage="search",
            suggestion_index=frozen.number + 1,
            unique_index=unique_index,
            is_duplicate=duplicate,
        )
        _clear_running_trial_state(
            state_path,
            config_hash=config_hash,
            lane=lane,
            trial_number=frozen.number,
            state_lock=state_lock,
        )
        if duplicate:
            duplicate_streak += 1
            if duplicate_streak >= unique_stall_limit:
                raise UniqueBudgetStalledError(
                    f"{lane} 连续 {duplicate_streak} 次未产生新候选"
                )
        else:
            seen.add(candidate.candidate_id)
            duplicate_streak = 0

    while len(seen) < unique_budget:
        trial = study.ask(fixed_distributions=distributions)
        candidate = _candidate_from_trial(
            trial,
            space=space,
            candidate_by_coordinate=candidate_by_coordinate,
        )
        duplicate = candidate.candidate_id in seen
        unique_index = None if duplicate else len(seen) + 1
        _write_running_trial_state(
            state_path,
            config_hash=config_hash,
            lane=lane,
            stage="search",
            trial_number=trial.number,
            candidate_id=candidate.candidate_id,
            state_lock=state_lock,
        )
        evaluation = evaluate(
            candidate,
            SearchRequestContext(
                lane=lane,
                seed=seed,
                trial_number=trial.number,
                stage="search",
                suggestion_index=trial.number + 1,
                unique_index=unique_index,
                is_duplicate=duplicate,
            ),
        )
        _finish_trial(
            study,
            trial=trial,
            candidate=candidate,
            evaluation=evaluation,
            lane=lane,
            seed=seed,
            stage="search",
            suggestion_index=trial.number + 1,
            unique_index=unique_index,
            is_duplicate=duplicate,
        )
        _clear_running_trial_state(
            state_path,
            config_hash=config_hash,
            lane=lane,
            trial_number=trial.number,
            state_lock=state_lock,
        )
        if duplicate:
            duplicate_streak += 1
            if duplicate_streak >= unique_stall_limit:
                raise UniqueBudgetStalledError(
                    f"{lane} 连续 {duplicate_streak} 次未产生新候选"
                )
        else:
            seen.add(candidate.candidate_id)
            duplicate_streak = 0
    history = _study_history(study, lane=lane, seed=seed, stage="search")
    return SeedLaneResult(
        seed=seed,
        history=history,
        unique_candidate_ids=tuple(
            sorted(
                {
                    row.candidate_id
                    for row in history
                    if not row.is_duplicate
                }
            )
        ),
    )


def _run_fill_study(
    *,
    space: BOSearchSpace,
    studies_dir: Path,
    state_path: Path,
    state_lock: threading.Lock,
    config_hash: str,
    seed: int,
    global_unique_budget: int,
    enumerate_entire_space: bool,
    n_startup_trials: int,
    unique_stall_limit: int,
    objective_version: str,
    constraints_version: str,
    seed_union: tuple[str, ...],
    evaluation_by_candidate: dict[str, SearchEvaluation],
    evaluate: Callable[
        [BOCandidate, SearchRequestContext],
        SearchEvaluation,
    ],
) -> tuple[SearchTrialRecord, ...]:
    if len(seed_union) >= global_unique_budget:
        return ()
    lane = "fill"
    study = _open_seed_study(
        path=studies_dir / "fill.sqlite3",
        name=f"phase2_{config_hash}_fill",
        seed=seed,
        n_startup_trials=n_startup_trials,
        config_hash=config_hash,
        objective_version=objective_version,
        constraints_version=constraints_version,
    )
    distributions = _coordinate_distributions(space)
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in space.candidates
    }
    candidate_by_coordinate = {
        candidate.coordinate: candidate for candidate in space.candidates
    }
    imported = {
        str(trial.user_attrs.get("candidate_id", ""))
        for trial in study.get_trials(deepcopy=False)
        if trial.user_attrs.get("stage") == "fill_import"
    }
    for import_index, candidate_id in enumerate(seed_union, start=1):
        if candidate_id in imported:
            continue
        candidate = candidate_by_id[candidate_id]
        evaluation = evaluation_by_candidate[candidate_id]
        study.add_trial(
            optuna.trial.create_trial(
                params=_trial_params(candidate, space),
                distributions=distributions,
                value=float(evaluation.objective),
                system_attrs={
                    "constraints": [
                        float(value)
                        for value in evaluation.constraints
                    ]
                },
                user_attrs=_trial_user_attrs(
                    candidate=candidate,
                    evaluation=evaluation,
                    lane=lane,
                    seed=seed,
                    stage="fill_import",
                    suggestion_index=import_index,
                    unique_index=None,
                    is_duplicate=False,
                ),
            )
        )
    history = list(_study_history(study, lane=lane, seed=seed, stage="fill"))
    global_seen = {
        *seed_union,
        *(
            row.candidate_id
            for row in history
            if not row.is_duplicate
        ),
    }
    fill_suggestion_index = len(history)
    duplicate_streak = _trailing_duplicate_count(history)
    running = sorted(
        study.get_trials(
            deepcopy=False,
            states=(optuna.trial.TrialState.RUNNING,),
        ),
        key=lambda trial: trial.number,
    )
    for frozen in running:
        candidate = _candidate_from_trial(
            frozen,
            space=space,
            candidate_by_coordinate=candidate_by_coordinate,
        )
        fill_suggestion_index += 1
        duplicate = candidate.candidate_id in global_seen
        evaluation = evaluate(
            candidate,
            SearchRequestContext(
                lane=lane,
                seed=seed,
                trial_number=frozen.number,
                stage="fill",
                suggestion_index=fill_suggestion_index,
                unique_index=(None if duplicate else len(global_seen) + 1),
                is_duplicate=duplicate,
            ),
        )
        evaluation_by_candidate[candidate.candidate_id] = evaluation
        _write_running_trial_state(
            state_path,
            config_hash=config_hash,
            lane=lane,
            stage="fill",
            trial_number=frozen.number,
            candidate_id=candidate.candidate_id,
            state_lock=state_lock,
        )
        _finish_trial(
            study,
            trial=frozen,
            candidate=candidate,
            evaluation=evaluation,
            lane=lane,
            seed=seed,
            stage="fill",
            suggestion_index=fill_suggestion_index,
            unique_index=(None if duplicate else len(global_seen) + 1),
            is_duplicate=duplicate,
        )
        _clear_running_trial_state(
            state_path,
            config_hash=config_hash,
            lane=lane,
            trial_number=frozen.number,
            state_lock=state_lock,
        )
        if duplicate:
            duplicate_streak += 1
        else:
            global_seen.add(candidate.candidate_id)
            duplicate_streak = 0

    def complete_with_deterministic_unseen(
        *,
        selection_reason: str,
    ) -> tuple[SearchTrialRecord, ...]:
        nonlocal fill_suggestion_index
        remaining = sorted(
            (
                candidate
                for candidate in space.candidates
                if candidate.candidate_id not in global_seen
            ),
            key=lambda candidate: candidate.candidate_id,
        )[: global_unique_budget - len(global_seen)]
        for candidate in remaining:
            study.enqueue_trial(
                _trial_params(candidate, space),
                user_attrs={
                    "fill_selection": selection_reason,
                },
                skip_if_exists=True,
            )
            trial = study.ask(fixed_distributions=distributions)
            suggested = _candidate_from_trial(
                trial,
                space=space,
                candidate_by_coordinate=candidate_by_coordinate,
            )
            if suggested.candidate_id != candidate.candidate_id:
                raise StudyStateMismatchError(
                    "确定性 fill 的队列候选顺序不一致"
                )
            fill_suggestion_index += 1
            _write_running_trial_state(
                state_path,
                config_hash=config_hash,
                lane=lane,
                stage="fill",
                trial_number=trial.number,
                candidate_id=candidate.candidate_id,
                state_lock=state_lock,
            )
            evaluation = evaluation_by_candidate.get(candidate.candidate_id)
            if evaluation is None:
                evaluation = evaluate(
                    candidate,
                    SearchRequestContext(
                        lane=lane,
                        seed=seed,
                        trial_number=trial.number,
                        stage="fill",
                        suggestion_index=fill_suggestion_index,
                        unique_index=len(global_seen) + 1,
                        is_duplicate=False,
                    ),
                )
                evaluation_by_candidate[candidate.candidate_id] = evaluation
            _finish_trial(
                study,
                trial=trial,
                candidate=candidate,
                evaluation=evaluation,
                lane=lane,
                seed=seed,
                stage="fill",
                suggestion_index=fill_suggestion_index,
                unique_index=len(global_seen) + 1,
                is_duplicate=False,
            )
            _clear_running_trial_state(
                state_path,
                config_hash=config_hash,
                lane=lane,
                trial_number=trial.number,
                state_lock=state_lock,
            )
            global_seen.add(candidate.candidate_id)
        if len(global_seen) != global_unique_budget:
            raise StudyStateMismatchError(
                "确定性 fill 完成后候选数与目标唯一预算不一致"
            )
        return _study_history(
            study,
            lane=lane,
            seed=seed,
            stage="fill",
        )

    if enumerate_entire_space:
        return complete_with_deterministic_unseen(
            selection_reason="full_enumeration_candidate_id_order",
        )

    while (
        len(global_seen) < global_unique_budget
        and duplicate_streak < unique_stall_limit
    ):
        trial = study.ask(fixed_distributions=distributions)
        candidate = _candidate_from_trial(
            trial,
            space=space,
            candidate_by_coordinate=candidate_by_coordinate,
        )
        fill_suggestion_index += 1
        duplicate = candidate.candidate_id in global_seen
        evaluation = evaluate(
            candidate,
            SearchRequestContext(
                lane=lane,
                seed=seed,
                trial_number=trial.number,
                stage="fill",
                suggestion_index=fill_suggestion_index,
                unique_index=(None if duplicate else len(global_seen) + 1),
                is_duplicate=duplicate,
            ),
        )
        evaluation_by_candidate[candidate.candidate_id] = evaluation
        _write_running_trial_state(
            state_path,
            config_hash=config_hash,
            lane=lane,
            stage="fill",
            trial_number=trial.number,
            candidate_id=candidate.candidate_id,
            state_lock=state_lock,
        )
        _finish_trial(
            study,
            trial=trial,
            candidate=candidate,
            evaluation=evaluation,
            lane=lane,
            seed=seed,
            stage="fill",
            suggestion_index=fill_suggestion_index,
            unique_index=(None if duplicate else len(global_seen) + 1),
            is_duplicate=duplicate,
        )
        _clear_running_trial_state(
            state_path,
            config_hash=config_hash,
            lane=lane,
            trial_number=trial.number,
            state_lock=state_lock,
        )
        if duplicate:
            duplicate_streak += 1
        else:
            global_seen.add(candidate.candidate_id)
            duplicate_streak = 0

    if len(global_seen) < global_unique_budget:
        return complete_with_deterministic_unseen(
            selection_reason=(
                "tpe_duplicate_stall_fallback_candidate_id_order"
            ),
        )
    return _study_history(study, lane=lane, seed=seed, stage="fill")


def _open_seed_study(
    *,
    path: Path,
    name: str,
    seed: int,
    n_startup_trials: int,
    config_hash: str,
    objective_version: str,
    constraints_version: str,
) -> optuna.study.Study:
    path.parent.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(
        seed=int(seed),
        n_startup_trials=int(n_startup_trials),
        constraints_func=_optuna_constraints,
    )
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{path.resolve().as_posix()}",
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )
    expected_attrs = {
        "driver_config_hash": config_hash,
        "sampler_seed": int(seed),
        "n_startup_trials": int(n_startup_trials),
        "objective_version": objective_version,
        "constraints_version": constraints_version,
    }
    for key, expected in expected_attrs.items():
        actual = study.user_attrs.get(key)
        if actual is not None and actual != expected:
            raise StudyStateMismatchError(
                f"study {name} 的 {key} 不匹配: {actual!r} != {expected!r}"
            )
        study.set_user_attr(key, expected)
    return study


def _finish_trial(
    study: optuna.study.Study,
    *,
    trial: optuna.trial.Trial | optuna.trial.FrozenTrial,
    candidate: BOCandidate,
    evaluation: SearchEvaluation,
    lane: str,
    seed: int,
    stage: Literal["search", "fill"],
    suggestion_index: int,
    unique_index: int | None,
    is_duplicate: bool,
) -> None:
    live_trial = _as_live_trial(study, trial)
    for key, value in _trial_user_attrs(
        candidate=candidate,
        evaluation=evaluation,
        lane=lane,
        seed=seed,
        stage=stage,
        suggestion_index=suggestion_index,
        unique_index=unique_index,
        is_duplicate=is_duplicate,
    ).items():
        live_trial.set_user_attr(key, value)
    study.tell(live_trial, float(evaluation.objective))


def _as_live_trial(
    study: optuna.study.Study,
    trial: optuna.trial.Trial | optuna.trial.FrozenTrial,
) -> optuna.trial.Trial:
    if isinstance(trial, optuna.trial.Trial):
        return trial
    trial_id = getattr(trial, "_trial_id", None)
    if not isinstance(trial_id, int):
        raise StudyStateMismatchError(
            "当前 Optuna 版本无法恢复 RUNNING trial；"
            f"version={optuna.__version__}"
        )
    return optuna.trial.Trial(study, trial_id)


def _trial_user_attrs(
    *,
    candidate: BOCandidate,
    evaluation: SearchEvaluation,
    lane: str,
    seed: int,
    stage: str,
    suggestion_index: int,
    unique_index: int | None,
    is_duplicate: bool,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "requested_params": _json_ready(candidate.requested_params),
        "actual_params": _json_ready(candidate.actual_params),
        "lane": lane,
        "seed": int(seed),
        "stage": stage,
        "suggestion_index": int(suggestion_index),
        "unique_index": unique_index,
        "is_duplicate": bool(is_duplicate),
        "constraints": [float(value) for value in evaluation.constraints],
        "metric_valid": bool(evaluation.metric_valid),
        "eligible": bool(evaluation.eligible),
        "failure_reason": str(evaluation.failure_reason),
    }


def _study_history(
    study: optuna.study.Study,
    *,
    lane: str,
    seed: int,
    stage: Literal["search", "fill"],
) -> tuple[SearchTrialRecord, ...]:
    rows: list[SearchTrialRecord] = []
    for trial in study.get_trials(
        deepcopy=False,
        states=(optuna.trial.TrialState.COMPLETE,),
    ):
        if trial.user_attrs.get("stage") != stage:
            continue
        rows.append(
            SearchTrialRecord(
                lane=lane,
                seed=int(seed),
                trial_number=int(trial.number),
                suggestion_index=int(trial.user_attrs["suggestion_index"]),
                unique_index=(
                    int(trial.user_attrs["unique_index"])
                    if trial.user_attrs.get("unique_index") is not None
                    else None
                ),
                candidate_id=str(trial.user_attrs["candidate_id"]),
                is_duplicate=bool(trial.user_attrs["is_duplicate"]),
                objective=float(trial.value),
                constraints=tuple(
                    float(value)
                    for value in trial.user_attrs.get("constraints", ())
                ),
                metric_valid=bool(trial.user_attrs["metric_valid"]),
                eligible=bool(trial.user_attrs["eligible"]),
                failure_reason=str(
                    trial.user_attrs.get("failure_reason", "")
                ),
                stage=stage,
            )
        )
    return tuple(sorted(rows, key=lambda row: row.trial_number))


def _candidate_from_trial(
    trial: optuna.trial.Trial | optuna.trial.FrozenTrial,
    *,
    space: BOSearchSpace,
    candidate_by_coordinate: Mapping[tuple[int, ...], BOCandidate],
) -> BOCandidate:
    try:
        coordinate = tuple(
            int(trial.params[name]) for name in space.parameter_names
        )
    except KeyError as exc:
        raise StudyStateMismatchError(
            f"trial {trial.number} 缺少已冻结参数 {exc.args[0]!r}"
        ) from exc
    candidate = candidate_by_coordinate.get(coordinate)
    if candidate is None:
        raise StudyStateMismatchError(
            f"trial {trial.number} 坐标不在冻结空间: {coordinate}"
        )
    return candidate


def _coordinate_distributions(
    space: BOSearchSpace,
) -> dict[str, optuna.distributions.IntDistribution]:
    return {
        name: optuna.distributions.IntDistribution(
            low=0,
            high=len(space.option_values[axis]) - 1,
        )
        for axis, name in enumerate(space.parameter_names)
    }


def _trial_params(
    candidate: BOCandidate,
    space: BOSearchSpace,
) -> dict[str, int]:
    return {
        name: int(candidate.coordinate[axis])
        for axis, name in enumerate(space.parameter_names)
    }


def _optuna_constraints(
    trial: optuna.trial.FrozenTrial,
) -> tuple[float, ...]:
    return tuple(
        float(value)
        for value in trial.user_attrs.get("constraints", ())
    )


def _evaluations_from_lanes(
    lanes: Sequence[SeedLaneResult],
) -> dict[str, SearchEvaluation]:
    evaluations: dict[str, SearchEvaluation] = {}
    for lane in lanes:
        for row in lane.history:
            evaluation = SearchEvaluation(
                objective=row.objective,
                constraints=row.constraints,
                metric_valid=row.metric_valid,
                eligible=row.eligible,
                failure_reason=row.failure_reason,
            )
            previous = evaluations.get(row.candidate_id)
            if previous is not None and previous != evaluation:
                raise StudyStateMismatchError(
                    f"候选 {row.candidate_id} 在 seed lane 间结果不一致"
                )
            evaluations[row.candidate_id] = evaluation
    return evaluations


def _trailing_duplicate_count(
    history: Sequence[SearchTrialRecord],
) -> int:
    count = 0
    for row in reversed(history):
        if not row.is_duplicate:
            break
        count += 1
    return count


def _validate_seed_search(
    space: BOSearchSpace,
    budget: SeedSearchBudget,
) -> None:
    if budget.lane_unique_budget > len(space.candidates):
        raise ValueError("lane_unique_budget 超过离散空间大小")
    candidate_by_coordinate = {
        candidate.coordinate: candidate for candidate in space.candidates
    }
    if len(candidate_by_coordinate) != len(space.candidates):
        raise ValueError("冻结空间包含重复候选坐标")
    for coordinate in candidate_by_coordinate:
        if len(coordinate) != len(space.parameter_names):
            raise ValueError("候选坐标维数与参数名不一致")
        for axis, option_idx in enumerate(coordinate):
            if not 0 <= int(option_idx) < len(space.option_values[axis]):
                raise ValueError(f"候选坐标越界: {coordinate}")


def _seed_search_config_hash(
    space: BOSearchSpace,
    budget: SeedSearchBudget,
    experiment_identity: SearchExperimentIdentity,
) -> str:
    payload = {
        "space_name": space.name,
        "parameter_names": space.parameter_names,
        "option_values": space.option_values,
        "candidate_ids": [
            candidate.candidate_id for candidate in space.candidates
        ],
        "budget": asdict(budget),
        "experiment_identity": {
            "input_sha256s": experiment_identity.input_sha256s,
            "reference_sha256s": experiment_identity.reference_sha256s,
            "git_commit": experiment_identity.git_commit,
            "run_config": _json_ready(experiment_identity.run_config),
            "evaluation_version": experiment_identity.evaluation_version,
        },
        "metric_contract_version": METRIC_CONTRACT_VERSION,
    }
    return hashlib.sha256(
        json.dumps(
            _json_ready(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _ensure_search_identity(
    path: Path,
    *,
    config_hash: str,
    experiment_identity: SearchExperimentIdentity,
) -> None:
    expected = {
        "config_hash": config_hash,
        "experiment_identity": {
            "input_sha256s": experiment_identity.input_sha256s,
            "reference_sha256s": experiment_identity.reference_sha256s,
            "git_commit": experiment_identity.git_commit,
            "run_config": _json_ready(experiment_identity.run_config),
            "evaluation_version": experiment_identity.evaluation_version,
        },
        "metric_contract_version": METRIC_CONTRACT_VERSION,
    }
    if path.exists():
        actual = _read_json(path)
        if actual != _json_ready(expected):
            raise StudyStateMismatchError(
                "search_identity.json 与当前冻结实验身份不一致"
            )
        return
    _atomic_write_json(path, expected)


@contextmanager
def _exclusive_search_directory(lock_path: Path) -> Iterator[None]:
    with _try_exclusive_file_lock(lock_path) as acquired:
        if not acquired:
            raise SearchAlreadyRunningError(
                f"搜索输出目录正在使用: {lock_path.parent}"
            )
        yield


def _write_running_trial_state(
    state_path: Path,
    *,
    config_hash: str,
    lane: str,
    stage: Literal["search", "fill"],
    trial_number: int,
    candidate_id: str,
    state_lock: threading.Lock,
) -> None:
    unresolved = {
        "lane": lane,
        "trial_number": int(trial_number),
        "candidate_id": candidate_id,
    }
    with state_lock:
        payload = _read_driver_state(state_path, config_hash=config_hash)
        current = [
            item
            for item in payload.get("unresolved_trials", [])
            if not (
                item.get("lane") == lane
                and int(item.get("trial_number", -1)) == trial_number
            )
        ]
        current.append(unresolved)
        _atomic_write_json(
            state_path,
            {
                **payload,
                "config_hash": config_hash,
                "stage": stage,
                "unresolved_trials": sorted(
                    current,
                    key=lambda item: (
                        str(item["lane"]),
                        int(item["trial_number"]),
                    ),
                ),
            },
        )


def _clear_running_trial_state(
    state_path: Path,
    *,
    config_hash: str,
    lane: str,
    trial_number: int,
    state_lock: threading.Lock,
) -> None:
    with state_lock:
        payload = _read_driver_state(state_path, config_hash=config_hash)
        unresolved = [
            item
            for item in payload.get("unresolved_trials", [])
            if not (
                item.get("lane") == lane
                and int(item.get("trial_number", -1)) == trial_number
            )
        ]
        _atomic_write_json(
            state_path,
            {
                **payload,
                "config_hash": config_hash,
                "unresolved_trials": unresolved,
            },
        )


def _read_driver_state(
    state_path: Path,
    *,
    config_hash: str,
) -> dict[str, Any]:
    if not state_path.exists():
        return {}
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    actual_hash = payload.get("config_hash")
    if actual_hash not in (None, config_hash):
        raise StudyStateMismatchError(
            "driver_state.json 与当前搜索配置不一致"
        )
    return dict(payload)


def build_solver_cache_key(identity: SolverCacheIdentity) -> SolverCacheKey:
    """由冻结输入事实构造稳定 SHA-256 求解缓存键。"""

    payload = {
        "data_sha256": str(identity.data_sha256),
        "reference_sha256": str(identity.reference_sha256),
        "git_commit": str(identity.git_commit),
        "run_config": _json_ready(identity.run_config),
        "candidate_id": identity.candidate.candidate_id,
        "space_name": identity.candidate.space_name,
        "requested_params": _json_ready(identity.candidate.requested_params),
        "actual_params": _json_ready(identity.candidate.actual_params),
        "fixed_params": _json_ready(identity.candidate.fixed_params),
        "reference_groups_order": [
            str(group) for group in identity.reference_groups_order
        ],
        "metric_contract_version": METRIC_CONTRACT_VERSION,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return SolverCacheKey(
        key=hashlib.sha256(canonical).hexdigest(),
        payload=MappingProxyType(payload),
    )


def build_bo_search_space(name: SpaceName) -> BOSearchSpace:
    """枚举一个冻结参数空间并返回稳定候选身份。"""

    if name == "legacy_full_v1":
        options = _LEGACY_FULL_OPTIONS
        fixed_params: Mapping[str, Any] = MappingProxyType(
            {"analysis_scope": "full", "lms_mu_min": 1e-6}
        )
    elif name == "legacy_reduced_v1":
        options = _LEGACY_REDUCED_OPTIONS
        fixed_params = _COMMON_FIXED_PARAMS
    elif name == "physical_v1":
        options = _PHYSICAL_OPTIONS
        fixed_params = _COMMON_FIXED_PARAMS
    else:
        raise ValueError(f"未知第二阶段参数空间: {name}")

    parameter_names = tuple(parameter_name for parameter_name, _ in options)
    option_values = tuple(values for _, values in options)
    candidates = tuple(
        _candidate_from_coordinate(
            name=name,
            parameter_names=parameter_names,
            option_values=option_values,
            coordinate=coordinate,
            fixed_params=fixed_params,
        )
        for coordinate in itertools.product(
            *(range(len(values)) for values in option_values)
        )
    )
    return BOSearchSpace(
        name=name,
        parameter_names=parameter_names,
        option_values=option_values,
        candidates=candidates,
    )


def evaluate_formal_metrics(
    result: V2SolverResult,
    *,
    ref_data: np.ndarray,
    time_bias: float,
    method_names: Sequence[str],
) -> FormalMetricResult:
    """按 ``lyx_bo_formal_metric_v1`` 重算一条候选的窗口级指标。

    该入口不读取 ``err_stats`` 或经典 error CSV，且在分母或预测不完整时
    失败关闭。
    """

    if str(result.metadata.get("analysis_scope", "")) != "full":
        raise FormalMetricContractError(
            "analysis_scope_not_full",
            "正式候选必须使用 analysis_scope=full",
        )
    final_method, reset_fft_method = _resolve_formal_method_identity(
        metadata=result.metadata,
        method_names=method_names,
    )
    hr = np.asarray(result.HR, dtype=float)
    if hr.ndim != 2 or hr.shape[1] < 5 or hr.shape[0] == 0:
        raise FormalMetricContractError(
            "invalid_hr_shape",
            f"期望至少五列且非空，实际 shape={hr.shape}",
        )
    reliable = _joined_reliable_mask(hr, result.window_table)
    if not bool(np.any(reliable)):
        raise FormalMetricContractError("no_reliable_windows")

    reference = _interpolate_raw_reference(
        ref_data,
        hr[:, 0] + float(time_bias),
    )
    overlap = np.isfinite(reference)
    motion = hr[:, 4] >= 0.5
    base_full = overlap & reliable
    base_motion = base_full & motion
    classic_motion = overlap & motion

    _require_window_count("base_full", base_full)
    _require_window_count("base_motion", base_motion)
    _require_window_count("classic_motion", classic_motion)

    final = hr[:, 3]
    reset_fft = hr[:, 2]
    _require_finite_prediction("final", final, "base_full", base_full)
    _require_finite_prediction("final", final, "base_motion", base_motion)
    _require_finite_prediction("reset_fft", reset_fft, "base_motion", base_motion)
    _require_finite_prediction("final", final, "classic_motion", classic_motion)
    _require_finite_prediction(
        "reset_fft",
        reset_fft,
        "classic_motion",
        classic_motion,
    )

    base_motion_final_finite = base_motion & np.isfinite(final)
    base_motion_reset_finite = base_motion & np.isfinite(reset_fft)
    classic_final_finite = classic_motion & np.isfinite(final)
    classic_reset_finite = classic_motion & np.isfinite(reset_fft)
    return FormalMetricResult(
        metric_contract_version=METRIC_CONTRACT_VERSION,
        final_method=final_method,
        reset_fft_method=reset_fft_method,
        base_full_window_count=int(np.count_nonzero(base_full)),
        base_motion_window_count=int(np.count_nonzero(base_motion)),
        classic_motion_window_count=int(np.count_nonzero(classic_motion)),
        base_full_final_finite_count=int(
            np.count_nonzero(base_full & np.isfinite(final))
        ),
        base_motion_final_finite_count=int(
            np.count_nonzero(base_motion_final_finite)
        ),
        base_motion_reset_fft_finite_count=int(
            np.count_nonzero(base_motion_reset_finite)
        ),
        base_motion_common_finite_count=int(
            np.count_nonzero(
                base_motion_final_finite & base_motion_reset_finite
            )
        ),
        classic_motion_final_finite_count=int(
            np.count_nonzero(classic_final_finite)
        ),
        classic_motion_reset_fft_finite_count=int(
            np.count_nonzero(classic_reset_finite)
        ),
        classic_motion_common_finite_count=int(
            np.count_nonzero(classic_final_finite & classic_reset_finite)
        ),
        base_full_window_sha256=_window_timestamp_sha256(hr[:, 0], base_full),
        base_motion_window_sha256=_window_timestamp_sha256(
            hr[:, 0],
            base_motion,
        ),
        classic_motion_window_sha256=_window_timestamp_sha256(
            hr[:, 0],
            classic_motion,
        ),
        full_final_mae_bpm=_mae(final, reference, base_full),
        reliable_motion_final_mae_bpm=_mae(final, reference, base_motion),
        reliable_motion_reset_fft_mae_bpm=_mae(
            reset_fft,
            reference,
            base_motion,
        ),
        classic_motion_final_mae_bpm=_mae(
            final,
            reference,
            classic_motion,
        ),
        classic_motion_reset_fft_mae_bpm=_mae(
            reset_fft,
            reference,
            classic_motion,
        ),
    )


def _candidate_from_coordinate(
    *,
    name: SpaceName,
    parameter_names: tuple[str, ...],
    option_values: tuple[tuple[int | float, ...], ...],
    coordinate: tuple[int, ...],
    fixed_params: Mapping[str, Any],
) -> BOCandidate:
    requested = {
        parameter_name: option_values[axis][option_idx]
        for axis, (parameter_name, option_idx) in enumerate(
            zip(parameter_names, coordinate, strict=True)
        )
    }
    if name == "physical_v1":
        actual = {
            "fs_target": int(requested["fs_target"]),
            "max_order": int(
                round(
                    float(requested["fs_target"])
                    * float(requested["memory_ms"])
                    / 1000.0
                )
            ),
            "lms_mu_base": float(requested["mu_base"]),
            "spec_penalty_width": (
                float(requested["exclusion_half_width_bpm"]) / 60.0
            ),
        }
    else:
        actual = dict(requested)
    actual.update(fixed_params)
    candidate_payload = {
        "space_name": name,
        "requested_params": requested,
        "actual_params": actual,
    }
    digest = hashlib.sha256(
        json.dumps(
            candidate_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return BOCandidate(
        space_name=name,
        candidate_id=f"{name}:{digest}",
        coordinate=coordinate,
        requested_params=MappingProxyType(requested),
        actual_params=MappingProxyType(actual),
        fixed_params=fixed_params,
    )


def _joined_reliable_mask(
    hr: np.ndarray,
    window_table: list[dict[str, Any]],
) -> np.ndarray:
    if len(window_table) != hr.shape[0]:
        raise FormalMetricContractError(
            "window_table_length_mismatch",
            f"HR={hr.shape[0]}, window_table={len(window_table)}",
        )
    rows_by_idx: dict[int, dict[str, Any]] = {}
    for row in window_table:
        if "window_idx" not in row:
            raise FormalMetricContractError("missing_window_idx")
        raw_idx = row["window_idx"]
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FormalMetricContractError(
                "invalid_window_idx",
                repr(raw_idx),
            ) from exc
        if idx in rows_by_idx:
            raise FormalMetricContractError(
                "duplicate_window_idx",
                str(idx),
            )
        rows_by_idx[idx] = row
    expected = set(range(hr.shape[0]))
    if set(rows_by_idx) != expected:
        raise FormalMetricContractError(
            "window_idx_set_mismatch",
            f"expected={sorted(expected)}, actual={sorted(rows_by_idx)}",
        )

    reliable = np.zeros(hr.shape[0], dtype=bool)
    for idx in range(hr.shape[0]):
        row = rows_by_idx[idx]
        if "center_s" not in row:
            raise FormalMetricContractError(
                "missing_center_s",
                f"window_idx={idx}",
            )
        try:
            center_s = float(row["center_s"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise FormalMetricContractError(
                "invalid_center_s",
                f"window_idx={idx}",
            ) from exc
        if not np.isfinite(center_s) or not np.isfinite(hr[idx, 0]):
            raise FormalMetricContractError(
                "nonfinite_center_s",
                f"window_idx={idx}",
            )
        if abs(center_s - float(hr[idx, 0])) > 1e-9:
            raise FormalMetricContractError(
                "center_s_mismatch",
                (
                    f"window_idx={idx}, HR={hr[idx, 0]:.17g}, "
                    f"window_table={center_s:.17g}"
                ),
            )
        if "reliable" not in row:
            raise FormalMetricContractError(
                "missing_reliable",
                f"window_idx={idx}",
            )
        reliable[idx] = bool(row["reliable"])
    return reliable


def _resolve_formal_method_identity(
    *,
    metadata: Mapping[str, Any],
    method_names: Sequence[str],
) -> tuple[str, str]:
    names = tuple(str(name).strip() for name in method_names)
    if not names or any(not name for name in names):
        raise FormalMetricContractError("invalid_method_identity")
    if len(set(names)) != len(names):
        raise FormalMetricContractError(
            "duplicate_method_identity",
            repr(names),
        )
    adaptive_filter = str(metadata.get("adaptive_filter", "")).strip().lower()
    raw_groups = metadata.get("reference_groups_order")
    if not adaptive_filter or not isinstance(raw_groups, list | tuple):
        raise FormalMetricContractError(
            "missing_expected_method_identity",
            "metadata 缺少 adaptive_filter/reference_groups_order",
        )
    if adaptive_filter not in _FORMAL_ADAPTIVE_FILTERS:
        raise FormalMetricContractError(
            "invalid_adaptive_filter_identity",
            repr(adaptive_filter),
        )
    try:
        groups = normalise_reference_order(tuple(str(item) for item in raw_groups))
        expected_final = method_label(adaptive_filter, groups)
    except ValueError as exc:
        raise FormalMetricContractError(
            "invalid_expected_method_identity",
            str(exc),
        ) from exc
    if expected_final not in names:
        raise FormalMetricContractError(
            "missing_final_method_identity",
            f"expected={expected_final!r}, available={names!r}",
        )
    reset_fft_method = (
        "reset FFT"
        if "reset FFT" in names
        else "FFT"
        if "FFT" in names
        else ""
    )
    if not reset_fft_method:
        raise FormalMetricContractError(
            "missing_reset_fft_method_identity",
            f"available={names!r}",
        )
    return expected_final, reset_fft_method


def _interpolate_raw_reference(
    ref_data: np.ndarray,
    aligned_times: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(ref_data, dtype=float)
    if reference.ndim != 2 or reference.shape[1] < 2:
        raise FormalMetricContractError(
            "invalid_reference_shape",
            f"shape={reference.shape}",
        )
    finite_rows = np.isfinite(reference[:, 0]) & np.isfinite(reference[:, 1])
    reference = reference[finite_rows, :2]
    if reference.shape[0] < 2:
        raise FormalMetricContractError("insufficient_reference_points")
    order = np.argsort(reference[:, 0], kind="stable")
    reference = reference[order]
    if bool(np.any(np.diff(reference[:, 0]) <= 0)):
        raise FormalMetricContractError("reference_time_not_strictly_increasing")

    aligned = np.asarray(aligned_times, dtype=float)
    output = np.full(aligned.shape, np.nan, dtype=float)
    start = float(reference[0, 0])
    end = float(reference[-1, 0])
    eps = max(1e-9, 1e-9 * max(abs(start), abs(end), 1.0))
    overlap = (
        np.isfinite(aligned)
        & (aligned >= start - eps)
        & (aligned <= end + eps)
    )
    output[overlap] = np.interp(
        aligned[overlap],
        reference[:, 0],
        reference[:, 1],
    )
    return output


def _require_window_count(
    scope: str,
    mask: np.ndarray,
) -> None:
    count = int(np.count_nonzero(mask))
    if count < FORMAL_MIN_WINDOW_COUNT:
        raise FormalMetricContractError(
            f"insufficient_{scope}_windows",
            f"minimum={FORMAL_MIN_WINDOW_COUNT}, actual={count}",
        )


def _require_finite_prediction(
    prediction_name: str,
    values: np.ndarray,
    scope: str,
    mask: np.ndarray,
) -> None:
    finite_count = int(np.count_nonzero(mask & np.isfinite(values)))
    base_count = int(np.count_nonzero(mask))
    if finite_count != base_count:
        raise FormalMetricContractError(
            f"nonfinite_{prediction_name}_on_{scope}",
            f"expected={base_count}, finite={finite_count}",
        )


def _window_timestamp_sha256(
    center_s: np.ndarray,
    mask: np.ndarray,
) -> str:
    payload = "\n".join(
        f"{idx}:{float(center_s[idx]):.17g}"
        for idx in np.flatnonzero(mask)
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _mae(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(prediction, dtype=float)[mask]
                - np.asarray(reference, dtype=float)[mask]
            )
        )
    )


def _write_cached_outcome(
    entry: Path,
    outcome: CandidateSolveOutcome,
) -> None:
    solver_payload: dict[str, Any] | None = None
    if outcome.solver_result is not None:
        solver_payload = {
            "err_stats": _cache_json_ready(outcome.solver_result.err_stats),
            "metadata": _cache_json_ready(outcome.solver_result.metadata),
            "window_table": _cache_json_ready(outcome.solver_result.window_table),
        }
        temp_npz = entry / f"solver_result.{uuid.uuid4().hex}.tmp"
        with temp_npz.open("wb") as handle:
            np.savez_compressed(
                handle,
                HR=np.asarray(outcome.solver_result.HR, dtype=float),
            )
        os.replace(temp_npz, entry / "solver_result.npz")
    _atomic_write_json(
        entry / "outcome.json",
        {
            "status": outcome.status,
            "failure_reason": outcome.failure_reason,
            "diagnostics": _cache_json_ready(outcome.diagnostics),
            "formal_metrics": (
                _json_ready(asdict(outcome.formal_metrics))
                if outcome.formal_metrics is not None
                else None
            ),
            "solver_result": solver_payload,
        },
    )


def _read_cached_outcome(entry: Path) -> CandidateSolveOutcome:
    payload = _cache_json_restore(_read_json(entry / "outcome.json"))
    formal_payload = payload.get("formal_metrics")
    formal_metrics = (
        FormalMetricResult(**formal_payload)
        if isinstance(formal_payload, dict)
        else None
    )
    solver_payload = payload.get("solver_result")
    solver_result: V2SolverResult | None = None
    if isinstance(solver_payload, dict):
        with np.load(entry / "solver_result.npz", allow_pickle=False) as arrays:
            hr = np.asarray(arrays["HR"], dtype=float)
        solver_result = V2SolverResult(
            HR=hr,
            err_stats=dict(solver_payload.get("err_stats", {})),
            metadata=dict(solver_payload.get("metadata", {})),
            window_table=list(solver_payload.get("window_table", [])),
        )
    status = str(payload.get("status", ""))
    if status not in {"valid", "invalid"}:
        raise ValueError(f"缓存 outcome 状态非法: {status!r}")
    return CandidateSolveOutcome(
        status=status,
        solver_result=solver_result,
        formal_metrics=formal_metrics,
        failure_reason=str(payload.get("failure_reason", "")),
        diagnostics=MappingProxyType(dict(payload.get("diagnostics", {}))),
    )


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
    retry_delays_seconds = (0.01, 0.05, 0.10, 0.25, 0.50)
    try:
        for delay_seconds in (*retry_delays_seconds, None):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if delay_seconds is None:
                    raise
                sleep(delay_seconds)
    finally:
        if temp.exists():
            with suppress(OSError):
                temp.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"缓存 JSON 顶层必须是对象: {path}")
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer | np.floating | np.bool_):
        return value.item()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"值无法进入缓存 JSON: {type(value).__name__}")


def _cache_json_ready(value: Any) -> Any:
    """Encode optional non-finite diagnostics without weakening identity JSON."""

    if isinstance(value, Mapping):
        if _CACHE_NONFINITE_FLOAT_KEY in value:
            raise ValueError(
                "diagnostic mapping contains reserved cache marker: "
                f"{_CACHE_NONFINITE_FLOAT_KEY}"
            )
        return {
            str(key): _cache_json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple | list):
        return [_cache_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _cache_json_ready(value.tolist())
    if isinstance(value, np.integer | np.floating | np.bool_):
        return _cache_json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "nan"
        elif value > 0:
            label = "positive_infinity"
        else:
            label = "negative_infinity"
        return {_CACHE_NONFINITE_FLOAT_KEY: label}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(
        "value cannot enter candidate-outcome cache JSON: "
        f"{type(value).__name__}"
    )


def _cache_json_restore(value: Any) -> Any:
    """Restore cache-only tagged non-finite diagnostics."""

    if isinstance(value, list):
        return [_cache_json_restore(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {_CACHE_NONFINITE_FLOAT_KEY}:
        label = value[_CACHE_NONFINITE_FLOAT_KEY]
        if label == "nan":
            return float("nan")
        if label == "positive_infinity":
            return float("inf")
        if label == "negative_infinity":
            return float("-inf")
        raise ValueError(f"unknown cached non-finite float label: {label!r}")
    return {
        str(key): _cache_json_restore(item)
        for key, item in value.items()
    }


def _formal_metric_failure_category(reason: str) -> str:
    if (
        reason in _METHOD_IDENTITY_CONTRACT_REASONS
        or "method" in reason
        or "adaptive_filter" in reason
    ):
        return "method_identity_mismatch"
    return "metric_window_contract_failed"
