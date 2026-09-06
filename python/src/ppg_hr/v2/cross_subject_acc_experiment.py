"""Frozen ACC-only response experiment for the 143-record cross-subject panel."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .cross_subject_acc_ledger import (
    AccCompactCellMetric,
    AccCompactResponseLedger,
    AccTechnicalAttemptEvent,
)
from .cross_subject_loso_metrics import (
    FIXED_TIME_METRIC_CONTRACT_ID,
    evaluate_fixed_time_metrics,
)
from .cross_subject_loso_runner import build_hf_run_config
from .cross_subject_loso_source import PanelRecord, PhysicalCoordinate
from .preprocess import load_v2_reference
from .solver import V2SolverResult, solve_v2
from .types import V2RunConfig

ACC_EXPERIMENT_ID = "cross_subject_multirecord_acc_independent_physical4d_v1"
PARENT_EXPERIMENT_ID = "cross_subject_multirecord_hf_loso_v1"
ACC_ROUTE_ID = "ACC"
EXPECTED_ACC_CALL_COUNT = 42_900
TIME_BIAS_S = 5.0
ACC_METRIC_CONTRACT = {
    "metric_id": "cross_subject_acc_route_native_fixed5_metrics_v1",
    "fixed_metric_contract_id": FIXED_TIME_METRIC_CONTRACT_ID,
    "time_bias_s": TIME_BIAS_S,
    "selection_support": "route_native_reliable_reference_overlap",
    "gate_policy": "no_hf_lite_six_gate",
}
ACC_METRIC_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(ACC_METRIC_CONTRACT, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class AccCallIdentity:
    experiment_id: str
    route_id: str
    record: PanelRecord
    coordinate: PhysicalCoordinate
    algorithm_sha256: str
    metric_contract_sha256: str
    call_identity_sha256: str


@dataclass(frozen=True)
class AccRunReceipt:
    requested: int
    skipped: int
    attempted: int
    completed: int
    technical_failures: int


@dataclass(frozen=True)
class _ProcessRequest:
    call: AccCallIdentity
    attempt_number: int
    dataset_sha256: str
    runner_sha256: str


def build_acc_run_config(record: PanelRecord, coordinate: PhysicalCoordinate) -> V2RunConfig:
    config = build_hf_run_config(record, coordinate)
    return V2RunConfig(
        **{
            field.name: (
                ("ACC",) if field.name == "reference_groups_order" else getattr(config, field.name)
            )
            for field in fields(V2RunConfig)
        }
    )


def build_acc_call_identities(
    *,
    records: Sequence[PanelRecord],
    coordinates: Sequence[PhysicalCoordinate],
    algorithm_sha256: str,
) -> tuple[AccCallIdentity, ...]:
    calls: list[AccCallIdentity] = []
    for record in records:
        for coordinate in coordinates:
            identity = {
                "experiment_id": ACC_EXPERIMENT_ID,
                "parent_experiment_id": PARENT_EXPERIMENT_ID,
                "route_id": ACC_ROUTE_ID,
                "physical_subject_id": record.physical_subject_id,
                "scene": record.scene,
                "record_id": record.record_id,
                "data_sha256": record.data_sha256,
                "ref_sha256": record.ref_sha256,
                "coordinate": asdict(coordinate),
                "algorithm_sha256": algorithm_sha256,
                "metric_contract_sha256": ACC_METRIC_CONTRACT_SHA256,
            }
            calls.append(
                AccCallIdentity(
                    experiment_id=ACC_EXPERIMENT_ID,
                    route_id=ACC_ROUTE_ID,
                    record=record,
                    coordinate=coordinate,
                    algorithm_sha256=algorithm_sha256,
                    metric_contract_sha256=ACC_METRIC_CONTRACT_SHA256,
                    call_identity_sha256=_semantic_sha256(identity),
                )
            )
    return tuple(calls)


def choose_acc_sentinel_calls(
    calls: Sequence[AccCallIdentity],
    *,
    sentinel_record_ids: Sequence[str],
    coordinate_index: int,
) -> tuple[AccCallIdentity, ...]:
    by_key = {(call.record.record_id, call.coordinate.coordinate_index): call for call in calls}
    selected = []
    for record_id in sentinel_record_ids:
        try:
            selected.append(by_key[(record_id, coordinate_index)])
        except KeyError as error:
            raise ValueError(f"acc_sentinel_call_missing:{record_id}:{coordinate_index}") from error
    return tuple(selected)


class AccCrossSubjectRunner:
    def __init__(
        self,
        *,
        ledger: AccCompactResponseLedger,
        dataset_sha256: str,
        runner_sha256: str,
        solve_call: Callable[[AccCallIdentity], V2SolverResult] | None = None,
        load_reference: Callable[[Path], np.ndarray] = load_v2_reference,
        now: Callable[[], str] | None = None,
    ) -> None:
        self.ledger = ledger
        self.dataset_sha256 = dataset_sha256
        self.runner_sha256 = runner_sha256
        self.solve_call = solve_call or _solve_acc_call
        self.load_reference = load_reference
        self.now = now or _now_iso
        self._uses_default_callbacks = (
            solve_call is None and load_reference is load_v2_reference and now is None
        )

    def run_calls(self, calls: Sequence[AccCallIdentity]) -> AccRunReceipt:
        pending = self._pending_calls(calls)
        self._verify_input_identities(pending)
        completed = 0
        failures = 0
        for call in pending:
            outcome = self._execute_call(call, self._next_attempt_number(call))
            if isinstance(outcome, AccCompactCellMetric):
                self.ledger.record_complete(outcome)
                completed += 1
            else:
                self.ledger.record_attempt(outcome)
                failures += 1
        return AccRunReceipt(
            len(calls), len(calls) - len(pending), len(pending), completed, failures
        )

    def run_calls_parallel(
        self,
        calls: Sequence[AccCallIdentity],
        *,
        workers: int,
        batch_size: int,
        executor_kind: str = "thread",
        on_progress: Callable[[AccRunReceipt], None] | None = None,
    ) -> AccRunReceipt:
        if workers < 1 or batch_size < 1:
            raise ValueError("workers_and_batch_size_must_be_positive")
        if executor_kind not in {"thread", "process"}:
            raise ValueError(f"unsupported_executor_kind:{executor_kind}")
        if executor_kind == "process" and not self._uses_default_callbacks:
            raise ValueError("process_executor_requires_default_callbacks")
        pending = self._pending_calls(calls)
        skipped = len(calls) - len(pending)
        self._verify_input_identities(pending)
        prior = self.ledger.attempt_counts()
        completed = 0
        failures = 0
        executor_type = ProcessPoolExecutor if executor_kind == "process" else ThreadPoolExecutor
        with executor_type(max_workers=workers) as executor:
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset : offset + batch_size]
                attempted = tuple(
                    (
                        call,
                        prior.get(
                            (
                                call.route_id,
                                call.record.physical_subject_id,
                                call.record.record_id,
                                call.coordinate.coordinate_id,
                            ),
                            0,
                        )
                        + 1,
                    )
                    for call in batch
                )
                if executor_kind == "process":
                    requests = tuple(
                        _ProcessRequest(call, attempt, self.dataset_sha256, self.runner_sha256)
                        for call, attempt in attempted
                    )
                    outcomes = tuple(executor.map(_execute_process_request, requests))
                else:
                    outcomes = tuple(
                        executor.map(lambda item: self._execute_call(*item), attempted)
                    )
                cells = tuple(row for row in outcomes if isinstance(row, AccCompactCellMetric))
                events = tuple(row for row in outcomes if isinstance(row, AccTechnicalAttemptEvent))
                self.ledger.record_outcomes_batch(cells, events)
                completed += len(cells)
                failures += len(events)
                if on_progress is not None:
                    on_progress(
                        AccRunReceipt(
                            len(calls), skipped, completed + failures, completed, failures
                        )
                    )
        return AccRunReceipt(len(calls), skipped, len(pending), completed, failures)

    def _pending_calls(self, calls: Sequence[AccCallIdentity]) -> list[AccCallIdentity]:
        complete = self.ledger.completed_keys(ACC_ROUTE_ID)
        return [
            call
            for call in calls
            if (
                call.record.physical_subject_id,
                call.record.record_id,
                call.coordinate.coordinate_id,
            )
            not in complete
        ]

    def _next_attempt_number(self, call: AccCallIdentity) -> int:
        return self.ledger.next_attempt_number(
            call.route_id,
            call.record.physical_subject_id,
            call.record.record_id,
            call.coordinate.coordinate_id,
        )

    def _execute_call(
        self, call: AccCallIdentity, attempt_number: int
    ) -> AccCompactCellMetric | AccTechnicalAttemptEvent:
        return _execute_call_impl(
            call=call,
            attempt_number=attempt_number,
            dataset_sha256=self.dataset_sha256,
            runner_sha256=self.runner_sha256,
            solve_call=self.solve_call,
            load_reference=self.load_reference,
            now=self.now,
        )

    @staticmethod
    def _verify_input_identities(calls: Sequence[AccCallIdentity]) -> None:
        for record in {call.record.record_id: call.record for call in calls}.values():
            if _file_sha256(record.data_path) != record.data_sha256:
                raise RuntimeError(f"data_sha256_mismatch:{record.record_id}")
            if _file_sha256(record.ref_path) != record.ref_sha256:
                raise RuntimeError(f"ref_sha256_mismatch:{record.record_id}")


def _execute_process_request(
    request: _ProcessRequest,
) -> AccCompactCellMetric | AccTechnicalAttemptEvent:
    return _execute_call_impl(
        call=request.call,
        attempt_number=request.attempt_number,
        dataset_sha256=request.dataset_sha256,
        runner_sha256=request.runner_sha256,
        solve_call=_solve_acc_call,
        load_reference=load_v2_reference,
        now=_now_iso,
    )


def _execute_call_impl(
    *,
    call: AccCallIdentity,
    attempt_number: int,
    dataset_sha256: str,
    runner_sha256: str,
    solve_call: Callable[[AccCallIdentity], V2SolverResult],
    load_reference: Callable[[Path], np.ndarray],
    now: Callable[[], str],
) -> AccCompactCellMetric | AccTechnicalAttemptEvent:
    started = time.perf_counter()
    try:
        result = solve_call(call)
        metric = evaluate_fixed_time_metrics(
            result,
            ref_data=load_reference(call.record.ref_path),
            time_bias_s=TIME_BIAS_S,
        )
        return AccCompactCellMetric(
            experiment_id=call.experiment_id,
            parent_experiment_id=PARENT_EXPERIMENT_ID,
            algorithm_sha256=call.algorithm_sha256,
            runner_sha256=runner_sha256,
            dataset_sha256=dataset_sha256,
            input_sha256=_input_sha256(call.record),
            metric_contract_sha256=call.metric_contract_sha256,
            call_identity_sha256=call.call_identity_sha256,
            route_id=call.route_id,
            physical_subject_id=call.record.physical_subject_id,
            scene=call.record.scene,
            record_id=call.record.record_id,
            coordinate_id=call.coordinate.coordinate_id,
            coordinate_index=call.coordinate.coordinate_index,
            fs_target_hz=call.coordinate.fs_target_hz,
            memory_ms=call.coordinate.memory_ms,
            mu_base=call.coordinate.mu_base,
            exclusion_half_width_bpm=call.coordinate.exclusion_half_width_bpm,
            mae_bpm=metric.mae_bpm,
            l10=metric.l10,
            l20=metric.l20,
            e10=metric.e10,
            e20=metric.e20,
            right_censored_recovery_count=metric.right_censored_recovery_count,
            full_window_count=metric.full_window_count,
            reliable_window_count=metric.reliable_window_count,
            motion_window_count=metric.motion_window_count,
            true_rise_applicable=metric.true_rise_applicable,
            true_rise_underestimate_bpm=metric.true_rise_underestimate_bpm,
            true_rise_episode_count=metric.true_rise_episode_count,
            spectral_gate_contract_v2=metric.spectral_gate_contract_v2,
            stability_pass=metric.stability_pass,
            reference_groups_order_json=json.dumps(
                list(metric.reference_groups_order), separators=(",", ":")
            ),
            adaptive_reference_stage_limit=metric.adaptive_reference_stage_limit,
            evaluation_window_sha256=metric.evaluation_window_sha256,
            solver_elapsed_s=time.perf_counter() - started,
            attempt_count=attempt_number,
            completed_at=now(),
        )
    except Exception as error:
        return AccTechnicalAttemptEvent(
            route_id=call.route_id,
            physical_subject_id=call.record.physical_subject_id,
            record_id=call.record.record_id,
            coordinate_id=call.coordinate.coordinate_id,
            call_identity_sha256=call.call_identity_sha256,
            attempt_number=attempt_number,
            reason_code=_reason_code(error),
            detail=f"{type(error).__name__}:{error}"[:4000],
            occurred_at=now(),
        )


def _solve_acc_call(call: AccCallIdentity) -> V2SolverResult:
    return solve_v2(build_acc_run_config(call.record, call.coordinate))


def _input_sha256(record: PanelRecord) -> str:
    return _semantic_sha256(
        {
            "physical_subject_id": record.physical_subject_id,
            "record_id": record.record_id,
            "data_sha256": record.data_sha256,
            "ref_sha256": record.ref_sha256,
        }
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reason_code(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return "input_missing"
    if type(error).__name__ == "FixedTimeMetricError":
        return "metric_contract_failure"
    return "solver_exception"


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
