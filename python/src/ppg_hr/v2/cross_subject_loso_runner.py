"""P1 sentinel runner for the compact cross-subject response ledger."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .cross_subject_loso_ledger import (
    CompactCellMetric,
    CompactResponseLedger,
    TechnicalAttemptEvent,
)
from .cross_subject_loso_metrics import (
    evaluate_fixed_time_metrics,
    evaluate_six_gates,
)
from .cross_subject_loso_source import HFCallIdentity, PanelRecord, PhysicalCoordinate
from .preprocess import load_v2_reference
from .solver import V2SolverResult, solve_v2
from .types import V2RunConfig

DEFAULT_P1_SENTINEL_RECORD_IDS = (
    "xiezi3_CGX_0710",
    "run1_HB_0711",
    "bobi2_LYX_0519",
    "run3_LZJ_0711",
    "bobi1_PJY_0714",
    "bobi1_QYC_0615",
    "xiezi3_TS_0709",
)
DEFAULT_P1_COORDINATE_INDEX = 0


@dataclass(frozen=True)
class RunReceipt:
    requested: int
    skipped: int
    attempted: int
    completed: int
    technical_failures: int


@dataclass(frozen=True)
class _ProcessRequest:
    call: HFCallIdentity
    attempt_number: int
    dataset_sha256: str
    runner_sha256: str


def build_hf_run_config(record: PanelRecord, coordinate: PhysicalCoordinate) -> V2RunConfig:
    return V2RunConfig(
        data_path=record.data_path,
        ref_path=record.ref_path,
        ppg_mode="green",
        ppg_input_transform="raw_bandpass",
        adaptive_filter="lms",
        algorithm_preset="lite",
        reference_groups_order=("HF",),
        adaptive_reference_stage_limit=None,
        rise_candidate_lineage_enable=True,
        rise_confirmation_policy_id="legacy_v1",
        penalty_candidate_id="suppressed_protected_continuous_visibility_v1",
        low_reacquire_candidate_id="bounded_low_owner_harmonic_support_v1",
        recovery_candidate_id="identity_blind_dual_high_lock_rescue_v1",
        post_motion_minimal_loss_fallback_hits=3,
        post_motion_delayed_raw_bootstrap_hits=2,
        postprocess_dynamics_enable=True,
        analysis_scope="full",
        fs_target=coordinate.fs_target_hz,
        max_order=round(coordinate.fs_target_hz * coordinate.memory_ms / 1000),
        lms_mu_base=coordinate.mu_base,
        lms_mu_min=1e-6,
        spec_penalty_width=coordinate.exclusion_half_width_bpm / 60.0,
        smooth_win_len=5,
        time_bias=5.0,
    )


def choose_p1_sentinel_calls(
    calls: Sequence[HFCallIdentity],
    *,
    sentinel_record_ids: Sequence[str] = DEFAULT_P1_SENTINEL_RECORD_IDS,
    coordinate_index: int = DEFAULT_P1_COORDINATE_INDEX,
) -> tuple[HFCallIdentity, ...]:
    by_key = {(call.record.record_id, call.coordinate.coordinate_index): call for call in calls}
    selected: list[HFCallIdentity] = []
    for record_id in sentinel_record_ids:
        try:
            selected.append(by_key[(record_id, coordinate_index)])
        except KeyError as error:
            raise ValueError(f"p1_sentinel_call_missing:{record_id}:{coordinate_index}") from error
    if len({call.record.physical_subject_id for call in selected}) != len(selected):
        raise ValueError("p1_sentinels_must_cover_distinct_subjects")
    return tuple(selected)


class CrossSubjectRunner:
    def __init__(
        self,
        *,
        ledger: CompactResponseLedger,
        dataset_sha256: str,
        runner_sha256: str,
        solve_call: Callable[[HFCallIdentity], V2SolverResult] | None = None,
        load_reference: Callable[[Path], np.ndarray] = load_v2_reference,
        now: Callable[[], str] | None = None,
    ) -> None:
        self.ledger = ledger
        self.dataset_sha256 = dataset_sha256
        self.runner_sha256 = runner_sha256
        self.solve_call = solve_call or _solve_hf_call
        self.load_reference = load_reference
        self.now = now or _now_iso
        self._uses_default_callbacks = (
            solve_call is None and load_reference is load_v2_reference and now is None
        )

    def run_calls(self, calls: Sequence[HFCallIdentity]) -> RunReceipt:
        pending = self._pending_calls(calls)
        skipped = len(calls) - len(pending)
        self._verify_input_identities(pending)
        completed = 0
        technical_failures = 0
        for call in pending:
            outcome = self._execute_call(call, self._next_attempt_number(call))
            if isinstance(outcome, CompactCellMetric):
                self.ledger.record_complete(outcome)
                completed += 1
            else:
                self.ledger.record_attempt(outcome)
                technical_failures += 1
        return RunReceipt(
            requested=len(calls),
            skipped=skipped,
            attempted=len(pending),
            completed=completed,
            technical_failures=technical_failures,
        )

    def run_calls_parallel(
        self,
        calls: Sequence[HFCallIdentity],
        *,
        workers: int,
        batch_size: int,
        executor_kind: str = "thread",
        on_progress: Callable[[RunReceipt], None] | None = None,
    ) -> RunReceipt:
        if workers < 1 or batch_size < 1:
            raise ValueError("workers_and_batch_size_must_be_positive")
        if executor_kind not in {"thread", "process"}:
            raise ValueError(f"unsupported_executor_kind:{executor_kind}")
        if executor_kind == "process" and not self._uses_default_callbacks:
            raise ValueError("process_executor_requires_default_callbacks")
        pending = self._pending_calls(calls)
        skipped = len(calls) - len(pending)
        self._verify_input_identities(pending)
        prior_attempts = self.ledger.attempt_counts()
        completed = 0
        technical_failures = 0
        executor_type = ProcessPoolExecutor if executor_kind == "process" else ThreadPoolExecutor
        with executor_type(max_workers=workers) as executor:
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset : offset + batch_size]
                attempted = tuple(
                    (
                        call,
                        prior_attempts.get(
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
                        _ProcessRequest(
                            call=call,
                            attempt_number=attempt_number,
                            dataset_sha256=self.dataset_sha256,
                            runner_sha256=self.runner_sha256,
                        )
                        for call, attempt_number in attempted
                    )
                    outcomes = tuple(executor.map(_execute_process_request, requests))
                else:
                    outcomes = tuple(
                        executor.map(lambda item: self._execute_call(*item), attempted)
                    )
                cells = tuple(
                    outcome for outcome in outcomes if isinstance(outcome, CompactCellMetric)
                )
                events = tuple(
                    outcome for outcome in outcomes if isinstance(outcome, TechnicalAttemptEvent)
                )
                self.ledger.record_outcomes_batch(cells, events)
                completed += len(cells)
                technical_failures += len(events)
                if on_progress is not None:
                    on_progress(
                        RunReceipt(
                            requested=len(calls),
                            skipped=skipped,
                            attempted=completed + technical_failures,
                            completed=completed,
                            technical_failures=technical_failures,
                        )
                    )
        return RunReceipt(
            requested=len(calls),
            skipped=skipped,
            attempted=len(pending),
            completed=completed,
            technical_failures=technical_failures,
        )

    def _pending_calls(self, calls: Sequence[HFCallIdentity]) -> list[HFCallIdentity]:
        complete_by_route = {
            route_id: self.ledger.completed_keys(route_id)
            for route_id in {call.route_id for call in calls}
        }
        return [
            call
            for call in calls
            if (
                call.record.physical_subject_id,
                call.record.record_id,
                call.coordinate.coordinate_id,
            )
            not in complete_by_route[call.route_id]
        ]

    def _next_attempt_number(self, call: HFCallIdentity) -> int:
        return self.ledger.next_attempt_number(
            call.route_id,
            call.record.physical_subject_id,
            call.record.record_id,
            call.coordinate.coordinate_id,
        )

    def _execute_call(
        self, call: HFCallIdentity, attempt_number: int
    ) -> CompactCellMetric | TechnicalAttemptEvent:
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
    def _verify_input_identities(calls: Sequence[HFCallIdentity]) -> None:
        records = {call.record.record_id: call.record for call in calls}
        for record in records.values():
            if _file_sha256(record.data_path) != record.data_sha256:
                raise RuntimeError(f"data_sha256_mismatch:{record.record_id}")
            if _file_sha256(record.ref_path) != record.ref_sha256:
                raise RuntimeError(f"ref_sha256_mismatch:{record.record_id}")


def _execute_process_request(
    request: _ProcessRequest,
) -> CompactCellMetric | TechnicalAttemptEvent:
    return _execute_call_impl(
        call=request.call,
        attempt_number=request.attempt_number,
        dataset_sha256=request.dataset_sha256,
        runner_sha256=request.runner_sha256,
        solve_call=_solve_hf_call,
        load_reference=load_v2_reference,
        now=_now_iso,
    )


def _execute_call_impl(
    *,
    call: HFCallIdentity,
    attempt_number: int,
    dataset_sha256: str,
    runner_sha256: str,
    solve_call: Callable[[HFCallIdentity], V2SolverResult],
    load_reference: Callable[[Path], np.ndarray],
    now: Callable[[], str],
) -> CompactCellMetric | TechnicalAttemptEvent:
    started = time.perf_counter()
    try:
        result = solve_call(call)
        reference = load_reference(call.record.ref_path)
        candidate = evaluate_fixed_time_metrics(
            result,
            ref_data=reference,
            time_bias_s=5.0,
        )
        gate = evaluate_six_gates(
            candidate=candidate,
            baseline=call.record.baseline.fixed5,
        )
        return CompactCellMetric(
            experiment_id=call.experiment_id,
            algorithm_sha256=call.algorithm_sha256,
            runner_sha256=runner_sha256,
            dataset_sha256=dataset_sha256,
            input_sha256=_input_sha256(call.record),
            baseline_sha256=call.record.baseline.fixed5_sha256,
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
            candidate_mae_bpm=candidate.mae_bpm,
            candidate_l10=candidate.l10,
            candidate_l20=candidate.l20,
            candidate_e10=candidate.e10,
            candidate_e20=candidate.e20,
            candidate_right_censored_recovery_count=candidate.right_censored_recovery_count,
            candidate_full_window_count=candidate.full_window_count,
            candidate_reliable_window_count=candidate.reliable_window_count,
            candidate_motion_window_count=candidate.motion_window_count,
            candidate_evaluation_window_sha256=candidate.evaluation_window_sha256,
            baseline_mae_bpm=call.record.baseline.fixed5.mae_bpm,
            baseline_l10=call.record.baseline.fixed5.l10,
            baseline_l20=call.record.baseline.fixed5.l20,
            baseline_right_censored_recovery_count=(
                call.record.baseline.fixed5.right_censored_recovery_count
            ),
            g1i_pass=gate.g1i_pass,
            g2_pass=gate.g2_pass,
            g3_pass=gate.g3_pass,
            g4_pass=gate.g4_pass,
            g5_pass=gate.g5_pass,
            g7_pass=gate.g7_pass,
            qualified=gate.qualified,
            g2_margin_s=gate.g2_margin_s,
            g3_margin_s=gate.g3_margin_s,
            g4_margin_bpm=gate.g4_margin_bpm,
            g5_right_censored_count=gate.g5_right_censored_count,
            g7_margin_s=gate.g7_margin_s,
            failed_gates_json=json.dumps(list(gate.failed_gates), separators=(",", ":")),
            solver_elapsed_s=time.perf_counter() - started,
            attempt_count=attempt_number,
            completed_at=now(),
        )
    except Exception as error:
        return TechnicalAttemptEvent(
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


def _solve_hf_call(call: HFCallIdentity) -> V2SolverResult:
    return solve_v2(build_hf_run_config(call.record, call.coordinate))


def _input_sha256(record: PanelRecord) -> str:
    payload = json.dumps(
        {
            "physical_subject_id": record.physical_subject_id,
            "record_id": record.record_id,
            "data_sha256": record.data_sha256,
            "ref_sha256": record.ref_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reason_code(error: Exception) -> str:
    name = type(error).__name__
    if isinstance(error, FileNotFoundError):
        return "input_missing"
    if str(error) in {"data_sha256_mismatch", "ref_sha256_mismatch"}:
        return "input_identity_mismatch"
    if name == "FixedTimeMetricError":
        return "metric_contract_failure"
    return "solver_exception"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
