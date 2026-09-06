"""Resumable ACC and HF+ACC Physical4D response-surface runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from .preprocess import load_v2_reference
from .reference_arm_ledger import (
    CompactCellMetric,
    CompactResponseLedger,
    TechnicalAttemptEvent,
    TechnicalCellError,
    evaluate_fixed5_mae,
)
from .reference_arm_source import (
    ImportedHFCell,
    PhysicalCoordinate,
    RecordIdentity,
    ReferenceArmSourceSnapshot,
    materialise_source_snapshot,
)
from .solver import solve_v2
from .types import V2RunConfig

EXPERIMENT_ID = "lyx_reference_arm_physical4d_threefold_20260828"
FROZEN_SOURCE_COMMIT = "38393d9b193517860db6e8b4cc741e4bacdee174"
ROUTE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "ACC": ("ACC",),
    "HF_ACC": ("HF", "ACC"),
}
METRIC_CONTRACT = {
    "metric_id": "reference_arm_fixed5_exact_window_keys_v1",
    "prediction_column": "Final",
    "time_bias_s": 5.0,
    "join_key": ["window_idx", "center_s"],
    "missing_policy": "technical_incomplete_same_identity_rerun",
}
METRIC_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(METRIC_CONTRACT, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class ReferenceArmCall:
    route_id: str
    record: RecordIdentity
    coordinate: PhysicalCoordinate
    config_sha256: str
    identity_sha256: str


@dataclass(frozen=True)
class ReferenceArmRunSummary:
    requested: int
    attempted: int
    complete: int
    skipped: int
    technical_failures: int
    timed_out: bool


@dataclass(frozen=True)
class ReferenceArmExperimentContext:
    repo_root: Path
    output_root: Path
    snapshot: ReferenceArmSourceSnapshot
    ledger: CompactResponseLedger
    calls: tuple[ReferenceArmCall, ...]
    experiment_identity: Mapping[str, Any]
    code_sha256: str


@dataclass(frozen=True)
class _AttemptedCall:
    call: ReferenceArmCall
    attempt_number: int


def build_run_config(
    record: RecordIdentity,
    coordinate: PhysicalCoordinate,
    route_id: str,
) -> V2RunConfig:
    try:
        reference_groups = ROUTE_GROUPS[route_id]
    except KeyError as error:
        raise ValueError(f"unknown reference route: {route_id}") from error
    return V2RunConfig(
        data_path=record.data_path,
        ref_path=record.ref_path,
        ppg_mode="green",
        ppg_input_transform="raw_bandpass",
        adaptive_filter="lms",
        algorithm_preset="lite",
        reference_groups_order=reference_groups,
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


def config_diff(left: V2RunConfig, right: V2RunConfig) -> set[str]:
    return {
        field.name
        for field in fields(V2RunConfig)
        if getattr(left, field.name) != getattr(right, field.name)
    }


def build_reference_arm_calls(
    snapshot: ReferenceArmSourceSnapshot,
    *,
    code_sha256: str,
) -> tuple[ReferenceArmCall, ...]:
    calls = []
    for route_id in ROUTE_GROUPS:
        for record in snapshot.records:
            for coordinate in snapshot.coordinates:
                config = build_run_config(record, coordinate, route_id)
                config_payload = _config_identity_payload(config, record)
                config_sha = _semantic_sha256(config_payload)
                identity = {
                    "experiment_id": EXPERIMENT_ID,
                    "source_commit": snapshot.source_commit,
                    "source_semantic_sha256": snapshot.semantic_sha256,
                    "code_sha256": code_sha256,
                    "metric_contract_sha256": METRIC_CONTRACT_SHA256,
                    "route_id": route_id,
                    "record_id": record.record_id,
                    "scene": record.scene,
                    "data_sha256": record.data_sha256,
                    "ref_sha256": record.ref_sha256,
                    "coordinate": asdict(coordinate),
                    "config": config_payload,
                }
                calls.append(
                    ReferenceArmCall(
                        route_id=route_id,
                        record=record,
                        coordinate=coordinate,
                        config_sha256=config_sha,
                        identity_sha256=_semantic_sha256(identity),
                    )
                )
    return tuple(calls)


class ReferenceArmRunner:
    def __init__(
        self,
        *,
        snapshot: ReferenceArmSourceSnapshot,
        ledger: CompactResponseLedger,
        experiment_id: str,
        code_sha256: str,
    ) -> None:
        self.snapshot = snapshot
        self.ledger = ledger
        self.experiment_id = experiment_id
        self.code_sha256 = code_sha256
        self.calls = build_reference_arm_calls(snapshot, code_sha256=code_sha256)
        self._timelines = dict(snapshot.timelines)

    def run_calls(
        self,
        calls: Sequence[ReferenceArmCall],
        *,
        workers: int,
        deadline_monotonic: float | None = None,
        on_progress: Callable[[ReferenceArmRunSummary], None] | None = None,
    ) -> ReferenceArmRunSummary:
        if workers < 1:
            raise ValueError("workers must be at least one")
        requested = len(calls)
        pending = [
            call
            for call in calls
            if not self.ledger.has_complete(
                call.route_id, call.record.record_id, call.coordinate.coordinate_id
            )
        ]
        skipped = requested - len(pending)
        attempted = 0
        complete = 0
        technical_failures = 0
        timed_out = False

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for offset in range(0, len(pending), workers):
                if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                    timed_out = True
                    break
                batch = pending[offset : offset + workers]
                attempted_batch = [
                    _AttemptedCall(call=call, attempt_number=self._next_attempt_number(call))
                    for call in batch
                ]
                outcomes = list(executor.map(self._execute_call, attempted_batch))
                for outcome in outcomes:
                    attempted += 1
                    if isinstance(outcome, CompactCellMetric):
                        self.ledger.record_complete(outcome)
                        complete += 1
                    else:
                        self.ledger.record_attempt(outcome)
                        technical_failures += 1
                    if on_progress is not None:
                        on_progress(
                            ReferenceArmRunSummary(
                                requested=requested,
                                attempted=attempted,
                                complete=complete,
                                skipped=skipped,
                                technical_failures=technical_failures,
                                timed_out=False,
                            )
                        )
        return ReferenceArmRunSummary(
            requested=requested,
            attempted=attempted,
            complete=complete,
            skipped=skipped,
            technical_failures=technical_failures,
            timed_out=timed_out,
        )

    def _execute_call(
        self, attempted_call: _AttemptedCall
    ) -> CompactCellMetric | TechnicalAttemptEvent:
        call = attempted_call.call
        attempt_number = attempted_call.attempt_number
        started = time.perf_counter()
        try:
            config = build_run_config(call.record, call.coordinate, call.route_id)
            result = solve_v2(config)
            ref_data = load_v2_reference(call.record.ref_path)
            metric = evaluate_fixed5_mae(
                result,
                ref_data,
                self._timelines[call.record.record_id],
            )
            elapsed = time.perf_counter() - started
            del result
            return CompactCellMetric(
                experiment_id=self.experiment_id,
                algorithm_sha256=call.config_sha256,
                source_sha256=self.snapshot.semantic_sha256,
                input_sha256=_input_sha256(call.record),
                metric_contract_sha256=METRIC_CONTRACT_SHA256,
                code_sha256=self.code_sha256,
                call_identity_sha256=call.identity_sha256,
                route_id=call.route_id,
                reference_groups_order=ROUTE_GROUPS[call.route_id],
                scene=call.record.scene,
                record_id=call.record.record_id,
                coordinate_id=call.coordinate.coordinate_id,
                coordinate_index=call.coordinate.coordinate_index,
                fs_target_hz=call.coordinate.fs_target_hz,
                memory_ms=call.coordinate.memory_ms,
                mu_base=call.coordinate.mu_base,
                exclusion_half_width_bpm=call.coordinate.exclusion_half_width_bpm,
                evaluation_window_count=metric.evaluation_window_count,
                evaluation_window_sha256=metric.evaluation_window_sha256,
                mae_bpm=metric.mae_bpm,
                solver_elapsed_s=elapsed,
                attempt_count=attempt_number,
                completed_at=_now_iso(),
            )
        except TechnicalCellError as error:
            return TechnicalAttemptEvent(
                route_id=call.route_id,
                record_id=call.record.record_id,
                coordinate_id=call.coordinate.coordinate_id,
                call_identity_sha256=call.identity_sha256,
                attempt_number=attempt_number,
                reason_code=error.reason_code,
                detail=error.detail,
                occurred_at=_now_iso(),
            )
        except Exception as error:  # Solver failures are technical evidence, not MAE.
            return TechnicalAttemptEvent(
                route_id=call.route_id,
                record_id=call.record.record_id,
                coordinate_id=call.coordinate.coordinate_id,
                call_identity_sha256=call.identity_sha256,
                attempt_number=attempt_number,
                reason_code="solver_exception",
                detail=f"{type(error).__name__}:{error}",
                occurred_at=_now_iso(),
            )

    def _next_attempt_number(self, call: ReferenceArmCall) -> int:
        prior_attempts = int(
            self.ledger.connection.execute(
                """
                SELECT COUNT(*) FROM attempt_events
                WHERE route_id=? AND record_id=? AND coordinate_id=?
                """,
                (call.route_id, call.record.record_id, call.coordinate.coordinate_id),
            ).fetchone()[0]
        )
        return prior_attempts + 1


def prepare_reference_arm_experiment(
    repo_root: Path,
    output_root: Path,
    *,
    source_commit: str = FROZEN_SOURCE_COMMIT,
    require_clean: bool = True,
) -> ReferenceArmExperimentContext:
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root).resolve()
    if require_clean:
        _require_clean_worktree(repo_root)
    snapshot = materialise_source_snapshot(
        repo_root,
        output_root / "source",
        source_commit,
    )
    code_sha = current_code_sha256(repo_root)
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "source_commit": snapshot.source_commit,
        "source_sha256": snapshot.semantic_sha256,
        "code_sha256": code_sha,
        "metric_contract_sha256": METRIC_CONTRACT_SHA256,
        "routes": {key: list(value) for key, value in ROUTE_GROUPS.items()},
    }
    ledger = CompactResponseLedger.create(output_root / "ledger.sqlite3", identity)
    _import_hf_cells(ledger, snapshot)
    calls = build_reference_arm_calls(snapshot, code_sha256=code_sha)
    context = ReferenceArmExperimentContext(
        repo_root=repo_root,
        output_root=output_root,
        snapshot=snapshot,
        ledger=ledger,
        calls=calls,
        experiment_identity=identity,
        code_sha256=code_sha,
    )
    _write_json(
        output_root / "preflight_receipt.json",
        {
            "schema_id": "lyx_reference_arm_preflight_v1",
            "experiment_identity": identity,
            "record_count": len(snapshot.records),
            "scene_count": len({row.scene for row in snapshot.records}),
            "coordinate_count": len(snapshot.coordinates),
            "hf_cell_count": ledger.complete_count("HF"),
            "hf_selection_count": len(snapshot.hf_selections),
            "timeline_count": len(snapshot.timelines),
            "new_call_count": len(calls),
            "solver_invocation_count": 0,
        },
    )
    return context


def current_code_sha256(repo_root: Path) -> str:
    tree = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD:python/src/ppg_hr"],
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    ).strip()
    return _semantic_sha256({"ppg_hr_tree": tree})


def _require_clean_worktree(repo_root: Path) -> None:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = completed.stdout.strip()
    if output:
        raise RuntimeError(f"worktree_not_clean:\n{output}")


def _import_hf_cells(ledger: CompactResponseLedger, snapshot: ReferenceArmSourceSnapshot) -> None:
    record_by_id = {row.record_id: row for row in snapshot.records}
    source_code_sha = _semantic_sha256({"source_commit": snapshot.source_commit})
    for cell in snapshot.hf_cells:
        record = record_by_id[cell.record_id]
        ledger.record_complete(_imported_hf_metric(cell, record, snapshot, source_code_sha))


def _imported_hf_metric(
    cell: ImportedHFCell,
    record: RecordIdentity,
    snapshot: ReferenceArmSourceSnapshot,
    source_code_sha: str,
) -> CompactCellMetric:
    identity = {
        "route_id": "HF",
        "record_id": cell.record_id,
        "coordinate_id": cell.coordinate_id,
        "source_partition_sha256": cell.source_partition_sha256,
        "source_commit": snapshot.source_commit,
    }
    return CompactCellMetric(
        experiment_id=EXPERIMENT_ID,
        algorithm_sha256=source_code_sha,
        source_sha256=snapshot.semantic_sha256,
        input_sha256=_input_sha256(record),
        metric_contract_sha256=METRIC_CONTRACT_SHA256,
        code_sha256=source_code_sha,
        call_identity_sha256=_semantic_sha256(identity),
        route_id="HF",
        reference_groups_order=("HF",),
        scene=cell.scene,
        record_id=cell.record_id,
        coordinate_id=cell.coordinate_id,
        coordinate_index=cell.coordinate_index,
        fs_target_hz=cell.fs_target_hz,
        memory_ms=cell.memory_ms,
        mu_base=cell.mu_base,
        exclusion_half_width_bpm=cell.exclusion_half_width_bpm,
        evaluation_window_count=cell.evaluation_window_count,
        evaluation_window_sha256=cell.evaluation_window_sha256,
        mae_bpm=cell.mae_bpm,
        solver_elapsed_s=0.0,
        attempt_count=1,
        completed_at="frozen-source-import",
    )


def _config_identity_payload(config: V2RunConfig, record: RecordIdentity) -> dict[str, Any]:
    payload = asdict(config)
    payload["data_path"] = record.data_relative_path
    payload["ref_path"] = record.ref_relative_path
    return _json_ready(payload)


def _input_sha256(record: RecordIdentity) -> str:
    return _semantic_sha256({"data_sha256": record.data_sha256, "ref_sha256": record.ref_sha256})


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
