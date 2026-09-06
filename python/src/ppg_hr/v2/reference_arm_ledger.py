"""Exact fixed-window metric and compact SQLite response ledger."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .reference_arm_source import EvaluationTimeline
from .solver import V2SolverResult


class TechnicalCellError(RuntimeError):
    """A cell cannot be evaluated without changing its frozen identity."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        self.detail = detail
        message = reason_code if not detail else f"{reason_code}:{detail}"
        super().__init__(message)


class LedgerIdentityError(RuntimeError):
    """The ledger identity or an existing primary-key value conflicts."""


@dataclass(frozen=True)
class Fixed5Metric:
    mae_bpm: float
    evaluation_window_count: int
    evaluation_window_sha256: str


@dataclass(frozen=True)
class CompactCellMetric:
    experiment_id: str
    algorithm_sha256: str
    source_sha256: str
    input_sha256: str
    metric_contract_sha256: str
    code_sha256: str
    call_identity_sha256: str
    route_id: str
    reference_groups_order: tuple[str, ...]
    scene: str
    record_id: str
    coordinate_id: str
    coordinate_index: int
    fs_target_hz: int
    memory_ms: int
    mu_base: float
    exclusion_half_width_bpm: int
    evaluation_window_count: int
    evaluation_window_sha256: str
    mae_bpm: float
    solver_elapsed_s: float
    attempt_count: int
    completed_at: str


@dataclass(frozen=True)
class TechnicalAttemptEvent:
    route_id: str
    record_id: str
    coordinate_id: str
    call_identity_sha256: str
    attempt_number: int
    reason_code: str
    detail: str
    occurred_at: str


CELL_COLUMNS = (
    "experiment_id",
    "algorithm_sha256",
    "source_sha256",
    "input_sha256",
    "metric_contract_sha256",
    "code_sha256",
    "call_identity_sha256",
    "route_id",
    "reference_groups_order",
    "scene",
    "record_id",
    "coordinate_id",
    "coordinate_index",
    "fs_target_hz",
    "memory_ms",
    "mu_base",
    "exclusion_half_width_bpm",
    "evaluation_window_count",
    "evaluation_window_sha256",
    "mae_bpm",
    "solver_elapsed_s",
    "attempt_count",
    "completed_at",
)


def evaluate_fixed5_mae(
    result: V2SolverResult,
    ref_data: np.ndarray,
    timeline: EvaluationTimeline,
) -> Fixed5Metric:
    """Evaluate Final on every exact frozen key at the frozen time bias."""

    hr = np.asarray(result.HR, dtype=float)
    if hr.ndim != 2 or hr.shape[1] < 4:
        raise TechnicalCellError("solver_hr_shape")
    predictions: list[float] = []
    for window_idx, center_s in timeline.window_keys:
        if window_idx < 0 or window_idx >= hr.shape[0]:
            raise TechnicalCellError("missing_window", str(window_idx))
        row = hr[window_idx]
        actual_center = float(row[0])
        if not math.isfinite(actual_center) or abs(actual_center - center_s) > 1e-9:
            raise TechnicalCellError(
                "center_mismatch",
                f"index={window_idx},expected={center_s},actual={actual_center}",
            )
        prediction = float(row[3])
        if not math.isfinite(prediction):
            raise TechnicalCellError("nonfinite_final", str(window_idx))
        predictions.append(prediction)

    reference = np.asarray(ref_data, dtype=float)
    if reference.ndim != 2 or reference.shape[1] < 2 or reference.shape[0] < 2:
        raise TechnicalCellError("reference_shape")
    times = reference[:, 0]
    values = reference[:, 1]
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise TechnicalCellError("nonfinite_reference")
    if np.any(np.diff(times) <= 0.0):
        raise TechnicalCellError("reference_time_not_strictly_increasing")
    targets = np.asarray(
        [center + timeline.time_bias_s for _, center in timeline.window_keys],
        dtype=float,
    )
    if targets.size == 0:
        raise TechnicalCellError("missing_window", "empty_timeline")
    if float(np.min(targets)) < float(times[0]) or float(np.max(targets)) > float(times[-1]):
        raise TechnicalCellError("reference_out_of_range")
    references = np.interp(targets, times, values)
    errors = np.abs(np.asarray(predictions, dtype=float) - references)
    if not np.all(np.isfinite(errors)):
        raise TechnicalCellError("nonfinite_final_or_reference")
    return Fixed5Metric(
        mae_bpm=float(np.mean(errors)),
        evaluation_window_count=len(timeline.window_keys),
        evaluation_window_sha256=timeline.window_sha256,
    )


class CompactResponseLedger:
    """Single-writer compact cell ledger with strict idempotence."""

    def __init__(self, path: Path, connection: sqlite3.Connection) -> None:
        self.path = path
        self.connection = connection
        self.connection.row_factory = sqlite3.Row

    @classmethod
    def create(cls, path: Path, identity: Mapping[str, Any]) -> CompactResponseLedger:
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        ledger = cls(path, connection)
        ledger._create_schema()
        ledger._bind_or_verify_identity(identity)
        return ledger

    @classmethod
    def open(cls, path: Path, expected_identity: Mapping[str, Any]) -> CompactResponseLedger:
        path = Path(path).resolve()
        if not path.is_file():
            raise LedgerIdentityError(f"ledger_missing:{path}")
        connection = sqlite3.connect(path)
        ledger = cls(path, connection)
        ledger._create_schema()
        ledger._verify_identity(expected_identity)
        return ledger

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS ledger_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cell_metrics (
                experiment_id TEXT NOT NULL,
                algorithm_sha256 TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,
                metric_contract_sha256 TEXT NOT NULL,
                code_sha256 TEXT NOT NULL,
                call_identity_sha256 TEXT NOT NULL,
                route_id TEXT NOT NULL,
                reference_groups_order TEXT NOT NULL,
                scene TEXT NOT NULL,
                record_id TEXT NOT NULL,
                coordinate_id TEXT NOT NULL,
                coordinate_index INTEGER NOT NULL CHECK(coordinate_index >= 0),
                fs_target_hz INTEGER NOT NULL CHECK(fs_target_hz > 0),
                memory_ms INTEGER NOT NULL CHECK(memory_ms > 0),
                mu_base REAL NOT NULL CHECK(mu_base > 0),
                exclusion_half_width_bpm INTEGER NOT NULL CHECK(exclusion_half_width_bpm > 0),
                evaluation_window_count INTEGER NOT NULL CHECK(evaluation_window_count > 0),
                evaluation_window_sha256 TEXT NOT NULL,
                mae_bpm REAL NOT NULL CHECK(mae_bpm >= 0),
                solver_elapsed_s REAL NOT NULL CHECK(solver_elapsed_s >= 0),
                attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
                completed_at TEXT NOT NULL,
                PRIMARY KEY(route_id, record_id, coordinate_id)
            );
            CREATE TABLE IF NOT EXISTS attempt_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_id TEXT NOT NULL,
                record_id TEXT NOT NULL,
                coordinate_id TEXT NOT NULL,
                call_identity_sha256 TEXT NOT NULL,
                attempt_number INTEGER NOT NULL CHECK(attempt_number >= 1),
                reason_code TEXT NOT NULL,
                detail TEXT NOT NULL,
                occurred_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def _bind_or_verify_identity(self, identity: Mapping[str, Any]) -> None:
        encoded = _canonical_json(identity)
        row = self.connection.execute(
            "SELECT value FROM ledger_metadata WHERE key='experiment_identity'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO ledger_metadata(key, value) VALUES (?, ?)",
                ("experiment_identity", encoded),
            )
            self.connection.commit()
            return
        if str(row[0]) != encoded:
            raise LedgerIdentityError("experiment_identity mismatch")

    def _verify_identity(self, identity: Mapping[str, Any]) -> None:
        row = self.connection.execute(
            "SELECT value FROM ledger_metadata WHERE key='experiment_identity'"
        ).fetchone()
        if row is None or str(row[0]) != _canonical_json(identity):
            raise LedgerIdentityError("experiment_identity mismatch")

    def record_complete(self, row: CompactCellMetric) -> None:
        values = _cell_values(row)
        if not math.isfinite(row.mae_bpm) or row.mae_bpm < 0.0:
            raise LedgerIdentityError("nonfinite_or_negative_mae")
        if not math.isfinite(row.solver_elapsed_s) or row.solver_elapsed_s < 0.0:
            raise LedgerIdentityError("nonfinite_or_negative_solver_elapsed")
        existing = self.connection.execute(
            f"SELECT {', '.join(CELL_COLUMNS)} FROM cell_metrics "
            "WHERE route_id=? AND record_id=? AND coordinate_id=?",
            (row.route_id, row.record_id, row.coordinate_id),
        ).fetchone()
        if existing is not None:
            if tuple(existing[column] for column in CELL_COLUMNS) != values:
                raise LedgerIdentityError(
                    f"conflicting_cell:{row.route_id}:{row.record_id}:{row.coordinate_id}"
                )
            return
        placeholders = ", ".join("?" for _ in CELL_COLUMNS)
        self.connection.execute(
            f"INSERT INTO cell_metrics ({', '.join(CELL_COLUMNS)}) VALUES ({placeholders})",
            values,
        )
        self.connection.commit()

    def record_attempt(self, event: TechnicalAttemptEvent) -> None:
        self.connection.execute(
            """
            INSERT INTO attempt_events (
                route_id, record_id, coordinate_id, call_identity_sha256,
                attempt_number, reason_code, detail, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.route_id,
                event.record_id,
                event.coordinate_id,
                event.call_identity_sha256,
                event.attempt_number,
                event.reason_code,
                event.detail,
                event.occurred_at,
            ),
        )
        self.connection.commit()

    def has_complete(self, route_id: str, record_id: str, coordinate_id: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM cell_metrics WHERE route_id=? AND record_id=? AND coordinate_id=?",
            (route_id, record_id, coordinate_id),
        ).fetchone()
        return row is not None

    def complete_count(self, route_id: str | None = None) -> int:
        if route_id is None:
            row = self.connection.execute("SELECT COUNT(*) FROM cell_metrics").fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) FROM cell_metrics WHERE route_id=?", (route_id,)
            ).fetchone()
        return int(row[0])

    def export_canonical_csv(self, path: Path) -> str:
        rows = self.connection.execute(
            f"SELECT {', '.join(CELL_COLUMNS)} FROM cell_metrics "
            "ORDER BY route_id, scene, record_id, coordinate_index"
        ).fetchall()
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(CELL_COLUMNS)
        for row in rows:
            writer.writerow([row[column] for column in CELL_COLUMNS])
        payload = stream.getvalue().encode("utf-8")
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
        return hashlib.sha256(payload).hexdigest()


def _cell_values(row: CompactCellMetric) -> tuple[Any, ...]:
    payload = asdict(row)
    payload["reference_groups_order"] = json.dumps(
        list(row.reference_groups_order), separators=(",", ":")
    )
    return tuple(payload[column] for column in CELL_COLUMNS)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
