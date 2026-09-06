"""Compact SQLite response ledger for cross-subject HF cells."""

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


class LedgerIdentityError(RuntimeError):
    """The ledger identity or a completed cell conflicts."""


@dataclass(frozen=True)
class CompactCellMetric:
    experiment_id: str
    algorithm_sha256: str
    runner_sha256: str
    dataset_sha256: str
    input_sha256: str
    baseline_sha256: str
    metric_contract_sha256: str
    call_identity_sha256: str
    route_id: str
    physical_subject_id: str
    scene: str
    record_id: str
    coordinate_id: str
    coordinate_index: int
    fs_target_hz: int
    memory_ms: int
    mu_base: float
    exclusion_half_width_bpm: int
    candidate_mae_bpm: float
    candidate_l10: int
    candidate_l20: int
    candidate_e10: int
    candidate_e20: int
    candidate_right_censored_recovery_count: int
    candidate_full_window_count: int
    candidate_reliable_window_count: int
    candidate_motion_window_count: int
    candidate_evaluation_window_sha256: str
    baseline_mae_bpm: float
    baseline_l10: int
    baseline_l20: int
    baseline_right_censored_recovery_count: int
    g1i_pass: bool
    g2_pass: bool
    g3_pass: bool
    g4_pass: bool
    g5_pass: bool
    g7_pass: bool
    qualified: bool
    g2_margin_s: float
    g3_margin_s: float
    g4_margin_bpm: float
    g5_right_censored_count: int
    g7_margin_s: float
    failed_gates_json: str
    solver_elapsed_s: float
    attempt_count: int
    completed_at: str


@dataclass(frozen=True)
class TechnicalAttemptEvent:
    route_id: str
    physical_subject_id: str
    record_id: str
    coordinate_id: str
    call_identity_sha256: str
    attempt_number: int
    reason_code: str
    detail: str
    occurred_at: str


CELL_COLUMNS = tuple(CompactCellMetric.__dataclass_fields__)


class CompactResponseLedger:
    """Single-writer ledger with exact idempotence and stable export."""

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
        ledger = cls(path, sqlite3.connect(path))
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
                runner_sha256 TEXT NOT NULL,
                dataset_sha256 TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,
                baseline_sha256 TEXT NOT NULL,
                metric_contract_sha256 TEXT NOT NULL,
                call_identity_sha256 TEXT NOT NULL,
                route_id TEXT NOT NULL,
                physical_subject_id TEXT NOT NULL,
                scene TEXT NOT NULL,
                record_id TEXT NOT NULL,
                coordinate_id TEXT NOT NULL,
                coordinate_index INTEGER NOT NULL CHECK(coordinate_index >= 0),
                fs_target_hz INTEGER NOT NULL CHECK(fs_target_hz > 0),
                memory_ms INTEGER NOT NULL CHECK(memory_ms > 0),
                mu_base REAL NOT NULL CHECK(mu_base > 0),
                exclusion_half_width_bpm INTEGER NOT NULL CHECK(exclusion_half_width_bpm > 0),
                candidate_mae_bpm REAL NOT NULL CHECK(candidate_mae_bpm >= 0),
                candidate_l10 INTEGER NOT NULL CHECK(candidate_l10 >= 0),
                candidate_l20 INTEGER NOT NULL CHECK(candidate_l20 >= 0),
                candidate_e10 INTEGER NOT NULL CHECK(candidate_e10 >= 0),
                candidate_e20 INTEGER NOT NULL CHECK(candidate_e20 >= 0),
                candidate_right_censored_recovery_count INTEGER NOT NULL CHECK(candidate_right_censored_recovery_count >= 0),
                candidate_full_window_count INTEGER NOT NULL CHECK(candidate_full_window_count > 0),
                candidate_reliable_window_count INTEGER NOT NULL CHECK(candidate_reliable_window_count > 0),
                candidate_motion_window_count INTEGER NOT NULL CHECK(candidate_motion_window_count >= 0),
                candidate_evaluation_window_sha256 TEXT NOT NULL,
                baseline_mae_bpm REAL NOT NULL CHECK(baseline_mae_bpm >= 0),
                baseline_l10 INTEGER NOT NULL CHECK(baseline_l10 >= 0),
                baseline_l20 INTEGER NOT NULL CHECK(baseline_l20 >= 0),
                baseline_right_censored_recovery_count INTEGER NOT NULL CHECK(baseline_right_censored_recovery_count >= 0),
                g1i_pass INTEGER NOT NULL CHECK(g1i_pass IN (0, 1)),
                g2_pass INTEGER NOT NULL CHECK(g2_pass IN (0, 1)),
                g3_pass INTEGER NOT NULL CHECK(g3_pass IN (0, 1)),
                g4_pass INTEGER NOT NULL CHECK(g4_pass IN (0, 1)),
                g5_pass INTEGER NOT NULL CHECK(g5_pass IN (0, 1)),
                g7_pass INTEGER NOT NULL CHECK(g7_pass IN (0, 1)),
                qualified INTEGER NOT NULL CHECK(qualified IN (0, 1)),
                g2_margin_s REAL NOT NULL,
                g3_margin_s REAL NOT NULL,
                g4_margin_bpm REAL NOT NULL,
                g5_right_censored_count INTEGER NOT NULL CHECK(g5_right_censored_count >= 0),
                g7_margin_s REAL NOT NULL,
                failed_gates_json TEXT NOT NULL,
                solver_elapsed_s REAL NOT NULL CHECK(solver_elapsed_s >= 0),
                attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
                completed_at TEXT NOT NULL,
                PRIMARY KEY(route_id, physical_subject_id, record_id, coordinate_id)
            );
            CREATE TABLE IF NOT EXISTS attempt_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_id TEXT NOT NULL,
                physical_subject_id TEXT NOT NULL,
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
        elif str(row[0]) != encoded:
            raise LedgerIdentityError("experiment_identity mismatch")

    def _verify_identity(self, identity: Mapping[str, Any]) -> None:
        row = self.connection.execute(
            "SELECT value FROM ledger_metadata WHERE key='experiment_identity'"
        ).fetchone()
        if row is None or str(row[0]) != _canonical_json(identity):
            raise LedgerIdentityError("experiment_identity mismatch")

    def record_complete(self, row: CompactCellMetric) -> None:
        self.record_outcomes_batch((row,), ())

    def record_complete_batch(self, rows: tuple[CompactCellMetric, ...]) -> None:
        self.record_outcomes_batch(rows, ())

    def record_attempt(self, event: TechnicalAttemptEvent) -> None:
        self.record_outcomes_batch((), (event,))

    def record_outcomes_batch(
        self,
        completed: tuple[CompactCellMetric, ...],
        attempts: tuple[TechnicalAttemptEvent, ...],
    ) -> None:
        try:
            for row in completed:
                self._record_complete_without_commit(row)
            for event in attempts:
                self.connection.execute(
                    """
                    INSERT INTO attempt_events (
                        route_id, physical_subject_id, record_id, coordinate_id,
                        call_identity_sha256, attempt_number, reason_code, detail, occurred_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.route_id,
                        event.physical_subject_id,
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
        except Exception:
            self.connection.rollback()
            raise

    def _record_complete_without_commit(self, row: CompactCellMetric) -> None:
        _validate_cell(row)
        values = _cell_values(row)
        key = (
            row.route_id,
            row.physical_subject_id,
            row.record_id,
            row.coordinate_id,
        )
        existing = self.connection.execute(
            f"SELECT {', '.join(CELL_COLUMNS)} FROM cell_metrics "
            "WHERE route_id=? AND physical_subject_id=? AND record_id=? AND coordinate_id=?",
            key,
        ).fetchone()
        if existing is not None:
            if tuple(existing[column] for column in CELL_COLUMNS) != values:
                raise LedgerIdentityError("conflicting_cell:" + ":".join(key))
            return
        placeholders = ", ".join("?" for _ in CELL_COLUMNS)
        self.connection.execute(
            f"INSERT INTO cell_metrics ({', '.join(CELL_COLUMNS)}) VALUES ({placeholders})",
            values,
        )

    def has_complete(
        self,
        route_id: str,
        physical_subject_id: str,
        record_id: str,
        coordinate_id: str,
    ) -> bool:
        row = self.connection.execute(
            """
            SELECT 1 FROM cell_metrics
            WHERE route_id=? AND physical_subject_id=? AND record_id=? AND coordinate_id=?
            """,
            (route_id, physical_subject_id, record_id, coordinate_id),
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

    def attempt_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM attempt_events").fetchone()
        return int(row[0])

    def completed_keys(self, route_id: str) -> set[tuple[str, str, str]]:
        rows = self.connection.execute(
            """
            SELECT physical_subject_id, record_id, coordinate_id
            FROM cell_metrics WHERE route_id=?
            """,
            (route_id,),
        ).fetchall()
        return {(str(row[0]), str(row[1]), str(row[2])) for row in rows}

    def read_complete_cells(self, route_id: str | None = None) -> tuple[CompactCellMetric, ...]:
        if route_id is None:
            rows = self.connection.execute(
                f"SELECT {', '.join(CELL_COLUMNS)} FROM cell_metrics "
                "ORDER BY route_id, scene, physical_subject_id, record_id, coordinate_index"
            ).fetchall()
        else:
            rows = self.connection.execute(
                f"SELECT {', '.join(CELL_COLUMNS)} FROM cell_metrics "
                "WHERE route_id=? "
                "ORDER BY route_id, scene, physical_subject_id, record_id, coordinate_index",
                (route_id,),
            ).fetchall()
        boolean_columns = {
            "g1i_pass",
            "g2_pass",
            "g3_pass",
            "g4_pass",
            "g5_pass",
            "g7_pass",
            "qualified",
        }
        return tuple(
            CompactCellMetric(
                **{
                    column: (bool(row[column]) if column in boolean_columns else row[column])
                    for column in CELL_COLUMNS
                }
            )
            for row in rows
        )

    def attempt_counts(self) -> dict[tuple[str, str, str, str], int]:
        rows = self.connection.execute(
            """
            SELECT route_id, physical_subject_id, record_id, coordinate_id, COUNT(*)
            FROM attempt_events
            GROUP BY route_id, physical_subject_id, record_id, coordinate_id
            """
        ).fetchall()
        return {(str(row[0]), str(row[1]), str(row[2]), str(row[3])): int(row[4]) for row in rows}

    def next_attempt_number(
        self,
        route_id: str,
        physical_subject_id: str,
        record_id: str,
        coordinate_id: str,
    ) -> int:
        row = self.connection.execute(
            """
            SELECT COUNT(*) FROM attempt_events
            WHERE route_id=? AND physical_subject_id=? AND record_id=? AND coordinate_id=?
            """,
            (route_id, physical_subject_id, record_id, coordinate_id),
        ).fetchone()
        return int(row[0]) + 1

    def export_canonical_csv(self, path: Path) -> str:
        rows = self.connection.execute(
            f"SELECT {', '.join(CELL_COLUMNS)} FROM cell_metrics "
            "ORDER BY route_id, scene, physical_subject_id, record_id, coordinate_index"
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


def _validate_cell(row: CompactCellMetric) -> None:
    values = (
        row.candidate_mae_bpm,
        row.baseline_mae_bpm,
        row.g2_margin_s,
        row.g3_margin_s,
        row.g4_margin_bpm,
        row.g7_margin_s,
        row.solver_elapsed_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise LedgerIdentityError("nonfinite_cell_metric")
    if row.candidate_mae_bpm < 0 or row.baseline_mae_bpm < 0:
        raise LedgerIdentityError("negative_mae")
    try:
        failed = json.loads(row.failed_gates_json)
    except json.JSONDecodeError as error:
        raise LedgerIdentityError("invalid_failed_gates_json") from error
    if not isinstance(failed, list):
        raise LedgerIdentityError("invalid_failed_gates_json")


def _cell_values(row: CompactCellMetric) -> tuple[Any, ...]:
    payload = asdict(row)
    return tuple(payload[column] for column in CELL_COLUMNS)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
