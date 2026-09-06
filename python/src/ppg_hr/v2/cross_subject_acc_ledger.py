"""Compact SQLite ledger for the independent cross-subject ACC response surface."""

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


class AccLedgerIdentityError(RuntimeError):
    """The ACC ledger identity or a completed cell conflicts."""


@dataclass(frozen=True)
class AccCompactCellMetric:
    experiment_id: str
    parent_experiment_id: str
    algorithm_sha256: str
    runner_sha256: str
    dataset_sha256: str
    input_sha256: str
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
    mae_bpm: float
    l10: int
    l20: int
    e10: int
    e20: int
    right_censored_recovery_count: int
    full_window_count: int
    reliable_window_count: int
    motion_window_count: int
    true_rise_applicable: bool
    true_rise_underestimate_bpm: float | None
    true_rise_episode_count: int
    spectral_gate_contract_v2: bool
    stability_pass: bool
    reference_groups_order_json: str
    adaptive_reference_stage_limit: int | None
    evaluation_window_sha256: str
    solver_elapsed_s: float
    attempt_count: int
    completed_at: str


@dataclass(frozen=True)
class AccTechnicalAttemptEvent:
    route_id: str
    physical_subject_id: str
    record_id: str
    coordinate_id: str
    call_identity_sha256: str
    attempt_number: int
    reason_code: str
    detail: str
    occurred_at: str


CELL_COLUMNS = tuple(AccCompactCellMetric.__dataclass_fields__)
BOOLEAN_COLUMNS = {
    "true_rise_applicable",
    "spectral_gate_contract_v2",
    "stability_pass",
}


class AccCompactResponseLedger:
    """Single-writer ACC ledger with strict idempotence and stable export."""

    def __init__(self, path: Path, connection: sqlite3.Connection) -> None:
        self.path = Path(path)
        self.connection = connection
        self.connection.row_factory = sqlite3.Row

    @classmethod
    def create(cls, path: Path, identity: Mapping[str, Any]) -> AccCompactResponseLedger:
        resolved = Path(path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        ledger = cls(resolved, sqlite3.connect(resolved))
        ledger._create_schema()
        ledger._bind_or_verify_identity(identity)
        return ledger

    @classmethod
    def open(cls, path: Path, expected_identity: Mapping[str, Any]) -> AccCompactResponseLedger:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise AccLedgerIdentityError(f"ledger_missing:{resolved}")
        ledger = cls(resolved, sqlite3.connect(resolved))
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
                parent_experiment_id TEXT NOT NULL,
                algorithm_sha256 TEXT NOT NULL,
                runner_sha256 TEXT NOT NULL,
                dataset_sha256 TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,
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
                mae_bpm REAL NOT NULL CHECK(mae_bpm >= 0),
                l10 INTEGER NOT NULL CHECK(l10 >= 0),
                l20 INTEGER NOT NULL CHECK(l20 >= 0),
                e10 INTEGER NOT NULL CHECK(e10 >= 0),
                e20 INTEGER NOT NULL CHECK(e20 >= 0),
                right_censored_recovery_count INTEGER NOT NULL CHECK(right_censored_recovery_count >= 0),
                full_window_count INTEGER NOT NULL CHECK(full_window_count > 0),
                reliable_window_count INTEGER NOT NULL CHECK(reliable_window_count > 0),
                motion_window_count INTEGER NOT NULL CHECK(motion_window_count >= 0),
                true_rise_applicable INTEGER NOT NULL CHECK(true_rise_applicable IN (0, 1)),
                true_rise_underestimate_bpm REAL,
                true_rise_episode_count INTEGER NOT NULL CHECK(true_rise_episode_count >= 0),
                spectral_gate_contract_v2 INTEGER NOT NULL CHECK(spectral_gate_contract_v2 IN (0, 1)),
                stability_pass INTEGER NOT NULL CHECK(stability_pass IN (0, 1)),
                reference_groups_order_json TEXT NOT NULL,
                adaptive_reference_stage_limit INTEGER,
                evaluation_window_sha256 TEXT NOT NULL,
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
            raise AccLedgerIdentityError("experiment_identity mismatch")

    def _verify_identity(self, identity: Mapping[str, Any]) -> None:
        row = self.connection.execute(
            "SELECT value FROM ledger_metadata WHERE key='experiment_identity'"
        ).fetchone()
        if row is None or str(row[0]) != _canonical_json(identity):
            raise AccLedgerIdentityError("experiment_identity mismatch")

    def record_complete(self, row: AccCompactCellMetric) -> None:
        self.record_outcomes_batch((row,), ())

    def record_complete_batch(self, rows: tuple[AccCompactCellMetric, ...]) -> None:
        self.record_outcomes_batch(rows, ())

    def record_attempt(self, event: AccTechnicalAttemptEvent) -> None:
        self.record_outcomes_batch((), (event,))

    def record_outcomes_batch(
        self,
        completed: tuple[AccCompactCellMetric, ...],
        attempts: tuple[AccTechnicalAttemptEvent, ...],
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

    def _record_complete_without_commit(self, row: AccCompactCellMetric) -> None:
        _validate_cell(row)
        key = (row.route_id, row.physical_subject_id, row.record_id, row.coordinate_id)
        existing = self.connection.execute(
            f"SELECT {', '.join(CELL_COLUMNS)} FROM cell_metrics "
            "WHERE route_id=? AND physical_subject_id=? AND record_id=? AND coordinate_id=?",
            key,
        ).fetchone()
        values = _cell_values(row)
        if existing is not None:
            if tuple(existing[column] for column in CELL_COLUMNS) != values:
                raise AccLedgerIdentityError("conflicting_cell:" + ":".join(key))
            return
        placeholders = ", ".join("?" for _ in CELL_COLUMNS)
        self.connection.execute(
            f"INSERT INTO cell_metrics ({', '.join(CELL_COLUMNS)}) VALUES ({placeholders})",
            values,
        )

    def complete_count(self, route_id: str | None = None) -> int:
        if route_id is None:
            row = self.connection.execute("SELECT COUNT(*) FROM cell_metrics").fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) FROM cell_metrics WHERE route_id=?", (route_id,)
            ).fetchone()
        return int(row[0])

    def attempt_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM attempt_events").fetchone()[0])

    def completed_keys(self, route_id: str) -> set[tuple[str, str, str]]:
        rows = self.connection.execute(
            """
            SELECT physical_subject_id, record_id, coordinate_id
            FROM cell_metrics WHERE route_id=?
            """,
            (route_id,),
        ).fetchall()
        return {(str(row[0]), str(row[1]), str(row[2])) for row in rows}

    def attempt_counts(self) -> dict[tuple[str, str, str, str], int]:
        rows = self.connection.execute(
            """
            SELECT route_id, physical_subject_id, record_id, coordinate_id, COUNT(*)
            FROM attempt_events
            GROUP BY route_id, physical_subject_id, record_id, coordinate_id
            """
        ).fetchall()
        return {(str(a), str(b), str(c), str(d)): int(count) for a, b, c, d, count in rows}

    def next_attempt_number(
        self, route_id: str, physical_subject_id: str, record_id: str, coordinate_id: str
    ) -> int:
        row = self.connection.execute(
            """
            SELECT COUNT(*) FROM attempt_events
            WHERE route_id=? AND physical_subject_id=? AND record_id=? AND coordinate_id=?
            """,
            (route_id, physical_subject_id, record_id, coordinate_id),
        ).fetchone()
        return int(row[0]) + 1

    def read_complete_cells(self, route_id: str | None = None) -> tuple[AccCompactCellMetric, ...]:
        where = "" if route_id is None else "WHERE route_id=? "
        parameters: tuple[str, ...] = () if route_id is None else (route_id,)
        rows = self.connection.execute(
            f"SELECT {', '.join(CELL_COLUMNS)} FROM cell_metrics {where}"
            "ORDER BY route_id, scene, physical_subject_id, record_id, coordinate_index",
            parameters,
        ).fetchall()
        return tuple(
            AccCompactCellMetric(
                **{
                    column: (bool(row[column]) if column in BOOLEAN_COLUMNS else row[column])
                    for column in CELL_COLUMNS
                }
            )
            for row in rows
        )

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
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
        return hashlib.sha256(payload).hexdigest()


def _validate_cell(row: AccCompactCellMetric) -> None:
    numeric = (row.mae_bpm, row.solver_elapsed_s)
    if not all(math.isfinite(float(value)) and float(value) >= 0 for value in numeric):
        raise AccLedgerIdentityError("nonfinite_or_negative_metric")
    try:
        groups = json.loads(row.reference_groups_order_json)
    except json.JSONDecodeError as error:
        raise AccLedgerIdentityError("invalid_reference_groups_json") from error
    if groups != ["ACC"] or row.route_id != "ACC":
        raise AccLedgerIdentityError("acc_route_identity_mismatch")


def _cell_values(row: AccCompactCellMetric) -> tuple[Any, ...]:
    payload = asdict(row)
    return tuple(payload[column] for column in CELL_COLUMNS)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
