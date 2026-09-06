from __future__ import annotations

from pathlib import Path

import pytest

from ppg_hr.v2.cross_subject_loso_ledger import (
    CELL_COLUMNS,
    CompactCellMetric,
    CompactResponseLedger,
    LedgerIdentityError,
    TechnicalAttemptEvent,
)


def _cell(*, mae_bpm: float = 1.25) -> CompactCellMetric:
    return CompactCellMetric(
        experiment_id="experiment",
        algorithm_sha256="a" * 64,
        runner_sha256="9" * 64,
        dataset_sha256="b" * 64,
        input_sha256="c" * 64,
        baseline_sha256="d" * 64,
        metric_contract_sha256="e" * 64,
        call_identity_sha256="f" * 64,
        route_id="HF",
        physical_subject_id="TS",
        scene="xiezi",
        record_id="xiezi3_TS_0709",
        coordinate_id="physical4d:fs025:m040:mu0006:w003",
        coordinate_index=0,
        fs_target_hz=25,
        memory_ms=40,
        mu_base=0.006,
        exclusion_half_width_bpm=3,
        candidate_mae_bpm=mae_bpm,
        candidate_l10=0,
        candidate_l20=0,
        candidate_e10=0,
        candidate_e20=0,
        candidate_right_censored_recovery_count=0,
        candidate_full_window_count=10,
        candidate_reliable_window_count=10,
        candidate_motion_window_count=5,
        candidate_evaluation_window_sha256="1" * 64,
        baseline_mae_bpm=1.5,
        baseline_l10=0,
        baseline_l20=0,
        baseline_right_censored_recovery_count=0,
        g1i_pass=True,
        g2_pass=True,
        g3_pass=True,
        g4_pass=True,
        g5_pass=True,
        g7_pass=True,
        qualified=True,
        g2_margin_s=10.0,
        g3_margin_s=2.0,
        g4_margin_bpm=2.25,
        g5_right_censored_count=0,
        g7_margin_s=20.0,
        failed_gates_json="[]",
        solver_elapsed_s=0.5,
        attempt_count=1,
        completed_at="2026-08-28T00:00:00+00:00",
    )


def test_ledger_is_idempotent_and_conflicts_fail(tmp_path: Path) -> None:
    identity = {"experiment_id": "experiment", "dataset_sha256": "b" * 64}
    ledger = CompactResponseLedger.create(tmp_path / "ledger.sqlite3", identity)
    try:
        ledger.record_complete(_cell())
        ledger.record_complete(_cell())
        assert ledger.complete_count("HF") == 1
        assert ledger.has_complete(
            "HF", "TS", "xiezi3_TS_0709", "physical4d:fs025:m040:mu0006:w003"
        )

        with pytest.raises(LedgerIdentityError, match="conflicting_cell"):
            ledger.record_complete(_cell(mae_bpm=9.0))
    finally:
        ledger.close()


def test_attempts_are_separate_and_export_is_canonical(tmp_path: Path) -> None:
    identity = {"experiment_id": "experiment", "dataset_sha256": "b" * 64}
    ledger = CompactResponseLedger.create(tmp_path / "ledger.sqlite3", identity)
    try:
        ledger.record_attempt(
            TechnicalAttemptEvent(
                route_id="HF",
                physical_subject_id="TS",
                record_id="xiezi3_TS_0709",
                coordinate_id="physical4d:fs025:m040:mu0006:w003",
                call_identity_sha256="f" * 64,
                attempt_number=1,
                reason_code="solver_exception",
                detail="RuntimeError:boom",
                occurred_at="2026-08-28T00:00:00+00:00",
            )
        )
        assert ledger.complete_count() == 0
        assert ledger.attempt_count() == 1

        ledger.record_complete(_cell())
        first = ledger.export_canonical_csv(tmp_path / "one.csv")
        second = ledger.export_canonical_csv(tmp_path / "two.csv")
        assert first == second
        assert (tmp_path / "one.csv").read_bytes() == (tmp_path / "two.csv").read_bytes()
        assert (tmp_path / "one.csv").read_text(encoding="utf-8").splitlines()[0] == ",".join(
            CELL_COLUMNS
        )
    finally:
        ledger.close()


def test_open_rejects_wrong_experiment_identity(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = CompactResponseLedger.create(path, {"experiment_id": "one"})
    ledger.close()

    with pytest.raises(LedgerIdentityError, match="experiment_identity mismatch"):
        CompactResponseLedger.open(path, {"experiment_id": "two"})


def test_batch_insert_and_completed_keys_are_transactional(tmp_path: Path) -> None:
    identity = {"experiment_id": "experiment"}
    ledger = CompactResponseLedger.create(tmp_path / "ledger.sqlite3", identity)
    second = CompactCellMetric(
        **{
            **_cell().__dict__,
            "physical_subject_id": "CGX",
            "record_id": "xiezi3_CGX_0710",
            "call_identity_sha256": "8" * 64,
        }
    )
    try:
        ledger.record_complete_batch((_cell(), second))
        assert ledger.complete_count("HF") == 2
        assert ledger.completed_keys("HF") == {
            ("TS", "xiezi3_TS_0709", "physical4d:fs025:m040:mu0006:w003"),
            ("CGX", "xiezi3_CGX_0710", "physical4d:fs025:m040:mu0006:w003"),
        }
        assert len(ledger.read_complete_cells("HF")) == 2
    finally:
        ledger.close()
