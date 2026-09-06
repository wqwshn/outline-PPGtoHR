from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.cross_subject_loso_ledger import CompactResponseLedger
from ppg_hr.v2.cross_subject_loso_metrics import FixedTimeMetrics
from ppg_hr.v2.cross_subject_loso_runner import (
    CrossSubjectRunner,
    choose_p1_sentinel_calls,
)
from ppg_hr.v2.cross_subject_loso_source import (
    BaselineBinding,
    HFCallIdentity,
    PanelRecord,
    PhysicalCoordinate,
)
from ppg_hr.v2.solver import V2SolverResult


def _metrics() -> FixedTimeMetrics:
    return FixedTimeMetrics(
        time_bias_s=5.0,
        mae_bpm=0.0,
        l10=0,
        l20=0,
        e10=0,
        e20=0,
        right_censored_recovery_count=0,
        full_window_count=3,
        reliable_window_count=3,
        motion_window_count=3,
        true_rise_applicable=False,
        true_rise_underestimate_bpm=None,
        true_rise_episode_count=0,
        spectral_gate_contract_v2=True,
        stability_pass=True,
        reference_groups_order=("HF",),
        adaptive_reference_stage_limit=None,
        evaluation_window_sha256="a" * 64,
    )


def _record(tmp_path: Path, subject: str, record_id: str) -> PanelRecord:
    data_path = tmp_path / f"{record_id}.csv"
    ref_path = tmp_path / f"{record_id}_HR_ref.csv"
    data_path.write_text("sensor\n", encoding="utf-8")
    ref_path.write_text("time,hr\n0,60\n9,69\n", encoding="utf-8")
    return PanelRecord(
        physical_subject_id=subject,
        scene="xiezi",
        repeat_index=1,
        source_repeat_label="xiezi3",
        record_id=record_id,
        data_path=data_path,
        ref_path=ref_path,
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        ref_sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
        baseline=BaselineBinding(
            batch_id="batch",
            report_path=tmp_path / f"{record_id}.json",
            report_sha256="d" * 64,
            selection_rule="newest_lexicographic_compatible_batch",
            historical_time_bias_s=4.0,
            bo_trial_count=120,
            qc_status="good",
            qc_reason="ok",
            fixed5=_metrics(),
            fixed5_sha256="e" * 64,
        ),
    )


def _call(tmp_path: Path, subject: str, record_id: str) -> HFCallIdentity:
    record = _record(tmp_path, subject, record_id)
    coordinate = PhysicalCoordinate(
        coordinate_id="physical4d:fs025:m040:mu0006:w003",
        coordinate_index=0,
        fs_target_hz=25,
        memory_ms=40,
        mu_base=0.006,
        exclusion_half_width_bpm=3,
    )
    return HFCallIdentity(
        experiment_id="experiment",
        route_id="HF",
        record=record,
        coordinate=coordinate,
        algorithm_sha256="f" * 64,
        metric_contract_sha256="1" * 64,
        call_identity_sha256=(subject.lower() + "0" * 64)[:64],
    )


def _solver_result() -> V2SolverResult:
    centers = np.asarray([0.0, 1.0, 2.0])
    final = 65.0 + centers
    return V2SolverResult(
        HR=np.column_stack([centers, final, final, final, np.ones(3), final]),
        err_stats={},
        metadata={
            "motion_segment": {"start_s": 0.0, "end_s": 2.0},
            "reference_groups_order": ["HF"],
            "adaptive_reference_stage_limit": None,
        },
        window_table=[
            {
                "window_idx": idx,
                "center_s": float(center),
                "reliable": True,
                "adaptive_stages": [{"group": "HF"}, {"group": "HF"}],
            }
            for idx, center in enumerate(centers)
        ],
    )


def test_sentinel_selection_uses_frozen_record_ids_and_coordinate_zero(
    tmp_path: Path,
) -> None:
    calls = (
        _call(tmp_path, "TS", "xiezi3_TS_0709"),
        _call(tmp_path, "CGX", "xiezi3_CGX_0710"),
        _call(tmp_path, "QYC", "bobi1_QYC_0615"),
    )

    selected = choose_p1_sentinel_calls(
        calls,
        sentinel_record_ids=("xiezi3_CGX_0710", "xiezi3_TS_0709"),
        coordinate_index=0,
    )

    assert [call.record.record_id for call in selected] == [
        "xiezi3_CGX_0710",
        "xiezi3_TS_0709",
    ]
    assert {call.coordinate.coordinate_index for call in selected} == {0}


def test_runner_writes_success_and_technical_failure_separately(tmp_path: Path) -> None:
    success = _call(tmp_path, "TS", "xiezi3_TS_0709")
    failure = _call(tmp_path, "CGX", "xiezi3_CGX_0710")
    reference = np.column_stack([np.arange(0.0, 10.0), 60.0 + np.arange(0.0, 10.0)])
    ledger = CompactResponseLedger.create(
        tmp_path / "ledger.sqlite3", {"experiment_id": "experiment"}
    )

    def solve(call: HFCallIdentity) -> V2SolverResult:
        if call.record.physical_subject_id == "CGX":
            raise RuntimeError("boom")
        return _solver_result()

    runner = CrossSubjectRunner(
        ledger=ledger,
        dataset_sha256="2" * 64,
        runner_sha256="3" * 64,
        solve_call=solve,
        load_reference=lambda _: reference,
        now=lambda: "2026-08-28T00:00:00+00:00",
    )
    try:
        receipt = runner.run_calls((success, failure))
        assert receipt.requested == 2
        assert receipt.completed == 1
        assert receipt.technical_failures == 1
        assert ledger.complete_count() == 1
        assert ledger.attempt_count() == 1
    finally:
        ledger.close()


def test_parallel_runner_resumes_and_reports_progress(tmp_path: Path) -> None:
    calls = tuple(
        _call(tmp_path, subject, f"xiezi{index}_{subject}_unit")
        for index, subject in enumerate(("TS", "CGX", "QYC", "PJY"), start=1)
    )
    reference = np.column_stack([np.arange(0.0, 10.0), 60.0 + np.arange(0.0, 10.0)])
    ledger = CompactResponseLedger.create(
        tmp_path / "parallel.sqlite3", {"experiment_id": "experiment"}
    )
    lock = threading.Lock()
    active = 0
    max_active = 0

    def solve(_: HFCallIdentity) -> V2SolverResult:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return _solver_result()

    progress = []
    runner = CrossSubjectRunner(
        ledger=ledger,
        dataset_sha256="2" * 64,
        runner_sha256="3" * 64,
        solve_call=solve,
        load_reference=lambda _: reference,
        now=lambda: "2026-08-28T00:00:00+00:00",
    )
    try:
        first = runner.run_calls_parallel(
            calls,
            workers=2,
            batch_size=2,
            on_progress=progress.append,
        )
        second = runner.run_calls_parallel(calls, workers=2, batch_size=2)
        assert first.completed == 4
        assert first.technical_failures == 0
        assert second.skipped == 4
        assert second.attempted == 0
        assert max_active >= 2
        assert progress[-1].completed == 4
        assert ledger.complete_count("HF") == 4
    finally:
        ledger.close()


def test_process_runner_rejects_unpicklable_test_callbacks(tmp_path: Path) -> None:
    call = _call(tmp_path, "TS", "xiezi3_TS_0709")
    reference = np.column_stack([np.arange(0.0, 10.0), 60.0 + np.arange(0.0, 10.0)])
    ledger = CompactResponseLedger.create(
        tmp_path / "process.sqlite3", {"experiment_id": "experiment"}
    )
    runner = CrossSubjectRunner(
        ledger=ledger,
        dataset_sha256="2" * 64,
        runner_sha256="3" * 64,
        solve_call=lambda _: _solver_result(),
        load_reference=lambda _: reference,
        now=lambda: "2026-08-28T00:00:00+00:00",
    )
    try:
        with pytest.raises(ValueError, match="process_executor_requires_default_callbacks"):
            runner.run_calls_parallel((call,), workers=1, batch_size=1, executor_kind="process")
    finally:
        ledger.close()
