from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.reference_arm_ledger import (
    CompactCellMetric,
    CompactResponseLedger,
    LedgerIdentityError,
    TechnicalCellError,
    evaluate_fixed5_mae,
)
from ppg_hr.v2.reference_arm_source import EvaluationTimeline, PhysicalCoordinate
from ppg_hr.v2.solver import V2SolverResult

EXPERIMENT_IDENTITY = {
    "experiment_id": "synthetic_reference_arm",
    "source_sha256": "a" * 64,
    "code_sha256": "b" * 64,
}


def solver_result(centers: list[float], final: list[float]) -> V2SolverResult:
    hr = np.asarray(
        [[center, 0.0, 0.0, prediction] for center, prediction in zip(centers, final, strict=True)],
        dtype=float,
    )
    return V2SolverResult(HR=hr, err_stats={}, metadata={}, window_table=[])


def timeline_for(keys: list[tuple[int, float]], *, time_bias_s: float = 5.0) -> EvaluationTimeline:
    return EvaluationTimeline(
        record_id="r1",
        time_bias_s=time_bias_s,
        window_keys=tuple(keys),
        window_sha256="c" * 64,
    )


def coordinate(index: int = 0) -> PhysicalCoordinate:
    return PhysicalCoordinate(
        coordinate_id=f"physical4d:c{index:03d}",
        coordinate_index=index,
        fs_target_hz=25,
        memory_ms=40,
        mu_base=0.006,
        exclusion_half_width_bpm=3,
    )


def compact_cell(*, route: str = "ACC", record: str = "r1", mae: float = 4.25) -> CompactCellMetric:
    point = coordinate()
    return CompactCellMetric(
        experiment_id="synthetic_reference_arm",
        algorithm_sha256="d" * 64,
        source_sha256="a" * 64,
        input_sha256="e" * 64,
        metric_contract_sha256="f" * 64,
        code_sha256="b" * 64,
        call_identity_sha256="1" * 64,
        route_id=route,
        reference_groups_order=(route,),
        scene="scene1",
        record_id=record,
        coordinate_id=point.coordinate_id,
        coordinate_index=point.coordinate_index,
        fs_target_hz=point.fs_target_hz,
        memory_ms=point.memory_ms,
        mu_base=point.mu_base,
        exclusion_half_width_bpm=point.exclusion_half_width_bpm,
        evaluation_window_count=2,
        evaluation_window_sha256="c" * 64,
        mae_bpm=mae,
        solver_elapsed_s=0.25,
        attempt_count=1,
        completed_at="2026-08-28T12:00:00+08:00",
    )


def broken_metric_fixture(
    mutation: str,
) -> tuple[V2SolverResult, np.ndarray, EvaluationTimeline]:
    result = solver_result([0.0, 1.0, 2.0], [70.0, 80.0, 90.0])
    ref = np.asarray([[5.0, 72.0], [7.0, 82.0]], dtype=float)
    timeline = timeline_for([(0, 0.0), (2, 2.0)])
    if mutation == "missing_window":
        timeline = timeline_for([(0, 0.0), (3, 3.0)])
    elif mutation == "center_mismatch":
        timeline = timeline_for([(0, 0.0), (2, 2.5)])
    elif mutation == "nonfinite_final":
        result.HR[2, 3] = np.nan
    else:
        raise AssertionError(mutation)
    return result, ref, timeline


def test_evaluate_fixed5_mae_uses_exact_frozen_window_keys() -> None:
    result = solver_result(centers=[0.0, 1.0, 2.0], final=[70.0, 80.0, 90.0])
    timeline = timeline_for([(0, 0.0), (2, 2.0)])
    ref = np.asarray([[5.0, 72.0], [7.0, 82.0]], dtype=float)
    metric = evaluate_fixed5_mae(result, ref, timeline)
    assert metric.mae_bpm == pytest.approx(5.0)
    assert metric.evaluation_window_count == 2
    assert metric.evaluation_window_sha256 == "c" * 64


@pytest.mark.parametrize("mutation", ["missing_window", "center_mismatch", "nonfinite_final"])
def test_evaluate_fixed5_mae_rejects_technical_incompleteness(mutation: str) -> None:
    result, ref, timeline = broken_metric_fixture(mutation)
    with pytest.raises(TechnicalCellError, match=mutation):
        evaluate_fixed5_mae(result, ref, timeline)


def test_evaluate_fixed5_mae_rejects_reference_extrapolation() -> None:
    result = solver_result([0.0, 1.0], [70.0, 80.0])
    timeline = timeline_for([(0, 0.0), (1, 1.0)], time_bias_s=10.0)
    ref = np.asarray([[5.0, 72.0], [7.0, 82.0]], dtype=float)
    with pytest.raises(TechnicalCellError, match="reference_out_of_range"):
        evaluate_fixed5_mae(result, ref, timeline)


def test_compact_ledger_is_idempotent_and_stores_no_solver_arrays(tmp_path: Path) -> None:
    ledger = CompactResponseLedger.create(tmp_path / "ledger.sqlite3", EXPERIMENT_IDENTITY)
    row = compact_cell()
    ledger.record_complete(row)
    ledger.record_complete(row)
    assert ledger.complete_count("ACC") == 1
    columns = ledger.connection.execute("PRAGMA table_info(cell_metrics)").fetchall()
    assert not {"hr", "window_table", "solver_result", "report_path"} & {
        column[1] for column in columns
    }


def test_compact_ledger_rejects_conflicting_primary_key(tmp_path: Path) -> None:
    ledger = CompactResponseLedger.create(tmp_path / "ledger.sqlite3", EXPERIMENT_IDENTITY)
    row = compact_cell()
    ledger.record_complete(row)
    with pytest.raises(LedgerIdentityError, match="conflicting_cell"):
        ledger.record_complete(replace(row, mae_bpm=row.mae_bpm + 1.0))


def test_compact_ledger_open_requires_exact_experiment_identity(tmp_path: Path) -> None:
    path = tmp_path / "ledger.sqlite3"
    CompactResponseLedger.create(path, EXPERIMENT_IDENTITY).close()
    with pytest.raises(LedgerIdentityError, match="experiment_identity"):
        CompactResponseLedger.open(path, {**EXPERIMENT_IDENTITY, "code_sha256": "0" * 64})


def test_canonical_export_is_sorted_and_hash_stable(tmp_path: Path) -> None:
    ledger = CompactResponseLedger.create(tmp_path / "ledger.sqlite3", EXPERIMENT_IDENTITY)
    ledger.record_complete(compact_cell(route="HF_ACC", record="r2", mae=2.0))
    ledger.record_complete(compact_cell(route="ACC", record="r1", mae=3.0))
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first_sha = ledger.export_canonical_csv(first)
    second_sha = ledger.export_canonical_csv(second)
    assert first.read_bytes() == second.read_bytes()
    assert first_sha == second_sha
    assert first.read_text(encoding="utf-8").splitlines()[1].split(",")[7] == "ACC"
