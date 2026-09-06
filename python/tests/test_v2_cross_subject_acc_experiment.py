from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.cross_subject_acc_experiment import (
    ACC_EXPERIMENT_ID,
    AccCrossSubjectRunner,
    build_acc_call_identities,
    build_acc_run_config,
)
from ppg_hr.v2.cross_subject_acc_ledger import (
    AccCompactCellMetric,
    AccCompactResponseLedger,
    AccLedgerIdentityError,
)
from ppg_hr.v2.cross_subject_acc_paired import (
    ACC_THETA_ACC,
    ACC_THETA_HF,
    HF_THETA_ACC,
    HF_THETA_HF,
    build_fold_results,
    evaluate_four_cell_common_support,
)
from ppg_hr.v2.cross_subject_acc_selection import (
    ACC_SELECTION_RULE_ID,
    AccSelectionCell,
    select_acc_coordinate,
)
from ppg_hr.v2.cross_subject_loso_metrics import FixedTimeMetrics
from ppg_hr.v2.cross_subject_loso_runner import build_hf_run_config
from ppg_hr.v2.cross_subject_loso_source import (
    BaselineBinding,
    PanelRecord,
    PhysicalCoordinate,
)
from ppg_hr.v2.solver import V2SolverResult


def _fixed_metrics() -> FixedTimeMetrics:
    return FixedTimeMetrics(
        time_bias_s=5.0,
        mae_bpm=1.0,
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
        evaluation_window_sha256="1" * 64,
    )


def _record(tmp_path: Path, subject: str = "TS") -> PanelRecord:
    data_path = tmp_path / f"xiezi1_{subject}.csv"
    ref_path = tmp_path / f"xiezi1_{subject}_HR_ref.csv"
    data_path.write_text("sensor\n", encoding="utf-8")
    ref_path.write_text("time,hr\n0,60\n9,69\n", encoding="utf-8")
    return PanelRecord(
        physical_subject_id=subject,
        scene="xiezi",
        repeat_index=1,
        source_repeat_label="xiezi1",
        record_id=f"xiezi1_{subject}_unit",
        data_path=data_path,
        ref_path=ref_path,
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        ref_sha256=hashlib.sha256(ref_path.read_bytes()).hexdigest(),
        baseline=BaselineBinding(
            batch_id="batch",
            report_path=tmp_path / "baseline.json",
            report_sha256="2" * 64,
            selection_rule="newest_lexicographic_compatible_batch",
            historical_time_bias_s=5.0,
            bo_trial_count=120,
            qc_status="good",
            qc_reason="ok",
            fixed5=_fixed_metrics(),
            fixed5_sha256="3" * 64,
        ),
    )


def _coordinate(index: int = 0) -> PhysicalCoordinate:
    return PhysicalCoordinate(
        coordinate_id=f"physical4d:test:{index}",
        coordinate_index=index,
        fs_target_hz=25,
        memory_ms=40,
        mu_base=0.006,
        exclusion_half_width_bpm=3,
    )


def _solver_result() -> V2SolverResult:
    centers = np.asarray([0.0, 1.0, 2.0])
    final = 65.0 + centers
    return V2SolverResult(
        HR=np.column_stack([centers, final, final, final, np.ones(3), final]),
        err_stats={},
        metadata={
            "reference_groups_order": ["ACC"],
            "adaptive_reference_stage_limit": None,
        },
        window_table=[
            {
                "window_idx": index,
                "center_s": float(center),
                "reliable": True,
                "adaptive_stages": [{"group": "ACC"}, {"group": "ACC"}],
            }
            for index, center in enumerate(centers)
        ],
    )


def test_acc_config_only_changes_reference_group_from_frozen_hf(tmp_path: Path) -> None:
    record = _record(tmp_path)
    coordinate = _coordinate()
    hf = build_hf_run_config(record, coordinate)
    acc = build_acc_run_config(record, coordinate)

    differences = {
        field: (getattr(hf, field), getattr(acc, field))
        for field in hf.__dataclass_fields__
        if getattr(hf, field) != getattr(acc, field)
    }

    assert differences == {"reference_groups_order": (("HF",), ("ACC",))}


def test_acc_call_identity_is_complete_and_route_specific(tmp_path: Path) -> None:
    records = (_record(tmp_path, "TS"), _record(tmp_path, "CGX"))
    coordinates = (_coordinate(0), _coordinate(1))

    calls = build_acc_call_identities(
        records=records,
        coordinates=coordinates,
        algorithm_sha256="4" * 64,
    )

    assert len(calls) == 4
    assert {call.experiment_id for call in calls} == {ACC_EXPERIMENT_ID}
    assert {call.route_id for call in calls} == {"ACC"}
    assert len({call.call_identity_sha256 for call in calls}) == 4


def test_acc_ledger_is_idempotent_and_rejects_conflicts(tmp_path: Path) -> None:
    row = AccCompactCellMetric(
        experiment_id=ACC_EXPERIMENT_ID,
        parent_experiment_id="cross_subject_multirecord_hf_loso_v1",
        algorithm_sha256="4" * 64,
        runner_sha256="6" * 64,
        dataset_sha256="5" * 64,
        input_sha256="7" * 64,
        metric_contract_sha256="8" * 64,
        call_identity_sha256="9" * 64,
        route_id="ACC",
        physical_subject_id="TS",
        scene="xiezi",
        record_id="xiezi1_TS_unit",
        coordinate_id="physical4d:test:0",
        coordinate_index=0,
        fs_target_hz=25,
        memory_ms=40,
        mu_base=0.006,
        exclusion_half_width_bpm=3,
        mae_bpm=1.5,
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
        reference_groups_order_json='["ACC"]',
        adaptive_reference_stage_limit=None,
        evaluation_window_sha256="a" * 64,
        solver_elapsed_s=0.1,
        attempt_count=1,
        completed_at="2026-08-30T00:00:00+00:00",
    )
    ledger = AccCompactResponseLedger.create(
        tmp_path / "acc.sqlite3", {"experiment_id": ACC_EXPERIMENT_ID}
    )
    try:
        ledger.record_complete(row)
        ledger.record_complete(row)
        assert ledger.complete_count("ACC") == 1
        with pytest.raises(AccLedgerIdentityError, match="conflicting_cell"):
            ledger.record_complete(
                AccCompactCellMetric(**{**row.__dict__, "mae_bpm": row.mae_bpm + 1.0})
            )
    finally:
        ledger.close()


def test_acc_runner_writes_route_neutral_metrics_without_hf_gates(tmp_path: Path) -> None:
    record = _record(tmp_path)
    call = build_acc_call_identities(
        records=(record,),
        coordinates=(_coordinate(),),
        algorithm_sha256="4" * 64,
    )[0]
    ledger = AccCompactResponseLedger.create(
        tmp_path / "runner.sqlite3", {"experiment_id": ACC_EXPERIMENT_ID}
    )
    reference = np.column_stack([np.arange(0.0, 10.0), 60.0 + np.arange(0.0, 10.0)])
    runner = AccCrossSubjectRunner(
        ledger=ledger,
        dataset_sha256="5" * 64,
        runner_sha256="6" * 64,
        solve_call=lambda _: _solver_result(),
        load_reference=lambda _: reference,
        now=lambda: "2026-08-30T00:00:00+00:00",
    )
    try:
        receipt = runner.run_calls((call,))
        cells = ledger.read_complete_cells("ACC")
        assert receipt.completed == 1
        assert len(cells) == 1
        assert cells[0].reference_groups_order_json == '["ACC"]'
        assert "g1i_pass" not in cells[0].__dict__
        assert cells[0].reliable_window_count == 3
    finally:
        ledger.close()


def test_acc_selector_balances_repeats_within_subject_before_minimax() -> None:
    cells = (
        _selection_cell("A", "a1", "c0", 0, 1.0),
        _selection_cell("A", "a2", "c0", 0, 1.0),
        _selection_cell("B", "b1", "c0", 0, 5.0),
        _selection_cell("B", "b2", "c0", 0, 5.0),
        _selection_cell("A", "a1", "c1", 1, 4.0),
        _selection_cell("A", "a2", "c1", 1, 4.0),
        _selection_cell("B", "b1", "c1", 1, 4.0),
        _selection_cell("B", "b2", "c1", 1, 4.0),
    )

    selected = select_acc_coordinate(cells, ("A", "B"))

    assert selected.selection_rule_id == ACC_SELECTION_RULE_ID
    assert selected.coordinate_id == "c1"
    assert selected.worst_subject_mean_mae_bpm == "4.0"
    assert [summary.record_count for summary in selected.subject_summaries] == [2, 2]


def test_acc_selector_uses_mean_then_coordinate_order_as_frozen_tiebreaks() -> None:
    cells = (
        _selection_cell("A", "a1", "c0", 0, 4.0),
        _selection_cell("B", "b1", "c0", 0, 2.0),
        _selection_cell("A", "a1", "c1", 1, 4.0),
        _selection_cell("B", "b1", "c1", 1, 3.0),
        _selection_cell("A", "a1", "c2", 2, 4.0),
        _selection_cell("B", "b1", "c2", 2, 2.0),
    )

    selected = select_acc_coordinate(cells, ("A", "B"))

    assert selected.coordinate_id == "c0"
    assert selected.pre_order_tied_coordinate_ids == ("c0", "c2")
    assert selected.coordinate_order_tiebreak_applied is True


def _selection_cell(
    subject_id: str,
    record_id: str,
    coordinate_id: str,
    coordinate_index: int,
    mae_bpm: float,
) -> AccSelectionCell:
    return AccSelectionCell(
        subject_id=subject_id,
        record_id=record_id,
        coordinate_id=coordinate_id,
        coordinate_index=coordinate_index,
        mae_bpm=mae_bpm,
    )


def test_four_cell_metrics_use_exact_reliable_window_intersection() -> None:
    reference = np.column_stack([np.arange(0.0, 20.0), np.arange(0.0, 20.0) + 60.0])
    results = {
        HF_THETA_HF: _paired_result((True, True, True), (66.0, 67.0, 68.0)),
        HF_THETA_ACC: _paired_result((True, False, True), (67.0, 70.0, 69.0)),
        ACC_THETA_HF: _paired_result((True, True, True), (68.0, 69.0, 70.0)),
        ACC_THETA_ACC: _paired_result((True, True, True), (69.0, 70.0, 71.0)),
    }

    paired = evaluate_four_cell_common_support(results, ref_data=reference)

    assert paired.common_window_count == 2
    assert paired.native_window_counts[HF_THETA_ACC] == 2
    assert paired.lost_window_counts[HF_THETA_HF] == 1
    assert paired.paired_mae_bpm[HF_THETA_HF] == pytest.approx(0.0)
    assert paired.paired_mae_bpm[ACC_THETA_ACC] == pytest.approx(3.0)


def test_fold_results_average_records_before_forming_frozen_contrasts() -> None:
    rows = [
        {
            "fold_id": "scene__holdout_A",
            "scene": "scene",
            "physical_subject_id": "A",
            "theta_hf_coordinate_id": "c0",
            "theta_acc_coordinate_id": "c1",
            "common_window_count": 3,
            f"{HF_THETA_HF}_mae_bpm": 1.0,
            f"{HF_THETA_ACC}_mae_bpm": 2.0,
            f"{ACC_THETA_HF}_mae_bpm": 4.0,
            f"{ACC_THETA_ACC}_mae_bpm": 3.0,
        },
        {
            "fold_id": "scene__holdout_A",
            "scene": "scene",
            "physical_subject_id": "A",
            "theta_hf_coordinate_id": "c0",
            "theta_acc_coordinate_id": "c1",
            "common_window_count": 2,
            f"{HF_THETA_HF}_mae_bpm": 3.0,
            f"{HF_THETA_ACC}_mae_bpm": 4.0,
            f"{ACC_THETA_HF}_mae_bpm": 8.0,
            f"{ACC_THETA_ACC}_mae_bpm": 7.0,
        },
    ]

    fold = build_fold_results(rows)[0]

    assert fold[f"{HF_THETA_HF}_mae_bpm"] == pytest.approx(2.0)
    assert fold[f"{ACC_THETA_ACC}_mae_bpm"] == pytest.approx(5.0)
    assert fold["delta_end_bpm"] == pytest.approx(3.0)
    assert fold["acc_selection_gain_bpm"] == pytest.approx(1.0)


def _paired_result(reliable: tuple[bool, ...], predictions: tuple[float, ...]) -> V2SolverResult:
    centers = np.arange(1.0, len(reliable) + 1.0)
    values = np.asarray(predictions, dtype=float)
    return V2SolverResult(
        HR=np.column_stack([centers, values, values, values, np.ones(len(values)), values]),
        err_stats={},
        metadata={},
        window_table=[
            {
                "window_idx": index,
                "center_s": float(center),
                "reliable": bool(is_reliable),
            }
            for index, (center, is_reliable) in enumerate(zip(centers, reliable, strict=True))
        ],
    )
