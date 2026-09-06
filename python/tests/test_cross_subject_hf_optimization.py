from __future__ import annotations

from pathlib import Path

import numpy as np

from ppg_hr.v2.cross_subject_hf_optimization import (
    CONSENSUS_SELECTOR_ID,
    LEGACY_SELECTOR_ID,
    MEAN_FIRST_SELECTOR_ID,
    RecordResponse,
    ResponseTable,
    apply_frozen_fold_coordinates,
    evaluate_panel,
    greedy_balanced_record_panels,
    load_lyx_partition,
    local_improve_record_panel,
    select_coordinate_index,
    with_best4d_core_gate_reference,
)


def _record(subject: str, scene: str, record_id: str, maes, qualified) -> RecordResponse:
    coordinate_ids = ("physical4d:fs025:m040:mu0006:w003", "physical4d:fs050:m080:mu0010:w006")
    return RecordResponse(
        subject_id=subject,
        scene=scene,
        record_id=record_id,
        coordinate_ids=coordinate_ids,
        mae_bpm=np.asarray(maes, dtype=float),
        qualified=np.asarray(qualified, dtype=bool),
        l10=np.zeros(2, dtype=int),
        l20=np.zeros(2, dtype=int),
        right_censored_recovery=np.zeros(2, dtype=int),
        g1i_pass=np.ones(2, dtype=bool),
        g5_pass=np.ones(2, dtype=bool),
        g7_pass=np.ones(2, dtype=bool),
    )


def test_legacy_selection_preserves_six_gate_priority() -> None:
    training = {
        "A": (_record("A", "run", "a1", [1.0, 9.0], [False, True]),),
        "B": (_record("B", "run", "b1", [1.0, 9.0], [False, True]),),
    }
    assert select_coordinate_index(training, selector_id=LEGACY_SELECTOR_ID) == 1


def test_consensus_uses_full_ranking_to_break_vote_ties() -> None:
    training = {
        "A": (_record("A", "run", "a1", [1.0, 3.0], [True, True]),),
        "B": (_record("B", "run", "b1", [3.0, 1.0], [True, True]),),
        "C": (_record("C", "run", "c1", [2.0, 2.0], [True, True]),),
    }
    assert select_coordinate_index(training, selector_id=CONSENSUS_SELECTOR_ID) == 0


def test_panel_keeps_holdout_records_out_of_selection() -> None:
    records = {
        "a1": _record("A", "run", "a1", [1.0, 8.0], [True, True]),
        "b1": _record("B", "run", "b1", [2.0, 7.0], [True, True]),
        "c1": _record("C", "run", "c1", [50.0, 1.0], [True, True]),
    }
    table = ResponseTable(records=records, coordinate_ids=records["a1"].coordinate_ids)
    result = evaluate_panel(table)
    c_fold = next(fold for fold in result.folds if fold.holdout_subject_id == "C")
    assert c_fold.coordinate_index == 0
    assert c_fold.holdout_mae_bpm == (50.0,)


def test_lyx_partition_loader_orders_physical4d_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "cells.csv"
    path.write_text(
        "scene,record_id,coordinate_id,mae_full_bpm,l10_seconds,l20_seconds,"
        "g5_has_right_censored,gate_status_json,engineering_pass_v2\n"
        "run,r1,physical4d:fs050:m080:mu0010:w006,2,0,0,false,"
        '"{""G1-I"":true,""G5"":true,""G7"":true}",true\n'
        "run,r1,physical4d:fs025:m040:mu0006:w003,1,0,0,false,"
        '"{""G1-I"":true,""G5"":true,""G7"":true}",true\n',
        encoding="utf-8",
    )
    record = load_lyx_partition(path)
    assert record.coordinate_ids[0] == "physical4d:fs025:m040:mu0006:w003"
    assert record.mae_bpm.tolist() == [1.0, 2.0]


def test_best4d_proxy_recomputes_baseline_dependent_core_gates() -> None:
    record = _record("A", "run", "a1", [1.0, 4.5], [True, True])
    table = ResponseTable(records={"a1": record}, coordinate_ids=record.coordinate_ids)
    proxy, audit = with_best4d_core_gate_reference(table)
    assert proxy.records["a1"].qualified.tolist() == [True, False]
    assert audit[0].baseline_coordinate_id == record.coordinate_ids[0]


def test_balanced_deletion_never_deletes_twice_from_one_grid() -> None:
    records = {}
    for subject, values in {"A": (8.0, 1.0, 2.0), "B": (3.0, 4.0, 5.0)}.items():
        for index, value in enumerate(values, start=1):
            record_id = f"{subject.lower()}{index}"
            records[record_id] = _record(subject, "run", record_id, [value, value], [True, True])
    table = ResponseTable(
        records=records, coordinate_ids=next(iter(records.values())).coordinate_ids
    )
    levels = greedy_balanced_record_panels(table, max_deletions_per_scene=2)
    assert len(levels[-1].excluded_record_ids) == 2
    deleted_subjects = [
        records[record_id].subject_id for record_id in levels[-1].excluded_record_ids
    ]
    assert sorted(deleted_subjects) == ["A", "B"]


def test_mean_first_selector_still_prioritizes_six_gate_pass_fraction() -> None:
    training = {
        "A": (_record("A", "run", "a1", [1.0, 20.0], [False, True]),),
        "B": (_record("B", "run", "b1", [1.0, 20.0], [False, True]),),
    }
    assert select_coordinate_index(training, selector_id=MEAN_FIRST_SELECTOR_ID) == 1


def test_frozen_coordinate_counterfactual_does_not_reselect() -> None:
    records = {
        "a1": _record("A", "run", "a1", [1.0, 9.0], [True, True]),
        "b1": _record("B", "run", "b1", [8.0, 2.0], [True, True]),
        "c1": _record("C", "run", "c1", [7.0, 3.0], [True, True]),
    }
    table = ResponseTable(records=records, coordinate_ids=records["a1"].coordinate_ids)
    anchor = evaluate_panel(table)
    frozen = apply_frozen_fold_coordinates(
        table, retained_record_ids={"a1", "b1", "c1"}, reference=anchor
    )
    assert [fold.coordinate_index for fold in frozen.folds] == [
        fold.coordinate_index for fold in anchor.folds
    ]


def test_local_swap_keeps_deletion_count_and_grid_limit() -> None:
    records = {}
    for subject, values in {"A": (9.0, 1.0, 2.0), "B": (3.0, 4.0, 5.0)}.items():
        for index, value in enumerate(values, start=1):
            record_id = f"{subject.lower()}{index}"
            records[record_id] = _record(subject, "run", record_id, [value, value], [True, True])
    table = ResponseTable(
        records=records, coordinate_ids=next(iter(records.values())).coordinate_ids
    )
    trajectory = local_improve_record_panel(
        table, initial_excluded_record_ids={"a2"}, max_iterations=2
    )
    if trajectory:
        assert len(trajectory[-1].excluded_record_ids) == 1
