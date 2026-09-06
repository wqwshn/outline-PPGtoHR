from __future__ import annotations

from dataclasses import replace

from ppg_hr.v2.cross_subject_loso_selection import (
    SelectionCell,
    freeze_training_inputs,
    reveal_holdout_results,
    select_coordinate,
    write_training_inputs,
)
from ppg_hr.v2.cross_subject_loso_source import GroupedFold


def _cells(
    rows: dict[str, dict[str, tuple[float, tuple[bool, ...]]]],
) -> tuple[SelectionCell, ...]:
    cells: list[SelectionCell] = []
    for coordinate_index, (coordinate_id, subjects) in enumerate(rows.items()):
        for subject_id, (mae_bpm, qualifications) in subjects.items():
            for repeat_index, qualified in enumerate(qualifications, start=1):
                cells.append(
                    SelectionCell(
                        subject_id=subject_id,
                        record_id=f"{subject_id}_{repeat_index}",
                        coordinate_id=coordinate_id,
                        coordinate_index=coordinate_index,
                        candidate_mae_bpm=mae_bpm,
                        qualified=qualified,
                    )
                )
    return tuple(cells)


def test_minimum_subject_pass_fraction_has_first_priority() -> None:
    cells = _cells(
        {
            "balanced": {
                "A": (5.0, (True, True)),
                "B": (5.0, (True, False)),
            },
            "unbalanced": {
                "A": (1.0, (True, True)),
                "B": (1.0, (False, False)),
            },
        }
    )

    selected = select_coordinate(cells, ("A", "B"))

    assert selected.coordinate_id == "balanced"
    assert selected.minimum_subject_pass_fraction == "1/2"


def test_mean_subject_pass_fraction_breaks_a_minimum_tie() -> None:
    cells = _cells(
        {
            "lower_mean": {
                "A": (1.0, (True, False)),
                "B": (1.0, (True, False)),
                "C": (1.0, (True, True)),
            },
            "higher_mean": {
                "A": (9.0, (True, False)),
                "B": (9.0, (True, True)),
                "C": (9.0, (True, True)),
            },
        }
    )

    selected = select_coordinate(cells, ("A", "B", "C"))

    assert selected.coordinate_id == "higher_mean"
    assert selected.mean_subject_pass_fraction == "5/6"


def test_mae_and_coordinate_order_complete_the_lexicographic_tie_break() -> None:
    cells = _cells(
        {
            "worst": {"A": (1.0, (True,)), "B": (8.0, (True,))},
            "mean": {"A": (4.0, (True,)), "B": (4.0, (True,))},
            "index": {"A": (4.0, (True,)), "B": (4.0, (True,))},
        }
    )

    selected = select_coordinate(cells, ("A", "B"))

    assert selected.coordinate_id == "mean"
    assert selected.worst_subject_mean_mae_bpm == "4.0"
    assert selected.mean_subject_mean_mae_bpm == "4.0"
    assert selected.coordinate_index == 1
    assert selected.coordinate_order_tiebreak_applied is True
    assert selected.pre_order_tied_coordinate_ids == ("mean", "index")


def test_subjects_are_equally_weighted_when_repeat_counts_differ() -> None:
    cells = _cells(
        {
            "equal_subject_weight": {
                "QYC": (2.0, (True, False)),
                "OTHER": (4.0, (True, True, True)),
            }
        }
    )

    selected = select_coordinate(cells, ("QYC", "OTHER"))

    assert selected.minimum_subject_pass_fraction == "1/2"
    assert selected.mean_subject_pass_fraction == "3/4"
    assert selected.mean_subject_mean_mae_bpm == "3.0"
    assert selected.subject_summaries[0].record_count == 3
    assert selected.subject_summaries[1].record_count == 2


def test_holdout_rows_are_rejected_from_training_input() -> None:
    cells = _cells({"only": {"TRAIN": (1.0, (True,)), "HOLDOUT": (1.0, (True,))}})

    try:
        select_coordinate(cells, ("TRAIN",))
    except ValueError as error:
        assert str(error) == "unexpected_training_subject:HOLDOUT"
    else:
        raise AssertionError("holdout row was accepted")


def test_training_artifact_contains_no_holdout_metric(tmp_path) -> None:
    fold = GroupedFold(
        fold_id="scene__holdout_B",
        scene="scene",
        holdout_subject_id="B",
        train_subject_ids=("A",),
        holdout_record_ids=("B_1",),
        train_record_ids=("A_1",),
    )
    cells = _cells(
        {
            "c0": {"A": (1.0, (True,)), "B": (999.0, (False,))},
            "c1": {"A": (2.0, (False,)), "B": (0.0, (True,))},
        }
    )

    receipt = write_training_inputs(
        (fold,), cells, tmp_path / "training", identity={"dataset_sha256": "dataset"}
    )

    training_text = (tmp_path / "training" / "scene__holdout_B.csv").read_text(encoding="utf-8")
    assert "B_1" not in training_text
    assert "999.0" not in training_text
    assert receipt["folds"][0]["holdout_record_ids"] == ["B_1"]


def test_freeze_then_reveal_requires_all_fold_receipts(tmp_path) -> None:
    folds = (
        GroupedFold(
            fold_id="scene__holdout_A",
            scene="scene",
            holdout_subject_id="A",
            train_subject_ids=("B",),
            holdout_record_ids=("A_1",),
            train_record_ids=("B_1",),
        ),
        GroupedFold(
            fold_id="scene__holdout_B",
            scene="scene",
            holdout_subject_id="B",
            train_subject_ids=("A",),
            holdout_record_ids=("B_1",),
            train_record_ids=("A_1",),
        ),
    )
    cells = _cells(
        {
            "c0": {"A": (1.0, (True,)), "B": (2.0, (True,))},
            "c1": {"A": (3.0, (False,)), "B": (4.0, (False,))},
        }
    )
    training_root = tmp_path / "training"
    selection_root = tmp_path / "selections"
    write_training_inputs(folds, cells, training_root, identity={"dataset_sha256": "d"})
    frozen = freeze_training_inputs(training_root, selection_root, expected_fold_count=2)

    assert frozen["status"] == "pass"
    cells = tuple(replace(cell, baseline_mae_bpm=cell.candidate_mae_bpm + 1.0) for cell in cells)
    holdout_rows, fold_rows = reveal_holdout_results(
        folds,
        cells,
        selection_root,
        expected_fold_count=2,
        record_metadata={
            "A_1": {"repeat_index": 1, "source_repeat_label": "scene1"},
            "B_1": {"repeat_index": 1, "source_repeat_label": "scene1"},
        },
    )
    assert len(holdout_rows) == 2
    assert len(fold_rows) == 2
    assert {row["record_id"] for row in holdout_rows} == {"A_1", "B_1"}
    assert all(row["mae_delta_vs_baseline_bpm"] == -1.0 for row in holdout_rows)
    assert all(row["repeat_index"] == 1 for row in holdout_rows)
    assert all(len(row["compact_cell_sha256"]) == 64 for row in holdout_rows)
    assert all(row["selected_coordinate_id"] == "c0" for row in fold_rows)


def test_reveal_rejects_an_incomplete_freeze(tmp_path) -> None:
    folds = (
        GroupedFold(
            fold_id="scene__holdout_A",
            scene="scene",
            holdout_subject_id="A",
            train_subject_ids=("B",),
            holdout_record_ids=("A_1",),
            train_record_ids=("B_1",),
        ),
    )
    cells = _cells({"c0": {"A": (1.0, (True,)), "B": (2.0, (True,))}})
    (tmp_path / "selections").mkdir()

    try:
        reveal_holdout_results(folds, cells, tmp_path / "selections", expected_fold_count=1)
    except ValueError as error:
        assert str(error) == "freeze_receipt_missing"
    else:
        raise AssertionError("incomplete freeze was revealed")
