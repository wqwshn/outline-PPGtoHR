from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ppg_hr.v2.cross_subject_matched_minimax import (
    RouteSelectionCell,
    evaluate_named_common_support,
    select_route_native_coordinate,
)


def test_matched_selector_weights_subjects_before_repeats() -> None:
    cells = (
        RouteSelectionCell("A", "a1", "c0", 0, 1.0),
        RouteSelectionCell("A", "a2", "c0", 0, 1.0),
        RouteSelectionCell("B", "b1", "c0", 0, 9.0),
        RouteSelectionCell("A", "a1", "c1", 1, 5.0),
        RouteSelectionCell("A", "a2", "c1", 1, 5.0),
        RouteSelectionCell("B", "b1", "c1", 1, 5.0),
    )
    selected = select_route_native_coordinate(cells, ("A", "B"))
    assert selected.coordinate_id == "c1"
    assert selected.worst_subject_mean_mae_bpm == "5.0"


def test_matched_selector_uses_coordinate_order_only_for_exact_metric_tie() -> None:
    cells = tuple(
        RouteSelectionCell(subject, f"{subject}1", coordinate, index, 2.0)
        for coordinate, index in (("late", 9), ("early", 2))
        for subject in ("A", "B")
    )
    selected = select_route_native_coordinate(cells, ("A", "B"))
    assert selected.coordinate_id == "early"
    assert selected.coordinate_order_tiebreak_applied
    assert selected.pre_order_tied_coordinate_ids == ("early", "late")


def test_named_common_support_uses_exact_intersection(monkeypatch) -> None:
    windows = {
        "a": {(0, 1.0): SimpleNamespace(prediction_bpm=70.0, reference_bpm=72.0)},
        "b": {
            (0, 1.0): SimpleNamespace(prediction_bpm=74.0, reference_bpm=72.0),
            (1, 2.0): SimpleNamespace(prediction_bpm=80.0, reference_bpm=80.0),
        },
        "c": {
            (0, 1.0): SimpleNamespace(prediction_bpm=73.0, reference_bpm=72.0),
            (2, 3.0): SimpleNamespace(prediction_bpm=90.0, reference_bpm=90.0),
        },
    }
    monkeypatch.setattr(
        "ppg_hr.v2.cross_subject_matched_minimax.extract_native_windows",
        lambda result, **_: windows[result],
    )
    result = evaluate_named_common_support(
        {"a": "a", "b": "b", "c": "c"},
        ref_data=np.asarray([[0.0, 70.0], [5.0, 70.0]]),
        required_labels=("a", "b", "c"),
    )
    assert result.common_window_count == 1
    assert result.lost_window_counts == {"a": 0, "b": 1, "c": 1}
    assert result.paired_mae_bpm == {"a": 2.0, "b": 2.0, "c": 1.0}
