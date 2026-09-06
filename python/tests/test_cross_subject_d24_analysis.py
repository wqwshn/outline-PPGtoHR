from __future__ import annotations

import numpy as np
import pytest

from ppg_hr.v2.cross_subject_d24_analysis import (
    AccRecordResponse,
    evaluate_acc_panel,
    nested_deletion_rounds,
    scene_iqr_audit,
)


def _record(subject: str, record_id: str, mae: tuple[float, float]) -> AccRecordResponse:
    return AccRecordResponse(
        subject_id=subject,
        scene="scene-a",
        record_id=record_id,
        coordinate_ids=("c0", "c1"),
        coordinate_indices=(0, 1),
        mae_bpm=np.asarray(mae, dtype=float),
        evaluation_window_sha256=(f"{record_id}-w0", f"{record_id}-w1"),
    )


def test_acc_panel_uses_training_subject_minimax_and_preserves_holdout_rows() -> None:
    records = {
        "a1": _record("A", "a1", (1.0, 2.0)),
        "b1": _record("B", "b1", (3.0, 4.0)),
        "c1": _record("C", "c1", (9.0, 1.0)),
    }

    result = evaluate_acc_panel(records, retained_record_ids=records)

    assert result.record_count == 3
    assert result.mean_mae_bpm == pytest.approx((2.0 + 4.0 + 9.0) / 3.0)
    by_holdout = {fold.holdout_subject_id: fold for fold in result.folds}
    assert by_holdout["A"].coordinate_id == "c1"
    assert by_holdout["B"].coordinate_id == "c1"
    assert by_holdout["C"].coordinate_id == "c0"
    assert by_holdout["A"].holdout_record_ids == ("a1",)


def test_scene_iqr_audit_uses_strict_classical_fences() -> None:
    rows = [
        {"scene": "s1", "record_id": f"a{i}", "mae_bpm": value}
        for i, value in enumerate((1.0, 2.0, 2.0, 3.0, 10.0))
    ] + [
        {"scene": "s2", "record_id": f"b{i}", "mae_bpm": value}
        for i, value in enumerate((0.0, 1.0, 2.0, 3.0, 6.0))
    ]

    audited, summaries = scene_iqr_audit(rows)

    by_record = {row["record_id"]: row for row in audited}
    by_scene = {row.scene: row for row in summaries}
    assert by_scene["s1"].q1 == pytest.approx(2.0)
    assert by_scene["s1"].q3 == pytest.approx(3.0)
    assert by_scene["s1"].upper_fence == pytest.approx(4.5)
    assert by_record["a4"]["is_upper_iqr_outlier"] is True
    assert by_scene["s2"].upper_fence == pytest.approx(6.0)
    assert by_record["b4"]["is_upper_iqr_outlier"] is False


def test_nested_deletion_rounds_require_eight_new_records_per_level() -> None:
    d8 = {f"r{i}" for i in range(8)}
    d16 = d8 | {f"r{i}" for i in range(8, 16)}
    d24 = d16 | {f"r{i}" for i in range(16, 24)}

    rounds = nested_deletion_rounds((d8, d16, d24), expected_increment=8)

    assert len(rounds) == 24
    assert all(rounds[f"r{i}"] == 1 for i in range(8))
    assert all(rounds[f"r{i}"] == 2 for i in range(8, 16))
    assert all(rounds[f"r{i}"] == 3 for i in range(16, 24))

    with pytest.raises(ValueError, match="deletion_levels_not_nested"):
        nested_deletion_rounds((d8, d16 - {"r0"}, d24), expected_increment=8)
