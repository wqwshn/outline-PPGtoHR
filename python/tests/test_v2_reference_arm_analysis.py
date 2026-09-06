from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ppg_hr.v2.reference_arm_analysis import (
    DiagonalRouteMetrics,
    TrainingCell,
    build_cross_evaluation_matrix,
    freeze_fold_selections,
    load_fold_selections,
    paired_effects,
    select_minimax_coordinate,
    write_freeze_package,
    write_reveal_package,
)
from ppg_hr.v2.reference_arm_ledger import CompactCellMetric, CompactResponseLedger
from ppg_hr.v2.reference_arm_source import (
    EvaluationTimeline,
    FrozenHFSelection,
    RecordIdentity,
    ReferenceArmSourceSnapshot,
    physical4d_coordinates,
)


def two_train_rows(**coordinates: tuple[float, float]) -> list[TrainingCell]:
    rows = []
    for index, (coordinate_id, values) in enumerate(coordinates.items()):
        rows.extend(
            [
                TrainingCell(
                    route_id="ACC",
                    record_id="train1",
                    coordinate_id=coordinate_id,
                    coordinate_index=index,
                    mae_bpm=values[0],
                ),
                TrainingCell(
                    route_id="ACC",
                    record_id="train2",
                    coordinate_id=coordinate_id,
                    coordinate_index=index,
                    mae_bpm=values[1],
                ),
            ]
        )
    return rows


def miniature_snapshot(tmp_path: Path, *, scene_count: int = 1) -> ReferenceArmSourceSnapshot:
    coordinates = physical4d_coordinates()[:3]
    records = []
    timelines = {}
    for index in range(scene_count * 3):
        scene = f"scene{index // 3 + 1}"
        record_id = f"{scene}_r{index % 3 + 1}"
        data = tmp_path / f"{record_id}.csv"
        ref = tmp_path / f"{record_id}_HR_ref.csv"
        data.write_text("x\n", encoding="utf-8")
        ref.write_text("elapsed_seconds,hr_bpm\n5,70\n", encoding="utf-8")
        records.append(
            RecordIdentity(
                scene=scene,
                record_id=record_id,
                fold_id=f"{scene}::{record_id}",
                data_path=data,
                ref_path=ref,
                data_relative_path=data.name,
                ref_relative_path=ref.name,
                data_sha256=f"{index + 1}" * 64,
                ref_sha256=f"{index + 4}" * 64,
            )
        )
        timelines[record_id] = EvaluationTimeline(
            record_id=record_id,
            time_bias_s=5.0,
            window_keys=((0, 0.0),),
            window_sha256="a" * 64,
        )
    hf_selections = tuple(
        FrozenHFSelection(
            route_id="HF",
            scene=record.scene,
            fold_id=record.fold_id,
            train_record_ids=tuple(
                item.record_id
                for item in records
                if item.scene == record.scene and item.record_id != record.record_id
            ),
            holdout_record_id=record.record_id,
            coordinate_id=coordinates[index % 3].coordinate_id,
            coordinate_index=index % 3,
            selector_id="frozen_hf",
            holdout_mae_bpm=1.0 + index,
            prior_common_window_count=1,
            prior_common_window_mask_sha256="9" * 64,
        )
        for index, record in enumerate(records)
    )
    return ReferenceArmSourceSnapshot(
        source_commit="1" * 40,
        records=tuple(records),
        coordinates=coordinates,
        timelines=timelines,
        hf_cells=(),
        hf_selections=hf_selections,
        semantic_sha256="b" * 64,
    )


def miniature_ledger(tmp_path: Path, snapshot: ReferenceArmSourceSnapshot) -> CompactResponseLedger:
    ledger = CompactResponseLedger.create(
        tmp_path / "ledger.sqlite3",
        {"experiment_id": "miniature", "source_sha256": snapshot.semantic_sha256},
    )
    for route_index, route in enumerate(("HF", "ACC", "HF_ACC")):
        groups = ("HF",) if route == "HF" else ("ACC",) if route == "ACC" else ("HF", "ACC")
        for record_index, record in enumerate(snapshot.records):
            for coordinate in snapshot.coordinates:
                ledger.record_complete(
                    CompactCellMetric(
                        experiment_id="miniature",
                        algorithm_sha256="c" * 64,
                        source_sha256=snapshot.semantic_sha256,
                        input_sha256="d" * 64,
                        metric_contract_sha256="e" * 64,
                        code_sha256="f" * 64,
                        call_identity_sha256=(
                            f"{route_index}{record_index}{coordinate.coordinate_index}".ljust(
                                64, "0"
                            )
                        ),
                        route_id=route,
                        reference_groups_order=groups,
                        scene="scene1",
                        record_id=record.record_id,
                        coordinate_id=coordinate.coordinate_id,
                        coordinate_index=coordinate.coordinate_index,
                        fs_target_hz=coordinate.fs_target_hz,
                        memory_ms=coordinate.memory_ms,
                        mu_base=coordinate.mu_base,
                        exclusion_half_width_bpm=coordinate.exclusion_half_width_bpm,
                        evaluation_window_count=1,
                        evaluation_window_sha256="a" * 64,
                        mae_bpm=float(
                            10 * route_index + 3 * record_index + coordinate.coordinate_index
                        ),
                        solver_elapsed_s=0.0,
                        attempt_count=1,
                        completed_at="frozen",
                    )
                )
    return ledger


def test_minimax_selection_uses_worst_then_mean_then_coordinate_order() -> None:
    rows = two_train_rows(c0=(3.0, 5.0), c1=(4.0, 4.0), c2=(4.0, 4.0))
    assert select_minimax_coordinate(rows).coordinate_id == "c1"


def test_hf_selection_is_imported_not_recomputed(tmp_path: Path) -> None:
    snapshot = miniature_snapshot(tmp_path)
    ledger = miniature_ledger(tmp_path, snapshot)
    selections = freeze_fold_selections(
        snapshot, ledger, expected_coordinate_count=3, frozen_at="2026-08-28T12:00:00+08:00"
    )
    assert [selection.coordinate_id for selection in selections if selection.route_id == "HF"] == [
        selection.coordinate_id for selection in snapshot.hf_selections
    ]


def test_cross_matrix_has_nine_values_and_diagonal_is_formal_result(tmp_path: Path) -> None:
    snapshot = miniature_snapshot(tmp_path)
    ledger = miniature_ledger(tmp_path, snapshot)
    all_selections = freeze_fold_selections(
        snapshot, ledger, expected_coordinate_count=3, frozen_at="2026-08-28T12:00:00+08:00"
    )
    fold = snapshot.hf_selections[0].fold_id
    selections = tuple(selection for selection in all_selections if selection.fold_id == fold)
    matrix = build_cross_evaluation_matrix(ledger, selections)
    assert len(matrix.cells) == 9
    assert matrix.value("ACC", "ACC") == pytest.approx(10.0)
    assert matrix.value("HF_ACC", "HF_ACC") == pytest.approx(20.0)


def test_primary_contrasts_have_fixed_sign_convention() -> None:
    diagonal = DiagonalRouteMetrics(
        scene="scene1",
        fold_id="scene1::r1",
        holdout_record_id="r1",
        hf=2.0,
        acc=5.0,
        hf_acc=1.0,
    )
    effects = paired_effects(diagonal)
    assert effects["ACC_minus_HF"] == pytest.approx(3.0)
    assert effects["HF_minus_HF_ACC"] == pytest.approx(1.0)
    assert effects["ACC_minus_HF_ACC"] == pytest.approx(4.0)


def test_cross_matrix_rejects_unfrozen_or_mixed_fold_selection(tmp_path: Path) -> None:
    snapshot = miniature_snapshot(tmp_path)
    ledger = miniature_ledger(tmp_path, snapshot)
    selections = freeze_fold_selections(
        snapshot, ledger, expected_coordinate_count=3, frozen_at="2026-08-28T12:00:00+08:00"
    )
    mixed = (selections[0], selections[1], selections[5])
    with pytest.raises(ValueError, match="one fold"):
        build_cross_evaluation_matrix(ledger, mixed)


def test_freeze_and_reveal_package_has_complete_predeclared_shape(tmp_path: Path) -> None:
    snapshot = miniature_snapshot(tmp_path, scene_count=8)
    ledger = miniature_ledger(tmp_path, snapshot)
    output_root = tmp_path / "experiment"
    selections = freeze_fold_selections(
        snapshot,
        ledger,
        expected_coordinate_count=3,
        frozen_at="2026-08-28T12:00:00+08:00",
    )

    freeze = write_freeze_package(selections, output_root)
    reloaded = load_fold_selections(output_root / "analysis" / "selections.csv")
    reveal = write_reveal_package(
        snapshot,
        ledger,
        reloaded,
        output_root,
        revealed_at="2026-08-28T12:01:00+08:00",
    )

    assert freeze["selection_count"] == 72
    assert freeze["new_selection_count"] == 48
    assert freeze["holdout_values_revealed"] is False
    assert reveal["matrix_count"] == 24
    assert reveal["matrix_cell_count"] == 216
    assert reveal["diagonal_row_count"] == 72
    assert reveal["paired_effect_row_count"] == 72
    assert reveal["scene_count"] == 8
    with (output_root / "analysis" / "overall_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary = list(csv.DictReader(handle))
    contrast_rows = [row for row in summary if row["kind"] == "contrast"]
    assert len(contrast_rows) == 3
    assert {int(row["scene_denominator"]) for row in contrast_rows} == {8}
    assert (
        json.loads((output_root / "analysis" / "freeze_manifest.json").read_text(encoding="utf-8"))[
            "aggregate_freeze_sha256"
        ]
        == freeze["aggregate_freeze_sha256"]
    )
