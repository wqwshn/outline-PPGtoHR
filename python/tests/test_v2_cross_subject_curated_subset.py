from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from ppg_hr.v2.cross_subject_curated_reporting import build_panel_report_row
from ppg_hr.v2.cross_subject_curated_subset import (
    CuratedPanelRecord,
    RecordExclusionCandidate,
    RecordPanelLevel,
    ResponseDifficultyCell,
    SubjectExclusionCandidate,
    bind_parent_sources,
    build_curated_folds,
    build_curated_panel_manifest,
    build_hf_acc_common_support,
    freeze_curated_panel_training,
    partition_hf_acc_common_support,
    plan_nested_record_exclusions,
    plan_subject_exclusions,
    prepare_curated_p0,
    summarize_record_difficulties,
    summarize_route_rows,
)
from ppg_hr.v2.cross_subject_curated_verifier import verify_panel_partition


def test_record_difficulty_summarizes_frozen_response_surface() -> None:
    cells = (
        ResponseDifficultyCell("S1", "scene", "r1", 9.0, False),
        ResponseDifficultyCell("S1", "scene", "r1", 1.0, True),
        ResponseDifficultyCell("S1", "scene", "r1", 5.0, True),
    )

    summaries = summarize_record_difficulties(cells, expected_coordinate_count=3)

    assert summaries == (
        {
            "physical_subject_id": "S1",
            "scene": "scene",
            "record_id": "r1",
            "qualified_coordinate_count": 2,
            "median_mae_bpm": 5.0,
            "minimum_mae_bpm": 1.0,
        },
    )


def test_record_difficulty_rejects_non_finite_mae() -> None:
    cells = (ResponseDifficultyCell("S1", "scene", "r1", float("nan"), False),)

    try:
        summarize_record_difficulties(cells, expected_coordinate_count=1)
    except ValueError as error:
        assert str(error) == "non_finite_mae:r1"
    else:
        raise AssertionError("non-finite response cell was accepted")


def test_nested_record_panels_use_the_hardest_globally_feasible_exclusions() -> None:
    candidates = (
        RecordExclusionCandidate("A", "s1", "a1", (10.0,)),
        RecordExclusionCandidate("B", "s1", "b1", (1.0,)),
        RecordExclusionCandidate("A", "s2", "a2", (2.0,)),
        RecordExclusionCandidate("B", "s2", "b2", (9.0,)),
    )
    levels = (
        RecordPanelLevel("r2", 1, {"A": 1, "B": 1}),
        RecordPanelLevel("r0", 2, {"A": 2, "B": 2}),
    )

    panels = plan_nested_record_exclusions(candidates, levels)

    assert panels == {
        "r2": ("a1", "b2"),
        "r0": ("a1", "a2", "b1", "b2"),
    }


def test_subject_panel_uses_the_hardest_feasible_scene_assignment() -> None:
    candidates = (
        SubjectExclusionCandidate("A", "s1", ("a1",), (10.0,)),
        SubjectExclusionCandidate("B", "s1", ("b1",), (1.0,)),
        SubjectExclusionCandidate("C", "s1", ("c1",), (0.0,)),
        SubjectExclusionCandidate("A", "s2", ("a2",), (2.0,)),
        SubjectExclusionCandidate("B", "s2", ("b2",), (9.0,)),
        SubjectExclusionCandidate("C", "s2", ("c2",), (0.0,)),
        SubjectExclusionCandidate("A", "s3", ("a3",), (8.0,)),
        SubjectExclusionCandidate("B", "s3", ("b3",), (3.0,)),
        SubjectExclusionCandidate("C", "s3", ("c3",), (7.0,)),
    )

    selected = plan_subject_exclusions(
        candidates,
        subject_minimums={"A": 1, "B": 1, "C": 0},
        subject_maximums={"A": 2, "B": 1, "C": 1},
    )

    assert tuple((row.scene, row.physical_subject_id) for row in selected) == (
        ("s1", "A"),
        ("s2", "B"),
        ("s3", "A"),
    )


def test_five_subject_panel_builds_four_train_one_holdout_folds() -> None:
    records = tuple(
        CuratedPanelRecord(subject, "scene", f"{subject}_{repeat}")
        for subject in ("A", "B", "C", "D", "E")
        for repeat in (1, 2)
    )

    folds = build_curated_folds(records, expected_subjects_per_scene=5)

    assert len(folds) == 5
    fold_a = next(row for row in folds if row.holdout_subject_id == "A")
    assert fold_a.holdout_record_ids == ("A_1", "A_2")
    assert fold_a.train_subject_ids == ("B", "C", "D", "E")
    assert len(fold_a.train_record_ids) == 8


def test_panel_manifest_freezes_retained_records_and_folds() -> None:
    records = tuple(
        CuratedPanelRecord(subject, scene, f"{scene}_{subject}")
        for scene in ("s1", "s2")
        for subject in ("A", "B", "C")
    )

    manifest = build_curated_panel_manifest(
        records,
        panel_id="subject_s2",
        excluded_record_ids=("s1_C", "s2_B"),
        expected_subjects_per_scene=2,
    )

    assert manifest["retained_record_count"] == 4
    assert manifest["excluded_record_ids"] == ["s1_C", "s2_B"]
    assert manifest["fold_count"] == 4
    assert {row["fold_id"] for row in manifest["folds"]} == {
        "s1__holdout_A",
        "s1__holdout_B",
        "s2__holdout_A",
        "s2__holdout_C",
    }


def test_parent_source_binding_verifies_hf_and_acc_receipts(tmp_path: Path) -> None:
    hf_root = tmp_path / "hf"
    acc_root = tmp_path / "acc"
    for root, cell_name in ((hf_root, "hf_cell_metrics.csv"), (acc_root, "acc_cell_metrics.csv")):
        (root / "p0").mkdir(parents=True)
        (root / "p2").mkdir()
        (root / "p3").mkdir()
        (root / "p2" / cell_name).write_text("cell\n", encoding="utf-8")
        (root / "p3" / "holdout_record_results.csv").write_text("record\n", encoding="utf-8")
        (root / "p3" / "fold_results.csv").write_text("fold\n", encoding="utf-8")

    common_p0 = {
        "status": "pass",
        "record_count": 1,
        "fold_count": 1,
        "coordinate_count": 1,
        "dataset_sha256": "dataset",
        "coordinate_order_sha256": "coordinates",
        "fold_manifest_sha256": "folds",
    }
    _write_json(hf_root / "p0" / "p0_receipt.json", {**common_p0, "experiment_id": "hf"})
    _write_json(acc_root / "p0" / "p0_receipt.json", {**common_p0, "experiment_id": "acc"})
    for root, cell_name in ((hf_root, "hf_cell_metrics.csv"), (acc_root, "acc_cell_metrics.csv")):
        _write_json(
            root / "p2" / "p2_receipt.json",
            {
                "status": "pass",
                "complete_cell_count": 1,
                "dataset_sha256": "dataset",
                "coordinate_order_sha256": "coordinates",
                "canonical_csv_sha256": _sha(root / "p2" / cell_name),
            },
        )
        _write_json(
            root / "p3" / "p3_reveal_receipt.json",
            {
                "status": "pass",
                "record_result_count": 1,
                "fold_result_count": 1,
                "holdout_record_results_sha256": _sha(root / "p3" / "holdout_record_results.csv"),
                "fold_results_sha256": _sha(root / "p3" / "fold_results.csv"),
            },
        )

    binding = bind_parent_sources(
        hf_root,
        acc_root,
        expected_record_count=1,
        expected_fold_count=1,
        expected_coordinate_count=1,
        expected_cell_count=1,
    )

    assert binding["dataset_sha256"] == "dataset"
    assert binding["hf_cell_metrics_sha256"] == _sha(hf_root / "p2" / "hf_cell_metrics.csv")
    assert binding["acc_cell_metrics_sha256"] == _sha(acc_root / "p2" / "acc_cell_metrics.csv")


def test_parent_source_binding_rejects_wrong_parent_experiment_id(tmp_path: Path) -> None:
    hf_root, acc_root = _write_parent_fixture(tmp_path)

    try:
        bind_parent_sources(
            hf_root,
            acc_root,
            expected_record_count=1,
            expected_fold_count=1,
            expected_coordinate_count=1,
            expected_cell_count=1,
            expected_hf_experiment_id="expected-hf",
            expected_acc_experiment_id="acc",
        )
    except ValueError as error:
        assert str(error) == "parent_source_mismatch:p0.experiment_id"
    else:
        raise AssertionError("wrong parent experiment id was accepted")


def test_parent_source_binding_rejects_dataset_manifest_tampering(tmp_path: Path) -> None:
    hf_root, acc_root = _write_parent_fixture(tmp_path)
    (hf_root / "p0" / "dataset_manifest.json").write_text("tampered\n", encoding="utf-8")

    try:
        bind_parent_sources(
            hf_root,
            acc_root,
            expected_record_count=1,
            expected_fold_count=1,
            expected_coordinate_count=1,
            expected_cell_count=1,
        )
    except ValueError as error:
        assert str(error) == "parent_source_hash_mismatch:p0.dataset_manifest"
    else:
        raise AssertionError("tampered dataset manifest was accepted")


def test_prepare_p0_seals_contract_and_parent_binding(tmp_path: Path) -> None:
    hf_root, acc_root = _write_parent_fixture(tmp_path)
    contract_path = tmp_path / "contract.json"
    output_root = tmp_path / "experiment" / "p0"
    contract = {
        "schema_id": "curated_contract_test",
        "experiment_id": "curated",
        "hf_parent_experiment_id": "hf",
        "acc_parent_experiment_id": "acc",
        "expected_record_count": 1,
        "expected_fold_count": 1,
        "expected_coordinate_count": 1,
        "expected_parent_cell_count": 1,
        "hf_selection_rule_id": "subject_balanced_lexicographic_physical4d_v1",
        "acc_selection_rule_id": "subject_balanced_acc_mae_minimax_v1",
    }
    _write_json(contract_path, contract)

    receipt = prepare_curated_p0(
        hf_root=hf_root,
        acc_root=acc_root,
        contract_path=contract_path,
        output_root=output_root,
    )

    assert receipt["status"] == "pass"
    assert receipt["experiment_id"] == "curated"
    assert receipt["source_binding_sha256"] == _sha(output_root / "source_binding.json")
    assert json.loads((output_root / "p0_receipt.json").read_text(encoding="utf-8")) == receipt


def test_panel_training_freezes_hf_and_acc_without_holdout_rows(tmp_path: Path) -> None:
    records = (
        CuratedPanelRecord("A", "scene", "a1"),
        CuratedPanelRecord("B", "scene", "b1"),
    )
    folds = build_curated_folds(records, expected_subjects_per_scene=2)
    hf_cells = tuple(
        SimpleNamespace(
            physical_subject_id=subject,
            record_id=record_id,
            coordinate_id=coordinate_id,
            coordinate_index=coordinate_index,
            candidate_mae_bpm=mae,
            qualified=qualified,
        )
        for subject, record_id in (("A", "a1"), ("B", "b1"))
        for coordinate_id, coordinate_index, mae, qualified in (
            ("c0", 0, 1.0, True),
            ("c1", 1, 2.0, False),
        )
    )
    acc_cells = tuple(
        SimpleNamespace(
            physical_subject_id=subject,
            record_id=record_id,
            coordinate_id=coordinate_id,
            coordinate_index=coordinate_index,
            mae_bpm=mae,
        )
        for subject, record_id in (("A", "a1"), ("B", "b1"))
        for coordinate_id, coordinate_index, mae in (
            ("c0", 0, 1.0),
            ("c1", 1, 2.0),
        )
    )

    receipt = freeze_curated_panel_training(
        panel_id="tiny",
        folds=folds,
        hf_cells=hf_cells,
        acc_cells=acc_cells,
        output_root=tmp_path / "p2" / "tiny",
        identity={"coordinate_order_sha256": "a" * 64},
    )

    assert receipt["status"] == "pass"
    assert receipt["fold_count"] == 2
    assert receipt["hf_selection_rule_id"] == "subject_balanced_lexicographic_physical4d_v1"
    assert receipt["acc_selection_rule_id"] == "subject_balanced_acc_mae_minimax_v1"
    hf_manifest = json.loads(
        (tmp_path / "p2" / "tiny" / "hf" / "training" / "training_input_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for fold in hf_manifest["folds"]:
        rows = (
            tmp_path / "p2" / "tiny" / "hf" / "training" / fold["training_input_file"]
        ).read_text(encoding="utf-8")
        assert fold["holdout_record_ids"][0] not in rows


def test_hf_acc_common_support_requires_identical_evaluation_windows() -> None:
    hf_rows = [
        {
            "fold_id": "scene__holdout_A",
            "holdout_subject_id": "A",
            "scene": "scene",
            "record_id": "a1",
            "selected_coordinate_id": "hf-coordinate",
            "candidate_mae_bpm": 2.0,
            "candidate_reliable_window_count": 12,
            "candidate_evaluation_window_sha256": "window",
        }
    ]
    acc_rows = [
        {
            "fold_id": "scene__holdout_A",
            "holdout_subject_id": "A",
            "scene": "scene",
            "record_id": "a1",
            "selected_coordinate_id": "acc-coordinate",
            "native_mae_bpm": 5.0,
            "native_reliable_window_count": 12,
            "native_evaluation_window_sha256": "window",
        }
    ]

    rows = build_hf_acc_common_support("panel", hf_rows, acc_rows)

    assert rows == [
        {
            "panel_id": "panel",
            "fold_id": "scene__holdout_A",
            "holdout_subject_id": "A",
            "scene": "scene",
            "record_id": "a1",
            "hf_selected_coordinate_id": "hf-coordinate",
            "acc_selected_coordinate_id": "acc-coordinate",
            "hf_mae_bpm": 2.0,
            "acc_mae_bpm": 5.0,
            "acc_minus_hf_mae_bpm": 3.0,
            "common_reliable_window_count": 12,
            "common_evaluation_window_sha256": "window",
        }
    ]

    acc_rows[0]["native_evaluation_window_sha256"] = "different"
    try:
        build_hf_acc_common_support("panel", hf_rows, acc_rows)
    except ValueError as error:
        assert str(error) == "curated_common_support_mismatch:a1"
    else:
        raise AssertionError("different HF/ACC evaluation windows were compared")

    matched, audit = partition_hf_acc_common_support("panel", hf_rows, acc_rows)
    assert matched == []
    assert audit == [
        {
            "panel_id": "panel",
            "fold_id": "scene__holdout_A",
            "holdout_subject_id": "A",
            "scene": "scene",
            "record_id": "a1",
            "hf_reliable_window_count": 12,
            "acc_reliable_window_count": 12,
            "hf_evaluation_window_sha256": "window",
            "acc_evaluation_window_sha256": "different",
            "common_support": False,
        }
    ]


def test_hf_summary_parses_csv_boolean_text() -> None:
    summary = summarize_route_rows(
        [
            {"candidate_mae_bpm": "2.0", "qualified": "False"},
            {"candidate_mae_bpm": "4.0", "qualified": "true"},
        ],
        route="hf",
    )

    assert summary["qualified_record_count"] == 1
    assert summary["qualified_record_fraction"] == 0.5


def test_independent_verifier_checks_panel_partition() -> None:
    source_records = (
        {"physical_subject_id": "A", "scene": "s", "record_id": "a"},
        {"physical_subject_id": "B", "scene": "s", "record_id": "b"},
        {"physical_subject_id": "C", "scene": "s", "record_id": "c"},
    )
    panel = {
        "panel_id": "tiny",
        "retained_record_count": 2,
        "excluded_record_count": 1,
        "retained_records": source_records[:2],
        "excluded_record_ids": ["c"],
    }

    facts = verify_panel_partition(source_records, panel)

    assert facts == {
        "source_record_count": 3,
        "retained_record_count": 2,
        "excluded_record_count": 1,
    }

    panel["excluded_record_ids"] = ["unknown"]
    try:
        verify_panel_partition(source_records, panel)
    except ValueError as error:
        assert str(error) == "verifier_panel_partition:tiny"
    else:
        raise AssertionError("invalid panel partition was accepted")


def test_report_row_decomposes_composition_and_reselection_effects() -> None:
    anchor = {"hf": {"mean_mae_bpm": 10.0}, "acc": {"mean_mae_bpm": 8.0}}
    panel = {
        "panel_id": "panel",
        "retained_record_count": 5,
        "fold_count": 2,
        "hf": {
            "mean_mae_bpm": 7.0,
            "median_mae_bpm": 6.0,
            "max_mae_bpm": 11.0,
            "qualified_record_count": 3,
            "qualified_record_fraction": 0.6,
        },
        "acc": {"mean_mae_bpm": 6.0, "median_mae_bpm": 5.0, "max_mae_bpm": 10.0},
        "hf_acc_common_support": {"record_count": 5, "mean_difference_bpm": -1.0},
        "hf_acc_support_mismatch_record_count": 0,
        "retained_parent_vs_curated": {
            "HF": {
                "parent_mean_mae_bpm": 8.0,
                "curated_minus_parent_mean_mae_bpm": -1.0,
            },
            "ACC": {
                "parent_mean_mae_bpm": 7.5,
                "curated_minus_parent_mean_mae_bpm": -1.5,
            },
        },
    }

    row = build_panel_report_row(anchor, panel)

    assert row["hf_composition_effect_bpm"] == -2.0
    assert row["hf_reselection_effect_bpm"] == -1.0
    assert row["hf_total_effect_bpm"] == -3.0
    assert row["acc_composition_effect_bpm"] == -0.5
    assert row["acc_reselection_effect_bpm"] == -1.5
    assert row["acc_total_effect_bpm"] == -2.0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_parent_fixture(tmp_path: Path) -> tuple[Path, Path]:
    hf_root = tmp_path / "hf"
    acc_root = tmp_path / "acc"
    for root, cell_name in ((hf_root, "hf_cell_metrics.csv"), (acc_root, "acc_cell_metrics.csv")):
        (root / "p0").mkdir(parents=True)
        (root / "p2").mkdir()
        (root / "p3").mkdir()
        (root / "p2" / cell_name).write_text("cell\n", encoding="utf-8")
        (root / "p3" / "holdout_record_results.csv").write_text("record\n", encoding="utf-8")
        (root / "p3" / "fold_results.csv").write_text("fold\n", encoding="utf-8")
        if root == hf_root:
            _write_json(root / "p0" / "dataset_manifest.json", {"records": []})
        _write_json(
            root / "p0" / "p0_receipt.json",
            {
                "status": "pass",
                "experiment_id": "hf" if root == hf_root else "acc",
                "record_count": 1,
                "fold_count": 1,
                "coordinate_count": 1,
                "dataset_sha256": "dataset",
                "coordinate_order_sha256": "coordinates",
                "fold_manifest_sha256": "folds",
                **(
                    {
                        "artifact_sha256": {
                            "dataset_manifest.json": _sha(root / "p0" / "dataset_manifest.json")
                        }
                    }
                    if root == hf_root
                    else {}
                ),
            },
        )
        _write_json(
            root / "p2" / "p2_receipt.json",
            {
                "status": "pass",
                "complete_cell_count": 1,
                "dataset_sha256": "dataset",
                "coordinate_order_sha256": "coordinates",
                "canonical_csv_sha256": _sha(root / "p2" / cell_name),
            },
        )
        _write_json(
            root / "p3" / "p3_reveal_receipt.json",
            {
                "status": "pass",
                "record_result_count": 1,
                "fold_result_count": 1,
                "holdout_record_results_sha256": _sha(root / "p3" / "holdout_record_results.csv"),
                "fold_results_sha256": _sha(root / "p3" / "fold_results.csv"),
            },
        )
    return hf_root, acc_root
