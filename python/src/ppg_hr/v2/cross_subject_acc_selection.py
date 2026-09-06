"""Training-only minimax selector for the independent ACC response surface."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any

from .cross_subject_acc_ledger import (
    BOOLEAN_COLUMNS,
    CELL_COLUMNS,
    AccCompactCellMetric,
)
from .cross_subject_loso_source import GroupedFold

ACC_SELECTION_RULE_ID = "subject_balanced_acc_mae_minimax_v1"


@dataclass(frozen=True)
class AccSelectionCell:
    subject_id: str
    record_id: str
    coordinate_id: str
    coordinate_index: int
    mae_bpm: float


@dataclass(frozen=True)
class AccSubjectSelectionSummary:
    subject_id: str
    record_count: int
    mean_mae_bpm: str


@dataclass(frozen=True)
class AccFoldSelection:
    selection_rule_id: str
    coordinate_id: str
    coordinate_index: int
    worst_subject_mean_mae_bpm: str
    mean_subject_mean_mae_bpm: str
    subject_summaries: tuple[AccSubjectSelectionSummary, ...]
    pre_order_tied_coordinate_ids: tuple[str, ...]
    coordinate_order_tiebreak_applied: bool


def select_acc_coordinate(
    cells: Sequence[AccSelectionCell],
    train_subject_ids: Sequence[str],
) -> AccFoldSelection:
    """Minimise worst subject mean, then mean subject mean, then coordinate order."""

    expected_subjects = tuple(train_subject_ids)
    if not cells:
        raise ValueError("empty_acc_training_cells")
    if not expected_subjects or len(set(expected_subjects)) != len(expected_subjects):
        raise ValueError("invalid_acc_training_subjects")
    expected_subject_set = set(expected_subjects)
    if any(cell.subject_id not in expected_subject_set for cell in cells):
        raise ValueError("unexpected_acc_training_subject")

    by_coordinate: dict[tuple[int, str], list[AccSelectionCell]] = defaultdict(list)
    for cell in cells:
        by_coordinate[(cell.coordinate_index, cell.coordinate_id)].append(cell)
    if len({coordinate_id for _, coordinate_id in by_coordinate}) != len(by_coordinate):
        raise ValueError("acc_coordinate_id_has_multiple_indices")

    ranked: list[tuple[tuple[Decimal, Decimal, int], AccFoldSelection]] = []
    expected_record_keys: set[tuple[str, str]] | None = None
    for (coordinate_index, coordinate_id), rows in by_coordinate.items():
        record_keys = [(row.subject_id, row.record_id) for row in rows]
        if len(record_keys) != len(set(record_keys)):
            raise ValueError(f"duplicate_acc_training_cell:{coordinate_id}")
        if expected_record_keys is None:
            expected_record_keys = set(record_keys)
        elif set(record_keys) != expected_record_keys:
            raise ValueError(f"incomplete_acc_coordinate_grid:{coordinate_id}")

        summaries: list[AccSubjectSelectionSummary] = []
        subject_means: list[Decimal] = []
        for subject_id in sorted(expected_subject_set):
            subject_rows = [row for row in rows if row.subject_id == subject_id]
            if not subject_rows:
                raise ValueError(f"missing_acc_training_subject:{coordinate_id}:{subject_id}")
            mean_mae = sum((Decimal(str(row.mae_bpm)) for row in subject_rows), Decimal()) / len(
                subject_rows
            )
            subject_means.append(mean_mae)
            summaries.append(
                AccSubjectSelectionSummary(
                    subject_id=subject_id,
                    record_count=len(subject_rows),
                    mean_mae_bpm=str(mean_mae),
                )
            )
        worst_mae = max(subject_means)
        mean_mae = sum(subject_means, Decimal()) / len(subject_means)
        selection = AccFoldSelection(
            selection_rule_id=ACC_SELECTION_RULE_ID,
            coordinate_id=coordinate_id,
            coordinate_index=coordinate_index,
            worst_subject_mean_mae_bpm=str(worst_mae),
            mean_subject_mean_mae_bpm=str(mean_mae),
            subject_summaries=tuple(summaries),
            pre_order_tied_coordinate_ids=(),
            coordinate_order_tiebreak_applied=False,
        )
        ranked.append(((worst_mae, mean_mae, coordinate_index), selection))

    best_key, selected = min(ranked, key=lambda item: item[0])
    tied = tuple(
        selection.coordinate_id
        for key, selection in sorted(ranked, key=lambda item: item[0][2])
        if key[:2] == best_key[:2]
    )
    return replace(
        selected,
        pre_order_tied_coordinate_ids=tied,
        coordinate_order_tiebreak_applied=len(tied) > 1,
    )


def load_acc_compact_cell_csv(path: Path) -> tuple[AccCompactCellMetric, ...]:
    integer_columns = {
        "coordinate_index",
        "fs_target_hz",
        "memory_ms",
        "exclusion_half_width_bpm",
        "l10",
        "l20",
        "e10",
        "e20",
        "right_censored_recovery_count",
        "full_window_count",
        "reliable_window_count",
        "motion_window_count",
        "true_rise_episode_count",
        "attempt_count",
    }
    optional_integer_columns = {"adaptive_reference_stage_limit"}
    float_columns = {"mu_base", "mae_bpm", "solver_elapsed_s"}
    optional_float_columns = {"true_rise_underestimate_bpm"}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CELL_COLUMNS:
            raise ValueError("acc_compact_cell_csv_columns_mismatch")
        rows: list[AccCompactCellMetric] = []
        for raw in reader:
            payload: dict[str, Any] = {}
            for column in CELL_COLUMNS:
                value = raw[column]
                if column in BOOLEAN_COLUMNS:
                    payload[column] = str(value).lower() in {"1", "true"}
                elif column in integer_columns:
                    payload[column] = int(value)
                elif column in optional_integer_columns:
                    payload[column] = None if value == "" else int(value)
                elif column in float_columns:
                    payload[column] = float(value)
                elif column in optional_float_columns:
                    payload[column] = None if value == "" else float(value)
                else:
                    payload[column] = str(value)
            rows.append(AccCompactCellMetric(**payload))
    return tuple(rows)


def write_acc_training_inputs(
    folds: Sequence[GroupedFold],
    cells: Sequence[AccCompactCellMetric],
    output_root: Path,
    *,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Write metric-minimal files containing only each fold's training records."""

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    by_record: dict[str, list[AccCompactCellMetric]] = defaultdict(list)
    for cell in cells:
        by_record[cell.record_id].append(cell)
    columns = (
        "physical_subject_id",
        "record_id",
        "coordinate_id",
        "coordinate_index",
        "mae_bpm",
    )
    fold_receipts: list[dict[str, Any]] = []
    for fold in folds:
        selected = [cell for record_id in fold.train_record_ids for cell in by_record[record_id]]
        selected.sort(
            key=lambda cell: (
                cell.physical_subject_id,
                cell.record_id,
                cell.coordinate_index,
                cell.coordinate_id,
            )
        )
        rows = [
            {
                "physical_subject_id": cell.physical_subject_id,
                "record_id": cell.record_id,
                "coordinate_id": cell.coordinate_id,
                "coordinate_index": cell.coordinate_index,
                "mae_bpm": cell.mae_bpm,
            }
            for cell in selected
        ]
        expected = len(fold.train_record_ids) * len(
            {(cell.coordinate_index, cell.coordinate_id) for cell in cells}
        )
        if len(rows) != expected:
            raise ValueError(f"acc_training_grid_incomplete:{fold.fold_id}:{len(rows)}")
        path = root / f"{fold.fold_id}.csv"
        csv_sha = write_dict_csv(path, rows, columns=columns)
        fold_receipts.append(
            {
                "fold_id": fold.fold_id,
                "scene": fold.scene,
                "holdout_subject_id": fold.holdout_subject_id,
                "train_subject_ids": list(fold.train_subject_ids),
                "train_record_ids": list(fold.train_record_ids),
                "holdout_record_ids": list(fold.holdout_record_ids),
                "training_row_count": len(rows),
                "training_input_file": path.name,
                "training_input_sha256": csv_sha,
            }
        )
    receipt = {
        "schema_id": "cross_subject_multirecord_acc_p3_training_inputs_v1",
        "status": "pass",
        "fold_count": len(fold_receipts),
        "identity": identity,
        "folds": fold_receipts,
    }
    write_json(root / "training_input_manifest.json", receipt)
    return receipt


def freeze_acc_training_inputs(
    training_root: Path,
    output_root: Path,
    *,
    expected_fold_count: int,
) -> dict[str, Any]:
    """Select all folds from training-only files and seal one unified receipt."""

    training_root = Path(training_root).resolve()
    output_root = Path(output_root).resolve()
    manifest_path = training_root / "training_input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = list(manifest.get("folds") or [])
    if manifest.get("status") != "pass" or len(folds) != expected_fold_count:
        raise ValueError("acc_training_input_manifest_incomplete")
    identity = dict(manifest.get("identity") or {})
    coordinate_order_sha256 = str(identity.get("coordinate_order_sha256") or "")
    if len(coordinate_order_sha256) != 64:
        raise ValueError("acc_coordinate_order_identity_missing")
    output_root.mkdir(parents=True, exist_ok=True)
    selector_sha = file_sha256(Path(__file__))
    selections = []
    for fold in folds:
        path = training_root / str(fold["training_input_file"])
        if file_sha256(path) != fold["training_input_sha256"]:
            raise ValueError(f"acc_training_input_hash_mismatch:{fold['fold_id']}")
        selection = select_acc_coordinate(
            _read_acc_training_cells(path), tuple(fold["train_subject_ids"])
        )
        receipt = {
            "schema_id": "cross_subject_multirecord_acc_fold_selection_v1",
            "fold_id": fold["fold_id"],
            "scene": fold["scene"],
            "holdout_subject_id": fold["holdout_subject_id"],
            "train_subject_ids": fold["train_subject_ids"],
            "train_record_ids": fold["train_record_ids"],
            "holdout_record_ids": fold["holdout_record_ids"],
            "training_record_set_sha256": _semantic_sha256(
                sorted(str(value) for value in fold["train_record_ids"])
            ),
            "training_input_sha256": fold["training_input_sha256"],
            "selector_source_sha256": selector_sha,
            "coordinate_order_sha256": coordinate_order_sha256,
            "sorting_key": [
                selection.worst_subject_mean_mae_bpm,
                selection.mean_subject_mean_mae_bpm,
                selection.coordinate_index,
            ],
            "selection": asdict(selection),
        }
        selection_path = output_root / f"{fold['fold_id']}.json"
        selection_sha = write_json(selection_path, receipt)
        selections.append(
            {
                "fold_id": fold["fold_id"],
                "selection_file": selection_path.name,
                "selection_sha256": selection_sha,
                "selected_coordinate_id": selection.coordinate_id,
                "selected_coordinate_index": selection.coordinate_index,
            }
        )
    freeze = {
        "schema_id": "cross_subject_multirecord_acc_p3_freeze_receipt_v1",
        "status": "pass",
        "fold_count": len(selections),
        "selection_rule_id": ACC_SELECTION_RULE_ID,
        "coordinate_order_sha256": coordinate_order_sha256,
        "training_input_manifest_sha256": file_sha256(manifest_path),
        "selector_source_sha256": selector_sha,
        "selections": selections,
    }
    freeze_sha = write_json(output_root / "p3_freeze_receipt.json", freeze)
    return {**freeze, "receipt_sha256": freeze_sha}


def reveal_acc_holdouts(
    folds: Sequence[GroupedFold],
    cells: Sequence[AccCompactCellMetric],
    selection_root: Path,
    *,
    expected_fold_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reveal route-native ACC holdouts after all fold selections are frozen."""

    selection_root = Path(selection_root).resolve()
    freeze = json.loads((selection_root / "p3_freeze_receipt.json").read_text(encoding="utf-8"))
    rows = list(freeze.get("selections") or [])
    if freeze.get("status") != "pass" or len(rows) != expected_fold_count:
        raise ValueError("acc_freeze_receipt_incomplete")
    expected_folds = {fold.fold_id for fold in folds}
    if {str(row["fold_id"]) for row in rows} != expected_folds:
        raise ValueError("acc_freeze_fold_set_mismatch")
    frozen: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = selection_root / str(row["selection_file"])
        if file_sha256(path) != row["selection_sha256"]:
            raise ValueError(f"acc_selection_hash_mismatch:{row['fold_id']}")
        frozen[str(row["fold_id"])] = json.loads(path.read_text(encoding="utf-8"))

    by_key = {(cell.record_id, cell.coordinate_id): cell for cell in cells}
    record_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        selection_receipt = frozen[fold.fold_id]
        selection = selection_receipt["selection"]
        coordinate_id = str(selection["coordinate_id"])
        selected_cells = []
        selection_sha = file_sha256(selection_root / f"{fold.fold_id}.json")
        for record_id in fold.holdout_record_ids:
            cell = by_key.get((record_id, coordinate_id))
            if cell is None:
                raise ValueError(f"acc_selected_holdout_missing:{fold.fold_id}:{record_id}")
            selected_cells.append(cell)
            record_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "holdout_subject_id": fold.holdout_subject_id,
                    "scene": fold.scene,
                    "record_id": record_id,
                    "selected_coordinate_id": coordinate_id,
                    "selected_coordinate_index": int(selection["coordinate_index"]),
                    "selection_sha256": selection_sha,
                    "call_identity_sha256": cell.call_identity_sha256,
                    "native_mae_bpm": cell.mae_bpm,
                    "native_reliable_window_count": cell.reliable_window_count,
                    "native_evaluation_window_sha256": cell.evaluation_window_sha256,
                }
            )
        maes = [float(cell.mae_bpm) for cell in selected_cells]
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "scene": fold.scene,
                "holdout_subject_id": fold.holdout_subject_id,
                "selected_coordinate_id": coordinate_id,
                "selected_coordinate_index": int(selection["coordinate_index"]),
                "worst_training_subject_mean_mae_bpm": selection["worst_subject_mean_mae_bpm"],
                "mean_training_subject_mean_mae_bpm": selection["mean_subject_mean_mae_bpm"],
                "coordinate_order_tiebreak_applied": selection["coordinate_order_tiebreak_applied"],
                "pre_order_tie_count": len(selection["pre_order_tied_coordinate_ids"]),
                "native_mean_mae_bpm": sum(maes) / len(maes),
                "native_median_mae_bpm": median(maes),
                "native_max_mae_bpm": max(maes),
                "heldout_records": len(maes),
            }
        )
    return record_rows, fold_rows


def write_dict_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    columns: Sequence[str] | None = None,
) -> str:
    if not rows:
        raise ValueError("empty_acc_result_rows")
    fieldnames = tuple(columns or rows[0].keys())
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    return file_sha256(target)


def write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_acc_training_cells(path: Path) -> tuple[AccSelectionCell, ...]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return tuple(
        AccSelectionCell(
            subject_id=str(row["physical_subject_id"]),
            record_id=str(row["record_id"]),
            coordinate_id=str(row["coordinate_id"]),
            coordinate_index=int(row["coordinate_index"]),
            mae_bpm=float(row["mae_bpm"]),
        )
        for row in rows
    )
