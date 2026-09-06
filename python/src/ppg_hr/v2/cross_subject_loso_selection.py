"""Training-only selector for grouped cross-subject LOSO folds."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any

from .cross_subject_loso_ledger import CELL_COLUMNS, CompactCellMetric
from .cross_subject_loso_source import GroupedFold

SELECTION_RULE_ID = "subject_balanced_lexicographic_physical4d_v1"


@dataclass(frozen=True)
class SelectionCell:
    subject_id: str
    record_id: str
    coordinate_id: str
    coordinate_index: int
    candidate_mae_bpm: float
    qualified: bool
    baseline_mae_bpm: float | None = None


@dataclass(frozen=True)
class SubjectSelectionSummary:
    subject_id: str
    record_count: int
    passed_record_count: int
    pass_fraction: str
    mean_mae_bpm: str


@dataclass(frozen=True)
class FoldSelection:
    selection_rule_id: str
    coordinate_id: str
    coordinate_index: int
    minimum_subject_pass_fraction: str
    mean_subject_pass_fraction: str
    worst_subject_mean_mae_bpm: str
    mean_subject_mean_mae_bpm: str
    subject_summaries: tuple[SubjectSelectionSummary, ...]
    pre_order_tied_coordinate_ids: tuple[str, ...]
    coordinate_order_tiebreak_applied: bool


def select_coordinate(
    cells: Sequence[SelectionCell],
    train_subject_ids: Sequence[str],
) -> FoldSelection:
    """Choose one coordinate using the frozen subject-balanced lexicographic rule."""

    if not cells:
        raise ValueError("empty_training_cells")
    expected_subjects = set(train_subject_ids)
    if not expected_subjects or len(expected_subjects) != len(train_subject_ids):
        raise ValueError("invalid_training_subjects")
    for cell in cells:
        if cell.subject_id not in expected_subjects:
            raise ValueError(f"unexpected_training_subject:{cell.subject_id}")

    by_coordinate: dict[tuple[int, str], list[SelectionCell]] = defaultdict(list)
    for cell in cells:
        by_coordinate[(cell.coordinate_index, cell.coordinate_id)].append(cell)
    coordinate_ids = [key[1] for key in by_coordinate]
    if len(coordinate_ids) != len(set(coordinate_ids)):
        raise ValueError("coordinate_id_has_multiple_indices")

    ranked: list[
        tuple[
            tuple[Fraction, Fraction, Decimal, Decimal, int],
            FoldSelection,
        ]
    ] = []
    expected_record_keys: set[tuple[str, str]] | None = None
    for (coordinate_index, coordinate_id), coordinate_cells in by_coordinate.items():
        record_keys = [(cell.subject_id, cell.record_id) for cell in coordinate_cells]
        if len(record_keys) != len(set(record_keys)):
            raise ValueError(f"duplicate_training_cell:{coordinate_id}")
        if expected_record_keys is None:
            expected_record_keys = set(record_keys)
        elif set(record_keys) != expected_record_keys:
            raise ValueError(f"incomplete_coordinate_grid:{coordinate_id}")

        subject_rows: list[SubjectSelectionSummary] = []
        pass_fractions: list[Fraction] = []
        subject_maes: list[Decimal] = []
        for subject_id in sorted(expected_subjects):
            rows = [cell for cell in coordinate_cells if cell.subject_id == subject_id]
            if not rows:
                raise ValueError(f"missing_training_subject:{coordinate_id}:{subject_id}")
            passed = sum(cell.qualified for cell in rows)
            pass_fraction = Fraction(passed, len(rows))
            mean_mae = sum(Decimal(str(cell.candidate_mae_bpm)) for cell in rows) / len(rows)
            pass_fractions.append(pass_fraction)
            subject_maes.append(mean_mae)
            subject_rows.append(
                SubjectSelectionSummary(
                    subject_id=subject_id,
                    record_count=len(rows),
                    passed_record_count=passed,
                    pass_fraction=_fraction_text(pass_fraction),
                    mean_mae_bpm=str(mean_mae),
                )
            )

        minimum_pass = min(pass_fractions)
        mean_pass = sum(pass_fractions, Fraction()) / len(pass_fractions)
        worst_mae = max(subject_maes)
        mean_mae = sum(subject_maes, Decimal()) / len(subject_maes)
        selection = FoldSelection(
            selection_rule_id=SELECTION_RULE_ID,
            coordinate_id=coordinate_id,
            coordinate_index=coordinate_index,
            minimum_subject_pass_fraction=_fraction_text(minimum_pass),
            mean_subject_pass_fraction=_fraction_text(mean_pass),
            worst_subject_mean_mae_bpm=str(worst_mae),
            mean_subject_mean_mae_bpm=str(mean_mae),
            subject_summaries=tuple(subject_rows),
            pre_order_tied_coordinate_ids=(),
            coordinate_order_tiebreak_applied=False,
        )
        key = (-minimum_pass, -mean_pass, worst_mae, mean_mae, coordinate_index)
        ranked.append((key, selection))

    best_key, selected = min(ranked, key=lambda item: item[0])
    tied = tuple(
        selection.coordinate_id
        for key, selection in sorted(ranked, key=lambda item: item[0][4])
        if key[:4] == best_key[:4]
    )
    return replace(
        selected,
        pre_order_tied_coordinate_ids=tied,
        coordinate_order_tiebreak_applied=len(tied) > 1,
    )


def write_training_inputs(
    folds: Sequence[GroupedFold],
    cells: Sequence[SelectionCell],
    output_root: Path,
    *,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Materialise one metric-minimal, holdout-free CSV for each fold."""

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cell_by_record: dict[str, list[SelectionCell]] = defaultdict(list)
    for cell in cells:
        cell_by_record[cell.record_id].append(cell)
    fold_receipts: list[dict[str, Any]] = []
    columns = (
        "physical_subject_id",
        "record_id",
        "coordinate_id",
        "coordinate_index",
        "candidate_mae_bpm",
        "qualified",
    )
    for fold in folds:
        selected_cells = [
            cell for record_id in fold.train_record_ids for cell in cell_by_record[record_id]
        ]
        expected_subjects = set(fold.train_subject_ids)
        for cell in selected_cells:
            if _subject_id(cell) not in expected_subjects:
                raise ValueError(f"training_subject_mismatch:{fold.fold_id}:{cell.record_id}")
        selected_cells.sort(
            key=lambda cell: (
                _subject_id(cell),
                cell.record_id,
                cell.coordinate_index,
                cell.coordinate_id,
            )
        )
        rows = [
            {
                "physical_subject_id": _subject_id(cell),
                "record_id": cell.record_id,
                "coordinate_id": cell.coordinate_id,
                "coordinate_index": cell.coordinate_index,
                "candidate_mae_bpm": cell.candidate_mae_bpm,
                "qualified": int(cell.qualified),
            }
            for cell in selected_cells
        ]
        path = output_root / f"{fold.fold_id}.csv"
        csv_sha = _write_csv(path, columns, rows)
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
        "schema_id": "cross_subject_multirecord_p3_training_inputs_v1",
        "status": "pass",
        "fold_count": len(fold_receipts),
        "identity": identity,
        "folds": fold_receipts,
    }
    _write_json(output_root / "training_input_manifest.json", receipt)
    return receipt


def freeze_training_inputs(
    training_root: Path,
    output_root: Path,
    *,
    expected_fold_count: int,
) -> dict[str, Any]:
    """Select and seal every fold without accepting a path to holdout metrics."""

    training_root = Path(training_root).resolve()
    output_root = Path(output_root).resolve()
    manifest_path = training_root / "training_input_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("training_input_manifest_missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = list(manifest.get("folds") or [])
    if manifest.get("status") != "pass" or len(folds) != expected_fold_count:
        raise ValueError("training_input_manifest_incomplete")
    output_root.mkdir(parents=True, exist_ok=True)
    selector_sha = _file_sha256(Path(__file__))
    selection_receipts: list[dict[str, Any]] = []
    for fold in folds:
        input_path = training_root / str(fold["training_input_file"])
        if _file_sha256(input_path) != str(fold["training_input_sha256"]):
            raise ValueError(f"training_input_hash_mismatch:{fold['fold_id']}")
        cells = _read_training_cells(input_path)
        selection = select_coordinate(cells, tuple(fold["train_subject_ids"]))
        selection_receipt = {
            "schema_id": "cross_subject_multirecord_fold_selection_v1",
            "fold_id": fold["fold_id"],
            "scene": fold["scene"],
            "holdout_subject_id": fold["holdout_subject_id"],
            "train_subject_ids": fold["train_subject_ids"],
            "train_record_ids": fold["train_record_ids"],
            "holdout_record_ids": fold["holdout_record_ids"],
            "training_input_sha256": fold["training_input_sha256"],
            "selector_source_sha256": selector_sha,
            "selection": asdict(selection),
        }
        path = output_root / f"{fold['fold_id']}.json"
        selection_sha = _write_json(path, selection_receipt)
        selection_receipts.append(
            {
                "fold_id": fold["fold_id"],
                "selection_file": path.name,
                "selection_sha256": selection_sha,
                "selected_coordinate_id": selection.coordinate_id,
                "selected_coordinate_index": selection.coordinate_index,
            }
        )
    receipt = {
        "schema_id": "cross_subject_multirecord_p3_freeze_receipt_v1",
        "status": "pass",
        "fold_count": len(selection_receipts),
        "training_input_manifest_sha256": _file_sha256(manifest_path),
        "selector_source_sha256": selector_sha,
        "selections": selection_receipts,
    }
    receipt_sha = _write_json(output_root / "p3_freeze_receipt.json", receipt)
    return {**receipt, "receipt_sha256": receipt_sha}


def reveal_holdout_results(
    folds: Sequence[GroupedFold],
    cells: Sequence[SelectionCell],
    selection_root: Path,
    *,
    expected_fold_count: int,
    record_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reveal selected holdouts only after the complete freeze receipt is verified."""

    selection_root = Path(selection_root).resolve()
    receipt_path = selection_root / "p3_freeze_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("freeze_receipt_missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    selections = list(receipt.get("selections") or [])
    if receipt.get("status") != "pass" or len(selections) != expected_fold_count:
        raise ValueError("freeze_receipt_incomplete")
    expected_fold_ids = {fold.fold_id for fold in folds}
    if {str(row["fold_id"]) for row in selections} != expected_fold_ids:
        raise ValueError("freeze_fold_set_mismatch")

    frozen: dict[str, dict[str, Any]] = {}
    for row in selections:
        path = selection_root / str(row["selection_file"])
        if _file_sha256(path) != str(row["selection_sha256"]):
            raise ValueError(f"selection_hash_mismatch:{row['fold_id']}")
        frozen[str(row["fold_id"])] = json.loads(path.read_text(encoding="utf-8"))

    cell_index = {(cell.record_id, cell.coordinate_id): cell for cell in cells}
    holdout_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        selection_receipt = frozen[fold.fold_id]
        selected = selection_receipt["selection"]
        coordinate_id = str(selected["coordinate_id"])
        selected_cells = []
        for record_id in fold.holdout_record_ids:
            cell = cell_index.get((record_id, coordinate_id))
            if cell is None:
                raise ValueError(f"selected_holdout_cell_missing:{fold.fold_id}:{record_id}")
            selected_cells.append(cell)
            payload = _cell_payload(cell)
            row = {
                "fold_id": fold.fold_id,
                "holdout_subject_id": fold.holdout_subject_id,
                "selected_coordinate_id": coordinate_id,
                "selection_sha256": _file_sha256(selection_root / f"{fold.fold_id}.json"),
                "compact_cell_sha256": compact_cell_sha256(cell),
                **payload,
            }
            baseline_mae = payload.get("baseline_mae_bpm")
            if baseline_mae is not None:
                row["mae_delta_vs_baseline_bpm"] = float(cell.candidate_mae_bpm) - float(
                    baseline_mae
                )
            if record_metadata is not None:
                try:
                    row.update(record_metadata[record_id])
                except KeyError as error:
                    raise ValueError(f"record_metadata_missing:{record_id}") from error
            holdout_rows.append(row)
        candidate_maes = [float(cell.candidate_mae_bpm) for cell in selected_cells]
        passed_count = sum(bool(cell.qualified) for cell in selected_cells)
        fold_row: dict[str, Any] = {
            "fold_id": fold.fold_id,
            "scene": fold.scene,
            "holdout_subject_id": fold.holdout_subject_id,
            "selected_coordinate_id": coordinate_id,
            "selected_coordinate_index": int(selected["coordinate_index"]),
            "minimum_training_subject_pass_fraction": selected["minimum_subject_pass_fraction"],
            "mean_training_subject_pass_fraction": selected["mean_subject_pass_fraction"],
            "worst_training_subject_mean_mae_bpm": selected["worst_subject_mean_mae_bpm"],
            "mean_training_subject_mean_mae_bpm": selected["mean_subject_mean_mae_bpm"],
            "coordinate_order_tiebreak_applied": selected["coordinate_order_tiebreak_applied"],
            "pre_order_tie_count": len(selected["pre_order_tied_coordinate_ids"]),
            "pre_order_tied_coordinate_ids_json": json.dumps(
                selected["pre_order_tied_coordinate_ids"], separators=(",", ":")
            ),
            "candidate_mean_mae_bpm": sum(candidate_maes) / len(candidate_maes),
            "candidate_median_mae_bpm": median(candidate_maes),
            "candidate_max_mae_bpm": max(candidate_maes),
            "passed_records": passed_count,
            "heldout_records": len(selected_cells),
            "strict_all_pass": passed_count == len(selected_cells),
        }
        if all(cell.baseline_mae_bpm is not None for cell in selected_cells):
            baseline_maes = [float(cell.baseline_mae_bpm) for cell in selected_cells]
            fold_row.update(
                {
                    "baseline_mean_mae_bpm": sum(baseline_maes) / len(baseline_maes),
                    "baseline_median_mae_bpm": median(baseline_maes),
                    "baseline_max_mae_bpm": max(baseline_maes),
                }
            )
        fold_rows.append(fold_row)
    return holdout_rows, fold_rows


def load_compact_cell_csv(path: Path) -> tuple[CompactCellMetric, ...]:
    """Load the sealed canonical P2 CSV using the ledger's public row contract."""

    boolean_columns = {
        "g1i_pass",
        "g2_pass",
        "g3_pass",
        "g4_pass",
        "g5_pass",
        "g7_pass",
        "qualified",
    }
    integer_columns = {
        "coordinate_index",
        "fs_target_hz",
        "memory_ms",
        "exclusion_half_width_bpm",
        "candidate_l10",
        "candidate_l20",
        "candidate_e10",
        "candidate_e20",
        "candidate_right_censored_recovery_count",
        "candidate_full_window_count",
        "candidate_reliable_window_count",
        "candidate_motion_window_count",
        "baseline_l10",
        "baseline_l20",
        "baseline_right_censored_recovery_count",
        "g5_right_censored_count",
        "attempt_count",
    }
    float_columns = {
        "mu_base",
        "candidate_mae_bpm",
        "baseline_mae_bpm",
        "g2_margin_s",
        "g3_margin_s",
        "g4_margin_bpm",
        "g7_margin_s",
        "solver_elapsed_s",
    }
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CELL_COLUMNS:
            raise ValueError("compact_cell_csv_columns_mismatch")
        rows = []
        for raw in reader:
            payload: dict[str, Any] = {}
            for column in CELL_COLUMNS:
                value = raw[column]
                if column in boolean_columns:
                    payload[column] = str(value).lower() in {"1", "true"}
                elif column in integer_columns:
                    payload[column] = int(value)
                elif column in float_columns:
                    payload[column] = float(value)
                else:
                    payload[column] = str(value)
            rows.append(CompactCellMetric(**payload))
    return tuple(rows)


def write_dict_csv(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    """Write deterministic dictionaries using the first row's stable field order."""

    if not rows:
        raise ValueError("empty_result_rows")
    return _write_csv(path, tuple(rows[0]), rows)


def compact_cell_sha256(cell: Any) -> str:
    """Hash one compact cell using the same canonical payload emitted at reveal."""

    return _semantic_sha256(_cell_payload(cell))


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _subject_id(cell: Any) -> str:
    if hasattr(cell, "subject_id"):
        return str(cell.subject_id)
    return str(cell.physical_subject_id)


def _cell_payload(cell: Any) -> dict[str, Any]:
    if is_dataclass(cell):
        return asdict(cell)
    return {
        "physical_subject_id": _subject_id(cell),
        "record_id": cell.record_id,
        "coordinate_id": cell.coordinate_id,
        "coordinate_index": cell.coordinate_index,
        "candidate_mae_bpm": cell.candidate_mae_bpm,
        "qualified": cell.qualified,
    }


def _read_training_cells(path: Path) -> tuple[SelectionCell, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return tuple(
        SelectionCell(
            subject_id=str(row["physical_subject_id"]),
            record_id=str(row["record_id"]),
            coordinate_id=str(row["coordinate_id"]),
            coordinate_index=int(row["coordinate_index"]),
            candidate_mae_bpm=float(row["candidate_mae_bpm"]),
            qualified=str(row["qualified"]).lower() in {"1", "true"},
        )
        for row in rows
    )


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode("utf-8")
    _atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    _atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
