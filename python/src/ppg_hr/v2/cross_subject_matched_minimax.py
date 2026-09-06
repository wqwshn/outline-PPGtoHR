"""Matched training-side MAE minimax utilities for HF and ACC LOSO follow-up."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .cross_subject_acc_paired import extract_native_windows
from .cross_subject_loso_source import GroupedFold
from .solver import V2SolverResult

EXPERIMENT_ID = "cross_subject_multirecord_hf_acc_matched_minimax_v1"
PARENT_HF_EXPERIMENT_ID = "cross_subject_multirecord_hf_loso_v1"
PARENT_ACC_EXPERIMENT_ID = "cross_subject_multirecord_acc_independent_physical4d_v1"
SELECTION_RULE_ID = "subject_balanced_route_native_mae_minimax_v1"
TIME_BIAS_S = 5.0

HF_THETA_HF_GATE = "hf_theta_hf_gate"
HF_THETA_ACC_MINIMAX = "hf_theta_acc_minimax"
HF_THETA_HF_MINIMAX = "hf_theta_hf_minimax"
ACC_THETA_ACC_MINIMAX = "acc_theta_acc_minimax"
ACC_THETA_HF_MINIMAX = "acc_theta_hf_minimax"
FIVE_CELL_LABELS = (
    HF_THETA_HF_GATE,
    HF_THETA_ACC_MINIMAX,
    HF_THETA_HF_MINIMAX,
    ACC_THETA_ACC_MINIMAX,
    ACC_THETA_HF_MINIMAX,
)
CONTRAST_COLUMNS = {
    "hf_minimax_gain_vs_gate_bpm": (HF_THETA_HF_GATE, HF_THETA_HF_MINIMAX),
    "hf_minimax_gain_vs_acc_coordinate_bpm": (
        HF_THETA_ACC_MINIMAX,
        HF_THETA_HF_MINIMAX,
    ),
    "independent_minimax_delta_bpm": (ACC_THETA_ACC_MINIMAX, HF_THETA_HF_MINIMAX),
    "same_hf_coordinate_reference_delta_bpm": (
        ACC_THETA_HF_MINIMAX,
        HF_THETA_HF_MINIMAX,
    ),
}


@dataclass(frozen=True)
class RouteSelectionCell:
    subject_id: str
    record_id: str
    coordinate_id: str
    coordinate_index: int
    mae_bpm: float


@dataclass(frozen=True)
class RouteSubjectSummary:
    subject_id: str
    record_count: int
    mean_mae_bpm: str


@dataclass(frozen=True)
class RouteFoldSelection:
    selection_rule_id: str
    coordinate_id: str
    coordinate_index: int
    worst_subject_mean_mae_bpm: str
    mean_subject_mean_mae_bpm: str
    subject_summaries: tuple[RouteSubjectSummary, ...]
    pre_order_tied_coordinate_ids: tuple[str, ...]
    coordinate_order_tiebreak_applied: bool


@dataclass(frozen=True)
class NamedCommonSupportResult:
    common_window_count: int
    common_window_sha256: str
    native_window_counts: dict[str, int]
    native_window_sha256: dict[str, str]
    lost_window_counts: dict[str, int]
    paired_mae_bpm: dict[str, float]


def select_route_native_coordinate(
    cells: Sequence[RouteSelectionCell], train_subject_ids: Sequence[str]
) -> RouteFoldSelection:
    """Minimise worst subject mean, mean subject mean, then frozen coordinate order."""

    subjects = tuple(train_subject_ids)
    if not cells:
        raise ValueError("matched_empty_training_cells")
    if not subjects or len(subjects) != len(set(subjects)):
        raise ValueError("matched_invalid_training_subjects")
    expected_subjects = set(subjects)
    if any(cell.subject_id not in expected_subjects for cell in cells):
        raise ValueError("matched_unexpected_training_subject")

    by_coordinate: dict[tuple[int, str], list[RouteSelectionCell]] = defaultdict(list)
    for cell in cells:
        by_coordinate[(cell.coordinate_index, cell.coordinate_id)].append(cell)
    if len({coordinate_id for _, coordinate_id in by_coordinate}) != len(by_coordinate):
        raise ValueError("matched_coordinate_id_has_multiple_indices")

    expected_record_keys: set[tuple[str, str]] | None = None
    ranked: list[tuple[tuple[Decimal, Decimal, int], RouteFoldSelection]] = []
    for (coordinate_index, coordinate_id), rows in by_coordinate.items():
        record_keys = [(row.subject_id, row.record_id) for row in rows]
        if len(record_keys) != len(set(record_keys)):
            raise ValueError(f"matched_duplicate_training_cell:{coordinate_id}")
        if expected_record_keys is None:
            expected_record_keys = set(record_keys)
        elif set(record_keys) != expected_record_keys:
            raise ValueError(f"matched_incomplete_coordinate_grid:{coordinate_id}")

        summaries = []
        subject_means = []
        for subject_id in sorted(expected_subjects):
            subject_rows = [row for row in rows if row.subject_id == subject_id]
            if not subject_rows:
                raise ValueError(f"matched_missing_training_subject:{coordinate_id}:{subject_id}")
            mean_mae = sum((Decimal(str(row.mae_bpm)) for row in subject_rows), Decimal()) / len(
                subject_rows
            )
            subject_means.append(mean_mae)
            summaries.append(
                RouteSubjectSummary(
                    subject_id=subject_id,
                    record_count=len(subject_rows),
                    mean_mae_bpm=str(mean_mae),
                )
            )
        worst = max(subject_means)
        mean_value = sum(subject_means, Decimal()) / len(subject_means)
        selection = RouteFoldSelection(
            selection_rule_id=SELECTION_RULE_ID,
            coordinate_id=coordinate_id,
            coordinate_index=coordinate_index,
            worst_subject_mean_mae_bpm=str(worst),
            mean_subject_mean_mae_bpm=str(mean_value),
            subject_summaries=tuple(summaries),
            pre_order_tied_coordinate_ids=(),
            coordinate_order_tiebreak_applied=False,
        )
        ranked.append(((worst, mean_value, coordinate_index), selection))

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


def write_training_inputs(
    folds: Sequence[GroupedFold],
    cells: Sequence[RouteSelectionCell],
    output_root: Path,
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one holdout-free, metric-minimal HF training file per fold."""

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    by_record: dict[str, list[RouteSelectionCell]] = defaultdict(list)
    for cell in cells:
        by_record[cell.record_id].append(cell)
    columns = (
        "physical_subject_id",
        "record_id",
        "coordinate_id",
        "coordinate_index",
        "mae_bpm",
    )
    receipts = []
    coordinate_count = len({(cell.coordinate_index, cell.coordinate_id) for cell in cells})
    for fold in folds:
        rows = [cell for record_id in fold.train_record_ids for cell in by_record[record_id]]
        rows.sort(
            key=lambda cell: (
                cell.subject_id,
                cell.record_id,
                cell.coordinate_index,
                cell.coordinate_id,
            )
        )
        if len(rows) != len(fold.train_record_ids) * coordinate_count:
            raise ValueError(f"matched_training_grid_incomplete:{fold.fold_id}")
        if {row.subject_id for row in rows} != set(fold.train_subject_ids):
            raise ValueError(f"matched_training_subject_set:{fold.fold_id}")
        path = root / f"{fold.fold_id}.csv"
        sha = write_csv(
            path,
            [
                {
                    "physical_subject_id": row.subject_id,
                    "record_id": row.record_id,
                    "coordinate_id": row.coordinate_id,
                    "coordinate_index": row.coordinate_index,
                    "mae_bpm": row.mae_bpm,
                }
                for row in rows
            ],
            columns=columns,
        )
        receipts.append(
            {
                "fold_id": fold.fold_id,
                "scene": fold.scene,
                "holdout_subject_id": fold.holdout_subject_id,
                "train_subject_ids": list(fold.train_subject_ids),
                "train_record_ids": list(fold.train_record_ids),
                "holdout_record_ids": list(fold.holdout_record_ids),
                "training_row_count": len(rows),
                "training_input_file": path.name,
                "training_input_sha256": sha,
            }
        )
    receipt = {
        "schema_id": "cross_subject_matched_minimax_training_inputs_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass",
        "fold_count": len(receipts),
        "identity": dict(identity),
        "folds": receipts,
    }
    write_json(root / "training_input_manifest.json", receipt)
    return receipt


def freeze_training_inputs(
    training_root: Path, selection_root: Path, *, expected_fold_count: int
) -> dict[str, Any]:
    """Select all folds from files that expose no holdout metric columns."""

    training_root = Path(training_root).resolve()
    selection_root = Path(selection_root).resolve()
    manifest_path = training_root / "training_input_manifest.json"
    manifest = read_json(manifest_path)
    folds = list(manifest.get("folds") or [])
    if manifest.get("status") != "pass" or len(folds) != expected_fold_count:
        raise ValueError("matched_training_manifest_incomplete")
    selection_root.mkdir(parents=True, exist_ok=True)
    source_sha = file_sha256(Path(__file__))
    selections = []
    for fold in folds:
        input_path = training_root / str(fold["training_input_file"])
        if file_sha256(input_path) != fold["training_input_sha256"]:
            raise ValueError(f"matched_training_hash_mismatch:{fold['fold_id']}")
        selection = select_route_native_coordinate(
            read_training_cells(input_path), tuple(fold["train_subject_ids"])
        )
        payload = {
            "schema_id": "cross_subject_matched_minimax_fold_selection_v1",
            "experiment_id": EXPERIMENT_ID,
            "fold_id": fold["fold_id"],
            "scene": fold["scene"],
            "holdout_subject_id": fold["holdout_subject_id"],
            "train_subject_ids": fold["train_subject_ids"],
            "train_record_ids": fold["train_record_ids"],
            "holdout_record_ids": fold["holdout_record_ids"],
            "training_input_sha256": fold["training_input_sha256"],
            "selector_source_sha256": source_sha,
            "selection": asdict(selection),
        }
        path = selection_root / f"{fold['fold_id']}.json"
        sha = write_json(path, payload)
        selections.append(
            {
                "fold_id": fold["fold_id"],
                "selection_file": path.name,
                "selection_sha256": sha,
                "selected_coordinate_id": selection.coordinate_id,
                "selected_coordinate_index": selection.coordinate_index,
            }
        )
    freeze = {
        "schema_id": "cross_subject_matched_minimax_freeze_receipt_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass",
        "fold_count": len(selections),
        "selection_rule_id": SELECTION_RULE_ID,
        "training_input_manifest_sha256": file_sha256(manifest_path),
        "selector_source_sha256": source_sha,
        "selections": selections,
    }
    sha = write_json(selection_root / "p1_freeze_receipt.json", freeze)
    return {**freeze, "receipt_sha256": sha}


def evaluate_named_common_support(
    results: Mapping[str, V2SolverResult],
    *,
    ref_data: np.ndarray,
    required_labels: Sequence[str] = FIVE_CELL_LABELS,
    time_bias_s: float = TIME_BIAS_S,
) -> NamedCommonSupportResult:
    """Evaluate arbitrary named trajectories on one exact reliable-window intersection."""

    labels = tuple(required_labels)
    if tuple(results) != labels:
        raise ValueError("matched_common_support_cell_order_mismatch")
    native = {
        label: extract_native_windows(results[label], ref_data=ref_data, time_bias_s=time_bias_s)
        for label in labels
    }
    ordered_keys = sorted(set.intersection(*(set(native[label]) for label in labels)))
    if not ordered_keys:
        raise ValueError("matched_empty_common_support")
    for key in ordered_keys:
        references = [native[label][key].reference_bpm for label in labels]
        if not all(
            math.isclose(references[0], value, abs_tol=1e-9, rel_tol=0.0)
            for value in references[1:]
        ):
            raise ValueError(f"matched_reference_mismatch:{key}")
    counts = {label: len(native[label]) for label in labels}
    hashes = {label: _window_sha(sorted(native[label])) for label in labels}
    maes = {
        label: float(
            np.mean(
                [
                    abs(native[label][key].prediction_bpm - native[label][key].reference_bpm)
                    for key in ordered_keys
                ]
            )
        )
        for label in labels
    }
    return NamedCommonSupportResult(
        common_window_count=len(ordered_keys),
        common_window_sha256=_window_sha(ordered_keys),
        native_window_counts=counts,
        native_window_sha256=hashes,
        lost_window_counts={label: counts[label] - len(ordered_keys) for label in labels},
        paired_mae_bpm=maes,
    )


def add_contrasts(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row)
    for name, (left, right) in CONTRAST_COLUMNS.items():
        output[name] = float(row[f"{left}_mae_bpm"]) - float(row[f"{right}_mae_bpm"])
    return output


def build_fold_results(record_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_fold: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in record_rows:
        by_fold[str(row["fold_id"])].append(row)
    outputs = []
    for fold_id, rows in sorted(by_fold.items()):
        first = rows[0]
        if any(
            row["scene"] != first["scene"]
            or row["physical_subject_id"] != first["physical_subject_id"]
            or row["theta_hf_minimax_coordinate_id"] != first["theta_hf_minimax_coordinate_id"]
            for row in rows
        ):
            raise ValueError(f"matched_fold_identity_mismatch:{fold_id}")
        output: dict[str, Any] = {
            "fold_id": fold_id,
            "scene": first["scene"],
            "holdout_subject_id": first["physical_subject_id"],
            "heldout_records": len(rows),
            "theta_hf_gate_coordinate_id": first["theta_hf_gate_coordinate_id"],
            "theta_hf_minimax_coordinate_id": first["theta_hf_minimax_coordinate_id"],
            "theta_acc_minimax_coordinate_id": first["theta_acc_minimax_coordinate_id"],
            "common_window_count": sum(int(row["common_window_count"]) for row in rows),
        }
        for label in FIVE_CELL_LABELS:
            output[f"{label}_mae_bpm"] = float(
                np.mean([float(row[f"{label}_mae_bpm"]) for row in rows])
            )
        outputs.append(add_contrasts(output))
    return outputs


def distribution_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("matched_invalid_summary_values")
    q1, q3 = np.percentile(array, [25.0, 75.0])
    return {
        "mean": float(np.mean(array)),
        "median": float(median(array.tolist())),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def read_training_cells(path: Path) -> tuple[RouteSelectionCell, ...]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = (
            "physical_subject_id",
            "record_id",
            "coordinate_id",
            "coordinate_index",
            "mae_bpm",
        )
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError("matched_training_columns_mismatch")
        return tuple(
            RouteSelectionCell(
                subject_id=str(row["physical_subject_id"]),
                record_id=str(row["record_id"]),
                coordinate_id=str(row["coordinate_id"]),
                coordinate_index=int(row["coordinate_index"]),
                mae_bpm=float(row["mae_bpm"]),
            )
            for row in reader
        )


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], *, columns: Sequence[str] | None = None
) -> str:
    if not rows:
        raise ValueError("matched_empty_csv_rows")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(columns or rows[0].keys())
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _window_sha(keys: Sequence[tuple[int, float]]) -> str:
    return semantic_sha256([[int(index), float(center)] for index, center in keys])
