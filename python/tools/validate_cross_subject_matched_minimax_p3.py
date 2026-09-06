"""Independently validate selection, common support, and fold aggregation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppg_hr.v2.cross_subject_matched_minimax import (
    ACC_THETA_ACC_MINIMAX,
    ACC_THETA_HF_MINIMAX,
    CONTRAST_COLUMNS,
    EXPERIMENT_ID,
    FIVE_CELL_LABELS,
    HF_THETA_ACC_MINIMAX,
    HF_THETA_HF_GATE,
    HF_THETA_HF_MINIMAX,
    PARENT_ACC_EXPERIMENT_ID,
    PARENT_HF_EXPERIMENT_ID,
    file_sha256,
    write_csv,
    write_json,
)

FLOAT_TOLERANCE = 1e-9
TRAINING_COLUMNS = (
    "physical_subject_id",
    "record_id",
    "coordinate_id",
    "coordinate_index",
    "mae_bpm",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--experiment-root", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = (
        args.experiment_root.resolve()
        if args.experiment_root
        else repo_root / "data" / "experiments" / EXPERIMENT_ID
    )
    hf_root = repo_root / "data" / "experiments" / PARENT_HF_EXPERIMENT_ID
    acc_root = repo_root / "data" / "experiments" / PARENT_ACC_EXPERIMENT_ID
    output_root = root / "p3"
    output_root.mkdir(parents=True, exist_ok=True)

    p2_root = root / "p2"
    p2_receipt = _read_json(p2_root / "p2_receipt.json")
    _require(p2_receipt.get("status") == "pass", "validator_p2_not_pass")
    _require(p2_receipt.get("record_count") == 143, "validator_p2_record_count")
    _require(p2_receipt.get("fold_count") == 48, "validator_p2_fold_count")
    _require(
        p2_receipt.get("full_response_recalculation_count") == 0,
        "validator_full_surface_was_recalculated",
    )
    _require(
        p2_receipt.get("technical_failure_event_count") == 0,
        "validator_technical_failure_event",
    )
    _require(
        p2_receipt.get("performance_exclusion_count") == 0,
        "validator_performance_exclusion",
    )
    for name, expected in p2_receipt["artifact_sha256"].items():
        _require(file_sha256(p2_root / name) == expected, f"validator_p2_hash:{name}")

    hf_selection = _validate_selection_set(
        root / "p1" / "training",
        root / "p1" / "selections",
        freeze_name="p1_freeze_receipt.json",
    )
    acc_selection = _validate_selection_set(
        acc_root / "p3" / "training",
        acc_root / "p3" / "selections",
        freeze_name="p3_freeze_receipt.json",
    )
    gate_selection = _load_hashed_selection_set(
        hf_root / "p3" / "selections", "p3_freeze_receipt.json"
    )

    request_path = p2_root / "five_cell_request_manifest.csv"
    requests = _read_csv(request_path)
    _require(len(requests) == 715, "validator_request_count")
    _validate_request_coordinates(requests, hf_selection, acc_selection, gate_selection)

    report_path = p2_root / "materialized_report_manifest.csv"
    reports = _read_csv(report_path)
    _require(len(reports) == 691, "validator_unique_report_count")
    report_index = {}
    for row in reports:
        key = (row["route_id"], row["record_id"], row["coordinate_id"])
        _require(key not in report_index, f"validator_duplicate_report:{key}")
        path = Path(row["report_path"]).resolve()
        _require(path.is_file(), f"validator_missing_report:{key}")
        _require(file_sha256(path) == row["report_sha256"], f"validator_report_hash:{key}")
        report_index[key] = path

    expected_records = _read_csv(p2_root / "matched_record_results.csv")
    independent_records = _reconstruct_records(requests, report_index, expected_records)
    independent_folds = _reconstruct_folds(independent_records)
    expected_folds = _read_csv(p2_root / "matched_fold_results.csv")
    _compare_fold_rows(independent_folds, expected_folds)
    _validate_summaries(independent_folds, p2_root)

    record_sha = write_csv(output_root / "independent_record_results.csv", independent_records)
    fold_sha = write_csv(output_root / "independent_fold_results.csv", independent_folds)
    receipt = {
        "schema_id": "cross_subject_matched_minimax_p3_validation_receipt_v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": "P3-validation",
        "status": "pass",
        "independent_hf_selection_match_count": len(hf_selection),
        "independent_acc_selection_match_count": len(acc_selection),
        "old_hf_gate_selection_hash_check_count": len(gate_selection),
        "logical_request_count": len(requests),
        "unique_report_hash_check_count": len(reports),
        "independent_record_reconstruction_count": len(independent_records),
        "independent_fold_reconstruction_count": len(independent_folds),
        "float_tolerance": FLOAT_TOLERANCE,
        "holdout_metric_columns_exposed_to_new_hf_selector": [],
        "full_response_recalculation_count": 0,
        "technical_failure_event_count": 0,
        "performance_exclusion_count": 0,
        "artifact_sha256": {
            "independent_record_results.csv": record_sha,
            "independent_fold_results.csv": fold_sha,
            "p2_receipt.json": file_sha256(p2_root / "p2_receipt.json"),
            "five_cell_request_manifest.csv": file_sha256(request_path),
            "materialized_report_manifest.csv": file_sha256(report_path),
        },
    }
    sha = write_json(output_root / "p3_validation_receipt.json", receipt)
    print(json.dumps({**receipt, "receipt_sha256": sha}, ensure_ascii=False, indent=2))
    return 0


def _validate_selection_set(
    training_root: Path, selection_root: Path, *, freeze_name: str
) -> dict[str, dict[str, Any]]:
    manifest_path = training_root / "training_input_manifest.json"
    manifest = _read_json(manifest_path)
    freeze = _read_json(selection_root / freeze_name)
    folds = list(manifest.get("folds") or [])
    frozen = {row["fold_id"]: row for row in freeze.get("selections") or []}
    _require(manifest.get("status") == "pass", "validator_training_manifest_status")
    _require(freeze.get("status") == "pass", "validator_selection_freeze_status")
    _require(len(folds) == len(frozen) == 48, "validator_selection_fold_count")
    output = {}
    for fold in folds:
        fold_id = str(fold["fold_id"])
        training_path = training_root / str(fold["training_input_file"])
        _require(
            file_sha256(training_path) == fold["training_input_sha256"],
            f"validator_training_hash:{fold_id}",
        )
        rows = _read_csv(training_path, expected_columns=TRAINING_COLUMNS)
        holdout_records = set(fold["holdout_record_ids"])
        _require(
            not holdout_records.intersection(row["record_id"] for row in rows),
            f"validator_holdout_leakage:{fold_id}",
        )
        selected = _independent_select(rows, tuple(fold["train_subject_ids"]))
        frozen_row = frozen[fold_id]
        selection_path = selection_root / str(frozen_row["selection_file"])
        _require(
            file_sha256(selection_path) == frozen_row["selection_sha256"],
            f"validator_selection_hash:{fold_id}",
        )
        payload = _read_json(selection_path)["selection"]
        for key in (
            "coordinate_id",
            "coordinate_index",
            "worst_subject_mean_mae_bpm",
            "mean_subject_mean_mae_bpm",
        ):
            _require(
                str(payload[key]) == str(selected[key]), f"validator_selection:{fold_id}:{key}"
            )
        output[fold_id] = {
            **selected,
            "selection_sha256": str(frozen_row["selection_sha256"]),
        }
    return output


def _independent_select(
    rows: list[dict[str, str]], train_subject_ids: tuple[str, ...]
) -> dict[str, Any]:
    subjects = set(train_subject_ids)
    _require(subjects and len(subjects) == len(train_subject_ids), "validator_train_subjects")
    by_coordinate: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        _require(row["physical_subject_id"] in subjects, "validator_unexpected_train_subject")
        by_coordinate[(int(row["coordinate_index"]), row["coordinate_id"])].append(row)
    ranked = []
    expected_keys = None
    for (index, coordinate_id), cells in by_coordinate.items():
        record_keys = {(row["physical_subject_id"], row["record_id"]) for row in cells}
        _require(len(record_keys) == len(cells), f"validator_duplicate_cell:{coordinate_id}")
        if expected_keys is None:
            expected_keys = record_keys
        _require(record_keys == expected_keys, f"validator_incomplete_grid:{coordinate_id}")
        subject_means = []
        for subject in sorted(subjects):
            values = [
                Decimal(row["mae_bpm"]) for row in cells if row["physical_subject_id"] == subject
            ]
            _require(bool(values), f"validator_missing_subject:{coordinate_id}:{subject}")
            subject_means.append(sum(values, Decimal()) / len(values))
        worst = max(subject_means)
        mean_value = sum(subject_means, Decimal()) / len(subject_means)
        ranked.append(((worst, mean_value, index), coordinate_id))
    key, coordinate_id = min(ranked)
    return {
        "coordinate_id": coordinate_id,
        "coordinate_index": key[2],
        "worst_subject_mean_mae_bpm": str(key[0]),
        "mean_subject_mean_mae_bpm": str(key[1]),
    }


def _load_hashed_selection_set(root: Path, freeze_name: str) -> dict[str, dict[str, Any]]:
    freeze = _read_json(root / freeze_name)
    rows = list(freeze.get("selections") or [])
    _require(freeze.get("status") == "pass" and len(rows) == 48, "validator_gate_freeze")
    output = {}
    for row in rows:
        path = root / row["selection_file"]
        _require(file_sha256(path) == row["selection_sha256"], "validator_gate_selection_hash")
        selection = _read_json(path)["selection"]
        output[row["fold_id"]] = {
            "coordinate_id": selection["coordinate_id"],
            "coordinate_index": int(selection["coordinate_index"]),
            "selection_sha256": row["selection_sha256"],
        }
    return output


def _validate_request_coordinates(
    rows: list[dict[str, str]],
    hf: dict[str, dict[str, Any]],
    acc: dict[str, dict[str, Any]],
    gate: dict[str, dict[str, Any]],
) -> None:
    expected = {
        HF_THETA_HF_GATE: ("HF", gate),
        HF_THETA_ACC_MINIMAX: ("HF", acc),
        HF_THETA_HF_MINIMAX: ("HF", hf),
        ACC_THETA_ACC_MINIMAX: ("ACC", acc),
        ACC_THETA_HF_MINIMAX: ("ACC", hf),
    }
    counts = defaultdict(int)
    for row in rows:
        label = row["cell_label"]
        fold_id = row["fold_id"]
        route, selections = expected[label]
        selected = selections[fold_id]
        _require(row["route_id"] == route, f"validator_request_route:{fold_id}:{label}")
        _require(
            row["coordinate_id"] == selected["coordinate_id"],
            f"validator_request_coordinate:{fold_id}:{label}",
        )
        _require(
            int(row["coordinate_index"]) == selected["coordinate_index"],
            f"validator_request_coordinate_index:{fold_id}:{label}",
        )
        _require(
            row["selection_sha256"] == selected["selection_sha256"],
            f"validator_request_selection_hash:{fold_id}:{label}",
        )
        counts[label] += 1
    _require(set(counts) == set(FIVE_CELL_LABELS), "validator_request_labels")
    _require(all(count == 143 for count in counts.values()), "validator_request_label_counts")


def _reconstruct_records(
    requests: list[dict[str, str]],
    report_index: dict[tuple[str, str, str], Path],
    expected_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    expected = {row["record_id"]: row for row in expected_rows}
    _require(len(expected) == len(expected_rows) == 143, "validator_expected_record_count")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in requests:
        grouped[row["record_id"]].append(row)
    report_cache = {}
    reference_cache = {}
    outputs = []
    for record_id, rows in sorted(grouped.items()):
        by_label = {row["cell_label"]: row for row in rows}
        _require(set(by_label) == set(FIVE_CELL_LABELS), f"validator_record_cells:{record_id}")
        native = {}
        for label in FIVE_CELL_LABELS:
            item = by_label[label]
            key = (item["route_id"], record_id, item["coordinate_id"])
            path = report_index[key]
            if path not in report_cache:
                report_cache[path] = _read_json(path)
            report = report_cache[path]
            ref_path = Path(report["ref_path"]).resolve()
            if ref_path not in reference_cache:
                reference_cache[ref_path] = _read_reference(ref_path)
            native[label] = _independent_native_windows(report, reference_cache[ref_path])
        common = sorted(set.intersection(*(set(native[label]) for label in FIVE_CELL_LABELS)))
        _require(bool(common), f"validator_empty_common_support:{record_id}")
        current: dict[str, Any] = {
            "fold_id": rows[0]["fold_id"],
            "physical_subject_id": rows[0]["physical_subject_id"],
            "scene": rows[0]["scene"],
            "record_id": record_id,
            "common_window_count": len(common),
            "common_window_sha256": _window_sha(common),
        }
        for label in FIVE_CELL_LABELS:
            values = native[label]
            current[f"{label}_native_window_count"] = len(values)
            current[f"{label}_native_window_sha256"] = _window_sha(sorted(values))
            current[f"{label}_lost_window_count"] = len(values) - len(common)
            current[f"{label}_mae_bpm"] = float(
                np.mean([abs(values[key][0] - values[key][1]) for key in common])
            )
        for name, (left, right) in CONTRAST_COLUMNS.items():
            current[name] = current[f"{left}_mae_bpm"] - current[f"{right}_mae_bpm"]
        _compare_record(current, expected[record_id])
        outputs.append(current)
    _require(len(outputs) == 143, "validator_reconstructed_record_count")
    return outputs


def _independent_native_windows(
    report: dict[str, Any], reference: np.ndarray
) -> dict[tuple[int, float], tuple[float, float]]:
    hr = np.asarray(report["hr"], dtype=float)
    windows = list(report["window_table"])
    _require(hr.ndim == 2 and hr.shape[1] >= 4, "validator_hr_shape")
    _require(len(windows) == hr.shape[0], "validator_window_table_length")
    centers = hr[:, 0]
    predictions = hr[:, 3]
    references = np.interp(
        centers + 5.0,
        reference[:, 0],
        reference[:, 1],
        left=np.nan,
        right=np.nan,
    )
    output = {}
    for index, row in enumerate(windows):
        center = float(centers[index])
        _require(int(row["window_idx"]) == index, "validator_window_index")
        _require(_close(float(row["center_s"]), center), "validator_window_center")
        prediction = float(predictions[index])
        reference_bpm = float(references[index])
        if bool(row["reliable"]) and math.isfinite(reference_bpm):
            _require(math.isfinite(prediction), "validator_nonfinite_prediction")
            key = (index, center)
            _require(key not in output, "validator_duplicate_window")
            output[key] = (prediction, reference_bpm)
    _require(bool(output), "validator_empty_native_support")
    return output


def _read_reference(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, skiprows=3, header=None)
    _require(frame.shape[1] >= 3, "validator_reference_columns")
    times = []
    for raw in frame.iloc[:, 1].astype(str).str.strip():
        try:
            times.append(pd.to_timedelta(raw).total_seconds())
        except (TypeError, ValueError):
            try:
                times.append(float(raw))
            except ValueError:
                times.append(float("nan"))
    seconds = np.asarray(times, dtype=float)
    bpm = pd.to_numeric(frame.iloc[:, 2], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(seconds) & np.isfinite(bpm)
    values = np.column_stack((seconds[finite], bpm[finite]))
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    _require(values.shape[0] >= 2 and np.all(np.diff(values[:, 0]) > 0), "validator_reference")
    return values


def _compare_record(actual: dict[str, Any], expected: dict[str, str]) -> None:
    text_columns = ("fold_id", "physical_subject_id", "scene", "record_id", "common_window_sha256")
    integer_columns = ("common_window_count",) + tuple(
        name
        for label in FIVE_CELL_LABELS
        for name in (f"{label}_native_window_count", f"{label}_lost_window_count")
    )
    hash_columns = tuple(f"{label}_native_window_sha256" for label in FIVE_CELL_LABELS)
    float_columns = tuple(f"{label}_mae_bpm" for label in FIVE_CELL_LABELS) + tuple(
        CONTRAST_COLUMNS
    )
    for column in text_columns + hash_columns:
        _require(str(actual[column]) == expected[column], f"validator_record_text:{column}")
    for column in integer_columns:
        _require(int(actual[column]) == int(expected[column]), f"validator_record_int:{column}")
    for column in float_columns:
        _require(
            _close(float(actual[column]), float(expected[column])),
            f"validator_record_float:{column}",
        )


def _reconstruct_folds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["fold_id"]].append(row)
    outputs = []
    for fold_id, rows in sorted(grouped.items()):
        first = rows[0]
        output: dict[str, Any] = {
            "fold_id": fold_id,
            "scene": first["scene"],
            "holdout_subject_id": first["physical_subject_id"],
            "heldout_records": len(rows),
            "common_window_count": sum(int(row["common_window_count"]) for row in rows),
        }
        for label in FIVE_CELL_LABELS:
            output[f"{label}_mae_bpm"] = float(
                np.mean([float(row[f"{label}_mae_bpm"]) for row in rows])
            )
        for name, (left, right) in CONTRAST_COLUMNS.items():
            output[name] = output[f"{left}_mae_bpm"] - output[f"{right}_mae_bpm"]
        outputs.append(output)
    _require(len(outputs) == 48, "validator_reconstructed_fold_count")
    return outputs


def _compare_fold_rows(actual: list[dict[str, Any]], expected: list[dict[str, str]]) -> None:
    by_fold = {row["fold_id"]: row for row in expected}
    _require(len(by_fold) == len(expected) == 48, "validator_expected_fold_count")
    for row in actual:
        saved = by_fold[row["fold_id"]]
        for column in ("scene", "holdout_subject_id"):
            _require(row[column] == saved[column], f"validator_fold_text:{column}")
        for column in ("heldout_records", "common_window_count"):
            _require(int(row[column]) == int(saved[column]), f"validator_fold_int:{column}")
        for column in tuple(f"{label}_mae_bpm" for label in FIVE_CELL_LABELS) + tuple(
            CONTRAST_COLUMNS
        ):
            _require(
                _close(float(row[column]), float(saved[column])), f"validator_fold_float:{column}"
            )


def _validate_summaries(folds: list[dict[str, Any]], p2_root: Path) -> None:
    cell_rows = {row["cell_label"]: row for row in _read_csv(p2_root / "five_cell_summary.csv")}
    for label in FIVE_CELL_LABELS:
        _compare_summary(
            [row[f"{label}_mae_bpm"] for row in folds], cell_rows[label], f"cell:{label}"
        )
    contrast_rows = {row["contrast_id"]: row for row in _read_csv(p2_root / "contrast_summary.csv")}
    for name in CONTRAST_COLUMNS:
        values = [row[name] for row in folds]
        saved = contrast_rows[name]
        _compare_summary(values, saved, f"contrast:{name}")
        _require(sum(value > 0 for value in values) == int(saved["positive_fold_count"]), name)
        _require(sum(value == 0 for value in values) == int(saved["zero_fold_count"]), name)
        _require(sum(value < 0 for value in values) == int(saved["negative_fold_count"]), name)


def _compare_summary(values: list[float], saved: dict[str, str], prefix: str) -> None:
    array = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(array, (25.0, 75.0))
    actual = {
        "mean": np.mean(array),
        "median": np.median(array),
        "sd": np.std(array, ddof=1),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": np.min(array),
        "max": np.max(array),
    }
    _require(int(saved["fold_count"]) == len(values), f"validator_summary_count:{prefix}")
    for key, value in actual.items():
        _require(_close(float(value), float(saved[key])), f"validator_summary:{prefix}:{key}")


def _read_csv(path: Path, expected_columns: tuple[str, ...] | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if expected_columns is not None:
            _require(
                tuple(reader.fieldnames or ()) == expected_columns, f"validator_columns:{path}"
            )
        return list(reader)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _window_sha(keys: list[tuple[int, float]]) -> str:
    value = [[int(index), float(center)] for index, center in keys]
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, abs_tol=FLOAT_TOLERANCE, rel_tol=0.0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":
    raise SystemExit(main())
