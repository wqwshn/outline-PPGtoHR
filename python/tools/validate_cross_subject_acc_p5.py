"""Independently recompute ACC selections, common support, contrasts, and summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from ppg_hr.v2.cross_subject_acc_experiment import (
    ACC_EXPERIMENT_ID,
    EXPECTED_ACC_CALL_COUNT,
    PARENT_EXPERIMENT_ID,
)
from ppg_hr.v2.cross_subject_acc_source import load_acc_p0_snapshot
from ppg_hr.v2.cross_subject_loso_source import EXPECTED_FOLD_COUNT, EXPECTED_RECORD_COUNT
from ppg_hr.v2.preprocess import load_v2_reference
from ppg_hr.v2.report import load_v2_report

FOUR_CELL_LABELS = (
    "hf_theta_hf",
    "hf_theta_acc",
    "acc_theta_hf",
    "acc_theta_acc",
)
CONTRASTS = {
    "delta_end_bpm": ("acc_theta_acc", "hf_theta_hf"),
    "delta_hf_coordinate_reference_bpm": ("acc_theta_hf", "hf_theta_hf"),
    "delta_acc_coordinate_reference_bpm": ("acc_theta_acc", "hf_theta_acc"),
    "acc_selection_gain_bpm": ("acc_theta_hf", "acc_theta_acc"),
    "hf_own_coordinate_gain_bpm": ("hf_theta_acc", "hf_theta_hf"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--parent-experiment-root", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    parent_root = (
        args.parent_experiment_root.resolve()
        if args.parent_experiment_root
        else repo_root / "data" / "experiments" / PARENT_EXPERIMENT_ID
    )
    experiment_root = (
        args.experiment_root.resolve()
        if args.experiment_root
        else repo_root / "data" / "experiments" / ACC_EXPERIMENT_ID
    )
    output_root = experiment_root / "p5"
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot = load_acc_p0_snapshot(experiment_root / "p0", parent_experiment_root=parent_root)
    p2_receipt = _read_json(experiment_root / "p2" / "p2_receipt.json")
    p4_receipt = _read_json(experiment_root / "p4" / "p4_receipt.json")
    if (
        p2_receipt.get("status") != "pass"
        or p2_receipt.get("complete_cell_count") != EXPECTED_ACC_CALL_COUNT
        or p4_receipt.get("status") != "pass"
        or p4_receipt.get("record_count") != EXPECTED_RECORD_COUNT
        or p4_receipt.get("fold_count") != EXPECTED_FOLD_COUNT
    ):
        raise ValueError("p5_upstream_not_sealed")
    for name, expected_sha in dict(p4_receipt.get("artifact_sha256") or {}).items():
        path = experiment_root / "p4" / name
        if not path.is_file() or _file_sha256(path) != expected_sha:
            raise ValueError(f"p5_p4_artifact_hash_mismatch:{name}")
    audit_ledger = _read_json(experiment_root / "p4" / "p4_audit_ledger.json")
    if (
        audit_ledger.get("performance_exclusion_count") != 0
        or audit_ledger.get("performance_exclusions") != []
    ):
        raise ValueError("p5_performance_exclusion_detected")

    acc_mae = _read_acc_response(
        experiment_root / "p2" / "acc_cell_metrics.csv",
        expected_sha=p2_receipt["canonical_csv_sha256"],
    )
    selected = _validate_selections(
        snapshot.parent.folds,
        snapshot.parent.records,
        snapshot.parent.coordinates,
        acc_mae,
        experiment_root / "p3" / "selections",
        coordinate_order_sha256=snapshot.parent.coordinate_order_sha256,
    )
    hf_selected = _read_selection_coordinates(parent_root / "p3" / "selections")
    requests = _read_csv(experiment_root / "p4" / "four_cell_request_manifest.csv")
    _validate_requests(snapshot.parent.folds, requests, hf_selected, selected)
    report_paths = _validate_report_manifest(
        experiment_root / "p4" / "materialized_report_manifest.csv",
        requests,
    )
    record_rows = _read_csv(experiment_root / "p4" / "paired_record_results.csv")
    _validate_record_results(
        record_rows,
        requests=requests,
        report_paths=report_paths,
        records={record.record_id: record for record in snapshot.parent.records},
    )
    fold_rows = _read_csv(experiment_root / "p4" / "paired_fold_results.csv")
    independent_folds = _validate_fold_results(record_rows, fold_rows)
    _validate_aggregate_tables(experiment_root / "p4", independent_folds)

    receipt = {
        "schema_id": "cross_subject_multirecord_acc_p5_validation_receipt_v1",
        "experiment_id": ACC_EXPERIMENT_ID,
        "stage": "P5_validation",
        "status": "pass",
        "independent_validator_source_sha256": _file_sha256(Path(__file__)),
        "validated_acc_response_cell_count": len(acc_mae),
        "validated_selection_count": len(selected),
        "validated_four_cell_request_count": len(requests),
        "validated_unique_report_count": len(report_paths),
        "validated_record_count": len(record_rows),
        "validated_fold_count": len(independent_folds),
        "validated_scene_count": len({row["scene"] for row in independent_folds}),
        "validated_physical_subject_count": len(
            {row["holdout_subject_id"] for row in independent_folds}
        ),
        "validated_contrast_ids": list(CONTRASTS),
        "production_selector_called": False,
        "production_common_support_evaluator_called": False,
        "performance_exclusion_count": 0,
        "upstream_sha256": {
            "p2_receipt.json": _file_sha256(experiment_root / "p2" / "p2_receipt.json"),
            "p3_freeze_receipt.json": _file_sha256(
                experiment_root / "p3" / "selections" / "p3_freeze_receipt.json"
            ),
            "p4_receipt.json": _file_sha256(experiment_root / "p4" / "p4_receipt.json"),
        },
    }
    receipt_sha = _write_json(output_root / "p5_validation_receipt.json", receipt)
    print(
        json.dumps(
            {**receipt, "receipt_sha256": receipt_sha},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


def _read_acc_response(path: Path, *, expected_sha: str) -> dict[tuple[str, str], Decimal]:
    if _file_sha256(path) != expected_sha:
        raise ValueError("p5_acc_response_hash_mismatch")
    rows = _read_csv(path)
    if len(rows) != EXPECTED_ACC_CALL_COUNT:
        raise ValueError(f"p5_acc_response_count:{len(rows)}")
    output: dict[tuple[str, str], Decimal] = {}
    for row in rows:
        if row["route_id"] != "ACC" or json.loads(row["reference_groups_order_json"]) != ["ACC"]:
            raise ValueError("p5_acc_route_identity_mismatch")
        key = (row["record_id"], row["coordinate_id"])
        if key in output:
            raise ValueError(f"p5_duplicate_acc_response:{key}")
        value = Decimal(row["mae_bpm"])
        if not value.is_finite() or value < 0:
            raise ValueError(f"p5_invalid_acc_mae:{key}")
        output[key] = value
    return output


def _validate_selections(
    folds: Any,
    records: Any,
    coordinates: Any,
    acc_mae: dict[tuple[str, str], Decimal],
    selection_root: Path,
    *,
    coordinate_order_sha256: str,
) -> dict[str, dict[str, Any]]:
    by_record = {record.record_id: record for record in records}
    freeze = _read_json(selection_root / "p3_freeze_receipt.json")
    receipt_rows = list(freeze.get("selections") or [])
    if (
        freeze.get("status") != "pass"
        or len(receipt_rows) != EXPECTED_FOLD_COUNT
        or freeze.get("coordinate_order_sha256") != coordinate_order_sha256
    ):
        raise ValueError("p5_acc_selection_freeze_invalid")
    receipt_by_fold = {str(row["fold_id"]): row for row in receipt_rows}
    output = {}
    for fold in folds:
        ranked = []
        subject_means_by_coordinate: dict[str, dict[str, Decimal]] = {}
        for coordinate in coordinates:
            subject_means = {}
            for subject_id in fold.train_subject_ids:
                record_ids = [
                    record_id
                    for record_id in fold.train_record_ids
                    if by_record[record_id].physical_subject_id == subject_id
                ]
                values = [
                    acc_mae[(record_id, coordinate.coordinate_id)] for record_id in record_ids
                ]
                if not values:
                    raise ValueError(f"p5_empty_training_subject:{fold.fold_id}:{subject_id}")
                subject_means[subject_id] = sum(values, Decimal()) / len(values)
            worst = max(subject_means.values())
            mean = sum(subject_means.values(), Decimal()) / len(subject_means)
            ranked.append((worst, mean, coordinate.coordinate_index, coordinate.coordinate_id))
            subject_means_by_coordinate[coordinate.coordinate_id] = subject_means
        expected = min(ranked)
        row = receipt_by_fold.get(fold.fold_id)
        if row is None:
            raise ValueError(f"p5_selection_receipt_missing:{fold.fold_id}")
        path = selection_root / str(row["selection_file"])
        if _file_sha256(path) != row["selection_sha256"]:
            raise ValueError(f"p5_selection_receipt_hash:{fold.fold_id}")
        payload = _read_json(path)
        selection = payload["selection"]
        if (
            selection["coordinate_id"] != expected[3]
            or int(selection["coordinate_index"]) != expected[2]
            or Decimal(selection["worst_subject_mean_mae_bpm"]) != expected[0]
            or Decimal(selection["mean_subject_mean_mae_bpm"]) != expected[1]
            or payload.get("coordinate_order_sha256") != coordinate_order_sha256
            or [Decimal(str(value)) for value in payload.get("sorting_key", [])[:2]]
            != [expected[0], expected[1]]
            or int(payload.get("sorting_key", [None, None, -1])[2]) != expected[2]
        ):
            raise ValueError(f"p5_selection_mismatch:{fold.fold_id}")
        actual_subjects = {
            str(summary["subject_id"]): Decimal(summary["mean_mae_bpm"])
            for summary in selection["subject_summaries"]
        }
        if actual_subjects != subject_means_by_coordinate[expected[3]]:
            raise ValueError(f"p5_selection_subject_means:{fold.fold_id}")
        output[fold.fold_id] = {
            "coordinate_id": expected[3],
            "coordinate_index": expected[2],
            "selection_sha256": row["selection_sha256"],
        }
    return output


def _read_selection_coordinates(root: Path) -> dict[str, dict[str, Any]]:
    freeze = _read_json(root / "p3_freeze_receipt.json")
    output = {}
    for row in freeze.get("selections") or []:
        path = root / str(row["selection_file"])
        if _file_sha256(path) != row["selection_sha256"]:
            raise ValueError(f"p5_parent_selection_hash:{row['fold_id']}")
        selection = _read_json(path)["selection"]
        output[str(row["fold_id"])] = {
            "coordinate_id": str(selection["coordinate_id"]),
            "coordinate_index": int(selection["coordinate_index"]),
        }
    if len(output) != EXPECTED_FOLD_COUNT:
        raise ValueError("p5_parent_selection_count")
    return output


def _validate_requests(
    folds: Any,
    actual_rows: list[dict[str, str]],
    hf_selected: dict[str, dict[str, Any]],
    acc_selected: dict[str, dict[str, Any]],
) -> None:
    expected = set()
    for fold in folds:
        specs = (
            ("hf_theta_hf", "HF", "HF", hf_selected[fold.fold_id]),
            ("hf_theta_acc", "HF", "ACC", acc_selected[fold.fold_id]),
            ("acc_theta_hf", "ACC", "HF", hf_selected[fold.fold_id]),
            ("acc_theta_acc", "ACC", "ACC", acc_selected[fold.fold_id]),
        )
        for record_id in fold.holdout_record_ids:
            for label, route, source, selection in specs:
                expected.add(
                    (
                        fold.fold_id,
                        record_id,
                        label,
                        route,
                        source,
                        selection["coordinate_id"],
                        selection["coordinate_index"],
                    )
                )
    actual = {
        (
            row["fold_id"],
            row["record_id"],
            row["cell_label"],
            row["route_id"],
            row["coordinate_source"],
            row["coordinate_id"],
            int(row["coordinate_index"]),
        )
        for row in actual_rows
    }
    if len(actual_rows) != EXPECTED_RECORD_COUNT * 4 or actual != expected:
        raise ValueError("p5_four_cell_request_mismatch")


def _validate_report_manifest(
    path: Path,
    requests: list[dict[str, str]],
) -> dict[tuple[str, str, str], Path]:
    rows = _read_csv(path)
    expected = {(row["route_id"], row["record_id"], row["coordinate_id"]) for row in requests}
    output = {}
    for row in rows:
        key = (row["route_id"], row["record_id"], row["coordinate_id"])
        report_path = Path(row["report_path"])
        if (
            key in output
            or not report_path.is_file()
            or _file_sha256(report_path) != row["report_sha256"]
        ):
            raise ValueError(f"p5_report_manifest_invalid:{key}")
        output[key] = report_path
    if set(output) != expected:
        raise ValueError("p5_report_request_set_mismatch")
    return output


def _validate_record_results(
    actual_rows: list[dict[str, str]],
    *,
    requests: list[dict[str, str]],
    report_paths: dict[tuple[str, str, str], Path],
    records: dict[str, Any],
) -> None:
    if len(actual_rows) != EXPECTED_RECORD_COUNT:
        raise ValueError(f"p5_record_count:{len(actual_rows)}")
    requests_by_record: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in requests:
        requests_by_record[row["record_id"]][row["cell_label"]] = row
    actual_by_record = {row["record_id"]: row for row in actual_rows}
    if len(actual_by_record) != EXPECTED_RECORD_COUNT:
        raise ValueError("p5_duplicate_record_result")
    for record_id, by_label in requests_by_record.items():
        if set(by_label) != set(FOUR_CELL_LABELS):
            raise ValueError(f"p5_record_request_cells:{record_id}")
        native = {}
        for label, request in by_label.items():
            path = report_paths[(request["route_id"], record_id, request["coordinate_id"])]
            native[label] = _direct_native_windows(
                load_v2_report(path),
                load_v2_reference(records[record_id].ref_path),
            )
        common = sorted(set.intersection(*(set(native[label]) for label in FOUR_CELL_LABELS)))
        if not common:
            raise ValueError(f"p5_empty_common_support:{record_id}")
        row = actual_by_record[record_id]
        if int(row["common_window_count"]) != len(common) or row[
            "common_window_sha256"
        ] != _window_hash(common):
            raise ValueError(f"p5_common_support_mismatch:{record_id}")
        for label in FOUR_CELL_LABELS:
            keys = sorted(native[label])
            mae = float(
                np.mean([abs(native[label][key][0] - native[label][key][1]) for key in common])
            )
            _assert_close(float(row[f"{label}_mae_bpm"]), mae, f"{record_id}:{label}:mae")
            if (
                int(row[f"{label}_native_window_count"]) != len(keys)
                or row[f"{label}_native_window_sha256"] != _window_hash(keys)
                or int(row[f"{label}_lost_window_count"]) != len(keys) - len(common)
            ):
                raise ValueError(f"p5_native_support_mismatch:{record_id}:{label}")
        for column, (left, right) in CONTRASTS.items():
            expected = float(row[f"{left}_mae_bpm"]) - float(row[f"{right}_mae_bpm"])
            _assert_close(float(row[column]), expected, f"{record_id}:{column}")


def _direct_native_windows(
    payload: dict[str, Any], reference: np.ndarray
) -> dict[tuple[int, float], tuple[float, float]]:
    hr = np.asarray(payload.get("hr") or [], dtype=float)
    rows = list(payload.get("window_table") or [])
    if hr.ndim != 2 or hr.shape[0] == 0 or hr.shape[1] < 5 or len(rows) != hr.shape[0]:
        raise ValueError("p5_invalid_report_trajectory")
    ref = np.asarray(reference, dtype=float)
    order = np.argsort(ref[:, 0], kind="stable")
    ref_t = ref[order, 0]
    ref_hr = ref[order, 1]
    finite = np.isfinite(ref_t) & np.isfinite(ref_hr)
    ref_t = ref_t[finite]
    ref_hr = ref_hr[finite]
    values = np.interp(hr[:, 0] + 5.0, ref_t, ref_hr, left=np.nan, right=np.nan)
    output = {}
    for index, row in enumerate(rows):
        window_idx = int(row.get("window_idx", -1))
        center = float(row.get("center_s", float("nan")))
        prediction = float(hr[index, 3])
        expected_center = float(hr[index, 0])
        if window_idx != index or not math.isclose(
            center, expected_center, abs_tol=1e-9, rel_tol=0.0
        ):
            raise ValueError("p5_window_identity_mismatch")
        if math.isfinite(values[index]) and not math.isfinite(prediction):
            raise ValueError("p5_nonfinite_prediction")
        if bool(row.get("reliable")) and math.isfinite(values[index]) and math.isfinite(prediction):
            output[(window_idx, center)] = (prediction, float(values[index]))
    if not output:
        raise ValueError("p5_empty_native_support")
    return output


def _validate_fold_results(
    record_rows: list[dict[str, str]], actual_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    if len(actual_rows) != EXPECTED_FOLD_COUNT:
        raise ValueError(f"p5_fold_count:{len(actual_rows)}")
    records_by_fold: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in record_rows:
        records_by_fold[row["fold_id"]].append(row)
    actual_by_fold = {row["fold_id"]: row for row in actual_rows}
    independent = []
    for fold_id, rows in sorted(records_by_fold.items()):
        actual = actual_by_fold[fold_id]
        expected: dict[str, Any] = {
            "fold_id": fold_id,
            "scene": rows[0]["scene"],
            "holdout_subject_id": rows[0]["physical_subject_id"],
            "heldout_records": len(rows),
        }
        for label in FOUR_CELL_LABELS:
            expected[f"{label}_mae_bpm"] = _mean([float(row[f"{label}_mae_bpm"]) for row in rows])
        for column, (left, right) in CONTRASTS.items():
            expected[column] = expected[f"{left}_mae_bpm"] - expected[f"{right}_mae_bpm"]
        for key, value in expected.items():
            if isinstance(value, float):
                _assert_close(float(actual[key]), value, f"{fold_id}:{key}")
            elif str(actual[key]) != str(value):
                raise ValueError(f"p5_fold_identity_mismatch:{fold_id}:{key}")
        independent.append(expected)
    return independent


def _validate_aggregate_tables(root: Path, folds: list[dict[str, Any]]) -> None:
    matrix = _read_csv(root / "paired_2x2_matrix.csv")
    matrix_by_label = {row["cell_label"]: row for row in matrix}
    if set(matrix_by_label) != set(FOUR_CELL_LABELS):
        raise ValueError("p5_matrix_cell_set")
    for label in FOUR_CELL_LABELS:
        expected = _summary([float(row[f"{label}_mae_bpm"]) for row in folds])
        for key, value in expected.items():
            _assert_close(
                float(matrix_by_label[label][f"mae_bpm__{key}"]),
                value,
                f"matrix:{label}:{key}",
            )

    contrast = _read_csv(root / "paired_contrast_summary.csv")
    contrast_by_id = {row["contrast_id"]: row for row in contrast}
    if set(contrast_by_id) != set(CONTRASTS):
        raise ValueError("p5_contrast_set")
    for column in CONTRASTS:
        values = [float(row[column]) for row in folds]
        expected = _summary(values)
        for key, value in expected.items():
            _assert_close(float(contrast_by_id[column][key]), value, f"contrast:{column}:{key}")
        counts = (
            sum(value > 0 for value in values),
            sum(value == 0 for value in values),
            sum(value < 0 for value in values),
        )
        actual_counts = tuple(
            int(contrast_by_id[column][name])
            for name in (
                "positive_fold_count",
                "zero_fold_count",
                "negative_fold_count",
            )
        )
        if counts != actual_counts:
            raise ValueError(f"p5_contrast_sign_counts:{column}")

    _validate_group_summary(root / "paired_scene_summary.csv", folds, "scene")
    _validate_group_summary(root / "paired_subject_summary.csv", folds, "holdout_subject_id")
    overall = _read_json(root / "paired_overall_summary.json")
    _validate_summary_row(overall, folds, "overall")


def _validate_group_summary(path: Path, folds: list[dict[str, Any]], group_column: str) -> None:
    actual = {row["group"]: row for row in _read_csv(path)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in folds:
        grouped[str(row[group_column])].append(row)
    if set(actual) != set(grouped):
        raise ValueError(f"p5_group_set:{group_column}")
    for group, rows in grouped.items():
        _validate_summary_row(actual[group], rows, group)


def _validate_summary_row(
    actual: dict[str, Any], rows: list[dict[str, Any]], expected_group: str
) -> None:
    if str(actual["group"]) != expected_group or int(actual["fold_count"]) != len(rows):
        raise ValueError(f"p5_summary_identity:{expected_group}")
    columns = tuple(f"{label}_mae_bpm" for label in FOUR_CELL_LABELS) + tuple(CONTRASTS)
    for column in columns:
        summary = _summary([float(row[column]) for row in rows])
        for key, value in summary.items():
            _assert_close(
                float(actual[f"{column}__{key}"]),
                value,
                f"summary:{expected_group}:{column}:{key}",
            )


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
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


def _window_hash(keys: list[tuple[int, float]]) -> str:
    return _semantic_sha256([[int(index), float(center)] for index, center in keys])


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, abs_tol=1e-9, rel_tol=0.0):
        raise ValueError(f"p5_numeric_mismatch:{label}:{actual}:{expected}")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
