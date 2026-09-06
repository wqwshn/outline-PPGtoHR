"""Four-cell common-support evaluation for the ACC cross-subject experiment."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

import numpy as np

from .solver import V2SolverResult

HF_THETA_HF = "hf_theta_hf"
HF_THETA_ACC = "hf_theta_acc"
ACC_THETA_HF = "acc_theta_hf"
ACC_THETA_ACC = "acc_theta_acc"
FOUR_CELL_LABELS = (HF_THETA_HF, HF_THETA_ACC, ACC_THETA_HF, ACC_THETA_ACC)
CONTRAST_COLUMNS = {
    "delta_end_bpm": (ACC_THETA_ACC, HF_THETA_HF),
    "delta_hf_coordinate_reference_bpm": (ACC_THETA_HF, HF_THETA_HF),
    "delta_acc_coordinate_reference_bpm": (ACC_THETA_ACC, HF_THETA_ACC),
    "acc_selection_gain_bpm": (ACC_THETA_HF, ACC_THETA_ACC),
    "hf_own_coordinate_gain_bpm": (HF_THETA_ACC, HF_THETA_HF),
}


@dataclass(frozen=True)
class NativeWindowValue:
    window_idx: int
    center_s: float
    prediction_bpm: float
    reference_bpm: float


@dataclass(frozen=True)
class PairedSupportResult:
    common_window_count: int
    common_window_sha256: str
    native_window_counts: dict[str, int]
    native_window_sha256: dict[str, str]
    lost_window_counts: dict[str, int]
    paired_mae_bpm: dict[str, float]


def extract_native_windows(
    result: V2SolverResult,
    *,
    ref_data: np.ndarray,
    time_bias_s: float = 5.0,
) -> dict[tuple[int, float], NativeWindowValue]:
    """Return reliable, finite-prediction windows with fixed-time reference overlap."""

    hr = np.asarray(result.HR, dtype=float)
    rows = list(result.window_table)
    if hr.ndim != 2 or hr.shape[0] == 0 or hr.shape[1] < 5:
        raise ValueError(f"paired_invalid_hr_shape:{hr.shape}")
    if len(rows) != hr.shape[0]:
        raise ValueError("paired_window_table_length_mismatch")
    centers = hr[:, 0]
    predictions = hr[:, 3]
    references = _interpolate_reference(ref_data, centers + float(time_bias_s))
    native: dict[tuple[int, float], NativeWindowValue] = {}
    for expected_idx, row in enumerate(rows):
        window_idx = int(row.get("window_idx", -1))
        center_s = float(row.get("center_s", float("nan")))
        if window_idx != expected_idx:
            raise ValueError("paired_window_index_mismatch")
        if not math.isclose(center_s, float(centers[expected_idx]), abs_tol=1e-9, rel_tol=0.0):
            raise ValueError("paired_window_center_mismatch")
        prediction = float(predictions[expected_idx])
        reference = float(references[expected_idx])
        if math.isfinite(reference) and not math.isfinite(prediction):
            raise ValueError("paired_nonfinite_prediction_in_reference_overlap")
        if bool(row.get("reliable")) and math.isfinite(reference) and math.isfinite(prediction):
            key = (window_idx, center_s)
            if key in native:
                raise ValueError("paired_duplicate_window_key")
            native[key] = NativeWindowValue(
                window_idx=window_idx,
                center_s=center_s,
                prediction_bpm=prediction,
                reference_bpm=reference,
            )
    if not native:
        raise ValueError("paired_empty_native_support")
    return native


def evaluate_four_cell_common_support(
    results: Mapping[str, V2SolverResult],
    *,
    ref_data: np.ndarray,
    time_bias_s: float = 5.0,
) -> PairedSupportResult:
    if set(results) != set(FOUR_CELL_LABELS):
        raise ValueError("paired_four_cell_set_mismatch")
    native = {
        label: extract_native_windows(results[label], ref_data=ref_data, time_bias_s=time_bias_s)
        for label in FOUR_CELL_LABELS
    }
    common_keys = set.intersection(*(set(native[label]) for label in FOUR_CELL_LABELS))
    ordered_keys = sorted(common_keys)
    if not ordered_keys:
        raise ValueError("paired_empty_common_support")
    for key in ordered_keys:
        reference_values = [native[label][key].reference_bpm for label in FOUR_CELL_LABELS]
        if not all(
            math.isclose(reference_values[0], value, abs_tol=1e-9, rel_tol=0.0)
            for value in reference_values[1:]
        ):
            raise ValueError(f"paired_reference_mismatch:{key}")
    native_counts = {label: len(native[label]) for label in FOUR_CELL_LABELS}
    native_hashes = {label: _window_key_sha256(sorted(native[label])) for label in FOUR_CELL_LABELS}
    paired_mae = {
        label: float(
            np.mean(
                [
                    abs(native[label][key].prediction_bpm - native[label][key].reference_bpm)
                    for key in ordered_keys
                ]
            )
        )
        for label in FOUR_CELL_LABELS
    }
    return PairedSupportResult(
        common_window_count=len(ordered_keys),
        common_window_sha256=_window_key_sha256(ordered_keys),
        native_window_counts=native_counts,
        native_window_sha256=native_hashes,
        lost_window_counts={
            label: native_counts[label] - len(ordered_keys) for label in FOUR_CELL_LABELS
        },
        paired_mae_bpm=paired_mae,
    )


def add_frozen_contrasts(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    for output, (left, right) in CONTRAST_COLUMNS.items():
        enriched[output] = float(row[f"{left}_mae_bpm"]) - float(row[f"{right}_mae_bpm"])
    return enriched


def build_fold_results(
    record_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_fold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in record_rows:
        by_fold[str(row["fold_id"])].append(row)
    results = []
    for fold_id, rows in sorted(by_fold.items()):
        first = rows[0]
        fold_row: dict[str, Any] = {
            "fold_id": fold_id,
            "scene": first["scene"],
            "holdout_subject_id": first["physical_subject_id"],
            "heldout_records": len(rows),
            "theta_hf_coordinate_id": first["theta_hf_coordinate_id"],
            "theta_acc_coordinate_id": first["theta_acc_coordinate_id"],
            "common_window_count": sum(int(row["common_window_count"]) for row in rows),
        }
        if any(
            row["scene"] != first["scene"]
            or row["physical_subject_id"] != first["physical_subject_id"]
            or row["theta_hf_coordinate_id"] != first["theta_hf_coordinate_id"]
            or row["theta_acc_coordinate_id"] != first["theta_acc_coordinate_id"]
            for row in rows
        ):
            raise ValueError(f"paired_fold_identity_mismatch:{fold_id}")
        for label in FOUR_CELL_LABELS:
            fold_row[f"{label}_mae_bpm"] = _mean([float(row[f"{label}_mae_bpm"]) for row in rows])
        results.append(add_frozen_contrasts(fold_row))
    return results


def summarise_fold_results(
    fold_rows: Sequence[dict[str, Any]],
    *,
    group_column: str | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if group_column is None:
        grouped["overall"] = list(fold_rows)
    else:
        for row in fold_rows:
            grouped[str(row[group_column])].append(row)
    outputs = []
    value_columns = tuple(f"{label}_mae_bpm" for label in FOUR_CELL_LABELS) + tuple(
        CONTRAST_COLUMNS
    )
    for group, rows in sorted(grouped.items()):
        output: dict[str, Any] = {
            "group": group,
            "fold_count": len(rows),
        }
        for column in value_columns:
            values = [float(row[column]) for row in rows]
            summary = distribution_summary(values)
            for statistic, value in summary.items():
                output[f"{column}__{statistic}"] = value
        if group_column is None:
            deltas = [float(row["delta_end_bpm"]) for row in rows]
            output.update(
                {
                    "delta_end_positive_fold_count": sum(value > 0 for value in deltas),
                    "delta_end_zero_fold_count": sum(value == 0 for value in deltas),
                    "delta_end_negative_fold_count": sum(value < 0 for value in deltas),
                }
            )
        outputs.append(output)
    return outputs


def distribution_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("paired_invalid_summary_values")
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


def _interpolate_reference(reference: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    source = np.asarray(reference, dtype=float)
    if source.ndim != 2 or source.shape[1] < 2 or source.shape[0] < 2:
        raise ValueError("paired_invalid_reference_shape")
    order = np.argsort(source[:, 0], kind="stable")
    ref_t = source[order, 0]
    ref_hr = source[order, 1]
    finite = np.isfinite(ref_t) & np.isfinite(ref_hr)
    ref_t = ref_t[finite]
    ref_hr = ref_hr[finite]
    if ref_t.size < 2 or np.any(np.diff(ref_t) <= 0.0):
        raise ValueError("paired_invalid_reference_timeline")
    return np.interp(times_s, ref_t, ref_hr, left=np.nan, right=np.nan)


def _window_key_sha256(keys: Sequence[tuple[int, float]]) -> str:
    return _semantic_sha256([[int(index), float(center)] for index, center in keys])


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("paired_empty_mean")
    return float(sum(values) / len(values))
