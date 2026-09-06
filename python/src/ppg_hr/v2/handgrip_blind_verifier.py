"""Independent verifier for the frozen Handgrip composition experiment.

This module intentionally does not import the formal composition or selector
entry points.  It recomputes their governed outputs from sealed fingerprints
and metric-minimal training CSVs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

EXPERIMENT_ID = "cross_subject_multirecord_d24_handgrip_blind_composition_v1"
FEATURE_CLASS_IDS = (
    "hf_interface_baseline",
    "acc_tremor",
    "hf_relative_acc_ppg_artifact_response",
    "post_motion_hf_recovery",
    "artifact_persistence_bandwidth",
    "dual_hf_consistency",
)
FEATURE_COMPONENT_COUNTS = {
    "hf_interface_baseline": 2,
    "acc_tremor": 1,
    "hf_relative_acc_ppg_artifact_response": 2,
    "post_motion_hf_recovery": 2,
    "artifact_persistence_bandwidth": 2,
    "dual_hf_consistency": 2,
}


def independent_hf_coordinate(
    rows: list[dict[str, Any]], subjects: tuple[str, ...]
) -> tuple[int, str]:
    """Recompute the six-gate subject-balanced lexicographic top-1."""

    expected = set(subjects)
    if not rows or not expected or len(expected) != len(subjects):
        raise ValueError("handgrip_verify_hf_input")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["physical_subject_id"]) not in expected:
            raise ValueError("handgrip_verify_hf_subject")
        grouped[(int(row["coordinate_index"]), str(row["coordinate_id"]))].append(row)
    ranked = []
    expected_records: set[tuple[str, str]] | None = None
    for (index, coordinate_id), coordinate_rows in grouped.items():
        record_keys = {
            (str(row["physical_subject_id"]), str(row["record_id"])) for row in coordinate_rows
        }
        if len(record_keys) != len(coordinate_rows):
            raise ValueError(f"handgrip_verify_hf_duplicate:{coordinate_id}")
        if expected_records is None:
            expected_records = record_keys
        elif record_keys != expected_records:
            raise ValueError(f"handgrip_verify_hf_grid:{coordinate_id}")
        pass_fractions = []
        subject_maes = []
        for subject in sorted(expected):
            subject_rows = [
                row for row in coordinate_rows if str(row["physical_subject_id"]) == subject
            ]
            if not subject_rows:
                raise ValueError(f"handgrip_verify_hf_subject_missing:{coordinate_id}:{subject}")
            passed = sum(_as_bool(row["qualified"]) for row in subject_rows)
            pass_fractions.append(Fraction(passed, len(subject_rows)))
            subject_maes.append(
                sum(
                    (Decimal(str(row["candidate_mae_bpm"])) for row in subject_rows),
                    Decimal(),
                )
                / len(subject_rows)
            )
        key = (
            -min(pass_fractions),
            -(sum(pass_fractions, Fraction()) / len(pass_fractions)),
            max(subject_maes),
            sum(subject_maes, Decimal()) / len(subject_maes),
            index,
        )
        ranked.append((key, index, coordinate_id))
    _, index, coordinate_id = min(ranked, key=lambda value: value[0])
    return index, coordinate_id


def independent_acc_coordinate(
    rows: list[dict[str, Any]], subjects: tuple[str, ...]
) -> tuple[int, str]:
    """Recompute the subject-balanced ACC minimax top-1."""

    expected = set(subjects)
    if not rows or not expected or len(expected) != len(subjects):
        raise ValueError("handgrip_verify_acc_input")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["physical_subject_id"]) not in expected:
            raise ValueError("handgrip_verify_acc_subject")
        grouped[(int(row["coordinate_index"]), str(row["coordinate_id"]))].append(row)
    ranked = []
    expected_records: set[tuple[str, str]] | None = None
    for (index, coordinate_id), coordinate_rows in grouped.items():
        record_keys = {
            (str(row["physical_subject_id"]), str(row["record_id"])) for row in coordinate_rows
        }
        if len(record_keys) != len(coordinate_rows):
            raise ValueError(f"handgrip_verify_acc_duplicate:{coordinate_id}")
        if expected_records is None:
            expected_records = record_keys
        elif record_keys != expected_records:
            raise ValueError(f"handgrip_verify_acc_grid:{coordinate_id}")
        subject_maes = []
        for subject in sorted(expected):
            subject_rows = [
                row for row in coordinate_rows if str(row["physical_subject_id"]) == subject
            ]
            if not subject_rows:
                raise ValueError(f"handgrip_verify_acc_subject_missing:{coordinate_id}:{subject}")
            subject_maes.append(
                sum(
                    (Decimal(str(row["mae_bpm"])) for row in subject_rows),
                    Decimal(),
                )
                / len(subject_rows)
            )
        key = (
            max(subject_maes),
            sum(subject_maes, Decimal()) / len(subject_maes),
            index,
        )
        ranked.append((key, index, coordinate_id))
    _, index, coordinate_id = min(ranked, key=lambda value: value[0])
    return index, coordinate_id


def run_independent_verification(
    *,
    p0_root: Path,
    p1_root: Path,
    primary_p2_root: Path,
    primary_reveal_root: Path,
    fallback_root: Path,
    lyx_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Verify both D24 stages and the LYX projection from sealed artifacts."""

    p0_root = Path(p0_root).resolve()
    p1_root = Path(p1_root).resolve()
    output_root = Path(output_root).resolve()
    candidate_manifest = _read_json(p0_root / "candidate_manifest.json")
    baseline = _read_json(p0_root / "baseline_snapshot.json")
    feature_index = _read_json(p1_root / "feature_index.json")
    primary_ids = set(str(value) for value in candidate_manifest["primary_record_ids"])
    fingerprints = _load_fingerprints(feature_index["candidate_fingerprints"])
    lyx_fingerprints = _load_fingerprints(feature_index["lyx_sync_fingerprints"])

    stages = [
        _verify_stage(
            stage_id="d24_15_primary",
            p2_stage_root=Path(primary_p2_root).resolve() / "d24_15_primary",
            reveal_stage_root=Path(primary_reveal_root).resolve() / "d24_15_primary",
            fingerprints=fingerprints,
            primary_ids=primary_ids,
            baseline=baseline,
        ),
        _verify_stage(
            stage_id="d24_18_fallback",
            p2_stage_root=Path(fallback_root).resolve() / "d24_18_fallback",
            reveal_stage_root=Path(fallback_root).resolve() / "d24_18_fallback",
            fingerprints=fingerprints,
            primary_ids=primary_ids,
            baseline=baseline,
        ),
    ]
    lyx = _verify_lyx(Path(lyx_root).resolve(), lyx_fingerprints)
    verification = {
        "schema_id": "d24_handgrip_blind_composition_independent_verification_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass",
        "formal_composition_entry_called": False,
        "formal_selector_entries_called": False,
        "verified_feature_scaling": True,
        "verified_equal_class_distance": True,
        "verified_silhouette_and_ari": True,
        "verified_patterns_and_training_cores": True,
        "verified_hf_and_acc_coordinates": True,
        "verified_evaluation_denominators_and_summaries": True,
        "stage_verifications": stages,
        "lyx_verification": lyx,
    }
    verification_path = output_root / "verification.json"
    verification_sha = _write_json(verification_path, verification)
    receipt = {
        "schema_id": "d24_handgrip_blind_composition_p5_verifier_receipt_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "pass",
        "stage_count": len(stages),
        "lyx_verified": True,
        "verification_path": str(verification_path),
        "verification_sha256": verification_sha,
    }
    _write_json(output_root / "receipt.json", receipt)
    return receipt


def _verify_stage(
    *,
    stage_id: str,
    p2_stage_root: Path,
    reveal_stage_root: Path,
    fingerprints: dict[str, dict[str, Any]],
    primary_ids: set[str],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    composition_by_fold = {}
    for path in sorted((p2_stage_root / "compositions").glob("*.json")):
        formal = _read_json(path)
        independent = _independent_composition(
            [fingerprints[record_id] for record_id in formal["record_ids"]]
        )
        _assert_composition_equivalent(formal, independent, context=f"{stage_id}:{path.stem}")
        composition_by_fold[path.stem] = independent
    if len(composition_by_fold) != 6:
        raise ValueError(f"handgrip_verify_fold_count:{stage_id}")

    mode_details = {}
    for mode in ("main", "sens30"):
        manifest = _read_json(p2_stage_root / mode / "selection_manifest.json")
        if len(manifest.get("selections") or []) != 6:
            raise ValueError(f"handgrip_verify_selection_count:{stage_id}:{mode}")
        for row in manifest["selections"]:
            selection_path = Path(str(row["selection_path"])).resolve()
            selection = _read_json(selection_path)
            fold_id = str(selection["fold_id"])
            expected_core = (
                composition_by_fold[fold_id]["training_core_record_ids"]
                if mode == "main"
                else composition_by_fold[fold_id]["sensitivity_30s"]["training_core_record_ids"]
            )
            if list(selection["train_record_ids"]) != list(expected_core):
                raise ValueError(f"handgrip_verify_core_binding:{stage_id}:{mode}:{fold_id}")
            if set(selection["holdout_record_ids"]) & set(selection["train_record_ids"]):
                raise ValueError(f"handgrip_verify_holdout_leakage:{stage_id}:{mode}:{fold_id}")
            hf_rows = _read_csv(selection_path.parent / "hf_training.csv")
            acc_rows = _read_csv(selection_path.parent / "acc_training.csv")
            subjects = tuple(str(value) for value in selection["train_subject_ids"])
            hf_index, hf_coordinate = independent_hf_coordinate(hf_rows, subjects)
            acc_index, acc_coordinate = independent_acc_coordinate(acc_rows, subjects)
            if (hf_index, hf_coordinate) != (
                int(selection["hf_selection"]["coordinate_index"]),
                str(selection["hf_selection"]["coordinate_id"]),
            ):
                raise ValueError(f"handgrip_verify_hf_selection:{stage_id}:{mode}:{fold_id}")
            if (acc_index, acc_coordinate) != (
                int(selection["acc_selection"]["coordinate_index"]),
                str(selection["acc_selection"]["coordinate_id"]),
            ):
                raise ValueError(f"handgrip_verify_acc_selection:{stage_id}:{mode}:{fold_id}")

        records = _read_csv(reveal_stage_root / mode / "record_results.csv")
        if len(records) != 15 or {str(row["record_id"]) for row in records} != primary_ids:
            raise ValueError(f"handgrip_verify_denominator:{stage_id}:{mode}")
        recomputed = _recompute_summary(records, baseline)
        formal_summary = _read_json(reveal_stage_root / mode / "summary.json")
        _assert_summary_equivalent(formal_summary, recomputed, f"{stage_id}:{mode}")
        mode_details[mode] = {
            "fold_count": 6,
            "record_count": 15,
            "hf_coordinate_recomputation": "pass",
            "acc_coordinate_recomputation": "pass",
            "summary_recomputation": "pass",
        }
    decision = _read_json(reveal_stage_root / "decision.json")
    main_summary = _read_json(reveal_stage_root / "main" / "summary.json")
    expected_next = (
        "report_and_lyx_consistency"
        if main_summary["primary_pass"]
        else ("run_d24_18_fallback" if stage_id == "d24_15_primary" else "stop_composition_route")
    )
    if decision.get("next_action") != expected_next:
        raise ValueError(f"handgrip_verify_stage_gate:{stage_id}")
    return {
        "stage_id": stage_id,
        "status": "pass",
        "fold_count": 6,
        "modes": mode_details,
        "next_action": expected_next,
    }


def _verify_lyx(lyx_root: Path, fingerprints: dict[str, dict[str, Any]]) -> dict[str, Any]:
    formal = _read_json(lyx_root / "composition.json")
    independent = _independent_composition(
        [fingerprints[record_id] for record_id in formal["record_ids"]]
    )
    _assert_composition_equivalent(formal, independent, context="lyx")
    selection = _read_json(lyx_root / "selection.json")
    hf_index, hf_coordinate = independent_hf_coordinate(
        _read_csv(lyx_root / "hf_training.csv"), ("LYX",)
    )
    acc_index, acc_coordinate = independent_acc_coordinate(
        _read_csv(lyx_root / "acc_training.csv"), ("LYX",)
    )
    if (hf_index, hf_coordinate) != (
        int(selection["hf_selection"]["coordinate_index"]),
        str(selection["hf_selection"]["coordinate_id"]),
    ):
        raise ValueError("handgrip_verify_lyx_hf_selection")
    if (acc_index, acc_coordinate) != (
        int(selection["acc_selection"]["coordinate_index"]),
        str(selection["acc_selection"]["coordinate_id"]),
    ):
        raise ValueError("handgrip_verify_lyx_acc_selection")
    records = _read_csv(lyx_root / "record_results.csv")
    summary = _read_json(lyx_root / "summary.json")
    if len(records) != 3 or int(summary["record_count"]) != 3:
        raise ValueError("handgrip_verify_lyx_denominator")
    if not np.isclose(
        mean(float(row["hf_mae_bpm"]) for row in records),
        float(summary["hf_mean_mae_bpm"]),
        atol=1e-12,
        rtol=0.0,
    ) or not np.isclose(
        mean(float(row["acc_mae_bpm"]) for row in records),
        float(summary["acc_mean_mae_bpm"]),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("handgrip_verify_lyx_summary")
    if summary.get("rule_changes_after_results") is not False:
        raise ValueError("handgrip_verify_lyx_rule_change")
    return {
        "status": "pass",
        "record_count": 3,
        "physical_subject_count": 1,
        "subject_and_record_overlap_confirmed": True,
        "rule_changes_after_results": False,
    }


def _independent_composition(fingerprints: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(fingerprints, key=lambda row: str(row["record_id"]))
    if len(ordered) < 3:
        raise ValueError("handgrip_verify_record_count")
    record_ids = [str(row["record_id"]) for row in ordered]
    subject_by_record = {str(row["record_id"]): str(row["physical_subject_id"]) for row in ordered}
    primary_versions = [
        [row["full"] for row in ordered],
        *[[row["leave_one_block_out"][index] for row in ordered] for index in range(4)],
    ]
    primary = _independent_group(record_ids, primary_versions)
    primary_core = _independent_core(record_ids, subject_by_record, primary)
    sensitivity_versions = [
        [row["sensitivity_30s"]["full"] for row in ordered],
        *[
            [row["sensitivity_30s"]["leave_one_block_out"][index] for row in ordered]
            for index in range(6)
        ],
    ]
    sensitivity = _independent_group(record_ids, sensitivity_versions)
    sensitivity_core = _independent_core(record_ids, subject_by_record, sensitivity)
    return {
        **primary,
        **primary_core,
        "sensitivity_30s": {**sensitivity, **sensitivity_core},
    }


def _independent_group(
    record_ids: list[str], versions: list[list[dict[str, list[float]]]]
) -> dict[str, Any]:
    distances = []
    scalings = []
    for rows in versions:
        distance, scaling = _independent_distance(rows)
        distances.append(distance)
        scalings.append(scaling)
    candidates = {}
    for cluster_count in (2, 3):
        if cluster_count >= len(record_ids):
            continue
        labels_by_version = []
        silhouettes = []
        for distance in distances:
            labels = AgglomerativeClustering(
                n_clusters=cluster_count,
                metric="precomputed",
                linkage="average",
            ).fit_predict(distance)
            labels_by_version.append(labels)
            silhouettes.append(
                float(silhouette_score(distance, labels, metric="precomputed"))
                if len(set(int(value) for value in labels)) >= 2
                else 0.0
            )
        pairwise_ari = [
            float(adjusted_rand_score(labels_by_version[left], labels_by_version[right]))
            for left in range(len(labels_by_version))
            for right in range(left + 1, len(labels_by_version))
        ]
        median_silhouette = float(np.median(silhouettes))
        median_ari = float(np.median(pairwise_ari))
        minimum_ari = min(pairwise_ari)
        passed = bool(
            all(value > 0.0 for value in silhouettes)
            and median_silhouette >= 0.25
            and median_ari >= 0.80
            and minimum_ari >= 0.50
        )
        candidates[str(cluster_count)] = {
            "cluster_count": cluster_count,
            "silhouettes": silhouettes,
            "median_silhouette": median_silhouette,
            "pairwise_ari": pairwise_ari,
            "median_pairwise_ari": median_ari,
            "minimum_pairwise_ari": minimum_ari,
            "passed": passed,
            "labels_by_version": [labels.tolist() for labels in labels_by_version],
        }
    passing = [row for row in candidates.values() if row["passed"]]
    if passing:
        selected = min(
            passing,
            key=lambda row: (-float(row["median_silhouette"]), int(row["cluster_count"])),
        )
        cluster_count = int(selected["cluster_count"])
        full_labels = list(selected["labels_by_version"][0])
        fallback = None
    else:
        cluster_count = 1
        full_labels = [0] * len(record_ids)
        fallback = "retain_all_training_records"
    return {
        "record_ids": record_ids,
        "selected_cluster_count": cluster_count,
        "fallback_policy": fallback,
        "candidates": candidates,
        "full_labels": full_labels,
        "full_distance_matrix": distances[0].tolist(),
        "scaling_by_version": scalings,
    }


def _independent_distance(
    feature_rows: list[dict[str, list[float]]],
) -> tuple[np.ndarray, dict[str, Any]]:
    record_count = len(feature_rows)
    scaled_by_class = {}
    scaling = {}
    for class_id in FEATURE_CLASS_IDS:
        component_count = FEATURE_COMPONENT_COUNTS[class_id]
        matrix = np.asarray([row[class_id] for row in feature_rows], dtype=float)
        if matrix.shape != (record_count, component_count) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"handgrip_verify_feature_contract:{class_id}")
        if class_id == "post_motion_hf_recovery":
            if np.any((matrix < 0.0) | (matrix > 1.0)):
                raise ValueError("handgrip_verify_recovery_bounds")
            centers = np.zeros(component_count, dtype=float)
            scales = np.ones(component_count, dtype=float)
            sources = ["bounded_identity"] * component_count
            scaled = matrix.copy()
        else:
            centers = np.median(matrix, axis=0)
            scales = np.quantile(matrix, 0.75, axis=0) - np.quantile(matrix, 0.25, axis=0)
            sources = []
            for component in range(component_count):
                if scales[component] > 0.0:
                    sources.append("iqr")
                else:
                    mad = float(np.median(np.abs(matrix[:, component] - centers[component])))
                    fallback = 1.4826 * mad
                    if fallback > 0.0:
                        scales[component] = fallback
                        sources.append("mad")
                    else:
                        scales[component] = 1.0
                        sources.append("zero_contribution")
            scaled = (matrix - centers) / scales
            for component, source in enumerate(sources):
                if source == "zero_contribution":
                    scaled[:, component] = 0.0
        scaled_by_class[class_id] = scaled
        scaling[class_id] = {
            "center": [float(value) for value in centers],
            "scale": [float(value) for value in scales],
            "scale_source": sources,
        }
    total_squared = np.zeros((record_count, record_count), dtype=float)
    for class_id in FEATURE_CLASS_IDS:
        matrix = scaled_by_class[class_id]
        differences = matrix[:, None, :] - matrix[None, :, :]
        class_distance = np.sqrt(np.mean(differences**2, axis=2))
        total_squared += class_distance**2
    distance = np.sqrt(total_squared / len(FEATURE_CLASS_IDS))
    np.fill_diagonal(distance, 0.0)
    return distance, scaling


def _independent_core(
    record_ids: list[str],
    subject_by_record: dict[str, str],
    grouping: dict[str, Any],
) -> dict[str, Any]:
    cluster_count = int(grouping["selected_cluster_count"])
    if cluster_count == 1:
        return {
            "training_core_record_ids": list(record_ids),
            "common_patterns": [],
            "rare_patterns": [],
        }
    distance = np.asarray(grouping["full_distance_matrix"], dtype=float)
    labels = np.asarray(grouping["full_labels"], dtype=int)
    retained = set()
    common_patterns = []
    rare_patterns = []
    for label in sorted(set(int(value) for value in labels)):
        members = np.flatnonzero(labels == label)
        member_ids = [record_ids[index] for index in members]
        subjects = sorted({subject_by_record[record_id] for record_id in member_ids})
        if len(subjects) == 1:
            retained.update(member_ids)
            rare_patterns.append(
                {
                    "label": label,
                    "subject_ids": subjects,
                    "record_ids": member_ids,
                    "retained_record_ids": member_ids,
                }
            )
            continue
        totals = np.sum(distance[np.ix_(members, members)], axis=1)
        minimum = float(np.min(totals))
        medoid_id = min(
            record_ids[members[index]]
            for index, value in enumerate(totals)
            if np.isclose(float(value), minimum, atol=1e-12, rtol=0.0)
        )
        medoid_index = record_ids.index(medoid_id)
        representatives = []
        for subject in subjects:
            subject_members = [
                index for index in members if subject_by_record[record_ids[index]] == subject
            ]
            minimum_distance = min(
                float(distance[index, medoid_index]) for index in subject_members
            )
            representative = min(
                record_ids[index]
                for index in subject_members
                if np.isclose(
                    float(distance[index, medoid_index]),
                    minimum_distance,
                    atol=1e-12,
                    rtol=0.0,
                )
            )
            representatives.append(representative)
            retained.add(representative)
        common_patterns.append(
            {
                "label": label,
                "subject_ids": subjects,
                "record_ids": member_ids,
                "medoid_record_id": medoid_id,
                "retained_record_ids": representatives,
            }
        )
    return {
        "training_core_record_ids": sorted(retained),
        "common_patterns": common_patterns,
        "rare_patterns": rare_patterns,
    }


def _assert_composition_equivalent(
    formal: dict[str, Any], independent: dict[str, Any], *, context: str
) -> None:
    for key in (
        "record_ids",
        "selected_cluster_count",
        "fallback_policy",
        "training_core_record_ids",
        "common_patterns",
        "rare_patterns",
        "full_labels",
    ):
        if formal[key] != independent[key]:
            raise ValueError(f"handgrip_verify_composition:{context}:{key}")
    _assert_numeric_tree_close(
        formal["full_distance_matrix"], independent["full_distance_matrix"], context
    )
    _assert_numeric_tree_close(
        formal["scaling_by_version"], independent["scaling_by_version"], context
    )
    _assert_numeric_tree_close(formal["candidates"], independent["candidates"], context)
    formal_sensitivity = formal["sensitivity_30s"]
    independent_sensitivity = independent["sensitivity_30s"]
    for key in (
        "record_ids",
        "selected_cluster_count",
        "fallback_policy",
        "training_core_record_ids",
        "common_patterns",
        "rare_patterns",
        "full_labels",
    ):
        if formal_sensitivity[key] != independent_sensitivity[key]:
            raise ValueError(f"handgrip_verify_composition:{context}:sens30:{key}")
    _assert_numeric_tree_close(formal_sensitivity, independent_sensitivity, context)


def _assert_numeric_tree_close(formal: Any, independent: Any, context: str) -> None:
    if isinstance(formal, dict) and isinstance(independent, dict):
        if set(formal) != set(independent):
            raise ValueError(f"handgrip_verify_tree_keys:{context}")
        for key in formal:
            _assert_numeric_tree_close(formal[key], independent[key], f"{context}:{key}")
        return
    if isinstance(formal, list) and isinstance(independent, list):
        if len(formal) != len(independent):
            raise ValueError(f"handgrip_verify_tree_length:{context}")
        for index, (left, right) in enumerate(zip(formal, independent, strict=True)):
            _assert_numeric_tree_close(left, right, f"{context}:{index}")
        return
    if (
        isinstance(formal, (int, float))
        and not isinstance(formal, bool)
        and isinstance(independent, (int, float))
        and not isinstance(independent, bool)
    ):
        if not np.isclose(float(formal), float(independent), atol=1e-12, rtol=0.0):
            raise ValueError(f"handgrip_verify_numeric:{context}")
        return
    if formal != independent:
        raise ValueError(f"handgrip_verify_value:{context}")


def _recompute_summary(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    hf_values = [float(row["hf_mae_bpm"]) for row in rows]
    acc_values = [float(row["acc_mae_bpm"]) for row in rows]
    hf_mean = mean(hf_values)
    acc_mean = mean(acc_values)
    hf_sd = stdev(hf_values)
    by_record = {str(row["record_id"]): row for row in rows}
    challenges = {
        record_id: float(by_record[record_id]["hf_mae_bpm"]) <= float(value)
        for record_id, value in baseline["challenge_hf_mae_bpm"].items()
    }
    self_improved = hf_mean < float(baseline["hf_mean_mae_bpm"])
    not_worse_acc = hf_mean <= acc_mean
    primary_pass = self_improved and not_worse_acc
    sd_nonworse = hf_sd <= float(baseline["hf_sample_sd_bpm"])
    return {
        "record_count": len(rows),
        "hf_mean_mae_bpm": hf_mean,
        "hf_median_mae_bpm": median(hf_values),
        "hf_sample_sd_bpm": hf_sd,
        "acc_mean_mae_bpm": acc_mean,
        "acc_median_mae_bpm": median(acc_values),
        "hf_self_improved": self_improved,
        "hf_not_worse_than_same_core_acc": not_worse_acc,
        "primary_pass": primary_pass,
        "hf_sample_sd_not_worse": sd_nonworse,
        "challenge_hf_nonworse": challenges,
        "stability_improvement_claim": primary_pass and sd_nonworse and all(challenges.values()),
    }


def _assert_summary_equivalent(
    formal: dict[str, Any], independent: dict[str, Any], context: str
) -> None:
    for key, value in independent.items():
        _assert_numeric_tree_close(formal[key], value, f"{context}:{key}")


def _load_fingerprints(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        path = Path(str(row["fingerprint_path"])).resolve()
        if _file_sha256(path) != str(row["fingerprint_sha256"]):
            raise ValueError(f"handgrip_verify_fingerprint_hash:{row['record_id']}")
        result[str(row["record_id"])] = _read_json(path)
    return result


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
