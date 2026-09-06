"""Verify the archived paper results, replaying selection without running the solver."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from multiperson_joint_screening import solver_result_from_payload
from multiperson_screening_contracts import select_full_mae_time_bias

from ppg_hr.v2.cross_subject_hf_optimization import (
    evaluate_panel,
    load_lyx_partition,
    load_parent_hf_cells,
)
from ppg_hr.v2.handgrip_paper_selector import (
    COORDINATE_PAIR,
    fingerprint_vector,
    fit_coordinate_selector,
    predict_coordinates,
)
from ppg_hr.v2.lyx_paper_selector import select_sparse_domain_four_tap_fallback
from ppg_hr.v2.lyx_paper_selector_core import (
    build_train_candidates,
    neighbor_map,
    read_cell_partition,
)
from ppg_hr.v2.preprocess import load_v2_reference


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def equal(actual: float, expected: float, detail: str) -> None:
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), detail)


def verify_archive(root: Path) -> int:
    entries = read_json(root / "artifact_manifest.json")["files"]
    for row in entries:
        path = root / row["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == row["sha256"], f"archive_hash:{row['path']}")
    return len(entries)


def verify_lyx(root: Path) -> dict[str, float | int]:
    rows = read_rows(root / "lyx24/results.csv")
    require(len(rows) == len({r["record_id"] for r in rows}) == 24, "lyx_record_count")
    coordinates = read_json(root / "lyx24/identity/coordinate_space.json")["coordinates"]
    physical = {
        r["coordinate_id"]: (
            r["fs_target"],
            r["memory_ms"],
            r["mu_base"],
            r["exclusion_half_width_bpm"],
        )
        for r in coordinates
    }
    order = {r["coordinate_id"]: i for i, r in enumerate(coordinates)}
    neighbors = neighbor_map(physical)
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row["record_id"])
    require(len(by_scene) == 8 and all(len(v) == 3 for v in by_scene.values()), "lyx_scenes")
    partitions = {
        r["record_id"]: read_cell_partition(root / "lyx24/partitions" / f"{r['record_id']}.csv")
        for r in rows
    }
    values = []
    for row in rows:
        record_id = row["record_id"]
        training = sorted(set(by_scene[row["scene"]]) - {record_id})
        candidates = build_train_candidates(
            [r for record in training for r in partitions[record]], training
        )
        selected = select_sparse_domain_four_tap_fallback(
            candidates, physical_by_coordinate=physical, neighbors=neighbors, coordinate_order=order
        )
        require(selected["coordinate_id"] == row["coordinate_id"], f"lyx_selection:{record_id}")
        payload = read_json(root / "lyx24/traces" / f"{record_id}_hf.json")
        reference = load_v2_reference(root / "raw" / f"{record_id}_HR_ref.csv")
        curve = select_full_mae_time_bias(solver_result_from_payload(payload), ref_data=reference)
        require(
            curve["common_window_count"] == int(row["hf_common_window_count"]),
            f"lyx_support:{record_id}",
        )
        indices = curve["common_window_indices"]
        digest = hashlib.sha256(
            json.dumps(
                indices, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        require(digest == row["hf_common_window_mask_sha256"], f"lyx_mask:{record_id}")
        selected_curve = next(
            r for r in curve["curve"] if r["bias_s"] == float(row["hf_v3_bias_s"])
        )
        value = selected_curve["common_mae_bpm"]
        equal(value, float(row["hf_v3_common_mae_bpm"]), f"lyx_mae:{record_id}")
        values.append(value)
    return {
        "record_count": len(values),
        "mean_mae_bpm": mean(values),
        "sample_sd_bpm": stdev(values),
        "selection_replays": 24,
        "trajectory_metric_replays": 24,
    }


def verify_cross_subject(root: Path) -> dict[str, float | int]:
    base = root / "cross_subject119"
    table = load_parent_hf_cells(base / "parent_hf/p2/hf_cell_metrics.csv")
    replacements = read_rows(base / "response/lyx_replacement_binding.csv")
    table = table.with_replacements(
        remove_record_ids=[r["old_record_id"] for r in replacements],
        additions=[
            load_lyx_partition(base / "response" / f"{r['new_record_id']}.csv")
            for r in replacements
        ],
    )
    panel = read_json(base / "baseline/panel/panel.json")
    original = evaluate_panel(table, retained_record_ids=panel["retained_record_ids"])
    baseline_rows = {
        r["record_id"]: r
        for r in read_rows(base / "baseline/d24_record_route_mae.csv")
        if r["route_id"] == "HF"
    }
    for fold in original.folds:
        for record_id, value in zip(fold.holdout_record_ids, fold.holdout_mae_bpm, strict=True):
            row = baseline_rows[record_id]
            require(
                fold.coordinate_id == row["selected_coordinate_id"],
                f"baseline_coordinate:{record_id}",
            )
            equal(value, float(row["mae_bpm"]), f"baseline_mae:{record_id}")
    current = {r["record_id"]: r for r in read_rows(base / "results.csv") if r["route_id"] == "HF"}
    require(len(current) == 119, "cross_subject_record_count")
    handgrip = read_rows(base / "handgrip/record_results.csv")
    subjects = {r["record_id"]: r["holdout_subject_id"] for r in handgrip}
    features = {
        record: fingerprint_vector(
            read_json(base / "handgrip_source/p1/features" / f"{record}.json")
        )
        for record in subjects
    }
    predicted = {}
    rules = {r["holdout_subject_id"]: r for r in read_json(base / "handgrip/fold_rules.json")}
    for holdout in sorted(set(subjects.values())):
        training = sorted(r for r, subject in subjects.items() if subject != holdout)
        testing = sorted(r for r, subject in subjects.items() if subject == holdout)
        require(training == rules[holdout]["train_record_ids"], f"handgrip_training:{holdout}")
        tree = fit_coordinate_selector(
            np.stack([features[r] for r in training]),
            np.stack([table.records[r].mae_bpm[list(COORDINATE_PAIR)] for r in training]),
        )
        rule = rules[holdout]["tree_rule"]
        if rule["feature_index"] is not None:
            require(
                int(tree.tree_.feature[0]) == rule["feature_index"], f"handgrip_feature:{holdout}"
            )
            equal(
                float(tree.tree_.threshold[0]), rule["threshold"], f"handgrip_threshold:{holdout}"
            )
        indices = predict_coordinates(tree, np.stack([features[r] for r in testing]))
        predicted.update(zip(testing, indices.tolist(), strict=True))
    values = []
    for record_id, row in current.items():
        if row["scene_id"] == "woli":
            index = predicted[record_id]
            require(
                index == int(row["selected_coordinate_index"]), f"handgrip_coordinate:{record_id}"
            )
            value = float(table.records[record_id].mae_bpm[index])
        else:
            baseline = baseline_rows[record_id]
            require(
                row["selected_coordinate_id"] == baseline["selected_coordinate_id"],
                f"current_coordinate:{record_id}",
            )
            value = float(baseline["mae_bpm"])
        equal(value, float(row["mae_bpm"]), f"current_mae:{record_id}")
        values.append(value)
    return {
        "record_count": len(values),
        "mean_mae_bpm": mean(values),
        "sample_sd_bpm": stdev(values),
        "baseline_loso_fold_replays": len(original.folds),
        "handgrip_fold_replays": len(rules),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    count = verify_archive(root)
    lyx = verify_lyx(root)
    cross_subject = verify_cross_subject(root)
    require(f"{lyx['mean_mae_bpm']:.3f}" == "2.076", "lyx_reported_result")
    require(f"{cross_subject['mean_mae_bpm']:.3f}" == "3.608", "cross_subject_reported_mean")
    require(f"{cross_subject['sample_sd_bpm']:.3f}" == "3.528", "cross_subject_reported_sd")
    receipt = {
        "status": "PASS",
        "verified_archive_files": count,
        "lyx24": lyx,
        "cross_subject119": cross_subject,
        "solver_invocations": 0,
    }
    (root / "verification.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
