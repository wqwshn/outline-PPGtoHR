"""Reveal native HF holdout metrics after all matched-minimax selections are frozen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np

from ppg_hr.v2.cross_subject_acc_source import load_acc_p0_snapshot
from ppg_hr.v2.cross_subject_loso_selection import load_compact_cell_csv
from ppg_hr.v2.cross_subject_matched_minimax import (
    EXPERIMENT_ID,
    PARENT_ACC_EXPERIMENT_ID,
    PARENT_HF_EXPERIMENT_ID,
    file_sha256,
    read_json,
    write_csv,
    write_json,
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
    freeze_path = root / "p1" / "selections" / "p1_freeze_receipt.json"
    freeze = read_json(freeze_path)
    selections = list(freeze.get("selections") or [])
    if freeze.get("status") != "pass" or len(selections) != 48:
        raise ValueError("matched_reveal_freeze_incomplete")
    snapshot = load_acc_p0_snapshot(acc_root / "p0", parent_experiment_root=hf_root)
    compact = load_compact_cell_csv(hf_root / "p2" / "hf_cell_metrics.csv")
    by_key = {(cell.record_id, cell.coordinate_id): cell for cell in compact}
    frozen = {}
    for row in selections:
        path = root / "p1" / "selections" / str(row["selection_file"])
        if file_sha256(path) != row["selection_sha256"]:
            raise ValueError(f"matched_reveal_selection_hash:{row['fold_id']}")
        frozen[str(row["fold_id"])] = read_json(path)
    records = []
    folds = []
    for fold in snapshot.parent.folds:
        selection = frozen[fold.fold_id]["selection"]
        coordinate_id = str(selection["coordinate_id"])
        values = []
        for record_id in fold.holdout_record_ids:
            cell = by_key[(record_id, coordinate_id)]
            values.append(float(cell.candidate_mae_bpm))
            records.append(
                {
                    "fold_id": fold.fold_id,
                    "scene": fold.scene,
                    "holdout_subject_id": fold.holdout_subject_id,
                    "record_id": record_id,
                    "selected_coordinate_id": coordinate_id,
                    "selected_coordinate_index": int(selection["coordinate_index"]),
                    "native_mae_bpm": cell.candidate_mae_bpm,
                    "native_reliable_window_count": cell.candidate_reliable_window_count,
                    "native_evaluation_window_sha256": cell.candidate_evaluation_window_sha256,
                }
            )
        folds.append(
            {
                "fold_id": fold.fold_id,
                "scene": fold.scene,
                "holdout_subject_id": fold.holdout_subject_id,
                "heldout_records": len(values),
                "selected_coordinate_id": coordinate_id,
                "selected_coordinate_index": int(selection["coordinate_index"]),
                "worst_training_subject_mean_mae_bpm": selection["worst_subject_mean_mae_bpm"],
                "mean_training_subject_mean_mae_bpm": selection["mean_subject_mean_mae_bpm"],
                "native_mean_mae_bpm": float(np.mean(values)),
                "native_median_mae_bpm": float(median(values)),
                "native_max_mae_bpm": max(values),
            }
        )
    p1_root = root / "p1"
    record_sha = write_csv(p1_root / "holdout_record_results.csv", records)
    fold_sha = write_csv(p1_root / "fold_results.csv", folds)
    receipt = {
        "schema_id": "cross_subject_matched_minimax_p1_reveal_receipt_v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": "P1_reveal",
        "status": "pass",
        "record_count": len(records),
        "fold_count": len(folds),
        "freeze_receipt_sha256": file_sha256(freeze_path),
        "holdout_record_results_sha256": record_sha,
        "fold_results_sha256": fold_sha,
    }
    sha = write_json(p1_root / "p1_reveal_receipt.json", receipt)
    print(json.dumps({**receipt, "receipt_sha256": sha}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
