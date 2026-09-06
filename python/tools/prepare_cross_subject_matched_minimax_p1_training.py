"""Write 48 HF-only training files for matched minimax selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppg_hr.v2.cross_subject_acc_source import load_acc_p0_snapshot
from ppg_hr.v2.cross_subject_loso_selection import load_compact_cell_csv
from ppg_hr.v2.cross_subject_matched_minimax import (
    EXPERIMENT_ID,
    PARENT_ACC_EXPERIMENT_ID,
    PARENT_HF_EXPERIMENT_ID,
    SELECTION_RULE_ID,
    RouteSelectionCell,
    file_sha256,
    write_training_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--experiment-root", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    experiment_root = (
        args.experiment_root.resolve()
        if args.experiment_root
        else repo_root / "data" / "experiments" / EXPERIMENT_ID
    )
    hf_root = repo_root / "data" / "experiments" / PARENT_HF_EXPERIMENT_ID
    acc_root = repo_root / "data" / "experiments" / PARENT_ACC_EXPERIMENT_ID
    p0_path = experiment_root / "p0" / "p0_receipt.json"
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    if p0.get("status") != "pass":
        raise ValueError("matched_training_p0_not_pass")
    snapshot = load_acc_p0_snapshot(acc_root / "p0", parent_experiment_root=hf_root)
    compact = load_compact_cell_csv(hf_root / "p2" / "hf_cell_metrics.csv")
    cells = tuple(
        RouteSelectionCell(
            subject_id=cell.physical_subject_id,
            record_id=cell.record_id,
            coordinate_id=cell.coordinate_id,
            coordinate_index=cell.coordinate_index,
            mae_bpm=cell.candidate_mae_bpm,
        )
        for cell in compact
    )
    receipt = write_training_inputs(
        snapshot.parent.folds,
        cells,
        experiment_root / "p1" / "training",
        identity={
            "selection_rule_id": SELECTION_RULE_ID,
            "route_id": "HF",
            "dataset_sha256": snapshot.parent.dataset_sha256,
            "fold_manifest_sha256": snapshot.parent.fold_manifest_sha256,
            "coordinate_order_sha256": snapshot.parent.coordinate_order_sha256,
            "hf_cell_metrics_sha256": file_sha256(hf_root / "p2" / "hf_cell_metrics.csv"),
            "p0_receipt_sha256": file_sha256(p0_path),
        },
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
