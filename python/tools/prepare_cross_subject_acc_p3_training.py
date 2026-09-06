"""Materialise holdout-free training inputs for all 48 ACC folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ppg_hr.v2.cross_subject_acc_experiment import (
    ACC_EXPERIMENT_ID,
    EXPECTED_ACC_CALL_COUNT,
    PARENT_EXPERIMENT_ID,
)
from ppg_hr.v2.cross_subject_acc_selection import (
    load_acc_compact_cell_csv,
    write_acc_training_inputs,
)
from ppg_hr.v2.cross_subject_acc_source import load_acc_p0_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--parent-experiment-root", type=Path)
    parser.add_argument("--p0-root", type=Path)
    parser.add_argument("--p2-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    experiment_root = repo_root / "data" / "experiments" / ACC_EXPERIMENT_ID
    parent_root = (
        args.parent_experiment_root.resolve()
        if args.parent_experiment_root
        else repo_root / "data" / "experiments" / PARENT_EXPERIMENT_ID
    )
    p0_root = args.p0_root.resolve() if args.p0_root else experiment_root / "p0"
    p2_root = args.p2_root.resolve() if args.p2_root else experiment_root / "p2"
    output_root = (
        args.output_root.resolve() if args.output_root else experiment_root / "p3" / "training"
    )
    snapshot = load_acc_p0_snapshot(p0_root, parent_experiment_root=parent_root)
    receipt = json.loads((p2_root / "p2_receipt.json").read_text(encoding="utf-8"))
    cell_path = p2_root / "acc_cell_metrics.csv"
    if (
        receipt.get("status") != "pass"
        or receipt.get("complete_cell_count") != EXPECTED_ACC_CALL_COUNT
        or _file_sha256(cell_path) != receipt.get("canonical_csv_sha256")
    ):
        raise ValueError("acc_p2_not_sealed")
    cells = load_acc_compact_cell_csv(cell_path)
    if len(cells) != EXPECTED_ACC_CALL_COUNT:
        raise ValueError(f"acc_p2_cell_count:{len(cells)}")
    manifest = write_acc_training_inputs(
        snapshot.parent.folds,
        cells,
        output_root,
        identity={
            "experiment_id": ACC_EXPERIMENT_ID,
            "p0_receipt_sha256": _file_sha256(p0_root / "p0_receipt.json"),
            "p2_receipt_sha256": _file_sha256(p2_root / "p2_receipt.json"),
            "p2_cell_metrics_sha256": _file_sha256(cell_path),
            "dataset_sha256": snapshot.parent.dataset_sha256,
            "coordinate_order_sha256": snapshot.parent.coordinate_order_sha256,
            "fold_manifest_sha256": snapshot.parent.fold_manifest_sha256,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
