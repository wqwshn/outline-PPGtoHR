"""Reveal route-native ACC holdout results after the 48-fold freeze."""

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
    reveal_acc_holdouts,
    write_dict_csv,
    write_json,
)
from ppg_hr.v2.cross_subject_acc_source import load_acc_p0_snapshot
from ppg_hr.v2.cross_subject_loso_source import EXPECTED_FOLD_COUNT, EXPECTED_RECORD_COUNT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--parent-experiment-root", type=Path)
    parser.add_argument("--p0-root", type=Path)
    parser.add_argument("--p2-root", type=Path)
    parser.add_argument("--selection-root", type=Path)
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
    selection_root = (
        args.selection_root.resolve()
        if args.selection_root
        else experiment_root / "p3" / "selections"
    )
    output_root = args.output_root.resolve() if args.output_root else experiment_root / "p3"
    snapshot = load_acc_p0_snapshot(p0_root, parent_experiment_root=parent_root)
    p2_receipt = json.loads((p2_root / "p2_receipt.json").read_text(encoding="utf-8"))
    cell_path = p2_root / "acc_cell_metrics.csv"
    if (
        p2_receipt.get("status") != "pass"
        or p2_receipt.get("complete_cell_count") != EXPECTED_ACC_CALL_COUNT
        or _file_sha256(cell_path) != p2_receipt.get("canonical_csv_sha256")
    ):
        raise ValueError("acc_p2_not_sealed")
    cells = load_acc_compact_cell_csv(cell_path)
    record_rows, fold_rows = reveal_acc_holdouts(
        snapshot.parent.folds,
        cells,
        selection_root,
        expected_fold_count=EXPECTED_FOLD_COUNT,
    )
    if len(record_rows) != EXPECTED_RECORD_COUNT or len(fold_rows) != EXPECTED_FOLD_COUNT:
        raise ValueError("acc_p3_reveal_count_mismatch")
    record_sha = write_dict_csv(output_root / "holdout_record_results.csv", record_rows)
    fold_sha = write_dict_csv(output_root / "fold_results.csv", fold_rows)
    receipt = {
        "schema_id": "cross_subject_multirecord_acc_p3_reveal_receipt_v1",
        "experiment_id": ACC_EXPERIMENT_ID,
        "stage": "P3",
        "status": "pass",
        "p0_receipt_sha256": _file_sha256(p0_root / "p0_receipt.json"),
        "p2_receipt_sha256": _file_sha256(p2_root / "p2_receipt.json"),
        "freeze_receipt_sha256": _file_sha256(selection_root / "p3_freeze_receipt.json"),
        "record_result_count": len(record_rows),
        "fold_result_count": len(fold_rows),
        "holdout_record_results_sha256": record_sha,
        "fold_results_sha256": fold_sha,
    }
    write_json(output_root / "p3_reveal_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
