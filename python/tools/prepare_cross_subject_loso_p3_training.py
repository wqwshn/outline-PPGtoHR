"""Materialise holdout-free training inputs from the sealed P2 ledger export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ppg_hr.v2.cross_subject_loso_selection import (
    load_compact_cell_csv,
    write_training_inputs,
)
from ppg_hr.v2.cross_subject_loso_source import (
    EXPECTED_HF_CALL_COUNT,
    EXPERIMENT_ID,
    load_p0_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--p0-root", type=Path)
    parser.add_argument("--p2-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    experiment_root = repo_root / "data" / "experiments" / EXPERIMENT_ID
    p0_root = args.p0_root.resolve() if args.p0_root else experiment_root / "p0"
    p2_root = args.p2_root.resolve() if args.p2_root else experiment_root / "p2"
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else experiment_root / "p3" / "training_inputs"
    )
    snapshot = load_p0_snapshot(p0_root)
    p2_receipt_path = p2_root / "p2_receipt.json"
    p2_receipt = json.loads(p2_receipt_path.read_text(encoding="utf-8"))
    csv_path = p2_root / "hf_cell_metrics.csv"
    _validate_p2_receipt(p2_receipt, snapshot, csv_path)
    cells = load_compact_cell_csv(csv_path)
    if len(cells) != EXPECTED_HF_CALL_COUNT:
        raise ValueError(f"p2_cell_count:{len(cells)}")
    receipt = write_training_inputs(
        snapshot.folds,
        cells,
        output_root,
        identity={
            "experiment_id": snapshot.experiment_id,
            "p0_receipt_sha256": _file_sha256(p0_root / "p0_receipt.json"),
            "p2_receipt_sha256": _file_sha256(p2_receipt_path),
            "p2_canonical_csv_sha256": _file_sha256(csv_path),
            "dataset_sha256": snapshot.dataset_sha256,
            "coordinate_order_sha256": snapshot.coordinate_order_sha256,
            "fold_manifest_sha256": snapshot.fold_manifest_sha256,
            "metric_contract_sha256": snapshot.metric_contract_sha256,
        },
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _validate_p2_receipt(receipt: dict, snapshot: object, csv_path: Path) -> None:
    expected = {
        "status": "pass",
        "complete_cell_count": EXPECTED_HF_CALL_COUNT,
        "dataset_sha256": snapshot.dataset_sha256,
        "coordinate_order_sha256": snapshot.coordinate_order_sha256,
        "call_manifest_sha256": snapshot.call_manifest_sha256,
        "algorithm_sha256": snapshot.algorithm_sha256,
        "metric_contract_sha256": snapshot.metric_contract_sha256,
        "canonical_csv_sha256": _file_sha256(csv_path),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"p2_receipt_mismatch:{key}")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
