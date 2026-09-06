"""Reveal selected holdout cells only after all 48 training selections are sealed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ppg_hr.v2.cross_subject_loso_selection import (
    load_compact_cell_csv,
    reveal_holdout_results,
    write_dict_csv,
)
from ppg_hr.v2.cross_subject_loso_source import (
    EXPECTED_FOLD_COUNT,
    EXPECTED_HF_CALL_COUNT,
    EXPECTED_RECORD_COUNT,
    EXPERIMENT_ID,
    load_p0_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--p0-root", type=Path)
    parser.add_argument("--p2-root", type=Path)
    parser.add_argument("--selection-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    experiment_root = repo_root / "data" / "experiments" / EXPERIMENT_ID
    p0_root = args.p0_root.resolve() if args.p0_root else experiment_root / "p0"
    p2_root = args.p2_root.resolve() if args.p2_root else experiment_root / "p2"
    p3_root = args.output_root.resolve() if args.output_root else experiment_root / "p3"
    selection_root = (
        args.selection_root.resolve() if args.selection_root else p3_root / "selections"
    )
    snapshot = load_p0_snapshot(p0_root)
    p2_receipt_path = p2_root / "p2_receipt.json"
    p2_receipt = json.loads(p2_receipt_path.read_text(encoding="utf-8"))
    if (
        p2_receipt.get("status") != "pass"
        or p2_receipt.get("complete_cell_count") != EXPECTED_HF_CALL_COUNT
    ):
        raise ValueError("p2_not_sealed")
    csv_path = p2_root / "hf_cell_metrics.csv"
    if _file_sha256(csv_path) != p2_receipt.get("canonical_csv_sha256"):
        raise ValueError("p2_csv_hash_mismatch")
    cells = load_compact_cell_csv(csv_path)
    holdout_rows, fold_rows = reveal_holdout_results(
        snapshot.folds,
        cells,
        selection_root,
        expected_fold_count=EXPECTED_FOLD_COUNT,
        record_metadata={
            record.record_id: {
                "repeat_index": record.repeat_index,
                "source_repeat_label": record.source_repeat_label,
            }
            for record in snapshot.records
        },
    )
    if len(holdout_rows) != EXPECTED_RECORD_COUNT or len(fold_rows) != EXPECTED_FOLD_COUNT:
        raise ValueError("p3_reveal_count_mismatch")
    p3_root.mkdir(parents=True, exist_ok=True)
    holdout_sha = write_dict_csv(p3_root / "holdout_record_results.csv", holdout_rows)
    fold_sha = write_dict_csv(p3_root / "fold_results.csv", fold_rows)
    receipt = {
        "schema_id": "cross_subject_multirecord_p3_reveal_receipt_v1",
        "experiment_id": snapshot.experiment_id,
        "status": "pass",
        "p0_receipt_sha256": _file_sha256(p0_root / "p0_receipt.json"),
        "p2_receipt_sha256": _file_sha256(p2_receipt_path),
        "freeze_receipt_sha256": _file_sha256(selection_root / "p3_freeze_receipt.json"),
        "record_result_count": len(holdout_rows),
        "fold_result_count": len(fold_rows),
        "holdout_record_results_sha256": holdout_sha,
        "fold_results_sha256": fold_sha,
    }
    _write_json(p3_root / "p3_reveal_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
