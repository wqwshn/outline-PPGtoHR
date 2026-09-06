"""Bind the matched-minimax follow-up to sealed HF and ACC parent evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppg_hr.v2.cross_subject_acc_source import load_acc_p0_snapshot
from ppg_hr.v2.cross_subject_matched_minimax import (
    EXPERIMENT_ID,
    PARENT_ACC_EXPERIMENT_ID,
    PARENT_HF_EXPERIMENT_ID,
    SELECTION_RULE_ID,
    file_sha256,
    write_json,
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
    snapshot = load_acc_p0_snapshot(acc_root / "p0", parent_experiment_root=hf_root)
    contract_path = (
        repo_root
        / "docs"
        / "contracts"
        / "acceptance"
        / "cross_subject_multirecord_hf_acc_matched_minimax_v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = {
        "expected_record_count": len(snapshot.parent.records),
        "expected_physical_subject_count": len(
            {record.physical_subject_id for record in snapshot.parent.records}
        ),
        "expected_scene_count": len({record.scene for record in snapshot.parent.records}),
        "expected_fold_count": len(snapshot.parent.folds),
        "expected_coordinate_count": len(snapshot.parent.coordinates),
    }
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("matched_p0_contract_experiment_id")
    for key, value in expected.items():
        if int(contract.get(key, -1)) != value:
            raise ValueError(f"matched_p0_contract_count:{key}")
    receipt_paths = {
        "hf_p0_receipt.json": hf_root / "p0" / "p0_receipt.json",
        "hf_p2_receipt.json": hf_root / "p2" / "p2_receipt.json",
        "hf_p3_freeze_receipt.json": hf_root / "p3" / "selections" / "p3_freeze_receipt.json",
        "hf_p4_receipt.json": hf_root / "p4" / "p4_receipt.json",
        "acc_p0_receipt.json": acc_root / "p0" / "p0_receipt.json",
        "acc_p2_receipt.json": acc_root / "p2" / "p2_receipt.json",
        "acc_p3_freeze_receipt.json": acc_root / "p3" / "selections" / "p3_freeze_receipt.json",
        "acc_p4_receipt.json": acc_root / "p4" / "p4_receipt.json",
        "acc_p5_completion_receipt.json": acc_root / "p5" / "p5_completion_receipt.json",
    }
    for name, path in receipt_paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            raise ValueError(f"matched_p0_parent_not_pass:{name}")
    hf_p2 = json.loads(receipt_paths["hf_p2_receipt.json"].read_text(encoding="utf-8"))
    acc_p2 = json.loads(receipt_paths["acc_p2_receipt.json"].read_text(encoding="utf-8"))
    if hf_p2.get("complete_cell_count") != 42_900:
        raise ValueError("matched_p0_hf_surface_incomplete")
    if acc_p2.get("complete_cell_count") != 42_900:
        raise ValueError("matched_p0_acc_surface_incomplete")

    p0_root = experiment_root / "p0"
    binding = {
        "schema_id": "cross_subject_matched_minimax_parent_binding_v1",
        "experiment_id": EXPERIMENT_ID,
        "selection_rule_id": SELECTION_RULE_ID,
        "dataset_sha256": snapshot.parent.dataset_sha256,
        "fold_manifest_sha256": snapshot.parent.fold_manifest_sha256,
        "coordinate_order_sha256": snapshot.parent.coordinate_order_sha256,
        "algorithm_sha256": snapshot.parent.algorithm_sha256,
        "parent_receipt_sha256": {name: file_sha256(path) for name, path in receipt_paths.items()},
        "hf_cell_metrics_sha256": file_sha256(hf_root / "p2" / "hf_cell_metrics.csv"),
        "acc_cell_metrics_sha256": file_sha256(acc_root / "p2" / "acc_cell_metrics.csv"),
        "acceptance_contract_sha256": file_sha256(contract_path),
    }
    binding_sha = write_json(p0_root / "parent_binding.json", binding)
    receipt = {
        "schema_id": "cross_subject_matched_minimax_p0_receipt_v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": "P0",
        "status": "pass",
        "record_count": expected["expected_record_count"],
        "physical_subject_count": expected["expected_physical_subject_count"],
        "scene_count": expected["expected_scene_count"],
        "fold_count": expected["expected_fold_count"],
        "coordinate_count": expected["expected_coordinate_count"],
        "hf_response_cell_count": hf_p2["complete_cell_count"],
        "acc_response_cell_count": acc_p2["complete_cell_count"],
        "full_response_recalculation_count": 0,
        "parent_binding_sha256": binding_sha,
        "acceptance_contract_sha256": binding["acceptance_contract_sha256"],
    }
    sha = write_json(p0_root / "p0_receipt.json", receipt)
    print(json.dumps({**receipt, "receipt_sha256": sha}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
