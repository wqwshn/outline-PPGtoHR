"""Bind the accepted ACC experiment to the completed 143-record HF parent."""

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
from ppg_hr.v2.cross_subject_acc_source import (
    build_acc_p0_snapshot,
    write_acc_p0_snapshot,
)
from ppg_hr.v2.cross_subject_loso_identity import solver_source_sha256

CONTRACT_PATH = Path(
    "docs/contracts/acceptance/cross_subject_multirecord_acc_independent_physical4d_v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--parent-experiment-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    contract_path = repo_root / CONTRACT_PATH
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": ACC_EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "expected_acc_call_count": EXPECTED_ACC_CALL_COUNT,
        "route_id": "ACC",
        "reference_groups_order": ["ACC"],
        "time_bias_s": 5.0,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"acc_acceptance_contract_mismatch:{key}")
    parent_root = (
        args.parent_experiment_root.resolve()
        if args.parent_experiment_root
        else repo_root / "data" / "experiments" / PARENT_EXPERIMENT_ID
    )
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else repo_root / "data" / "experiments" / ACC_EXPERIMENT_ID / "p0"
    )
    algorithm_sha = solver_source_sha256(repo_root / "python" / "src" / "ppg_hr")
    snapshot = build_acc_p0_snapshot(
        parent_experiment_root=parent_root, algorithm_sha256=algorithm_sha
    )
    receipt = write_acc_p0_snapshot(
        snapshot,
        output_root,
        acceptance_contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        parent_experiment_root=parent_root,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
