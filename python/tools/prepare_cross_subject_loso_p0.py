"""Materialise and verify the frozen 143-record P0 source contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ppg_hr.v2.cross_subject_loso_identity import solver_source_sha256
from ppg_hr.v2.cross_subject_loso_source import (
    EXPECTED_COORDINATE_COUNT,
    EXPECTED_FOLD_COUNT,
    EXPECTED_HF_CALL_COUNT,
    EXPECTED_RECORD_COUNT,
    EXPERIMENT_ID,
    PJY_BOBI_LABELS,
    RUN_ROSTER,
    STANDARD_ROSTER,
    TIME_BIAS_S,
    build_p0_snapshot,
    write_p0_snapshot,
)

CONTRACT_PATH = Path("docs/contracts/acceptance/cross_subject_multirecord_hf_loso_v1.json")
INVENTORY_RELATIVE_ROOT = Path("data/experiments/multiperson_joint_physical4d_screening_20260819")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--non-lyx-inventory", type=Path)
    parser.add_argument("--lyx-inventory", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    contract = _load_and_validate_contract(repo_root / CONTRACT_PATH)
    inventory_root = _default_inventory_root(repo_root)
    non_lyx = (
        args.non_lyx_inventory.resolve()
        if args.non_lyx_inventory
        else inventory_root / "baseline_inventory_non_lyx.json"
    )
    lyx = (
        args.lyx_inventory.resolve()
        if args.lyx_inventory
        else inventory_root / "baseline_inventory_lyx.json"
    )
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else repo_root / "data" / "experiments" / EXPERIMENT_ID / "p0"
    )
    algorithm_sha = solver_source_sha256(repo_root / "python" / "src" / "ppg_hr")
    snapshot = build_p0_snapshot(
        non_lyx_inventory_path=non_lyx,
        lyx_inventory_path=lyx,
        algorithm_sha256=algorithm_sha,
    )
    receipt = write_p0_snapshot(
        snapshot,
        output_root,
        receipt_metadata={
            "acceptance_contract_sha256": _file_sha256(repo_root / CONTRACT_PATH),
            "acceptance_contract_id": contract["schema_id"],
            "baseline_inventory_sources": {
                "non_lyx": {
                    "path": str(non_lyx),
                    "sha256": _file_sha256(non_lyx),
                },
                "lyx": {
                    "path": str(lyx),
                    "sha256": _file_sha256(lyx),
                },
            },
        },
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_inventory_root(repo_root: Path) -> Path:
    candidates = (
        repo_root.parent / "multiperson-joint-bo-screening" / INVENTORY_RELATIVE_ROOT,
        repo_root / ".worktrees" / "multiperson-joint-bo-screening" / INVENTORY_RELATIVE_ROOT,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("baseline_inventory_root_not_found")


def _load_and_validate_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "time_bias_s": TIME_BIAS_S,
        "expected_record_count": EXPECTED_RECORD_COUNT,
        "expected_fold_count": EXPECTED_FOLD_COUNT,
        "expected_coordinate_count": EXPECTED_COORDINATE_COUNT,
        "expected_hf_call_count": EXPECTED_HF_CALL_COUNT,
        "standard_scene_roster": list(STANDARD_ROSTER),
        "run_scene_roster": list(RUN_ROSTER),
        "pjy_bobi_labels": list(PJY_BOBI_LABELS),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"acceptance_contract_mismatch:{key}")
    return payload


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
