"""Run the independent hard-stop verifier for the curated subset experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppg_hr.v2.cross_subject_curated_verifier import write_verification_receipt

EXPERIMENT_ID = "cross_subject_multirecord_curated_subset_loso_v1"
CONTRACT_PATH = Path(
    "docs/contracts/acceptance/cross_subject_multirecord_curated_subset_loso_v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    experiment_root = (
        args.experiment_root.resolve()
        if args.experiment_root
        else repo_root / "data" / "experiments" / EXPERIMENT_ID
    )
    output_path = (
        args.output_path.resolve()
        if args.output_path
        else experiment_root / "p4" / "verification_receipt.json"
    )
    receipt = write_verification_receipt(
        experiment_root,
        repo_root / CONTRACT_PATH,
        output_path,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
