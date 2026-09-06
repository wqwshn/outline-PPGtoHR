"""Write the independently verified P4 report and Python publication figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppg_hr.v2.cross_subject_curated_reporting import write_curated_p4_package

EXPERIMENT_ID = "cross_subject_multirecord_curated_subset_loso_v1"
CONTRACT_PATH = Path(
    "docs/contracts/acceptance/cross_subject_multirecord_curated_subset_loso_v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    experiment_root = (
        args.experiment_root.resolve()
        if args.experiment_root
        else repo_root / "data" / "experiments" / EXPERIMENT_ID
    )
    output_root = args.output_root.resolve() if args.output_root else experiment_root / "p4"
    receipt = write_curated_p4_package(
        experiment_root=experiment_root,
        contract_path=repo_root / CONTRACT_PATH,
        output_root=output_root,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
