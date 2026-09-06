"""Materialise training-only inputs and freeze HF/ACC selections for all panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppg_hr.v2.cross_subject_curated_subset import prepare_curated_p2

EXPERIMENT_ID = "cross_subject_multirecord_curated_subset_loso_v1"
CONTRACT_PATH = Path(
    "docs/contracts/acceptance/cross_subject_multirecord_curated_subset_loso_v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--p0-root", type=Path)
    parser.add_argument("--p1-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    experiment_root = repo_root / "data" / "experiments" / EXPERIMENT_ID
    receipt = prepare_curated_p2(
        p0_root=args.p0_root.resolve() if args.p0_root else experiment_root / "p0",
        p1_root=args.p1_root.resolve() if args.p1_root else experiment_root / "p1",
        contract_path=repo_root / CONTRACT_PATH,
        output_root=args.output_root.resolve() if args.output_root else experiment_root / "p2",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
