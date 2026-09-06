"""Bind the curated subset experiment to its completed HF and ACC parents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppg_hr.v2.cross_subject_curated_subset import prepare_curated_p0

EXPERIMENT_ID = "cross_subject_multirecord_curated_subset_loso_v1"
HF_PARENT_ID = "cross_subject_multirecord_hf_loso_v1"
ACC_PARENT_ID = "cross_subject_multirecord_acc_independent_physical4d_v1"
CONTRACT_PATH = Path(
    "docs/contracts/acceptance/cross_subject_multirecord_curated_subset_loso_v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--hf-parent-root", type=Path)
    parser.add_argument("--acc-parent-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    parent_base = _default_parent_base(repo_root)
    receipt = prepare_curated_p0(
        hf_root=(args.hf_parent_root or parent_base / HF_PARENT_ID).resolve(),
        acc_root=(args.acc_parent_root or parent_base / ACC_PARENT_ID).resolve(),
        contract_path=repo_root / CONTRACT_PATH,
        output_root=(
            args.output_root.resolve()
            if args.output_root
            else repo_root / "data" / "experiments" / EXPERIMENT_ID / "p0"
        ),
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_parent_base(repo_root: Path) -> Path:
    candidates = (
        repo_root / "data" / "experiments",
        repo_root.parent / "cross-subject-multirecord-hf-loso" / "data" / "experiments",
    )
    for candidate in candidates:
        if (candidate / HF_PARENT_ID).is_dir() and (candidate / ACC_PARENT_ID).is_dir():
            return candidate.resolve()
    raise FileNotFoundError("curated_parent_experiment_base_not_found")


if __name__ == "__main__":
    raise SystemExit(main())
