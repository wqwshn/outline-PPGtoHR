"""Freeze all 48 grouped-LOSO selections from training-only inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppg_hr.v2.cross_subject_loso_selection import freeze_training_inputs
from ppg_hr.v2.cross_subject_loso_source import EXPECTED_FOLD_COUNT, EXPERIMENT_ID


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    p3_root = repo_root / "data" / "experiments" / EXPERIMENT_ID / "p3"
    training_root = (
        args.training_root.resolve() if args.training_root else p3_root / "training_inputs"
    )
    output_root = args.output_root.resolve() if args.output_root else p3_root / "selections"
    receipt = freeze_training_inputs(
        training_root,
        output_root,
        expected_fold_count=EXPECTED_FOLD_COUNT,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
