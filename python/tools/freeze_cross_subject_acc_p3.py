"""Freeze all 48 ACC minimax selections before revealing any holdout metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppg_hr.v2.cross_subject_acc_experiment import ACC_EXPERIMENT_ID
from ppg_hr.v2.cross_subject_acc_selection import freeze_acc_training_inputs
from ppg_hr.v2.cross_subject_loso_source import EXPECTED_FOLD_COUNT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    experiment_root = args.repo_root.resolve() / "data" / "experiments" / ACC_EXPERIMENT_ID
    training_root = (
        args.training_root.resolve() if args.training_root else experiment_root / "p3" / "training"
    )
    output_root = (
        args.output_root.resolve() if args.output_root else experiment_root / "p3" / "selections"
    )
    receipt = freeze_acc_training_inputs(
        training_root,
        output_root,
        expected_fold_count=EXPECTED_FOLD_COUNT,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
