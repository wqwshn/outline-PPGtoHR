"""Freeze all 48 HF matched-minimax selections before holdout reveal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppg_hr.v2.cross_subject_matched_minimax import (
    EXPERIMENT_ID,
    freeze_training_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--experiment-root", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    root = (
        args.experiment_root.resolve()
        if args.experiment_root
        else repo_root / "data" / "experiments" / EXPERIMENT_ID
    )
    receipt = freeze_training_inputs(
        root / "p1" / "training", root / "p1" / "selections", expected_fold_count=48
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
