"""Stable interface for the LYX Stage F experiment.

Stage R v3 froze its evaluation source closure before these modules were
introduced. Stage F reuses those frozen cache/config interfaces without
modifying the Stage R implementation.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .recovery_stage_f_contracts import (
    StageFPlanError,
    StageFProgressCallback,
)
from .recovery_stage_f_execution import execute_stage_f_proposal
from .recovery_stage_f_plan import (
    build_stage_f_proposal,
    propose_stage_f_execution,
)

__all__ = [
    "StageFPlanError",
    "StageFProgressCallback",
    "build_stage_f_proposal",
    "execute_stage_f_proposal",
    "propose_stage_f_execution",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="冻结 LYX Stage F 8×12 双矩阵零运行 proposal",
    )
    parser.add_argument("--stage-r-proposal", required=True, type=Path)
    parser.add_argument("--stage-r-completion", required=True, type=Path)
    parser.add_argument("--profile-library", required=True, type=Path)
    parser.add_argument(
        "--profile-library-completion",
        required=True,
        type=Path,
    )
    parser.add_argument("--baseline-metrics", required=True, type=Path)
    parser.add_argument(
        "--baseline-contract-receipt",
        required=True,
        type=Path,
    )
    parser.add_argument("--recovery-registry", required=True, type=Path)
    parser.add_argument("--penalty-registry", required=True, type=Path)
    parser.add_argument("--budget-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--parent-experiment-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = propose_stage_f_execution(
        stage_r_proposal_path=args.stage_r_proposal,
        stage_r_completion_path=args.stage_r_completion,
        profile_library_path=args.profile_library,
        profile_library_completion_path=(
            args.profile_library_completion
        ),
        baseline_metrics_path=args.baseline_metrics,
        baseline_contract_receipt_path=(
            args.baseline_contract_receipt
        ),
        recovery_registry_path=args.recovery_registry,
        penalty_registry_path=args.penalty_registry,
        budget_contract_path=args.budget_contract,
        output_dir=args.output_dir,
        source_root=args.source_root,
        parent_experiment_id=args.parent_experiment_id,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
