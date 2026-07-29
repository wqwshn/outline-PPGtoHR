"""CLI for proposing and executing the LYX twelve-slot fold replay."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .recovery_fold_replay_execution import (
    execute_fold_replay_proposal,
)
from .recovery_fold_replay_plan import (
    propose_fold_replay_execution,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LYX twelve-slot leakage-safe fold replay",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    propose = subparsers.add_parser("propose")
    propose.add_argument("--final-interaction-audit", type=Path, required=True)
    propose.add_argument("--pre-fold-gate-receipt", type=Path, required=True)
    propose.add_argument("--pre-fold-human-decision", type=Path)
    propose.add_argument("--budget-contract", type=Path, required=True)
    propose.add_argument("--output-dir", type=Path, required=True)
    propose.add_argument("--source-root", type=Path, required=True)
    propose.add_argument("--parent-experiment-id", required=True)

    execute = subparsers.add_parser("execute")
    execute.add_argument("--proposal-dir", type=Path, required=True)
    execute.add_argument("--governance-dir", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--source-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "propose":
        result = propose_fold_replay_execution(
            final_interaction_audit_path=args.final_interaction_audit,
            pre_fold_gate_receipt_path=args.pre_fold_gate_receipt,
            pre_fold_human_decision_path=args.pre_fold_human_decision,
            budget_contract_path=args.budget_contract,
            output_dir=args.output_dir,
            source_root=args.source_root,
            parent_experiment_id=args.parent_experiment_id,
        )
    else:
        result = execute_fold_replay_proposal(
            proposal_dir=args.proposal_dir,
            governance_dir=args.governance_dir,
            output_dir=args.output_dir,
            source_root=args.source_root,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
