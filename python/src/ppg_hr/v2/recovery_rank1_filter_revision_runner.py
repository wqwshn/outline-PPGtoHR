"""CLI for the bounded p25-short-low rank-1 filter revision."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .recovery_rank1_filter_revision import (
    execute_rank1_filter_revision,
    prepare_rank1_filter_revision_governance,
    propose_rank1_filter_revision,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="提案、登记或执行 12 条记录的 rank-1 滤波修订诊断",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    propose = commands.add_parser("propose")
    propose.add_argument("--mechanism-proposal", required=True, type=Path)
    propose.add_argument("--mechanism-completion", required=True, type=Path)
    propose.add_argument("--mechanism-decision", required=True, type=Path)
    propose.add_argument("--mechanism-manifest", required=True, type=Path)
    propose.add_argument("--source-budget-contract", required=True, type=Path)
    propose.add_argument("--spectral-gate-contract", required=True, type=Path)
    propose.add_argument("--spec", required=True, type=Path)
    propose.add_argument("--output-dir", required=True, type=Path)
    propose.add_argument("--source-root", required=True, type=Path)
    propose.add_argument("--parent-experiment-id", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--proposal-dir", required=True, type=Path)
    prepare.add_argument("--source-governance-dir", required=True, type=Path)
    prepare.add_argument("--governance-dir", required=True, type=Path)
    prepare.add_argument("--source-root", required=True, type=Path)

    execute = commands.add_parser("execute")
    execute.add_argument("--proposal-dir", required=True, type=Path)
    execute.add_argument("--governance-dir", required=True, type=Path)
    execute.add_argument("--output-dir", required=True, type=Path)
    execute.add_argument("--source-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "propose":
        result = propose_rank1_filter_revision(
            mechanism_proposal_path=args.mechanism_proposal,
            mechanism_completion_path=args.mechanism_completion,
            mechanism_decision_path=args.mechanism_decision,
            mechanism_manifest_path=args.mechanism_manifest,
            source_budget_contract_path=args.source_budget_contract,
            spectral_gate_contract_path=args.spectral_gate_contract,
            spec_path=args.spec,
            output_dir=args.output_dir,
            source_root=args.source_root,
            parent_experiment_id=args.parent_experiment_id,
        )
    elif args.command == "prepare":
        result = prepare_rank1_filter_revision_governance(
            proposal_dir=args.proposal_dir,
            source_governance_dir=args.source_governance_dir,
            governance_dir=args.governance_dir,
            source_root=args.source_root,
        )
    else:
        result = execute_rank1_filter_revision(
            proposal_dir=args.proposal_dir,
            governance_dir=args.governance_dir,
            output_dir=args.output_dir,
            source_root=args.source_root,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
