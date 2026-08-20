"""CLI for the bounded p25-short-low filter-mechanism decomposition."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .recovery_filter_mechanism_decomposition import (
    execute_filter_mechanism_decomposition,
    prepare_filter_mechanism_decomposition_governance,
    propose_filter_mechanism_decomposition,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="提案、登记或执行 12 条记录的滤波机制分解诊断",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    propose = commands.add_parser("propose")
    propose.add_argument("--p25-proposal", required=True, type=Path)
    propose.add_argument("--p25-completion", required=True, type=Path)
    propose.add_argument("--p25-decision", required=True, type=Path)
    propose.add_argument("--p25-manifest", required=True, type=Path)
    propose.add_argument("--source-budget-contract", required=True, type=Path)
    propose.add_argument("--spectral-gate-contract", required=True, type=Path)
    propose.add_argument("--spec", required=True, type=Path)
    propose.add_argument("--output-dir", required=True, type=Path)
    propose.add_argument("--source-root", required=True, type=Path)
    propose.add_argument("--parent-experiment-id", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--proposal-dir", required=True, type=Path)
    prepare.add_argument("--authorization-receipt", required=True, type=Path)
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
        result = propose_filter_mechanism_decomposition(
            p25_proposal_path=args.p25_proposal,
            p25_completion_path=args.p25_completion,
            p25_decision_path=args.p25_decision,
            p25_manifest_path=args.p25_manifest,
            source_budget_contract_path=args.source_budget_contract,
            spectral_gate_contract_path=args.spectral_gate_contract,
            spec_path=args.spec,
            output_dir=args.output_dir,
            source_root=args.source_root,
            parent_experiment_id=args.parent_experiment_id,
        )
    elif args.command == "prepare":
        result = prepare_filter_mechanism_decomposition_governance(
            proposal_dir=args.proposal_dir,
            authorization_receipt_path=args.authorization_receipt,
            source_governance_dir=args.source_governance_dir,
            governance_dir=args.governance_dir,
            source_root=args.source_root,
        )
    else:
        result = execute_filter_mechanism_decomposition(
            proposal_dir=args.proposal_dir,
            governance_dir=args.governance_dir,
            output_dir=args.output_dir,
            source_root=args.source_root,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
