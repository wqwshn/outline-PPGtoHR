"""Command-line entry points for the bounded p25 spectral diagnostic."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .recovery_p25_spectral_diagnostic import (
    execute_p25_spectral_diagnostic,
    prepare_p25_spectral_diagnostic_governance,
    propose_p25_spectral_diagnostic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="提案、登记或执行 3×12 个 p25 频谱诊断身份",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    propose = commands.add_parser("propose")
    propose.add_argument("--stage-r-proposal", required=True, type=Path)
    propose.add_argument("--stage-r-completion", required=True, type=Path)
    propose.add_argument("--profile-library", required=True, type=Path)
    propose.add_argument("--budget-contract", required=True, type=Path)
    propose.add_argument("--metric-contract", required=True, type=Path)
    propose.add_argument(
        "--spectral-gate-contract",
        required=True,
        type=Path,
    )
    propose.add_argument("--output-dir", required=True, type=Path)
    propose.add_argument("--source-root", required=True, type=Path)
    propose.add_argument("--parent-experiment-id", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--proposal-dir", required=True, type=Path)
    prepare.add_argument(
        "--authorization-receipt",
        required=True,
        type=Path,
    )
    prepare.add_argument(
        "--source-governance-dir",
        required=True,
        type=Path,
    )
    prepare.add_argument("--governance-dir", required=True, type=Path)

    execute = commands.add_parser("execute")
    execute.add_argument("--proposal-dir", required=True, type=Path)
    execute.add_argument("--governance-dir", required=True, type=Path)
    execute.add_argument("--output-dir", required=True, type=Path)
    execute.add_argument("--source-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "propose":
        result = propose_p25_spectral_diagnostic(
            stage_r_proposal_path=args.stage_r_proposal,
            stage_r_completion_path=args.stage_r_completion,
            profile_library_path=args.profile_library,
            budget_contract_path=args.budget_contract,
            metric_contract_path=args.metric_contract,
            spectral_gate_contract_path=args.spectral_gate_contract,
            output_dir=args.output_dir,
            source_root=args.source_root,
            parent_experiment_id=args.parent_experiment_id,
        )
    elif args.command == "prepare":
        result = prepare_p25_spectral_diagnostic_governance(
            proposal_dir=args.proposal_dir,
            authorization_receipt_path=args.authorization_receipt,
            source_governance_dir=args.source_governance_dir,
            governance_dir=args.governance_dir,
        )
    else:
        result = execute_p25_spectral_diagnostic(
            proposal_dir=args.proposal_dir,
            governance_dir=args.governance_dir,
            output_dir=args.output_dir,
            source_root=args.source_root,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
