"""CLI for the bounded rank-1 Stage R repair."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .recovery_stage_r_rank1_replan import (
    execute_stage_r_rank1_replan,
    prepare_stage_r_rank1_replan_governance,
    propose_stage_r_rank1_replan,
    publish_stage_r_rank1_failure_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="提案、登记或执行 36 个 rank-1 Stage R 正式身份",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    propose = commands.add_parser("propose")
    propose.add_argument("--rank1-proposal", required=True, type=Path)
    propose.add_argument("--rank1-completion", required=True, type=Path)
    propose.add_argument("--rank1-decision", required=True, type=Path)
    propose.add_argument("--rank1-manifest", required=True, type=Path)
    propose.add_argument(
        "--prior-stage-r-proposal",
        required=True,
        type=Path,
    )
    propose.add_argument(
        "--prior-stage-r-governance-receipt",
        required=True,
        type=Path,
    )
    propose.add_argument(
        "--recovery-registry",
        required=True,
        type=Path,
    )
    propose.add_argument(
        "--penalty-registry",
        required=True,
        type=Path,
    )
    propose.add_argument(
        "--baseline-manifest",
        required=True,
        type=Path,
    )
    propose.add_argument(
        "--baseline-metrics",
        required=True,
        type=Path,
    )
    propose.add_argument(
        "--source-budget-contract",
        required=True,
        type=Path,
    )
    propose.add_argument("--spec", required=True, type=Path)
    propose.add_argument("--output-dir", required=True, type=Path)
    propose.add_argument("--source-root", required=True, type=Path)
    propose.add_argument("--parent-experiment-id", required=True)
    propose.add_argument("--superseded-proposal", type=Path)
    propose.add_argument("--superseded-authorization", type=Path)
    propose.add_argument("--source-governance-receipt", type=Path)
    propose.add_argument("--source-exploration-registry", type=Path)
    propose.add_argument("--failed-execution-binding", type=Path)
    propose.add_argument("--runtime-failure-receipt", type=Path)

    failure = commands.add_parser("record-failure")
    failure.add_argument(
        "--superseded-proposal",
        required=True,
        type=Path,
    )
    failure.add_argument(
        "--superseded-authorization",
        required=True,
        type=Path,
    )
    failure.add_argument(
        "--source-governance-receipt",
        required=True,
        type=Path,
    )
    failure.add_argument(
        "--source-attempt-registry",
        required=True,
        type=Path,
    )
    failure.add_argument(
        "--source-exploration-registry",
        required=True,
        type=Path,
    )
    failure.add_argument(
        "--failed-execution-binding",
        required=True,
        type=Path,
    )
    failure.add_argument("--output", required=True, type=Path)

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
    prepare.add_argument(
        "--governance-dir",
        required=True,
        type=Path,
    )
    prepare.add_argument("--source-root", required=True, type=Path)

    execute = commands.add_parser("execute")
    execute.add_argument("--proposal-dir", required=True, type=Path)
    execute.add_argument(
        "--authorization-receipt",
        required=True,
        type=Path,
    )
    execute.add_argument(
        "--governance-dir",
        required=True,
        type=Path,
    )
    execute.add_argument("--output-dir", required=True, type=Path)
    execute.add_argument("--source-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "propose":
        result = propose_stage_r_rank1_replan(
            rank1_proposal_path=args.rank1_proposal,
            rank1_completion_path=args.rank1_completion,
            rank1_decision_path=args.rank1_decision,
            rank1_manifest_path=args.rank1_manifest,
            prior_stage_r_proposal_path=args.prior_stage_r_proposal,
            prior_stage_r_governance_receipt_path=(
                args.prior_stage_r_governance_receipt
            ),
            recovery_registry_path=args.recovery_registry,
            penalty_registry_path=args.penalty_registry,
            baseline_manifest_path=args.baseline_manifest,
            baseline_metrics_path=args.baseline_metrics,
            source_budget_contract_path=args.source_budget_contract,
            spec_path=args.spec,
            output_dir=args.output_dir,
            source_root=args.source_root,
            parent_experiment_id=args.parent_experiment_id,
            superseded_proposal_path=args.superseded_proposal,
            superseded_authorization_path=(
                args.superseded_authorization
            ),
            source_governance_receipt_path=(
                args.source_governance_receipt
            ),
            source_exploration_registry_path=(
                args.source_exploration_registry
            ),
            failed_execution_binding_path=(
                args.failed_execution_binding
            ),
            runtime_failure_receipt_path=(
                args.runtime_failure_receipt
            ),
        )
    elif args.command == "record-failure":
        result = publish_stage_r_rank1_failure_receipt(
            superseded_proposal_path=args.superseded_proposal,
            superseded_authorization_path=(
                args.superseded_authorization
            ),
            source_governance_receipt_path=(
                args.source_governance_receipt
            ),
            source_attempt_registry_path=(
                args.source_attempt_registry
            ),
            source_exploration_registry_path=(
                args.source_exploration_registry
            ),
            failed_execution_binding_path=(
                args.failed_execution_binding
            ),
            output_path=args.output,
        )
    elif args.command == "prepare":
        result = prepare_stage_r_rank1_replan_governance(
            proposal_dir=args.proposal_dir,
            authorization_receipt_path=args.authorization_receipt,
            source_governance_dir=args.source_governance_dir,
            governance_dir=args.governance_dir,
            source_root=args.source_root,
        )
    else:
        result = execute_stage_r_rank1_replan(
            proposal_dir=args.proposal_dir,
            authorization_receipt_path=args.authorization_receipt,
            governance_dir=args.governance_dir,
            output_dir=args.output_dir,
            source_root=args.source_root,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
