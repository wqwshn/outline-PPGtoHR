"""CLI for publishing the reporting-only post-fold package."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .recovery_post_fold_experiment import publish_post_fold_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="发布 LYX 折后人工门、开发内报告与挑战场景交接包",
    )
    parser.add_argument("--fold-replay-proposal", required=True, type=Path)
    parser.add_argument("--pre-fold-gate-receipt", required=True, type=Path)
    parser.add_argument("--fold-replay-report", required=True, type=Path)
    parser.add_argument("--fold-selection-receipt", required=True, type=Path)
    parser.add_argument("--final-interaction-audit", required=True, type=Path)
    parser.add_argument("--historical-ab-report", required=True, type=Path)
    parser.add_argument("--current-role-matrix", required=True, type=Path)
    parser.add_argument("--review-context", required=True, type=Path)
    parser.add_argument("--budget-contract", required=True, type=Path)
    parser.add_argument("--exploration-registry", required=True, type=Path)
    parser.add_argument("--attempt-registry", required=True, type=Path)
    parser.add_argument(
        "--budget-amendment-authorization",
        required=True,
        type=Path,
        nargs="+",
    )
    parser.add_argument("--challenge-scene-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    completion = publish_post_fold_package(
        fold_replay_proposal_path=args.fold_replay_proposal,
        pre_fold_gate_receipt_path=args.pre_fold_gate_receipt,
        fold_replay_report_path=args.fold_replay_report,
        fold_selection_receipt_path=args.fold_selection_receipt,
        final_interaction_audit_path=args.final_interaction_audit,
        historical_ab_report_path=args.historical_ab_report,
        current_role_matrix_path=args.current_role_matrix,
        review_context_path=args.review_context,
        budget_contract_path=args.budget_contract,
        exploration_registry_path=args.exploration_registry,
        attempt_registry_path=args.attempt_registry,
        budget_amendment_authorization_paths=(args.budget_amendment_authorization),
        challenge_scene_manifest_path=args.challenge_scene_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(completion, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
