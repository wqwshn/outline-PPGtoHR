"""Stable public facade for LYX Stage P interaction execution."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .recovery_interaction_execution import (
    execute_rollback_backup_proposal,
    propose_rollback_backup_execution,
)
from .recovery_interaction_resolution import (
    build_final_interaction_audit,
    build_rollback_backup_proposal,
    resolve_recovery_interaction,
)
from .recovery_pre_fold_execution import (
    execute_historical_recovery_ab_proposal,
    propose_historical_recovery_ab_execution,
)
from .recovery_pre_fold_gate import (
    build_historical_recovery_ab_proposal,
    build_historical_recovery_ab_report,
    evaluate_pre_fold_independent_bo_gate,
    freeze_historical_parameter_replay_manifest,
    publish_pre_fold_independent_bo_gate,
)
from .recovery_stage_p_contracts import (
    StagePPlanError,
    StagePProgressCallback,
)
from .recovery_stage_p_execution import execute_stage_p_proposal
from .recovery_stage_p_plan import (
    build_stage_p_proposal,
    propose_stage_p_execution,
)
from .recovery_stage_p_reporting import build_penalty_interaction_report

__all__ = [
    "StagePPlanError",
    "StagePProgressCallback",
    "build_rollback_backup_proposal",
    "build_penalty_interaction_report",
    "build_historical_recovery_ab_proposal",
    "build_historical_recovery_ab_report",
    "build_final_interaction_audit",
    "build_stage_p_proposal",
    "execute_rollback_backup_proposal",
    "execute_historical_recovery_ab_proposal",
    "execute_stage_p_proposal",
    "evaluate_pre_fold_independent_bo_gate",
    "freeze_historical_parameter_replay_manifest",
    "publish_pre_fold_independent_bo_gate",
    "propose_rollback_backup_execution",
    "propose_historical_recovery_ab_execution",
    "propose_stage_p_execution",
    "resolve_recovery_interaction",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="冻结 LYX Stage P 3×8×12 惩罚交互零运行 proposal",
    )
    parser.add_argument("--stage-f-proposal", required=True, type=Path)
    parser.add_argument("--stage-f-completion", required=True, type=Path)
    parser.add_argument(
        "--stage-f-profile-matrix",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--stage-f-current-role-matrix",
        required=True,
        type=Path,
    )
    parser.add_argument("--penalty-registry", required=True, type=Path)
    parser.add_argument("--budget-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--parent-experiment-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = propose_stage_p_execution(
        stage_f_proposal_path=args.stage_f_proposal,
        stage_f_completion_path=args.stage_f_completion,
        stage_f_profile_matrix_path=args.stage_f_profile_matrix,
        stage_f_current_role_matrix_path=(
            args.stage_f_current_role_matrix
        ),
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
