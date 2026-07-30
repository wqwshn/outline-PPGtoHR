"""CLI for the governed recovery independent-BO experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .phase2_experiment_io import atomic_write_json, read_json
from .recovery_contracts import canonical_sha256
from .recovery_independent_bo_experiment import (
    build_recovery_independent_bo_proposal,
    execute_recovery_independent_bo_proposal,
    prepare_recovery_independent_bo_governance,
    validate_recovery_independent_bo_preflight,
)


def _write_proposal_artifacts(
    *,
    proposal: dict[str, Any],
    output_dir: Path,
) -> None:
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            f"independent_bo_proposal_dir_exists:{destination}"
        )
    destination.mkdir(parents=True)
    atomic_write_json(
        destination / "recovery_independent_bo_proposal.json",
        proposal,
    )
    atomic_write_json(
        destination / "budget_contract_v13.json",
        dict(proposal["budget_contract"]),
    )
    atomic_write_json(
        destination / "budget_amendment_request.json",
        dict(proposal["budget_amendment_request"]),
    )
    atomic_write_json(
        destination / "independent_bo_request.json",
        dict(proposal["independent_bo_request"]),
    )
    atomic_write_json(
        destination / "search_space.json",
        dict(proposal["search_space"]),
    )
    atomic_write_json(
        destination / "seed_manifest.json",
        dict(proposal["seed_manifest"]),
    )
    atomic_write_json(
        destination / "metric_contract.json",
        dict(proposal["metric_contract"]),
    )
    receipt = {
        "receipt_version": (
            "lyx_recovery_independent_bo_proposal_receipt_v1"
        ),
        "status": "awaiting_human_independent_bo_decision",
        "proposal_sha256": proposal["proposal_sha256"],
        "unique_budget": proposal["unique_budget"],
        "formal_solver_run_count": 0,
        "independent_bo_run_count": 0,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    atomic_write_json(destination / "proposal_receipt.json", receipt)


def _record_blanket_authorization(
    *,
    proposal_dir: Path,
    approved_at: str,
    expires_at: str,
) -> None:
    root = Path(proposal_dir).resolve()
    proposal = read_json(
        root / "recovery_independent_bo_proposal.json"
    )
    request = dict(proposal["independent_bo_request"])
    execution = {
        "approved": True,
        "decision_state": (
            "awaiting_human_independent_bo_decision"
        ),
        **request,
        "proposal_sha256": proposal["proposal_sha256"],
        "budget_contract_hash": proposal[
            "budget_contract_hash"
        ],
        "approved_at": approved_at,
        "approved_by": "user",
        "authorization_basis": (
            "blanket_proposal_authorization_until_deadline"
        ),
        "blanket_authorization_expires_at": expires_at,
        "user_authorization": (
            "在7月31日10:00前，所有需要人工批准的proposal"
            "均授权通过并可继续执行"
        ),
    }
    execution["authorization_sha256"] = canonical_sha256(
        execution
    )
    atomic_write_json(
        root / "execution_authorization.json",
        execution,
    )
    amendment = dict(proposal["budget_amendment_request"])
    budget = {
        "approved": True,
        "decision_state": "awaiting_human_budget_decision",
        **amendment,
        "independent_bo_authorized": False,
        "approved_at": approved_at,
        "approved_by": "user",
        "authorization_basis": (
            "blanket_proposal_authorization_until_deadline"
        ),
        "blanket_authorization_expires_at": expires_at,
        "proposal_sha256": proposal["proposal_sha256"],
    }
    budget["authorization_sha256"] = canonical_sha256(budget)
    atomic_write_json(
        root / "budget_authorization.json",
        budget,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    propose = subparsers.add_parser("propose")
    for name in (
        "stage_r_proposal",
        "stage_r_completion",
        "stage_r_selection",
        "stage_r_result_index",
        "repository_root",
        "output_dir",
    ):
        propose.add_argument(f"--{name.replace('_', '-')}", required=True)
    authorize = subparsers.add_parser("authorize-blanket")
    authorize.add_argument("--proposal-dir", required=True)
    authorize.add_argument("--approved-at", required=True)
    authorize.add_argument("--expires-at", required=True)
    prepare = subparsers.add_parser("prepare")
    for name in (
        "proposal_dir",
        "source_governance_dir",
        "target_governance_dir",
        "repository_root",
    ):
        prepare.add_argument(f"--{name.replace('_', '-')}", required=True)
    execute = subparsers.add_parser("execute")
    for name in (
        "proposal_dir",
        "governance_dir",
        "output_dir",
        "repository_root",
    ):
        execute.add_argument(f"--{name.replace('_', '-')}", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--proposal-dir", required=True)
    preflight.add_argument("--repository-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "propose":
        proposal = build_recovery_independent_bo_proposal(
            stage_r_proposal=read_json(
                Path(args.stage_r_proposal).resolve()
            ),
            stage_r_completion=read_json(
                Path(args.stage_r_completion).resolve()
            ),
            stage_r_selection=read_json(
                Path(args.stage_r_selection).resolve()
            ),
            stage_r_result_index=read_json(
                Path(args.stage_r_result_index).resolve()
            ),
            repository_root=Path(args.repository_root),
        )
        _write_proposal_artifacts(
            proposal=proposal,
            output_dir=Path(args.output_dir),
        )
        print(
            json.dumps(
                {
                    "status": proposal["status"],
                    "proposal_sha256": proposal[
                        "proposal_sha256"
                    ],
                    "unique_budget": proposal["unique_budget"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "authorize-blanket":
        _record_blanket_authorization(
            proposal_dir=Path(args.proposal_dir),
            approved_at=args.approved_at,
            expires_at=args.expires_at,
        )
        print('{"status":"authorized"}')
        return 0
    if args.command == "prepare":
        root = Path(args.proposal_dir).resolve()
        receipt = prepare_recovery_independent_bo_governance(
            proposal=read_json(
                root / "recovery_independent_bo_proposal.json"
            ),
            execution_authorization=read_json(
                root / "execution_authorization.json"
            ),
            budget_authorization=read_json(
                root / "budget_authorization.json"
            ),
            source_governance_dir=Path(
                args.source_governance_dir
            ),
            target_governance_dir=Path(
                args.target_governance_dir
            ),
            repository_root=Path(args.repository_root),
        )
        print(json.dumps(receipt, ensure_ascii=False))
        return 0
    if args.command == "preflight":
        root = Path(args.proposal_dir).resolve()
        validate_recovery_independent_bo_preflight(
            proposal=read_json(
                root / "recovery_independent_bo_proposal.json"
            ),
            repository_root=Path(args.repository_root),
        )
        print('{"status":"preflight_passed"}')
        return 0
    if args.command == "execute":
        root = Path(args.proposal_dir).resolve()

        def progress(event: dict[str, Any]) -> None:
            print(json.dumps(event, ensure_ascii=False), flush=True)

        completion = execute_recovery_independent_bo_proposal(
            proposal_path=(
                root / "recovery_independent_bo_proposal.json"
            ),
            authorization_path=(
                root / "execution_authorization.json"
            ),
            governance_dir=Path(args.governance_dir),
            output_dir=Path(args.output_dir),
            repository_root=Path(args.repository_root),
            progress_callback=progress,
        )
        print(json.dumps(completion, ensure_ascii=False))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
