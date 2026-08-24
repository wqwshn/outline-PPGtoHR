"""Govern the exact traceback evidence written by the short-circuit runner.

The numerical repair contract intentionally permits only the missing cell
completion and its paired receipt.  The scheduler must nevertheless preserve
the exact traceback that proves a future direct-call ``mappingproxy`` failure.
This module gives that diagnostic file a separate, zero-budget authorization;
it never runs a solver or mutates numerical artifacts.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from recovery_short_circuit_mappingproxy_repair import (
    is_short_circuit_mappingproxy_completion_failure,
    validate_direct_repair_authorization,
    validate_direct_repair_proposal,
)
from recovery_short_circuit_runner import (
    validate_authorization as validate_short_circuit_authorization,
)
from recovery_short_circuit_runner import (
    validate_proposal as validate_short_circuit_proposal,
)

from ppg_hr.v2.experiment_freeze_utils import file_sha256
from ppg_hr.v2.phase2_experiment_io import atomic_write_json, read_json
from ppg_hr.v2.recovery_contracts import canonical_sha256

PROPOSAL_VERSION = (
    "lyx_recovery_short_circuit_failure_evidence_supplement_proposal_v1"
)
AUTHORIZATION_VERSION = (
    "lyx_recovery_short_circuit_failure_evidence_supplement_authorization_v1"
)
PROPOSAL_RECEIPT_VERSION = (
    "lyx_recovery_short_circuit_failure_evidence_supplement_receipt_v1"
)
AUTHORIZATION_SCOPE_END = "experiment_complete"
ALLOWED_WRITE = (
    "exact_short_circuit_mappingproxy_failure_evidence_logs_only"
)
USER_AUTHORIZATION_TEXT = (
    "批准编写并执行短路调度器直接调用栈的零运行替换修复 proposal，批准生成绑定"
    "修复后源码的新短路 proposal，并授权继续执行 Gate A 和必要的 Gate B，直至"
    "本轮大实验完成；不得重算已完成身份或自动重试失败身份。"
)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


class FailureEvidenceSupplementError(RuntimeError):
    """Raised when the failure-evidence authorization drifts."""


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FailureEvidenceSupplementError(f"{name}_invalid")
    return value


def _list(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise FailureEvidenceSupplementError(f"{name}_invalid")
    return value


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    artifact_name: str,
) -> None:
    frozen = dict(payload)
    observed = frozen.pop(hash_field, None)
    if not isinstance(observed, str) or observed != canonical_sha256(frozen):
        raise FailureEvidenceSupplementError(
            f"{artifact_name}_hash_invalid"
        )


def _relative_to_root(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise FailureEvidenceSupplementError(
            "failure_evidence_path_outside_repository"
        )
    return resolved.relative_to(root).as_posix()


def expected_failure_evidence_paths(
    *,
    short_circuit_proposal: Mapping[str, Any],
    execution_output_dir: str,
) -> list[str]:
    """Return the only per-cell traceback paths covered by this supplement."""

    output = PurePosixPath(execution_output_dir)
    if output.is_absolute() or ".." in output.parts:
        raise FailureEvidenceSupplementError(
            "execution_output_dir_invalid"
        )
    paths: list[str] = []
    coordinates: set[tuple[str, str]] = set()
    for raw in _list(
        "gate_a_cell_bindings",
        short_circuit_proposal.get("gate_a_cell_bindings"),
    ):
        binding = _mapping("gate_a_cell_binding", raw)
        recovery_id = str(binding.get("recovery_candidate_id", ""))
        record_id = str(binding.get("record_id", ""))
        if (
            not _SAFE_COMPONENT.fullmatch(recovery_id)
            or not _SAFE_COMPONENT.fullmatch(record_id)
            or (recovery_id, record_id) in coordinates
        ):
            raise FailureEvidenceSupplementError(
                "gate_a_failure_evidence_coordinate_invalid"
            )
        coordinates.add((recovery_id, record_id))
        paths.append(
            (
                output
                / "direct_repair_failures"
                / f"{recovery_id}__{record_id}.stderr.log"
            ).as_posix()
        )
    return sorted(paths)


def validate_observed_failure_evidence(path: Path) -> None:
    failure_path = Path(path)
    if not failure_path.is_file() or not (
        is_short_circuit_mappingproxy_completion_failure(
            failure_path.read_text(encoding="utf-8", errors="replace")
        )
    ):
        raise FailureEvidenceSupplementError(
            "failure_evidence_signature_invalid"
        )


def build_proposal(
    *,
    short_circuit_proposal_path: Path,
    short_circuit_authorization_path: Path,
    direct_repair_proposal_path: Path,
    direct_repair_authorization_path: Path,
    scheduler_path: Path,
    spec_path: Path,
    execution_output_dir: Path,
    repository_root: Path,
    tool_path: Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    short_path = Path(short_circuit_proposal_path).resolve()
    short_auth_path = Path(short_circuit_authorization_path).resolve()
    direct_path = Path(direct_repair_proposal_path).resolve()
    direct_auth_path = Path(direct_repair_authorization_path).resolve()
    scheduler = Path(scheduler_path).resolve()
    spec = Path(spec_path).resolve()
    tool = Path(tool_path).resolve()
    output = Path(execution_output_dir).resolve()
    for path in (
        short_path,
        short_auth_path,
        direct_path,
        direct_auth_path,
        scheduler,
        spec,
        tool,
    ):
        if not path.is_file():
            raise FailureEvidenceSupplementError(
                "failure_evidence_bound_file_missing"
            )
    if not output.is_dir() or not output.is_relative_to(root):
        raise FailureEvidenceSupplementError(
            "failure_evidence_output_dir_invalid"
        )
    short = validate_short_circuit_proposal(
        proposal=read_json(short_path),
        repository_root=root,
    )
    short_auth = validate_short_circuit_authorization(
        proposal=short,
        receipt=read_json(short_auth_path),
    )
    direct = validate_direct_repair_proposal(
        proposal=read_json(direct_path),
        repository_root=root,
    )
    direct_auth = validate_direct_repair_authorization(
        proposal=direct,
        receipt=read_json(direct_auth_path),
    )
    scheduler_relative = _relative_to_root(scheduler, root)
    if (
        scheduler_relative != short.get("scheduler_path")
        or file_sha256(scheduler) != short.get("scheduler_file_sha256")
        or direct.get("proposal_sha256")
        != short.get("direct_repair_proposal_sha256")
        or canonical_sha256(direct_auth)
        != short.get("direct_repair_authorization_sha256")
    ):
        raise FailureEvidenceSupplementError(
            "failure_evidence_parent_binding_drift"
        )
    output_relative = _relative_to_root(output, root)
    allowed_paths = expected_failure_evidence_paths(
        short_circuit_proposal=short,
        execution_output_dir=output_relative,
    )
    if len(allowed_paths) != 12:
        raise FailureEvidenceSupplementError(
            "failure_evidence_gate_a_panel_invalid"
        )
    proposal: dict[str, Any] = {
        "proposal_version": PROPOSAL_VERSION,
        "status": "frozen_zero_solver_runs",
        "short_circuit_proposal_path": _relative_to_root(
            short_path, root
        ),
        "short_circuit_proposal_file_sha256": file_sha256(short_path),
        "short_circuit_proposal_sha256": short["proposal_sha256"],
        "short_circuit_authorization_path": _relative_to_root(
            short_auth_path, root
        ),
        "short_circuit_authorization_file_sha256": file_sha256(
            short_auth_path
        ),
        "short_circuit_authorization_sha256": short_auth[
            "authorization_sha256"
        ],
        "direct_repair_proposal_path": _relative_to_root(
            direct_path, root
        ),
        "direct_repair_proposal_file_sha256": file_sha256(direct_path),
        "direct_repair_proposal_sha256": direct["proposal_sha256"],
        "direct_repair_authorization_path": _relative_to_root(
            direct_auth_path, root
        ),
        "direct_repair_authorization_file_sha256": file_sha256(
            direct_auth_path
        ),
        "direct_repair_authorization_sha256": direct_auth[
            "authorization_sha256"
        ],
        "scheduler_path": scheduler_relative,
        "scheduler_file_sha256": file_sha256(scheduler),
        "spec_path": _relative_to_root(spec, root),
        "spec_file_sha256": file_sha256(spec),
        "tool_path": _relative_to_root(tool, root),
        "tool_file_sha256": file_sha256(tool),
        "execution_output_dir": output_relative,
        "allowed_failure_evidence_paths": allowed_paths,
        "allowed_failure_evidence_path_count": len(allowed_paths),
        "allowed_write": ALLOWED_WRITE,
        "repair_added_unique_identity_budget": 0,
        "repair_added_solver_run_count": 0,
        "automatic_retry_count": 0,
        "completed_identity_recomputation_allowed": False,
        "authorization_scope_end": AUTHORIZATION_SCOPE_END,
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def validate_proposal(
    *,
    proposal: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    frozen = deepcopy(dict(proposal))
    _verify_embedded_hash(
        frozen,
        hash_field="proposal_sha256",
        artifact_name="failure_evidence_supplement_proposal",
    )
    if (
        frozen.get("proposal_version") != PROPOSAL_VERSION
        or frozen.get("status") != "frozen_zero_solver_runs"
        or frozen.get("allowed_write") != ALLOWED_WRITE
        or frozen.get("allowed_failure_evidence_path_count") != 12
        or frozen.get("repair_added_unique_identity_budget") != 0
        or frozen.get("repair_added_solver_run_count") != 0
        or frozen.get("automatic_retry_count") != 0
        or frozen.get("completed_identity_recomputation_allowed") is not False
        or frozen.get("authorization_scope_end")
        != AUTHORIZATION_SCOPE_END
    ):
        raise FailureEvidenceSupplementError(
            "failure_evidence_proposal_invalid"
        )
    root = Path(repository_root).resolve()
    bound_files = (
        (
            "short_circuit_proposal_path",
            "short_circuit_proposal_file_sha256",
        ),
        (
            "short_circuit_authorization_path",
            "short_circuit_authorization_file_sha256",
        ),
        (
            "direct_repair_proposal_path",
            "direct_repair_proposal_file_sha256",
        ),
        (
            "direct_repair_authorization_path",
            "direct_repair_authorization_file_sha256",
        ),
        ("scheduler_path", "scheduler_file_sha256"),
        ("spec_path", "spec_file_sha256"),
        ("tool_path", "tool_file_sha256"),
    )
    for path_field, hash_field in bound_files:
        path = (root / str(frozen.get(path_field, ""))).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or file_sha256(path) != frozen.get(hash_field)
        ):
            raise FailureEvidenceSupplementError(
                f"failure_evidence_bound_file_drift:{path_field}"
            )
    short = validate_short_circuit_proposal(
        proposal=read_json(
            root / str(frozen["short_circuit_proposal_path"])
        ),
        repository_root=root,
    )
    short_auth = validate_short_circuit_authorization(
        proposal=short,
        receipt=read_json(
            root / str(frozen["short_circuit_authorization_path"])
        ),
    )
    direct = validate_direct_repair_proposal(
        proposal=read_json(
            root / str(frozen["direct_repair_proposal_path"])
        ),
        repository_root=root,
    )
    direct_auth = validate_direct_repair_authorization(
        proposal=direct,
        receipt=read_json(
            root / str(frozen["direct_repair_authorization_path"])
        ),
    )
    scheduler = (root / str(frozen["scheduler_path"])).resolve()
    if (
        short.get("proposal_sha256")
        != frozen.get("short_circuit_proposal_sha256")
        or short_auth.get("authorization_sha256")
        != frozen.get("short_circuit_authorization_sha256")
        or direct.get("proposal_sha256")
        != frozen.get("direct_repair_proposal_sha256")
        or direct_auth.get("authorization_sha256")
        != frozen.get("direct_repair_authorization_sha256")
        or short.get("direct_repair_proposal_sha256")
        != direct.get("proposal_sha256")
        or short.get("scheduler_path") != frozen.get("scheduler_path")
        or short.get("scheduler_file_sha256") != file_sha256(scheduler)
    ):
        raise FailureEvidenceSupplementError(
            "failure_evidence_parent_binding_drift"
        )
    output = (root / str(frozen.get("execution_output_dir", ""))).resolve()
    if not output.is_dir() or not output.is_relative_to(root):
        raise FailureEvidenceSupplementError(
            "failure_evidence_output_dir_drift"
        )
    expected = expected_failure_evidence_paths(
        short_circuit_proposal=short,
        execution_output_dir=_relative_to_root(output, root),
    )
    if frozen.get("allowed_failure_evidence_paths") != expected:
        raise FailureEvidenceSupplementError(
            "failure_evidence_path_panel_drift"
        )
    for relative in expected:
        path = (root / relative).resolve()
        if not path.is_relative_to(output):
            raise FailureEvidenceSupplementError(
                "failure_evidence_path_outside_output"
            )
        if path.exists():
            validate_observed_failure_evidence(path)
    return frozen


def build_authorization(
    *,
    proposal: Mapping[str, Any],
    approved_at: str,
) -> dict[str, Any]:
    approved = datetime.fromisoformat(approved_at)
    if approved.tzinfo is None:
        raise FailureEvidenceSupplementError(
            "authorization_time_invalid"
        )
    receipt: dict[str, Any] = {
        "authorization_version": AUTHORIZATION_VERSION,
        "status": "authorized",
        "proposal_sha256": proposal["proposal_sha256"],
        "approved_at": approved.isoformat(timespec="seconds"),
        "approved_by": "user",
        "user_authorization_text": USER_AUTHORIZATION_TEXT,
        "authorization_scope_end": AUTHORIZATION_SCOPE_END,
        "allowed_write": ALLOWED_WRITE,
        "repair_added_unique_identity_budget": 0,
        "repair_added_solver_run_count": 0,
        "automatic_retry_count": 0,
        "completed_identity_recomputation_allowed": False,
    }
    receipt["authorization_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_authorization(
    *,
    proposal: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = deepcopy(dict(receipt))
    _verify_embedded_hash(
        frozen,
        hash_field="authorization_sha256",
        artifact_name="failure_evidence_supplement_authorization",
    )
    try:
        approved = datetime.fromisoformat(str(frozen["approved_at"]))
    except (KeyError, TypeError, ValueError) as error:
        raise FailureEvidenceSupplementError(
            "authorization_invalid"
        ) from error
    expected = {
        "authorization_version": AUTHORIZATION_VERSION,
        "status": "authorized",
        "proposal_sha256": proposal["proposal_sha256"],
        "approved_by": "user",
        "user_authorization_text": USER_AUTHORIZATION_TEXT,
        "authorization_scope_end": AUTHORIZATION_SCOPE_END,
        "allowed_write": ALLOWED_WRITE,
        "repair_added_unique_identity_budget": 0,
        "repair_added_solver_run_count": 0,
        "automatic_retry_count": 0,
        "completed_identity_recomputation_allowed": False,
    }
    if (
        approved.tzinfo is None
        or any(frozen.get(key) != value for key, value in expected.items())
    ):
        raise FailureEvidenceSupplementError(
            "authorization_invalid"
        )
    return frozen


def _write_proposal(output_dir: Path, proposal: Mapping[str, Any]) -> None:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "proposal.json", dict(proposal))
    receipt: dict[str, Any] = {
        "receipt_version": PROPOSAL_RECEIPT_VERSION,
        "status": "frozen_zero_solver_runs",
        "proposal_sha256": proposal["proposal_sha256"],
        "allowed_failure_evidence_path_count": proposal[
            "allowed_failure_evidence_path_count"
        ],
        "repair_added_solver_run_count": 0,
        "repair_added_unique_identity_budget": 0,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    atomic_write_json(output / "proposal_receipt.json", receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    propose = commands.add_parser("propose")
    propose.add_argument("--short-circuit-proposal", required=True)
    propose.add_argument("--short-circuit-authorization", required=True)
    propose.add_argument("--direct-repair-proposal", required=True)
    propose.add_argument("--direct-repair-authorization", required=True)
    propose.add_argument("--scheduler", required=True)
    propose.add_argument("--spec", required=True)
    propose.add_argument("--execution-output-dir", required=True)
    propose.add_argument("--repository-root", required=True)
    propose.add_argument("--output-dir", required=True)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--proposal-dir", required=True)
    authorize.add_argument("--approved-at", required=True)
    authorize.add_argument("--repository-root", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--proposal-dir", required=True)
    audit.add_argument("--repository-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "propose":
        proposal = build_proposal(
            short_circuit_proposal_path=Path(
                args.short_circuit_proposal
            ),
            short_circuit_authorization_path=Path(
                args.short_circuit_authorization
            ),
            direct_repair_proposal_path=Path(
                args.direct_repair_proposal
            ),
            direct_repair_authorization_path=Path(
                args.direct_repair_authorization
            ),
            scheduler_path=Path(args.scheduler),
            spec_path=Path(args.spec),
            execution_output_dir=Path(args.execution_output_dir),
            repository_root=Path(args.repository_root),
            tool_path=Path(__file__),
        )
        validate_proposal(
            proposal=proposal,
            repository_root=Path(args.repository_root),
        )
        _write_proposal(Path(args.output_dir), proposal)
        print(
            {
                "status": proposal["status"],
                "proposal_sha256": proposal["proposal_sha256"],
                "allowed_failure_evidence_path_count": proposal[
                    "allowed_failure_evidence_path_count"
                ],
                "repair_added_solver_run_count": 0,
            }
        )
        return 0
    proposal_dir = Path(args.proposal_dir).resolve()
    proposal = validate_proposal(
        proposal=read_json(proposal_dir / "proposal.json"),
        repository_root=Path(args.repository_root),
    )
    if args.command == "authorize":
        receipt = build_authorization(
            proposal=proposal,
            approved_at=args.approved_at,
        )
        validate_authorization(proposal=proposal, receipt=receipt)
        atomic_write_json(proposal_dir / "authorization.json", receipt)
        print(
            {
                "status": receipt["status"],
                "authorization_sha256": receipt[
                    "authorization_sha256"
                ],
                "authorization_scope_end": AUTHORIZATION_SCOPE_END,
            }
        )
        return 0
    authorization = validate_authorization(
        proposal=proposal,
        receipt=read_json(proposal_dir / "authorization.json"),
    )
    root = Path(args.repository_root).resolve()
    observed = sum(
        (root / path).is_file()
        for path in proposal["allowed_failure_evidence_paths"]
    )
    print(
        {
            "status": "valid",
            "proposal_sha256": proposal["proposal_sha256"],
            "authorization_sha256": authorization[
                "authorization_sha256"
            ],
            "observed_failure_evidence_count": observed,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
