"""Zero-solver completion repair for the short-circuit scheduler.

The numerical cell is already complete when ``_execute_search_cell`` fails
while hashing a runtime ``mappingproxy``.  This module accepts only the exact
direct scheduler traceback, verifies all 150 registered identities and frozen
JSON artifacts, then reconstructs the missing completion and repair receipt.
It never invokes a numerical runner or registers a retry.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from recovery_independent_bo_supervisor import (
    MAPPINGPROXY_FAILURE_SIGNATURE,
    RepairContractError,
    _cell_dir,
    _exclusive_supervisor_lock,
    _list,
    _mapping,
    _relative_to_root,
    _require_runner_stopped,
    _try_exclusive_file_lock,
    _validate_ready_artifacts,
    _verify_embedded_hash,
    build_cell_completion_payload,
)

from ppg_hr.v2.experiment_freeze_utils import file_sha256
from ppg_hr.v2.phase2_experiment_io import atomic_write_json, read_json
from ppg_hr.v2.recovery_contracts import canonical_sha256
from ppg_hr.v2.recovery_experiment_governance import (
    AttemptRegistry,
    BudgetContract,
)
from ppg_hr.v2.recovery_independent_bo_experiment import (
    _exploration_from_payload,
    validate_recovery_independent_bo_preflight,
)

DIRECT_REPAIR_PROPOSAL_VERSION = (
    "lyx_recovery_short_circuit_mappingproxy_direct_repair_proposal_v1"
)
DIRECT_REPAIR_AUTHORIZATION_VERSION = (
    "lyx_recovery_short_circuit_mappingproxy_direct_repair_authorization_v1"
)
DIRECT_REPAIR_RECEIPT_VERSION = (
    "lyx_recovery_short_circuit_mappingproxy_direct_repair_receipt_v1"
)
DIRECT_FAILURE_CONTEXT = "short_circuit_direct_execute_search_cell_v1"
USER_AUTHORIZATION_TEXT = (
    "批准编写并执行短路调度器直接调用栈的零运行替换修复 proposal，批准生成绑定修复后源码的"
    "新短路 proposal，并授权继续执行 Gate A 和必要的 Gate B，直至本轮大实验完成；不得重算"
    "已完成身份或自动重试失败身份。"
)


def is_short_circuit_mappingproxy_completion_failure(
    stderr_text: str,
) -> bool:
    """Accept only the exact direct-scheduler post-search hash failure."""

    ordered_context = (
        "recovery_short_circuit_runner.py",
        "in execute_gate_a",
        "in _execute_or_repair_cell",
        "in _execute_search_cell",
        '"seed_stability_audit_sha256": canonical_sha256(',
    )
    stripped = stderr_text.strip()
    lines = stripped.splitlines()
    traceback_marker = "Traceback (most recent call last):"
    chained_markers = (
        "During handling of the above exception",
        "The above exception was the direct cause",
    )
    if (
        not lines
        or lines.count(traceback_marker) != 1
        or lines[-1] != MAPPINGPROXY_FAILURE_SIGNATURE
        or any(marker in stripped for marker in chained_markers)
    ):
        return False
    exception_lines = [
        line.strip()
        for line in lines
        if re.match(
            r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):",
            line.strip(),
        )
    ]
    if exception_lines != [MAPPINGPROXY_FAILURE_SIGNATURE]:
        return False
    position = lines.index(traceback_marker)
    for token in ordered_context:
        try:
            position = next(
                index
                for index in range(position + 1, len(lines))
                if token in lines[index]
            )
        except StopIteration:
            return False
    return position < len(lines) - 1


def _covered_cells(
    *,
    original_proposal: Mapping[str, Any],
    short_circuit_proposal: Mapping[str, Any],
) -> list[dict[str, Any]]:
    original_by_hash = {
        str(cell["cell_sha256"]): _mapping("direct_original_cell", cell)
        for cell in _list(
            "direct_original_cells",
            original_proposal.get("search_cells"),
        )
    }
    bindings = _list(
        "direct_gate_a_bindings",
        short_circuit_proposal.get("gate_a_cell_bindings"),
    )
    covered: list[dict[str, Any]] = []
    for raw in bindings:
        binding = _mapping("direct_gate_a_binding", raw)
        cell = original_by_hash.get(str(binding.get("cell_sha256")))
        if (
            cell is None
            or binding.get("record_id") != cell.get("record_id")
            or binding.get("recovery_candidate_id")
            != cell.get("recovery_candidate_id")
            or binding.get("unique_budget") != 150
        ):
            raise RepairContractError(
                "direct_mappingproxy_gate_a_binding_drift"
            )
        covered.append(
            {
                "cell_sha256": cell["cell_sha256"],
                "record_id": cell["record_id"],
                "scene": cell["scene"],
                "recovery_candidate_id": cell[
                    "recovery_candidate_id"
                ],
            }
        )
    coordinates = {
        (item["recovery_candidate_id"], item["record_id"])
        for item in covered
    }
    if len(covered) != 12 or len(coordinates) != 12:
        raise RepairContractError(
            "direct_mappingproxy_covered_panel_invalid"
        )
    return covered


def _incident_artifacts(
    *,
    covered_cells: Sequence[Mapping[str, Any]],
    execution_dir: Path,
    repository_root: Path,
) -> list[dict[str, Any]]:
    incidents: list[dict[str, Any]] = []
    for cell in covered_cells:
        cell_dir = _cell_dir(execution_dir, cell)
        completion_path = cell_dir / "cell_completion.json"
        state_path = cell_dir / "search" / "driver_state.json"
        results_path = cell_dir / "candidate_results.json"
        audit_path = cell_dir / "seed_stability_audit.json"
        if completion_path.is_file() or not all(
            path.is_file()
            for path in (state_path, results_path, audit_path)
        ):
            continue
        if read_json(state_path).get("stage") != "complete":
            continue
        incidents.append(
            {
                "cell_sha256": cell["cell_sha256"],
                "record_id": cell["record_id"],
                "recovery_candidate_id": cell[
                    "recovery_candidate_id"
                ],
                "driver_state_path": _relative_to_root(
                    state_path,
                    repository_root,
                ),
                "driver_state_file_sha256": file_sha256(state_path),
                "candidate_results_path": _relative_to_root(
                    results_path,
                    repository_root,
                ),
                "candidate_results_file_sha256": file_sha256(
                    results_path
                ),
                "seed_stability_audit_path": _relative_to_root(
                    audit_path,
                    repository_root,
                ),
                "seed_stability_audit_file_sha256": file_sha256(
                    audit_path
                ),
            }
        )
    return incidents


def build_direct_repair_proposal(
    *,
    original_proposal_path: Path,
    short_circuit_proposal_path: Path,
    short_circuit_authorization_path: Path,
    failure_log_path: Path,
    spec_path: Path,
    scheduler_path: Path,
    tool_path: Path,
    governance_dir: Path,
    execution_dir: Path,
    repository_root: Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    original_path = Path(original_proposal_path).resolve()
    short_path = Path(short_circuit_proposal_path).resolve()
    short_auth_path = Path(short_circuit_authorization_path).resolve()
    failure_path = Path(failure_log_path).resolve()
    spec = Path(spec_path).resolve()
    scheduler = Path(scheduler_path).resolve()
    tool = Path(tool_path).resolve()
    governance = Path(governance_dir).resolve()
    execution = Path(execution_dir).resolve()
    original = read_json(original_path)
    validate_recovery_independent_bo_preflight(
        proposal=original,
        repository_root=root,
    )
    short = read_json(short_path)
    short_auth = read_json(short_auth_path)
    _verify_embedded_hash(
        short,
        hash_field="proposal_sha256",
        artifact_name="direct_short_circuit_proposal",
    )
    _verify_embedded_hash(
        short_auth,
        hash_field="authorization_sha256",
        artifact_name="direct_short_circuit_authorization",
    )
    if (
        short_auth.get("proposal_sha256")
        != short.get("proposal_sha256")
        or _relative_to_root(scheduler, root)
        != short.get("scheduler_path")
    ):
        raise RepairContractError(
            "direct_mappingproxy_short_circuit_binding_drift"
        )
    if not is_short_circuit_mappingproxy_completion_failure(
        failure_path.read_text(encoding="utf-8", errors="replace")
    ):
        raise RepairContractError(
            "direct_mappingproxy_failure_signature_not_observed"
        )
    covered = _covered_cells(
        original_proposal=original,
        short_circuit_proposal=short,
    )
    incidents = _incident_artifacts(
        covered_cells=covered,
        execution_dir=execution,
        repository_root=root,
    )
    if len(incidents) != 1:
        raise RepairContractError(
            "direct_mappingproxy_requires_one_incident_cell"
        )
    budget_path = governance / "budget_contract.json"
    registry_path = governance / "attempt_registry.json"
    if not budget_path.is_file() or not registry_path.is_file():
        raise RepairContractError(
            "direct_mappingproxy_governance_binding_missing"
        )
    proposal: dict[str, Any] = {
        "proposal_version": DIRECT_REPAIR_PROPOSAL_VERSION,
        "status": "frozen_zero_solver_runs",
        "failure_context": DIRECT_FAILURE_CONTEXT,
        "failure_signature": MAPPINGPROXY_FAILURE_SIGNATURE,
        "original_proposal_sha256": original["proposal_sha256"],
        "original_proposal_path": _relative_to_root(
            original_path, root
        ),
        "original_proposal_file_sha256": file_sha256(original_path),
        "short_circuit_proposal_sha256": short["proposal_sha256"],
        "short_circuit_proposal_path": _relative_to_root(
            short_path, root
        ),
        "short_circuit_proposal_file_sha256": file_sha256(short_path),
        "short_circuit_authorization_path": _relative_to_root(
            short_auth_path, root
        ),
        "short_circuit_authorization_file_sha256": file_sha256(
            short_auth_path
        ),
        "failure_log_path": _relative_to_root(failure_path, root),
        "failure_log_file_sha256": file_sha256(failure_path),
        "spec_path": _relative_to_root(spec, root),
        "spec_file_sha256": file_sha256(spec),
        "failed_scheduler_file_sha256": short[
            "scheduler_file_sha256"
        ],
        "scheduler_path": _relative_to_root(scheduler, root),
        "scheduler_file_sha256": file_sha256(scheduler),
        "repair_tool_path": _relative_to_root(tool, root),
        "repair_tool_file_sha256": file_sha256(tool),
        "governance_dir": _relative_to_root(governance, root),
        "governance_budget_contract_file_sha256": file_sha256(
            budget_path
        ),
        "execution_dir": _relative_to_root(execution, root),
        "covered_cell_count": len(covered),
        "covered_cell_identities": covered,
        "covered_cell_identity_panel_sha256": canonical_sha256(
            covered
        ),
        "incident_artifacts": incidents,
        "allowed_write": (
            "missing_gate_a_cell_completion_and_paired_repair_receipt"
        ),
        "repair_added_unique_identity_budget": 0,
        "repair_added_solver_run_count": 0,
        "automatic_retry_count": 0,
        "completed_identity_recomputation_allowed": False,
        "authorization_scope_end": "experiment_complete",
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "next_state": "awaiting_user_authorization",
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def validate_direct_repair_proposal(
    *,
    proposal: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    frozen = deepcopy(dict(proposal))
    _verify_embedded_hash(
        frozen,
        hash_field="proposal_sha256",
        artifact_name="direct_mappingproxy_repair_proposal",
    )
    if (
        frozen.get("proposal_version")
        != DIRECT_REPAIR_PROPOSAL_VERSION
        or frozen.get("status") != "frozen_zero_solver_runs"
        or frozen.get("failure_context") != DIRECT_FAILURE_CONTEXT
        or frozen.get("failure_signature")
        != MAPPINGPROXY_FAILURE_SIGNATURE
        or frozen.get("covered_cell_count") != 12
        or frozen.get("allowed_write")
        != "missing_gate_a_cell_completion_and_paired_repair_receipt"
        or frozen.get("repair_added_unique_identity_budget") != 0
        or frozen.get("repair_added_solver_run_count") != 0
        or frozen.get("automatic_retry_count") != 0
        or frozen.get("completed_identity_recomputation_allowed")
        is not False
        or frozen.get("authorization_scope_end")
        != "experiment_complete"
    ):
        raise RepairContractError(
            "direct_mappingproxy_repair_proposal_invalid"
        )
    root = Path(repository_root).resolve()
    bound_files = (
        ("original_proposal_path", "original_proposal_file_sha256"),
        (
            "short_circuit_proposal_path",
            "short_circuit_proposal_file_sha256",
        ),
        (
            "short_circuit_authorization_path",
            "short_circuit_authorization_file_sha256",
        ),
        ("failure_log_path", "failure_log_file_sha256"),
        ("spec_path", "spec_file_sha256"),
        ("scheduler_path", "scheduler_file_sha256"),
        ("repair_tool_path", "repair_tool_file_sha256"),
    )
    for path_field, hash_field in bound_files:
        path = (root / str(frozen.get(path_field, ""))).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or file_sha256(path) != frozen.get(hash_field)
        ):
            raise RepairContractError(
                f"direct_mappingproxy_bound_file_drift:{path_field}"
            )
    failure_path = (
        root / str(frozen["failure_log_path"])
    ).resolve()
    if not is_short_circuit_mappingproxy_completion_failure(
        failure_path.read_text(encoding="utf-8", errors="replace")
    ):
        raise RepairContractError(
            "direct_mappingproxy_failure_evidence_drift"
        )
    original = read_json(
        (root / str(frozen["original_proposal_path"])).resolve()
    )
    short = read_json(
        (root / str(frozen["short_circuit_proposal_path"])).resolve()
    )
    short_auth = read_json(
        (
            root
            / str(frozen["short_circuit_authorization_path"])
        ).resolve()
    )
    _verify_embedded_hash(
        short,
        hash_field="proposal_sha256",
        artifact_name="direct_bound_short_proposal",
    )
    _verify_embedded_hash(
        short_auth,
        hash_field="authorization_sha256",
        artifact_name="direct_bound_short_authorization",
    )
    if (
        original.get("proposal_sha256")
        != frozen.get("original_proposal_sha256")
        or short.get("proposal_sha256")
        != frozen.get("short_circuit_proposal_sha256")
        or short_auth.get("proposal_sha256")
        != short.get("proposal_sha256")
        or frozen.get("failed_scheduler_file_sha256")
        != short.get("scheduler_file_sha256")
    ):
        raise RepairContractError(
            "direct_mappingproxy_parent_binding_drift"
        )
    covered = _covered_cells(
        original_proposal=original,
        short_circuit_proposal=short,
    )
    if (
        frozen.get("covered_cell_identities") != covered
        or frozen.get("covered_cell_identity_panel_sha256")
        != canonical_sha256(covered)
    ):
        raise RepairContractError(
            "direct_mappingproxy_covered_panel_drift"
        )
    incidents = _list(
        "direct_mappingproxy_incident_artifacts",
        frozen.get("incident_artifacts"),
    )
    if len(incidents) != 1:
        raise RepairContractError(
            "direct_mappingproxy_incident_count_invalid"
        )
    incident = _mapping("direct_mappingproxy_incident", incidents[0])
    covered_by_hash = {
        str(item["cell_sha256"]): item for item in covered
    }
    cell = covered_by_hash.get(str(incident.get("cell_sha256")))
    if (
        cell is None
        or incident.get("record_id") != cell.get("record_id")
        or incident.get("recovery_candidate_id")
        != cell.get("recovery_candidate_id")
    ):
        raise RepairContractError(
            "direct_mappingproxy_incident_cell_drift"
        )
    execution = (root / str(frozen["execution_dir"])).resolve()
    incident_bindings = (
        (
            "driver_state_path",
            "driver_state_file_sha256",
            Path("search") / "driver_state.json",
        ),
        (
            "candidate_results_path",
            "candidate_results_file_sha256",
            Path("candidate_results.json"),
        ),
        (
            "seed_stability_audit_path",
            "seed_stability_audit_file_sha256",
            Path("seed_stability_audit.json"),
        ),
    )
    for path_field, hash_field, relative_path in incident_bindings:
        path = (root / str(incident.get(path_field, ""))).resolve()
        if (
            path != (_cell_dir(execution, cell) / relative_path).resolve()
            or not path.is_file()
            or file_sha256(path) != incident.get(hash_field)
        ):
            raise RepairContractError(
                "direct_mappingproxy_incident_artifact_drift:"
                + path_field
            )
    incident_state = read_json(
        (root / str(incident["driver_state_path"])).resolve()
    )
    incident_results = read_json(
        (root / str(incident["candidate_results_path"])).resolve()
    )
    if (
        incident_state.get("stage") != "complete"
        or incident_results.get("proposal_sha256")
        != original.get("proposal_sha256")
        or incident_results.get("cell_sha256")
        != incident.get("cell_sha256")
        or incident_results.get("result_count") != 150
    ):
        raise RepairContractError(
            "direct_mappingproxy_incident_semantic_drift"
        )
    governance = (root / str(frozen["governance_dir"])).resolve()
    budget_path = governance / "budget_contract.json"
    if (
        not budget_path.is_file()
        or file_sha256(budget_path)
        != frozen.get("governance_budget_contract_file_sha256")
        or read_json(budget_path)
        != BudgetContract.proposed_v13_recovery_independent_bo().to_dict()
    ):
        raise RepairContractError(
            "direct_mappingproxy_governance_drift"
        )
    return frozen


def build_direct_repair_authorization(
    *,
    proposal: Mapping[str, Any],
    approved_at: str,
) -> dict[str, Any]:
    approved = datetime.fromisoformat(approved_at)
    if approved.tzinfo is None:
        raise RepairContractError(
            "direct_mappingproxy_authorization_time_invalid"
        )
    receipt: dict[str, Any] = {
        "authorization_version": DIRECT_REPAIR_AUTHORIZATION_VERSION,
        "status": "authorized",
        "proposal_sha256": proposal["proposal_sha256"],
        "approved_at": approved.isoformat(timespec="seconds"),
        "approved_by": "user",
        "user_authorization_text": USER_AUTHORIZATION_TEXT,
        "authorization_scope_end": "experiment_complete",
        "allowed_write": proposal["allowed_write"],
        "repair_added_unique_identity_budget": 0,
        "repair_added_solver_run_count": 0,
        "automatic_retry_count": 0,
        "completed_identity_recomputation_allowed": False,
    }
    receipt["authorization_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_direct_repair_authorization(
    *,
    proposal: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = deepcopy(dict(receipt))
    _verify_embedded_hash(
        frozen,
        hash_field="authorization_sha256",
        artifact_name="direct_mappingproxy_authorization",
    )
    try:
        approved = datetime.fromisoformat(str(frozen["approved_at"]))
    except (KeyError, TypeError, ValueError) as error:
        raise RepairContractError(
            "direct_mappingproxy_authorization_invalid"
        ) from error
    expected = {
        "authorization_version": DIRECT_REPAIR_AUTHORIZATION_VERSION,
        "status": "authorized",
        "proposal_sha256": proposal["proposal_sha256"],
        "approved_by": "user",
        "user_authorization_text": USER_AUTHORIZATION_TEXT,
        "authorization_scope_end": "experiment_complete",
        "allowed_write": (
            "missing_gate_a_cell_completion_and_paired_repair_receipt"
        ),
        "repair_added_unique_identity_budget": 0,
        "repair_added_solver_run_count": 0,
        "automatic_retry_count": 0,
        "completed_identity_recomputation_allowed": False,
    }
    if (
        approved.tzinfo is None
        or any(frozen.get(key) != value for key, value in expected.items())
    ):
        raise RepairContractError(
            "direct_mappingproxy_authorization_invalid"
        )
    return frozen


def _build_direct_completion(
    *,
    original: Mapping[str, Any],
    cell: Mapping[str, Any],
    results: Mapping[str, Any],
    seed_audit: Mapping[str, Any],
    matrix: Mapping[str, Any],
    repair_proposal: Mapping[str, Any],
    repair_authorization: Mapping[str, Any],
    failure_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    completion = build_cell_completion_payload(
        proposal=original,
        cell=cell,
        candidate_results=results,
        seed_audit=seed_audit,
        matrix=matrix,
        repair_proposal=repair_proposal,
        repair_authorization=repair_authorization,
        observed_failure_log_path=failure_path,
        repository_root=repository_root,
    )
    reporting = dict(
        _mapping("direct_completion_reporting", completion["reporting_repair"])
    )
    reporting["repair_version"] = DIRECT_REPAIR_RECEIPT_VERSION
    reporting["failure_context"] = DIRECT_FAILURE_CONTEXT
    completion["reporting_repair"] = reporting
    completion.pop("completion_sha256", None)
    completion["completion_sha256"] = canonical_sha256(completion)
    return completion


def _build_direct_receipt(
    *,
    cell: Mapping[str, Any],
    cell_dir: Path,
    completion: Mapping[str, Any],
    repair_proposal: Mapping[str, Any],
    repair_authorization: Mapping[str, Any],
    failure_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    reporting = _mapping(
        "direct_receipt_reporting",
        completion.get("reporting_repair"),
    )
    receipt: dict[str, Any] = {
        "receipt_version": DIRECT_REPAIR_RECEIPT_VERSION,
        "status": "cell_completion_repaired",
        "failure_context": DIRECT_FAILURE_CONTEXT,
        "cell_sha256": cell["cell_sha256"],
        "record_id": cell["record_id"],
        "recovery_candidate_id": cell["recovery_candidate_id"],
        "repair_proposal_sha256": repair_proposal["proposal_sha256"],
        "repair_authorization_sha256": repair_authorization[
            "authorization_sha256"
        ],
        "candidate_results_file_sha256": file_sha256(
            cell_dir / "candidate_results.json"
        ),
        "seed_stability_audit_file_sha256": file_sha256(
            cell_dir / "seed_stability_audit.json"
        ),
        "observed_failure_log_path": _relative_to_root(
            failure_path, repository_root
        ),
        "observed_failure_log_file_sha256": file_sha256(failure_path),
        "cell_completion_sha256": completion["completion_sha256"],
        "repair_added_solver_run_count": 0,
        "repair_added_unique_identity_count": 0,
        "original_runner_cell_solver_attempt_count": reporting[
            "original_runner_cell_solver_attempt_count"
        ],
        "original_runner_cell_cache_only_identity_count": reporting[
            "original_runner_cell_cache_only_identity_count"
        ],
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_existing_direct_completion(
    *,
    completion_path: Path,
    cell: Mapping[str, Any],
    original_proposal: Mapping[str, Any],
    repair_proposal: Mapping[str, Any],
    repair_authorization: Mapping[str, Any],
    repository_root: Path,
) -> None:
    completion = read_json(completion_path)
    _verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="direct_existing_completion",
    )
    reporting = _mapping(
        "direct_existing_reporting",
        completion.get("reporting_repair"),
    )
    receipt_path = completion_path.parent / "cell_completion_repair_receipt.json"
    if not receipt_path.is_file():
        raise RepairContractError("direct_mappingproxy_receipt_missing")
    receipt = read_json(receipt_path)
    _verify_embedded_hash(
        receipt,
        hash_field="receipt_sha256",
        artifact_name="direct_existing_receipt",
    )
    expected = {
        "cell_sha256": cell["cell_sha256"],
        "record_id": cell["record_id"],
        "recovery_candidate_id": cell["recovery_candidate_id"],
        "repair_proposal_sha256": repair_proposal["proposal_sha256"],
        "repair_authorization_sha256": repair_authorization[
            "authorization_sha256"
        ],
        "cell_completion_sha256": completion["completion_sha256"],
        "repair_added_solver_run_count": 0,
        "repair_added_unique_identity_count": 0,
        "failure_context": DIRECT_FAILURE_CONTEXT,
    }
    if (
        completion.get("proposal_sha256")
        != original_proposal.get("proposal_sha256")
        or completion.get("cell_sha256") != cell.get("cell_sha256")
        or completion.get("unique_candidate_count") != 150
        or receipt.get("receipt_version")
        != DIRECT_REPAIR_RECEIPT_VERSION
        or receipt.get("status") != "cell_completion_repaired"
        or any(receipt.get(key) != value for key, value in expected.items())
        or reporting.get("repair_version")
        != DIRECT_REPAIR_RECEIPT_VERSION
        or reporting.get("failure_context") != DIRECT_FAILURE_CONTEXT
        or reporting.get("failure_signature")
        != MAPPINGPROXY_FAILURE_SIGNATURE
        or reporting.get("repair_proposal_sha256")
        != repair_proposal.get("proposal_sha256")
        or reporting.get("repair_authorization_sha256")
        != repair_authorization.get("authorization_sha256")
        or reporting.get("repair_added_solver_run_count") != 0
        or reporting.get("repair_added_unique_identity_count") != 0
        or reporting.get("source_artifacts_were_json_round_tripped")
        is not True
    ):
        raise RepairContractError(
            "direct_mappingproxy_existing_pair_drift"
        )
    cell_dir = completion_path.parent
    if (
        receipt.get("candidate_results_file_sha256")
        != file_sha256(cell_dir / "candidate_results.json")
        or receipt.get("seed_stability_audit_file_sha256")
        != file_sha256(cell_dir / "seed_stability_audit.json")
    ):
        raise RepairContractError(
            "direct_mappingproxy_existing_source_drift"
        )
    root = Path(repository_root).resolve()
    failure_path = (
        root / str(receipt.get("observed_failure_log_path", ""))
    ).resolve()
    if (
        not failure_path.is_relative_to(root)
        or not failure_path.is_file()
        or file_sha256(failure_path)
        != receipt.get("observed_failure_log_file_sha256")
        or reporting.get("observed_failure_log_path")
        != receipt.get("observed_failure_log_path")
        or reporting.get("observed_failure_log_file_sha256")
        != receipt.get("observed_failure_log_file_sha256")
        or not is_short_circuit_mappingproxy_completion_failure(
            failure_path.read_text(encoding="utf-8", errors="replace")
        )
    ):
        raise RepairContractError(
            "direct_mappingproxy_existing_failure_evidence_drift"
        )


def finalize_direct_ready_cell_locked(
    *,
    original_proposal_path: Path,
    repair_proposal_path: Path,
    repair_authorization_path: Path,
    governance_dir: Path,
    execution_dir: Path,
    repository_root: Path,
    observed_failure_log_path: Path,
    expected_cell_sha256: str,
) -> dict[str, Any]:
    """Finalize exactly one covered ready cell without numerical work."""

    _require_runner_stopped(execution_dir)
    root = Path(repository_root).resolve()
    failure_path = Path(observed_failure_log_path).resolve()
    if (
        not failure_path.is_file()
        or not is_short_circuit_mappingproxy_completion_failure(
            failure_path.read_text(encoding="utf-8", errors="replace")
        )
    ):
        raise RepairContractError(
            "direct_mappingproxy_observed_failure_invalid"
        )
    original = read_json(Path(original_proposal_path).resolve())
    validate_recovery_independent_bo_preflight(
        proposal=original,
        repository_root=root,
    )
    repair_proposal = validate_direct_repair_proposal(
        proposal=read_json(Path(repair_proposal_path).resolve()),
        repository_root=root,
    )
    repair_authorization = validate_direct_repair_authorization(
        proposal=repair_proposal,
        receipt=read_json(Path(repair_authorization_path).resolve()),
    )
    if repair_proposal.get("original_proposal_sha256") != original.get(
        "proposal_sha256"
    ):
        raise RepairContractError(
            "direct_mappingproxy_original_proposal_drift"
        )
    execution = Path(execution_dir).resolve()
    if (
        root / str(repair_proposal.get("execution_dir"))
    ).resolve() != execution:
        raise RepairContractError(
            "direct_mappingproxy_execution_binding_drift"
        )
    covered_hashes = {
        str(item["cell_sha256"])
        for item in _list(
            "direct_mappingproxy_covered_cells",
            repair_proposal.get("covered_cell_identities"),
        )
    }
    ready: list[tuple[Mapping[str, Any], Path]] = []
    for raw in _list("direct_original_cells", original.get("search_cells")):
        cell = _mapping("direct_original_cell", raw)
        if str(cell.get("cell_sha256")) not in covered_hashes:
            continue
        cell_dir = _cell_dir(execution, cell)
        completion_path = cell_dir / "cell_completion.json"
        if completion_path.is_file():
            continue
        required = (
            cell_dir / "search" / "driver_state.json",
            cell_dir / "candidate_results.json",
            cell_dir / "seed_stability_audit.json",
        )
        if all(path.is_file() for path in required) and read_json(
            required[0]
        ).get("stage") == "complete":
            ready.append((cell, cell_dir))
    if len(ready) != 1:
        raise RepairContractError(
            "direct_mappingproxy_expected_one_ready_cell"
        )
    cell, cell_dir = ready[0]
    if str(cell.get("cell_sha256")) != expected_cell_sha256:
        raise RepairContractError(
            "direct_mappingproxy_unexpected_ready_cell"
        )
    with _try_exclusive_file_lock(
        cell_dir / "search" / ".driver.lock"
    ) as acquired:
        if not acquired:
            raise RepairContractError(
                "direct_mappingproxy_search_driver_active"
            )
        _require_runner_stopped(execution)
        governance = Path(governance_dir).resolve()
        budget = BudgetContract.proposed_v13_recovery_independent_bo()
        exploration = _exploration_from_payload(
            read_json(governance / "exploration_registry.json")
        )
        registry = AttemptRegistry.open(
            governance / "attempt_registry.json",
            budget_contract=budget,
            exploration_registry=exploration,
        )
        results, seed_audit, identities = _validate_ready_artifacts(
            proposal=original,
            cell=cell,
            cell_dir=cell_dir,
        )
        registry.assert_complete_matrix(identities)
        matrix = registry.matrix_execution_summary(identities)
        completion = _build_direct_completion(
            original=original,
            cell=cell,
            results=results,
            seed_audit=seed_audit,
            matrix=matrix,
            repair_proposal=repair_proposal,
            repair_authorization=repair_authorization,
            failure_path=failure_path,
            repository_root=root,
        )
        receipt = _build_direct_receipt(
            cell=cell,
            cell_dir=cell_dir,
            completion=completion,
            repair_proposal=repair_proposal,
            repair_authorization=repair_authorization,
            failure_path=failure_path,
            repository_root=root,
        )
        receipt_path = cell_dir / "cell_completion_repair_receipt.json"
        if receipt_path.is_file():
            if read_json(receipt_path) != receipt:
                raise RepairContractError(
                    "direct_mappingproxy_pending_receipt_drift"
                )
        else:
            atomic_write_json(receipt_path, receipt)
        completion_path = cell_dir / "cell_completion.json"
        atomic_write_json(completion_path, completion)
        validate_existing_direct_completion(
            completion_path=completion_path,
            cell=cell,
            original_proposal=original,
            repair_proposal=repair_proposal,
            repair_authorization=repair_authorization,
            repository_root=root,
        )
        return receipt


def _write_proposal(
    *,
    proposal: Mapping[str, Any],
    output_dir: Path,
) -> None:
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            "direct_mappingproxy_proposal_dir_exists:" + str(destination)
        )
    destination.mkdir(parents=True)
    atomic_write_json(destination / "proposal.json", dict(proposal))
    receipt: dict[str, Any] = {
        "receipt_version": (
            "lyx_recovery_short_circuit_mappingproxy_direct_"
            "repair_proposal_receipt_v1"
        ),
        "status": "awaiting_user_authorization",
        "proposal_sha256": proposal["proposal_sha256"],
        "repair_added_unique_identity_budget": 0,
        "repair_added_solver_run_count": 0,
        "created_at": proposal["created_at"],
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    atomic_write_json(destination / "proposal_receipt.json", receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    propose = commands.add_parser("propose")
    for name in (
        "original_proposal",
        "short_circuit_proposal",
        "short_circuit_authorization",
        "failure_log",
        "spec",
        "scheduler",
        "governance_dir",
        "execution_dir",
        "repository_root",
        "output_dir",
    ):
        propose.add_argument(f"--{name.replace('_', '-')}", required=True)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--proposal-dir", required=True)
    authorize.add_argument("--approved-at", required=True)
    authorize.add_argument("--repository-root", required=True)
    finalize = commands.add_parser("finalize")
    for name in (
        "original_proposal",
        "proposal_dir",
        "failure_log",
        "governance_dir",
        "execution_dir",
        "repository_root",
        "expected_cell_sha256",
    ):
        finalize.add_argument(f"--{name.replace('_', '-')}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "propose":
        proposal = build_direct_repair_proposal(
            original_proposal_path=Path(args.original_proposal),
            short_circuit_proposal_path=Path(args.short_circuit_proposal),
            short_circuit_authorization_path=Path(
                args.short_circuit_authorization
            ),
            failure_log_path=Path(args.failure_log),
            spec_path=Path(args.spec),
            scheduler_path=Path(args.scheduler),
            tool_path=Path(__file__),
            governance_dir=Path(args.governance_dir),
            execution_dir=Path(args.execution_dir),
            repository_root=Path(args.repository_root),
        )
        _write_proposal(proposal=proposal, output_dir=Path(args.output_dir))
        print(
            json.dumps(
                {
                    "status": proposal["status"],
                    "proposal_sha256": proposal["proposal_sha256"],
                    "repair_added_unique_identity_budget": 0,
                    "repair_added_solver_run_count": 0,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "authorize":
        proposal_dir = Path(args.proposal_dir).resolve()
        proposal = validate_direct_repair_proposal(
            proposal=read_json(proposal_dir / "proposal.json"),
            repository_root=Path(args.repository_root),
        )
        receipt = build_direct_repair_authorization(
            proposal=proposal,
            approved_at=args.approved_at,
        )
        path = proposal_dir / "authorization.json"
        if path.exists():
            raise FileExistsError(
                "direct_mappingproxy_authorization_exists:" + str(path)
            )
        atomic_write_json(path, receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "authorization_sha256": receipt[
                        "authorization_sha256"
                    ],
                    "authorization_scope_end": (
                        receipt["authorization_scope_end"]
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 0
    proposal_dir = Path(args.proposal_dir).resolve()
    with _exclusive_supervisor_lock(Path(args.execution_dir).resolve()):
        receipt = finalize_direct_ready_cell_locked(
            original_proposal_path=Path(args.original_proposal),
            repair_proposal_path=proposal_dir / "proposal.json",
            repair_authorization_path=proposal_dir / "authorization.json",
            governance_dir=Path(args.governance_dir),
            execution_dir=Path(args.execution_dir),
            repository_root=Path(args.repository_root),
            observed_failure_log_path=Path(args.failure_log),
            expected_cell_sha256=args.expected_cell_sha256,
        )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
