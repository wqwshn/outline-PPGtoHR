"""Fail-closed supervisor for the independent-BO mappingproxy receipt defect.

This tool deliberately lives outside ``python/src``.  It does not change the
frozen numerical or search source identity.  It may only reconstruct a missing
``cell_completion.json`` after the original runner has completed every
identity in that cell and exited with the exact mappingproxy hashing error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import psutil

from ppg_hr.v2.bo_space_generalization import (
    _try_exclusive_file_lock,
    build_bo_search_space,
)
from ppg_hr.v2.experiment_freeze_utils import file_sha256
from ppg_hr.v2.phase2_experiment_io import atomic_write_json, read_json
from ppg_hr.v2.recovery_contracts import canonical_sha256
from ppg_hr.v2.recovery_experiment_governance import (
    AttemptRegistry,
    BudgetContract,
)
from ppg_hr.v2.recovery_independent_bo_experiment import (
    BLANKET_AUTHORIZATION_EXPIRES_AT,
    BLANKET_AUTHORIZATION_USER_TEXT,
    EXPECTED_SEARCH_CELL_COUNT,
    _attempt_identity_from_item,
    _exploration_from_payload,
    build_recovery_independent_bo_identity,
    validate_recovery_independent_bo_execution_authorization,
    validate_recovery_independent_bo_preflight,
)

REPAIR_PROPOSAL_VERSION = (
    "lyx_recovery_independent_bo_mappingproxy_repair_v1"
)
REPAIR_AUTHORIZATION_VERSION = (
    "lyx_recovery_independent_bo_mappingproxy_repair_authorization_v1"
)
REPAIR_RECEIPT_VERSION = (
    "lyx_recovery_independent_bo_cell_completion_repair_v1"
)
MAPPINGPROXY_FAILURE_SIGNATURE = (
    "TypeError: Object of type mappingproxy is not JSON serializable"
)


class RepairContractError(RuntimeError):
    """The zero-run receipt repair violates its frozen contract."""


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RepairContractError(f"{name}_must_be_object")
    return value


def _list(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise RepairContractError(f"{name}_must_be_list")
    return value


def _relative_to_root(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    repository_root = Path(root).resolve()
    if not resolved.is_relative_to(repository_root):
        raise RepairContractError(
            f"repair_path_outside_repository:{resolved}"
        )
    return resolved.relative_to(repository_root).as_posix()


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    artifact_name: str,
) -> str:
    declared = payload.get(hash_field)
    if not isinstance(declared, str) or len(declared) != 64:
        raise RepairContractError(
            f"{artifact_name}_{hash_field}_missing"
        )
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != hash_field
    }
    if canonical_sha256(unsigned) != declared:
        raise RepairContractError(
            f"{artifact_name}_{hash_field}_mismatch"
        )
    return declared


@contextmanager
def _exclusive_supervisor_lock(
    execution_dir: Path,
) -> Iterator[None]:
    """Hold one OS lock for the complete supervise-or-repair protocol."""

    root = Path(execution_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".mappingproxy_repair_supervisor.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+b") as handle:
        if lock_path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(
                    handle.fileno(),
                    msvcrt.LK_NBLCK,
                    1,
                )
            except OSError as error:
                raise RepairContractError(
                    "mappingproxy_supervisor_already_running"
                ) from error
        else:
            import fcntl

            try:
                fcntl.flock(
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError as error:
                raise RepairContractError(
                    "mappingproxy_supervisor_already_running"
                ) from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(
                    handle.fileno(),
                    msvcrt.LK_UNLCK,
                    1,
                )
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _matching_external_runner_pids(
    execution_dir: Path,
) -> tuple[int, ...]:
    """Find original runner processes targeting this execution root."""

    expected = Path(execution_dir).resolve()
    matches: list[int] = []
    for process in psutil.process_iter(
        attrs=("pid", "cmdline", "cwd"),
    ):
        try:
            cmdline = process.info.get("cmdline") or []
            if (
                "ppg_hr.v2.recovery_independent_bo_runner"
                not in cmdline
                or "execute" not in cmdline
                or "--output-dir" not in cmdline
            ):
                continue
            output_index = cmdline.index("--output-dir") + 1
            if output_index >= len(cmdline):
                continue
            raw_output = Path(cmdline[output_index])
            if raw_output.is_absolute():
                actual = raw_output.resolve()
            else:
                raw_cwd = process.info.get("cwd")
                if not raw_cwd:
                    matches.append(int(process.info["pid"]))
                    continue
                actual = (Path(raw_cwd) / raw_output).resolve()
            if actual == expected:
                matches.append(int(process.info["pid"]))
        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            ValueError,
        ):
            continue
    return tuple(sorted(matches))


def _require_runner_stopped(execution_dir: Path) -> None:
    pids = _matching_external_runner_pids(execution_dir)
    if pids:
        raise RepairContractError(
            "mappingproxy_original_runner_still_running:"
            + ",".join(str(pid) for pid in pids)
        )


def is_mappingproxy_completion_failure(stderr_text: str) -> bool:
    """Accept only the observed post-search receipt hashing failure."""

    ordered_context = (
        "recovery_independent_bo_runner.py",
        "execute_recovery_independent_bo_proposal",
        "_execute_search_cell",
        '"seed_stability_audit_sha256": canonical_sha256(',
    )
    stripped = stderr_text.strip()
    lines = stripped.splitlines()
    chained_markers = (
        "During handling of the above exception",
        "The above exception was the direct cause",
    )
    traceback_marker = "Traceback (most recent call last):"
    if (
        lines.count(traceback_marker) != 1
        or not lines
        or lines[-1] != MAPPINGPROXY_FAILURE_SIGNATURE
        or any(marker in stripped for marker in chained_markers)
    ):
        return False
    traceback_index = lines.index(traceback_marker)
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
    position = traceback_index
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


def _incident_artifacts(
    *,
    original_proposal: Mapping[str, Any],
    execution_dir: Path | None,
    repository_root: Path,
) -> list[dict[str, Any]]:
    if execution_dir is None:
        return []
    root = Path(execution_dir).resolve()
    incidents: list[dict[str, Any]] = []
    for raw_cell in _list(
        "repair_search_cells",
        original_proposal.get("search_cells"),
    ):
        cell = _mapping("repair_search_cell", raw_cell)
        cell_dir = (
            root
            / "cells"
            / str(cell["recovery_candidate_id"])
            / str(cell["record_id"])
        )
        completion_path = cell_dir / "cell_completion.json"
        state_path = cell_dir / "search" / "driver_state.json"
        results_path = cell_dir / "candidate_results.json"
        audit_path = cell_dir / "seed_stability_audit.json"
        if completion_path.is_file():
            continue
        if not (
            state_path.is_file()
            and results_path.is_file()
            and audit_path.is_file()
        ):
            continue
        state = read_json(state_path)
        if state.get("stage") != "complete":
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


def _covered_cell_identities(
    original_proposal: Mapping[str, Any],
) -> list[dict[str, Any]]:
    panel = [
        {
            "cell_sha256": cell["cell_sha256"],
            "record_id": cell["record_id"],
            "scene": cell["scene"],
            "recovery_candidate_id": cell[
                "recovery_candidate_id"
            ],
        }
        for cell in (
            _mapping("repair_covered_cell", raw)
            for raw in _list(
                "repair_covered_cells",
                original_proposal.get("search_cells"),
            )
        )
    ]
    coordinates = {
        (
            str(item["recovery_candidate_id"]),
            str(item["record_id"]),
        )
        for item in panel
    }
    hashes = {str(item["cell_sha256"]) for item in panel}
    if (
        len(panel) != EXPECTED_SEARCH_CELL_COUNT
        or len(coordinates) != EXPECTED_SEARCH_CELL_COUNT
        or len(hashes) != EXPECTED_SEARCH_CELL_COUNT
    ):
        raise RepairContractError(
            "mappingproxy_repair_cell_panel_invalid"
        )
    return panel


def build_repair_proposal(
    *,
    original_proposal_path: Path,
    original_authorization_path: Path,
    failure_log_path: Path,
    tool_path: Path,
    repository_root: Path,
    execution_dir: Path,
    governance_dir: Path,
) -> dict[str, Any]:
    """Freeze a zero-solver repair for missing cell completion receipts."""

    root = Path(repository_root).resolve()
    original_path = Path(original_proposal_path).resolve()
    original_authorization = Path(
        original_authorization_path
    ).resolve()
    failure_path = Path(failure_log_path).resolve()
    repair_tool_path = Path(tool_path).resolve()
    bound_governance_dir = Path(governance_dir).resolve()
    original = read_json(original_path)
    stderr_text = failure_path.read_text(encoding="utf-8")
    if not is_mappingproxy_completion_failure(stderr_text):
        raise RepairContractError(
            "repair_failure_signature_not_observed"
        )
    cells = _covered_cell_identities(original)
    incidents = _incident_artifacts(
        original_proposal=original,
        execution_dir=execution_dir,
        repository_root=root,
    )
    if len(incidents) != 1:
        raise RepairContractError(
            "mappingproxy_repair_requires_one_incident_cell"
        )
    governance_budget = bound_governance_dir / "budget_contract.json"
    governance_authorization = (
        bound_governance_dir / "execution_authorization.json"
    )
    if not (
        original_authorization.is_file()
        and governance_budget.is_file()
        and governance_authorization.is_file()
    ):
        raise RepairContractError(
            "mappingproxy_repair_governance_binding_missing"
        )
    proposal: dict[str, Any] = {
        "proposal_version": REPAIR_PROPOSAL_VERSION,
        "status": "frozen_zero_solver_runs",
        "original_proposal_sha256": original[
            "proposal_sha256"
        ],
        "original_proposal_path": _relative_to_root(
            original_path,
            root,
        ),
        "original_proposal_file_sha256": file_sha256(
            original_path
        ),
        "original_authorization_path": _relative_to_root(
            original_authorization,
            root,
        ),
        "original_authorization_file_sha256": file_sha256(
            original_authorization
        ),
        "failure_signature": MAPPINGPROXY_FAILURE_SIGNATURE,
        "failure_log_path": _relative_to_root(
            failure_path,
            root,
        ),
        "failure_log_file_sha256": file_sha256(failure_path),
        "repair_tool_path": _relative_to_root(
            repair_tool_path,
            root,
        ),
        "repair_tool_file_sha256": file_sha256(
            repair_tool_path
        ),
        "execution_dir": _relative_to_root(
            Path(execution_dir),
            root,
        ),
        "governance_dir": _relative_to_root(
            bound_governance_dir,
            root,
        ),
        "governance_budget_contract_file_sha256": file_sha256(
            governance_budget
        ),
        "governance_execution_authorization_file_sha256": (
            file_sha256(governance_authorization)
        ),
        "covered_cell_count": len(cells),
        "covered_cell_identities": cells,
        "covered_cell_identity_panel_sha256": canonical_sha256(
            cells
        ),
        "incident_artifacts": incidents,
        "allowed_read": [
            "completed_search_driver_state",
            "candidate_results_json",
            "seed_stability_audit_json",
            "attempt_registry_after_runner_exit",
        ],
        "allowed_write": "missing_cell_completion_only",
        "original_runner_unique_identity_budget": int(
            original["unique_budget"]
        ),
        "original_runner_may_continue": True,
        "repair_added_unique_identity_budget": 0,
        "repair_added_solver_run_count": 0,
        "frozen_runtime_source_change_count": 0,
        "auxiliary_supervisor_tool_count": 1,
        "automatic_retry_count": 0,
        "expected_runner_resume_count": len(cells),
        "failure_policy": (
            "any_non_mappingproxy_runner_failure_stops_supervisor"
        ),
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def validate_repair_proposal(
    *,
    proposal: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    frozen = deepcopy(dict(proposal))
    _verify_embedded_hash(
        frozen,
        hash_field="proposal_sha256",
        artifact_name="mappingproxy_repair_proposal",
    )
    if (
        frozen.get("proposal_version")
        != REPAIR_PROPOSAL_VERSION
        or frozen.get("status") != "frozen_zero_solver_runs"
        or frozen.get("covered_cell_count")
        != EXPECTED_SEARCH_CELL_COUNT
        or frozen.get("allowed_write")
        != "missing_cell_completion_only"
        or frozen.get("repair_added_unique_identity_budget") != 0
        or frozen.get("repair_added_solver_run_count") != 0
        or frozen.get("original_runner_may_continue") is not True
        or frozen.get("original_runner_unique_identity_budget")
        != 5400
        or frozen.get("frozen_runtime_source_change_count") != 0
        or frozen.get("auxiliary_supervisor_tool_count") != 1
        or frozen.get("automatic_retry_count") != 0
        or frozen.get("failure_signature")
        != MAPPINGPROXY_FAILURE_SIGNATURE
    ):
        raise RepairContractError(
            "mappingproxy_repair_proposal_invalid"
        )
    root = Path(repository_root).resolve()
    bound_files = (
        ("original_proposal_path", "original_proposal_file_sha256"),
        (
            "original_authorization_path",
            "original_authorization_file_sha256",
        ),
        ("failure_log_path", "failure_log_file_sha256"),
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
                f"mappingproxy_repair_bound_file_drift:{path_field}"
            )
    failure_path = (
        root / str(frozen["failure_log_path"])
    ).resolve()
    if not is_mappingproxy_completion_failure(
        failure_path.read_text(encoding="utf-8")
    ):
        raise RepairContractError(
            "mappingproxy_repair_failure_evidence_drift"
        )
    original_path = (
        root / str(frozen["original_proposal_path"])
    ).resolve()
    original = read_json(original_path)
    if original.get("proposal_sha256") != frozen.get(
        "original_proposal_sha256"
    ):
        raise RepairContractError(
            "mappingproxy_repair_original_proposal_drift"
        )
    original_authorization_path = (
        root / str(frozen["original_authorization_path"])
    ).resolve()
    original_authorization = (
        validate_recovery_independent_bo_execution_authorization(
            original,
            receipt=read_json(original_authorization_path),
        )
    )
    execution_root = (
        root / str(frozen.get("execution_dir", ""))
    ).resolve()
    governance_root = (
        root / str(frozen.get("governance_dir", ""))
    ).resolve()
    if (
        not execution_root.is_relative_to(root)
        or not governance_root.is_relative_to(root)
        or not governance_root.is_dir()
    ):
        raise RepairContractError(
            "mappingproxy_repair_execution_binding_invalid"
        )
    governance_budget_path = (
        governance_root / "budget_contract.json"
    )
    governance_authorization_path = (
        governance_root / "execution_authorization.json"
    )
    if (
        not governance_budget_path.is_file()
        or file_sha256(governance_budget_path)
        != frozen.get(
            "governance_budget_contract_file_sha256"
        )
        or read_json(governance_budget_path)
        != BudgetContract.proposed_v13_recovery_independent_bo().to_dict()
        or not governance_authorization_path.is_file()
        or file_sha256(governance_authorization_path)
        != frozen.get(
            "governance_execution_authorization_file_sha256"
        )
        or read_json(governance_authorization_path)
        != original_authorization
    ):
        raise RepairContractError(
            "mappingproxy_repair_governance_drift"
        )
    expected_panel = _covered_cell_identities(original)
    actual_panel = _list(
        "mappingproxy_repair_covered_cell_identities",
        frozen.get("covered_cell_identities"),
    )
    if (
        actual_panel != expected_panel
        or frozen.get("covered_cell_identity_panel_sha256")
        != canonical_sha256(expected_panel)
    ):
        raise RepairContractError(
            "mappingproxy_repair_cell_panel_drift"
        )
    incidents = _list(
        "mappingproxy_repair_incident_artifacts",
        frozen.get("incident_artifacts"),
    )
    if len(incidents) != 1:
        raise RepairContractError(
            "mappingproxy_repair_incident_count_invalid"
        )
    incident = _mapping(
        "mappingproxy_repair_incident_artifact",
        incidents[0],
    )
    panel_by_hash = {
        str(item["cell_sha256"]): item
        for item in expected_panel
    }
    panel_item = panel_by_hash.get(str(incident.get("cell_sha256")))
    if (
        panel_item is None
        or incident.get("record_id") != panel_item["record_id"]
        or incident.get("recovery_candidate_id")
        != panel_item["recovery_candidate_id"]
    ):
        raise RepairContractError(
            "mappingproxy_repair_incident_cell_drift"
        )
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
    incident_cell_dir = _cell_dir(execution_root, panel_item)
    for path_field, hash_field, relative_path in incident_bindings:
        incident_path = (
            root / str(incident.get(path_field, ""))
        ).resolve()
        if (
            incident_path
            != (incident_cell_dir / relative_path).resolve()
            or not incident_path.is_file()
            or file_sha256(incident_path)
            != incident.get(hash_field)
        ):
            raise RepairContractError(
                "mappingproxy_repair_incident_artifact_drift:"
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
    ):
        raise RepairContractError(
            "mappingproxy_repair_incident_semantic_drift"
        )
    return frozen


def build_repair_authorization(
    *,
    proposal: Mapping[str, Any],
    approved_at: str,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "authorization_version": REPAIR_AUTHORIZATION_VERSION,
        "approved": True,
        "proposal_sha256": proposal["proposal_sha256"],
        "original_proposal_sha256": proposal[
            "original_proposal_sha256"
        ],
        "allowed_write": proposal["allowed_write"],
        "repair_added_unique_identity_budget": proposal[
            "repair_added_unique_identity_budget"
        ],
        "repair_added_solver_run_count": proposal[
            "repair_added_solver_run_count"
        ],
        "original_runner_unique_identity_budget": proposal[
            "original_runner_unique_identity_budget"
        ],
        "approved_at": approved_at,
        "approved_by": "user",
        "authorization_basis": (
            "blanket_proposal_authorization_until_deadline"
        ),
        "blanket_authorization_expires_at": (
            BLANKET_AUTHORIZATION_EXPIRES_AT
        ),
        "user_authorization": BLANKET_AUTHORIZATION_USER_TEXT,
    }
    receipt["authorization_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_repair_authorization(
    *,
    proposal: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = deepcopy(dict(receipt))
    try:
        _verify_embedded_hash(
            frozen,
            hash_field="authorization_sha256",
            artifact_name="mappingproxy_repair_authorization",
        )
        approved_at = datetime.fromisoformat(
            str(frozen["approved_at"])
        )
        deadline = datetime.fromisoformat(
            BLANKET_AUTHORIZATION_EXPIRES_AT
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RepairContractError(
            "repair_authorization_invalid"
        ) from error
    expected = {
        "authorization_version": REPAIR_AUTHORIZATION_VERSION,
        "approved": True,
        "proposal_sha256": proposal["proposal_sha256"],
        "original_proposal_sha256": proposal[
            "original_proposal_sha256"
        ],
        "allowed_write": "missing_cell_completion_only",
        "repair_added_unique_identity_budget": 0,
        "repair_added_solver_run_count": 0,
        "original_runner_unique_identity_budget": 5400,
        "approved_by": "user",
        "authorization_basis": (
            "blanket_proposal_authorization_until_deadline"
        ),
        "blanket_authorization_expires_at": (
            BLANKET_AUTHORIZATION_EXPIRES_AT
        ),
        "user_authorization": BLANKET_AUTHORIZATION_USER_TEXT,
    }
    if (
        any(frozen.get(key) != value for key, value in expected.items())
        or approved_at.tzinfo is None
        or approved_at.utcoffset() != deadline.utcoffset()
        or approved_at >= deadline
    ):
        raise RepairContractError(
            "repair_authorization_invalid"
        )
    return frozen


def build_cell_completion_payload(
    *,
    proposal: Mapping[str, Any],
    cell: Mapping[str, Any],
    candidate_results: Mapping[str, Any],
    seed_audit: Mapping[str, Any],
    matrix: Mapping[str, Any],
    repair_proposal: Mapping[str, Any],
    repair_authorization: Mapping[str, Any],
    observed_failure_log_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Build the original completion shape from JSON-safe artifacts."""

    expected_count = int(
        _mapping(
            "repair_seed_manifest",
            proposal.get("seed_manifest"),
        )["unique_budget_per_cell"]
    )
    rows = [
        dict(_mapping("repair_candidate_row", row))
        for row in _list(
            "repair_candidate_rows",
            candidate_results.get("results"),
        )
    ]
    candidate_ids = [str(row["candidate_id"]) for row in rows]
    if (
        candidate_results.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or candidate_results.get("cell_sha256")
        != cell.get("cell_sha256")
        or candidate_results.get("result_count") != expected_count
        or len(rows) != expected_count
        or len(set(candidate_ids)) != expected_count
        or int(seed_audit.get("global_candidate_count", -1))
        != expected_count
    ):
        raise RepairContractError(
            "mappingproxy_repair_cell_artifact_mismatch"
        )
    if (
        int(matrix.get("planned_identity_count", -1))
        != expected_count
        or int(
            matrix.get(
                "identity_with_solver_attempt_count",
                -1,
            )
        )
        + int(matrix.get("cache_only_identity_count", -1))
        != expected_count
        or int(matrix.get("failed_attempt_count", -1)) != 0
        or int(matrix.get("retry_count", -1)) != 0
    ):
        raise RepairContractError(
            "mappingproxy_repair_matrix_incomplete"
        )
    selected = min(
        rows,
        key=lambda row: (
            not bool(row["eligible"]),
            float(row["objective"]),
            str(row["candidate_id"]),
        ),
    )
    stability_ids = _list(
        "repair_seed_stability_candidate_ids",
        seed_audit.get("seed_stability_candidate_ids"),
    )
    completion: dict[str, Any] = {
        "completion_version": (
            "lyx_recovery_independent_bo_cell_completion_v1"
        ),
        "status": "complete",
        "proposal_sha256": proposal["proposal_sha256"],
        "cell_sha256": cell["cell_sha256"],
        "record_id": cell["record_id"],
        "scene": cell["scene"],
        "recovery_candidate_id": cell[
            "recovery_candidate_id"
        ],
        "unique_candidate_count": len(rows),
        "eligible_candidate_count": sum(
            bool(row["eligible"]) for row in rows
        ),
        "seed_stability_candidate_count": len(stability_ids),
        "selected": selected,
        "matrix_execution_summary": dict(matrix),
        "candidate_results_sha256": candidate_results[
            "result_sha256"
        ],
        "seed_stability_audit_sha256": canonical_sha256(
            dict(seed_audit)
        ),
        "reporting_repair": {
            "repair_version": REPAIR_RECEIPT_VERSION,
            "repair_proposal_sha256": repair_proposal[
                "proposal_sha256"
            ],
            "repair_authorization_sha256": (
                repair_authorization["authorization_sha256"]
            ),
            "failure_signature": MAPPINGPROXY_FAILURE_SIGNATURE,
            "observed_failure_log_path": _relative_to_root(
                observed_failure_log_path,
                repository_root,
            ),
            "observed_failure_log_file_sha256": file_sha256(
                observed_failure_log_path
            ),
            "source_artifacts_were_json_round_tripped": True,
            "repair_added_solver_run_count": 0,
            "repair_added_unique_identity_count": 0,
            "original_runner_cell_solver_attempt_count": int(
                matrix["identity_with_solver_attempt_count"]
            ),
            "original_runner_cell_cache_only_identity_count": int(
                matrix["cache_only_identity_count"]
            ),
        },
    }
    completion["completion_sha256"] = canonical_sha256(completion)
    return completion


def _validate_ready_artifacts(
    *,
    proposal: Mapping[str, Any],
    cell: Mapping[str, Any],
    cell_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[Any, ...],
]:
    state = read_json(cell_dir / "search" / "driver_state.json")
    results = read_json(cell_dir / "candidate_results.json")
    seed_audit = read_json(
        cell_dir / "seed_stability_audit.json"
    )
    if state.get("stage") != "complete":
        raise RepairContractError(
            "mappingproxy_repair_search_not_complete"
        )
    _verify_embedded_hash(
        results,
        hash_field="result_sha256",
        artifact_name="mappingproxy_repair_candidate_results",
    )
    rows = [
        _mapping("mappingproxy_repair_candidate_row", row)
        for row in _list(
            "mappingproxy_repair_candidate_rows",
            results.get("results"),
        )
    ]
    state_ids = {
        str(value)
        for value in _list(
            "mappingproxy_repair_global_candidate_ids",
            state.get("global_candidate_ids"),
        )
    }
    row_ids = {str(row["candidate_id"]) for row in rows}
    audit_ids = {
        str(value)
        for value in _list(
            "mappingproxy_repair_stability_ids",
            seed_audit.get("seed_stability_candidate_ids"),
        )
    }
    expected_count = int(
        _mapping(
            "mappingproxy_repair_seed_manifest",
            proposal["seed_manifest"],
        )["unique_budget_per_cell"]
    )
    if (
        state.get("requested_global_unique_budget")
        != expected_count
        or state.get("effective_global_unique_budget")
        != expected_count
        or len(state_ids) != expected_count
        or row_ids != state_ids
        or not audit_ids.issubset(row_ids)
        or seed_audit.get("global_candidate_count")
        != expected_count
        or seed_audit.get("requested_global_unique_budget")
        != expected_count
        or seed_audit.get("effective_global_unique_budget")
        != expected_count
        or results.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or results.get("cell_sha256") != cell.get("cell_sha256")
    ):
        raise RepairContractError(
            "mappingproxy_repair_ready_artifact_invalid"
        )
    candidates = {
        candidate.candidate_id: candidate
        for candidate in build_bo_search_space(
            "physical_v1"
        ).candidates
    }
    identities = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise RepairContractError(
                "mappingproxy_repair_candidate_unknown"
            )
        expected_identity = build_recovery_independent_bo_identity(
            proposal=proposal,
            cell=cell,
            candidate=candidate,
        )
        identity_payload = dict(
            _mapping(
                "mappingproxy_repair_identity_item",
                row["identity"],
            )
        )
        identity = _attempt_identity_from_item(
            identity_payload
        )
        if (
            identity_payload != expected_identity
            or row.get("identity_sha256") != identity.sha256
            or candidate_id != identity_payload["bo_candidate_id"]
        ):
            raise RepairContractError(
                "mappingproxy_repair_identity_drift"
            )
        identities.append(identity)
    if len({identity.sha256 for identity in identities}) != expected_count:
        raise RepairContractError(
            "mappingproxy_repair_identity_count_mismatch"
        )
    return results, seed_audit, tuple(identities)


def _cell_dir(execution_dir: Path, cell: Mapping[str, Any]) -> Path:
    return (
        Path(execution_dir).resolve()
        / "cells"
        / str(cell["recovery_candidate_id"])
        / str(cell["record_id"])
    )


def _ready_missing_cell_sha256s(
    *,
    original_proposal: Mapping[str, Any],
    execution_dir: Path,
) -> tuple[str, ...]:
    ready: list[str] = []
    for raw_cell in _list(
        "mappingproxy_repair_ready_scan_cells",
        original_proposal.get("search_cells"),
    ):
        cell = _mapping(
            "mappingproxy_repair_ready_scan_cell",
            raw_cell,
        )
        cell_dir = _cell_dir(execution_dir, cell)
        if (cell_dir / "cell_completion.json").is_file():
            continue
        required = (
            cell_dir / "search" / "driver_state.json",
            cell_dir / "candidate_results.json",
            cell_dir / "seed_stability_audit.json",
        )
        if not all(path.is_file() for path in required):
            continue
        if read_json(required[0]).get("stage") == "complete":
            ready.append(str(cell["cell_sha256"]))
    return tuple(ready)


def _pending_repair_evidence(
    *,
    original_proposal: Mapping[str, Any],
    execution_dir: Path,
    repository_root: Path,
) -> tuple[str, Path] | None:
    """Return the one receipt-first, completion-missing repair."""

    root = Path(repository_root).resolve()
    pending: list[tuple[str, Path]] = []
    for raw_cell in _list(
        "mappingproxy_repair_pending_scan_cells",
        original_proposal.get("search_cells"),
    ):
        cell = _mapping(
            "mappingproxy_repair_pending_scan_cell",
            raw_cell,
        )
        cell_dir = _cell_dir(execution_dir, cell)
        completion_path = cell_dir / "cell_completion.json"
        receipt_path = (
            cell_dir / "cell_completion_repair_receipt.json"
        )
        if completion_path.is_file() or not receipt_path.is_file():
            continue
        receipt = read_json(receipt_path)
        _verify_embedded_hash(
            receipt,
            hash_field="receipt_sha256",
            artifact_name="mappingproxy_repair_pending_receipt",
        )
        if (
            receipt.get("receipt_version") != REPAIR_RECEIPT_VERSION
            or receipt.get("status") != "cell_completion_repaired"
            or receipt.get("cell_sha256") != cell.get("cell_sha256")
            or receipt.get("record_id") != cell.get("record_id")
            or receipt.get("recovery_candidate_id")
            != cell.get("recovery_candidate_id")
        ):
            raise RepairContractError(
                "mappingproxy_repair_pending_receipt_cell_drift"
            )
        failure_path = (
            root
            / str(receipt.get("observed_failure_log_path", ""))
        ).resolve()
        if (
            not failure_path.is_relative_to(root)
            or not failure_path.is_file()
            or file_sha256(failure_path)
            != receipt.get(
                "observed_failure_log_file_sha256"
            )
            or not is_mappingproxy_completion_failure(
                failure_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )
        ):
            raise RepairContractError(
                "mappingproxy_repair_pending_failure_evidence_drift"
            )
        pending.append((str(cell["cell_sha256"]), failure_path))
    if len(pending) > 1:
        raise RepairContractError(
            "mappingproxy_repair_multiple_pending_receipts"
        )
    return pending[0] if pending else None


def _observed_failure_state_evidence(
    *,
    state_path: Path,
    original_proposal: Mapping[str, Any],
    repository_root: Path,
) -> tuple[str, Path] | None:
    if not Path(state_path).is_file():
        return None
    state = read_json(state_path)
    if state.get("status") != "mappingproxy_failure_observed":
        return None
    _verify_embedded_hash(
        state,
        hash_field="state_sha256",
        artifact_name="mappingproxy_repair_supervisor_state",
    )
    cell_sha256 = str(state.get("ready_cell_sha256", ""))
    cell_hashes = {
        str(
            _mapping(
                "mappingproxy_repair_state_cell",
                raw_cell,
            )["cell_sha256"]
        )
        for raw_cell in _list(
            "mappingproxy_repair_state_cells",
            original_proposal.get("search_cells"),
        )
    }
    root = Path(repository_root).resolve()
    failure_path = (
        root
        / str(state.get("observed_failure_log_path", ""))
    ).resolve()
    if (
        cell_sha256 not in cell_hashes
        or not failure_path.is_relative_to(root)
        or not failure_path.is_file()
        or file_sha256(failure_path)
        != state.get("observed_failure_log_file_sha256")
        or not is_mappingproxy_completion_failure(
            failure_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
    ):
        raise RepairContractError(
            "mappingproxy_repair_supervisor_state_drift"
        )
    return cell_sha256, failure_path


def _promote_finished_launch_state(
    *,
    state_path: Path,
    original_proposal: Mapping[str, Any],
    execution_dir: Path,
    repository_root: Path,
) -> bool:
    """Recover the exit-to-observed-state crash window."""

    path = Path(state_path)
    if not path.is_file():
        return False
    state = read_json(path)
    if state.get("status") != "launching_original_runner":
        return False
    _verify_embedded_hash(
        state,
        hash_field="state_sha256",
        artifact_name="mappingproxy_repair_launching_state",
    )
    try:
        attempt_number = int(state["attempt_number"])
    except (KeyError, TypeError, ValueError) as error:
        raise RepairContractError(
            "mappingproxy_repair_launching_state_invalid"
        ) from error
    if attempt_number <= 0:
        raise RepairContractError(
            "mappingproxy_repair_launching_state_invalid"
        )
    stderr_path = (
        Path(execution_dir).resolve()
        / "supervisor_attempts"
        / f"attempt_{attempt_number:03d}.stderr.log"
    )
    ready_cell_sha256s = _ready_missing_cell_sha256s(
        original_proposal=original_proposal,
        execution_dir=execution_dir,
    )
    if not stderr_path.is_file():
        if ready_cell_sha256s:
            raise RepairContractError(
                "mappingproxy_repair_launching_state_missing_log"
            )
        return False
    stderr_text = stderr_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    if is_mappingproxy_completion_failure(stderr_text):
        if len(ready_cell_sha256s) != 1:
            raise RepairContractError(
                "mappingproxy_repair_launching_state_ready_mismatch"
            )
        observed: dict[str, Any] = {
            "state_version": (
                "lyx_recovery_independent_bo_supervisor_state_v1"
            ),
            "status": "mappingproxy_failure_observed",
            "attempt_number": attempt_number,
            "runner_returncode_observed": False,
            "ready_cell_sha256": ready_cell_sha256s[0],
            "observed_failure_log_path": _relative_to_root(
                stderr_path,
                repository_root,
            ),
            "observed_failure_log_file_sha256": file_sha256(
                stderr_path
            ),
        }
        observed["state_sha256"] = canonical_sha256(observed)
        atomic_write_json(path, observed)
        return True
    if stderr_text.strip() or ready_cell_sha256s:
        raise RepairContractError(
            "mappingproxy_repair_unrecoverable_launching_state"
        )
    return False


def _build_repair_receipt(
    *,
    cell: Mapping[str, Any],
    cell_dir: Path,
    completion: Mapping[str, Any],
    repair_proposal: Mapping[str, Any],
    repair_authorization: Mapping[str, Any],
    observed_failure_log_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    reporting = _mapping(
        "mappingproxy_repair_receipt_reporting",
        completion.get("reporting_repair"),
    )
    repair_receipt: dict[str, Any] = {
        "receipt_version": REPAIR_RECEIPT_VERSION,
        "status": "cell_completion_repaired",
        "cell_sha256": cell["cell_sha256"],
        "record_id": cell["record_id"],
        "recovery_candidate_id": cell["recovery_candidate_id"],
        "repair_proposal_sha256": repair_proposal[
            "proposal_sha256"
        ],
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
            observed_failure_log_path,
            repository_root,
        ),
        "observed_failure_log_file_sha256": file_sha256(
            observed_failure_log_path
        ),
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
    repair_receipt["receipt_sha256"] = canonical_sha256(
        repair_receipt
    )
    return repair_receipt


def _validate_existing_completion(
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
        artifact_name="mappingproxy_repair_existing_completion",
    )
    if (
        completion.get("proposal_sha256")
        != original_proposal.get("proposal_sha256")
        or completion.get("cell_sha256") != cell.get("cell_sha256")
        or completion.get("unique_candidate_count") != 150
    ):
        raise RepairContractError(
            "mappingproxy_repair_existing_completion_drift"
        )
    receipt_path = (
        completion_path.parent
        / "cell_completion_repair_receipt.json"
    )
    reporting_repair = completion.get("reporting_repair")
    if reporting_repair is None:
        if receipt_path.exists():
            raise RepairContractError(
                "mappingproxy_repair_orphan_receipt"
            )
        return
    reporting = _mapping(
        "mappingproxy_repair_existing_reporting_repair",
        reporting_repair,
    )
    if not receipt_path.is_file():
        raise RepairContractError(
            "mappingproxy_repair_receipt_missing"
        )
    receipt = read_json(receipt_path)
    _verify_embedded_hash(
        receipt,
        hash_field="receipt_sha256",
        artifact_name="mappingproxy_repair_existing_receipt",
    )
    expected_pairs = {
        "cell_sha256": cell["cell_sha256"],
        "record_id": cell["record_id"],
        "recovery_candidate_id": cell["recovery_candidate_id"],
        "repair_proposal_sha256": repair_proposal[
            "proposal_sha256"
        ],
        "repair_authorization_sha256": repair_authorization[
            "authorization_sha256"
        ],
        "cell_completion_sha256": completion[
            "completion_sha256"
        ],
        "repair_added_solver_run_count": 0,
        "repair_added_unique_identity_count": 0,
        "original_runner_cell_solver_attempt_count": reporting.get(
            "original_runner_cell_solver_attempt_count"
        ),
        "original_runner_cell_cache_only_identity_count": reporting.get(
            "original_runner_cell_cache_only_identity_count"
        ),
    }
    if (
        receipt.get("receipt_version") != REPAIR_RECEIPT_VERSION
        or receipt.get("status") != "cell_completion_repaired"
        or any(
            receipt.get(key) != value
            for key, value in expected_pairs.items()
        )
        or reporting.get("repair_proposal_sha256")
        != expected_pairs["repair_proposal_sha256"]
        or reporting.get("repair_authorization_sha256")
        != expected_pairs["repair_authorization_sha256"]
        or reporting.get("repair_version") != REPAIR_RECEIPT_VERSION
        or reporting.get("failure_signature")
        != MAPPINGPROXY_FAILURE_SIGNATURE
        or reporting.get("source_artifacts_were_json_round_tripped")
        is not True
        or reporting.get("repair_added_solver_run_count") != 0
        or reporting.get("repair_added_unique_identity_count") != 0
    ):
        raise RepairContractError(
            "mappingproxy_repair_existing_pair_drift"
        )
    cell_dir = completion_path.parent
    current_artifact_hashes = {
        "candidate_results_file_sha256": file_sha256(
            cell_dir / "candidate_results.json"
        ),
        "seed_stability_audit_file_sha256": file_sha256(
            cell_dir / "seed_stability_audit.json"
        ),
    }
    if any(
        receipt.get(key) != value
        for key, value in current_artifact_hashes.items()
    ):
        raise RepairContractError(
            "mappingproxy_repair_existing_source_drift"
        )
    root = Path(repository_root).resolve()
    observed_path = (
        root / str(receipt.get("observed_failure_log_path", ""))
    ).resolve()
    if (
        not observed_path.is_relative_to(root)
        or not observed_path.is_file()
        or file_sha256(observed_path)
        != receipt.get("observed_failure_log_file_sha256")
        or reporting.get("observed_failure_log_path")
        != receipt.get("observed_failure_log_path")
        or reporting.get("observed_failure_log_file_sha256")
        != receipt.get("observed_failure_log_file_sha256")
        or not is_mappingproxy_completion_failure(
            observed_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
    ):
        raise RepairContractError(
            "mappingproxy_repair_existing_failure_evidence_drift"
        )


def _finalize_ready_cell_locked(
    *,
    original_proposal_path: Path,
    repair_proposal_path: Path,
    repair_authorization_path: Path,
    governance_dir: Path,
    execution_dir: Path,
    repository_root: Path,
    observed_failure_log_path: Path,
    require_ready: bool,
    expected_cell_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Finalize at most one ready cell under the supervisor OS lock."""

    _require_runner_stopped(execution_dir)
    failure_path = Path(observed_failure_log_path).resolve()
    if (
        not failure_path.is_file()
        or not is_mappingproxy_completion_failure(
            failure_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
    ):
        raise RepairContractError(
            "mappingproxy_repair_observed_failure_invalid"
        )
    original = read_json(Path(original_proposal_path).resolve())
    validate_recovery_independent_bo_preflight(
        proposal=original,
        repository_root=Path(repository_root),
    )
    repair_proposal = validate_repair_proposal(
        proposal=read_json(Path(repair_proposal_path).resolve()),
        repository_root=Path(repository_root),
    )
    repair_authorization = validate_repair_authorization(
        proposal=repair_proposal,
        receipt=read_json(
            Path(repair_authorization_path).resolve()
        ),
    )
    if repair_proposal.get("original_proposal_sha256") != (
        original.get("proposal_sha256")
    ):
        raise RepairContractError(
            "mappingproxy_repair_proposal_binding_mismatch"
        )
    bound_execution = repair_proposal.get("execution_dir")
    if bound_execution is None or (
        Path(repository_root).resolve() / str(bound_execution)
    ).resolve() != Path(execution_dir).resolve():
        raise RepairContractError(
            "mappingproxy_repair_execution_binding_mismatch"
        )
    ready: list[tuple[Mapping[str, Any], Path]] = []
    for raw_cell in _list(
        "mappingproxy_repair_search_cells",
        original.get("search_cells"),
    ):
        cell = _mapping("mappingproxy_repair_search_cell", raw_cell)
        cell_dir = _cell_dir(Path(execution_dir), cell)
        completion_path = cell_dir / "cell_completion.json"
        if completion_path.is_file():
            _validate_existing_completion(
                completion_path=completion_path,
                cell=cell,
                original_proposal=original,
                repair_proposal=repair_proposal,
                repair_authorization=repair_authorization,
                repository_root=repository_root,
            )
            continue
        required = (
            cell_dir / "search" / "driver_state.json",
            cell_dir / "candidate_results.json",
            cell_dir / "seed_stability_audit.json",
        )
        if not all(path.is_file() for path in required):
            continue
        state = read_json(required[0])
        if state.get("stage") != "complete":
            continue
        ready.append((cell, cell_dir))
    if len(ready) > 1:
        raise RepairContractError(
            "mappingproxy_repair_multiple_ready_cells"
        )
    if not ready:
        if require_ready:
            raise RepairContractError(
                "mappingproxy_repair_expected_one_ready_cell"
            )
        return None
    cell, cell_dir = ready[0]
    if (
        expected_cell_sha256 is not None
        and cell.get("cell_sha256") != expected_cell_sha256
    ):
        raise RepairContractError(
            "mappingproxy_repair_unexpected_ready_cell"
        )
    driver_lock_path = cell_dir / "search" / ".driver.lock"
    with _try_exclusive_file_lock(driver_lock_path) as acquired:
        if not acquired:
            raise RepairContractError(
                "mappingproxy_repair_search_driver_active"
            )
        _require_runner_stopped(execution_dir)
        completion_path = cell_dir / "cell_completion.json"
        if completion_path.is_file():
            _validate_existing_completion(
                completion_path=completion_path,
                cell=cell,
                original_proposal=original,
                repair_proposal=repair_proposal,
                repair_authorization=repair_authorization,
                repository_root=repository_root,
            )
            if require_ready:
                raise RepairContractError(
                    "mappingproxy_repair_ready_cell_changed"
                )
            return None
        required = (
            cell_dir / "search" / "driver_state.json",
            cell_dir / "candidate_results.json",
            cell_dir / "seed_stability_audit.json",
        )
        if (
            not all(path.is_file() for path in required)
            or read_json(required[0]).get("stage") != "complete"
        ):
            raise RepairContractError(
                "mappingproxy_repair_ready_cell_changed"
            )
        governance_root = Path(governance_dir).resolve()
        budget = (
            BudgetContract.proposed_v13_recovery_independent_bo()
        )
        exploration = _exploration_from_payload(
            read_json(
                governance_root / "exploration_registry.json"
            )
        )
        registry = AttemptRegistry.open(
            governance_root / "attempt_registry.json",
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
        repair_receipt = _build_repair_receipt(
            cell=cell,
            cell_dir=cell_dir,
            completion=completion,
            repair_proposal=repair_proposal,
            repair_authorization=repair_authorization,
            observed_failure_log_path=failure_path,
            repository_root=repository_root,
        )
        receipt_path = (
            cell_dir / "cell_completion_repair_receipt.json"
        )
        if receipt_path.is_file():
            existing_receipt = read_json(receipt_path)
            _verify_embedded_hash(
                existing_receipt,
                hash_field="receipt_sha256",
                artifact_name="mappingproxy_repair_pending_receipt",
            )
            if existing_receipt != repair_receipt:
                raise RepairContractError(
                    "mappingproxy_repair_pending_receipt_drift"
                )
        else:
            atomic_write_json(
                receipt_path,
                repair_receipt,
            )
        atomic_write_json(
            completion_path,
            completion,
        )
        _validate_existing_completion(
            completion_path=completion_path,
            cell=cell,
            original_proposal=original,
            repair_proposal=repair_proposal,
            repair_authorization=repair_authorization,
            repository_root=repository_root,
        )
        return repair_receipt


def _run_original_runner(
    *,
    original_proposal_dir: Path,
    governance_dir: Path,
    execution_dir: Path,
    repository_root: Path,
    attempt_number: int,
) -> subprocess.CompletedProcess[bytes]:
    attempts_dir = (
        Path(execution_dir).resolve() / "supervisor_attempts"
    )
    attempts_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = attempts_dir / f"attempt_{attempt_number:03d}.stdout.log"
    stderr_path = attempts_dir / f"attempt_{attempt_number:03d}.stderr.log"
    command = [
        sys.executable,
        "-u",
        "-m",
        "ppg_hr.v2.recovery_independent_bo_runner",
        "execute",
        "--proposal-dir",
        str(Path(original_proposal_dir).resolve()),
        "--governance-dir",
        str(Path(governance_dir).resolve()),
        "--output-dir",
        str(Path(execution_dir).resolve()),
        "--repository-root",
        str(Path(repository_root).resolve()),
    ]
    with stdout_path.open("wb") as stdout_handle, stderr_path.open(
        "wb"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=Path(repository_root).resolve(),
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            env=os.environ.copy(),
        )
    return completed


def supervise(
    *,
    original_proposal_dir: Path,
    repair_proposal_dir: Path,
    governance_dir: Path,
    execution_dir: Path,
    repository_root: Path,
) -> int:
    """Run, repair the one known receipt defect, and resume fail-closed."""

    execution_root = Path(execution_dir).resolve()
    original_path = (
        Path(original_proposal_dir).resolve()
        / "recovery_independent_bo_proposal.json"
    )
    repair_path = (
        Path(repair_proposal_dir).resolve()
        / "mappingproxy_repair_proposal.json"
    )
    authorization_path = (
        Path(repair_proposal_dir).resolve()
        / "repair_authorization.json"
    )
    state_path = (
        execution_root / "supervisor_state.json"
    )
    with _exclusive_supervisor_lock(execution_root):
        _require_runner_stopped(execution_root)
        repair_proposal = validate_repair_proposal(
            proposal=read_json(repair_path),
            repository_root=repository_root,
        )
        original = read_json(original_path)
        cells_by_hash = {
            str(cell["cell_sha256"]): cell
            for cell in (
                _mapping(
                    "mappingproxy_repair_supervisor_cell",
                    raw_cell,
                )
                for raw_cell in _list(
                    "mappingproxy_repair_supervisor_cells",
                    original.get("search_cells"),
                )
            )
        }
        _promote_finished_launch_state(
            state_path=state_path,
            original_proposal=original,
            execution_dir=execution_root,
            repository_root=repository_root,
        )
        pending = _pending_repair_evidence(
            original_proposal=original,
            execution_dir=execution_root,
            repository_root=repository_root,
        )
        if pending is not None:
            pending_cell_sha256, pending_failure_path = pending
            _finalize_ready_cell_locked(
                original_proposal_path=original_path,
                repair_proposal_path=repair_path,
                repair_authorization_path=authorization_path,
                governance_dir=governance_dir,
                execution_dir=execution_root,
                repository_root=repository_root,
                observed_failure_log_path=pending_failure_path,
                require_ready=True,
                expected_cell_sha256=pending_cell_sha256,
            )
        observed_state = _observed_failure_state_evidence(
            state_path=state_path,
            original_proposal=original,
            repository_root=repository_root,
        )
        if observed_state is not None:
            observed_cell_sha256, observed_failure_path = (
                observed_state
            )
            _finalize_ready_cell_locked(
                original_proposal_path=original_path,
                repair_proposal_path=repair_path,
                repair_authorization_path=authorization_path,
                governance_dir=governance_dir,
                execution_dir=execution_root,
                repository_root=repository_root,
                observed_failure_log_path=observed_failure_path,
                require_ready=False,
                expected_cell_sha256=observed_cell_sha256,
            )
            observed_cell = cells_by_hash[observed_cell_sha256]
            observed_completion_path = (
                _cell_dir(execution_root, observed_cell)
                / "cell_completion.json"
            )
            if not observed_completion_path.is_file():
                raise RepairContractError(
                    "mappingproxy_repair_state_not_recovered"
                )
        incident = _mapping(
            "mappingproxy_repair_initial_incident",
            _list(
                "mappingproxy_repair_initial_incidents",
                repair_proposal["incident_artifacts"],
            )[0],
        )
        initial_failure_path = (
            Path(repository_root).resolve()
            / str(repair_proposal["failure_log_path"])
        ).resolve()
        initial_repair = _finalize_ready_cell_locked(
            original_proposal_path=original_path,
            repair_proposal_path=repair_path,
            repair_authorization_path=authorization_path,
            governance_dir=governance_dir,
            execution_dir=execution_root,
            repository_root=repository_root,
            observed_failure_log_path=initial_failure_path,
            require_ready=False,
            expected_cell_sha256=str(incident["cell_sha256"]),
        )
        attempts_dir = execution_root / "supervisor_attempts"
        existing_attempts = [
            path.stem.split(".")[0]
            for path in attempts_dir.glob("attempt_*.stderr.log")
        ] if attempts_dir.is_dir() else []
        attempt_numbers = [
            int(name.removeprefix("attempt_"))
            for name in existing_attempts
            if name.removeprefix("attempt_").isdigit()
        ]
        first_attempt_number = (
            max(attempt_numbers, default=0) + 1
        )
        for offset in range(EXPECTED_SEARCH_CELL_COUNT + 2):
            attempt_number = first_attempt_number + offset
            _require_runner_stopped(execution_root)
            state = {
                "state_version": (
                    "lyx_recovery_independent_bo_supervisor_state_v1"
                ),
                "status": "launching_original_runner",
                "attempt_number": attempt_number,
                "initial_repair_performed": initial_repair is not None,
                "cell_completion_count": len(
                    list(
                        (execution_root / "cells").rglob(
                            "cell_completion.json"
                        )
                    )
                ),
            }
            state["state_sha256"] = canonical_sha256(state)
            atomic_write_json(state_path, state)
            completed = _run_original_runner(
                original_proposal_dir=original_proposal_dir,
                governance_dir=governance_dir,
                execution_dir=execution_root,
                repository_root=repository_root,
                attempt_number=attempt_number,
            )
            _require_runner_stopped(execution_root)
            stderr_path = (
                execution_root
                / "supervisor_attempts"
                / f"attempt_{attempt_number:03d}.stderr.log"
            )
            stderr_text = stderr_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            if completed.returncode == 0:
                final_state = {
                    "state_version": (
                        "lyx_recovery_independent_bo_supervisor_state_v1"
                    ),
                    "status": "complete",
                    "attempt_number": attempt_number,
                    "cell_completion_count": len(
                        list(
                            (execution_root / "cells").rglob(
                                "cell_completion.json"
                            )
                        )
                    ),
                    "runner_returncode": 0,
                }
                final_state["state_sha256"] = canonical_sha256(
                    final_state
                )
                atomic_write_json(state_path, final_state)
                return 0
            if not is_mappingproxy_completion_failure(stderr_text):
                failure = {
                    "state_version": (
                        "lyx_recovery_independent_bo_supervisor_state_v1"
                    ),
                    "status": "failed_closed",
                    "attempt_number": attempt_number,
                    "runner_returncode": completed.returncode,
                    "stderr_file_sha256": file_sha256(stderr_path),
                    "reason": "unexpected_runner_failure",
                }
                failure["state_sha256"] = canonical_sha256(failure)
                atomic_write_json(state_path, failure)
                raise RepairContractError(
                    "supervisor_unexpected_runner_failure"
                )
            ready_cell_sha256s = _ready_missing_cell_sha256s(
                original_proposal=original,
                execution_dir=execution_root,
            )
            if len(ready_cell_sha256s) != 1:
                raise RepairContractError(
                    "supervisor_expected_exactly_one_ready_cell"
                )
            observed_state_payload = {
                "state_version": (
                    "lyx_recovery_independent_bo_supervisor_state_v1"
                ),
                "status": "mappingproxy_failure_observed",
                "attempt_number": attempt_number,
                "runner_returncode": completed.returncode,
                "ready_cell_sha256": ready_cell_sha256s[0],
                "observed_failure_log_path": _relative_to_root(
                    stderr_path,
                    repository_root,
                ),
                "observed_failure_log_file_sha256": file_sha256(
                    stderr_path
                ),
            }
            observed_state_payload["state_sha256"] = (
                canonical_sha256(observed_state_payload)
            )
            atomic_write_json(state_path, observed_state_payload)
            _finalize_ready_cell_locked(
                original_proposal_path=original_path,
                repair_proposal_path=repair_path,
                repair_authorization_path=authorization_path,
                governance_dir=governance_dir,
                execution_dir=execution_root,
                repository_root=repository_root,
                observed_failure_log_path=stderr_path,
                require_ready=True,
                expected_cell_sha256=ready_cell_sha256s[0],
            )
        raise RepairContractError(
            "supervisor_attempt_bound_exhausted"
        )


def _write_proposal(
    *,
    proposal: Mapping[str, Any],
    output_dir: Path,
) -> None:
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(
            f"mappingproxy_repair_proposal_dir_exists:{root}"
        )
    root.mkdir(parents=True)
    atomic_write_json(
        root / "mappingproxy_repair_proposal.json",
        dict(proposal),
    )
    receipt = {
        "receipt_version": (
            "lyx_recovery_independent_bo_mappingproxy_repair_proposal_receipt_v1"
        ),
        "status": "awaiting_blanket_authorization",
        "proposal_sha256": proposal["proposal_sha256"],
        "repair_added_unique_identity_budget": 0,
        "repair_added_solver_run_count": 0,
        "original_runner_unique_identity_budget": proposal[
            "original_runner_unique_identity_budget"
        ],
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    atomic_write_json(root / "proposal_receipt.json", receipt)


def _record_authorization(
    *,
    proposal_dir: Path,
    repository_root: Path,
) -> dict[str, Any]:
    root = Path(proposal_dir).resolve()
    proposal = validate_repair_proposal(
        proposal=read_json(
            root / "mappingproxy_repair_proposal.json"
        ),
        repository_root=repository_root,
    )
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    deadline = datetime.fromisoformat(
        BLANKET_AUTHORIZATION_EXPIRES_AT
    )
    if now >= deadline:
        raise RepairContractError(
            "blanket_proposal_authorization_deadline_passed"
        )
    receipt = build_repair_authorization(
        proposal=proposal,
        approved_at=now.isoformat(timespec="seconds"),
    )
    validate_repair_authorization(
        proposal=proposal,
        receipt=receipt,
    )
    atomic_write_json(
        root / "repair_authorization.json",
        receipt,
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    propose = subparsers.add_parser("propose")
    for name in (
        "original_proposal",
        "original_authorization",
        "failure_log",
        "governance_dir",
        "execution_dir",
        "repository_root",
        "output_dir",
    ):
        propose.add_argument(
            f"--{name.replace('_', '-')}",
            required=True,
        )
    authorize = subparsers.add_parser("authorize-blanket")
    authorize.add_argument("--proposal-dir", required=True)
    authorize.add_argument("--repository-root", required=True)
    run = subparsers.add_parser("supervise")
    for name in (
        "original_proposal_dir",
        "repair_proposal_dir",
        "governance_dir",
        "execution_dir",
        "repository_root",
    ):
        run.add_argument(
            f"--{name.replace('_', '-')}",
            required=True,
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "propose":
        proposal = build_repair_proposal(
            original_proposal_path=Path(
                args.original_proposal
            ),
            original_authorization_path=Path(
                args.original_authorization
            ),
            failure_log_path=Path(args.failure_log),
            tool_path=Path(__file__),
            execution_dir=Path(args.execution_dir),
            governance_dir=Path(args.governance_dir),
            repository_root=Path(args.repository_root),
        )
        _write_proposal(
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
                    "repair_added_unique_identity_budget": 0,
                    "original_runner_unique_identity_budget": (
                        proposal[
                            "original_runner_unique_identity_budget"
                        ]
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "authorize-blanket":
        receipt = _record_authorization(
            proposal_dir=Path(args.proposal_dir),
            repository_root=Path(args.repository_root),
        )
        print(
            json.dumps(
                {
                    "status": "authorized",
                    "authorization_sha256": receipt[
                        "authorization_sha256"
                    ],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "supervise":
        return supervise(
            original_proposal_dir=Path(
                args.original_proposal_dir
            ),
            repair_proposal_dir=Path(
                args.repair_proposal_dir
            ),
            governance_dir=Path(args.governance_dir),
            execution_dir=Path(args.execution_dir),
            repository_root=Path(args.repository_root),
        )
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
