from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPOSITORY_ROOT / "python" / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from recovery_independent_bo_supervisor import (  # noqa: E402
    MAPPINGPROXY_FAILURE_SIGNATURE,
    RepairContractError,
    _build_repair_receipt,
    _exclusive_supervisor_lock,
    _observed_failure_state_evidence,
    _pending_repair_evidence,
    _promote_finished_launch_state,
    _validate_existing_completion,
    build_cell_completion_payload,
    build_repair_authorization,
    build_repair_proposal,
    is_mappingproxy_completion_failure,
    validate_repair_authorization,
)

from ppg_hr.v2.recovery_contracts import (  # noqa: E402
    canonical_sha256,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _mappingproxy_traceback() -> str:
    return "\n".join(
        (
            "runner warning before failure",
            "Traceback (most recent call last):",
            (
                '  File "recovery_independent_bo_runner.py", '
                "line 99, in main"
            ),
            "    execute_recovery_independent_bo_proposal(",
            (
                '  File "recovery_independent_bo_experiment.py", '
                "line 99, in _execute_search_cell"
            ),
            (
                '    "seed_stability_audit_sha256": '
                "canonical_sha256("
            ),
            MAPPINGPROXY_FAILURE_SIGNATURE,
            "",
        )
    )


def _synthetic_cells() -> list[dict[str, object]]:
    return [
        {
            "cell_sha256": f"{index + 1:064x}",
            "record_id": f"record_{index % 12:02d}",
            "scene": f"scene_{index % 4}",
            "recovery_candidate_id": f"recovery_{index // 12}",
        }
        for index in range(36)
    ]


def _prepare_proposal_inputs(
    tmp_path: Path,
    *,
    ready_cell_count: int = 1,
) -> dict[str, Path]:
    cells = _synthetic_cells()
    original = {
        "proposal_sha256": "a" * 64,
        "unique_budget": 5400,
        "search_cells": cells,
    }
    original_path = tmp_path / "original.json"
    original_authorization_path = tmp_path / "original_auth.json"
    failure_path = tmp_path / "failure.log"
    tool_path = tmp_path / "tool.py"
    execution_dir = tmp_path / "execution"
    governance_dir = tmp_path / "governance"
    _write_json(original_path, original)
    _write_json(original_authorization_path, {"approved": True})
    _write_json(governance_dir / "budget_contract.json", {})
    _write_json(governance_dir / "execution_authorization.json", {})
    failure_path.write_text(
        _mappingproxy_traceback(),
        encoding="utf-8",
    )
    tool_path.write_text("print('repair')\n", encoding="utf-8")
    for cell in cells[:ready_cell_count]:
        cell_dir = (
            execution_dir
            / "cells"
            / str(cell["recovery_candidate_id"])
            / str(cell["record_id"])
        )
        _write_json(
            cell_dir / "search" / "driver_state.json",
            {"stage": "complete"},
        )
        _write_json(
            cell_dir / "candidate_results.json",
            {
                "proposal_sha256": "a" * 64,
                "cell_sha256": cell["cell_sha256"],
            },
        )
        _write_json(
            cell_dir / "seed_stability_audit.json",
            {"global_candidate_count": 150},
        )
    return {
        "original": original_path,
        "original_authorization": original_authorization_path,
        "failure": failure_path,
        "tool": tool_path,
        "execution": execution_dir,
        "governance": governance_dir,
    }


def test_repair_proposal_is_zero_run_and_binds_one_incident(
    tmp_path: Path,
) -> None:
    paths = _prepare_proposal_inputs(tmp_path)

    proposal = build_repair_proposal(
        original_proposal_path=paths["original"],
        original_authorization_path=paths[
            "original_authorization"
        ],
        failure_log_path=paths["failure"],
        tool_path=paths["tool"],
        repository_root=tmp_path,
        execution_dir=paths["execution"],
        governance_dir=paths["governance"],
    )

    assert proposal["status"] == "frozen_zero_solver_runs"
    assert proposal["original_proposal_sha256"] == "a" * 64
    assert proposal["covered_cell_count"] == 36
    assert len(proposal["incident_artifacts"]) == 1
    assert proposal["repair_added_unique_identity_budget"] == 0
    assert proposal["repair_added_solver_run_count"] == 0
    assert proposal["original_runner_unique_identity_budget"] == 5400
    assert proposal["allowed_write"] == "missing_cell_completion_only"
    unsigned = {
        key: value
        for key, value in proposal.items()
        if key != "proposal_sha256"
    }
    assert proposal["proposal_sha256"] == canonical_sha256(unsigned)


@pytest.mark.parametrize("ready_cell_count", [0, 2])
def test_repair_proposal_requires_exactly_one_incident(
    tmp_path: Path,
    ready_cell_count: int,
) -> None:
    paths = _prepare_proposal_inputs(
        tmp_path,
        ready_cell_count=ready_cell_count,
    )
    with pytest.raises(
        RepairContractError,
        match="requires_one_incident_cell",
    ):
        build_repair_proposal(
            original_proposal_path=paths["original"],
            original_authorization_path=paths[
                "original_authorization"
            ],
            failure_log_path=paths["failure"],
            tool_path=paths["tool"],
            repository_root=tmp_path,
            execution_dir=paths["execution"],
            governance_dir=paths["governance"],
        )


def test_cell_completion_is_rebuilt_from_json_safe_artifacts(
    tmp_path: Path,
) -> None:
    proposal = {
        "proposal_sha256": "a" * 64,
        "seed_manifest": {"unique_budget_per_cell": 2},
    }
    cell = {
        "cell_sha256": "b" * 64,
        "record_id": "jianpan1_LYX_0708",
        "scene": "jianpan",
        "recovery_candidate_id": "control",
    }
    rows = [
        {
            "candidate_id": "candidate-b",
            "identity": {"identity_sha256": "1" * 64},
            "identity_sha256": "1" * 64,
            "eligible": False,
            "objective": 1.0,
        },
        {
            "candidate_id": "candidate-a",
            "identity": {"identity_sha256": "2" * 64},
            "identity_sha256": "2" * 64,
            "eligible": True,
            "objective": 3.0,
        },
    ]
    candidate_results = {
        "proposal_sha256": "a" * 64,
        "cell_sha256": "b" * 64,
        "result_count": 2,
        "results": rows,
        "result_sha256": "c" * 64,
    }
    seed_audit = {
        "global_candidate_count": 2,
        "seed_stability_candidate_ids": [
            "candidate-a",
            "candidate-b",
        ],
    }
    matrix = {
        "planned_identity_count": 2,
        "identity_with_solver_attempt_count": 2,
        "cache_only_identity_count": 0,
        "failed_attempt_count": 0,
        "retry_count": 0,
    }
    repair_proposal = {"proposal_sha256": "d" * 64}
    repair_authorization = {"authorization_sha256": "e" * 64}
    failure_path = tmp_path / "failure.log"
    failure_path.write_text(
        _mappingproxy_traceback(),
        encoding="utf-8",
    )

    completion = build_cell_completion_payload(
        proposal=proposal,
        cell=cell,
        candidate_results=candidate_results,
        seed_audit=seed_audit,
        matrix=matrix,
        repair_proposal=repair_proposal,
        repair_authorization=repair_authorization,
        observed_failure_log_path=failure_path,
        repository_root=tmp_path,
    )

    assert completion["status"] == "complete"
    assert completion["unique_candidate_count"] == 2
    assert completion["eligible_candidate_count"] == 1
    assert completion["selected"]["candidate_id"] == "candidate-a"
    assert (
        completion["reporting_repair"][
            "repair_added_solver_run_count"
        ]
        == 0
    )
    unsigned = {
        key: value
        for key, value in completion.items()
        if key != "completion_sha256"
    }
    assert completion["completion_sha256"] == canonical_sha256(unsigned)


def test_repair_authorization_and_failure_match_are_fail_closed() -> None:
    proposal = {
        "proposal_sha256": "d" * 64,
        "original_proposal_sha256": "a" * 64,
        "repair_added_unique_identity_budget": 0,
        "repair_added_solver_run_count": 0,
        "original_runner_unique_identity_budget": 5400,
        "allowed_write": "missing_cell_completion_only",
    }
    receipt = build_repair_authorization(
        proposal=proposal,
        approved_at="2026-07-31T02:30:00+08:00",
    )

    assert validate_repair_authorization(
        proposal=proposal,
        receipt=receipt,
    ) == receipt
    assert is_mappingproxy_completion_failure(
        _mappingproxy_traceback()
    )
    assert not is_mappingproxy_completion_failure(
        _mappingproxy_traceback()
        + "\nPermissionError: [WinError 5]"
    )
    assert not is_mappingproxy_completion_failure(
        "PermissionError: [WinError 5]\n"
        + _mappingproxy_traceback()
    )
    assert not is_mappingproxy_completion_failure(
        _mappingproxy_traceback().replace(
            MAPPINGPROXY_FAILURE_SIGNATURE,
            "PermissionError: [WinError 5]",
        )
    )

    expired = build_repair_authorization(
        proposal=proposal,
        approved_at="2026-07-31T10:00:00+08:00",
    )
    with pytest.raises(
        RepairContractError,
        match="repair_authorization_invalid",
    ):
        validate_repair_authorization(
            proposal=proposal,
            receipt=expired,
        )


def test_supervisor_lock_rejects_a_second_owner(
    tmp_path: Path,
) -> None:
    with _exclusive_supervisor_lock(tmp_path):
        with pytest.raises(
            RepairContractError,
            match="supervisor_already_running",
        ):
            with _exclusive_supervisor_lock(tmp_path):
                pass


def test_observed_failure_state_binds_next_ready_cell(
    tmp_path: Path,
) -> None:
    failure_path = tmp_path / "attempt.stderr.log"
    failure_path.write_text(
        _mappingproxy_traceback(),
        encoding="utf-8",
    )
    state = {
        "state_version": (
            "lyx_recovery_independent_bo_supervisor_state_v1"
        ),
        "status": "mappingproxy_failure_observed",
        "attempt_number": 2,
        "runner_returncode": 1,
        "ready_cell_sha256": "b" * 64,
        "observed_failure_log_path": "attempt.stderr.log",
        "observed_failure_log_file_sha256": hashlib.sha256(
            failure_path.read_bytes()
        ).hexdigest(),
    }
    state["state_sha256"] = canonical_sha256(state)
    state_path = tmp_path / "supervisor_state.json"
    _write_json(state_path, state)

    assert _observed_failure_state_evidence(
        state_path=state_path,
        original_proposal={
            "search_cells": [{"cell_sha256": "b" * 64}]
        },
        repository_root=tmp_path,
    ) == ("b" * 64, failure_path)


def test_finished_launch_state_is_promoted_before_recovery(
    tmp_path: Path,
) -> None:
    cell = {
        "cell_sha256": "b" * 64,
        "record_id": "record",
        "recovery_candidate_id": "recovery",
    }
    cell_dir = tmp_path / "cells" / "recovery" / "record"
    _write_json(
        cell_dir / "search" / "driver_state.json",
        {"stage": "complete"},
    )
    _write_json(cell_dir / "candidate_results.json", {})
    _write_json(cell_dir / "seed_stability_audit.json", {})
    stderr_path = (
        tmp_path
        / "supervisor_attempts"
        / "attempt_002.stderr.log"
    )
    stderr_path.parent.mkdir(parents=True)
    stderr_path.write_text(
        _mappingproxy_traceback(),
        encoding="utf-8",
    )
    launching = {
        "state_version": (
            "lyx_recovery_independent_bo_supervisor_state_v1"
        ),
        "status": "launching_original_runner",
        "attempt_number": 2,
        "initial_repair_performed": True,
        "cell_completion_count": 1,
    }
    launching["state_sha256"] = canonical_sha256(launching)
    state_path = tmp_path / "supervisor_state.json"
    _write_json(state_path, launching)
    proposal = {"search_cells": [cell]}

    assert _promote_finished_launch_state(
        state_path=state_path,
        original_proposal=proposal,
        execution_dir=tmp_path,
        repository_root=tmp_path,
    )
    assert _observed_failure_state_evidence(
        state_path=state_path,
        original_proposal=proposal,
        repository_root=tmp_path,
    ) == ("b" * 64, stderr_path)


def test_launching_state_without_log_rejects_ready_cell(
    tmp_path: Path,
) -> None:
    cell = {
        "cell_sha256": "b" * 64,
        "record_id": "record",
        "recovery_candidate_id": "recovery",
    }
    cell_dir = tmp_path / "cells" / "recovery" / "record"
    _write_json(
        cell_dir / "search" / "driver_state.json",
        {"stage": "complete"},
    )
    _write_json(cell_dir / "candidate_results.json", {})
    _write_json(cell_dir / "seed_stability_audit.json", {})
    launching = {
        "state_version": (
            "lyx_recovery_independent_bo_supervisor_state_v1"
        ),
        "status": "launching_original_runner",
        "attempt_number": 2,
        "initial_repair_performed": True,
        "cell_completion_count": 1,
    }
    launching["state_sha256"] = canonical_sha256(launching)
    state_path = tmp_path / "supervisor_state.json"
    _write_json(state_path, launching)

    with pytest.raises(
        RepairContractError,
        match="launching_state_missing_log",
    ):
        _promote_finished_launch_state(
            state_path=state_path,
            original_proposal={"search_cells": [cell]},
            execution_dir=tmp_path,
            repository_root=tmp_path,
        )


def test_repaired_completion_requires_matching_receipt(
    tmp_path: Path,
) -> None:
    cell = {
        "cell_sha256": "b" * 64,
        "record_id": "record",
        "scene": "scene",
        "recovery_candidate_id": "recovery",
    }
    cell_dir = tmp_path / "cells" / "recovery" / "record"
    _write_json(cell_dir / "candidate_results.json", {"rows": []})
    _write_json(cell_dir / "seed_stability_audit.json", {"ids": []})
    failure_path = tmp_path / "failure.log"
    failure_path.write_text(
        _mappingproxy_traceback(),
        encoding="utf-8",
    )
    original_proposal = {"proposal_sha256": "a" * 64}
    repair_proposal = {"proposal_sha256": "d" * 64}
    repair_authorization = {"authorization_sha256": "e" * 64}
    completion = {
        "completion_version": (
            "lyx_recovery_independent_bo_cell_completion_v1"
        ),
        "status": "complete",
        "proposal_sha256": "a" * 64,
        "cell_sha256": "b" * 64,
        "record_id": "record",
        "scene": "scene",
        "recovery_candidate_id": "recovery",
        "unique_candidate_count": 150,
        "reporting_repair": {
            "repair_version": (
                "lyx_recovery_independent_bo_cell_completion_repair_v1"
            ),
            "repair_proposal_sha256": "d" * 64,
            "repair_authorization_sha256": "e" * 64,
            "failure_signature": MAPPINGPROXY_FAILURE_SIGNATURE,
            "source_artifacts_were_json_round_tripped": True,
            "repair_added_solver_run_count": 0,
            "repair_added_unique_identity_count": 0,
            "original_runner_cell_solver_attempt_count": 150,
            "original_runner_cell_cache_only_identity_count": 0,
            "observed_failure_log_path": "failure.log",
            "observed_failure_log_file_sha256": (
                hashlib.sha256(failure_path.read_bytes()).hexdigest()
            ),
        },
    }
    completion["completion_sha256"] = canonical_sha256(completion)
    receipt = _build_repair_receipt(
        cell=cell,
        cell_dir=cell_dir,
        completion=completion,
        repair_proposal=repair_proposal,
        repair_authorization=repair_authorization,
        observed_failure_log_path=failure_path,
        repository_root=tmp_path,
    )
    completion_path = cell_dir / "cell_completion.json"
    receipt_path = cell_dir / "cell_completion_repair_receipt.json"
    _write_json(receipt_path, receipt)
    _write_json(completion_path, completion)

    _validate_existing_completion(
        completion_path=completion_path,
        cell=cell,
        original_proposal=original_proposal,
        repair_proposal=repair_proposal,
        repair_authorization=repair_authorization,
        repository_root=tmp_path,
    )
    completion_path.unlink()
    pending = _pending_repair_evidence(
        original_proposal={"search_cells": [cell]},
        execution_dir=tmp_path,
        repository_root=tmp_path,
    )
    assert pending == ("b" * 64, failure_path)
    _write_json(completion_path, completion)
    receipt_path.unlink()
    with pytest.raises(
        RepairContractError,
        match="receipt_missing",
    ):
        _validate_existing_completion(
            completion_path=completion_path,
            cell=cell,
            original_proposal=original_proposal,
            repair_proposal=repair_proposal,
            repair_authorization=repair_authorization,
            repository_root=tmp_path,
        )
