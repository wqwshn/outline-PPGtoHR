"""Governed short-circuit scheduler for the LYX recovery experiment.

This tool does not change the frozen numerical identity, metric contract, BO
space, or per-cell search budget from ``recovery_independent_bo_v1``.  It only
changes scheduling:

* stop the already-eliminated control recovery after its ninth sealed cell;
* run the two remaining recoveries on a frozen hard-record order;
* eliminate a recovery immediately after a 0/150 eligible cell;
* admit at most one survivor to the separately governed shared-profile phase.

The original runner has one known JSON ``mappingproxy`` receipt defect.  This
tool uses the already-authorized zero-run repair path after observing the same
exact failure and never retries a solver identity.
"""

from __future__ import annotations

import argparse
import json
import statistics
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from recovery_independent_bo_supervisor import (
    _exclusive_supervisor_lock,
    _finalize_ready_cell_locked,
    _validate_existing_completion,
    is_mappingproxy_completion_failure,
    validate_repair_authorization,
    validate_repair_proposal,
)

from ppg_hr.v2 import recovery_stage_r_cache as stage_r_cache
from ppg_hr.v2.bo_space_generalization import (
    BOCandidate,
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
    _attempt_identity_from_item,
    _constraint_values,
    _execute_search_cell,
    _exploration_from_payload,
    build_recovery_independent_bo_identity,
    validate_recovery_independent_bo_execution_authorization,
    validate_recovery_independent_bo_preflight,
)
from ppg_hr.v2.recovery_stage_r_experiment import (
    run_stage_r_numerical_identity,
)


class RecoveryShortCircuitError(RuntimeError):
    """The short-circuit experiment violates its frozen contract."""


PROPOSAL_VERSION = "lyx_recovery_short_circuit_shared_validation_proposal_v1"
AUTHORIZATION_VERSION = (
    "lyx_recovery_short_circuit_shared_validation_authorization_v1"
)
GATE_A_COMPLETION_VERSION = "lyx_recovery_short_circuit_gate_a_completion_v1"
GATE_B_COMPLETION_VERSION = (
    "lyx_recovery_scene_shared_validation_completion_v1"
)
CONTROL_RECOVERY_ID = "current_fixed_floor_control_v1"
REMAINING_RECOVERY_IDS = (
    "relative_gap_timeout_v1",
    "relative_gap_rise_guard_v1",
)
HARD_RECORD_ORDER = (
    "kaihe3_LYX_0613",
    "kaihe1_LYX_0617",
    "run3_LYX_0708",
    "xiezi4_LYX_0708",
    "kaihe1_LYX_0613",
    "jianpan1_LYX_0708",
)
REQUIRED_CONTROL_RECORDS = (
    "jianpan1_LYX_0708",
    "jianpan2_LYX_0708",
    "jianpan3_LYX_0708",
    "kaihe1_LYX_0613",
    "kaihe1_LYX_0617",
    "kaihe3_LYX_0613",
    "run1_LYX_0708",
    "run2_LYX_0708",
    "run3_LYX_0708",
)
SHARED_FS_TARGET = 25
SHARED_GRID_SIZE = 100
EXISTING_IDENTITY_UPPER_BOUND = 1350
GATE_A_IDENTITY_UPPER_BOUND = 1800
GATE_B_IDENTITY_UPPER_BOUND = 1200
TOTAL_IDENTITY_UPPER_BOUND = (
    EXISTING_IDENTITY_UPPER_BOUND
    + GATE_A_IDENTITY_UPPER_BOUND
    + GATE_B_IDENTITY_UPPER_BOUND
)
V13_IDENTITY_LIMIT = 5400
USER_AUTHORIZATION_TEXT = (
    "好的，先按照你提出的：不再批准“机械跑满 36”，改为“run3 封存后停止 + "
    "困难样本优先短路 + 直接场景内共享参数验证”。调整实验方向，我批准编写计划后"
    "可以直接执行。\n\n在北京时间20：00前你拥有完全授权，可以自行执行。"
)
AUTHORIZATION_CUTOFF = "2026-07-31T20:00:00+08:00"


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryShortCircuitError(f"{name}_must_be_mapping")
    return value


def _verify_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    artifact: str,
) -> None:
    expected = payload.get(hash_field)
    body = dict(payload)
    body.pop(hash_field, None)
    if expected != canonical_sha256(body):
        raise RecoveryShortCircuitError(f"{artifact}_hash_mismatch")


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _cell_index(
    original: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    cells = original.get("search_cells")
    if not isinstance(cells, list):
        raise RecoveryShortCircuitError("original_search_cells_invalid")
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in cells:
        cell = _mapping("original_search_cell", raw)
        key = (
            str(cell.get("recovery_candidate_id")),
            str(cell.get("record_id")),
        )
        if key in index:
            raise RecoveryShortCircuitError("original_search_cell_duplicate")
        index[key] = cell
    return index


def _completion_path(
    execution_dir: Path,
    *,
    recovery_id: str,
    record_id: str,
) -> Path:
    return (
        Path(execution_dir)
        / "cells"
        / recovery_id
        / record_id
        / "cell_completion.json"
    )


def _validate_bound_completion(
    *,
    completion_path: Path,
    cell: Mapping[str, Any],
    original: Mapping[str, Any],
    repair_proposal: Mapping[str, Any],
    repair_authorization: Mapping[str, Any],
    repository_root: Path,
) -> Mapping[str, Any]:
    if not completion_path.is_file():
        raise RecoveryShortCircuitError(
            "required_cell_completion_missing:"
            + str(completion_path)
        )
    completion = read_json(completion_path)
    _validate_existing_completion(
        completion_path=completion_path,
        cell=cell,
        original_proposal=original,
        repair_proposal=repair_proposal,
        repair_authorization=repair_authorization,
        repository_root=repository_root,
    )
    return _mapping("validated_cell_completion", completion)


def build_proposal(
    *,
    original_proposal_path: Path,
    original_authorization_path: Path,
    repair_proposal_path: Path,
    repair_authorization_path: Path,
    governance_dir: Path,
    execution_dir: Path,
    spec_path: Path,
    repository_root: Path,
    tool_path: Path,
) -> dict[str, Any]:
    """Build the exact scheduling amendment after nine sealed control cells."""

    root = Path(repository_root).resolve()
    original_path = Path(original_proposal_path).resolve()
    original = read_json(original_path)
    validate_recovery_independent_bo_preflight(
        proposal=original,
        repository_root=root,
    )
    original_authorization = (
        validate_recovery_independent_bo_execution_authorization(
            original,
            receipt=read_json(
                Path(original_authorization_path).resolve()
            ),
        )
    )
    repair_proposal = validate_repair_proposal(
        proposal=read_json(Path(repair_proposal_path).resolve()),
        repository_root=root,
    )
    repair_authorization = validate_repair_authorization(
        proposal=repair_proposal,
        receipt=read_json(Path(repair_authorization_path).resolve()),
    )
    if (
        repair_proposal.get("original_proposal_sha256")
        != original.get("proposal_sha256")
    ):
        raise RecoveryShortCircuitError(
            "repair_original_proposal_binding_mismatch"
        )
    cells = _cell_index(original)
    completion_bindings: list[dict[str, Any]] = []
    for record_id in REQUIRED_CONTROL_RECORDS:
        key = (CONTROL_RECOVERY_ID, record_id)
        if key not in cells:
            raise RecoveryShortCircuitError(
                "required_control_cell_missing:" + record_id
            )
        path = _completion_path(
            Path(execution_dir).resolve(),
            recovery_id=CONTROL_RECOVERY_ID,
            record_id=record_id,
        )
        completion = _validate_bound_completion(
            completion_path=path,
            cell=cells[key],
            original=original,
            repair_proposal=repair_proposal,
            repair_authorization=repair_authorization,
            repository_root=root,
        )
        completion_bindings.append(
            {
                "record_id": record_id,
                "cell_sha256": cells[key]["cell_sha256"],
                "completion_sha256": completion["completion_sha256"],
                "eligible_candidate_count": completion[
                    "eligible_candidate_count"
                ],
            }
        )
    kaihe3 = next(
        item
        for item in completion_bindings
        if item["record_id"] == "kaihe3_LYX_0613"
    )
    if kaihe3["eligible_candidate_count"] != 0:
        raise RecoveryShortCircuitError(
            "control_elimination_evidence_changed"
        )
    if TOTAL_IDENTITY_UPPER_BOUND > V13_IDENTITY_LIMIT:
        raise RecoveryShortCircuitError(
            "short_circuit_budget_exceeds_v13"
        )
    cell_bindings: list[dict[str, Any]] = []
    for recovery_id in REMAINING_RECOVERY_IDS:
        for order, record_id in enumerate(HARD_RECORD_ORDER, start=1):
            cell = cells.get((recovery_id, record_id))
            if cell is None:
                raise RecoveryShortCircuitError(
                    "hard_cell_missing:"
                    + recovery_id
                    + "/"
                    + record_id
                )
            cell_bindings.append(
                {
                    "recovery_candidate_id": recovery_id,
                    "record_id": record_id,
                    "order": order,
                    "cell_sha256": cell["cell_sha256"],
                    "unique_budget": 150,
                }
            )
    budget_contract_path = (
        Path(governance_dir).resolve() / "budget_contract.json"
    )
    budget_contract = read_json(budget_contract_path)
    expected_budget = (
        BudgetContract.proposed_v13_recovery_independent_bo()
    )
    if budget_contract != expected_budget.to_dict():
        raise RecoveryShortCircuitError("v13_budget_contract_drift")
    spec = Path(spec_path).resolve()
    tool = Path(tool_path).resolve()
    proposal: dict[str, Any] = {
        "proposal_version": PROPOSAL_VERSION,
        "status": "awaiting_user_blanket_authorization",
        "parent_experiment_id": original["parent_experiment_id"],
        "original_proposal_sha256": original["proposal_sha256"],
        "original_authorization_sha256": canonical_sha256(
            original_authorization
        ),
        "repair_proposal_sha256": repair_proposal["proposal_sha256"],
        "repair_authorization_sha256": canonical_sha256(
            repair_authorization
        ),
        "spec_path": str(spec.relative_to(root)).replace("\\", "/"),
        "spec_file_sha256": file_sha256(spec),
        "scheduler_path": str(tool.relative_to(root)).replace("\\", "/"),
        "scheduler_file_sha256": file_sha256(tool),
        "execution_dir": str(
            Path(execution_dir).resolve().relative_to(root)
        ).replace("\\", "/"),
        "governance_dir": str(
            Path(governance_dir).resolve().relative_to(root)
        ).replace("\\", "/"),
        "v13_budget_contract_hash": expected_budget.sha256,
        "control_recovery_id": CONTROL_RECOVERY_ID,
        "control_elimination_evidence": {
            "record_id": "kaihe3_LYX_0613",
            "eligible_candidate_count": 0,
            "completion_sha256": kaihe3["completion_sha256"],
        },
        "sealed_control_completions": completion_bindings,
        "remaining_recovery_ids": list(REMAINING_RECOVERY_IDS),
        "hard_record_order": list(HARD_RECORD_ORDER),
        "gate_a_cell_bindings": cell_bindings,
        "gate_a_stop_rule": "first_zero_eligible_candidate_count",
        "gate_a_survivor_ranking": [
            "max_selected_eligible_mae_asc",
            "eligible_candidate_count_sum_desc",
            "mechanism_complexity_asc",
            "recovery_candidate_id_asc",
        ],
        "gate_b_admit_count": 1,
        "gate_b_evidence_role": "scene_shared_validation",
        "gate_b_fs_target": SHARED_FS_TARGET,
        "gate_b_grid_size_per_record": SHARED_GRID_SIZE,
        "gate_b_scene_count": 4,
        "gate_b_fold_count": 12,
        "identity_budget": {
            "existing_sealed_upper_bound": (
                EXISTING_IDENTITY_UPPER_BOUND
            ),
            "gate_a_new_upper_bound": GATE_A_IDENTITY_UPPER_BOUND,
            "gate_b_new_upper_bound": GATE_B_IDENTITY_UPPER_BOUND,
            "combined_upper_bound": TOTAL_IDENTITY_UPPER_BOUND,
            "v13_stage_limit": V13_IDENTITY_LIMIT,
            "unused_v13_capacity_not_authorized": (
                V13_IDENTITY_LIMIT - TOTAL_IDENTITY_UPPER_BOUND
            ),
            "budget_expansion": 0,
            "automatic_retry": False,
        },
        "automatic_stage_f_execution": False,
        "created_at": _iso_now(),
        "authorization_cutoff": AUTHORIZATION_CUTOFF,
        "next_state": "awaiting_authorized_gate_a_execution",
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def validate_proposal(
    *,
    proposal: Mapping[str, Any],
    repository_root: Path,
) -> Mapping[str, Any]:
    """Validate immutable scheduling and budget fields before any solve."""

    _verify_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact="short_circuit_proposal",
    )
    if (
        proposal.get("proposal_version") != PROPOSAL_VERSION
        or proposal.get("remaining_recovery_ids")
        != list(REMAINING_RECOVERY_IDS)
        or proposal.get("hard_record_order")
        != list(HARD_RECORD_ORDER)
        or proposal.get("gate_a_stop_rule")
        != "first_zero_eligible_candidate_count"
        or proposal.get("gate_b_admit_count") != 1
        or proposal.get("gate_b_fs_target") != SHARED_FS_TARGET
        or proposal.get("gate_b_grid_size_per_record")
        != SHARED_GRID_SIZE
        or proposal.get("authorization_cutoff")
        != AUTHORIZATION_CUTOFF
    ):
        raise RecoveryShortCircuitError(
            "short_circuit_proposal_contract_invalid"
        )
    budget = _mapping(
        "short_circuit_identity_budget",
        proposal.get("identity_budget"),
    )
    if (
        budget.get("combined_upper_bound")
        != TOTAL_IDENTITY_UPPER_BOUND
        or budget.get("v13_stage_limit") != V13_IDENTITY_LIMIT
        or budget.get("budget_expansion") != 0
        or budget.get("automatic_retry") is not False
    ):
        raise RecoveryShortCircuitError(
            "short_circuit_budget_contract_invalid"
        )
    root = Path(repository_root).resolve()
    for path_field, hash_field in (
        ("spec_path", "spec_file_sha256"),
        ("scheduler_path", "scheduler_file_sha256"),
    ):
        path = (root / str(proposal[path_field])).resolve()
        if (
            not path.is_file()
            or file_sha256(path) != proposal.get(hash_field)
        ):
            raise RecoveryShortCircuitError(
                f"short_circuit_{path_field}_drift"
            )
    return proposal


def build_authorization(
    *,
    proposal: Mapping[str, Any],
    granted_at: str,
) -> dict[str, Any]:
    validate_proposal(
        proposal=proposal,
        repository_root=Path.cwd(),
    )
    granted = datetime.fromisoformat(granted_at)
    cutoff = datetime.fromisoformat(AUTHORIZATION_CUTOFF)
    if (
        granted.tzinfo is None
        or granted > cutoff
        or granted.date() != cutoff.date()
    ):
        raise RecoveryShortCircuitError(
            "short_circuit_authorization_outside_window"
        )
    receipt: dict[str, Any] = {
        "authorization_version": AUTHORIZATION_VERSION,
        "status": "authorized",
        "proposal_sha256": proposal["proposal_sha256"],
        "granted_at": granted.isoformat(timespec="seconds"),
        "expires_at": AUTHORIZATION_CUTOFF,
        "user_authorization_text": USER_AUTHORIZATION_TEXT,
        "authorized_actions": [
            "stop_original_matrix_after_run3_sealed",
            "execute_gate_a_with_candidate_short_circuit",
            "admit_at_most_one_survivor_to_gate_b",
            "execute_scene_shared_validation",
        ],
        "authorized_new_unique_identity_upper_bound": (
            GATE_A_IDENTITY_UPPER_BOUND
            + GATE_B_IDENTITY_UPPER_BOUND
        ),
        "budget_expansion": 0,
        "automatic_retry": False,
    }
    receipt["authorization_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_authorization(
    *,
    proposal: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    _verify_hash(
        receipt,
        hash_field="authorization_sha256",
        artifact="short_circuit_authorization",
    )
    if (
        receipt.get("authorization_version")
        != AUTHORIZATION_VERSION
        or receipt.get("status") != "authorized"
        or receipt.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or receipt.get("expires_at") != AUTHORIZATION_CUTOFF
        or receipt.get("user_authorization_text")
        != USER_AUTHORIZATION_TEXT
        or receipt.get("budget_expansion") != 0
        or receipt.get("automatic_retry") is not False
    ):
        raise RecoveryShortCircuitError(
            "short_circuit_authorization_invalid"
        )
    granted = datetime.fromisoformat(str(receipt["granted_at"]))
    cutoff = datetime.fromisoformat(AUTHORIZATION_CUTOFF)
    if granted.tzinfo is None or granted > cutoff:
        raise RecoveryShortCircuitError(
            "short_circuit_authorization_time_invalid"
        )
    return receipt


def _write_proposal_artifacts(
    *,
    proposal: Mapping[str, Any],
    output_dir: Path,
) -> None:
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            "short_circuit_proposal_dir_exists:" + str(destination)
        )
    destination.mkdir(parents=True)
    atomic_write_json(destination / "proposal.json", dict(proposal))
    receipt: dict[str, Any] = {
        "receipt_version": (
            "lyx_recovery_short_circuit_shared_validation_"
            "proposal_receipt_v1"
        ),
        "status": "awaiting_user_blanket_authorization",
        "proposal_sha256": proposal["proposal_sha256"],
        "created_at": proposal["created_at"],
        "new_unique_identity_upper_bound": (
            GATE_A_IDENTITY_UPPER_BOUND
            + GATE_B_IDENTITY_UPPER_BOUND
        ),
        "budget_expansion": 0,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    atomic_write_json(destination / "proposal_receipt.json", receipt)


def _write_authorization_artifact(
    *,
    proposal_dir: Path,
    receipt: Mapping[str, Any],
) -> None:
    path = Path(proposal_dir).resolve() / "authorization.json"
    if path.exists():
        raise FileExistsError(
            "short_circuit_authorization_exists:" + str(path)
        )
    atomic_write_json(path, dict(receipt))


def _validated_runtime(
    *,
    proposal_dir: Path,
    original_proposal_path: Path,
    original_authorization_path: Path,
    repair_proposal_path: Path,
    repair_authorization_path: Path,
    governance_dir: Path,
    repository_root: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    AttemptRegistry,
]:
    root = Path(repository_root).resolve()
    amendment = read_json(Path(proposal_dir).resolve() / "proposal.json")
    validate_proposal(proposal=amendment, repository_root=root)
    validate_authorization(
        proposal=amendment,
        receipt=read_json(
            Path(proposal_dir).resolve() / "authorization.json"
        ),
    )
    original = read_json(Path(original_proposal_path).resolve())
    validate_recovery_independent_bo_preflight(
        proposal=original,
        repository_root=root,
    )
    validate_recovery_independent_bo_execution_authorization(
        original,
        receipt=read_json(
            Path(original_authorization_path).resolve()
        ),
    )
    if (
        original.get("proposal_sha256")
        != amendment.get("original_proposal_sha256")
    ):
        raise RecoveryShortCircuitError(
            "short_circuit_original_proposal_drift"
        )
    repair_proposal = validate_repair_proposal(
        proposal=read_json(Path(repair_proposal_path).resolve()),
        repository_root=root,
    )
    repair_authorization = validate_repair_authorization(
        proposal=repair_proposal,
        receipt=read_json(Path(repair_authorization_path).resolve()),
    )
    if (
        repair_proposal.get("proposal_sha256")
        != amendment.get("repair_proposal_sha256")
        or canonical_sha256(repair_authorization)
        != amendment.get("repair_authorization_sha256")
    ):
        raise RecoveryShortCircuitError(
            "short_circuit_repair_binding_drift"
        )
    governance = Path(governance_dir).resolve()
    budget = BudgetContract.proposed_v13_recovery_independent_bo()
    if (
        read_json(governance / "budget_contract.json")
        != budget.to_dict()
    ):
        raise RecoveryShortCircuitError(
            "short_circuit_governance_budget_drift"
        )
    exploration = _exploration_from_payload(
        read_json(governance / "exploration_registry.json")
    )
    registry = AttemptRegistry.open(
        governance / "attempt_registry.json",
        budget_contract=budget,
        exploration_registry=exploration,
    )
    return (
        amendment,
        original,
        repair_proposal,
        repair_authorization,
        registry,
    )


def _execute_or_repair_cell(
    *,
    original: Mapping[str, Any],
    cell: Mapping[str, Any],
    registry: AttemptRegistry,
    execution_dir: Path,
    repair_proposal_path: Path,
    repair_authorization_path: Path,
    original_proposal_path: Path,
    governance_dir: Path,
    repository_root: Path,
    failure_log_dir: Path,
) -> Mapping[str, Any]:
    completion_path = _completion_path(
        execution_dir,
        recovery_id=str(cell["recovery_candidate_id"]),
        record_id=str(cell["record_id"]),
    )
    if completion_path.is_file():
        return _mapping(
            "short_circuit_existing_completion",
            read_json(completion_path),
        )
    try:
        return _execute_search_cell(
            proposal=original,
            cell=cell,
            space=build_bo_search_space("physical_v1"),
            registry=registry,
            output_dir=Path(execution_dir) / "cells",
            parallel_lanes=False,
            progress_callback=None,
        )
    except TypeError as error:
        failure_text = traceback.format_exc()
        if not is_mappingproxy_completion_failure(failure_text):
            raise
        failure_root = Path(failure_log_dir).resolve()
        failure_root.mkdir(parents=True, exist_ok=True)
        failure_path = (
            failure_root
            / (
                str(cell["recovery_candidate_id"])
                + "__"
                + str(cell["record_id"])
                + ".stderr.log"
            )
        )
        failure_path.write_text(failure_text, encoding="utf-8")
        _finalize_ready_cell_locked(
            original_proposal_path=original_proposal_path,
            repair_proposal_path=repair_proposal_path,
            repair_authorization_path=repair_authorization_path,
            governance_dir=governance_dir,
            execution_dir=execution_dir,
            repository_root=repository_root,
            observed_failure_log_path=failure_path,
            require_ready=True,
            expected_cell_sha256=str(cell["cell_sha256"]),
        )
        if not completion_path.is_file():
            raise RecoveryShortCircuitError(
                "short_circuit_repair_did_not_create_completion"
            ) from error
        return _mapping(
            "short_circuit_repaired_completion",
            read_json(completion_path),
        )


def _candidate_summary(
    *,
    recovery_id: str,
    completions: Sequence[Mapping[str, Any]],
    mechanism_complexity: int,
) -> dict[str, Any]:
    selected_maes = [
        float(
            _mapping(
                "short_circuit_selected",
                completion["selected"],
            )["metrics"]["final_motion_mae_bpm"]
        )
        for completion in completions
    ]
    return {
        "recovery_candidate_id": recovery_id,
        "status": (
            "survivor"
            if len(completions) == len(HARD_RECORD_ORDER)
            and all(
                int(item["eligible_candidate_count"]) > 0
                for item in completions
            )
            else "eliminated"
        ),
        "completed_hard_cell_count": len(completions),
        "eliminated_at_record_id": next(
            (
                str(item["record_id"])
                for item in completions
                if int(item["eligible_candidate_count"]) == 0
            ),
            None,
        ),
        "max_selected_eligible_mae": max(selected_maes),
        "eligible_candidate_count_sum": sum(
            int(item["eligible_candidate_count"])
            for item in completions
        ),
        "mechanism_complexity": mechanism_complexity,
        "cell_results": [
            {
                "record_id": item["record_id"],
                "cell_sha256": item["cell_sha256"],
                "completion_sha256": item["completion_sha256"],
                "eligible_candidate_count": item[
                    "eligible_candidate_count"
                ],
                "selected_eligible": item["selected"]["eligible"],
                "selected_mae": item["selected"]["metrics"][
                    "final_motion_mae_bpm"
                ],
            }
            for item in completions
        ],
    }


def execute_gate_a(
    *,
    proposal_dir: Path,
    original_proposal_path: Path,
    original_authorization_path: Path,
    repair_proposal_path: Path,
    repair_authorization_path: Path,
    governance_dir: Path,
    execution_dir: Path,
    repository_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute the exact hard-record order and stop each loser immediately."""

    (
        amendment,
        original,
        _repair_proposal,
        _repair_authorization,
        registry,
    ) = _validated_runtime(
        proposal_dir=proposal_dir,
        original_proposal_path=original_proposal_path,
        original_authorization_path=original_authorization_path,
        repair_proposal_path=repair_proposal_path,
        repair_authorization_path=repair_authorization_path,
        governance_dir=governance_dir,
        repository_root=repository_root,
    )
    cells = _cell_index(original)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    completion_path = destination / "gate_a_completion.json"
    if completion_path.is_file():
        completion = read_json(completion_path)
        _verify_hash(
            completion,
            hash_field="completion_sha256",
            artifact="short_circuit_gate_a_completion",
        )
        return completion
    recovery_complexity = {
        str(item["candidate_id"]): int(item["mechanism_complexity"])
        for item in original["recovery_candidates"]
    }
    summaries: list[dict[str, Any]] = []
    with _exclusive_supervisor_lock(Path(execution_dir).resolve()):
        for recovery_id in REMAINING_RECOVERY_IDS:
            completed: list[Mapping[str, Any]] = []
            for record_id in HARD_RECORD_ORDER:
                cell = cells[(recovery_id, record_id)]
                cell_completion = _execute_or_repair_cell(
                    original=original,
                    cell=cell,
                    registry=registry,
                    execution_dir=Path(execution_dir).resolve(),
                    repair_proposal_path=Path(
                        repair_proposal_path
                    ).resolve(),
                    repair_authorization_path=Path(
                        repair_authorization_path
                    ).resolve(),
                    original_proposal_path=Path(
                        original_proposal_path
                    ).resolve(),
                    governance_dir=Path(governance_dir).resolve(),
                    repository_root=Path(repository_root).resolve(),
                    failure_log_dir=destination / "failure_logs",
                )
                completed.append(cell_completion)
                progress: dict[str, Any] = {
                    "progress_version": (
                        "lyx_recovery_short_circuit_gate_a_progress_v1"
                    ),
                    "proposal_sha256": amendment["proposal_sha256"],
                    "recovery_candidate_id": recovery_id,
                    "last_record_id": record_id,
                    "completed_hard_cell_count": len(completed),
                    "eligible_candidate_count": cell_completion[
                        "eligible_candidate_count"
                    ],
                    "stopped": (
                        int(
                            cell_completion[
                                "eligible_candidate_count"
                            ]
                        )
                        == 0
                    ),
                    "updated_at": _iso_now(),
                }
                progress["progress_sha256"] = canonical_sha256(
                    progress
                )
                atomic_write_json(
                    destination
                    / f"{recovery_id}_progress.json",
                    progress,
                )
                if int(
                    cell_completion["eligible_candidate_count"]
                ) == 0:
                    break
            summaries.append(
                _candidate_summary(
                    recovery_id=recovery_id,
                    completions=completed,
                    mechanism_complexity=recovery_complexity[
                        recovery_id
                    ],
                )
            )
    survivors = [
        item for item in summaries if item["status"] == "survivor"
    ]
    survivors.sort(
        key=lambda item: (
            float(item["max_selected_eligible_mae"]),
            -int(item["eligible_candidate_count_sum"]),
            int(item["mechanism_complexity"]),
            str(item["recovery_candidate_id"]),
        )
    )
    selected = (
        str(survivors[0]["recovery_candidate_id"])
        if survivors
        else None
    )
    completion = {
        "completion_version": GATE_A_COMPLETION_VERSION,
        "status": (
            "survivor_selected"
            if selected is not None
            else "no_recovery_survivor"
        ),
        "proposal_sha256": amendment["proposal_sha256"],
        "original_proposal_sha256": original["proposal_sha256"],
        "control_recovery_status": "eliminated_before_gate_a",
        "candidate_summaries": summaries,
        "survivor_ids": [
            item["recovery_candidate_id"] for item in survivors
        ],
        "selected_recovery_candidate_id": selected,
        "automatic_gate_b_execution": False,
        "completed_at": _iso_now(),
        "next_state": (
            "ready_for_authorized_gate_b"
            if selected is not None
            else "terminal_no_safe_recovery_candidate"
        ),
    }
    completion["completion_sha256"] = canonical_sha256(completion)
    atomic_write_json(completion_path, completion)
    return completion


def _shared_candidates() -> tuple[BOCandidate, ...]:
    candidates = tuple(
        candidate
        for candidate in build_bo_search_space(
            "physical_v1"
        ).candidates
        if int(candidate.requested_params["fs_target"])
        == SHARED_FS_TARGET
    )
    if (
        len(candidates) != SHARED_GRID_SIZE
        or len({item.candidate_id for item in candidates})
        != SHARED_GRID_SIZE
        or len({item.coordinate for item in candidates})
        != SHARED_GRID_SIZE
    ):
        raise RecoveryShortCircuitError(
            "shared_validation_grid_invalid"
        )
    return tuple(sorted(candidates, key=lambda item: item.coordinate))


def _evaluate_shared_candidate(
    *,
    original: Mapping[str, Any],
    cell: Mapping[str, Any],
    candidate: BOCandidate,
    registry: AttemptRegistry,
    spectral_dir: Path,
) -> dict[str, Any]:
    item = build_recovery_independent_bo_identity(
        proposal=original,
        cell=cell,
        candidate=candidate,
    )
    identity = _attempt_identity_from_item(item)
    registry.register_identity(identity)
    before = registry.matrix_execution_summary((identity,))
    if (
        before["failed_attempt_count"] != 0
        or before["retry_count"] != 0
    ):
        raise RecoveryShortCircuitError(
            "shared_validation_retry_requires_new_proposal:"
            + identity.sha256
        )
    result = stage_r_cache.execute_stage_r_identity(
        registry=registry,
        item=item,
        numerical_runner=run_stage_r_numerical_identity,
        spectral_audit_dir=Path(spectral_dir),
        allow_retry=False,
    )
    metrics = _mapping(
        "shared_validation_metrics",
        result.get("metrics"),
    )
    spectral = _mapping(
        "shared_validation_spectral",
        result.get("spectral_audit"),
    )
    constraints = _constraint_values(
        metrics=metrics,
        spectral=spectral,
        current=_mapping(
            "shared_validation_current_metrics",
            cell.get("current_metrics"),
        ),
        independent=_mapping(
            "shared_validation_independent_metrics",
            cell.get("independent_metrics"),
        ),
        true_rise_applicable=bool(
            cell["true_rise_applicable"]
        ),
    )
    row = {
        "candidate_id": candidate.candidate_id,
        "coordinate": list(candidate.coordinate),
        "requested_params": dict(candidate.requested_params),
        "identity_sha256": identity.sha256,
        "cache_hit": bool(result["cache_hit"]),
        "metrics": dict(metrics),
        "spectral_audit": dict(spectral),
        "constraints": [float(value) for value in constraints],
        "eligible": all(value <= 0.0 for value in constraints),
        "objective": float(metrics["final_motion_mae_bpm"]),
    }
    # Do not allow MappingProxyType or another runtime view into any receipt.
    return json.loads(json.dumps(row, ensure_ascii=False))


def _direct_shared_neighbors(
    candidate: BOCandidate,
    *,
    by_coordinate: Mapping[tuple[int, ...], BOCandidate],
) -> tuple[BOCandidate, ...]:
    neighbors: list[BOCandidate] = []
    coordinate = candidate.coordinate
    for dimension in (1, 2, 3):
        for delta in (-1, 1):
            adjacent = list(coordinate)
            adjacent[dimension] += delta
            found = by_coordinate.get(tuple(adjacent))
            if found is not None:
                neighbors.append(found)
    return tuple(
        sorted(neighbors, key=lambda item: item.coordinate)
    )


def _rank_training_candidates(
    *,
    candidates: Sequence[BOCandidate],
    train_record_ids: Sequence[str],
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(train_record_ids) != 2:
        raise RecoveryShortCircuitError(
            "shared_validation_train_pair_invalid"
        )
    by_coordinate = {
        candidate.coordinate: candidate for candidate in candidates
    }
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        train_rows = [
            rows[(record_id, candidate.candidate_id)]
            for record_id in train_record_ids
        ]
        if not all(bool(row["eligible"]) for row in train_rows):
            continue
        train_maes = [
            float(row["metrics"]["final_motion_mae_bpm"])
            for row in train_rows
        ]
        worst = max(train_maes)
        mean = statistics.fmean(train_maes)
        neighbors = _direct_shared_neighbors(
            candidate,
            by_coordinate=by_coordinate,
        )
        supported: list[str] = []
        cliffs: list[str] = []
        for neighbor in neighbors:
            neighbor_rows = [
                rows[(record_id, neighbor.candidate_id)]
                for record_id in train_record_ids
            ]
            neighbor_maes = [
                float(
                    row["metrics"]["final_motion_mae_bpm"]
                )
                for row in neighbor_rows
            ]
            if (
                all(bool(row["eligible"]) for row in neighbor_rows)
                and max(neighbor_maes) <= worst + 1.0
            ):
                supported.append(neighbor.candidate_id)
            if worst <= 5.0 and max(neighbor_maes) >= 10.0:
                cliffs.append(neighbor.candidate_id)
        support_ratio = (
            len(supported) / len(neighbors) if neighbors else 0.0
        )
        ranked.append(
            {
                "candidate_id": candidate.candidate_id,
                "coordinate": list(candidate.coordinate),
                "requested_params": dict(
                    candidate.requested_params
                ),
                "worst_train_mae": worst,
                "mean_train_mae": mean,
                "support_neighbor_count": len(supported),
                "neighbor_count": len(neighbors),
                "support_ratio": support_ratio,
                "supported_neighbor_ids": supported,
                "parameter_cliff_count": len(cliffs),
                "parameter_cliff_neighbor_ids": cliffs,
                "train_rows": [
                    {
                        "record_id": record_id,
                        "identity_sha256": row["identity_sha256"],
                        "eligible": row["eligible"],
                        "mae": row["metrics"][
                            "final_motion_mae_bpm"
                        ],
                        "longest_e10_run_windows": row[
                            "metrics"
                        ]["longest_e10_run_windows"],
                        "longest_e20_run_windows": row[
                            "metrics"
                        ]["longest_e20_run_windows"],
                    }
                    for record_id, row in zip(
                        train_record_ids,
                        train_rows,
                        strict=True,
                    )
                ],
            }
        )
    ranked.sort(
        key=lambda item: (
            float(item["worst_train_mae"]),
            float(item["mean_train_mae"]),
            -float(item["support_ratio"]),
            int(item["parameter_cliff_count"]),
            tuple(item["coordinate"]),
        )
    )
    return ranked


def _scene_shared_decision(
    *,
    scene: str,
    folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    revealed = [
        fold for fold in folds if fold["status"] == "revealed"
    ]
    if len(revealed) != 3:
        return {
            "scene": scene,
            "status": "no_safe_shared_parameter",
            "passed": False,
            "reason": "one_or_more_folds_have_no_training_candidate",
            "fold_count": len(folds),
            "revealed_fold_count": len(revealed),
        }
    held_maes = [
        float(fold["held_out"]["mae"]) for fold in revealed
    ]
    train_gaps = [
        float(fold["held_out"]["mae"])
        - float(fold["selection"]["worst_train_mae"])
        for fold in revealed
    ]
    lite_deltas = [
        float(fold["held_out"]["mae"])
        - float(fold["held_out"]["independent_bo_lite_mae"])
        for fold in revealed
    ]
    all_eligible = all(
        bool(fold["held_out"]["eligible"])
        for fold in revealed
    )
    mean_mae = statistics.fmean(held_maes)
    median_gap = statistics.median(train_gaps)
    max_gap = max(train_gaps)
    no_disaster = max(held_maes) < 10.0
    run_delta_pass = (
        True
        if scene != "run"
        else (
            statistics.fmean(lite_deltas) <= 1.0
            and max(lite_deltas) <= 2.0
        )
    )
    passed = (
        all_eligible
        and mean_mae <= 5.0
        and median_gap <= 2.0
        and max_gap <= 5.0
        and no_disaster
        and run_delta_pass
    )
    return {
        "scene": scene,
        "status": (
            "scene_shared_validation_passed"
            if passed
            else "scene_shared_validation_failed"
        ),
        "passed": passed,
        "fold_count": 3,
        "revealed_fold_count": 3,
        "all_held_out_eligible": all_eligible,
        "mean_held_out_mae": mean_mae,
        "max_held_out_mae": max(held_maes),
        "median_train_to_test_gap": median_gap,
        "max_train_to_test_gap": max_gap,
        "mean_independent_bo_lite_delta": statistics.fmean(
            lite_deltas
        ),
        "max_independent_bo_lite_delta": max(lite_deltas),
        "no_new_mae_disaster": no_disaster,
        "run_delta_pass": run_delta_pass,
    }


def execute_gate_b(
    *,
    proposal_dir: Path,
    original_proposal_path: Path,
    original_authorization_path: Path,
    repair_proposal_path: Path,
    repair_authorization_path: Path,
    governance_dir: Path,
    execution_dir: Path,
    repository_root: Path,
    gate_a_output_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute deterministic scene-wise LOO selection for one survivor."""

    (
        amendment,
        original,
        _repair_proposal,
        _repair_authorization,
        registry,
    ) = _validated_runtime(
        proposal_dir=proposal_dir,
        original_proposal_path=original_proposal_path,
        original_authorization_path=original_authorization_path,
        repair_proposal_path=repair_proposal_path,
        repair_authorization_path=repair_authorization_path,
        governance_dir=governance_dir,
        repository_root=repository_root,
    )
    gate_a_path = (
        Path(gate_a_output_dir).resolve()
        / "gate_a_completion.json"
    )
    gate_a = read_json(gate_a_path)
    _verify_hash(
        gate_a,
        hash_field="completion_sha256",
        artifact="short_circuit_gate_a_completion",
    )
    if (
        gate_a.get("proposal_sha256")
        != amendment.get("proposal_sha256")
        or gate_a.get("status") != "survivor_selected"
    ):
        raise RecoveryShortCircuitError(
            "shared_validation_gate_a_not_ready"
        )
    recovery_id = str(
        gate_a["selected_recovery_candidate_id"]
    )
    if recovery_id not in REMAINING_RECOVERY_IDS:
        raise RecoveryShortCircuitError(
            "shared_validation_recovery_not_authorized"
        )
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    completion_path = destination / "gate_b_completion.json"
    if completion_path.is_file():
        completion = read_json(completion_path)
        _verify_hash(
            completion,
            hash_field="completion_sha256",
            artifact="shared_validation_completion",
        )
        return completion
    candidates = _shared_candidates()
    candidate_by_id = {
        item.candidate_id: item for item in candidates
    }
    cells = _cell_index(original)
    scene_records: dict[str, list[str]] = {}
    for (candidate_id, record_id), cell in cells.items():
        if candidate_id != recovery_id:
            continue
        scene_records.setdefault(str(cell["scene"]), []).append(
            record_id
        )
    if (
        set(scene_records) != {"xiezi", "jianpan", "run", "kaihe"}
        or any(len(records) != 3 for records in scene_records.values())
    ):
        raise RecoveryShortCircuitError(
            "shared_validation_record_panel_invalid"
        )
    rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    folds: list[dict[str, Any]] = []
    spectral_dir = destination / "spectral_audits"
    fold_root = destination / "folds"
    with _exclusive_supervisor_lock(Path(execution_dir).resolve()):
        for scene in ("xiezi", "jianpan", "run", "kaihe"):
            records = sorted(scene_records[scene])
            for held_out_id in records:
                train_ids = [
                    item for item in records if item != held_out_id
                ]
                fold_id = f"{scene}__held_out__{held_out_id}"
                fold_dir = fold_root / fold_id
                freeze_path = fold_dir / "freeze_receipt.json"
                reveal_path = fold_dir / "reveal_receipt.json"
                if reveal_path.is_file():
                    reveal = read_json(reveal_path)
                    _verify_hash(
                        reveal,
                        hash_field="receipt_sha256",
                        artifact="shared_validation_reveal",
                    )
                    folds.append(reveal)
                    continue
                for train_id in train_ids:
                    cell = cells[(recovery_id, train_id)]
                    for candidate in candidates:
                        key = (train_id, candidate.candidate_id)
                        if key not in rows:
                            rows[key] = _evaluate_shared_candidate(
                                original=original,
                                cell=cell,
                                candidate=candidate,
                                registry=registry,
                                spectral_dir=spectral_dir,
                            )
                ranked = _rank_training_candidates(
                    candidates=candidates,
                    train_record_ids=train_ids,
                    rows=rows,
                )
                fold_dir.mkdir(parents=True, exist_ok=True)
                if not ranked:
                    freeze = {
                        "receipt_version": (
                            "lyx_recovery_scene_shared_freeze_v1"
                        ),
                        "status": "no_safe_training_candidate",
                        "proposal_sha256": amendment[
                            "proposal_sha256"
                        ],
                        "gate_a_completion_sha256": gate_a[
                            "completion_sha256"
                        ],
                        "recovery_candidate_id": recovery_id,
                        "fold_id": fold_id,
                        "scene": scene,
                        "train_record_ids": train_ids,
                        "held_out_record_id": held_out_id,
                        "candidate_count": SHARED_GRID_SIZE,
                        "eligible_training_candidate_count": 0,
                        "selected_candidate_id": None,
                        "frozen_at": _iso_now(),
                    }
                    freeze["receipt_sha256"] = canonical_sha256(
                        freeze
                    )
                    atomic_write_json(freeze_path, freeze)
                    fold_result = {
                        "receipt_version": (
                            "lyx_recovery_scene_shared_reveal_v1"
                        ),
                        "status": "no_safe_training_candidate",
                        "proposal_sha256": amendment[
                            "proposal_sha256"
                        ],
                        "freeze_receipt_sha256": freeze[
                            "receipt_sha256"
                        ],
                        "fold_id": fold_id,
                        "scene": scene,
                        "train_record_ids": train_ids,
                        "held_out_record_id": held_out_id,
                        "selection": None,
                        "held_out": None,
                        "revealed_at": None,
                    }
                    fold_result["receipt_sha256"] = (
                        canonical_sha256(fold_result)
                    )
                    atomic_write_json(reveal_path, fold_result)
                    folds.append(fold_result)
                    continue
                selected = ranked[0]
                freeze = {
                    "receipt_version": (
                        "lyx_recovery_scene_shared_freeze_v1"
                    ),
                    "status": "parameter_frozen",
                    "proposal_sha256": amendment[
                        "proposal_sha256"
                    ],
                    "gate_a_completion_sha256": gate_a[
                        "completion_sha256"
                    ],
                    "recovery_candidate_id": recovery_id,
                    "fold_id": fold_id,
                    "scene": scene,
                    "train_record_ids": train_ids,
                    "held_out_record_id": held_out_id,
                    "candidate_count": SHARED_GRID_SIZE,
                    "eligible_training_candidate_count": len(
                        ranked
                    ),
                    "selected_candidate_id": selected[
                        "candidate_id"
                    ],
                    "selection": selected,
                    "frozen_at": _iso_now(),
                }
                freeze["receipt_sha256"] = canonical_sha256(freeze)
                atomic_write_json(freeze_path, freeze)
                selected_candidate = candidate_by_id[
                    str(selected["candidate_id"])
                ]
                held_key = (
                    held_out_id,
                    selected_candidate.candidate_id,
                )
                if held_key not in rows:
                    rows[held_key] = _evaluate_shared_candidate(
                        original=original,
                        cell=cells[(recovery_id, held_out_id)],
                        candidate=selected_candidate,
                        registry=registry,
                        spectral_dir=spectral_dir,
                    )
                held = rows[held_key]
                independent = _mapping(
                    "shared_validation_held_independent",
                    cells[(recovery_id, held_out_id)][
                        "independent_metrics"
                    ],
                )
                fold_result = {
                    "receipt_version": (
                        "lyx_recovery_scene_shared_reveal_v1"
                    ),
                    "status": "revealed",
                    "proposal_sha256": amendment[
                        "proposal_sha256"
                    ],
                    "freeze_receipt_sha256": freeze[
                        "receipt_sha256"
                    ],
                    "fold_id": fold_id,
                    "scene": scene,
                    "train_record_ids": train_ids,
                    "held_out_record_id": held_out_id,
                    "selection": selected,
                    "held_out": {
                        "identity_sha256": held[
                            "identity_sha256"
                        ],
                        "eligible": held["eligible"],
                        "mae": held["metrics"][
                            "final_motion_mae_bpm"
                        ],
                        "independent_bo_lite_mae": independent[
                            "final_motion_mae_bpm"
                        ],
                        "longest_e10_run_windows": held[
                            "metrics"
                        ]["longest_e10_run_windows"],
                        "longest_e20_run_windows": held[
                            "metrics"
                        ]["longest_e20_run_windows"],
                        "right_censored_recovery_count": held[
                            "metrics"
                        ]["right_censored_recovery_count"],
                        "max_rise_underestimate_bpm": held[
                            "metrics"
                        ]["max_rise_underestimate_bpm"],
                        "spectral_gate_pass": held[
                            "spectral_audit"
                        ]["spectral_gate_pass"],
                        "stability_pass": held[
                            "spectral_audit"
                        ]["stability_pass"],
                        "constraints": held["constraints"],
                    },
                    "revealed_at": _iso_now(),
                }
                fold_result["receipt_sha256"] = canonical_sha256(
                    fold_result
                )
                atomic_write_json(reveal_path, fold_result)
                folds.append(fold_result)
                progress = {
                    "progress_version": (
                        "lyx_recovery_scene_shared_progress_v1"
                    ),
                    "proposal_sha256": amendment[
                        "proposal_sha256"
                    ],
                    "recovery_candidate_id": recovery_id,
                    "completed_fold_count": len(folds),
                    "total_fold_count": 12,
                    "last_fold_id": fold_id,
                    "updated_at": _iso_now(),
                }
                progress["progress_sha256"] = canonical_sha256(
                    progress
                )
                atomic_write_json(
                    destination / "progress.json",
                    progress,
                )
    scene_decisions = [
        _scene_shared_decision(
            scene=scene,
            folds=[
                fold for fold in folds if fold["scene"] == scene
            ],
        )
        for scene in ("xiezi", "jianpan", "run", "kaihe")
    ]
    completion = {
        "completion_version": GATE_B_COMPLETION_VERSION,
        "status": (
            "all_scenes_passed"
            if all(item["passed"] for item in scene_decisions)
            else "one_or_more_scenes_failed"
        ),
        "proposal_sha256": amendment["proposal_sha256"],
        "gate_a_completion_sha256": gate_a[
            "completion_sha256"
        ],
        "recovery_candidate_id": recovery_id,
        "fold_count": len(folds),
        "scene_decisions": scene_decisions,
        "all_scene_shared_validation_passed": all(
            item["passed"] for item in scene_decisions
        ),
        "completed_at": _iso_now(),
        "evidence_class": "development_reuse_pilot",
        "automatic_stage_f_execution": False,
        "next_state": "ready_for_final_reporting",
    }
    completion["completion_sha256"] = canonical_sha256(
        completion
    )
    atomic_write_json(completion_path, completion)
    return completion


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
        "repair_proposal",
        "repair_authorization",
        "governance_dir",
        "execution_dir",
        "spec",
        "repository_root",
        "output_dir",
    ):
        propose.add_argument(
            f"--{name.replace('_', '-')}",
            required=True,
        )
    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--proposal-dir", required=True)
    authorize.add_argument("--repository-root", required=True)
    authorize.add_argument("--granted-at", required=True)
    gate_a = subparsers.add_parser("execute-gate-a")
    for name in (
        "proposal_dir",
        "original_proposal",
        "original_authorization",
        "repair_proposal",
        "repair_authorization",
        "governance_dir",
        "execution_dir",
        "repository_root",
        "output_dir",
    ):
        gate_a.add_argument(
            f"--{name.replace('_', '-')}",
            required=True,
        )
    gate_b = subparsers.add_parser("execute-gate-b")
    for name in (
        "proposal_dir",
        "original_proposal",
        "original_authorization",
        "repair_proposal",
        "repair_authorization",
        "governance_dir",
        "execution_dir",
        "repository_root",
        "gate_a_output_dir",
        "output_dir",
    ):
        gate_b.add_argument(
            f"--{name.replace('_', '-')}",
            required=True,
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "propose":
        proposal = build_proposal(
            original_proposal_path=Path(args.original_proposal),
            original_authorization_path=Path(
                args.original_authorization
            ),
            repair_proposal_path=Path(args.repair_proposal),
            repair_authorization_path=Path(
                args.repair_authorization
            ),
            governance_dir=Path(args.governance_dir),
            execution_dir=Path(args.execution_dir),
            spec_path=Path(args.spec),
            repository_root=Path(args.repository_root),
            tool_path=Path(__file__),
        )
        _write_proposal_artifacts(
            proposal=proposal,
            output_dir=Path(args.output_dir),
        )
        print(
            json.dumps(
                {
                    "status": proposal["status"],
                    "proposal_sha256": proposal["proposal_sha256"],
                    "new_unique_identity_upper_bound": (
                        GATE_A_IDENTITY_UPPER_BOUND
                        + GATE_B_IDENTITY_UPPER_BOUND
                    ),
                    "budget_expansion": 0,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "authorize":
        proposal_dir = Path(args.proposal_dir).resolve()
        proposal = read_json(proposal_dir / "proposal.json")
        validate_proposal(
            proposal=proposal,
            repository_root=Path(args.repository_root),
        )
        receipt = build_authorization(
            proposal=proposal,
            granted_at=args.granted_at,
        )
        _write_authorization_artifact(
            proposal_dir=proposal_dir,
            receipt=receipt,
        )
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "authorization_sha256": receipt[
                        "authorization_sha256"
                    ],
                    "expires_at": receipt["expires_at"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "execute-gate-a":
        completion = execute_gate_a(
            proposal_dir=Path(args.proposal_dir),
            original_proposal_path=Path(args.original_proposal),
            original_authorization_path=Path(
                args.original_authorization
            ),
            repair_proposal_path=Path(args.repair_proposal),
            repair_authorization_path=Path(
                args.repair_authorization
            ),
            governance_dir=Path(args.governance_dir),
            execution_dir=Path(args.execution_dir),
            repository_root=Path(args.repository_root),
            output_dir=Path(args.output_dir),
        )
        print(
            json.dumps(
                {
                    "status": completion["status"],
                    "selected_recovery_candidate_id": completion[
                        "selected_recovery_candidate_id"
                    ],
                    "completion_sha256": completion[
                        "completion_sha256"
                    ],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "execute-gate-b":
        completion = execute_gate_b(
            proposal_dir=Path(args.proposal_dir),
            original_proposal_path=Path(args.original_proposal),
            original_authorization_path=Path(
                args.original_authorization
            ),
            repair_proposal_path=Path(args.repair_proposal),
            repair_authorization_path=Path(
                args.repair_authorization
            ),
            governance_dir=Path(args.governance_dir),
            execution_dir=Path(args.execution_dir),
            repository_root=Path(args.repository_root),
            gate_a_output_dir=Path(args.gate_a_output_dir),
            output_dir=Path(args.output_dir),
        )
        print(
            json.dumps(
                {
                    "status": completion["status"],
                    "recovery_candidate_id": completion[
                        "recovery_candidate_id"
                    ],
                    "completion_sha256": completion[
                        "completion_sha256"
                    ],
                },
                ensure_ascii=False,
            )
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
