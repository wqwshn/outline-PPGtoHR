"""Fail-closed artifact freezing for Stage R recovery contracts."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .experiment_freeze_utils import (
    file_sha256,
    filesystem_path,
    runtime_config_schema_identity,
    runtime_source_identity,
    write_json_atomically,
)
from .recovery_candidates import (
    RecoveryCandidate,
    RecoveryCandidateError,
    recovery_candidates_v1,
)
from .recovery_contracts import (
    RECOVERY_PREFLIGHT_HASH_FIELDS,
    canonical_sha256,
    require_sha256,
)
from .recovery_selection import recovery_selection_contract_v1

_EXPECTED_IDS = (
    "current_fixed_floor_control_v1",
    "relative_gap_timeout_v1",
    "relative_gap_rise_guard_v1",
)

def freeze_recovery_candidate_registry(
    candidates: Sequence[RecoveryCandidate],
    *,
    solver_hash: str,
    config_schema_hash: str,
) -> dict[str, Any]:
    """Freeze exactly one control and two new candidates without solver runs."""

    try:
        require_sha256("solver_hash", solver_hash)
        require_sha256("config_schema_hash", config_schema_hash)
    except ValueError as exc:
        raise RecoveryCandidateError("invalid_recovery_registry_hash") from exc
    frozen = tuple(candidates)
    if len(frozen) != 3:
        raise RecoveryCandidateError("candidate_count_must_be_three")
    if tuple(candidate.candidate_id for candidate in frozen) != _EXPECTED_IDS:
        raise RecoveryCandidateError("candidate_identity_or_order_mismatch")
    if sum(candidate.design_role == "control" for candidate in frozen) != 1:
        raise RecoveryCandidateError("exactly_one_control_required")
    if len({candidate.sha256 for candidate in frozen}) != len(frozen):
        raise RecoveryCandidateError("duplicate_candidate_identity")
    payload = {
        "registry_version": "lyx_recovery_candidate_registry_v1",
        "status": "frozen_zero_formal_runs",
        "solver_hash": solver_hash,
        "config_schema_hash": config_schema_hash,
        "candidate_count": 3,
        "control_candidate_id": _EXPECTED_IDS[0],
        "new_candidate_count": 2,
        "formal_solver_run_count": 0,
        "uses_reference_hr_online": False,
        "candidates": [
            {
                **candidate.to_dict(),
                "candidate_sha256": candidate.sha256,
            }
            for candidate in frozen
        ],
    }
    payload["registry_sha256"] = canonical_sha256(payload)
    return payload


def freeze_recovery_candidate_artifacts(
    *,
    output_dir: Path,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Atomically freeze Stage R contracts without evaluating any record."""

    if output_dir.exists():
        raise RecoveryCandidateError(
            "recovery_candidate_freeze_output_already_exists"
        )
    if source_root is None:
        source_root = Path(__file__).resolve().parents[2]
    source_root = Path(source_root).resolve()
    source_identity = runtime_source_identity(source_root)
    source_files = source_identity["source_files"]
    source_bundle_sha256 = source_identity["source_bundle_sha256"]

    config_identity = runtime_config_schema_identity()
    config_schemas = config_identity["config_schemas"]
    config_schema_sha256 = config_identity["config_schema_sha256"]
    registry = freeze_recovery_candidate_registry(
        recovery_candidates_v1(),
        solver_hash=source_bundle_sha256,
        config_schema_hash=config_schema_sha256,
    )
    selection = recovery_selection_contract_v1()

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(
        f".{output_dir.name}.{uuid.uuid4().hex}.staging"
    )
    try:
        os.makedirs(filesystem_path(staging))
        registry_path = staging / "recovery_candidate_registry.json"
        selection_path = staging / "recovery_selection_contract.json"
        write_json_atomically(registry_path, registry)
        write_json_atomically(selection_path, selection)
        receipt = {
            "receipt_version": "lyx_recovery_candidate_freeze_receipt_v1",
            "status": "frozen_zero_formal_runs",
            "source_files": source_files,
            "source_bundle_sha256": source_bundle_sha256,
            "config_schemas": config_schemas,
            "config_schema_sha256": config_schema_sha256,
            "recovery_candidate_registry_sha256": (
                registry["registry_sha256"]
            ),
            "recovery_selection_contract_sha256": (
                selection["contract_sha256"]
            ),
            "formal_solver_run_count": 0,
            "diagnostic_solver_run_count": 0,
            "independent_bo_run_count": 0,
            "preflight_status": "awaiting_full_preflight_contracts",
            "required_preflight_hashes": list(
                RECOVERY_PREFLIGHT_HASH_FIELDS
            ),
            "artifacts": {
                registry_path.name: file_sha256(registry_path),
                selection_path.name: file_sha256(selection_path),
            },
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        write_json_atomically(
            staging / "recovery_candidate_freeze_receipt.json",
            receipt,
        )
        os.replace(filesystem_path(staging), filesystem_path(output_dir))
    except BaseException:
        if staging.exists():
            shutil.rmtree(filesystem_path(staging))
        raise
    return receipt
