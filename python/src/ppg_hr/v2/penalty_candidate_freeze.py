"""Fail-closed freezing for the Stage P motion-penalty registry."""

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
from .penalty_candidates import (
    PenaltyCandidate,
    PenaltyCandidateError,
    penalty_candidates_v1,
)
from .recovery_contracts import canonical_sha256, require_sha256

_EXPECTED_IDS = (
    "current_soft_penalty_control_v1",
    "resolution_adaptive_width_v1",
    "trusted_history_corridor_v1",
)


def freeze_penalty_candidate_registry(
    candidates: Sequence[PenaltyCandidate],
    *,
    solver_hash: str,
    config_schema_hash: str,
) -> dict[str, Any]:
    """Freeze exactly one control and two new penalty strategies."""

    try:
        require_sha256("solver_hash", solver_hash)
        require_sha256("config_schema_hash", config_schema_hash)
    except ValueError as exc:
        raise PenaltyCandidateError("invalid_penalty_registry_hash") from exc
    frozen = tuple(candidates)
    if tuple(candidate.penalty_id for candidate in frozen) != _EXPECTED_IDS:
        raise PenaltyCandidateError("penalty_identity_or_order_mismatch")
    if len(frozen) != 3:
        raise PenaltyCandidateError("penalty_count_must_be_three")
    if sum(candidate.design_role == "control" for candidate in frozen) != 1:
        raise PenaltyCandidateError("exactly_one_penalty_control_required")
    if len({candidate.sha256 for candidate in frozen}) != 3:
        raise PenaltyCandidateError("duplicate_penalty_candidate_identity")

    payload = {
        "registry_version": "lyx_penalty_registry_v1",
        "status": "frozen_zero_formal_runs",
        "solver_hash": solver_hash,
        "config_schema_hash": config_schema_hash,
        "penalty_count": 3,
        "control_penalty_id": _EXPECTED_IDS[0],
        "new_penalty_count": 2,
        "formal_solver_run_count": 0,
        "uses_reference_hr_runtime": False,
        "causal_online_ready": False,
        "runtime_information_boundary": (
            "offline_v2_motion_segmentation_with_causal_local_penalty_state"
        ),
        "existing_capabilities_not_claimed_as_new": [
            "confidence_scaled_penalty_weight",
            "conditional_harmonic_penalty",
            "single_previous_track_protection",
            "motion_core_challenger_protection_suppression",
        ],
        "selection_hard_gate_order": [
            "spectral_gate_contract_v1",
            "l10_l20_gates",
            "mae_gate",
            "right_censored_recovery_gate",
            "true_rise_protection_gate",
        ],
        "selection_ranking_key": [
            "hard_gate_failure_count",
            "right_censored_recovery_count",
            "worst_l10",
            "worst_mae",
            "mean_mae",
            "mechanism_complexity",
            "penalty_id",
        ],
        "selection_sort_direction": "ascending_all_fields",
        "tie_rule": "penalty_id_ascending",
        "no_fourth_strategy_after_freeze": True,
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


def freeze_penalty_candidate_artifacts(
    *,
    output_dir: Path,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Atomically freeze Stage P without evaluating any development record."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise PenaltyCandidateError(
            "penalty_candidate_freeze_output_already_exists"
        )
    if source_root is None:
        source_root = Path(__file__).resolve().parents[2]
    source_identity = runtime_source_identity(Path(source_root))
    config_identity = runtime_config_schema_identity()
    registry = freeze_penalty_candidate_registry(
        penalty_candidates_v1(),
        solver_hash=source_identity["source_bundle_sha256"],
        config_schema_hash=config_identity["config_schema_sha256"],
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(
        f".{output_dir.name}.{uuid.uuid4().hex}.staging"
    )
    try:
        os.makedirs(filesystem_path(staging))
        registry_path = staging / "penalty_registry.json"
        write_json_atomically(registry_path, registry)
        receipt = {
            "receipt_version": "lyx_penalty_freeze_receipt_v1",
            "status": "frozen_zero_formal_runs",
            **source_identity,
            **config_identity,
            "penalty_registry_sha256": registry["registry_sha256"],
            "formal_solver_run_count": 0,
            "diagnostic_solver_run_count": 0,
            "independent_bo_run_count": 0,
            "development_record_read_count": 0,
            "preflight_status": "awaiting_interaction_matrix",
            "artifacts": {
                registry_path.name: file_sha256(registry_path),
            },
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        write_json_atomically(
            staging / "penalty_freeze_receipt.json",
            receipt,
        )
        os.replace(filesystem_path(staging), filesystem_path(output_dir))
    except BaseException:
        if staging.exists():
            shutil.rmtree(filesystem_path(staging))
        raise
    return receipt
