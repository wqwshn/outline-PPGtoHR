"""Frozen contracts shared by the LYX twelve-slot fold replay."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import BudgetContract
from .recovery_stage_p_contracts import StagePPlanError

FOLD_REPLAY_STAGE = "fold_replay"
EXPECTED_SCENE_COUNTS = {
    "jianpan": 3,
    "kaihe": 3,
    "run": 3,
    "xiezi": 3,
}
EXPECTED_RECORD_COUNT = 12
EXPECTED_PROFILE_COUNT = 8
EXPECTED_LOGICAL_SLOT_COUNT = 12
MAX_SUPPLEMENTAL_IDENTITY_COUNT = 12

TRAINING_SOURCE_FIELDS = ("record_id", "scene", "profile_rows")
TARGET_IDENTITY_SOURCE_FIELDS = ("sample_id", "record_id")
TARGET_RESULT_SOURCE_FIELDS = ("record_id", "scene", "selected_row")

SELECTION_RANKING_KEY = (
    "worst_training_l10",
    "worst_training_recovery_delay_s",
    "worst_training_mae_bpm",
    "mean_training_mae_bpm",
    "spectral_invalid_window_count",
    "negative_spectral_valid_window_count",
    "actual_taps_runtime_complexity_proxy",
    "filter_profile_id",
)

TARGET_FAILURE_CATEGORIES = frozenset(
    {
        "no_safe_shared_candidate",
        "metric_or_mask_contract_failure",
        "spectral_gate_contract_v1",
        "independent_l10_gate",
        "independent_l20_gate",
        "independent_mae_gate",
        "new_right_censored_recovery",
        "current_l10_catastrophic_regression",
        "current_mae_gate",
        "true_rise_underestimate_gate",
        "identity_mismatch_requires_supplement",
    }
)


class FoldReplayError(StagePPlanError):
    """A fold replay artifact violates the frozen experiment contract."""


def require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FoldReplayError(f"{name}_must_be_object")
    return value


def require_list(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise FoldReplayError(f"{name}_must_be_array")
    return value


def require_hash(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FoldReplayError(f"{name}_must_be_lowercase_sha256")
    return value


def verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    artifact_name: str,
) -> str:
    declared = require_hash(hash_field, payload.get(hash_field))
    body = {key: value for key, value in payload.items() if key != hash_field}
    if canonical_sha256(body) != declared:
        raise FoldReplayError(f"{artifact_name}_hash_mismatch")
    return declared


def finite_float(name: str, value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise FoldReplayError(f"{name}_must_be_finite") from error
    if not math.isfinite(number):
        raise FoldReplayError(f"{name}_must_be_finite")
    return number


def nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise FoldReplayError(f"{name}_must_be_nonnegative_integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise FoldReplayError(f"{name}_must_be_nonnegative_integer") from error
    if number < 0 or number != value:
        raise FoldReplayError(f"{name}_must_be_nonnegative_integer")
    return number


def validate_scene_panel(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Validate the exact four-scene, three-record development panel."""

    scene_by_record: dict[str, str] = {}
    coordinates: set[tuple[str, str]] = set()
    profile_ids: set[str] = set()
    for raw in rows:
        row = require_mapping("fold_replay_final_row", raw)
        record_id = str(row.get("record_id", ""))
        scene = str(row.get("scene", ""))
        profile_id = str(row.get("filter_profile_id", ""))
        if not record_id or not scene or not profile_id:
            raise FoldReplayError("fold_replay_row_coordinate_missing")
        existing_scene = scene_by_record.setdefault(record_id, scene)
        if existing_scene != scene:
            raise FoldReplayError("fold_replay_record_scene_mismatch")
        coordinate = (record_id, profile_id)
        if coordinate in coordinates:
            raise FoldReplayError("fold_replay_duplicate_coordinate")
        coordinates.add(coordinate)
        profile_ids.add(profile_id)
    counts = {
        scene: sum(value == scene for value in scene_by_record.values())
        for scene in EXPECTED_SCENE_COUNTS
    }
    if (
        len(rows) != EXPECTED_RECORD_COUNT * EXPECTED_PROFILE_COUNT
        or len(scene_by_record) != EXPECTED_RECORD_COUNT
        or len(profile_ids) != EXPECTED_PROFILE_COUNT
        or counts != EXPECTED_SCENE_COUNTS
        or any(
            sum(coordinate[0] == record_id for coordinate in coordinates) != EXPECTED_PROFILE_COUNT
            for record_id in scene_by_record
        )
    ):
        raise FoldReplayError("fold_replay_panel_mismatch")
    return scene_by_record, tuple(sorted(profile_ids))


def budget_contract_from_payload(
    payload: Mapping[str, Any],
) -> BudgetContract:
    return BudgetContract(
        contract_version=str(payload["contract_version"]),
        stage_unique_limits=dict(
            require_mapping(
                "fold_replay_stage_unique_limits",
                payload.get("stage_unique_limits"),
            )
        ),
        normal_unique_identity_limit=payload.get("normal_unique_identity_limit"),
        supplemental_stage=(
            None
            if payload.get("supplemental_stage") is None
            else str(payload["supplemental_stage"])
        ),
        stage_attempt_kinds=dict(
            require_mapping(
                "fold_replay_stage_attempt_kinds",
                payload.get("stage_attempt_kinds"),
            )
        ),
        max_unique_identities=int(payload["max_unique_identities"]),
        max_attempts=int(payload["max_attempts"]),
        retry_limit=int(payload["retry_limit"]),
    )


def selection_contract_v1() -> dict[str, Any]:
    """Return the executable selector and target-audit contract."""

    contract = {
        "contract_version": "lyx_fold_replay_selection_contract_v1",
        "evidence_class": "development_replay_audit",
        "algorithm_level_holdout": False,
        "training_hard_gates": {
            "per_record_final_qualification_must_pass": True,
            "mean_independent_delta_mae_bpm_max": 1.0,
            "metric_failure_policy": "fail_closed",
        },
        "ranking_key": list(SELECTION_RANKING_KEY),
        "ranking_direction": "ascending_lexicographic",
        "runtime_complexity_policy": {
            "field": "actual_taps",
            "interpretation": (
                "deterministic compute/implementation complexity proxy; "
                "wall-clock runtime is not present in the frozen matrix"
            ),
        },
        "target_audit": {
            "evaluate_only_after_selection_receipt_is_frozen": True,
            "same_per_record_qualification_contract": True,
            "failure_categories": sorted(TARGET_FAILURE_CATEGORIES),
            "failed_slots_remain_in_denominator": True,
        },
        "read_barrier": {
            "training_fields": list(TRAINING_SOURCE_FIELDS),
            "target_preselection_fields": list(TARGET_IDENTITY_SOURCE_FIELDS),
            "target_postselection_fields": list(TARGET_RESULT_SOURCE_FIELDS),
            "target_result_granularity": "one_record_one_selected_profile",
        },
        "supplement_policy": {
            "stage": FOLD_REPLAY_STAGE,
            "max_unique_identities": MAX_SUPPLEMENTAL_IDENTITY_COUNT,
            "only_reason": "selected_target_numerical_identity_mismatch",
            "candidate_or_threshold_revision_allowed": False,
        },
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract
