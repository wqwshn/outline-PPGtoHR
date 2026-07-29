"""Stage R completion validation and independent-BO review reporting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bo_space_generalization import (
    SeedSearchBudget,
    build_bo_search_space,
)
from .phase2_experiment_io import file_sha256, read_json
from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import (
    AttemptIdentity,
    AttemptRegistry,
)
from .recovery_stage_r_common import StageRPlanError


def _require_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageRPlanError(f"{name}_must_be_object")
    return value


def _require_list(name: str, value: object) -> list[Any]:
    if not isinstance(value, list):
        raise StageRPlanError(f"{name}_must_be_array")
    return value


def _verify_embedded_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    artifact_name: str,
) -> None:
    declared = payload.get(hash_field)
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != hash_field
    }
    if declared != canonical_sha256(unsigned):
        raise StageRPlanError(
            f"{artifact_name}_embedded_hash_mismatch"
        )


def build_independent_bo_review_package(
    *,
    proposal_sha256: str,
    authorization_sha256: str,
    selection: Mapping[str, Any],
    candidate_evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the spec §15 packet without authorizing or running BO."""

    if selection.get("status") != "no_safe_recovery_candidate":
        raise StageRPlanError(
            "independent_bo_review_requires_no_safe_selection"
        )
    eliminated = dict(
        _require_mapping(
            "eliminated_candidates",
            selection.get("eliminated_candidates"),
        )
    )
    evaluation_rows: list[dict[str, Any]] = []
    trigger_map: dict[str, dict[str, Any]] = {}
    for candidate in candidate_evaluations:
        candidate_id = str(candidate["candidate_id"])
        reasons = [
            str(reason)
            for reason in _require_list(
                f"eliminated_reasons:{candidate_id}",
                eliminated.get(candidate_id),
            )
        ]
        for raw_record in _require_list(
            f"candidate_records:{candidate_id}",
            candidate.get("records"),
        ):
            record = _require_mapping(
                f"candidate_record:{candidate_id}",
                raw_record,
            )
            record_id = str(record["record_id"])
            sentinel_id = str(record["sentinel_id"])
            coordinate_reasons = [
                reason
                for reason in reasons
                if (
                    reason.startswith(
                        f"{sentinel_id}/{record_id}:"
                    )
                    or (
                        record_id in reason
                        and f"sentinel:{sentinel_id}:" in reason
                    )
                )
            ]
            if coordinate_reasons:
                item = trigger_map.setdefault(
                    record_id,
                    {
                        "record_id": record_id,
                        "scene": str(record["scene"]),
                        "candidate_reasons": [],
                    },
                )
                item["candidate_reasons"].append(
                    {
                        "candidate_id": candidate_id,
                        "sentinel_id": sentinel_id,
                        "reasons": coordinate_reasons,
                    }
                )
            evaluation_rows.append(
                {
                    "candidate_id": candidate_id,
                    "sentinel_id": sentinel_id,
                    "record_id": record_id,
                    "scene": str(record["scene"]),
                    "trigger_reasons": coordinate_reasons,
                    "candidate_minus_historical_independent_mae_bpm": (
                        float(record["mae"])
                        - float(record["independent_mae"])
                    ),
                    "candidate_minus_same_sentinel_current_mechanism_mae_bpm": (
                        float(record["mae"])
                        - float(record["current_mae"])
                    ),
                    "historical_parameter_new_mechanism_replay_delta_bpm": None,
                    "combination_library_in_sample_upper_delta_bpm": None,
                }
            )
    if not trigger_map:
        raise StageRPlanError(
            "independent_bo_review_has_no_trigger_records"
        )
    reason_counts: Counter[str] = Counter()
    for reasons in eliminated.values():
        for reason in _require_list(
            "independent_bo_elimination_reasons",
            reasons,
        ):
            reason_counts[str(reason).rsplit(":", 1)[-1]] += 1
    mechanism_categories = {
        "spectral_harm_or_untraceable_evidence": sum(
            count
            for reason, count in reason_counts.items()
            if "spectral" in reason
        ),
        "long_tail_or_recovery_failure": sum(
            count
            for reason, count in reason_counts.items()
            if any(
                token in reason
                for token in (
                    "l10",
                    "l20",
                    "right_censored",
                    "recovery",
                )
            )
        ),
        "accuracy_gap": sum(
            count
            for reason, count in reason_counts.items()
            if "mae" in reason
        ),
        "physiological_rise_suppression": sum(
            count
            for reason, count in reason_counts.items()
            if "true_rise" in reason
        ),
    }
    search_space = build_bo_search_space("physical_v1")
    lane_budget = SeedSearchBudget()
    candidate_count = len(candidate_evaluations)
    record_count = len(
        {str(row["record_id"]) for row in evaluation_rows}
    )
    unique_budget = (
        candidate_count
        * record_count
        * lane_budget.global_unique_budget
    )
    observed_cache_bytes_per_identity = 1_013_117
    estimated_cache_bytes = (
        unique_budget * observed_cache_bytes_per_identity
    )
    search_space_payload = {
        "space_name": search_space.name,
        "parameter_names": list(search_space.parameter_names),
        "option_values": [
            list(values) for values in search_space.option_values
        ],
        "discrete_candidate_count": len(search_space.candidates),
    }
    proposed_budget = {
        "recovery_candidate_count": candidate_count,
        "record_count": record_count,
        "lane_seeds": list(lane_budget.lane_seeds),
        "lane_unique_budget": lane_budget.lane_unique_budget,
        "global_unique_budget_per_candidate_record": (
            lane_budget.global_unique_budget
        ),
        "maximum_unique_solver_config_record_identities": unique_budget,
        "retry_budget": "must_be_frozen_in_separate_proposal",
    }
    payload = {
        "package_version": "lyx_stage_r_independent_bo_review_v1",
        "status": "awaiting_human_independent_bo_decision",
        "proposal_sha256": proposal_sha256,
        "authorization_sha256": authorization_sha256,
        "trigger": "no_safe_recovery_candidate",
        "trigger_selection_sha256": selection[
            "selection_sha256"
        ],
        "trigger_records": [
            trigger_map[record_id]
            for record_id in sorted(trigger_map)
        ],
        "baseline_differences": {
            "available_rows": evaluation_rows,
            "historical_parameter_new_mechanism_replay": {
                "status": "not_available",
                "reason": (
                    "Stage R did not register a historical-per-record-"
                    "parameter replay matrix for each new mechanism"
                ),
            },
            "combination_library_in_sample_upper": {
                "status": "not_available",
                "reason": (
                    "no_safe at Stage R stops before Stage F by spec"
                ),
            },
        },
        "eliminated_candidates": eliminated,
        "mechanism_reason_summary": {
            "category_counts": mechanism_categories,
            "exact_reason_counts": dict(sorted(reason_counts.items())),
            "interpretation": (
                "The exact reason distribution distinguishes spectral/"
                "mechanism failure from a parameter-retuning gap; it does "
                "not treat a lower BO objective as a safety override."
            ),
        },
        "proposed_search_space": {
            **search_space_payload,
            "search_space_sha256": canonical_sha256(
                search_space_payload
            ),
        },
        "proposed_unique_budget": proposed_budget,
        "resource_estimate": {
            "basis": (
                "formal_phase2_20260725T110252:s21 archive; observed "
                "solver runtimes approximately 1.1-4.3 s and mean cache "
                "volume 1,013,117 bytes per completed identity"
            ),
            "estimated_serial_runtime_hours_low": round(
                unique_budget * 1.1 / 3600.0,
                2,
            ),
            "estimated_serial_runtime_hours_high": round(
                unique_budget * 4.3 / 3600.0,
                2,
            ),
            "estimated_cache_bytes": estimated_cache_bytes,
            "estimated_cache_gib": round(
                estimated_cache_bytes / (1024**3),
                2,
            ),
            "expected_reusable_cache_identity_count": 0,
            "cache_reuse_caveat": (
                "reuse remains zero until a separate proposal proves exact "
                "solver/config/metric/data identity equality"
            ),
        },
        "decision_consequences": {
            "if_run": (
                "Tests whether per-record retuning can rescue any recovery "
                "mechanism and separates search-space mismatch from a "
                "mechanism-level failure."
            ),
            "if_not_run": (
                "Retains the fail-closed conclusion that no mechanism is "
                "safe under the frozen sentinel matrix, but cannot tell "
                "whether independent retuning would rescue it."
            ),
            "cannot_answer_even_if_run": (
                "Does not establish unseen-scene or cross-person "
                "generalization because all 12 records remain development "
                "data."
            ),
        },
        "recommendation": {
            "decision": "prepare_exact_independent_bo_proposal",
            "reason": (
                "no_safe_recovery_candidate blocks Stage F and the full "
                "independent BO is the registered way to test whether the "
                "failure is caused by parameter interaction rather than the "
                "recovery mechanism itself"
            ),
            "automatic_execution": False,
        },
        "requested_human_decision": (
            "whether_to_prepare_a_separate_exact_independent_bo_proposal"
        ),
        "independent_bo_authorized": False,
        "independent_bo_run_count": 0,
        "execution_identity_count": unique_budget,
        "execution_budget": proposed_budget,
        "execution_policy": (
            "no BO may run until identities, search budget, data scope, "
            "source closure, and a new authorization hash are frozen"
        ),
    }
    payload["package_sha256"] = canonical_sha256(payload)
    return payload


def validate_completed_stage_r_execution(
    *,
    completion_path: Path,
    proposal: Mapping[str, Any],
    authorization_sha256: str,
    governance_root: Path,
    destination: Path,
    registry: AttemptRegistry,
    identities: Sequence[AttemptIdentity],
) -> dict[str, Any]:
    """Verify the completion commit marker and immutable Stage R snapshot."""

    completion = read_json(completion_path)
    _verify_embedded_hash(
        completion,
        hash_field="completion_sha256",
        artifact_name="stage_r_completion",
    )
    if (
        completion.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or completion.get("authorization_sha256")
        != authorization_sha256
    ):
        raise StageRPlanError(
            "stage_r_completion_authorization_mismatch"
        )
    artifacts = _require_mapping(
        "stage_r_completion_artifacts",
        completion.get("artifacts"),
    )
    for name, expected_hash in artifacts.items():
        path = (destination / str(name)).resolve()
        if (
            not path.is_relative_to(destination)
            or not path.is_file()
            or file_sha256(path) != expected_hash
        ):
            raise StageRPlanError(
                f"stage_r_completion_artifact_mismatch:{name}"
            )
    governance_path = (
        governance_root / "stage_r_governance_receipt.json"
    )
    if (
        not governance_path.is_file()
        or file_sha256(governance_path)
        != completion.get("governance_receipt_file_sha256")
    ):
        raise StageRPlanError(
            "stage_r_completion_governance_receipt_mismatch"
        )
    governance = read_json(governance_path)
    _verify_embedded_hash(
        governance,
        hash_field="receipt_sha256",
        artifact_name="stage_r_governance_receipt",
    )
    if (
        governance.get("proposal_sha256")
        != proposal.get("proposal_sha256")
        or governance.get("authorization_sha256")
        != authorization_sha256
        or governance.get("receipt_sha256")
        != completion.get("governance_receipt_sha256")
        or governance.get("status") != completion.get("status")
        or governance.get("artifacts") != dict(artifacts)
    ):
        raise StageRPlanError(
            "stage_r_governance_receipt_binding_mismatch"
        )
    snapshot_path = (
        destination / "attempt_registry_stage_r_snapshot.json"
    )
    if "attempt_registry_stage_r_snapshot.json" not in artifacts:
        raise StageRPlanError(
            "stage_r_completion_matrix_snapshot_missing"
        )
    snapshot = read_json(snapshot_path)
    registry.assert_matrix_matches_snapshot(identities, snapshot)
    current_matrix_summary = registry.matrix_execution_summary(
        identities
    )
    if (
        snapshot.get("snapshot_sha256")
        != governance.get("attempt_registry_matrix_snapshot_sha256")
        or current_matrix_summary
        != governance.get("matrix_execution_summary")
        or current_matrix_summary
        != completion.get("matrix_execution_summary")
        or governance.get("identity_matrix_sha256")
        != canonical_sha256(
            [identity.sha256 for identity in identities]
        )
    ):
        raise StageRPlanError(
            "stage_r_completion_attempt_registry_mismatch"
        )
    return completion
