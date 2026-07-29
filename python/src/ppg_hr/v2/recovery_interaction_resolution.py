"""Final-penalty backup planning and one-shot recovery rollback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .recovery_contracts import canonical_sha256
from .recovery_experiment_governance import AttemptIdentity
from .recovery_profile_upper_bound import (
    build_sample_in_upper_bound_payloads,
)
from .recovery_stage_f_reporting import _qualification
from .recovery_stage_p_contracts import (
    ROLLBACK_BACKUP_STAGE,
    StagePPlanError,
    require_hash,
    require_list,
    require_mapping,
    validate_recovery_candidate_registry,
    verify_embedded_hash,
)


def _backup_identity(
    *,
    template: Mapping[str, Any],
    backup: Mapping[str, Any],
    penalty: Mapping[str, Any],
    solver_hash: str,
    metric_contract_hash: str,
    evaluation_hash: str,
    parent_experiment_id: str,
) -> dict[str, Any]:
    config = deepcopy(dict(require_mapping("rollback_template_config", template.get("config"))))
    parameters = dict(
        require_mapping(
            "rollback_template_parameters",
            config.get("parameters"),
        )
    )
    parameters.update(
        {
            "recovery_candidate_id": backup["candidate_id"],
            "penalty_candidate_id": penalty["penalty_id"],
        }
    )
    config["parameters"] = parameters
    attempt = AttemptIdentity(
        solver_hash=solver_hash,
        config_hash=canonical_sha256(config),
        metric_contract_hash=metric_contract_hash,
        evaluation_hash=evaluation_hash,
        data_sha256=str(template["data_sha256"]),
        record_id=str(template["record_id"]),
        stage=ROLLBACK_BACKUP_STAGE,
        attempt_kind="formal",
        parent_experiment_id=parent_experiment_id,
    )
    ignored = {
        "solver_hash",
        "config_hash",
        "metric_contract_hash",
        "evaluation_hash",
        "data_sha256",
        "record_id",
        "stage",
        "attempt_kind",
        "parent_experiment_id",
        "identity_sha256",
        "cache_identity_sha256",
        "matrix_role",
        "recovery_candidate_id",
        "recovery_candidate_sha256",
        "candidate_min_bpm",
        "penalty_candidate_id",
        "penalty_candidate_sha256",
        "config",
    }
    constants = require_mapping(
        "rollback_recovery_constants",
        backup.get("constants"),
    )
    return {
        **{key: deepcopy(value) for key, value in template.items() if key not in ignored},
        **attempt.to_dict(),
        "matrix_role": "fixed_rollback_backup",
        "config": config,
        "recovery_candidate_id": backup["candidate_id"],
        "recovery_candidate_sha256": backup["candidate_sha256"],
        "candidate_min_bpm": constants.get("candidate_min_bpm"),
        "penalty_candidate_id": penalty["penalty_id"],
        "penalty_candidate_sha256": penalty["candidate_sha256"],
    }


def build_rollback_backup_proposal(
    *,
    stage_f_proposal: Mapping[str, Any],
    stage_p_proposal: Mapping[str, Any],
    stage_p_completion: Mapping[str, Any],
    penalty_interaction_report: Mapping[str, Any],
    recovery_registry: Mapping[str, Any],
    budget_contract: Mapping[str, Any],
    parent_experiment_id: str,
    solver_hash: str,
    metric_contract_hash: str,
    evaluation_hash: str,
) -> dict[str, Any]:
    """Freeze the optional 8×12 backup matrix under the final penalty."""

    for name, value in (
        ("solver_hash", solver_hash),
        ("metric_contract_hash", metric_contract_hash),
        ("evaluation_hash", evaluation_hash),
    ):
        require_hash(name, value)
    stage_f_sha = verify_embedded_hash(
        stage_f_proposal,
        hash_field="proposal_sha256",
        artifact_name="rollback_stage_f_proposal",
    )
    stage_p_sha = verify_embedded_hash(
        stage_p_proposal,
        hash_field="proposal_sha256",
        artifact_name="rollback_stage_p_proposal",
    )
    verify_embedded_hash(
        stage_p_completion,
        hash_field="completion_sha256",
        artifact_name="rollback_stage_p_completion",
    )
    verify_embedded_hash(
        penalty_interaction_report,
        hash_field="report_sha256",
        artifact_name="rollback_penalty_interaction_report",
    )
    selected_penalty_id = str(penalty_interaction_report.get("selected_penalty_id"))
    penalties = {
        str(candidate["penalty_id"]): candidate
        for candidate in (
            dict(require_mapping("rollback_penalty_candidate", raw))
            for raw in require_list(
                "rollback_penalties",
                stage_p_proposal.get("penalties"),
            )
        )
    }
    backup_id = stage_f_proposal.get("rollback_backup_id")
    provisional_id = str(stage_f_proposal.get("provisional_recovery_id"))
    candidates = validate_recovery_candidate_registry(
        recovery_registry,
        artifact_name="rollback_recovery_registry",
    )
    selected_rows = [
        row
        for row in (
            dict(require_mapping("rollback_selected_penalty_row", raw))
            for raw in require_list(
                "rollback_interaction_rows",
                penalty_interaction_report.get("rows"),
            )
        )
        if row.get("penalty_candidate_id") == selected_penalty_id
    ]
    limits = require_mapping(
        "rollback_budget_stage_unique_limits",
        budget_contract.get("stage_unique_limits"),
    )
    stage_f_frozen = require_mapping(
        "rollback_stage_f_frozen_contracts",
        stage_f_proposal.get("frozen_contracts"),
    )
    if (
        stage_p_completion.get("completion_version") != "lyx_stage_p_completion_v1"
        or stage_p_completion.get("status") != "selected"
        or stage_p_completion.get("proposal_sha256") != stage_p_sha
        or stage_p_completion.get("selected_penalty_id") != selected_penalty_id
        or penalty_interaction_report.get("proposal_sha256") != stage_p_sha
        or len(selected_rows) != 96
        or selected_penalty_id not in penalties
        or provisional_id not in candidates
        or (backup_id is not None and backup_id not in candidates)
        or backup_id == provisional_id
        or limits.get(ROLLBACK_BACKUP_STAGE) != 96
        or stage_f_sha != stage_p_proposal.get("stage_f_proposal_sha256")
        or stage_f_frozen.get("recovery_candidate_registry_hash")
        != recovery_registry.get("registry_sha256")
        or stage_f_frozen.get("budget_contract_hash") != canonical_sha256(budget_contract)
    ):
        raise StagePPlanError("rollback_upstream_contract_mismatch")
    templates = [
        dict(require_mapping("rollback_stage_f_identity", raw))
        for raw in require_list(
            "rollback_stage_f_identities",
            stage_f_proposal.get("identities"),
        )
        if require_mapping(
            "rollback_stage_f_identity",
            raw,
        ).get("matrix_role")
        == "provisional_recovery"
    ]
    if len(templates) != 96:
        raise StagePPlanError("rollback_stage_f_templates_incomplete")
    if any(
        template.get("solver_hash") != solver_hash
        or template.get("metric_contract_hash") != metric_contract_hash
        for template in templates
    ):
        raise StagePPlanError("rollback_runtime_identity_mismatch")
    identities: list[dict[str, Any]] = []
    if backup_id is not None:
        identities = [
            _backup_identity(
                template=template,
                backup=candidates[str(backup_id)],
                penalty=penalties[selected_penalty_id],
                solver_hash=solver_hash,
                metric_contract_hash=metric_contract_hash,
                evaluation_hash=evaluation_hash,
                parent_experiment_id=parent_experiment_id,
            )
            for template in templates
        ]
    if len(identities) not in {0, 96} or len(
        {str(identity["identity_sha256"]) for identity in identities}
    ) != len(identities):
        raise StagePPlanError("rollback_identity_matrix_mismatch")
    proposal = {
        "proposal_version": "lyx_rollback_backup_proposal_v1",
        "status": ("ready_for_execution" if identities else "no_backup_execution_required"),
        "parent_experiment_id": parent_experiment_id,
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "independent_bo_authorized": False,
        "stage_f_proposal_sha256": stage_f_sha,
        "stage_p_proposal_sha256": stage_p_sha,
        "stage_p_completion_sha256": stage_p_completion["completion_sha256"],
        "penalty_interaction_report_sha256": (penalty_interaction_report["report_sha256"]),
        "provisional_recovery_id": provisional_id,
        "rollback_backup_id": backup_id,
        "selected_penalty_id": selected_penalty_id,
        "profile_count": 8,
        "record_count": 12,
        "record_panel": deepcopy(stage_f_proposal["record_panel"]),
        "planned_unique_identity_count": len(identities),
        "rollback_limit": 1,
        "penalty_reselection_allowed": False,
        "candidate_reselection_allowed": False,
        "frozen_contracts": {
            "metric_contract_hash": metric_contract_hash,
            "recovery_candidate_registry_hash": recovery_registry["registry_sha256"],
            "penalty_registry_hash": stage_p_proposal["frozen_contracts"]["penalty_registry_hash"],
            "budget_contract_hash": canonical_sha256(budget_contract),
            "rollback_evaluation_hash": evaluation_hash,
        },
        "identities": identities,
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    return proposal


def _awaiting_interaction_receipt(
    *,
    proposal: Mapping[str, Any],
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed with an auditable receipt when rollback evidence is unsafe."""

    receipt = {
        "receipt_version": "lyx_recovery_rollback_receipt_v1",
        "status": "awaiting_human_interaction_decision",
        "proposal_sha256": proposal["proposal_sha256"],
        "stage_p_proposal_sha256": proposal["stage_p_proposal_sha256"],
        "penalty_interaction_report_sha256": proposal["penalty_interaction_report_sha256"],
        "provisional_recovery_id": proposal["provisional_recovery_id"],
        "rollback_backup_id": proposal.get("rollback_backup_id"),
        "selected_penalty_id": proposal["selected_penalty_id"],
        "rollback_triggered": False,
        "rollback_count": 0,
        "final_recovery_id": None,
        "candidate_reselection_count": 0,
        "penalty_reselection_count": 0,
        "trigger_evidence": {
            "rule_1_coordinates": [],
            "rule_2_coordinates": [],
            "rule_3_coordinates": [],
            "evidence_failure": {
                "reason": reason,
                "details": dict(details or {}),
            },
        },
        "independent_bo_run_count": 0,
        "next_state": "awaiting_human_interaction_decision",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _index_interaction_rows(
    *,
    name: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], str | None]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        try:
            row = dict(require_mapping(f"{name}_row", raw))
        except StagePPlanError:
            return {}, f"{name}_row_not_object"
        profile_id = row.get("filter_profile_id")
        record_id = row.get("record_id")
        if not isinstance(profile_id, str) or not profile_id:
            return {}, f"{name}_profile_id_missing"
        if not isinstance(record_id, str) or not record_id:
            return {}, f"{name}_record_id_missing"
        coordinate = (profile_id, record_id)
        if coordinate in indexed:
            return {}, f"{name}_duplicate_coordinate"
        indexed[coordinate] = row
    return indexed, None


def resolve_recovery_interaction(
    *,
    proposal: Mapping[str, Any],
    provisional_rows: Sequence[Mapping[str, Any]],
    backup_rows: Sequence[Mapping[str, Any]],
    current_role_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the pre-registered rollback trigger exactly once."""

    verify_embedded_hash(
        proposal,
        hash_field="proposal_sha256",
        artifact_name="rollback_resolution_proposal",
    )
    provisional_id = str(proposal["provisional_recovery_id"])
    backup_id = proposal.get("rollback_backup_id")
    selected_penalty_id = str(proposal["selected_penalty_id"])
    current_by_coordinate, current_error = _index_interaction_rows(
        name="rollback_current",
        rows=current_role_rows,
    )
    provisional_by_coordinate, provisional_error = _index_interaction_rows(
        name="rollback_provisional",
        rows=provisional_rows,
    )
    backup_by_coordinate, backup_error = _index_interaction_rows(
        name="rollback_backup",
        rows=backup_rows,
    )
    indexing_errors = [
        error for error in (current_error, provisional_error, backup_error) if error is not None
    ]
    if indexing_errors:
        return _awaiting_interaction_receipt(
            proposal=proposal,
            reason="rollback_evidence_incomplete_or_malformed",
            details={"errors": indexing_errors},
        )
    if (
        len(provisional_by_coordinate) != 96
        or len(current_by_coordinate) != 96
        or set(current_by_coordinate) != set(provisional_by_coordinate)
        or any(
            row.get("recovery_candidate_id") != provisional_id
            or row.get("penalty_candidate_id") != selected_penalty_id
            for row in provisional_by_coordinate.values()
        )
    ):
        return _awaiting_interaction_receipt(
            proposal=proposal,
            reason="rollback_provisional_or_current_pairing_mismatch",
            details={
                "provisional_coordinate_count": len(provisional_by_coordinate),
                "current_coordinate_count": len(current_by_coordinate),
            },
        )
    if backup_id is None:
        return _awaiting_interaction_receipt(
            proposal=proposal,
            reason="rollback_backup_unavailable",
            details={"backup_row_count": len(backup_by_coordinate)},
        )
    if (
        len(backup_by_coordinate) != 96
        or set(backup_by_coordinate) != set(provisional_by_coordinate)
        or any(
            row.get("recovery_candidate_id") != backup_id
            or row.get("penalty_candidate_id") != selected_penalty_id
            for row in backup_by_coordinate.values()
        )
    ):
        return _awaiting_interaction_receipt(
            proposal=proposal,
            reason="rollback_backup_pairing_identity_mismatch",
            details={
                "backup_coordinate_count": len(backup_by_coordinate),
                "expected_coordinate_count": len(provisional_by_coordinate),
            },
        )
    evaluated_backup: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}
    for coordinate, row in backup_by_coordinate.items():
        try:
            evaluated_backup[coordinate] = {
                **row,
                "qualification": _qualification(
                    candidate=row,
                    current=current_by_coordinate[coordinate],
                    independent=next(
                        require_mapping(
                            "rollback_independent_metrics",
                            record["independent_metrics"],
                        )
                        for record in proposal.get("record_panel", [])
                        if record["record_id"] == coordinate[1]
                    )
                    if proposal.get("record_panel")
                    else require_mapping(
                        "rollback_provisional_qualification_context",
                        provisional_by_coordinate[coordinate].get("independent_metrics"),
                    ),
                    true_rise_applicable=bool(
                        provisional_by_coordinate[coordinate].get(
                            "true_rise_applicable",
                            provisional_by_coordinate[coordinate].get("scene") in {"run", "kaihe"},
                        )
                    ),
                ),
            }
        except (KeyError, StopIteration, TypeError, ValueError, StagePPlanError) as error:
            return _awaiting_interaction_receipt(
                proposal=proposal,
                reason="rollback_backup_evaluation_evidence_invalid",
                details={
                    "coordinate": f"{coordinate[0]}:{coordinate[1]}",
                    "error": str(error),
                },
            )
    hard_gate_candidates: set[tuple[str, str]] = set()
    dynamic_candidates: set[tuple[str, str]] = set()
    catastrophe_candidates: set[tuple[str, str]] = set()
    try:
        for coordinate, provisional in provisional_by_coordinate.items():
            reasons = set(
                require_mapping(
                    "rollback_provisional_qualification",
                    provisional.get("qualification"),
                ).get("elimination_reasons", [])
            )
            if reasons & {"independent_l10_gate", "independent_l20_gate"}:
                hard_gate_candidates.add(coordinate)
            if provisional.get("scene") in {"run", "kaihe"} and reasons & {
                "new_right_censored_recovery",
                "true_rise_underestimate_gate",
            }:
                dynamic_candidates.add(coordinate)
            provisional_l10 = int(
                require_mapping(
                    "rollback_provisional_metrics",
                    provisional.get("metrics"),
                )["longest_e10_run_windows"]
            )
            current_l10 = int(
                require_mapping(
                    "rollback_current_metrics",
                    current_by_coordinate[coordinate].get("metrics"),
                )["longest_e10_run_windows"]
            )
            if current_l10 <= 10 and provisional_l10 >= 20:
                catastrophe_candidates.add(coordinate)
    except (KeyError, TypeError, ValueError, StagePPlanError) as error:
        return _awaiting_interaction_receipt(
            proposal=proposal,
            reason="rollback_trigger_evidence_invalid",
            details={"error": str(error)},
        )
    qualifying_hard = {
        coordinate
        for coordinate in hard_gate_candidates
        if evaluated_backup[coordinate]["qualification"]["qualified"]
    }
    hard_records = {coordinate[1] for coordinate in qualifying_hard}
    hard_scenes = {
        str(provisional_by_coordinate[coordinate]["scene"]) for coordinate in qualifying_hard
    }
    rule_1 = len(hard_records) >= 2 and len(hard_scenes) >= 2
    qualifying_dynamic = {
        coordinate
        for coordinate in dynamic_candidates
        if evaluated_backup[coordinate]["qualification"]["qualified"]
    }
    rule_2 = bool(qualifying_dynamic)
    qualifying_catastrophe = {
        coordinate
        for coordinate in catastrophe_candidates
        if int(
            require_mapping(
                "rollback_backup_metrics",
                evaluated_backup[coordinate].get("metrics"),
            )["longest_e10_run_windows"]
        )
        < 20
    }
    rule_3 = bool(qualifying_catastrophe)
    trigger = rule_1 or rule_2 or rule_3
    status = "rolled_back" if trigger else "retained"
    final_recovery_id = backup_id if trigger else provisional_id
    receipt = {
        "receipt_version": "lyx_recovery_rollback_receipt_v1",
        "status": status,
        "proposal_sha256": proposal["proposal_sha256"],
        "stage_p_proposal_sha256": proposal["stage_p_proposal_sha256"],
        "penalty_interaction_report_sha256": proposal["penalty_interaction_report_sha256"],
        "provisional_recovery_id": provisional_id,
        "rollback_backup_id": backup_id,
        "selected_penalty_id": selected_penalty_id,
        "rollback_triggered": trigger,
        "rollback_count": 1 if status == "rolled_back" else 0,
        "final_recovery_id": final_recovery_id,
        "candidate_reselection_count": 0,
        "penalty_reselection_count": 0,
        "trigger_evidence": {
            "rule_1_coordinates": sorted(
                f"{profile_id}:{record_id}" for profile_id, record_id in qualifying_hard
            ),
            "rule_2_coordinates": sorted(
                f"{profile_id}:{record_id}" for profile_id, record_id in qualifying_dynamic
            ),
            "rule_3_coordinates": sorted(
                f"{profile_id}:{record_id}" for profile_id, record_id in qualifying_catastrophe
            ),
        },
        "independent_bo_run_count": 0,
        "next_state": (
            "awaiting_human_interaction_decision"
            if status == "awaiting_human_interaction_decision"
            else "ready_for_historical_recovery_ab_proposal"
        ),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def build_final_interaction_audit(
    *,
    proposal: Mapping[str, Any],
    rollback_receipt: Mapping[str, Any],
    penalty_interaction_report: Mapping[str, Any],
    backup_rows: Sequence[Mapping[str, Any]],
    current_role_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Regenerate final profile receipts and the sample-in upper bound."""

    verify_embedded_hash(
        rollback_receipt,
        hash_field="receipt_sha256",
        artifact_name="final_interaction_rollback_receipt",
    )
    verify_embedded_hash(
        penalty_interaction_report,
        hash_field="report_sha256",
        artifact_name="final_interaction_penalty_report",
    )
    if (
        rollback_receipt.get("status") not in {"retained", "rolled_back"}
        or rollback_receipt.get("proposal_sha256") != proposal.get("proposal_sha256")
        or rollback_receipt.get("selected_penalty_id") != proposal.get("selected_penalty_id")
        or rollback_receipt.get("stage_p_proposal_sha256")
        != proposal.get("stage_p_proposal_sha256")
        or rollback_receipt.get("penalty_interaction_report_sha256")
        != proposal.get("penalty_interaction_report_sha256")
        or penalty_interaction_report.get("report_sha256")
        != proposal.get("penalty_interaction_report_sha256")
        or rollback_receipt.get("rollback_count") not in {0, 1}
        or rollback_receipt.get("candidate_reselection_count") != 0
        or rollback_receipt.get("penalty_reselection_count") != 0
    ):
        raise StagePPlanError("final_interaction_upstream_mismatch")
    final_recovery_id = str(rollback_receipt["final_recovery_id"])
    selected_penalty_id = str(rollback_receipt["selected_penalty_id"])
    if rollback_receipt["status"] == "rolled_back":
        raw_final_rows = [dict(row) for row in backup_rows]
    else:
        raw_final_rows = [
            dict(require_mapping("final_interaction_penalty_row", raw))
            for raw in penalty_interaction_report["rows"]
            if require_mapping(
                "final_interaction_penalty_row",
                raw,
            ).get("penalty_candidate_id")
            == selected_penalty_id
        ]
    current_by_coordinate = {
        (
            str(row["filter_profile_id"]),
            str(row["record_id"]),
        ): row
        for row in current_role_rows
    }
    record_panel = {str(record["record_id"]): record for record in proposal["record_panel"]}
    coordinates = {
        (
            str(row["filter_profile_id"]),
            str(row["record_id"]),
        )
        for row in raw_final_rows
    }
    if (
        len(raw_final_rows) != 96
        or len(coordinates) != 96
        or set(current_by_coordinate) != coordinates
        or {coordinate[1] for coordinate in coordinates} != set(record_panel)
        or any(
            row.get("recovery_candidate_id") != final_recovery_id
            or row.get("penalty_candidate_id") != selected_penalty_id
            for row in raw_final_rows
        )
    ):
        raise StagePPlanError("final_interaction_matrix_mismatch")
    final_rows: list[dict[str, Any]] = []
    for row in raw_final_rows:
        coordinate = (
            str(row["filter_profile_id"]),
            str(row["record_id"]),
        )
        record = record_panel[coordinate[1]]
        final_rows.append(
            {
                **row,
                "qualification": _qualification(
                    candidate=row,
                    current=current_by_coordinate[coordinate],
                    independent=require_mapping(
                        "final_interaction_independent_metrics",
                        record["independent_metrics"],
                    ),
                    true_rise_applicable=bool(record["true_rise_applicable"]),
                ),
            }
        )
    profile_receipts: dict[str, dict[str, Any]] = {}
    for profile_id in sorted({str(row["filter_profile_id"]) for row in final_rows}):
        profile_rows = [row for row in final_rows if row["filter_profile_id"] == profile_id]
        receipt = {
            "receipt_version": "lyx_final_filter_profile_receipt_v1",
            "filter_profile_id": profile_id,
            "final_recovery_id": final_recovery_id,
            "selected_penalty_id": selected_penalty_id,
            "record_count": len(profile_rows),
            "qualified_record_count": sum(
                bool(row["qualification"]["qualified"]) for row in profile_rows
            ),
            "hard_gate_failure_count": sum(
                not bool(row["qualification"]["qualified"]) for row in profile_rows
            ),
            "identity_sha256": sorted(str(row["identity_sha256"]) for row in profile_rows),
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        profile_receipts[profile_id] = receipt
    if len(profile_receipts) != 8 or any(
        receipt["record_count"] != 12 for receipt in profile_receipts.values()
    ):
        raise StagePPlanError("final_interaction_profile_receipt_mismatch")
    upper_bounds = build_sample_in_upper_bound_payloads(
        final_profile_rows=final_rows,
        scene_by_record={
            record_id: str(record["scene"]) for record_id, record in record_panel.items()
        },
    )
    audit = {
        "audit_version": "lyx_final_interaction_audit_v1",
        "status": "complete",
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "proposal_sha256": proposal["proposal_sha256"],
        "rollback_receipt_sha256": rollback_receipt["receipt_sha256"],
        "penalty_interaction_report_sha256": (penalty_interaction_report["report_sha256"]),
        "final_recovery_id": final_recovery_id,
        "selected_penalty_id": selected_penalty_id,
        "rollback_count": rollback_receipt["rollback_count"],
        "row_count": 96,
        "rows": final_rows,
        "independent_metrics_by_record": {
            record_id: deepcopy(record["independent_metrics"])
            for record_id, record in sorted(record_panel.items())
        },
        "profile_receipts": profile_receipts,
        **upper_bounds,
        "independent_bo_run_count": 0,
        "next_state": "ready_for_historical_recovery_ab_proposal",
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return audit
