from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ppg_hr.v2.phase2_experiment_io import atomic_write_json, read_json
from ppg_hr.v2.recovery_contracts import canonical_sha256
from ppg_hr.v2.recovery_experiment_governance import (
    AttemptRegistry,
    BudgetContract,
    ExplorationRegistry,
    GovernanceError,
    HumanGateRequiredError,
)
from ppg_hr.v2.recovery_fold_replay_contracts import FoldReplayError
from ppg_hr.v2.recovery_post_fold_experiment import (
    build_challenge_scene_handoff,
    build_final_development_report,
    default_challenge_scene_manifest,
    evaluate_post_fold_independent_bo_gate,
    publish_post_fold_package,
    render_final_development_report_markdown,
    validate_post_fold_independent_bo_authorization,
)
from ppg_hr.v2.recovery_profile_upper_bound import (
    build_sample_in_upper_bound_payloads,
)

_SCENES = ("jianpan", "kaihe", "run", "xiezi")
_PROFILE_IDS = tuple(f"p{index}" for index in range(8))


def _with_hash(
    payload: dict[str, object],
    field: str,
) -> dict[str, object]:
    payload[field] = canonical_sha256(payload)
    return payload


def _metrics(
    *,
    mae: float,
    l10: int = 2,
    l20: int = 0,
    right_censored: int = 0,
    rise: float | None = None,
) -> dict[str, object]:
    return {
        "final_motion_mae_bpm": mae,
        "longest_e10_run_windows": l10,
        "longest_e20_run_windows": l20,
        "right_censored_recovery_count": right_censored,
        "max_recovered_delay_s": 1.0,
        "max_rise_underestimate_bpm": rise,
    }


def _review_context() -> dict[str, object]:
    return {
        "solver_hash": "a" * 64,
        "search_space_hash": "b" * 64,
        "metric_contract_hash": "c" * 64,
        "seed_manifest_hash": "d" * 64,
        "unique_budget": 5400,
        "estimated_runtime": "1.65-6.45 serial hours",
        "estimated_cache_size": "5.1 GiB",
        "plausible_mechanism_causes": [
            "scene_shared_selector_loss",
            "recovery_filter_parameter_interaction",
        ],
        "recommendation": "先人工审核，再决定是否运行。",
        "run_answers": "检验完整搜索是否能消除冻结档位覆盖不足。",
        "no_run_answers": "保留本轮机制停止结论，不产生新工程锚点。",
    }


def _budget_authorizations() -> list[dict[str, object]]:
    common = {
        "approved": True,
        "decision_state": "awaiting_human_budget_decision",
        "independent_bo_authorized": False,
        "approved_by": "test-user",
        "profile_design_rule_hash": "e" * 64,
        "record_manifest_hash": "f" * 64,
    }
    return [
        {
            **common,
            "approved_at": "2026-07-28T22:00:00+08:00",
            "stage": "filter_profile_stability_audit",
            "added_unique_identities": 32,
            "normal_unique_identity_limit": 704,
            "max_unique_identities": 716,
            "max_attempts": 1432,
        },
        {
            **common,
            "approved_at": "2026-07-28T23:00:00+08:00",
            "stage": "filter_profile_stability_audit",
            "added_unique_identities": 8,
            "normal_unique_identity_limit": 712,
            "max_unique_identities": 724,
            "max_attempts": 1448,
            "proposal_sha256": "1" * 64,
        },
        {
            **common,
            "approved_at": "2026-07-29T09:00:00+08:00",
            "stage": "filter_profile_stability_audit",
            "added_unique_identities": 24,
            "normal_unique_identity_limit": 736,
            "max_unique_identities": 748,
            "max_attempts": 1496,
            "proposal_sha256": "2" * 64,
        },
        {
            **common,
            "approved_at": "2026-07-29T11:00:00+08:00",
            "stage": "filter_profile_rate_normalization_exploration",
            "added_unique_identities": 8,
            "normal_unique_identity_limit": 744,
            "max_unique_identities": 756,
            "max_attempts": 1512,
            "attempt_kind": "exploration",
            "exploration_unique_budget": 8,
            "proposal_sha256": "3" * 64,
        },
    ]


def _bundle(
    *,
    gap_record_ids: set[str] | None = None,
    failed_record_ids: set[str] | None = None,
    unsafe_sample_record_ids: set[str] | None = None,
) -> dict[str, object]:
    gaps = set(gap_record_ids or ())
    failures = set(failed_record_ids or ())
    unsafe = set(unsafe_sample_record_ids or ())
    records = [
        (scene, f"{scene}{record_index}") for scene in _SCENES for record_index in range(1, 4)
    ]
    final_rows: list[dict[str, object]] = []
    current_role_rows: list[dict[str, object]] = []
    independent_by_record: dict[str, dict[str, object]] = {}
    historical_records: list[dict[str, object]] = []
    for scene, record_id in records:
        rise = 1.0 if scene in {"kaihe", "run"} else None
        independent_by_record[record_id] = _metrics(
            mae=1.0,
            rise=rise,
        )
        historical_records.append(
            {
                "record_id": record_id,
                "scene": scene,
                "current_metrics": _metrics(
                    mae=1.4,
                    rise=rise,
                ),
                "final_metrics": _metrics(
                    mae=1.2,
                    rise=rise,
                ),
                "spectral_gate_pass": True,
            }
        )
        for profile_index, profile_id in enumerate(_PROFILE_IDS):
            qualified = record_id not in unsafe
            row_metrics = _metrics(
                mae=1.0 + profile_index / 10.0,
                rise=rise,
            )
            final_rows.append(
                {
                    "identity_sha256": canonical_sha256(
                        {
                            "role": "final",
                            "record_id": record_id,
                            "profile_id": profile_id,
                        }
                    ),
                    "record_id": record_id,
                    "scene": scene,
                    "filter_profile_id": profile_id,
                    "solver_hash": "4" * 64,
                    "config_hash": canonical_sha256(
                        {
                            "record_id": record_id,
                            "profile_id": profile_id,
                        }
                    ),
                    "metric_contract_hash": "5" * 64,
                    "evaluation_hash": "6" * 64,
                    "data_sha256": canonical_sha256({"record_id": record_id}),
                    "parent_experiment_id": "parent-v1",
                    "actual_taps": 10 + profile_index,
                    "metrics": row_metrics,
                    "qualification": {
                        "qualified": qualified,
                        "elimination_reasons": ([] if qualified else ["spectral_gate_contract_v1"]),
                    },
                }
            )
            current_role_rows.append(
                {
                    "identity_sha256": canonical_sha256(
                        {
                            "role": "current",
                            "record_id": record_id,
                            "profile_id": profile_id,
                        }
                    ),
                    "record_id": record_id,
                    "scene": scene,
                    "filter_profile_id": profile_id,
                    "metrics": _metrics(
                        mae=1.5 + profile_index / 10.0,
                        rise=rise,
                    ),
                }
            )
    upper_bounds = build_sample_in_upper_bound_payloads(
        final_profile_rows=final_rows,
        scene_by_record={record_id: scene for scene, record_id in records},
    )
    profile_receipts: dict[str, dict[str, object]] = {}
    for profile_id in _PROFILE_IDS:
        profile_rows = [row for row in final_rows if row["filter_profile_id"] == profile_id]
        profile_receipts[profile_id] = _with_hash(
            {
                "receipt_version": "lyx_final_filter_profile_receipt_v1",
                "filter_profile_id": profile_id,
                "record_count": 12,
                "identity_sha256": sorted(str(row["identity_sha256"]) for row in profile_rows),
            },
            "receipt_sha256",
        )
    final_audit = _with_hash(
        {
            "audit_version": "lyx_final_interaction_audit_v1",
            "status": "complete",
            "evidence_class": "development_reuse_pilot",
            "algorithm_level_holdout": False,
            "final_recovery_id": "recovery-final-v1",
            "selected_penalty_id": "penalty-final-v1",
            "row_count": 96,
            "rows": final_rows,
            "independent_metrics_by_record": independent_by_record,
            "profile_receipts": profile_receipts,
            **upper_bounds,
            "independent_bo_run_count": 0,
        },
        "audit_sha256",
    )
    historical = _with_hash(
        {
            "report_version": "lyx_historical_recovery_ab_report_v1",
            "status": "complete",
            "logical_result_count": 24,
            "formal_result_count": 24,
            "records": historical_records,
            "independent_bo_run_count": 0,
        },
        "report_sha256",
    )
    current_role = _with_hash(
        {
            "matrix_version": "lyx_stage_f_current_role_matrix_v1",
            "matrix_role": "same_role_current_control",
            "algorithm_level_holdout": False,
            "row_count": 96,
            "unique_spectral_audit_count": 96,
            "rows": current_role_rows,
        },
        "matrix_sha256",
    )
    pre_fold_gate = _with_hash(
        {
            "receipt_version": "lyx_pre_fold_independent_bo_gate_v1",
            "status": "ready_for_fold_replay",
            "triggered": False,
            "final_interaction_audit_sha256": final_audit["audit_sha256"],
            "independent_bo_authorized": False,
            "independent_bo_run_count": 0,
            "next_state": "ready_for_fold_replay",
        },
        "receipt_sha256",
    )
    fold_proposal = _with_hash(
        {
            "proposal_version": "lyx_fold_replay_execution_proposal_v1",
            "status": "ready_for_execution",
            "parent_experiment_id": "parent-v1",
            "evidence_class": "development_replay_audit",
            "algorithm_level_holdout": False,
            "final_interaction_audit_sha256": final_audit["audit_sha256"],
            "pre_fold_gate_receipt_sha256": pre_fold_gate["receipt_sha256"],
            "pre_fold_gate_resolution": "pre_fold_gate_not_triggered",
            "pre_fold_human_decision_sha256": None,
            "evaluation_hash": "7" * 64,
            "independent_bo_run_count": 0,
        },
        "proposal_sha256",
    )
    summaries: list[dict[str, object]] = []
    target_audits: dict[str, dict[str, object]] = {}
    for scene, record_id in records:
        fold_id = f"{scene}-fold-{record_id}"
        passed = record_id not in failures
        shared_mae = 4.1 if record_id in gaps else 1.2
        rise = 1.0 if scene in {"kaihe", "run"} else None
        audit = _with_hash(
            {
                "receipt_version": "lyx_fold_target_audit_receipt_v1",
                "fold_id": fold_id,
                "selection_sha256": canonical_sha256(
                    {
                        "fold_id": fold_id,
                        "selected_profile_id": "p0",
                    }
                ),
                "status": "passed" if passed else "failed",
                "audit_pass": passed,
                "failure_reasons": ([] if passed else ["independent_l10_gate"]),
                "target_performance_read_count": 1,
                "selected_filter_profile_id": "p0",
                "audit_target_record_id": record_id,
                "metrics": _metrics(
                    mae=shared_mae,
                    l10=2 if passed else 11,
                    rise=rise,
                ),
                "qualification": {
                    "qualified": passed,
                    "elimination_reasons": ([] if passed else ["independent_l10_gate"]),
                },
            },
            "receipt_sha256",
        )
        summary = {
            "fold_id": fold_id,
            "scene": scene,
            "training_record_ids": [
                other_record
                for other_scene, other_record in records
                if other_scene == scene and other_record != record_id
            ],
            "audit_target_record_id": record_id,
            "selection_status": "selected",
            "selected_filter_profile_id": "p0",
            "selection_sha256": audit["selection_sha256"],
            "target_audit_status": audit["status"],
            "audit_pass": passed,
            "failure_reasons": audit["failure_reasons"],
            "target_performance_read_count": 1,
            "selection_receipt": (f"folds/{fold_id}/fold_selection_receipt.json"),
            "read_barrier_receipt": (f"folds/{fold_id}/read_barrier_receipt.json"),
            "target_audit_receipt": (f"folds/{fold_id}/target_audit_receipt.json"),
        }
        summaries.append(summary)
        target_audits[fold_id] = audit
    selection = _with_hash(
        {
            "receipt_version": "lyx_fold_selection_aggregate_receipt_v1",
            "proposal_sha256": fold_proposal["proposal_sha256"],
            "evidence_class": "development_replay_audit",
            "algorithm_level_holdout": False,
            "logical_slot_count": 12,
            "denominator_slot_count": 12,
            "folds": summaries,
            "candidate_or_threshold_revision_count": 0,
        },
        "receipt_sha256",
    )
    passed_count = sum(bool(summary["audit_pass"]) for summary in summaries)
    fold_report = _with_hash(
        {
            "report_version": "lyx_fold_replay_report_v1",
            "status": "complete",
            "proposal_sha256": fold_proposal["proposal_sha256"],
            "fold_selection_receipt_sha256": selection["receipt_sha256"],
            "evidence_class": "development_replay_audit",
            "algorithm_level_holdout": False,
            "logical_slot_count": 12,
            "denominator_slot_count": 12,
            "passed_slot_count": passed_count,
            "failed_slot_count": 12 - passed_count,
            "no_safe_shared_candidate_count": 0,
            "target_result_reuse_count": 12,
            "initial_planned_unique_identity_count": 0,
            "planned_unique_identity_count": 0,
            "registered_unique_identity_count": 0,
            "actual_unique_run_count": 0,
            "supplemental_identity_count": 0,
            "formal_solver_run_count": 0,
            "cache_hit_count": 12,
            "independent_bo_run_count": 0,
            "candidate_or_threshold_revision_count": 0,
            "folds": summaries,
            "supplemental_identities": [],
            "supplement_registry_snapshot_sha256": None,
            "next_state": "ready_for_post_fold_independent_bo_gate",
        },
        "report_sha256",
    )
    return {
        "fold_proposal": fold_proposal,
        "pre_fold_gate": pre_fold_gate,
        "final_audit": final_audit,
        "historical": historical,
        "current_role": current_role,
        "selection": selection,
        "fold_report": fold_report,
        "target_audits": target_audits,
        "review_context": _review_context(),
    }


def _evaluate(bundle: dict[str, object]) -> dict[str, object]:
    return evaluate_post_fold_independent_bo_gate(
        fold_replay_proposal=bundle["fold_proposal"],
        pre_fold_gate_receipt=bundle["pre_fold_gate"],
        fold_replay_report=bundle["fold_report"],
        fold_selection_receipt=bundle["selection"],
        target_audits_by_fold=bundle["target_audits"],
        final_interaction_audit=bundle["final_audit"],
        historical_ab_report=bundle["historical"],
        current_role_matrix=bundle["current_role"],
        review_context=bundle["review_context"],
    )


def test_post_fold_gate_has_exactly_two_false_conditions_when_all_slots_pass() -> None:
    gate = _evaluate(_bundle())

    assert gate["triggered"] is False
    assert gate["status"] == "ready_for_final_development_report"
    assert set(gate["conditions"]) == {
        "shared_replay_vs_sample_in_upper_bound_gap",
        "failed_slots_with_safe_sample_in_upper_bound",
    }
    assert all(condition["triggered"] is False for condition in gate["conditions"].values())
    assert gate["independent_bo_authorized"] is False
    assert gate["independent_bo_run_count"] == 0


def test_gap_condition_requires_three_records_across_two_scenes() -> None:
    two_records = _bundle(
        gap_record_ids={"jianpan1", "kaihe1"},
    )
    three_one_scene = _bundle(
        gap_record_ids={"jianpan1", "jianpan2", "jianpan3"},
    )
    three_two_scenes = _bundle(
        gap_record_ids={"jianpan1", "jianpan2", "kaihe1"},
    )

    assert _evaluate(two_records)["triggered"] is False
    assert _evaluate(three_one_scene)["triggered"] is False
    gate = _evaluate(three_two_scenes)
    assert gate["triggered"] is True
    assert gate["conditions"]["shared_replay_vs_sample_in_upper_bound_gap"]["triggered"] is True
    assert gate["review_packet"]["independent_bo_request"]["unique_budget"] == 5400


def test_failed_slot_condition_requires_every_corresponding_upper_bound_safe() -> None:
    failed = {"xiezi1", "jianpan1", "run1"}
    safe_bundle = _bundle(failed_record_ids=failed)
    unsafe_bundle = _bundle(
        failed_record_ids=failed,
        unsafe_sample_record_ids={"run1"},
    )

    safe_gate = _evaluate(safe_bundle)
    unsafe_gate = _evaluate(unsafe_bundle)

    assert (
        safe_gate["conditions"]["failed_slots_with_safe_sample_in_upper_bound"]["triggered"] is True
    )
    assert unsafe_gate["triggered"] is False
    assert (
        unsafe_gate["conditions"]["failed_slots_with_safe_sample_in_upper_bound"][
            "all_corresponding_sample_in_upper_bounds_pass"
        ]
        is False
    )


def test_triggered_gate_fails_closed_on_incomplete_review_context() -> None:
    bundle = _bundle(
        gap_record_ids={"jianpan1", "jianpan2", "kaihe1"},
    )
    bundle["review_context"] = deepcopy(bundle["review_context"])
    bundle["review_context"].pop("seed_manifest_hash")

    with pytest.raises(
        FoldReplayError,
        match="post_fold_review_context_field_set_mismatch",
    ):
        _evaluate(bundle)


def test_post_fold_gate_refuses_a_human_resolved_but_triggered_pre_fold_gate() -> None:
    bundle = _bundle()
    triggered_pre_gate = _with_hash(
        {
            **{
                key: value
                for key, value in bundle["pre_fold_gate"].items()
                if key != "receipt_sha256"
            },
            "status": "awaiting_human_independent_bo_decision",
            "triggered": True,
            "next_state": "awaiting_human_independent_bo_decision",
        },
        "receipt_sha256",
    )
    bundle["pre_fold_gate"] = triggered_pre_gate
    bundle["fold_proposal"] = _with_hash(
        {
            **{
                key: value
                for key, value in bundle["fold_proposal"].items()
                if key != "proposal_sha256"
            },
            "pre_fold_gate_receipt_sha256": triggered_pre_gate["receipt_sha256"],
            "pre_fold_gate_resolution": ("human_approved_current_non_bo_flow"),
            "pre_fold_human_decision_sha256": "8" * 64,
        },
        "proposal_sha256",
    )
    bundle["fold_report"] = _with_hash(
        {
            **{
                key: value for key, value in bundle["fold_report"].items() if key != "report_sha256"
            },
            "proposal_sha256": bundle["fold_proposal"]["proposal_sha256"],
        },
        "report_sha256",
    )

    with pytest.raises(
        FoldReplayError,
        match="post_fold_pre_fold_gate_provenance_mismatch",
    ):
        _evaluate(bundle)


def test_future_bo_authorization_must_match_all_five_request_fields() -> None:
    gate = _evaluate(
        _bundle(
            gap_record_ids={"jianpan1", "jianpan2", "kaihe1"},
        )
    )
    context = _review_context()
    valid_receipt = {
        "approved": True,
        "decision_state": "awaiting_human_independent_bo_decision",
        "solver_hash": context["solver_hash"],
        "search_space_hash": context["search_space_hash"],
        "metric_contract_hash": context["metric_contract_hash"],
        "seed_manifest_hash": context["seed_manifest_hash"],
        "unique_budget": context["unique_budget"],
        "approved_at": "2026-07-30T03:00:00+08:00",
        "approved_by": "test-user",
    }

    with pytest.raises(HumanGateRequiredError):
        validate_post_fold_independent_bo_authorization(
            gate_receipt=gate,
            authorization_receipt=None,
        )
    mismatched = {**valid_receipt, "unique_budget": 5399}
    with pytest.raises(
        GovernanceError,
        match="authorization_identity_mismatch:unique_budget",
    ):
        validate_post_fold_independent_bo_authorization(
            gate_receipt=gate,
            authorization_receipt=mismatched,
        )
    assert (
        validate_post_fold_independent_bo_authorization(
            gate_receipt=gate,
            authorization_receipt=valid_receipt,
        )
        == valid_receipt
    )


def test_final_report_separates_five_layers_and_limits_claims() -> None:
    bundle = _bundle()
    gate = _evaluate(bundle)
    budget = BudgetContract.approved_v5()
    report = build_final_development_report(
        gate_receipt=gate,
        fold_replay_proposal=bundle["fold_proposal"],
        pre_fold_gate_receipt=bundle["pre_fold_gate"],
        fold_replay_report=bundle["fold_report"],
        fold_selection_receipt=bundle["selection"],
        target_audits_by_fold=bundle["target_audits"],
        final_interaction_audit=bundle["final_audit"],
        historical_ab_report=bundle["historical"],
        current_role_matrix=bundle["current_role"],
        budget_contract=budget.to_dict(),
        attempt_registry_summary={
            "logical_task_count": 0,
            "planned_unique_identity_count": 0,
            "actual_unique_run_count": 0,
            "cache_evidence_count": 0,
            "cache_hit_count": 0,
            "failed_attempt_count": 0,
            "retry_count": 0,
            "total_attempt_count": 0,
            "by_stage": {},
        },
        budget_amendment_authorizations=_budget_authorizations(),
    )
    markdown = render_final_development_report_markdown(report)

    assert report["status"] == "complete"
    assert report["comparison_layer_order"] == [
        "historical_independent_bo_lite",
        "same_identity_recovery_ab",
        "same_role_current_mechanism",
        "combination_library_sample_in_upper_bound",
        "scene_shared_profile_replay",
    ]
    assert report["trace_rescue_treatment"]["included_in_primary_baseline_tables"] is False
    assert report["conclusion"]["claim_status"] == ("development_replay_audit_only")
    assert report["conclusion"]["unseen_scene_generalization_passed"] is False
    assert "TraceRescue 只保留为历史探索背景" in markdown
    assert "未见场景或跨个体泛化证据" in markdown
    assert "同身份当前恢复 MAE" in markdown
    assert report["provenance"]["parent_experiment_id"] == "parent-v1"
    assert len(report["provenance"]["configuration_hashes"]) == 96
    assert (
        report["budget_audit"]["original_mechanism_body_contract"]["within_original_body_contract"]
        is True
    )

    with pytest.raises(
        FoldReplayError,
        match="post_fold_budget_amendment_chain_incomplete",
    ):
        build_final_development_report(
            gate_receipt=gate,
            fold_replay_proposal=bundle["fold_proposal"],
            pre_fold_gate_receipt=bundle["pre_fold_gate"],
            fold_replay_report=bundle["fold_report"],
            fold_selection_receipt=bundle["selection"],
            target_audits_by_fold=bundle["target_audits"],
            final_interaction_audit=bundle["final_audit"],
            historical_ab_report=bundle["historical"],
            current_role_matrix=bundle["current_role"],
            budget_contract=budget.to_dict(),
            attempt_registry_summary={
                "logical_task_count": 0,
                "planned_unique_identity_count": 0,
                "actual_unique_run_count": 0,
                "cache_evidence_count": 0,
                "cache_hit_count": 0,
                "failed_attempt_count": 0,
                "retry_count": 0,
                "total_attempt_count": 0,
                "by_stage": {},
            },
            budget_amendment_authorizations=(_budget_authorizations()[:-1]),
        )

    handoff = build_challenge_scene_handoff(
        final_report=report,
        final_interaction_audit=bundle["final_audit"],
        fold_selection_receipt=bundle["selection"],
        challenge_scene_manifest=default_challenge_scene_manifest(),
    )
    assert handoff["status"] == "ready_for_unseen_scene_validation"
    assert handoff["challenge_protocol"]["reserved_scene_ids"] == ["bobi"]
    assert handoff["challenge_protocol"]["challenge_result_read_count_at_freeze"] == 0
    assert "cross_person_generalization_passed" in handoff["claims_not_yet_allowed"]


def _write_publisher_inputs(
    tmp_path: Path,
    bundle: dict[str, object],
) -> dict[str, object]:
    fold_root = tmp_path / "fold-output"
    fold_root.mkdir()
    paths = {
        "fold_proposal": tmp_path / "fold_replay_proposal.json",
        "pre_fold_gate": tmp_path / "pre_fold_independent_bo_gate_receipt.json",
        "fold_report": fold_root / "fold_replay_report.json",
        "selection": fold_root / "fold_selection_receipt.json",
        "final_audit": tmp_path / "final_interaction_audit.json",
        "historical": tmp_path / "historical_ab_report.json",
        "current_role": tmp_path / "current_role_matrix.json",
        "review_context": tmp_path / "review_context.json",
        "budget": tmp_path / "budget_contract.json",
        "exploration": tmp_path / "exploration_registry.json",
        "challenge": tmp_path / "challenge_scene_manifest.json",
    }
    atomic_write_json(paths["fold_proposal"], bundle["fold_proposal"])
    atomic_write_json(paths["pre_fold_gate"], bundle["pre_fold_gate"])
    atomic_write_json(paths["fold_report"], bundle["fold_report"])
    atomic_write_json(paths["selection"], bundle["selection"])
    atomic_write_json(paths["final_audit"], bundle["final_audit"])
    atomic_write_json(paths["historical"], bundle["historical"])
    atomic_write_json(paths["current_role"], bundle["current_role"])
    atomic_write_json(paths["review_context"], bundle["review_context"])
    budget = BudgetContract.approved_v5()
    exploration = ExplorationRegistry.zero_budget_v1()
    atomic_write_json(paths["budget"], budget.to_dict())
    atomic_write_json(paths["exploration"], exploration.to_dict())
    atomic_write_json(
        paths["challenge"],
        default_challenge_scene_manifest(),
    )
    paths["budget_authorizations"] = []
    for index, authorization in enumerate(_budget_authorizations()):
        path = tmp_path / f"budget-amendment-{index}.json"
        atomic_write_json(path, authorization)
        paths["budget_authorizations"].append(path)
    for summary in bundle["fold_report"]["folds"]:
        audit_path = fold_root / summary["target_audit_receipt"]
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            audit_path,
            bundle["target_audits"][summary["fold_id"]],
        )
    governance = tmp_path / "governance"
    governance.mkdir()
    paths["registry"] = governance / "attempt_registry.json"
    AttemptRegistry.create(
        paths["registry"],
        budget_contract=budget,
        exploration_registry=exploration,
    )
    return paths


def test_source_bound_publisher_adds_no_solver_identity_and_writes_handoff(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    paths = _write_publisher_inputs(tmp_path, bundle)
    output = tmp_path / "post-fold-output"

    completion = publish_post_fold_package(
        fold_replay_proposal_path=paths["fold_proposal"],
        pre_fold_gate_receipt_path=paths["pre_fold_gate"],
        fold_replay_report_path=paths["fold_report"],
        fold_selection_receipt_path=paths["selection"],
        final_interaction_audit_path=paths["final_audit"],
        historical_ab_report_path=paths["historical"],
        current_role_matrix_path=paths["current_role"],
        review_context_path=paths["review_context"],
        budget_contract_path=paths["budget"],
        exploration_registry_path=paths["exploration"],
        attempt_registry_path=paths["registry"],
        budget_amendment_authorization_paths=paths["budget_authorizations"],
        challenge_scene_manifest_path=paths["challenge"],
        output_dir=output,
    )

    assert completion["status"] == "complete"
    assert completion["planned_unique_identity_count"] == 0
    assert completion["actual_unique_run_count"] == 0
    assert completion["independent_bo_run_count"] == 0
    assert (output / "challenge_scene_handoff.json").is_file()
    assert not (output / "independent_bo_review_packet.json").exists()
    assert read_json(output / "post_fold_completion.json") == completion
    registry = read_json(paths["registry"])
    assert registry["summary"]["planned_unique_identity_count"] == 0


def test_publisher_writes_review_packet_but_never_authorizes_bo(
    tmp_path: Path,
) -> None:
    bundle = _bundle(
        gap_record_ids={"jianpan1", "jianpan2", "kaihe1"},
    )
    paths = _write_publisher_inputs(tmp_path, bundle)
    output = tmp_path / "post-fold-output"

    completion = publish_post_fold_package(
        fold_replay_proposal_path=paths["fold_proposal"],
        pre_fold_gate_receipt_path=paths["pre_fold_gate"],
        fold_replay_report_path=paths["fold_report"],
        fold_selection_receipt_path=paths["selection"],
        final_interaction_audit_path=paths["final_audit"],
        historical_ab_report_path=paths["historical"],
        current_role_matrix_path=paths["current_role"],
        review_context_path=paths["review_context"],
        budget_contract_path=paths["budget"],
        exploration_registry_path=paths["exploration"],
        attempt_registry_path=paths["registry"],
        budget_amendment_authorization_paths=paths["budget_authorizations"],
        challenge_scene_manifest_path=paths["challenge"],
        output_dir=output,
    )

    packet = read_json(output / "independent_bo_review_packet.json")
    assert completion["status"] == "awaiting_human_independent_bo_decision"
    assert completion["next_state"] == ("awaiting_human_independent_bo_decision")
    assert packet["independent_bo_authorized"] is False
    assert packet["independent_bo_run_count"] == 0
    assert not (output / "challenge_scene_handoff.json").exists()
