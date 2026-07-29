from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ppg_hr.v2.phase2_experiment_io import (
    atomic_write_json,
    read_json,
)
from ppg_hr.v2.recovery_contracts import canonical_sha256
from ppg_hr.v2.recovery_experiment_governance import (
    AttemptIdentity,
    AttemptRegistry,
    BudgetContract,
    ExplorationRegistry,
)
from ppg_hr.v2.recovery_fold_replay_experiment import (
    FoldReplayError,
    audit_selected_target,
    build_fold_replay_proposal,
    execute_fold_replay_proposal,
    propose_fold_replay_execution,
    select_fold_profile,
)

_SCENES = ("jianpan", "kaihe", "run", "xiezi")
_PROFILE_IDS = tuple(f"p{index}" for index in range(8))


def _with_hash(
    payload: dict[str, object],
    field: str,
) -> dict[str, object]:
    payload[field] = canonical_sha256(payload)
    return payload


def _row(
    *,
    record_id: str,
    scene: str,
    profile_id: str,
    profile_index: int,
    qualified: bool = True,
    reasons: list[str] | None = None,
) -> dict[str, object]:
    config = {
        "parameters": {
            "profile_id": profile_id,
            "record_id": record_id,
        }
    }
    config_hash = canonical_sha256(config)
    data_sha256 = canonical_sha256({"record_id": record_id})
    attempt = AttemptIdentity(
        solver_hash="1" * 64,
        config_hash=config_hash,
        metric_contract_hash="2" * 64,
        evaluation_hash="3" * 64,
        data_sha256=data_sha256,
        record_id=record_id,
        stage="penalty_interaction",
        attempt_kind="formal",
        parent_experiment_id="parent-v1",
    )
    identity_sha = attempt.sha256
    elimination_reasons = (
        [] if reasons is None and qualified else list(reasons or ["independent_mae_gate"])
    )
    return {
        "solver_hash": "1" * 64,
        "config_hash": config_hash,
        "metric_contract_hash": "2" * 64,
        "evaluation_hash": "3" * 64,
        "data_sha256": data_sha256,
        "record_id": record_id,
        "stage": "penalty_interaction",
        "attempt_kind": "formal",
        "parent_experiment_id": "parent-v1",
        "identity_sha256": identity_sha,
        "cache_identity_sha256": identity_sha,
        "config": config,
        "data_path": f"C:/data/{record_id}.csv",
        "reference_path": f"C:/data/{record_id}-ref.csv",
        "raw_data_sha256": canonical_sha256({"raw_record_id": record_id}),
        "reference_sha256": canonical_sha256({"reference_record_id": record_id}),
        "method_names": ["HF"],
        "scene": scene,
        "true_rise_applicable": scene in {"run", "kaihe"},
        "filter_profile_id": profile_id,
        "filter_profile_sha256": canonical_sha256({"profile_id": profile_id}),
        "filter_profile_design_role": "core",
        "physical_memory_ms": 100,
        "actual_taps": 10 + profile_index,
        "nominal_mu": 0.01,
        "recovery_candidate_id": "recovery-final-v1",
        "recovery_candidate_sha256": "4" * 64,
        "candidate_min_bpm": None,
        "penalty_candidate_id": "penalty-final-v1",
        "penalty_candidate_sha256": "5" * 64,
        "metrics": {
            "longest_e10_run_windows": 2 + profile_index,
            "longest_e20_run_windows": 0,
            "final_motion_mae_bpm": 2.0 + profile_index / 10.0,
            "right_censored_recovery_count": 0,
            "max_recovered_delay_s": 1.0 + profile_index / 10.0,
            "max_rise_underestimate_bpm": (1.0 if scene in {"run", "kaihe"} else None),
            "recovery_episode_count": 1,
            "total_window_count": 30,
        },
        "spectral_audit": {
            "stability_pass": True,
            "spectral_gate_pass": True,
            "stage_r_spectral_gate": {
                "spectral_gate_pass": True,
                "valid_window_count": 2,
                "invalid_window_count": 0,
            },
        },
        "qualification": {
            "qualified": qualified,
            "elimination_reasons": elimination_reasons,
            "independent_delta_mae_bpm": profile_index / 10.0,
            "current_delta_mae_bpm": 0.0,
        },
    }


def _final_audit(
    *,
    mutate_rows=None,
) -> dict[str, object]:
    rows = [
        _row(
            record_id=f"{scene}{record_index}",
            scene=scene,
            profile_id=profile_id,
            profile_index=profile_index,
        )
        for scene in _SCENES
        for record_index in range(1, 4)
        for profile_index, profile_id in enumerate(_PROFILE_IDS)
    ]
    if mutate_rows is not None:
        mutate_rows(rows)
    profile_receipts: dict[str, dict[str, object]] = {}
    for profile_id in _PROFILE_IDS:
        profile_rows = [row for row in rows if row["filter_profile_id"] == profile_id]
        profile_receipts[profile_id] = _with_hash(
            {
                "receipt_version": ("lyx_final_filter_profile_receipt_v1"),
                "filter_profile_id": profile_id,
                "final_recovery_id": "recovery-final-v1",
                "selected_penalty_id": "penalty-final-v1",
                "record_count": 12,
                "qualified_record_count": sum(
                    row["qualification"]["qualified"] for row in profile_rows
                ),
                "hard_gate_failure_count": sum(
                    not row["qualification"]["qualified"] for row in profile_rows
                ),
                "identity_sha256": sorted(row["identity_sha256"] for row in profile_rows),
            },
            "receipt_sha256",
        )
    return _with_hash(
        {
            "audit_version": "lyx_final_interaction_audit_v1",
            "status": "complete",
            "evidence_class": "development_reuse_pilot",
            "algorithm_level_holdout": False,
            "final_recovery_id": "recovery-final-v1",
            "selected_penalty_id": "penalty-final-v1",
            "row_count": 96,
            "rows": rows,
            "profile_receipts": profile_receipts,
            "independent_bo_run_count": 0,
        },
        "audit_sha256",
    )


def _pre_fold_gate(
    audit: dict[str, object],
    *,
    triggered: bool = False,
) -> dict[str, object]:
    status = "awaiting_human_independent_bo_decision" if triggered else "ready_for_fold_replay"
    return _with_hash(
        {
            "receipt_version": "lyx_pre_fold_independent_bo_gate_v1",
            "status": status,
            "triggered": triggered,
            "final_interaction_audit_sha256": audit["audit_sha256"],
            "independent_bo_run_count": 0,
            "independent_bo_authorized": False,
            "next_state": status,
        },
        "receipt_sha256",
    )


def _human_decision(
    gate: dict[str, object],
) -> dict[str, object]:
    return _with_hash(
        {
            "decision_version": ("lyx_pre_fold_independent_bo_human_decision_v1"),
            "gate_receipt_sha256": gate["receipt_sha256"],
            "continue_current_non_bo_flow": True,
            "run_independent_bo_now": False,
            "independent_bo_run_count": 0,
            "decided_by": "test-user",
            "decided_at": "2026-07-29T00:00:00+08:00",
        },
        "decision_sha256",
    )


def _budget() -> dict[str, object]:
    return BudgetContract.approved_v5().to_dict()


def _training_payload(
    *,
    record_id: str,
    scene: str,
    qualified: bool = True,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "scene": scene,
        "profile_rows": [
            _row(
                record_id=record_id,
                scene=scene,
                profile_id=profile_id,
                profile_index=profile_index,
                qualified=qualified,
            )
            for profile_index, profile_id in enumerate(_PROFILE_IDS)
        ],
    }


def _write_inputs(
    tmp_path: Path,
    *,
    audit: dict[str, object],
    gate: dict[str, object],
    decision: dict[str, object] | None = None,
) -> dict[str, Path]:
    paths = {
        "audit": tmp_path / "final-interaction-audit.json",
        "gate": tmp_path / "pre-fold-gate.json",
        "budget": tmp_path / "budget.json",
    }
    atomic_write_json(paths["audit"], audit)
    atomic_write_json(paths["gate"], gate)
    atomic_write_json(paths["budget"], _budget())
    if decision is not None:
        paths["decision"] = tmp_path / "human-decision.json"
        atomic_write_json(paths["decision"], decision)
    return paths


def _write_governance(tmp_path: Path) -> Path:
    governance = tmp_path / "governance"
    governance.mkdir()
    budget = BudgetContract.approved_v5()
    exploration = ExplorationRegistry.zero_budget_v1()
    atomic_write_json(
        governance / "budget_contract.json",
        budget.to_dict(),
    )
    atomic_write_json(
        governance / "exploration_registry.json",
        exploration.to_dict(),
    )
    AttemptRegistry.create(
        governance / "attempt_registry.json",
        budget_contract=budget,
        exploration_registry=exploration,
    )
    return governance


def _publish_package(
    tmp_path: Path,
    *,
    audit: dict[str, object] | None = None,
    gate: dict[str, object] | None = None,
    decision: dict[str, object] | None = None,
) -> Path:
    frozen_audit = _final_audit() if audit is None else audit
    frozen_gate = _pre_fold_gate(frozen_audit) if gate is None else gate
    paths = _write_inputs(
        tmp_path,
        audit=frozen_audit,
        gate=frozen_gate,
        decision=decision,
    )
    proposal_dir = tmp_path / "proposal"
    propose_fold_replay_execution(
        final_interaction_audit_path=paths["audit"],
        pre_fold_gate_receipt_path=paths["gate"],
        pre_fold_human_decision_path=paths.get("decision"),
        budget_contract_path=paths["budget"],
        output_dir=proposal_dir,
        source_root=Path(__file__).parents[1] / "src",
        parent_experiment_id="parent-v1",
    )
    return proposal_dir


def test_selector_uses_only_two_training_records_and_lexicographic_key() -> None:
    selection = select_fold_profile(
        fold_id="run-fold-1-run3",
        scene="run",
        training_record_payloads=[
            _training_payload(record_id="run1", scene="run"),
            _training_payload(record_id="run2", scene="run"),
        ],
        audit_target_record_id="run3",
        profile_ids=_PROFILE_IDS,
    )

    assert selection["status"] == "selected"
    assert selection["selected_filter_profile_id"] == "p0"
    assert selection["training_record_ids"] == ["run1", "run2"]
    assert selection["target_performance_read_count_before_freeze"] == 0
    assert len(selection["candidate_elimination_chain"]) == 8


def test_training_pair_mean_gate_is_not_hidden_by_per_record_passes() -> None:
    payloads = [
        _training_payload(record_id="xiezi1", scene="xiezi"),
        _training_payload(record_id="xiezi2", scene="xiezi"),
    ]
    for payload in payloads:
        for row in payload["profile_rows"]:
            row["qualification"]["independent_delta_mae_bpm"] = 1.1

    selection = select_fold_profile(
        fold_id="xiezi-fold-3-xiezi3",
        scene="xiezi",
        training_record_payloads=payloads,
        audit_target_record_id="xiezi3",
        profile_ids=_PROFILE_IDS,
    )

    assert selection["status"] == "no_safe_shared_candidate"
    assert {
        reason
        for candidate in selection["candidate_elimination_chain"]
        for reason in candidate["elimination_reasons"]
    } == {"training_pair_mean_independent_mae_gate"}


def test_target_audit_is_bound_to_frozen_profile_and_identity() -> None:
    selection = select_fold_profile(
        fold_id="jianpan-fold-1-jianpan3",
        scene="jianpan",
        training_record_payloads=[
            _training_payload(record_id="jianpan1", scene="jianpan"),
            _training_payload(record_id="jianpan2", scene="jianpan"),
        ],
        audit_target_record_id="jianpan3",
        profile_ids=_PROFILE_IDS,
    )
    selected = _row(
        record_id="jianpan3",
        scene="jianpan",
        profile_id="p0",
        profile_index=0,
    )
    passed = audit_selected_target(
        selection_receipt=selection,
        target_result_payload={
            "record_id": "jianpan3",
            "selected_row": selected,
        },
        expected_identity_sha256=selected["identity_sha256"],
    )
    mismatched = audit_selected_target(
        selection_receipt=selection,
        target_result_payload={
            "record_id": "jianpan3",
            "selected_row": selected,
        },
        expected_identity_sha256="f" * 64,
    )

    assert passed["audit_pass"] is True
    assert mismatched["status"] == ("identity_mismatch_requires_supplement")
    assert mismatched["failure_reasons"] == ["identity_mismatch_requires_supplement"]


def test_proposal_freezes_12_slots_and_physically_separates_target_data(
    tmp_path: Path,
) -> None:
    proposal_dir = _publish_package(tmp_path)
    proposal = read_json(proposal_dir / "fold_replay_proposal.json")
    manifest = read_json(proposal_dir / "data_role_manifest.json")
    source_index = read_json(proposal_dir / "source_index.json")

    assert proposal["logical_task_count"] == 12
    assert proposal["planned_unique_identity_count"] == 0
    assert len(proposal["folds"]) == 12
    assert source_index["source_count"] == 120
    assert manifest["algorithm_level_holdout"] is False
    assert manifest["evidence_class"] == "development_replay_audit"
    target_source = read_json(
        proposal_dir / manifest["folds"][0]["target_preselection_source"]["path"]
    )
    assert set(target_source) == {
        "source_version",
        "sample_id",
        "record_id",
        "source_sha256",
    }
    assert manifest["folds"][0]["target_preselection_source"]["denied_field_classes"] == [
        "mae",
        "spectral",
        "long_tail",
        "independent_bo_parameter_summary",
        "derived_performance",
    ]


def test_triggered_pre_fold_gate_requires_exact_human_decision() -> None:
    audit = _final_audit()
    gate = _pre_fold_gate(audit, triggered=True)

    with pytest.raises(
        FoldReplayError,
        match="fold_replay_pre_fold_gate_awaiting_human_decision",
    ):
        build_fold_replay_proposal(
            final_interaction_audit=audit,
            pre_fold_gate_receipt=gate,
            budget_contract=_budget(),
            parent_experiment_id="parent-v1",
            evaluation_hash="4" * 64,
        )

    proposal, _ = build_fold_replay_proposal(
        final_interaction_audit=audit,
        pre_fold_gate_receipt=gate,
        pre_fold_human_decision=_human_decision(gate),
        budget_contract=_budget(),
        parent_experiment_id="parent-v1",
        evaluation_hash="4" * 64,
    )
    assert proposal["pre_fold_gate_resolution"] == ("human_approved_current_non_bo_flow")
    assert proposal["independent_bo_run_count"] == 0


def test_execution_completes_12_slots_with_zero_solver_runs(
    tmp_path: Path,
) -> None:
    proposal_dir = _publish_package(tmp_path)
    output_dir = tmp_path / "output"

    completion = execute_fold_replay_proposal(
        proposal_dir=proposal_dir,
        governance_dir=tmp_path / "unused-governance",
        output_dir=output_dir,
        source_root=Path(__file__).parents[1] / "src",
    )

    assert completion["status"] == "complete"
    assert completion["logical_slot_count"] == 12
    assert completion["denominator_slot_count"] == 12
    assert completion["passed_slot_count"] == 12
    assert completion["failed_slot_count"] == 0
    assert completion["planned_unique_identity_count"] == 0
    assert completion["registered_unique_identity_count"] == 0
    assert completion["actual_unique_run_count"] == 0
    assert completion["formal_solver_run_count"] == 0
    assert completion["cache_hit_count"] == 12
    report = read_json(output_dir / "fold_replay_report.json")
    assert report["target_result_reuse_count"] == 12
    assert report["supplemental_identity_count"] == 0
    first_fold = report["folds"][0]
    barrier = read_json(output_dir / first_fold["read_barrier_receipt"])
    target_access = next(
        access for access in barrier["accesses"] if access["role"] == "audit_target"
    )
    assert target_access["fields"] == ["sample_id", "record_id"]
    target_audit = read_json(output_dir / first_fold["target_audit_receipt"])
    assert target_audit["target_access"]["fields"] == [
        "record_id",
        "scene",
        "selected_row",
    ]
    assert target_audit["target_access"]["selected_profile_result_count"] == 1
    assert (
        execute_fold_replay_proposal(
            proposal_dir=proposal_dir,
            governance_dir=tmp_path / "unused-governance",
            output_dir=output_dir,
            source_root=Path(__file__).parents[1] / "src",
        )
        == completion
    )


def test_failed_slots_remain_in_denominator(
    tmp_path: Path,
) -> None:
    def mutate(rows: list[dict[str, object]]) -> None:
        for row in rows:
            if row["record_id"] == "xiezi1":
                row["qualification"] = {
                    "qualified": False,
                    "elimination_reasons": ["independent_l10_gate"],
                    "independent_delta_mae_bpm": 0.0,
                    "current_delta_mae_bpm": 0.0,
                }

    audit = _final_audit(mutate_rows=mutate)
    proposal_dir = _publish_package(
        tmp_path,
        audit=audit,
        gate=_pre_fold_gate(audit),
    )
    output_dir = tmp_path / "output"
    completion = execute_fold_replay_proposal(
        proposal_dir=proposal_dir,
        governance_dir=tmp_path / "unused-governance",
        output_dir=output_dir,
        source_root=Path(__file__).parents[1] / "src",
    )
    report = read_json(output_dir / "fold_replay_report.json")

    assert completion["denominator_slot_count"] == 12
    assert completion["failed_slot_count"] == 3
    assert report["no_safe_shared_candidate_count"] == 2
    failed = [fold for fold in report["folds"] if not fold["audit_pass"]]
    assert len(failed) == 3
    assert any("independent_l10_gate" in fold["failure_reasons"] for fold in failed)
    assert sum("no_safe_shared_candidate" in fold["failure_reasons"] for fold in failed) == 2


def test_identity_mismatch_registers_one_bounded_supplement(
    tmp_path: Path,
) -> None:
    def mutate(rows: list[dict[str, object]]) -> None:
        row = next(
            row
            for row in rows
            if row["record_id"] == "jianpan1" and row["filter_profile_id"] == "p0"
        )
        row["identity_sha256"] = "f" * 64
        row["cache_identity_sha256"] = "f" * 64

    audit = _final_audit(mutate_rows=mutate)
    proposal_dir = _publish_package(
        tmp_path,
        audit=audit,
        gate=_pre_fold_gate(audit),
    )
    governance = _write_governance(tmp_path)
    output_dir = tmp_path / "output"

    completion = execute_fold_replay_proposal(
        proposal_dir=proposal_dir,
        governance_dir=governance,
        output_dir=output_dir,
        source_root=Path(__file__).parents[1] / "src",
    )
    report = read_json(output_dir / "fold_replay_report.json")
    registry = read_json(governance / "attempt_registry.json")

    assert completion["status"] == ("awaiting_fold_replay_supplement_execution")
    assert completion["supplemental_identity_count"] == 1
    assert completion["planned_unique_identity_count"] == 1
    assert completion["registered_unique_identity_count"] == 1
    assert completion["actual_unique_run_count"] == 0
    assert completion["formal_solver_run_count"] == 0
    assert completion["cache_hit_count"] == 11
    assert report["target_result_reuse_count"] == 11
    assert report["supplemental_identities"][0]["stage"] == ("fold_replay")
    assert report["supplemental_identities"][0]["reason"] == (
        "selected_target_numerical_identity_mismatch"
    )
    execution_item = report["supplemental_identities"][0]["execution_item"]
    assert execution_item["stage"] == "fold_replay"
    assert execution_item["config"]["parameters"]["profile_id"] == "p0"
    assert len(registry["entries"]) == 1
    entry = next(iter(registry["entries"].values()))
    assert entry["identity"]["stage"] == "fold_replay"
    assert entry["status"] == "registered"


def test_package_tampering_fails_before_any_output(
    tmp_path: Path,
) -> None:
    proposal_dir = _publish_package(tmp_path)
    target = proposal_dir / "target_identity_sources" / "run1.json"
    tampered = deepcopy(read_json(target))
    tampered["sample_id"] = "changed"
    atomic_write_json(target, tampered)
    output_dir = tmp_path / "output"

    with pytest.raises(
        FoldReplayError,
        match="fold_replay_internal_source_changed",
    ):
        execute_fold_replay_proposal(
            proposal_dir=proposal_dir,
            governance_dir=tmp_path / "unused-governance",
            output_dir=output_dir,
            source_root=Path(__file__).parents[1] / "src",
        )

    assert not output_dir.exists()
