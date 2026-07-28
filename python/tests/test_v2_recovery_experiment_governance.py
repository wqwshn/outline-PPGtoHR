from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import ppg_hr.v2.recovery_experiment_governance as governance
from ppg_hr.v2.recovery_experiment_governance import (
    AttemptIdentity,
    AttemptRegistry,
    BudgetContract,
    CacheEvidence,
    DataRoleManifest,
    ExplorationRegistry,
    FoldReadBarrier,
    GovernanceError,
    HumanGateRequiredError,
    IndependentBORequest,
    RecordSource,
    initialize_recovery_experiment_governance,
    validate_human_gate,
    validate_independent_bo_authorization,
)


def _identity(
    *,
    record_id: str = "run1_LYX_0708",
    stage: str = "historical_recovery_ab",
    attempt_kind: str = "formal",
) -> AttemptIdentity:
    return AttemptIdentity(
        solver_hash="a" * 64,
        config_hash="b" * 64,
        metric_contract_hash="c" * 64,
        evaluation_hash="e" * 64,
        data_sha256="d" * 64,
        record_id=record_id,
        stage=stage,
        attempt_kind=attempt_kind,
        parent_experiment_id="lyx_recovery_filter_profile_v1",
    )


def test_attempt_registry_rejects_unregistered_and_limits_retry(
    tmp_path: Path,
) -> None:
    registry = AttemptRegistry.create(
        tmp_path / "attempt_registry.json",
        budget_contract=BudgetContract.frozen_v1(),
        exploration_registry=ExplorationRegistry.zero_budget_v1(),
    )
    identity = _identity()

    with pytest.raises(GovernanceError, match="unregistered_identity"):
        registry.begin_attempt(identity)

    registry.register_identity(identity)
    first = registry.begin_attempt(identity)
    registry.finish_attempt(first, status="failed", failure_reason="boom")
    second = registry.begin_attempt(identity)
    registry.finish_attempt(second, status="failed", failure_reason="boom")

    with pytest.raises(GovernanceError, match="retry_limit_exceeded"):
        registry.begin_attempt(identity)
    summary = registry.summary()
    assert summary["planned_unique_identity_count"] == 1
    assert summary["actual_unique_run_count"] == 1
    assert summary["failed_attempt_count"] == 2
    assert summary["retry_count"] == 1


def test_registered_execution_is_only_nominatable_after_evidence(
    tmp_path: Path,
) -> None:
    registry = AttemptRegistry.create(
        tmp_path / "attempt_registry.json",
        budget_contract=BudgetContract.frozen_v1(),
        exploration_registry=ExplorationRegistry.zero_budget_v1(),
    )
    identity = _identity()
    operation_called = False

    def operation() -> int:
        nonlocal operation_called
        operation_called = True
        return 42

    with pytest.raises(GovernanceError, match="unregistered_identity"):
        registry.execute_registered(identity, operation)
    assert operation_called is False

    registry.register_identity(identity)
    with pytest.raises(GovernanceError, match="nomination_without_evidence"):
        registry.assert_nominatable(identity)
    assert registry.execute_registered(identity, operation) == 42
    registry.assert_nominatable(identity)


def test_exploration_is_denied_by_default_zero_budget(tmp_path: Path) -> None:
    registry = AttemptRegistry.create(
        tmp_path / "attempt_registry.json",
        budget_contract=BudgetContract.frozen_v1(),
        exploration_registry=ExplorationRegistry.zero_budget_v1(),
    )

    with pytest.raises(GovernanceError, match="exploration_not_authorized"):
        registry.register_identity(_identity(stage="exploration", attempt_kind="exploration"))


def test_stage_policy_rejects_self_reported_attempt_kind(tmp_path: Path) -> None:
    registry = AttemptRegistry.create(
        tmp_path / "attempt_registry.json",
        budget_contract=BudgetContract.frozen_v1(),
        exploration_registry=ExplorationRegistry.zero_budget_v1(),
    )

    with pytest.raises(GovernanceError, match="stage_attempt_kind_mismatch"):
        registry.register_identity(
            _identity(
                stage="fixed_lower_bound_diagnostic",
                attempt_kind="formal",
            )
        )


def test_identity_binds_evaluation_and_cache_identity() -> None:
    first = _identity()
    changed = AttemptIdentity(
        **{
            **first.to_identity_dict(),
            "evaluation_hash": "f" * 64,
        }
    )

    assert first.sha256 != changed.sha256
    assert first.cache_identity_sha256 == first.sha256


def test_attempt_registry_rejects_stage_budget_overflow(tmp_path: Path) -> None:
    contract = BudgetContract(
        stage_unique_limits={"tiny": 1},
        max_unique_identities=1,
        max_attempts=2,
        retry_limit=1,
    )
    registry = AttemptRegistry.create(
        tmp_path / "attempt_registry.json",
        budget_contract=contract,
        exploration_registry=ExplorationRegistry.zero_budget_v1(),
    )
    registry.register_identity(_identity(stage="tiny", record_id="one"))

    with pytest.raises(HumanGateRequiredError) as overflow:
        registry.register_identity(_identity(stage="tiny", record_id="two"))
    assert overflow.value.state == "awaiting_human_budget_decision"
    assert "unique_budget_exceeded" in str(overflow.value)


def test_only_fold_replay_can_use_672_to_684_supplement(tmp_path: Path) -> None:
    contract = BudgetContract(
        stage_unique_limits={"formal": 1, "fold_replay": 1},
        normal_unique_identity_limit=1,
        supplemental_stage="fold_replay",
        max_unique_identities=2,
        max_attempts=4,
        retry_limit=1,
    )
    registry = AttemptRegistry.create(
        tmp_path / "attempt_registry.json",
        budget_contract=contract,
        exploration_registry=ExplorationRegistry.zero_budget_v1(),
    )
    registry.register_identity(_identity(stage="formal", record_id="one"))

    with pytest.raises(HumanGateRequiredError):
        registry.register_identity(_identity(stage="formal", record_id="two"))
    registry.register_identity(_identity(stage="fold_replay", record_id="fold"))
    assert registry.summary()["planned_unique_identity_count"] == 2


def test_attempt_registry_reopens_with_bound_contract_and_counts_cache(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attempt_registry.json"
    contract = BudgetContract.frozen_v1()
    exploration = ExplorationRegistry.zero_budget_v1()
    registry = AttemptRegistry.create(
        path,
        budget_contract=contract,
        exploration_registry=exploration,
    )
    identity = _identity()
    registry.register_identity(identity)
    cache_path = tmp_path / "cache_receipt.json"
    cache_path.write_text(
        json.dumps(
            {
                "identity_sha256": identity.sha256,
                "result_sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    registry.record_cache_hit(
        identity,
        evidence=CacheEvidence.from_path(cache_path),
    )

    reopened = AttemptRegistry.open(
        path,
        budget_contract=contract,
        exploration_registry=exploration,
    )

    assert reopened.summary() == {
        "logical_task_count": 1,
        "planned_unique_identity_count": 1,
        "actual_unique_run_count": 0,
        "cache_hit_count": 1,
        "failed_attempt_count": 0,
        "retry_count": 0,
    }


def test_cache_evidence_must_bind_complete_identity(tmp_path: Path) -> None:
    path = tmp_path / "attempt_registry.json"
    registry = AttemptRegistry.create(
        path,
        budget_contract=BudgetContract.frozen_v1(),
        exploration_registry=ExplorationRegistry.zero_budget_v1(),
    )
    identity = _identity()
    registry.register_identity(identity)
    cache_path = tmp_path / "cache_receipt.json"
    cache_path.write_text(
        json.dumps(
            {
                "identity_sha256": "0" * 64,
                "result_sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GovernanceError, match="cache_identity_mismatch"):
        registry.record_cache_hit(
            identity,
            evidence=CacheEvidence.from_path(cache_path),
        )


def test_attempt_registry_refuses_contract_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "attempt_registry.json"
    AttemptRegistry.create(
        path,
        budget_contract=BudgetContract.frozen_v1(),
        exploration_registry=ExplorationRegistry.zero_budget_v1(),
    )
    incompatible = BudgetContract(
        stage_unique_limits={"other": 1},
        max_unique_identities=1,
        max_attempts=2,
        retry_limit=1,
    )

    with pytest.raises(GovernanceError, match="budget_contract_mismatch"):
        AttemptRegistry.open(
            path,
            budget_contract=incompatible,
            exploration_registry=ExplorationRegistry.zero_budget_v1(),
        )


def test_attempt_registry_refuses_tampered_state_machine(tmp_path: Path) -> None:
    path = tmp_path / "attempt_registry.json"
    contract = BudgetContract.frozen_v1()
    exploration = ExplorationRegistry.zero_budget_v1()
    registry = AttemptRegistry.create(
        path,
        budget_contract=contract,
        exploration_registry=exploration,
    )
    identity = _identity()
    registry.register_identity(identity)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][identity.sha256]["attempts"] = [
        {
            "attempt_number": 3,
            "token": "",
            "status": "succeeded",
            "failure_reason": None,
        }
    ]
    payload["summary"]["actual_unique_run_count"] = 1
    payload["summary"]["logical_task_count"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GovernanceError, match="invalid_attempt_state"):
        AttemptRegistry.open(
            path,
            budget_contract=contract,
            exploration_registry=exploration,
        )


def test_stale_registry_instance_reloads_before_mutation(tmp_path: Path) -> None:
    path = tmp_path / "attempt_registry.json"
    contract = BudgetContract.frozen_v1()
    exploration = ExplorationRegistry.zero_budget_v1()
    first = AttemptRegistry.create(
        path,
        budget_contract=contract,
        exploration_registry=exploration,
    )
    stale = AttemptRegistry.open(
        path,
        budget_contract=contract,
        exploration_registry=exploration,
    )

    first.register_identity(_identity(record_id="one"))
    stale.register_identity(_identity(record_id="two"))

    reopened = AttemptRegistry.open(
        path,
        budget_contract=contract,
        exploration_registry=exploration,
    )
    assert reopened.summary()["planned_unique_identity_count"] == 2
    assert stale.summary()["planned_unique_identity_count"] == 2


def test_stale_lock_file_does_not_block_registry_creation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attempt_registry.json"
    path.with_name(f".{path.name}.lock").write_text(
        "dead-process",
        encoding="ascii",
    )

    registry = AttemptRegistry.create(
        path,
        budget_contract=BudgetContract.frozen_v1(),
        exploration_registry=ExplorationRegistry.zero_budget_v1(),
    )

    assert registry.summary()["planned_unique_identity_count"] == 0


def test_failed_persist_does_not_mutate_in_memory_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = AttemptRegistry.create(
        tmp_path / "attempt_registry.json",
        budget_contract=BudgetContract.frozen_v1(),
        exploration_registry=ExplorationRegistry.zero_budget_v1(),
    )

    def fail_write(path: Path, payload: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(governance, "atomic_write_json", fail_write)
    with pytest.raises(OSError, match="disk full"):
        registry.register_identity(_identity())
    assert registry.summary()["planned_unique_identity_count"] == 0


def test_budget_stage_limits_are_immutable() -> None:
    contract = BudgetContract.frozen_v1()

    with pytest.raises(TypeError):
        contract.stage_unique_limits["historical_recovery_ab"] = 999
    with pytest.raises(FrozenInstanceError):
        contract.max_unique_identities = 999


def test_fold_read_barrier_denies_target_performance_fields(
    tmp_path: Path,
) -> None:
    training_path = tmp_path / "train.json"
    target_path = tmp_path / "target.json"
    training_path.write_text(
        json.dumps({"sample_id": "train", "mae": 1.0}),
        encoding="utf-8",
    )
    target_path.write_text(
        json.dumps({"sample_id": "target", "mae": 2.0}),
        encoding="utf-8",
    )
    barrier = FoldReadBarrier(
        DataRoleManifest(
            fold_id="run-fold-1",
            training_record_ids=("run1", "run2"),
            audit_target_record_id="run3",
            record_sources={
                "run1": RecordSource.from_path(training_path),
                "run2": RecordSource.from_path(training_path),
                "run3": RecordSource.from_path(target_path),
            },
        )
    )

    assert (
        barrier.read_json_fields(
            record_id="run1",
            fields=("sample_id", "mae"),
        )["mae"]
        == 1.0
    )
    assert barrier.read_json_fields(
        record_id="run3",
        fields=("sample_id",),
    ) == {"sample_id": "target"}
    with pytest.raises(GovernanceError, match="audit_target_field_denied"):
        barrier.read_json_fields(
            record_id="run3",
            fields=("mae",),
        )

    receipt = barrier.receipt()
    assert receipt["algorithm_level_holdout"] is False
    assert receipt["evidence_class"] == "development_replay_audit"
    assert receipt["accesses"][1]["fields"] == ["sample_id"]


def test_fold_read_barrier_binds_record_to_source_hash(tmp_path: Path) -> None:
    training_path = tmp_path / "train.json"
    target_path = tmp_path / "target.json"
    training_path.write_text('{"sample_id":"train","mae":1.0}', encoding="utf-8")
    target_path.write_text('{"sample_id":"target","mae":99.0}', encoding="utf-8")
    manifest = DataRoleManifest(
        fold_id="fold",
        training_record_ids=("train",),
        audit_target_record_id="target",
        record_sources={
            "train": RecordSource.from_path(training_path),
            "target": RecordSource.from_path(target_path),
        },
    )
    target_path.write_text('{"sample_id":"target","mae":98.0}', encoding="utf-8")
    barrier = FoldReadBarrier(manifest)

    with pytest.raises(GovernanceError, match="record_source_hash_mismatch"):
        barrier.read_json_fields(record_id="target", fields=("sample_id",))

    with pytest.raises(
        ValueError,
        match="target_identity_fields_cannot_expand_whitelist",
    ):
        FoldReadBarrier(manifest, target_identity_fields=("mae",))


def test_fold_read_barrier_hashes_and_parses_same_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_path = tmp_path / "train.json"
    target_path = tmp_path / "target.json"
    training_path.write_text('{"sample_id":"train","mae":1.0}', encoding="utf-8")
    target_path.write_text('{"sample_id":"target","mae":99.0}', encoding="utf-8")
    manifest = DataRoleManifest(
        fold_id="fold",
        training_record_ids=("train",),
        audit_target_record_id="target",
        record_sources={
            "train": RecordSource.from_path(training_path),
            "target": RecordSource.from_path(target_path),
        },
    )

    def swap_during_old_hash(path: Path) -> str:
        path.write_text('{"sample_id":"swapped","mae":0.0}', encoding="utf-8")
        return manifest.record_sources["target"].sha256

    monkeypatch.setattr(governance, "file_sha256", swap_during_old_hash)
    result = FoldReadBarrier(manifest).read_json_fields(
        record_id="target",
        fields=("sample_id",),
    )

    assert result == {"sample_id": "target"}


def test_independent_bo_requires_exact_machine_authorization() -> None:
    request = IndependentBORequest(
        solver_hash="a" * 64,
        search_space_hash="b" * 64,
        metric_contract_hash="c" * 64,
        seed_manifest_hash="d" * 64,
        unique_budget=120,
    )

    with pytest.raises(HumanGateRequiredError) as missing:
        validate_independent_bo_authorization(request, receipt=None)
    assert missing.value.state == "awaiting_human_independent_bo_decision"

    with pytest.raises(GovernanceError, match="authorization_identity_mismatch"):
        validate_independent_bo_authorization(
            request,
            receipt={
                "approved": True,
                "decision_state": "awaiting_human_independent_bo_decision",
                "solver_hash": "f" * 64,
                "search_space_hash": "b" * 64,
                "metric_contract_hash": "c" * 64,
                "seed_manifest_hash": "d" * 64,
                "unique_budget": 120,
                "approved_at": "2026-07-28T12:00:00+08:00",
                "approved_by": "user",
            },
        )


def test_independent_bo_accepts_only_complete_exact_receipt() -> None:
    request = IndependentBORequest(
        solver_hash="a" * 64,
        search_space_hash="b" * 64,
        metric_contract_hash="c" * 64,
        seed_manifest_hash="d" * 64,
        unique_budget=120,
    )
    receipt = {
        "approved": True,
        "decision_state": "awaiting_human_independent_bo_decision",
        "solver_hash": "a" * 64,
        "search_space_hash": "b" * 64,
        "metric_contract_hash": "c" * 64,
        "seed_manifest_hash": "d" * 64,
        "unique_budget": 120,
        "approved_at": "2026-07-28T12:00:00+08:00",
        "approved_by": "user",
    }

    assert (
        validate_independent_bo_authorization(
            request,
            receipt=receipt,
        )
        == receipt
    )

    incomplete = dict(receipt)
    incomplete.pop("seed_manifest_hash")
    with pytest.raises(GovernanceError, match="authorization_missing_fields"):
        validate_independent_bo_authorization(
            request,
            receipt=incomplete,
        )


@pytest.mark.parametrize(
    "state",
    [
        "awaiting_human_interaction_decision",
        "awaiting_human_budget_decision",
    ],
)
def test_general_human_gates_fail_closed(
    state: str,
) -> None:
    with pytest.raises(HumanGateRequiredError) as waiting:
        validate_human_gate(state=state, receipt=None)
    assert waiting.value.state == state

    with pytest.raises(GovernanceError, match="authorization_state_mismatch"):
        validate_human_gate(
            state=state,
            receipt={
                "approved": True,
                "decision_state": "awaiting_human_independent_bo_decision",
                "approved_at": "2026-07-28T12:00:00+08:00",
                "approved_by": "user",
            },
        )


def test_initialize_governance_writes_frozen_zero_run_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "governance"

    receipt = initialize_recovery_experiment_governance(
        output_dir=output,
        parent_experiment_id="lyx_recovery_filter_profile_v1",
    )

    assert receipt["status"] == "complete"
    assert receipt["planned_unique_identity_limit"] == 684
    assert receipt["worst_case_attempt_limit"] == 1368
    assert sorted(path.name for path in output.iterdir()) == [
        "attempt_registry.json",
        "budget_contract.json",
        "exploration_registry.json",
        "governance_receipt.json",
    ]
    attempts = json.loads((output / "attempt_registry.json").read_text(encoding="utf-8"))
    assert attempts["entries"] == {}
