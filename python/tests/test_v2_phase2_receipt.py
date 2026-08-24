from __future__ import annotations

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace

import pytest

from ppg_hr.v2.phase2_receipt import (
    FrozenReplayContext,
    FrozenReplayOutcome,
    NeighborhoodEvidence,
    ReceiptConflictError,
    ReceiptIntegrityError,
    RecordIdentity,
    ReplayAlreadyRunningError,
    ReplayIdentity,
    ReplayInfrastructureError,
    SearchBudgetEvidence,
    SelectionEvidence,
    SelectionReceiptMismatchError,
    TrainingMetricEvidence,
    freeze_selection,
    load_selection_receipt,
    replay_frozen_selection,
)


def _sha(character: str) -> str:
    return character * 64


def _record(record_id: str, character: str) -> RecordIdentity:
    return RecordIdentity(
        record_id=record_id,
        data_path=f"D:/data/{record_id}.csv",
        data_sha256=_sha(character),
        reference_path=f"D:/reference/{record_id}.csv",
        reference_sha256=_sha(character.upper().lower()),
    )


def _selection_evidence() -> SelectionEvidence:
    return SelectionEvidence(
        experiment_name="xiezi-fold-0-k1",
        arm="K1",
        scene="xiezi",
        fold=0,
        code_commit="abc123",
        code_dirty=False,
        training_records=(
            _record("xiezi-1", "a"),
            _record("xiezi-2", "b"),
        ),
        heldout_record=_record("xiezi-3", "c"),
        space_name="legacy_full_v1",
        space_sha256=_sha("d"),
        metric_contract_version="lyx_bo_formal_metric_v1",
        study_identities=("seed-42", "seed-43", "seed-44", "fill"),
        budget=SearchBudgetEvidence(
            lane_unique_budget=40,
            requested_global_unique_budget=120,
            actual_global_unique_count=120,
            requested_neighborhood_budget=30,
            actual_neighborhood_count=24,
        ),
        selected_candidate_id="candidate-1",
        selected_requested_params={"fs_target": 25, "max_order": 8},
        selected_actual_params={"fs_target": 25, "max_order": 8},
        selected_fixed_params={"time_bias": 5.0, "smooth_win_len": 5},
        training_metrics=TrainingMetricEvidence(
            eligible=True,
            common_window_counts=(18, 20),
            common_window_sha256s=(_sha("e"), _sha("f")),
            worst_train_mae_bpm=7.0,
            mean_train_mae_bpm=6.5,
            nonharm_deltas_bpm=(1.0, 0.5),
        ),
        neighborhood_evidence=NeighborhoodEvidence(
            status="complete",
            reviewed_neighbor_count=8,
            support_ratio=0.75,
            has_cliff=False,
            truncated_center_count=0,
        ),
        candidate_history_sha256=_sha("1"),
        evidence_level="development_reuse_pilot",
    )


def _replay_identity() -> ReplayIdentity:
    return ReplayIdentity(
        heldout_record=_selection_evidence().heldout_record,
        reference_groups_order=("HF", "ACC"),
    )


def _success_outcome() -> FrozenReplayOutcome:
    return FrozenReplayOutcome.success(
        metrics={"hf_final_mae_bpm": 5.0},
        artifact_sha256s={
            "hf": _sha("2"),
            "reset_fft": _sha("3"),
            "acc": _sha("4"),
        },
    )


def test_selection_receipt_hash_is_stable_and_freezes_fold_without_test_metrics(
    tmp_path,
) -> None:
    path = tmp_path / "selection_receipt.json"
    first = freeze_selection(path, _selection_evidence())
    second = freeze_selection(path, _selection_evidence())

    assert second == first
    assert len(first.selection_hash) == 64
    assert first.evidence.heldout_record.record_id == "xiezi-3"
    payload_text = path.read_text(encoding="utf-8")
    assert "heldout_record" in payload_text
    assert "test_metrics" not in payload_text
    assert first.evidence.evidence_level == "development_reuse_pilot"


def test_training_selection_api_has_no_test_result_seam() -> None:
    evidence_fields = {field.name for field in fields(SelectionEvidence)}
    freeze_parameters = set(inspect.signature(freeze_selection).parameters)

    assert "test_metrics" not in evidence_fields
    assert "test_result" not in evidence_fields
    assert not any(name.startswith("test") for name in freeze_parameters)
    assert "replay_config" not in {
        field.name for field in fields(ReplayIdentity)
    }


def test_selection_receipt_rejects_overwrite_and_tampering(tmp_path) -> None:
    path = tmp_path / "selection_receipt.json"
    freeze_selection(path, _selection_evidence())

    changed = replace(
        _selection_evidence(),
        selected_candidate_id="candidate-2",
    )
    with pytest.raises(ReceiptConflictError):
        freeze_selection(path, changed)

    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("candidate-1", "candidate-x"),
        encoding="utf-8",
    )
    with pytest.raises(ReceiptIntegrityError):
        load_selection_receipt(path)


def test_selection_receipt_strictly_rejects_wrong_json_types(tmp_path) -> None:
    path = tmp_path / "selection_receipt.json"
    freeze_selection(path, _selection_evidence())
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace('"code_dirty": false', '"code_dirty": "false"'),
        encoding="utf-8",
    )

    with pytest.raises(ReceiptIntegrityError, match="code_dirty"):
        load_selection_receipt(path)


def test_selection_receipt_rejects_malformed_sha_as_integrity_failure(
    tmp_path,
) -> None:
    path = tmp_path / "selection_receipt.json"
    freeze_selection(path, _selection_evidence())
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            _sha("d"),
            "not-a-sha",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReceiptIntegrityError):
        load_selection_receipt(path)


def test_selection_evidence_mappings_are_deeply_immutable() -> None:
    evidence = _selection_evidence()

    with pytest.raises(TypeError):
        evidence.selected_actual_params["fs_target"] = 100


def test_selection_rejects_heldout_alias_with_different_id_same_data() -> None:
    aliased = replace(
        _selection_evidence().training_records[0],
        record_id="different-name",
        data_path="D:/alias/same-content.csv",
    )

    with pytest.raises(ValueError, match="相同数据内容"):
        replace(_selection_evidence(), heldout_record=aliased)


def test_concurrent_selection_freeze_keeps_one_complete_immutable_receipt(
    tmp_path,
) -> None:
    path = tmp_path / "selection_receipt.json"
    barrier = threading.Barrier(2)

    def freeze(candidate_id: str):
        barrier.wait(timeout=5)
        return freeze_selection(
            path,
            replace(
                _selection_evidence(),
                selected_candidate_id=candidate_id,
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(freeze, candidate_id)
            for candidate_id in ("candidate-a", "candidate-b")
        ]
        results = []
        errors = []
        for future in futures:
            try:
                results.append(future.result(timeout=10))
            except ReceiptConflictError as exc:
                errors.append(exc)

    assert len(results) == 1
    assert len(errors) == 1
    assert load_selection_receipt(path) == results[0]


def test_replay_requires_matching_hash_and_predeclared_heldout_record(
    tmp_path,
) -> None:
    receipt_path = tmp_path / "selection_receipt.json"
    receipt = freeze_selection(receipt_path, _selection_evidence())
    seen: list[FrozenReplayContext] = []

    def replay(context: FrozenReplayContext) -> FrozenReplayOutcome:
        seen.append(context)
        return _success_outcome()

    with pytest.raises(SelectionReceiptMismatchError):
        replay_frozen_selection(
            receipt_path=receipt_path,
            expected_selection_hash="0" * 64,
            replay_identity=_replay_identity(),
            replay_receipt_path=tmp_path / "replay.json",
            replay=replay,
        )
    changed_test = replace(
        _replay_identity(),
        heldout_record=_record("xiezi-4", "9"),
    )
    with pytest.raises(SelectionReceiptMismatchError, match="留出记录"):
        replay_frozen_selection(
            receipt_path=receipt_path,
            expected_selection_hash=receipt.selection_hash,
            replay_identity=changed_test,
            replay_receipt_path=tmp_path / "replay.json",
            replay=replay,
        )
    assert seen == []

    replay_receipt = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=_replay_identity(),
        replay_receipt_path=tmp_path / "replay.json",
        replay=replay,
    )
    assert replay_receipt.status == "success"
    assert len(seen) == 1
    assert seen[0].candidate_id == "candidate-1"
    assert seen[0].actual_params == {"fs_target": 25, "max_order": 8}
    assert seen[0].reference_groups_order == ("HF", "ACC")


def test_only_explicit_infrastructure_exception_can_retry(tmp_path) -> None:
    receipt_path = tmp_path / "selection_receipt.json"
    receipt = freeze_selection(receipt_path, _selection_evidence())
    identity = _replay_identity()
    infrastructure_path = tmp_path / "infrastructure.json"

    failed = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=identity,
        replay_receipt_path=infrastructure_path,
        replay=lambda _: (_ for _ in ()).throw(
            ReplayInfrastructureError("solver_timeout")
        ),
    )
    assert failed.status == "infrastructure_failed"
    recovered = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=identity,
        replay_receipt_path=infrastructure_path,
        replay=lambda _: _success_outcome(),
    )
    assert recovered.status == "success"

    algorithm_path = tmp_path / "algorithm.json"
    invalid = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=identity,
        replay_receipt_path=algorithm_path,
        replay=lambda _: (_ for _ in ()).throw(
            RuntimeError("metric contract bug")
        ),
    )
    assert invalid.status == "invalid"
    assert invalid.failure_reason == "unclassified_replay_exception:RuntimeError"
    same_invalid = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=identity,
        replay_receipt_path=algorithm_path,
        replay=lambda _: pytest.fail("algorithm failure is immutable"),
    )
    assert same_invalid == invalid

    with pytest.raises(ValueError, match="未知基础设施"):
        ReplayInfrastructureError("metric_contract_failed")


def test_successful_or_declared_invalid_replay_is_immutable(tmp_path) -> None:
    receipt_path = tmp_path / "selection_receipt.json"
    receipt = freeze_selection(receipt_path, _selection_evidence())

    replay_path = tmp_path / "success.json"
    first = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=_replay_identity(),
        replay_receipt_path=replay_path,
        replay=lambda _: _success_outcome(),
    )
    second = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=_replay_identity(),
        replay_receipt_path=replay_path,
        replay=lambda _: pytest.fail("completed replay must be reused"),
    )
    assert second == first

    invalid_path = tmp_path / "invalid.json"
    invalid = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=_replay_identity(),
        replay_receipt_path=invalid_path,
        replay=lambda _: FrozenReplayOutcome.invalid(
            "metric_contract_failed"
        ),
    )
    same_invalid = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=_replay_identity(),
        replay_receipt_path=invalid_path,
        replay=lambda _: pytest.fail("invalid replay is immutable"),
    )
    assert same_invalid == invalid

    with pytest.raises(ValueError, match="失败回放不得"):
        FrozenReplayOutcome(
            status="invalid",
            metrics={"hf_final_mae_bpm": 5.0},
            artifact_sha256s={},
            failure_reason="metric_contract_failed",
        )


def test_concurrent_replay_allows_only_one_solver_execution(tmp_path) -> None:
    receipt_path = tmp_path / "selection_receipt.json"
    receipt = freeze_selection(receipt_path, _selection_evidence())
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def blocking_replay(_: FrozenReplayContext) -> FrozenReplayOutcome:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5)
        return _success_outcome()

    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(
            replay_frozen_selection,
            receipt_path=receipt_path,
            expected_selection_hash=receipt.selection_hash,
            replay_identity=_replay_identity(),
            replay_receipt_path=tmp_path / "replay.json",
            replay=blocking_replay,
        )
        assert entered.wait(timeout=5)
        with pytest.raises(ReplayAlreadyRunningError):
            replay_frozen_selection(
                receipt_path=receipt_path,
                expected_selection_hash=receipt.selection_hash,
                replay_identity=_replay_identity(),
                replay_receipt_path=tmp_path / "replay.json",
                replay=lambda _: pytest.fail("second solver must not run"),
            )
        release.set()
        running.result(timeout=10)
    assert calls == 1


def test_changed_candidate_history_cannot_reuse_old_replay(tmp_path) -> None:
    first_path = tmp_path / "selection-1.json"
    first = freeze_selection(first_path, _selection_evidence())
    replay_path = tmp_path / "replay.json"
    replay_frozen_selection(
        receipt_path=first_path,
        expected_selection_hash=first.selection_hash,
        replay_identity=_replay_identity(),
        replay_receipt_path=replay_path,
        replay=lambda _: _success_outcome(),
    )

    second_path = tmp_path / "selection-2.json"
    second = freeze_selection(
        second_path,
        replace(
            _selection_evidence(),
            candidate_history_sha256=_sha("8"),
        ),
    )
    assert second.selection_hash != first.selection_hash
    with pytest.raises(ReceiptConflictError):
        replay_frozen_selection(
            receipt_path=second_path,
            expected_selection_hash=second.selection_hash,
            replay_identity=_replay_identity(),
            replay_receipt_path=replay_path,
            replay=lambda _: pytest.fail("old replay must not be reused"),
        )


def test_changed_predeclared_heldout_record_changes_selection_hash(
    tmp_path,
) -> None:
    first = freeze_selection(
        tmp_path / "selection-1.json",
        _selection_evidence(),
    )
    second = freeze_selection(
        tmp_path / "selection-2.json",
        replace(
            _selection_evidence(),
            heldout_record=_record("xiezi-4", "9"),
        ),
    )

    assert second.selection_hash != first.selection_hash


def test_replay_receipt_tampering_fails_closed(tmp_path) -> None:
    selection_path = tmp_path / "selection.json"
    selection = freeze_selection(selection_path, _selection_evidence())
    replay_path = tmp_path / "replay.json"
    replay_frozen_selection(
        receipt_path=selection_path,
        expected_selection_hash=selection.selection_hash,
        replay_identity=_replay_identity(),
        replay_receipt_path=replay_path,
        replay=lambda _: _success_outcome(),
    )
    replay_path.write_text(
        replay_path.read_text(encoding="utf-8").replace("5.0", "9.0"),
        encoding="utf-8",
    )

    with pytest.raises(ReceiptIntegrityError):
        replay_frozen_selection(
            receipt_path=selection_path,
            expected_selection_hash=selection.selection_hash,
            replay_identity=_replay_identity(),
            replay_receipt_path=replay_path,
            replay=lambda _: pytest.fail("tampered replay must not run"),
        )
