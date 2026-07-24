from __future__ import annotations

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace

import pytest

from ppg_hr.v2.phase2_receipt import (
    FrozenReplayContext,
    FrozenReplayOutcome,
    ReceiptConflictError,
    ReceiptIntegrityError,
    ReplayIdentity,
    SelectionEvidence,
    SelectionReceiptMismatchError,
    freeze_selection,
    load_selection_receipt,
    replay_frozen_selection,
)


def _selection_evidence() -> SelectionEvidence:
    return SelectionEvidence(
        experiment_name="xiezi-fold-0-k1",
        code_commit="abc123",
        code_dirty=False,
        training_input_sha256s=("train-a", "train-b"),
        training_reference_sha256s=("ref-a", "ref-b"),
        space_name="legacy_full_v1",
        space_sha256="space-sha",
        metric_contract_version="lyx_bo_formal_metric_v1",
        study_identities=("seed-42", "seed-43", "seed-44", "fill"),
        budget={"lane_unique_budget": 40, "global_unique_budget": 120},
        selected_candidate_id="candidate-1",
        selected_requested_params={"fs_target": 25, "max_order": 8},
        selected_actual_params={"fs_target": 25, "max_order": 8},
        selected_fixed_params={"time_bias": 5.0, "smooth_win_len": 5},
        training_metrics={"worst_motion_mae_bpm": 7.0},
        neighborhood_evidence={"status": "complete", "radius": 1},
        candidate_history_sha256="history-sha",
        evidence_level="development_pilot",
    )


def _replay_identity() -> ReplayIdentity:
    return ReplayIdentity(
        test_record_id="xiezi-3",
        test_input_sha256="test-data",
        test_reference_sha256="test-ref",
        replay_config={"analysis_scope": "full"},
        reference_groups_order=("HF", "ACC"),
    )


def test_selection_receipt_hash_is_stable_and_contains_training_only(
    tmp_path,
) -> None:
    path = tmp_path / "selection_receipt.json"
    first = freeze_selection(path, _selection_evidence())
    second = freeze_selection(path, _selection_evidence())

    assert second == first
    assert len(first.selection_hash) == 64
    payload_text = path.read_text(encoding="utf-8")
    assert "test_record" not in payload_text
    assert first.evidence.training_input_sha256s == ("train-a", "train-b")


def test_training_selection_api_has_no_test_input_seam() -> None:
    evidence_fields = {field.name for field in fields(SelectionEvidence)}
    freeze_parameters = set(inspect.signature(freeze_selection).parameters)

    assert not any(name.startswith("test") for name in evidence_fields)
    assert not any(name.startswith("test") for name in freeze_parameters)


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
    persisted = load_selection_receipt(path)
    assert persisted == results[0]


def test_replay_requires_matching_frozen_hash_and_passes_only_frozen_params(
    tmp_path,
) -> None:
    receipt = freeze_selection(
        tmp_path / "selection_receipt.json",
        _selection_evidence(),
    )
    seen: list[FrozenReplayContext] = []

    def replay(context: FrozenReplayContext) -> FrozenReplayOutcome:
        seen.append(context)
        return FrozenReplayOutcome.success(
            metrics={"hf_final_mae_bpm": 5.0},
            artifact_sha256s={
                "hf": "hf-sha",
                "reset_fft": "reset-sha",
                "acc": "acc-sha",
            },
        )

    with pytest.raises(SelectionReceiptMismatchError):
        replay_frozen_selection(
            receipt_path=tmp_path / "selection_receipt.json",
            expected_selection_hash="0" * 64,
            replay_identity=_replay_identity(),
            replay_receipt_path=tmp_path / "replay.json",
            replay=replay,
        )
    assert seen == []

    replay_receipt = replay_frozen_selection(
        receipt_path=tmp_path / "selection_receipt.json",
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


def test_successful_or_invalid_replay_is_immutable_but_infrastructure_can_retry(
    tmp_path,
) -> None:
    receipt_path = tmp_path / "selection_receipt.json"
    receipt = freeze_selection(receipt_path, _selection_evidence())
    identity = _replay_identity()

    calls = 0

    def success(_: FrozenReplayContext) -> FrozenReplayOutcome:
        nonlocal calls
        calls += 1
        return FrozenReplayOutcome.success(
            metrics={"hf_final_mae_bpm": 5.0},
            artifact_sha256s={"hf": "hf-sha"},
        )

    replay_path = tmp_path / "success.json"
    first = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=identity,
        replay_receipt_path=replay_path,
        replay=success,
    )
    second = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=identity,
        replay_receipt_path=replay_path,
        replay=lambda _: pytest.fail("completed replay must be reused"),
    )
    assert second == first
    assert calls == 1

    infrastructure_path = tmp_path / "infrastructure.json"
    failed = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=identity,
        replay_receipt_path=infrastructure_path,
        replay=lambda _: FrozenReplayOutcome.infrastructure_failed(
            "solver_timeout"
        ),
    )
    assert failed.status == "infrastructure_failed"

    recovered = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=identity,
        replay_receipt_path=infrastructure_path,
        replay=success,
    )
    assert recovered.status == "success"

    invalid_path = tmp_path / "invalid.json"
    invalid = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=identity,
        replay_receipt_path=invalid_path,
        replay=lambda _: FrozenReplayOutcome.invalid("metric_contract_failed"),
    )
    assert invalid.status == "invalid"
    same_invalid = replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=identity,
        replay_receipt_path=invalid_path,
        replay=lambda _: pytest.fail("invalid replay is immutable"),
    )
    assert same_invalid == invalid


def test_replay_receipt_rejects_changed_test_identity(tmp_path) -> None:
    receipt_path = tmp_path / "selection_receipt.json"
    receipt = freeze_selection(receipt_path, _selection_evidence())
    replay_path = tmp_path / "replay.json"
    replay_frozen_selection(
        receipt_path=receipt_path,
        expected_selection_hash=receipt.selection_hash,
        replay_identity=_replay_identity(),
        replay_receipt_path=replay_path,
        replay=lambda _: FrozenReplayOutcome.success(
            metrics={"hf_final_mae_bpm": 5.0},
            artifact_sha256s={"hf": "hf-sha"},
        ),
    )

    changed_test = replace(_replay_identity(), test_input_sha256="changed")
    with pytest.raises(ReceiptConflictError):
        replay_frozen_selection(
            receipt_path=receipt_path,
            expected_selection_hash=receipt.selection_hash,
            replay_identity=changed_test,
            replay_receipt_path=replay_path,
            replay=lambda _: pytest.fail("mismatched replay must not run"),
        )


def test_changed_candidate_history_cannot_reuse_old_replay(tmp_path) -> None:
    first_path = tmp_path / "selection-1.json"
    first = freeze_selection(first_path, _selection_evidence())
    replay_path = tmp_path / "replay.json"
    replay_frozen_selection(
        receipt_path=first_path,
        expected_selection_hash=first.selection_hash,
        replay_identity=_replay_identity(),
        replay_receipt_path=replay_path,
        replay=lambda _: FrozenReplayOutcome.success(
            metrics={"hf_final_mae_bpm": 5.0},
            artifact_sha256s={"hf": "hf-sha"},
        ),
    )

    second_path = tmp_path / "selection-2.json"
    second = freeze_selection(
        second_path,
        replace(
            _selection_evidence(),
            candidate_history_sha256="changed-history",
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


def test_replay_receipt_tampering_fails_closed(tmp_path) -> None:
    selection_path = tmp_path / "selection.json"
    selection = freeze_selection(selection_path, _selection_evidence())
    replay_path = tmp_path / "replay.json"
    replay_frozen_selection(
        receipt_path=selection_path,
        expected_selection_hash=selection.selection_hash,
        replay_identity=_replay_identity(),
        replay_receipt_path=replay_path,
        replay=lambda _: FrozenReplayOutcome.success(
            metrics={"hf_final_mae_bpm": 5.0},
            artifact_sha256s={"hf": "hf-sha"},
        ),
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
