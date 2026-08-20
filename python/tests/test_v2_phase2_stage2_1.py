from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import ppg_hr.v2.phase2_stage2_1 as stage2_1
from ppg_hr.v2.phase2_stage2_1 import (
    FrozenIndependentRecord,
    Stage21GateResult,
    evaluate_stage2_1_acceptance,
    load_frozen_independent_records,
)


def _row(
    sample_id: str,
    scene: str,
    *,
    historical_classic: float,
    legacy_reliable: float,
    legacy_classic: float,
    physical_reliable: float,
    physical_classic: float,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "scene": scene,
        "historical_classic_motion_mae_bpm": historical_classic,
        "legacy_reliable_motion_mae_bpm": legacy_reliable,
        "legacy_classic_motion_mae_bpm": legacy_classic,
        "physical_reliable_motion_mae_bpm": physical_reliable,
        "physical_classic_motion_mae_bpm": physical_classic,
    }


def test_stage2_1_acceptance_applies_each_metric_column_independently() -> None:
    rows = [
        _row(
            "run1",
            "run",
            historical_classic=4.0,
            legacy_reliable=4.0,
            legacy_classic=4.0,
            physical_reliable=4.4,
            physical_classic=4.4,
        ),
        _row(
            "run2",
            "run",
            historical_classic=5.0,
            legacy_reliable=5.0,
            legacy_classic=5.0,
            physical_reliable=5.3,
            physical_classic=5.3,
        ),
        _row(
            "walk1",
            "walk",
            historical_classic=6.0,
            legacy_reliable=6.0,
            legacy_classic=6.0,
            physical_reliable=6.2,
            physical_classic=6.2,
        ),
    ]

    decision = evaluate_stage2_1_acceptance(rows)

    assert decision.passed is True
    assert {gate.comparison for gate in decision.gates} == {
        "physical_vs_historical_classic",
        "physical_vs_legacy_reliable",
        "physical_vs_legacy_classic",
    }
    assert all(isinstance(gate, Stage21GateResult) for gate in decision.gates)
    assert all(gate.passed for gate in decision.gates)


def test_stage2_1_acceptance_fails_closed_on_one_record_delta() -> None:
    rows = [
        _row(
            "run1",
            "run",
            historical_classic=4.0,
            legacy_reliable=4.0,
            legacy_classic=4.0,
            physical_reliable=6.1,
            physical_classic=4.0,
        ),
        _row(
            "run2",
            "run",
            historical_classic=4.0,
            legacy_reliable=4.0,
            legacy_classic=4.0,
            physical_reliable=4.0,
            physical_classic=4.0,
        ),
    ]

    decision = evaluate_stage2_1_acceptance(rows)

    failed = [
        gate
        for gate in decision.gates
        if not gate.passed
        and gate.comparison == "physical_vs_legacy_reliable"
    ]
    assert decision.passed is False
    assert any(gate.gate == "max_record_delta_bpm" for gate in failed)
    assert decision.stage2_2_authorized is False


def test_stage2_1_acceptance_detects_new_disaster_and_scene_regression() -> None:
    rows = [
        _row(
            "xiezi1",
            "xiezi",
            historical_classic=5.0,
            legacy_reliable=5.0,
            legacy_classic=5.0,
            physical_reliable=10.0,
            physical_classic=10.0,
        ),
        _row(
            "xiezi2",
            "xiezi",
            historical_classic=6.0,
            legacy_reliable=6.0,
            legacy_classic=6.0,
            physical_reliable=8.0,
            physical_classic=8.0,
        ),
    ]

    decision = evaluate_stage2_1_acceptance(rows)

    assert decision.passed is False
    assert any(
        gate.gate == "new_disaster_count"
        and gate.comparison == "physical_vs_historical_classic"
        and not gate.passed
        for gate in decision.gates
    )
    assert any(
        gate.gate == "scene_mean_delta_bpm"
        and gate.scope == "xiezi"
        and not gate.passed
        for gate in decision.gates
    )


def test_stage2_1_acceptance_rejects_nonfinite_or_missing_values() -> None:
    row = _row(
        "run1",
        "run",
        historical_classic=4.0,
        legacy_reliable=4.0,
        legacy_classic=4.0,
        physical_reliable=4.0,
        physical_classic=4.0,
    )
    row["physical_reliable_motion_mae_bpm"] = float("nan")

    with pytest.raises(ValueError, match="有限数"):
        evaluate_stage2_1_acceptance([row])


def test_load_frozen_independent_records_requires_passed_authorization(
    tmp_path: Path,
) -> None:
    report = tmp_path / "sample.json"
    error = tmp_path / "sample-error.csv"
    data = tmp_path / "sample.csv"
    reference = tmp_path / "sample-ref.csv"
    for path in (report, error, data, reference):
        path.write_text(path.name, encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "status": "passed",
                "stage2_1_authorized": True,
                "git": {"head": "abc123"},
                "checks": [
                    {
                        "name": "frozen_lyx_record_identities",
                        "status": "passed",
                        "details": [
                            {
                                "sample": "run1",
                                "scene": "run",
                                "files": {
                                    "data": {"path": str(data)},
                                    "reference": {"path": str(reference)},
                                    "historical_report": {"path": str(report)},
                                    "historical_error_csv": {"path": str(error)},
                                },
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    records = load_frozen_independent_records(
        preflight,
        expected_git_commit="abc123",
        expected_record_count=1,
    )

    assert records[0].sample_id == "run1"
    assert records[0].scene == "run"
    assert records[0].historical_report_path == report.resolve()


def test_load_frozen_independent_records_rejects_commit_or_count_mismatch(
    tmp_path: Path,
) -> None:
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "status": "passed",
                "stage2_1_authorized": True,
                "git": {"head": "frozen"},
                "checks": [
                    {
                        "name": "frozen_lyx_record_identities",
                        "status": "passed",
                        "details": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="commit"):
        load_frozen_independent_records(
            preflight,
            expected_git_commit="other",
            expected_record_count=0,
        )
    with pytest.raises(ValueError, match="记录数量"):
        load_frozen_independent_records(
            preflight,
            expected_git_commit="frozen",
            expected_record_count=24,
        )


def test_actual_git_state_rejects_wrong_head_and_dirty_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        stage2_1,
        "_git_output",
        lambda _root, *args: (
            "actual\n" if args == ("rev-parse", "HEAD") else ""
        ),
    )
    with pytest.raises(ValueError, match="实际 HEAD"):
        stage2_1._validate_actual_git_state(
            tmp_path,
            expected_git_commit="expected",
        )

    monkeypatch.setattr(
        stage2_1,
        "_git_output",
        lambda _root, *args: (
            "expected\n"
            if args == ("rev-parse", "HEAD")
            else " M tracked.py\n"
        ),
    )
    with pytest.raises(ValueError, match="干净工作树"):
        stage2_1._validate_actual_git_state(
            tmp_path,
            expected_git_commit="expected",
        )


def test_completed_record_receipt_binds_core_artifact_hashes(
    tmp_path: Path,
) -> None:
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    inputs = {}
    for name in ("data", "reference", "historical_report", "historical_error_csv"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        inputs[name] = path
    plots = {}
    for name in ("historical_plot", "legacy_plot", "physical_plot"):
        path = tmp_path / f"{name}.png"
        path.write_bytes(name.encode())
        plots[name] = path
    core = tmp_path / "candidate_history.csv"
    core.write_text("candidate_id\nc1\n", encoding="utf-8")
    record = FrozenIndependentRecord(
        sample_id="run1",
        scene="run",
        data_path=inputs["data"],
        reference_path=inputs["reference"],
        historical_report_path=inputs["historical_report"],
        historical_error_csv=inputs["historical_error_csv"],
    )
    receipt = tmp_path / "record_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "sample_id": "run1",
                "scene": "run",
                "git_commit": "commit",
                "input_sha256": {
                    name: digest(path) for name, path in inputs.items()
                },
                "artifacts": {
                    "candidate_history": {
                        "path": str(core),
                        "sha256": digest(core),
                    }
                },
                "record_metric": {
                    name: str(path) for name, path in plots.items()
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        stage2_1._load_completed_record_receipt(
            receipt,
            record=record,
            git_commit="commit",
        )
        is not None
    )
    core.write_text("candidate_id\nchanged\n", encoding="utf-8")
    assert (
        stage2_1._load_completed_record_receipt(
            receipt,
            record=record,
            git_commit="commit",
        )
        is None
    )


def test_failure_classification_does_not_guess_infrastructure_from_text() -> None:
    assert (
        stage2_1._classify_stage2_1_exception(
            stage2_1.Stage21AuditError(
                "study_state_mismatch",
                "记录身份错配",
            )
        )
        == "study_state_mismatch"
    )


def test_batch_fail_closes_startup_and_terminal_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    formal_root = tmp_path / "formal"
    formal_root.mkdir()
    (formal_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "phase2_run_manifest_v1",
                "status": "preflight_passed",
                "stage2_2_authorized": False,
            }
        ),
        encoding="utf-8",
    )

    def fail(_config):
        raise stage2_1.Stage21AuditError(
            "preflight_failed",
            "冻结输入错误",
        )

    monkeypatch.setattr(stage2_1, "_run_stage2_1_batch_inner", fail)
    with pytest.raises(stage2_1.Stage21AuditError, match="冻结输入错误"):
        stage2_1.run_stage2_1_batch(
            stage2_1.Stage21BatchConfig(
                formal_root=formal_root,
                git_commit="commit",
            )
        )

    failure = json.loads(
        (formal_root / "s21" / "stage2_1_failed.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (formal_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert failure["failure_classification"] == "preflight_failed"
    assert manifest["current_stage"] == "stage_2_1_failed"
    assert manifest["stage2_2_authorized"] is False
    assert (
        stage2_1._classify_stage2_1_exception(
            stage2_1.IndependentMethodIdentityMismatchError(
                "经典心率图缺少必需方法曲线"
            )
        )
        == "method_identity_mismatch"
    )
    assert (
        stage2_1._classify_stage2_1_exception(ValueError("未知算法错误"))
        == "study_state_mismatch"
    )
