from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppg_hr.v2.phase2_stage2_1 import (
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
