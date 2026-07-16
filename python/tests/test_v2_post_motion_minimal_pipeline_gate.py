from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppg_hr.v2.post_motion_minimal_pipeline_gate import (
    build_bo_decision,
    build_fixed_validation_decision,
    require_fixed_validation_go,
    write_stopped_pipeline_decisions,
)


def test_ablation_no_go_blocks_fixed_hb24_validation() -> None:
    decision = build_fixed_validation_decision(
        {
            "verdict": "NO_GO",
            "reason": "all_runtime_candidates_failed_frozen_acceptance",
            "failed_gates_by_candidate": {
                "minimal_reanchor": "lost_existing_sub3_rescue"
            },
        }
    )

    assert decision["verdict"] == "NO_GO"
    assert decision["hb24_run_started"] is False
    assert decision["bo_allowed"] is False
    assert decision["reason"] == "upstream_relocation_ablation_no_go"


def test_fixed_no_go_proves_expected_bo_batch_was_not_created(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "20260717_minimal_handoff_hb24_lite_1x40"

    decision = build_bo_decision(
        {"verdict": "NO_GO", "bo_allowed": False},
        expected_bo_dir=expected,
    )

    assert decision["verdict"] == "NO_GO"
    assert decision["bo_batch_started"] is False
    assert decision["expected_output_absent"] is True
    assert decision["budget_consumed_iterations"] == 0


def test_stopped_bo_gate_fails_if_output_exists(tmp_path: Path) -> None:
    expected = tmp_path / "unexpected_batch"
    expected.mkdir()

    with pytest.raises(RuntimeError, match="must not exist"):
        build_bo_decision(
            {"verdict": "NO_GO", "bo_allowed": False},
            expected_bo_dir=expected,
        )


def test_gate_writer_persists_both_machine_decisions(tmp_path: Path) -> None:
    ablation = tmp_path / "ablation.json"
    ablation.write_text(
        json.dumps({"verdict": "NO_GO", "reason": "failed"}),
        encoding="utf-8",
    )
    output = tmp_path / "gate"

    result = write_stopped_pipeline_decisions(
        ablation_decision_path=ablation,
        output_dir=output,
        expected_bo_dir=tmp_path / "not_started_bo",
    )

    assert result["fixed_validation"]["hb24_run_started"] is False
    assert result["bo"]["bo_batch_started"] is False
    assert (output / "fixed_validation_decision.json").exists()
    assert (output / "bo_decision.json").exists()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"verdict": "PENDING"},
        {"verdict": "NO_GO", "selected_candidate": "minimal_reanchor"},
    ],
)
def test_ablation_gate_rejects_ambiguous_or_contradictory_input(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        build_fixed_validation_decision(payload)


def test_hb_lite_entry_rejects_fixed_validation_no_go(tmp_path: Path) -> None:
    decision = tmp_path / "fixed.json"
    decision.write_text(
        json.dumps({"verdict": "NO_GO", "bo_allowed": False}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="explicit fixed-validation GO"):
        require_fixed_validation_go(decision)


def test_hb_lite_entry_accepts_only_explicit_go_contract(tmp_path: Path) -> None:
    decision = tmp_path / "fixed.json"
    decision.write_text(
        json.dumps(
            {
                "verdict": "GO",
                "bo_allowed": True,
                "selected_candidate": "minimal_reanchor",
            }
        ),
        encoding="utf-8",
    )

    loaded = require_fixed_validation_go(decision)

    assert loaded["selected_candidate"] == "minimal_reanchor"
