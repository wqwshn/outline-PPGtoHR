from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppg_hr.v2.plot_lyx_short_circuit_final import (
    build_lyx_short_circuit_final_report_assets,
    collect_lyx_short_circuit_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "data" / "experiments" / "lyx_recovery_filter_profile"
SHORT_EXECUTION_DIR = EXPERIMENT_ROOT / "recovery_short_circuit_selector_replay_execution_v2"
CELL_ROOT = REPO_ROOT / "data" / "experiments" / "ribo_exec_v1" / "cells"
BASELINE_SUMMARY = EXPERIMENT_ROOT / "independent_bo_baseline_v1" / "summary.json"
METRIC_CONTRACT = EXPERIMENT_ROOT / "recovery_independent_bo_v1" / "metric_contract.json"
ASSET_DIR = (
    REPO_ROOT / "docs" / "reports" / "assets" / "2026-08-01-lyx-bo-space-generalization-final"
)


def test_committed_short_circuit_result_builds_final_report_assets(
    tmp_path: Path,
) -> None:
    rows, metadata = collect_lyx_short_circuit_rows(
        SHORT_EXECUTION_DIR,
        CELL_ROOT,
        BASELINE_SUMMARY,
        METRIC_CONTRACT,
    )

    assert len(rows) == 3
    assert all(row["eligible_candidate_count"] == 0 for row in rows)
    assert all(row["unique_candidate_count"] == 150 for row in rows)
    assert metadata["completed_cell_count"] == 11
    assert metadata["zero_run_repair_receipt_count"] == 11
    assert metadata["skipped_cell_count"] == 25
    assert metadata["completed_identity_count"] == 1650
    assert metadata["avoided_identity_count"] == 3750

    summary = build_lyx_short_circuit_final_report_assets(
        SHORT_EXECUTION_DIR,
        CELL_ROOT,
        BASELINE_SUMMARY,
        METRIC_CONTRACT,
        tmp_path,
    )

    assert summary["status"] == "no_recovery_survivor"
    assert summary["gate_b_executed"] is False
    assert summary["shared_parameter_mechanism_validated"] is False
    assert summary["gate_b_skip_reason"] == "gate_a_no_fs25_survivor"
    for artifact_name in (
        "lyx_short_circuit_final_metrics.csv",
        "lyx_short_circuit_final_summary.json",
        "lyx_short_circuit_final.png",
        "lyx_short_circuit_final.svg",
        "lyx_short_circuit_final.pdf",
        "_preview.png",
        "_grayscale.png",
    ):
        assert (tmp_path / artifact_name).is_file()
    for artifact_name in (
        "lyx_short_circuit_final_metrics.csv",
        "lyx_short_circuit_final_summary.json",
    ):
        assert (tmp_path / artifact_name).read_bytes() == (ASSET_DIR / artifact_name).read_bytes()


def test_short_circuit_report_loader_fails_closed_on_gate_tamper(
    tmp_path: Path,
) -> None:
    execution_copy = tmp_path / "execution"
    execution_copy.mkdir()
    source = SHORT_EXECUTION_DIR / "gate_a_completion.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["status"] = "survivor"
    (execution_copy / "gate_a_completion.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="embedded_hash_mismatch:lyx_short_circuit_gate_a_completion",
    ):
        collect_lyx_short_circuit_rows(
            execution_copy,
            CELL_ROOT,
            BASELINE_SUMMARY,
            METRIC_CONTRACT,
        )
