from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ppg_hr.v2.plot_filter_mechanism_decomposition import (
    build_filter_mechanism_decomposition_report_assets,
    collect_filter_mechanism_decomposition_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_DIR = (
    REPO_ROOT
    / "data"
    / "experiments"
    / "lyx_recovery_filter_profile"
    / "filter_mechanism_decomposition_execution_v1"
)
ASSET_DIR = (
    REPO_ROOT
    / "docs"
    / "reports"
    / "assets"
    / "2026-07-30-lyx-filter-mechanism-decomposition"
)


def test_committed_filter_mechanism_decomposition_builds_report_assets(
    tmp_path: Path,
) -> None:
    rows = collect_filter_mechanism_decomposition_rows(EXECUTION_DIR)

    assert len(rows) == 72
    assert {row["scene"] for row in rows} == {
        "jianpan",
        "xiezi",
        "run",
        "kaihe",
    }
    assert {row["lane"] for row in rows} == {
        "raw_bypass",
        "two_stage_zero_update",
        "rank1_only_adaptive",
        "rank2_only_adaptive",
        "ranked_cascade_adaptive",
        "reverse_cascade_adaptive",
    }
    assert sum(row["spectral_gate_pass"] for row in rows) == 68

    summary = build_filter_mechanism_decomposition_report_assets(
        EXECUTION_DIR,
        tmp_path,
    )

    assert summary["decision"] == "rank1_single_stage_mechanism_candidate"
    assert summary["record_count"] == 12
    assert summary["lane_complete_pass_counts"] == {
        "raw_bypass": 12,
        "two_stage_zero_update": 12,
        "rank1_only_adaptive": 12,
        "rank2_only_adaptive": 12,
        "ranked_cascade_adaptive": 10,
        "reverse_cascade_adaptive": 10,
    }
    assert summary["rank1_vs_forward_transitions"] == {
        "fail_to_pass": 2,
        "pass_to_fail": 0,
        "pass_to_pass": 10,
        "fail_to_fail": 0,
    }
    assert summary["forward_failure_record_ids"] == [
        "jianpan1_LYX_0708",
        "xiezi2_LYX_0708",
    ]
    assert summary["rank1_failure_record_ids"] == []
    assert summary["parameter_search_run_count"] == 0
    assert summary["independent_bo_run_count"] == 0
    for artifact_name in (
        "filter_mechanism_decomposition_record_metrics.csv",
        "filter_mechanism_decomposition_summary.json",
        "filter_mechanism_decomposition.png",
        "filter_mechanism_decomposition.svg",
        "filter_mechanism_decomposition.pdf",
    ):
        assert (tmp_path / artifact_name).is_file()
    for artifact_name in (
        "filter_mechanism_decomposition_record_metrics.csv",
        "filter_mechanism_decomposition_summary.json",
    ):
        assert (tmp_path / artifact_name).read_bytes() == (
            ASSET_DIR / artifact_name
        ).read_bytes()


def test_mechanism_report_loader_fails_closed_on_manifested_result_tamper(
    tmp_path: Path,
) -> None:
    execution_copy = tmp_path / "execution"
    shutil.copytree(EXECUTION_DIR, execution_copy)
    result_path = (
        execution_copy
        / "record_mechanism_audits"
        / "jianpan1_LYX_0708.json"
    )
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="mechanism_manifest_file_hash_mismatch",
    ):
        collect_filter_mechanism_decomposition_rows(execution_copy)


def test_mechanism_report_loader_fails_closed_on_unmanifested_result(
    tmp_path: Path,
) -> None:
    execution_copy = tmp_path / "execution"
    shutil.copytree(EXECUTION_DIR, execution_copy)
    source = (
        execution_copy
        / "record_mechanism_audits"
        / "jianpan1_LYX_0708.json"
    )
    shutil.copyfile(source, source.with_name("unmanifested.json"))

    with pytest.raises(
        ValueError,
        match="mechanism_manifest_result_file_set_mismatch",
    ):
        collect_filter_mechanism_decomposition_rows(execution_copy)
