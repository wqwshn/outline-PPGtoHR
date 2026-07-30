from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ppg_hr.v2.plot_p25_spectral_diagnostic import (
    build_p25_spectral_report_assets,
    collect_record_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_DIR = (
    REPO_ROOT
    / "data"
    / "experiments"
    / "lyx_recovery_filter_profile"
    / "p25_spectral_diagnostic_execution_v1"
)


def test_committed_p25_spectral_execution_builds_report_assets(
    tmp_path: Path,
) -> None:
    rows = collect_record_rows(EXECUTION_DIR)

    assert len(rows) == 36
    assert {row["scene"] for row in rows} == {
        "jianpan",
        "xiezi",
        "run",
        "kaihe",
    }
    assert not any(row["pulse_power_retention_pass"] for row in rows)
    assert not any(row["spectral_gate_pass"] for row in rows)

    summary = build_p25_spectral_report_assets(EXECUTION_DIR, tmp_path)

    assert summary["record_result_count"] == 36
    assert summary["decision"] == "spectral_metric_control_audit_required"
    assert summary["pulse_power_retention_overall"]["pass_count"] == 0
    assert summary["overall_gate_pass_counts"] == {
        "prominence_db_delta_pass": 32,
        "visible_top3_rate_delta_pass": 36,
        "hr_band_share_delta_pass": 18,
        "pulse_power_retention_pass": 0,
        "residual_artifact_corr_delta_pass": 36,
        "complete_window_evidence_pass": 36,
    }
    assert (tmp_path / "p25_spectral_record_metrics.csv").is_file()
    assert (tmp_path / "p25_spectral_summary.json").is_file()
    assert (tmp_path / "p25_spectral_diagnostic.png").is_file()
    assert (tmp_path / "p25_spectral_diagnostic.svg").is_file()
    assert (tmp_path / "p25_spectral_diagnostic.pdf").is_file()


def test_report_loader_fails_closed_on_manifested_audit_tamper(
    tmp_path: Path,
) -> None:
    execution_copy = tmp_path / "execution"
    shutil.copytree(EXECUTION_DIR, execution_copy)
    audit_path = (
        execution_copy
        / "spectral_audits"
        / "p25-short-low"
        / "jianpan1_LYX_0708.json"
    )
    audit_path.write_text(
        audit_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="p25_manifest_file_hash_mismatch"):
        collect_record_rows(execution_copy)


def test_report_loader_fails_closed_on_unmanifested_audit(
    tmp_path: Path,
) -> None:
    execution_copy = tmp_path / "execution"
    shutil.copytree(EXECUTION_DIR, execution_copy)
    source = (
        execution_copy
        / "spectral_audits"
        / "p25-short-low"
        / "jianpan1_LYX_0708.json"
    )
    shutil.copyfile(source, source.with_name("unmanifested.json"))

    with pytest.raises(
        ValueError,
        match="p25_manifest_materialized_file_set_mismatch",
    ):
        collect_record_rows(execution_copy)
