from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ppg_hr.v2.plot_p25_spectral_recheck import (
    build_p25_spectral_recheck_report_assets,
    collect_p25_spectral_recheck_rows,
)
from ppg_hr.v2.recovery_contracts import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_DIR = (
    REPO_ROOT
    / "data"
    / "experiments"
    / "lyx_recovery_filter_profile"
    / "p25_spectral_recheck_execution_v2"
)
ASSET_DIR = (
    REPO_ROOT
    / "docs"
    / "reports"
    / "assets"
    / "2026-07-30-lyx-p25-spectral-recheck"
)


def test_committed_p25_spectral_recheck_builds_report_assets(
    tmp_path: Path,
) -> None:
    rows = collect_p25_spectral_recheck_rows(EXECUTION_DIR)

    assert len(rows) == 36
    assert {row["scene"] for row in rows} == {
        "jianpan",
        "xiezi",
        "run",
        "kaihe",
    }
    assert all(row["stability_pass"] for row in rows)
    assert all(row["pulse_power_retention_pass"] for row in rows)
    assert all(row["visible_top3_rate_delta_pass"] for row in rows)
    assert all(row["residual_artifact_corr_delta_pass"] for row in rows)
    assert sum(row["hr_band_share_delta_pass"] for row in rows) == 18
    assert sum(row["prominence_db_delta_pass"] for row in rows) == 32

    summary = build_p25_spectral_recheck_report_assets(
        EXECUTION_DIR,
        tmp_path,
    )

    assert summary["record_result_count"] == 36
    assert summary["decision"] == "p25_failure_review_required"
    assert summary["complete_pass_profile_ids"] == []
    assert summary["overall_gate_pass_counts"] == {
        "complete_window_evidence_pass": 36,
        "hr_band_share_delta_pass": 18,
        "prominence_db_delta_pass": 32,
        "pulse_power_retention_pass": 36,
        "residual_artifact_corr_delta_pass": 36,
        "visible_top3_rate_delta_pass": 36,
    }
    assert {
        profile: values["complete_pass_count"]
        for profile, values in summary["profiles"].items()
    } == {
        "p25-short-low": 10,
        "p25-short-mid": 5,
        "p25-long-mid": 3,
    }
    assert (
        tmp_path / "p25_spectral_recheck_record_metrics.csv"
    ).is_file()
    assert (tmp_path / "p25_spectral_recheck_summary.json").is_file()
    assert (tmp_path / "p25_spectral_recheck.png").is_file()
    assert (tmp_path / "p25_spectral_recheck.svg").is_file()
    assert (tmp_path / "p25_spectral_recheck.pdf").is_file()
    for artifact_name in (
        "p25_spectral_recheck_record_metrics.csv",
        "p25_spectral_recheck_summary.json",
    ):
        assert (tmp_path / artifact_name).read_bytes() == (
            ASSET_DIR / artifact_name
        ).read_bytes()


@pytest.mark.parametrize(
    "artifact_name",
    ("decision_receipt.json", "result_manifest.json"),
)
def test_recheck_report_builder_binds_completion_artifact_file_hashes(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    execution_copy = tmp_path / "execution"
    shutil.copytree(EXECUTION_DIR, execution_copy)
    artifact_path = execution_copy / artifact_name
    artifact_path.write_text(
        artifact_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "p25_recheck_completion_artifact_hash_mismatch:"
            f"{artifact_name}"
        ),
    ):
        build_p25_spectral_recheck_report_assets(
            execution_copy,
            tmp_path / "report",
        )


def test_recheck_report_builder_binds_completion_to_decision_hash(
    tmp_path: Path,
) -> None:
    execution_copy = tmp_path / "execution"
    shutil.copytree(EXECUTION_DIR, execution_copy)
    completion_path = execution_copy / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["decision_sha256"] = "0" * 64
    completion.pop("completion_sha256")
    completion["completion_sha256"] = canonical_sha256(completion)
    completion_path.write_text(
        json.dumps(
            completion,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="p25_recheck_completion_decision_hash_mismatch",
    ):
        build_p25_spectral_recheck_report_assets(
            execution_copy,
            tmp_path / "report",
        )


def test_recheck_report_loader_fails_closed_on_manifested_result_tamper(
    tmp_path: Path,
) -> None:
    execution_copy = tmp_path / "execution"
    shutil.copytree(EXECUTION_DIR, execution_copy)
    result_path = (
        execution_copy
        / "profile_record_audits"
        / "p25-short-low"
        / "jianpan1_LYX_0708.json"
    )
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="p25_recheck_manifest_file_hash_mismatch",
    ):
        collect_p25_spectral_recheck_rows(execution_copy)


def test_recheck_report_loader_fails_closed_on_unmanifested_result(
    tmp_path: Path,
) -> None:
    execution_copy = tmp_path / "execution"
    shutil.copytree(EXECUTION_DIR, execution_copy)
    source = (
        execution_copy
        / "profile_record_audits"
        / "p25-short-low"
        / "jianpan1_LYX_0708.json"
    )
    shutil.copyfile(source, source.with_name("unmanifested.json"))

    with pytest.raises(
        ValueError,
        match="p25_recheck_manifest_result_file_set_mismatch",
    ):
        collect_p25_spectral_recheck_rows(execution_copy)
