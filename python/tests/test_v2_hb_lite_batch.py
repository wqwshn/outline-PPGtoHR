from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import ppg_hr.v2.hb_lite_batch as hb_lite_batch
from ppg_hr.v2.batch_pipeline import V2BatchRecord
from ppg_hr.v2.hb_lite_batch import (
    HB24_SAMPLE_STEMS,
    _audit_acc_comparison,
    _audit_artifact_sets,
    _audit_summary_samples,
    audit_hb_lite_batch,
    run_audited_hb_lite_batch,
)
from ppg_hr.v2.optimizer import V2BayesConfig


def test_audited_runner_rejects_duplicate_samples_before_running(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate samples"):
        run_audited_hb_lite_batch(
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            sample_stems=("bobi1", "bobi1"),
            fixed_validation_decision_path=tmp_path / "not-read.json",
        )


def test_audited_runner_rejects_non_frozen_bo_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires exactly 1x40"):
        run_audited_hb_lite_batch(
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            sample_stems=("bobi1",),
            fixed_validation_decision_path=tmp_path / "not-read.json",
            bayes_cfg=V2BayesConfig(
                max_iterations=20,
                num_seed_points=10,
                num_repeats=2,
                random_state=42,
            ),
        )


def test_artifact_audit_rejects_extra_sample_in_every_output_set(tmp_path: Path) -> None:
    patterns = {
        "json": "-green-raw_bandpass-lms-full-HF-v2.json",
        "png": "-green-raw_bandpass-lms-full-HF-v2-hr.png",
        "csv_hr": "-green-raw_bandpass-lms-full-HF-v2-hr.csv",
        "csv_error": "-green-raw_bandpass-lms-full-HF-v2-error.csv",
        "csv_trace": "-green-raw_bandpass-lms-full-HF-v2-window-trace.csv",
        "csv_history": "-green-raw_bandpass-lms-full-HF-v2-history.csv",
    }
    for directory in (tmp_path / "json", tmp_path / "png", tmp_path / "csv"):
        directory.mkdir()
    for sample in ("bobi1_HB_0711", "extra_HB_0711"):
        (tmp_path / "json" / f"{sample}{patterns['json']}").write_text("{}")
        (tmp_path / "png" / f"{sample}{patterns['png']}").write_bytes(b"png")
        for key in ("csv_hr", "csv_error", "csv_trace", "csv_history"):
            (tmp_path / "csv" / f"{sample}{patterns[key]}").write_text("x")

    failures: list[str] = []
    _audit_artifact_sets(
        output_dir=tmp_path,
        requested_samples=["bobi1"],
        failures=failures,
    )

    assert len(failures) == 6
    assert all("artifact_multiset_mismatch" in failure for failure in failures)


def test_direct_audit_rejects_non_frozen_bo_budget(tmp_path: Path) -> None:
    audit = audit_hb_lite_batch(
        records=[],
        requested_samples=(),
        bayes_cfg=V2BayesConfig(max_iterations=20, num_repeats=2),
        output_dir=tmp_path,
    )

    assert audit["status"] == "fail"
    assert "non_frozen_bayes_config" in audit["failures"]


def test_summary_audit_rejects_duplicate_or_extra_rows(tmp_path: Path) -> None:
    summary = tmp_path / "v2_batch_summary.csv"
    summary.write_text(
        "sample,status\n"
        "bobi1_HB_0711.csv,ok\n"
        "bobi1_HB_0711.csv,ok\n"
        "extra_HB_0711.csv,ok\n",
        encoding="utf-8-sig",
    )
    failures: list[str] = []

    _audit_summary_samples(
        summary_path=summary,
        requested_samples=["bobi1"],
        failures=failures,
    )

    assert failures == [
        "summary_sample_multiset_mismatch: "
        "expected=['bobi1'], actual=['bobi1', 'bobi1', 'extra']"
    ]


def test_audited_runner_keeps_hf_primary_and_enables_acc_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = tmp_path / "fixed.json"
    decision.write_text(
        '{"verdict":"GO","bo_allowed":true,'
        '"selected_candidate":"minimal_none",'
        '"selected_relocation_mode":"none"}',
        encoding="utf-8",
    )
    output = tmp_path / "out"
    output.mkdir()
    captured: dict[str, Any] = {}

    def fake_pipeline(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"records": [], "output_dir": output}

    monkeypatch.setattr(hb_lite_batch, "run_v2_batch_pipeline", fake_pipeline)
    monkeypatch.setattr(
        hb_lite_batch,
        "audit_hb_lite_batch",
        lambda **_kwargs: {"status": "pass", "failures": []},
    )

    run_audited_hb_lite_batch(
        input_dir=tmp_path,
        output_dir=output,
        sample_stems=HB24_SAMPLE_STEMS,
        fixed_validation_decision_path=decision,
    )

    assert captured["reference_groups_order"] == ("HF",)
    assert captured["comparison_groups"] == (("ACC",),)


def test_audited_runner_rejects_incomplete_hb24_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact HB24 manifest"):
        run_audited_hb_lite_batch(
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            sample_stems=("bobi1",),
            fixed_validation_decision_path=tmp_path / "not-read.json",
        )


def test_acc_comparison_audit_fails_closed_when_curve_is_missing(
    tmp_path: Path,
) -> None:
    hr_csv = tmp_path / "hr.csv"
    hr_csv.write_text("time_s,final_bpm\n0,80\n", encoding="utf-8")
    error_csv = tmp_path / "error.csv"
    error_csv.write_text(
        "method,total_aae\nLMS+H,1.0\n",
        encoding="utf-8",
    )
    record = V2BatchRecord(
        sample="bobi1_HB_0711.csv",
        ppg_mode="green",
        ppg_input_transform="raw_bandpass",
        adaptive_filter="lms",
        analysis_scope="full",
        reference_order_key="HF",
        qc_status="good",
        report_path=tmp_path / "report.json",
        best_error=1.0,
        hr_csv=hr_csv,
        error_csv=error_csv,
    )
    failures: list[str] = []

    result = _audit_acc_comparison(record, prefix="bobi1", failures=failures)

    assert failures == [
        "bobi1:missing_acc_timeline_column",
        "bobi1:missing_acc_metrics",
    ]
    assert result["timeline_finite_windows"] == 0


def test_acc_comparison_audit_accepts_timeline_and_metrics(tmp_path: Path) -> None:
    hr_csv = tmp_path / "hr.csv"
    hr_csv.write_text(
        "time_s,final_bpm,LMS+A_bpm\n0,80,79\n",
        encoding="utf-8",
    )
    error_csv = tmp_path / "error.csv"
    error_csv.write_text(
        "method,total_aae\nLMS+A,1.5\n",
        encoding="utf-8",
    )
    record = V2BatchRecord(
        sample="bobi1_HB_0711.csv",
        ppg_mode="green",
        ppg_input_transform="raw_bandpass",
        adaptive_filter="lms",
        analysis_scope="full",
        reference_order_key="HF",
        qc_status="good",
        report_path=tmp_path / "report.json",
        best_error=1.0,
        hr_csv=hr_csv,
        error_csv=error_csv,
    )
    failures: list[str] = []

    result = _audit_acc_comparison(record, prefix="bobi1", failures=failures)

    assert failures == []
    assert result["timeline_finite_windows"] == 1
    assert result["metric_rows"] == 1
