from __future__ import annotations

from pathlib import Path

import pytest

from ppg_hr.v2.hb_lite_batch import (
    _audit_artifact_sets,
    run_audited_hb_lite_batch,
)
from ppg_hr.v2.optimizer import V2BayesConfig


def test_audited_runner_rejects_duplicate_samples_before_running(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate samples"):
        run_audited_hb_lite_batch(
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            sample_stems=("bobi1", "bobi1"),
        )


def test_audited_runner_rejects_non_frozen_bo_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires exactly 1x40"):
        run_audited_hb_lite_batch(
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            sample_stems=("bobi1",),
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
