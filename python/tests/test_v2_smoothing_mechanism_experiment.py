from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.smoothing_mechanism_experiment import (
    DEFAULT_SMOOTH_DURATIONS_S,
    audit_method_identity,
    compute_smoothing_metrics,
    discover_smoothing_anchors,
    validate_smoothing_durations,
    write_human_decision_template,
)


def _write_report(
    path: Path,
    *,
    data_path: Path,
    ref_path: Path,
    smooth_win_len: int = 7,
    time_bias: float = 6.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "v2",
                "data_path": str(data_path),
                "ref_path": str(ref_path),
                "algorithm_preset": "lite",
                "adaptive_filter": "lms",
                "reference_groups_order": ["HF"],
                "time_bias": time_bias,
                "best_params": {
                    "smooth_win_len": smooth_win_len,
                    "time_bias": time_bias,
                },
                "err_stats": {
                    "fft_aae_bpm": 2.0,
                    "final_aae_bpm": 1.0,
                },
                "unreliable_windows": 0,
            }
        ),
        encoding="utf-8",
    )


def _write_error_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "method,total_aae,rest_aae,motion_aae\n"
        "reset FFT,2,2,2\n"
        "LMS+H,1,1,1\n",
        encoding="utf-8",
    )


def _synthetic_anchor_roots(tmp_path: Path) -> tuple[Path, Path]:
    data_path = tmp_path / "xiezi1_LYX_0708.csv"
    ref_path = tmp_path / "xiezi1_LYX_0708_HR_ref.csv"
    data_path.write_text("data", encoding="utf-8")
    ref_path.write_text("ref", encoding="utf-8")

    independent = tmp_path / "independent"
    independent_report = (
        independent
        / "json"
        / "xiezi1_LYX_0708-green-raw_bandpass-lms-full-HF-v2.json"
    )
    _write_report(
        independent_report,
        data_path=data_path,
        ref_path=ref_path,
        smooth_win_len=9,
    )
    _write_error_csv(
        independent
        / "csv"
        / "xiezi1_LYX_0708-green-raw_bandpass-lms-full-HF-v2-error.csv"
    )

    generalization = tmp_path / "generalization"
    shared_report = generalization / "kfold" / "fold_01" / "json" / "test_xz1-v2.json"
    shared_error = generalization / "kfold" / "fold_01" / "csv" / "test_xz1-v2-error.csv"
    _write_report(
        shared_report,
        data_path=data_path,
        ref_path=ref_path,
        smooth_win_len=5,
    )
    _write_error_csv(shared_error)
    generalization.mkdir(parents=True, exist_ok=True)
    with (generalization / "v2_generalization_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "motion_type",
                "evaluation_mode",
                "fold_id",
                "split",
                "sample",
                "fft_aae_bpm",
                "final_aae_bpm",
                "report_path",
                "error_csv",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "motion_type": "xiezi",
                "evaluation_mode": "k_fold_holdout",
                "fold_id": "fold_01",
                "split": "test",
                "sample": data_path.name,
                "fft_aae_bpm": "2",
                "final_aae_bpm": "1",
                "report_path": str(shared_report),
                "error_csv": str(shared_error),
                "status": "ok",
            }
        )
    return independent, generalization


def test_discover_smoothing_anchors_pairs_independent_and_shared_reports(
    tmp_path: Path,
) -> None:
    independent, generalization = _synthetic_anchor_roots(tmp_path)

    anchors = discover_smoothing_anchors(
        independent,
        generalization,
        expected_record_count=1,
    )

    assert [(row.anchor_type, row.sample) for row in anchors] == [
        ("independent_bo", "xiezi1_LYX_0708"),
        ("shared_holdout", "xiezi1_LYX_0708"),
    ]
    assert anchors[0].scene == "xiezi"
    assert anchors[0].source_smooth_win_len == 9
    assert anchors[1].source_smooth_win_len == 5
    assert anchors[1].fold_id == "fold_01"
    assert all(row.data_path.is_absolute() for row in anchors)


def test_discover_smoothing_anchors_fails_closed_on_sample_mismatch(
    tmp_path: Path,
) -> None:
    independent, generalization = _synthetic_anchor_roots(tmp_path)
    summary = generalization / "v2_generalization_summary.csv"
    text = summary.read_text(encoding="utf-8")
    summary.write_text(
        text.replace("xiezi1_LYX_0708.csv", "xiezi2_LYX_0708.csv"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="样本.*不一致"):
        discover_smoothing_anchors(
            independent,
            generalization,
            expected_record_count=1,
        )


def test_identity_audit_separates_method_identity_from_known_mask_difference(
    tmp_path: Path,
) -> None:
    independent, generalization = _synthetic_anchor_roots(tmp_path)
    report = next((independent / "json").glob("*.json"))
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["err_stats"] = {"fft_aae_bpm": 8.0, "final_aae_bpm": 9.0}
    payload["unreliable_windows"] = 3
    report.write_text(json.dumps(payload), encoding="utf-8")

    anchors = discover_smoothing_anchors(
        independent,
        generalization,
        expected_record_count=1,
    )
    row = next(
        item
        for item in audit_method_identity(anchors)
        if item["anchor_type"] == "independent_bo"
    )

    assert row["identity_ok"] is True
    assert row["numeric_reconciled"] is False
    assert row["numeric_difference_reason"].startswith(
        "solver_summary_excludes_unreliable_windows"
    )
    assert row["audit_ok"] is True


def test_compute_smoothing_metrics_uses_time_bias_and_strict_motion_mask() -> None:
    hr = np.asarray(
        [
            [0.0, 60.0, 71.0, 70.0, 0.0, 1.0],
            [1.0, 70.0, 79.0, 80.0, 1.0, 1.0],
            [2.0, 80.0, 90.0, 90.0, 1.0, 1.0],
        ]
    )

    metrics = compute_smoothing_metrics(
        hr,
        time_bias=1.0,
        reference_bounds=(0.0, 2.0),
        reliable_mask=np.asarray([True, False, True]),
    )

    assert metrics["full_final_mae_bpm"] == pytest.approx(0.0)
    assert metrics["motion_final_mae_bpm"] == pytest.approx(0.0)
    assert metrics["full_reset_fft_mae_bpm"] == pytest.approx(1.0)
    assert metrics["valid_full_windows"] == 2
    assert metrics["valid_motion_windows"] == 1
    assert metrics["full_final_mae_reliable_bpm"] == pytest.approx(0.0)
    assert metrics["valid_reliable_windows"] == 1


def test_smoothing_durations_are_positive_odd_and_include_control() -> None:
    assert validate_smoothing_durations(DEFAULT_SMOOTH_DURATIONS_S) == (
        1,
        3,
        5,
        7,
        9,
        11,
    )
    with pytest.raises(ValueError, match="1 s"):
        validate_smoothing_durations((3, 5, 7))
    with pytest.raises(ValueError, match="正奇数"):
        validate_smoothing_durations((1, 4, 5))


def test_human_decision_template_stays_pending(tmp_path: Path) -> None:
    path = write_human_decision_template(
        tmp_path,
        evidence_shortlist=(3, 5),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["status"] == "pending_human_review"
    assert payload["selected_smooth_win_len_s"] is None
    assert payload["formal_experiment_authorized"] is False
    assert payload["evidence_shortlist_s"] == [3, 5]
