from __future__ import annotations

import json

import pytest

from ppg_hr.v2.cross_subject_loso_reporting import (
    CompactVerificationError,
    build_reporting_tables,
    verify_materialized_facts,
)


def _compact() -> dict:
    return {
        "candidate_mae_bpm": 3.25,
        "candidate_l10": 4,
        "candidate_l20": 2,
        "candidate_e10": 8,
        "candidate_e20": 3,
        "candidate_right_censored_recovery_count": 0,
        "candidate_full_window_count": 100,
        "candidate_reliable_window_count": 90,
        "candidate_motion_window_count": 20,
        "candidate_evaluation_window_sha256": "window",
        "g1i_pass": True,
        "g2_pass": True,
        "g3_pass": True,
        "g4_pass": True,
        "g5_pass": True,
        "g7_pass": True,
        "qualified": True,
        "g2_margin_s": 6.0,
        "g3_margin_s": 0.0,
        "g4_margin_bpm": 1.0,
        "g5_right_censored_count": 0,
        "g7_margin_s": 16.0,
        "failed_gates_json": "[]",
    }


def _candidate() -> dict:
    return {
        "mae_bpm": 3.25,
        "l10": 4,
        "l20": 2,
        "e10": 8,
        "e20": 3,
        "right_censored_recovery_count": 0,
        "full_window_count": 100,
        "reliable_window_count": 90,
        "motion_window_count": 20,
        "evaluation_window_sha256": "window",
    }


def _gate() -> dict:
    return {
        "g1i_pass": True,
        "g2_pass": True,
        "g3_pass": True,
        "g4_pass": True,
        "g5_pass": True,
        "g7_pass": True,
        "qualified": True,
        "g2_margin_s": 6.0,
        "g3_margin_s": 0.0,
        "g4_margin_bpm": 1.0,
        "g5_right_censored_count": 0,
        "g7_margin_s": 16.0,
        "failed_gates": (),
    }


def test_materialized_full_report_facts_must_match_compact_cell() -> None:
    verify_materialized_facts(_compact(), _candidate(), _gate())

    changed = _candidate()
    changed["mae_bpm"] = 3.2501
    with pytest.raises(CompactVerificationError, match="candidate_mae_bpm"):
        verify_materialized_facts(_compact(), changed, _gate())


def test_failed_gate_identity_is_verified() -> None:
    compact = _compact()
    compact["failed_gates_json"] = json.dumps(["G4"])
    compact["qualified"] = False
    compact["g4_pass"] = False
    gate = _gate()
    gate["failed_gates"] = ("G2",)
    gate["qualified"] = False
    gate["g4_pass"] = False

    with pytest.raises(CompactVerificationError, match="failed_gates"):
        verify_materialized_facts(compact, _candidate(), gate)


def test_primary_overall_mae_is_unweighted_across_folds() -> None:
    record_rows = [
        {
            "scene": "s",
            "fold_id": "f1",
            "candidate_mae_bpm": 1.0,
            "baseline_mae_bpm": 2.0,
            "qualified": True,
            "g1i_pass": True,
            "g2_pass": True,
            "g3_pass": True,
            "g4_pass": True,
            "g5_pass": True,
            "g7_pass": True,
        },
        {
            "scene": "s",
            "fold_id": "f2",
            "candidate_mae_bpm": 10.0,
            "baseline_mae_bpm": 12.0,
            "qualified": False,
            "g1i_pass": True,
            "g2_pass": False,
            "g3_pass": True,
            "g4_pass": True,
            "g5_pass": True,
            "g7_pass": True,
        },
        {
            "scene": "s",
            "fold_id": "f2",
            "candidate_mae_bpm": 20.0,
            "baseline_mae_bpm": 22.0,
            "qualified": False,
            "g1i_pass": True,
            "g2_pass": False,
            "g3_pass": True,
            "g4_pass": True,
            "g5_pass": True,
            "g7_pass": True,
        },
    ]
    fold_rows = [
        {
            "scene": "s",
            "fold_id": "f1",
            "selected_coordinate_id": "c0",
            "candidate_mean_mae_bpm": 1.0,
            "baseline_mean_mae_bpm": 2.0,
            "passed_records": 1,
            "heldout_records": 1,
            "strict_all_pass": True,
        },
        {
            "scene": "s",
            "fold_id": "f2",
            "selected_coordinate_id": "c1",
            "candidate_mean_mae_bpm": 15.0,
            "baseline_mean_mae_bpm": 17.0,
            "passed_records": 0,
            "heldout_records": 2,
            "strict_all_pass": False,
        },
    ]

    tables = build_reporting_tables(record_rows, fold_rows)

    assert tables["overall"]["primary_mean_of_fold_mean_mae_bpm"] == 8.0
    assert tables["overall"]["raw_record_mean_mae_bpm"] == pytest.approx(31 / 3)
    assert tables["overall"]["passed_records"] == 1
    assert tables["coordinate_frequency"] == [
        {"coordinate_id": "c0", "selected_fold_count": 1},
        {"coordinate_id": "c1", "selected_fold_count": 1},
    ]
