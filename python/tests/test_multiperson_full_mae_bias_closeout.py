from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOL_PATH = Path(__file__).parents[1] / "tools" / "multiperson_full_mae_bias_closeout.py"
SPEC = importlib.util.spec_from_file_location("multiperson_full_mae_bias_closeout", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
CLOSEOUT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLOSEOUT
SPEC.loader.exec_module(CLOSEOUT)


def test_updated_dataset_card_keeps_records_and_coordinates_frozen() -> None:
    original = {
        "schema_id": "screened_cross_subject_dataset_card_v1",
        "status": "complete",
        "scene_count": 1,
        "records": [
            {
                "record_id": "run1_LYX",
                "scene": "run",
                "subject": "LYX",
                "coordinate_id": "physical4d:a",
                "selected_bias_s": 6.0,
                "best_gate_mae_bpm": 2.0,
            },
            {
                "record_id": "run1_TS",
                "scene": "run",
                "subject": "TS",
                "coordinate_id": "physical4d:b",
                "selected_bias_s": 4.0,
                "best_gate_mae_bpm": 1.0,
            },
        ],
        "scene_composition": [
            {
                "scene": "run",
                "record_ids": ["run1_LYX", "run1_TS"],
                "subjects": ["LYX", "TS"],
            }
        ],
    }
    evaluations = {
        "run1_LYX": {
            "record_id": "run1_LYX",
            "coordinate_id": "physical4d:a",
            "previous_r_all_bias_s": 6.0,
            "raw_full_mae_selected_bias_s": 5.5,
            "raw_full_mae_selected_common_mae_bpm": 1.8,
            "selected_bias_s": 5.5,
            "common_window_count": 100,
            "fixed_5s_common_mae_bpm": 1.9,
            "selected_common_mae_bpm": 1.8,
            "improvement_vs_5s_bpm": 0.1,
            "compatibility_metrics": {"mae_bpm": 1.81},
            "gate_diagnostic": {"qualified": True, "failed_gates": []},
            "raw_full_mae_gate_diagnostic": {
                "qualified": True,
                "failed_gates": [],
            },
            "previous_r_all_gate_diagnostic": {
                "qualified": True,
                "failed_gates": [],
            },
            "gate_passing_bias_candidates_s": [4.0, 5.5],
            "gate_passing_candidate_count": 2,
            "selected_from_gate_passing_candidates": True,
            "no_gate_passing_candidate_risk": False,
            "risk_flags": [],
            "selection_reason": "minimum_full_mae_among_gate_passing_candidates",
            "candidate_gate_diagnostics": [],
        },
        "run1_TS": {
            "record_id": "run1_TS",
            "coordinate_id": "physical4d:b",
            "previous_r_all_bias_s": 4.0,
            "raw_full_mae_selected_bias_s": 4.5,
            "raw_full_mae_selected_common_mae_bpm": 0.9,
            "selected_bias_s": 4.5,
            "common_window_count": 99,
            "fixed_5s_common_mae_bpm": 1.2,
            "selected_common_mae_bpm": 0.9,
            "improvement_vs_5s_bpm": 0.3,
            "compatibility_metrics": {"mae_bpm": 0.91},
            "gate_diagnostic": {
                "qualified": False,
                "failed_gates": ["absolute_l10"],
            },
            "raw_full_mae_gate_diagnostic": {
                "qualified": False,
                "failed_gates": ["absolute_l10"],
            },
            "previous_r_all_gate_diagnostic": {
                "qualified": False,
                "failed_gates": ["absolute_l10"],
            },
            "gate_passing_bias_candidates_s": [],
            "gate_passing_candidate_count": 0,
            "selected_from_gate_passing_candidates": False,
            "no_gate_passing_candidate_risk": True,
            "risk_flags": ["no_gate_passing_time_bias_candidate"],
            "selection_reason": (
                "global_full_mae_minimum_with_no_gate_passing_candidate_risk"
            ),
            "candidate_gate_diagnostics": [],
        },
    }

    updated = CLOSEOUT.build_updated_dataset_card(
        original,
        evaluations=evaluations,
        updated_at="2026-08-19T12:00:00+08:00",
    )

    assert updated["schema_id"] == "screened_cross_subject_dataset_card_v2"
    assert [row["record_id"] for row in updated["records"]] == [
        "run1_LYX",
        "run1_TS",
    ]
    assert [row["coordinate_id"] for row in updated["records"]] == [
        "physical4d:a",
        "physical4d:b",
    ]
    assert updated["records"][0]["selected_bias_s"] == 5.5
    assert updated["records"][0]["final_common_mae_bpm"] == 1.8
    assert updated["records"][1]["gate_diagnostic"]["qualified"] is False
    assert updated["records"][1]["no_gate_passing_candidate_risk"] is True
    assert updated["scene_composition"] == original["scene_composition"]


def test_updated_dataset_card_fails_if_coordinate_would_change() -> None:
    original = {
        "records": [
            {
                "record_id": "run1_LYX",
                "coordinate_id": "physical4d:frozen",
                "selected_bias_s": 5.0,
            }
        ]
    }
    evaluation = {
        "run1_LYX": {
            "coordinate_id": "physical4d:different",
        }
    }

    with pytest.raises(
        CLOSEOUT.FullMaeBiasCloseoutError,
        match="frozen_coordinate_mismatch",
    ):
        CLOSEOUT.build_updated_dataset_card(
            original,
            evaluations=evaluation,
            updated_at="2026-08-19T12:00:00+08:00",
        )


def test_recomputed_selection_rejects_a_tampered_selected_bias() -> None:
    recomputed = {
        "bias_candidates_s": [4.0, 4.5, 5.0, 5.5, 6.0],
        "selection_rule": "minimum_common_reliable_full_mae_nearest_5s_then_smaller",
        "common_window_indices": [0, 1],
        "common_window_count": 2,
        "curve": [
            {"bias_s": bias, "common_mae_bpm": abs(bias - 4.5), "window_count": 2}
            for bias in (4.0, 4.5, 5.0, 5.5, 6.0)
        ],
        "selected_bias_s": 4.5,
        "selected_common_mae_bpm": 0.0,
        "fixed_5s_common_mae_bpm": 0.5,
        "improvement_vs_5s_bpm": 0.5,
    }
    stored = dict(recomputed)
    stored["selected_bias_s"] = 5.0

    with pytest.raises(
        CLOSEOUT.FullMaeBiasCloseoutError,
        match="recomputed_selection_mismatch:run1_LYX",
    ):
        CLOSEOUT.assert_recomputed_selection_matches(
            record_id="run1_LYX",
            stored=stored,
            recomputed=recomputed,
        )
