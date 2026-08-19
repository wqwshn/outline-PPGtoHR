from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.solver import V2SolverResult

TOOL_PATH = Path(__file__).parents[1] / "tools" / "multiperson_screening_contracts.py"
SPEC = importlib.util.spec_from_file_location("multiperson_screening_contracts", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
CONTRACTS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONTRACTS
SPEC.loader.exec_module(CONTRACTS)

calibrate_rest_time_bias = CONTRACTS.calibrate_rest_time_bias
evaluate_aligned_metrics = CONTRACTS.evaluate_aligned_metrics
evaluate_screening_gate = CONTRACTS.evaluate_screening_gate
select_scene_panel = CONTRACTS.select_scene_panel
select_full_mae_time_bias = CONTRACTS.select_full_mae_time_bias

ORCHESTRATOR_PATH = (
    Path(__file__).parents[1] / "tools" / "multiperson_joint_screening.py"
)
ORCHESTRATOR_SPEC = importlib.util.spec_from_file_location(
    "multiperson_joint_screening", ORCHESTRATOR_PATH
)
assert ORCHESTRATOR_SPEC is not None and ORCHESTRATOR_SPEC.loader is not None
ORCHESTRATOR = importlib.util.module_from_spec(ORCHESTRATOR_SPEC)
sys.modules[ORCHESTRATOR_SPEC.name] = ORCHESTRATOR
ORCHESTRATOR_SPEC.loader.exec_module(ORCHESTRATOR)


def _result(
    centers: np.ndarray,
    final: np.ndarray,
    *,
    motion_start: float = 10.0,
    motion_end: float = 15.0,
    reliable: np.ndarray | None = None,
) -> V2SolverResult:
    mask = (centers >= motion_start) & (centers <= motion_end)
    hr = np.column_stack(
        [centers, np.zeros_like(centers), final, final, mask.astype(float), final]
    )
    if reliable is None:
        reliable = np.ones_like(centers, dtype=bool)
    rows = [
        {
            "window_idx": idx,
            "center_s": float(center),
            "reliable": bool(reliable[idx]),
            "adaptive_stages": [{"group": "HF"}, {"group": "HF"}],
        }
        for idx, center in enumerate(centers)
    ]
    return V2SolverResult(
        HR=hr,
        err_stats={"final_aae_bpm": 999.0},
        metadata={
            "analysis_scope": "full",
            "motion_segment": {"start_s": motion_start, "end_s": motion_end},
            "reference_groups_order": ["HF"],
            "adaptive_reference_stage_limit": None,
        },
        window_table=rows,
    )


def test_full_mae_time_bias_uses_one_common_reliable_window_set() -> None:
    centers = np.arange(0.0, 9.0)
    reliable = np.ones(centers.size, dtype=bool)
    reliable[2] = False
    ref_t = np.arange(0.0, 13.0)
    ref = np.column_stack([ref_t, 60.0 + ref_t])
    result = _result(centers, 66.0 + centers, reliable=reliable)
    result.HR[:, 1] = -999.0
    result.err_stats = {"final_aae_bpm": -999.0}

    selected = select_full_mae_time_bias(result, ref_data=ref)

    assert selected["schema_id"] == "full_mae_evaluation_time_bias_v1"
    assert selected["common_window_indices"] == [0, 1, 3, 4, 5, 6]
    assert selected["common_window_count"] == 6
    assert {row["window_count"] for row in selected["curve"]} == {6}
    assert selected["selected_bias_s"] == pytest.approx(6.0)
    assert selected["selected_common_mae_bpm"] == pytest.approx(0.0)
    assert selected["fixed_5s_common_mae_bpm"] == pytest.approx(1.0)
    assert selected["improvement_vs_5s_bpm"] == pytest.approx(1.0)


def test_full_mae_time_bias_tie_prefers_nearest_5s_then_smaller() -> None:
    ref = np.asarray(
        [
            [4.0, 90.0],
            [4.5, 70.0],
            [5.0, 80.0],
            [5.5, 70.0],
            [6.0, 90.0],
        ]
    )

    selected = select_full_mae_time_bias(
        _result(np.asarray([0.0]), np.asarray([70.0])),
        ref_data=ref,
        biases_s=(4.5, 5.0, 5.5),
    )

    assert selected["selected_bias_s"] == pytest.approx(4.5)
    assert selected["selected_common_mae_bpm"] == pytest.approx(0.0)


def test_rest_calibration_uses_dynamic_post_rest_but_not_motion_error() -> None:
    ref_t = np.arange(0.0, 101.0)
    ref_hr = np.where(ref_t < 30.0, 70.0, 70.0 + ref_t - 30.0)
    ref = np.column_stack([ref_t, ref_hr])
    centers = np.arange(0.0, 80.0)
    final = np.interp(centers + 6.0, ref_t, ref_hr)
    # Make the motion interval favor 4 s. It must remain diagnostic only.
    motion = (centers >= 10.0) & (centers <= 15.0)
    final[motion] = np.interp(centers[motion] + 4.0, ref_t, ref_hr)

    calibrated = calibrate_rest_time_bias(
        _result(centers, final),
        ref_data=ref,
        post_motion_guard_seconds=2.0,
    )

    assert calibrated["r_all"]["selected_bias_s"] == pytest.approx(6.0)
    assert calibrated["r_pre"]["selected_bias_s"] == pytest.approx(5.0)
    assert calibrated["r_all"]["identifiable"] is True


def test_rest_calibration_falls_back_when_nondefault_gain_is_too_small() -> None:
    ref_t = np.arange(0.0, 101.0)
    ref_hr = 70.0 + 0.04 * ref_t
    ref = np.column_stack([ref_t, ref_hr])
    centers = np.arange(0.0, 80.0)
    final = np.interp(centers + 6.0, ref_t, ref_hr)

    calibrated = calibrate_rest_time_bias(
        _result(centers, final),
        ref_data=ref,
        post_motion_guard_seconds=2.0,
    )

    assert calibrated["r_all"]["raw_winner_bias_s"] == pytest.approx(6.0)
    assert calibrated["r_all"]["selected_bias_s"] == pytest.approx(5.0)
    assert "insufficient_improvement_vs_5s" in calibrated["r_all"][
        "fallback_reason"
    ]


def test_rest_calibration_requires_ten_windows_and_falls_back() -> None:
    ref_t = np.arange(0.0, 31.0)
    ref = np.column_stack([ref_t, 60.0 + ref_t])
    centers = np.arange(10.0, 16.0)

    calibrated = calibrate_rest_time_bias(
        _result(centers, 60.0 + centers),
        ref_data=ref,
    )

    assert calibrated["r_all"]["selected_bias_s"] == pytest.approx(5.0)
    assert calibrated["r_all"]["identifiable"] is False


def test_aligned_metrics_ignore_report_reference_column_and_err_stats() -> None:
    centers = np.arange(0.0, 30.0)
    ref_t = np.arange(0.0, 50.0)
    ref_hr = 60.0 + ref_t
    result = _result(centers, centers + 65.0)
    result.HR[:, 1] = -999.0

    metrics = evaluate_aligned_metrics(
        result,
        ref_data=np.column_stack([ref_t, ref_hr]),
        time_bias_s=5.0,
    )

    assert metrics["mae_bpm"] == pytest.approx(0.0)
    assert metrics["e10"] == 0
    assert metrics["l10"] == 0


def test_aligned_metrics_l10_uses_full_continuous_timeline() -> None:
    centers = np.arange(0.0, 30.0)
    ref = np.column_stack([np.arange(0.0, 50.0), np.full(50, 70.0)])
    final = np.full(30, 70.0)
    final[2:14] = 55.0
    reliable = np.ones(30, dtype=bool)
    reliable[5:8] = False

    metrics = evaluate_aligned_metrics(
        _result(centers, final, reliable=reliable),
        ref_data=ref,
        time_bias_s=5.0,
    )

    assert metrics["l10"] == 12
    assert metrics["reliable_window_count"] == 27


def _metrics(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "mae_bpm": 4.0,
        "l10": 8,
        "l20": 1,
        "right_censored_recovery_count": 0,
        "true_rise_applicable": False,
        "true_rise_underestimate_bpm": None,
        "spectral_gate_contract_v2": True,
        "stability_pass": True,
        "reference_groups_order": ["HF"],
        "adaptive_reference_stage_limit": None,
    }
    values.update(overrides)
    return values


def test_screening_gate_enforces_absolute_l10_and_zero_right_censor() -> None:
    baseline = _metrics(mae_bpm=3.0, l10=30, l20=3)

    failed = evaluate_screening_gate(
        candidate=_metrics(l10=21, l20=2, right_censored_recovery_count=1),
        baseline=baseline,
    )

    assert failed["qualified"] is False
    assert "absolute_l10" in failed["failed_gates"]
    assert "right_censored_e10" in failed["failed_gates"]


def test_screening_gate_uses_reference_defined_true_rise_applicability() -> None:
    passed = evaluate_screening_gate(
        candidate=_metrics(
            true_rise_applicable=True,
            true_rise_underestimate_bpm=4.0,
        ),
        baseline=_metrics(
            true_rise_applicable=True,
            true_rise_underestimate_bpm=3.0,
        ),
    )
    failed = evaluate_screening_gate(
        candidate=_metrics(
            true_rise_applicable=True,
            true_rise_underestimate_bpm=6.1,
        ),
        baseline=_metrics(
            true_rise_applicable=True,
            true_rise_underestimate_bpm=4.0,
        ),
    )

    assert passed["qualified"] is True
    assert failed["qualified"] is False
    assert "true_rise_underestimate" in failed["failed_gates"]


def test_panel_requires_lyx_and_prefers_six_distinct_subjects() -> None:
    rows = [
        {
            "scene": "run",
            "subject": subject,
            "record_id": f"run1_{subject}",
            "qualified": True,
            "best_gate_mae_bpm": mae,
        }
        for subject, mae in zip(
            ("LYX", "QYC", "TS", "CGX", "HB", "LZJ", "PJY"),
            (5.0, 4.0, 3.0, 6.0, 7.0, 8.0, 9.0),
            strict=True,
        )
    ]

    panel = select_scene_panel(rows, scene="run")

    assert panel["status"] == "complete_six_subjects"
    assert len({row["subject"] for row in panel["selected"]}) == 6
    assert any(row["subject"] == "LYX" for row in panel["selected"])


def test_panel_fails_closed_without_qualified_lyx() -> None:
    panel = select_scene_panel(
        [
            {
                "scene": "run",
                "subject": "TS",
                "record_id": "run1_TS",
                "qualified": True,
                "best_gate_mae_bpm": 2.0,
            }
        ],
        scene="run",
    )

    assert panel["status"] == "failed_no_qualified_development_subject"
    assert panel["selected"] == []


def test_bo120_replay_is_deterministic_and_counts_logical_duplicates() -> None:
    rows = []
    index = 0
    for fs in (25, 50, 100):
        for memory in (40, 80, 120, 160, 200):
            for mu in (0.006, 0.008, 0.010, 0.012, 0.016):
                for width in (3, 6, 12, 18):
                    rows.append(
                        {
                            "coordinate_id": (
                                f"physical4d:fs{fs:03d}:m{memory:03d}:"
                                f"mu{int(round(mu * 1000)):04d}:w{width:03d}"
                            ),
                            "metrics": {"mae_bpm": float(index + 1)},
                            "gate": {"qualified": index % 7 == 0},
                        }
                    )
                    index += 1

    first = ORCHESTRATOR.replay_bo120(rows)
    second = ORCHESTRATOR.replay_bo120(rows)

    assert first["logical_trial_count"] == 120
    assert first["new_solver_count"] == 0
    assert first["cache_hit_count"] == 120
    assert first["duplicate_logical_trial_count"] > 0
    assert [row["coordinate_id"] for row in first["history"]] == [
        row["coordinate_id"] for row in second["history"]
    ]
