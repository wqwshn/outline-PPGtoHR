from __future__ import annotations

import numpy as np
import pytest

from ppg_hr.v2.bo_space_generalization import (
    FormalMetricContractError,
    build_bo_search_space,
    evaluate_formal_metrics,
)
from ppg_hr.v2.solver import V2SolverResult


def test_phase2_search_spaces_match_frozen_candidate_contract() -> None:
    legacy_full = build_bo_search_space("legacy_full_v1")
    legacy_reduced = build_bo_search_space("legacy_reduced_v1")
    physical = build_bo_search_space("physical_v1")

    assert legacy_full.parameter_names == (
        "fs_target",
        "max_order",
        "lms_mu_base",
        "smooth_win_len",
        "spec_penalty_width",
        "time_bias",
    )
    assert legacy_reduced.parameter_names == (
        "fs_target",
        "max_order",
        "lms_mu_base",
        "spec_penalty_width",
    )
    assert physical.parameter_names == (
        "fs_target",
        "memory_ms",
        "mu_base",
        "exclusion_half_width_bpm",
    )
    assert len(legacy_full.candidates) == 1620
    assert len(legacy_reduced.candidates) == 108
    assert len(physical.candidates) == 300
    assert len({candidate.candidate_id for candidate in legacy_full.candidates}) == 1620
    assert len({candidate.candidate_id for candidate in legacy_reduced.candidates}) == 108
    assert len({candidate.candidate_id for candidate in physical.candidates}) == 300


def test_physical_space_preserves_requested_meaning_and_solver_mapping() -> None:
    physical = build_bo_search_space("physical_v1")
    mapped = {
        (
            candidate.requested_params["fs_target"],
            candidate.requested_params["memory_ms"],
        ): candidate
        for candidate in physical.candidates
        if candidate.requested_params["mu_base"] == 0.006
        and candidate.requested_params["exclusion_half_width_bpm"] == 3
    }

    expected_orders = {
        (25, 40): 1,
        (25, 80): 2,
        (25, 120): 3,
        (25, 160): 4,
        (25, 200): 5,
        (50, 40): 2,
        (50, 80): 4,
        (50, 120): 6,
        (50, 160): 8,
        (50, 200): 10,
        (100, 40): 4,
        (100, 80): 8,
        (100, 120): 12,
        (100, 160): 16,
        (100, 200): 20,
    }
    assert set(mapped) == set(expected_orders)
    for coordinate, expected_order in expected_orders.items():
        candidate = mapped[coordinate]
        assert candidate.actual_params["max_order"] == expected_order
        assert candidate.actual_params["lms_mu_base"] == 0.006
        assert candidate.actual_params["spec_penalty_width"] == 0.05
        assert candidate.fixed_params == {
            "analysis_scope": "full",
            "smooth_win_len": 5,
            "time_bias": 5.0,
            "lms_mu_min": 1e-6,
        }


def test_reduced_and_physical_spaces_cannot_override_five_second_settings() -> None:
    for space_name in ("legacy_reduced_v1", "physical_v1"):
        space = build_bo_search_space(space_name)
        assert all(
            candidate.actual_params["smooth_win_len"] == 5
            and candidate.actual_params["time_bias"] == 5.0
            and candidate.actual_params["lms_mu_min"] == 1e-6
            for candidate in space.candidates
        )
        assert all(
            "smooth_win_len" not in candidate.requested_params
            and "time_bias" not in candidate.requested_params
            for candidate in space.candidates
        )


def _formal_metric_fixture(
    *,
    row_count: int = 12,
    reliable: bool = True,
) -> tuple[V2SolverResult, np.ndarray]:
    centers = np.arange(row_count, dtype=float)
    hr = np.column_stack(
        [
            centers,
            np.full(row_count, 999.0),
            np.full(row_count, 102.0),
            np.full(row_count, 101.0),
            np.ones(row_count),
            np.ones(row_count),
        ]
    )
    window_table = [
        {
            "window_idx": idx,
            "center_s": float(center),
            "reliable": bool(reliable and idx < 10),
        }
        for idx, center in enumerate(centers)
    ]
    ref_data = np.column_stack(
        [
            np.arange(0.0, 30.0),
            np.full(30, 100.0),
        ]
    )
    return (
        V2SolverResult(
            HR=hr,
            err_stats={},
            metadata={"analysis_scope": "full"},
            window_table=window_table,
        ),
        ref_data,
    )


def test_formal_metrics_use_raw_reference_and_frozen_common_masks() -> None:
    result, ref_data = _formal_metric_fixture()

    metrics = evaluate_formal_metrics(
        result,
        ref_data=ref_data,
        time_bias=5.0,
    )

    assert metrics.metric_contract_version == "lyx_bo_formal_metric_v1"
    assert metrics.base_full_window_count == 10
    assert metrics.base_motion_window_count == 10
    assert metrics.classic_motion_window_count == 12
    assert metrics.full_final_mae_bpm == 1.0
    assert metrics.reliable_motion_final_mae_bpm == 1.0
    assert metrics.reliable_motion_reset_fft_mae_bpm == 2.0
    assert metrics.classic_motion_final_mae_bpm == 1.0
    assert metrics.classic_motion_reset_fft_mae_bpm == 2.0
    assert metrics.base_motion_final_finite_count == 10
    assert metrics.base_motion_reset_fft_finite_count == 10
    assert metrics.base_motion_common_finite_count == 10
    assert len(metrics.base_motion_window_sha256) == 64


def test_formal_metrics_fail_when_window_rows_do_not_join_exactly() -> None:
    result, ref_data = _formal_metric_fixture()
    result.window_table[3]["center_s"] = 3.00000001

    with pytest.raises(
        FormalMetricContractError,
        match="center_s",
    ):
        evaluate_formal_metrics(result, ref_data=ref_data, time_bias=5.0)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("too_few", "insufficient_base_full_windows"),
        ("no_reliable", "no_reliable_windows"),
        ("nonfinite_final", "nonfinite_final_on_base_full"),
        ("nonfinite_reset", "nonfinite_reset_fft_on_base_motion"),
    ],
)
def test_formal_metrics_fail_closed_on_invalid_denominators(
    mutation: str,
    reason: str,
) -> None:
    result, ref_data = _formal_metric_fixture()
    if mutation == "too_few":
        result.HR = result.HR[:9]
        result.window_table = result.window_table[:9]
    elif mutation == "no_reliable":
        for row in result.window_table:
            row["reliable"] = False
    elif mutation == "nonfinite_final":
        result.HR[0, 3] = np.nan
    elif mutation == "nonfinite_reset":
        result.HR[0, 2] = np.nan

    with pytest.raises(FormalMetricContractError) as error:
        evaluate_formal_metrics(result, ref_data=ref_data, time_bias=5.0)

    assert error.value.reason == reason
