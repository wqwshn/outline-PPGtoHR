from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ppg_hr.v2.phase2_independent as phase2_independent
from ppg_hr.v2.bo_space_generalization import (
    CandidateSolveOutcome,
    FormalMetricResult,
    SeedSearchBudget,
)
from ppg_hr.v2.phase2_independent import (
    IndependentRecordRuntime,
    IndependentStudyConfig,
    run_independent_bo_study,
)
from ppg_hr.v2.solver import V2SolverResult


def _metric(
    *,
    full: float,
    reliable_motion: float,
    classic_motion: float,
) -> FormalMetricResult:
    return FormalMetricResult(
        metric_contract_version="lyx_bo_formal_metric_v1",
        final_method="LMS+H",
        reset_fft_method="reset FFT",
        base_full_window_count=12,
        base_motion_window_count=10,
        classic_motion_window_count=12,
        base_full_final_finite_count=12,
        base_motion_final_finite_count=10,
        base_motion_reset_fft_finite_count=10,
        base_motion_common_finite_count=10,
        classic_motion_final_finite_count=12,
        classic_motion_reset_fft_finite_count=12,
        classic_motion_common_finite_count=12,
        base_full_window_sha256="a" * 64,
        base_motion_window_sha256="b" * 64,
        classic_motion_window_sha256="c" * 64,
        full_final_mae_bpm=full,
        reliable_motion_final_mae_bpm=reliable_motion,
        reliable_motion_reset_fft_mae_bpm=8.0,
        classic_motion_final_mae_bpm=classic_motion,
        classic_motion_reset_fft_mae_bpm=8.5,
    )


def _solver_result() -> V2SolverResult:
    centers = np.arange(12, dtype=float)
    return V2SolverResult(
        HR=np.column_stack(
            [
                centers,
                np.full(12, 100.0),
                np.full(12, 101.0),
                np.full(12, 102.0),
                np.ones(12),
            ]
        ),
        err_stats={},
        metadata={
            "analysis_scope": "full",
            "adaptive_filter": "lms",
            "reference_groups_order": ["HF"],
        },
        window_table=[
            {
                "window_idx": idx,
                "center_s": float(idx),
                "reliable": True,
            }
            for idx in range(12)
        ],
    )


def _runtime(tmp_path: Path) -> IndependentRecordRuntime:
    solve_counts: dict[str, int] = {}

    def solve(candidate) -> CandidateSolveOutcome:
        solve_counts[candidate.candidate_id] = (
            solve_counts.get(candidate.candidate_id, 0) + 1
        )
        full = float(sum(candidate.coordinate))
        return CandidateSolveOutcome.valid(
            _solver_result(),
            _metric(
                full=full,
                reliable_motion=full + 1.0,
                classic_motion=full + 1.5,
            ),
            diagnostics={
                "solver_runtime_seconds": 0.01,
                "lms_configured_max_order": int(
                    candidate.actual_params["max_order"]
                ),
            },
        )

    def render(arm, candidate, _outcome, output_dir: Path) -> Path:
        path = output_dir / "png" / f"{arm}-{candidate.candidate_id[-8:]}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return path

    historical_plot = tmp_path / "historical.png"
    historical_plot.write_bytes(b"png")
    return IndependentRecordRuntime(
        sample_id="xiezi-1",
        data_sha256="1" * 64,
        reference_sha256="2" * 64,
        run_config={"analysis_scope": "full", "reference_groups_order": ["HF"]},
        historical_metrics=_metric(
            full=2.0,
            reliable_motion=2.5,
            classic_motion=3.0,
        ),
        historical_method_names=("reset FFT", "LMS+H"),
        historical_plot=historical_plot,
        solve_candidate=solve,
        render_selected=render,
        diagnostics={"solve_counts": solve_counts},
    )


def _config(tmp_path: Path) -> IndependentStudyConfig:
    small = SeedSearchBudget(
        lane_seeds=(42, 43, 44),
        lane_unique_budget=1,
        global_unique_budget=3,
        n_startup_trials=1,
    )
    return IndependentStudyConfig(
        historical_report_path=tmp_path / "historical.json",
        historical_error_csv=tmp_path / "historical-error.csv",
        output_dir=tmp_path / "study",
        git_commit="test-commit",
        legacy_budget=small,
        physical_budget=replace(
            small,
            objective_version="phase2_independent_physical_v1",
        ),
    )


def test_independent_study_writes_dual_baseline_history_and_classic_plots(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    result = run_independent_bo_study(
        _config(tmp_path),
        runtime=runtime,
    )

    assert result.sample_id == "xiezi-1"
    assert result.historical_metrics.classic_motion_final_mae_bpm == 3.0
    assert result.comparison[
        "physical_vs_historical_classic_delta_bpm"
    ] == (
        result.physical.selected_metrics.classic_motion_final_mae_bpm - 3.0
    )
    assert result.comparison[
        "physical_vs_legacy_reliable_delta_bpm"
    ] == (
        result.physical.selected_metrics.reliable_motion_final_mae_bpm
        - result.legacy.selected_metrics.reliable_motion_final_mae_bpm
    )
    assert result.comparison[
        "physical_vs_legacy_classic_delta_bpm"
    ] == (
        result.physical.selected_metrics.classic_motion_final_mae_bpm
        - result.legacy.selected_metrics.classic_motion_final_mae_bpm
    )
    assert result.legacy.classic_plot.is_file()
    assert result.physical.classic_plot.is_file()
    assert runtime.historical_plot.is_file()
    assert result.comparison_table.is_file()
    assert result.acceptance_preview.is_file()

    legacy_history = result.legacy.candidate_history.read_text(encoding="utf-8")
    assert "seed_42" in legacy_history
    assert "scene" in legacy_history
    assert "fold" in legacy_history
    assert "stage" in legacy_history
    assert "cache_hit" in legacy_history
    assert "base_motion_window_sha256" in legacy_history
    assert "diagnostic_solver_runtime_seconds" in legacy_history
    assert "requested_fs_target" in legacy_history
    assert "actual_fs_target" in legacy_history
    legacy_stability = json.loads(
        result.legacy.seed_stability.read_text(encoding="utf-8")
    )
    assert all(lane["best_so_far"] for lane in legacy_stability["lanes"])
    assert "cross_lane_overlap_count" in legacy_stability
    assert "pairwise_lane_overlap_counts" in legacy_stability
    assert "seed_best_parameter_differences" in legacy_stability
    assert legacy_stability["cache_statistics"]["logical_request_count"] > 0
    assert (
        legacy_stability["cache_statistics"]["physical_solve_count"]
        == result.legacy.cache_summary["physical_solve_count"]
    )
    selected = json.loads(
        (
            result.legacy.candidate_history.parent / "selected_candidate.json"
        ).read_text(encoding="utf-8")
    )
    assert "diagnostics" in selected


def test_independent_study_reuses_cache_and_is_numerically_repeatable(
    tmp_path,
) -> None:
    runtime = _runtime(tmp_path)
    config = _config(tmp_path)

    first = run_independent_bo_study(config, runtime=runtime)
    counts_after_first = dict(runtime.diagnostics["solve_counts"])
    second = run_independent_bo_study(config, runtime=runtime)

    assert second.legacy.selected_candidate_id == first.legacy.selected_candidate_id
    assert second.physical.selected_candidate_id == first.physical.selected_candidate_id
    assert second.comparison == first.comparison
    assert runtime.diagnostics["solve_counts"] == counts_after_first
    assert second.legacy.search_result == first.legacy.search_result
    assert second.physical.search_result == first.physical.search_result


def test_independent_study_keeps_fill_out_of_seed_stability(tmp_path) -> None:
    result = run_independent_bo_study(
        _config(tmp_path),
        runtime=_runtime(tmp_path),
    )

    for arm in (result.legacy, result.physical):
        seed_union = {
            candidate_id
            for lane in arm.search_result.lanes
            for candidate_id in lane.unique_candidate_ids
        }
        assert set(arm.search_result.seed_stability_candidate_ids) == seed_union
        assert all(row.stage == "fill" for row in arm.search_result.fill_history)


def test_classic_plot_method_validation_requires_acc_curve(tmp_path) -> None:
    incomplete = tmp_path / "incomplete-error.csv"
    incomplete.write_text(
        "method,mae_bpm\nreset FFT,2.0\nLMS+H,3.0\n",
        encoding="utf-8-sig",
    )

    with pytest.raises(ValueError, match=r"LMS\+A"):
        phase2_independent._validate_classic_plot_methods(incomplete)

    complete = tmp_path / "complete-error.csv"
    complete.write_text(
        "method,mae_bpm\nreset FFT,2.0\nLMS+H,3.0\nLMS+A,4.0\n",
        encoding="utf-8-sig",
    )
    phase2_independent._validate_classic_plot_methods(complete)


def test_default_runtime_rejects_report_paths_outside_frozen_inputs(
    tmp_path,
    monkeypatch,
) -> None:
    report = tmp_path / "historical.json"
    report.write_text("{}", encoding="utf-8")
    error = tmp_path / "historical-error.csv"
    error.write_text("method\nLMS+H\n", encoding="utf-8")
    report_data = tmp_path / "report-data.csv"
    report_ref = tmp_path / "report-ref.csv"
    frozen_data = tmp_path / "frozen-data.csv"
    frozen_ref = tmp_path / "frozen-ref.csv"
    monkeypatch.setattr(
        phase2_independent,
        "load_v2_report",
        lambda _path: {},
    )
    monkeypatch.setattr(
        phase2_independent,
        "load_lite_report_config",
        lambda _payload: SimpleNamespace(
            data_path=report_data,
            ref_path=report_ref,
        ),
    )
    config = IndependentStudyConfig(
        historical_report_path=report,
        historical_error_csv=error,
        output_dir=tmp_path / "out",
        git_commit="commit",
        expected_data_path=frozen_data,
        expected_reference_path=frozen_ref,
    )

    with pytest.raises(ValueError, match="preflight 冻结路径"):
        phase2_independent._build_default_runtime(config)


def test_historical_method_identity_requires_exact_reset_fft_name() -> None:
    with pytest.raises(
        phase2_independent.IndependentMethodIdentityMismatchError,
        match="reset FFT",
    ):
        phase2_independent._validate_historical_method_names(
            ("FFT", "LMS+H", "LMS+A")
        )
