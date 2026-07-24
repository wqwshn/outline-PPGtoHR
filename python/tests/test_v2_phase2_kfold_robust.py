from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import ppg_hr.v2.phase2_kfold_runtime as kfold_runtime
from ppg_hr.v2.bo_space_generalization import (
    CandidateSolveOutcome,
    FormalMetricResult,
    SeedSearchBudget,
)
from ppg_hr.v2.phase2_kfold_robust import (
    K1AuditIntegrityError,
    K1DriverIdentityConflictError,
    K1FoldConfig,
    K3FoldConfig,
    run_k1_fold_study,
    run_k3_fold_study,
)
from ppg_hr.v2.phase2_kfold_runtime import (
    ClassicPlotArtifact,
    KFoldRuntime,
    KFoldTrainingRecordRuntime,
)
from ppg_hr.v2.phase2_receipt import (
    FrozenReplayOutcome,
    RecordIdentity,
    ReplayInfrastructureError,
)
from ppg_hr.v2.solver import V2SolverResult


def _sha(character: str) -> str:
    return character * 64


def _record(record_id: str, character: str) -> RecordIdentity:
    return RecordIdentity(
        record_id=record_id,
        data_path=f"D:/data/{record_id}.csv",
        data_sha256=_sha(character),
        reference_path=f"D:/ref/{record_id}.csv",
        reference_sha256=_sha(character),
    )


def _metric(
    *,
    final_motion: float,
    reset_motion: float,
) -> FormalMetricResult:
    return FormalMetricResult(
        metric_contract_version="lyx_bo_formal_metric_v1",
        final_method="LMS+H",
        reset_fft_method="reset FFT",
        base_full_window_count=12,
        base_motion_window_count=10,
        classic_motion_window_count=10,
        base_full_final_finite_count=12,
        base_motion_final_finite_count=10,
        base_motion_reset_fft_finite_count=10,
        base_motion_common_finite_count=10,
        classic_motion_final_finite_count=10,
        classic_motion_reset_fft_finite_count=10,
        classic_motion_common_finite_count=10,
        base_full_window_sha256=_sha("d"),
        base_motion_window_sha256=_sha("e"),
        classic_motion_window_sha256=_sha("f"),
        full_final_mae_bpm=final_motion + 0.5,
        reliable_motion_final_mae_bpm=final_motion,
        reliable_motion_reset_fft_mae_bpm=reset_motion,
        classic_motion_final_mae_bpm=final_motion + 0.25,
        classic_motion_reset_fft_mae_bpm=reset_motion + 0.25,
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
                "window_idx": index,
                "center_s": float(index),
                "reliable": True,
            }
            for index in range(12)
        ],
    )


def test_kfold_runtime_reports_lms_order_cap_and_numeric_diagnostics() -> None:
    result = _solver_result()
    result.window_table[0]["adaptive_stages"] = [
        {"M": 4, "delay_samples": -4},
        {"M": 8, "delay_samples": -12},
    ]
    result.HR[0, 1] = np.nan

    diagnostics = kfold_runtime._solver_diagnostics(
        result,
        max_order=8,
        solver_runtime_seconds=0.25,
    )

    assert diagnostics["solver_runtime_seconds"] == 0.25
    assert diagnostics["lms_stage_count"] == 2
    assert diagnostics["lms_delay_derived_order_min"] == 4
    assert diagnostics["lms_delay_derived_order_max"] == 8
    assert diagnostics["lms_max_order_hit"] is True
    assert diagnostics["lms_max_order_hit_count"] == 1
    assert diagnostics["nonfinite_hr_value_count"] == 1


def test_k3_plot_title_preserves_physical_request_and_actual_mapping() -> None:
    title = kfold_runtime.kfold_plot_title(
        arm="K3",
        training_record_ids=("run1", "run2"),
        heldout_record_id="run3",
        view_role="test",
        view_record_id="run3",
        actual_params={
            "fs_target": 100,
            "max_order": 20,
            "lms_mu_base": 0.008,
            "smooth_win_len": 5,
            "spec_penalty_width": 0.1,
            "time_bias": 5.0,
        },
        requested_params={
            "fs_target": 100,
            "memory_ms": 200,
            "mu_base": 0.008,
            "exclusion_half_width_bpm": 6,
        },
    )

    assert "order=20taps" in title
    assert "memory=200ms" in title
    assert "exclusion=6BPM" in title


def _training_runtime(
    *,
    record: RecordIdentity,
    offset: float,
) -> KFoldTrainingRecordRuntime:
    def solve(candidate) -> CandidateSolveOutcome:
        distance = float(sum(candidate.coordinate)) / 100.0
        final = 3.0 + offset + distance
        return CandidateSolveOutcome.valid(
            _solver_result(),
            _metric(
                final_motion=final,
                reset_motion=final - 1.0,
            ),
            diagnostics={
                "solver_runtime_seconds": 0.25,
                "lms_stage_count": 10,
                "lms_delay_derived_order_max": int(
                    candidate.actual_params["max_order"]
                ),
                "lms_max_order_hit": True,
                "lms_max_order_hit_count": 2,
                "nonfinite_hr_value_count": 0,
            },
        )

    def render(
        _candidate,
        _outcome,
        output_dir: Path,
    ) -> ClassicPlotArtifact:
        figure = output_dir / f"{record.record_id}.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        figure.write_bytes(b"png")
        return ClassicPlotArtifact(
            figure_png=figure,
            method_names=("reset FFT", "LMS+H", "LMS+A"),
        )

    return KFoldTrainingRecordRuntime(
        identity=record,
        run_config={"record_id": record.record_id},
        solve_candidate=solve,
        render_selected=render,
    )


def _config(tmp_path: Path) -> K1FoldConfig:
    return K1FoldConfig(
        output_dir=tmp_path / "k1",
        scene="xiezi",
        fold=0,
        git_commit="test-commit",
        budget=SeedSearchBudget(
            lane_seeds=(42, 43, 44),
            lane_unique_budget=1,
            global_unique_budget=3,
            n_startup_trials=1,
            objective_version="phase2_robust_worst_motion_v1",
            constraints_version="phase2_nonharm_per_record_v1",
        ),
        neighborhood_budget=30,
    )


def _k3_config(tmp_path: Path) -> K3FoldConfig:
    return K3FoldConfig(
        output_dir=tmp_path / "k3",
        scene="xiezi",
        fold=0,
        git_commit="test-commit",
        budget=SeedSearchBudget(
            lane_seeds=(42, 43, 44),
            lane_unique_budget=1,
            global_unique_budget=3,
            n_startup_trials=1,
            objective_version="phase2_robust_worst_motion_v1",
            constraints_version="phase2_nonharm_per_record_v1",
        ),
        neighborhood_budget=30,
    )


def test_k3_defaults_freeze_120_plus_30_budget(tmp_path) -> None:
    config = K3FoldConfig(
        output_dir=tmp_path / "k3-defaults",
        scene="run",
        fold=0,
        git_commit="test-commit",
    )

    assert config.budget.lane_seeds == (42, 43, 44)
    assert config.budget.lane_unique_budget == 40
    assert config.budget.global_unique_budget == 120
    assert config.budget.n_startup_trials == 10
    assert config.neighborhood_budget == 30


def test_k3_reuses_robust_contract_and_preserves_physical_parameter_meaning(
    tmp_path,
) -> None:
    config = _k3_config(tmp_path)
    replay_calls: list[str] = []

    def replay(context) -> FrozenReplayOutcome:
        replay_calls.append(context.candidate_id)
        return FrozenReplayOutcome.success(
            metrics={"reliable_motion_final_mae_bpm": 4.5},
            artifact_sha256s={
                "hf": _sha("1"),
                "reset_fft": _sha("2"),
                "acc": _sha("3"),
            },
        )

    result = run_k3_fold_study(
        config,
        runtime=KFoldRuntime(
            training_records=(
                _training_runtime(
                    record=_record("xiezi-1", "a"),
                    offset=0.0,
                ),
                _training_runtime(
                    record=_record("xiezi-2", "b"),
                    offset=0.5,
                ),
            ),
            heldout_record=_record("xiezi-3", "c"),
            replay_heldout=replay,
        ),
    )

    assert result.arm == "K3"
    assert result.replay_status == "success"
    assert replay_calls == [result.selected_candidate_id]
    assert all(
        candidate_id.startswith("physical_v1:")
        for candidate_id in result.search_result.global_candidate_ids
    )
    assert result.neighborhood_candidate_count <= 30
    assert result.manifest.name == "k3_fold_manifest.json"

    with result.candidate_history.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    required_fields = {
        "requested_memory_ms",
        "requested_mu_base",
        "requested_exclusion_half_width_bpm",
        "actual_max_order",
        "actual_lms_mu_base",
        "actual_spec_penalty_width",
        "fixed_smooth_win_len",
        "fixed_time_bias",
        "constraint_train_0_bpm",
        "constraint_train_1_bpm",
        "diagnostic_train_0_solver_runtime_seconds",
        "diagnostic_train_0_lms_delay_derived_order_max",
        "diagnostic_train_0_lms_max_order_hit",
        "diagnostic_train_0_nonfinite_hr_value_count",
    }
    assert required_fields <= set(rows[0])
    for row in rows:
        expected_order = round(
            float(row["requested_fs_target"])
            * float(row["requested_memory_ms"])
            / 1000.0
        )
        assert int(row["actual_max_order"]) == expected_order
        assert float(row["actual_spec_penalty_width"]) == pytest.approx(
            float(row["requested_exclusion_half_width_bpm"]) / 60.0
        )
        assert row["fixed_smooth_win_len"] == "5"
        assert row["fixed_time_bias"] == "5.0"

    selection = json.loads(
        result.selection_receipt.read_text(encoding="utf-8")
    )
    evidence = selection["evidence"]
    assert evidence["arm"] == "K3"
    assert evidence["space_name"] == "physical_v1"
    assert "memory_ms" in evidence["selected_requested_params"]
    assert "max_order" in evidence["selected_actual_params"]
    assert [
        row["record_id"]
        for row in evidence["selected_diagnostics"]["training_records"]
    ] == ["xiezi-1", "xiezi-2"]
    assert all(
        "lms_max_order_hit" in row
        for row in evidence["selected_diagnostics"]["training_records"]
    )
    params = json.loads(result.selected_params.read_text(encoding="utf-8"))
    assert params["selected_diagnostics"] == evidence["selected_diagnostics"]
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["comparison_scope"] == "operational_workflow_only"
    assert manifest["causal_claim_allowed"] is False
    assert manifest["confirmatory_claim_allowed"] is False
    assert manifest["space_candidate_count"] == 300
    assert manifest["global_search_candidate_count"] == 3
    assert manifest["neighborhood_candidate_count"] <= 30
    assert manifest["reviewed_unique_candidate_count"] == (
        manifest["global_search_candidate_count"]
        + manifest["neighborhood_candidate_count"]
    )
    assert manifest["coverage_ratio"] == pytest.approx(
        manifest["reviewed_unique_candidate_count"] / 300
    )
    comparison = manifest["k2_k3_comparison_context"]
    assert comparison["k2_max_reviewed_candidate_count"] == 108
    assert comparison["k2_max_coverage_ratio"] == 1.0
    assert comparison["k3_max_reviewed_candidate_count"] == 150
    assert comparison["k3_max_coverage_ratio"] == 0.5
    assert comparison["k3_neighborhood_geometry"] == (
        "budgeted_direct_neighbors_primary_band_first_"
        "then_diagnostic_band_if_budget_remains"
    )
    assert (
        comparison["single_factor_causal_attribution_allowed"]
        is False
    )

    terminal_paths = (
        result.candidate_history,
        result.selected_params,
        result.selection_receipt,
        result.replay_receipt,
        result.manifest,
    )
    before = {
        path.name: path.read_bytes()
        for path in terminal_paths
    }
    recovered = run_k3_fold_study(
        config,
        runtime=KFoldRuntime(
            training_records=(
                _training_runtime(
                    record=_record("xiezi-1", "a"),
                    offset=0.0,
                ),
                _training_runtime(
                    record=_record("xiezi-2", "b"),
                    offset=0.5,
                ),
            ),
            heldout_record=_record("xiezi-3", "c"),
            replay_heldout=replay,
        ),
    )
    assert recovered.selected_candidate_id == result.selected_candidate_id
    assert replay_calls == [result.selected_candidate_id]
    assert {
        path.name: path.read_bytes()
        for path in terminal_paths
    } == before


def test_k1_uses_worst_motion_constraints_and_frozen_neighborhood(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    heldout = _record("xiezi-3", "c")
    replay_calls: list[str] = []

    def replay(context) -> FrozenReplayOutcome:
        assert config.output_dir.joinpath(
            "selection_receipt.json"
        ).is_file()
        replay_calls.append(context.candidate_id)
        return FrozenReplayOutcome.success(
            metrics={
                "reliable_motion_final_mae_bpm": 4.5,
            },
            artifact_sha256s={
                "hf": _sha("1"),
                "reset_fft": _sha("2"),
                "acc": _sha("3"),
            },
        )

    result = run_k1_fold_study(
        config,
        runtime=KFoldRuntime(
            training_records=(
                _training_runtime(
                    record=_record("xiezi-1", "a"),
                    offset=0.0,
                ),
                _training_runtime(
                    record=_record("xiezi-2", "b"),
                    offset=0.5,
                ),
            ),
            heldout_record=heldout,
            replay_heldout=replay,
        ),
    )

    assert result.arm == "K1"
    assert result.replay_status == "success"
    assert replay_calls == [result.selected_candidate_id]
    assert (
        result.selected_candidate_id
        in result.search_result.global_candidate_ids
    )
    assert result.neighborhood_candidate_count <= 30
    assert result.neighborhood_evidence.is_file()
    assert all(path.is_file() for path in result.training_plots)

    with result.candidate_history.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {"search", "neighborhood"} <= {
        row["stage"] for row in rows
    }
    assert {
        row["lane"]
        for row in rows
        if row["stage"] == "neighborhood"
    } == {"enumeration"}
    assert {
        "scene",
        "fold",
        "suggestion_index",
        "unique_index",
        "constraint_train_0_bpm",
        "constraint_train_1_bpm",
        "constraint_r1",
        "constraint_r2",
        "nonharm_delta_train_0_bpm",
        "tpe_objective",
        "cache_hit",
        "cache_key",
        "worst_train_mae_bpm",
        "mean_train_mae_bpm",
        "worst_train_mae",
        "mean_train_mae",
        "w_star_bpm",
        "w_star",
        "in_primary_band",
        "in_diagnostic_band",
        "center_candidate_id",
        "is_direct_neighbor",
        "support_neighbor",
        "parameter_cliff",
        "runtime_seconds",
    } <= set(rows[0])
    assert all(
        float(row["constraint_train_0_bpm"]) == pytest.approx(-1.0)
        for row in rows
    )

    receipt = json.loads(
        result.selection_receipt.read_text(encoding="utf-8")
    )
    manifest = json.loads(
        result.manifest.read_text(encoding="utf-8")
    )
    assert {
        "candidate_history",
        "selected_params",
        "training_metrics",
        "neighborhood_evidence",
        "selection_receipt",
        "replay_receipt",
        "cache_summary",
        "failure_classification",
    } == set(manifest["artifacts"]["files"])
    assert len(manifest["artifacts"]["training_plots"]) == 2
    training = receipt["evidence"]["training_metrics"]
    assert training["worst_train_mae_bpm"] >= (
        training["mean_train_mae_bpm"]
    )
    neighborhood = receipt["evidence"]["neighborhood_evidence"]
    assert neighborhood["status"] == "complete"
    assert neighborhood["reviewed_neighbor_count"] > 0
    neighborhood_hash_identity = next(
        identity
        for identity in receipt["evidence"]["study_identities"]
        if "neighborhood:" in identity
    )
    assert len(neighborhood_hash_identity.rsplit(
        "neighborhood:",
        maxsplit=1,
    )[1]) == 64
    cache_before = result.cache_summary.read_text(encoding="utf-8")
    history_before = result.candidate_history.read_bytes()

    repeat = run_k1_fold_study(
        config,
        runtime=KFoldRuntime(
            training_records=(
                _training_runtime(
                    record=_record("xiezi-1", "a"),
                    offset=0.0,
                ),
                _training_runtime(
                    record=_record("xiezi-2", "b"),
                    offset=0.5,
                ),
            ),
            heldout_record=heldout,
            replay_heldout=replay,
        ),
    )
    assert repeat.selected_candidate_id == result.selected_candidate_id
    assert repeat.selected_worst_train_mae_bpm == pytest.approx(
        result.selected_worst_train_mae_bpm
    )
    assert replay_calls == [result.selected_candidate_id]
    assert repeat.cache_summary.read_text(encoding="utf-8") == cache_before
    assert repeat.candidate_history.read_bytes() == history_before

    with pytest.raises(K1DriverIdentityConflictError):
        run_k1_fold_study(
            replace(config, neighborhood_budget=29),
            runtime=KFoldRuntime(
                training_records=(
                    _training_runtime(
                        record=_record("xiezi-1", "a"),
                        offset=0.0,
                    ),
                    _training_runtime(
                        record=_record("xiezi-2", "b"),
                        offset=0.5,
                    ),
                ),
                heldout_record=heldout,
                replay_heldout=replay,
            ),
        )


def test_k1_retries_only_infrastructure_failed_terminal_replay(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    heldout = _record("xiezi-3", "c")
    replay_attempts = 0

    def replay(_context) -> FrozenReplayOutcome:
        nonlocal replay_attempts
        replay_attempts += 1
        if replay_attempts == 1:
            raise ReplayInfrastructureError("solver_timeout")
        return FrozenReplayOutcome.success(
            metrics={"reliable_motion_final_mae_bpm": 4.5},
            artifact_sha256s={
                "hf": _sha("1"),
                "reset_fft": _sha("2"),
                "acc": _sha("3"),
            },
        )

    runtime = KFoldRuntime(
        training_records=(
            _training_runtime(
                record=_record("xiezi-1", "a"),
                offset=0.0,
            ),
            _training_runtime(
                record=_record("xiezi-2", "b"),
                offset=0.5,
            ),
        ),
        heldout_record=heldout,
        replay_heldout=replay,
    )
    first = run_k1_fold_study(config, runtime=runtime)
    selection_hash = json.loads(
        first.selection_receipt.read_text(encoding="utf-8")
    )["selection_hash"]
    assert first.replay_status == "infrastructure_failed"

    recovered = run_k1_fold_study(config, runtime=runtime)

    assert recovered.replay_status == "success"
    assert replay_attempts == 2
    assert json.loads(
        recovered.selection_receipt.read_text(encoding="utf-8")
    )["selection_hash"] == selection_hash


def test_k1_completed_recovery_rejects_tampered_terminal_artifacts(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    runtime = KFoldRuntime(
        training_records=(
            _training_runtime(
                record=_record("xiezi-1", "a"),
                offset=0.0,
            ),
            _training_runtime(
                record=_record("xiezi-2", "b"),
                offset=0.5,
            ),
        ),
        heldout_record=_record("xiezi-3", "c"),
        replay_heldout=lambda _context: FrozenReplayOutcome.success(
            metrics={"reliable_motion_final_mae_bpm": 4.5},
            artifact_sha256s={
                "hf": _sha("1"),
                "reset_fft": _sha("2"),
                "acc": _sha("3"),
            },
        ),
    )
    result = run_k1_fold_study(config, runtime=runtime)
    params_before = result.selected_params.read_text(encoding="utf-8")
    params = json.loads(params_before)
    params["actual_params"]["fs_target"] = 25
    result.selected_params.write_text(
        json.dumps(params, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(K1AuditIntegrityError):
        run_k1_fold_study(config, runtime=runtime)

    result.selected_params.write_text(
        params_before,
        encoding="utf-8",
    )
    with result.training_metrics.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write("\ntruncated")

    with pytest.raises(K1AuditIntegrityError):
        run_k1_fold_study(config, runtime=runtime)


@pytest.mark.parametrize(
    ("config_factory", "run_study", "manifest_name"),
    (
        (_config, run_k1_fold_study, "k1_fold_manifest.json"),
        (_k3_config, run_k3_fold_study, "k3_fold_manifest.json"),
    ),
)
def test_robust_fold_fails_closed_before_heldout_when_all_candidates_unsafe(
    tmp_path,
    config_factory,
    run_study,
    manifest_name,
) -> None:
    config = config_factory(tmp_path)
    replay_called = False

    def unsafe_runtime(
        record: RecordIdentity,
    ) -> KFoldTrainingRecordRuntime:
        runtime = _training_runtime(record=record, offset=0.0)

        def solve(_candidate) -> CandidateSolveOutcome:
            return CandidateSolveOutcome.valid(
                _solver_result(),
                _metric(
                    final_motion=8.0,
                    reset_motion=5.0,
                ),
            )

        return KFoldTrainingRecordRuntime(
            identity=runtime.identity,
            run_config=runtime.run_config,
            solve_candidate=solve,
            render_selected=runtime.render_selected,
        )

    def replay(_context) -> FrozenReplayOutcome:
        nonlocal replay_called
        replay_called = True
        raise AssertionError("不应回放留出记录")

    with pytest.raises(
        RuntimeError,
        match="no_safe_shared_candidate",
    ):
        run_study(
            config,
            runtime=KFoldRuntime(
                training_records=(
                    unsafe_runtime(_record("xiezi-1", "a")),
                    unsafe_runtime(_record("xiezi-2", "b")),
                ),
                heldout_record=_record("xiezi-3", "c"),
                replay_heldout=replay,
            ),
        )

    assert replay_called is False
    failure = json.loads(
        config.output_dir.joinpath(
            "failure_classification.json"
        ).read_text(encoding="utf-8")
    )
    assert failure["failure_reason"] == "no_safe_shared_candidate"
    assert not config.output_dir.joinpath(
        "selection_receipt.json"
    ).exists()
    assert config.output_dir.joinpath(manifest_name).is_file()


def test_k1_rejects_misaligned_neighborhood_audit_on_recovery(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    heldout = _record("xiezi-3", "c")
    runtime = KFoldRuntime(
        training_records=(
            _training_runtime(
                record=_record("xiezi-1", "a"),
                offset=0.0,
            ),
            _training_runtime(
                record=_record("xiezi-2", "b"),
                offset=0.5,
            ),
        ),
        heldout_record=heldout,
        replay_heldout=lambda _context: FrozenReplayOutcome.success(
            metrics={"reliable_motion_final_mae_bpm": 4.5},
            artifact_sha256s={
                "hf": _sha("1"),
                "reset_fft": _sha("2"),
                "acc": _sha("3"),
            },
        ),
    )
    run_k1_fold_study(config, runtime=runtime)
    for name in (
        "selection_receipt.json",
        "replay_receipt.json",
        "k1_fold_manifest.json",
    ):
        config.output_dir.joinpath(name).unlink()
    audit_path = config.output_dir.joinpath(
        "trial_audit",
        "neighborhood-000.json",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["candidate_id"] = "tampered-candidate"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(K1AuditIntegrityError):
        run_k1_fold_study(config, runtime=runtime)


def test_k1_rejects_audit_metrics_that_conflict_with_outcomes(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    runtime = KFoldRuntime(
        training_records=(
            _training_runtime(
                record=_record("xiezi-1", "a"),
                offset=0.0,
            ),
            _training_runtime(
                record=_record("xiezi-2", "b"),
                offset=0.5,
            ),
        ),
        heldout_record=_record("xiezi-3", "c"),
        replay_heldout=lambda _context: FrozenReplayOutcome.success(
            metrics={"reliable_motion_final_mae_bpm": 4.5},
            artifact_sha256s={
                "hf": _sha("1"),
                "reset_fft": _sha("2"),
                "acc": _sha("3"),
            },
        ),
    )
    run_k1_fold_study(config, runtime=runtime)
    for name in (
        "selection_receipt.json",
        "replay_receipt.json",
        "k1_fold_manifest.json",
    ):
        config.output_dir.joinpath(name).unlink()
    audit_path = config.output_dir.joinpath(
        "trial_audit",
        "band-evidence-000.json",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["training_outcomes"][0]["formal_metrics"][
        "reliable_motion_final_mae_bpm"
    ] += 1.0
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(
        K1AuditIntegrityError,
        match="robust_evidence",
    ):
        run_k1_fold_study(config, runtime=runtime)
