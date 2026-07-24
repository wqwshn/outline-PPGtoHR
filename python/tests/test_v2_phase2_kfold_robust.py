from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.bo_space_generalization import (
    CandidateSolveOutcome,
    FormalMetricResult,
    SeedSearchBudget,
)
from ppg_hr.v2.phase2_kfold_robust import (
    K1AuditIntegrityError,
    K1DriverIdentityConflictError,
    K1FoldConfig,
    run_k1_fold_study,
)
from ppg_hr.v2.phase2_kfold_runtime import (
    ClassicPlotArtifact,
    KFoldRuntime,
    KFoldTrainingRecordRuntime,
)
from ppg_hr.v2.phase2_receipt import (
    FrozenReplayOutcome,
    RecordIdentity,
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


def test_k1_fails_closed_before_heldout_when_all_candidates_unsafe(
    tmp_path,
) -> None:
    config = _config(tmp_path)
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
        run_k1_fold_study(
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
