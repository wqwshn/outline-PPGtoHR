from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.bo_space_generalization import (
    CandidateSolveOutcome,
    FormalMetricResult,
    build_bo_search_space,
)
from ppg_hr.v2.phase2_kfold_enumeration import (
    K2FoldConfig,
    run_k2_fold_study,
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
    unsafe: bool = False,
) -> KFoldTrainingRecordRuntime:
    def solve(candidate) -> CandidateSolveOutcome:
        distance = float(sum(candidate.coordinate)) / 100.0
        final = 3.0 + offset + distance
        reset = final - (3.0 if unsafe else 1.0)
        return CandidateSolveOutcome.valid(
            _solver_result(),
            _metric(
                final_motion=final,
                reset_motion=reset,
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


def _runtime(
    *,
    replay_calls: list[str],
) -> KFoldRuntime:
    heldout = _record("xiezi-3", "c")

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

    return KFoldRuntime(
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


def _config(output_dir: Path) -> K2FoldConfig:
    return K2FoldConfig(
        output_dir=output_dir,
        scene="xiezi",
        fold=0,
        git_commit="test-commit",
    )


def test_k2_enumerates_108_and_is_order_invariant(tmp_path) -> None:
    canonical_calls: list[str] = []
    canonical = run_k2_fold_study(
        _config(tmp_path / "canonical"),
        runtime=_runtime(replay_calls=canonical_calls),
    )
    space = build_bo_search_space("legacy_reduced_v1")
    reverse_calls: list[str] = []
    reversed_result = run_k2_fold_study(
        _config(tmp_path / "reversed"),
        runtime=_runtime(replay_calls=reverse_calls),
        enumeration_order=tuple(
            reversed(
                [
                    candidate.candidate_id
                    for candidate in space.candidates
                ]
            )
        ),
    )

    assert canonical.arm == "K2"
    assert canonical.enumeration_count == 108
    assert canonical.coverage_ratio == pytest.approx(1.0)
    assert canonical.selected_candidate_id == (
        reversed_result.selected_candidate_id
    )
    assert canonical.selected_worst_train_mae_bpm == pytest.approx(
        reversed_result.selected_worst_train_mae_bpm
    )
    assert canonical_calls == [canonical.selected_candidate_id]
    assert reverse_calls == [reversed_result.selected_candidate_id]

    with canonical.candidate_history.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 108
    assert len({row["candidate_id"] for row in rows}) == 108
    assert {row["stage"] for row in rows} == {"enumeration"}
    assert {row["lane"] for row in rows} == {"enumeration"}
    assert {
        float(row["fixed_time_bias"])
        for row in rows
    } == {5.0}
    assert {
        int(float(row["fixed_smooth_win_len"]))
        for row in rows
    } == {5}
    assert all(row["requested_time_bias"] == "" for row in rows)
    assert all(
        row["requested_smooth_win_len"] == ""
        for row in rows
    )

    receipt = json.loads(
        canonical.selection_receipt.read_text(encoding="utf-8")
    )
    assert receipt["evidence"]["space_name"] == (
        "legacy_reduced_v1"
    )
    assert len(receipt["evidence"]["study_identities"]) == 1
    assert receipt["evidence"]["budget"] == {
        "lane_unique_budget": 108,
        "requested_global_unique_budget": 108,
        "actual_global_unique_count": 108,
        "requested_neighborhood_budget": 0,
        "actual_neighborhood_count": 0,
    }
    neighborhood = json.loads(
        canonical.neighborhood_evidence.read_text(
            encoding="utf-8"
        )
    )
    assert neighborhood["coverage"] == {
        "enumerated_candidate_count": 108,
        "space_candidate_count": 108,
        "coverage_ratio": 1.0,
        "additional_neighborhood_candidate_count": 0,
    }
    assert neighborhood["plan"][
        "candidate_ids_to_evaluate"
    ] == []
    assert neighborhood["plan"][
        "truncated_primary_center_ids"
    ] == []
    manifest = json.loads(
        canonical.manifest.read_text(encoding="utf-8")
    )
    assert manifest["causal_claim_allowed"] is False
    assert manifest["comparison_scope"] == (
        "operational_workflow_only"
    )

    hashes_before = {
        path.name: path.read_bytes()
        for path in (
            canonical.cache_summary,
            canonical.candidate_history,
            canonical.selection_receipt,
            canonical.neighborhood_evidence,
            canonical.manifest,
        )
    }
    repeated = run_k2_fold_study(
        _config(tmp_path / "canonical"),
        runtime=_runtime(replay_calls=canonical_calls),
    )
    assert repeated.selected_candidate_id == (
        canonical.selected_candidate_id
    )
    assert canonical_calls == [canonical.selected_candidate_id]
    assert {
        path.name: path.read_bytes()
        for path in (
            repeated.cache_summary,
            repeated.candidate_history,
            repeated.selection_receipt,
            repeated.neighborhood_evidence,
            repeated.manifest,
        )
    } == hashes_before


def test_k2_fails_closed_when_all_108_candidates_are_unsafe(
    tmp_path,
) -> None:
    replay_called = False

    def replay(_context) -> FrozenReplayOutcome:
        nonlocal replay_called
        replay_called = True
        raise AssertionError("不应回放留出记录")

    runtime = KFoldRuntime(
        training_records=(
            _training_runtime(
                record=_record("xiezi-1", "a"),
                offset=0.0,
                unsafe=True,
            ),
            _training_runtime(
                record=_record("xiezi-2", "b"),
                offset=0.5,
                unsafe=True,
            ),
        ),
        heldout_record=_record("xiezi-3", "c"),
        replay_heldout=replay,
    )

    with pytest.raises(
        RuntimeError,
        match="no_safe_shared_candidate",
    ):
        run_k2_fold_study(
            _config(tmp_path / "unsafe"),
            runtime=runtime,
        )

    assert replay_called is False
    assert not (tmp_path / "unsafe" / "selection_receipt.json").exists()
    with (
        tmp_path
        / "unsafe"
        / "candidate_history.csv"
    ).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 108
    assert all(row["eligible"] == "False" for row in rows)
