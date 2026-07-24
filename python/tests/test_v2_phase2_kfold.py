from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import ppg_hr.v2.phase2_kfold as phase2_kfold
import ppg_hr.v2.phase2_kfold_runtime as phase2_kfold_runtime
from ppg_hr.v2.bo_space_generalization import (
    CandidateSolveOutcome,
    FormalMetricResult,
    SeedSearchBudget,
)
from ppg_hr.v2.phase2_kfold import (
    ClassicPlotArtifact,
    K0FoldConfig,
    K0FoldRuntime,
    K0RecordInput,
    K0TrainingRecordRuntime,
    build_k0_default_runtime,
    run_k0_fold_study,
)
from ppg_hr.v2.phase2_receipt import (
    FrozenReplayOutcome,
    RecordIdentity,
)
from ppg_hr.v2.solver import V2SolverResult
from ppg_hr.v2.types import V2RunConfig


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


def _metric(full_mae: float) -> FormalMetricResult:
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
        full_final_mae_bpm=full_mae,
        reliable_motion_final_mae_bpm=full_mae + 1.0,
        reliable_motion_reset_fft_mae_bpm=full_mae + 2.0,
        classic_motion_final_mae_bpm=full_mae + 1.5,
        classic_motion_reset_fft_mae_bpm=full_mae + 2.5,
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
    tmp_path: Path,
    *,
    record: RecordIdentity,
    offset: float,
    solve_counts: dict[str, int],
) -> K0TrainingRecordRuntime:
    def solve(candidate) -> CandidateSolveOutcome:
        solve_counts[record.record_id] = solve_counts.get(record.record_id, 0) + 1
        return CandidateSolveOutcome.valid(
            _solver_result(),
            _metric(float(sum(candidate.coordinate)) + offset),
        )

    def render(_candidate, _outcome, output_dir: Path) -> ClassicPlotArtifact:
        figure = output_dir / f"{record.record_id}.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        figure.write_bytes(b"png")
        return ClassicPlotArtifact(
            figure_png=figure,
            method_names=("reset FFT", "LMS+H", "LMS+A"),
        )

    return K0TrainingRecordRuntime(
        identity=record,
        run_config={"record_id": record.record_id, "analysis_scope": "full"},
        solve_candidate=solve,
        render_selected=render,
    )


def _config(tmp_path: Path) -> K0FoldConfig:
    return K0FoldConfig(
        output_dir=tmp_path / "k0",
        scene="xiezi",
        fold=0,
        git_commit="test-commit",
        budget=SeedSearchBudget(
            lane_seeds=(42, 43, 44),
            lane_unique_budget=1,
            global_unique_budget=3,
            n_startup_trials=1,
            objective_version="phase2_k0_mean_full_final_v1",
        ),
    )


def test_k0_fold_freezes_mean_training_selection_before_heldout_replay(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    solve_counts: dict[str, int] = {}
    training = (
        _training_runtime(
            tmp_path,
            record=_record("xiezi-1", "a"),
            offset=1.0,
            solve_counts=solve_counts,
        ),
        _training_runtime(
            tmp_path,
            record=_record("xiezi-2", "b"),
            offset=3.0,
            solve_counts=solve_counts,
        ),
    )
    heldout = _record("xiezi-3", "c")
    replay_calls: list[str] = []

    def replay(context) -> FrozenReplayOutcome:
        assert (config.output_dir / "selection_receipt.json").is_file()
        payload = json.loads(
            (config.output_dir / "selection_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        assert "test_metrics" not in payload["evidence"]
        assert context.heldout_record == heldout
        replay_calls.append(context.candidate_id)
        return FrozenReplayOutcome.success(
            metrics={
                "full_final_mae_bpm": 4.0,
                "reliable_motion_final_mae_bpm": 5.0,
            },
            artifact_sha256s={
                "hf": _sha("1"),
                "reset_fft": _sha("2"),
                "acc": _sha("3"),
            },
        )

    result = run_k0_fold_study(
        config,
        runtime=K0FoldRuntime(
            training_records=training,
            heldout_record=heldout,
            replay_heldout=replay,
        ),
    )

    assert result.arm == "K0"
    assert result.flow_label == "完整旧空间简单平均流程基线"
    assert result.selection_receipt.is_file()
    assert result.replay_receipt.is_file()
    assert result.replay_status == "success"
    assert replay_calls == [result.selected_candidate_id]
    assert solve_counts == {"xiezi-1": 3, "xiezi-2": 3}
    assert len(result.training_plots) == 2
    assert all(path.is_file() for path in result.training_plots)

    with result.candidate_history.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        history = list(csv.DictReader(handle))
    assert history
    assert {
        "stage",
        "train_0_full_final_mae_bpm",
        "train_1_full_final_mae_bpm",
        "mean_train_full_final_mae_bpm",
        "cache_hit_train_0",
        "cache_hit_train_1",
    } <= set(history[0])
    assert {row["stage"] for row in history} <= {"search", "fill"}
    for row in history:
        if row["metric_valid"] == "True":
            expected_mean = (
                float(row["train_0_full_final_mae_bpm"])
                + float(row["train_1_full_final_mae_bpm"])
            ) / 2.0
            assert float(row["mean_train_full_final_mae_bpm"]) == pytest.approx(
                expected_mean
            )

    receipt_payload = json.loads(
        result.selection_receipt.read_text(encoding="utf-8")
    )
    assert receipt_payload["evidence"]["arm"] == "K0"
    assert (
        receipt_payload["evidence"]["training_metrics"][
            "mean_train_mae_bpm"
        ]
        == pytest.approx(result.selected_mean_train_mae_bpm)
    )
    assert (
        receipt_payload["evidence"]["neighborhood_evidence"]["status"]
        == "not_required"
    )
    failure_payload = json.loads(
        result.failure_classification.read_text(encoding="utf-8")
    )
    assert failure_payload["replay_status"] == "success"
    assert failure_payload["invalid_candidate_count"] == 0


def test_k0_plot_title_names_fold_roles_and_frozen_params() -> None:
    title = phase2_kfold._k0_plot_title(
        training_record_ids=("run-1", "run-2"),
        heldout_record_id="run-3",
        view_role="test",
        view_record_id="run-3",
        actual_params={
            "fs_target": 50,
            "max_order": 20,
            "lms_mu_base": 0.012,
            "smooth_win_len": 5,
            "spec_penalty_width": 0.2,
            "time_bias": 5.0,
        },
    )

    assert "K0" in title
    assert "train: run-1 + run-2" in title
    assert "test: run-3" in title
    assert "view: test run-3" in title
    assert "fs=50Hz" in title
    assert "order=20taps" in title
    assert "mu=0.012" in title
    assert "smooth=5" in title
    assert "width=0.2Hz" in title
    assert "bias=5s" in title


def test_classic_plot_artifact_rejects_missing_acc_curve(tmp_path) -> None:
    figure = tmp_path / "classic.png"
    figure.write_bytes(b"png")

    with pytest.raises(ValueError, match=r"LMS\+A"):
        ClassicPlotArtifact(
            figure_png=figure,
            method_names=("reset FFT", "LMS+H"),
        )


def test_default_k0_runtime_defers_loading_heldout_data_until_replay(
    tmp_path,
    monkeypatch,
) -> None:
    records = []
    for index in range(3):
        data_path = tmp_path / f"record-{index}.csv"
        reference_path = tmp_path / f"record-{index}-ref.csv"
        data_path.write_text(f"data-{index}", encoding="utf-8")
        reference_path.write_text(f"ref-{index}", encoding="utf-8")
        records.append(
            K0RecordInput(
                record_id=f"xiezi-{index + 1}",
                data_path=data_path,
                reference_path=reference_path,
            )
        )
    loaded_data_paths: list[Path] = []

    def load_dataset(data_path, _reference_path, *, fs_origin):
        assert fs_origin == 100
        loaded_data_paths.append(Path(data_path))
        return object()

    monkeypatch.setattr(
        phase2_kfold_runtime,
        "load_v2_dataset",
        load_dataset,
    )
    runtime = build_k0_default_runtime(
        base_config=V2RunConfig(
            data_path=records[0].data_path,
            ref_path=records[0].reference_path,
        ),
        training_records=(records[0], records[1]),
        heldout_record=records[2],
        output_dir=tmp_path / "default-runtime",
    )

    assert loaded_data_paths == [
        records[0].data_path,
        records[1].data_path,
    ]
    assert runtime.heldout_record.record_id == "xiezi-3"
