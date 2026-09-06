from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import ppg_hr.v2.reference_arm_experiment as experiment
from ppg_hr.v2.reference_arm_experiment import (
    ReferenceArmRunner,
    build_reference_arm_calls,
    build_run_config,
    config_diff,
)
from ppg_hr.v2.reference_arm_ledger import CompactResponseLedger
from ppg_hr.v2.reference_arm_source import (
    EvaluationTimeline,
    FrozenHFSelection,
    RecordIdentity,
    ReferenceArmSourceSnapshot,
    physical4d_coordinates,
)
from ppg_hr.v2.solver import V2SolverResult


def snapshot_fixture(tmp_path: Path) -> ReferenceArmSourceSnapshot:
    records = []
    timelines = {}
    selections = []
    for scene_index in range(8):
        scene = f"scene{scene_index + 1}"
        scene_ids = [f"{scene}_record{index + 1}" for index in range(3)]
        for record_index, record_id in enumerate(scene_ids):
            data = tmp_path / f"{record_id}.csv"
            ref = tmp_path / f"{record_id}_HR_ref.csv"
            data.write_text("unused\n", encoding="utf-8")
            ref.write_text("elapsed_seconds,hr_bpm\n5,72\n7,82\n", encoding="utf-8")
            records.append(
                RecordIdentity(
                    scene=scene,
                    record_id=record_id,
                    fold_id=f"{scene}::{record_id}",
                    data_path=data,
                    ref_path=ref,
                    data_relative_path=data.name,
                    ref_relative_path=ref.name,
                    data_sha256="a" * 64,
                    ref_sha256="b" * 64,
                )
            )
            timelines[record_id] = EvaluationTimeline(
                record_id=record_id,
                time_bias_s=5.0,
                window_keys=((0, 0.0), (2, 2.0)),
                window_sha256="c" * 64,
            )
            train = tuple(item for item in scene_ids if item != record_id)
            selections.append(
                FrozenHFSelection(
                    route_id="HF",
                    scene=scene,
                    fold_id=f"{scene}::{record_id}",
                    train_record_ids=(train[0], train[1]),
                    holdout_record_id=record_id,
                    coordinate_id=physical4d_coordinates()[record_index].coordinate_id,
                    coordinate_index=record_index,
                    selector_id="frozen",
                    holdout_mae_bpm=1.0,
                    prior_common_window_count=2,
                    prior_common_window_mask_sha256="9" * 64,
                )
            )
    return ReferenceArmSourceSnapshot(
        source_commit="1" * 40,
        records=tuple(records),
        coordinates=physical4d_coordinates(),
        timelines=timelines,
        hf_cells=(),
        hf_selections=tuple(selections),
        semantic_sha256="d" * 64,
    )


def solver_result_with_large_arrays() -> V2SolverResult:
    return V2SolverResult(
        HR=np.asarray(
            [[0.0, 0.0, 0.0, 70.0], [1.0, 0.0, 0.0, 80.0], [2.0, 0.0, 0.0, 90.0]],
            dtype=float,
        ),
        err_stats={"unused": 1.0},
        metadata={"large": np.zeros(1000)},
        window_table=[{"large": np.zeros(1000)}],
    )


def runner_fixture(tmp_path: Path, snapshot: ReferenceArmSourceSnapshot) -> ReferenceArmRunner:
    identity = {
        "experiment_id": "synthetic_reference_arm",
        "source_sha256": snapshot.semantic_sha256,
        "code_sha256": "e" * 64,
    }
    ledger = CompactResponseLedger.create(tmp_path / "ledger.sqlite3", identity)
    return ReferenceArmRunner(
        snapshot=snapshot,
        ledger=ledger,
        experiment_id="synthetic_reference_arm",
        code_sha256="e" * 64,
    )


def test_build_calls_is_exactly_24_by_300_by_two(tmp_path: Path) -> None:
    snapshot = snapshot_fixture(tmp_path)
    calls = build_reference_arm_calls(snapshot, code_sha256="e" * 64)
    assert len(calls) == 14_400
    assert Counter(call.route_id for call in calls) == {"ACC": 7200, "HF_ACC": 7200}
    assert len({call.identity_sha256 for call in calls}) == 14_400


def test_run_config_changes_only_reference_route_and_physical_coordinate(
    tmp_path: Path,
) -> None:
    snapshot = snapshot_fixture(tmp_path)
    acc = build_run_config(snapshot.records[0], snapshot.coordinates[0], "ACC")
    mixed = build_run_config(snapshot.records[0], snapshot.coordinates[0], "HF_ACC")
    assert acc.reference_groups_order == ("ACC",)
    assert mixed.reference_groups_order == ("HF", "ACC")
    assert config_diff(acc, mixed) == {"reference_groups_order"}


def test_runner_releases_full_result_after_writing_one_compact_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = snapshot_fixture(tmp_path)
    result = solver_result_with_large_arrays()
    monkeypatch.setattr(experiment, "solve_v2", lambda _config: result)
    runner = runner_fixture(tmp_path, snapshot)
    runner.run_calls(runner.calls[:1], workers=1)
    assert runner.ledger.complete_count("ACC") == 1
    assert not list(tmp_path.rglob("report-v2.json"))
    assert not list(tmp_path.rglob("*.npz"))


def test_runner_records_technical_error_without_penalty_mae(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = snapshot_fixture(tmp_path)
    broken = solver_result_with_large_arrays()
    broken.HR[2, 3] = np.nan
    monkeypatch.setattr(experiment, "solve_v2", lambda _config: broken)
    runner = runner_fixture(tmp_path, snapshot)
    summary = runner.run_calls(runner.calls[:1], workers=1)
    assert summary.complete == 0
    assert summary.technical_failures == 1
    assert runner.ledger.complete_count() == 0
    event = runner.ledger.connection.execute(
        "SELECT reason_code, detail FROM attempt_events"
    ).fetchone()
    assert event[0] == "nonfinite_final", event[1]


def test_clean_worktree_ignores_git_warning_on_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    warning = "warning: unable to access global ignore file"
    monkeypatch.setattr(
        experiment.subprocess,
        "check_output",
        lambda *args, **kwargs: warning,
    )
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="",
            stderr=warning,
        ),
    )

    experiment._require_clean_worktree(tmp_path)
