from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from ppg_hr.v2.reference_arm_source import (
    SourceContract,
    SourceIdentityError,
    materialise_source_snapshot,
    physical4d_coordinates,
)

FINAL_ROWS = (
    "data/experiments/lyx_tiaosheng_curated_panel_threefold_summary_20260824/"
    "final/eight_scene_performance_rows.csv"
)
INPUT_MANIFEST = (
    "data/experiments/lyx_eight_scene_identity_blind_unified_physical4d_response_20260820/"
    "input_manifest.json"
)
PARTITION_ROOT = (
    "data/experiments/lyx_eight_scene_identity_blind_unified_physical4d_response_20260820/"
    "response/partitions"
)


@dataclass(frozen=True)
class SyntheticSource:
    repo: Path
    commit: str
    contract: SourceContract

    def commit_corrupt_final_rows(self) -> str:
        path = self.repo / FINAL_ROWS
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        _git(self.repo, "add", FINAL_ROWS)
        _git(self.repo, "commit", "-m", "corrupt final rows")
        return _git(self.repo, "rev-parse", "HEAD")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8").strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_synthetic_source(tmp_path: Path) -> SyntheticSource:
    repo = tmp_path / "source-repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "codex@example.invalid")
    _git(repo, "config", "user.name", "Codex Test")

    coordinates = physical4d_coordinates()
    input_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    for scene_index in range(8):
        scene = f"scene{scene_index + 1}"
        for record_index in range(3):
            record_id = f"{scene}_record{record_index + 1}"
            data_relative = f"data/raw/{record_id}.csv"
            ref_relative = f"data/raw/{record_id}_HR_ref.csv"
            data_path = repo / data_relative
            ref_path = repo / ref_relative
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.write_text("time,green\n0,1\n", encoding="utf-8")
            ref_path.write_text("time,hr\n5,70\n", encoding="utf-8")
            input_rows.append(
                {
                    "scene": scene,
                    "record_id": record_id,
                    "data_relative_path": data_relative,
                    "ref_relative_path": ref_relative,
                    "data_sha256": _sha256(data_path),
                    "ref_sha256": _sha256(ref_path),
                }
            )

            window_keys = [
                {"window_idx": index, "center_s": float(5 + index)} for index in range(3)
            ]
            window_sha = _semantic_sha256(window_keys)
            prior_comparison_indices = [0, 2]
            window_mask_sha = _semantic_sha256(prior_comparison_indices)
            partition_rows = [
                {
                    "scene": scene,
                    "record_id": record_id,
                    "coordinate_id": coordinate.coordinate_id,
                    "fs_target": coordinate.fs_target_hz,
                    "memory_ms": coordinate.memory_ms,
                    "mu_base": coordinate.mu_base,
                    "exclusion_half_width_bpm": coordinate.exclusion_half_width_bpm,
                    "effective_window_count": 3,
                    "first_effective_window_idx": 0,
                    "last_effective_window_idx": 2,
                    "first_effective_center_s": 5.0,
                    "last_effective_center_s": 7.0,
                    "effective_window_keys_sha256": window_sha,
                    "mae_full_bpm": 1.0 + coordinate.coordinate_index / 1000.0,
                }
                for coordinate in coordinates
            ]
            _write_csv(
                repo / PARTITION_ROOT / scene / record_id / "cell_rows.csv",
                partition_rows,
            )
            selected = coordinates[record_index]
            final_rows.append(
                {
                    "scene": scene,
                    "record_id": record_id,
                    "fold_id": f"{scene}::{record_id}",
                    "coordinate_id": selected.coordinate_id,
                    "candidate_rule": "frozen_hf_selector",
                    "hf_fixed_5s_mae_bpm": 1.0 + selected.coordinate_index / 1000.0,
                    "hf_common_window_count": len(prior_comparison_indices),
                    "hf_common_window_mask_sha256": window_mask_sha,
                    "ref_sha256": _sha256(ref_path),
                }
            )

    manifest_path = repo / INPUT_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"records": input_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(repo / FINAL_ROWS, final_rows)
    final_sha = _sha256(repo / FINAL_ROWS)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "synthetic frozen evidence")
    commit = _git(repo, "rev-parse", "HEAD")
    contract = SourceContract(
        source_final_rows_sha256=final_sha,
        expected_record_count=24,
        expected_scene_count=8,
        expected_hf_cell_count=7200,
        time_bias_s=5.0,
    )
    return SyntheticSource(repo=repo, commit=commit, contract=contract)


def test_physical4d_coordinates_are_the_frozen_300_row_rectangle() -> None:
    rows = physical4d_coordinates()
    assert len(rows) == 300
    assert [row.coordinate_index for row in rows] == list(range(300))
    assert rows[0].coordinate_id == "physical4d:fs025:m040:mu0006:w003"
    assert rows[-1].coordinate_id == "physical4d:fs100:m200:mu0016:w018"


def test_source_snapshot_requires_eight_scenes_three_records_and_300_hf_rows_each(
    tmp_path: Path,
) -> None:
    source = make_synthetic_source(tmp_path)
    snapshot = materialise_source_snapshot(
        source.repo,
        tmp_path / "snapshot",
        source.commit,
        contract=source.contract,
    )
    assert len(snapshot.records) == 24
    assert len(snapshot.hf_cells) == 7200
    assert len(snapshot.hf_selections) == 24
    assert {len(snapshot.timelines[row.record_id].window_keys) for row in snapshot.records} == {3}
    assert {selection.prior_common_window_count for selection in snapshot.hf_selections} == {2}
    assert {selection.prior_common_window_mask_sha256 for selection in snapshot.hf_selections} == {
        _semantic_sha256([0, 2])
    }
    assert (tmp_path / "snapshot" / "source_snapshot.json").is_file()


def test_source_snapshot_rejects_hash_mismatched_git_payload(tmp_path: Path) -> None:
    source = make_synthetic_source(tmp_path)
    corrupt_commit = source.commit_corrupt_final_rows()
    with pytest.raises(SourceIdentityError, match="source_final_rows_sha256"):
        materialise_source_snapshot(
            source.repo,
            tmp_path / "snapshot",
            corrupt_commit,
            contract=source.contract,
        )


def test_source_snapshot_reads_commit_not_dirty_worktree(tmp_path: Path) -> None:
    source = make_synthetic_source(tmp_path)
    final_path = source.repo / FINAL_ROWS
    final_path.write_text("dirty working copy\n", encoding="utf-8")
    snapshot = materialise_source_snapshot(
        source.repo,
        tmp_path / "snapshot",
        source.commit,
        contract=source.contract,
    )
    assert len(snapshot.hf_cells) == 7200
