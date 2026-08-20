from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ppg_hr.v2.recovery_profile_baseline import (
    BaselineRebuildError,
    rebuild_independent_bo_baseline,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_archive_fixture(tmp_path: Path) -> tuple[Path, Path]:
    formal_root = tmp_path / "formal"
    source_root = tmp_path / "source"
    formal_root.mkdir()
    source_root.mkdir()
    records: list[dict[str, object]] = []
    scenes = ("xiezi", "jianpan", "run", "kaihe")
    for record_index in range(12):
        scene = scenes[record_index // 3]
        sample_id = f"{scene}{record_index % 3 + 1}_LYX_test"
        sensor_path = source_root / f"{sample_id}.csv"
        sensor_path.write_text("frozen-sensor-input\n", encoding="utf-8")
        reference_path = source_root / f"{sample_id}_HR_ref.csv"
        with reference_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["elapsed_seconds", "hr_bpm"])
            writer.writerows((second, 100.0) for second in range(30))

        candidate_id = f"physical_v1:test-{record_index:02d}"
        actual_params = {
            "analysis_scope": "full",
            "smooth_win_len": 5,
            "time_bias": 5.0,
        }
        requested_params = {"fs_target": 50}
        fixed_params = {"smooth_win_len": 5, "time_bias": 5.0}
        solver_identity = {
            "data_sha256": _sha256(sensor_path),
            "reference_sha256": _sha256(reference_path),
            "git_commit": "a" * 40,
            "run_config": {"analysis_scope": "full"},
            "candidate_id": candidate_id,
            "space_name": "physical_v1",
            "requested_params": requested_params,
            "actual_params": actual_params,
            "fixed_params": fixed_params,
            "reference_groups_order": ["HF"],
            "metric_contract_version": "lyx_bo_formal_metric_v1",
        }
        cache_key = hashlib.sha256(
            json.dumps(
                solver_identity,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        record_root = formal_root / "records" / sample_id
        cache_entry = record_root / "cache" / cache_key
        cache_entry.mkdir(parents=True)
        selected_path = record_root / "selected_candidate.json"
        selected_path.write_text(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "requested_params": requested_params,
                    "actual_params": actual_params,
                    "fixed_params": fixed_params,
                }
            ),
            encoding="utf-8",
        )
        selection_audit_path = record_root / "selection_audit.json"
        selection_audit_path.write_text(
            json.dumps(
                {
                    "candidate_id": candidate_id,
                    "cache_key": cache_key,
                }
            ),
            encoding="utf-8",
        )
        centers = np.arange(12, dtype=float)
        hr = np.column_stack(
            [
                centers,
                np.full(12, 999.0),
                np.full(12, 102.0),
                np.full(12, 101.0),
                np.ones(12),
                np.ones(12),
            ]
        )
        np.savez_compressed(cache_entry / "solver_result.npz", HR=hr)
        (cache_entry / "reservation.json").write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "identity": solver_identity,
                }
            ),
            encoding="utf-8",
        )
        (cache_entry / "outcome.json").write_text(
            json.dumps(
                {
                    "status": "valid",
                    "failure_reason": "",
                    "diagnostics": {},
                    "formal_metrics": {
                        "metric_contract_version": "lyx_bo_formal_metric_v1",
                        "final_method": "LMS+H",
                        "reset_fft_method": "reset FFT",
                        "base_full_window_count": 12,
                        "base_motion_window_count": 12,
                        "classic_motion_window_count": 12,
                        "base_full_final_finite_count": 12,
                        "base_motion_final_finite_count": 12,
                        "base_motion_reset_fft_finite_count": 12,
                        "base_motion_common_finite_count": 12,
                        "classic_motion_final_finite_count": 12,
                        "classic_motion_reset_fft_finite_count": 12,
                        "classic_motion_common_finite_count": 12,
                        "base_full_window_sha256": "a" * 64,
                        "base_motion_window_sha256": "b" * 64,
                        "classic_motion_window_sha256": "b" * 64,
                        "full_final_mae_bpm": 1.0,
                        "reliable_motion_final_mae_bpm": 1.0,
                        "reliable_motion_reset_fft_mae_bpm": 2.0,
                        "classic_motion_final_mae_bpm": 1.0,
                        "classic_motion_reset_fft_mae_bpm": 2.0,
                    },
                    "solver_result": {
                        "err_stats": {},
                        "metadata": {
                            "analysis_scope": "full",
                            "adaptive_filter": "lms",
                            "reference_groups_order": ["HF"],
                            "time_bias": 5.0,
                        },
                        "window_table": [
                            {
                                "window_idx": index,
                                "center_s": float(center),
                                "reliable": True,
                            }
                            for index, center in enumerate(centers)
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        records.append(
            {
                "sample_id": sample_id,
                "scene": scene,
                "candidate_id": candidate_id,
                "cache_entry": str(cache_entry.relative_to(formal_root)),
                "selected_candidate": str(
                    selected_path.relative_to(formal_root)
                ),
                "selection_audit": str(
                    selection_audit_path.relative_to(formal_root)
                ),
                "sensor_path": str(sensor_path),
                "reference_path": str(reference_path),
                "data_sha256": _sha256(sensor_path),
                "reference_sha256": _sha256(reference_path),
                "method_names": ["reset FFT", "LMS+H"],
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": (
                    "lyx_recovery_profile_baseline_manifest_v1"
                ),
                "parent_experiment_id": "synthetic-formal:s21",
                "archive_git_commit": "a" * 40,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return formal_root, manifest_path


def test_rebuild_independent_bo_baseline_writes_complete_atomic_artifact(
    tmp_path: Path,
) -> None:
    formal_root, manifest_path = _write_archive_fixture(tmp_path)
    output_dir = tmp_path / "baseline"

    receipt = rebuild_independent_bo_baseline(
        formal_root=formal_root,
        record_manifest=manifest_path,
        output_dir=output_dir,
    )

    assert receipt["status"] == "complete"
    assert receipt["record_count"] == 12
    assert receipt["scene_counts"] == {
        "jianpan": 3,
        "kaihe": 3,
        "run": 3,
        "xiezi": 3,
    }
    assert receipt["parent_experiment_id"] == "synthetic-formal:s21"
    assert receipt["archive_git_commit"] == "a" * 40
    assert len(receipt["evaluation_code_sha256"]) == 64
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "contract_receipt.json",
        "record_metrics.csv",
        "record_metrics.json",
        "summary.json",
    ]
    rows = json.loads(
        (output_dir / "record_metrics.json").read_text(encoding="utf-8")
    )["records"]
    assert len(rows) == 12
    assert rows[0]["metrics"]["metric_contract_version"] == (
        "lyx_recovery_profile_metric_v1"
    )
    assert rows[0]["actual_params"]["smooth_win_len"] == 5
    assert len(rows[0]["selected_candidate_sha256"]) == 64
    assert len(rows[0]["selection_audit_sha256"]) == 64
    assert len(rows[0]["solver_identity_sha256"]) == 64
    assert len(rows[0]["solver_outcome_sha256"]) == 64
    assert len(rows[0]["solver_result_sha256"]) == 64


def test_rebuild_independent_bo_baseline_fails_closed_without_partial_output(
    tmp_path: Path,
) -> None:
    formal_root, manifest_path = _write_archive_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["records"][7]["reference_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "baseline"

    with pytest.raises(BaselineRebuildError, match="reference_sha256"):
        rebuild_independent_bo_baseline(
            formal_root=formal_root,
            record_manifest=manifest_path,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_rebuild_independent_bo_baseline_rejects_wrong_cache_identity(
    tmp_path: Path,
) -> None:
    formal_root, manifest_path = _write_archive_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_entry = formal_root / payload["records"][0]["cache_entry"]
    reservation_path = cache_entry / "reservation.json"
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    reservation["identity"]["candidate_id"] = "physical_v1:wrong"
    reservation_path.write_text(json.dumps(reservation), encoding="utf-8")

    with pytest.raises(BaselineRebuildError, match="完整 identity"):
        rebuild_independent_bo_baseline(
            formal_root=formal_root,
            record_manifest=manifest_path,
            output_dir=tmp_path / "baseline",
        )


def test_rebuild_independent_bo_baseline_rejects_tampered_identity_payload(
    tmp_path: Path,
) -> None:
    formal_root, manifest_path = _write_archive_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_entry = formal_root / payload["records"][0]["cache_entry"]
    reservation_path = cache_entry / "reservation.json"
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    reservation["identity"]["run_config"]["analysis_scope"] = "motion"
    reservation_path.write_text(json.dumps(reservation), encoding="utf-8")

    with pytest.raises(BaselineRebuildError, match="完整 identity"):
        rebuild_independent_bo_baseline(
            formal_root=formal_root,
            record_manifest=manifest_path,
            output_dir=tmp_path / "baseline",
        )
