from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import ppg_hr.v2.phase2_preflight as preflight
from ppg_hr.v2.phase2_preflight import (
    DATA_REUSE_REASON,
    EVIDENCE_LEVEL,
    Phase2PreflightConfig,
    Phase2PreflightError,
    PreflightTestEvidence,
    run_phase2_preflight,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _anchor_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "anchor_manifest.csv"
    rows: list[dict[str, str]] = []
    scenes = (
        ("bobi", 3),
        ("kaihe", 3),
        ("quanji", 3),
        ("tiaosheng", 3),
        ("woli", 3),
        ("xiezi", 3),
        ("jianpan", 3),
        ("run", 3),
    )
    for scene, count in scenes:
        for index in range(1, count + 1):
            sample = f"{scene}{index}"
            files: dict[str, Path] = {}
            for key in ("data", "ref", "report", "error"):
                file_path = tmp_path / "inputs" / f"{sample}-{key}.txt"
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(f"{sample}-{key}", encoding="utf-8")
                files[key] = file_path
            rows.append(
                {
                    "anchor_type": "independent_bo",
                    "sample": sample,
                    "scene": scene,
                    "data_path": str(files["data"]),
                    "ref_path": str(files["ref"]),
                    "report_path": str(files["report"]),
                    "error_csv": str(files["error"]),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def _independent_smoke(tmp_path: Path, sample: str) -> Path:
    root = tmp_path / f"independent-{sample}"
    _write_json(
        root / "independent_study_manifest.json",
        {"sample_id": sample},
    )
    _write_json(root / "acceptance_preview.json", {"sample_id": sample})
    (root / "independent_dual_baseline.csv").write_text(
        "sample\n" + sample,
        encoding="utf-8",
    )
    for arm in ("legacy_same_code", "physical_new"):
        arm_dir = root / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        (arm_dir / "candidate_history.csv").write_text(
            "candidate_id\ncandidate",
            encoding="utf-8",
        )
        _write_json(arm_dir / "seed_stability.json", {"ok": True})
        (arm_dir / "plot.png").write_bytes(b"png")
    (root / "historical.png").write_bytes(b"png")
    return root


def _kfold_smoke(tmp_path: Path, arm: str) -> Path:
    root = tmp_path / arm
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / f"{arm.lower()}_fold_manifest.json", {"arm": arm})
    selection = {
        "selection_hash": f"{arm}-hash",
        "evidence": {
            "heldout_record": {"record_id": "run3"},
            "training_records": [
                {"record_id": "run1"},
                {"record_id": "run2"},
            ],
        },
    }
    _write_json(root / "selection_receipt.json", selection)
    _write_json(
        root / "replay_receipt.json",
        {"selection_hash": f"{arm}-hash"},
    )
    for name in (
        "candidate_history.csv",
        "params.json",
        "training_metrics.csv",
        "cache_summary.json",
        "failure_classification.json",
    ):
        (root / name).write_text("{}", encoding="utf-8")
    for index in range(3):
        (root / f"plot-{index}.png").write_bytes(b"png")
    return root


def _test_evidence() -> tuple[PreflightTestEvidence, ...]:
    digest = hashlib.sha256(b"passed").hexdigest()
    return tuple(
        PreflightTestEvidence(
            suite=suite,
            command=f"run {suite}",
            exit_code=0,
            passed_count=1,
            output_sha256=digest,
        )
        for suite in (
            "p0_method_mapping_and_lms_floor",
            "phase2_contracts",
            "related_regressions",
            "plotting_regressions",
            "ruff",
        )
    )


def _config(tmp_path: Path) -> Phase2PreflightConfig:
    decision = tmp_path / "decision.json"
    _write_json(
        decision,
        {
            "status": "approved",
            "selected_smooth_win_len_s": 5,
            "formal_experiment_authorized": True,
        },
    )
    return Phase2PreflightConfig(
        repo_root=tmp_path,
        output_dir=tmp_path / "formal",
        git_commit="a" * 40,
        smoothing_decision=decision,
        anchor_manifest=_anchor_manifest(tmp_path),
        independent_smoke_dirs=(
            _independent_smoke(tmp_path, "sample-a"),
            _independent_smoke(tmp_path, "sample-b"),
        ),
        kfold_smoke_dirs={
            arm: _kfold_smoke(tmp_path, arm)
            for arm in ("K0", "K1", "K2", "K3")
        },
        test_evidence=_test_evidence(),
    )


def test_preflight_passes_only_after_all_machine_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        preflight,
        "_validate_git_state",
        lambda _root, commit: {
            "head": commit,
            "porcelain": "",
            "clean": True,
        },
    )

    result = run_phase2_preflight(config)

    payload = json.loads(result.preflight.read_text(encoding="utf-8"))
    manifest = json.loads(result.run_manifest.read_text(encoding="utf-8"))
    assert result.record_count == 24
    assert result.stage2_1_authorized is True
    assert payload["status"] == "passed"
    assert payload["stage2_1_authorized"] is True
    assert payload["evidence_level"] == EVIDENCE_LEVEL
    assert payload["confirmatory_claim_allowed"] is False
    assert payload["data_reuse_reason"] == DATA_REUSE_REASON
    assert all(item["status"] == "passed" for item in payload["checks"])
    assert manifest["stage2_1_authorized"] is True
    assert manifest["stage2_2_authorized"] is False
    assert len(manifest["pilot_scenes"]["xiezi"]) == 3
    assert len(manifest["pilot_scenes"]["jianpan"]) == 3
    assert len(manifest["pilot_scenes"]["run"]) == 3


def test_preflight_fails_closed_and_records_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        preflight,
        "_validate_git_state",
        lambda _root, _commit: (_ for _ in ()).throw(
            ValueError("dirty")
        ),
    )

    with pytest.raises(Phase2PreflightError, match="preflight_failed"):
        run_phase2_preflight(config)

    payload = json.loads(
        (config.output_dir / "preflight.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "preflight_failed"
    assert payload["stage2_1_authorized"] is False
    assert payload["checks"][0]["status"] == "failed"
