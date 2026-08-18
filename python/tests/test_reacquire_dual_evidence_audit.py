from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_tool() -> ModuleType:
    path = Path(__file__).parents[1] / "tools/run_reacquire_dual_evidence_audit.py"
    spec = importlib.util.spec_from_file_location("run_reacquire_dual_evidence_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_tool()


def _hashed(payload: dict[str, object], field: str) -> dict[str, object]:
    result = dict(payload)
    result[field] = audit._canonical_sha256(result)
    return result


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    cache_root = source_root / "execution/cache/solver"
    entry_dir = cache_root / "abc123"
    report_path = entry_dir / "report-v2.json"
    complete_path = entry_dir / "complete.json"
    _write_json(report_path, {"report": "original"})
    _write_json(complete_path, {"status": "complete"})

    proposal = _hashed({"data_panel": {"resolved_lite_records": []}}, "proposal_sha256")
    receipt = _hashed(
        {
            "entries": [
                {
                    "entry": str(entry_dir),
                    "cache_key": "abc123full-key",
                    "key_prefix_matches_entry": True,
                    "report_sha256": _file_hash(report_path),
                    "complete_sha256": _file_hash(complete_path),
                }
            ],
            "imported_solver_entry_count": 1,
            "proposal_sha256": proposal["proposal_sha256"],
        },
        "receipt_sha256",
    )
    lite_receipt = _hashed(
        {
            "stage": "lite_baseline",
            "decision": "stop",
            "stop_reasons": ["fixed_replay_confirms_mechanism_regression"],
        },
        "receipt_sha256",
    )
    completion = _hashed(
        {
            "status": "stopped_after_lite_audit",
            "proposal_sha256": proposal["proposal_sha256"],
            "lite_receipt": lite_receipt,
        },
        "completion_sha256",
    )
    _write_json(source_root / "proposal.json", proposal)
    _write_json(source_root / "execution/cache/cache_import_receipt.json", receipt)
    _write_json(source_root / "completion.json", completion)
    return source_root, cache_root


def test_candidate_deltas_capture_only_completed_challenge() -> None:
    payload = {
        "window_table": [
            {
                "spectrum_tracking": {
                    "reacquire_reason": "candidate_challenge_pending",
                    "reacquire_candidate_bpm": 102.0,
                    "reacquire_count": 1,
                }
            },
            {
                "spectrum_tracking": {
                    "reacquire_reason": "candidate_challenge_pending",
                    "reacquire_candidate_bpm": 100.0,
                    "reacquire_count": 2,
                }
            },
            {
                "spectrum_tracking": {
                    "reacquire_reason": "confirmed_upward_candidate",
                    "reacquire_candidate_bpm": 99.5,
                    "reacquire_count": 3,
                }
            },
        ]
    }
    assert audit._candidate_deltas(payload) == [-2.5]


def test_embedded_hash_verification_fails_closed_on_tamper() -> None:
    payload = _hashed({"value": 1}, "receipt_sha256")
    assert audit._verify_embedded_hash(payload, "receipt_sha256", "fixture")
    payload["value"] = 2
    with pytest.raises(RuntimeError, match="fixture_embedded_hash_mismatch"):
        audit._verify_embedded_hash(payload, "receipt_sha256", "fixture")


def test_source_binding_verification_rejects_report_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, cache_root = _source_fixture(tmp_path)
    monkeypatch.setattr(audit, "SOURCE_EXPERIMENT_ROOT", source_root)
    monkeypatch.setattr(audit, "EXPECTED_SOLVER_REPORTS", 1)
    audit._load_and_validate_source_bindings(cache_root)

    report_path = cache_root / "abc123/report-v2.json"
    _write_json(report_path, {"report": "tampered"})
    with pytest.raises(RuntimeError, match="v7_cache_report_hash_mismatch:abc123"):
        audit._load_and_validate_source_bindings(cache_root)


def test_input_binding_verification_rejects_hash_drift(tmp_path: Path) -> None:
    data_path = tmp_path / "record.csv"
    ref_path = tmp_path / "record_HR_ref.csv"
    data_path.write_text("data", encoding="utf-8")
    ref_path.write_text("reference", encoding="utf-8")
    actual = {
        "record": {
            "record_id": "record",
            "data_path": str(data_path),
            "data_sha256": _file_hash(data_path),
            "reference_path": str(ref_path),
            "reference_sha256": _file_hash(ref_path),
        }
    }
    proposal = {
        "data_panel": {
            "resolved_lite_records": [
                {
                    "record_id": "record",
                    "data_path": str(data_path),
                    "data_sha256": "wrong",
                    "ref_path": str(ref_path),
                    "ref_sha256": _file_hash(ref_path),
                }
            ]
        }
    }
    with pytest.raises(RuntimeError, match="v7_input_binding_mismatch:record:data_sha256"):
        audit._validate_input_bindings(actual, proposal)


def test_process_receipts_reject_pid_reuse_and_count_drift() -> None:
    code_hashes = {
        "runner_sha256": "runner",
        "source_files_sha256": "source",
    }
    with pytest.raises(RuntimeError, match="record_process_pid_reuse"):
        audit._validate_process_receipts(
            [
                {"pid": 10, "coordinate_count": 1},
                {"pid": 10, "coordinate_count": 1},
            ],
            2,
            code_hashes,
        )
    with pytest.raises(RuntimeError, match="record_process_coordinate_count_drift"):
        audit._validate_process_receipts(
            [{"pid": 10, "coordinate_count": 1}],
            2,
            code_hashes,
        )
    with pytest.raises(RuntimeError, match="record_process_code_hash_mismatch:record"):
        audit._validate_process_receipts(
            [
                {
                    "record_id": "record",
                    "pid": 10,
                    "coordinate_count": 1,
                    "runner_sha256": "wrong",
                    "source_files_sha256": "source",
                }
            ],
            1,
            code_hashes,
        )


def test_record_best_gate_rejects_regression() -> None:
    with pytest.raises(RuntimeError, match="record_best_regression:record"):
        audit._assert_no_record_best_regressions(
            [{"record_id": "record", "best_delta_bpm": 0.01}]
        )


def test_failed_rerun_overwrites_stale_success_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    completion_path = output_dir / "completion.json"
    _write_json(completion_path, {"status": "complete", "old": True})
    prior_hash = _file_hash(completion_path)
    monkeypatch.setattr(audit, "DEFAULT_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(audit, "DEFAULT_CACHE_ROOT", tmp_path / "cache")

    def fail(*_args: object, **_kwargs: object) -> int:
        running = json.loads(completion_path.read_text(encoding="utf-8"))
        assert running["status"] == "running"
        raise RuntimeError("fixture_failure")

    with pytest.raises(RuntimeError, match="fixture_failure"):
        audit._run_with_completion_guard(argparse.Namespace(workers=2), audit=fail)

    failed = json.loads(completion_path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["prior_completion_file_sha256"] == prior_hash
    assert failed["error_type"] == "RuntimeError"
    audit._verify_embedded_hash(failed, "receipt_sha256", "failed_receipt")


def test_second_instance_is_rejected_without_overwriting_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    completion_path = output_dir / "completion.json"
    original = {"status": "complete", "completion_sha256": "preserve"}
    _write_json(completion_path, original)
    monkeypatch.setattr(audit, "DEFAULT_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(audit, "DEFAULT_CACHE_ROOT", tmp_path / "cache")

    with audit._exclusive_run_lock(output_dir):
        with pytest.raises(RuntimeError, match="audit_run_already_active"):
            audit._run_with_completion_guard(
                argparse.Namespace(workers=1),
                audit=lambda *_args, **_kwargs: pytest.fail("audit must not start"),
            )

    assert json.loads(completion_path.read_text(encoding="utf-8")) == original
    assert not (output_dir / ".audit.lock").exists()


def test_runtime_code_hash_verification_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"runner_sha256": "runner", "source_files_sha256": "source"}
    monkeypatch.setattr(audit, "_runtime_code_hashes", lambda: dict(expected))
    assert audit._verify_runtime_code_hashes(expected, label="fixture") == expected
    monkeypatch.setattr(
        audit,
        "_runtime_code_hashes",
        lambda: {"runner_sha256": "changed", "source_files_sha256": "source"},
    )
    with pytest.raises(RuntimeError, match="fixture_runtime_code_hash_mismatch"):
        audit._verify_runtime_code_hashes(expected, label="fixture")
