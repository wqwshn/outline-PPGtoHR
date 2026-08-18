"""Build a hash-closed audit for the low-lock reacquire dual-evidence rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python" / "src"))

from ppg_hr.v2.solver import solve_v2  # noqa: E402
from ppg_hr.v2.types import V2RunConfig  # noqa: E402

SOURCE_EXPERIMENT_ROOT = (
    REPO_ROOT / "data/experiments/lyx_current_source_lite_shared_20260802_formal_v7"
)
DEFAULT_CACHE_ROOT = SOURCE_EXPERIMENT_ROOT / "execution/cache/solver"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/experiments/jianpan2_reacquire_dual_evidence_20260802"
MIN_CANDIDATE_DRIFT_BPM = -1.0
EXPECTED_SOLVER_REPORTS = 2394
EXPECTED_AFFECTED_REPORTS = 139
EXPECTED_RECORDS = 24


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return _file_sha256(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected_json_object:{path}")
    return payload


def _verify_embedded_hash(payload: dict[str, Any], field: str, label: str) -> str:
    body = dict(payload)
    expected = str(body.pop(field, ""))
    if not expected:
        raise RuntimeError(f"{label}_missing_{field}")
    actual = _canonical_sha256(body)
    if actual != expected:
        raise RuntimeError(f"{label}_embedded_hash_mismatch:{actual}!={expected}")
    return expected


def _git_text(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _source_closure() -> dict[str, Any]:
    files = _source_file_entries()
    return {
        "schema_version": "reacquire_source_closure_v1",
        "git_commit": _git_text("rev-parse", "HEAD"),
        "git_status_porcelain": _git_text("status", "--porcelain"),
        "files": files,
        "files_sha256": _canonical_sha256(files),
    }


def _source_file_entries() -> list[dict[str, Any]]:
    paths = sorted((REPO_ROOT / "python/src/ppg_hr").rglob("*.py"))
    paths.append(REPO_ROOT / "python/pyproject.toml")
    return [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]


def _runtime_code_hashes() -> dict[str, str]:
    return {
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "source_files_sha256": _canonical_sha256(_source_file_entries()),
    }


def _verify_runtime_code_hashes(
    expected: dict[str, str],
    *,
    label: str,
) -> dict[str, str]:
    actual = _runtime_code_hashes()
    if actual != expected:
        raise RuntimeError(f"{label}_runtime_code_hash_mismatch:{actual}!={expected}")
    return actual


def _normal_path(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _load_and_validate_source_bindings(
    cache_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    proposal_path = SOURCE_EXPERIMENT_ROOT / "proposal.json"
    receipt_path = SOURCE_EXPERIMENT_ROOT / "execution/cache/cache_import_receipt.json"
    completion_path = SOURCE_EXPERIMENT_ROOT / "completion.json"
    proposal = _read_json(proposal_path)
    receipt = _read_json(receipt_path)
    completion = _read_json(completion_path)

    proposal_sha256 = _verify_embedded_hash(proposal, "proposal_sha256", "v7_proposal")
    receipt_sha256 = _verify_embedded_hash(receipt, "receipt_sha256", "v7_cache_receipt")
    completion_sha256 = _verify_embedded_hash(
        completion,
        "completion_sha256",
        "v7_completion",
    )
    if receipt.get("proposal_sha256") != proposal_sha256:
        raise RuntimeError("v7_cache_receipt_proposal_binding_mismatch")
    if completion.get("proposal_sha256") != proposal_sha256:
        raise RuntimeError("v7_completion_proposal_binding_mismatch")
    if completion.get("status") != "stopped_after_lite_audit":
        raise RuntimeError(f"v7_completion_status_drift:{completion.get('status')}")
    lite_receipt = completion.get("lite_receipt")
    if not isinstance(lite_receipt, dict):
        raise RuntimeError("v7_completion_missing_lite_receipt")
    lite_receipt_sha256 = _verify_embedded_hash(
        lite_receipt,
        "receipt_sha256",
        "v7_lite_receipt",
    )
    if (
        lite_receipt.get("stage") != "lite_baseline"
        or lite_receipt.get("decision") != "stop"
        or "fixed_replay_confirms_mechanism_regression"
        not in (lite_receipt.get("stop_reasons") or [])
    ):
        raise RuntimeError("v7_lite_stop_receipt_semantics_drift")

    receipt_entries = receipt.get("entries") or []
    if len(receipt_entries) != EXPECTED_SOLVER_REPORTS:
        raise RuntimeError(
            "v7_receipt_entry_count_drift:"
            f"{len(receipt_entries)}!={EXPECTED_SOLVER_REPORTS}"
        )
    if int(receipt.get("imported_solver_entry_count", -1)) != EXPECTED_SOLVER_REPORTS:
        raise RuntimeError("v7_receipt_imported_solver_count_drift")

    entries_by_cache_id: dict[str, dict[str, Any]] = {}
    for entry in receipt_entries:
        cache_id = Path(str(entry.get("entry") or "")).name
        if not cache_id or cache_id in entries_by_cache_id:
            raise RuntimeError(f"v7_receipt_duplicate_or_blank_cache_id:{cache_id}")
        if not bool(entry.get("key_prefix_matches_entry")):
            raise RuntimeError(f"v7_receipt_prefix_flag_false:{cache_id}")
        if not str(entry.get("cache_key") or "").startswith(cache_id):
            raise RuntimeError(f"v7_receipt_cache_key_prefix_mismatch:{cache_id}")
        entries_by_cache_id[cache_id] = entry

    cache_dirs = {path.name for path in cache_root.iterdir() if path.is_dir()}
    if cache_dirs != set(entries_by_cache_id):
        missing = sorted(set(entries_by_cache_id) - cache_dirs)
        extra = sorted(cache_dirs - set(entries_by_cache_id))
        raise RuntimeError(f"v7_cache_directory_set_mismatch:missing={missing}:extra={extra}")
    for cache_id, entry in entries_by_cache_id.items():
        entry_dir = cache_root / cache_id
        report_path = entry_dir / "report-v2.json"
        complete_path = entry_dir / "complete.json"
        if _file_sha256(report_path) != entry.get("report_sha256"):
            raise RuntimeError(f"v7_cache_report_hash_mismatch:{cache_id}")
        if _file_sha256(complete_path) != entry.get("complete_sha256"):
            raise RuntimeError(f"v7_cache_complete_hash_mismatch:{cache_id}")

    validation = {
        "source_proposal_embedded_sha256": proposal_sha256,
        "source_cache_receipt_embedded_sha256": receipt_sha256,
        "source_completion_embedded_sha256": completion_sha256,
        "source_lite_receipt_embedded_sha256": lite_receipt_sha256,
        "source_completion_status": completion["status"],
        "source_proposal_binding_mismatch_count": 0,
        "source_cache_entry_count": len(entries_by_cache_id),
        "source_cache_directory_mismatch_count": 0,
        "source_cache_report_hash_mismatch_count": 0,
        "source_cache_complete_hash_mismatch_count": 0,
    }
    return proposal, receipt, completion, validation


def _validate_input_bindings(
    input_entries: dict[str, dict[str, Any]],
    source_proposal: dict[str, Any],
) -> dict[str, Any]:
    expected_records = {
        str(row["record_id"]): row
        for row in source_proposal["data_panel"]["resolved_lite_records"]
    }
    if set(input_entries) != set(expected_records):
        missing = sorted(set(expected_records) - set(input_entries))
        extra = sorted(set(input_entries) - set(expected_records))
        raise RuntimeError(f"v7_input_record_set_mismatch:missing={missing}:extra={extra}")
    for record_id, actual in input_entries.items():
        expected = expected_records[record_id]
        comparisons = {
            "data_path": (
                _normal_path(actual["data_path"]),
                _normal_path(expected["data_path"]),
            ),
            "data_sha256": (actual["data_sha256"], expected["data_sha256"]),
            "reference_path": (
                _normal_path(actual["reference_path"]),
                _normal_path(expected["ref_path"]),
            ),
            "reference_sha256": (
                actual["reference_sha256"],
                expected["ref_sha256"],
            ),
        }
        for field, (actual_value, expected_value) in comparisons.items():
            if actual_value != expected_value:
                raise RuntimeError(f"v7_input_binding_mismatch:{record_id}:{field}")
    return {
        "source_input_record_count": len(input_entries),
        "source_input_record_set_mismatch_count": 0,
        "source_input_path_mismatch_count": 0,
        "source_input_hash_mismatch_count": 0,
    }


def _motion_mae(rows: list[dict[str, Any]]) -> float:
    errors = [
        abs(float(row["final_hr_bpm"]) - float(row["ref_hr_bpm"]))
        for row in rows
        if row.get("window_kind") == "motion"
        and np.isfinite(float(row["final_hr_bpm"]))
        and np.isfinite(float(row["ref_hr_bpm"]))
    ]
    return float(np.mean(errors))


def _candidate_deltas(payload: dict[str, Any]) -> list[float]:
    sequence: list[float] = []
    values: list[float] = []
    for row in payload.get("window_table") or []:
        trace = row.get("spectrum_tracking") or {}
        reason = str(trace.get("reacquire_reason") or "")
        candidate = trace.get("reacquire_candidate_bpm")
        count = int(trace.get("reacquire_count") or 0)
        if reason == "candidate_challenge_pending" and candidate is not None:
            if count <= 1:
                sequence = [float(candidate)]
            else:
                sequence.append(float(candidate))
            continue
        if reason == "confirmed_upward_candidate" and candidate is not None:
            if count <= 1:
                sequence = [float(candidate)]
            else:
                sequence.append(float(candidate))
            if len(sequence) >= 2:
                values.append(sequence[-1] - sequence[0])
            sequence = []
            continue
        if str(trace.get("reacquire_action") or "") in {
            "reset",
            "reset_candidate",
            "complete",
        }:
            sequence = []
    return values


def _run_affected(item: tuple[Path, dict[str, Any], list[float]]) -> dict[str, Any]:
    report_path, payload, candidate_deltas = item
    params = dict(payload.get("best_params") or {})
    result = solve_v2(
        V2RunConfig(
            data_path=Path(payload["data_path"]),
            ref_path=Path(payload["ref_path"]),
            ppg_mode="green",
            ppg_input_transform="raw_bandpass",
            analysis_scope="full",
            adaptive_filter="lms",
            algorithm_preset="lite",
            reference_groups_order=("HF",),
            motion_gate_filter_allowlist=("lms",),
            **params,
        )
    )
    reasons = Counter(
        str((row.get("spectrum_tracking") or {}).get("reacquire_reason") or "")
        for row in result.window_table
        if row.get("window_kind") == "motion"
    )
    routes = Counter(
        str((row.get("spectrum_tracking") or {}).get("reacquire_evidence_route") or "")
        for row in result.window_table
        if row.get("window_kind") == "motion"
    )
    old_full = float(payload["err_stats"]["final_aae_bpm"])
    new_full = float(result.err_stats["final_aae_bpm"])
    old_motion = _motion_mae(payload["window_table"])
    new_motion = _motion_mae(result.window_table)
    return {
        "cache_id": report_path.parent.name,
        "record_id": Path(payload["data_path"]).stem,
        "params": params,
        "old_candidate_deltas_bpm": candidate_deltas,
        "old_full_mae_bpm": old_full,
        "new_full_mae_bpm": new_full,
        "full_delta_bpm": new_full - old_full,
        "old_motion_mae_bpm": old_motion,
        "new_motion_mae_bpm": new_motion,
        "motion_delta_bpm": new_motion - old_motion,
        "new_trigger_count": int(reasons["confirmed_upward_candidate"]),
        "new_insufficient_evidence_count": int(reasons["insufficient_reacquire_evidence"]),
        "new_evidence_route_counts": {
            name: int(count) for name, count in sorted(routes.items()) if name
        },
    }


def _run_record_group(
    record_id: str,
    items: list[tuple[Path, dict[str, Any], list[float]]],
    expected_code_hashes: dict[str, str],
) -> dict[str, Any]:
    started_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    started = time.perf_counter()
    runtime_code_hashes = _verify_runtime_code_hashes(
        expected_code_hashes,
        label=f"record_process_start:{record_id}",
    )
    rows = [_run_affected(item) for item in items]
    if any(row["record_id"] != record_id for row in rows):
        raise RuntimeError(f"record_process_identity_drift:{record_id}")
    _verify_runtime_code_hashes(
        expected_code_hashes,
        label=f"record_process_end:{record_id}",
    )
    return {
        "record_id": record_id,
        "pid": os.getpid(),
        "coordinate_count": len(rows),
        "started_at": started_at,
        "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        **runtime_code_hashes,
        "rows": rows,
    }


def _record_process_entry(
    connection: Any,
    record_id: str,
    items: list[tuple[Path, dict[str, Any], list[float]]],
    expected_code_hashes: dict[str, str],
) -> None:
    try:
        message = {
            "status": "ok",
            "record_id": record_id,
            "result": _run_record_group(record_id, items, expected_code_hashes),
        }
    except BaseException as exc:
        message = {
            "status": "error",
            "record_id": record_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    try:
        connection.send(message)
    finally:
        connection.close()


def _run_record_processes(
    by_record: dict[str, list[tuple[Path, dict[str, Any], list[float]]]],
    *,
    max_workers: int,
    expected_code_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    record_ids = sorted(by_record)
    results: list[dict[str, Any]] = []
    for offset in range(0, len(record_ids), max_workers):
        wave = record_ids[offset : offset + max_workers]
        processes: dict[str, Any] = {}
        parent_connections: dict[Any, str] = {}
        try:
            for record_id in wave:
                parent_connection, child_connection = context.Pipe(duplex=False)
                process = context.Process(
                    target=_record_process_entry,
                    args=(
                        child_connection,
                        record_id,
                        by_record[record_id],
                        expected_code_hashes,
                    ),
                    name=f"reacquire-audit-{record_id}",
                )
                process.start()
                child_connection.close()
                processes[record_id] = process
                parent_connections[parent_connection] = record_id

            pending = set(parent_connections)
            while pending:
                for connection in wait(pending):
                    record_id = parent_connections[connection]
                    try:
                        message = connection.recv()
                    except EOFError as exc:
                        raise RuntimeError(f"record_process_no_receipt:{record_id}") from exc
                    finally:
                        connection.close()
                    pending.remove(connection)
                    if message.get("status") != "ok":
                        raise RuntimeError(
                            "record_process_failed:"
                            f"{record_id}:{message.get('error_type')}:{message.get('error')}"
                        )
                    results.append(message["result"])

            for record_id, process in processes.items():
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(
                        f"record_process_nonzero_exit:{record_id}:{process.exitcode}"
                    )
        finally:
            for connection in parent_connections:
                connection.close()
            for process in processes.values():
                if process.is_alive():
                    process.terminate()
                process.join()
    results.sort(key=lambda result: result["record_id"])
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise RuntimeError(f"refusing_empty_csv:{path.name}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    return _file_sha256(path)


def _build_best_rows(
    all_metrics: list[dict[str, Any]],
    affected_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    overrides = {row["cache_id"]: row["new_full_mae_bpm"] for row in affected_rows}
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_metrics:
        by_record[item["record_id"]].append(item)
    rows: list[dict[str, Any]] = []
    for record_id, items in sorted(by_record.items()):
        old_best = min(items, key=lambda item: item["old_full_mae_bpm"])
        new_best = min(
            items,
            key=lambda item: overrides.get(item["cache_id"], item["old_full_mae_bpm"]),
        )
        new_best_mae = float(overrides.get(new_best["cache_id"], new_best["old_full_mae_bpm"]))
        rows.append(
            {
                "record_id": record_id,
                "old_best_cache_id": old_best["cache_id"],
                "old_best_mae_bpm": old_best["old_full_mae_bpm"],
                "new_best_cache_id": new_best["cache_id"],
                "new_best_mae_bpm": new_best_mae,
                "best_delta_bpm": new_best_mae - old_best["old_full_mae_bpm"],
                "best_coordinate_changed": new_best["cache_id"] != old_best["cache_id"],
                "new_best_params": new_best["params"],
            }
        )
    return rows


def _assert_no_record_best_regressions(best_rows: list[dict[str, Any]]) -> None:
    regressions = [row for row in best_rows if float(row["best_delta_bpm"]) > 1e-12]
    if regressions:
        raise RuntimeError(
            "record_best_regression:"
            + ",".join(str(row["record_id"]) for row in regressions)
        )


def _validate_process_receipts(
    process_receipts: list[dict[str, Any]],
    expected_coordinate_count: int,
    expected_code_hashes: dict[str, str],
) -> None:
    if len({receipt["pid"] for receipt in process_receipts}) != len(process_receipts):
        raise RuntimeError("record_process_pid_reuse")
    if sum(int(receipt["coordinate_count"]) for receipt in process_receipts) != (
        expected_coordinate_count
    ):
        raise RuntimeError("record_process_coordinate_count_drift")
    for receipt in process_receipts:
        actual = {
            "runner_sha256": receipt.get("runner_sha256"),
            "source_files_sha256": receipt.get("source_files_sha256"),
        }
        if actual != expected_code_hashes:
            raise RuntimeError(
                f"record_process_code_hash_mismatch:{receipt.get('record_id')}"
            )


def _command(workers: int) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "--workers",
        str(workers),
    ]


@contextmanager
def _exclusive_run_lock(output_dir: Path) -> Iterator[Path]:
    lock_path = output_dir / ".audit.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise RuntimeError(f"audit_run_already_active:{lock_path}") from exc
    try:
        lock_payload = {
            "schema_version": "reacquire_audit_lock_v1",
            "pid": os.getpid(),
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        }
        os.write(
            descriptor,
            (json.dumps(lock_payload, ensure_ascii=False) + "\n").encode("utf-8"),
        )
        os.close(descriptor)
        descriptor = -1
        yield lock_path
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _write_run_receipt(path: Path, payload: dict[str, Any]) -> str:
    body = dict(payload)
    body["receipt_sha256"] = _canonical_sha256(body)
    return _write_json(path, body)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def _run_audit(
    args: argparse.Namespace,
    *,
    started: float,
    started_at: str,
) -> int:
    cache_root = DEFAULT_CACHE_ROOT.resolve()
    output_dir = DEFAULT_OUTPUT_DIR.resolve()
    reports = sorted(cache_root.glob("*/report-v2.json"))
    if len(reports) != EXPECTED_SOLVER_REPORTS:
        raise RuntimeError(f"solver_report_count_drift:{len(reports)}!={EXPECTED_SOLVER_REPORTS}")

    source_proposal, source_receipt, source_completion, binding_validation = (
        _load_and_validate_source_bindings(cache_root)
    )
    source_closure = _source_closure()
    expected_code_hashes = {
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "source_files_sha256": source_closure["files_sha256"],
    }
    _verify_runtime_code_hashes(expected_code_hashes, label="parent_process_start")
    source_closure_path = output_dir / "source_closure.json"
    source_closure_file_sha256 = _write_json(source_closure_path, source_closure)

    input_entries: dict[str, dict[str, Any]] = {}
    cache_entries: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    affected: list[tuple[Path, dict[str, Any], list[float]]] = []
    for report_path in reports:
        payload = _read_json(report_path)
        data_path = Path(payload["data_path"]).resolve()
        ref_path = Path(payload["ref_path"]).resolve()
        record_id = data_path.stem
        if record_id not in input_entries:
            input_entries[record_id] = {
                "record_id": record_id,
                "data_path": str(data_path),
                "data_sha256": _file_sha256(data_path),
                "data_bytes": data_path.stat().st_size,
                "reference_path": str(ref_path),
                "reference_sha256": _file_sha256(ref_path),
                "reference_bytes": ref_path.stat().st_size,
            }
        deltas = _candidate_deltas(payload)
        cache_id = report_path.parent.name
        params = dict(payload.get("best_params") or {})
        cache_entries.append(
            {
                "cache_id": cache_id,
                "report_path": report_path.relative_to(REPO_ROOT).as_posix(),
                "report_sha256": _file_sha256(report_path),
                "record_id": record_id,
                "params": params,
                "old_full_mae_bpm": float(payload["err_stats"]["final_aae_bpm"]),
                "candidate_deltas_bpm": deltas,
            }
        )
        all_metrics.append(
            {
                "cache_id": cache_id,
                "record_id": record_id,
                "old_full_mae_bpm": float(payload["err_stats"]["final_aae_bpm"]),
                "params": params,
            }
        )
        if any(value < MIN_CANDIDATE_DRIFT_BPM for value in deltas):
            affected.append((report_path, payload, deltas))

    if len(input_entries) != EXPECTED_RECORDS:
        raise RuntimeError(f"record_count_drift:{len(input_entries)}!={EXPECTED_RECORDS}")
    if len(affected) != EXPECTED_AFFECTED_REPORTS:
        raise RuntimeError(
            f"affected_report_count_drift:{len(affected)}!={EXPECTED_AFFECTED_REPORTS}"
        )
    binding_validation.update(_validate_input_bindings(input_entries, source_proposal))

    input_manifest = {
        "schema_version": "reacquire_input_manifest_v1",
        "record_count": len(input_entries),
        "records": [input_entries[key] for key in sorted(input_entries)],
    }
    input_manifest["records_sha256"] = _canonical_sha256(input_manifest["records"])
    input_manifest_path = output_dir / "input_manifest.json"
    input_manifest_file_sha256 = _write_json(input_manifest_path, input_manifest)

    source_paths = {
        "v7_proposal": SOURCE_EXPERIMENT_ROOT / "proposal.json",
        "v7_completion": SOURCE_EXPERIMENT_ROOT / "completion.json",
        "v7_cache_import_receipt": (
            SOURCE_EXPERIMENT_ROOT / "execution/cache/cache_import_receipt.json"
        ),
    }
    source_artifacts = {
        name: {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _file_sha256(path),
        }
        for name, path in source_paths.items()
    }
    cache_receipt = {
        "schema_version": "reacquire_cache_import_receipt_v2",
        "source_experiment": "lyx_current_source_lite_shared_20260802_formal_v7",
        "source_artifacts": source_artifacts,
        "source_binding_validation": binding_validation,
        "source_embedded_hashes": {
            "proposal": source_proposal["proposal_sha256"],
            "cache_receipt": source_receipt["receipt_sha256"],
            "completion": source_completion["completion_sha256"],
        },
        "solver_report_count": len(cache_entries),
        "entries_sha256": _canonical_sha256(cache_entries),
        "entries": cache_entries,
    }
    cache_receipt_path = output_dir / "cache_import_receipt.json"
    cache_receipt_file_sha256 = _write_json(cache_receipt_path, cache_receipt)

    command = _command(args.workers)
    proposal = {
        "schema_version": "reacquire_dual_evidence_proposal_v2",
        "question": (
            "Does the fixed dual-evidence confirmation repair jianpan2 while "
            "preserving each record's Lite upper bound?"
        ),
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "fixed_rule": {
            "min_candidate_drift_bpm": MIN_CANDIDATE_DRIFT_BPM,
            "min_low_track_drift_step_fraction": 0.75,
            "min_low_track_drift_target_gap_ratio": 0.12,
            "bo_parameter_added": False,
        },
        "expected_counts": {
            "solver_reports": EXPECTED_SOLVER_REPORTS,
            "affected_reports": EXPECTED_AFFECTED_REPORTS,
            "records": EXPECTED_RECORDS,
        },
        "cache_root": str(cache_root),
        "output_dir": str(output_dir),
        "source_closure_file_sha256": source_closure_file_sha256,
        "source_files_sha256": source_closure["files_sha256"],
        "input_manifest_file_sha256": input_manifest_file_sha256,
        "input_records_sha256": input_manifest["records_sha256"],
        "cache_import_receipt_file_sha256": cache_receipt_file_sha256,
        "cache_entries_sha256": cache_receipt["entries_sha256"],
        "runner_sha256": expected_code_hashes["runner_sha256"],
        "command": command,
        "command_text": subprocess.list2cmdline(command),
    }
    proposal["proposal_sha256"] = _canonical_sha256(proposal)
    proposal_path = output_dir / "proposal.json"
    proposal_file_sha256 = _write_json(proposal_path, proposal)

    by_record: dict[str, list[tuple[Path, dict[str, Any], list[float]]]] = defaultdict(list)
    for item in affected:
        by_record[Path(item[1]["data_path"]).stem].append(item)
    max_workers = min(args.workers, len(by_record))
    process_results = _run_record_processes(
        by_record,
        max_workers=max_workers,
        expected_code_hashes=expected_code_hashes,
    )
    _verify_runtime_code_hashes(expected_code_hashes, label="parent_process_end")
    process_receipts = [
        {key: value for key, value in result.items() if key != "rows"}
        for result in process_results
    ]
    _validate_process_receipts(
        process_receipts,
        len(affected),
        expected_code_hashes,
    )
    process_receipts_path = output_dir / "process_receipts.json"
    process_receipts_payload = {
        "schema_version": "reacquire_record_process_receipts_v1",
        "process_isolation": "explicit_spawn_one_process_per_record",
        "runtime_code_hashes": expected_code_hashes,
        "record_process_count": len(process_receipts),
        "records": process_receipts,
    }
    process_receipts_payload["records_sha256"] = _canonical_sha256(process_receipts)
    process_receipts_file_sha256 = _write_json(
        process_receipts_path,
        process_receipts_payload,
    )

    affected_rows = [row for result in process_results for row in result["rows"]]
    affected_rows.sort(key=lambda row: (row["record_id"], row["cache_id"]))
    affected_csv_path = output_dir / "affected_coordinates.csv"
    affected_csv_sha256 = _write_csv(affected_csv_path, affected_rows)

    best_rows = _build_best_rows(all_metrics, affected_rows)
    best_csv_path = output_dir / "record_best_summary.csv"
    best_csv_sha256 = _write_csv(best_csv_path, best_rows)
    _assert_no_record_best_regressions(best_rows)

    completed_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    completion = {
        "schema_version": "reacquire_dual_evidence_completion_v2",
        "status": "complete",
        "decision": "proceed_lightweight_algorithm_fix",
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "started_at": started_at,
        "completed_at": completed_at,
        "wall_seconds": time.perf_counter() - started,
        "command": command,
        "command_text": subprocess.list2cmdline(command),
        "cache_root": str(cache_root),
        "output_dir": str(output_dir),
        "proposal_sha256": proposal["proposal_sha256"],
        "proposal_file_sha256": proposal_file_sha256,
        "source_closure_file_sha256": source_closure_file_sha256,
        "input_manifest_file_sha256": input_manifest_file_sha256,
        "cache_import_receipt_file_sha256": cache_receipt_file_sha256,
        "source_binding_validation": binding_validation,
        "solver_report_count": len(reports),
        "identity_equivalent_report_count": len(reports) - len(affected_rows),
        "affected_report_count": len(affected_rows),
        "replayed_report_count": len(affected_rows),
        "record_count": len(best_rows),
        "record_process_count": len(process_receipts),
        "record_process_pid_reuse_count": 0,
        "record_best_any_regression_count": 0,
        "full_coordinate_improved_count": sum(
            float(row["full_delta_bpm"]) < -1e-12 for row in affected_rows
        ),
        "full_coordinate_same_count": sum(
            abs(float(row["full_delta_bpm"])) <= 1e-12 for row in affected_rows
        ),
        "full_coordinate_worse_count": sum(
            float(row["full_delta_bpm"]) > 1e-12 for row in affected_rows
        ),
        "artifacts": {
            "affected_coordinates.csv": affected_csv_sha256,
            "record_best_summary.csv": best_csv_sha256,
            "process_receipts.json": process_receipts_file_sha256,
            "proposal.json": proposal_file_sha256,
            "source_closure.json": source_closure_file_sha256,
            "input_manifest.json": input_manifest_file_sha256,
            "cache_import_receipt.json": cache_receipt_file_sha256,
        },
        "boundary": (
            "LYX development-reuse replay over cached six-dimensional Lite "
            "coordinates; not unseen-subject or cross-dataset generalization."
        ),
    }
    completion["completion_sha256"] = _canonical_sha256(completion)
    _write_json(output_dir / "completion.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return 0


def _run_with_completion_guard(
    args: argparse.Namespace,
    audit: Callable[..., int] = _run_audit,
) -> int:
    output_dir = DEFAULT_OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_run_lock(output_dir):
        completion_path = output_dir / "completion.json"
        prior_completion_file_sha256 = (
            _file_sha256(completion_path) if completion_path.exists() else None
        )
        started = time.perf_counter()
        started_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        command = _command(args.workers)
        common = {
            "schema_version": "reacquire_dual_evidence_run_receipt_v1",
            "started_at": started_at,
            "command": command,
            "command_text": subprocess.list2cmdline(command),
            "cache_root": str(DEFAULT_CACHE_ROOT.resolve()),
            "output_dir": str(output_dir),
            "prior_completion_file_sha256": prior_completion_file_sha256,
        }
        _write_run_receipt(completion_path, {**common, "status": "running"})
        try:
            return audit(args, started=started, started_at=started_at)
        except BaseException as exc:
            failed = {
                **common,
                "status": "failed",
                "failed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "wall_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_run_receipt(completion_path, failed)
            raise


def main() -> int:
    return _run_with_completion_guard(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
