"""Evaluate the seven LYX replacement records on the full ACC Physical4D grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ppg_hr.v2.cross_subject_acc_experiment import (
    ACC_METRIC_CONTRACT_SHA256,
    ACC_ROUTE_ID,
    PARENT_EXPERIMENT_ID,
    AccCallIdentity,
    AccCrossSubjectRunner,
)
from ppg_hr.v2.cross_subject_acc_ledger import AccCompactResponseLedger
from ppg_hr.v2.cross_subject_loso_identity import runtime_bundle_sha256
from ppg_hr.v2.cross_subject_loso_source import PanelRecord, load_p0_snapshot

EXPERIMENT_ID = "cross_subject_multirecord_d24_hf_acc_analysis_20260831_v1"
PARENT_ALGORITHM_SHA256 = "02cbe2a3c35a8442cf0326964767999619eaa3a86596ca9e3eafce73ece13f76"
REPLACEMENTS = {
    "tiaosheng1_LYX_0613": "tiaosheng10_LYX_0607",
    "tiaosheng1_LYX_0617": "tiaosheng11_LYX_0607",
    "tiaosheng2_LYX_0613": "tiaosheng2_LYX_0617",
    "woli1_LYX_0708": "woli1_LYX_0823",
    "woli3_LYX_0708": "woli2_LYX_0823",
    "xiezi2_LYX_0708": "xiezi1_LYX_0823",
    "xiezi4_LYX_0708": "xiezi2_LYX_0823",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--parent-worktree", type=Path)
    parser.add_argument("--lyx-worktree", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--executor", choices=("thread", "process"), default="process")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    worktrees_root = repo_root.parent
    parent_worktree = (
        args.parent_worktree.resolve()
        if args.parent_worktree
        else worktrees_root / "cross-subject-multirecord-hf-loso"
    )
    lyx_worktree = (
        args.lyx_worktree.resolve()
        if args.lyx_worktree
        else worktrees_root / "lyx-bo-space-generalization"
    )
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else repo_root / "data" / "experiments" / EXPERIMENT_ID / "acc_supplement"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    parent_root = parent_worktree / "data" / "experiments" / "cross_subject_multirecord_hf_loso_v1"
    snapshot = load_p0_snapshot(parent_root / "p0")
    if snapshot.algorithm_sha256 != PARENT_ALGORITHM_SHA256:
        raise RuntimeError("parent_algorithm_sha256_mismatch")

    manifest_records, manifest_hashes = _load_lyx_manifest_records(lyx_worktree)
    parent_by_id = {record.record_id: record for record in snapshot.records}
    replacement_records = tuple(
        _replacement_record(parent_by_id[old_id], manifest_records[new_id])
        for old_id, new_id in REPLACEMENTS.items()
    )
    calls = _build_calls(replacement_records, snapshot.coordinates)
    expected_count = len(replacement_records) * len(snapshot.coordinates)
    if len(calls) != expected_count or expected_count != 2_100:
        raise RuntimeError(f"unexpected_acc_supplement_call_count:{len(calls)}")

    synced_records = [
        record for record in snapshot.records if record.record_id not in REPLACEMENTS
    ] + list(replacement_records)
    dataset_sha = _semantic_sha256(
        [
            {
                "physical_subject_id": record.physical_subject_id,
                "scene": record.scene,
                "repeat_index": record.repeat_index,
                "record_id": record.record_id,
                "data_sha256": record.data_sha256,
                "ref_sha256": record.ref_sha256,
            }
            for record in sorted(synced_records, key=lambda item: item.record_id)
        ]
    )
    runner_sha = _runner_sha256(repo_root)
    call_manifest_sha = _semantic_sha256([_call_row(call) for call in calls])
    identity = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "ACC_SUPPLEMENT",
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "dataset_sha256": dataset_sha,
        "coordinate_order_sha256": snapshot.coordinate_order_sha256,
        "call_manifest_sha256": call_manifest_sha,
        "algorithm_sha256": PARENT_ALGORITHM_SHA256,
        "runner_sha256": runner_sha,
        "metric_contract_sha256": ACC_METRIC_CONTRACT_SHA256,
        "route_id": ACC_ROUTE_ID,
    }
    ledger = AccCompactResponseLedger.create(output_root / "acc_ledger.sqlite3", identity)
    started = time.monotonic()
    last_print = 0.0
    try:
        runner = AccCrossSubjectRunner(
            ledger=ledger,
            dataset_sha256=dataset_sha,
            runner_sha256=runner_sha,
        )

        def progress(receipt: Any) -> None:
            nonlocal last_print
            completed = receipt.skipped + receipt.completed
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = completed / elapsed
            payload = {
                "schema_id": "d24_acc_supplement_progress_v1",
                "experiment_id": EXPERIMENT_ID,
                "updated_at": datetime.now(UTC).isoformat(),
                "completed": completed,
                "expected": expected_count,
                "technical_failures_this_pass": receipt.technical_failures,
                "elapsed_s": elapsed,
                "rate_cells_s": rate,
                "eta_s": None if rate <= 0 else (expected_count - completed) / rate,
            }
            _write_json(output_root / "progress.json", payload)
            if elapsed - last_print >= 30.0 or completed == expected_count:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
                last_print = elapsed

        pass_receipt = runner.run_calls_parallel(
            calls,
            workers=args.workers,
            batch_size=args.batch_size,
            executor_kind=args.executor,
            on_progress=progress,
        )
        complete = ledger.complete_count(ACC_ROUTE_ID)
        attempts = ledger.attempt_count()
        csv_sha = ledger.export_canonical_csv(output_root / "acc_cell_metrics.csv")
    finally:
        ledger.close()

    status = "pass" if complete == expected_count and attempts == 0 else "fail"
    receipt = {
        "schema_id": "d24_acc_supplement_receipt_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "claim_boundary": "response_completion_for_posthoc_d24_sensitivity_analysis",
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "replacement_mapping": REPLACEMENTS,
        "record_count": len(replacement_records),
        "coordinate_count": len(snapshot.coordinates),
        "expected_cell_count": expected_count,
        "complete_cell_count": complete,
        "attempt_event_count": attempts,
        "dataset_sha256": dataset_sha,
        "coordinate_order_sha256": snapshot.coordinate_order_sha256,
        "call_manifest_sha256": call_manifest_sha,
        "algorithm_sha256": PARENT_ALGORITHM_SHA256,
        "algorithm_identity_note": (
            "The frozen parent solver identity is retained; the current branch only adds "
            "post-processing modules excluded from the solver runtime."
        ),
        "runner_sha256": runner_sha,
        "metric_contract_sha256": ACC_METRIC_CONTRACT_SHA256,
        "source_manifest_sha256": manifest_hashes,
        "baseline_metadata_used_by_solver": False,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "executor": args.executor,
        "run_receipt": asdict(pass_receipt),
        "canonical_csv_sha256": csv_sha,
        "elapsed_s": time.monotonic() - started,
        "sealed_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output_root / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0 if status == "pass" else 2


def _load_lyx_manifest_records(
    lyx_worktree: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    roots = (
        lyx_worktree / "data" / "experiments" / "lyx_curated_panel_threefold_summary_20260823",
        lyx_worktree
        / "data"
        / "experiments"
        / "lyx_tiaosheng_curated_panel_threefold_summary_20260824",
    )
    records: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for root in roots:
        path = root / "input_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        hashes[root.name] = _file_sha256(path)
        for row in payload["records"]:
            records[str(row["record_id"])] = dict(row)
    missing = set(REPLACEMENTS.values()) - set(records)
    if missing:
        raise RuntimeError(f"lyx_manifest_records_missing:{sorted(missing)}")
    return records, hashes


def _replacement_record(parent: PanelRecord, row: dict[str, Any]) -> PanelRecord:
    if parent.physical_subject_id != "LYX" or parent.scene != str(row["scene"]):
        raise RuntimeError(f"replacement_identity_mismatch:{parent.record_id}")
    record = replace(
        parent,
        source_repeat_label=str(row["record_id"]),
        record_id=str(row["record_id"]),
        data_path=Path(str(row["data_path"])).resolve(),
        ref_path=Path(str(row["ref_path"])).resolve(),
        data_sha256=str(row["data_sha256"]),
        ref_sha256=str(row["ref_sha256"]),
    )
    if _file_sha256(record.data_path) != record.data_sha256:
        raise RuntimeError(f"replacement_data_sha256_mismatch:{record.record_id}")
    if _file_sha256(record.ref_path) != record.ref_sha256:
        raise RuntimeError(f"replacement_ref_sha256_mismatch:{record.record_id}")
    return record


def _build_calls(
    records: tuple[PanelRecord, ...], coordinates: tuple[Any, ...]
) -> tuple[AccCallIdentity, ...]:
    calls: list[AccCallIdentity] = []
    for record in records:
        for coordinate in coordinates:
            identity = {
                "experiment_id": EXPERIMENT_ID,
                "parent_experiment_id": PARENT_EXPERIMENT_ID,
                "route_id": ACC_ROUTE_ID,
                "physical_subject_id": record.physical_subject_id,
                "scene": record.scene,
                "record_id": record.record_id,
                "data_sha256": record.data_sha256,
                "ref_sha256": record.ref_sha256,
                "coordinate": asdict(coordinate),
                "algorithm_sha256": PARENT_ALGORITHM_SHA256,
                "metric_contract_sha256": ACC_METRIC_CONTRACT_SHA256,
            }
            calls.append(
                AccCallIdentity(
                    experiment_id=EXPERIMENT_ID,
                    route_id=ACC_ROUTE_ID,
                    record=record,
                    coordinate=coordinate,
                    algorithm_sha256=PARENT_ALGORITHM_SHA256,
                    metric_contract_sha256=ACC_METRIC_CONTRACT_SHA256,
                    call_identity_sha256=_semantic_sha256(identity),
                )
            )
    return tuple(calls)


def _runner_sha256(repo_root: Path) -> str:
    root = repo_root / "python" / "src" / "ppg_hr" / "v2"
    return runtime_bundle_sha256(
        tuple(
            root / name
            for name in (
                "cross_subject_acc_experiment.py",
                "cross_subject_acc_ledger.py",
                "cross_subject_acc_source.py",
                "cross_subject_loso_identity.py",
                "cross_subject_loso_metrics.py",
                "cross_subject_loso_runner.py",
                "cross_subject_loso_source.py",
            )
        )
    )


def _call_row(call: AccCallIdentity) -> dict[str, Any]:
    return {
        "physical_subject_id": call.record.physical_subject_id,
        "scene": call.record.scene,
        "record_id": call.record.record_id,
        "coordinate_id": call.coordinate.coordinate_id,
        "coordinate_index": call.coordinate.coordinate_index,
        "call_identity_sha256": call.call_identity_sha256,
    }


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
