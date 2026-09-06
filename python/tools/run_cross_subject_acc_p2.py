"""Build and seal the complete 42,900-cell ACC compact response ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ppg_hr.v2.cross_subject_acc_experiment import (
    ACC_EXPERIMENT_ID,
    EXPECTED_ACC_CALL_COUNT,
    AccCrossSubjectRunner,
)
from ppg_hr.v2.cross_subject_acc_ledger import AccCompactResponseLedger
from ppg_hr.v2.cross_subject_acc_source import load_acc_p0_snapshot
from ppg_hr.v2.cross_subject_loso_identity import runtime_bundle_sha256

CONTRACT_PATH = Path(
    "docs/contracts/acceptance/cross_subject_multirecord_acc_independent_physical4d_v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--parent-experiment-root", type=Path)
    parser.add_argument("--p0-root", type=Path)
    parser.add_argument("--p1-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--executor", choices=("thread", "process"), default="process")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    experiment_root = repo_root / "data" / "experiments" / ACC_EXPERIMENT_ID
    parent_root = (
        args.parent_experiment_root.resolve()
        if args.parent_experiment_root
        else repo_root / "data" / "experiments" / "cross_subject_multirecord_hf_loso_v1"
    )
    p0_root = args.p0_root.resolve() if args.p0_root else experiment_root / "p0"
    p1_root = args.p1_root.resolve() if args.p1_root else experiment_root / "p1"
    output_root = args.output_root.resolve() if args.output_root else experiment_root / "p2"
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot = load_acc_p0_snapshot(p0_root, parent_experiment_root=parent_root)
    runner_sha = _runner_sha256(repo_root)
    identity = _p2_identity(snapshot, runner_sha)
    ledger = AccCompactResponseLedger.create(output_root / "acc_ledger.sqlite3", identity)
    started = time.monotonic()
    imported_p1 = 0
    last_print = 0.0
    try:
        if ledger.complete_count("ACC") == 0:
            imported_p1 = _import_p1(
                p1_root=p1_root,
                target=ledger,
                snapshot=snapshot,
                runner_sha=runner_sha,
                repo_root=repo_root,
            )
        runner = AccCrossSubjectRunner(
            ledger=ledger,
            dataset_sha256=snapshot.parent.dataset_sha256,
            runner_sha256=runner_sha,
        )
        initial = ledger.complete_count("ACC")

        def progress(receipt: Any) -> None:
            nonlocal last_print
            completed = receipt.skipped + receipt.completed
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = max((completed - initial) / elapsed, 0.0)
            payload = {
                "schema_id": "cross_subject_multirecord_acc_p2_progress_v1",
                "experiment_id": ACC_EXPERIMENT_ID,
                "updated_at": datetime.now(UTC).isoformat(),
                "completed": completed,
                "expected": EXPECTED_ACC_CALL_COUNT,
                "technical_failures_this_pass": receipt.technical_failures,
                "elapsed_s": elapsed,
                "rate_cells_s": rate,
                "eta_s": None if rate <= 0 else (EXPECTED_ACC_CALL_COUNT - completed) / rate,
            }
            _write_json(output_root / "p2_progress.json", payload)
            if elapsed - last_print >= 30.0 or completed == EXPECTED_ACC_CALL_COUNT:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
                last_print = elapsed

        passes = []
        for pass_index in range(1, args.max_passes + 1):
            before = ledger.complete_count("ACC")
            if before == EXPECTED_ACC_CALL_COUNT:
                break
            receipt = runner.run_calls_parallel(
                snapshot.calls,
                workers=args.workers,
                batch_size=args.batch_size,
                executor_kind=args.executor,
                on_progress=progress,
            )
            after = ledger.complete_count("ACC")
            passes.append(
                {"pass_index": pass_index, "before": before, "after": after, **receipt.__dict__}
            )
            if after == before:
                break
        complete = ledger.complete_count("ACC")
        attempt_count = ledger.attempt_count()
        csv_sha = ledger.export_canonical_csv(output_root / "acc_cell_metrics.csv")
    finally:
        ledger.close()
    status = "pass" if complete == EXPECTED_ACC_CALL_COUNT and attempt_count == 0 else "fail"
    receipt = {
        "schema_id": "cross_subject_multirecord_acc_p2_receipt_v1",
        "experiment_id": ACC_EXPERIMENT_ID,
        "stage": "P2",
        "status": status,
        "p0_receipt_sha256": _file_sha256(p0_root / "p0_receipt.json"),
        "p1_receipt_sha256": _file_sha256(p1_root / "p1_receipt.json"),
        "acceptance_contract_sha256": _file_sha256(repo_root / CONTRACT_PATH),
        "dataset_sha256": snapshot.parent.dataset_sha256,
        "coordinate_order_sha256": snapshot.parent.coordinate_order_sha256,
        "call_manifest_sha256": snapshot.call_manifest_sha256,
        "algorithm_sha256": snapshot.parent.algorithm_sha256,
        "runner_sha256": runner_sha,
        "metric_contract_sha256": snapshot.calls[0].metric_contract_sha256,
        "expected_cell_count": EXPECTED_ACC_CALL_COUNT,
        "complete_cell_count": complete,
        "attempt_event_count": attempt_count,
        "imported_p1_cell_count": imported_p1,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "executor": args.executor,
        "pass_receipts": passes,
        "canonical_csv_sha256": csv_sha,
        "ledger_sha256": _file_sha256(output_root / "acc_ledger.sqlite3"),
        "elapsed_s": time.monotonic() - started,
        "sealed_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output_root / "p2_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0 if status == "pass" else 2


def _import_p1(
    *,
    p1_root: Path,
    target: AccCompactResponseLedger,
    snapshot: Any,
    runner_sha: str,
    repo_root: Path,
) -> int:
    contract = json.loads((repo_root / CONTRACT_PATH).read_text(encoding="utf-8"))
    ids = tuple(str(value) for value in contract["p1_sentinel_record_ids"])
    index = int(contract["p1_coordinate_index"])
    identity = {
        "experiment_id": ACC_EXPERIMENT_ID,
        "stage": "P1",
        "dataset_sha256": snapshot.parent.dataset_sha256,
        "coordinate_order_sha256": snapshot.parent.coordinate_order_sha256,
        "call_manifest_sha256": snapshot.call_manifest_sha256,
        "algorithm_sha256": snapshot.parent.algorithm_sha256,
        "runner_sha256": runner_sha,
        "metric_contract_sha256": snapshot.calls[0].metric_contract_sha256,
        "route_id": "ACC",
        "sentinel_record_ids": list(ids),
        "coordinate_index": index,
    }
    source = AccCompactResponseLedger.open(p1_root / "sentinel_ledger.sqlite3", identity)
    try:
        cells = source.read_complete_cells("ACC")
    finally:
        source.close()
    if len(cells) != len(ids):
        raise RuntimeError(f"acc_p1_seed_count:{len(cells)}")
    target.record_complete_batch(cells)
    return len(cells)


def _p2_identity(snapshot: Any, runner_sha: str) -> dict[str, Any]:
    return {
        "experiment_id": ACC_EXPERIMENT_ID,
        "stage": "P2",
        "dataset_sha256": snapshot.parent.dataset_sha256,
        "coordinate_order_sha256": snapshot.parent.coordinate_order_sha256,
        "call_manifest_sha256": snapshot.call_manifest_sha256,
        "algorithm_sha256": snapshot.parent.algorithm_sha256,
        "runner_sha256": runner_sha,
        "metric_contract_sha256": snapshot.calls[0].metric_contract_sha256,
        "route_id": "ACC",
    }


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


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
