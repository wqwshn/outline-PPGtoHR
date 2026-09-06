"""Run the frozen seven-cell technical sentinel stage without starting P2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ppg_hr.v2.cross_subject_loso_identity import runtime_bundle_sha256
from ppg_hr.v2.cross_subject_loso_ledger import CompactResponseLedger
from ppg_hr.v2.cross_subject_loso_runner import (
    CrossSubjectRunner,
    choose_p1_sentinel_calls,
)
from ppg_hr.v2.cross_subject_loso_source import EXPERIMENT_ID, load_p0_snapshot

CONTRACT_PATH = Path("docs/contracts/acceptance/cross_subject_multirecord_hf_loso_v1.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--p0-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--executor", choices=("thread", "process"), default="thread")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    contract_path = repo_root / CONTRACT_PATH
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("p1_contract_experiment_mismatch")
    p0_root = (
        args.p0_root.resolve()
        if args.p0_root
        else repo_root / "data" / "experiments" / EXPERIMENT_ID / "p0"
    )
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else repo_root / "data" / "experiments" / EXPERIMENT_ID / "p1"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    snapshot = load_p0_snapshot(p0_root)
    runner_sha = _runner_sha256(repo_root)
    sentinel_ids = tuple(str(value) for value in contract["p1_sentinel_record_ids"])
    coordinate_index = int(contract["p1_coordinate_index"])
    sentinel_calls = choose_p1_sentinel_calls(
        snapshot.calls,
        sentinel_record_ids=sentinel_ids,
        coordinate_index=coordinate_index,
    )
    ledger_identity = {
        "experiment_id": snapshot.experiment_id,
        "stage": "P1",
        "dataset_sha256": snapshot.dataset_sha256,
        "coordinate_order_sha256": snapshot.coordinate_order_sha256,
        "call_manifest_sha256": snapshot.call_manifest_sha256,
        "algorithm_sha256": snapshot.algorithm_sha256,
        "runner_sha256": runner_sha,
        "metric_contract_sha256": snapshot.metric_contract_sha256,
        "route_id": "HF",
        "sentinel_record_ids": list(sentinel_ids),
        "coordinate_index": coordinate_index,
    }
    ledger = CompactResponseLedger.create(output_root / "sentinel_ledger.sqlite3", ledger_identity)
    try:
        runner = CrossSubjectRunner(
            ledger=ledger,
            dataset_sha256=snapshot.dataset_sha256,
            runner_sha256=runner_sha,
        )
        if args.workers == 1 and args.executor == "thread":
            run_receipt = runner.run_calls(sentinel_calls)
        else:
            run_receipt = runner.run_calls_parallel(
                sentinel_calls,
                workers=args.workers,
                batch_size=len(sentinel_calls),
                executor_kind=args.executor,
            )
        csv_sha = ledger.export_canonical_csv(output_root / "sentinel_cell_metrics.csv")
        complete_count = ledger.complete_count("HF")
        attempt_count = ledger.attempt_count()
    finally:
        ledger.close()

    status = "pass" if complete_count == len(sentinel_calls) else "fail"
    receipt = {
        "schema_id": "cross_subject_multirecord_p1_receipt_v1",
        "experiment_id": snapshot.experiment_id,
        "stage": "P1",
        "status": status,
        "p0_receipt_sha256": _file_sha256(p0_root / "p0_receipt.json"),
        "acceptance_contract_sha256": _file_sha256(contract_path),
        "dataset_sha256": snapshot.dataset_sha256,
        "algorithm_sha256": snapshot.algorithm_sha256,
        "runner_sha256": runner_sha,
        "metric_contract_sha256": snapshot.metric_contract_sha256,
        "sentinel_record_ids": list(sentinel_ids),
        "coordinate_index": coordinate_index,
        "sentinel_call_identity_sha256": [call.call_identity_sha256 for call in sentinel_calls],
        "requested": run_receipt.requested,
        "skipped": run_receipt.skipped,
        "attempted": run_receipt.attempted,
        "completed_this_run": run_receipt.completed,
        "technical_failures_this_run": run_receipt.technical_failures,
        "executor": args.executor,
        "workers": args.workers,
        "ledger_complete_count": complete_count,
        "ledger_attempt_event_count": attempt_count,
        "canonical_csv_sha256": csv_sha,
        "ledger_sha256": _file_sha256(output_root / "sentinel_ledger.sqlite3"),
    }
    _write_json(output_root / "p1_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if status == "pass" else 2


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runner_sha256(repo_root: Path) -> str:
    source_root = repo_root / "python" / "src" / "ppg_hr" / "v2"
    return runtime_bundle_sha256(
        tuple(
            source_root / name
            for name in (
                "cross_subject_loso_identity.py",
                "cross_subject_loso_ledger.py",
                "cross_subject_loso_metrics.py",
                "cross_subject_loso_runner.py",
                "cross_subject_loso_source.py",
            )
        )
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
