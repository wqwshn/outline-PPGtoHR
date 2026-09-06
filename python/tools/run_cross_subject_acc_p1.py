"""Run the performance-blind seven-cell ACC technical sentinel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ppg_hr.v2.cross_subject_acc_experiment import (
    ACC_EXPERIMENT_ID,
    AccCrossSubjectRunner,
    choose_acc_sentinel_calls,
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
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--executor", choices=("thread", "process"), default="thread")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    experiment_root = repo_root / "data" / "experiments" / ACC_EXPERIMENT_ID
    parent_root = (
        args.parent_experiment_root.resolve()
        if args.parent_experiment_root
        else repo_root / "data" / "experiments" / "cross_subject_multirecord_hf_loso_v1"
    )
    p0_root = args.p0_root.resolve() if args.p0_root else experiment_root / "p0"
    output_root = args.output_root.resolve() if args.output_root else experiment_root / "p1"
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = repo_root / CONTRACT_PATH
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    snapshot = load_acc_p0_snapshot(p0_root, parent_experiment_root=parent_root)
    runner_sha = _runner_sha256(repo_root)
    sentinel_ids = tuple(str(value) for value in contract["p1_sentinel_record_ids"])
    coordinate_index = int(contract["p1_coordinate_index"])
    calls = choose_acc_sentinel_calls(
        snapshot.calls,
        sentinel_record_ids=sentinel_ids,
        coordinate_index=coordinate_index,
    )
    identity = _ledger_identity(snapshot, runner_sha, sentinel_ids, coordinate_index, "P1")
    ledger = AccCompactResponseLedger.create(output_root / "sentinel_ledger.sqlite3", identity)
    try:
        runner = AccCrossSubjectRunner(
            ledger=ledger,
            dataset_sha256=snapshot.parent.dataset_sha256,
            runner_sha256=runner_sha,
        )
        if args.workers == 1 and args.executor == "thread":
            run_receipt = runner.run_calls(calls)
        else:
            run_receipt = runner.run_calls_parallel(
                calls,
                workers=args.workers,
                batch_size=len(calls),
                executor_kind=args.executor,
            )
        csv_sha = ledger.export_canonical_csv(output_root / "sentinel_cell_metrics.csv")
        complete = ledger.complete_count("ACC")
        attempts = ledger.attempt_count()
    finally:
        ledger.close()
    status = "pass" if complete == len(calls) and attempts == 0 else "fail"
    receipt = {
        "schema_id": "cross_subject_multirecord_acc_p1_receipt_v1",
        "experiment_id": ACC_EXPERIMENT_ID,
        "stage": "P1",
        "status": status,
        "p0_receipt_sha256": _file_sha256(p0_root / "p0_receipt.json"),
        "acceptance_contract_sha256": _file_sha256(contract_path),
        "dataset_sha256": snapshot.parent.dataset_sha256,
        "algorithm_sha256": snapshot.parent.algorithm_sha256,
        "runner_sha256": runner_sha,
        "metric_contract_sha256": snapshot.calls[0].metric_contract_sha256,
        "sentinel_record_ids": list(sentinel_ids),
        "coordinate_index": coordinate_index,
        "sentinel_call_identity_sha256": [call.call_identity_sha256 for call in calls],
        "requested": run_receipt.requested,
        "completed_this_run": run_receipt.completed,
        "technical_failures_this_run": run_receipt.technical_failures,
        "ledger_complete_count": complete,
        "ledger_attempt_event_count": attempts,
        "canonical_csv_sha256": csv_sha,
        "ledger_sha256": _file_sha256(output_root / "sentinel_ledger.sqlite3"),
    }
    _write_json(output_root / "p1_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if status == "pass" else 2


def _ledger_identity(
    snapshot: Any, runner_sha: str, ids: tuple[str, ...], index: int, stage: str
) -> dict[str, Any]:
    return {
        "experiment_id": ACC_EXPERIMENT_ID,
        "stage": stage,
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
