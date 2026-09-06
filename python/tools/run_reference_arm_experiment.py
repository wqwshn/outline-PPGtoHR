from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from ppg_hr.v2.reference_arm_experiment import (
    FROZEN_SOURCE_COMMIT,
    ReferenceArmRunner,
    ReferenceArmRunSummary,
    prepare_reference_arm_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen LYX reference-arm rectangle")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--output-root", type=Path, required=True)
    preflight.add_argument("--source-commit", default=FROZEN_SOURCE_COMMIT)
    sentinel = subparsers.add_parser("sentinel")
    sentinel.add_argument("--output-root", type=Path, required=True)
    sentinel.add_argument("--workers", type=int, default=2)
    run = subparsers.add_parser("run")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--timeout-hours", type=float, default=12.0)
    status = subparsers.add_parser("status")
    status.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    source_commit = getattr(args, "source_commit", FROZEN_SOURCE_COMMIT)
    context = prepare_reference_arm_experiment(
        repo_root,
        args.output_root,
        source_commit=source_commit,
    )
    if args.command == "preflight":
        print((context.output_root / "preflight_receipt.json").read_text(encoding="utf-8"))
        return 0
    if args.command == "status":
        payload = _status_payload(context)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if payload["missing_new_cells"] == 0 else 1

    runner = ReferenceArmRunner(
        snapshot=context.snapshot,
        ledger=context.ledger,
        experiment_id=str(context.experiment_identity["experiment_id"]),
        code_sha256=context.code_sha256,
    )
    calls = runner.calls
    if args.command == "sentinel":
        sentinel_records = {"jianpan1_LYX_0708", "tiaosheng2_LYX_0617"}
        calls = tuple(
            call
            for call in calls
            if call.record.record_id in sentinel_records and call.coordinate.coordinate_index == 0
        )
        if len(calls) != 4:
            raise RuntimeError(f"sentinel_call_count:{len(calls)}")
        deadline = None
    else:
        deadline = time.monotonic() + float(args.timeout_hours) * 3600.0

    progress_path = context.output_root / "progress.json"

    def on_progress(summary: ReferenceArmRunSummary) -> None:
        _write_json(progress_path, {**asdict(summary), **_status_payload(context)})

    summary = runner.run_calls(
        calls,
        workers=int(args.workers),
        deadline_monotonic=deadline,
        on_progress=on_progress,
    )
    payload = {**asdict(summary), **_status_payload(context)}
    _write_json(progress_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "sentinel":
        return (
            0 if summary.complete + summary.skipped == 4 and summary.technical_failures == 0 else 1
        )
    return 0 if payload["missing_new_cells"] == 0 and summary.technical_failures == 0 else 1


def _status_payload(context: object) -> dict[str, int]:
    ledger = context.ledger
    counts = {route: ledger.complete_count(route) for route in ("HF", "ACC", "HF_ACC")}
    technical = int(ledger.connection.execute("SELECT COUNT(*) FROM attempt_events").fetchone()[0])
    return {
        "hf_cells": counts["HF"],
        "acc_cells": counts["ACC"],
        "hf_acc_cells": counts["HF_ACC"],
        "missing_new_cells": 14_400 - counts["ACC"] - counts["HF_ACC"],
        "technical_attempt_events": technical,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


if __name__ == "__main__":
    sys.exit(main())
