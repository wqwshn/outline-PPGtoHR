"""Materialise 143 selected full reports and seal publication-facing tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ppg_hr.v2.cross_subject_loso_metrics import (
    evaluate_fixed_time_metrics,
    evaluate_six_gates,
    solver_result_from_report,
)
from ppg_hr.v2.cross_subject_loso_reporting import (
    build_reporting_tables,
    verify_materialized_facts,
)
from ppg_hr.v2.cross_subject_loso_runner import build_hf_run_config
from ppg_hr.v2.cross_subject_loso_selection import (
    compact_cell_sha256,
    load_compact_cell_csv,
    write_dict_csv,
)
from ppg_hr.v2.cross_subject_loso_source import (
    EXPECTED_FOLD_COUNT,
    EXPECTED_HF_CALL_COUNT,
    EXPECTED_RECORD_COUNT,
    EXPERIMENT_ID,
    HFCallIdentity,
    load_p0_snapshot,
)
from ppg_hr.v2.preprocess import load_v2_reference
from ppg_hr.v2.report import load_v2_report, save_v2_report
from ppg_hr.v2.solver import V2SolverResult, solve_v2


@dataclass(frozen=True)
class MaterializationTask:
    fold_id: str
    selection_sha256: str
    call: HFCallIdentity
    compact: Any
    output_path: Path


@dataclass(frozen=True)
class MaterializationOutcome:
    task: MaterializationTask
    result: V2SolverResult | None
    candidate: Any | None
    gate: Any | None
    elapsed_s: float
    error: str | None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--p0-root", type=Path)
    parser.add_argument("--p2-root", type=Path)
    parser.add_argument("--p3-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-passes", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.max_passes < 1:
        raise ValueError("positive_runtime_options_required")

    repo_root = args.repo_root.resolve()
    experiment_root = repo_root / "data" / "experiments" / EXPERIMENT_ID
    p0_root = args.p0_root.resolve() if args.p0_root else experiment_root / "p0"
    p2_root = args.p2_root.resolve() if args.p2_root else experiment_root / "p2"
    p3_root = args.p3_root.resolve() if args.p3_root else experiment_root / "p3"
    output_root = args.output_root.resolve() if args.output_root else experiment_root / "p4"
    output_root.mkdir(parents=True, exist_ok=True)

    snapshot = load_p0_snapshot(p0_root)
    p2_receipt = _read_json(p2_root / "p2_receipt.json")
    p3_receipt = _read_json(p3_root / "p3_reveal_receipt.json")
    _validate_upstream_receipts(p2_root, p3_root, p2_receipt, p3_receipt)
    compact_cells = load_compact_cell_csv(p2_root / "hf_cell_metrics.csv")
    compact_by_identity = {cell.call_identity_sha256: cell for cell in compact_cells}
    call_by_identity = {call.call_identity_sha256: call for call in snapshot.calls}
    holdout_rows = _read_csv(p3_root / "holdout_record_results.csv")
    fold_rows = _read_csv(p3_root / "fold_results.csv")
    if len(holdout_rows) != EXPECTED_RECORD_COUNT or len(fold_rows) != EXPECTED_FOLD_COUNT:
        raise ValueError("p3_result_count_mismatch")

    tasks = _build_tasks(
        holdout_rows,
        compact_by_identity,
        call_by_identity,
        output_root / "full_reports",
    )
    technical_events: list[dict[str, Any]] = []
    started = time.monotonic()
    for pass_index in range(1, args.max_passes + 1):
        pending = [task for task in tasks if not _existing_report_valid(task)]
        if not pending:
            break
        complete_count = len(tasks) - len(pending)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for outcome in executor.map(_solve_selected, pending):
                if outcome.error is not None:
                    technical_events.append(
                        {
                            "pass_index": pass_index,
                            "call_identity_sha256": outcome.task.call.call_identity_sha256,
                            "record_id": outcome.task.call.record.record_id,
                            "coordinate_id": outcome.task.call.coordinate.coordinate_id,
                            "error": outcome.error,
                            "occurred_at": datetime.now(UTC).isoformat(),
                        }
                    )
                    continue
                _save_verified_report(outcome)
                complete_count += 1
                if complete_count % 8 == 0 or complete_count == len(tasks):
                    _write_json(
                        output_root / "p4_progress.json",
                        {
                            "schema_id": "cross_subject_multirecord_p4_progress_v1",
                            "experiment_id": snapshot.experiment_id,
                            "completed": complete_count,
                            "expected": len(tasks),
                            "technical_failure_events": len(technical_events),
                            "elapsed_s": time.monotonic() - started,
                            "updated_at": datetime.now(UTC).isoformat(),
                        },
                    )

    manifest_rows = [
        row for task in tasks if (row := _existing_manifest_row(task, output_root)) is not None
    ]
    if len(manifest_rows) != EXPECTED_RECORD_COUNT:
        _write_json(
            output_root / "p4_failure_receipt.json",
            {
                "schema_id": "cross_subject_multirecord_p4_failure_v1",
                "status": "fail",
                "completed": len(manifest_rows),
                "expected": EXPECTED_RECORD_COUNT,
                "technical_failure_events": technical_events,
            },
        )
        return 2

    report_manifest_sha = write_dict_csv(output_root / "report_manifest.csv", manifest_rows)
    tables = build_reporting_tables(holdout_rows, fold_rows)
    scene_sha = write_dict_csv(output_root / "scene_summary.csv", tables["scene_summary"])
    gate_sha = write_dict_csv(output_root / "gate_summary.csv", tables["gate_summary"])
    coordinate_sha = write_dict_csv(
        output_root / "selected_coordinate_frequency.csv", tables["coordinate_frequency"]
    )
    overall_sha = _write_json(output_root / "overall_summary.json", tables["overall"])
    audit = _build_audit_ledger(snapshot, p2_receipt, technical_events)
    audit_sha = _write_json(output_root / "audit_ledger.json", audit)
    receipt = {
        "schema_id": "cross_subject_multirecord_p4_receipt_v1",
        "experiment_id": snapshot.experiment_id,
        "status": "pass",
        "p0_receipt_sha256": _file_sha256(p0_root / "p0_receipt.json"),
        "p2_receipt_sha256": _file_sha256(p2_root / "p2_receipt.json"),
        "p3_reveal_receipt_sha256": _file_sha256(p3_root / "p3_reveal_receipt.json"),
        "full_report_count": len(manifest_rows),
        "fold_result_count": len(fold_rows),
        "technical_failure_event_count": len(technical_events),
        "p2_attempt_event_count": p2_receipt["attempt_event_count"],
        "performance_exclusion_count": 0,
        "workers": args.workers,
        "elapsed_s": time.monotonic() - started,
        "sealed_at": datetime.now(UTC).isoformat(),
        "artifact_sha256": {
            "report_manifest.csv": report_manifest_sha,
            "scene_summary.csv": scene_sha,
            "gate_summary.csv": gate_sha,
            "selected_coordinate_frequency.csv": coordinate_sha,
            "overall_summary.json": overall_sha,
            "audit_ledger.json": audit_sha,
        },
    }
    receipt_sha = _write_json(output_root / "p4_receipt.json", receipt)
    print(
        json.dumps(
            {**receipt, "receipt_sha256": receipt_sha},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


def _build_tasks(
    holdout_rows: list[dict[str, str]],
    compact_by_identity: dict[str, Any],
    call_by_identity: dict[str, HFCallIdentity],
    report_root: Path,
) -> list[MaterializationTask]:
    tasks = []
    seen: set[str] = set()
    for row in holdout_rows:
        identity = row["call_identity_sha256"]
        if identity in seen:
            raise ValueError(f"duplicate_selected_call:{identity}")
        seen.add(identity)
        compact = compact_by_identity.get(identity)
        call = call_by_identity.get(identity)
        if compact is None or call is None:
            raise ValueError(f"selected_call_missing:{identity}")
        if row["compact_cell_sha256"] != compact_cell_sha256(compact):
            raise ValueError(f"selected_compact_cell_hash_mismatch:{identity}")
        if (
            row["record_id"] != call.record.record_id
            or row["selected_coordinate_id"] != call.coordinate.coordinate_id
        ):
            raise ValueError(f"selected_call_identity_mismatch:{identity}")
        output_path = (
            report_root
            / call.record.scene
            / f"{call.record.record_id}__c{call.coordinate.coordinate_index:03d}.json"
        )
        tasks.append(
            MaterializationTask(
                fold_id=row["fold_id"],
                selection_sha256=row["selection_sha256"],
                call=call,
                compact=compact,
                output_path=output_path,
            )
        )
    return tasks


def _solve_selected(task: MaterializationTask) -> MaterializationOutcome:
    started = time.perf_counter()
    try:
        config = build_hf_run_config(task.call.record, task.call.coordinate)
        result = solve_v2(config)
        reference = load_v2_reference(task.call.record.ref_path)
        candidate = evaluate_fixed_time_metrics(result, ref_data=reference, time_bias_s=5.0)
        gate = evaluate_six_gates(candidate=candidate, baseline=task.call.record.baseline.fixed5)
        verify_materialized_facts(asdict(task.compact), asdict(candidate), asdict(gate))
        return MaterializationOutcome(
            task=task,
            result=result,
            candidate=candidate,
            gate=gate,
            elapsed_s=time.perf_counter() - started,
            error=None,
        )
    except Exception as error:
        return MaterializationOutcome(
            task=task,
            result=None,
            candidate=None,
            gate=None,
            elapsed_s=time.perf_counter() - started,
            error=f"{type(error).__name__}:{error}"[:4000],
        )


def _save_verified_report(outcome: MaterializationOutcome) -> None:
    if outcome.result is None or outcome.candidate is None or outcome.gate is None:
        raise ValueError("cannot_save_failed_materialization")
    task = outcome.task
    config = build_hf_run_config(task.call.record, task.call.coordinate)
    saved = save_v2_report(
        task.output_path,
        outcome.result,
        best_params={
            "selected_coordinate": asdict(task.call.coordinate),
            "resolved_run_config": asdict(config),
        },
        history=[],
        qc={
            "source_baseline_qc_status": task.call.record.baseline.qc_status,
            "source_baseline_qc_reason": task.call.record.baseline.qc_reason,
            "materialization_verification": "pass",
        },
        artefacts={
            "schema_id": "cross_subject_multirecord_selected_full_report_v1",
            "experiment_id": task.call.experiment_id,
            "stage": "P4",
            "fold_id": task.fold_id,
            "selection_sha256": task.selection_sha256,
            "call_identity_sha256": task.call.call_identity_sha256,
            "compact_cell": asdict(task.compact),
            "fixed5_metrics": asdict(outcome.candidate),
            "six_gate_evaluation": asdict(outcome.gate),
            "solver_elapsed_s": outcome.elapsed_s,
        },
    )
    if saved.resolve() != task.output_path.resolve():
        raise RuntimeError(f"report_path_changed:{saved}")


def _existing_report_valid(task: MaterializationTask) -> bool:
    try:
        payload = load_v2_report(task.output_path)
        artefacts = payload["artefacts"]
        if (
            artefacts["call_identity_sha256"] != task.call.call_identity_sha256
            or artefacts["selection_sha256"] != task.selection_sha256
        ):
            return False
        result = solver_result_from_report(payload)
        reference = load_v2_reference(task.call.record.ref_path)
        candidate = evaluate_fixed_time_metrics(result, ref_data=reference, time_bias_s=5.0)
        gate = evaluate_six_gates(candidate=candidate, baseline=task.call.record.baseline.fixed5)
        verify_materialized_facts(asdict(task.compact), asdict(candidate), asdict(gate))
        return True
    except Exception:
        return False


def _existing_manifest_row(task: MaterializationTask, output_root: Path) -> dict[str, Any] | None:
    if not _existing_report_valid(task):
        return None
    payload = load_v2_report(task.output_path)
    metrics = payload["artefacts"]["fixed5_metrics"]
    gate = payload["artefacts"]["six_gate_evaluation"]
    return {
        "fold_id": task.fold_id,
        "physical_subject_id": task.call.record.physical_subject_id,
        "scene": task.call.record.scene,
        "record_id": task.call.record.record_id,
        "coordinate_id": task.call.coordinate.coordinate_id,
        "coordinate_index": task.call.coordinate.coordinate_index,
        "candidate_mae_bpm": metrics["mae_bpm"],
        "qualified": gate["qualified"],
        "call_identity_sha256": task.call.call_identity_sha256,
        "selection_sha256": task.selection_sha256,
        "report_path": str(task.output_path.relative_to(output_root)),
        "report_sha256": _file_sha256(task.output_path),
    }


def _validate_upstream_receipts(
    p2_root: Path,
    p3_root: Path,
    p2_receipt: dict[str, Any],
    p3_receipt: dict[str, Any],
) -> None:
    if (
        p2_receipt.get("status") != "pass"
        or p2_receipt.get("complete_cell_count") != EXPECTED_HF_CALL_COUNT
    ):
        raise ValueError("p2_not_sealed")
    if (
        p3_receipt.get("status") != "pass"
        or p3_receipt.get("record_result_count") != EXPECTED_RECORD_COUNT
    ):
        raise ValueError("p3_not_sealed")
    checks = (
        (p2_root / "hf_cell_metrics.csv", p2_receipt["canonical_csv_sha256"]),
        (
            p3_root / "holdout_record_results.csv",
            p3_receipt["holdout_record_results_sha256"],
        ),
        (p3_root / "fold_results.csv", p3_receipt["fold_results_sha256"]),
    )
    for path, expected in checks:
        if _file_sha256(path) != expected:
            raise ValueError(f"upstream_hash_mismatch:{path.name}")


def _build_audit_ledger(
    snapshot: Any,
    p2_receipt: dict[str, Any],
    technical_events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_id": "cross_subject_multirecord_audit_ledger_v1",
        "experiment_id": snapshot.experiment_id,
        "raw_record_count": len(snapshot.records),
        "fold_count": len(snapshot.folds),
        "physical_subject_count": len({row.physical_subject_id for row in snapshot.records}),
        "subject_record_counts": dict(
            sorted(Counter(row.physical_subject_id for row in snapshot.records).items())
        ),
        "scene_record_counts": dict(sorted(Counter(row.scene for row in snapshot.records).items())),
        "historical_baseline_bo_trial_count_distribution": dict(
            sorted(Counter(row.baseline.bo_trial_count for row in snapshot.records).items())
        ),
        "historical_baseline_qc_status_counts": dict(
            sorted(Counter(row.baseline.qc_status for row in snapshot.records).items())
        ),
        "p2_complete_cell_count": p2_receipt["complete_cell_count"],
        "p2_attempt_event_count": p2_receipt["attempt_event_count"],
        "performance_exclusions": [],
        "performance_exclusion_count": 0,
        "qc_warning_record_ids_retained": [
            row.record_id for row in snapshot.records if row.baseline.qc_status != "good"
        ],
        "panel_design_notes": [
            "PJY bobi uses Bobi1, Bobi3, and Bobi4; Bobi2 is outside the frozen panel.",
            "Run uses HB in place of QYC because QYC has no run records.",
            "QYC kaihe contains two records, so that subject-scene fold holds out two records.",
            "No record was removed according to observed P2 or P3 performance.",
        ],
        "hard_stop_scope": [
            "No ACC response surface or ACC-specific tuning was run.",
            "No performance-conditioned sensitivity subset was run.",
            "No record-specific time-bias tuning was run.",
            "No three-parallel-group LOSO variant was run.",
        ],
        "technical_failure_events": technical_events,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
