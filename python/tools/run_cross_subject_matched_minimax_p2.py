"""Materialise five frozen cells and evaluate one exact record-level common support."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ppg_hr.v2.cross_subject_acc_experiment import build_acc_run_config
from ppg_hr.v2.cross_subject_acc_selection import load_acc_compact_cell_csv
from ppg_hr.v2.cross_subject_acc_source import load_acc_p0_snapshot
from ppg_hr.v2.cross_subject_loso_metrics import (
    evaluate_fixed_time_metrics,
    solver_result_from_report,
)
from ppg_hr.v2.cross_subject_loso_runner import build_hf_run_config
from ppg_hr.v2.cross_subject_loso_selection import load_compact_cell_csv
from ppg_hr.v2.cross_subject_loso_source import PanelRecord, PhysicalCoordinate
from ppg_hr.v2.cross_subject_matched_minimax import (
    ACC_THETA_ACC_MINIMAX,
    ACC_THETA_HF_MINIMAX,
    CONTRAST_COLUMNS,
    EXPERIMENT_ID,
    FIVE_CELL_LABELS,
    HF_THETA_ACC_MINIMAX,
    HF_THETA_HF_GATE,
    HF_THETA_HF_MINIMAX,
    PARENT_ACC_EXPERIMENT_ID,
    PARENT_HF_EXPERIMENT_ID,
    add_contrasts,
    build_fold_results,
    distribution_summary,
    evaluate_named_common_support,
    file_sha256,
    read_json,
    write_csv,
    write_json,
)
from ppg_hr.v2.preprocess import load_v2_reference
from ppg_hr.v2.report import load_v2_report, save_v2_report
from ppg_hr.v2.solver import V2SolverResult, solve_v2


@dataclass(frozen=True)
class ReplayTask:
    route_id: str
    record: PanelRecord
    coordinate: PhysicalCoordinate
    compact: Any
    output_path: Path
    source: str

    @property
    def key(self) -> tuple[str, str, str]:
        return self.route_id, self.record.record_id, self.coordinate.coordinate_id


@dataclass(frozen=True)
class ReplayOutcome:
    task: ReplayTask
    error: str | None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-passes", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.max_passes < 1:
        raise ValueError("matched_p2_positive_runtime_options_required")
    repo_root = args.repo_root.resolve()
    root = (
        args.experiment_root.resolve()
        if args.experiment_root
        else repo_root / "data" / "experiments" / EXPERIMENT_ID
    )
    hf_root = repo_root / "data" / "experiments" / PARENT_HF_EXPERIMENT_ID
    acc_root = repo_root / "data" / "experiments" / PARENT_ACC_EXPERIMENT_ID
    output_root = root / "p2"
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    _validate_upstream(root, hf_root, acc_root)
    snapshot = load_acc_p0_snapshot(acc_root / "p0", parent_experiment_root=hf_root)
    records = {record.record_id: record for record in snapshot.parent.records}
    coordinates = {
        coordinate.coordinate_id: coordinate for coordinate in snapshot.parent.coordinates
    }
    hf_compact = {
        (cell.record_id, cell.coordinate_id): cell
        for cell in load_compact_cell_csv(hf_root / "p2" / "hf_cell_metrics.csv")
    }
    acc_compact = {
        (cell.record_id, cell.coordinate_id): cell
        for cell in load_acc_compact_cell_csv(acc_root / "p2" / "acc_cell_metrics.csv")
    }
    if len(hf_compact) != 42_900 or len(acc_compact) != 42_900:
        raise ValueError("matched_p2_compact_surface_incomplete")
    hf_gate = _load_selections(hf_root / "p3" / "selections", "p3_freeze_receipt.json")
    acc_minimax = _load_selections(acc_root / "p3" / "selections", "p3_freeze_receipt.json")
    hf_minimax = _load_selections(root / "p1" / "selections", "p1_freeze_receipt.json")
    prior_reports = _load_prior_reports(acc_root / "p4")
    request_rows, tasks = _build_requests(
        snapshot.parent.folds,
        records=records,
        coordinates=coordinates,
        hf_gate=hf_gate,
        acc_minimax=acc_minimax,
        hf_minimax=hf_minimax,
        hf_compact=hf_compact,
        acc_compact=acc_compact,
        prior_reports=prior_reports,
        output_root=output_root,
    )
    request_sha = write_csv(output_root / "five_cell_request_manifest.csv", request_rows)
    technical_events = []
    valid = {task.key for task in tasks if _valid_report(task)}
    for pass_index in range(1, args.max_passes + 1):
        pending = [task for task in tasks if task.key not in valid]
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for outcome in executor.map(_materialise, pending):
                if outcome.error is None and _valid_report(outcome.task):
                    valid.add(outcome.task.key)
                else:
                    technical_events.append(
                        {
                            "pass_index": pass_index,
                            "route_id": outcome.task.route_id,
                            "record_id": outcome.task.record.record_id,
                            "coordinate_id": outcome.task.coordinate.coordinate_id,
                            "error": outcome.error or "post_save_verification_failed",
                            "occurred_at": datetime.now(UTC).isoformat(),
                        }
                    )
        write_json(
            output_root / "p2_progress.json",
            {
                "schema_id": "cross_subject_matched_minimax_p2_progress_v1",
                "experiment_id": EXPERIMENT_ID,
                "completed_unique_reports": len(valid),
                "expected_unique_reports": len(tasks),
                "targeted_replay_count": sum(
                    task.source == "targeted_replay" and task.key in valid for task in tasks
                ),
                "technical_failure_event_count": len(technical_events),
                "elapsed_s": time.monotonic() - started,
            },
        )
    invalid = [task for task in tasks if task.key not in valid]
    if invalid:
        write_json(
            output_root / "p2_failure_receipt.json",
            {
                "schema_id": "cross_subject_matched_minimax_p2_failure_v1",
                "status": "fail",
                "invalid_unique_report_count": len(invalid),
                "technical_failure_events": technical_events,
            },
        )
        return 2

    report_rows = [_report_row(task) for task in tasks]
    report_sha = write_csv(output_root / "materialized_report_manifest.csv", report_rows)
    paths = {task.key: task.output_path for task in tasks}
    record_rows = _evaluate_records(
        request_rows,
        paths=paths,
        records=records,
        hf_compact=hf_compact,
        acc_compact=acc_compact,
    )
    fold_rows = build_fold_results(record_rows)
    if len(record_rows) != 143 or len(fold_rows) != 48:
        raise ValueError("matched_p2_result_count")
    record_sha = write_csv(output_root / "matched_record_results.csv", record_rows)
    fold_sha = write_csv(output_root / "matched_fold_results.csv", fold_rows)
    cell_rows = _cell_summary(fold_rows)
    contrast_rows = _contrast_summary(fold_rows)
    scene_rows = _group_summary(fold_rows, "scene")
    subject_rows = _group_summary(fold_rows, "holdout_subject_id")
    cell_sha = write_csv(output_root / "five_cell_summary.csv", cell_rows)
    contrast_sha = write_csv(output_root / "contrast_summary.csv", contrast_rows)
    scene_sha = write_csv(output_root / "scene_summary.csv", scene_rows)
    subject_sha = write_csv(output_root / "subject_summary.csv", subject_rows)
    audit_rows = _support_audit(record_rows)
    audit_sha = write_csv(output_root / "common_support_audit.csv", audit_rows)
    ledger_sha = write_json(
        output_root / "p2_audit_ledger.json",
        {
            "schema_id": "cross_subject_matched_minimax_p2_audit_ledger_v1",
            "experiment_id": EXPERIMENT_ID,
            "logical_request_count": len(request_rows),
            "unique_report_count": len(tasks),
            "prior_report_reuse_count": sum(task.source == "prior_report_reuse" for task in tasks),
            "targeted_replay_count": sum(task.source == "targeted_replay" for task in tasks),
            "full_response_recalculation_count": 0,
            "technical_failure_events": technical_events,
            "performance_exclusions": [],
            "subject_record_counts": dict(
                sorted(Counter(record.physical_subject_id for record in records.values()).items())
            ),
            "forbidden_extensions_run": [],
        },
    )
    receipt = {
        "schema_id": "cross_subject_matched_minimax_p2_receipt_v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": "P2",
        "status": "pass",
        "record_count": len(record_rows),
        "fold_count": len(fold_rows),
        "logical_request_count": len(request_rows),
        "unique_report_count": len(tasks),
        "prior_report_reuse_count": sum(task.source == "prior_report_reuse" for task in tasks),
        "targeted_replay_count": sum(task.source == "targeted_replay" for task in tasks),
        "full_response_recalculation_count": 0,
        "technical_failure_event_count": len(technical_events),
        "performance_exclusion_count": 0,
        "elapsed_s": time.monotonic() - started,
        "artifact_sha256": {
            "five_cell_request_manifest.csv": request_sha,
            "materialized_report_manifest.csv": report_sha,
            "matched_record_results.csv": record_sha,
            "matched_fold_results.csv": fold_sha,
            "five_cell_summary.csv": cell_sha,
            "contrast_summary.csv": contrast_sha,
            "scene_summary.csv": scene_sha,
            "subject_summary.csv": subject_sha,
            "common_support_audit.csv": audit_sha,
            "p2_audit_ledger.json": ledger_sha,
        },
    }
    sha = write_json(output_root / "p2_receipt.json", receipt)
    print(json.dumps({**receipt, "receipt_sha256": sha}, ensure_ascii=False, indent=2))
    return 0


def _validate_upstream(root: Path, hf_root: Path, acc_root: Path) -> None:
    paths = (
        root / "p0" / "p0_receipt.json",
        root / "p1" / "selections" / "p1_freeze_receipt.json",
        root / "p1" / "p1_reveal_receipt.json",
        hf_root / "p2" / "p2_receipt.json",
        acc_root / "p2" / "p2_receipt.json",
        acc_root / "p5" / "p5_completion_receipt.json",
    )
    if any(read_json(path).get("status") != "pass" for path in paths):
        raise ValueError("matched_p2_upstream_not_pass")


def _load_selections(root: Path, freeze_name: str) -> dict[str, dict[str, Any]]:
    freeze = read_json(root / freeze_name)
    rows = list(freeze.get("selections") or [])
    if freeze.get("status") != "pass" or len(rows) != 48:
        raise ValueError(f"matched_p2_selection_incomplete:{root}")
    output = {}
    for row in rows:
        path = root / str(row["selection_file"])
        if file_sha256(path) != row["selection_sha256"]:
            raise ValueError(f"matched_p2_selection_hash:{row['fold_id']}")
        payload = read_json(path)["selection"]
        output[str(row["fold_id"])] = {
            "coordinate_id": str(payload["coordinate_id"]),
            "coordinate_index": int(payload["coordinate_index"]),
            "selection_sha256": str(row["selection_sha256"]),
        }
    return output


def _load_prior_reports(root: Path) -> dict[tuple[str, str, str], Path]:
    receipt = read_json(root / "p4_receipt.json")
    path = root / "materialized_report_manifest.csv"
    if file_sha256(path) != receipt["artifact_sha256"]["materialized_report_manifest.csv"]:
        raise ValueError("matched_p2_prior_report_manifest_hash")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["route_id"], row["record_id"], row["coordinate_id"]): Path(
            row["report_path"]
        ).resolve()
        for row in rows
    }


def _build_requests(
    folds: Any,
    *,
    records: dict[str, PanelRecord],
    coordinates: dict[str, PhysicalCoordinate],
    hf_gate: dict[str, dict[str, Any]],
    acc_minimax: dict[str, dict[str, Any]],
    hf_minimax: dict[str, dict[str, Any]],
    hf_compact: dict[tuple[str, str], Any],
    acc_compact: dict[tuple[str, str], Any],
    prior_reports: dict[tuple[str, str, str], Path],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[ReplayTask]]:
    requests = []
    tasks: dict[tuple[str, str, str], ReplayTask] = {}
    for fold in folds:
        specs = (
            (HF_THETA_HF_GATE, "HF", "HF_GATE", hf_gate[fold.fold_id]),
            (HF_THETA_ACC_MINIMAX, "HF", "ACC_MINIMAX", acc_minimax[fold.fold_id]),
            (HF_THETA_HF_MINIMAX, "HF", "HF_MINIMAX", hf_minimax[fold.fold_id]),
            (ACC_THETA_ACC_MINIMAX, "ACC", "ACC_MINIMAX", acc_minimax[fold.fold_id]),
            (ACC_THETA_HF_MINIMAX, "ACC", "HF_MINIMAX", hf_minimax[fold.fold_id]),
        )
        for record_id in fold.holdout_record_ids:
            record = records[record_id]
            for label, route, coordinate_source, selection in specs:
                coordinate = coordinates[selection["coordinate_id"]]
                compact_map = hf_compact if route == "HF" else acc_compact
                compact = compact_map[(record_id, coordinate.coordinate_id)]
                key = (route, record_id, coordinate.coordinate_id)
                prior = prior_reports.get(key)
                if prior is None:
                    path = (
                        output_root
                        / "full_reports"
                        / route.lower()
                        / record.scene
                        / f"{record_id}__c{coordinate.coordinate_index:03d}.json"
                    ).resolve()
                    source = "targeted_replay"
                else:
                    path = prior
                    source = "prior_report_reuse"
                task = ReplayTask(route, record, coordinate, compact, path, source)
                existing = tasks.get(key)
                if existing is not None and existing.output_path != path:
                    raise ValueError(f"matched_p2_task_path_conflict:{key}")
                tasks[key] = task
                requests.append(
                    {
                        "fold_id": fold.fold_id,
                        "physical_subject_id": record.physical_subject_id,
                        "scene": record.scene,
                        "record_id": record_id,
                        "cell_label": label,
                        "route_id": route,
                        "coordinate_source": coordinate_source,
                        "coordinate_id": coordinate.coordinate_id,
                        "coordinate_index": coordinate.coordinate_index,
                        "selection_sha256": selection["selection_sha256"],
                        "report_source": source,
                    }
                )
    if len(requests) != 143 * len(FIVE_CELL_LABELS):
        raise ValueError("matched_p2_logical_request_count")
    return requests, sorted(tasks.values(), key=lambda task: task.key)


def _materialise(task: ReplayTask) -> ReplayOutcome:
    try:
        if task.source == "prior_report_reuse":
            if not _valid_report(task):
                raise ValueError("prior_report_compact_verification_failed")
        else:
            config = _build_config(task)
            result = solve_v2(config)
            _verify_compact(task, result)
            save_v2_report(
                task.output_path,
                result,
                best_params={
                    "selected_coordinate": asdict(task.coordinate),
                    "resolved_run_config": asdict(config),
                },
                history=[],
                qc={"materialization_verification": "pass"},
                artefacts={
                    "schema_id": "cross_subject_matched_minimax_full_report_v1",
                    "experiment_id": EXPERIMENT_ID,
                    "stage": "P2",
                    "route_id": task.route_id,
                    "record_id": task.record.record_id,
                    "coordinate_id": task.coordinate.coordinate_id,
                    "compact_call_identity_sha256": task.compact.call_identity_sha256,
                },
            )
        return ReplayOutcome(task, None)
    except Exception as error:
        return ReplayOutcome(task, f"{type(error).__name__}:{error}"[:4000])


def _valid_report(task: ReplayTask) -> bool:
    try:
        payload = load_v2_report(task.output_path)
        _verify_compact(task, solver_result_from_report(payload))
        if task.source == "targeted_replay":
            artefacts = payload.get("artefacts") or {}
            return (
                artefacts.get("experiment_id") == EXPERIMENT_ID
                and artefacts.get("route_id") == task.route_id
                and artefacts.get("record_id") == task.record.record_id
                and artefacts.get("coordinate_id") == task.coordinate.coordinate_id
                and artefacts.get("compact_call_identity_sha256")
                == task.compact.call_identity_sha256
            )
        return True
    except Exception:
        return False


def _build_config(task: ReplayTask) -> Any:
    if task.route_id == "HF":
        return build_hf_run_config(task.record, task.coordinate)
    return build_acc_run_config(task.record, task.coordinate)


def _verify_compact(task: ReplayTask, result: V2SolverResult) -> None:
    metric = evaluate_fixed_time_metrics(
        result, ref_data=load_v2_reference(task.record.ref_path), time_bias_s=5.0
    )
    if task.route_id == "HF":
        expected = {
            "mae_bpm": task.compact.candidate_mae_bpm,
            "l10": task.compact.candidate_l10,
            "l20": task.compact.candidate_l20,
            "e10": task.compact.candidate_e10,
            "e20": task.compact.candidate_e20,
            "reliable_window_count": task.compact.candidate_reliable_window_count,
            "evaluation_window_sha256": task.compact.candidate_evaluation_window_sha256,
            "reference_groups_order": ("HF",),
        }
    else:
        expected = {
            "mae_bpm": task.compact.mae_bpm,
            "l10": task.compact.l10,
            "l20": task.compact.l20,
            "e10": task.compact.e10,
            "e20": task.compact.e20,
            "reliable_window_count": task.compact.reliable_window_count,
            "evaluation_window_sha256": task.compact.evaluation_window_sha256,
            "reference_groups_order": ("ACC",),
        }
    actual = asdict(metric)
    for key, value in expected.items():
        observed = actual[key]
        if isinstance(value, float):
            if not math.isclose(float(observed), value, abs_tol=1e-9, rel_tol=0.0):
                raise ValueError(f"matched_p2_compact_float:{key}")
        elif observed != value:
            raise ValueError(f"matched_p2_compact:{key}")


def _report_row(task: ReplayTask) -> dict[str, Any]:
    return {
        "route_id": task.route_id,
        "physical_subject_id": task.record.physical_subject_id,
        "scene": task.record.scene,
        "record_id": task.record.record_id,
        "coordinate_id": task.coordinate.coordinate_id,
        "coordinate_index": task.coordinate.coordinate_index,
        "compact_call_identity_sha256": task.compact.call_identity_sha256,
        "source": task.source,
        "report_path": str(task.output_path),
        "report_sha256": file_sha256(task.output_path),
    }


def _evaluate_records(
    request_rows: list[dict[str, Any]],
    *,
    paths: dict[tuple[str, str, str], Path],
    records: dict[str, PanelRecord],
    hf_compact: dict[tuple[str, str], Any],
    acc_compact: dict[tuple[str, str], Any],
) -> list[dict[str, Any]]:
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in request_rows:
        by_record[str(row["record_id"])].append(row)
    cache: dict[Path, V2SolverResult] = {}
    outputs = []
    for record_id, rows in sorted(by_record.items()):
        by_label = {str(row["cell_label"]): row for row in rows}
        if tuple(by_label) != FIVE_CELL_LABELS:
            raise ValueError(f"matched_p2_record_cell_order:{record_id}")
        record = records[record_id]
        results = {}
        for label in FIVE_CELL_LABELS:
            item = by_label[label]
            key = (str(item["route_id"]), record_id, str(item["coordinate_id"]))
            path = paths[key]
            if path not in cache:
                cache[path] = solver_result_from_report(load_v2_report(path))
            results[label] = cache[path]
        paired = evaluate_named_common_support(results, ref_data=load_v2_reference(record.ref_path))
        output: dict[str, Any] = {
            "fold_id": rows[0]["fold_id"],
            "physical_subject_id": record.physical_subject_id,
            "scene": record.scene,
            "record_id": record_id,
            "repeat_index": record.repeat_index,
            "source_repeat_label": record.source_repeat_label,
            "theta_hf_gate_coordinate_id": by_label[HF_THETA_HF_GATE]["coordinate_id"],
            "theta_hf_minimax_coordinate_id": by_label[HF_THETA_HF_MINIMAX]["coordinate_id"],
            "theta_acc_minimax_coordinate_id": by_label[ACC_THETA_ACC_MINIMAX]["coordinate_id"],
            "common_window_count": paired.common_window_count,
            "common_window_sha256": paired.common_window_sha256,
        }
        for label in FIVE_CELL_LABELS:
            output[f"{label}_native_window_count"] = paired.native_window_counts[label]
            output[f"{label}_native_window_sha256"] = paired.native_window_sha256[label]
            output[f"{label}_lost_window_count"] = paired.lost_window_counts[label]
            output[f"{label}_mae_bpm"] = paired.paired_mae_bpm[label]
            item = by_label[label]
            compact_map = hf_compact if str(item["route_id"]) == "HF" else acc_compact
            compact = compact_map[(record_id, str(item["coordinate_id"]))]
            native = (
                float(compact.candidate_mae_bpm)
                if str(item["route_id"]) == "HF"
                else float(compact.mae_bpm)
            )
            output[f"{label}_compact_native_mae_bpm"] = native
            output[f"{label}_paired_minus_native_mae_bpm"] = paired.paired_mae_bpm[label] - native
        outputs.append(add_contrasts(output))
    return outputs


def _cell_summary(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label in FIVE_CELL_LABELS:
        summary = distribution_summary([float(row[f"{label}_mae_bpm"]) for row in folds])
        rows.append({"cell_label": label, "fold_count": len(folds), **summary})
    return rows


def _contrast_summary(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name in CONTRAST_COLUMNS:
        values = [float(row[name]) for row in folds]
        rows.append(
            {
                "contrast_id": name,
                "fold_count": len(folds),
                **distribution_summary(values),
                "positive_fold_count": sum(value > 0 for value in values),
                "zero_fold_count": sum(value == 0 for value in values),
                "negative_fold_count": sum(value < 0 for value in values),
            }
        )
    return rows


def _group_summary(folds: list[dict[str, Any]], group_column: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in folds:
        grouped[str(row[group_column])].append(row)
    output = []
    columns = tuple(f"{label}_mae_bpm" for label in FIVE_CELL_LABELS) + tuple(CONTRAST_COLUMNS)
    for group, rows in sorted(grouped.items()):
        item: dict[str, Any] = {"group": group, "fold_count": len(rows)}
        for column in columns:
            summary = distribution_summary([float(row[column]) for row in rows])
            for statistic, value in summary.items():
                item[f"{column}__{statistic}"] = value
        output.append(item)
    return output


def _support_audit(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = (
        "fold_id",
        "physical_subject_id",
        "scene",
        "record_id",
        "common_window_count",
        "common_window_sha256",
    ) + tuple(
        name
        for label in FIVE_CELL_LABELS
        for name in (
            f"{label}_native_window_count",
            f"{label}_native_window_sha256",
            f"{label}_lost_window_count",
        )
    )
    return [{column: row[column] for column in columns} for row in records]


if __name__ == "__main__":
    raise SystemExit(main())
