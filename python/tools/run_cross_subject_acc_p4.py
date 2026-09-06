"""Materialise the frozen 2x2 cells and evaluate exact per-record common support."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ppg_hr.v2.cross_subject_acc_experiment import (
    ACC_EXPERIMENT_ID,
    EXPECTED_ACC_CALL_COUNT,
    PARENT_EXPERIMENT_ID,
    build_acc_run_config,
)
from ppg_hr.v2.cross_subject_acc_paired import (
    ACC_THETA_ACC,
    ACC_THETA_HF,
    CONTRAST_COLUMNS,
    FOUR_CELL_LABELS,
    HF_THETA_ACC,
    HF_THETA_HF,
    build_fold_results,
    distribution_summary,
    evaluate_four_cell_common_support,
    summarise_fold_results,
)
from ppg_hr.v2.cross_subject_acc_selection import (
    file_sha256,
    load_acc_compact_cell_csv,
    write_dict_csv,
    write_json,
)
from ppg_hr.v2.cross_subject_acc_source import load_acc_p0_snapshot
from ppg_hr.v2.cross_subject_loso_metrics import (
    evaluate_fixed_time_metrics,
    solver_result_from_report,
)
from ppg_hr.v2.cross_subject_loso_runner import build_hf_run_config
from ppg_hr.v2.cross_subject_loso_selection import load_compact_cell_csv
from ppg_hr.v2.cross_subject_loso_source import (
    EXPECTED_FOLD_COUNT,
    EXPECTED_HF_CALL_COUNT,
    EXPECTED_RECORD_COUNT,
    PanelRecord,
    PhysicalCoordinate,
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
    elapsed_s: float
    error: str | None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--parent-experiment-root", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-passes", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.max_passes < 1:
        raise ValueError("positive_acc_p4_runtime_options_required")
    repo_root = args.repo_root.resolve()
    parent_root = (
        args.parent_experiment_root.resolve()
        if args.parent_experiment_root
        else repo_root / "data" / "experiments" / PARENT_EXPERIMENT_ID
    )
    experiment_root = (
        args.experiment_root.resolve()
        if args.experiment_root
        else repo_root / "data" / "experiments" / ACC_EXPERIMENT_ID
    )
    output_root = experiment_root / "p4"
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    snapshot = load_acc_p0_snapshot(experiment_root / "p0", parent_experiment_root=parent_root)
    _validate_upstream(parent_root, experiment_root)
    hf_selections = _load_selections(parent_root / "p3" / "selections")
    acc_selections = _load_selections(experiment_root / "p3" / "selections")
    records = {record.record_id: record for record in snapshot.parent.records}
    coordinates = {
        coordinate.coordinate_id: coordinate for coordinate in snapshot.parent.coordinates
    }
    hf_cells = load_compact_cell_csv(parent_root / "p2" / "hf_cell_metrics.csv")
    acc_cells = load_acc_compact_cell_csv(experiment_root / "p2" / "acc_cell_metrics.csv")
    hf_compact = {(cell.record_id, cell.coordinate_id): cell for cell in hf_cells}
    acc_compact = {(cell.record_id, cell.coordinate_id): cell for cell in acc_cells}
    if len(hf_compact) != EXPECTED_HF_CALL_COUNT or len(acc_compact) != EXPECTED_ACC_CALL_COUNT:
        raise ValueError("p4_compact_surface_incomplete")
    parent_reports = _load_parent_report_manifest(parent_root / "p4")

    request_rows, tasks = _build_requests_and_tasks(
        snapshot.parent.folds,
        records=records,
        coordinates=coordinates,
        hf_selections=hf_selections,
        acc_selections=acc_selections,
        hf_compact=hf_compact,
        acc_compact=acc_compact,
        parent_reports=parent_reports,
        output_root=output_root,
    )
    request_sha = write_dict_csv(output_root / "four_cell_request_manifest.csv", request_rows)
    technical_events: list[dict[str, Any]] = []
    valid_keys = {task.key for task in tasks if _valid_report(task)}
    for pass_index in range(1, args.max_passes + 1):
        pending = [task for task in tasks if task.key not in valid_keys]
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for outcome in executor.map(_materialise_task, pending):
                if outcome.error is not None:
                    technical_events.append(
                        {
                            "pass_index": pass_index,
                            "route_id": outcome.task.route_id,
                            "record_id": outcome.task.record.record_id,
                            "coordinate_id": outcome.task.coordinate.coordinate_id,
                            "error": outcome.error,
                            "occurred_at": datetime.now(UTC).isoformat(),
                        }
                    )
                elif _valid_report(outcome.task):
                    valid_keys.add(outcome.task.key)
                else:
                    technical_events.append(
                        {
                            "pass_index": pass_index,
                            "route_id": outcome.task.route_id,
                            "record_id": outcome.task.record.record_id,
                            "coordinate_id": outcome.task.coordinate.coordinate_id,
                            "error": "post_save_report_verification_failed",
                            "occurred_at": datetime.now(UTC).isoformat(),
                        }
                    )
        complete = len(valid_keys)
        write_json(
            output_root / "p4_progress.json",
            {
                "schema_id": "cross_subject_acc_p4_progress_v1",
                "experiment_id": ACC_EXPERIMENT_ID,
                "completed_unique_reports": complete,
                "expected_unique_reports": len(tasks),
                "technical_failure_events": len(technical_events),
                "elapsed_s": time.monotonic() - started,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    invalid = [task for task in tasks if task.key not in valid_keys]
    if invalid:
        write_json(
            output_root / "p4_failure_receipt.json",
            {
                "schema_id": "cross_subject_acc_p4_failure_v1",
                "status": "fail",
                "expected_unique_reports": len(tasks),
                "valid_unique_reports": len(tasks) - len(invalid),
                "technical_failure_events": technical_events,
            },
        )
        return 2

    report_rows = [_report_manifest_row(task) for task in tasks]
    report_sha = write_dict_csv(output_root / "materialized_report_manifest.csv", report_rows)
    path_by_key = {task.key: task.output_path for task in tasks}
    record_rows = _evaluate_records(
        request_rows,
        path_by_key=path_by_key,
        records=records,
        hf_compact=hf_compact,
        acc_compact=acc_compact,
    )
    if len(record_rows) != EXPECTED_RECORD_COUNT:
        raise ValueError(f"p4_paired_record_count:{len(record_rows)}")
    fold_rows = build_fold_results(record_rows)
    if len(fold_rows) != EXPECTED_FOLD_COUNT:
        raise ValueError(f"p4_paired_fold_count:{len(fold_rows)}")
    record_sha = write_dict_csv(output_root / "paired_record_results.csv", record_rows)
    fold_sha = write_dict_csv(output_root / "paired_fold_results.csv", fold_rows)
    scene_rows = summarise_fold_results(fold_rows, group_column="scene")
    subject_rows = summarise_fold_results(fold_rows, group_column="holdout_subject_id")
    overall_rows = summarise_fold_results(fold_rows)
    scene_sha = write_dict_csv(output_root / "paired_scene_summary.csv", scene_rows)
    subject_sha = write_dict_csv(output_root / "paired_subject_summary.csv", subject_rows)
    overall_sha = write_json(output_root / "paired_overall_summary.json", overall_rows[0])
    matrix_rows = _matrix_rows(fold_rows)
    contrast_rows = _contrast_rows(fold_rows)
    matrix_sha = write_dict_csv(output_root / "paired_2x2_matrix.csv", matrix_rows)
    contrast_sha = write_dict_csv(output_root / "paired_contrast_summary.csv", contrast_rows)
    audit_rows = _common_support_audit_rows(record_rows)
    audit_sha = write_dict_csv(output_root / "common_support_audit.csv", audit_rows)
    audit_ledger_sha = write_json(
        output_root / "p4_audit_ledger.json",
        {
            "schema_id": "cross_subject_acc_p4_audit_ledger_v1",
            "experiment_id": ACC_EXPERIMENT_ID,
            "raw_record_count": len(snapshot.parent.records),
            "fold_count": len(snapshot.parent.folds),
            "four_cell_request_count": len(request_rows),
            "unique_report_count": len(tasks),
            "parent_hf_report_reuse_count": sum(task.source == "parent_hf_reuse" for task in tasks),
            "targeted_replay_count": sum(task.source == "targeted_replay" for task in tasks),
            "technical_failure_events": technical_events,
            "performance_exclusions": [],
            "performance_exclusion_count": 0,
            "subject_record_counts": dict(
                sorted(Counter(record.physical_subject_id for record in records.values()).items())
            ),
            "hard_stop_scope": [
                "No HF+ACC route was run.",
                "No ACC time-bias selection was run.",
                "No ACC six-gate selection was run.",
                "No Physical4D expansion was run.",
                "No record was excluded according to observed performance.",
            ],
        },
    )
    receipt = {
        "schema_id": "cross_subject_multirecord_acc_p4_receipt_v1",
        "experiment_id": ACC_EXPERIMENT_ID,
        "stage": "P4",
        "status": "pass",
        "record_count": len(record_rows),
        "fold_count": len(fold_rows),
        "four_cell_request_count": len(request_rows),
        "unique_report_count": len(tasks),
        "technical_failure_event_count": len(technical_events),
        "performance_exclusion_count": 0,
        "elapsed_s": time.monotonic() - started,
        "sealed_at": datetime.now(UTC).isoformat(),
        "upstream_sha256": {
            "acc_p2_receipt.json": file_sha256(experiment_root / "p2" / "p2_receipt.json"),
            "acc_p3_freeze_receipt.json": file_sha256(
                experiment_root / "p3" / "selections" / "p3_freeze_receipt.json"
            ),
            "acc_p3_reveal_receipt.json": file_sha256(
                experiment_root / "p3" / "p3_reveal_receipt.json"
            ),
            "parent_p4_receipt.json": file_sha256(parent_root / "p4" / "p4_receipt.json"),
        },
        "artifact_sha256": {
            "four_cell_request_manifest.csv": request_sha,
            "materialized_report_manifest.csv": report_sha,
            "paired_record_results.csv": record_sha,
            "paired_fold_results.csv": fold_sha,
            "paired_scene_summary.csv": scene_sha,
            "paired_subject_summary.csv": subject_sha,
            "paired_overall_summary.json": overall_sha,
            "paired_2x2_matrix.csv": matrix_sha,
            "paired_contrast_summary.csv": contrast_sha,
            "common_support_audit.csv": audit_sha,
            "p4_audit_ledger.json": audit_ledger_sha,
        },
    }
    receipt_sha = write_json(output_root / "p4_receipt.json", receipt)
    print(
        json.dumps(
            {**receipt, "receipt_sha256": receipt_sha},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


def _validate_upstream(parent_root: Path, experiment_root: Path) -> None:
    checks = (
        (parent_root / "p2" / "p2_receipt.json", EXPECTED_HF_CALL_COUNT),
        (experiment_root / "p2" / "p2_receipt.json", EXPECTED_ACC_CALL_COUNT),
    )
    for path, count in checks:
        receipt = _read_json(path)
        if receipt.get("status") != "pass" or receipt.get("complete_cell_count") != count:
            raise ValueError(f"p4_response_surface_not_sealed:{path}")
    for path in (
        parent_root / "p3" / "selections" / "p3_freeze_receipt.json",
        parent_root / "p3" / "p3_reveal_receipt.json",
        parent_root / "p4" / "p4_receipt.json",
        experiment_root / "p3" / "selections" / "p3_freeze_receipt.json",
        experiment_root / "p3" / "p3_reveal_receipt.json",
    ):
        if _read_json(path).get("status") != "pass":
            raise ValueError(f"p4_upstream_not_pass:{path}")


def _load_selections(root: Path) -> dict[str, dict[str, Any]]:
    freeze = _read_json(root / "p3_freeze_receipt.json")
    rows = list(freeze.get("selections") or [])
    if freeze.get("status") != "pass" or len(rows) != EXPECTED_FOLD_COUNT:
        raise ValueError(f"p4_selection_freeze_incomplete:{root}")
    output = {}
    for row in rows:
        path = root / str(row["selection_file"])
        if file_sha256(path) != row["selection_sha256"]:
            raise ValueError(f"p4_selection_hash_mismatch:{row['fold_id']}")
        payload = _read_json(path)
        output[str(row["fold_id"])] = {
            "coordinate_id": str(payload["selection"]["coordinate_id"]),
            "coordinate_index": int(payload["selection"]["coordinate_index"]),
            "selection_sha256": str(row["selection_sha256"]),
        }
    return output


def _load_parent_report_manifest(root: Path) -> dict[tuple[str, str], Path]:
    receipt = _read_json(root / "p4_receipt.json")
    path = root / "report_manifest.csv"
    if file_sha256(path) != receipt["artifact_sha256"]["report_manifest.csv"]:
        raise ValueError("parent_hf_report_manifest_hash_mismatch")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_RECORD_COUNT:
        raise ValueError("parent_hf_report_manifest_count")
    return {
        (row["record_id"], row["coordinate_id"]): root / Path(row["report_path"]) for row in rows
    }


def _build_requests_and_tasks(
    folds: Any,
    *,
    records: dict[str, PanelRecord],
    coordinates: dict[str, PhysicalCoordinate],
    hf_selections: dict[str, dict[str, Any]],
    acc_selections: dict[str, dict[str, Any]],
    hf_compact: dict[tuple[str, str], Any],
    acc_compact: dict[tuple[str, str], Any],
    parent_reports: dict[tuple[str, str], Path],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[ReplayTask]]:
    request_rows: list[dict[str, Any]] = []
    tasks_by_key: dict[tuple[str, str, str], ReplayTask] = {}
    for fold in folds:
        hf_selection = hf_selections[fold.fold_id]
        acc_selection = acc_selections[fold.fold_id]
        cell_specs = (
            (HF_THETA_HF, "HF", "HF", hf_selection),
            (HF_THETA_ACC, "HF", "ACC", acc_selection),
            (ACC_THETA_HF, "ACC", "HF", hf_selection),
            (ACC_THETA_ACC, "ACC", "ACC", acc_selection),
        )
        for record_id in fold.holdout_record_ids:
            record = records[record_id]
            for label, route_id, coordinate_source, selection in cell_specs:
                coordinate = coordinates[selection["coordinate_id"]]
                compact_map = hf_compact if route_id == "HF" else acc_compact
                compact = compact_map.get((record_id, coordinate.coordinate_id))
                if compact is None:
                    raise ValueError(f"p4_compact_request_missing:{route_id}:{record_id}")
                parent_path = parent_reports.get((record_id, coordinate.coordinate_id))
                if route_id == "HF" and parent_path is not None:
                    output_path = parent_path.resolve()
                    source = "parent_hf_reuse"
                else:
                    output_path = (
                        output_root
                        / "full_reports"
                        / route_id.lower()
                        / record.scene
                        / f"{record_id}__c{coordinate.coordinate_index:03d}.json"
                    ).resolve()
                    source = "targeted_replay"
                task = ReplayTask(
                    route_id=route_id,
                    record=record,
                    coordinate=coordinate,
                    compact=compact,
                    output_path=output_path,
                    source=source,
                )
                existing = tasks_by_key.get(task.key)
                if existing is not None and existing.output_path != task.output_path:
                    raise ValueError(f"p4_dedup_path_conflict:{task.key}")
                tasks_by_key[task.key] = task
                request_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "physical_subject_id": record.physical_subject_id,
                        "scene": record.scene,
                        "record_id": record_id,
                        "cell_label": label,
                        "route_id": route_id,
                        "coordinate_source": coordinate_source,
                        "coordinate_id": coordinate.coordinate_id,
                        "coordinate_index": coordinate.coordinate_index,
                        "selection_sha256": selection["selection_sha256"],
                        "report_source": source,
                    }
                )
    if len(request_rows) != EXPECTED_RECORD_COUNT * len(FOUR_CELL_LABELS):
        raise ValueError(f"p4_request_count:{len(request_rows)}")
    return request_rows, sorted(tasks_by_key.values(), key=lambda task: task.key)


def _materialise_task(task: ReplayTask) -> ReplayOutcome:
    started = time.perf_counter()
    try:
        if task.source == "parent_hf_reuse":
            if not _valid_report(task):
                raise ValueError("parent_hf_report_failed_compact_verification")
        else:
            config = _build_config(task)
            result = solve_v2(config)
            _verify_compact(task, result)
            saved = save_v2_report(
                task.output_path,
                result,
                best_params={
                    "selected_coordinate": asdict(task.coordinate),
                    "resolved_run_config": asdict(config),
                },
                history=[],
                qc={"materialization_verification": "pass"},
                artefacts={
                    "schema_id": "cross_subject_acc_four_cell_full_report_v1",
                    "experiment_id": ACC_EXPERIMENT_ID,
                    "stage": "P4",
                    "route_id": task.route_id,
                    "record_id": task.record.record_id,
                    "coordinate_id": task.coordinate.coordinate_id,
                    "compact_call_identity_sha256": task.compact.call_identity_sha256,
                    "solver_elapsed_s": time.perf_counter() - started,
                },
            )
            if saved.resolve() != task.output_path.resolve():
                raise ValueError(f"p4_report_path_changed:{saved}")
        return ReplayOutcome(task=task, elapsed_s=time.perf_counter() - started, error=None)
    except Exception as error:
        return ReplayOutcome(
            task=task,
            elapsed_s=time.perf_counter() - started,
            error=f"{type(error).__name__}:{error}"[:4000],
        )


def _valid_report(task: ReplayTask) -> bool:
    try:
        payload = load_v2_report(task.output_path)
        result = solver_result_from_report(payload)
        _verify_compact(task, result)
        if task.source == "targeted_replay":
            artefacts = payload.get("artefacts") or {}
            if (
                artefacts.get("route_id") != task.route_id
                or artefacts.get("record_id") != task.record.record_id
                or artefacts.get("coordinate_id") != task.coordinate.coordinate_id
                or artefacts.get("compact_call_identity_sha256")
                != task.compact.call_identity_sha256
            ):
                return False
        return True
    except Exception:
        return False


def _verify_compact(task: ReplayTask, result: V2SolverResult) -> None:
    metric = evaluate_fixed_time_metrics(
        result,
        ref_data=load_v2_reference(task.record.ref_path),
        time_bias_s=5.0,
    )
    if task.route_id == "HF":
        expected = {
            "mae_bpm": task.compact.candidate_mae_bpm,
            "l10": task.compact.candidate_l10,
            "l20": task.compact.candidate_l20,
            "e10": task.compact.candidate_e10,
            "e20": task.compact.candidate_e20,
            "right_censored_recovery_count": task.compact.candidate_right_censored_recovery_count,
            "full_window_count": task.compact.candidate_full_window_count,
            "reliable_window_count": task.compact.candidate_reliable_window_count,
            "motion_window_count": task.compact.candidate_motion_window_count,
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
            "right_censored_recovery_count": task.compact.right_censored_recovery_count,
            "full_window_count": task.compact.full_window_count,
            "reliable_window_count": task.compact.reliable_window_count,
            "motion_window_count": task.compact.motion_window_count,
            "evaluation_window_sha256": task.compact.evaluation_window_sha256,
            "reference_groups_order": ("ACC",),
        }
    actual = asdict(metric)
    for key, value in expected.items():
        observed = actual[key]
        if isinstance(value, float):
            if not math.isclose(float(observed), value, abs_tol=1e-9, rel_tol=0.0):
                raise ValueError(f"p4_compact_float_mismatch:{key}:{observed}:{value}")
        elif observed != value:
            raise ValueError(f"p4_compact_mismatch:{key}:{observed}:{value}")


def _build_config(task: ReplayTask) -> Any:
    if task.route_id == "HF":
        return build_hf_run_config(task.record, task.coordinate)
    if task.route_id == "ACC":
        return build_acc_run_config(task.record, task.coordinate)
    raise ValueError(f"p4_unknown_route:{task.route_id}")


def _report_manifest_row(task: ReplayTask) -> dict[str, Any]:
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
    path_by_key: dict[tuple[str, str, str], Path],
    records: dict[str, PanelRecord],
    hf_compact: dict[tuple[str, str], Any],
    acc_compact: dict[tuple[str, str], Any],
) -> list[dict[str, Any]]:
    by_record: dict[str, list[dict[str, Any]]] = {}
    for row in request_rows:
        by_record.setdefault(str(row["record_id"]), []).append(row)
    result_cache: dict[Path, V2SolverResult] = {}
    output = []
    for record_id, rows in sorted(by_record.items()):
        if len(rows) != len(FOUR_CELL_LABELS):
            raise ValueError(f"p4_record_cell_count:{record_id}:{len(rows)}")
        by_label = {str(row["cell_label"]): row for row in rows}
        if set(by_label) != set(FOUR_CELL_LABELS):
            raise ValueError(f"p4_record_cell_set:{record_id}")
        record = records[record_id]
        results = {}
        for label, row in by_label.items():
            key = (str(row["route_id"]), record_id, str(row["coordinate_id"]))
            path = path_by_key[key]
            if path not in result_cache:
                result_cache[path] = solver_result_from_report(load_v2_report(path))
            results[label] = result_cache[path]
        paired = evaluate_four_cell_common_support(
            results,
            ref_data=load_v2_reference(record.ref_path),
            time_bias_s=5.0,
        )
        theta_hf = str(by_label[HF_THETA_HF]["coordinate_id"])
        theta_acc = str(by_label[ACC_THETA_ACC]["coordinate_id"])
        row: dict[str, Any] = {
            "fold_id": rows[0]["fold_id"],
            "physical_subject_id": record.physical_subject_id,
            "scene": record.scene,
            "record_id": record_id,
            "repeat_index": record.repeat_index,
            "source_repeat_label": record.source_repeat_label,
            "theta_hf_coordinate_id": theta_hf,
            "theta_hf_coordinate_index": by_label[HF_THETA_HF]["coordinate_index"],
            "theta_acc_coordinate_id": theta_acc,
            "theta_acc_coordinate_index": by_label[ACC_THETA_ACC]["coordinate_index"],
            "common_window_count": paired.common_window_count,
            "common_window_sha256": paired.common_window_sha256,
        }
        for label in FOUR_CELL_LABELS:
            row[f"{label}_native_window_count"] = paired.native_window_counts[label]
            row[f"{label}_native_window_sha256"] = paired.native_window_sha256[label]
            row[f"{label}_lost_window_count"] = paired.lost_window_counts[label]
            row[f"{label}_mae_bpm"] = paired.paired_mae_bpm[label]
            compact_map = hf_compact if label.startswith("hf_") else acc_compact
            coordinate_id = str(by_label[label]["coordinate_id"])
            compact = compact_map[(record_id, coordinate_id)]
            native_mae = (
                float(compact.candidate_mae_bpm)
                if label.startswith("hf_")
                else float(compact.mae_bpm)
            )
            row[f"{label}_compact_native_mae_bpm"] = native_mae
            row[f"{label}_paired_minus_native_mae_bpm"] = paired.paired_mae_bpm[label] - native_mae
        for output_column, (left, right) in CONTRAST_COLUMNS.items():
            row[output_column] = row[f"{left}_mae_bpm"] - row[f"{right}_mae_bpm"]
        output.append(row)
    return output


def _matrix_rows(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells = (
        ("HF", "HF", HF_THETA_HF),
        ("HF", "ACC", HF_THETA_ACC),
        ("ACC", "HF", ACC_THETA_HF),
        ("ACC", "ACC", ACC_THETA_ACC),
    )
    rows = []
    for route_id, coordinate_source, label in cells:
        summary = distribution_summary([float(row[f"{label}_mae_bpm"]) for row in fold_rows])
        rows.append(
            {
                "route_id": route_id,
                "coordinate_source": coordinate_source,
                "cell_label": label,
                "fold_count": len(fold_rows),
                **{f"mae_bpm__{key}": value for key, value in summary.items()},
            }
        )
    return rows


def _contrast_rows(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for column in CONTRAST_COLUMNS:
        summary = distribution_summary([float(row[column]) for row in fold_rows])
        rows.append(
            {
                "contrast_id": column,
                "fold_count": len(fold_rows),
                **summary,
                "positive_fold_count": sum(float(row[column]) > 0 for row in fold_rows),
                "zero_fold_count": sum(float(row[column]) == 0 for row in fold_rows),
                "negative_fold_count": sum(float(row[column]) < 0 for row in fold_rows),
            }
        )
    return rows


def _common_support_audit_rows(record_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = (
        "fold_id",
        "physical_subject_id",
        "scene",
        "record_id",
        "common_window_count",
        "common_window_sha256",
    ) + tuple(
        name
        for label in FOUR_CELL_LABELS
        for name in (
            f"{label}_native_window_count",
            f"{label}_native_window_sha256",
            f"{label}_lost_window_count",
        )
    )
    return [{column: row[column] for column in columns} for row in record_rows]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
