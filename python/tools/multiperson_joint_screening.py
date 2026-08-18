"""Resumable joint-mechanism multiperson dataset screening orchestrator.

Stage ``lyx`` is a read-only consumer of the external 24 x 300 cache.  Later
stages add per-record BO caches under this experiment root; no file is ever
written below the external dependency root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna

PYTHON_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PYTHON_SRC))

from multiperson_screening_contracts import (  # noqa: E402
    BIAS_DEFAULT_S,
    calibrate_rest_time_bias,
    evaluate_aligned_metrics,
    evaluate_screening_gate,
    select_scene_panel,
)

from ppg_hr.v2.preprocess import load_v2_reference  # noqa: E402
from ppg_hr.v2.report import save_v2_report  # noqa: E402
from ppg_hr.v2.solver import V2SolverResult, solve_v2  # noqa: E402
from ppg_hr.v2.types import V2RunConfig  # noqa: E402

EXPERIMENT_ID = "multiperson_joint_physical4d_screening_20260819"
EXPECTED_HEAD = "0cd2a82a309112866e1727a0ad5761fe92d4e91c"
EXPECTED_PYTHON_SRC_TREE = "5c3a6717dc80252f3919a1b4ebf6ce6efc352dfb"
EXPECTED_PPG_HR_TREE = "71362a84a5814d2fddf2b58046b33c542f83cfa0"
EXPECTED_EXTERNAL_EXPERIMENT = (
    "lyx_eight_scene_joint_mechanism_physical4d_cache_20260818"
)
EXPECTED_PENALTY_ID = "suppressed_protected_continuous_visibility_v1"
ANCHOR_COORDINATE_ID = "physical4d:fs025:m200:mu0010:w006"
SCENES = (
    "bobi",
    "jianpan",
    "kaihe",
    "quanji",
    "run",
    "tiaosheng",
    "woli",
    "xiezi",
)
BO_SEEDS = (42, 43, 44)
BO_TRIALS_PER_REPEAT = 40
BO_STARTUP_TRIALS = 10
WRITE_LOCK = threading.Lock()


class ScreeningRunError(RuntimeError):
    """The frozen screening execution contract was violated."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, path)


def git(worktree: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=worktree, stderr=subprocess.DEVNULL
    ).decode("utf-8", errors="replace").strip()


def source_identity(worktree: Path) -> dict[str, Any]:
    checks = {
        "head": git(worktree, "rev-parse", "HEAD"),
        "python_src_tree": git(worktree, "rev-parse", "HEAD:python/src"),
        "ppg_hr_tree": git(worktree, "rev-parse", "HEAD:python/src/ppg_hr"),
        "tracked_python_src_diff": git(
            worktree, "diff", "--name-only", "--", "python/src"
        ),
    }
    expected = {
        "head": EXPECTED_HEAD,
        "python_src_tree": EXPECTED_PYTHON_SRC_TREE,
        "ppg_hr_tree": EXPECTED_PPG_HR_TREE,
        "tracked_python_src_diff": "",
    }
    if checks != expected:
        raise ScreeningRunError(
            "source_identity_mismatch:" + json.dumps(checks, sort_keys=True)
        )
    return {
        "schema_id": "multiperson_screening_source_identity_v1",
        "captured_at": now_iso(),
        "checks": checks,
        "algorithm_identity": {
            "adaptive_filter": "lms",
            "adaptive_reference_stage_limit": None,
            "algorithm_preset": "lite",
            "analysis_scope": "full",
            "penalty_candidate_id": EXPECTED_PENALTY_ID,
            "ppg_input_transform": "raw_bandpass",
            "ppg_mode": "green",
            "reference_groups_order": ["HF"],
            "rise_candidate_lineage_enable": True,
            "rise_confirmation_policy_id": "legacy_v1",
            "smooth_win_len": 5,
            "solver_time_bias_s": 5.0,
        },
    }


def validate_external_cache(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    required = (
        "completion.json",
        "proposal.json",
        "source_identity.json",
        "input_manifest.json",
        "coordinate_space.json",
        "call_manifest.json",
    )
    for name in required:
        if not (root / name).is_file():
            raise ScreeningRunError(f"external_cache_required_file_missing:{name}")
    completion = read_json(root / "completion.json")
    if completion.get("status") != "complete":
        raise ScreeningRunError("external_cache_not_complete")
    if int(completion.get("complete", -1)) != 7200:
        raise ScreeningRunError("external_cache_complete_count_mismatch")
    if int(completion.get("failure_count", -1)) != 0:
        raise ScreeningRunError("external_cache_has_failures")
    if completion.get("all_report_evidence_present") is not True:
        raise ScreeningRunError("external_cache_report_evidence_not_asserted")
    proposal = read_json(root / "proposal.json")
    if proposal.get("experiment_id") != EXPECTED_EXTERNAL_EXPERIMENT:
        raise ScreeningRunError("external_cache_experiment_id_mismatch")
    source = read_json(root / "source_identity.json")
    checks = source.get("checks") or {}
    if checks.get("head") != EXPECTED_HEAD:
        raise ScreeningRunError("external_cache_head_mismatch")
    if checks.get("python_src_tree") != EXPECTED_PYTHON_SRC_TREE:
        raise ScreeningRunError("external_cache_python_src_tree_mismatch")
    if checks.get("ppg_hr_tree") != EXPECTED_PPG_HR_TREE:
        raise ScreeningRunError("external_cache_ppg_hr_tree_mismatch")
    for field, filename in (
        ("input_manifest_sha256", "input_manifest.json"),
        ("coordinate_space_sha256", "coordinate_space.json"),
        ("call_manifest_sha256", "call_manifest.json"),
    ):
        if completion.get(field) != file_sha256(root / filename):
            raise ScreeningRunError(f"external_cache_{field}_mismatch")
    input_manifest = read_json(root / "input_manifest.json")
    coordinate_space = read_json(root / "coordinate_space.json")
    call_manifest = read_json(root / "call_manifest.json")
    records = list(input_manifest.get("records") or [])
    coordinates = list(coordinate_space.get("coordinates") or [])
    calls = list(call_manifest.get("calls") or [])
    if len(records) != 24 or len(coordinates) != 300 or len(calls) != 7200:
        raise ScreeningRunError("external_cache_manifest_cardinality_mismatch")
    if len({str(row["cache_key"]) for row in calls}) != 7200:
        raise ScreeningRunError("external_cache_key_cardinality_mismatch")
    return {
        "root": str(root),
        "completion": completion,
        "completion_sha256": file_sha256(root / "completion.json"),
        "proposal_sha256": file_sha256(root / "proposal.json"),
        "source_identity": source,
        "records": records,
        "coordinates": coordinates,
        "calls": calls,
        "manifest_hashes": {
            name: file_sha256(root / name) for name in required
        },
        "access_mode": "read_only_external_dependency",
    }


def solver_result_from_payload(payload: Mapping[str, Any]) -> V2SolverResult:
    if payload.get("schema_version") != "v2":
        raise ScreeningRunError("report_schema_mismatch")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"hr", "window_table", "err_stats", "history", "qc"}
    }
    return V2SolverResult(
        HR=np.asarray(payload.get("hr") or [], dtype=float),
        err_stats=dict(payload.get("err_stats") or {}),
        metadata=metadata,
        window_table=list(payload.get("window_table") or []),
    )


def validate_joint_report_identity(payload: Mapping[str, Any]) -> None:
    checks = {
        "algorithm_preset": payload.get("algorithm_preset") == "lite",
        "analysis_scope": payload.get("analysis_scope") == "full",
        "adaptive_filter": payload.get("adaptive_filter") == "lms",
        "reference_groups_order": list(payload.get("reference_groups_order") or [])
        == ["HF"],
        "adaptive_reference_stage_limit": payload.get(
            "adaptive_reference_stage_limit"
        )
        is None,
        "rise_candidate_lineage_enable": payload.get(
            "rise_candidate_lineage_enable"
        )
        is True,
        "rise_confirmation_policy_id": payload.get("rise_confirmation_policy_id")
        == "legacy_v1",
        "penalty_candidate_id": (payload.get("motion_penalty") or {}).get(
            "penalty_id"
        )
        == EXPECTED_PENALTY_ID,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ScreeningRunError("joint_report_identity_mismatch:" + ",".join(failed))


def read_verified_external_report(call: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    entry = Path(str(call["cache_entry"]))
    complete_path = entry / "complete.json"
    report_path = entry / "report-v2.json"
    if not complete_path.is_file() or not report_path.is_file():
        raise ScreeningRunError(f"external_cell_missing:{call['cache_key']}")
    complete = read_json(complete_path)
    if complete.get("status") != "complete":
        raise ScreeningRunError(f"external_cell_not_complete:{call['cache_key']}")
    if complete.get("cache_key") != call.get("cache_key"):
        raise ScreeningRunError(f"external_cell_cache_key_mismatch:{call['cache_key']}")
    raw = report_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != complete.get("report_sha256"):
        raise ScreeningRunError(f"external_cell_report_hash_mismatch:{call['cache_key']}")
    payload = json.loads(raw.decode("utf-8"))
    validate_joint_report_identity(payload)
    return payload, digest


def compatible_baseline(payload: Mapping[str, Any], record_id: str) -> bool:
    try:
        data_id = Path(str(payload["data_path"])).stem
    except (KeyError, TypeError, ValueError):
        return False
    return (
        payload.get("schema_version") == "v2"
        and data_id == record_id
        and payload.get("ppg_mode") == "green"
        and payload.get("ppg_input_transform") == "raw_bandpass"
        and payload.get("algorithm_preset") == "lite"
        and payload.get("analysis_scope") == "full"
        and payload.get("adaptive_filter") == "lms"
        and list(payload.get("reference_groups_order") or []) == ["HF"]
        and bool(payload.get("hr"))
        and bool(payload.get("window_table"))
    )


def find_historical_baseline(subject_root: Path, record_id: str) -> dict[str, Any]:
    batch_root = Path(subject_root) / "v2_batch_outputs"
    if not batch_root.is_dir():
        raise ScreeningRunError(f"baseline_batch_root_missing:{record_id}")
    batches = sorted(
        (path for path in batch_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    inspected: list[str] = []
    for batch in batches:
        json_root = batch / "json"
        if not json_root.is_dir():
            continue
        candidates = sorted(json_root.glob(f"{record_id}*.json"))
        for path in candidates:
            inspected.append(str(path))
            try:
                payload = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if compatible_baseline(payload, record_id):
                return {
                    "batch_id": batch.name,
                    "report_path": str(path.resolve()),
                    "report_sha256": file_sha256(path),
                    "payload": payload,
                    "selection_rule": "newest_lexicographic_compatible_batch",
                }
    raise ScreeningRunError(
        f"compatible_historical_baseline_missing:{record_id}:{len(inspected)}"
    )


def replay_bo120(
    cell_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_coordinate = {str(row["coordinate_id"]): row for row in cell_rows}
    if len(by_coordinate) != 300:
        raise ScreeningRunError("bo_replay_coordinate_count_mismatch")
    seen: set[str] = set()
    history: list[dict[str, Any]] = []
    repeat_best: list[dict[str, Any]] = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for repeat_idx, seed in enumerate(BO_SEEDS):
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=BO_STARTUP_TRIALS,
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(
            trial: optuna.trial.Trial,
            *,
            repeat_idx: int = repeat_idx,
            seed: int = seed,
        ) -> float:
            fs_target = trial.suggest_categorical("fs_target", [25, 50, 100])
            memory_ms = trial.suggest_categorical(
                "memory_ms", [40, 80, 120, 160, 200]
            )
            mu_base = trial.suggest_categorical(
                "mu_base", [0.006, 0.008, 0.010, 0.012, 0.016]
            )
            half_width = trial.suggest_categorical(
                "exclusion_half_width_bpm", [3, 6, 12, 18]
            )
            coordinate_id = (
                f"physical4d:fs{int(fs_target):03d}:m{int(memory_ms):03d}:"
                f"mu{int(round(float(mu_base) * 1000)):04d}:w{int(half_width):03d}"
            )
            row = by_coordinate[coordinate_id]
            duplicate = coordinate_id in seen
            seen.add(coordinate_id)
            history.append(
                {
                    "repeat_idx": repeat_idx,
                    "seed": seed,
                    "trial_idx": int(trial.number),
                    "global_trial_idx": len(history),
                    "coordinate_id": coordinate_id,
                    "params": {
                        "fs_target": int(fs_target),
                        "memory_ms": int(memory_ms),
                        "mu_base": float(mu_base),
                        "exclusion_half_width_bpm": int(half_width),
                    },
                    "objective_mae_bpm": float(row["metrics"]["mae_bpm"]),
                    "qualified": bool(row["gate"]["qualified"]),
                    "duplicate_coordinate": duplicate,
                    "cache_hit": True,
                    "new_solver": False,
                }
            )
            return float(row["metrics"]["mae_bpm"])

        study.optimize(objective, n_trials=BO_TRIALS_PER_REPEAT)
        best = min(
            (row for row in history if int(row["repeat_idx"]) == repeat_idx),
            key=lambda row: (float(row["objective_mae_bpm"]), int(row["trial_idx"])),
        )
        repeat_best.append(dict(best))
    gate_rows = [row for row in history if row["qualified"]]
    best_observed = min(
        history,
        key=lambda row: (
            float(row["objective_mae_bpm"]),
            int(row["global_trial_idx"]),
        ),
    )
    best_gate = (
        None
        if not gate_rows
        else min(
            gate_rows,
            key=lambda row: (
                float(row["objective_mae_bpm"]),
                int(row["global_trial_idx"]),
            ),
        )
    )
    return {
        "schema_id": "physical4d_bo120_replay_v1",
        "logical_trial_count": len(history),
        "unique_coordinate_count": len(seen),
        "duplicate_logical_trial_count": len(history) - len(seen),
        "new_solver_count": 0,
        "cache_hit_count": len(history),
        "repeat_best": repeat_best,
        "best_observed": best_observed,
        "best_gate_passing_observed": best_gate,
        "history": history,
    }


def run_lyx_stage(
    *,
    worktree: Path,
    external_cache_root: Path,
    data_root: Path,
    output_root: Path,
    preflight_only: bool = False,
) -> dict[str, Any]:
    identity = source_identity(worktree)
    external = validate_external_cache(external_cache_root)
    output_root = Path(output_root).resolve()
    proposal = {
        "schema_id": "multiperson_joint_screening_proposal_v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at": now_iso(),
        "stage": "lyx",
        "source_identity": identity,
        "external_cache_binding": {
            key: value
            for key, value in external.items()
            if key not in {"records", "coordinates", "calls", "completion"}
        },
        "execution_boundary": "dataset_screening_only_no_loso",
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    write_json(output_root / "proposal.json", proposal)
    if preflight_only:
        completion = {
            "schema_id": "multiperson_joint_screening_preflight_v1",
            "status": "preflight_complete",
            "proposal_sha256": proposal["proposal_sha256"],
        }
        write_json(output_root / "preflight.json", completion)
        return completion

    records = list(external["records"])
    coordinates = list(external["coordinates"])
    calls = list(external["calls"])
    calls_by_record: dict[str, dict[str, Mapping[str, Any]]] = {}
    for call in calls:
        calls_by_record.setdefault(str(call["record_id"]), {})[
            str(call["coordinate_id"])
        ] = call
    coordinate_ids = {str(row["coordinate_id"]) for row in coordinates}
    if any(set(value) != coordinate_ids for value in calls_by_record.values()):
        raise ScreeningRunError("external_cache_record_coordinate_coverage_mismatch")

    bias_manifest_rows: list[dict[str, Any]] = []
    baseline_inventory: list[dict[str, Any]] = []
    record_summaries: list[dict[str, Any]] = []
    verified_report_count = 0
    verified_report_hashes: list[str] = []

    for record_index, record in enumerate(records, start=1):
        record_id = str(record["record_id"])
        scene = str(record["scene"])
        ref_path = Path(str(record["ref_path"]))
        if file_sha256(ref_path) != str(record["ref_sha256"]):
            raise ScreeningRunError(f"live_reference_hash_mismatch:{record_id}")
        ref_data = load_v2_reference(ref_path)
        anchor_call = calls_by_record[record_id][ANCHOR_COORDINATE_ID]
        anchor_payload, _ = read_verified_external_report(anchor_call)
        calibration = calibrate_rest_time_bias(
            solver_result_from_payload(anchor_payload),
            ref_data=ref_data,
        )
        selected_bias = float(calibration["r_all"]["selected_bias_s"])
        bias_row = {
            "record_id": record_id,
            "scene": scene,
            "subject": "LYX",
            "data_sha256": str(record["data_sha256"]),
            "ref_sha256": str(record["ref_sha256"]),
            "anchor_coordinate_id": ANCHOR_COORDINATE_ID,
            "anchor_cache_key": str(anchor_call["cache_key"]),
            "selected_bias_s": selected_bias,
            "calibration": calibration,
        }
        bias_manifest_rows.append(bias_row)

        subject_root = Path(str(record["data_path"])).parent
        baseline = find_historical_baseline(subject_root, record_id)
        baseline_result = solver_result_from_payload(baseline.pop("payload"))
        baseline_metrics = evaluate_aligned_metrics(
            baseline_result,
            ref_data=ref_data,
            time_bias_s=selected_bias,
        )
        baseline_item = {
            "record_id": record_id,
            "scene": scene,
            "subject": "LYX",
            **baseline,
            "metrics": baseline_metrics,
        }
        baseline_inventory.append(baseline_item)

        cell_rows: list[dict[str, Any]] = []
        for coordinate in coordinates:
            coordinate_id = str(coordinate["coordinate_id"])
            payload, report_sha = read_verified_external_report(
                calls_by_record[record_id][coordinate_id]
            )
            verified_report_count += 1
            verified_report_hashes.append(report_sha)
            metrics = evaluate_aligned_metrics(
                solver_result_from_payload(payload),
                ref_data=ref_data,
                time_bias_s=selected_bias,
            )
            gate = evaluate_screening_gate(
                candidate=metrics,
                baseline=baseline_metrics,
            )
            cell_rows.append(
                {
                    "record_id": record_id,
                    "scene": scene,
                    "subject": "LYX",
                    "coordinate_id": coordinate_id,
                    "cache_key": str(
                        calls_by_record[record_id][coordinate_id]["cache_key"]
                    ),
                    "report_sha256": report_sha,
                    "time_bias_s": selected_bias,
                    "metrics": metrics,
                    "gate": gate,
                }
            )
        write_jsonl(output_root / "lyx" / "cells" / f"{record_id}.jsonl", cell_rows)
        best_unconstrained = min(
            cell_rows,
            key=lambda row: (float(row["metrics"]["mae_bpm"]), row["coordinate_id"]),
        )
        passing = [row for row in cell_rows if row["gate"]["qualified"]]
        best_gate = (
            None
            if not passing
            else min(
                passing,
                key=lambda row: (
                    float(row["metrics"]["mae_bpm"]),
                    row["coordinate_id"],
                ),
            )
        )
        bo = replay_bo120(cell_rows)
        selected_for_effect = best_gate or best_unconstrained
        selected_call = calls_by_record[record_id][
            str(selected_for_effect["coordinate_id"])
        ]
        selected_payload, _ = read_verified_external_report(selected_call)
        fixed_five_metrics = evaluate_aligned_metrics(
            solver_result_from_payload(selected_payload),
            ref_data=ref_data,
            time_bias_s=BIAS_DEFAULT_S,
        )
        summary = {
            "schema_id": "lyx_joint_screening_record_v1",
            "record_id": record_id,
            "scene": scene,
            "subject": "LYX",
            "selected_bias_s": selected_bias,
            "bias_fallback_reason": calibration["r_all"]["fallback_reason"],
            "baseline": baseline_item,
            "coordinate_count": len(cell_rows),
            "gate_passing_coordinate_count": len(passing),
            "failed_gate_counts": dict(
                Counter(
                    reason
                    for row in cell_rows
                    for reason in row["gate"]["failed_gates"]
                )
            ),
            "best_unconstrained": best_unconstrained,
            "best_gate_passing": best_gate,
            "bo120": bo,
            "selected_coordinate_bias_effect": {
                "coordinate_id": selected_for_effect["coordinate_id"],
                "selected_bias_metrics": selected_for_effect["metrics"],
                "fixed_5s_metrics": fixed_five_metrics,
                "mae_improvement_bpm": float(fixed_five_metrics["mae_bpm"])
                - float(selected_for_effect["metrics"]["mae_bpm"]),
            },
        }
        write_json(output_root / "lyx" / "records" / record_id / "summary.json", summary)
        record_summaries.append(summary)
        print(
            json.dumps(
                {
                    "stage": "lyx",
                    "record_index": record_index,
                    "record_count": len(records),
                    "record_id": record_id,
                    "selected_bias_s": selected_bias,
                    "gate_passing_coordinates": len(passing),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    scenes: list[dict[str, Any]] = []
    for scene in SCENES:
        scene_records = [row for row in record_summaries if row["scene"] == scene]
        passing_records = [
            row for row in scene_records if row["best_gate_passing"] is not None
        ]
        selected = (
            None
            if not passing_records
            else min(
                passing_records,
                key=lambda row: (
                    float(row["best_gate_passing"]["metrics"]["mae_bpm"]),
                    row["record_id"],
                ),
            )
        )
        scenes.append(
            {
                "scene": scene,
                "status": (
                    "qualified_lyx_available"
                    if selected is not None
                    else "failed_no_qualified_lyx"
                ),
                "selected_record_id": None if selected is None else selected["record_id"],
                "selected_coordinate_id": (
                    None
                    if selected is None
                    else selected["best_gate_passing"]["coordinate_id"]
                ),
                "selected_mae_bpm": (
                    None
                    if selected is None
                    else selected["best_gate_passing"]["metrics"]["mae_bpm"]
                ),
                "eligible_record_count": len(passing_records),
            }
        )

    bias_manifest = {
        "schema_id": "record_level_rest_calibrated_time_bias_manifest_v1",
        "record_count": len(bias_manifest_rows),
        "records": bias_manifest_rows,
    }
    bias_manifest["manifest_sha256"] = canonical_sha256(bias_manifest)
    write_json(output_root / "bias_manifest.json", bias_manifest)
    baseline_payload = {
        "schema_id": "historical_baseline_inventory_v1",
        "record_count": len(baseline_inventory),
        "records": baseline_inventory,
    }
    baseline_payload["manifest_sha256"] = canonical_sha256(baseline_payload)
    write_json(output_root / "baseline_inventory_lyx.json", baseline_payload)
    write_json(output_root / "lyx_scene_selection.json", {"scenes": scenes})
    status = (
        "complete_lyx_qualified_all_scenes"
        if all(row["status"] == "qualified_lyx_available" for row in scenes)
        else "complete_negative_lyx_missing_scene"
    )
    completion = {
        "schema_id": "multiperson_joint_screening_lyx_completion_v1",
        "status": status,
        "completed_at": now_iso(),
        "proposal_sha256": proposal["proposal_sha256"],
        "record_count": len(records),
        "coordinate_count_per_record": 300,
        "verified_external_report_count": verified_report_count,
        "verified_external_report_hash_set_sha256": canonical_sha256(
            sorted(verified_report_hashes)
        ),
        "scenes": scenes,
        "bias_manifest_sha256": file_sha256(output_root / "bias_manifest.json"),
        "baseline_inventory_sha256": file_sha256(
            output_root / "baseline_inventory_lyx.json"
        ),
        "no_loso_executed": True,
    }
    write_json(output_root / "lyx_completion.json", completion)
    return completion


def discover_multiperson_records(data_root: Path) -> list[dict[str, Any]]:
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise ScreeningRunError(f"multiperson_data_root_missing:{root}")
    records: list[dict[str, Any]] = []
    for subject_root in sorted(path for path in root.iterdir() if path.is_dir()):
        subject = subject_root.name.split("-", 1)[-1]
        for data_path in sorted(subject_root.glob("*.csv")):
            if data_path.name.endswith(("_HR_ref.csv", "_ref.csv")):
                continue
            record_id = data_path.stem
            scene_token = record_id.split("_", 1)[0]
            scene = "".join(
                character for character in scene_token if character.isalpha()
            ).lower()
            if scene not in SCENES:
                continue
            ref_path = subject_root / f"{record_id}_HR_ref.csv"
            if not ref_path.is_file():
                raise ScreeningRunError(f"paired_reference_missing:{record_id}")
            records.append(
                {
                    "record_id": record_id,
                    "scene": scene,
                    "subject": subject,
                    "subject_root": str(subject_root.resolve()),
                    "data_path": str(data_path.resolve()),
                    "ref_path": str(ref_path.resolve()),
                    "data_sha256": file_sha256(data_path),
                    "ref_sha256": file_sha256(ref_path),
                }
            )
    if len(records) != 184:
        raise ScreeningRunError(f"multiperson_record_count_mismatch:{len(records)}")
    if len({str(row["record_id"]) for row in records}) != len(records):
        raise ScreeningRunError("multiperson_record_id_not_unique")
    return records


def local_cache_identity(
    record: Mapping[str, Any], coordinate: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_id": "multiperson_joint_solver_cell_v1",
        "algorithm": {
            "python_src_tree": EXPECTED_PYTHON_SRC_TREE,
            "ppg_hr_tree": EXPECTED_PPG_HR_TREE,
            "algorithm_preset": "lite",
            "adaptive_filter": "lms",
            "reference_groups_order": ["HF"],
            "rise_candidate_lineage_enable": True,
            "rise_confirmation_policy_id": "legacy_v1",
            "penalty_candidate_id": EXPECTED_PENALTY_ID,
        },
        "input": {
            "record_id": record["record_id"],
            "scene": record["scene"],
            "subject": record["subject"],
            "data_sha256": record["data_sha256"],
            "ref_sha256": record["ref_sha256"],
        },
        "coordinate": dict(coordinate),
        "solver_params": {
            "analysis_scope": "full",
            "fs_target": int(coordinate["fs_target"]),
            "max_order": int(coordinate["max_order"]),
            "lms_mu_base": float(coordinate["lms_mu_base"]),
            "lms_mu_min": 1e-6,
            "smooth_win_len": 5,
            "spec_penalty_width": float(coordinate["spec_penalty_width"]),
            "time_bias": 5.0,
        },
    }


def solve_local_cell(
    *,
    record: Mapping[str, Any],
    coordinate: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    identity = local_cache_identity(record, coordinate)
    cache_key = canonical_sha256(identity)
    entry = Path(output_root) / "execution" / "cache" / "solver" / cache_key[:24]
    complete_path = entry / "complete.json"
    report_path = entry / "report-v2.json"
    failed_path = entry / "failed.json"
    if complete_path.is_file():
        complete = read_json(complete_path)
        if complete.get("cache_key") != cache_key:
            raise ScreeningRunError(f"local_cache_key_collision:{cache_key}")
        if not report_path.is_file() or file_sha256(report_path) != complete.get(
            "report_sha256"
        ):
            raise ScreeningRunError(f"local_cache_report_mismatch:{cache_key}")
        return {
            "cache_key": cache_key,
            "cache_hit": True,
            "new_solver": False,
            "report_path": str(report_path.resolve()),
            "report_sha256": str(complete["report_sha256"]),
            "elapsed_s": float(complete.get("elapsed_s", 0.0)),
        }
    if report_path.exists() or failed_path.exists() or entry.exists():
        raise ScreeningRunError(f"incomplete_local_cache_entry:{entry}")
    entry.mkdir(parents=True, exist_ok=False)
    params = identity["solver_params"]
    config = V2RunConfig(
        data_path=Path(str(record["data_path"])),
        ref_path=Path(str(record["ref_path"])),
        ppg_mode="green",
        ppg_input_transform="raw_bandpass",
        adaptive_filter="lms",
        algorithm_preset="lite",
        reference_groups_order=("HF",),
        adaptive_reference_stage_limit=None,
        rise_candidate_lineage_enable=True,
        rise_confirmation_policy_id="legacy_v1",
        penalty_candidate_id=EXPECTED_PENALTY_ID,
        **params,
    )
    started = time.monotonic()
    try:
        result = solve_v2(config)
        save_v2_report(
            report_path,
            result,
            best_params={
                **params,
                "rise_candidate_lineage_enable": True,
                "rise_confirmation_policy_id": "legacy_v1",
                "penalty_candidate_id": EXPECTED_PENALTY_ID,
            },
            history=[],
            artefacts={
                "experiment_id": EXPERIMENT_ID,
                "record_id": record["record_id"],
                "subject": record["subject"],
                "scene": record["scene"],
                "coordinate_id": coordinate["coordinate_id"],
                "cache_key": cache_key,
            },
        )
        payload = read_json(report_path)
        validate_joint_report_identity(payload)
        elapsed_s = time.monotonic() - started
        report_sha = file_sha256(report_path)
        complete = {
            "schema_id": "multiperson_joint_solver_complete_v1",
            "status": "complete",
            "completed_at": now_iso(),
            "cache_key": cache_key,
            "cache_identity": identity,
            "record_id": record["record_id"],
            "subject": record["subject"],
            "scene": record["scene"],
            "coordinate_id": coordinate["coordinate_id"],
            "elapsed_s": elapsed_s,
            "report_path": str(report_path.resolve()),
            "report_sha256": report_sha,
        }
        write_json(complete_path, complete)
        return {
            "cache_key": cache_key,
            "cache_hit": False,
            "new_solver": True,
            "report_path": str(report_path.resolve()),
            "report_sha256": report_sha,
            "elapsed_s": elapsed_s,
        }
    except Exception as exc:
        write_json(
            failed_path,
            {
                "schema_id": "multiperson_joint_solver_failure_v1",
                "status": "failed",
                "failed_at": now_iso(),
                "cache_key": cache_key,
                "record_id": record["record_id"],
                "coordinate_id": coordinate["coordinate_id"],
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def read_local_report(solve_receipt: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(solve_receipt["report_path"]))
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != solve_receipt.get("report_sha256"):
        raise ScreeningRunError(f"local_report_hash_mismatch:{path}")
    payload = json.loads(raw.decode("utf-8"))
    validate_joint_report_identity(payload)
    return payload


def build_non_lyx_preselection(
    *,
    records: Sequence[Mapping[str, Any]],
    coordinates_by_id: Mapping[str, Mapping[str, Any]],
    output_root: Path,
    workers: int,
) -> dict[str, Any]:
    non_lyx = [dict(row) for row in records if str(row["subject"]) != "LYX"]
    anchor = coordinates_by_id[ANCHOR_COORDINATE_ID]
    receipts: dict[str, dict[str, Any]] = {}
    anchor_failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[dict[str, Any]], dict[str, Any]] = {
            pool.submit(
                solve_local_cell,
                record=record,
                coordinate=anchor,
                output_root=output_root,
            ): record
            for record in non_lyx
        }
        completed = 0
        for future in as_completed(futures):
            record = futures[future]
            try:
                receipt = future.result()
            except Exception as exc:
                anchor_failures[str(record["record_id"])] = (
                    f"{type(exc).__name__}:{exc}"
                )
            else:
                receipts[str(record["record_id"])] = receipt
            completed += 1
            if completed == 1 or completed % 20 == 0 or completed == len(non_lyx):
                print(
                    json.dumps(
                        {
                            "stage": "non_lyx_anchor",
                            "complete": completed,
                            "expected": len(non_lyx),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    bias_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    priority_rows: list[dict[str, Any]] = []
    for record in non_lyx:
        record_id = str(record["record_id"])
        if record_id not in receipts:
            failed_quality = {
                "hard_quality_pass": False,
                "baseline_prior_pass": False,
                "fallback_count": 1,
                "unreliable_windows": 0,
                "dropped_window_ratio": 1.0,
                "channel_outlier_ratio": 1.0,
                "valid_windows": 0,
            }
            failure_reason = anchor_failures.get(record_id, "anchor_receipt_missing")
            bias_rows.append(
                {
                    **record,
                    "status": "failed_closed",
                    "failure_reason": failure_reason,
                    "selected_bias_s": BIAS_DEFAULT_S,
                }
            )
            baseline_rows.append(
                {
                    **record,
                    "selected_bias_s": BIAS_DEFAULT_S,
                    "status": "failed_closed",
                    "failure_reason": failure_reason,
                    "metrics": None,
                    "quality": failed_quality,
                }
            )
            priority_rows.append(
                {
                    **record,
                    "selected_bias_s": BIAS_DEFAULT_S,
                    "baseline_report_sha256": None,
                    "baseline_metrics": None,
                    "quality": failed_quality,
                    "priority_key": [1, 1, 1e9, 1, 0, 1.0, 1.0, 0, record_id],
                }
            )
            continue
        ref_data = load_v2_reference(Path(str(record["ref_path"])))
        anchor_payload = read_local_report(receipts[record_id])
        calibration = calibrate_rest_time_bias(
            solver_result_from_payload(anchor_payload), ref_data=ref_data
        )
        selected_bias = float(calibration["r_all"]["selected_bias_s"])
        bias_rows.append(
            {
                **record,
                "selected_bias_s": selected_bias,
                "anchor_coordinate_id": ANCHOR_COORDINATE_ID,
                "anchor_cache_key": receipts[record_id]["cache_key"],
                "anchor_report_sha256": receipts[record_id]["report_sha256"],
                "calibration": calibration,
            }
        )
        try:
            baseline = find_historical_baseline(
                Path(str(record["subject_root"])), record_id
            )
            baseline_payload = baseline.pop("payload")
            baseline_metrics = evaluate_aligned_metrics(
                solver_result_from_payload(baseline_payload),
                ref_data=ref_data,
                time_bias_s=selected_bias,
            )
        except Exception as exc:
            failed_quality = {
                "hard_quality_pass": False,
                "baseline_prior_pass": False,
                "fallback_count": 1,
                "unreliable_windows": 0,
                "dropped_window_ratio": 1.0,
                "channel_outlier_ratio": 1.0,
                "valid_windows": 0,
            }
            baseline_rows.append(
                {
                    **record,
                    "selected_bias_s": selected_bias,
                    "status": "failed_closed",
                    "failure_reason": f"{type(exc).__name__}:{exc}",
                    "metrics": None,
                    "quality": failed_quality,
                }
            )
            priority_rows.append(
                {
                    **record,
                    "selected_bias_s": selected_bias,
                    "baseline_report_sha256": None,
                    "baseline_metrics": None,
                    "quality": failed_quality,
                    "priority_key": [1, 1, 1e9, 1, 0, 1.0, 1.0, 0, record_id],
                }
            )
            continue
        qc = dict(baseline_payload.get("qc") or {})
        overlap = dict(baseline_payload.get("reference_overlap") or {})
        valid_windows = int(overlap.get("valid_windows") or 0)
        dropped_windows = int(overlap.get("dropped_windows") or 0)
        total_overlap = valid_windows + dropped_windows
        dropped_ratio = (
            float(dropped_windows / total_overlap) if total_overlap else 1.0
        )
        outlier_ratio = max(
            float(qc.get("outlier_ratio_ut1") or 0.0),
            float(qc.get("outlier_ratio_ut2") or 0.0),
        )
        fallback_count = int(bool(str(baseline_payload.get("fallback_reason") or "")))
        unreliable = int(baseline_payload.get("unreliable_windows") or 0)
        hard_quality_pass = bool(qc.get("is_good") is not False and valid_windows > 0)
        baseline_prior_pass = bool(
            hard_quality_pass
            and baseline_metrics["right_censored_recovery_count"] == 0
            and baseline_metrics["l10"] <= 20
        )
        baseline_row = {
            **record,
            **baseline,
            "selected_bias_s": selected_bias,
            "metrics": baseline_metrics,
            "quality": {
                "hard_quality_pass": hard_quality_pass,
                "baseline_prior_pass": baseline_prior_pass,
                "fallback_count": fallback_count,
                "unreliable_windows": unreliable,
                "dropped_window_ratio": dropped_ratio,
                "channel_outlier_ratio": outlier_ratio,
                "valid_windows": valid_windows,
            },
        }
        baseline_rows.append(baseline_row)
        priority_rows.append(
            {
                **record,
                "selected_bias_s": selected_bias,
                "baseline_report_sha256": baseline["report_sha256"],
                "baseline_metrics": baseline_metrics,
                "quality": baseline_row["quality"],
                "priority_key": [
                    0 if hard_quality_pass else 1,
                    0 if baseline_prior_pass else 1,
                    float(baseline_metrics["mae_bpm"]),
                    fallback_count,
                    unreliable,
                    dropped_ratio,
                    outlier_ratio,
                    -valid_windows,
                    record_id,
                ],
            }
        )

    priority_rows.sort(
        key=lambda row: (
            str(row["scene"]),
            str(row["subject"]),
            tuple(row["priority_key"]),
        )
    )
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in priority_rows:
        by_group.setdefault((str(row["scene"]), str(row["subject"])), []).append(row)
    for values in by_group.values():
        for rank, row in enumerate(values, start=1):
            row["subject_scene_rank"] = rank
            row["schedule_role"] = "primary" if rank == 1 else "backup"
    bias_payload = {
        "schema_id": "record_level_rest_calibrated_time_bias_manifest_v1",
        "record_count": len(bias_rows),
        "records": bias_rows,
    }
    bias_payload["manifest_sha256"] = canonical_sha256(bias_payload)
    write_json(output_root / "bias_manifest_non_lyx.json", bias_payload)
    baseline_payload = {
        "schema_id": "historical_baseline_inventory_v1",
        "record_count": len(baseline_rows),
        "records": baseline_rows,
    }
    baseline_payload["manifest_sha256"] = canonical_sha256(baseline_payload)
    write_json(output_root / "baseline_inventory_non_lyx.json", baseline_payload)
    priority_payload = {
        "schema_id": "multiperson_screening_priority_v1",
        "selection_inputs": [
            "pairing_and_qc",
            "historical_baseline_only",
            "rest_calibrated_bias",
        ],
        "joint_anchor_performance_not_used_for_ranking": True,
        "record_count": len(priority_rows),
        "records": priority_rows,
    }
    priority_payload["manifest_sha256"] = canonical_sha256(priority_payload)
    write_json(output_root / "preselection_priority.json", priority_payload)
    completion = {
        "schema_id": "multiperson_non_lyx_preselection_completion_v1",
        "status": "complete",
        "completed_at": now_iso(),
        "record_count": len(non_lyx),
        "anchor_new_solver_count": sum(
            int(receipt["new_solver"]) for receipt in receipts.values()
        ),
        "anchor_cache_hit_count": sum(
            int(receipt["cache_hit"]) for receipt in receipts.values()
        ),
        "anchor_failure_count": len(anchor_failures),
        "anchor_failures": anchor_failures,
        "bias_manifest_sha256": file_sha256(
            output_root / "bias_manifest_non_lyx.json"
        ),
        "baseline_inventory_sha256": file_sha256(
            output_root / "baseline_inventory_non_lyx.json"
        ),
        "priority_manifest_sha256": file_sha256(
            output_root / "preselection_priority.json"
        ),
    }
    write_json(output_root / "non_lyx_preselection_completion.json", completion)
    return {
        "completion": completion,
        "bias_rows": bias_rows,
        "baseline_rows": baseline_rows,
        "priority_rows": priority_rows,
    }


def run_record_bo120(
    *,
    record: Mapping[str, Any],
    preselection: Mapping[str, Any],
    baseline: Mapping[str, Any],
    coordinates_by_id: Mapping[str, Mapping[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    record_id = str(record["record_id"])
    record_dir = Path(output_root) / "non_lyx" / "records" / record_id
    summary_path = record_dir / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        if (
            summary.get("status") == "complete"
            and summary.get("data_sha256") == record.get("data_sha256")
            and summary.get("ref_sha256") == record.get("ref_sha256")
        ):
            return summary
        raise ScreeningRunError(f"record_summary_identity_mismatch:{record_id}")
    selected_bias = float(preselection["selected_bias_s"])
    baseline_metrics = dict(baseline["metrics"])
    ref_data = load_v2_reference(Path(str(record["ref_path"])))
    history: list[dict[str, Any]] = []
    seen: set[str] = set()
    new_solver_count = 0
    cache_hit_count = 0
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for repeat_idx, seed in enumerate(BO_SEEDS):
        sampler = optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=BO_STARTUP_TRIALS
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(
            trial: optuna.trial.Trial,
            *,
            repeat_idx: int = repeat_idx,
            seed: int = seed,
        ) -> float:
            nonlocal new_solver_count, cache_hit_count
            fs_target = trial.suggest_categorical("fs_target", [25, 50, 100])
            memory_ms = trial.suggest_categorical(
                "memory_ms", [40, 80, 120, 160, 200]
            )
            mu_base = trial.suggest_categorical(
                "mu_base", [0.006, 0.008, 0.010, 0.012, 0.016]
            )
            half_width = trial.suggest_categorical(
                "exclusion_half_width_bpm", [3, 6, 12, 18]
            )
            coordinate_id = (
                f"physical4d:fs{int(fs_target):03d}:m{int(memory_ms):03d}:"
                f"mu{int(round(float(mu_base) * 1000)):04d}:w{int(half_width):03d}"
            )
            receipt = solve_local_cell(
                record=record,
                coordinate=coordinates_by_id[coordinate_id],
                output_root=output_root,
            )
            new_solver_count += int(receipt["new_solver"])
            cache_hit_count += int(receipt["cache_hit"])
            metrics = evaluate_aligned_metrics(
                solver_result_from_payload(read_local_report(receipt)),
                ref_data=ref_data,
                time_bias_s=selected_bias,
            )
            gate = evaluate_screening_gate(
                candidate=metrics, baseline=baseline_metrics
            )
            duplicate = coordinate_id in seen
            seen.add(coordinate_id)
            row = {
                "repeat_idx": repeat_idx,
                "seed": seed,
                "trial_idx": int(trial.number),
                "global_trial_idx": len(history),
                "coordinate_id": coordinate_id,
                "params": {
                    "fs_target": int(fs_target),
                    "memory_ms": int(memory_ms),
                    "mu_base": float(mu_base),
                    "exclusion_half_width_bpm": int(half_width),
                },
                "objective_mae_bpm": float(metrics["mae_bpm"]),
                "metrics": metrics,
                "gate": gate,
                "duplicate_coordinate": duplicate,
                "cache_hit": bool(receipt["cache_hit"]),
                "new_solver": bool(receipt["new_solver"]),
                "cache_key": receipt["cache_key"],
                "report_sha256": receipt["report_sha256"],
            }
            history.append(row)
            write_json(
                record_dir / "checkpoint.json",
                {
                    "schema_id": "multiperson_bo120_checkpoint_v1",
                    "record_id": record_id,
                    "logical_trial_count": len(history),
                    "history": history,
                },
            )
            return float(metrics["mae_bpm"])

        study.optimize(objective, n_trials=BO_TRIALS_PER_REPEAT)
    if len(history) != 120:
        raise ScreeningRunError(f"bo120_logical_count_mismatch:{record_id}")
    best_observed = min(
        history,
        key=lambda row: (
            float(row["objective_mae_bpm"]),
            int(row["global_trial_idx"]),
        ),
    )
    passing = [row for row in history if row["gate"]["qualified"]]
    best_gate = (
        None
        if not passing
        else min(
            passing,
            key=lambda row: (
                float(row["objective_mae_bpm"]),
                int(row["global_trial_idx"]),
            ),
        )
    )
    selected_for_effect = best_gate or best_observed
    selected_receipt = solve_local_cell(
        record=record,
        coordinate=coordinates_by_id[str(selected_for_effect["coordinate_id"])],
        output_root=output_root,
    )
    fixed_five_metrics = evaluate_aligned_metrics(
        solver_result_from_payload(read_local_report(selected_receipt)),
        ref_data=ref_data,
        time_bias_s=BIAS_DEFAULT_S,
    )
    summary = {
        "schema_id": "multiperson_joint_bo120_record_v1",
        "status": "complete",
        "record_id": record_id,
        "scene": record["scene"],
        "subject": record["subject"],
        "data_path": record["data_path"],
        "ref_path": record["ref_path"],
        "data_sha256": record["data_sha256"],
        "ref_sha256": record["ref_sha256"],
        "selected_bias_s": selected_bias,
        "baseline": baseline,
        "logical_trial_count": len(history),
        "unique_coordinate_count": len(seen),
        "duplicate_logical_trial_count": len(history) - len(seen),
        "new_solver_count": new_solver_count,
        "cache_hit_count": cache_hit_count,
        "best_observed": best_observed,
        "best_gate_passing": best_gate,
        "gate_passing_logical_trial_count": len(passing),
        "selected_coordinate_bias_effect": {
            "coordinate_id": selected_for_effect["coordinate_id"],
            "selected_bias_metrics": selected_for_effect["metrics"],
            "fixed_5s_metrics": fixed_five_metrics,
            "mae_improvement_bpm": float(fixed_five_metrics["mae_bpm"])
            - float(selected_for_effect["metrics"]["mae_bpm"]),
        },
        "history": history,
    }
    write_json(summary_path, summary)
    return summary


def run_bo_batch(
    *,
    record_ids: Sequence[str],
    records_by_id: Mapping[str, Mapping[str, Any]],
    preselection_by_id: Mapping[str, Mapping[str, Any]],
    baseline_by_id: Mapping[str, Mapping[str, Any]],
    coordinates_by_id: Mapping[str, Mapping[str, Any]],
    output_root: Path,
    workers: int,
) -> list[dict[str, Any]]:
    ids = list(dict.fromkeys(str(value) for value in record_ids))
    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_record_bo120,
                record=records_by_id[record_id],
                preselection=preselection_by_id[record_id],
                baseline=baseline_by_id[record_id],
                coordinates_by_id=coordinates_by_id,
                output_root=output_root,
            ): record_id
            for record_id in ids
        }
        completed = 0
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            completed += 1
            print(
                json.dumps(
                    {
                        "stage": "non_lyx_bo120",
                        "complete": completed,
                        "expected": len(ids),
                        "record_id": summary["record_id"],
                        "subject": summary["subject"],
                        "scene": summary["scene"],
                        "unique_coordinates": summary["unique_coordinate_count"],
                        "qualified": summary["best_gate_passing"] is not None,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return summaries


def candidate_rows_from_summaries(
    *, output_root: Path, non_lyx_summaries: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((Path(output_root) / "lyx" / "records").glob("*/summary.json")):
        summary = read_json(path)
        best = summary.get("best_gate_passing")
        rows.append(
            {
                "scene": summary["scene"],
                "subject": "LYX",
                "record_id": summary["record_id"],
                "qualified": best is not None,
                "best_gate_mae_bpm": (
                    None if best is None else float(best["metrics"]["mae_bpm"])
                ),
                "coordinate_id": None if best is None else best["coordinate_id"],
                "search_evidence": "complete_300_grid",
            }
        )
    for summary in non_lyx_summaries:
        best = summary.get("best_gate_passing")
        rows.append(
            {
                "scene": summary["scene"],
                "subject": summary["subject"],
                "record_id": summary["record_id"],
                "qualified": best is not None,
                "best_gate_mae_bpm": (
                    None if best is None else float(best["metrics"]["mae_bpm"])
                ),
                "coordinate_id": None if best is None else best["coordinate_id"],
                "search_evidence": "bo120_observed",
            }
        )
    return rows


def run_all_stage(
    *,
    worktree: Path,
    external_cache_root: Path,
    data_root: Path,
    output_root: Path,
    workers: int,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    lyx_completion_path = output_root / "lyx_completion.json"
    if not lyx_completion_path.is_file():
        run_lyx_stage(
            worktree=worktree,
            external_cache_root=external_cache_root,
            data_root=data_root,
            output_root=output_root,
        )
    lyx_completion = read_json(lyx_completion_path)
    if lyx_completion.get("status") != "complete_lyx_qualified_all_scenes":
        final = {
            "schema_id": "multiperson_joint_screening_completion_v1",
            "status": "complete_negative_lyx_requirement_failed",
            "no_loso_executed": True,
            "lyx_completion_sha256": file_sha256(lyx_completion_path),
        }
        write_json(output_root / "completion.json", final)
        return final
    identity = source_identity(worktree)
    external = validate_external_cache(external_cache_root)
    coordinates_by_id = {
        str(row["coordinate_id"]): row for row in external["coordinates"]
    }
    records = discover_multiperson_records(data_root)
    records_by_id = {str(row["record_id"]): row for row in records}
    data_manifest = {
        "schema_id": "multiperson_184_record_manifest_v1",
        "record_count": len(records),
        "subject_count": len({str(row["subject"]) for row in records}),
        "scene_count": len({str(row["scene"]) for row in records}),
        "records": records,
    }
    data_manifest["manifest_sha256"] = canonical_sha256(data_manifest)
    write_json(output_root / "data_manifest.json", data_manifest)

    preselection = build_non_lyx_preselection(
        records=records,
        coordinates_by_id=coordinates_by_id,
        output_root=output_root,
        workers=workers,
    )
    preselection_by_id = {
        str(row["record_id"]): row for row in preselection["priority_rows"]
    }
    baseline_by_id = {
        str(row["record_id"]): row for row in preselection["baseline_rows"]
    }
    priorities_by_group: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in preselection["priority_rows"]:
        priorities_by_group.setdefault(
            (str(row["scene"]), str(row["subject"])), []
        ).append(row)
    for values in priorities_by_group.values():
        values.sort(key=lambda row: int(row["subject_scene_rank"]))
    primary_ids = [
        str(values[0]["record_id"])
        for _, values in sorted(priorities_by_group.items())
        if values and values[0]["quality"]["hard_quality_pass"]
    ]
    all_summaries: dict[str, dict[str, Any]] = {}
    for summary in run_bo_batch(
        record_ids=primary_ids,
        records_by_id=records_by_id,
        preselection_by_id=preselection_by_id,
        baseline_by_id=baseline_by_id,
        coordinates_by_id=coordinates_by_id,
        output_root=output_root,
        workers=workers,
    ):
        all_summaries[str(summary["record_id"])] = summary

    while True:
        candidate_rows = candidate_rows_from_summaries(
            output_root=output_root,
            non_lyx_summaries=list(all_summaries.values()),
        )
        panels = [select_scene_panel(candidate_rows, scene=scene) for scene in SCENES]
        deficient = {
            str(panel["scene"])
            for panel in panels
            if int(panel["distinct_subject_count"]) < 6
        }
        if not deficient:
            break
        next_ids: list[str] = []
        for scene in sorted(deficient):
            scene_rows = [row for row in candidate_rows if row["scene"] == scene]
            passed_subjects = {
                str(row["subject"])
                for row in scene_rows
                if row.get("qualified") is True
            }
            for (group_scene, subject), values in sorted(priorities_by_group.items()):
                if group_scene != scene or subject in passed_subjects:
                    continue
                next_row = next(
                    (
                        row
                        for row in values
                        if str(row["record_id"]) not in all_summaries
                        and row["quality"]["hard_quality_pass"]
                    ),
                    None,
                )
                if next_row is not None:
                    next_ids.append(str(next_row["record_id"]))
        next_ids = list(dict.fromkeys(next_ids))
        if not next_ids:
            break
        for summary in run_bo_batch(
            record_ids=next_ids,
            records_by_id=records_by_id,
            preselection_by_id=preselection_by_id,
            baseline_by_id=baseline_by_id,
            coordinates_by_id=coordinates_by_id,
            output_root=output_root,
            workers=workers,
        ):
            all_summaries[str(summary["record_id"])] = summary

    candidate_rows = candidate_rows_from_summaries(
        output_root=output_root,
        non_lyx_summaries=list(all_summaries.values()),
    )
    panels = [select_scene_panel(candidate_rows, scene=scene) for scene in SCENES]
    write_json(
        output_root / "panel_selection.json",
        {
            "schema_id": "multiperson_joint_panel_selection_v1",
            "scenes": panels,
            "candidate_rows": candidate_rows,
        },
    )
    success = all(int(panel["distinct_subject_count"]) >= 5 for panel in panels)
    dataset_records: list[dict[str, Any]] = []
    for panel in panels:
        for row in panel["selected"]:
            record = records_by_id[str(row["record_id"])]
            dataset_records.append(
                {
                    "scene": panel["scene"],
                    "subject": row["subject"],
                    "record_id": row["record_id"],
                    "role": "development_anchor" if row["subject"] == "LYX" else "screened_subject",
                    "data_path": record["data_path"],
                    "ref_path": record["ref_path"],
                    "data_sha256": record["data_sha256"],
                    "ref_sha256": record["ref_sha256"],
                    "selected_bias_s": (
                        next(
                            item["selected_bias_s"]
                            for item in read_json(output_root / "bias_manifest.json")["records"]
                            if item["record_id"] == row["record_id"]
                        )
                        if row["subject"] == "LYX"
                        else preselection_by_id[str(row["record_id"])]["selected_bias_s"]
                    ),
                    "coordinate_id": row["coordinate_id"],
                    "best_gate_mae_bpm": row["best_gate_mae_bpm"],
                    "search_evidence": row["search_evidence"],
                }
            )
    dataset_card = {
        "schema_id": "screened_cross_subject_dataset_card_v1",
        "status": "complete" if success else "failed_insufficient_subjects",
        "created_at": now_iso(),
        "development_subject": "LYX",
        "scene_count": len(panels),
        "records": dataset_records,
        "scene_composition": [
            {
                "scene": panel["scene"],
                "status": panel["status"],
                "subject_count": panel["distinct_subject_count"],
                "subjects": [row["subject"] for row in panel["selected"]],
                "record_ids": [row["record_id"] for row in panel["selected"]],
            }
            for panel in panels
        ],
        "limitations": [
            "performance_screened_development_panel",
            "lyx_is_algorithm_development_subject",
            "not_random_population_sample",
            "not_independent_external_validation",
            "no_loso_executed",
        ],
    }
    dataset_card["dataset_card_sha256"] = canonical_sha256(dataset_card)
    write_json(output_root / "dataset_card.json", dataset_card)
    status = "complete_panel_screened" if success else "requires_fullgrid_escalation"
    completion = {
        "schema_id": "multiperson_joint_screening_completion_v1",
        "status": status,
        "completed_at": now_iso(),
        "source_identity": identity,
        "record_bo_count": len(all_summaries),
        "scene_count": len(panels),
        "successful_scene_count": sum(
            int(int(panel["distinct_subject_count"]) >= 5) for panel in panels
        ),
        "dataset_card_sha256": file_sha256(output_root / "dataset_card.json"),
        "panel_selection_sha256": file_sha256(output_root / "panel_selection.json"),
        "no_loso_executed": True,
    }
    write_json(output_root / "completion.json", completion)
    return completion


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def validate_final_outputs(
    *, output_root: Path, external_cache_root: Path
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    completion_path = root / "completion.json"
    if not completion_path.is_file():
        raise ScreeningRunError("final_completion_missing")
    completion = read_json(completion_path)
    if completion.get("status") != "complete_panel_screened":
        raise ScreeningRunError("final_completion_status_not_success")
    if completion.get("no_loso_executed") is not True:
        raise ScreeningRunError("final_completion_loso_boundary_missing")
    validate_external_cache(external_cache_root)
    if completion.get("dataset_card_sha256") != file_sha256(
        root / "dataset_card.json"
    ):
        raise ScreeningRunError("dataset_card_hash_mismatch")
    if completion.get("panel_selection_sha256") != file_sha256(
        root / "panel_selection.json"
    ):
        raise ScreeningRunError("panel_selection_hash_mismatch")

    data_manifest = read_json(root / "data_manifest.json")
    lyx_bias = read_json(root / "bias_manifest.json")
    non_lyx_bias = read_json(root / "bias_manifest_non_lyx.json")
    lyx_baseline = read_json(root / "baseline_inventory_lyx.json")
    non_lyx_baseline = read_json(root / "baseline_inventory_non_lyx.json")
    panel_selection = read_json(root / "panel_selection.json")
    dataset_card = read_json(root / "dataset_card.json")
    if int(data_manifest.get("record_count", -1)) != 184:
        raise ScreeningRunError("validation_data_record_count")
    bias_rows = list(lyx_bias.get("records") or []) + list(
        non_lyx_bias.get("records") or []
    )
    if len(bias_rows) != 184 or len(
        {str(row["record_id"]) for row in bias_rows}
    ) != 184:
        raise ScreeningRunError("validation_bias_manifest_count")
    baseline_rows = list(lyx_baseline.get("records") or []) + list(
        non_lyx_baseline.get("records") or []
    )
    if len(baseline_rows) != 184:
        raise ScreeningRunError("validation_baseline_inventory_count")
    baseline_failures = [
        row for row in baseline_rows if row.get("status") == "failed_closed"
    ]

    summaries: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "non_lyx" / "records").glob("*/summary.json")):
        summary = read_json(path)
        record_id = str(summary["record_id"])
        if summary.get("status") != "complete":
            raise ScreeningRunError(f"validation_record_not_complete:{record_id}")
        history = list(summary.get("history") or [])
        if int(summary.get("logical_trial_count", -1)) != 120 or len(history) != 120:
            raise ScreeningRunError(f"validation_bo120_count:{record_id}")
        unique = len({str(row["coordinate_id"]) for row in history})
        if unique != int(summary.get("unique_coordinate_count", -1)):
            raise ScreeningRunError(f"validation_unique_count:{record_id}")
        if 120 - unique != int(summary.get("duplicate_logical_trial_count", -1)):
            raise ScreeningRunError(f"validation_duplicate_count:{record_id}")
        summaries[record_id] = summary
    if len(summaries) != 57 or int(completion.get("record_bo_count", -1)) != 57:
        raise ScreeningRunError("validation_record_bo_count")

    local_complete_paths = sorted(
        (root / "execution" / "cache" / "solver").glob("*/complete.json")
    )
    expected_local_solves = 160 + sum(
        int(summary["new_solver_count"]) for summary in summaries.values()
    )
    if len(local_complete_paths) != expected_local_solves:
        raise ScreeningRunError(
            f"validation_local_cache_count:{len(local_complete_paths)}:{expected_local_solves}"
        )
    local_report_hashes: list[str] = []
    for complete_path_item in local_complete_paths:
        complete_item = read_json(complete_path_item)
        report_path = complete_path_item.parent / "report-v2.json"
        if complete_item.get("status") != "complete" or not report_path.is_file():
            raise ScreeningRunError(f"validation_local_cache_incomplete:{complete_path_item}")
        report_sha = file_sha256(report_path)
        if report_sha != complete_item.get("report_sha256"):
            raise ScreeningRunError(f"validation_local_report_hash:{report_path}")
        local_report_hashes.append(report_sha)

    scenes = list(panel_selection.get("scenes") or [])
    if len(scenes) != 8 or len(dataset_card.get("records") or []) != 48:
        raise ScreeningRunError("validation_panel_cardinality")
    for scene in scenes:
        selected = list(scene.get("selected") or [])
        subjects = [str(row["subject"]) for row in selected]
        if (
            scene.get("status") != "complete_six_subjects"
            or len(selected) != 6
            or len(set(subjects)) != 6
            or subjects.count("LYX") != 1
        ):
            raise ScreeningRunError(f"validation_scene_panel:{scene.get('scene')}")

    forbidden_names = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and any(
            token in path.name.lower()
            for token in ("loso", "leave_one", "leave-one", "fold_output", "heldout")
        )
    ]
    if forbidden_names:
        raise ScreeningRunError("validation_loso_artifact_present")

    selected_bias_counts = Counter(float(row["selected_bias_s"]) for row in bias_rows)
    identifiable_count = sum(
        int(bool((row.get("calibration") or {}).get("r_all", {}).get("identifiable")))
        for row in bias_rows
    )
    rest_improvements: list[float] = []
    r_pre_agreement = 0
    for row in bias_rows:
        calibration = row.get("calibration") or {}
        selected_bias = float(row["selected_bias_s"])
        curve = list(calibration.get("curve") or [])
        scores = {
            float(item["bias_s"]): item.get("score_all_bpm") for item in curve
        }
        default_score = scores.get(5.0)
        selected_score = scores.get(selected_bias)
        if default_score is not None and selected_score is not None:
            rest_improvements.append(float(default_score) - float(selected_score))
        r_pre = calibration.get("r_pre") or {}
        if r_pre.get("selected_bias_s") is not None and math.isclose(
            float(r_pre["selected_bias_s"]), selected_bias
        ):
            r_pre_agreement += 1

    selected_record_ids = {
        str(row["record_id"]) for row in dataset_card.get("records") or []
    }
    lyx_summaries = {
        str(summary["record_id"]): summary
        for summary in (
            read_json(path)
            for path in sorted((root / "lyx" / "records").glob("*/summary.json"))
        )
    }
    all_summaries = {**lyx_summaries, **summaries}
    selected_effects = [
        float(all_summaries[record_id]["selected_coordinate_bias_effect"]["mae_improvement_bpm"])
        for record_id in sorted(selected_record_ids)
    ]
    selected_maes = [
        float(row["best_gate_mae_bpm"]) for row in dataset_card.get("records") or []
    ]
    bo_unique = [int(summary["unique_coordinate_count"]) for summary in summaries.values()]
    bo_duplicates = [
        int(summary["duplicate_logical_trial_count"]) for summary in summaries.values()
    ]
    bo_gate_records = sum(
        int(summary.get("best_gate_passing") is not None)
        for summary in summaries.values()
    )
    result_summary = {
        "schema_id": "multiperson_joint_screening_result_summary_v1",
        "material_passport": {
            "schema": "ARS-9",
            "artifact_type": "experiment_result",
            "experiment_id": EXPERIMENT_ID,
            "verification_status": "VERIFIED",
            "source_identity": completion["source_identity"],
        },
        "time_bias": {
            "record_count": len(bias_rows),
            "selected_bias_counts": {
                str(key): value for key, value in sorted(selected_bias_counts.items())
            },
            "identifiable_count": identifiable_count,
            "nonidentifiable_fallback_count": len(bias_rows) - identifiable_count,
            "selected_default_5s_count": int(
                selected_bias_counts.get(BIAS_DEFAULT_S, 0)
            ),
            "r_all_r_pre_agreement_count": r_pre_agreement,
            "rest_score_improvement_vs_5s_bpm": _numeric_summary(rest_improvements),
            "selected_panel_same_coordinate_full_mae_improvement_vs_5s_bpm": {
                **_numeric_summary(selected_effects),
                "improved_count": sum(int(value > 0.0) for value in selected_effects),
                "unchanged_count": sum(int(math.isclose(value, 0.0, abs_tol=1e-12)) for value in selected_effects),
                "worsened_count": sum(int(value < 0.0) for value in selected_effects),
            },
        },
        "bo120": {
            "record_count": len(summaries),
            "logical_trial_count": 120 * len(summaries),
            "unique_coordinate_count": sum(bo_unique),
            "duplicate_logical_trial_count": sum(bo_duplicates),
            "unique_per_record": _numeric_summary(bo_unique),
            "gate_passing_record_count": bo_gate_records,
            "new_solver_count_excluding_anchors": sum(
                int(summary["new_solver_count"]) for summary in summaries.values()
            ),
            "anchor_solver_count": 160,
            "local_physical_solver_count": len(local_complete_paths),
        },
        "dataset": {
            "status": dataset_card["status"],
            "scene_count": 8,
            "record_slot_count": 48,
            "unique_record_count": len(selected_record_ids),
            "subject_count": len(
                {str(row["subject"]) for row in dataset_card.get("records") or []}
            ),
            "selected_mae_bpm": _numeric_summary(selected_maes),
            "scene_composition": dataset_card["scene_composition"],
        },
        "quality": {
            "baseline_failure_count": len(baseline_failures),
            "verified_external_report_count": 7200,
            "verified_local_report_count": len(local_complete_paths),
            "local_report_hash_set_sha256": canonical_sha256(
                sorted(local_report_hashes)
            ),
            "no_loso_executed": True,
        },
    }
    write_json(root / "result_summary.json", result_summary)
    receipt = {
        "schema_id": "multiperson_joint_screening_validation_v1",
        "status": "verified_complete",
        "validated_at": now_iso(),
        "completion_sha256": file_sha256(completion_path),
        "result_summary_sha256": file_sha256(root / "result_summary.json"),
        "dataset_card_sha256": file_sha256(root / "dataset_card.json"),
        "panel_selection_sha256": file_sha256(root / "panel_selection.json"),
        "verified_requirements": {
            "input_records_184": True,
            "time_bias_records_184": True,
            "historical_baselines_184": True,
            "lyx_external_reports_7200": True,
            "bo120_records_57": True,
            "bo120_logical_trials_per_record_120": True,
            "local_report_hashes": True,
            "eight_six_subject_scenes": True,
            "one_lyx_per_scene": True,
            "no_loso_artifacts": True,
        },
    }
    write_json(root / "validation_receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("lyx", "all"), default="all")
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--external-cache-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ScreeningRunError("workers_must_be_positive")
    if args.validate_only:
        completion = validate_final_outputs(
            output_root=args.output_root,
            external_cache_root=args.external_cache_root,
        )
    elif args.stage == "lyx" or args.preflight_only:
        completion = run_lyx_stage(
            worktree=args.worktree,
            external_cache_root=args.external_cache_root,
            data_root=args.data_root,
            output_root=args.output_root,
            preflight_only=bool(args.preflight_only),
        )
    else:
        completion = run_all_stage(
            worktree=args.worktree,
            external_cache_root=args.external_cache_root,
            data_root=args.data_root,
            output_root=args.output_root,
            workers=int(args.workers),
        )
    print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
