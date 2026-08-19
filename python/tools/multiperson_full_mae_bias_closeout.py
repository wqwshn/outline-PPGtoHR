"""Finalize one frozen cross-subject panel with post-solver full-MAE bias tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PYTHON_SRC = Path(__file__).parents[1] / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))
TOOLS_DIR = Path(__file__).parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from multiperson_joint_screening import (  # noqa: E402
    EXPERIMENT_ID,
    canonical_sha256,
    evaluate_aligned_metrics,
    evaluate_screening_gate,
    file_sha256,
    read_json,
    read_verified_external_report,
    solver_result_from_payload,
    validate_external_cache,
    validate_joint_report_identity,
    write_json,
)
from multiperson_screening_contracts import (  # noqa: E402
    BIAS_DEFAULT_S,
    select_full_mae_time_bias,
)

from ppg_hr.v2.preprocess import load_v2_reference  # noqa: E402

CLOSEOUT_CONTRACT = "full_mae_evaluation_time_bias_v1"
CLOSEOUT_DIR = Path("execution") / "full_mae_bias_closeout"


class FullMaeBiasCloseoutError(RuntimeError):
    """Raised when closeout would change a frozen screening decision."""


def build_updated_dataset_card(
    original: Mapping[str, Any],
    *,
    evaluations: Mapping[str, Mapping[str, Any]],
    updated_at: str,
) -> dict[str, Any]:
    """Replace only final alignment fields on a frozen dataset card."""

    original_records = list(original.get("records") or [])
    original_ids = [str(row["record_id"]) for row in original_records]
    if set(original_ids) != set(evaluations):
        raise FullMaeBiasCloseoutError("evaluation_record_set_mismatch")
    updated_records: list[dict[str, Any]] = []
    for source in original_records:
        record_id = str(source["record_id"])
        evaluation = evaluations[record_id]
        if str(evaluation.get("coordinate_id")) != str(source["coordinate_id"]):
            raise FullMaeBiasCloseoutError(f"frozen_coordinate_mismatch:{record_id}")
        updated = dict(source)
        updated.pop("best_gate_mae_bpm", None)
        updated.update(
            {
                "previous_r_all_bias_s": float(evaluation["previous_r_all_bias_s"]),
                "selected_bias_s": float(evaluation["selected_bias_s"]),
                "time_bias_contract": "full_mae_evaluation_time_bias_v1",
                "common_window_count": int(evaluation["common_window_count"]),
                "fixed_5s_common_mae_bpm": float(evaluation["fixed_5s_common_mae_bpm"]),
                "final_common_mae_bpm": float(evaluation["selected_common_mae_bpm"]),
                "improvement_vs_5s_bpm": float(evaluation["improvement_vs_5s_bpm"]),
                "compatibility_metrics": dict(evaluation["compatibility_metrics"]),
                "gate_diagnostic": dict(evaluation["gate_diagnostic"]),
            }
        )
        updated_records.append(updated)
    header = dict(original)
    header.pop("dataset_card_sha256", None)
    return {
        **header,
        "schema_id": "screened_cross_subject_dataset_card_v2",
        "updated_at": str(updated_at),
        "time_bias_contract": "full_mae_evaluation_time_bias_v1",
        "records": updated_records,
    }


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git(worktree: Path, *args: str) -> str:
    return (
        subprocess.check_output(["git", *args], cwd=worktree, stderr=subprocess.DEVNULL)
        .decode("utf-8", errors="replace")
        .strip()
    )


def evaluator_identity(worktree: Path) -> dict[str, Any]:
    root = Path(worktree).resolve()
    evaluator_files = (
        "python/tools/multiperson_screening_contracts.py",
        "python/tools/multiperson_joint_screening.py",
        "python/tools/multiperson_full_mae_bias_closeout.py",
    )
    tracked_diff = _git(
        root,
        "diff",
        "--name-only",
        "--",
        *evaluator_files,
    )
    staged_diff = _git(
        root,
        "diff",
        "--cached",
        "--name-only",
        "--",
        *evaluator_files,
    )
    if tracked_diff or staged_diff:
        raise FullMaeBiasCloseoutError(
            "evaluator_source_not_committed:"
            + ",".join(
                value.replace("\n", ",")
                for value in (tracked_diff, staged_diff)
                if value
            )
        )
    paths = tuple(root / relative for relative in evaluator_files)
    return {
        "schema_id": "full_mae_bias_evaluator_identity_v1",
        "git_head": _git(root, "rev-parse", "HEAD"),
        "ppg_hr_tree": _git(root, "rev-parse", "HEAD:python/src/ppg_hr"),
        "files": {
            str(path.relative_to(root)).replace("\\", "/"): file_sha256(path) for path in paths
        },
    }


def _local_solver_inventory(output_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for complete_path in sorted(
        (Path(output_root) / "execution" / "cache" / "solver").glob("*/complete.json")
    ):
        complete = read_json(complete_path)
        report_path = complete_path.parent / "report-v2.json"
        if complete.get("status") != "complete" or not report_path.is_file():
            raise FullMaeBiasCloseoutError(f"local_solver_entry_incomplete:{complete_path}")
        actual_report_sha256 = file_sha256(report_path)
        if actual_report_sha256 != str(complete.get("report_sha256")):
            raise FullMaeBiasCloseoutError(
                f"local_solver_report_hash_mismatch:{report_path}"
            )
        rows.append(
            {
                "cache_key": str(complete["cache_key"]),
                "record_id": str(complete["record_id"]),
                "coordinate_id": str(complete["coordinate_id"]),
                "report_sha256": actual_report_sha256,
                "report_size": int(report_path.stat().st_size),
            }
        )
    if len(rows) != 4043:
        raise FullMaeBiasCloseoutError(f"local_solver_count_mismatch:{len(rows)}")
    return {
        "complete_count": len(rows),
        "index_sha256": canonical_sha256(rows),
    }


def _bo_inventory(output_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted((Path(output_root) / "non_lyx" / "records").glob("*/summary.json")):
        summary = read_json(summary_path)
        history = list(summary.get("history") or [])
        if len(history) != 120 or int(summary.get("logical_trial_count", -1)) != 120:
            raise FullMaeBiasCloseoutError(f"bo_history_count_mismatch:{summary.get('record_id')}")
        rows.append(
            {
                "record_id": str(summary["record_id"]),
                "coordinates": [str(row["coordinate_id"]) for row in history],
                "report_hashes": [str(row["report_sha256"]) for row in history],
            }
        )
    if len(rows) != 57:
        raise FullMaeBiasCloseoutError(f"bo_record_count_mismatch:{len(rows)}")
    return {
        "record_count": len(rows),
        "logical_trial_count": 120 * len(rows),
        "history_sha256": canonical_sha256(rows),
    }


def _frozen_record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "record_id",
            "scene",
            "subject",
            "coordinate_id",
            "data_sha256",
            "ref_sha256",
        )
    }


def _capture_frozen_inputs(
    *,
    output_root: Path,
    external_cache: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    snapshot_path = root / CLOSEOUT_DIR / "frozen_inputs.json"
    dataset_card = read_json(root / "dataset_card.json")
    records = list(dataset_card.get("records") or [])
    if len(records) != 48 or len({str(row["record_id"]) for row in records}) != 48:
        raise FullMaeBiasCloseoutError("frozen_dataset_record_count_mismatch")
    frozen_records = [
        {
            **_frozen_record_identity(row),
            "previous_r_all_bias_s": float(
                row.get("previous_r_all_bias_s", row["selected_bias_s"])
            ),
            "screening_best_gate_mae_bpm": float(
                row.get(
                    "screening_best_gate_mae_bpm",
                    row.get("best_gate_mae_bpm", row.get("final_common_mae_bpm")),
                )
            ),
        }
        for row in records
    ]
    live_identity = [_frozen_record_identity(row) for row in records]
    local_inventory = _local_solver_inventory(root)
    bo_inventory = _bo_inventory(root)
    if snapshot_path.is_file():
        snapshot = read_json(snapshot_path)
        if snapshot.get("snapshot_sha256") != canonical_sha256(
            {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
        ):
            raise FullMaeBiasCloseoutError("frozen_snapshot_hash_mismatch")
        if list(snapshot.get("record_identity") or []) != live_identity:
            raise FullMaeBiasCloseoutError("frozen_dataset_identity_changed")
        if snapshot.get("local_solver_inventory") != local_inventory:
            raise FullMaeBiasCloseoutError("local_solver_inventory_changed")
        if snapshot.get("bo_inventory") != bo_inventory:
            raise FullMaeBiasCloseoutError("bo_inventory_changed")
        if snapshot.get("external_completion_sha256") != external_cache.get("completion_sha256"):
            raise FullMaeBiasCloseoutError("external_cache_identity_changed")
        return snapshot
    snapshot = {
        "schema_id": "full_mae_bias_frozen_inputs_v1",
        "captured_at": now_iso(),
        "dataset_card_before_sha256": file_sha256(root / "dataset_card.json"),
        "panel_selection_before_sha256": file_sha256(root / "panel_selection.json"),
        "completion_before_sha256": file_sha256(root / "completion.json"),
        "record_identity": live_identity,
        "frozen_records": frozen_records,
        "local_solver_inventory": local_inventory,
        "bo_inventory": bo_inventory,
        "external_completion_sha256": external_cache["completion_sha256"],
    }
    snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
    write_json(snapshot_path, snapshot)
    return snapshot


def _read_payload_with_hash(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise FullMaeBiasCloseoutError(f"report_hash_mismatch:{path}")
    return json.loads(raw.decode("utf-8"))


def _record_summary_path(output_root: Path, record: Mapping[str, Any]) -> Path:
    group = "lyx" if str(record["subject"]) == "LYX" else "non_lyx"
    return Path(output_root) / group / "records" / str(record["record_id"]) / "summary.json"


def _resolve_selected_report(
    *,
    output_root: Path,
    record: Mapping[str, Any],
    external_calls: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    record_id = str(record["record_id"])
    coordinate_id = str(record["coordinate_id"])
    summary_path = _record_summary_path(output_root, record)
    if not summary_path.is_file():
        raise FullMaeBiasCloseoutError(f"record_summary_missing:{record_id}")
    summary = read_json(summary_path)
    selected = summary.get("best_gate_passing") or {}
    if str(selected.get("coordinate_id")) != coordinate_id:
        raise FullMaeBiasCloseoutError(
            f"selected_coordinate_mismatch:{record_id}:{selected.get('coordinate_id')}"
        )
    expected_report_sha = str(selected.get("report_sha256"))
    if str(record["subject"]) == "LYX":
        call = external_calls.get((record_id, coordinate_id))
        if call is None:
            raise FullMaeBiasCloseoutError(
                f"external_selected_call_missing:{record_id}:{coordinate_id}"
            )
        payload, report_sha = read_verified_external_report(call)
        report_path = Path(str(call["cache_entry"])) / "report-v2.json"
        cache_key = str(call["cache_key"])
    else:
        cache_key = str(selected.get("cache_key"))
        entry = Path(output_root) / "execution" / "cache" / "solver" / cache_key[:24]
        complete_path = entry / "complete.json"
        report_path = entry / "report-v2.json"
        if not complete_path.is_file() or not report_path.is_file():
            raise FullMaeBiasCloseoutError(f"local_selected_report_missing:{record_id}")
        complete = read_json(complete_path)
        if (
            complete.get("cache_key") != cache_key
            or complete.get("record_id") != record_id
            or complete.get("coordinate_id") != coordinate_id
        ):
            raise FullMaeBiasCloseoutError(f"local_selected_complete_mismatch:{record_id}")
        report_sha = str(complete["report_sha256"])
        payload = _read_payload_with_hash(report_path, report_sha)
        validate_joint_report_identity(payload)
    if report_sha != expected_report_sha:
        raise FullMaeBiasCloseoutError(f"selected_report_sha_mismatch:{record_id}")
    return (
        payload,
        summary,
        {
            "cache_key": cache_key,
            "report_path": str(report_path.resolve()),
            "report_sha256": report_sha,
        },
    )


def _evaluate_frozen_record(
    *,
    output_root: Path,
    record: Mapping[str, Any],
    frozen_record: Mapping[str, Any],
    external_calls: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    record_id = str(record["record_id"])
    for path_key, hash_key in (
        ("data_path", "data_sha256"),
        ("ref_path", "ref_sha256"),
    ):
        path = Path(str(record[path_key]))
        if not path.is_file() or file_sha256(path) != str(record[hash_key]):
            raise FullMaeBiasCloseoutError(f"live_input_hash_mismatch:{record_id}:{path_key}")
    payload, summary, report = _resolve_selected_report(
        output_root=output_root,
        record=record,
        external_calls=external_calls,
    )
    result = solver_result_from_payload(payload)
    ref_data = load_v2_reference(Path(str(record["ref_path"])))
    selection = select_full_mae_time_bias(result, ref_data=ref_data)
    if (
        float(selection["selected_common_mae_bpm"])
        > float(selection["fixed_5s_common_mae_bpm"]) + 1e-12
    ):
        raise FullMaeBiasCloseoutError(f"selected_mae_worse_than_5s:{record_id}")
    selected_bias = float(selection["selected_bias_s"])
    compatibility = evaluate_aligned_metrics(
        result,
        ref_data=ref_data,
        time_bias_s=selected_bias,
    )
    fixed_five_compatibility = evaluate_aligned_metrics(
        result,
        ref_data=ref_data,
        time_bias_s=BIAS_DEFAULT_S,
    )
    baseline = dict(summary.get("baseline") or {})
    baseline_path = Path(str(baseline.get("report_path")))
    baseline_sha = str(baseline.get("report_sha256"))
    if not baseline_path.is_file():
        raise FullMaeBiasCloseoutError(f"baseline_report_missing:{record_id}")
    baseline_payload = _read_payload_with_hash(baseline_path, baseline_sha)
    baseline_metrics = evaluate_aligned_metrics(
        solver_result_from_payload(baseline_payload),
        ref_data=ref_data,
        time_bias_s=selected_bias,
    )
    gate = evaluate_screening_gate(
        candidate=compatibility,
        baseline=baseline_metrics,
    )
    previous_bias = float(frozen_record["previous_r_all_bias_s"])
    curve_by_bias = {
        float(row["bias_s"]): float(row["common_mae_bpm"]) for row in selection["curve"]
    }
    return {
        "schema_id": "full_mae_bias_record_result_v1",
        "record_id": record_id,
        "scene": str(record["scene"]),
        "subject": str(record["subject"]),
        "coordinate_id": str(record["coordinate_id"]),
        "data_sha256": str(record["data_sha256"]),
        "ref_sha256": str(record["ref_sha256"]),
        "previous_r_all_bias_s": previous_bias,
        "previous_r_all_common_mae_bpm": curve_by_bias[previous_bias],
        "screening_best_gate_mae_bpm": float(frozen_record["screening_best_gate_mae_bpm"]),
        "bias_candidates_s": list(selection["bias_candidates_s"]),
        "selection_rule": selection["selection_rule"],
        "common_window_indices": list(selection["common_window_indices"]),
        "common_window_count": int(selection["common_window_count"]),
        "common_window_mask_sha256": canonical_sha256(selection["common_window_indices"]),
        "curve": list(selection["curve"]),
        "selected_bias_s": selected_bias,
        "selected_common_mae_bpm": float(selection["selected_common_mae_bpm"]),
        "fixed_5s_common_mae_bpm": float(selection["fixed_5s_common_mae_bpm"]),
        "improvement_vs_5s_bpm": float(selection["improvement_vs_5s_bpm"]),
        "compatibility_metrics": compatibility,
        "fixed_5s_compatibility_metrics": fixed_five_compatibility,
        "baseline_compatibility_metrics": baseline_metrics,
        "gate_diagnostic": gate,
        "previous_gate_qualified": bool(
            (summary.get("best_gate_passing") or {}).get("gate", {}).get("qualified")
        ),
        "solver_report": report,
        "baseline_report": {
            "report_path": str(baseline_path.resolve()),
            "report_sha256": baseline_sha,
        },
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    if not np.all(np.isfinite(array)):
        raise FullMaeBiasCloseoutError("nonfinite_summary_value")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _screening_history_row(row: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    if "best_gate_mae_bpm" in updated:
        updated["screening_r_all_best_gate_mae_bpm"] = updated.pop("best_gate_mae_bpm")
    updated["result_role"] = "screening_history_not_finalized"
    return updated


def _build_updated_panel_selection(
    original: Mapping[str, Any],
    *,
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scenes: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    for source_scene in list(original.get("scenes") or []):
        scene = dict(source_scene)
        selected: list[dict[str, Any]] = []
        for source in list(source_scene.get("selected") or []):
            record_id = str(source["record_id"])
            evaluation = evaluations.get(record_id)
            if evaluation is None:
                raise FullMaeBiasCloseoutError(f"panel_selected_evaluation_missing:{record_id}")
            if str(source.get("coordinate_id")) != str(evaluation["coordinate_id"]):
                raise FullMaeBiasCloseoutError(f"panel_selected_coordinate_changed:{record_id}")
            row = dict(source)
            row.pop("best_gate_mae_bpm", None)
            row.update(
                {
                    "screening_r_all_best_gate_mae_bpm": float(
                        evaluation["screening_best_gate_mae_bpm"]
                    ),
                    "previous_r_all_bias_s": float(evaluation["previous_r_all_bias_s"]),
                    "selected_bias_s": float(evaluation["selected_bias_s"]),
                    "final_common_mae_bpm": float(evaluation["selected_common_mae_bpm"]),
                    "fixed_5s_common_mae_bpm": float(evaluation["fixed_5s_common_mae_bpm"]),
                    "improvement_vs_5s_bpm": float(evaluation["improvement_vs_5s_bpm"]),
                    "gate_diagnostic": dict(evaluation["gate_diagnostic"]),
                    "time_bias_contract": CLOSEOUT_CONTRACT,
                }
            )
            selected.append(row)
            selected_ids.append(record_id)
        scene["selected"] = selected
        scene["backups"] = [
            _screening_history_row(row) for row in list(source_scene.get("backups") or [])
        ]
        scenes.append(scene)
    if set(selected_ids) != set(evaluations) or len(selected_ids) != len(evaluations):
        raise FullMaeBiasCloseoutError("panel_selected_record_set_changed")
    return {
        "schema_id": "multiperson_panel_selection_v2",
        "time_bias_contract": CLOSEOUT_CONTRACT,
        "candidate_rows": [
            _screening_history_row(row) for row in list(original.get("candidate_rows") or [])
        ],
        "scenes": scenes,
    }


def _manifest(rows: Sequence[Mapping[str, Any]], *, schema_id: str) -> dict[str, Any]:
    value = {
        "schema_id": schema_id,
        "time_bias_contract": CLOSEOUT_CONTRACT,
        "record_count": len(rows),
        "records": list(rows),
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def _build_result_summary(
    *,
    previous: Mapping[str, Any],
    dataset_card: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
    frozen_snapshot: Mapping[str, Any],
    evaluator: Mapping[str, Any],
) -> dict[str, Any]:
    selected_biases = [float(row["selected_bias_s"]) for row in evaluations]
    final_maes = [float(row["selected_common_mae_bpm"]) for row in evaluations]
    improvements = [float(row["improvement_vs_5s_bpm"]) for row in evaluations]
    compatibility_maes = [float(row["compatibility_metrics"]["mae_bpm"]) for row in evaluations]
    common_counts = [int(row["common_window_count"]) for row in evaluations]
    current_gate = [bool(row["gate_diagnostic"].get("qualified")) for row in evaluations]
    previous_gate = [bool(row["previous_gate_qualified"]) for row in evaluations]
    counts = Counter(selected_biases)
    improved = sum(int(value > 1e-12) for value in improvements)
    unchanged = sum(int(math.isclose(value, 0.0, abs_tol=1e-12)) for value in improvements)
    worsened = sum(int(value < -1e-12) for value in improvements)
    if worsened:
        raise FullMaeBiasCloseoutError("official_common_mae_has_regression")
    material_passport = dict(previous.get("material_passport") or {})
    previous_source_identity = material_passport.get("source_identity")
    if isinstance(previous_source_identity, Mapping):
        frozen_solver_identity = previous_source_identity.get(
            "frozen_solver_identity", previous_source_identity
        )
    else:
        frozen_solver_identity = previous_source_identity
    material_passport.update(
        {
            "verification_status": "VERIFIED",
            "source_identity": {
                "frozen_solver_identity": frozen_solver_identity,
                "post_solver_evaluator_identity": evaluator,
            },
        }
    )
    quality = dict(previous.get("quality") or {})
    quality.update(
        {
            "no_loso_executed": True,
            "zero_new_solver_reports": True,
            "local_solver_inventory": frozen_snapshot["local_solver_inventory"],
            "bo_inventory": frozen_snapshot["bo_inventory"],
            "selected_solver_report_set_sha256": canonical_sha256(
                sorted(str(row["solver_report"]["report_sha256"]) for row in evaluations)
            ),
        }
    )
    dataset = dict(previous.get("dataset") or {})
    dataset.update(
        {
            "selected_mae_contract": CLOSEOUT_CONTRACT,
            "selected_mae_bpm": _numeric_summary(final_maes),
            "scene_composition": list(dataset_card["scene_composition"]),
        }
    )
    return {
        "schema_id": "multiperson_joint_screening_result_summary_v2",
        "material_passport": material_passport,
        "time_bias": {
            "contract": CLOSEOUT_CONTRACT,
            "record_count": len(evaluations),
            "candidates_s": [4.0, 4.5, 5.0, 5.5, 6.0],
            "selected_bias_counts": {str(key): value for key, value in sorted(counts.items())},
            "selected_default_5s_count": int(counts.get(BIAS_DEFAULT_S, 0)),
            "common_window_count": _numeric_summary(common_counts),
            "official_common_window_mae_bpm": _numeric_summary(final_maes),
            "improvement_vs_5s_bpm": {
                **_numeric_summary(improvements),
                "improved_count": improved,
                "unchanged_count": unchanged,
                "worsened_count": worsened,
            },
            "maximum_overlap_compatibility_mae_bpm": _numeric_summary(compatibility_maes),
            "gate_diagnostic": {
                "qualified_count": sum(current_gate),
                "not_qualified_count": len(current_gate) - sum(current_gate),
                "previous_qualified_count": sum(previous_gate),
                "changed_count": sum(
                    int(before != after)
                    for before, after in zip(previous_gate, current_gate, strict=True)
                ),
                "qualified_to_not_qualified_count": sum(
                    int(before and not after)
                    for before, after in zip(previous_gate, current_gate, strict=True)
                ),
                "not_qualified_to_qualified_count": sum(
                    int(not before and after)
                    for before, after in zip(previous_gate, current_gate, strict=True)
                ),
                "selection_effect": "diagnostic_only_no_reselection",
            },
        },
        "bo120": dict(previous.get("bo120") or {}),
        "dataset": dataset,
        "quality": quality,
    }


def _external_calls_by_key(
    external: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    calls: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in list(external.get("calls") or []):
        key = (str(row["record_id"]), str(row["coordinate_id"]))
        if key in calls:
            raise FullMaeBiasCloseoutError(f"duplicate_external_call:{key[0]}:{key[1]}")
        calls[key] = row
    return calls


def _completion_payload(
    *,
    original_completion: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    frozen_snapshot: Mapping[str, Any],
    output_root: Path,
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = Path(output_root)
    return {
        "schema_id": "multiperson_joint_screening_completion_v2",
        "status": "complete_panel_full_mae_bias_finalized",
        "completed_at": now_iso(),
        "experiment_id": EXPERIMENT_ID,
        "time_bias_contract": CLOSEOUT_CONTRACT,
        "frozen_solver_source_identity": original_completion.get(
            "frozen_solver_source_identity",
            original_completion.get("source_identity"),
        ),
        "post_solver_evaluator_identity": evaluator,
        "frozen_inputs_snapshot_sha256": frozen_snapshot["snapshot_sha256"],
        "dataset_card_sha256": file_sha256(root / "dataset_card.json"),
        "panel_selection_sha256": file_sha256(root / "panel_selection.json"),
        "bias_manifest_sha256": file_sha256(root / "bias_manifest.json"),
        "bias_manifest_non_lyx_sha256": file_sha256(root / "bias_manifest_non_lyx.json"),
        "result_summary_sha256": file_sha256(root / "result_summary.json"),
        "selected_solver_report_set_sha256": canonical_sha256(
            sorted(str(row["solver_report"]["report_sha256"]) for row in evaluations)
        ),
        "record_count": len(evaluations),
        "scene_count": 8,
        "successful_scene_count": 8,
        "record_bo_count": int(frozen_snapshot["bo_inventory"]["record_count"]),
        "new_solver_report_count": 0,
        "new_bo_logical_trial_count": 0,
        "no_loso_executed": True,
    }


def run_closeout(
    *,
    worktree: Path,
    external_cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    external = validate_external_cache(external_cache_root)
    evaluator = evaluator_identity(worktree)
    original_card = read_json(root / "dataset_card.json")
    original_panel = read_json(root / "panel_selection.json")
    original_result = read_json(root / "result_summary.json")
    original_completion = read_json(root / "completion.json")
    snapshot = _capture_frozen_inputs(
        output_root=root,
        external_cache=external,
    )
    frozen_by_id = {str(row["record_id"]): row for row in list(snapshot["frozen_records"])}
    external_calls = _external_calls_by_key(external)
    evaluations = [
        _evaluate_frozen_record(
            output_root=root,
            record=record,
            frozen_record=frozen_by_id[str(record["record_id"])],
            external_calls=external_calls,
        )
        for record in list(original_card.get("records") or [])
    ]
    evaluation_by_id = {str(row["record_id"]): row for row in evaluations}
    if len(evaluation_by_id) != 48:
        raise FullMaeBiasCloseoutError("evaluation_record_count_mismatch")
    updated_at = now_iso()
    updated_card = build_updated_dataset_card(
        original_card,
        evaluations=evaluation_by_id,
        updated_at=updated_at,
    )
    updated_panel = _build_updated_panel_selection(
        original_panel,
        evaluations=evaluation_by_id,
    )
    combined_manifest = _manifest(
        evaluations,
        schema_id="multiperson_full_mae_bias_manifest_v1",
    )
    non_lyx_manifest = _manifest(
        [row for row in evaluations if str(row["subject"]) != "LYX"],
        schema_id="multiperson_non_lyx_full_mae_bias_manifest_v1",
    )
    result_summary = _build_result_summary(
        previous=original_result,
        dataset_card=updated_card,
        evaluations=evaluations,
        frozen_snapshot=snapshot,
        evaluator=evaluator,
    )
    write_json(root / "dataset_card.json", updated_card)
    write_json(root / "panel_selection.json", updated_panel)
    write_json(root / "bias_manifest.json", combined_manifest)
    write_json(root / "bias_manifest_non_lyx.json", non_lyx_manifest)
    write_json(root / "result_summary.json", result_summary)
    completion = _completion_payload(
        original_completion=original_completion,
        evaluator=evaluator,
        frozen_snapshot=snapshot,
        output_root=root,
        evaluations=evaluations,
    )
    write_json(root / "completion.json", completion)
    return validate_closeout(
        worktree=worktree,
        external_cache_root=external_cache_root,
        output_root=root,
    )


def _validate_file_hash(
    completion: Mapping[str, Any],
    *,
    field: str,
    path: Path,
) -> None:
    if str(completion.get(field)) != file_sha256(path):
        raise FullMaeBiasCloseoutError(f"completion_hash_mismatch:{field}")


def assert_recomputed_selection_matches(
    *,
    record_id: str,
    stored: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> None:
    keys = (
        "bias_candidates_s",
        "selection_rule",
        "common_window_indices",
        "common_window_count",
        "curve",
        "selected_bias_s",
        "selected_common_mae_bpm",
        "fixed_5s_common_mae_bpm",
        "improvement_vs_5s_bpm",
    )
    stored_selection = {key: stored.get(key) for key in keys}
    recomputed_selection = {key: recomputed.get(key) for key in keys}
    if canonical_sha256(stored_selection) != canonical_sha256(
        recomputed_selection
    ):
        raise FullMaeBiasCloseoutError(
            f"recomputed_selection_mismatch:{record_id}"
        )


def validate_closeout(
    *,
    worktree: Path,
    external_cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    external = validate_external_cache(external_cache_root)
    evaluator = evaluator_identity(worktree)
    snapshot = read_json(root / CLOSEOUT_DIR / "frozen_inputs.json")
    snapshot_without_hash = {
        key: value for key, value in snapshot.items() if key != "snapshot_sha256"
    }
    if snapshot.get("snapshot_sha256") != canonical_sha256(snapshot_without_hash):
        raise FullMaeBiasCloseoutError("frozen_snapshot_hash_mismatch")
    if snapshot.get("external_completion_sha256") != external.get("completion_sha256"):
        raise FullMaeBiasCloseoutError("external_cache_identity_changed")
    if snapshot.get("local_solver_inventory") != _local_solver_inventory(root):
        raise FullMaeBiasCloseoutError("local_solver_inventory_changed")
    if snapshot.get("bo_inventory") != _bo_inventory(root):
        raise FullMaeBiasCloseoutError("bo_inventory_changed")
    completion = read_json(root / "completion.json")
    if (
        completion.get("status") != "complete_panel_full_mae_bias_finalized"
        or completion.get("time_bias_contract") != CLOSEOUT_CONTRACT
        or completion.get("no_loso_executed") is not True
        or int(completion.get("new_solver_report_count", -1)) != 0
        or int(completion.get("new_bo_logical_trial_count", -1)) != 0
    ):
        raise FullMaeBiasCloseoutError("completion_contract_mismatch")
    if completion.get("post_solver_evaluator_identity") != evaluator:
        raise FullMaeBiasCloseoutError("evaluator_identity_mismatch")
    for field, name in (
        ("dataset_card_sha256", "dataset_card.json"),
        ("panel_selection_sha256", "panel_selection.json"),
        ("bias_manifest_sha256", "bias_manifest.json"),
        ("bias_manifest_non_lyx_sha256", "bias_manifest_non_lyx.json"),
        ("result_summary_sha256", "result_summary.json"),
    ):
        _validate_file_hash(completion, field=field, path=root / name)
    card = read_json(root / "dataset_card.json")
    card_records = list(card.get("records") or [])
    identity = [_frozen_record_identity(row) for row in card_records]
    if identity != list(snapshot.get("record_identity") or []):
        raise FullMaeBiasCloseoutError("final_dataset_identity_changed")
    if len(card_records) != 48 or card.get("time_bias_contract") != CLOSEOUT_CONTRACT:
        raise FullMaeBiasCloseoutError("final_dataset_contract_mismatch")
    for row in card_records:
        record_id = str(row["record_id"])
        for path_key, hash_key in (
            ("data_path", "data_sha256"),
            ("ref_path", "ref_sha256"),
        ):
            path = Path(str(row[path_key]))
            if not path.is_file() or file_sha256(path) != str(row[hash_key]):
                raise FullMaeBiasCloseoutError(
                    f"final_input_hash_mismatch:{record_id}:{path_key}"
                )
    scene_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in card_records:
        scene_groups.setdefault(str(row["scene"]), []).append(row)
    if len(scene_groups) != 8:
        raise FullMaeBiasCloseoutError("final_scene_count_mismatch")
    for scene, rows in scene_groups.items():
        subjects = [str(row["subject"]) for row in rows]
        if len(rows) != 6 or len(set(subjects)) != 6 or subjects.count("LYX") != 1:
            raise FullMaeBiasCloseoutError(f"final_scene_composition:{scene}")
    manifest = read_json(root / "bias_manifest.json")
    manifest_no_hash = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != canonical_sha256(manifest_no_hash):
        raise FullMaeBiasCloseoutError("bias_manifest_self_hash_mismatch")
    rows = list(manifest.get("records") or [])
    if len(rows) != 48:
        raise FullMaeBiasCloseoutError("bias_manifest_record_count_mismatch")
    rows_by_id = {str(row["record_id"]): row for row in rows}
    if len(rows_by_id) != 48:
        raise FullMaeBiasCloseoutError("bias_manifest_duplicate_record")
    non_lyx_manifest = read_json(root / "bias_manifest_non_lyx.json")
    non_lyx_without_hash = {
        key: value
        for key, value in non_lyx_manifest.items()
        if key != "manifest_sha256"
    }
    if (
        non_lyx_manifest.get("manifest_sha256")
        != canonical_sha256(non_lyx_without_hash)
        or int(non_lyx_manifest.get("record_count", -1)) != 40
        or any(
            str(row.get("subject")) == "LYX"
            for row in list(non_lyx_manifest.get("records") or [])
        )
    ):
        raise FullMaeBiasCloseoutError("non_lyx_bias_manifest_mismatch")
    frozen_by_id = {
        str(row["record_id"]): row
        for row in list(snapshot.get("frozen_records") or [])
    }
    external_calls = _external_calls_by_key(external)
    recomputed_rows = [
        _evaluate_frozen_record(
            output_root=root,
            record=card_row,
            frozen_record=frozen_by_id[str(card_row["record_id"])],
            external_calls=external_calls,
        )
        for card_row in card_records
    ]
    recomputed_by_id = {
        str(row["record_id"]): row for row in recomputed_rows
    }
    for record_id, recomputed in recomputed_by_id.items():
        stored = rows_by_id.get(record_id)
        if stored is None:
            raise FullMaeBiasCloseoutError(
                f"recomputed_record_missing:{record_id}"
            )
        assert_recomputed_selection_matches(
            record_id=record_id,
            stored=stored,
            recomputed=recomputed,
        )
        if canonical_sha256(stored) != canonical_sha256(recomputed):
            raise FullMaeBiasCloseoutError(
                f"recomputed_record_result_mismatch:{record_id}"
            )
    report_hashes: list[str] = []
    for card_row in card_records:
        record_id = str(card_row["record_id"])
        row = rows_by_id.get(record_id)
        if row is None or str(row["coordinate_id"]) != str(card_row["coordinate_id"]):
            raise FullMaeBiasCloseoutError(f"bias_manifest_identity_mismatch:{record_id}")
        curve = list(row.get("curve") or [])
        if [float(item["bias_s"]) for item in curve] != [
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
        ]:
            raise FullMaeBiasCloseoutError(f"bias_curve_mismatch:{record_id}")
        if any(int(item["window_count"]) != int(row["common_window_count"]) for item in curve):
            raise FullMaeBiasCloseoutError(f"bias_common_window_mismatch:{record_id}")
        if float(row["selected_common_mae_bpm"]) > float(row["fixed_5s_common_mae_bpm"]) + 1e-12:
            raise FullMaeBiasCloseoutError(f"bias_selected_worse_than_5s:{record_id}")
        if not math.isclose(
            float(card_row["final_common_mae_bpm"]),
            float(row["selected_common_mae_bpm"]),
            abs_tol=1e-12,
        ):
            raise FullMaeBiasCloseoutError(f"dataset_manifest_mae_mismatch:{record_id}")
        for report_key in ("solver_report", "baseline_report"):
            report = row[report_key]
            report_path = Path(str(report["report_path"]))
            if not report_path.is_file() or file_sha256(report_path) != str(
                report["report_sha256"]
            ):
                raise FullMaeBiasCloseoutError(
                    f"selected_input_report_changed:{record_id}:{report_key}"
                )
        report_hashes.append(str(row["solver_report"]["report_sha256"]))
    report_set_hash = canonical_sha256(sorted(report_hashes))
    if report_set_hash != completion.get("selected_solver_report_set_sha256"):
        raise FullMaeBiasCloseoutError("selected_solver_report_set_changed")
    panel = read_json(root / "panel_selection.json")
    panel_selected = [
        row
        for scene in list(panel.get("scenes") or [])
        for row in list(scene.get("selected") or [])
    ]
    if [str(row["record_id"]) for row in panel_selected] != [
        str(row["record_id"]) for row in card_records
    ]:
        raise FullMaeBiasCloseoutError("panel_dataset_record_order_changed")
    result = read_json(root / "result_summary.json")
    expected_card = build_updated_dataset_card(
        card,
        evaluations=recomputed_by_id,
        updated_at=str(card["updated_at"]),
    )
    if canonical_sha256(card) != canonical_sha256(expected_card):
        raise FullMaeBiasCloseoutError("recomputed_dataset_card_mismatch")
    expected_panel = _build_updated_panel_selection(
        panel,
        evaluations=recomputed_by_id,
    )
    if canonical_sha256(panel) != canonical_sha256(expected_panel):
        raise FullMaeBiasCloseoutError("recomputed_panel_selection_mismatch")
    expected_result = _build_result_summary(
        previous=result,
        dataset_card=card,
        evaluations=recomputed_rows,
        frozen_snapshot=snapshot,
        evaluator=evaluator,
    )
    if canonical_sha256(result) != canonical_sha256(expected_result):
        raise FullMaeBiasCloseoutError("recomputed_result_summary_mismatch")
    official = (result.get("time_bias") or {}).get("official_common_window_mae_bpm") or {}
    improvement = (result.get("time_bias") or {}).get("improvement_vs_5s_bpm") or {}
    if (
        int(official.get("count", -1)) != 48
        or int(improvement.get("count", -1)) != 48
        or int(improvement.get("worsened_count", -1)) != 0
    ):
        raise FullMaeBiasCloseoutError("result_summary_contract_mismatch")
    forbidden_names = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and any(
            token in path.name.lower()
            for token in (
                "loso",
                "leave_one",
                "leave-one",
                "fold_output",
                "heldout",
            )
        )
    ]
    if forbidden_names:
        raise FullMaeBiasCloseoutError("loso_artifact_present")
    receipt = {
        "schema_id": "multiperson_full_mae_bias_closeout_validation_v1",
        "status": "verified_complete",
        "validated_at": now_iso(),
        "time_bias_contract": CLOSEOUT_CONTRACT,
        "evaluator_identity": evaluator,
        "frozen_inputs_snapshot_sha256": snapshot["snapshot_sha256"],
        "completion_sha256": file_sha256(root / "completion.json"),
        "result_summary_sha256": file_sha256(root / "result_summary.json"),
        "dataset_card_sha256": file_sha256(root / "dataset_card.json"),
        "panel_selection_sha256": file_sha256(root / "panel_selection.json"),
        "selected_solver_report_set_sha256": report_set_hash,
        "verified_requirements": {
            "frozen_records_and_coordinates_48": True,
            "five_biases_on_one_common_window": True,
            "official_mae_not_worse_than_fixed_5s": True,
            "zero_new_solver_reports": True,
            "zero_new_bo_logical_trials": True,
            "eight_six_subject_scenes": True,
            "one_lyx_per_scene": True,
            "gate_changes_diagnostic_only": True,
            "no_loso_artifacts": True,
        },
    }
    write_json(root / "validation_receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--external-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.validate_only:
        result = validate_closeout(
            worktree=args.worktree,
            external_cache_root=args.external_cache_root,
            output_root=args.output_root,
        )
    else:
        result = run_closeout(
            worktree=args.worktree,
            external_cache_root=args.external_cache_root,
            output_root=args.output_root,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
