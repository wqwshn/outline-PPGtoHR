"""Governed runner for the 2026-08-02 LYX Lite/shared-parameter experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

PYTHON_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(PYTHON_SRC_ROOT) in sys.path:
    sys.path.remove(str(PYTHON_SRC_ROOT))
sys.path.insert(0, str(PYTHON_SRC_ROOT))

AUTHORIZATION_DEADLINE = "2026-08-02T08:30:00+08:00"
EXPERIMENT_ID = "2026-08-02-lyx-current-source-lite-shared-parameter"
PROPOSAL_VERSION = "lyx_current_source_lite_shared_proposal_v1"
AUTHORIZATION_VERSION = "lyx_current_source_lite_shared_authorization_v1"
PAUSE_COMPLETION_VERSION = "lyx_current_source_lite_shared_pause_completion_v1"
DUAL_CASCADE_IDENTITY = "dual_cascade_two_hf_v1"
PHYSICAL_25HZ_SPACE_NAME = "physical_25hz_extended_v1"
SOLVER_CACHE_CONTRACT_VERSION = "lyx_current_source_solver_cache_v1"
EVALUATION_CACHE_CONTRACT_VERSION = "lyx_current_source_evaluation_cache_v1"
GATE_CONTRACT_VERSION = "lyx_current_source_shared_record_gates_v1"
SELECTOR_CONTRACT_VERSION = "lyx_current_source_near_optimal_selector_v1"
CACHE_IMPORT_RECEIPT_VERSION = "lyx_current_source_cache_import_receipt_v1"
LITE_REFRESH_CONTRACT_VERSION = "lyx_current_source_lite_refresh_v1"
FOLD_FREEZE_VERSION = "lyx_current_source_fold_selection_freeze_v1"
FOLD_REVEAL_VERSION = "lyx_current_source_fold_reveal_v1"
EXECUTION_CHECKPOINT_VERSION = "lyx_current_source_execution_checkpoint_v1"
CLOCK_OVERRIDE_ENV = "LYX_CURRENT_SOURCE_ALLOW_CLOCK_OVERRIDE_FOR_TESTS"
LYX_PANEL_RELATIVE = Path("202607-multiperson") / "0708-LYX"
OLD_LITE_BATCH_NAME = "20260724_001816_lite_raw_bandpass_full_LMS+H"
OLD_LITE_REPORT_ALIASES = {
    "kaihe3_LYX_0613": ("multi_kaihe2_0613",),
}
LITE_PARAMETER_NAMES = (
    "fs_target",
    "max_order",
    "lms_mu_base",
    "smooth_win_len",
    "spec_penalty_width",
    "time_bias",
)

LITE_PANEL_RECORDS = (
    "bobi2_LYX_0519",
    "bobi2_LYX_0613",
    "bobi2_LYX_0617",
    "jianpan1_LYX_0708",
    "jianpan2_LYX_0708",
    "jianpan3_LYX_0708",
    "kaihe1_LYX_0613",
    "kaihe1_LYX_0617",
    "kaihe3_LYX_0613",
    "quanji1_LYX_0708",
    "quanji2_LYX_0708",
    "quanji4_LYX_0708",
    "run1_LYX_0708",
    "run2_LYX_0708",
    "run3_LYX_0708",
    "tiaosheng1_LYX_0613",
    "tiaosheng1_LYX_0617",
    "tiaosheng2_LYX_0613",
    "woli1_LYX_0708",
    "woli2_LYX_0708",
    "woli3_LYX_0708",
    "xiezi2_LYX_0708",
    "xiezi3_LYX_0708",
    "xiezi4_LYX_0708",
)

SHARED_SCENES = {
    "xiezi": ("xiezi2_LYX_0708", "xiezi4_LYX_0708", "xiezi3_LYX_0708"),
    "jianpan": (
        "jianpan1_LYX_0708",
        "jianpan2_LYX_0708",
        "jianpan3_LYX_0708",
    ),
    "run": ("run3_LYX_0708", "run1_LYX_0708", "run2_LYX_0708"),
    "kaihe": ("kaihe3_LYX_0613", "kaihe1_LYX_0617", "kaihe1_LYX_0613"),
}

PUBLISHED_TICKETS = (
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
)


class LYXExperimentError(RuntimeError):
    """The governed LYX experiment contract was violated."""


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        _json_ready(payload),
        ensure_ascii=True,
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


def _external_canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        _json_ready(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _verify_external_embedded_hash(
    payload: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    body = dict(payload)
    expected = str(body.pop(field, ""))
    if not expected:
        raise LYXExperimentError(f"{label}_missing_{field}")
    actual = _external_canonical_sha256(body)
    if actual != expected:
        raise LYXExperimentError(f"{label}_embedded_hash_mismatch")
    return expected


def _build_lite_refresh_binding(
    *,
    source_lite_dir: Path | None,
    audit_dir: Path | None,
) -> dict[str, Any] | None:
    if source_lite_dir is None and audit_dir is None:
        return None
    if source_lite_dir is None or audit_dir is None:
        raise LYXExperimentError("lite_refresh_requires_source_and_audit")
    source_root = Path(source_lite_dir).resolve()
    audit_root = Path(audit_dir).resolve()
    source_receipt = source_root / "lite_audit_receipt.json"
    audit_completion = audit_root / "completion.json"
    if not source_receipt.is_file():
        raise LYXExperimentError(f"lite_refresh_source_receipt_missing:{source_receipt}")
    if not audit_completion.is_file():
        raise LYXExperimentError(f"lite_refresh_audit_completion_missing:{audit_completion}")
    record_receipts = sorted(
        (source_root / "records").glob("*/lite_record_receipt.json")
    )
    if len(record_receipts) != len(LITE_PANEL_RECORDS):
        raise LYXExperimentError(
            f"lite_refresh_record_receipt_count:{len(record_receipts)}"
        )
    record_manifest = [
        {
            "record_id": path.parent.name,
            "path": str(path),
            "sha256": file_sha256(path),
        }
        for path in record_receipts
    ]
    return {
        "contract_version": LITE_REFRESH_CONTRACT_VERSION,
        "source_lite_dir": str(source_root),
        "source_lite_receipt_sha256": file_sha256(source_receipt),
        "source_record_receipts": record_manifest,
        "source_record_receipts_sha256": canonical_sha256(record_manifest),
        "audit_dir": str(audit_root),
        "audit_completion_file_sha256": file_sha256(audit_completion),
        "method": "exhaustive_cached_coordinate_replay_plus_fixed_anchor_refresh",
        "logical_trial_count": 3600,
        "new_bo_trial_count": 0,
    }


def _validate_proposal_runtime(proposal: Mapping[str, Any]) -> None:
    repository_root = Path(str(proposal["repository_root"])).resolve()
    if _git(repository_root, "rev-parse", "HEAD") != proposal.get("git_head"):
        raise LYXExperimentError("proposal_git_head_drift")
    if (
        _git(repository_root, "rev-parse", "HEAD:python/src/ppg_hr")
        != proposal.get("ppg_hr_source_tree")
    ):
        raise LYXExperimentError("proposal_ppg_hr_source_tree_drift")
    runner = proposal.get("runner") or {}
    runner_path = Path(str(runner.get("path") or ""))
    if not runner_path.is_file() or file_sha256(runner_path) != runner.get("sha256"):
        raise LYXExperimentError("proposal_runner_hash_drift")


def resolve_lite_panel_records(
    data_root: Path | str,
    *,
    record_ids: Sequence[str] = LITE_PANEL_RECORDS,
) -> list[dict[str, Any]]:
    """Resolve the frozen LYX panel, including files parked in ``nouse-data``."""

    panel_root = _resolve_panel_root(Path(data_root))
    records: list[dict[str, Any]] = []
    for record_id in record_ids:
        data_path = _first_existing(
            panel_root / f"{record_id}.csv",
            panel_root / "nouse-data" / f"{record_id}.csv",
        )
        ref_path = _first_existing(
            panel_root / f"{record_id}_HR_ref.csv",
            panel_root / f"{record_id}_ref.csv",
            panel_root / "nouse-data" / f"{record_id}_HR_ref.csv",
            panel_root / "nouse-data" / f"{record_id}_ref.csv",
        )
        if data_path is None:
            raise LYXExperimentError(f"record_data_missing:{record_id}")
        if ref_path is None:
            raise LYXExperimentError(f"record_ref_missing:{record_id}")
        records.append(
            {
                "record_id": str(record_id),
                "scene": _scene_from_record_id(str(record_id)),
                "data_path": str(data_path),
                "ref_path": str(ref_path),
                "data_sha256": file_sha256(data_path),
                "ref_sha256": file_sha256(ref_path),
            }
        )
    return records


def physical_25hz_extended_candidates() -> list[dict[str, Any]]:
    """Return the frozen 25 Hz, 9 x 5 x 4 physical grid."""

    memories_ms = (40, 80, 120, 160, 200, 320, 480, 640, 800)
    mus = (0.006, 0.008, 0.010, 0.012, 0.016)
    half_widths_bpm = (3, 6, 12, 18)
    parameter_names = (
        "fs_target",
        "memory_ms",
        "mu_base",
        "exclusion_half_width_bpm",
    )
    candidates: list[dict[str, Any]] = []
    for coordinate, (memory_ms, mu, half_width) in enumerate(
        itertools.product(memories_ms, mus, half_widths_bpm)
    ):
        requested = {
            "fs_target": 25,
            "memory_ms": int(memory_ms),
            "mu_base": float(mu),
            "exclusion_half_width_bpm": int(half_width),
        }
        actual = {
            "fs_target": 25,
            "max_order": int(round(25 * int(memory_ms) / 1000.0)),
            "lms_mu_base": float(mu),
            "spec_penalty_width": float(half_width) / 60.0,
        }
        fixed = {
            "analysis_scope": "full",
            "smooth_win_len": 5,
            "time_bias": 5.0,
            "lms_mu_min": 1e-6,
        }
        actual.update(fixed)
        payload = {
            "space_name": PHYSICAL_25HZ_SPACE_NAME,
            "requested_params": requested,
            "actual_params": actual,
            "fixed_params": fixed,
        }
        candidates.append(
            {
                "space_name": PHYSICAL_25HZ_SPACE_NAME,
                "candidate_id": (
                    f"{PHYSICAL_25HZ_SPACE_NAME}:"
                    f"{canonical_sha256(payload)}"
                ),
                "coordinate": [
                    memories_ms.index(memory_ms),
                    mus.index(mu),
                    half_widths_bpm.index(half_width),
                ],
                "coordinate_index": coordinate,
                "parameter_names": list(parameter_names),
                "requested_params": requested,
                "actual_params": actual,
                "fixed_params": fixed,
            }
        )
    return candidates


def validate_dual_cascade_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    order = tuple(str(item) for item in receipt.get("reference_groups_order", ()))
    stage_limit = receipt.get("adaptive_reference_stage_limit")
    stage_count = receipt.get("actual_adaptive_hf_stage_count")
    if order != ("HF",) or stage_limit is not None or int(stage_count or -1) != 2:
        raise LYXExperimentError("dual_cascade_identity_mismatch")
    return {
        "identity": DUAL_CASCADE_IDENTITY,
        "reference_groups_order": ["HF"],
        "adaptive_reference_stage_limit": None,
        "actual_adaptive_hf_stage_count": 2,
    }


def build_solver_cache_key(
    *,
    algorithm_source_sha256: str,
    data_sha256: str,
    candidate: Mapping[str, Any],
    mechanism_identity: Mapping[str, Any],
    logical_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_sha256("algorithm_source_sha256", algorithm_source_sha256)
    _require_sha256("data_sha256", data_sha256)
    payload = {
        "contract_version": SOLVER_CACHE_CONTRACT_VERSION,
        "algorithm_source_sha256": algorithm_source_sha256,
        "data_sha256": data_sha256,
        "actual_params": candidate["actual_params"],
        "fixed_params": candidate["fixed_params"],
        "mechanism_identity": dict(mechanism_identity),
        "logical_context_ignored": True,
        "logical_candidate_identity_ignored": True,
    }
    key = canonical_sha256(payload)
    return {
        "key": key,
        "payload": payload,
        "logical_context": logical_context,
        "candidate_identity": {
            "space_name": candidate.get("space_name"),
            "candidate_id": candidate.get("candidate_id"),
            "requested_params": candidate.get("requested_params"),
        },
    }


def build_evaluation_cache_key(
    *,
    solver_result_sha256: str,
    reference_sha256: str,
    metric_contract_sha256: str,
    gate_contract_sha256: str,
) -> dict[str, Any]:
    for name, value in (
        ("solver_result_sha256", solver_result_sha256),
        ("reference_sha256", reference_sha256),
        ("metric_contract_sha256", metric_contract_sha256),
        ("gate_contract_sha256", gate_contract_sha256),
    ):
        _require_sha256(name, value)
    payload = {
        "contract_version": EVALUATION_CACHE_CONTRACT_VERSION,
        "solver_result_sha256": solver_result_sha256,
        "reference_sha256": reference_sha256,
        "metric_contract_sha256": metric_contract_sha256,
        "gate_contract_sha256": gate_contract_sha256,
    }
    return {"key": canonical_sha256(payload), "payload": payload}


def _metric_contract_sha256() -> str:
    return canonical_sha256(
        {
            "contract_version": EVALUATION_CACHE_CONTRACT_VERSION,
            "metric_source": "ppg_hr.v2.report-v2.json",
            "metric_fields": [
                "mae_bpm",
                "motion_mae_bpm",
                "e10",
                "e20",
                "l10",
                "l20",
                "post_motion_60s_mae_bpm",
                "post_motion_60s_e10_count",
                "post_motion_60s_e20_count",
                "right_censored_recovery_count",
                "true_rise_underestimate_bpm",
                "true_rise_applicable",
                "spectral_gate_contract_v2",
                "stability_pass",
            ],
        }
    )


def _gate_contract_sha256() -> str:
    return canonical_sha256(
        {
            "shared_record_gates": GATE_CONTRACT_VERSION,
            "lite_non_regression_audit": "lyx_current_source_lite_audit_v2",
            "dual_cascade_identity": DUAL_CASCADE_IDENTITY,
        }
    )


def _evaluation_cache_root_from_solver_cache(cache_root: Path) -> Path:
    return Path(cache_root).parent / "evaluation"


def _evaluation_cache_entry_path(cache_root: Path, key: str) -> Path:
    return Path(cache_root) / key[:24]


def _cached_metrics_from_report(
    *,
    report_path: Path,
    record: Mapping[str, Any],
    scene: str,
    evaluation_cache_root: Path,
) -> dict[str, Any]:
    report = Path(report_path)
    if not report.is_file():
        raise LYXExperimentError(f"evaluation_report_missing:{report}")
    evaluation_cache_root.mkdir(parents=True, exist_ok=True)
    key_info = build_evaluation_cache_key(
        solver_result_sha256=file_sha256(report),
        reference_sha256=str(record["ref_sha256"]),
        metric_contract_sha256=_metric_contract_sha256(),
        gate_contract_sha256=_gate_contract_sha256(),
    )
    key = str(key_info["key"])
    entry = _evaluation_cache_entry_path(evaluation_cache_root, key)
    metrics_path = entry / "metrics.json"
    complete_path = entry / "complete.json"
    if metrics_path.is_file() and complete_path.is_file():
        complete = _read_json(complete_path)
        if complete.get("cache_key") != key:
            raise LYXExperimentError(
                f"evaluation_cache_key_prefix_collision:{entry}:{key}"
            )
        metrics = _read_json(metrics_path)
        metrics.update(
            {
                "evaluation_cache_key": key,
                "evaluation_cache_hit": True,
                "evaluation_cache_path": str(metrics_path),
            }
        )
        return metrics

    payload = _read_json(report)
    metrics = _metrics_from_report_payload(payload, scene=scene)
    entry.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(metrics_path, metrics)
    _atomic_write_json(
        complete_path,
        {
            "cache_key": key,
            "status": "complete",
            "payload": key_info["payload"],
            "record_id": record.get("record_id"),
            "scene": scene,
            "report_path": str(report),
        },
    )
    metrics = dict(metrics)
    metrics.update(
        {
            "evaluation_cache_key": key,
            "evaluation_cache_hit": False,
            "evaluation_cache_path": str(metrics_path),
        }
    )
    return metrics


def evaluate_record_gates(
    *,
    candidate: Mapping[str, Any],
    independent: Mapping[str, Any],
    current: Mapping[str, Any],
    scene: str,
) -> dict[str, Any]:
    try:
        validate_dual_cascade_receipt(candidate)
        candidate_mae = _finite_metric(candidate, "mae_bpm")
        independent_mae = _finite_metric(independent, "mae_bpm")
        current_mae = _finite_metric(current, "mae_bpm")
        candidate_l10 = int(_finite_metric(candidate, "l10"))
        candidate_l20 = int(_finite_metric(candidate, "l20"))
        independent_l10 = int(_finite_metric(independent, "l10"))
        independent_l20 = int(_finite_metric(independent, "l20"))
        current_l10 = int(_finite_metric(current, "l10"))
        candidate_right = int(
            _finite_metric(candidate, "right_censored_recovery_count")
        )
        current_right = int(
            _finite_metric(current, "right_censored_recovery_count")
        )
    except (KeyError, TypeError, ValueError, LYXExperimentError):
        return _gate_result(False, ["nonfinite_or_missing_metric"])

    failed: list[str] = []
    if (
        candidate.get("spectral_gate_contract_v2") is not True
        or candidate.get("stability_pass") is not True
    ):
        failed.append("spectral_gate_contract_v2")
    if candidate_l10 > max(10, independent_l10 + 2):
        failed.append("independent_l10")
    if candidate_l20 > max(2, independent_l20):
        failed.append("independent_l20")
    if candidate_mae - independent_mae > 2.0:
        failed.append("independent_mae_delta")
    if candidate_right > current_right:
        failed.append("right_censored_recovery")
    if _true_rise_required(scene):
        try:
            candidate_rise = _finite_metric(
                candidate,
                "true_rise_underestimate_bpm",
            )
            current_rise = _finite_metric(current, "true_rise_underestimate_bpm")
        except (KeyError, TypeError, ValueError):
            failed.append("true_rise_underestimate")
        else:
            if candidate.get("true_rise_applicable") is not True:
                failed.append("true_rise_underestimate")
            elif current.get("true_rise_applicable") is not True:
                failed.append("true_rise_underestimate")
            elif candidate_rise - current_rise > 2.0:
                failed.append("true_rise_underestimate")
    elif not _explicit_not_applicable(candidate) or not _explicit_not_applicable(
        current
    ):
        failed.append("true_rise_not_applicable_contract")
    if current_l10 <= 10 and candidate_l10 >= 20:
        failed.append("current_l10_catastrophic_regression")
    if candidate_mae - current_mae > 2.0:
        failed.append("current_mae_delta")
    return _gate_result(not failed, failed)


def select_near_optimal_candidate(
    *,
    candidates: Sequence[Mapping[str, Any]],
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    train_record_ids: Sequence[str],
) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        record_rows = [rows.get((record_id, candidate_id)) for record_id in train_record_ids]
        if any(row is None or row.get("qualified") is not True for row in record_rows):
            continue
        maes = [_finite_metric(row, "mae_bpm") for row in record_rows if row is not None]
        try:
            mean_independent_delta = _required_mean_independent_delta(record_rows)
        except (KeyError, TypeError, ValueError):
            continue
        if mean_independent_delta > 1.0:
            continue
        eligible.append(
            {
                "candidate_id": candidate_id,
                "coordinate": list(candidate["coordinate"]),
                "worst_train_mae": max(maes),
                "mean_train_mae": sum(maes) / len(maes),
            }
        )
    if not eligible:
        raise LYXExperimentError("no_safe_training_candidate")

    best_worst = min(item["worst_train_mae"] for item in eligible)
    near = [
        item for item in eligible if item["worst_train_mae"] <= best_worst + 0.5
    ]
    row_index = {
        item["candidate_id"]: item
        for item in eligible
    }
    candidates_by_id = {str(item["candidate_id"]): item for item in candidates}
    for item in near:
        support = _support_neighbors(
            center=item,
            row_index=row_index,
            candidates_by_id=candidates_by_id,
        )
        cliffs = _cliff_count(
            center=item,
            row_index=row_index,
            rows=rows,
            train_record_ids=train_record_ids,
            candidates_by_id=candidates_by_id,
        )
        possible_neighbors = _possible_neighbor_count(
            item["candidate_id"],
            candidates_by_id=candidates_by_id,
        )
        item["support_neighbor_count"] = support
        item["support_neighbor_fraction"] = (
            0.0 if possible_neighbors == 0 else support / possible_neighbors
        )
        item["cliff_count"] = cliffs
    ranking = sorted(
        near,
        key=lambda item: (
            -float(item["support_neighbor_fraction"]),
            -int(item["support_neighbor_count"]),
            int(item["cliff_count"]),
            float(item["mean_train_mae"]),
            tuple(int(value) for value in item["coordinate"]),
        ),
    )
    return {
        "selector_contract_version": SELECTOR_CONTRACT_VERSION,
        "selected_candidate_id": ranking[0]["candidate_id"],
        "best_worst_train_mae": best_worst,
        "near_optimal_candidate_count": len(near),
        "eligible_candidate_count": len(eligible),
        "ranking": ranking,
    }


def analyse_common_platforms(
    *,
    candidates: Sequence[Mapping[str, Any]],
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    record_ids: Sequence[str],
) -> dict[str, Any]:
    """Derive safe/flat 2x2 platforms from already evaluated common rows."""

    candidates_by_id = {str(item["candidate_id"]): item for item in candidates}
    common_ids = [
        candidate_id
        for candidate_id in candidates_by_id
        if all(
            rows.get((record_id, candidate_id), {}).get("qualified") is True
            for record_id in record_ids
        )
    ]
    components = _connected_components(common_ids, candidates_by_id)
    common_set = set(common_ids)
    boundary_ids = [
        candidate_id
        for candidate_id in common_ids
        if _support_neighbor_ids(
            candidate_id,
            candidates_by_id=candidates_by_id,
        )
        - common_set
    ]
    platforms: list[dict[str, Any]] = []
    for quad in _candidate_2x2_quads(candidates):
        if not set(quad).issubset(common_set):
            continue
        per_record_ranges = {}
        for record_id in record_ids:
            values = [
                _finite_metric(rows[(record_id, candidate_id)], "mae_bpm")
                for candidate_id in quad
            ]
            per_record_ranges[str(record_id)] = max(values) - min(values)
        platforms.append(
            {
                "candidate_ids": list(quad),
                "per_record_mae_range_bpm": per_record_ranges,
                "strong_flat": all(value <= 1.0 for value in per_record_ranges.values()),
                "sensitivity_2bpm": all(
                    value <= 2.0 for value in per_record_ranges.values()
                ),
            }
        )
    return {
        "common_safe_candidate_count": len(common_ids),
        "common_safe_candidate_ids": common_ids,
        "common_safe_coordinates": [
            list(candidates_by_id[candidate_id]["coordinate"])
            for candidate_id in common_ids
        ],
        "common_safe_parameter_coverage": {
            name: sorted(
                {
                    candidates_by_id[candidate_id]
                    .get("requested_params", {})
                    .get(name, candidates_by_id[candidate_id]["coordinate"][axis])
                    for candidate_id in common_ids
                }
            )
            for axis, name in enumerate(
                ("memory_ms", "mu_base", "exclusion_half_width_bpm")
            )
        },
        "boundary_candidate_ids": boundary_ids,
        "largest_component_size": max((len(item) for item in components), default=0),
        "common_components": components,
        "safe_platform_count": len(platforms),
        "strong_flat_platform_count": sum(
            bool(item["strong_flat"]) for item in platforms
        ),
        "sensitivity_2bpm_platform_count": sum(
            bool(item["sensitivity_2bpm"]) for item in platforms
        ),
        "platforms": platforms,
    }


def evaluate_scene_reference_line(
    *,
    fold_results: Sequence[Mapping[str, Any]],
    scene: str,
) -> dict[str, Any]:
    failures: list[str] = []
    evaluated = [row for row in fold_results if row.get("status") == "evaluated"]
    if len(fold_results) != 3 or len(evaluated) != 3:
        failures.append("incomplete_three_fold_evidence")
    if any(not bool(row.get("heldout_pass")) for row in fold_results):
        failures.append("heldout_hard_gate_failure")
    heldout_maes = [float(row["heldout_mae_bpm"]) for row in evaluated]
    train_deltas = [
        float(row["heldout_mae_bpm"]) - float(row["selected_worst_train_mae"])
        for row in evaluated
    ]
    current_deltas = [
        float(row["heldout_current_delta_mae_bpm"]) for row in evaluated
    ]
    if any(value >= 10.0 for value in heldout_maes):
        failures.append("heldout_new_catastrophe_ge_10_bpm")
    if heldout_maes and sum(heldout_maes) / len(heldout_maes) > 5.0:
        failures.append("scene_mean_heldout_mae_gt_5_bpm")
    if train_deltas:
        ordered = sorted(train_deltas)
        median = ordered[len(ordered) // 2]
        if median > 2.0:
            failures.append("median_heldout_delta_gt_2_bpm")
        if max(train_deltas) > 5.0:
            failures.append("fold_heldout_delta_gt_5_bpm")
    if scene == "run" and current_deltas:
        if sum(current_deltas) / len(current_deltas) > 1.0:
            failures.append("run_mean_current_delta_gt_1_bpm")
        if max(current_deltas) > 2.0:
            failures.append("run_record_current_delta_gt_2_bpm")
    return {
        "passed": not failures,
        "failures": failures,
        "evaluated_fold_count": len(evaluated),
        "heldout_mae_bpm": heldout_maes,
        "heldout_minus_worst_train_bpm": train_deltas,
        "heldout_minus_current_bpm": current_deltas,
    }


def classify_scene_fold_results(
    *,
    fold_results: Sequence[Mapping[str, Any]],
    common_components: Sequence[Sequence[str]],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not fold_results:
        return {"functional_shared": False, "level": "not_run"}
    if not all(bool(item.get("heldout_pass")) for item in fold_results):
        return {"functional_shared": False, "level": "fold_failure"}
    coordinates = [
        tuple(int(value) for value in item["selected_coordinate"])
        for item in fold_results
    ]
    if len(set(coordinates)) == 1:
        return {"functional_shared": True, "level": "stable_shared_parameter"}

    per_axis_span = [
        max(coord[axis] for coord in coordinates)
        - min(coord[axis] for coord in coordinates)
        for axis in range(len(coordinates[0]))
    ]
    selected_ids = [
        str(item.get("selected_candidate_id", ""))
        or _candidate_id_for_coordinate(candidates_by_id, coord)
        for item, coord in zip(fold_results, coordinates, strict=True)
    ]
    same_component = any(
        set(selected_ids).issubset(set(component)) for component in common_components
    )
    if max(per_axis_span, default=0) <= 1 and same_component:
        return {
            "functional_shared": True,
            "level": "stable_shared_neighborhood",
            "per_axis_span": per_axis_span,
        }
    return {
        "functional_shared": True,
        "level": "selection_unstable",
        "per_axis_span": per_axis_span,
    }


def validate_authorization_window(
    proposal: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    now: str | None = None,
) -> dict[str, Any]:
    if receipt.get("approved") is not True:
        raise LYXExperimentError("authorization_not_approved")
    if receipt.get("proposal_sha256") != proposal.get("proposal_sha256"):
        raise LYXExperimentError("authorization_proposal_mismatch")
    expires_at = str(receipt.get("expires_at") or proposal.get("authorization_deadline"))
    now_dt = _parse_datetime(now or datetime.now().astimezone().isoformat())
    expiry_dt = _parse_datetime(expires_at)
    if now_dt >= expiry_dt:
        raise LYXExperimentError("authorization_expired")
    validated = dict(receipt)
    validated["validated_at"] = now_dt.isoformat()
    return validated


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build-proposal":
            proposal = build_proposal(
                output_dir=args.output_dir,
                spec_path=args.spec_path,
                data_root=args.data_root,
                repository_root=args.repository_root,
                source_lite_dir=args.source_lite_dir,
                lite_refresh_audit_dir=args.lite_refresh_audit_dir,
            )
            print(json.dumps(proposal, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "authorize-window":
            receipt = authorize_window(
                proposal_dir=args.proposal_dir,
                approved_at=args.approved_at,
                expires_at=args.expires_at,
                allow_clock_override=args.allow_clock_override_for_tests,
            )
            print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "run":
            return run_from_proposal(
                args.proposal_dir,
                now=args.now,
                allow_clock_override=args.allow_clock_override_for_tests,
                phase=args.phase,
                max_lite_records=args.max_lite_records,
                max_shared_scenes=args.max_shared_scenes,
            )
        if args.command == "report":
            report = write_report(args.proposal_dir)
            print(str(report))
            return 0
    except LYXExperimentError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled_command:{args.command}")


def build_proposal(
    *,
    output_dir: Path,
    spec_path: Path,
    data_root: Path,
    repository_root: Path,
    source_lite_dir: Path | None = None,
    lite_refresh_audit_dir: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    spec = Path(spec_path).resolve()
    data = Path(data_root).resolve()
    repo = Path(repository_root).resolve()
    if not spec.is_file():
        raise LYXExperimentError(f"spec_missing:{spec}")
    if not data.exists():
        raise LYXExperimentError(f"data_root_missing:{data}")
    resolved_records: list[dict[str, Any]] = []
    old_lite_batch_dir = ""
    try:
        panel_root = _resolve_panel_root(data)
        resolved_records = resolve_lite_panel_records(data)
        old_lite_batch_dir = str(
            panel_root / "v2_batch_outputs" / OLD_LITE_BATCH_NAME
        )
    except LYXExperimentError:
        resolved_records = []
        old_lite_batch_dir = ""
    lite_refresh = _build_lite_refresh_binding(
        source_lite_dir=source_lite_dir,
        audit_dir=lite_refresh_audit_dir,
    )

    proposal = {
        "proposal_version": PROPOSAL_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "ready_for_authorization",
        "evidence_class": "development_reuse_pilot",
        "algorithm_level_holdout": False,
        "authorization_deadline": AUTHORIZATION_DEADLINE,
        "repository_root": str(repo),
        "git_head": _git(repo, "rev-parse", "HEAD"),
        "ppg_hr_source_tree": _git(repo, "rev-parse", "HEAD:python/src/ppg_hr"),
        "spec": {
            "path": str(spec),
            "sha256": file_sha256(spec),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "data_panel": {
            "root": str(data),
            "panel_root": str(_resolve_panel_root(data)) if resolved_records else "",
            "lite_record_count": len(LITE_PANEL_RECORDS),
            "lite_records": list(LITE_PANEL_RECORDS),
            "resolved_lite_records": resolved_records,
            "old_lite_batch_dir": old_lite_batch_dir,
            "shared_scenes": {
                scene: list(records) for scene, records in SHARED_SCENES.items()
            },
        },
        "algorithm_identity": {
            "algorithm_preset": "lite",
            "adaptive_filter": "lms",
            "dual_cascade_identity": DUAL_CASCADE_IDENTITY,
            "reference_groups_order": ["HF"],
            "adaptive_reference_stage_limit": None,
        },
        "lite_search": {
            "repeat_total": 3,
            "trial_total": 50,
            "seeds": [42, 43, 44],
            "num_seed_points": 10,
            "logical_trial_count": 3600,
            "space": [
                "fs_target",
                "max_order",
                "lms_mu_base",
                "smooth_win_len",
                "spec_penalty_width",
                "time_bias",
            ],
            "repeated_parameters_allowed": True,
        },
        "lite_refresh": lite_refresh,
        "physical_space": {
            "space_name": PHYSICAL_25HZ_SPACE_NAME,
            "candidate_count": 180,
            "sha256": canonical_sha256(physical_25hz_extended_candidates()),
        },
        "contracts": {
            "solver_cache": SOLVER_CACHE_CONTRACT_VERSION,
            "evaluation_cache": EVALUATION_CACHE_CONTRACT_VERSION,
            "record_gates": GATE_CONTRACT_VERSION,
            "selector": SELECTOR_CONTRACT_VERSION,
        },
        "budget": {
            "lite_logical_trials": 3600,
            "fixed_replays": 24,
            "physical_requests": 2160,
            "max_solver_requests": 5784,
            "wall_clock_hours": 12,
            "execution_workers": 8,
        },
        "published_tickets": list(PUBLISHED_TICKETS),
        "execution_baseline": {
            "captured_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "git_status_short": _git_status_short(repo),
            "preexisting_untracked_preserve_rules": [
                "do_not_touch_existing_dot_t_directories",
                "do_not_touch_existing_report_preview_or_grayscale_assets",
                "do_not_stage_unrelated_untracked_assets",
            ],
        },
        "formal_solver_run_count": 0,
    }
    proposal["proposal_sha256"] = canonical_sha256(proposal)
    _atomic_write_json(output_root / "proposal.json", proposal)
    _atomic_write_json(
        output_root / "physical_25hz_extended_candidates.json",
        {
            "space_name": PHYSICAL_25HZ_SPACE_NAME,
            "candidate_count": 180,
            "candidates": physical_25hz_extended_candidates(),
        },
    )
    return proposal


def authorize_window(
    *,
    proposal_dir: Path,
    approved_at: str | None = None,
    expires_at: str | None = None,
    allow_clock_override: bool = False,
) -> dict[str, Any]:
    proposal_root = Path(proposal_dir).resolve()
    proposal = _read_json(proposal_root / "proposal.json")
    if approved_at is None:
        approved_at = datetime.now().astimezone().isoformat(timespec="seconds")
    elif not _clock_override_allowed(allow_clock_override):
        raise LYXExperimentError("clock_override_requires_test_opt_in")
    approved_dt = _parse_datetime(approved_at)
    expiry_dt = _parse_datetime(expires_at or AUTHORIZATION_DEADLINE)
    if approved_dt >= expiry_dt:
        raise LYXExperimentError("authorization_approved_after_deadline")
    receipt = {
        "authorization_version": AUTHORIZATION_VERSION,
        "approved": True,
        "proposal_sha256": proposal["proposal_sha256"],
        "approved_at": approved_dt.isoformat(),
        "expires_at": expiry_dt.isoformat(),
        "legacy_deadline": AUTHORIZATION_DEADLINE,
        "scope": (
            "prepare and execute the governed LYX Lite/shared-parameter "
            "experiment until the bound expiration"
        ),
        "formal_solver_run_count_at_authorization": 0,
    }
    receipt["authorization_sha256"] = canonical_sha256(receipt)
    _atomic_write_json(proposal_root / "authorization.json", receipt)
    return receipt


def run_from_proposal(
    proposal_dir: Path,
    *,
    now: str | None = None,
    allow_clock_override: bool = False,
    phase: str = "all",
    max_lite_records: int | None = None,
    max_shared_scenes: int | None = None,
) -> int:
    if now is not None and not _clock_override_allowed(allow_clock_override):
        raise LYXExperimentError("clock_override_requires_test_opt_in")
    if (
        max_lite_records is not None or max_shared_scenes is not None
    ) and not _clock_override_allowed(allow_clock_override):
        raise LYXExperimentError("reduced_denominator_requires_test_opt_in")
    proposal_root = Path(proposal_dir).resolve()
    proposal = _read_json(proposal_root / "proposal.json")
    _validate_proposal_runtime(proposal)
    authorization_path = proposal_root / "authorization.json"
    if not authorization_path.is_file():
        if _deadline_expired(proposal, now=now):
            completion = _paused_completion(
                proposal=proposal,
                authorization_sha256=None,
                paused_at=now
                or datetime.now().astimezone().isoformat(timespec="seconds"),
            )
            _atomic_write_json(proposal_root / "completion.json", completion)
            return 0
        raise LYXExperimentError("authorization_missing")
    receipt = _read_json(authorization_path)
    try:
        validate_authorization_window(proposal, receipt, now=now)
    except LYXExperimentError as exc:
        if str(exc) == "authorization_proposal_mismatch" and _deadline_expired(
            proposal,
            now=now,
        ):
            completion = _paused_completion(
                proposal=proposal,
                authorization_sha256=None,
                paused_at=now
                or datetime.now().astimezone().isoformat(timespec="seconds"),
            )
            _atomic_write_json(proposal_root / "completion.json", completion)
            return 0
        if str(exc) != "authorization_expired":
            raise
        completion = _paused_completion(
            proposal=proposal,
            authorization_sha256=receipt.get("authorization_sha256"),
            paused_at=now
            or datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        _atomic_write_json(proposal_root / "completion.json", completion)
        return 0
    execute_experiment(
        proposal_root=proposal_root,
        proposal=proposal,
        authorization=receipt,
        phase=phase,
        max_lite_records=max_lite_records,
        max_shared_scenes=max_shared_scenes,
    )
    return 0


def _lite_stage_mode(proposal: Mapping[str, Any]) -> str:
    """Let the frozen proposal, rather than a CLI default, select Lite mode."""

    return "certified_refresh" if proposal.get("lite_refresh") else "fresh_bo"


def execute_experiment(
    *,
    proposal_root: Path,
    proposal: Mapping[str, Any],
    authorization: Mapping[str, Any],
    phase: str = "all",
    max_lite_records: int | None = None,
    max_shared_scenes: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    execution_root = proposal_root / "execution"
    tables = execution_root / "tables"
    figures = execution_root / "figures"
    cache_root = execution_root / "cache" / "solver"
    for path in (execution_root, tables, figures, cache_root):
        path.mkdir(parents=True, exist_ok=True)
    _write_cache_import_receipt(
        proposal=proposal,
        execution_root=execution_root,
    )

    records = _proposal_records(proposal)
    if max_lite_records is not None:
        records = records[: max(0, int(max_lite_records))]

    lite_receipt: dict[str, Any] | None = None
    shared_receipt: dict[str, Any] | None = None
    try:
        if phase in {"all", "lite"}:
            if _lite_stage_mode(proposal) == "certified_refresh":
                lite_receipt = run_lite_refresh_stage(
                    proposal=proposal,
                    records=records,
                    execution_root=execution_root,
                    cache_root=cache_root,
                    started_monotonic=started,
                )
            else:
                lite_receipt = run_lite_baseline_stage(
                    proposal=proposal,
                    records=records,
                    proposal_root=proposal_root,
                    execution_root=execution_root,
                    cache_root=cache_root,
                    started_monotonic=started,
                )
            if lite_receipt["decision"] == "stop":
                completion = _full_completion(
                    proposal=proposal,
                    authorization=authorization,
                    status="stopped_after_lite_audit",
                    started_monotonic=started,
                    execution_root=execution_root,
                    lite_receipt=lite_receipt,
                    shared_receipt=None,
                )
                _atomic_write_json(proposal_root / "completion.json", completion)
                return completion

        if phase in {"all", "shared"}:
            lite_receipt_path = execution_root / "lite" / "lite_audit_receipt.json"
            if lite_receipt is None and lite_receipt_path.is_file():
                lite_receipt = _read_json(lite_receipt_path)
            if lite_receipt is None and proposal.get("lite_refresh"):
                lite_receipt = run_lite_refresh_stage(
                    proposal=proposal,
                    records=records,
                    execution_root=execution_root,
                    cache_root=cache_root,
                    started_monotonic=started,
                )
            if lite_receipt is None:
                raise LYXExperimentError("lite_known_best_missing")
            if lite_receipt.get("decision") != "proceed":
                completion = _full_completion(
                    proposal=proposal,
                    authorization=authorization,
                    status="stopped_after_lite_audit",
                    started_monotonic=started,
                    execution_root=execution_root,
                    lite_receipt=lite_receipt,
                    shared_receipt=None,
                )
                _atomic_write_json(proposal_root / "completion.json", completion)
                return completion
            shared_receipt = run_shared_parameter_stage(
                proposal=proposal,
                proposal_root=proposal_root,
                execution_root=execution_root,
                cache_root=cache_root,
                lite_receipt=lite_receipt,
                max_shared_scenes=max_shared_scenes,
                started_monotonic=started,
            )
    except LYXExperimentError as exc:
        if not str(exc).startswith("wall_clock_budget_exceeded:"):
            raise
        completion = _full_completion(
            proposal=proposal,
            authorization=authorization,
            status="paused_wall_clock_budget",
            started_monotonic=started,
            execution_root=execution_root,
            lite_receipt=lite_receipt,
            shared_receipt=shared_receipt,
            pause_reason=str(exc),
        )
        _atomic_write_json(proposal_root / "completion.json", completion)
        return completion

    completion = _full_completion(
        proposal=proposal,
        authorization=authorization,
        status="completed",
        started_monotonic=started,
        execution_root=execution_root,
        lite_receipt=lite_receipt,
        shared_receipt=shared_receipt,
    )
    _atomic_write_json(proposal_root / "completion.json", completion)
    return completion


def _ensure_wall_clock_budget(
    proposal: Mapping[str, Any],
    *,
    started_monotonic: float,
    checkpoint: str,
) -> None:
    raw_hours = proposal.get("budget", {}).get("wall_clock_hours", 12)
    wall_clock_hours = float(12 if raw_hours is None else raw_hours)
    if time.monotonic() - started_monotonic > wall_clock_hours * 3600:
        raise LYXExperimentError(f"wall_clock_budget_exceeded:{checkpoint}")


def run_lite_baseline_stage(
    *,
    proposal: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    proposal_root: Path,
    execution_root: Path,
    cache_root: Path,
    started_monotonic: float,
) -> dict[str, Any]:
    lite_root = execution_root / "lite"
    record_root = lite_root / "records"
    figures = execution_root / "figures" / "lite"
    tables = execution_root / "tables"
    for path in (lite_root, record_root, figures, tables):
        path.mkdir(parents=True, exist_ok=True)
    old_batch = Path(str(proposal["data_panel"].get("old_lite_batch_dir", "")))
    if not old_batch.is_dir():
        raise LYXExperimentError(f"old_lite_batch_missing:{old_batch}")

    record_summaries: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        _ensure_wall_clock_budget(
            proposal,
            started_monotonic=started_monotonic,
            checkpoint=f"lite_record_{index}",
        )
        record_id = str(record["record_id"])
        out_dir = record_root / _safe_name(record_id)
        receipt_path = out_dir / "lite_record_receipt.json"
        if receipt_path.is_file():
            receipt = _read_json(receipt_path)
        else:
            print(f"[lite {index}/{len(records)}] {record_id}", flush=True)
            receipt = _run_one_lite_record(
                proposal=proposal,
                record=record,
                old_batch=old_batch,
                out_dir=out_dir,
                cache_root=cache_root,
            )
        record_summaries.append(receipt["summary"])
        repeat_rows.extend(receipt["repeat_rows"])
        _write_dict_csv(tables / "lite_record_summary.csv", record_summaries)
        _write_dict_csv(tables / "lite_repeat_summary.csv", repeat_rows)
        _write_execution_checkpoint(
            execution_root=execution_root,
            status="running",
            stage="lite_baseline",
            last_transaction=f"lite_record_{record_id}",
            details={"completed_record_count": len(record_summaries)},
        )

    _write_dict_csv(tables / "lite_record_summary.csv", record_summaries)
    _write_dict_csv(tables / "lite_repeat_summary.csv", repeat_rows)
    _plot_lite_delta(
        record_summaries,
        figures / "lite_mae_delta_pairplot.png",
    )
    overview_path = figures / "lite_24_record_overview.png"
    _plot_lite_overview_grid(
        [
            {
                "record_id": summary["record_id"],
                "old_report": str(_old_lite_report_path(old_batch, record)),
                "new_report": summary["lite_150_report"],
            }
            for record, summary in zip(records, record_summaries, strict=True)
        ],
        overview_path,
    )
    decision = "proceed"
    stop_reasons = [
        row["audit_reason"]
        for row in record_summaries
        if str(row.get("audit_decision")) == "stop"
    ]
    if stop_reasons:
        decision = "stop"
    receipt = {
        "stage": "lite_baseline",
        "decision": decision,
        "record_count": len(record_summaries),
        "logical_trial_count": sum(int(row["logical_trials"]) for row in record_summaries),
        "unique_solver_count": sum(int(row["unique_solver_count"]) for row in record_summaries),
        "cache_hit_count": sum(int(row["cache_hit_count"]) for row in record_summaries),
        "logical_solver_time_estimate_s": sum(
            float(row["logical_solver_time_estimate_s"]) for row in record_summaries
        ),
        "unique_solver_time_estimate_s": sum(
            float(row["unique_solver_time_estimate_s"]) for row in record_summaries
        ),
        "cache_saved_time_estimate_s": sum(
            float(row["cache_saved_time_estimate_s"]) for row in record_summaries
        ),
        "stop_reasons": stop_reasons,
        "summary_csv": str(tables / "lite_record_summary.csv"),
        "repeat_csv": str(tables / "lite_repeat_summary.csv"),
        "figure": str(figures / "lite_mae_delta_pairplot.png"),
        "overview_figure": str(overview_path),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_write_json(lite_root / "lite_audit_receipt.json", receipt)
    return receipt


def _validate_lite_refresh_sources(
    *,
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    binding = dict(proposal.get("lite_refresh") or {})
    if binding.get("contract_version") != LITE_REFRESH_CONTRACT_VERSION:
        raise LYXExperimentError("lite_refresh_contract_mismatch")
    source_lite_root = Path(str(binding["source_lite_dir"])).resolve()
    audit_root = Path(str(binding["audit_dir"])).resolve()
    source_receipt_path = source_lite_root / "lite_audit_receipt.json"
    audit_completion_path = audit_root / "completion.json"
    if file_sha256(source_receipt_path) != binding.get("source_lite_receipt_sha256"):
        raise LYXExperimentError("lite_refresh_source_receipt_hash_mismatch")
    if file_sha256(audit_completion_path) != binding.get(
        "audit_completion_file_sha256"
    ):
        raise LYXExperimentError("lite_refresh_audit_completion_file_hash_mismatch")

    source_receipt = _read_json(source_receipt_path)
    source_receipt_body = dict(source_receipt)
    source_receipt_sha = str(source_receipt_body.pop("receipt_sha256", ""))
    if not source_receipt_sha or canonical_sha256(source_receipt_body) != source_receipt_sha:
        raise LYXExperimentError("lite_refresh_source_receipt_embedded_hash_mismatch")
    if int(source_receipt.get("record_count", -1)) != len(LITE_PANEL_RECORDS):
        raise LYXExperimentError("lite_refresh_source_record_count_mismatch")

    record_manifest = list(binding.get("source_record_receipts") or [])
    if canonical_sha256(record_manifest) != binding.get(
        "source_record_receipts_sha256"
    ):
        raise LYXExperimentError("lite_refresh_record_manifest_hash_mismatch")
    source_records: dict[str, dict[str, Any]] = {}
    for item in record_manifest:
        path = Path(str(item["path"])).resolve()
        if file_sha256(path) != item.get("sha256"):
            raise LYXExperimentError(
                f"lite_refresh_record_receipt_hash_mismatch:{item.get('record_id')}"
            )
        receipt = _read_json(path)
        record_id = str(receipt.get("record_id") or "")
        if not record_id or record_id in source_records:
            raise LYXExperimentError(f"lite_refresh_duplicate_record:{record_id}")
        if len(receipt.get("history") or []) != 150:
            raise LYXExperimentError(f"lite_refresh_history_count:{record_id}")
        source_records[record_id] = receipt
    if set(source_records) != set(LITE_PANEL_RECORDS):
        raise LYXExperimentError("lite_refresh_record_set_mismatch")

    audit_completion = _read_json(audit_completion_path)
    _verify_external_embedded_hash(
        audit_completion,
        field="completion_sha256",
        label="lite_refresh_audit_completion",
    )
    if (
        audit_completion.get("status") != "complete"
        or int(audit_completion.get("solver_report_count", -1)) != 2394
        or int(audit_completion.get("record_count", -1)) != len(LITE_PANEL_RECORDS)
        or int(audit_completion.get("record_best_any_regression_count", -1)) != 0
    ):
        raise LYXExperimentError("lite_refresh_audit_completion_semantics_mismatch")
    for name, expected_sha in (audit_completion.get("artifacts") or {}).items():
        path = audit_root / str(name)
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise LYXExperimentError(f"lite_refresh_audit_artifact_hash_mismatch:{name}")

    source_closure = _read_json(audit_root / "source_closure.json")
    repository_root = Path(str(proposal["repository_root"])).resolve()
    for item in source_closure.get("files") or []:
        path = repository_root / str(item["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or file_sha256(path) != item["sha256"]
        ):
            raise LYXExperimentError(
                f"lite_refresh_current_source_mismatch:{item.get('path')}"
            )

    proposal_records = {
        str(item["record_id"]): item
        for item in proposal["data_panel"]["resolved_lite_records"]
    }
    audit_inputs = _read_json(audit_root / "input_manifest.json")
    input_records = {
        str(item["record_id"]): item for item in audit_inputs.get("records") or []
    }
    if set(input_records) != set(proposal_records):
        raise LYXExperimentError("lite_refresh_input_record_set_mismatch")
    for record_id, record in proposal_records.items():
        audited = input_records[record_id]
        if (
            audited.get("data_sha256") != record.get("data_sha256")
            or audited.get("reference_sha256") != record.get("ref_sha256")
        ):
            raise LYXExperimentError(f"lite_refresh_input_hash_mismatch:{record_id}")

    affected_rows = _read_dict_csv(audit_root / "affected_coordinates.csv")
    if len(affected_rows) != int(audit_completion["affected_report_count"]):
        raise LYXExperimentError("lite_refresh_affected_row_count_mismatch")
    affected_mae = {
        str(row["cache_id"]): float(row["new_full_mae_bpm"])
        for row in affected_rows
    }
    best_rows = _read_dict_csv(audit_root / "record_best_summary.csv")
    best_by_record = {str(row["record_id"]): row for row in best_rows}
    if set(best_by_record) != set(proposal_records):
        raise LYXExperimentError("lite_refresh_best_record_set_mismatch")
    if any(float(row["best_delta_bpm"]) > 1e-12 for row in best_rows):
        raise LYXExperimentError("lite_refresh_record_best_regression")
    return {
        "binding": binding,
        "source_receipt": source_receipt,
        "source_records": source_records,
        "audit_completion": audit_completion,
        "affected_mae": affected_mae,
        "best_by_record": best_by_record,
    }


def _refresh_lite_history(
    history: Sequence[Mapping[str, Any]],
    affected_mae: Mapping[str, float],
) -> tuple[list[dict[str, Any]], int]:
    refreshed: list[dict[str, Any]] = []
    best_by_repeat: dict[int, float] = {}
    updated_count = 0
    ordered = sorted(history, key=lambda row: int(row["global_trial"]))
    for source_row in ordered:
        row = dict(source_row)
        cache_id = str(row["cache_key"])[:24]
        if cache_id in affected_mae:
            row["value"] = float(affected_mae[cache_id])
            updated_count += 1
        repeat_idx = int(row["repeat_idx"])
        best_by_repeat[repeat_idx] = min(
            best_by_repeat.get(repeat_idx, float("inf")),
            float(row["value"]),
        )
        row["best_in_repeat"] = best_by_repeat[repeat_idx]
        refreshed.append(row)
    return refreshed, updated_count


def _params_from_lite_history(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: row[name] for name in LITE_PARAMETER_NAMES}


def run_lite_refresh_stage(
    *,
    proposal: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    execution_root: Path,
    cache_root: Path,
    started_monotonic: float,
) -> dict[str, Any]:
    validated = _validate_lite_refresh_sources(proposal=proposal)
    if len(records) != len(LITE_PANEL_RECORDS):
        raise LYXExperimentError(f"lite_refresh_denominator:{len(records)}")
    lite_root = execution_root / "lite"
    record_root = lite_root / "records"
    figures = execution_root / "figures" / "lite"
    tables = execution_root / "tables"
    for path in (lite_root, record_root, figures, tables):
        path.mkdir(parents=True, exist_ok=True)
    old_batch = Path(str(proposal["data_panel"].get("old_lite_batch_dir", "")))
    if not old_batch.is_dir():
        raise LYXExperimentError(f"old_lite_batch_missing:{old_batch}")

    record_summaries: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    stop_reasons: list[str] = []
    total_cache_hits = 0
    total_unique_solves = 0
    total_updated_trials = 0
    source_solver_cache = (
        Path(str(validated["binding"]["source_lite_dir"])).parent
        / "cache"
        / "solver"
    )
    for index, record in enumerate(records, start=1):
        _ensure_wall_clock_budget(
            proposal,
            started_monotonic=started_monotonic,
            checkpoint=f"lite_refresh_record_{index}",
        )
        record_started = time.monotonic()
        record_id = str(record["record_id"])
        print(f"[lite-refresh {index}/{len(records)}] {record_id}", flush=True)
        source = validated["source_records"][record_id]
        history, updated_trial_count = _refresh_lite_history(
            source["history"],
            validated["affected_mae"],
        )
        timing = _lite_solver_time_summary(history, cache_root=source_solver_cache)
        if timing["timed_logical_trial_count"] != len(history):
            raise LYXExperimentError(f"lite_timing_coverage_incomplete:{record_id}")
        total_updated_trials += updated_trial_count
        lite_best = min(
            history,
            key=lambda row: (float(row["value"]), int(row["global_trial"])),
        )
        lite_params = _params_from_lite_history(lite_best)
        best_row = validated["best_by_record"][record_id]
        known_params = json.loads(str(best_row["new_best_params"]))

        lite_solve = _solve_cached_report(
            proposal=proposal,
            record=record,
            params=lite_params,
            cache_root=cache_root,
            logical_reference={
                "stage": "lite_refresh_anchor",
                "record_id": record_id,
            },
            candidate_identity={
                "space_name": "lite_150_anchor_refreshed",
                "candidate_id": f"{record_id}:lite_refresh",
                "requested_params": lite_params,
            },
        )
        known_solve = _solve_cached_report(
            proposal=proposal,
            record=record,
            params=known_params,
            cache_root=cache_root,
            logical_reference={
                "stage": "known_best_current_source_anchor_refresh",
                "record_id": record_id,
            },
            candidate_identity={
                "space_name": "known_best_current_source_anchor_refreshed",
                "candidate_id": str(best_row["new_best_cache_id"]),
                "requested_params": known_params,
            },
        )
        solves_by_key = {
            str(lite_solve["cache_key"]): lite_solve,
            str(known_solve["cache_key"]): known_solve,
        }
        total_cache_hits += sum(bool(item["cache_hit"]) for item in solves_by_key.values())
        total_unique_solves += sum(not bool(item["cache_hit"]) for item in solves_by_key.values())

        evaluation_root = _evaluation_cache_root_from_solver_cache(cache_root)
        lite_metrics = _cached_metrics_from_report(
            report_path=Path(str(lite_solve["report_path"])),
            record=record,
            scene=str(record["scene"]),
            evaluation_cache_root=evaluation_root,
        )
        known_metrics = _cached_metrics_from_report(
            report_path=Path(str(known_solve["report_path"])),
            record=record,
            scene=str(record["scene"]),
            evaluation_cache_root=evaluation_root,
        )
        if abs(float(lite_metrics["mae_bpm"]) - float(lite_best["value"])) > 1e-9:
            raise LYXExperimentError(f"lite_refresh_anchor_replay_mismatch:{record_id}")
        if abs(float(known_metrics["mae_bpm"]) - float(best_row["new_best_mae_bpm"])) > 1e-9:
            raise LYXExperimentError(f"lite_refresh_known_best_replay_mismatch:{record_id}")

        old_metrics = dict(source["old_metrics"])
        prior_known_metrics = dict(source["known_best_metrics"])
        historical_audit = _audit_lite_non_regression(old_metrics, known_metrics)
        prior_current_audit = _audit_lite_non_regression(
            prior_known_metrics,
            known_metrics,
        )
        severe = bool(
            historical_audit["severe_regression"]
            or prior_current_audit["severe_regression"]
            or float(best_row["best_delta_bpm"]) > 1e-12
        )
        audit_decision = "stop" if severe else "proceed"
        audit_reason = ";".join(
            reason
            for reason in (
                str(historical_audit["reason"]),
                str(prior_current_audit["reason"]),
            )
            if reason != "ok"
        ) or "ok"
        if severe:
            stop_reasons.append(f"{record_id}:{audit_reason}")

        out_dir = record_root / _safe_name(record_id)
        rendered_lite_report = _save_rendered_lite_best_report(
            record_id=record_id,
            out_dir=out_dir,
            best_result=_solver_result_from_payload(
                _read_json(Path(str(lite_solve["report_path"])))
            ),
            best_params=lite_params,
            history=history,
        )
        old_payload = _read_json(_old_lite_report_path(old_batch, record))
        lite_payload = _read_json(Path(str(lite_solve["report_path"])))
        known_payload = _read_json(Path(str(known_solve["report_path"])))
        overlay = _plot_lite_hr_overlay(
            record_id=record_id,
            old_payload=old_payload,
            new_payload=lite_payload,
            fixed_payload=(
                None
                if lite_solve["cache_key"] == known_solve["cache_key"]
                else known_payload
            ),
            path=out_dir / "png" / f"{_safe_name(record_id)}-old-new-overlay.png",
        )
        record_repeat_rows = _lite_repeat_rows(record_id, history)
        repeat_rows.extend(record_repeat_rows)
        summary = {
            "record_id": record_id,
            "scene": record["scene"],
            "old_mae_bpm": old_metrics["mae_bpm"],
            "new_lite_150_mae_bpm": lite_metrics["mae_bpm"],
            "known_best_current_source_mae_bpm": known_metrics["mae_bpm"],
            "delta_new_minus_old_mae_bpm": (
                float(lite_metrics["mae_bpm"]) - float(old_metrics["mae_bpm"])
            ),
            "known_best_delta_minus_old_mae_bpm": (
                float(known_metrics["mae_bpm"]) - float(old_metrics["mae_bpm"])
            ),
            "old_motion_mae_bpm": old_metrics["motion_mae_bpm"],
            "new_motion_mae_bpm": lite_metrics["motion_mae_bpm"],
            "known_best_motion_mae_bpm": known_metrics["motion_mae_bpm"],
            "old_l10": old_metrics["l10"],
            "new_l10": lite_metrics["l10"],
            "known_best_l10": known_metrics["l10"],
            "old_l20": old_metrics["l20"],
            "new_l20": lite_metrics["l20"],
            "known_best_l20": known_metrics["l20"],
            "logical_trials": len(history),
            "updated_logical_trials": updated_trial_count,
            "unique_coordinate_count": len({str(row["cache_key"]) for row in history}),
            "duplicate_coordinate_count": len(history)
            - len({str(row["cache_key"]) for row in history}),
            "refresh_unique_solver_count": sum(
                not bool(item["cache_hit"]) for item in solves_by_key.values()
            ),
            "refresh_cache_hit_count": sum(
                bool(item["cache_hit"]) for item in solves_by_key.values()
            ),
            **timing,
            "best_global_trial": int(lite_best["global_trial"]),
            "best_repeat_idx": int(lite_best["repeat_idx"]),
            "best_trial_idx": int(lite_best["trial_idx"]),
            "lite_150_report": str(rendered_lite_report),
            "known_best_report": str(known_solve["report_path"]),
            "fixed_replay_report": (
                ""
                if lite_solve["cache_key"] == known_solve["cache_key"]
                else str(known_solve["report_path"])
            ),
            "overlay_figure": str(overlay),
            "audit_decision": audit_decision,
            "audit_reason": audit_reason,
            "refresh_wall_seconds": time.monotonic() - record_started,
        }
        record_summaries.append(summary)
        receipt = {
            "record_id": record_id,
            "summary": summary,
            "old_metrics": old_metrics,
            "new_metrics": lite_metrics,
            "known_best_metrics": known_metrics,
            "prior_known_best_metrics": prior_known_metrics,
            "fixed_replay": (
                None
                if lite_solve["cache_key"] == known_solve["cache_key"]
                else {"report_path": str(known_solve["report_path"]), "metrics": known_metrics}
            ),
            "history": history,
            "repeat_rows": record_repeat_rows,
            "historical_audit": historical_audit,
            "prior_current_audit": prior_current_audit,
            "audit": {
                "decision": audit_decision,
                "reason": audit_reason,
                "source_completion_sha256": validated["audit_completion"][
                    "completion_sha256"
                ],
            },
        }
        _atomic_write_json(out_dir / "lite_record_receipt.json", receipt)
        _write_dict_csv(tables / "lite_record_summary.csv", record_summaries)
        _write_dict_csv(tables / "lite_repeat_summary.csv", repeat_rows)
        _write_execution_checkpoint(
            execution_root=execution_root,
            status="running",
            stage="lite_refresh",
            last_transaction=f"lite_refresh_record_{record_id}",
            details={"completed_record_count": len(record_summaries)},
        )

    _write_dict_csv(tables / "lite_record_summary.csv", record_summaries)
    _write_dict_csv(tables / "lite_repeat_summary.csv", repeat_rows)
    _plot_lite_delta(record_summaries, figures / "lite_mae_delta_pairplot.png")
    overview_path = figures / "lite_24_record_overview.png"
    _plot_lite_overview_grid(
        [
            {
                "record_id": summary["record_id"],
                "old_report": str(_old_lite_report_path(old_batch, record)),
                "new_report": summary["lite_150_report"],
            }
            for record, summary in zip(records, record_summaries, strict=True)
        ],
        overview_path,
    )
    receipt = {
        "stage": "lite_baseline",
        "mode": "certified_exhaustive_coordinate_refresh",
        "contract_version": LITE_REFRESH_CONTRACT_VERSION,
        "decision": "stop" if stop_reasons else "proceed",
        "record_count": len(record_summaries),
        "logical_trial_count": sum(int(row["logical_trials"]) for row in record_summaries),
        "new_bo_trial_count": 0,
        "updated_logical_trial_count": total_updated_trials,
        "unique_solver_count": total_unique_solves,
        "cache_hit_count": total_cache_hits,
        "logical_solver_time_estimate_s": sum(
            float(row["logical_solver_time_estimate_s"]) for row in record_summaries
        ),
        "unique_solver_time_estimate_s": sum(
            float(row["unique_solver_time_estimate_s"]) for row in record_summaries
        ),
        "cache_saved_time_estimate_s": sum(
            float(row["cache_saved_time_estimate_s"]) for row in record_summaries
        ),
        "stop_reasons": stop_reasons,
        "source_lite_receipt_sha256": validated["source_receipt"]["receipt_sha256"],
        "refresh_audit_completion_sha256": validated["audit_completion"][
            "completion_sha256"
        ],
        "summary_csv": str(tables / "lite_record_summary.csv"),
        "repeat_csv": str(tables / "lite_repeat_summary.csv"),
        "figure": str(figures / "lite_mae_delta_pairplot.png"),
        "overview_figure": str(overview_path),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_write_json(lite_root / "lite_audit_receipt.json", receipt)
    return receipt


def _run_one_lite_record(
    *,
    proposal: Mapping[str, Any],
    record: Mapping[str, Any],
    old_batch: Path,
    out_dir: Path,
    cache_root: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    record_id = str(record["record_id"])
    evaluation_cache_root = _evaluation_cache_root_from_solver_cache(cache_root)
    old_report = _old_lite_report_path(old_batch, record)
    if not old_report.is_file():
        raise LYXExperimentError(f"old_lite_report_missing:{record_id}")
    old_payload = _read_json(old_report)
    old_metrics = _cached_metrics_from_report(
        report_path=old_report,
        record=record,
        scene=str(record["scene"]),
        evaluation_cache_root=evaluation_cache_root,
    )

    optimise = _optimise_lite_record_with_cache(
        proposal=proposal,
        record=record,
        out_dir=out_dir,
        cache_root=cache_root,
    )
    new_payload = _read_json(Path(optimise["best_report"]))
    new_metrics = _cached_metrics_from_report(
        report_path=Path(str(optimise["best_report"])),
        record=record,
        scene=str(record["scene"]),
        evaluation_cache_root=evaluation_cache_root,
    )
    audit = _audit_lite_non_regression(old_metrics, new_metrics)

    fixed_replay: dict[str, Any] | None = None
    known_best_report = str(optimise["best_report"])
    known_best_metrics = dict(new_metrics)
    if audit["requires_fixed_replay"]:
        old_params = dict(old_payload.get("best_params") or {})
        if old_params:
            fixed_replay = _solve_cached_report(
                proposal=proposal,
                record=record,
                params=old_params,
                cache_root=cache_root,
                logical_reference={
                    "stage": "lite_fixed_replay",
                    "record_id": record_id,
                },
                candidate_identity={
                    "space_name": "old_lite_best_fixed_replay",
                    "candidate_id": f"{record_id}:old_lite_best",
                    "requested_params": old_params,
                },
            )
            replay_metrics = _cached_metrics_from_report(
                report_path=Path(str(fixed_replay["report_path"])),
                record=record,
                scene=str(record["scene"]),
                evaluation_cache_root=evaluation_cache_root,
            )
            fixed_replay["metrics"] = replay_metrics
            replay_audit = _audit_lite_non_regression(old_metrics, replay_metrics)
            fixed_replay["audit"] = replay_audit
            if replay_metrics["mae_bpm"] < known_best_metrics["mae_bpm"]:
                known_best_metrics = replay_metrics
                known_best_report = str(fixed_replay["report_path"])
            if replay_audit["severe_regression"]:
                audit["decision"] = "stop"
                audit["reason"] = "fixed_replay_confirms_mechanism_regression"

    overlay_figure = _plot_lite_hr_overlay(
        record_id=record_id,
        old_payload=old_payload,
        new_payload=new_payload,
        fixed_payload=(
            None
            if fixed_replay is None
            else _read_json(Path(str(fixed_replay["report_path"])))
        ),
        path=out_dir / "png" / f"{_safe_name(record_id)}-old-new-overlay.png",
    )
    repeat_rows = _lite_repeat_rows(record_id, optimise["history"])
    summary = {
        "record_id": record_id,
        "scene": record["scene"],
        "old_mae_bpm": old_metrics["mae_bpm"],
        "new_lite_150_mae_bpm": new_metrics["mae_bpm"],
        "known_best_current_source_mae_bpm": known_best_metrics["mae_bpm"],
        "delta_new_minus_old_mae_bpm": new_metrics["mae_bpm"] - old_metrics["mae_bpm"],
        "old_motion_mae_bpm": old_metrics["motion_mae_bpm"],
        "new_motion_mae_bpm": new_metrics["motion_mae_bpm"],
        "delta_new_minus_old_motion_mae_bpm": (
            new_metrics["motion_mae_bpm"] - old_metrics["motion_mae_bpm"]
        ),
        "old_l10": old_metrics["l10"],
        "new_l10": new_metrics["l10"],
        "old_l20": old_metrics["l20"],
        "new_l20": new_metrics["l20"],
        "logical_trials": len(optimise["history"]),
        "unique_coordinate_count": optimise["unique_coordinate_count"],
        "duplicate_coordinate_count": optimise["duplicate_coordinate_count"],
        "unique_solver_count": optimise["unique_solver_count"],
        "cache_hit_count": optimise["cache_hit_count"],
        "timing_contract": optimise["timing_contract"],
        "timed_logical_trial_count": optimise["timed_logical_trial_count"],
        "timed_unique_coordinate_count": optimise["timed_unique_coordinate_count"],
        "logical_solver_time_estimate_s": optimise[
            "logical_solver_time_estimate_s"
        ],
        "unique_solver_time_estimate_s": optimise[
            "unique_solver_time_estimate_s"
        ],
        "cache_saved_time_estimate_s": optimise["cache_saved_time_estimate_s"],
        "best_global_trial": optimise["best_global_trial"],
        "best_repeat_idx": optimise["best_repeat_idx"],
        "best_trial_idx": optimise["best_trial_idx"],
        "lite_150_report": str(optimise["best_report"]),
        "known_best_report": known_best_report,
        "fixed_replay_report": "" if fixed_replay is None else fixed_replay["report_path"],
        "overlay_figure": str(overlay_figure),
        "audit_decision": audit["decision"],
        "audit_reason": audit["reason"],
    }
    receipt = {
        "record_id": record_id,
        "summary": summary,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "known_best_metrics": known_best_metrics,
        "fixed_replay": fixed_replay,
        "history": optimise["history"],
        "repeat_rows": repeat_rows,
        "audit": audit,
    }
    _atomic_write_json(out_dir / "lite_record_receipt.json", receipt)
    return receipt


def _optimise_lite_record_with_cache(
    *,
    proposal: Mapping[str, Any],
    record: Mapping[str, Any],
    out_dir: Path,
    cache_root: Path,
) -> dict[str, Any]:
    import optuna

    from ppg_hr.v2.algorithm_presets import v2_search_space_for_preset
    from ppg_hr.v2.search_space import decode_v2

    space = v2_search_space_for_preset("lms", "lite")
    trials_per_repeat = 50
    repeat_total = 3
    seed0 = 42
    history: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    seen_coordinates: set[tuple[Any, ...]] = set()
    cache_hits = 0
    unique_solves = 0

    for repeat_idx0 in range(repeat_total):
        repeat_best = [float("inf")]

        def objective(
            trial: Any,
            *,
            _repeat_idx0: int = repeat_idx0,
            _repeat_best: list[float] = repeat_best,
        ) -> float:
            nonlocal best_row, cache_hits, unique_solves
            idx_map = {
                name: trial.suggest_int(name, 0, len(space.options(name)) - 1)
                for name in space.names()
            }
            params = decode_v2(space, idx_map)
            coordinate = tuple(params[name] for name in space.names())
            is_duplicate = coordinate in seen_coordinates
            seen_coordinates.add(coordinate)
            solve = _solve_cached_report(
                proposal=proposal,
                record=record,
                params=params,
                cache_root=cache_root,
                logical_reference={
                    "stage": "lite_150_anchor",
                    "record_id": record["record_id"],
                    "repeat_idx": _repeat_idx0 + 1,
                    "trial_idx": trial.number + 1,
                },
                candidate_identity={
                    "space_name": "lite_150_anchor",
                    "candidate_id": canonical_sha256(
                        {
                            "record_id": record["record_id"],
                            "params": params,
                        }
                    ),
                    "requested_params": params,
                },
            )
            cache_hits += int(bool(solve["cache_hit"]))
            unique_solves += int(not bool(solve["cache_hit"]))
            payload = _read_json(Path(solve["report_path"]))
            value = _finite_objective(payload["err_stats"].get("final_aae_bpm"))
            _repeat_best[0] = min(_repeat_best[0], value)
            row = {
                "repeat_idx": _repeat_idx0 + 1,
                "repeat_total": repeat_total,
                "trial": trial.number,
                "trial_idx": trial.number + 1,
                "trial_total": trials_per_repeat,
                "global_trial": _repeat_idx0 * trials_per_repeat + trial.number + 1,
                "global_total": trials_per_repeat * repeat_total,
                "value": value,
                "best_in_repeat": _repeat_best[0],
                "cache_key": solve["cache_key"],
                "cache_hit": solve["cache_hit"],
                "solver_elapsed_s": solve["solver_elapsed_s"],
                "is_duplicate_coordinate": is_duplicate,
                **params,
            }
            history.append(row)
            if best_row is None or (
                value,
                int(row["global_trial"]),
            ) < (
                float(best_row["value"]),
                int(best_row["global_trial"]),
            ):
                best_row = row
            return value

        sampler = optuna.samplers.TPESampler(
            seed=seed0 + repeat_idx0,
            n_startup_trials=10,
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=trials_per_repeat, show_progress_bar=False)

    if best_row is None:
        raise LYXExperimentError(f"lite_no_trials:{record['record_id']}")
    best_params = {name: best_row[name] for name in space.names()}
    best_cached = _solve_cached_report(
        proposal=proposal,
        record=record,
        params=best_params,
        cache_root=cache_root,
        logical_reference={
            "stage": "lite_150_anchor_best_report",
            "record_id": record["record_id"],
        },
        candidate_identity={
            "space_name": "lite_150_anchor",
            "candidate_id": str(best_row["cache_key"]),
            "requested_params": best_params,
        },
    )
    best_result = _solver_result_from_payload(_read_json(Path(best_cached["report_path"])))
    final_report = _save_rendered_lite_best_report(
        record_id=str(record["record_id"]),
        out_dir=out_dir,
        best_result=best_result,
        best_params=best_params,
        history=history,
    )
    timing = _lite_solver_time_summary(history)
    return {
        "best_report": str(final_report),
        "best_params": best_params,
        "history": history,
        "unique_coordinate_count": len(seen_coordinates),
        "duplicate_coordinate_count": len(history) - len(seen_coordinates),
        "unique_solver_count": unique_solves,
        "cache_hit_count": cache_hits,
        **timing,
        "best_global_trial": int(best_row["global_trial"]),
        "best_repeat_idx": int(best_row["repeat_idx"]),
        "best_trial_idx": int(best_row["trial_idx"]),
    }


def _save_rendered_lite_best_report(
    *,
    record_id: str,
    out_dir: Path,
    best_result: Any,
    best_params: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> Path:
    from ppg_hr.v2.plotting import render_v2_report
    from ppg_hr.v2.report import save_v2_report

    json_dir = out_dir / "json"
    png_dir = out_dir / "png"
    csv_dir = out_dir / "csv"
    for path in (json_dir, png_dir, csv_dir):
        path.mkdir(parents=True, exist_ok=True)
    requested_report = json_dir / f"{_safe_name(record_id)}-lite-current-v2.json"
    saved_report = save_v2_report(
        requested_report,
        best_result,
        best_params=dict(best_params),
        history=[dict(row) for row in history],
    )
    render_v2_report(saved_report, out_dir=png_dir, csv_dir=csv_dir)
    _write_dict_csv(csv_dir / "lite_trial_history.csv", history)
    return saved_report


def _write_shared_progress_tables(
    *,
    tables: Path,
    all_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    control_metrics: Mapping[str, Mapping[str, Any]],
    scene_summaries: Sequence[Mapping[str, Any]],
    fold_rows: Sequence[Mapping[str, Any]],
    funnel_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidate_rows = sorted(
        (dict(row) for row in all_rows.values()),
        key=lambda row: (
            str(row.get("scene", "")),
            str(row.get("record_id", "")),
            str(row.get("candidate_id", "")),
        ),
    )
    controls = sorted(
        (dict(row) for row in control_metrics.values()),
        key=lambda row: str(row.get("record_id", "")),
    )
    _write_dict_csv(tables / "shared_candidate_rows.csv", candidate_rows)
    _write_dict_csv(tables / "shared_current_controls.csv", controls)
    _write_dict_csv(tables / "shared_scene_summary.csv", scene_summaries)
    _write_dict_csv(tables / "shared_fold_results.csv", fold_rows)
    _write_dict_csv(tables / "shared_candidate_funnel.csv", funnel_rows)
    return candidate_rows


def run_shared_parameter_stage(
    *,
    proposal: Mapping[str, Any],
    proposal_root: Path,
    execution_root: Path,
    cache_root: Path,
    lite_receipt: Mapping[str, Any],
    started_monotonic: float,
    max_shared_scenes: int | None = None,
) -> dict[str, Any]:
    del proposal_root
    shared_root = execution_root / "shared"
    tables = execution_root / "tables"
    figures = execution_root / "figures" / "shared"
    for path in (shared_root, tables, figures):
        path.mkdir(parents=True, exist_ok=True)
    records_by_id = {str(item["record_id"]): item for item in _proposal_records(proposal)}
    lite_by_record = _load_lite_known_best(execution_root / "lite")
    candidates = physical_25hz_extended_candidates()
    workers = max(1, int(proposal.get("budget", {}).get("execution_workers", 1)))
    scenes = list(SHARED_SCENES.items())
    if max_shared_scenes is not None:
        scenes = scenes[: max(0, int(max_shared_scenes))]

    all_rows: dict[tuple[str, str], dict[str, Any]] = {}
    scene_summaries: list[dict[str, Any]] = []
    funnel_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    control_metrics: dict[str, dict[str, Any]] = {}
    control_record_ids = list(
        dict.fromkeys(rid for _, scene_records in scenes for rid in scene_records)
    )
    for record_id in control_record_ids:
        _ensure_wall_clock_budget(
            proposal,
            started_monotonic=started_monotonic,
            checkpoint=f"shared_control_{record_id}",
        )
        control_candidate = _current_control_candidate(candidates)
        control_metrics[record_id] = _physical_metrics_for_candidate(
            proposal=proposal,
            record=records_by_id[record_id],
            candidate=control_candidate,
            cache_root=cache_root,
            logical_stage="current_control",
        )
        _write_shared_progress_tables(
            tables=tables,
            all_rows=all_rows,
            control_metrics=control_metrics,
            scene_summaries=scene_summaries,
            fold_rows=fold_rows,
            funnel_rows=funnel_rows,
        )
        _write_execution_checkpoint(
            execution_root=execution_root,
            status="running",
            stage="shared_controls",
            last_transaction=f"shared_control_{record_id}",
            details={"completed_control_count": len(control_metrics)},
        )

    for scene_index, (scene, record_order) in enumerate(scenes, start=1):
        _ensure_wall_clock_budget(
            proposal,
            started_monotonic=started_monotonic,
            checkpoint=f"shared_scene_{scene}",
        )
        print(f"[shared {scene_index}/{len(scenes)}] {scene}", flush=True)
        active_ids = [str(item["candidate_id"]) for item in candidates]
        scene_first_failures: dict[str, str] = {}
        for record_index, record_id in enumerate(record_order, start=1):
            incoming_count = len(active_ids)
            next_active: list[str] = []
            active_set = set(active_ids)
            active_candidates = [
                candidate
                for candidate in candidates
                if str(candidate["candidate_id"]) in active_set
            ]
            _ensure_wall_clock_budget(
                proposal,
                started_monotonic=started_monotonic,
                checkpoint=f"shared_candidate_{scene}_{record_id}",
            )
            evaluated_rows = _evaluate_shared_candidate_batch(
                proposal=proposal,
                scene=scene,
                record=records_by_id[record_id],
                candidates=active_candidates,
                independent=lite_by_record[record_id]["known_best_metrics"],
                current=control_metrics[record_id],
                cache_root=cache_root,
                logical_stage="short_circuit",
                workers=workers,
            )
            for row in evaluated_rows:
                candidate_id = str(row["candidate_id"])
                all_rows[(record_id, candidate_id)] = row
                if row["qualified"]:
                    next_active.append(candidate_id)
                else:
                    scene_first_failures.setdefault(
                        candidate_id,
                        str(row["first_failed_gate"]),
                    )
            active_ids = next_active
            funnel_rows.append(
                {
                    "scene": scene,
                    "record_order_index": record_index,
                    "record_id": record_id,
                    "incoming_candidate_count": incoming_count,
                    "evaluated_candidate_count": len(evaluated_rows),
                    "surviving_candidate_count": len(active_ids),
                    "failed_candidate_count": incoming_count - len(active_ids),
                }
            )
            _write_shared_progress_tables(
                tables=tables,
                all_rows=all_rows,
                control_metrics=control_metrics,
                scene_summaries=scene_summaries,
                fold_rows=fold_rows,
                funnel_rows=funnel_rows,
            )
            _write_execution_checkpoint(
                execution_root=execution_root,
                status="running",
                stage="shared_short_circuit",
                last_transaction=f"shared_candidate_{scene}_{record_id}",
                details={
                    "scene": scene,
                    "surviving_candidate_count": len(active_ids),
                    "candidate_row_count": len(all_rows),
                },
            )
            if not active_ids:
                break

        platform = analyse_common_platforms(
            candidates=candidates,
            rows=all_rows,
            record_ids=record_order,
        )
        fold_results: list[dict[str, Any]] = []
        if platform["common_safe_candidate_count"] > 0:
            for heldout_idx, heldout_id in enumerate(record_order):
                train_ids = tuple(
                    record_id
                    for idx, record_id in enumerate(record_order)
                    if idx != heldout_idx
                )
                fold_result = _run_one_scene_fold(
                    proposal=proposal,
                    scene=scene,
                    records_by_id=records_by_id,
                    candidates=candidates,
                    lite_by_record=lite_by_record,
                    control_metrics=control_metrics,
                    cache_root=cache_root,
                    all_rows=all_rows,
                    train_ids=train_ids,
                    heldout_id=heldout_id,
                    fold_root=shared_root / "folds" / scene,
                    workers=workers,
                )
                fold_results.append(fold_result)
                fold_rows.append({"scene": scene, **fold_result})
                _write_shared_progress_tables(
                    tables=tables,
                    all_rows=all_rows,
                    control_metrics=control_metrics,
                    scene_summaries=scene_summaries,
                    fold_rows=fold_rows,
                    funnel_rows=funnel_rows,
                )
                _write_execution_checkpoint(
                    execution_root=execution_root,
                    status="running",
                    stage="shared_fold",
                    last_transaction=f"shared_fold_{scene}_{heldout_id}",
                    details={"completed_fold_count": len(fold_rows)},
                )
        classification = classify_scene_fold_results(
            fold_results=fold_results,
            common_components=platform["common_components"],
            candidates_by_id={str(item["candidate_id"]): item for item in candidates},
        )
        reference_line = evaluate_scene_reference_line(
            fold_results=fold_results,
            scene=scene,
        )
        final_candidate_id = ""
        if classification.get("level") == "stable_shared_parameter":
            final_candidate_id = str(fold_results[0]["selected_candidate_id"])
        elif classification.get("level") == "stable_shared_neighborhood":
            final = select_near_optimal_candidate(
                candidates=candidates,
                rows=all_rows,
                train_record_ids=record_order,
            )
            final_candidate_id = str(final["selected_candidate_id"])
        final_candidate = next(
            (
                item
                for item in candidates
                if str(item["candidate_id"]) == final_candidate_id
            ),
            None,
        )
        scene_summaries.append(
            {
                "scene": scene,
                "record_order": " -> ".join(record_order),
                "common_safe_candidate_count": platform["common_safe_candidate_count"],
                "safe_platform_count": platform["safe_platform_count"],
                "strong_flat_platform_count": platform["strong_flat_platform_count"],
                "sensitivity_2bpm_platform_count": platform[
                    "sensitivity_2bpm_platform_count"
                ],
                "largest_component_size": platform["largest_component_size"],
                "fold_count": len(fold_results),
                "fold_pass_count": sum(bool(row.get("heldout_pass")) for row in fold_results),
                "classification": classification.get("level"),
                "reference_line_pass": reference_line["passed"],
                "reference_line_failures": json.dumps(
                    reference_line["failures"],
                    ensure_ascii=False,
                ),
                "final_candidate_id": final_candidate_id,
                "final_requested_params_json": (
                    ""
                    if final_candidate is None
                    else json.dumps(
                        final_candidate["requested_params"],
                        ensure_ascii=False,
                    )
                ),
                "first_failure_kinds": json.dumps(
                    _count_values(scene_first_failures.values()),
                    ensure_ascii=False,
                ),
            }
        )
        _atomic_write_json(
            shared_root / f"{scene}_platform.json",
            {
                **platform,
                "classification": classification,
                "reference_line": reference_line,
                "final_candidate_id": final_candidate_id,
                "final_candidate": final_candidate,
            },
        )
        _write_shared_progress_tables(
            tables=tables,
            all_rows=all_rows,
            control_metrics=control_metrics,
            scene_summaries=scene_summaries,
            fold_rows=fold_rows,
            funnel_rows=funnel_rows,
        )
        _write_execution_checkpoint(
            execution_root=execution_root,
            status="running",
            stage="shared_scene",
            last_transaction=f"shared_scene_{scene}",
            details={"completed_scene_count": len(scene_summaries)},
        )

    candidate_rows = _write_shared_progress_tables(
        tables=tables,
        all_rows=all_rows,
        control_metrics=control_metrics,
        scene_summaries=scene_summaries,
        fold_rows=fold_rows,
        funnel_rows=funnel_rows,
    )
    _plot_shared_scene_summary(
        scene_summaries,
        figures / "shared_scene_common_safe_counts.png",
    )
    _plot_shared_candidate_funnel(
        funnel_rows,
        figures / "shared_candidate_survival_funnel.png",
    )
    _render_shared_fold_figures(
        fold_rows=fold_rows,
        lite_by_record=lite_by_record,
        control_metrics=control_metrics,
        path_root=figures / "folds",
    )
    logical_summary = _shared_logical_request_summary(tables)
    receipt = {
        "stage": "shared_parameters",
        "source_lite_receipt_sha256": lite_receipt.get("receipt_sha256"),
        "scene_count": len(scene_summaries),
        "candidate_row_count": len(candidate_rows),
        **logical_summary,
        "fold_count": len(fold_rows),
        "fold_freeze_count": sum(bool(row.get("freeze_receipt")) for row in fold_rows),
        "fold_reveal_count": sum(bool(row.get("reveal_receipt")) for row in fold_rows),
        "any_scene_failure": any(
            row["classification"] not in {
                "stable_shared_parameter",
                "stable_shared_neighborhood",
            }
            for row in scene_summaries
        ),
        "summary_csv": str(tables / "shared_scene_summary.csv"),
        "candidate_rows_csv": str(tables / "shared_candidate_rows.csv"),
        "fold_results_csv": str(tables / "shared_fold_results.csv"),
        "candidate_funnel_csv": str(tables / "shared_candidate_funnel.csv"),
        "figure": str(figures / "shared_scene_common_safe_counts.png"),
        "funnel_figure": str(figures / "shared_candidate_survival_funnel.png"),
        "scene_decisions": scene_summaries,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_write_json(shared_root / "shared_parameter_receipt.json", receipt)
    return receipt


def write_report(proposal_dir: Path) -> Path:
    proposal_root = Path(proposal_dir).resolve()
    proposal = _read_json(proposal_root / "proposal.json")
    completion_path = proposal_root / "completion.json"
    completion = _read_json(completion_path) if completion_path.is_file() else {}
    report = proposal_root / "execution_status_report.md"
    lines = [
        "# LYX 当前源码 Lite/共享参数实验执行状态",
        "",
        f"- Proposal SHA：`{proposal['proposal_sha256']}`",
        f"- 当前状态：`{completion.get('status', 'not_started')}`",
        f"- 正式 solver 运行数：`{completion.get('formal_solver_run_count', 0)}`",
        f"- 下一状态：`{completion.get('next_state', 'build_authorization')}`",
        "",
        "本状态文件是受治理的执行回执，不等同于最终科学结论。",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-proposal")
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--spec-path", required=True, type=Path)
    build.add_argument("--data-root", required=True, type=Path)
    build.add_argument("--repository-root", required=True, type=Path)
    build.add_argument("--source-lite-dir", type=Path)
    build.add_argument("--lite-refresh-audit-dir", type=Path)
    authorize = sub.add_parser("authorize-window")
    authorize.add_argument("--proposal-dir", required=True, type=Path)
    authorize.add_argument("--expires-at")
    authorize.add_argument("--approved-at", help=argparse.SUPPRESS)
    authorize.add_argument(
        "--allow-clock-override-for-tests",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    run = sub.add_parser("run")
    run.add_argument("--proposal-dir", required=True, type=Path)
    run.add_argument("--phase", choices=("all", "lite", "shared"), default="all")
    run.add_argument("--max-lite-records", type=int)
    run.add_argument("--max-shared-scenes", type=int)
    run.add_argument("--now", help=argparse.SUPPRESS)
    run.add_argument(
        "--allow-clock-override-for-tests",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    report = sub.add_parser("report")
    report.add_argument("--proposal-dir", required=True, type=Path)
    return parser


def _proposal_records(proposal: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = list(proposal.get("data_panel", {}).get("resolved_lite_records") or [])
    if not records:
        data_root = Path(str(proposal.get("data_panel", {}).get("root", "")))
        records = resolve_lite_panel_records(data_root)
    return [dict(item) for item in records]


def _old_lite_report_path(old_batch: Path, record: Mapping[str, Any]) -> Path:
    record_id = str(record["record_id"])
    json_dir = old_batch / "json"
    aliases = (record_id, *OLD_LITE_REPORT_ALIASES.get(record_id, ()))
    identity_candidates = [
        json_dir / f"{alias}-green-raw_bandpass-lms-full-HF-v2.json"
        for alias in aliases
    ]
    for report in identity_candidates:
        if report.is_file() and _old_lite_report_hash_matches(
            report,
            record,
            allow_record_path_fallback=True,
        ):
            return report

    hash_matches = [
        report
        for report in sorted(json_dir.glob("*-green-raw_bandpass-lms-full-HF-v2.json"))
        if report not in identity_candidates
        and _old_lite_report_hash_matches(
            report,
            record,
            allow_record_path_fallback=False,
        )
    ]
    if len(hash_matches) == 1:
        return hash_matches[0]
    if len(hash_matches) > 1:
        raise LYXExperimentError(f"old_lite_hash_alias_ambiguous:{record_id}")
    if any(report.is_file() for report in identity_candidates):
        raise LYXExperimentError(f"old_lite_hash_mismatch:{record_id}")
    return identity_candidates[0]


def _old_lite_report_hash_matches(
    report: Path,
    record: Mapping[str, Any],
    *,
    allow_record_path_fallback: bool,
) -> bool:
    record_id = str(record["record_id"])
    payload = _read_json(report)
    data_candidates = [
        Path(str(payload.get("data_path") or "")),
    ]
    ref_candidates = [
        Path(str(payload.get("ref_path") or "")),
    ]
    if allow_record_path_fallback:
        data_candidates.append(Path(str(record.get("data_path") or "")))
        ref_candidates.append(Path(str(record.get("ref_path") or "")))
    data_sources = [path for path in data_candidates if path.is_file()]
    ref_sources = [path for path in ref_candidates if path.is_file()]
    if not data_sources or not ref_sources:
        if not allow_record_path_fallback:
            return False
        raise LYXExperimentError(f"old_lite_hash_source_missing:{record_id}")
    if not any(
        file_sha256(path) == str(record["data_sha256"]) for path in data_sources
    ):
        return False
    if not any(file_sha256(path) == str(record["ref_sha256"]) for path in ref_sources):
        return False
    return True


def _solve_cached_report(
    *,
    proposal: Mapping[str, Any],
    record: Mapping[str, Any],
    params: Mapping[str, Any],
    cache_root: Path,
    logical_reference: Mapping[str, Any],
    candidate_identity: Mapping[str, Any],
) -> dict[str, Any]:
    from ppg_hr.v2.report import save_v2_report
    from ppg_hr.v2.solver import solve_v2
    from ppg_hr.v2.types import V2RunConfig

    effective_params = _effective_solver_params(params)
    algorithm_sha = canonical_sha256(
        {
            "ppg_hr_source_tree": proposal.get("ppg_hr_source_tree"),
            "algorithm_identity": proposal.get("algorithm_identity"),
        }
    )
    cache_candidate = {
        "actual_params": effective_params,
        "fixed_params": {},
        **dict(candidate_identity),
    }
    key = build_solver_cache_key(
        algorithm_source_sha256=algorithm_sha,
        data_sha256=str(record["data_sha256"]),
        candidate=cache_candidate,
        mechanism_identity={
            "dual_cascade_identity": DUAL_CASCADE_IDENTITY,
            "algorithm_preset": "lite",
            "adaptive_filter": "lms",
            "reference_groups_order": ["HF"],
            "adaptive_reference_stage_limit": None,
        },
        logical_context=logical_reference,
    )["key"]
    entry = _solver_cache_entry_path(cache_root, key)
    report_path = entry / "report-v2.json"
    complete_path = entry / "complete.json"
    if report_path.is_file() and complete_path.is_file():
        complete = _read_json(complete_path)
        if complete.get("cache_key") != key:
            raise LYXExperimentError(
                f"solver_cache_key_prefix_collision:{entry}:{key}"
            )
        return {
            "cache_key": key,
            "cache_hit": True,
            "report_path": str(report_path),
            "solver_elapsed_s": float(complete.get("elapsed_s", 0.0)),
        }
    entry.mkdir(parents=True, exist_ok=True)
    cfg_values = {
        "data_path": Path(str(record["data_path"])),
        "ref_path": Path(str(record["ref_path"])),
        "ppg_mode": "green",
        "ppg_input_transform": "raw_bandpass",
        "analysis_scope": "full",
        "adaptive_filter": "lms",
        "algorithm_preset": "lite",
        "reference_groups_order": ("HF",),
    }
    cfg_values.update(effective_params)
    started = time.monotonic()
    try:
        result = solve_v2(V2RunConfig(**cfg_values))
        saved = save_v2_report(
            report_path,
            result,
            best_params=effective_params,
            history=[],
            artefacts={
                "solver_cache_key": key,
                "logical_reference": dict(logical_reference),
            },
        )
    except Exception as exc:
        entry.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            entry / "failed.json",
            {
                "cache_key": key,
                "status": "failed",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "logical_reference": dict(logical_reference),
            },
        )
        raise
    elapsed_s = time.monotonic() - started
    _atomic_write_json(
        entry / "complete.json",
        {
            "cache_key": key,
            "status": "complete",
            "elapsed_s": elapsed_s,
            "report_path": str(saved),
            "logical_reference": dict(logical_reference),
            "candidate_identity": dict(candidate_identity),
        },
    )
    return {
        "cache_key": key,
        "cache_hit": False,
        "report_path": str(saved),
        "solver_elapsed_s": elapsed_s,
    }


def _solver_cache_entry_path(cache_root: Path, key: str) -> Path:
    """Use an audited short directory name to stay under Windows path budgets."""

    return cache_root / key[:24]


def _lite_solver_time_summary(
    history: Sequence[Mapping[str, Any]],
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Estimate logical, unique-coordinate and cache-saved solver time.

    ``elapsed_s`` is the measured duration stored with the solver identity.  A
    duplicate logical trial inherits that identity's measured duration, so the
    difference between the logical sum and the one-per-key sum is the compute
    avoided by deterministic cache reuse.
    """

    elapsed_by_key: dict[str, float] = {}
    logical_elapsed: list[float] = []
    for row in history:
        key = str(row.get("cache_key") or "")
        if not key:
            continue
        elapsed_raw = row.get("solver_elapsed_s")
        if elapsed_raw is None and cache_root is not None:
            complete_path = _solver_cache_entry_path(cache_root, key) / "complete.json"
            if complete_path.is_file():
                complete = _read_json(complete_path)
                if str(complete.get("cache_key") or "") != key:
                    raise LYXExperimentError(
                        f"lite_timing_cache_key_mismatch:{complete_path}:{key}"
                    )
                elapsed_raw = complete.get("elapsed_s")
        elapsed = _float_or_nan(elapsed_raw)
        if not math.isfinite(elapsed) or elapsed < 0:
            continue
        elapsed_by_key.setdefault(key, elapsed)
        logical_elapsed.append(elapsed)
    logical_s = float(sum(logical_elapsed))
    unique_s = float(sum(elapsed_by_key.values()))
    return {
        "timing_contract": "measured_solver_identity_elapsed_v1",
        "timed_logical_trial_count": len(logical_elapsed),
        "timed_unique_coordinate_count": len(elapsed_by_key),
        "logical_solver_time_estimate_s": logical_s,
        "unique_solver_time_estimate_s": unique_s,
        "cache_saved_time_estimate_s": max(0.0, logical_s - unique_s),
    }


def _write_cache_import_receipt(
    *,
    proposal: Mapping[str, Any],
    execution_root: Path,
) -> dict[str, Any]:
    cache_dir = Path(execution_root) / "cache"
    solver_root = cache_dir / "solver"
    receipt_path = cache_dir / "cache_import_receipt.json"
    if receipt_path.is_file():
        return _read_json(receipt_path)
    entries: list[dict[str, Any]] = []
    source_identities: set[str] = set()
    for complete_path in sorted(solver_root.glob("*/complete.json")):
        report_path = complete_path.with_name("report-v2.json")
        if not report_path.is_file():
            continue
        complete = _read_json(complete_path)
        cache_key = str(complete.get("cache_key") or "")
        source_report = str(complete.get("report_path") or "")
        source_identity = _cache_source_identity(source_report)
        source_identities.add(source_identity)
        entries.append(
            {
                "entry": str(complete_path.parent),
                "cache_key": cache_key,
                "key_prefix_matches_entry": complete_path.parent.name
                == cache_key[:24],
                "source_identity": source_identity,
                "source_report_path": source_report,
                "report_sha256": file_sha256(report_path),
                "complete_sha256": file_sha256(complete_path),
            }
        )
    receipt = {
        "cache_import_receipt_version": CACHE_IMPORT_RECEIPT_VERSION,
        "proposal_sha256": proposal.get("proposal_sha256"),
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "imported_solver_entry_count": len(entries),
        "source_identities": sorted(source_identities),
        "entries": entries,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_write_json(receipt_path, receipt)
    return receipt


def _cache_source_identity(source_report_path: str) -> str:
    source = str(source_report_path or "")
    for marker in ("data\\experiments\\", "data/experiments/"):
        if marker in source:
            tail = source.split(marker, 1)[1]
            return tail.replace("/", "\\").split("\\", 1)[0]
    return source[:200] if source else "unknown"


def _effective_solver_params(params: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "fs_target",
        "max_order",
        "lms_mu_base",
        "smooth_win_len",
        "spec_penalty_width",
        "time_bias",
        "lms_mu_min",
        "analysis_scope",
    }
    return {key: value for key, value in dict(params).items() if key in allowed}


def _solver_result_from_payload(payload: Mapping[str, Any]) -> Any:
    import numpy as np

    from ppg_hr.v2.solver import V2SolverResult

    return V2SolverResult(
        HR=np.asarray(payload.get("hr", []), dtype=float),
        err_stats=dict(payload.get("err_stats") or {}),
        metadata={
            key: value
            for key, value in payload.items()
            if key not in {"hr", "err_stats", "history", "qc", "window_table"}
        },
        window_table=list(payload.get("window_table") or []),
    )


def _metrics_from_report_payload(
    payload: Mapping[str, Any],
    *,
    scene: str,
) -> dict[str, Any]:
    import numpy as np

    hr = np.asarray(payload.get("hr", []), dtype=float)
    if hr.ndim != 2 or hr.shape[0] == 0 or hr.shape[1] < 6:
        raise LYXExperimentError("report_hr_table_invalid")
    err_stats = dict(payload.get("err_stats") or {})
    ref = hr[:, 1]
    final = hr[:, 3]
    motion = hr[:, 4] >= 0.5
    finite = np.isfinite(ref) & np.isfinite(final)
    errors = np.abs(final - ref)
    full_errors = errors[finite]
    motion_errors = errors[finite & motion]
    e10 = finite & (errors >= 10.0)
    e20 = finite & (errors >= 20.0)
    window_table = [
        row for row in payload.get("window_table", []) if isinstance(row, Mapping)
    ]
    stage_counts = sorted(
        {
            len(row.get("adaptive_stages") or [])
            for row in window_table
            if row.get("adaptive_stages")
        }
    )
    actual_stage_count = stage_counts[0] if stage_counts == [2] else -1
    right_censored = _right_censored_recovery_count(e10[finite & motion])
    true_rise = _true_rise_metric(ref[finite & motion], final[finite & motion], scene)
    official_mae = _float_or_nan(err_stats.get("final_aae_bpm"))
    if not math.isfinite(official_mae):
        official_mae = _mean_or_nan(full_errors)
    return {
        "mae_bpm": official_mae,
        "motion_mae_bpm": _mean_or_nan(motion_errors),
        "e10": int(np.count_nonzero(e10)),
        "e20": int(np.count_nonzero(e20)),
        "l10": _longest_bool_run(e10),
        "l20": _longest_bool_run(e20),
        "post_motion_60s_mae_bpm": _float_or_nan(
            err_stats.get("post_motion_60s_mae_bpm")
        ),
        "post_motion_60s_e10_count": int(
            _float_or_nan(err_stats.get("post_motion_60s_e10_count"))
            if math.isfinite(_float_or_nan(err_stats.get("post_motion_60s_e10_count")))
            else 0
        ),
        "post_motion_60s_e20_count": int(
            _float_or_nan(err_stats.get("post_motion_60s_e20_count"))
            if math.isfinite(_float_or_nan(err_stats.get("post_motion_60s_e20_count")))
            else 0
        ),
        "right_censored_recovery_count": int(right_censored),
        "true_rise_underestimate_bpm": true_rise["value"],
        "true_rise_applicable": true_rise["applicable"],
        "spectral_gate_contract_v2": actual_stage_count == 2,
        "stability_pass": bool(len(full_errors) > 0 and np.all(np.isfinite(final[finite]))),
        "actual_adaptive_hf_stage_count": actual_stage_count,
        "actual_adaptive_hf_stage_count_set": stage_counts,
        "reference_groups_order": list(payload.get("reference_groups_order") or []),
        "adaptive_reference_stage_limit": payload.get("adaptive_reference_stage_limit"),
    }


def _audit_lite_non_regression(
    old_metrics: Mapping[str, Any],
    new_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    severe = False
    full_delta = _finite_metric(new_metrics, "mae_bpm") - _finite_metric(
        old_metrics,
        "mae_bpm",
    )
    motion_delta = _finite_metric(new_metrics, "motion_mae_bpm") - _finite_metric(
        old_metrics,
        "motion_mae_bpm",
    )
    if full_delta > 0.5:
        reasons.append("full_mae_regression_gt_0_5")
    if motion_delta > 0.5:
        reasons.append("motion_mae_regression_gt_0_5")
    post_motion_delta = _finite_metric(
        new_metrics,
        "post_motion_60s_mae_bpm",
    ) - _finite_metric(old_metrics, "post_motion_60s_mae_bpm")
    if post_motion_delta > 0.5:
        reasons.append("post_motion_60s_mae_regression_gt_0_5")
    if int(new_metrics["l10"]) > int(old_metrics["l10"]):
        reasons.append("l10_worse")
    if int(new_metrics["l20"]) > int(old_metrics["l20"]):
        reasons.append("l20_worse")
    if int(new_metrics["e10"]) > int(old_metrics["e10"]):
        reasons.append("e10_worse")
    if int(new_metrics["e20"]) > int(old_metrics["e20"]):
        reasons.append("e20_worse")
    if int(new_metrics["post_motion_60s_e10_count"]) > int(
        old_metrics["post_motion_60s_e10_count"]
    ):
        reasons.append("post_motion_60s_e10_worse")
    if int(new_metrics["post_motion_60s_e20_count"]) > int(
        old_metrics["post_motion_60s_e20_count"]
    ):
        reasons.append("post_motion_60s_e20_worse")
    if int(new_metrics["right_censored_recovery_count"]) > int(
        old_metrics["right_censored_recovery_count"]
    ):
        reasons.append("right_censored_recovery_worse")
    if new_metrics.get("stability_pass") is not True:
        reasons.append("curve_stability_failed")
        severe = True
    if full_delta > 2.0 or motion_delta > 2.0 or post_motion_delta > 2.0:
        severe = True
        reasons.append("mae_regression_gt_2")
    if int(old_metrics["l10"]) <= 10 and int(new_metrics["l10"]) >= 20:
        severe = True
        reasons.append("new_severe_l10")
    if int(old_metrics["e20"]) <= 2 and int(new_metrics["e20"]) >= 10:
        severe = True
        reasons.append("new_severe_e20")
    return {
        "requires_fixed_replay": bool(reasons),
        "severe_regression": severe,
        "decision": "stop" if severe else "proceed",
        "reason": "ok" if not reasons else ";".join(reasons),
    }


def _lite_repeat_rows(
    record_id: str,
    history: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat_idx in sorted({int(row["repeat_idx"]) for row in history}):
        repeat = [row for row in history if int(row["repeat_idx"]) == repeat_idx]
        best = min(repeat, key=lambda row: (float(row["value"]), int(row["trial_idx"])))
        values = sorted(float(row["value"]) for row in repeat)
        rows.append(
            {
                "record_id": record_id,
                "repeat_idx": repeat_idx,
                "best_mae_bpm": float(best["value"]),
                "median_mae_bpm": values[len(values) // 2],
                "range_mae_bpm": max(values) - min(values),
                "best_trial_idx": int(best["trial_idx"]),
                "best_params_json": json.dumps(
                    {
                        key: best[key]
                        for key in (
                            "fs_target",
                            "max_order",
                            "lms_mu_base",
                            "smooth_win_len",
                            "spec_penalty_width",
                            "time_bias",
                        )
                    },
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def _current_control_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    for candidate in candidates:
        if candidate["requested_params"] == {
            "fs_target": 25,
            "memory_ms": 200,
            "mu_base": 0.010,
            "exclusion_half_width_bpm": 6,
        }:
            return candidate
    raise LYXExperimentError("current_control_candidate_missing")


def _physical_metrics_for_candidate(
    *,
    proposal: Mapping[str, Any],
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    cache_root: Path,
    logical_stage: str,
) -> dict[str, Any]:
    solved = _solve_cached_report(
        proposal=proposal,
        record=record,
        params=candidate["actual_params"],
        cache_root=cache_root,
        logical_reference={
            "stage": logical_stage,
            "record_id": record["record_id"],
            "candidate_id": candidate["candidate_id"],
        },
        candidate_identity={
            "space_name": candidate["space_name"],
            "candidate_id": candidate["candidate_id"],
            "requested_params": candidate["requested_params"],
        },
    )
    metrics = _cached_metrics_from_report(
        report_path=Path(str(solved["report_path"])),
        record=record,
        scene=str(record["scene"]),
        evaluation_cache_root=_evaluation_cache_root_from_solver_cache(cache_root),
    )
    metrics.update(
        {
            "record_id": record["record_id"],
            "candidate_id": candidate["candidate_id"],
            "coordinate": list(candidate["coordinate"]),
            "cache_key": solved["cache_key"],
            "cache_hit": bool(solved["cache_hit"]),
            "report_path": solved["report_path"],
        }
    )
    return metrics


def _shared_candidate_row(
    *,
    proposal: Mapping[str, Any],
    scene: str,
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    independent: Mapping[str, Any],
    current: Mapping[str, Any],
    cache_root: Path,
    logical_stage: str,
) -> dict[str, Any]:
    metrics = _physical_metrics_for_candidate(
        proposal=proposal,
        record=record,
        candidate=candidate,
        cache_root=cache_root,
        logical_stage=logical_stage,
    )
    gate = evaluate_record_gates(
        candidate=metrics,
        independent=independent,
        current=current,
        scene=scene,
    )
    row = {
        **metrics,
        **gate,
        "scene": scene,
        "independent_mae_bpm": independent["mae_bpm"],
        "current_mae_bpm": current["mae_bpm"],
        "independent_delta_mae_bpm": metrics["mae_bpm"] - independent["mae_bpm"],
        "current_delta_mae_bpm": metrics["mae_bpm"] - current["mae_bpm"],
        "first_failed_gate": (
            "" if gate["qualified"] else str(gate["failed_gates"][0])
        ),
        "failed_gates_json": json.dumps(gate["failed_gates"], ensure_ascii=False),
        "coordinate_json": json.dumps(list(candidate["coordinate"])),
        "requested_params_json": json.dumps(
            candidate["requested_params"],
            ensure_ascii=False,
        ),
    }
    return row


def _evaluate_shared_candidate_batch(
    *,
    proposal: Mapping[str, Any],
    scene: str,
    record: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    independent: Mapping[str, Any],
    current: Mapping[str, Any],
    cache_root: Path,
    logical_stage: str,
    workers: int,
) -> list[dict[str, Any]]:
    def evaluate(candidate: Mapping[str, Any]) -> dict[str, Any]:
        return _shared_candidate_row(
            proposal=proposal,
            scene=scene,
            record=record,
            candidate=candidate,
            independent=independent,
            current=current,
            cache_root=cache_root,
            logical_stage=logical_stage,
        )

    if workers <= 1 or len(candidates) <= 1:
        return [evaluate(candidate) for candidate in candidates]
    with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as pool:
        return list(pool.map(evaluate, candidates))


def _write_fold_freeze(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt = {
        "receipt_version": FOLD_FREEZE_VERSION,
        **dict(payload),
        "heldout_performance_read_count_at_freeze": 0,
    }
    forbidden = {
        key
        for key in receipt
        if key.startswith("heldout_")
        and key not in {"heldout_record", "heldout_performance_read_count_at_freeze"}
    }
    if forbidden:
        raise LYXExperimentError(f"fold_freeze_contains_heldout_metrics:{sorted(forbidden)}")
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    if path.is_file():
        existing = _read_json(path)
        if existing != receipt:
            raise LYXExperimentError(f"fold_freeze_rebinding:{path}")
        return existing
    _atomic_write_json(path, receipt)
    return receipt


def _write_fold_reveal(
    path: Path,
    *,
    freeze: Mapping[str, Any],
    heldout: Mapping[str, Any],
) -> dict[str, Any]:
    freeze_body = dict(freeze)
    freeze_sha = str(freeze_body.pop("receipt_sha256", ""))
    if not freeze_sha or canonical_sha256(freeze_body) != freeze_sha:
        raise LYXExperimentError("fold_freeze_hash_mismatch_before_reveal")
    receipt = {
        "receipt_version": FOLD_REVEAL_VERSION,
        "freeze_receipt_sha256": freeze_sha,
        "heldout_record": freeze["heldout_record"],
        "selected_candidate_id": freeze.get("selected_candidate_id", ""),
        "heldout_metrics": _strict_json_ready(dict(heldout)),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_write_json(path, receipt)
    return receipt


def _run_one_scene_fold(
    *,
    proposal: Mapping[str, Any],
    scene: str,
    records_by_id: Mapping[str, Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    lite_by_record: Mapping[str, Mapping[str, Any]],
    control_metrics: Mapping[str, Mapping[str, Any]],
    cache_root: Path,
    all_rows: dict[tuple[str, str], dict[str, Any]],
    train_ids: Sequence[str],
    heldout_id: str,
    fold_root: Path,
    workers: int,
) -> dict[str, Any]:
    for train_id in train_ids:
        missing = [
            candidate
            for candidate in candidates
            if (train_id, str(candidate["candidate_id"])) not in all_rows
        ]
        evaluated = _evaluate_shared_candidate_batch(
            proposal=proposal,
            scene=scene,
            record=records_by_id[train_id],
            candidates=missing,
            independent=lite_by_record[train_id]["known_best_metrics"],
            current=control_metrics[train_id],
            cache_root=cache_root,
            logical_stage="fold_train",
            workers=workers,
        )
        for row in evaluated:
            all_rows[(train_id, str(row["candidate_id"]))] = row
    train_id_set = set(train_ids)
    training_rows = {
        key: value for key, value in all_rows.items() if key[0] in train_id_set
    }
    fold_id = f"{'+'.join(train_ids)}__holdout_{heldout_id}"
    fold_dir = fold_root / _fold_directory_name(
        train_ids=train_ids,
        heldout_id=heldout_id,
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    try:
        selected = select_near_optimal_candidate(
            candidates=candidates,
            rows=training_rows,
            train_record_ids=train_ids,
        )
    except LYXExperimentError as exc:
        freeze = _write_fold_freeze(
            fold_dir / "selection_freeze.json",
            {
                "scene": scene,
                "fold": fold_id,
                "train_records": list(train_ids),
                "heldout_record": heldout_id,
                "concept_candidate_count": len(candidates),
                "concept_candidates_sha256": canonical_sha256(list(candidates)),
                "selection_status": str(exc),
                "selected_candidate_id": "",
            },
        )
        return {
            "fold": fold_id,
            "train_records": "+".join(train_ids),
            "heldout_record": heldout_id,
            "status": str(exc),
            "heldout_pass": False,
            "selected_candidate_id": "",
            "selected_coordinate": [],
            "freeze_receipt": str(fold_dir / "selection_freeze.json"),
            "freeze_receipt_sha256": freeze["receipt_sha256"],
        }
    selected_id = str(selected["selected_candidate_id"])
    selected_candidate = next(
        item for item in candidates if str(item["candidate_id"]) == selected_id
    )
    freeze = _write_fold_freeze(
        fold_dir / "selection_freeze.json",
        {
            "scene": scene,
            "fold": fold_id,
            "train_records": list(train_ids),
            "heldout_record": heldout_id,
            "concept_candidate_count": len(candidates),
            "concept_candidates_sha256": canonical_sha256(list(candidates)),
            "selection_status": "selected",
            "selector_contract_version": selected["selector_contract_version"],
            "selected_candidate_id": selected_id,
            "selected_coordinate": list(selected_candidate["coordinate"]),
            "selector_receipt": selected,
        },
    )
    heldout_key = (heldout_id, selected_id)
    if heldout_key not in all_rows:
        all_rows[heldout_key] = _shared_candidate_row(
            proposal=proposal,
            scene=scene,
            record=records_by_id[heldout_id],
            candidate=selected_candidate,
            independent=lite_by_record[heldout_id]["known_best_metrics"],
            current=control_metrics[heldout_id],
            cache_root=cache_root,
            logical_stage="fold_heldout_reveal",
        )
    heldout = all_rows[heldout_key]
    reveal = _write_fold_reveal(
        fold_dir / "heldout_reveal.json",
        freeze=freeze,
        heldout=heldout,
    )
    selected_ranking = selected["ranking"][0]
    return {
        "fold": fold_id,
        "train_records": "+".join(train_ids),
        "heldout_record": heldout_id,
        "status": "evaluated",
        "heldout_pass": bool(heldout["qualified"]),
        "selected_candidate_id": selected_id,
        "selected_coordinate": list(selected_candidate["coordinate"]),
        "selected_coordinate_json": json.dumps(list(selected_candidate["coordinate"])),
        "eligible_candidate_count": selected["eligible_candidate_count"],
        "near_optimal_candidate_count": selected["near_optimal_candidate_count"],
        "best_worst_train_mae": selected["best_worst_train_mae"],
        "selected_worst_train_mae": selected_ranking["worst_train_mae"],
        "heldout_mae_bpm": heldout["mae_bpm"],
        "heldout_current_delta_mae_bpm": heldout["current_delta_mae_bpm"],
        "heldout_failed_gates": heldout["failed_gates_json"],
        "heldout_report_path": heldout["report_path"],
        "freeze_receipt": str(fold_dir / "selection_freeze.json"),
        "freeze_receipt_sha256": freeze["receipt_sha256"],
        "reveal_receipt": str(fold_dir / "heldout_reveal.json"),
        "reveal_receipt_sha256": reveal["receipt_sha256"],
        "selector_receipt_json": json.dumps(selected, ensure_ascii=False),
    }


def _load_lite_known_best(lite_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for receipt_path in sorted((lite_root / "records").glob("*/lite_record_receipt.json")):
        receipt = _read_json(receipt_path)
        out[str(receipt["record_id"])] = receipt
    if not out:
        raise LYXExperimentError("lite_known_best_missing")
    return out


def _shared_logical_request_summary(tables: Path) -> dict[str, int]:
    candidate_rows = _read_dict_csv(Path(tables) / "shared_candidate_rows.csv")
    control_rows = _read_dict_csv(Path(tables) / "shared_current_controls.csv")
    candidate_by_identity = {
        (str(row.get("record_id", "")), str(row.get("candidate_id", ""))): row
        for row in candidate_rows
    }
    control_by_identity = {
        (str(row.get("record_id", "")), str(row.get("candidate_id", ""))): row
        for row in control_rows
    }
    overlap = set(candidate_by_identity) & set(control_by_identity)
    control_only = set(control_by_identity) - set(candidate_by_identity)
    physical_identities = set(candidate_by_identity) | set(control_by_identity)
    candidate_cache_hits = sum(
        str(row.get("cache_hit", "")).lower() == "true"
        for row in candidate_by_identity.values()
    )
    control_cache_hits = sum(
        str(row.get("cache_hit", "")).lower() == "true"
        for row in control_by_identity.values()
    )
    identity_cache_hits = sum(
        str(control_by_identity[identity].get("cache_hit", "")).lower() == "true"
        for identity in overlap | control_only
    ) + sum(
        str(candidate_by_identity[identity].get("cache_hit", "")).lower() == "true"
        for identity in set(candidate_by_identity) - set(control_by_identity)
    )
    return {
        "candidate_row_count": len(candidate_by_identity),
        "control_record_count": len(control_by_identity),
        "control_candidate_overlap_count": len(overlap),
        "control_only_logical_request_count": len(control_only),
        "physical_logical_request_count": len(physical_identities),
        "physical_solver_request_count": len(candidate_rows) + len(control_rows),
        "physical_solver_cache_hit_count": int(
            candidate_cache_hits + control_cache_hits
        ),
        "physical_identity_cache_hit_count": int(identity_cache_hits),
    }


def _write_execution_checkpoint(
    *,
    execution_root: Path,
    status: str,
    stage: str,
    last_transaction: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    execution_root = Path(execution_root)
    tables = execution_root / "tables"
    table_state = []
    for path in sorted(tables.glob("*.csv")):
        table_state.append(
            {
                "path": path.relative_to(execution_root).as_posix(),
                "row_count": _csv_row_count(path),
                "sha256": file_sha256(path),
            }
        )
    checkpoint = {
        "checkpoint_version": EXECUTION_CHECKPOINT_VERSION,
        "status": status,
        "stage": stage,
        "last_transaction": last_transaction,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "lite_record_receipt_count": len(
            list((execution_root / "lite" / "records").glob("*/lite_record_receipt.json"))
        ),
        "shared_fold_freeze_count": len(
            list((execution_root / "shared" / "folds").glob("**/selection_freeze.json"))
        ),
        "shared_fold_reveal_count": len(
            list((execution_root / "shared" / "folds").glob("**/heldout_reveal.json"))
        ),
        "tables": table_state,
        "details": _strict_json_ready(dict(details or {})),
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(checkpoint)
    _atomic_write_json(execution_root / "checkpoint.json", checkpoint)
    return checkpoint


def _full_completion(
    *,
    proposal: Mapping[str, Any],
    authorization: Mapping[str, Any],
    status: str,
    started_monotonic: float,
    execution_root: Path,
    lite_receipt: Mapping[str, Any] | None,
    shared_receipt: Mapping[str, Any] | None,
    pause_reason: str | None = None,
) -> dict[str, Any]:
    last_transaction = (
        str(pause_reason).split(":", 1)[-1]
        if pause_reason
        else "finalize_completion"
    )
    checkpoint = _write_execution_checkpoint(
        execution_root=execution_root,
        status=status,
        stage="completion",
        last_transaction=last_transaction,
        details={"pause_reason": pause_reason or ""},
    )
    cache_reports = list((execution_root / "cache" / "solver").glob("*/report-v2.json"))
    cache_import = _read_cache_import_receipt(execution_root)
    final_cache_receipt = _write_final_cache_receipt(execution_root=execution_root)
    artifact_manifest = _write_execution_artifact_manifest(
        execution_root=execution_root,
    )
    imported_solver_count = int(cache_import.get("imported_solver_entry_count", 0))
    unique_new_solver_count = max(0, len(cache_reports) - imported_solver_count)
    shared_logical_summary = _shared_logical_request_summary(
        execution_root / "tables"
    )
    shared_logical = shared_logical_summary["physical_logical_request_count"]
    logical_request_count = int((lite_receipt or {}).get("logical_trial_count", 0)) + shared_logical
    cache_hit_count = int((lite_receipt or {}).get("cache_hit_count", 0)) + int(
        shared_logical_summary["physical_solver_cache_hit_count"]
    )
    elapsed_s = time.monotonic() - started_monotonic
    budget = dict(proposal.get("budget") or {})
    wall_clock_hours = float(
        12 if budget.get("wall_clock_hours") is None else budget.get("wall_clock_hours")
    )
    logical_budget = int(
        budget.get("lite_logical_trials", 0)
        + budget.get("physical_requests", 0)
    )
    solver_budget = int(budget.get("max_solver_requests", 0))
    scene_decisions = list((shared_receipt or {}).get("scene_decisions") or [])
    stable_levels = {"stable_shared_parameter", "stable_shared_neighborhood"}
    stable_scene_count = sum(
        str(row.get("classification")) in stable_levels for row in scene_decisions
    )
    reference_line_pass_count = sum(
        bool(row.get("reference_line_pass")) for row in scene_decisions
    )
    if status == "stopped_after_lite_audit":
        decision = "stop_lite_non_regression_gate"
    elif not scene_decisions:
        decision = "shared_stage_not_completed"
    elif stable_scene_count == len(scene_decisions):
        decision = "all_executed_scenes_stable_shared_support"
    elif stable_scene_count:
        decision = "partial_scene_stable_shared_support"
    else:
        decision = "no_scene_stable_shared_support"
    completion = {
        "completion_version": "lyx_current_source_lite_shared_completion_v3",
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": authorization.get("authorization_sha256"),
        "status": status,
        "decision": decision,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "wall_clock_s": elapsed_s,
        "wall_clock_budget_s": wall_clock_hours * 3600,
        "wall_clock_budget_remaining_s": max(
            0.0,
            wall_clock_hours * 3600 - elapsed_s,
        ),
        "formal_solver_run_count": len(cache_reports),
        "imported_solver_cache_count": imported_solver_count,
        "unique_new_solver_count": unique_new_solver_count,
        "logical_request_count": logical_request_count,
        "logical_request_budget_remaining": max(
            0,
            logical_budget - logical_request_count,
        ),
        "cache_hit_count": cache_hit_count,
        "shared_logical_request_summary": shared_logical_summary,
        "solver_budget_remaining": max(0, solver_budget - unique_new_solver_count),
        "cache_import_receipt": cache_import or None,
        "final_cache_receipt": final_cache_receipt,
        "artifact_manifest": artifact_manifest,
        "checkpoint": checkpoint,
        "execution_root": str(execution_root),
        "lite_receipt": lite_receipt,
        "shared_receipt": shared_receipt,
        "stable_scene_count": stable_scene_count,
        "reference_line_pass_count": reference_line_pass_count,
        "next_state": "final_report_and_code_review" if status == "completed" else "diagnose_or_reauthorize",
    }
    if pause_reason:
        completion["pause_reason"] = str(pause_reason)
    completion["completion_sha256"] = canonical_sha256(completion)
    _atomic_write_json(execution_root / "budget_ledger.json", completion)
    return completion


def _read_cache_import_receipt(execution_root: Path) -> dict[str, Any]:
    receipt_path = Path(execution_root) / "cache" / "cache_import_receipt.json"
    if not receipt_path.is_file():
        return {}
    return _read_json(receipt_path)


def _write_final_cache_receipt(*, execution_root: Path) -> dict[str, Any]:
    cache_root = Path(execution_root) / "cache"
    entries: list[dict[str, Any]] = []
    for cache_kind, payload_name in (
        ("solver", "report-v2.json"),
        ("evaluation", "metrics.json"),
    ):
        for payload_path in sorted((cache_root / cache_kind).glob(f"*/{payload_name}")):
            complete_path = payload_path.with_name("complete.json")
            if not complete_path.is_file():
                continue
            complete = _read_json(complete_path)
            cache_key = str(complete.get("cache_key") or "")
            entries.append(
                {
                    "cache_kind": cache_kind,
                    "entry": str(payload_path.parent),
                    "cache_key": cache_key,
                    "key_prefix_matches_entry": bool(cache_key)
                    and payload_path.parent.name == cache_key[:24],
                    "payload_sha256": file_sha256(payload_path),
                    "complete_sha256": file_sha256(complete_path),
                }
            )
    receipt = {
        "receipt_version": "lyx_final_cache_receipt_v1",
        "solver_entry_count": sum(row["cache_kind"] == "solver" for row in entries),
        "evaluation_entry_count": sum(
            row["cache_kind"] == "evaluation" for row in entries
        ),
        "entries_sha256": canonical_sha256(entries),
        "entries": entries,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_write_json(cache_root / "final_cache_receipt.json", receipt)
    return receipt


def _write_execution_artifact_manifest(*, execution_root: Path) -> dict[str, Any]:
    execution_root = Path(execution_root)
    paths: set[Path] = set()
    for relative_root in ("tables", "figures", "lite", "shared"):
        root = execution_root / relative_root
        if root.is_dir():
            paths.update(path for path in root.rglob("*") if path.is_file())
    for receipt_name in ("cache_import_receipt.json", "final_cache_receipt.json"):
        receipt_path = execution_root / "cache" / receipt_name
        if receipt_path.is_file():
            paths.add(receipt_path)
    checkpoint_path = execution_root / "checkpoint.json"
    if checkpoint_path.is_file():
        paths.add(checkpoint_path)
    entries = [
        {
            "path": path.relative_to(execution_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(paths)
    ]
    manifest = {
        "manifest_version": "lyx_execution_artifact_manifest_v1",
        "artifact_count": len(entries),
        "entries_sha256": canonical_sha256(entries),
        "entries": entries,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _atomic_write_json(execution_root / "artifact_manifest.json", manifest)
    return manifest


def _configure_plot_fonts(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans SC",
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )


def _plot_lite_delta(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_plot_fonts(plt)
    labels = [str(row["record_id"]) for row in rows]
    deltas = [float(row["delta_new_minus_old_mae_bpm"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 4.8))
    colors = ["#d55e00" if value > 0 else "#0072b2" for value in deltas]
    ax.bar(range(len(labels)), deltas, color=colors)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_ylabel("当前 Lite - 旧 Lite MAE (BPM)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_title("LYX 24 记录当前源码 Lite 非退化审核")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def _plot_lite_overview_grid(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    """Render the full Lite panel with a fixed physiological y-axis."""

    if not rows:
        return
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_plot_fonts(plt)
    column_count = 4
    row_count = math.ceil(len(rows) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(13.0, max(3.0, row_count * 2.2)),
        squeeze=False,
        sharey=True,
    )
    for ax, row in zip(axes.flat, rows, strict=False):
        old = np.asarray(_read_json(Path(str(row["old_report"]))).get("hr", []))
        new = np.asarray(_read_json(Path(str(row["new_report"]))).get("hr", []))
        if (
            old.ndim != 2
            or new.ndim != 2
            or old.shape[0] == 0
            or new.shape[0] == 0
            or old.shape[1] < 4
            or new.shape[1] < 4
        ):
            raise LYXExperimentError(
                f"lite_overview_invalid_hr:{row.get('record_id')}"
            )
        ax.plot(old[:, 0], old[:, 1], color="#4d4d4d", linewidth=0.75)
        ax.plot(
            old[:, 0],
            old[:, 3],
            color="#8a8a8a",
            linewidth=0.65,
            linestyle="--",
        )
        ax.plot(new[:, 0], new[:, 3], color="#d55e00", linewidth=0.75)
        ax.set_ylim(40, 220)
        ax.set_title(str(row["record_id"]), fontsize=7)
        ax.grid(True, color="#e6e6e6", linewidth=0.3)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(rows) :]:
        ax.axis("off")
    fig.supxlabel("Window center (s)", fontsize=9)
    fig.supylabel("Heart rate (BPM)", fontsize=9)
    handles = [
        plt.Line2D([], [], color="#4d4d4d", label="Reference"),
        plt.Line2D([], [], color="#8a8a8a", linestyle="--", label="Old Lite"),
        plt.Line2D([], [], color="#d55e00", label="Current Lite-150"),
    ]
    fig.suptitle("LYX Lite-150 full-panel overview", y=0.997, fontsize=11)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.978),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.947))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def _plot_shared_scene_summary(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_plot_fonts(plt)
    labels = [str(row["scene"]) for row in rows]
    values = [int(row["common_safe_candidate_count"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(labels, values, color="#e69f00")
    ax.set_ylabel("三记录公共安全候选数")
    ax.set_title("25 Hz 双级联物理空间公共可行性")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def _plot_shared_candidate_funnel(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_plot_fonts(plt)
    labels = [f"{row['scene']}\n{row['record_id']}" for row in rows]
    incoming = [int(row["incoming_candidate_count"]) for row in rows]
    surviving = [int(row["surviving_candidate_count"]) for row in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(max(8.0, len(rows) * 0.75), 4.8))
    ax.bar(x, incoming, color="#b9c8d4", label="进入本级")
    ax.bar(x, surviving, color="#e69f00", label="累计存活")
    ax.set_ylabel("候选数")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title("场景内 180 组合短路存活漏斗")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def _render_shared_fold_figures(
    *,
    fold_rows: Sequence[Mapping[str, Any]],
    lite_by_record: Mapping[str, Mapping[str, Any]],
    control_metrics: Mapping[str, Mapping[str, Any]],
    path_root: Path,
) -> None:
    for row in fold_rows:
        if row.get("status") != "evaluated":
            continue
        heldout_id = str(row["heldout_record"])
        _plot_shared_fold_hr(
            fold=str(row["fold"]),
            heldout_id=heldout_id,
            selected_report=Path(str(row["heldout_report_path"])),
            known_best_report=Path(
                str(lite_by_record[heldout_id]["summary"]["known_best_report"])
            ),
            control_report=Path(str(control_metrics[heldout_id]["report_path"])),
            path=path_root
            / _fold_figure_name(
                scene=str(row["scene"]),
                fold=str(row["fold"]),
            ),
        )


def _plot_shared_fold_hr(
    *,
    fold: str,
    heldout_id: str,
    selected_report: Path,
    known_best_report: Path,
    control_report: Path,
    path: Path,
) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_plot_fonts(plt)
    selected = np.asarray(_read_json(selected_report)["hr"], dtype=float)
    known = np.asarray(_read_json(known_best_report)["hr"], dtype=float)
    control = np.asarray(_read_json(control_report)["hr"], dtype=float)
    for label, hr in (("selected", selected), ("known", known), ("control", control)):
        if hr.ndim != 2 or hr.shape[0] == 0 or hr.shape[1] < 4:
            raise LYXExperimentError(f"fold_figure_invalid_hr:{fold}:{label}")
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(selected[:, 0], selected[:, 1], color="#4d4d4d", linewidth=1.3, label="参考")
    ax.plot(
        known[:, 0],
        known[:, 3],
        color="#0072b2",
        linewidth=1.0,
        label="逐记录 Lite 锚点",
    )
    ax.plot(
        control[:, 0],
        control[:, 3],
        color="#8a8a8a",
        linewidth=0.9,
        linestyle="--",
        label="200 ms 统一控制",
    )
    ax.plot(
        selected[:, 0],
        selected[:, 3],
        color="#e69f00",
        linewidth=1.1,
        label="fold 冻结参数",
    )
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("心率 (BPM)")
    ax.set_title(f"{heldout_id}：{fold}")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def _plot_lite_hr_overlay(
    *,
    record_id: str,
    old_payload: Mapping[str, Any],
    new_payload: Mapping[str, Any],
    fixed_payload: Mapping[str, Any] | None,
    path: Path,
) -> Path:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_plot_fonts(plt)
    def curve(payload: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        hr = np.asarray(payload.get("hr", []), dtype=float)
        if hr.ndim != 2 or hr.shape[0] == 0 or hr.shape[1] < 4:
            raise LYXExperimentError(f"lite_overlay_invalid_hr:{record_id}")
        return hr[:, 0], hr[:, 1], hr[:, 3]

    old_t, old_ref, old_final = curve(old_payload)
    new_t, new_ref, new_final = curve(new_payload)
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.plot(old_t, old_ref, color="#4d4d4d", linewidth=1.0, label="Reference")
    ax.plot(old_t, old_final, color="#999999", linewidth=0.9, linestyle="--", label="Old Lite")
    ax.plot(new_t, new_final, color="#d55e00", linewidth=1.0, label="Current Lite")
    if fixed_payload is not None:
        fixed_t, _fixed_ref, fixed_final = curve(fixed_payload)
        ax.plot(
            fixed_t,
            fixed_final,
            color="#0072b2",
            linewidth=0.9,
            linestyle=":",
            label="Old best replay",
        )
    ax.set_title(f"{record_id} Lite HR overlay")
    ax.set_xlabel("Window center (s)")
    ax.set_ylabel("Heart rate (BPM)")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, color="#dddddd", linewidth=0.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return path


def _write_dict_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    # Keep the same-directory temp name short: Windows installations without
    # long-path support can reject an otherwise valid target when a UUID and
    # the full target basename push only the temporary path beyond MAX_PATH.
    temporary = path.parent / f".t-{uuid.uuid4().hex[:12]}.tmp"
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: _csv_value(row.get(field, "")) for field in fields}
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_dict_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_row_count(path: Path) -> int:
    return len(_read_dict_csv(path))


def _csv_value(value: Any) -> Any:
    if isinstance(value, Mapping | list | tuple):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _safe_name(value: str) -> str:
    keep = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value]
    return "".join(keep).strip("._") or "item"


def _fold_directory_name(
    *,
    train_ids: Sequence[str],
    heldout_id: str,
) -> str:
    identity = {
        "train_records": list(train_ids),
        "heldout_record": heldout_id,
    }
    return f"fold-{canonical_sha256(identity)[:16]}"


def _fold_figure_name(*, scene: str, fold: str) -> str:
    identity_sha = canonical_sha256({"scene": scene, "fold": fold})[:16]
    return f"{_safe_name(scene)}-fold-{identity_sha}.png"


def _count_values(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _finite_objective(value: Any) -> float:
    parsed = _float_or_nan(value)
    return parsed if math.isfinite(parsed) else 1e9


def _float_or_nan(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def _mean_or_nan(values: Any) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _longest_bool_run(mask: Any) -> int:
    import numpy as np

    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _right_censored_recovery_count(e10_motion: Any) -> int:
    import numpy as np

    e10 = np.asarray(e10_motion, dtype=bool)
    count = 0
    idx = 0
    while idx < len(e10):
        if not bool(e10[idx]):
            idx += 1
            continue
        recovered = False
        cursor = idx + 1
        while cursor + 2 < len(e10):
            if not bool(np.any(e10[cursor : cursor + 3])):
                recovered = True
                break
            cursor += 1
        if not recovered:
            count += 1
            break
        idx = cursor + 3
    return count


def _true_rise_metric(ref: Any, final: Any, scene: str) -> dict[str, Any]:
    import numpy as np

    if not _true_rise_required(scene):
        return {"applicable": False, "value": "not_applicable"}
    ref_arr = np.asarray(ref, dtype=float)
    final_arr = np.asarray(final, dtype=float)
    if len(ref_arr) < 10:
        return {"applicable": True, "value": 0.0}
    values: list[float] = []
    for start in range(0, len(ref_arr) - 9):
        for end in range(start + 10, len(ref_arr) + 1):
            segment = ref_arr[start:end]
            if float(np.max(segment) - segment[0]) < 15.0:
                continue
            if float(np.median(np.diff(segment))) <= 0.0:
                continue
            values.append(float(np.median(segment - final_arr[start:end])))
    return {"applicable": True, "value": max(values) if values else 0.0}


def _support_neighbors(
    *,
    center: Mapping[str, Any],
    row_index: Mapping[str, Mapping[str, Any]],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
) -> int:
    count = 0
    center_coord = tuple(int(value) for value in center["coordinate"])
    center_worst = float(center["worst_train_mae"])
    for candidate_id, candidate in candidates_by_id.items():
        if candidate_id == center["candidate_id"]:
            continue
        coord = tuple(int(value) for value in candidate["coordinate"])
        if not _is_one_level_neighbor(center_coord, coord):
            continue
        row = row_index.get(candidate_id)
        if row is not None and float(row["worst_train_mae"]) <= center_worst + 1.0:
            count += 1
    return count


def _cliff_count(
    *,
    center: Mapping[str, Any],
    row_index: Mapping[str, Mapping[str, Any]],
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    train_record_ids: Sequence[str],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
) -> int:
    count = 0
    center_id = str(center["candidate_id"])
    center_coord = tuple(int(value) for value in center["coordinate"])
    for candidate_id, candidate in candidates_by_id.items():
        if candidate_id == center_id or candidate_id not in row_index:
            continue
        coord = tuple(int(value) for value in candidate["coordinate"])
        if not _is_one_level_neighbor(center_coord, coord):
            continue
        for record_id in train_record_ids:
            center_row = rows[(record_id, center_id)]
            neighbor_row = rows[(record_id, candidate_id)]
            if (
                _finite_metric(center_row, "mae_bpm") <= 5.0
                and _finite_metric(neighbor_row, "mae_bpm") >= 10.0
            ):
                count += 1
    return count


def _possible_neighbor_count(
    candidate_id: str,
    *,
    candidates_by_id: Mapping[str, Mapping[str, Any]],
) -> int:
    center = candidates_by_id[candidate_id]
    center_coord = tuple(int(value) for value in center["coordinate"])
    return sum(
        1
        for other_id, other in candidates_by_id.items()
        if other_id != candidate_id
        and _is_one_level_neighbor(
            center_coord,
            tuple(int(value) for value in other["coordinate"]),
        )
    )


def _support_neighbor_ids(
    candidate_id: str,
    *,
    candidates_by_id: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    center_coord = tuple(
        int(value) for value in candidates_by_id[candidate_id]["coordinate"]
    )
    return {
        other_id
        for other_id, other in candidates_by_id.items()
        if other_id != candidate_id
        and _is_one_level_neighbor(
            center_coord,
            tuple(int(value) for value in other["coordinate"]),
        )
    }


def _candidate_2x2_quads(
    candidates: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, str, str]]:
    by_coord = {
        tuple(int(value) for value in candidate["coordinate"]): str(
            candidate["candidate_id"]
        )
        for candidate in candidates
    }
    quads: set[tuple[str, str, str, str]] = set()
    for coord in by_coord:
        for axis_a, axis_b in itertools.combinations(range(len(coord)), 2):
            other = list(coord)
            other[axis_a] += 1
            a = tuple(other)
            other = list(coord)
            other[axis_b] += 1
            b = tuple(other)
            other = list(coord)
            other[axis_a] += 1
            other[axis_b] += 1
            ab = tuple(other)
            if a in by_coord and b in by_coord and ab in by_coord:
                quads.add(
                    tuple(
                        sorted(
                            (
                                by_coord[coord],
                                by_coord[a],
                                by_coord[b],
                                by_coord[ab],
                            )
                        )
                    )
                )
    return sorted(quads)


def _connected_components(
    candidate_ids: Sequence[str],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
) -> list[list[str]]:
    remaining = set(candidate_ids)
    components: list[list[str]] = []
    while remaining:
        start = remaining.pop()
        queue = [start]
        component = {start}
        while queue:
            current = queue.pop()
            current_coord = tuple(
                int(value) for value in candidates_by_id[current]["coordinate"]
            )
            for other in list(remaining):
                other_coord = tuple(
                    int(value) for value in candidates_by_id[other]["coordinate"]
                )
                if _is_one_level_neighbor(current_coord, other_coord):
                    remaining.remove(other)
                    component.add(other)
                    queue.append(other)
        components.append(sorted(component))
    return sorted(components, key=lambda item: (-len(item), item))


def _candidate_id_for_coordinate(
    candidates_by_id: Mapping[str, Mapping[str, Any]],
    coordinate: tuple[int, ...],
) -> str:
    for candidate_id, candidate in candidates_by_id.items():
        if tuple(int(value) for value in candidate["coordinate"]) == coordinate:
            return str(candidate_id)
    return ""


def _is_one_level_neighbor(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    deltas = [abs(a - b) for a, b in zip(left, right, strict=True)]
    return sum(delta > 0 for delta in deltas) == 1 and max(deltas, default=0) == 1


def _required_mean_independent_delta(
    rows: Sequence[Mapping[str, Any] | None],
) -> float:
    values: list[float] = []
    for row in rows:
        if row is None:
            raise KeyError("independent_delta_mae_bpm")
        values.append(_finite_metric(row, "independent_delta_mae_bpm"))
    return sum(values) / len(values)


def _paused_completion(
    *,
    proposal: Mapping[str, Any],
    authorization_sha256: str | None,
    paused_at: str,
) -> dict[str, Any]:
    completion: dict[str, Any] = {
        "completion_version": PAUSE_COMPLETION_VERSION,
        "proposal_sha256": proposal["proposal_sha256"],
        "authorization_sha256": authorization_sha256,
        "status": "paused_authorization_expired_before_start",
        "paused_at": _parse_datetime(paused_at).isoformat(),
        "formal_solver_run_count": 0,
        "logical_request_count": 0,
        "cache_hit_count": 0,
        "next_state": "requires_fresh_execution_authorization",
    }
    completion["completion_sha256"] = canonical_sha256(completion)
    return completion


def _clock_override_allowed(allow_clock_override: bool) -> bool:
    return allow_clock_override or os.environ.get(CLOCK_OVERRIDE_ENV) == "1"


def _deadline_expired(proposal: Mapping[str, Any], *, now: str | None = None) -> bool:
    now_dt = _parse_datetime(
        now or datetime.now().astimezone().isoformat(timespec="seconds")
    )
    deadline_dt = _parse_datetime(str(proposal.get("authorization_deadline")))
    return now_dt >= deadline_dt


def _true_rise_required(scene: str) -> bool:
    return str(scene) in {"run", "kaihe"}


def _explicit_not_applicable(metrics: Mapping[str, Any]) -> bool:
    return (
        metrics.get("true_rise_applicable") is False
        and metrics.get("true_rise_underestimate_bpm") == "not_applicable"
    )


def _finite_metric(metrics: Mapping[str, Any] | None, name: str) -> float:
    if metrics is None:
        raise KeyError(name)
    value = metrics[name]
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(name)
    return number


def _gate_result(qualified: bool, failed: Sequence[str]) -> dict[str, Any]:
    return {
        "gate_contract_version": GATE_CONTRACT_VERSION,
        "qualified": bool(qualified),
        "failed_gates": list(failed),
    }


def _require_sha256(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LYXExperimentError(f"{name}_must_be_sha256")


def _resolve_panel_root(data_root: Path) -> Path:
    root = data_root.resolve()
    if (root / "v2_batch_outputs").is_dir() or (root / "nouse-data").is_dir():
        return root
    candidate = root / LYX_PANEL_RELATIVE
    if candidate.is_dir():
        return candidate
    raise LYXExperimentError(f"lyx_panel_root_missing:{root}")


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        resolved = path.resolve()
        if resolved.is_file():
            return resolved
    return None


def _scene_from_record_id(record_id: str) -> str:
    return record_id.split("_", 1)[0].rstrip("0123456789")


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise LYXExperimentError(f"invalid_datetime:{value}") from exc
    if parsed.tzinfo is None:
        raise LYXExperimentError(f"datetime_requires_timezone:{value}")
    return parsed


def _git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LYXExperimentError(f"git_identity_failed:{' '.join(args)}") from exc
    return completed.stdout.strip()


def _git_status_short(repo: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.splitlines()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(
        json.dumps(
            _json_ready(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _strict_json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_strict_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_strict_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "nonfinite"
    return value


if __name__ == "__main__":
    raise SystemExit(main())
